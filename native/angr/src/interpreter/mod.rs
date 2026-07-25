//! Callback-aware VEX IR interpreter.
//!
//! This interpreter uses Python callbacks for memory operations instead of
//! local SymbolicMemory. It can run multiple blocks in a loop, returning
//! to Python only when an event requires Python handling.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;

use lru::LruCache;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::arch::{
    RegisterFile, arch_from_vex, calling_conventions::CallingConvention, default_cc_for_arch,
};
use crate::callbacks::{DeferredFork, ExecutionConfig, PythonCallbacks, RunErrorKind, RunResult};
use crate::claripy_bridge::{claripy_to_rustbv, is_claripy_ast, try_handle_to_rustbv};
use crate::concretize::{AddressConcretizer, ConcretizationResult};
use crate::memory::{MemoryError, Permission, SymbolicMemory};
use crate::symbolic::{
    BVOp, RustBV, RustSymbolTable, SymContext, record_mem_load, record_mem_store, record_vex_binop,
    record_vex_qop, record_vex_triop, record_vex_unop,
};
use crate::vex::ccall;
use crate::vex::dirty::DirtyHelperDispatch;
use crate::vex::ir::{
    IRConst, IRExpr, IRLoadGOp, IROp, IRSB, IRStmt, IRType, JumpKind, TypeEnv, VexArch,
};
use crate::vex::ops::{OpError, VEXOps, iropclass};
use crate::vex::{Endness, deserialize_irsb};

/// VEX block (IRSB) LRU cache capacity in entries.
///
/// Each cached `IRSB` is Arc-shared, so this caps unique IRSBs simultaneously
/// resident across an interpreter's lifetime. 4096 covers the working set of
/// every tracked bench in `baseline_timings.json`; adjust only with a counter
/// dump showing eviction churn (`mgr.stats()` does not expose hit/miss today;
/// see angr-l4bs).
pub(crate) const BLOCK_CACHE_CAPACITY: usize = 4096;

/// Start a profiling timer iff `self.profiling_enabled`. Yields `Option<Instant>`.
///
/// Pair with [`profile_add!`] to fold the elapsed nanoseconds into a `u64`
/// stats field. When profiling is off both macros collapse to a single branch.
macro_rules! profile_start {
    ($self:expr) => {
        if $self.profiling_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        }
    };
}

/// Add the elapsed time since `$start` (an `Option<Instant>` from
/// [`profile_start!`]) to a `u64` field. No-op when `$start` is `None`.
macro_rules! profile_add {
    ($start:expr, $field:expr) => {
        if let Some(__profile_start) = $start {
            $field += __profile_start.elapsed().as_nanos() as u64;
        }
    };
}

mod code_invalidation;
mod concrete_memory;
mod concretize_cache;
mod execution;
mod exits;
mod expressions;
mod fork_state;
mod helpers;
mod pending_store;
mod prefetch;
mod simprocedures;
mod statements;
mod statements_cas;
mod statements_inspect;
mod statements_store;

use helpers::bytes_to_bv;
use pending_store::PendingStoreBuffer;

pub use concrete_memory::ConcreteMemoryRegion;
pub use simprocedures::SimProcedureInfo;

/// Full state snapshot at a symbolic branch point.
/// Used by deferred forks to create correct alternate-path states
/// with solver, registers, and memory from the branch point.
pub struct BranchSnapshot {
    pub solver: SymContext,
    pub registers: RegisterFile,
    pub memory: Option<SymbolicMemory>,
}

/// Generates `ExecutionStats` plus its `to_hashmap` / `merge` impls from a
/// single field list. `: sum` accumulates on merge; `: snapshot` overwrites.
/// Adding a stat means editing only the invocation below.
macro_rules! define_execution_stats {
    (
        $(
            $(#[$attr:meta])*
            $field:ident: $mode:ident,
        )*
    ) => {
        /// Execution statistics for profiling.
        ///
        /// Tracks timing and counts for various operations during VEX execution.
        /// Times are in nanoseconds for precision.
        #[derive(Debug, Clone, Default)]
        pub struct ExecutionStats {
            $(
                $(#[$attr])*
                pub $field: u64,
            )*
        }

        impl ExecutionStats {
            /// Convert stats to a HashMap for Python exposure.
            pub fn to_hashmap(&self) -> HashMap<String, u64> {
                let mut map = HashMap::new();
                $(
                    map.insert(stringify!($field).to_string(), self.$field);
                )*
                map
            }

            /// Reset all statistics to zero.
            pub fn reset(&mut self) {
                *self = Self::default();
            }

            /// Merge another stats instance into this one.
            pub fn merge(&mut self, other: &ExecutionStats) {
                $(
                    define_execution_stats!(@merge_field self.$field, other.$field, $mode);
                )*
            }
        }
    };
    (@merge_field $self_field:expr, $other_field:expr, sum) => {
        $self_field += $other_field;
    };
    (@merge_field $self_field:expr, $other_field:expr, snapshot) => {
        $self_field = $other_field;
    };
}

define_execution_stats! {
    /// Number of load statements executed.
    load_stmt_count: sum,
    /// Time spent in load statements (nanoseconds).
    load_stmt_time_ns: sum,
    /// Number of store statements executed.
    store_stmt_count: sum,
    /// Time spent in store statements (nanoseconds).
    store_stmt_time_ns: sum,
    /// Number of exit statements executed.
    exit_stmt_count: sum,
    /// Time spent in exit statements (nanoseconds).
    exit_stmt_time_ns: sum,
    /// Number of Python callback invocations.
    python_callback_count: sum,
    /// Time spent in Python callbacks (nanoseconds).
    python_callback_time_ns: sum,
    /// Number of address concretizations performed.
    concretize_count: sum,
    /// Time spent in address concretization (nanoseconds).
    concretize_time_ns: sum,
    /// Number of IRSB cache hits.
    cache_hit_count: sum,
    /// Number of IRSB cache misses (lifts needed).
    cache_miss_count: sum,
    /// Number of IRSB cache evictions (capacity reached, LRU entry dropped).
    /// Tallied at every `block_cache.put()` site whose return value is `Some`
    /// AND the key was not already present (an overwrite, not an eviction).
    /// Together with `cache_hit_count`/`cache_miss_count` this gives the data
    /// needed to tune `BLOCK_CACHE_CAPACITY`: a high eviction-to-miss ratio
    /// indicates capacity pressure (working set exceeds cache); near-zero
    /// evictions on a benchmark mean the cache is oversized for that workload.
    cache_eviction_count: sum,
    /// Time spent lifting blocks (nanoseconds).
    lift_time_ns: sum,
    /// Number of Rust memory loads (vs callback fallback).
    rust_memory_load_count: sum,
    /// Number of Python fallback memory loads.
    fallback_memory_load_count: sum,
    /// Number of Rust memory stores.
    rust_memory_store_count: sum,
    /// Number of Python fallback memory stores.
    fallback_memory_store_count: sum,
    /// Number of expression evaluations.
    expr_eval_count: sum,
    /// Time spent evaluating expressions (nanoseconds).
    expr_eval_time_ns: sum,
    /// Number of blocks executed.
    blocks_executed: sum,
    /// Total execution time (nanoseconds).
    total_time_ns: sum,
    /// Time spent setting up interpreter per step (nanoseconds).
    step_setup_time_ns: sum,
    /// Number of exploration steps executed.
    step_count: sum,
    /// Number of solver satisfiability checks.
    solver_sat_count: sum,
    /// Time spent in solver satisfiability checks (nanoseconds).
    solver_sat_time_ns: sum,
    /// Time spent executing blocks (nanoseconds) — the inner VEX execution.
    block_exec_time_ns: sum,
    /// Number of statements executed.
    stmt_count: sum,
    /// Number of deferred forks PRESENTED to the exploration loop's
    /// post-block processing (input length of the `deferred_forks` Vec,
    /// summed across all processing sites). NOT every entry produces a
    /// solver clone: if a stored/reconstructed condition is unavailable
    /// the entry is silently skipped in the run_loop callback-resume path
    /// (run_loop.rs ~line 477), or routed through a no-condition
    /// conservative `state.fork()` in the stepping paths (which DOES clone
    /// the solver but is not tallied below). This counter is therefore
    /// NOT directly comparable to `solver_fork_count` — they measure
    /// orthogonal but overlapping concepts. See angr-95up.2.
    deferred_fork_count: sum,
    /// Time spent processing deferred forks (nanoseconds).
    deferred_fork_time_ns: sum,
    /// Time spent in solver fork/clone operations (nanoseconds).
    solver_fork_time_ns: sum,
    /// Number of solver fork operations whose Z3-clone cost is timed by
    /// `solver_fork_time_ns`. Tallied at three sites: (1) pre-callback
    /// state-snapshot fork in the SimProcedure path (stepping.rs ~line
    /// 151, NOT a deferred fork); (2) per-deferred-fork creation in
    /// `handle_block_end_or_max_blocks` when a condition is available
    /// (stepping.rs ~line 493); (3) per-deferred-fork creation in the
    /// callback-resume path when a condition is available (run_loop.rs
    /// ~line 513). NOT incremented by the no-condition conservative-fork
    /// fallback in `process_deferred_forks_into` (stepping.rs ~line 1200)
    /// or the main stepping fallback (stepping.rs ~line 531). Because of
    /// (1), this counter generally exceeds the deferred-fork-with-
    /// condition subset of `deferred_fork_count`; because of the
    /// fallbacks, the relationship is not a simple sum. See angr-95up.2.
    solver_fork_count: sum,
    /// Number of active states at end of run.
    active_states_count: snapshot,
    /// Time spent in the main run() loop overhead (nanoseconds).
    run_loop_time_ns: sum,
    /// Number of dirty-helper invocations that fell back to the Python
    /// `call_dirty_call` callback (no native handler matched).
    python_dirty_call_count: sum,
    /// Number of VEX op evaluations (Unop/Binop/Triop/Qop) that hit the
    /// silent symbolic-synthesis fallback because `VEXOps::*` returned an
    /// `OpError`. These never invoke Python; the value is replaced with a
    /// fresh symbolic (when any input was symbolic) or zero.
    python_vex_op_fallback_count: sum,
    /// Subset of `python_vex_op_fallback_count` that came from `Unop`.
    python_vex_unop_fallback_count: sum,
    /// Subset of `python_vex_op_fallback_count` that came from `Binop`.
    python_vex_binop_fallback_count: sum,
    /// Subset of `python_vex_op_fallback_count` that came from `Triop`.
    python_vex_triop_fallback_count: sum,
    /// Subset of `python_vex_op_fallback_count` that came from `Qop`.
    python_vex_qop_fallback_count: sum,
    /// Number of Unop/Binop evaluations that took the *fabricate-fresh-symbolic*
    /// BYPASS (the `any_sym` arm of `eval_unop`/`eval_binop`): the op returned an
    /// `OpError`, an input was symbolic, so a fresh unconstrained symbolic stood
    /// in for the real value. A strict subset of `python_vex_op_fallback_count`
    /// (excludes the concrete-arg arm, which propagates a typed error). Since
    /// angr-oyzvj this BYPASS is OPT-IN — it fires only when
    /// `ANGR_RUST_FABRICATE_UNSUPPORTED_IROP` is set; by default the symbolic arm
    /// routes to Python (`NeedPythonFallback`), so this counter stays 0 unless
    /// the escape hatch is enabled. The three dispatch-fabricate families
    /// (VPerm/Pclmul*/Crc32C, angr-s6miz) are routed to Python fallback *before*
    /// this point, so they do NOT increment this counter. See bd
    /// `vex-dispatch-bypass-inventory`.
    vex_bypass_fabricate_count: sum,
    /// Number of cold-block lifts served natively via the `libvex-ffi`
    /// `NativeLibVEXLifter` (feature-gated, off by default). Each hit is a
    /// Python `lift_block` callback + JSON round-trip that did NOT happen.
    native_lift_count: sum,
    /// Number of native-lift *attempts* that fell back to the Python callback
    /// (feature off / disabled paths do not count — only a live attempt that
    /// could not complete: missing-or-symbolic block bytes, unsupported arch,
    /// or a libVEX lift error). A high fallback:hit ratio means the native
    /// path is not paying off for that workload.
    native_lift_fallback_count: sum,
    /// Subset of `native_lift_fallback_count`: misses at an address that lies
    /// outside every loaded binary region, so no lifter — native or pyvex — can
    /// produce a block. The Python callback returns the `"{}"` sentinel and the
    /// state deadends; nothing was lost by falling back. Subtract this from
    /// `native_lift_fallback_count` to get the misses that are genuinely lost
    /// native-lift wins. Measured on `cow_fork_scaling` (angr-op0dn.2.3), all
    /// 256 apparent fallbacks were return-to-0x0 deadend probes of exactly this
    /// kind — the native path was in fact serving 21 of 21 real blocks.
    native_lift_deadend_probe_count: sum,
}

/// Reason string used by the CAS handler when it sees a double-CAS (cmpxchg16b).
/// Shared with `exploration::mod` so the manager can identify DCAS in
/// `PythonVEXFallback` events and bump a dedicated visibility counter.
pub const DCAS_UNSUPPORTED_REASON: &str = "double compare-and-swap";

/// Reason marker used by the VECRET/GSPTR fallback site
/// (`expressions.rs::eval_expr_with_callbacks`). The manager scans for this
/// substring in `PythonVEXFallback` reasons and bumps
/// `vecret_gsptr_fallback_count` so we can measure how often the corpus
/// actually exercises these vector-call/global-state pointer holders.
/// See bd `angr-2iow` — prevalence drives whether to implement natively or
/// document as a corpus-absent limitation.
pub const VECRET_GSPTR_REASON: &str = "VECRET/GSPTR";

/// Reason marker for the three dispatch-fabricate binop families
/// (`Iop_Perm8x*` => `VPerm`, `Iop_Pclmul*`, `Iop_Crc32C`). These parse to a
/// concrete IROp but have no native dispatch arm, so `VEXOps::binop` returns
/// `OpError::NotBinary`. Rather than fabricate a wrong fresh symbolic
/// (`eval_binop`'s BYPASS arm), `eval_binop` routes them to Python's VEX engine
/// — deterministic ops Python models exactly. See bd `angr-s6miz` /
/// `vex-dispatch-bypass-inventory`.
pub const DISPATCH_FABRICATE_REASON: &str = "dispatch-fabricate bypass";

/// How an error variant should be handled by the top-level interpreter loop.
///
/// Every [`CbExecutionError`] variant maps to one of these via
/// [`CbExecutionError::strategy`]. Adding a new variant requires an explicit
/// strategy decision — there is no default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackStrategy {
    /// Hand the failing block to Python's VEX engine and resume from there.
    /// Used for VEX features the Rust interpreter doesn't model
    /// (e.g. unsupported CCalls, VECRET/GSPTR, oversized symbolic addresses).
    PythonCallback,
    /// Surface the error to the caller as `RunResult::Error`. The state
    /// moves to the errored stash; no recovery is attempted. Used for
    /// genuine bugs (TypeMismatch, UnknownTemp, InvalidIR, lifter errors,
    /// callback-side failures).
    ///
    /// NOTE: `Op` / `TypeMismatch` / `InvalidIR` are exactly the failures
    /// that Python's `HeavyResilienceMixin` would catch and substitute a
    /// default for when `BYPASS_ERRORED_IROP` / `_IRCCALL` / `_IRSTMT` is
    /// set. Because Rust terminates here instead of handing the block to
    /// Python, those bypasses cannot fire — a silent divergence. Rather
    /// than diverge silently, the Python wrapper raises `NotImplementedError`
    /// at manager construction if any `BYPASS_ERRORED_*` option is set (see
    /// `_RAISE_OPTION_NAMES` in `angr/exploration/rust_manager.py`). Wiring a
    /// real bypass would mean routing these variants through `PythonCallback`
    /// (and re-classifying them as recoverable) so the resilience mixin can
    /// act; until then the raise is the contract.
    Panic,
}

/// Errors during callback-based VEX execution.
///
/// Each variant has a documented [`FallbackStrategy`]. The dispatcher in
/// `execution.rs::run` consults [`Self::strategy`] to decide whether the
/// error becomes `RunResult::NeedPythonVEX` (recoverable) or
/// `RunResult::Error` (terminal).
#[derive(Debug, Clone, thiserror::Error)]
pub enum CbExecutionError {
    /// Memory error from callback. Strategy: [`FallbackStrategy::Panic`].
    /// These come from underlying memory-model failures (unmapped, perms,
    /// solver timeout) that the interpreter can't paper over.
    #[error("memory error: {0}")]
    Memory(String),
    /// Operation error. Strategy: [`FallbackStrategy::Panic`].
    /// VEX op execution failed in a non-recoverable way; lifting to Python
    /// would just rerun the same op.
    #[error("operation error: {0}")]
    Op(#[from] OpError),
    /// Invalid VEX IR. Strategy: [`FallbackStrategy::Panic`].
    #[error("invalid VEX IR: {0}")]
    InvalidIR(String),
    /// Unsupported feature. Strategy: [`FallbackStrategy::PythonCallback`].
    /// Triggered when the Rust interpreter encounters VEX it doesn't model
    /// (e.g. complex symbolic memory operations, certain DirtyHelpers,
    /// symbolic exit targets mid-block).
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// Unknown temporary variable. Strategy: [`FallbackStrategy::Panic`].
    #[error("unknown temporary t{0}")]
    UnknownTemp(u32),
    /// Python callback error. Strategy: [`FallbackStrategy::Panic`].
    /// The Python side already had its chance and raised; rerunning the
    /// block via the VEX engine would not help.
    #[error("callback error: {0}")]
    Callback(String),
    /// Block lifting error. Strategy: [`FallbackStrategy::Panic`].
    #[error("lift error: {0}")]
    LiftError(String),
    /// Needs Python fallback for special expressions (P7 fix).
    /// Strategy: [`FallbackStrategy::PythonCallback`]. Distinct from
    /// `Unsupported` so call sites can request fallback explicitly without
    /// having to invent a "feature missing" message (e.g. VECRET/GSPTR,
    /// non-eflags CCalls).
    #[error("need Python fallback: {0}")]
    NeedPythonFallback(String),
}

impl CbExecutionError {
    /// Map this error to its declared [`FallbackStrategy`]. The match is
    /// exhaustive so adding a variant forces an explicit strategy choice.
    pub fn strategy(&self) -> FallbackStrategy {
        match self {
            CbExecutionError::Unsupported(_) | CbExecutionError::NeedPythonFallback(_) => {
                FallbackStrategy::PythonCallback
            }
            CbExecutionError::Memory(_)
            | CbExecutionError::Op(_)
            | CbExecutionError::InvalidIR(_)
            | CbExecutionError::UnknownTemp(_)
            | CbExecutionError::Callback(_)
            | CbExecutionError::LiftError(_) => FallbackStrategy::Panic,
        }
    }

    /// Classify this error for the exploration stepping loop (angr-zzju9).
    ///
    /// `LiftError` is the designed signal that a block could not be lifted —
    /// the Python lift callback returned the empty-IRSB sentinel (e.g. on
    /// `SimEngineError: No bytes in memory`) or the callback itself failed.
    /// Such states gracefully deadend, matching the vanilla Python engine.
    /// Every other `Panic`-strategy variant — including `InvalidIR`, which a
    /// genuinely malformed IRSB now maps to — is a real error that moves the
    /// state to the errored stash.
    pub fn run_error_kind(&self) -> RunErrorKind {
        match self {
            CbExecutionError::LiftError(_) => RunErrorKind::Deadend,
            _ => RunErrorKind::Fatal,
        }
    }
}

/// Result of concretizing a symbolic jump target.
enum ConcretizedJump {
    /// Single concrete address (common case for deterministic jumps).
    Single(u64),
    /// Multiple concrete addresses (for symbolic ret/call/jmp).
    /// Contains the list of targets and the original symbolic expression.
    Multiple { targets: Vec<u64>, expr: RustBV },
    /// Too many targets - exceeds max_symbolic_ip_targets limit.
    /// State should be marked as unconstrained.
    TooMany { min: u64, max: u64, limit: usize },
}

/// Result of executing a single statement.
enum StmtResult {
    /// Continue to next statement.
    Continue,
    /// Exit the block early.
    Exit { target: u64, jumpkind: JumpKind },
    /// Symbolic branch detected - need to fork.
    SymbolicBranch {
        condition: RustBV,
        true_target: u64,
        false_target: u64,
    },
}

/// Result of executing a single block.
#[derive(Debug)]
pub enum BlockResult {
    /// Continue to the next block at given address.
    Continue { next_addr: u64 },
    /// Syscall encountered. `num` is `None` when the syscall-number register
    /// is symbolic (angr-gffd); callers must route those cases to Python so
    /// `engines/successors.py::_resolve_syscall` can enumerate or honor
    /// `NO_SYMBOLIC_SYSCALL_RESOLUTION` instead of silently dispatching to
    /// `read` (amd64 syscall 0).
    Syscall { num: Option<u64> },
    /// Symbolic branch - need to fork.
    SymbolicBranch {
        condition_id: u64,
        true_target: u64,
        false_target: u64,
    },
    /// Hook address hit.
    Hook { addr: u64 },
    /// Block execution error.
    Error { message: String },
    /// Normal block end with jumpkind.
    BlockEnd { next_addr: u64, jumpkind: JumpKind },
    /// Symbolic jump target with multiple concrete targets after concretization.
    /// The exploration manager should fork states for each target.
    SymbolicJumpTarget {
        /// Concrete target addresses after concretization.
        targets: Vec<u64>,
        /// ID for the stored symbolic expression (for constraint addition).
        condition_id: u64,
        /// The symbolic expression for the jump target.
        target_expr: RustBV,
        /// Jump kind (Ijk_Ret, Ijk_Call, etc.).
        jumpkind: JumpKind,
    },
    /// Unconstrained jump - too many targets, exceeds limit.
    /// The state should be moved to the "unconstrained" stash.
    UnconstrainedJump {
        /// Minimum possible target address.
        min_target: u64,
        /// Maximum possible target address.
        max_target: u64,
        /// The configured limit that was exceeded.
        limit: usize,
        /// Jump kind.
        jumpkind: JumpKind,
    },
    /// Unmodeled function call - target is not hooked but is a CALL.
    /// Need Python to check if a SimProcedure can be resolved.
    UnmodeledCall {
        /// Address of the unmodeled function.
        addr: u64,
        /// Return address (from stack).
        return_addr: u64,
        /// Symbol name if available.
        symbol_name: Option<String>,
    },
}

/// Callback-aware VEX IR interpreter.
///
/// This interpreter uses Python callbacks for memory and register access,
/// allowing it to work with angr's symbolic memory model.
///
/// When `use_rust_memory` is true, the interpreter uses `rust_memory` for
/// memory operations, falling back to Python callbacks only for unmapped pages.
/// This provides significant performance improvement for memory-intensive code.
pub struct VEXInterpreter<'a> {
    /// Register file (local cache, synced via callbacks).
    pub registers: RegisterFile,
    /// Temporary variables for current block.
    temps: Vec<Option<RustBV>>,
    /// Solver context.
    ctx: &'a SymContext,
    /// Symbol table for handle-based claripy bypass.
    /// When present, Python callbacks can return RustBVHandle instead of claripy ASTs.
    symbol_table: Option<&'a RustSymbolTable>,
    /// Current program counter.
    pub pc: u64,
    /// Current instruction address (within block).
    current_insn_addr: u64,
    /// Current instruction length (from IMark).
    current_insn_len: u32,
    /// Hook addresses (return to Python when hit).
    /// Arc-shared on fork (O(1) clone). Mutators use Arc::make_mut for CoW.
    hook_addrs: Arc<FxHashSet<u64>>,
    /// VEX architecture.
    arch: VexArch,
    /// Native libVEX lifter (feature `libvex-ffi`). Stateless (holds only a
    /// process-wide lift lock); a fresh `Default` is fine per fork.
    #[cfg(feature = "libvex-ffi")]
    native_lifter: crate::vex::libvex_lifter::NativeLibVEXLifter,
    /// When true (and the arch/bytes qualify), cold-block lifts are attempted
    /// in-process via `native_lifter` before the Python `lift_block` callback.
    /// Off by default — Stage-2 flag-gated engine use (angr-z087y).
    #[cfg(feature = "libvex-ffi")]
    native_lift_enabled: bool,
    /// Block cache (shared across runs) using Arc for O(1) cloning.
    block_cache: LruCache<u64, Arc<IRSB>>,
    /// Whether to use callbacks for memory (vs local registers).
    use_memory_callbacks: bool,
    /// Deferred forks collected during execution.
    /// Each fork represents a branch where we took one path and deferred the other.
    deferred_forks: Vec<DeferredFork>,
    /// Whether a deferred fork was already created this step.
    /// Limits to one deferred fork per run_until_event call to prevent
    /// solver corruption from under-constrained multi-block execution.
    deferred_fork_this_step: bool,
    /// Whether we've pushed the solver for incremental branch constraint tracking.
    /// When true, the solver has accumulated taken-path conditions from prior
    /// deferred forks in this block. Must pop at block end.
    block_solver_pushed: bool,
    /// Number of deferred_forks conditions already asserted in the pushed context.
    block_forks_asserted: usize,
    /// Execution configuration.
    config: ExecutionConfig,
    /// Counter for alternating branch policy.
    branch_counter: u64,
    /// Next condition ID for tracking branch conditions.
    next_condition_id: u64,
    /// Current solver push level for constraint tracking.
    /// Incremented when we push before adding a branch constraint.
    push_level: u32,
    /// Concrete memory regions cached locally for fast access.
    /// These are read-only regions (e.g., binary .text/.rodata sections).
    /// Arc-shared on fork (O(1) clone). Mutators use Arc::make_mut for CoW.
    concrete_memory: Arc<Vec<ConcreteMemoryRegion>>,
    /// Address concretizer for handling symbolic addresses.
    concretizer: AddressConcretizer,
    /// Bitset tracking which register offsets have been modified.
    /// Each bit represents a 4-byte aligned offset (offset / 4).
    /// A u128 covers 512 bytes of register space (128 * 4 = 512).
    dirty_registers: u128,
    /// Pending concrete stores to batch for efficiency.
    /// Each entry is (address, data_bytes). Wrapped in a buffer that maintains
    /// a per-byte-address index so loads can fast-skip the reverse scan.
    pending_stores: PendingStoreBuffer,
    /// All stores flushed during this step (accumulated across block boundaries).
    /// Used for same-step cross-block load forwarding and for applying to state memory.
    /// HashMap for O(1) lookup by address. Value is the most recent store data.
    all_flushed_stores: FxHashMap<u64, Vec<u8>>,
    /// All symbolic stores flushed during this step (accumulated across block boundaries).
    /// Preserves symbolic RustBV values for cross-block load forwarding.
    /// Takes priority over all_flushed_stores (concrete) during loads.
    all_flushed_symbolic_stores: FxHashMap<u64, RustBV>,
    /// Pending symbolic stores - maps address to symbolic RustBV.
    /// These override the concrete bytes in pending_stores for load forwarding.
    pending_symbolic_stores: FxHashMap<u64, RustBV>,
    /// Maximum pending stores before auto-flush.
    max_pending_stores: usize,
    /// Rust-native symbolic memory (replaces Python callbacks when enabled).
    /// When Some, memory operations try Rust first before falling back to callbacks.
    rust_memory: Option<SymbolicMemory>,
    /// Whether to use Rust-native memory (vs Python callbacks).
    /// When true and rust_memory is Some, memory ops use Rust directly.
    use_rust_memory: bool,
    /// When true, skip Z3 feasibility checks during deferred fork creation.
    /// This mirrors angr's LAZY_SOLVES option for binaries with expensive constraints.
    pub lazy_solves: bool,
    /// When true, do not enumerate symbolic jump targets — short-circuit the
    /// IP-concretization path to UnconstrainedJump (state goes to the
    /// unconstrained stash without warning). Mirrors angr's
    /// NO_IP_CONCRETIZATION option (engines/successors.py:292-296).
    pub no_ip_concretization: bool,
    /// When true, route any symbolic jump target to UnconstrainedJump before
    /// AddressConcretizer enumeration is attempted. Mirrors angr's
    /// NO_SYMBOLIC_JUMP_RESOLUTION option (engines/successors.py:234-239).
    /// Functionally identical to `no_ip_concretization` for the symbolic-IP
    /// case in Rust; kept as a separate flag so Python option semantics are
    /// preserved.
    pub no_symbolic_jump_resolution: bool,
    /// When true and the next pc was concretized from a symbolic expression,
    /// store that expression in `symbolic_ip_at_exit` so the manager can
    /// write it back to the IP register after the block (mirroring Python's
    /// `split_state.regs.ip = target` at engines/successors.py:328). Also
    /// suppresses the per-Single `assume_true(target == addr)` constraint
    /// that `eval_next_addr_concretized` would otherwise add. Mirrors angr's
    /// KEEP_IP_SYMBOLIC option.
    pub keep_ip_symbolic: bool,
    /// Populated by `eval_next_addr_concretized` when `keep_ip_symbolic` is
    /// set and the default exit's `next` expression was symbolic. The
    /// manager extracts this via `take_symbolic_ip_at_exit()` and writes it
    /// back to the state's IP register after `set_pc`. None otherwise.
    pub symbolic_ip_at_exit: Option<RustBV>,
    /// Number of pages to prefetch in each direction when fetching a page.
    /// 0 = no prefetching, 1 = fetch 3 pages (main + 1 before + 1 after), etc.
    /// Default is 2 for good locality on stack/heap access patterns.
    page_prefetch_count: u32,
    /// Dirty helper dispatch table for native handling of common helpers.
    dirty_dispatch: DirtyHelperDispatch,
    /// Registry mapping hook addresses to SimProcedure info.
    /// When a hook is hit, we can extract arguments using this info.
    /// Arc-shared on fork (O(1) clone). Mutators use Arc::make_mut for CoW.
    simprocedure_registry: Arc<FxHashMap<u64, SimProcedureInfo>>,
    /// Calling convention for argument extraction.
    calling_convention: Box<dyn CallingConvention>,
    /// Last branch condition encountered (for symbolic branch handling).
    /// Stored when a SymbolicBranch is created so callers can retrieve it.
    last_branch_condition: Option<RustBV>,
    /// Stored branch conditions by ID for deferred fork handling.
    /// When a deferred fork is created, we store the condition here so
    /// callers can retrieve it to properly constrain forked states.
    stored_conditions: FxHashMap<u64, RustBV>,
    /// Full state snapshots taken BEFORE branch constraints were added.
    /// Keyed by condition_id, these enable correct alternate-path forking
    /// with solver, registers, and memory from the branch point.
    fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    /// Execution statistics for profiling.
    stats: ExecutionStats,
    /// Whether profiling is enabled.
    profiling_enabled: bool,
    /// Concrete memory regions sorted by base address for binary search.
    /// This is rebuilt when regions are added.
    concrete_memory_sorted: bool,
    /// Per-block concretization cache.
    /// Maps BV id to cached ConcretizationResult.
    /// Cleared at the start of each block since constraints don't change within a block.
    concretize_cache: FxHashMap<u64, Arc<ConcretizationResult>>,
    /// Function call stack. Pushed on Ijk_Call, popped on Ijk_Ret.
    /// Transferred to/from RustSimState before/after interpreter runs.
    pub call_stack: Vec<crate::state::CallStackEntry>,
    /// Detailed execution history. Transferred to/from RustSimState.
    pub detailed_history: Vec<crate::state::HistoryEntry>,
    /// VEX optimization level (None = pyvex default).
    pub vex_opt_level: Option<i32>,
    /// Per-address VEX optimization level overrides.
    /// Arc-shared on fork (O(1) clone). Setters replace the Arc wholesale.
    pub vex_opt_level_overrides: Arc<FxHashMap<u64, i32>>,
    /// Page numbers (addr >> 12) inside loaded binary regions that have
    /// been overwritten by a store. Used to invalidate cached IRSBs and
    /// to redirect the Python lift callback to read fresh bytes from
    /// `rust_memory` instead of the original (now stale) binary image.
    dirtied_code_pages: FxHashSet<u64>,
    /// State id of the state currently being stepped. Forwarded to
    /// `PythonCallbacks::call_inspect_mem_*` so the Python dispatcher can
    /// build a `RustStateProxy` for the BP action. -1 means "unknown" —
    /// e.g. fresh interpreter from tests, or fork paths that never set it.
    pub current_state_id: i64,
}

// NOTE (angr-vh834 Phase 4): `VEXInterpreter` is intentionally NOT asserted
// `Send + Sync`. It borrows a `&SymContext` whose Z3 model/solver caches use
// `Cell`/`RefCell` interior mutability (non-`Sync`), so the interpreter cannot
// cross threads. That is by design: the work-stealing worker carries the
// `Send + Sync` `StepContext` (asserted in `exploration/step_core.rs`) across
// the thread boundary and constructs a fresh interpreter from it *inside* the
// worker, under `allow_threads`. With `py` no longer threaded through the step,
// the worker re-acquires the GIL only inside the self-attaching
// `PythonCallbacks` methods that actually fire.

impl<'a> VEXInterpreter<'a> {
    /// Create a new callback-aware interpreter.
    pub fn new(arch: VexArch, ctx: &'a SymContext) -> Self {
        Self::with_config(arch, ctx, ExecutionConfig::default())
    }

    /// Create a new callback-aware interpreter with custom config.
    pub fn with_config(arch: VexArch, ctx: &'a SymContext, config: ExecutionConfig) -> Self {
        let arch_box = arch_from_vex(arch);
        let arch_name = arch_box.name();
        let cc = default_cc_for_arch(arch_name);

        VEXInterpreter {
            registers: RegisterFile::new(arch_box),
            temps: Vec::with_capacity(64), // Pre-allocate for typical block size
            ctx,
            symbol_table: None, // Set via set_symbol_table() when using handles
            pc: 0,
            current_insn_addr: 0,
            current_insn_len: 0,
            hook_addrs: Arc::new(FxHashSet::default()),
            arch,
            #[cfg(feature = "libvex-ffi")]
            native_lifter: crate::vex::libvex_lifter::NativeLibVEXLifter,
            #[cfg(feature = "libvex-ffi")]
            native_lift_enabled: false,
            // Bounded default: this is the interpreter's own cache when it runs
            // standalone (engine.rs single-block exec, tests). It MUST stay
            // bounded/clone-safe because `fork()` clones it and lru's `Clone`
            // does `LruCache::new(self.cap())` — an `unbounded()` cap of
            // `usize::MAX` overflows `HashMap::with_capacity` (angr-4xaga.1).
            // In the exploration hot path this default is discarded when
            // run_interpreter_step_core swaps the shared cache in, but the two
            // *placeholder* caches there ARE zero-preallocation `unbounded()`.
            block_cache: LruCache::new(
                NonZeroUsize::new(BLOCK_CACHE_CAPACITY).expect("BLOCK_CACHE_CAPACITY is non-zero"),
            ),
            use_memory_callbacks: true,
            deferred_forks: Vec::new(),
            deferred_fork_this_step: false,
            block_solver_pushed: false,
            block_forks_asserted: 0,
            config,
            branch_counter: 0,
            next_condition_id: 0,
            push_level: 0,
            concrete_memory: Arc::new(Vec::new()),
            concretizer: AddressConcretizer::new(),
            dirty_registers: 0,
            pending_stores: PendingStoreBuffer::with_capacity(256),
            all_flushed_stores: FxHashMap::default(),
            all_flushed_symbolic_stores: FxHashMap::default(),
            pending_symbolic_stores: FxHashMap::default(),
            max_pending_stores: 256,
            rust_memory: None,
            use_rust_memory: false,
            lazy_solves: false,
            no_ip_concretization: false,
            no_symbolic_jump_resolution: false,
            keep_ip_symbolic: false,
            symbolic_ip_at_exit: None,
            page_prefetch_count: 2, // Prefetch 2 pages in each direction by default
            dirty_dispatch: DirtyHelperDispatch::new(),
            simprocedure_registry: Arc::new(FxHashMap::default()),
            calling_convention: cc,
            last_branch_condition: None,
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
            stats: ExecutionStats::default(),
            profiling_enabled: false,
            concrete_memory_sorted: false,
            concretize_cache: FxHashMap::default(),
            call_stack: Vec::new(),
            detailed_history: Vec::new(),
            vex_opt_level: None,
            vex_opt_level_overrides: Arc::new(FxHashMap::default()),
            dirtied_code_pages: FxHashSet::default(),
            current_state_id: -1,
        }
    }

    /// Set the symbol table for handle-based claripy bypass.
    ///
    /// When set, Python callbacks can return RustBVHandle instead of claripy ASTs,
    /// providing significant performance improvement by bypassing AST conversion.
    pub fn set_symbol_table(&mut self, table: &'a RustSymbolTable) {
        self.symbol_table = Some(table);
    }

    /// Enable or disable profiling.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling_enabled = enabled;
        if enabled {
            self.stats.reset();
        }
    }

    /// Get execution statistics.
    pub fn stats(&self) -> &ExecutionStats {
        &self.stats
    }

    /// Get mutable execution statistics.
    pub fn stats_mut(&mut self) -> &mut ExecutionStats {
        &mut self.stats
    }

    /// Take the execution statistics, replacing with default.
    pub fn take_stats(&mut self) -> ExecutionStats {
        std::mem::take(&mut self.stats)
    }

    /// Fall back to Python's full symbolic load callback for a symbolic
    /// address that we can't resolve to a useful concretization shape
    /// (TooLarge / Failed / Strided in callers that don't enumerate).
    ///
    /// Syncs pending constraints first, then passes the address AST to
    /// Python's memory model. The returned AST is converted back to a
    /// `RustBV` via the handle table (fast path) or claripy bridge.
    ///
    /// `context` is a short label included in the Unsupported error when
    /// the callback isn't wired up — e.g. "Load", "LoadG", "store".
    pub(super) fn fallback_load_symbolic_full(
        &self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        size: usize,
        context: &str,
        addr_descr: &str,
    ) -> Result<RustBV, CbExecutionError> {
        if !callbacks.has_memory_load_symbolic_full() {
            return Err(CbExecutionError::Unsupported(format!(
                "{context} with symbolic address ({addr_descr}): no memory_load_symbolic_full callback"
            )));
        }

        let result_ast = callbacks
            .call_memory_load_symbolic_full(addr_val, size as u32)
            .map_err(|e| {
                CbExecutionError::Callback(format!(
                    "{context} symbolic load full callback failed ({addr_descr}): {e}"
                ))
            })?;

        // handle fast path -> claripy slow path -> fresh symbolic. (The prior
        // open-coded copy logged a warn! on claripy-conversion failure; that
        // diagnostic is dropped in favor of the shared ladder.)
        Ok(
            self.try_convert_symbolic_value(Some(&result_ast), (size * 8) as u32, || {
                format!("sym_pyref_{addr_descr}_{size}")
            }),
        )
    }

    /// Fall back to Python's full symbolic store callback for a store
    /// where the address concretization didn't produce a usable shape.
    ///
    /// Syncs pending constraints first, then hands the address AST and
    /// data to Python's memory model.
    ///
    /// `context` is a short label included in the Unsupported error when
    /// the callback isn't wired up.
    pub(super) fn fallback_store_symbolic_full(
        &self,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        data_val: &RustBV,
        context: &str,
        addr_descr: &str,
    ) -> Result<(), CbExecutionError> {
        if !callbacks.has_memory_store_symbolic_full() {
            return Err(CbExecutionError::Unsupported(format!(
                "{context} with symbolic address ({addr_descr}): no memory_store_symbolic_full callback"
            )));
        }

        callbacks
            .call_memory_store_symbolic_full(addr_val, data_val)
            .map_err(|e| {
                CbExecutionError::Callback(format!(
                    "{context} symbolic store full callback failed ({addr_descr}): {e}"
                ))
            })?;
        Ok(())
    }

    /// Serve a load neither side can back with real bytes, without crossing
    /// the GIL (angr-gorvf.4.7).
    ///
    /// A load only reaches `load_from_callback` once Rust's own memory has
    /// declined it. If Python *also* has no page there, the only answer it
    /// could give is an unconstrained filler (angr's `filler_mixin` default) —
    /// so Rust mints that filler itself and skips the crossing. Measured: this
    /// is 100% of the `memory_load` crossings on the five ZeroPy benches whose
    /// sole residual GIL was this site — all of them the `fs:[0x28]` stack
    /// canary or another unmapped low address.
    ///
    /// The oracle is `python_has_page`, NOT the `python_can_serve_page` set
    /// from angr-gorvf.4.6 — those answer different questions, and using the
    /// latter here is a silent-corruption bug (it briefly was one). A page
    /// holding symbolic bytes is *declined* for a whole-page concrete fetch
    /// yet serves a load fine, so gating on fetch-servability synthesizes
    /// fillers over real data.
    ///
    /// `python_has_page` fails *open* in exactly the cases that would make the
    /// filler wrong: `_install_python_servable_pages` declines to install a
    /// snapshot at all under `ZERO_FILL_UNCONSTRAINED_MEMORY`, where Python
    /// answers with zeros rather than a symbol. With no snapshot every load
    /// crosses, as before.
    ///
    /// The symbol name is the address-derived `mem_{addr}_{size}` already used
    /// by the fresh-symbol fallback below, so repeated loads of the same
    /// address mint the *same* Z3 constant. The canary depends on this: it is
    /// read twice (store, then compare), and two distinct symbols would make
    /// the `__stack_chk_fail` branch spuriously feasible.
    fn synthesize_unservable_load(
        &self,
        callbacks: &PythonCallbacks,
        addr_concrete: u64,
        size: usize,
    ) -> Option<RustBV> {
        let page_addr = addr_concrete & !(crate::memory::PAGE_SIZE - 1);
        if callbacks.python_has_page(page_addr) {
            return None;
        }
        // Rust having the page mapped means the bytes are real and the decline
        // came from elsewhere in the load path — don't paper over that with a
        // filler.
        if let Some(rust_mem) = self.rust_memory.as_ref()
            && rust_mem.is_mapped(page_addr)
        {
            return None;
        }

        let name = format!("mem_{addr_concrete:x}_{size}");
        let bits = (size * 8) as u32;
        let bv = RustBV::symbolic(self.ctx, &name, bits);
        self.dispatch_symbolic_variable_inspect(callbacks, &name, bits, &bv);
        Some(bv)
    }

    /// Load from memory via Python callback.
    /// This handles the common case of loading from a concrete address.
    fn load_from_callback(
        &self,
        callbacks: &PythonCallbacks,
        addr_concrete: u64,
        size: usize,
    ) -> Result<RustBV, CbExecutionError> {
        if let Some(bv) = self.synthesize_unservable_load(callbacks, addr_concrete, size) {
            return Ok(bv);
        }

        let (data, is_symbolic, symbolic_ast) = callbacks
            .call_memory_load(addr_concrete, size as u32)
            .map_err(|e| CbExecutionError::Callback(e.to_string()))?;

        if is_symbolic {
            // Try to convert to RustBV - check handle first (fast path), then claripy (slow path).
            // Self-attach for the bridge work (angr-vh834 Phase 4): no caller
            // GIL token is threaded in; re-entrant no-op when already held.
            if let Some(ast_obj) = symbolic_ast {
                let converted: Option<RustBV> = Python::attach(|py| {
                    let ast = ast_obj.bind(py);

                    // Fast path: check for RustBVHandle first
                    if let Some(table) = self.symbol_table
                        && let Some(bv) = try_handle_to_rustbv(ast, table)
                    {
                        return Some(bv);
                    }

                    // Slow path: claripy AST conversion (can fail for
                    // complex/unsupported ops -> fall through to fresh symbolic).
                    if is_claripy_ast(ast)
                        && let Ok(bv) = claripy_to_rustbv(py, ast, self.ctx)
                    {
                        return Some(bv);
                    }
                    None
                });
                if let Some(bv) = converted {
                    return Ok(bv);
                }
            }
            // Fallback: create a fresh symbolic value
            let name = format!("mem_{addr_concrete:x}_{size}");
            let bits = (size * 8) as u32;
            let bv = RustBV::symbolic(self.ctx, &name, bits);
            // angr-vfst: symbolic_variable BP_AFTER for engine-internal fresh
            // BVS minting. Mirrors Python's `solver.py:432-439` BP_AFTER
            // signature. The user-callable `state.solver.BVS()` path still
            // fires the same event from Python directly.
            self.dispatch_symbolic_variable_inspect(callbacks, &name, bits, &bv);
            Ok(bv)
        } else {
            // Convert bytes to concrete value
            Ok(bytes_to_bv(&data, (size * 8) as u32))
        }
    }

    /// Get the execution configuration.
    pub fn config(&self) -> &ExecutionConfig {
        &self.config
    }

    /// Set the execution configuration.
    pub fn set_config(&mut self, config: ExecutionConfig) {
        self.config = config;
    }

    /// Get the next condition ID.
    fn next_cond_id(&mut self) -> u64 {
        let id = self.next_condition_id;
        self.next_condition_id += 1;
        id
    }

    /// Get the current solver push level.
    pub fn push_level(&self) -> u32 {
        self.push_level
    }

    /// Get the solver context.
    pub fn context(&self) -> &SymContext {
        self.ctx
    }

    /// Get list of dirty register offsets (registers modified since last clear).
    /// Returns offsets in 4-byte granularity.
    pub fn get_dirty_register_offsets(&self) -> Vec<u32> {
        let mut offsets = Vec::new();
        for bit in 0..128u32 {
            if (self.dirty_registers & (1u128 << bit)) != 0 {
                offsets.push(bit * 4);
            }
        }
        offsets
    }

    /// Get the raw dirty register bitset.
    pub fn dirty_registers(&self) -> u128 {
        self.dirty_registers
    }

    /// Clear dirty register tracking (called after sync).
    pub fn clear_dirty_registers(&mut self) {
        self.dirty_registers = 0;
    }

    /// Mark the 4-byte register slot containing `offset` as dirty.
    /// Bitset is 128 bits wide (covers offsets [0, 512)); writes beyond
    /// that fall back to the always-sync slow path.
    #[inline]
    pub(crate) fn mark_register_dirty(&mut self, offset: u32) {
        let bit_index = offset / 4;
        if bit_index < 128 {
            self.dirty_registers |= 1u128 << bit_index;
        }
    }

    /// Flush pending stores to Python via batch callback.
    ///
    /// This sends all buffered stores in a single callback, reducing
    /// FFI overhead compared to individual store callbacks.
    fn flush_stores(&mut self, callbacks: &PythonCallbacks) -> Result<(), CbExecutionError> {
        if self.pending_stores.is_empty() && self.pending_symbolic_stores.is_empty() {
            return Ok(());
        }

        if self.use_rust_memory {
            // When Rust owns memory, flush stores to rust_memory instead of Python.
            if let Some(ref mut rust_mem) = self.rust_memory {
                for (addr, data) in self.pending_stores.iter() {
                    if let Err(e) = rust_mem.store_concrete_le_bytes_automap_internal(*addr, data) {
                        log::error!("dropped concrete store at {addr:#x}: {e:?}");
                    }
                }
                // Also flush symbolic stores to rust_memory so that subsequent
                // loads via load_concrete_lazy_inner find the symbolic values
                // instead of returning concrete zeros from the page fill.
                //
                // Drain in this loop and insert into all_flushed_symbolic_stores too,
                // sharing one clone instead of doing iter+clone followed by drain.
                for (addr, bv) in self.pending_symbolic_stores.drain() {
                    rust_mem.import_symbolic_value(addr, bv.clone(), None);
                    self.all_flushed_symbolic_stores.insert(addr, bv);
                }
            } else {
                for (addr, bv) in self.pending_symbolic_stores.drain() {
                    self.all_flushed_symbolic_stores.insert(addr, bv);
                }
            }
            // Drain pending_stores by move into all_flushed_stores (no Vec<u8> clone).
            for (addr, data) in self.pending_stores.drain() {
                self.all_flushed_stores.insert(addr, data);
            }
            return Ok(());
        }

        // Python callback path (when Rust memory is not used)
        callbacks
            .call_memory_store_batch(self.pending_stores.as_slice())
            .map_err(|e| CbExecutionError::Callback(e.to_string()))?;

        // Drain pending_stores into all_flushed_stores by move (no Vec<u8> clone).
        for (addr, data) in self.pending_stores.drain() {
            self.all_flushed_stores.insert(addr, data);
        }
        // Preserve symbolic values across block boundaries
        for (addr, bv) in self.pending_symbolic_stores.drain() {
            self.all_flushed_symbolic_stores.insert(addr, bv);
        }
        Ok(())
    }

    /// Flush all pending stores into rust_memory.
    /// Called before extracting rust_memory back to the state.
    pub fn flush_stores_to_rust_memory(&mut self) {
        if let Some(ref mut rust_mem) = self.rust_memory {
            // Flush concrete pending stores
            for (addr, data) in self.pending_stores.drain() {
                if let Err(e) = rust_mem.store_concrete_le_bytes_automap_internal(addr, &data) {
                    log::error!("dropped concrete store at {addr:#x}: {e:?}");
                }
            }
            // Flush symbolic pending stores
            for (addr, bv) in self.pending_symbolic_stores.drain() {
                rust_mem.import_symbolic_value(addr, bv, None);
            }
            self.all_flushed_stores.clear();
            self.all_flushed_symbolic_stores.clear();
        }
    }

    /// Take all stores from this step (both pending and previously flushed).
    pub fn take_all_stores(&mut self) -> Vec<(u64, Vec<u8>)> {
        self.pending_symbolic_stores.clear();
        self.all_flushed_symbolic_stores.clear();
        // Merge pending into flushed
        for (addr, data) in self.pending_stores.drain() {
            self.all_flushed_stores.insert(addr, data);
        }
        std::mem::take(&mut self.all_flushed_stores)
            .into_iter()
            .collect()
    }

    /// Set the program counter.
    pub fn set_pc(&mut self, addr: u64) {
        self.pc = addr;
        let pc_bv = RustBV::concrete(addr as u128, self.registers.arch().bits());
        self.registers.set_ip(pc_bv);
    }

    /// Get the program counter.
    #[inline]
    pub fn get_pc(&self) -> u64 {
        self.pc
    }

    /// Add a hook address.
    pub fn add_hook(&mut self, addr: u64) {
        Arc::make_mut(&mut self.hook_addrs).insert(addr);
    }

    /// Remove a hook address.
    pub fn remove_hook(&mut self, addr: u64) {
        Arc::make_mut(&mut self.hook_addrs).remove(&addr);
    }

    /// Add multiple hooks at once.
    pub fn add_hooks(&mut self, addrs: &[u64]) {
        let hooks = Arc::make_mut(&mut self.hook_addrs);
        for &addr in addrs {
            hooks.insert(addr);
        }
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        Arc::make_mut(&mut self.hook_addrs).clear();
    }

    /// Check if an address is hooked.
    pub fn is_hooked(&self, addr: u64) -> bool {
        self.hook_addrs.contains(&addr)
    }

    /// Check if we have a cached block at the given address.
    pub fn has_cached_block(&self, addr: u64) -> bool {
        self.block_cache.contains(&addr)
    }

    /// Add a block to the cache.
    pub fn cache_block(&mut self, addr: u64, irsb: IRSB) {
        self.block_cache.put(addr, Arc::new(irsb));
    }

    /// Get a block from the cache.
    pub fn get_cached_block(&mut self, addr: u64) -> Option<&IRSB> {
        self.block_cache.get(&addr).map(std::convert::AsRef::as_ref)
    }

    /// Swap in a shared block cache, returning the interpreter's current cache.
    pub fn swap_block_cache(
        &mut self,
        cache: LruCache<u64, Arc<IRSB>>,
    ) -> LruCache<u64, Arc<IRSB>> {
        std::mem::replace(&mut self.block_cache, cache)
    }

    /// Take the symbolic IP expression recorded at the most recent default
    /// exit (set only when `keep_ip_symbolic` is enabled and the exit's `next`
    /// was symbolic). The internal slot is cleared. Mirrors Python's
    /// `split_state.regs.ip = target` write at engines/successors.py:328.
    pub fn take_symbolic_ip_at_exit(&mut self) -> Option<RustBV> {
        self.symbolic_ip_at_exit.take()
    }

    /// Fork the interpreter state.
    pub fn fork(&self) -> VEXInterpreter<'a> {
        // Clone the calling convention based on its type
        let arch_box = arch_from_vex(self.arch);
        let cc = default_cc_for_arch(arch_box.name());

        VEXInterpreter {
            registers: self.registers.fork(),
            temps: self.temps.clone(),
            ctx: self.ctx,
            symbol_table: self.symbol_table, // Share symbol table reference
            pc: self.pc,
            current_insn_addr: self.current_insn_addr,
            current_insn_len: self.current_insn_len,
            hook_addrs: Arc::clone(&self.hook_addrs),
            arch: self.arch,
            #[cfg(feature = "libvex-ffi")]
            native_lifter: crate::vex::libvex_lifter::NativeLibVEXLifter,
            #[cfg(feature = "libvex-ffi")]
            native_lift_enabled: self.native_lift_enabled,
            block_cache: self.block_cache.clone(), // Share lifted blocks with parent (Arc values = cheap clone)
            use_memory_callbacks: self.use_memory_callbacks,
            deferred_forks: Vec::new(), // Fresh deferred forks for fork
            deferred_fork_this_step: false,
            block_solver_pushed: false,
            block_forks_asserted: 0,
            config: self.config.clone(),
            branch_counter: self.branch_counter,
            next_condition_id: self.next_condition_id,
            push_level: self.push_level, // Inherit push level for forked interpreter
            concrete_memory: Arc::clone(&self.concrete_memory), // Share concrete memory (read-only)
            concretizer: self.concretizer.clone(), // Share concretizer settings
            dirty_registers: 0,          // Fresh dirty tracking for fork
            pending_stores: PendingStoreBuffer::with_capacity(256), // Fresh store buffer for fork
            all_flushed_stores: FxHashMap::default(),
            all_flushed_symbolic_stores: FxHashMap::default(),
            pending_symbolic_stores: FxHashMap::default(),
            max_pending_stores: self.max_pending_stores,
            // Fork Rust memory with O(1) CoW
            rust_memory: self
                .rust_memory
                .as_ref()
                .map(super::memory::SymbolicMemory::fork),
            use_rust_memory: self.use_rust_memory,
            lazy_solves: self.lazy_solves,
            no_ip_concretization: self.no_ip_concretization,
            no_symbolic_jump_resolution: self.no_symbolic_jump_resolution,
            keep_ip_symbolic: self.keep_ip_symbolic,
            symbolic_ip_at_exit: None,
            page_prefetch_count: self.page_prefetch_count, // Inherit page prefetch count
            dirty_dispatch: DirtyHelperDispatch::new(),    // Fresh dispatch (stateless)
            simprocedure_registry: Arc::clone(&self.simprocedure_registry), // Share SimProcedure registry
            calling_convention: cc,
            last_branch_condition: None,               // Fresh for fork
            stored_conditions: FxHashMap::default(),   // Fresh for fork
            fork_snapshots: FxHashMap::default(),      // Fresh for fork
            stats: ExecutionStats::default(),          // Fresh stats for fork
            profiling_enabled: self.profiling_enabled, // Inherit profiling setting
            concrete_memory_sorted: self.concrete_memory_sorted, // Inherit sorted flag
            concretize_cache: FxHashMap::default(),    // Fresh cache for fork
            call_stack: self.call_stack.clone(),       // Clone call stack for fork
            detailed_history: self.detailed_history.clone(), // Clone history for fork
            vex_opt_level: self.vex_opt_level,         // Inherit VEX opt level
            vex_opt_level_overrides: Arc::clone(&self.vex_opt_level_overrides), // Inherit overrides
            dirtied_code_pages: self.dirtied_code_pages.clone(), // Inherit SMC tracking
            current_state_id: self.current_state_id,
        }
    }

    /// Enable Rust-native memory mode.
    ///
    /// When enabled, memory operations will try to use the Rust SymbolicMemory
    /// first, falling back to Python callbacks only for unmapped regions.
    /// This can significantly improve performance for memory-intensive code.
    pub fn enable_rust_memory(&mut self, endness: Endness) {
        self.rust_memory = Some(SymbolicMemory::new(endness));
        self.use_rust_memory = true;
    }

    /// Enable or disable native (in-process) libVEX cold-block lifting.
    ///
    /// When enabled, [`Self::get_or_lift_block`] attempts an FFI lift via
    /// `NativeLibVEXLifter` before the Python `lift_block` callback, falling
    /// back cleanly on any miss (unsupported arch, missing/symbolic bytes, or
    /// a libVEX error). Requires the `libvex-ffi` build feature; a no-op stub
    /// exists for the default build so Python wiring compiles unconditionally.
    #[cfg(feature = "libvex-ffi")]
    pub fn set_native_lift_enabled(&mut self, enabled: bool) {
        self.native_lift_enabled = enabled;
    }

    /// No-op stub when the `libvex-ffi` feature is not compiled in.
    #[cfg(not(feature = "libvex-ffi"))]
    pub fn set_native_lift_enabled(&mut self, _enabled: bool) {}

    /// Disable Rust-native memory mode.
    pub fn disable_rust_memory(&mut self) {
        self.use_rust_memory = false;
    }

    /// Get mutable reference to Rust memory (for initialization).
    pub fn rust_memory_mut(&mut self) -> Option<&mut SymbolicMemory> {
        self.rust_memory.as_mut()
    }

    /// Get reference to Rust memory.
    pub fn rust_memory(&self) -> Option<&SymbolicMemory> {
        self.rust_memory.as_ref()
    }

    /// Set the Rust memory instance.
    pub fn set_rust_memory(&mut self, memory: SymbolicMemory) {
        self.rust_memory = Some(memory);
        self.use_rust_memory = true;
    }

    /// Take the Rust memory instance (for transferring to engine).
    pub fn take_rust_memory(&mut self) -> Option<SymbolicMemory> {
        self.rust_memory.take()
    }

    pub fn add_lazy_region(&mut self, start_addr: u64, size: u64) {
        if let Some(ref mut rust_mem) = self.rust_memory {
            rust_mem.add_lazy_region(start_addr, size);
        }
    }

    /// Get statistics about Rust memory usage.
    pub fn rust_memory_stats(&self) -> Option<(usize, usize, usize)> {
        self.rust_memory.as_ref().map(|m| {
            (
                m.page_count(),
                m.lazy_region_count(),
                m.get_dirty_pages().len(),
            )
        })
    }
}

#[cfg(test)]
#[path = "smc_tests.rs"]
mod smc_tests;
