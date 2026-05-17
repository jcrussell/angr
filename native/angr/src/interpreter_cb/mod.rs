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
use crate::callbacks::{DeferredFork, ExecutionConfig, PythonCallbacks, RunResult};
use crate::claripy_bridge::{claripy_to_rustbv, is_claripy_ast, try_handle_to_rustbv};
use crate::concretize::{AddressConcretizer, ConcretizationResult};
use crate::memory::{MemoryError, Permission, SymbolicMemory};
use crate::symbolic::{BVOp, RustBV, RustSymbolTable, SymContext};
use crate::vex::ccall;
use crate::vex::dirty::DirtyHelperDispatch;
use crate::vex::ir::{
    IRConst, IRExpr, IRLoadGOp, IROp, IRSB, IRStmt, IRType, JumpKind, TypeEnv, VexArch,
};
use crate::vex::ops::{OpError, VEXOps};
use crate::vex::{Endness, deserialize_irsb};

mod constraints;
mod execution;
mod exits;
mod expressions;
mod helpers;
mod pending_store;
mod prefetch;
mod statements;

use helpers::bytes_to_bv;
use pending_store::PendingStoreBuffer;

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
    /// Time spent in prefetch loads per block (nanoseconds).
    prefetch_time_ns: sum,
    /// Number of statements executed.
    stmt_count: sum,
    /// Number of deferred forks processed in exploration loop.
    deferred_fork_count: sum,
    /// Time spent processing deferred forks (nanoseconds).
    deferred_fork_time_ns: sum,
    /// Time spent in solver fork/clone operations (nanoseconds).
    solver_fork_time_ns: sum,
    /// Number of solver fork operations.
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
}

/// Reason string used by the CAS handler when it sees a double-CAS (cmpxchg16b).
/// Shared with `exploration::mod` so the manager can identify DCAS in
/// `PythonVEXFallback` events and bump a dedicated visibility counter.
pub const DCAS_UNSUPPORTED_REASON: &str = "double compare-and-swap";

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
    Panic,
    /// Reserved: no current `CbExecutionError` variant uses this. The
    /// interpreter does have *non-error* silent substitution paths (e.g.
    /// the `or_else` fallbacks for unsupported binops in
    /// `expressions.rs`); those return `Ok(...)` and never reach the
    /// strategy dispatcher. This variant exists so future variants can
    /// opt into a "log and synthesize a sound default" policy explicitly
    /// instead of silently swallowing.
    #[allow(dead_code)]
    Silent,
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
    /// Type mismatch. Strategy: [`FallbackStrategy::Panic`].
    #[error("type mismatch: expected {expected:?}, got {got:?}")]
    TypeMismatch { expected: IRType, got: IRType },
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
            | CbExecutionError::TypeMismatch { .. }
            | CbExecutionError::UnknownTemp(_)
            | CbExecutionError::Callback(_)
            | CbExecutionError::LiftError(_) => FallbackStrategy::Panic,
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
    /// Syscall encountered.
    Syscall { num: u64 },
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

/// A concrete memory region cached locally in Rust.
/// Uses Arc<Vec<u8>> for O(1) cloning — binary data is shared, not copied.
#[derive(Clone)]
pub struct ConcreteMemoryRegion {
    /// Base address of the region.
    pub base: u64,
    /// Size of the region in bytes.
    pub size: u64,
    /// The concrete data (shared via Arc to avoid copying per step).
    pub data: Arc<Vec<u8>>,
}

impl ConcreteMemoryRegion {
    /// Check if this region contains the given address range.
    #[inline]
    pub fn contains(&self, addr: u64, size: u64) -> bool {
        addr >= self.base && addr + size <= self.base + self.size
    }

    /// Read bytes from this region. Returns None if out of bounds.
    #[inline]
    pub fn read(&self, addr: u64, size: usize) -> Option<&[u8]> {
        if addr < self.base {
            return None;
        }
        let offset = (addr - self.base) as usize;
        if offset + size > self.data.len() {
            return None;
        }
        Some(&self.data[offset..offset + size])
    }
}

/// Cached result of a prefetched memory load.
#[derive(Clone)]
pub struct PrefetchedLoad {
    /// The loaded bitvector value.
    pub value: RustBV,
    /// Whether the value is symbolic.
    pub is_symbolic: bool,
}

/// A constraint that was added in Rust and needs to be synced to Python.
///
/// When Rust concretizes a symbolic address or makes a branch decision,
/// it adds constraints to its Z3 context. These constraints must be
/// communicated to Python's claripy solver to maintain consistency
/// when falling back to Python for complex operations.
#[derive(Clone, Debug)]
pub struct PendingConstraint {
    /// The symbolic expression that was constrained.
    /// For address concretization: the address expression
    /// For branch: the condition
    pub expression: RustBV,
    /// The concrete value it was constrained to.
    pub concrete_value: u128,
    /// Description for debugging.
    pub description: String,
    /// Handle ID for looking up the original claripy AST in Python.
    /// If the expression came from a Python callback that returned a handle,
    /// this ID can be used to look up the original AST for constraint sync.
    pub handle_id: Option<u64>,
}

impl PendingConstraint {
    /// Create a new pending constraint for address concretization.
    pub fn address_concretization(addr_expr: RustBV, concrete_addr: u64) -> Self {
        PendingConstraint {
            expression: addr_expr,
            concrete_value: concrete_addr as u128,
            description: format!("addr_concretize_0x{:x}", concrete_addr),
            handle_id: None,
        }
    }

    /// Create a new pending constraint for address concretization with handle_id.
    pub fn address_concretization_with_handle(
        addr_expr: RustBV,
        concrete_addr: u64,
        handle_id: Option<u64>,
    ) -> Self {
        PendingConstraint {
            expression: addr_expr,
            concrete_value: concrete_addr as u128,
            description: format!("addr_concretize_0x{:x}", concrete_addr),
            handle_id,
        }
    }

    /// Create a new pending constraint for a branch taken (cond == 1).
    pub fn branch_true(cond: RustBV) -> Self {
        PendingConstraint {
            expression: cond,
            concrete_value: 1,
            description: "branch_true".to_string(),
            handle_id: None,
        }
    }

    /// Create a new pending constraint for a branch not taken (cond == 0).
    pub fn branch_false(cond: RustBV) -> Self {
        PendingConstraint {
            expression: cond,
            concrete_value: 0,
            description: "branch_false".to_string(),
            handle_id: None,
        }
    }
}

/// Information about a registered SimProcedure.
#[derive(Clone, Debug)]
pub struct SimProcedureInfo {
    /// Name of the SimProcedure (e.g., "strlen", "malloc").
    pub name: String,
    /// Number of arguments to extract.
    pub num_args: usize,
    /// Whether this is a no-return procedure (e.g., "exit", "abort").
    pub no_return: bool,
}

/// Callback-aware VEX IR interpreter.
///
/// This interpreter uses Python callbacks for memory and register access,
/// allowing it to work with angr's symbolic memory model.
///
/// When `use_rust_memory` is true, the interpreter uses `rust_memory` for
/// memory operations, falling back to Python callbacks only for unmapped pages.
/// This provides significant performance improvement for memory-intensive code.
pub struct CallbackInterpreter<'a> {
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
    /// Prefetch cache for batched memory loads.
    /// Key is (address, size), value is the prefetched result.
    /// This is populated at block start and used during Load expression evaluation.
    load_prefetch_cache: FxHashMap<(u64, usize), PrefetchedLoad>,
    /// Whether load prefetching is enabled.
    use_load_prefetch: bool,
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
    /// Pending constraints that need to be synced to Python.
    /// These accumulate when Rust adds constraints (e.g., address concretization)
    /// and are synced to Python before falling back to Python callbacks.
    pending_python_constraints: Vec<PendingConstraint>,
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
    /// Scratch buffers reused across `prefetch_loads_for_block` calls to avoid
    /// per-block allocator churn. Each is cleared (not reallocated) at block start.
    prefetch_loads_scratch: Vec<(u64, usize)>,
    prefetch_unique_scratch: Vec<(u64, usize)>,
    prefetch_dedup_scratch: HashSet<(u64, usize)>,
    prefetch_callback_scratch: Vec<(u64, u32)>,
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
    /// skip the native-lift fast path (which reads from immutable
    /// `concrete_memory` and would otherwise use stale bytes).
    dirtied_code_pages: FxHashSet<u64>,
    /// State id of the state currently being stepped. Forwarded to
    /// `PythonCallbacks::call_inspect_mem_*` so the Python dispatcher can
    /// build a `RustStateProxy` for the BP action. -1 means "unknown" —
    /// e.g. fresh interpreter from tests, or fork paths that never set it.
    pub current_state_id: i64,
}
impl<'a> CallbackInterpreter<'a> {
    /// Create a new callback-aware interpreter.
    pub fn new(arch: VexArch, ctx: &'a SymContext) -> Self {
        Self::with_config(arch, ctx, ExecutionConfig::default())
    }

    /// Create a new callback-aware interpreter with custom config.
    pub fn with_config(arch: VexArch, ctx: &'a SymContext, config: ExecutionConfig) -> Self {
        let arch_box = arch_from_vex(arch);
        let arch_name = arch_box.name();
        let cc = default_cc_for_arch(arch_name);

        CallbackInterpreter {
            registers: RegisterFile::new(arch_box),
            temps: Vec::with_capacity(64), // Pre-allocate for typical block size
            ctx,
            symbol_table: None, // Set via set_symbol_table() when using handles
            pc: 0,
            current_insn_addr: 0,
            current_insn_len: 0,
            hook_addrs: Arc::new(FxHashSet::default()),
            arch,
            block_cache: LruCache::new(NonZeroUsize::new(4096).expect("nonzero literal")),
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
            load_prefetch_cache: FxHashMap::default(),
            use_load_prefetch: false, // Disabled by default - adds overhead for most workloads
            page_prefetch_count: 2,   // Prefetch 2 pages in each direction by default
            dirty_dispatch: DirtyHelperDispatch::new(),
            simprocedure_registry: Arc::new(FxHashMap::default()),
            calling_convention: cc,
            last_branch_condition: None,
            pending_python_constraints: Vec::new(),
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
            stats: ExecutionStats::default(),
            profiling_enabled: false,
            concrete_memory_sorted: false,
            concretize_cache: FxHashMap::default(),
            prefetch_loads_scratch: Vec::new(),
            prefetch_unique_scratch: Vec::new(),
            prefetch_dedup_scratch: HashSet::new(),
            prefetch_callback_scratch: Vec::new(),
            call_stack: Vec::new(),
            detailed_history: Vec::new(),
            vex_opt_level: None,
            vex_opt_level_overrides: Arc::new(FxHashMap::default()),
            dirtied_code_pages: FxHashSet::default(),
            current_state_id: -1,
        }
    }

    /// Get the address concretizer.
    pub fn concretizer(&self) -> &AddressConcretizer {
        &self.concretizer
    }

    /// Set custom concretizer settings.
    pub fn set_concretizer(&mut self, concretizer: AddressConcretizer) {
        self.concretizer = concretizer;
    }

    /// Concretize for read with per-block caching.
    /// Uses read_range_limit and falls back to Any (single solution) if range is too large.
    ///
    /// Returns an `Arc<ConcretizationResult>` so cache hits / inserts only pay an
    /// atomic refcount bump rather than cloning a `Vec<u64>` for the Multiple variant.
    fn concretize_cached_read(&mut self, addr: &RustBV) -> Arc<ConcretizationResult> {
        if let Some(concrete_addr) = addr.as_u64() {
            return Arc::new(ConcretizationResult::Single(concrete_addr));
        }

        let cache_key = Self::bv_cache_key(addr);
        // Note: read and write may produce different results for same address,
        // but within a block they're typically used consistently for a given address.
        // Cache the raw result and apply fallback after cache lookup.
        if let Some(cached) = self.concretize_cache.get(&cache_key) {
            let result = Arc::clone(cached);
            // Apply read fallback to cached result
            return match &*result {
                ConcretizationResult::TooLarge { .. } if self.concretizer.read_fallback_any => {
                    if let Some(val) = self.ctx.eval(addr) {
                        Arc::new(ConcretizationResult::Single(val as u64))
                    } else {
                        result
                    }
                }
                _ => result,
            };
        }

        let conc_start = std::time::Instant::now();
        let result = Arc::new(self.concretizer.concretize_read(addr, self.ctx));
        let conc_elapsed = conc_start.elapsed();
        if self.profiling_enabled {
            self.stats.concretize_count += 1;
            self.stats.concretize_time_ns += conc_elapsed.as_nanos() as u64;
        }
        self.concretize_cache.insert(cache_key, Arc::clone(&result));
        result
    }

    /// Concretize for write with per-block caching.
    /// Uses write_range_limit and falls back to Max solution if range is too large.
    ///
    /// Returns an `Arc<ConcretizationResult>` (see `concretize_cached_read`).
    fn concretize_cached_write(&mut self, addr: &RustBV) -> Arc<ConcretizationResult> {
        if let Some(concrete_addr) = addr.as_u64() {
            return Arc::new(ConcretizationResult::Single(concrete_addr));
        }

        let cache_key = Self::bv_cache_key(addr);
        if let Some(cached) = self.concretize_cache.get(&cache_key) {
            let result = Arc::clone(cached);
            // Apply write fallback to cached result
            return match &*result {
                ConcretizationResult::TooLarge { .. } if self.concretizer.write_fallback_max => {
                    if let Some((_min, max)) = self.ctx.range(addr) {
                        Arc::new(ConcretizationResult::Single(max as u64))
                    } else if let Some(val) = self.ctx.eval(addr) {
                        Arc::new(ConcretizationResult::Single(val as u64))
                    } else {
                        result
                    }
                }
                _ => result,
            };
        }

        let conc_start = std::time::Instant::now();
        let result = Arc::new(self.concretizer.concretize_write(addr, self.ctx));
        let conc_elapsed = conc_start.elapsed();
        if self.profiling_enabled {
            self.stats.concretize_count += 1;
            self.stats.concretize_time_ns += conc_elapsed.as_nanos() as u64;
        }
        self.concretize_cache.insert(cache_key, Arc::clone(&result));
        result
    }

    /// Compute a cache key for a RustBV value.
    /// Uses the symbolic id for Symbolic/Constrained, and a hash of op+operand structure for Expression.
    fn bv_cache_key(bv: &RustBV) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        match bv {
            RustBV::Concrete { value, .. } => *value as u64,
            RustBV::Symbolic { id, .. } => *id,
            RustBV::Constrained { id, .. } => *id,
            RustBV::Expression {
                op,
                operands,
                width,
                ..
            } => {
                let mut hasher = DefaultHasher::new();
                // Hash op discriminant + width + operand keys recursively (1 level deep)
                std::mem::discriminant(op).hash(&mut hasher);
                width.hash(&mut hasher);
                for operand in operands.iter() {
                    match operand {
                        RustBV::Concrete { value, .. } => {
                            value.hash(&mut hasher);
                        }
                        RustBV::Symbolic { id, .. } => {
                            id.hash(&mut hasher);
                        }
                        RustBV::Constrained { id, .. } => {
                            id.hash(&mut hasher);
                        }
                        RustBV::Expression {
                            op: sub_op,
                            width: sub_w,
                            ..
                        } => {
                            std::mem::discriminant(sub_op).hash(&mut hasher);
                            sub_w.hash(&mut hasher);
                        }
                    }
                }
                hasher.finish()
            }
        }
    }

    /// Set the symbol table for handle-based claripy bypass.
    ///
    /// When set, Python callbacks can return RustBVHandle instead of claripy ASTs,
    /// providing significant performance improvement by bypassing AST conversion.
    pub fn set_symbol_table(&mut self, table: &'a RustSymbolTable) {
        self.symbol_table = Some(table);
    }

    /// Add a concrete memory region for fast local access.
    ///
    /// This allows the interpreter to read from binary sections (e.g., .text, .rodata)
    /// without going through Python callbacks, significantly improving performance.
    pub fn add_concrete_memory(&mut self, base: u64, data: Vec<u8>) {
        let size = data.len() as u64;
        Arc::make_mut(&mut self.concrete_memory).push(ConcreteMemoryRegion {
            base,
            size,
            data: Arc::new(data),
        });
        self.concrete_memory_sorted = false;
    }

    /// Add a concrete memory region using pre-shared Arc data (O(1) clone).
    pub fn add_concrete_memory_shared(&mut self, base: u64, data: Arc<Vec<u8>>) {
        let size = data.len() as u64;
        Arc::make_mut(&mut self.concrete_memory).push(ConcreteMemoryRegion { base, size, data });
        self.concrete_memory_sorted = false;
    }

    /// Sort concrete memory regions by base address for binary search.
    fn sort_concrete_memory(&mut self) {
        if !self.concrete_memory_sorted && self.concrete_memory.len() > 1 {
            Arc::make_mut(&mut self.concrete_memory).sort_by_key(|r| r.base);
            self.concrete_memory_sorted = true;
        }
    }

    /// Clear all concrete memory regions.
    pub fn clear_concrete_memory(&mut self) {
        Arc::make_mut(&mut self.concrete_memory).clear();
        self.concrete_memory_sorted = false;
    }

    /// Try to read from concrete memory cache using binary search.
    /// Returns Some(data) if the address range is fully contained in a cached region.
    #[inline]
    fn try_read_concrete_memory(&self, addr: u64, size: usize) -> Option<&[u8]> {
        if self.concrete_memory.is_empty() {
            return None;
        }

        // Use binary search if we have many regions
        if self.concrete_memory.len() > 4 && self.concrete_memory_sorted {
            // Binary search: find the region where base <= addr
            let idx = self.concrete_memory.partition_point(|r| r.base <= addr);
            if idx > 0 {
                // Check the region just before this index
                let region = &self.concrete_memory[idx - 1];
                if let Some(data) = region.read(addr, size) {
                    return Some(data);
                }
            }
            return None;
        }

        // Linear scan for small number of regions
        for region in self.concrete_memory.iter() {
            if let Some(data) = region.read(addr, size) {
                return Some(data);
            }
        }
        None
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
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        size: usize,
        context: &str,
        addr_descr: &str,
    ) -> Result<RustBV, CbExecutionError> {
        if !callbacks.has_memory_load_symbolic_full() {
            return Err(CbExecutionError::Unsupported(format!(
                "{} with symbolic address ({}): no memory_load_symbolic_full callback",
                context, addr_descr
            )));
        }

        self.sync_before_callback(py, callbacks)?;

        let result_ast = callbacks
            .call_memory_load_symbolic_full(py, addr_val, size as u32)
            .map_err(|e| {
                CbExecutionError::Callback(format!(
                    "{} symbolic load full callback failed ({}): {}",
                    context, addr_descr, e
                ))
            })?;

        let ast = result_ast.bind(py);

        if let Some(ref table) = self.symbol_table {
            if let Some(bv) = try_handle_to_rustbv(&ast, table) {
                return Ok(bv);
            }
        }

        if is_claripy_ast(&ast) {
            if let Ok(bv) = claripy_to_rustbv(py, &ast, self.ctx) {
                return Ok(bv);
            }
            log::warn!(
                "{} symbolic load ({}, size={}): AST conversion failed; using fresh symbol",
                context,
                addr_descr,
                size
            );
        }

        Ok(RustBV::symbolic(
            self.ctx,
            format!("sym_pyref_{}_{}", addr_descr, size),
            (size * 8) as u32,
        ))
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
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addr_val: &RustBV,
        data_val: &RustBV,
        context: &str,
        addr_descr: &str,
    ) -> Result<(), CbExecutionError> {
        if !callbacks.has_memory_store_symbolic_full() {
            return Err(CbExecutionError::Unsupported(format!(
                "{} with symbolic address ({}): no memory_store_symbolic_full callback",
                context, addr_descr
            )));
        }

        self.sync_before_callback(py, callbacks)?;
        callbacks
            .call_memory_store_symbolic_full(py, addr_val, data_val)
            .map_err(|e| {
                CbExecutionError::Callback(format!(
                    "{} symbolic store full callback failed ({}): {}",
                    context, addr_descr, e
                ))
            })?;
        Ok(())
    }

    /// Load from memory via Python callback.
    /// This handles the common case of loading from a concrete address.
    fn load_from_callback(
        &self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addr_concrete: u64,
        size: usize,
    ) -> Result<RustBV, CbExecutionError> {
        let (data, is_symbolic, symbolic_ast) = callbacks
            .call_memory_load(py, addr_concrete, size as u32)
            .map_err(|e| CbExecutionError::Callback(e.to_string()))?;

        if is_symbolic {
            // Try to convert to RustBV - check handle first (fast path), then claripy (slow path)
            if let Some(ast_obj) = symbolic_ast {
                let ast = ast_obj.bind(py);

                // Fast path: check for RustBVHandle first
                if let Some(ref table) = self.symbol_table {
                    if let Some(bv) = try_handle_to_rustbv(&ast, table) {
                        return Ok(bv);
                    }
                }

                // Slow path: claripy AST conversion
                if is_claripy_ast(&ast) {
                    match claripy_to_rustbv(py, &ast, self.ctx) {
                        Ok(bv) => {
                            return Ok(bv);
                        }
                        Err(_e) => {
                            // Fall back to creating a fresh symbolic value
                            // (claripy conversion can fail for complex/unsupported ops)
                        }
                    }
                } else {
                }
            } else {
            }
            // Fallback: create a fresh symbolic value
            let bv = RustBV::symbolic(
                self.ctx,
                &format!("mem_{:x}_{}", addr_concrete, size),
                (size * 8) as u32,
            );
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

    /// Get the deferred forks collected during execution.
    pub fn deferred_forks(&self) -> &[DeferredFork] {
        &self.deferred_forks
    }

    /// Take the deferred forks, leaving an empty vector.
    pub fn take_deferred_forks(&mut self) -> Vec<DeferredFork> {
        std::mem::take(&mut self.deferred_forks)
    }

    /// Clear the deferred forks.
    pub fn clear_deferred_forks(&mut self) {
        self.deferred_forks.clear();
    }

    /// Get the number of deferred forks.
    pub fn num_deferred_forks(&self) -> usize {
        self.deferred_forks.len()
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

    /// Flush pending stores to Python via batch callback.
    ///
    /// This sends all buffered stores in a single callback, reducing
    /// FFI overhead compared to individual store callbacks.
    fn flush_stores(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
    ) -> Result<(), CbExecutionError> {
        if self.pending_stores.is_empty() && self.pending_symbolic_stores.is_empty() {
            return Ok(());
        }

        if self.use_rust_memory {
            // When Rust owns memory, flush stores to rust_memory instead of Python.
            if let Some(ref mut rust_mem) = self.rust_memory {
                for (addr, data) in self.pending_stores.iter() {
                    let width = (data.len() * 8) as u32;
                    let mut val: u128 = 0;
                    for (i, &b) in data.iter().enumerate() {
                        val |= (b as u128) << (i * 8);
                    }
                    let bv = RustBV::concrete(val, width);
                    let _ = rust_mem.store_concrete_automap_internal(*addr, bv);
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
            .call_memory_store_batch(py, self.pending_stores.as_slice())
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
                let width = (data.len() * 8) as u32;
                let mut val: u128 = 0;
                for (i, &b) in data.iter().enumerate() {
                    val |= (b as u128) << (i * 8);
                }
                let bv = RustBV::concrete(val, width);
                let _ = rust_mem.store_concrete_automap_internal(addr, bv);
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

    /// Check if an address is within loaded binary (concrete memory) regions.
    /// Used to distinguish internal function calls from external/library calls.
    pub fn is_in_binary(&self, addr: u64) -> bool {
        self.concrete_memory
            .iter()
            .any(|region| addr >= region.base && addr < region.base + region.size)
    }

    /// Whether the page containing `addr` has been overwritten via a store.
    /// Native-lift callers must avoid this page since they read from the
    /// immutable `concrete_memory` buffer that does not see the new bytes.
    pub fn is_code_page_dirtied(&self, addr: u64) -> bool {
        self.dirtied_code_pages.contains(&(addr >> 12))
    }

    /// Whether any page intersecting `[addr, addr + len)` has been written.
    pub fn is_code_range_dirtied(&self, addr: u64, len: u64) -> bool {
        if self.dirtied_code_pages.is_empty() || len == 0 {
            return false;
        }
        let first_page = addr >> 12;
        let last_page = (addr.saturating_add(len - 1)) >> 12;
        for page_num in first_page..=last_page {
            if self.dirtied_code_pages.contains(&page_num) {
                return true;
            }
        }
        false
    }

    /// Mark code pages overlapping `[addr, addr + size)` as dirtied and
    /// invalidate cached IRSBs whose byte ranges cover any of those bytes.
    /// Caller must verify the address is in a binary region first.
    pub fn invalidate_code_at(&mut self, addr: u64, size: usize) {
        if size == 0 {
            return;
        }
        let end = addr.saturating_add(size as u64 - 1);
        let first_page = addr >> 12;
        let last_page = end >> 12;
        for page_num in first_page..=last_page {
            self.dirtied_code_pages.insert(page_num);
        }
        // Find cached IRSBs whose [start, start + irsb.size()) overlaps the
        // write. LruCache::iter is O(N) but N <= 4096 and this fires only
        // on rare in-binary stores, so the cost is bounded.
        let mut to_remove: Vec<u64> = Vec::new();
        for (block_addr, irsb) in self.block_cache.iter() {
            let block_end = block_addr.saturating_add(irsb.size() as u64);
            if *block_addr <= end && block_end > addr {
                to_remove.push(*block_addr);
            }
        }
        for block_addr in to_remove {
            self.block_cache.pop(&block_addr);
        }
    }

    /// Register a SimProcedure at an address.
    ///
    /// This allows the interpreter to pre-extract arguments when the hook is hit,
    /// reducing Python callback overhead.
    pub fn register_simprocedure(
        &mut self,
        addr: u64,
        name: String,
        num_args: usize,
        no_return: bool,
    ) {
        Arc::make_mut(&mut self.hook_addrs).insert(addr);
        Arc::make_mut(&mut self.simprocedure_registry).insert(
            addr,
            SimProcedureInfo {
                name,
                num_args,
                no_return,
            },
        );
    }

    /// Register multiple SimProcedures at once.
    ///
    /// Each tuple is (address, name, num_args, no_return).
    pub fn register_simprocedures(&mut self, procs: &[(u64, String, usize, bool)]) {
        let hooks = Arc::make_mut(&mut self.hook_addrs);
        for (addr, _, _, _) in procs {
            hooks.insert(*addr);
        }
        let registry = Arc::make_mut(&mut self.simprocedure_registry);
        for (addr, name, num_args, no_return) in procs {
            registry.insert(
                *addr,
                SimProcedureInfo {
                    name: name.clone(),
                    num_args: *num_args,
                    no_return: *no_return,
                },
            );
        }
    }

    /// Get SimProcedure info for an address, if registered.
    pub fn get_simprocedure_info(&self, addr: u64) -> Option<&SimProcedureInfo> {
        self.simprocedure_registry.get(&addr)
    }

    /// Clear all SimProcedure registrations.
    pub fn clear_simprocedures(&mut self) {
        Arc::make_mut(&mut self.simprocedure_registry).clear();
    }

    /// Extract arguments for a SimProcedure call.
    ///
    /// Uses the calling convention to extract arguments from registers and stack.
    pub fn extract_simprocedure_args(&self, num_args: usize) -> Vec<RustBV> {
        self.calling_convention.extract_args(
            &self.registers,
            self.rust_memory.as_ref(),
            self.ctx,
            num_args,
        )
    }

    /// Get the return address for a function call.
    ///
    /// Checks pending_stores and all_flushed_stores first, since the call
    /// instruction pushes the return address via VEX stores before the
    /// interpreter detects the SimProcedure hook.
    pub fn get_return_addr(&self) -> Option<u64> {
        let ptr_size = self.calling_convention.pointer_size();
        let sp = self
            .registers
            .get(self.registers.arch().sp_offset(), ptr_size, self.ctx);
        let sp_val = sp.as_u64()?;

        // Check pending_stores first (most recent writes, same block)
        if let Some(data) = self
            .pending_stores
            .try_load_exact(sp_val, ptr_size as usize)
        {
            let mut bytes = [0u8; 8];
            let len = std::cmp::min(ptr_size as usize, 8);
            bytes[..len].copy_from_slice(&data[..len]);
            return Some(u64::from_le_bytes(bytes));
        }

        // Check all_flushed_stores (cross-block within same step)
        if let Some(data) = self.all_flushed_stores.get(&sp_val) {
            if data.len() >= ptr_size as usize {
                let mut bytes = [0u8; 8];
                let len = std::cmp::min(ptr_size as usize, 8);
                bytes[..len].copy_from_slice(&data[..len]);
                return Some(u64::from_le_bytes(bytes));
            }
        }

        // Fall back to rust_memory
        self.calling_convention.get_return_addr(
            &self.registers,
            self.rust_memory.as_ref(),
            self.ctx,
        )
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
        self.block_cache.get(&addr).map(|arc| arc.as_ref())
    }

    /// Swap in a shared block cache, returning the interpreter's current cache.
    pub fn swap_block_cache(
        &mut self,
        cache: LruCache<u64, Arc<IRSB>>,
    ) -> LruCache<u64, Arc<IRSB>> {
        std::mem::replace(&mut self.block_cache, cache)
    }

    /// Take the last branch condition, if any.
    ///
    /// This is set when a SymbolicBranch result is created, and can be retrieved
    /// by callers who need to add constraints for forked states.
    /// The condition is cleared after being retrieved.
    pub fn take_last_branch_condition(&mut self) -> Option<RustBV> {
        self.last_branch_condition.take()
    }

    /// Get a stored condition by ID.
    ///
    /// Returns the branch condition associated with the given condition ID,
    /// if one was stored. This is used for deferred fork handling.
    pub fn get_stored_condition(&self, condition_id: u64) -> Option<&RustBV> {
        self.stored_conditions.get(&condition_id)
    }

    /// Take all stored conditions.
    ///
    /// Returns all stored conditions as a HashMap. The internal map is cleared.
    /// This is useful for bulk retrieval when processing multiple deferred forks.
    pub fn take_stored_conditions(&mut self) -> FxHashMap<u64, RustBV> {
        std::mem::take(&mut self.stored_conditions)
    }

    /// Take all branch snapshots for deferred forks.
    ///
    /// Returns full state snapshots captured BEFORE branch constraints were added,
    /// keyed by condition_id. Used for correct alternate-path forking.
    pub fn take_fork_snapshots(&mut self) -> FxHashMap<u64, BranchSnapshot> {
        std::mem::take(&mut self.fork_snapshots)
    }

    /// Run the execution loop until an event requires Python handling.

    /// Fork the interpreter state.
    pub fn fork(&self) -> CallbackInterpreter<'a> {
        // Clone the calling convention based on its type
        let arch_box = arch_from_vex(self.arch);
        let cc = default_cc_for_arch(arch_box.name());

        CallbackInterpreter {
            registers: self.registers.fork(),
            temps: self.temps.clone(),
            ctx: self.ctx,
            symbol_table: self.symbol_table, // Share symbol table reference
            pc: self.pc,
            current_insn_addr: self.current_insn_addr,
            current_insn_len: self.current_insn_len,
            hook_addrs: Arc::clone(&self.hook_addrs),
            arch: self.arch,
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
            rust_memory: self.rust_memory.as_ref().map(|m| m.fork()),
            use_rust_memory: self.use_rust_memory,
            lazy_solves: self.lazy_solves,
            no_ip_concretization: self.no_ip_concretization,
            load_prefetch_cache: FxHashMap::default(), // Fresh prefetch cache for fork
            use_load_prefetch: self.use_load_prefetch,
            page_prefetch_count: self.page_prefetch_count, // Inherit page prefetch count
            dirty_dispatch: DirtyHelperDispatch::new(),    // Fresh dispatch (stateless)
            simprocedure_registry: Arc::clone(&self.simprocedure_registry), // Share SimProcedure registry
            calling_convention: cc,
            last_branch_condition: None,               // Fresh for fork
            pending_python_constraints: Vec::new(),    // Fresh constraints for fork
            stored_conditions: FxHashMap::default(),   // Fresh for fork
            fork_snapshots: FxHashMap::default(),      // Fresh for fork
            stats: ExecutionStats::default(),          // Fresh stats for fork
            profiling_enabled: self.profiling_enabled, // Inherit profiling setting
            concrete_memory_sorted: self.concrete_memory_sorted, // Inherit sorted flag
            concretize_cache: FxHashMap::default(),    // Fresh cache for fork
            prefetch_loads_scratch: Vec::new(),
            prefetch_unique_scratch: Vec::new(),
            prefetch_dedup_scratch: HashSet::new(),
            prefetch_callback_scratch: Vec::new(),
            call_stack: self.call_stack.clone(), // Clone call stack for fork
            detailed_history: self.detailed_history.clone(), // Clone history for fork
            vex_opt_level: self.vex_opt_level,   // Inherit VEX opt level
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
mod smc_tests {
    use super::*;

    fn make_irsb(addr: u64, len_bytes: u32) -> IRSB {
        let mut irsb = IRSB::new(addr, VexArch::AMD64);
        irsb.statements.push(IRStmt::IMark {
            addr,
            len: len_bytes,
            delta: 0,
        });
        irsb
    }

    #[test]
    fn invalidate_removes_overlapping_block_and_marks_page_dirty() {
        let ctx = SymContext::new_mock();
        let mut interp = CallbackInterpreter::new(VexArch::AMD64, &ctx);
        interp.add_concrete_memory(0x1000, vec![0u8; 0x1000]);
        // Block at 0x1010 covering 8 bytes -> [0x1010, 0x1018).
        interp.cache_block(0x1010, make_irsb(0x1010, 8));
        assert!(interp.has_cached_block(0x1010));
        // Write a single byte at 0x1014 (inside the block range).
        interp.invalidate_code_at(0x1014, 1);
        assert!(!interp.has_cached_block(0x1010));
        assert!(interp.is_code_page_dirtied(0x1014));
        assert!(interp.is_code_page_dirtied(0x1010));
    }

    #[test]
    fn invalidate_skips_non_overlapping_blocks() {
        let ctx = SymContext::new_mock();
        let mut interp = CallbackInterpreter::new(VexArch::AMD64, &ctx);
        interp.add_concrete_memory(0x1000, vec![0u8; 0x2000]);
        interp.cache_block(0x1010, make_irsb(0x1010, 8));
        // Write a byte at 0x1100 — different bytes, but same page (0x1).
        interp.invalidate_code_at(0x1100, 1);
        // The block at 0x1010 doesn't overlap the write range, so it stays.
        assert!(interp.has_cached_block(0x1010));
        // But the page is now marked dirty (0x1100 >> 12 == 0x1).
        assert!(interp.is_code_page_dirtied(0x1100));
        assert!(interp.is_code_page_dirtied(0x1010));
    }

    #[test]
    fn invalidate_handles_multi_page_writes() {
        let ctx = SymContext::new_mock();
        let mut interp = CallbackInterpreter::new(VexArch::AMD64, &ctx);
        interp.add_concrete_memory(0x1000, vec![0u8; 0x4000]);
        // 16-byte write straddles 0x1ff8..0x2008 — pages 0x1 and 0x2.
        interp.invalidate_code_at(0x1ff8, 16);
        assert!(interp.is_code_page_dirtied(0x1ff8));
        assert!(interp.is_code_page_dirtied(0x2000));
    }

    #[test]
    fn is_code_range_dirtied_spans_pages() {
        let ctx = SymContext::new_mock();
        let mut interp = CallbackInterpreter::new(VexArch::AMD64, &ctx);
        interp.add_concrete_memory(0x1000, vec![0u8; 0x4000]);
        // Dirty just page 0x2.
        interp.invalidate_code_at(0x2000, 1);
        // A lift window at 0x1ff0 size 32 crosses pages 0x1 and 0x2.
        assert!(interp.is_code_range_dirtied(0x1ff0, 32));
        // A lift window at 0x1000 size 16 stays within page 0x1.
        assert!(!interp.is_code_range_dirtied(0x1000, 16));
    }

    #[test]
    fn fork_inherits_dirtied_pages() {
        let ctx = SymContext::new_mock();
        let mut interp = CallbackInterpreter::new(VexArch::AMD64, &ctx);
        interp.add_concrete_memory(0x1000, vec![0u8; 0x1000]);
        interp.invalidate_code_at(0x1500, 1);
        let child = interp.fork();
        assert!(child.is_code_page_dirtied(0x1500));
    }
}
