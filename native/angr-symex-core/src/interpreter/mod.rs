//! Callback-aware VEX IR interpreter.
//!
//! This interpreter uses Python callbacks for memory operations instead of
//! local SymbolicMemory. It can run multiple blocks in a loop, returning
//! to Python only when an event requires Python handling.
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** every value
//! this interpreter touches is guest-derived, so it carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` and IR it cannot handle
//! returns a `CbExecutionError` instead. There are no `unwrap`/`expect` sites
//! left in this file: the block-cache capacity is validated at compile time by
//! [`BLOCK_CACHE_CAPACITY_NZ`].
//!
//! **Layout (angr-9ke6b.91):** this root holds the [`VEXInterpreter`] struct
//! and its core methods. Three concerns that used to live here have their own
//! files, all re-exported below so `crate::interpreter::X` paths are unchanged:
//! [`ExecutionStats`] in `execution_stats`, the
//! [`CbExecutionError`]/[`FallbackStrategy`] taxonomy plus its reason markers
//! in `execution_error`, and the [`BlockResult`]/`StmtResult`/
//! `ConcretizedJump` control-flow results in `block_result`.
//!
//! **Host boundary (angr-mx00p.75):** `host` holds the [`EngineHost`]
//! trait the interpreter calls out through, and the value types that cross it
//! sit beside it: [`InspectBit`] in `inspect_bits`, [`RunResult`] /
//! [`RunErrorKind`] / [`ErrorRoute`] in `run_result`, and
//! [`ExecutionConfig`] / [`DeferredFork`] in `config`. None of them holds a
//! Python type; the PyO3 shell implements the trait in `callbacks/`.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use lru::LruCache;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::arch::{
    AddrOrSymbolic, RegisterCheckpoint, RegisterFile, arch_from_vex,
    calling_conventions::CallingConvention, default_cc_for_arch,
};
use crate::memory::{MemoryError, Permission, SymbolicMemory};
use crate::symbolic::concretize::{AddressConcretizer, ConcretizationResult, strided_addrs};
use crate::symbolic::{
    BVOp, RustBV, SymContext, record_mem_load, record_mem_store, record_vex_binop, record_vex_qop,
    record_vex_triop, record_vex_unop,
};
use angr_vex::ccall;
use angr_vex::dirty::DirtyHelperDispatch;
use angr_vex::ir::{
    IRConst, IRExpr, IRLoadGOp, IROp, IRSB, IRStmt, IRType, JumpKind, TypeEnv, VexArch,
};
use angr_vex::ops::{OpError, VEXOps, iropclass};
use angr_vex::{Endness, VEX_MAX_BYTES};

/// VEX block (IRSB) LRU cache capacity in entries.
///
/// Each cached `IRSB` is Arc-shared, so this caps unique IRSBs simultaneously
/// resident across an interpreter's lifetime. 4096 covers the working set of
/// every tracked bench in `baseline_timings.json`; adjust only with a counter
/// dump showing eviction churn (`mgr.stats()` does not expose
/// hit/miss; angr-l4bs).
pub const BLOCK_CACHE_CAPACITY: usize = 4096;

/// [`BLOCK_CACHE_CAPACITY`] pre-validated as the `NonZeroUsize` every
/// `LruCache::new` call site needs.
///
/// Doing the conversion once in a `const` moves the non-zero proof to compile
/// time: a zero capacity fails the build here instead of surfacing as a runtime
/// `.expect` in each of the constructors that build a block cache
/// (`ExecutionEnv::new`, `VEXInterpreter::new`, and
/// `exploration::scheduler_pool::worker_thread`).
pub const BLOCK_CACHE_CAPACITY_NZ: std::num::NonZeroUsize =
    match std::num::NonZeroUsize::new(BLOCK_CACHE_CAPACITY) {
        Some(n) => n,
        None => panic!("BLOCK_CACHE_CAPACITY must be non-zero"),
    };

/// Maximum number of candidate addresses for which a symbolic memory access is
/// resolved by building an ITE chain in Rust.
///
/// A `Multiple`/`Strided` concretization can carry up to `Concretizer::max_solutions`
/// (256) addresses, and an in-Rust ITE chain costs one Python `memory_load_batch`
/// entry plus one ITE node per address — an expression the solver then has to
/// carry through every downstream constraint. Beyond this cap the load
/// (`dispatch_multi_load`) hands the whole access to Python's memory model,
/// which has purpose-built machinery for wide symbolic accesses (angr-sqfj8.68).
/// The store (`dispatch_multi_store`) has no cap: a proxy-backed host cannot
/// write Rust memory from its full store (angr-vmi2l.324).
pub const MAX_ITE_ADDRS: usize = 16;

/// Start a profiling timer iff `self.profiling_enabled`. Yields `Option<Instant>`.
///
/// Pair with `profile_add!` to fold the elapsed nanoseconds into a `u64`
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
            $field += crate::elapsed_ns(__profile_start);
        }
    };
}

mod block_result;
pub mod bv_utils;
mod code_invalidation;
mod concrete_memory;
mod concretize_cache;
mod config;
mod execution;
mod execution_error;
mod execution_stats;
mod exits;
mod expressions;
mod expressions_arith;
mod expressions_inspect;
mod expressions_loads;
mod fork_state;
mod host;
mod inspect_bits;
mod inspect_view;
mod pending_store;
mod prefetch;
mod run_result;
mod simprocedures;
mod statements;
mod statements_cas;
mod statements_dirty;
mod statements_inspect;
mod statements_store;
mod transcendental_pins;

// Shared test helpers for the `statements*` test modules. Registered here
// rather than from a prod file because its consumers are children of three
// DIFFERENT prod modules (`statements.rs`, `statements_cas.rs`,
// `statements_dirty.rs`), so no `super::` path reaches across them
// (angr-fs8kb.96).
test_submod!("statements_tests_support.rs" => statements_tests_support);

// The plain-Rust `EngineHost` every interpreter test runs against, so none of
// them starts Python (angr-mx00p.75).
test_submod!(test_host);
#[cfg(test)]
pub use test_host::{SymbolicValueStore, TestHost};

use block_result::{ConcretizedJump, StmtResult};
use bv_utils::bytes_to_bv;
use pending_store::PendingStoreBuffer;

pub use block_result::BlockResult;
pub use concrete_memory::ConcreteMemoryRegion;
pub use config::{DeferredFork, ExecutionConfig, TrapSide};
pub use execution_error::{
    CbExecutionError, DCAS_UNSUPPORTED_REASON, DISPATCH_FABRICATE_REASON, FallbackStrategy,
    VECRET_GSPTR_REASON,
};
pub use execution_stats::ExecutionStats;
pub use host::{EngineHost, HostLoad, HostPage, LiftBlockOutcome};
pub use inspect_bits::InspectBit;
pub use inspect_view::InspectView;
pub use run_result::{ErrorRoute, RunErrorKind, RunResult};
pub use simprocedures::SimProcedureInfo;

/// Re-exported from `state`, which owns it: `RustSimState::fork_from_snapshot`
/// is the consumer, and the interpreter only fills one in at a branch.
pub use crate::state::BranchSnapshot;

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
    /// Current program counter.
    pub pc: u64,
    /// Current instruction address (within block).
    current_insn_addr: u64,
    /// Current instruction length (from IMark).
    current_insn_len: u32,
    /// Hook addresses (return to Python when hit).
    /// Arc-shared on fork (O(1) clone). Mutators use Arc::make_mut for CoW.
    hook_addrs: Arc<FxHashSet<u64>>,
    /// Addresses a block ends before, as Python's `extra_stop_points` split a
    /// block, without stopping a state that starts there: the explore targets
    /// of a round stepped with find/avoid disarmed (angr-mx00p.132.60). See
    /// [`Self::lift_window`] and the `IMark` arm of `execute_stmt_with_callbacks`.
    lift_cuts: Arc<FxHashSet<u64>>,
    /// A hook to step over once: the zero-length hook the run resumes at,
    /// already serviced by its callback. [`Self::is_hooked`] reports it
    /// unhooked until the run's first `IMark` at that address consumes it,
    /// so a loop that comes back to it later in the same run stops there
    /// again (angr-mx00p.132.5).
    skip_hook_once: Option<u64>,
    /// VEX architecture.
    arch: VexArch,
    /// Native lifter for this arch. When set (and the bytes qualify),
    /// cold-block lifts are attempted in-process before the Python
    /// `lift_block` callback. `None` by default; Python enables it through
    /// `RustExplorationManager.set_native_lift` (angr-z087y). Shared, and
    /// called only on a cache miss (angr-mx00p.51.7).
    native_lifter: Option<Arc<dyn angr_vex::VEXLifter>>,
    /// Block cache (shared across runs) using Arc for O(1) cloning.
    block_cache: LruCache<u64, Arc<IRSB>>,
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
    /// Next condition ID for tracking branch conditions.
    next_condition_id: u64,
    /// Concrete memory regions cached locally for fast access.
    /// These are read-only regions (e.g., binary .text/.rodata sections).
    /// Arc-shared on fork (O(1) clone). Mutators use Arc::make_mut for CoW.
    concrete_memory: Arc<Vec<ConcreteMemoryRegion>>,
    /// `[start, end)` of every loaded object (see
    /// [`Self::set_loader_object_ranges`]). Arc-shared on fork.
    loader_object_ranges: Arc<[(u64, u64)]>,
    /// Address concretizer for handling symbolic addresses.
    concretizer: AddressConcretizer,
    /// Pending concrete stores to batch for efficiency.
    /// Each entry is (address, data_bytes). Wrapped in a buffer that maintains
    /// a per-byte-address index so loads can fast-skip the reverse scan.
    ///
    /// Under `use_rust_memory` this and `pending_symbolic_stores` hold only
    /// writes `rust_memory` refused: every store, guarded and CAS writebacks
    /// included, tries `try_rust_memory_store` first, because `load_layered`
    /// reads these buffers only after `rust_memory` declines and
    /// `flush_stores` replays them after any newer direct store
    /// (angr-vmi2l.168).
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
    /// NO_IP_CONCRETIZATION option (`SimSuccessors._categorize_successor`).
    pub no_ip_concretization: bool,
    /// When true, route any symbolic jump target to UnconstrainedJump before
    /// AddressConcretizer enumeration is attempted. Mirrors angr's
    /// NO_SYMBOLIC_JUMP_RESOLUTION option (`SimSuccessors._categorize_successor`).
    /// Functionally identical to `no_ip_concretization` for the symbolic-IP
    /// case in Rust; kept as a separate flag so Python option semantics are
    /// preserved.
    pub no_symbolic_jump_resolution: bool,
    /// When true and the next pc was concretized from a symbolic expression,
    /// store that expression in `symbolic_ip_at_exit` so the manager can
    /// write it back to the IP register after the block (mirroring Python's
    /// `split_state.regs.ip = target` in `SimSuccessors._categorize_successor`). Also
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
    /// A syscall (or trap-side sibling's trap) exit the step stopped at without running it. Seeded from
    /// and written back to `RustSimState::pending_syscall` around each step;
    /// see that field and `run_until_event`.
    pub pending_syscall: Option<JumpKind>,
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
    /// Pages inside loaded binary regions that have been overwritten by a
    /// store. Used to invalidate cached IRSBs and to redirect the Python lift
    /// callback to read fresh bytes from `rust_memory` instead of the original
    /// (now stale) binary image.
    dirtied_code_pages: FxHashSet<crate::memory::PageIndex>,
    /// State id of the state currently being stepped. Forwarded to
    /// `EngineHost::inspect_mem_*` so the Python dispatcher can
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
    /// Create a new little-endian callback-aware interpreter.
    ///
    /// Test-only: the two production entry points (the engine shell's `engine` module's
    /// single-block exec and `exploration/step_core.rs`) both know their
    /// target's byte order and go through [`Self::with_config_endian`]
    /// (angr-21cz6).
    #[cfg(test)]
    pub fn new(arch: VexArch, ctx: &'a SymContext) -> Self {
        Self::with_config(arch, ctx, ExecutionConfig::default())
    }

    /// Create a new callback-aware interpreter with custom config.
    ///
    /// Little-endian by default; see [`Self::with_config_endian`] for why that
    /// is a caller-supplied fact rather than one derivable from `arch`.
    pub fn with_config(arch: VexArch, ctx: &'a SymContext, config: ExecutionConfig) -> Self {
        Self::with_config_endian(arch, ctx, config, true)
    }

    /// Create a new callback-aware interpreter for a target of known byte order.
    ///
    /// `is_le` is the *target's* endianness, not `Arch::is_little_endian` (which
    /// is hardcoded true on every arch this crate models). MIPS32/MIPS64/ARM are
    /// either-endian families, so only the caller knows: the exploration path
    /// gets it from the state (`RustSimState::with_solver_endian`, whose
    /// register file `exploration/step_core.rs` forks straight over this one),
    /// and `engine::execute_irsb_for_test` takes it as a parameter.
    ///
    /// It reaches exactly one thing: the register file's `mirror_offset`
    /// adapter, which is what makes a VEX Get/Put *narrower* than its containing
    /// register pick the same half angr's Python engine does on a big-endian
    /// target (angr-fuhmm). Storage stays little-endian either way — see
    /// `RegisterFile::mirror_offset`.
    pub fn with_config_endian(
        arch: VexArch,
        ctx: &'a SymContext,
        config: ExecutionConfig,
        is_le: bool,
    ) -> Self {
        let arch_box = arch_from_vex(arch);
        let arch_name = arch_box.name();
        let cc = default_cc_for_arch(arch_name);

        VEXInterpreter {
            registers: RegisterFile::new_with_endian(arch_box, is_le),
            temps: Vec::with_capacity(64), // Pre-allocate for typical block size
            ctx,
            pc: 0,
            current_insn_addr: 0,
            current_insn_len: 0,
            hook_addrs: Arc::new(FxHashSet::default()),
            lift_cuts: Arc::new(FxHashSet::default()),
            skip_hook_once: None,
            arch,
            native_lifter: None,
            // Bounded default: this is the interpreter's own cache when it runs
            // standalone (engine.rs single-block exec, tests). It MUST stay
            // bounded/clone-safe because `fork()` clones it and lru's `Clone`
            // does `LruCache::new(self.cap())` — an `unbounded()` cap of
            // `usize::MAX` overflows `HashMap::with_capacity` (angr-4xaga.1).
            // In the exploration hot path this default is discarded when
            // run_interpreter_step_core swaps the shared cache in, but the two
            // *placeholder* caches there ARE zero-preallocation `unbounded()`.
            block_cache: LruCache::new(BLOCK_CACHE_CAPACITY_NZ),
            deferred_forks: Vec::new(),
            deferred_fork_this_step: false,
            block_solver_pushed: false,
            block_forks_asserted: 0,
            config,
            next_condition_id: 0,
            concrete_memory: Arc::new(Vec::new()),
            loader_object_ranges: Arc::from([]),
            concretizer: AddressConcretizer::new(),
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
            pending_syscall: None,
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

    /// Enable or disable profiling.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling_enabled = enabled;
        if enabled {
            self.stats.reset();
        }
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
        callbacks: &dyn EngineHost,
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

        let value = callbacks
            .memory_load_symbolic_full(self.ctx, addr_val, size as u32)
            .map_err(|e| {
                CbExecutionError::Callback(format!(
                    "{context} symbolic load full callback failed ({addr_descr}): {e}"
                ))
            })?;

        // The host's value, or a fresh symbol when it did not convert.
        Ok(value.unwrap_or_else(|| {
            RustBV::symbol(format!("sym_pyref_{addr_descr}_{size}"), (size * 8) as u32)
        }))
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
        callbacks: &dyn EngineHost,
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
            .memory_store_symbolic_full(addr_val, data_val)
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
    /// is every `memory_load` crossing on the ZeroPy benches whose sole
    /// residual GIL was this site — all of them the `fs:[0x28]` stack canary
    /// or another unmapped low address.
    ///
    /// The oracle is `python_has_page`, NOT the `python_can_serve_page` set
    /// (angr-gorvf.4.6) — those answer different questions, and using the
    /// latter here is a silent-corruption bug. A page
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
    /// the `__stack_chk_fail` branch spuriously feasible. With `rust_memory`
    /// on, `install_unservable_filler` has already stored a filler before
    /// the load gets here, so this rung only serves the callback-only mode
    /// and a load that one could not map.
    fn synthesize_unservable_load(
        &self,
        callbacks: &dyn EngineHost,
        addr_concrete: u64,
        size: usize,
    ) -> Option<RustBV> {
        let page_addr = addr_concrete & !(crate::memory::PAGE_SIZE - 1);
        if callbacks.has_page(page_addr) {
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
        let bv = RustBV::symbol(&name, bits);
        self.dispatch_symbolic_variable_inspect(callbacks, &name, bits, &bv);
        Some(bv)
    }

    /// Mint the filler for a concrete load neither side can back and write
    /// it into `rust_memory`, so memory stays a function (angr-vmi2l.179).
    ///
    /// `synthesize_unservable_load` hands back a filler without storing it,
    /// so a narrower or shifted re-read of the same bytes mints an unrelated
    /// symbol and the export fills them with a third. Python's
    /// `DefaultFillerMixin` stores its filler; this does the same: map the
    /// pages as `map_fresh_pages_for_store` does for a store, store a uniquely
    /// named `mem_<addr>_rs<id>_<bits>` filler, and re-read it. The `rs` id
    /// keeps the name disjoint from Python's `mem_<addr>_<id>_<bits>` counter.
    /// The stack-canary double read now agrees because the second read hits
    /// the stored bytes, not because two mints share a name. Returns `None`
    /// whenever Rust or the host may hold any page under the load, leaving
    /// the caller on the callback ladder.
    pub(super) fn install_unservable_filler(
        &mut self,
        callbacks: &dyn EngineHost,
        addr_concrete: u64,
        size: usize,
    ) -> Option<RustBV> {
        // A store still buffered outside `rust_memory` (one that missed it
        // and went to the callback ladder) holds bytes the filler would
        // shadow; leave such a load to the ladder, which reads the buffers.
        if !self.store_buffers_empty() {
            return None;
        }
        // A straddling load whose other page Rust already holds would have
        // the filler overwrite that page's real bytes.
        let rust_mem = self.rust_memory.as_ref()?;
        if crate::memory::PageIndex::range_covering(addr_concrete, size as u64)
            .any(|page| rust_mem.is_mapped(page.base_addr()))
        {
            return None;
        }
        if !self.map_fresh_pages_for_store(callbacks, addr_concrete, size) {
            return None;
        }
        let bits = (size * 8) as u32;
        let name = crate::memory::mem_filler_name(addr_concrete, bits);
        let filler = RustBV::symbol(&name, bits);
        let rust_mem = self.rust_memory.as_mut()?;
        if let Err(e) = rust_mem.store_concrete(addr_concrete, filler.clone()) {
            // SILENT(cat-b): the pages were just mapped writable, so this
            // cannot fail short of a wrapping range; the caller falls back to
            // the uninstalled `synthesize_unservable_load` filler.
            log::debug!("filler store at {addr_concrete:#x} failed: {e:?}");
            return None;
        }
        rust_mem.record_filler(addr_concrete, name.clone(), bits);
        self.dispatch_symbolic_variable_inspect(callbacks, &name, bits, &filler);
        Some(filler)
    }

    /// Under STRICT_PAGE_ACCESS, the fault for an access to `[addr, addr +
    /// size)` that touches a page neither side holds (angr-vmi2l.258).
    ///
    /// Python's `PrivilegedPagingMixin` raises `SimSegfaultException(page,
    /// "unmapped")` rather than create such a page. Without this the
    /// `Unmapped` arms fell back to the callback ladder, which minted a
    /// filler (`synthesize_unservable_load`) or wrote the store into the
    /// host, and the state kept running. The error names the page base, as
    /// Python's does. A page the host holds is left to the fetch paths.
    pub(super) fn strict_unmapped_fault(
        &self,
        callbacks: &dyn EngineHost,
        addr: u64,
        size: usize,
    ) -> Option<MemoryError> {
        let rust_mem = self.rust_memory.as_ref()?;
        if !rust_mem.enforce_permissions() {
            return None;
        }
        crate::memory::PageIndex::range_covering(addr, size as u64)
            .map(|page| page.base_addr())
            .find(|&base| !rust_mem.is_mapped(base) && !callbacks.has_page(base))
            .map(|base| MemoryError::Unmapped {
                addr: base,
                size: size as u64,
            })
    }

    /// Whether every store buffer outside `rust_memory` is empty, so a
    /// filler written into it cannot shadow a buffered write.
    fn store_buffers_empty(&self) -> bool {
        self.pending_stores.is_empty()
            && self.pending_symbolic_stores.is_empty()
            && self.all_flushed_stores.is_empty()
            && self.all_flushed_symbolic_stores.is_empty()
    }

    /// Before a concrete load, mint and store a filler over each run of
    /// never-written bytes under it on a page Rust mapped with no host bytes
    /// (angr-vmi2l.325), as Python's `DefaultFillerMixin` does per gap. The
    /// load then reads the stored fillers, and so does every later read and
    /// the export. Each run fires the symbolic_variable inspect, like
    /// `install_unservable_filler`. Does nothing while a store is buffered
    /// outside `rust_memory`, for the reason given there.
    pub(super) fn fill_uninit_under_load(
        &mut self,
        callbacks: &dyn EngineHost,
        addr_concrete: u64,
        size: usize,
    ) {
        let Some(rust_mem) = self.rust_memory.as_ref() else {
            return;
        };
        let runs = rust_mem.uninit_runs(addr_concrete, size as u64);
        if runs.is_empty() || !self.store_buffers_empty() {
            return;
        }
        for (run_addr, run_len) in runs {
            let bits = (run_len * 8) as u32;
            let name = crate::memory::mem_filler_name(run_addr, bits);
            let filler = RustBV::symbol(&name, bits);
            let Some(rust_mem) = self.rust_memory.as_mut() else {
                return;
            };
            if let Err(e) = rust_mem.store_concrete(run_addr, filler.clone()) {
                // SILENT(cat-b): the run lies on a mapped page Rust owns, so
                // only a write-protected page refuses; the load then reads
                // the page's zero, its reading before angr-vmi2l.325.
                log::debug!("uninit filler store at {run_addr:#x} failed: {e:?}");
                continue;
            }
            rust_mem.record_filler(run_addr, name.clone(), bits);
            self.dispatch_symbolic_variable_inspect(callbacks, &name, bits, &filler);
        }
    }

    /// Load from memory via Python callback.
    /// This handles the common case of loading from a concrete address.
    ///
    /// Bottom rung of the load-resolution ladder: callers reach it from
    /// `load_layered` (through `load_concrete_addr`, after the pending/flushed
    /// store buffers and the prefetch and concrete-memory caches, or through
    /// `load_symbolic_addr`'s concretized arms), never in place of it.
    /// `synthesize_unservable_load` above gets the last word before the GIL
    /// crossing.
    fn load_from_callback(
        &self,
        callbacks: &dyn EngineHost,
        addr_concrete: u64,
        size: usize,
    ) -> Result<RustBV, CbExecutionError> {
        if let Some(bv) = self.synthesize_unservable_load(callbacks, addr_concrete, size) {
            return Ok(bv);
        }

        let load = callbacks
            .memory_load(self.ctx, addr_concrete, size as u32)
            .map_err(CbExecutionError::Callback)?;

        if load.symbolic {
            // The host's symbolic value, when it converted (a complex or
            // unsupported AST falls through to a fresh symbol).
            if let Some(bv) = load.value {
                return Ok(bv);
            }
            // Fallback: create a fresh symbolic value
            let name = format!("mem_{addr_concrete:x}_{size}");
            let bits = (size * 8) as u32;
            let bv = RustBV::symbol(&name, bits);
            // angr-vfst: symbolic_variable BP_AFTER for engine-internal fresh
            // BVS minting. Mirrors Python's `SimSolver.BVS` BP_AFTER
            // signature. The user-callable `state.solver.BVS()` path still
            // fires the same event from Python directly.
            self.dispatch_symbolic_variable_inspect(callbacks, &name, bits, &bv);
            Ok(bv)
        } else {
            // Convert bytes to concrete value
            Ok(bytes_to_bv(&load.bytes, (size * 8) as u32))
        }
    }

    /// Set the execution configuration.
    /// Production builds the interpreter with its final `ExecutionConfig`; only `exits_tests` swaps one in afterwards (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn set_config(&mut self, config: ExecutionConfig) {
        self.config = config;
    }

    /// Get the next condition ID.
    fn next_cond_id(&mut self) -> u64 {
        let id = self.next_condition_id;
        self.next_condition_id += 1;
        id
    }

    /// Undo the register writes and the still-buffered stores of a block
    /// that bailed out mid-way, so the Python re-run of the whole block does
    /// not apply its leading statements twice: a `push` before a `cpuid`
    /// moved the stack pointer by two slots (angr-mx00p.132.33). Stores
    /// already flushed (an auto-flush, or a mid-block deferred fork's flush)
    /// and Python store callbacks are not undone.
    fn discard_partial_block(&mut self, pre_block_registers: RegisterCheckpoint) {
        self.registers.restore(pre_block_registers);
        self.pending_stores.clear();
        self.pending_symbolic_stores.clear();
    }

    /// Flush pending stores to Python via batch callback.
    ///
    /// This sends all buffered stores in a single callback, reducing
    /// FFI overhead compared to individual store callbacks.
    fn flush_stores(&mut self, callbacks: &dyn EngineHost) -> Result<(), CbExecutionError> {
        if self.pending_stores.is_empty() && self.pending_symbolic_stores.is_empty() {
            return Ok(());
        }

        if self.use_rust_memory {
            // When Rust owns memory, flush stores to rust_memory instead of Python.
            if let Some(ref mut rust_mem) = self.rust_memory {
                for (addr, data) in self.pending_stores.iter() {
                    // SILENT(cat-c): the only failure here is an unmapped
                    // target outside every lazy region (or a range that wraps
                    // the address space), which the store path cannot repair.
                    // The write is dropped while the drain below still records
                    // it in `all_flushed_stores`, so a later load served from
                    // that sidecar would see a value `rust_memory` never got —
                    // log loudly and keep going.
                    if let Err(e) = rust_mem.store_concrete_le_bytes_automap_internal(*addr, data) {
                        log::error!("dropped concrete store at {addr:#x}: {e:?}");
                    }
                }
                // Also flush symbolic stores to rust_memory so that subsequent
                // loads via load_concrete_lazy_inner find the symbolic values
                // instead of returning concrete zeros from the page fill.
                // `install_*`, not `import_*`: these are guest-computed values
                // Python does not hold, so they must stay exportable
                // (angr-ccp2w).
                //
                // Drain in this loop and insert into all_flushed_symbolic_stores too,
                // sharing one clone instead of doing iter+clone followed by drain.
                for (addr, bv) in self.pending_symbolic_stores.drain() {
                    // SILENT(cat-c): a non-byte-multiple width cannot be
                    // placed in byte-addressed memory at all. VEX stores are
                    // always byte-multiple (Ity_I8 and wider), so this is a
                    // "shouldn't happen" arm — log loudly and keep the value
                    // in all_flushed_symbolic_stores so the block-boundary
                    // path still sees it. Different failure mode from the
                    // concrete-store arm above, same cat-c handling.
                    if let Err(e) = rust_mem.install_symbolic_value(addr, bv.clone()) {
                        log::error!("dropped symbolic store at {addr:#x}: {e:?}");
                    }
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
            .memory_store_batch(self.pending_stores.as_slice())
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
                // SILENT(cat-c): same unmapped-target arm as the concrete
                // branch of `flush_stores`, but here it is terminal — both
                // sidecars are cleared below and `rust_memory` is about to be
                // handed back to the state, so the dropped write is simply
                // gone. Nothing at this layer can repair it; log loudly.
                if let Err(e) = rust_mem.store_concrete_le_bytes_automap_internal(addr, &data) {
                    log::error!("dropped concrete store at {addr:#x}: {e:?}");
                }
            }
            // Flush symbolic pending stores. Same "shouldn't happen" arm as
            // `flush_stores`: SILENT(cat-c), a width VEX cannot produce.
            for (addr, bv) in self.pending_symbolic_stores.drain() {
                if let Err(e) = rust_mem.install_symbolic_value(addr, bv) {
                    log::error!("dropped symbolic store at {addr:#x}: {e:?}");
                }
            }
            self.all_flushed_stores.clear();
            self.all_flushed_symbolic_stores.clear();
        }
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
    /// Production never un-hooks mid-run (hook sets are rebuilt per step); only `interpreter_tests` removes one (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn remove_hook(&mut self, addr: u64) {
        Arc::make_mut(&mut self.hook_addrs).remove(&addr);
    }

    /// Add an address blocks end before; see `Self::lift_cuts`.
    pub fn add_lift_cut(&mut self, addr: u64) {
        Arc::make_mut(&mut self.lift_cuts).insert(addr);
    }

    /// True when a lift cut lies strictly inside `[addr, addr + size)`, so a
    /// block of that span must be lifted again, ending at the cut.
    pub(super) fn spans_lift_cut(&self, addr: u64, size: u64) -> bool {
        // A distance, as `lift_window` measures one: a cut below `addr` wraps
        // to a huge delta and fails the `< size` test.
        self.lift_cuts.iter().any(|&cut| {
            let delta = cut.wrapping_sub(addr);
            delta > 0 && delta < size
        })
    }

    /// Step over the hook at `addr` once; see `Self::skip_hook_once`.
    pub fn set_skip_hook_once(&mut self, addr: u64) {
        self.skip_hook_once = Some(addr);
    }

    /// Check if an address is hooked.
    pub fn is_hooked(&self, addr: u64) -> bool {
        self.skip_hook_once != Some(addr) && self.hook_addrs.contains(&addr)
    }

    /// Check if we have a cached block at the given address.
    /// Block-cache introspection used only by `smc_tests` / `execution_tests` / `statements_tests` to assert invalidation (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn has_cached_block(&self, addr: u64) -> bool {
        self.block_cache.contains(&addr)
    }

    /// Add a block to the cache.
    /// Direct cache priming used only by the block-cache tests; production populates the cache through `lift_block` (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn cache_block(&mut self, addr: u64, irsb: IRSB) {
        self.block_cache.put(addr, Arc::new(irsb));
    }

    /// Get a block from the cache.
    /// Same as `cache_block` — test-side cache introspection (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
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
    /// `split_state.regs.ip = target` write in `SimSuccessors._categorize_successor`.
    pub fn take_symbolic_ip_at_exit(&mut self) -> Option<RustBV> {
        self.symbolic_ip_at_exit.take()
    }

    /// Fork the interpreter state.
    /// Interpreter forking is unused: states fork via `RustSimState::fork` and a fresh interpreter is built per step. Only the interpreter tests exercise it (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn fork(&self) -> VEXInterpreter<'a> {
        // Clone the calling convention based on its type
        let arch_box = arch_from_vex(self.arch);
        let cc = default_cc_for_arch(arch_box.name());

        VEXInterpreter {
            registers: self.registers.fork(),
            temps: self.temps.clone(),
            ctx: self.ctx,
            pc: self.pc,
            current_insn_addr: self.current_insn_addr,
            current_insn_len: self.current_insn_len,
            hook_addrs: Arc::clone(&self.hook_addrs),
            lift_cuts: Arc::clone(&self.lift_cuts),
            skip_hook_once: self.skip_hook_once,
            arch: self.arch,
            native_lifter: self.native_lifter.clone(),
            block_cache: self.block_cache.clone(), // Share lifted blocks with parent (Arc values = cheap clone)
            deferred_forks: Vec::new(),            // Fresh deferred forks for fork
            deferred_fork_this_step: false,
            block_solver_pushed: false,
            block_forks_asserted: 0,
            config: self.config.clone(),
            next_condition_id: self.next_condition_id,
            concrete_memory: Arc::clone(&self.concrete_memory), // Share concrete memory (read-only)
            loader_object_ranges: Arc::clone(&self.loader_object_ranges),
            concretizer: self.concretizer.clone(), // Share concretizer settings
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
            pending_syscall: self.pending_syscall,
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

    /// Set (or, with `None`, clear) the native in-process libVEX lifter.
    ///
    /// When set, `Self::get_or_lift_block` attempts an FFI lift before the
    /// Python `lift_block` callback, falling back cleanly on any miss
    /// (missing/symbolic bytes, or a libVEX error).
    pub fn set_native_lifter(&mut self, lifter: Option<Arc<dyn angr_vex::VEXLifter>>) {
        self.native_lifter = lifter;
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
}

test_submod!("interpreter_tests.rs" => interpreter_tests);

test_submod!("smc_tests.rs" => smc_tests);

test_submod!("rm_dispatch_tests.rs" => rm_dispatch_tests);
