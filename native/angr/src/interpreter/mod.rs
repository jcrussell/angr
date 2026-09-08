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
//! [`ExecutionStats`] in [`execution_stats`], the
//! [`CbExecutionError`]/[`FallbackStrategy`] taxonomy plus its reason markers
//! in [`execution_error`], and the [`BlockResult`]/`StmtResult`/
//! `ConcretizedJump` control-flow results in [`block_result`].
#![deny(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use lru::LruCache;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::arch::{
    AddrOrSymbolic, RegisterFile, arch_from_vex, calling_conventions::CallingConvention,
    default_cc_for_arch,
};
use crate::callbacks::{
    DeferredFork, ExecutionConfig, InspectBit, PythonCallbacks, RunResult, note_inspect_error,
};
use crate::claripy_bridge::{claripy_to_rustbv, is_claripy_ast, try_handle_to_rustbv};
use crate::concretize::{AddressConcretizer, ConcretizationResult, strided_addrs};
use crate::memory::{MemoryError, Permission, SymbolicMemory};
use crate::symbolic::{
    BVOp, RustBV, RustSymbolTable, SymContext, record_mem_load, record_mem_store, record_vex_binop,
    record_vex_qop, record_vex_triop, record_vex_unop,
};
use crate::vex::ccall;
use crate::vex::dirty::{DirtyHelperDispatch, DirtyHelperState};
use crate::vex::ir::{
    IRConst, IRExpr, IRLoadGOp, IROp, IRSB, IRStmt, IRType, JumpKind, TypeEnv, VexArch,
};
use crate::vex::ops::{OpError, VEXOps, iropclass};
use crate::vex::{Endness, VEX_MAX_BYTES, deserialize_irsb};

/// VEX block (IRSB) LRU cache capacity in entries.
///
/// Each cached `IRSB` is Arc-shared, so this caps unique IRSBs simultaneously
/// resident across an interpreter's lifetime. 4096 covers the working set of
/// every tracked bench in `baseline_timings.json`; adjust only with a counter
/// dump showing eviction churn (`mgr.stats()` does not expose hit/miss today;
/// see angr-l4bs).
pub(crate) const BLOCK_CACHE_CAPACITY: usize = 4096;

/// [`BLOCK_CACHE_CAPACITY`] pre-validated as the `NonZeroUsize` every
/// `LruCache::new` call site needs.
///
/// Doing the conversion once in a `const` moves the non-zero proof to compile
/// time: a zero capacity fails the build here instead of surfacing as a runtime
/// `.expect` in each of the constructors that build a block cache
/// (`ExecutionEnv::new`, `VEXInterpreter::new`, and
/// `exploration::scheduler_pool::worker_thread`).
pub(crate) const BLOCK_CACHE_CAPACITY_NZ: std::num::NonZeroUsize =
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
/// carry through every downstream constraint. Beyond this cap both the load
/// (`dispatch_multi_load`) and the store (`dispatch_multi_store`) hand the whole
/// access to Python's memory model, which has purpose-built machinery for
/// wide symbolic accesses. Keeping one constant for both keeps the two paths from
/// drifting apart (angr-sqfj8.68).
pub(crate) const MAX_ITE_ADDRS: usize = 16;

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
pub(crate) mod bv_utils;
mod code_invalidation;
mod concrete_memory;
mod concretize_cache;
mod execution;
mod execution_error;
mod execution_stats;
mod exits;
mod expressions;
mod expressions_inspect;
mod fork_state;
mod pending_store;
mod prefetch;
mod simprocedures;
mod statements;
mod statements_cas;
mod statements_inspect;
mod statements_store;

use block_result::{ConcretizedJump, StmtResult};
use bv_utils::bytes_to_bv;
use pending_store::PendingStoreBuffer;

pub(crate) use block_result::BlockResult;
pub(crate) use concrete_memory::ConcreteMemoryRegion;
pub(crate) use execution_error::{
    CbExecutionError, DCAS_UNSUPPORTED_REASON, DISPATCH_FABRICATE_REASON, FallbackStrategy,
    VECRET_GSPTR_REASON,
};
pub(crate) use execution_stats::ExecutionStats;
pub(crate) use simprocedures::SimProcedureInfo;

/// Full state snapshot at a symbolic branch point.
/// Used by deferred forks to create correct alternate-path states
/// with solver, registers, and memory from the branch point.
pub struct BranchSnapshot {
    pub solver: SymContext,
    pub registers: RegisterFile,
    pub memory: Option<SymbolicMemory>,
}

/// Callback-aware VEX IR interpreter.
///
/// This interpreter uses Python callbacks for memory and register access,
/// allowing it to work with angr's symbolic memory model.
///
/// When `use_rust_memory` is true, the interpreter uses `rust_memory` for
/// memory operations, falling back to Python callbacks only for unmapped pages.
/// This provides significant performance improvement for memory-intensive code.
pub(crate) struct VEXInterpreter<'a> {
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
    /// Address concretizer for handling symbolic addresses.
    concretizer: AddressConcretizer,
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
    /// Per-state scratch space for stateful dirty helpers (the simulated TSC).
    /// Seeded from `RustSimState::tsc_counter` before the step and written
    /// back after it, so RDTSC is deterministic per execution path instead of
    /// depending on a process-wide counter (angr-9ke6b.173).
    pub dirty_helper_state: DirtyHelperState,
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
    /// Create a new little-endian callback-aware interpreter.
    ///
    /// Test-only: the two production entry points ([`crate::engine`]'s
    /// single-block exec and `exploration/step_core.rs`) both know their
    /// target's byte order and go through [`Self::with_config_endian`]
    /// (angr-21cz6).
    #[cfg(test)]
    pub(crate) fn new(arch: VexArch, ctx: &'a SymContext) -> Self {
        Self::with_config(arch, ctx, ExecutionConfig::default())
    }

    /// Create a new callback-aware interpreter with custom config.
    ///
    /// Little-endian by default; see [`Self::with_config_endian`] for why that
    /// is a caller-supplied fact rather than one derivable from `arch`.
    pub(crate) fn with_config(arch: VexArch, ctx: &'a SymContext, config: ExecutionConfig) -> Self {
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
    pub(crate) fn with_config_endian(
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
            block_cache: LruCache::new(BLOCK_CACHE_CAPACITY_NZ),
            deferred_forks: Vec::new(),
            deferred_fork_this_step: false,
            block_solver_pushed: false,
            block_forks_asserted: 0,
            config,
            next_condition_id: 0,
            concrete_memory: Arc::new(Vec::new()),
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
            dirty_helper_state: DirtyHelperState::default(),
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
    pub(crate) fn set_profiling(&mut self, enabled: bool) {
        self.profiling_enabled = enabled;
        if enabled {
            self.stats.reset();
        }
    }

    /// Get mutable execution statistics.
    pub(crate) fn stats_mut(&mut self) -> &mut ExecutionStats {
        &mut self.stats
    }

    /// Take the execution statistics, replacing with default.
    pub(crate) fn take_stats(&mut self) -> ExecutionStats {
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
    ///
    /// Bottom rung of the load-resolution ladder: every caller arrives here
    /// through `load_concrete_addr` in `expressions.rs`, which has already
    /// missed the pending/flushed store buffers and the prefetch and
    /// concrete-memory caches. `synthesize_unservable_load` above gets the
    /// last word before the GIL crossing.
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

    /// Set the execution configuration.
    /// Production builds the interpreter with its final `ExecutionConfig`; only `exits_tests` swaps one in afterwards (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn set_config(&mut self, config: ExecutionConfig) {
        self.config = config;
    }

    /// Get the next condition ID.
    fn next_cond_id(&mut self) -> u64 {
        let id = self.next_condition_id;
        self.next_condition_id += 1;
        id
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
                    // SILENT(cat-c): a non-byte-multiple width cannot be
                    // placed in byte-addressed memory at all. VEX stores are
                    // always byte-multiple (Ity_I8 and wider), so this is a
                    // "shouldn't happen" arm — log loudly and keep the value
                    // in all_flushed_symbolic_stores so the block-boundary
                    // path still sees it, mirroring the concrete-store arm
                    // above.
                    if let Err(e) = rust_mem.import_symbolic_value(addr, bv.clone(), None) {
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
    pub(crate) fn flush_stores_to_rust_memory(&mut self) {
        if let Some(ref mut rust_mem) = self.rust_memory {
            // Flush concrete pending stores
            for (addr, data) in self.pending_stores.drain() {
                if let Err(e) = rust_mem.store_concrete_le_bytes_automap_internal(addr, &data) {
                    log::error!("dropped concrete store at {addr:#x}: {e:?}");
                }
            }
            // Flush symbolic pending stores. Same "shouldn't happen" arm as
            // `flush_stores`: SILENT(cat-c), a width VEX cannot produce.
            for (addr, bv) in self.pending_symbolic_stores.drain() {
                if let Err(e) = rust_mem.import_symbolic_value(addr, bv, None) {
                    log::error!("dropped symbolic store at {addr:#x}: {e:?}");
                }
            }
            self.all_flushed_stores.clear();
            self.all_flushed_symbolic_stores.clear();
        }
    }

    /// Set the program counter.
    pub(crate) fn set_pc(&mut self, addr: u64) {
        self.pc = addr;
        let pc_bv = RustBV::concrete(addr as u128, self.registers.arch().bits());
        self.registers.set_ip(pc_bv);
    }

    /// Get the program counter.
    #[inline]
    pub(crate) fn get_pc(&self) -> u64 {
        self.pc
    }

    /// Add a hook address.
    pub(crate) fn add_hook(&mut self, addr: u64) {
        Arc::make_mut(&mut self.hook_addrs).insert(addr);
    }

    /// Remove a hook address.
    /// Production never un-hooks mid-run (hook sets are rebuilt per step); only `interpreter_tests` removes one (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn remove_hook(&mut self, addr: u64) {
        Arc::make_mut(&mut self.hook_addrs).remove(&addr);
    }

    /// Check if an address is hooked.
    pub(crate) fn is_hooked(&self, addr: u64) -> bool {
        self.hook_addrs.contains(&addr)
    }

    /// Check if we have a cached block at the given address.
    /// Block-cache introspection used only by `smc_tests` / `execution_tests` / `statements_tests` to assert invalidation (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn has_cached_block(&self, addr: u64) -> bool {
        self.block_cache.contains(&addr)
    }

    /// Add a block to the cache.
    /// Direct cache priming used only by the block-cache tests; production populates the cache through `lift_block` (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn cache_block(&mut self, addr: u64, irsb: IRSB) {
        self.block_cache.put(addr, Arc::new(irsb));
    }

    /// Get a block from the cache.
    /// Same as `cache_block` — test-side cache introspection (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn get_cached_block(&mut self, addr: u64) -> Option<&IRSB> {
        self.block_cache.get(&addr).map(std::convert::AsRef::as_ref)
    }

    /// Swap in a shared block cache, returning the interpreter's current cache.
    pub(crate) fn swap_block_cache(
        &mut self,
        cache: LruCache<u64, Arc<IRSB>>,
    ) -> LruCache<u64, Arc<IRSB>> {
        std::mem::replace(&mut self.block_cache, cache)
    }

    /// Take the symbolic IP expression recorded at the most recent default
    /// exit (set only when `keep_ip_symbolic` is enabled and the exit's `next`
    /// was symbolic). The internal slot is cleared. Mirrors Python's
    /// `split_state.regs.ip = target` write at engines/successors.py:328.
    pub(crate) fn take_symbolic_ip_at_exit(&mut self) -> Option<RustBV> {
        self.symbolic_ip_at_exit.take()
    }

    /// Fork the interpreter state.
    /// Interpreter forking is unused: states fork via `RustSimState::fork` and a fresh interpreter is built per step. Only the interpreter tests exercise it (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn fork(&self) -> VEXInterpreter<'a> {
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
            deferred_forks: Vec::new(),            // Fresh deferred forks for fork
            deferred_fork_this_step: false,
            block_solver_pushed: false,
            block_forks_asserted: 0,
            config: self.config.clone(),
            next_condition_id: self.next_condition_id,
            concrete_memory: Arc::clone(&self.concrete_memory), // Share concrete memory (read-only)
            concretizer: self.concretizer.clone(),              // Share concretizer settings
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
            // Stateful helper scratch (TSC) DOES carry over: a forked path
            // continues the parent's timeline rather than restarting it.
            dirty_helper_state: self.dirty_helper_state.clone(),
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

    /// Enable or disable native (in-process) libVEX cold-block lifting.
    ///
    /// When enabled, [`Self::get_or_lift_block`] attempts an FFI lift via
    /// `NativeLibVEXLifter` before the Python `lift_block` callback, falling
    /// back cleanly on any miss (unsupported arch, missing/symbolic bytes, or
    /// a libVEX error). Requires the `libvex-ffi` build feature; a no-op stub
    /// exists for the default build so Python wiring compiles unconditionally.
    #[cfg(feature = "libvex-ffi")]
    pub(crate) fn set_native_lift_enabled(&mut self, enabled: bool) {
        self.native_lift_enabled = enabled;
    }

    /// No-op stub when the `libvex-ffi` feature is not compiled in.
    #[cfg(not(feature = "libvex-ffi"))]
    pub(crate) fn set_native_lift_enabled(&mut self, _enabled: bool) {}

    /// Set the Rust memory instance.
    pub(crate) fn set_rust_memory(&mut self, memory: SymbolicMemory) {
        self.rust_memory = Some(memory);
        self.use_rust_memory = true;
    }

    /// Take the Rust memory instance (for transferring to engine).
    pub(crate) fn take_rust_memory(&mut self) -> Option<SymbolicMemory> {
        self.rust_memory.take()
    }
}

test_submod!("interpreter_tests.rs" => interpreter_tests);

test_submod!("smc_tests.rs" => smc_tests);
