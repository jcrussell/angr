//! Rust-native exploration manager for symbolic execution.
//!
//! `RustExplorationManager` provides a Rust-first exploration loop that:
//! - Manages states entirely in Rust (using `RustSimState`)
//! - Processes symbolic branches with deferred forks
//! - Only returns to Python for SimProcedures and syscalls
//! - Implements find/avoid address checking in Rust
//!
//! This achieves ~3x speedup by keeping state management in Rust and
//! minimizing Python callback overhead.

use crate::stash::{
    STASH_ACTIVE, STASH_AVOID, STASH_DEADENDED, STASH_ERRORED, STASH_FOUND, STASH_PRUNED,
    STASH_UNCONSTRAINED, StashManager,
};
use rustc_hash::FxHashMap;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use pyo3::class::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use self::selection_policy::SelectionPolicy;
use crate::arch::{ExtractionError, arch_from_name, default_cc_for_arch};
use crate::callbacks::{DeferredFork, ExecutionConfig, PythonCallbacks, RunResult};
use crate::claripy_bridge::{claripy_to_rustbv, rustbv_to_claripy};
use crate::interpreter::{DCAS_UNSUPPORTED_REASON, ExecutionStats, VECRET_GSPTR_REASON};
use crate::memory::Permission;
use crate::procedures::{NativeProcedureRegistry, ProcOutcome, ProcedureError};
use crate::solver::RustSolverContext;
use crate::state::{RustSimState, StateChanges};
#[cfg(feature = "vex-engine-z3")]
use crate::symbolic::Z3AstPtr;
use crate::symbolic::{RustBV, SymContext};
use crate::syscalls::NativeSyscallRegistry;

use std::cell::Cell;

mod callback_types;
mod constraints;
mod core_outcome;
mod event;
mod execution_env;
mod helpers;
mod memory_config;
mod native_technique;
mod pending_api;
mod profiling;
mod resume;
mod run_loop;
/// Work-stealing scheduler machinery for parallel exploration (angr-1ilq.3).
/// Z3-gated: it transports [`crate::state::StateMigrationPayload`], which only
/// exists with the Z3-backed engine.
#[cfg(feature = "vex-engine-z3")]
mod scheduler;
pub(crate) mod selection_policy;
/// SI-B (angr-1ilq.3 increment 2b'): opt-in shadow probe measuring the real
/// state-migration serde tax. Z3-gated: the scratch thread builds states into
/// its own `z3::Context`, so it only exists with the Z3-backed engine.
#[cfg(feature = "vex-engine-z3")]
mod shadow_probe;
mod state_api;
mod state_id;
mod state_lifecycle;
mod stats_api;
mod step_core;
mod stepping;

use self::constraints::{ConstraintSolver, ConstraintTracker};
use self::execution_env::ExecutionEnvironment;
use self::memory_config::MemoryConfiguration;
use self::profiling::ProfilingCollector;
use self::state_id::StateId;
use self::stepping::StepError;

// Thread-local stepping state ID, accessible from callbacks without borrow conflicts.
thread_local! {
    static STEPPING_STATE_ID: Cell<Option<u64>> = const { Cell::new(None) };
}

// Shared return-type aliases for state-inspection PyO3 methods. The thin
// pyclass wrappers in this file and the pub(crate) bodies in `state_api.rs`
// must use identical signatures — keep the aliases here so both impls see
// them via `use super::*`.
pub(crate) type HeapMetadataReturn = (Vec<(u64, u64)>, Vec<u64>);
pub(crate) type OpenFdInfo = (u32, String, u64, u32, usize, bool);
pub(crate) type InspectionEventInfo = (u8, String, u64, u32, u64);

/// Get the current stepping state ID (safe to call from callbacks).
#[pyfunction]
pub fn get_stepping_state_id() -> Option<u64> {
    STEPPING_STATE_ID.with(std::cell::Cell::get)
}

pub use self::callback_types::CallbackReason;
pub(crate) use self::callback_types::PendingCallback;
pub(crate) use self::callback_types::apply_deferred_fork_constraints;
pub use self::event::ExplorationEvent;
pub(crate) use self::native_technique::NativeTechnique;

/// Rust-native exploration manager.
///
/// Manages states entirely in Rust with O(1) forking.
/// Returns to Python only for SimProcedures, syscalls, and predicates.
#[pyclass(unsendable)]
pub struct RustExplorationManager {
    /// Per-binary execution environment: arch metadata, calling convention,
    /// binary regions, block cache, endianness, history cap.
    pub(crate) environment: ExecutionEnvironment,
    /// Stash system (mirrors Python's).
    pub(crate) sm: StashManager,
    /// Find addresses.
    pub(crate) find_addrs: HashSet<u64>,
    /// Avoid addresses.
    pub(crate) avoid_addrs: HashSet<u64>,
    /// Union of find + avoid addresses. Passed to the VEX interpreter so it
    /// breaks its internal block chain when it reaches one, instead of running
    /// past an address-based target (angr-027h). Kept in sync by
    /// `set_find_addrs` / `set_avoid_addrs`.
    pub(crate) stop_addrs: HashSet<u64>,
    /// Block-granular stepping mode (angr-bmyx). When `true`, the VEX
    /// interpreter breaks its internal block chain at *every* basic-block
    /// boundary, so each `step(n=1)` advances exactly one block and every
    /// interior pc becomes observable at a step boundary — matching Python
    /// angr's block-granular `step()`. This is what makes idioms like CADET
    /// solve.py phase 3 (`while True: sm.step(); break if any active.addr ==
    /// TARGET`) reach a mid-path target that the chained interpreter would
    /// otherwise run straight through. Default `false`: chaining stays on so
    /// `explore()`/benchmark step paths keep their throughput. Toggled via
    /// `set_block_granular`.
    pub(crate) block_granular: bool,
    /// When true, loop-exit deferred forks accumulated at an
    /// `UnconstrainedJump` are MATERIALIZED (eagerly, force_eager) and routed
    /// to the active stash instead of being dropped while
    /// `exec_config.use_deferred_forks` is true. This keeps the active stash
    /// non-empty so a bare step-loop (CADET solve.py phase 3) can keep making
    /// progress toward a target behind a symbolic loop exit instead of
    /// collapsing to `active_empty` and spinning forever. Default `false`:
    /// the drop behavior is what `explore()`'s two-phase eager retry
    /// (angr-027h) relies on to detect `active_empty` and re-seed in eager
    /// mode, so this MUST stay opt-in. Combine with `block_granular` so the
    /// egg block is observable before the materialized subtree explodes.
    /// Toggled via `set_materialize_unconstrained_forks` (angr-ckdy).
    pub(crate) materialize_unconstrained_forks: bool,
    /// Whether find condition has callable predicates.
    pub(crate) find_needs_python: bool,
    /// Whether avoid condition has callable predicates.
    pub(crate) avoid_needs_python: bool,
    /// Execution config for deferred forks.
    pub(crate) exec_config: ExecutionConfig,
    /// Python callbacks for memory/lifting.
    pub(crate) callbacks: Option<PythonCallbacks>,
    /// Hook addresses.
    pub(crate) hooks: HashSet<u64>,
    /// SimProcedures: address -> (name, num_args, no_return).
    pub(crate) simprocedures: HashMap<u64, (String, usize, bool)>,
    /// States waiting for a Python callback result, keyed by `StateId`.
    ///
    /// In the single-threaded engine this holds exactly one live entry at a
    /// time (the state currently parked across a Python callback). It is a map
    /// rather than an `Option` so multiple outstanding callbacks can coexist
    /// once the work-stealing scheduler (angr-1ilq.3) drives several workers,
    /// each addressing its own pending state explicitly by `state_id`.
    pub(crate) pending_callbacks: FxHashMap<StateId, PendingCallback>,
    /// ID of the state currently being stepped (for Python callbacks to identify)
    pub(crate) current_stepping_state_id: Option<StateId>,
    /// Total steps executed.
    pub(crate) steps: u64,
    /// Error log: (addr, message, state_id).
    pub(crate) errors: Vec<(u64, String, u64)>,
    /// Number of finds required before stopping.
    pub(crate) num_find: usize,
    /// Maximum steps per run iteration.
    pub(crate) max_steps_per_run: u32,
    /// Native procedure registry.
    ///
    /// `Arc`-wrapped so the persistent parallel pool (angr-vh834 Work Item 2)
    /// can snapshot it into a per-wave `WaveJob` with an O(1) `Arc::clone`.
    /// Mutated at setup time via `Arc::make_mut` (refcount is 1 outside a wave,
    /// so that is an in-place mutation, not a deep clone); read via `Deref`.
    pub(crate) native_procedures: Arc<NativeProcedureRegistry>,
    /// Native syscall registry (skip Python `_handle_syscall_callback` round-trip).
    /// `Arc`-wrapped for the same per-wave O(1) snapshot; never mutated after
    /// construction, so no `Arc::make_mut` sites exist for it.
    pub(crate) native_syscalls: Arc<NativeSyscallRegistry>,
    /// VEX fallback tracking: count and unique addresses.
    pub(crate) vex_fallback_count: u64,
    pub(crate) vex_fallback_addrs: HashMap<u64, String>,
    /// Visibility counter for `IRStmt::CAS` double-CAS (cmpxchg16b) fallbacks.
    /// Incremented alongside `vex_fallback_count` whenever the reason carries
    /// `DCAS_UNSUPPORTED_REASON`. Surfaced via `stats()` and
    /// `get_fallback_stats()` so DCAS-driven deadends are diagnosable.
    pub(crate) dcas_unsupported_count: u64,
    /// Visibility counter for `IRExpr::VECRET`/`IRExpr::GSPTR` Python
    /// fallbacks. Incremented alongside `vex_fallback_count` whenever the
    /// reason carries `VECRET_GSPTR_REASON`. angr-2iow: prevalence drives
    /// whether to implement these natively (non-zero) or downgrade to a
    /// documented limitation (corpus-wide zero).
    pub(crate) vecret_gsptr_fallback_count: u64,
    /// Total count of SimProcedure invocations that were dispatched to the
    /// Python `_handle_simprocedure_callback` (rather than handled natively).
    /// This includes: native handler missing, native handler returned `Err`,
    /// and addresses inside the binary (user-placed Python hooks). Surfaced
    /// via `stats()` so a regression that flips a hot procedure off the
    /// native path is visible without rebuilding.
    pub(crate) simprocedure_python_fallback_count: u64,
    /// Per-procedure breakdown of `simprocedure_python_fallback_count`. Keyed
    /// by the SimProcedure name (as registered via `add_simprocedure`).
    /// Surfaced via `stats()` and `get_fallback_stats()` under
    /// `simprocedure_fallback_by_name` so we can see which native procedures
    /// to implement next without running a separate profiling pass.
    pub(crate) simprocedure_fallback_by_name: HashMap<String, u64>,
    /// Total count of syscalls dispatched to the Python
    /// `_handle_syscall_callback` (rather than handled by `NativeSyscall`).
    /// Includes: no native handler registered for (arch, num) and native
    /// handler returned `Err`. Surfaced via `stats()`.
    pub(crate) syscall_python_fallback_count: u64,
    /// Per-syscall-number breakdown of `syscall_python_fallback_count`.
    /// Key is the syscall number (architecture-specific), or `-1` when the
    /// syscall register was symbolic at dispatch time (the native registry is
    /// skipped without consulting any concrete number). Used by `stats()` to
    /// surface which native handlers would close the next gap.
    pub(crate) syscall_python_fallback_by_num: HashMap<i64, u64>,
    /// Total count of syscalls handled by a native `NativeSyscall` handler
    /// (the fast path that never round-trips to Python). Incremented only when
    /// `handler.call` returns `Ok` — a native handler that returns `Err` falls
    /// through to the Python callback and is counted under
    /// `syscall_python_fallback_count` instead. Surfaced via `stats()`.
    pub(crate) syscall_native_count: u64,
    /// Per-syscall-number breakdown of `syscall_native_count`. Key is the
    /// syscall number (architecture- or DECREE/CGC-specific). Lets `stats()`
    /// confirm which native handlers actually fired on a workload.
    pub(crate) syscall_native_by_num: HashMap<i64, u64>,
    /// State IDs that have already produced a DCAS warning. We log the first
    /// DCAS hit per state to avoid spamming the log on tight DCAS loops.
    pub(crate) dcas_warned_states: HashSet<u64>,
    /// Stack of (address, expiry_step) for zero-length hook skip tracking.
    /// Each entry represents an address to skip, valid until the specified step.
    /// This prevents infinite loops when a hook with length=0 runs and
    /// returns to the same address. Stack-based to handle nested hooks.
    pub(crate) skip_hook_stack: Vec<(u64, u64)>,
    // state_roots is now in self.sm (StashManager)
    /// Active-state selection / fork-insertion policy (angr-a32jl.1).
    /// Governs which active state is stepped next and where new forks land.
    /// Defaults to [`Fifo`] (BFS); `set_state_selection_lifo` swaps in [`Lifo`]
    /// (DFS). Replaces the former `use_lifo: bool`.
    pub(crate) policy: Arc<dyn SelectionPolicy>,
    /// Solver configuration: lazy_solves flag and Z3 timeout.
    pub(crate) constraint_solver: ConstraintSolver,
    /// Memory and VEX configuration: zero-fill, concretizer, vex opt levels.
    pub(crate) memory_config: MemoryConfiguration,
    /// Maximum number of states in the active stash. None = unlimited.
    pub(crate) max_active_states: Option<usize>,
    /// One-shot guard so we emit a single `warn!` (not a per-state `debug!`)
    /// the first time `max_active_states` prunes a state — makes a runaway
    /// explosion visible in the log without spamming on tight fork loops.
    pub(crate) max_active_warned: bool,
    /// Cumulative count of loop-exit deferred forks DROPPED at an
    /// `UnconstrainedJump` while `exec_config.use_deferred_forks` is true
    /// (the angr-027h phase-1 behavior). Non-zero means a step-driven run
    /// reached `active_empty` only because egg-reaching loop-exit forks were
    /// discarded — the precise trigger Python's `_maybe_step_eager_retry`
    /// uses to flip to eager mode and re-seed (angr-ckdy). Always tracked
    /// (not gated on profiling) so the bare `step()` loop in CADET's solve.py
    /// phase 3 can read it. Read (non-resetting) via
    /// `deferred_forks_dropped()`.
    pub(crate) deferred_forks_dropped: u64,
    // drop_terminal_states, avoided_count, pruned_count, deadended_count
    // are now in self.sm (StashManager)
    /// Native exploration techniques that run entirely in Rust.
    pub(crate) native_techniques: Vec<NativeTechnique>,
    /// Constraint-related per-run tracking: uniqueness filter sets and
    /// the find/avoid predicate skip lists.
    pub(crate) constraint_tracker: ConstraintTracker,
    /// Profiling state: enable flag, per-step `ExecutionStats`, and
    /// `NativeProcStats` accumulator.
    pub(crate) profiling: ProfilingCollector,
    // state_index is now in self.sm (StashManager)
    /// DS-instr (angr-11djq.16): state-reconvergence instrumentation.
    /// Cumulative count, summed over per-step samples of the active stash, of
    /// active states that share a `(pc, callstack)` key with ≥1 other active
    /// state at the same step. High values mean lots of states reconverging on
    /// the same program point — the signal that directed-search pruning
    /// (.14.2) and automatic state merging (.10) have real headroom; near-zero
    /// (e.g. deep divergent paths) means they have nothing to merge. Counters
    /// only; no behaviour change. Surfaced via `stats()`.
    pub(crate) reconvergence_collision_states: u64,
    /// Cumulative sum of `active_count` over the same per-step samples — the
    /// denominator for `reconvergence_rate` (collision_states / observed).
    pub(crate) reconvergence_active_observed: u64,
    /// Number of per-step samples taken (steps where the active stash was
    /// non-empty). Lets a consumer distinguish "no collisions over many steps"
    /// from "barely sampled".
    pub(crate) reconvergence_samples: u64,
    /// Largest `(pc, callstack)`-sharing group size observed in a single step.
    pub(crate) reconvergence_max_group: u64,
    /// angr-panhl.1 (Phase 0 kill-gate): number of hypothetical workers used by
    /// the work-stealing migration model. Read once from `ANGR_PARALLEL_WORKERS`
    /// at construction (default 4). See `record_migration_sample`.
    pub(crate) parallel_num_workers: usize,
    /// angr-1ilq.3 increment 2a: number of REAL work-stealing worker threads
    /// for the parallel run-loop coordinator. DISTINCT from `parallel_num_workers`
    /// (the panhl.1 migration *model*). Read once from `RUST_PARALLEL_WORKERS` at
    /// construction (default 1 = single-threaded, no behaviour change). The
    /// coordinator path is wired in a later increment; at default 1 the run loop
    /// takes the verbatim single-threaded path.
    pub(crate) parallel_real_workers: usize,
    /// Cumulative count of modelled work-stealing migrations ("steals"): per
    /// dispatched task, +1 when the state's home worker has a backlog while
    /// another worker is idle. The kill-gate's <10/bench migration target.
    pub(crate) parallel_migrations: u64,
    /// Number of dispatched tasks sampled (≈ steps where a state was stepped).
    /// Denominator for the per-task duration the kill-gate's >100ms target uses
    /// (avg task duration = wall_time / parallel_tasks, computed Python-side).
    pub(crate) parallel_tasks: u64,
    /// angr-vh834 steady-state redesign (Phase 1) — duplex-protocol accounting,
    /// folded from `SchedulerStats` after each wave. `parallel_reattaches` counts
    /// injector-steal reattaches (observability of the migration path);
    /// `parallel_bounce_roundtrips` and `parallel_resume_reinjects` are wired in
    /// later phases and stay 0 for now.
    pub(crate) parallel_reattaches: u64,
    pub(crate) parallel_bounce_roundtrips: u64,
    pub(crate) parallel_resume_reinjects: u64,
    /// Largest schedulable frontier width (active stash + dispatched state)
    /// observed in a single step — the independent-state count over time.
    pub(crate) parallel_max_active_width: u64,
    /// angr-panhl.3 (concurrent-width audit): step-weighted histogram of the
    /// schedulable frontier width. Peak width (`parallel_max_active_width`) is
    /// misleading — one brief fork to width 5 looks identical to a sustained
    /// width-5 sweep. This counts dispatched-task samples per width bucket so
    /// the *sustained* (step-weighted) width is recoverable: the fraction of
    /// steps with ≥2/≥3/≥5 concurrent states is the true parallel-favorability
    /// signal panhl.1 omitted. Buckets: [width==1, ==2, 3–4, 5–8, ≥9].
    pub(crate) parallel_width_hist: [u64; 5],
    /// angr-op0dn.13.9: dispatches per real worker thread, folded from
    /// `SchedulerStats::worker_dispatches` after each wave / session. Empty on
    /// the serial path (the modelled homes live in `parallel_worker_of`);
    /// under the parallel loops its max/min is the load-balance column the S7
    /// find-all gate reports.
    pub(crate) parallel_worker_dispatch: Vec<u64>,
    /// Sticky home-worker assignment per active state id, rebuilt each sample
    /// from the surviving frontier (bounds memory to the active width).
    pub(crate) parallel_worker_of: HashMap<u64, usize>,
    /// SI-B (angr-1ilq.3 increment 2b'): opt-in "shadow probe" that measures the
    /// REAL state-migration serde tax on the live benchmark workload. Read ONCE
    /// from `RUST_PARALLEL_SHADOW_PROBE` at construction (default false). When
    /// false the probe is a no-op and exploration behaviour is byte-identical.
    /// When true, each dispatched state is round-tripped through
    /// `to_serialized` (on the main thread) + `from_serialized` (in a DIFFERENT
    /// Z3 context on a persistent scratch thread, faithfully modelling
    /// `StateMigrationPayload::reattach`'s cross-context AST minting) and the
    /// result is discarded — it measures the migration cost, never routes it.
    /// Feeds the 2b' overhead GO/NO-GO gate (SI-C).
    pub(crate) shadow_probe: bool,
    /// Cumulative nanoseconds spent in the shadow-probe migration round-trip:
    /// `to_serialized` time (measured on the main thread) + `from_serialized`
    /// time (measured on the scratch thread). Channel transit time is excluded.
    pub(crate) parallel_shadow_migration_ns: u64,
    /// Number of states round-tripped by the shadow probe (one per dispatch
    /// while the probe is on). Denominator for the per-state migration cost.
    pub(crate) parallel_shadow_migration_states: u64,
    /// Cumulative serialized-envelope bytes produced by the shadow probe across
    /// all round-tripped states. Denominator for the per-state payload size.
    pub(crate) parallel_shadow_migration_bytes: u64,
    /// Lazily-spawned persistent scratch-thread channel endpoints for the
    /// shadow probe: `(send serialized bytes, receive deserialize ns)`. `None`
    /// until the first probe step spawns the thread. The thread owns its own
    /// `z3::Context` for its whole life and exits cleanly when this Sender drops
    /// at manager teardown (channel close ends its `recv` loop) — no JoinHandle
    /// or Drop impl is needed.
    pub(crate) shadow_probe_chan: Option<(
        std::sync::mpsc::Sender<Vec<u8>>,
        std::sync::mpsc::Receiver<u64>,
    )>,
    /// angr-vh834 Phase 5 (M3): bounce states a parallel wave discovered but
    /// could not dispatch yet, because an earlier bounce in the same wave already
    /// surfaced its Python `need_callback` event (only one callback is surfaced
    /// per `run()`). Carried on the manager — NOT pushed back to `STASH_ACTIVE` —
    /// so the next `run()` dispatches them straight through `dispatch_bounce`
    /// instead of letting a worker RE-STEP a state already parked at a hook, which
    /// would re-run `run_post_step_core` and double-fold its fallback counters
    /// (`simprocedure_python_fallback_count`, …). Each entry is
    /// `(bounce state, BounceKind, lineage root)`. Empty in single-threaded mode.
    pub(crate) pending_parallel_bounces: Vec<(RustSimState, self::core_outcome::BounceKind, u64)>,
    /// angr-vh834 Work Item 2: the persistent work-stealing worker pool. `None`
    /// until the first parallel wave lazily spawns it (`run_loop_parallel`); the
    /// N long-lived worker threads each own a Z3 context for the pool's whole
    /// life, so subsequent waves pay no thread-spawn / context-creation tax. Only
    /// ever populated on the `parallel_real_workers >= 2` path; the pool's `Drop`
    /// broadcasts shutdown and joins the workers at manager teardown.
    #[cfg(feature = "vex-engine-z3")]
    pub(crate) parallel_pool: Option<self::scheduler::PersistentPool>,
    /// angr-nkoct steady-state: the live cross-`run()` parallel session, when
    /// one exists. `Some` only while the steady loop is running or parked
    /// across a `need_callback` return (workers keep stepping their local
    /// frontiers while Python services the callback). Finalized — residual
    /// frontier drained back to `STASH_ACTIVE` — on every other event return,
    /// on any exploration-config mutation (`steady_config_guard`), and at
    /// manager teardown.
    #[cfg(feature = "vex-engine-z3")]
    pub(crate) parallel_session: Option<self::run_loop::SteadySession>,
    /// The Python driver's promise that nothing reads or mutates the active
    /// stash between `run()` calls (set per explore loop by rust_manager.py;
    /// default false). One of the steady-state engagement conditions — without
    /// it a live resident frontier would be invisible to driver-side cache
    /// cleanup / state-index rebuilds.
    pub(crate) parallel_frontier_residency: bool,
    /// `RUST_PARALLEL_STEADY` env flag, read once at construction (same
    /// per-manager pattern as `RUST_PARALLEL_WORKERS`, so tests can
    /// monkeypatch it). Opt-in for the steady-state loop during angr-nkoct
    /// phases C-D; flipped to opt-out once the measurement gate passes.
    pub(crate) parallel_steady_env: bool,
    /// Residual live frontier states a cancelled parallel run returned to
    /// `STASH_ACTIVE` instead of dropping (the Bug M1 cancel-drain; both the
    /// wave loop and a steady-session finalize), folded from
    /// `SchedulerStats::residual_drains`.
    pub(crate) parallel_residual_drains: u64,
    /// Post-find speculative steps: work committed on a worker after another
    /// origin already requested cancel (angr-1ilq.8 measure-first), folded from
    /// `SchedulerStats::post_cancel_steps`. Quantifies the num_find=1
    /// speculative waste the find-aware dispatch bead (angr-1ilq.9) targets.
    pub(crate) parallel_post_cancel_steps: u64,
}

/// The manager's entire `#[pymethods]` surface lives in this child module
/// (angr-nbim4.1) to keep `mod.rs` focused on the struct definition and
/// module wiring. See the module doc for the PyO3 single-block constraint.
#[path = "manager_methods.rs"]
mod manager_methods;

/// Steady-state Drop safety (angr-nkoct). `PersistentPool::drop` broadcasts
/// `Shutdown` and JOINS the workers; with a live steady session a worker may be
/// mid-dispatch blocked in `Python::attach` (cold VEX lift), which deadlocks if
/// the dropping thread holds the GIL across the join (pyclass dealloc runs
/// GIL-held). So: cancel the session and wake parked workers so every worker
/// reaches its ctl channel, then release the GIL around the pool teardown. The
/// wave-mode path (no session) keeps the previous plain field-drop behaviour —
/// its workers are always parked between waves, so the join cannot block on
/// the GIL there.
impl Drop for RustExplorationManager {
    fn drop(&mut self) {
        #[cfg(feature = "vex-engine-z3")]
        {
            let had_session = self.parallel_session.is_some();
            if let Some(sess) = self.parallel_session.take() {
                sess.cancel_and_wake(self.parallel_pool.as_ref());
                // `sess` (and its up_rx) drops here; late worker sends fail
                // silently by design.
            }
            if had_session && let Some(pool) = self.parallel_pool.take() {
                Python::attach(|py| py.detach(move || drop(pool)));
            }
        }
    }
}

/// Register the exploration module with Python.
pub fn register_exploration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustExplorationManager>()?;
    m.add_class::<ExplorationEvent>()?;
    m.add_function(pyo3::wrap_pyfunction!(get_stepping_state_id, m)?)?;
    Ok(())
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
