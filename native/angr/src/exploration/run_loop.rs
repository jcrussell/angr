//! Main exploration run loop.
//!
//! `RustExplorationManager::run_loop` is a thin **driver**: it owns the loop
//! frame (pop a state from the active stash, the I8 termination checks, and the
//! post-step bookkeeping) and delegates the entire per-state body to
//! `step_one`, matching on the `StepOutcome` it returns to route results.
//!
//! `step_one` is the **single stepping decision point**: it evaluates find/avoid
//! predicates, dispatches SimProcedures (native fast path or Python fallback),
//! steps the interpreter, handles deferred forks at find/avoid addresses, and
//! classifies the result into a `StepOutcome`. Extracting it (bead 1ilq.6) gives
//! the work-stealing scheduler (`scheduler.rs`) one place to share, and keeps the
//! driver small. It is a pure refactor — behavior is byte-identical to the former
//! monolithic `run_loop`.
//!
//! Note: `step_one` is `&mut self` (it mutates stashes, counters, profiling, and
//! `pending_callback`), so a `Send + Sync` scheduler worker cannot call it
//! directly — reconciling that is the 1ilq.7 GIL-strategy spike. This bead only
//! lands the seam.
//!
//! The pyclass-facing `run` thin wrapper lives in `mod.rs` and just calls
//! `self.run_loop(py, n)`. PyO3 0.27.2 without `multiple-pymethods` only
//! permits a single `#[pymethods]` impl per class, so the body is extracted
//! here as `pub(crate)` methods on `RustExplorationManager`, mirroring the
//! `helpers.rs` / `stepping.rs` extension-impl pattern used elsewhere in
//! `exploration/`.
//!
//! **Invariant I8 (cross-mixin termination, mirror of
//! rust_manager.py:98):** the run loop must terminate on EITHER (a)
//! `found_count() >= num_find` (checked at the top of every iteration),
//! OR (b) the active stash exhausting itself (`pop_*` returns `None`,
//! emitting an `active_empty` event). `found_count()` covers both
//! Rust-native finds (forks routed to the found stash by the address
//! check) and Python-predicate-derived finds (added via the need_callback
//! resume path). The earlier predicate-only termination check infinite-
//! looped when `find=int` was combined with a non-predicate technique
//! like DFS — the technique made `_active_techniques` non-empty, routing
//! through the Python predicate path, which never saw the Rust find.
//! See module-level I8 in `state.rs`.

use super::*;

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use lru::LruCache;

use super::core_outcome::{
    BounceKind, CoreReturn, ParallelProfiling, PendingBounce, PostStepInputs,
    materialize_bounce_forks, run_post_step_core,
};
use super::scheduler::{
    CancelToken, PersistentPool, ProcessFn, SchedulerCounters, TaskOutcome,
    TerminalDisposition as SchedDisposition, TerminalSummary, WaveJob,
};
use super::step_core::run_interpreter_step_core;
use crate::state::StateMigrationPayload;
use crate::vex::IRSB;

/// Materialized-terminal disposition the parallel worker stamps into
/// [`ParallelShared::kind_map`] so the coordinator can route a state recovered
/// across the `thread::scope` join (which arrives as an untagged
/// [`StateMigrationPayload`]). Found states go to `STASH_FOUND`; `Unconstrained`
/// to that stash; `Bounce` carries the [`BounceKind`] the coordinator replays
/// through `dispatch_bounce` (with empty deferred-fork data — the worker already
/// materialized those forks locally).
#[derive(Clone)]
pub(crate) enum MatKind {
    Found,
    Unconstrained,
    Bounce(BounceKind),
    /// A still-live frontier state drained back across the cancel boundary
    /// (Phase 3's cancel-drain). Added now so the enum is stable for the
    /// steady-state coordinator; not produced by any path yet.
    #[allow(dead_code)]
    ActiveResidual,
}

// ---------------------------------------------------------------------------
// Steady-state duplex protocol (angr-vh834 redesign, Phase 1 scaffolding).
//
// PURE SCAFFOLDING: these types define the coordinator<->worker message shapes
// and the persistent per-run session the steady-state loop will own. Nothing
// below is wired into `run_loop_parallel` yet (Phase 2+). They are gated with
// `#[allow(dead_code)]` and exist mainly to PROVE — via the compile-time
// assertion at the end of this block — that the redesign's shared/session type
// (`RunSession`) is `Send + Sync` and its channel payloads (`WorkerUp` /
// `WorkerCtl`) are `Send`, before any behaviour is built on them.
// ---------------------------------------------------------------------------

/// Downstream control message: coordinator -> worker.
#[allow(dead_code)]
pub(crate) enum WorkerCtl {
    /// Begin/continue processing against a shared run session.
    Run(Arc<RunSession>),
    /// Resume a set of states migrated back from the coordinator (e.g. after a
    /// Python bounce round-trip).
    Resume(Vec<StateMigrationPayload>),
    /// Stop pulling work at the next task boundary and report `Paused`.
    Pause,
    /// Terminate the worker thread.
    Shutdown,
}

/// Upstream report message: worker -> coordinator.
#[allow(dead_code)]
pub(crate) enum WorkerUp {
    /// A materialized terminal the coordinator must route (found / unconstrained
    /// / bounce), tagged with its [`MatKind`] and lineage root.
    Terminal {
        payload: StateMigrationPayload,
        kind: MatKind,
        root: u64,
    },
    /// Per-step manager-level counters to fold into the manager after the wave.
    Counters(super::core_outcome::CoreCounters),
    /// Acknowledgement of a `Pause` at a task boundary.
    Paused { worker_id: usize },
    /// The worker observed global quiescence (its view of no outstanding work).
    Quiesced { worker_id: usize },
}

/// The persistent per-run session shared (by `Arc`) across all steady-state
/// workers. Interior-mutable throughout so the `Fn + Sync` worker body can touch
/// it without `&mut`. The `Send + Sync` proof below is the whole point of Phase
/// 1: it certifies the redesign's central shared type is thread-safe before the
/// steady-state loop is built on it.
#[allow(dead_code)]
pub(crate) struct RunSession {
    /// Shared work queue of migratable states.
    injector: crossbeam_deque::Injector<StateMigrationPayload>,
    /// Count of dispatched-but-not-yet-completed tasks (quiescence detector).
    outstanding: AtomicUsize,
    /// Cooperative cancellation, checked at task boundaries.
    cancel: CancelToken,
    /// Number of workers currently blocked waiting for work (starvation signal).
    idle_workers: AtomicUsize,
    /// Duplex-protocol accounting (reattaches / bounce round-trips / resume
    /// reinjects), folded into the manager after the run.
    counters: SchedulerCounters,
    /// Upstream channel each worker reports on.
    up_tx: std::sync::mpsc::Sender<WorkerUp>,
    /// The per-state processor invoked once per dispatched state.
    process: Box<ProcessFn>,
}

// Compile-time proof (angr-vh834 Phase 1): the steady-state session is
// `Send + Sync` and both channel payloads are `Send`. If any of these fail to
// compile, a field/variant type is not thread-safe and the redesign must adjust
// that type — do NOT paper over it with `unsafe`.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_send<T: Send>() {}
    assert_send_sync::<RunSession>();
    assert_send::<WorkerUp>();
    assert_send::<WorkerCtl>();
};

/// The find/avoid-checkable target address a materialized bounce carries.
///
/// Mirrors single-threaded `step_one`'s NeedCallback special case, which only
/// inspects `CallbackReason::SimProcedure { addr, .. }`. A `Hook` bounce and a
/// `SimProcedurePython` bounce are the two `BounceKind`s `dispatch_bounce` turns
/// into a `SimProcedure` callback reason, so only those are eligible for the
/// find/avoid short-circuit (Bug C1); every other bounce kind returns `None` and
/// always falls through to a real Python bounce, exactly as single-threaded does.
fn bounce_target_addr(kind: &BounceKind) -> Option<u64> {
    match kind {
        BounceKind::Hook { addr } | BounceKind::SimProcedurePython { addr, .. } => Some(*addr),
        _ => None,
    }
}

/// Per-wave state shared (by `&`) across all scheduler workers. Every field is
/// interior-mutable (atomics / `Mutex`) so the `Fn + Sync` worker closure can
/// touch it without `&mut`.
struct ParallelShared {
    /// `state_id -> lineage root`. Seeded with each drained active state's
    /// `root_or_self`; workers insert every successor / materialized terminal so
    /// the coordinator can replay `sm.set_root` and descendants inherit the root
    /// across the local re-dispatch + migration boundary.
    root_map: Mutex<FxHashMap<u64, u64>>,
    /// `state_id -> MatKind` for materialized terminals (found / unconstrained /
    /// bounce), so the coordinator routes each recovered payload correctly.
    kind_map: Mutex<FxHashMap<u64, MatKind>>,
    /// Per-step `CoreCounters` the coordinator folds into the manager after the
    /// wave (native-proc / syscall / simproc fallback tallies).
    counters: Mutex<Vec<super::core_outcome::CoreCounters>>,
    /// Running found count (seeded with the manager's current `found_count`);
    /// reaching `num_find` requests cancellation of the whole pool.
    found_counter: AtomicUsize,
    num_find: usize,
    /// angr-vh834 Phase 5 (M2): number of dispatches that actually took an
    /// interpreter step AND produced a `Successors`/`Terminal`-equivalent outcome
    /// (`run_post_step_core` returned `Continue`/`Deadended`/`Errored`/
    /// `Unconstrained`). EXCLUDES pre-step find/avoid routes and `NeedsPython`
    /// bounces — exactly the dispatches single-threaded `step_one` counts with
    /// `self.steps += 1` (Successors/Terminal only; Routed/NeedCallback skip it).
    /// The coordinator folds this into `self.steps`, NOT `stats.dispatches()`
    /// (which over-counts by including pre-step routes and bounces).
    stepped: AtomicUsize,
}

/// GIL-free analogue of `step_one` + `run_post_step_core` run on a scheduler
/// worker thread (angr-vh834 Phase 5). Returns the [`TaskOutcome`] the scheduler
/// routes: live successors stay LOCAL (the f≈0 path), found/unconstrained/bounce
/// terminals are materialized across the join, and dead paths are cheap
/// summaries.
///
/// How it replicates `step_one`:
/// * Pre-step address-based find/avoid (`step_one` ~296-361): PC in `avoid_addrs`
///   → an `Avoided` summary; PC in `find_addrs` + satisfiable → a materialized
///   `Found` (bumping the shared found counter, requesting cancel at `num_find`);
///   PC in `find_addrs` + UNSAT → a `Pruned` summary. Callable predicates never
///   reach here (the coordinator routes them single-threaded).
/// * Otherwise it steps via the now-GIL-free `run_interpreter_step_core` and
///   classifies with `run_post_step_core`, exactly the `step_state_with_skip`
///   pipeline minus `&mut self`. `Continue` successors (incl. native simproc /
///   syscall returns the core dispatched in-thread) stay local; `Deadended` /
///   `Errored` become summaries; `Unconstrained` materializes its main state and
///   keeps its loop-exit forks local; `NeedsPython` materializes the bounce
///   state for the coordinator's `dispatch_bounce` and keeps its deferred forks
///   local (re-materialized in-thread so no path is lost across the `!Send`
///   loose-condition boundary).
#[allow(clippy::too_many_arguments)]
fn parallel_process_state(
    mut state: RustSimState,
    cancel: &CancelToken,
    block_cache: &mut LruCache<u64, std::sync::Arc<IRSB>>,
    ctx: &super::step_core::StepContext,
    callbacks: &PythonCallbacks,
    prof: &ParallelProfiling,
    native_procs: &NativeProcedureRegistry,
    native_syscalls: &NativeSyscallRegistry,
    shared: &ParallelShared,
) -> TaskOutcome {
    let pc = state.pc();
    let id = state.state_id();
    let root_hint = shared
        .root_map
        .lock()
        .expect("root_map poisoned")
        .get(&id)
        .copied()
        .unwrap_or(id);

    // --- Pre-step address-based find / avoid (mirror of step_one). ---
    if ctx.avoid_addrs.contains(&pc) {
        return TaskOutcome::summarized(vec![TerminalSummary::of(
            &state,
            SchedDisposition::Avoided,
        )]);
    }
    if ctx.find_addrs.contains(&pc) {
        if ctx.lazy_solves || state.satisfiable() {
            {
                let mut rm = shared.root_map.lock().expect("root_map poisoned");
                rm.insert(id, root_hint);
            }
            shared
                .kind_map
                .lock()
                .expect("kind_map poisoned")
                .insert(id, MatKind::Found);
            let reached =
                shared.found_counter.fetch_add(1, Ordering::SeqCst) + 1 >= shared.num_find;
            return TaskOutcome {
                continue_states: Vec::new(),
                terminal_states: vec![state],
                terminal_summaries: Vec::new(),
                request_cancel: reached,
            };
        }
        return TaskOutcome::summarized(vec![TerminalSummary::of(
            &state,
            SchedDisposition::Pruned,
        )]);
    }

    if cancel.is_cancelled() {
        // Bail cheaply; the scheduler is winding down.
        return TaskOutcome::summarized(Vec::new());
    }

    // --- Step (PC is not a find/avoid address). ---
    // Warm per-worker block cache (angr-vh834 Work Item 3): `block_cache` is
    // owned by the worker thread and threaded in here, so lifted blocks persist
    // across dispatches AND waves. Cache hits avoid a GIL-serialized re-lift; the
    // warm cache returns identical (pure) `Arc<IRSB>` lifts, so the found set and
    // content fingerprints are unchanged.
    let step = run_interpreter_step_core(ctx, callbacks, &mut state, pc, None, None, block_cache);

    // Fold this step's block-cache hit/miss into the shared profiling accumulator
    // so the warm-cache win surfaces in `mgr.stats()` (block_cache_hits /
    // block_cache_misses). Cache counters are always-on in the interpreter (not
    // gated by profiling), so accumulate unconditionally; the coordinator adds
    // these into `accumulated_stats` via `prof.fold_into` after the wave barrier.
    prof.cache_hit_count
        .fetch_add(step.step_stats.cache_hit_count, Ordering::Relaxed);
    prof.cache_miss_count
        .fetch_add(step.step_stats.cache_miss_count, Ordering::Relaxed);

    // Restore the warm cache: `run_interpreter_step_core` swapped a fresh empty
    // placeholder into `*block_cache` and handed the now-populated cache back in
    // `updated_block_cache`. Write it back so the newly-lifted blocks persist for
    // this worker's next dispatch (mirrors the single-threaded restore in
    // `step_state_with_skip`: `self.environment.block_cache = step.updated_block_cache`).
    *block_cache = step.updated_block_cache;

    // State-update preamble (mirror of step_state_with_skip). Worker drops the
    // rest of the per-call step_stats (no shared accumulation in the MVP).
    if let Some(mem) = step.recovered_memory {
        state.replace_memory(mem);
    }
    state.set_registers(step.new_registers);
    state.set_pc(step.new_pc);
    if let Some(sym_ip) = step.symbolic_ip_at_exit {
        state.set_ip(sym_ip);
    }
    state.set_call_stack(step.new_call_stack);
    state.set_detailed_history(step.new_detailed_history);
    state.add_to_history(state.pc());

    let inputs = PostStepInputs {
        result: step.result,
        deferred_forks: step.deferred_forks,
        last_condition: step.last_condition,
        stored_conditions: step.stored_conditions,
        fork_snapshots: step.fork_snapshots,
    };
    let outcome = run_post_step_core(
        ctx,
        prof,
        native_procs,
        native_syscalls,
        state,
        inputs,
        root_hint,
    );

    // Dead-path side effects (UNSAT pruned forks, no-return deadended main):
    // recorded as cheap summaries; their full symbolic state is not recoverable
    // through the parallel path (the scheduler's selective drop_terminal_states).
    let mut summaries: Vec<TerminalSummary> = Vec::new();
    for s in &outcome.pruned {
        summaries.push(TerminalSummary::of(s, SchedDisposition::Pruned));
    }
    for (s, _stash) in &outcome.terminal_pushes {
        summaries.push(TerminalSummary::of(s, SchedDisposition::Deadended));
    }

    // Fold the step's manager-level counters (deferred to the coordinator).
    shared
        .counters
        .lock()
        .expect("counters poisoned")
        .push(outcome.counters);

    // M2: a stepped dispatch counts toward `self.steps` ONLY when it produced a
    // Successors/Terminal-equivalent outcome — mirror of single-threaded
    // `step_one`, which does `self.steps += 1` for `Successors`/`Terminal` but
    // skips it for `NeedCallback` (a bounce). A `NeedsPython` outcome is a bounce,
    // so it must not be counted here (the coordinator counts a bounce only if
    // `dispatch_bounce` later resolves it to successors / a terminal).
    if !matches!(outcome.ret, CoreReturn::NeedsPython(_)) {
        shared.stepped.fetch_add(1, Ordering::Relaxed);
    }

    match outcome.ret {
        CoreReturn::Continue(succ) => {
            {
                let mut rm = shared.root_map.lock().expect("root_map poisoned");
                for (s, _tag) in &succ {
                    rm.insert(s.state_id(), root_hint);
                }
            }
            let continue_states: Vec<RustSimState> = succ.into_iter().map(|(s, _)| s).collect();
            TaskOutcome {
                continue_states,
                terminal_states: Vec::new(),
                terminal_summaries: summaries,
                request_cancel: false,
            }
        }
        CoreReturn::Deadended(s) => {
            summaries.push(TerminalSummary::of(&s, SchedDisposition::Deadended));
            TaskOutcome::summarized(summaries)
        }
        CoreReturn::Errored(s, _msg) => {
            summaries.push(TerminalSummary::of(&s, SchedDisposition::Errored));
            TaskOutcome::summarized(summaries)
        }
        CoreReturn::Unconstrained(main, forks) => {
            let main_id = main.state_id();
            {
                let mut rm = shared.root_map.lock().expect("root_map poisoned");
                rm.insert(main_id, root_hint);
                for f in &forks {
                    rm.insert(f.state_id(), root_hint);
                }
            }
            shared
                .kind_map
                .lock()
                .expect("kind_map poisoned")
                .insert(main_id, MatKind::Unconstrained);
            TaskOutcome {
                continue_states: forks,
                terminal_states: vec![main],
                terminal_summaries: summaries,
                request_cancel: false,
            }
        }
        CoreReturn::NeedsPython(bounce) => {
            let PendingBounce {
                kind,
                state: bstate,
                deferred_forks,
                last_condition: _,
                stored_conditions,
                fork_snapshots,
            } = bounce;
            // Materialize the bounce's deferred forks in-thread so no unexplored
            // branch is lost across the (!Send loose-condition) bounce boundary.
            let (forks, pruned2, _ids) = materialize_bounce_forks(
                ctx,
                prof,
                &bstate,
                deferred_forks,
                &stored_conditions,
                fork_snapshots,
                root_hint,
            );
            let bid = bstate.state_id();
            {
                let mut rm = shared.root_map.lock().expect("root_map poisoned");
                rm.insert(bid, root_hint);
                for f in &forks {
                    rm.insert(f.state_id(), root_hint);
                }
            }
            shared
                .kind_map
                .lock()
                .expect("kind_map poisoned")
                .insert(bid, MatKind::Bounce(kind));
            for s in &pruned2 {
                summaries.push(TerminalSummary::of(s, SchedDisposition::Pruned));
            }
            TaskOutcome {
                continue_states: forks,
                terminal_states: vec![bstate],
                terminal_summaries: summaries,
                request_cancel: false,
            }
        }
    }
}

/// Outcome of stepping a single state in `step_one`. The thin `run_loop` driver
/// matches on this both to route results and to gate the post-step bookkeeping
/// (`steps += 1`, uniqueness filter, native techniques, reconvergence sample),
/// which runs ONLY on `Successors`/`Terminal` — never on `Routed`/`NeedCallback`
/// (mirroring the former `continue`/`return` paths before the bookkeeping block).
///
/// Carries `PendingCallback` / `RustSimState` inline by value, exactly like
/// `StepError` (which these variants are reclassified from). Boxing to satisfy
/// `large_enum_variant` would add a heap allocation on every step termination —
/// the wrong call on the hot path; the variants are intentionally inline. See
/// the matching rationale on `StepError` in `stepping.rs`.
#[allow(clippy::large_enum_variant)]
pub(crate) enum StepOutcome {
    /// `step_one` already pushed the state into its terminal/found/active stash
    /// via pre-step or find/avoid policy (avoid-addr, find-addr, native
    /// return/subcall/no_return, NeedCallback→find/avoid routing incl. deferred
    /// forks). Driver just advances — NO post-step bookkeeping.
    Routed,
    /// Live successors from a completed interpreter step. Driver routes each via
    /// `route_successor`, THEN runs post-step bookkeeping.
    Successors(Vec<RustSimState>),
    /// A terminal disposition from a completed step. Driver applies it via
    /// `apply_terminal`, THEN runs post-step bookkeeping.
    Terminal(TerminalDisposition),
    /// A Python callback is pending. The `PendingCallback` is returned as a value
    /// (NOT yet stored in `self.pending_callback`) so the driver can build the
    /// event from this local — letting the `PythonVEXFallback` counter mutations
    /// touch `self` while only `pending` is borrowed (no E0502) — before storing.
    NeedCallback(PendingCallback),
}

/// The three terminal step outcomes, each carrying the data the driver needs to
/// reproduce the original per-stash push (and its side effects) byte-for-byte.
pub(crate) enum TerminalDisposition {
    /// `push_or_drop_terminal(STASH_DEADENDED, state)`.
    Deadended(RustSimState),
    /// `errors.push((pc, message, state_id))` then a direct `push_back` into
    /// `STASH_ERRORED` (errored states are never dropped — not push_or_drop).
    Errored {
        state: RustSimState,
        pc: u64,
        message: String,
        state_id: u64,
    },
    /// `sm.push_or_drop_terminal(STASH_UNCONSTRAINED, state)` then route each
    /// loop-exit fork like a normal successor.
    Unconstrained {
        state: RustSimState,
        forks: Vec<RustSimState>,
    },
}

impl RustExplorationManager {
    /// Run-loop entry point. Dispatches to the verbatim single-threaded loop
    /// (default, zero-regression) or the parallel coordinator when
    /// `RUST_PARALLEL_WORKERS > 1`. The coordinator is scaffolding in 2a — it
    /// currently delegates to the single-threaded path (no behaviour change);
    /// the real wave loop lands in angr-vh834 (1ilq.3c).
    pub(crate) fn run_loop(
        &mut self,
        py: Python<'_>,
        n: Option<u32>,
    ) -> PyResult<ExplorationEvent> {
        if self.parallel_real_workers <= 1 {
            return self.run_loop_single_threaded(n);
        }
        self.run_loop_parallel(py, n)
    }

    /// Parallel coordinator path (angr-vh834 Phase 5): a real work-stealing wave
    /// loop. Each wave drains the entire active stash into migration seeds, runs
    /// the [`ParallelScheduler`] pool (GIL released) to quiescence — workers keep
    /// their live successors thread-local (the f≈0 deep-exploration path) and
    /// materialize only found/bounce/unconstrained terminals across the join —
    /// then the coordinator (`&mut self`, GIL held) applies every deferred
    /// mutation: routes found states to `STASH_FOUND` with `set_root`, re-runs
    /// the Python bounce tail for `NeedsPython` states, and folds the workers'
    /// profiling + counters back into the manager.
    ///
    /// Correctness contract (MVP): the FOUND set (by satisfying content) matches
    /// the single-threaded path when exploring exhaustively. See the worker in
    /// `parallel_process_state`.
    ///
    /// Predicate-driven find/avoid (`find_needs_python` / `avoid_needs_python`)
    /// falls back to the single-threaded loop — its cross-callback skip-state
    /// tracking has no clean parallel analogue and addresses-based exploration is
    /// the parallel-favourable case.
    ///
    /// **Known limitation — `num_find` early-exit does not preserve the active
    /// frontier (Bug M1).** When a wave reaches `num_find`, a worker trips the
    /// shared [`CancelToken`]; every worker stops at its next *task boundary*,
    /// leaving its un-dispatched worker-local live states and the injector's
    /// surplus undrained — those states are dropped at the `thread::scope` join
    /// (`scheduler.rs`), never returned to `STASH_ACTIVE`. The FOUND set is still
    /// correct, but unlike the single-threaded loop (which leaves the active stash
    /// intact when `found_count >= num_find`), the parallel path's post-explore
    /// `active_count()` is smaller and the un-explored frontier is NOT resumable.
    /// A clean fix would have the scheduler drain + materialize the remaining
    /// frontier on cancel, paying serde for states it is about to discard; that
    /// is deferred (the byte-stable scheduler tests pin the current drop-on-cancel
    /// behaviour). See the `CancelToken` doc and `scheduler.rs::worker_loop`.
    pub(crate) fn run_loop_parallel(
        &mut self,
        py: Python<'_>,
        n: Option<u32>,
    ) -> PyResult<ExplorationEvent> {
        // Callable find/avoid predicates need the coordinator's skip-state
        // tracking across resume callbacks; route them to the verbatim
        // single-threaded loop rather than risk a predicate-eval infinite loop.
        if self.find_needs_python || self.avoid_needs_python {
            return self.run_loop_single_threaded(n);
        }

        let max_steps = n.unwrap_or(self.max_steps_per_run) as u64;
        let workers = self.parallel_real_workers.max(2);

        let callbacks = self
            .callbacks
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("callbacks not set"))?
            .clone();
        if !callbacks.is_ready() {
            return Err(PyRuntimeError::new_err("callbacks not ready"));
        }

        let _gil_wall = crate::gil_profile::RunLoopWallGuard::new(self.profiling.profiling_enabled);
        let run_loop_start = if self.profiling.profiling_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let mut dispatched_total: u64 = 0;

        loop {
            // I8 termination path (a): enough solutions.
            if self.found_count() >= self.num_find {
                return Ok(ExplorationEvent::found(
                    self.found_count(),
                    self.active_count(),
                    self.steps,
                ));
            }

            // M3: dispatch any bounces a PRIOR wave parked because it had already
            // surfaced one Python callback this `run()` (only one callback is
            // returned per call). These go STRAIGHT to `dispatch_bounce` — they
            // are never re-stepped by a worker, so their fallback counters are
            // folded exactly once (the bug was re-queuing them to STASH_ACTIVE,
            // where the next wave re-ran `run_post_step_core` and re-folded the
            // counts). The first one needing a real callback returns its event;
            // the rest stay parked for the next `run()`.
            if !self.pending_parallel_bounces.is_empty() {
                let queue = std::mem::take(&mut self.pending_parallel_bounces);
                if let Some(event) = self.process_parallel_bounce_queue(&callbacks, queue) {
                    return Ok(event);
                }
            }

            // Drain the ENTIRE active stash into migration seeds. Each seed's
            // lineage root is stamped now (on the coordinator, with `&self.sm`)
            // and threaded to the workers via `root_map` so descendants inherit
            // it without reaching back into the manager.
            let drained: Vec<RustSimState> = match self.sm.get_mut(STASH_ACTIVE) {
                Some(s) => s.drain(..).collect(),
                None => Vec::new(),
            };
            if drained.is_empty() {
                // I8 termination path (b): active exhausted.
                if let Some(start) = run_loop_start {
                    self.profiling.accumulated_stats.run_loop_time_ns +=
                        start.elapsed().as_nanos() as u64;
                    self.profiling.accumulated_stats.active_states_count =
                        self.active_count() as u64;
                }
                if self.found_count() > 0 {
                    return Ok(ExplorationEvent::found(self.found_count(), 0, self.steps));
                }
                return Ok(ExplorationEvent::active_empty(
                    self.found_count(),
                    self.steps,
                ));
            }

            // Lazily spawn the persistent worker pool on the first wave. Its N
            // long-lived threads each own a Z3 context for the pool's whole life,
            // so the thread-spawn + context-creation cost is paid ONCE per manager
            // instead of once per wave (angr-vh834 Work Item 2).
            if self.parallel_pool.is_none() {
                self.parallel_pool = Some(PersistentPool::new(workers));
            }

            // Build the per-wave shared context. `prof` and `shared` are `Arc` so
            // both the GIL-free `'static` worker closure and this coordinator can
            // reach them: the closure holds a clone for the duration of the wave;
            // once it is dropped (after the barrier) the coordinator recovers sole
            // ownership to fold `prof` and read `shared`'s maps.
            let ctx = self.step_context();
            let prof = Arc::new(ParallelProfiling::default());
            let shared = Arc::new(ParallelShared {
                root_map: Mutex::new(FxHashMap::default()),
                kind_map: Mutex::new(FxHashMap::default()),
                counters: Mutex::new(Vec::new()),
                found_counter: AtomicUsize::new(self.found_count()),
                num_find: self.num_find,
                stepped: AtomicUsize::new(0),
            });
            let mut seeds: Vec<StateMigrationPayload> = Vec::with_capacity(drained.len());
            {
                let mut rm = shared.root_map.lock().expect("root_map poisoned");
                for state in drained {
                    let root = self.sm.root_or_self(state.state_id());
                    rm.insert(state.state_id(), root);
                    seeds.push(state.detach_for_migration());
                }
            }

            // Snapshot the two native registries (O(1) `Arc::clone`) and clone the
            // callbacks WHILE HOLDING THE GIL — `Py<T>::clone` needs the GIL, so
            // it must never run inside a worker.
            let wave_procs = Arc::clone(&self.native_procedures);
            let wave_syscalls = Arc::clone(&self.native_syscalls);
            let wave_callbacks = callbacks.clone();
            let wave_prof = Arc::clone(&prof);
            let wave_shared = Arc::clone(&shared);

            // The GIL-free per-state processor. It owns `ctx` and `Arc`-shares the
            // rest, so it is `'static` and lives in the `Arc<WaveJob>` the
            // persistent workers share. Its callbacks self-acquire the GIL via
            // `Python::attach` (Phase 4); it touches no `&mut self`.
            let process: Box<ProcessFn> = Box::new(move |state, cancel, block_cache| {
                parallel_process_state(
                    state,
                    cancel,
                    block_cache,
                    &ctx,
                    &wave_callbacks,
                    &wave_prof,
                    &wave_procs,
                    &wave_syscalls,
                    &wave_shared,
                )
            });
            let job = WaveJob::new(seeds, process);

            // Release the GIL and run the wave to quiescence on the persistent
            // pool. Workers keep live successors thread-local (the f≈0 path) and
            // materialize only found/bounce/unconstrained terminals across the
            // barrier.
            let pool = self
                .parallel_pool
                .as_ref()
                .expect("parallel pool just created");
            let (mut job, stats) = py.detach(|| pool.run_wave(job));

            // ---- Back on the GIL thread: apply deferred mutations. ----

            // Recover the materialized terminals, then DROP the wave (and its
            // process closure) to release the closure's `Arc` clones of `prof` /
            // `shared` — required before `Arc::into_inner` can recover sole
            // ownership of `shared` below.
            let materialized = job.take_results();
            drop(job);

            // M2 + counter fold: drain the workers' `stepped` count and queued
            // `CoreCounters` through the shared reference (lock + take, safe to
            // call while workers still hold `Arc` clones — the steady-state loop
            // reuses this mid-session), then fold the solver timing and recover
            // sole ownership of `shared` to drain its maps.
            let worker_stepped = self.fold_parallel_shared_counters(&shared);
            prof.fold_into(&mut self.profiling.accumulated_stats);
            let shared = Arc::into_inner(shared)
                .expect("BUG: ParallelShared still referenced after the wave barrier");

            let mut kind_map = shared.kind_map.into_inner().expect("kind_map poisoned");
            let root_map = shared.root_map.into_inner().expect("root_map poisoned");

            // Reattach + route each materialized terminal. Found / unconstrained
            // route immediately; bounces are queued so we surface at most one
            // need_callback event per wave (mirroring single-threaded, which
            // returns on the first state needing Python).
            let main_ctx = z3::Context::thread_local();
            let mut bounce_queue: Vec<(RustSimState, BounceKind, u64)> = Vec::new();
            for payload in materialized {
                let state = payload
                    .reattach(&main_ctx)
                    .map_err(|e| PyRuntimeError::new_err(format!("reattach failed: {e:?}")))?;
                let id = state.state_id();
                let root = root_map.get(&id).copied().unwrap_or(id);
                let kind = kind_map.remove(&id);
                self.route_materialized_terminal(state, kind, root, &mut bounce_queue);
            }

            // Per-wave bookkeeping (mirror of the single-threaded post-step
            // block). `steps` advances by the number of dispatches that actually
            // stepped (M2); `parallel_tasks` / `dispatched_total` keep counting
            // every dispatch (the scheduler-level metric + max_steps budget,
            // whose single-threaded analog is the loop-iteration count).
            self.steps += worker_stepped;
            self.parallel_tasks += stats.dispatches() as u64;
            self.parallel_migrations +=
                (stats.surplus_offloaded + stats.materialized_terminals) as u64;
            // angr-vh834 Phase 1 duplex-protocol accounting (observability only):
            // `bounce_roundtrips` / `resume_reinjects` stay 0 until later phases.
            self.parallel_reattaches += stats.reattaches as u64;
            self.parallel_bounce_roundtrips += stats.bounce_roundtrips as u64;
            self.parallel_resume_reinjects += stats.resume_reinjects as u64;
            dispatched_total += stats.dispatches() as u64;
            self.apply_uniqueness_filter();
            self.apply_native_techniques();
            self.record_reconvergence_sample();

            // Process this wave's true (non-find/avoid) bounces. The first one
            // needing a real callback returns its event; the rest are parked in
            // `self.pending_parallel_bounces` for the next `run()` (M3).
            if let Some(event) = self.process_parallel_bounce_queue(&callbacks, bounce_queue) {
                return Ok(event);
            }

            if dispatched_total >= max_steps {
                if let Some(start) = run_loop_start {
                    self.profiling.accumulated_stats.run_loop_time_ns +=
                        start.elapsed().as_nanos() as u64;
                    self.profiling.accumulated_stats.active_states_count =
                        self.active_count() as u64;
                }
                return Ok(ExplorationEvent::step_complete(
                    self.found_count(),
                    self.active_count(),
                    self.steps,
                ));
            }
        }
    }

    /// Drain the per-dispatch accounting out of a [`ParallelShared`] through the
    /// shared reference: takes the queued `CoreCounters` (lock + `mem::take`) and
    /// swaps the `stepped` counter to zero, returning its value (M2 semantics —
    /// see the `stepped` field doc). Deliberately does NOT require sole ownership
    /// of the `Arc`, so the steady-state coordinator can fold incrementally at
    /// every event return while workers still hold clones; the wave loop calls it
    /// once per wave right before `Arc::into_inner`.
    fn fold_parallel_shared_counters(&mut self, shared: &ParallelShared) -> u64 {
        let worker_stepped = shared.stepped.swap(0, Ordering::SeqCst) as u64;
        let drained = std::mem::take(&mut *shared.counters.lock().expect("counters poisoned"));
        for counters in drained {
            self.fold_core_counters(counters);
        }
        worker_stepped
    }

    /// Route one reattached materialized terminal by its [`MatKind`] tag —
    /// the per-payload body of the wave loop's post-barrier routing, extracted
    /// so the steady-state coordinator can route terminals one at a time as they
    /// stream in. The caller looks up (and REMOVES — entries must not accumulate
    /// over a long steady session) the state's `kind_map`/`root_map` entries and
    /// passes them per-state; true bounces are pushed to `bounce_queue` for
    /// `process_parallel_bounce_queue` rather than dispatched inline, so both
    /// loops surface at most one Python callback per `run()`.
    fn route_materialized_terminal(
        &mut self,
        mut state: RustSimState,
        kind: Option<MatKind>,
        root: u64,
        bounce_queue: &mut Vec<(RustSimState, BounceKind, u64)>,
    ) {
        let id = state.state_id();
        match kind {
            Some(MatKind::Found) => {
                self.sm.set_root(id, root);
                self.sm
                    .stashes_mut()
                    .entry(STASH_FOUND.to_string())
                    .or_default()
                    .push_back(state);
            }
            Some(MatKind::Unconstrained) => {
                self.sm.set_root(id, root);
                self.sm.push_or_drop_terminal(STASH_UNCONSTRAINED, state);
            }
            Some(MatKind::Bounce(kind)) => {
                // Bug C1: a bounce whose target is a find/avoid ADDRESS
                // must route to FOUND/AVOID/PRUNED exactly as
                // single-threaded `step_one`'s NeedCallback→find/avoid
                // special case does — NOT to a spurious Python bounce,
                // which would LOSE the found state. The worker already
                // materialized this bounce's deferred forks as
                // `continue_states` (explored locally), so we route ONLY
                // the parked main state here; no fork is re-created. The
                // `set_pc(addr)` + `add_to_history(addr)` mirrors
                // `dispatch_bounce` so the routed state reports the target.
                match bounce_target_addr(&kind) {
                    Some(addr) if self.find_addrs.contains(&addr) => {
                        state.set_pc(addr);
                        state.add_to_history(addr);
                        self.sm.set_root(id, root);
                        if self.constraint_solver.lazy_solves || state.satisfiable() {
                            self.sm
                                .stashes_mut()
                                .entry(STASH_FOUND.to_string())
                                .or_default()
                                .push_back(state);
                        } else {
                            log::debug!(
                                "parallel: bounce at find addr 0x{:x} is UNSAT, pruning",
                                addr
                            );
                            self.push_or_drop_terminal(STASH_PRUNED, state);
                        }
                    }
                    Some(addr) if self.avoid_addrs.contains(&addr) => {
                        state.set_pc(addr);
                        state.add_to_history(addr);
                        self.sm.set_root(id, root);
                        self.push_or_drop_terminal(STASH_AVOID, state);
                    }
                    _ => bounce_queue.push((state, kind, root)),
                }
            }
            Some(MatKind::ActiveResidual) => {
                // Phase 1 scaffolding: no path produces this yet. Its
                // future semantics (a live frontier state drained back on
                // cancel) is a bare active successor, so route it as one —
                // nothing is silently lost if a later phase emits it before
                // wiring the full cancel-drain handler.
                self.sm.set_root(id, root);
                self.route_successor(state, true);
            }
            None => {
                // Untagged materialized state — should not happen; treat
                // as a bare active successor so nothing is silently lost.
                log::error!("parallel: untagged materialized state {id}; re-routing");
                self.route_successor(state, true);
            }
        }
    }

    /// Dispatch a queue of parallel-wave bounce states through `dispatch_bounce`
    /// (angr-vh834 Phase 5; Bugs M2/M3). Every bounce goes STRAIGHT to Python —
    /// it is never handed back to a worker to re-step — so its native/syscall/
    /// simproc fallback counters are folded exactly once (M3: re-queuing parked
    /// bounces to `STASH_ACTIVE` made the next wave re-run `run_post_step_core`
    /// and double-fold them).
    ///
    /// Returns `Some(event)` for the FIRST bounce that needs a real Python
    /// callback (single-threaded surfaces at most one callback per `run()`); the
    /// remaining, undispatched bounces are stored in
    /// `self.pending_parallel_bounces` so the next `run()` dispatches them the
    /// same way (still no re-step).
    ///
    /// `self.steps` is bumped per bounce that `dispatch_bounce` resolves to live
    /// successors or a terminal — the single-threaded analog, where such a bounce
    /// surfaces from `step_state_with_skip` as a `Successors`/`Terminal`
    /// `StepOutcome` and runs `self.steps += 1` (M2). A bounce that needs a
    /// callback is a `NeedCallback` and does NOT increment, and the worker already
    /// excluded every `NeedsPython` outcome from its `stepped` counter, so each
    /// stepped dispatch is counted exactly once across the two sites.
    fn process_parallel_bounce_queue(
        &mut self,
        callbacks: &PythonCallbacks,
        queue: Vec<(RustSimState, BounceKind, u64)>,
    ) -> Option<ExplorationEvent> {
        let mut iter = queue.into_iter();
        let mut event = None;
        for (state, kind, root) in iter.by_ref() {
            self.sm.set_root(state.state_id(), root);
            let bounce = PendingBounce {
                kind,
                state,
                deferred_forks: Vec::new(),
                last_condition: None,
                stored_conditions: FxHashMap::default(),
                fork_snapshots: FxHashMap::default(),
            };
            match self.dispatch_bounce(callbacks, bounce) {
                Ok(successors) => {
                    for s in successors {
                        self.route_successor(s, true);
                    }
                    self.steps += 1;
                }
                Err(StepError::NeedCallback(pending)) => {
                    let ev = self.callback_event(&pending);
                    self.pending_callbacks
                        .insert(StateId::new(pending.state.state_id()), pending);
                    event = Some(ev);
                    break;
                }
                Err(StepError::Deadended(s)) => {
                    self.apply_terminal(TerminalDisposition::Deadended(s));
                    self.steps += 1;
                }
                Err(StepError::Error(s, message)) => {
                    let pc = s.pc();
                    let state_id = s.state_id();
                    self.apply_terminal(TerminalDisposition::Errored {
                        state: s,
                        pc,
                        message,
                        state_id,
                    });
                    self.steps += 1;
                }
                Err(StepError::Unconstrained(s, forks)) => {
                    self.apply_terminal(TerminalDisposition::Unconstrained { state: s, forks });
                    self.steps += 1;
                }
            }
        }
        // M3: any bounce left undispatched (we surfaced an event and broke) is
        // parked for the next run() — NOT pushed to STASH_ACTIVE, so no worker
        // re-steps it and re-folds its counters. When the queue drained fully,
        // `iter.collect()` is empty and this clears the parked list.
        self.pending_parallel_bounces = iter.collect();
        event
    }

    /// Inner body of the pymethods-exposed `run`. See `run` in `mod.rs`.
    ///
    /// Thin driver over `step_one`: owns the loop frame (I8 termination,
    /// state pop, post-step bookkeeping) and routes each `StepOutcome`.
    pub(crate) fn run_loop_single_threaded(
        &mut self,
        n: Option<u32>,
    ) -> PyResult<ExplorationEvent> {
        let max_steps = n.unwrap_or(self.max_steps_per_run);

        // Ensure callbacks are set and clone to avoid borrow issues
        let callbacks = self
            .callbacks
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("callbacks not set"))?
            .clone();

        if !callbacks.is_ready() {
            return Err(PyRuntimeError::new_err("callbacks not ready"));
        }

        // GIL-work timing (angr-1ilq.7): bracket the whole run-loop wall time —
        // the denominator for the GIL/solver fractions. While live it also arms
        // the GIL-region timer (see `gil_profile`), so only Python-touch work
        // *during stepping* is counted (state export after `run()` is excluded,
        // keeping `gil_work_ns <= run_wall_ns`). The `Drop` fires on every exit
        // path (early `return`, `?`, panic) without borrowing `self`.
        let _gil_wall = crate::gil_profile::RunLoopWallGuard::new(self.profiling.profiling_enabled);

        let run_loop_start = if self.profiling.profiling_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };

        for _ in 0..max_steps {
            // angr-v5ht: runtime thrash detection for the
            // `use_shared_lineage_solver` opt-in. Hooks at the TOP of
            // the for-loop iteration (BEFORE any early-return path)
            // because callback-heavy workloads (e.g.
            // google2016_unbreakable_0: every iteration returns via
            // `need_simprocedure`) never reach `self.steps += 1` and
            // would otherwise never sample (`bd recall
            // v5ht-sampler-tick-bottleneck`). `tick_and_sample_for_thrash`
            // uses its own internal tick counter so the sampling cadence
            // is independent of `self.steps`. Always-on: cheap (single
            // atomic load + branch on `LINEAGE_DISMANTLED`) and a no-op
            // until the kill switch is turned on. Sample every 10 ticks;
            // dismantle when ≥20 lineage_switch events show <35% hot
            // ratio over a window — threshold calibrated on N=4 workloads
            // (`bd recall v5ht-threshold-justification-2026-05-25`).
            #[cfg(feature = "vex-engine-z3")]
            crate::symbolic::lineage::tick_and_sample_for_thrash(10, 20, 35);

            // Check if we have enough solutions.
            // I8 termination path (a): `found_count()` covers both
            // Rust-native finds (`found` stash via address check) and
            // Python-predicate finds (need_callback resume). See module
            // header for the full contract.
            if self.found_count() >= self.num_find {
                return Ok(ExplorationEvent::found(
                    self.found_count(),
                    self.active_count(),
                    self.steps,
                ));
            }

            // Get next state from active stash
            // P9 fix: Use LIFO (pop_back) for DFS or FIFO (pop_front) for BFS
            let state = match self.sm.get_mut(STASH_ACTIVE).and_then(|s| {
                if self.use_lifo {
                    s.pop_back() // DFS: LIFO (most recent state first)
                } else {
                    s.pop_front() // BFS: FIFO (oldest state first)
                }
            }) {
                Some(s) => {
                    let sid = s.state_id();
                    self.current_stepping_state_id = Some(sid.into());
                    STEPPING_STATE_ID.with(|cell| cell.set(Some(sid)));
                    // angr-panhl.1: model a work-stealing migration at dispatch.
                    self.record_migration_sample(sid);
                    // SI-B (angr-1ilq.3 increment 2b'): the REAL analog of the
                    // MODEL sample above — measure the migration serde tax on
                    // this pre-step state and discard the result. No-op (and
                    // byte-identical behaviour) unless RUST_PARALLEL_SHADOW_PROBE
                    // is set.
                    #[cfg(feature = "vex-engine-z3")]
                    self.shadow_probe_migrate(&s);
                    s
                }
                None => {
                    // No active states.
                    // I8 termination path (b): active stash exhausted —
                    // emit `found` if we picked up any solutions, else
                    // `active_empty`. Either way the loop exits here
                    // rather than spinning. See module header.
                    if self.found_count() > 0 {
                        return Ok(ExplorationEvent::found(self.found_count(), 0, self.steps));
                    } else {
                        return Ok(ExplorationEvent::active_empty(
                            self.found_count(),
                            self.steps,
                        ));
                    }
                }
            };

            match self.step_one(&callbacks, state)? {
                // Pre-step / find-avoid policy already routed the state. Advance
                // without post-step bookkeeping (former `continue` paths).
                StepOutcome::Routed => continue,
                // Build the event from the LOCAL `pending` (so the
                // PythonVEXFallback counter mutations are free of a borrow
                // conflict), THEN store it. Mirrors the former order exactly.
                StepOutcome::NeedCallback(pending) => {
                    let event = self.callback_event(&pending);
                    self.pending_callbacks
                        .insert(StateId::new(pending.state.state_id()), pending);
                    return Ok(event);
                }
                // A real interpreter step completed — route successors / apply
                // the terminal disposition, then fall through to bookkeeping.
                StepOutcome::Successors(successors) => {
                    // Add successors to appropriate stashes, checking find/avoid
                    for successor in successors {
                        self.route_successor(successor, true);
                    }
                }
                StepOutcome::Terminal(disposition) => self.apply_terminal(disposition),
            }

            self.steps += 1;

            // Apply native uniqueness filter if enabled
            self.apply_uniqueness_filter();
            // Apply native techniques (LengthLimiter, Timeout, LoopBound)
            self.apply_native_techniques();
            // DS-instr (angr-11djq.16): sample (pc, callstack) reconvergence
            // over the post-filter active frontier. Counters only.
            self.record_reconvergence_sample();
        }

        // Record run loop timing and active state count
        if let Some(start) = run_loop_start {
            self.profiling.accumulated_stats.run_loop_time_ns += start.elapsed().as_nanos() as u64;
            self.profiling.accumulated_stats.active_states_count = self.active_count() as u64;
        }

        // Max steps reached
        Ok(ExplorationEvent::step_complete(
            self.found_count(),
            self.active_count(),
            self.steps,
        ))
    }

    /// Step a single popped state to its next outcome — the shared stepping
    /// decision point. Returns a `StepOutcome` the driver routes; this method
    /// performs all the per-state work (find/avoid checks, SimProcedure
    /// dispatch, interpreter step, deferred-fork routing) but leaves loop-frame
    /// concerns (state pop, post-step bookkeeping, event storage) to the driver.
    pub(crate) fn step_one(
        &mut self,
        callbacks: &PythonCallbacks,
        mut state: RustSimState,
    ) -> PyResult<StepOutcome> {
        // Check find/avoid before stepping
        let pc = state.pc();

        // P7 fix: Check if callable avoid predicate needs Python evaluation
        // When avoid is a callable (lambda/function), we must return to Python
        // to evaluate it for each state, not just check addresses.
        // Skip if this state was just checked (resume_avoid_predicate(false)
        // sets skip_avoid_predicate_states to prevent infinite loop).
        if self.avoid_needs_python {
            let state_id = state.state_id();
            if self
                .constraint_tracker
                .skip_avoid_predicate_states
                .remove(&state_id)
            {
                // Fall through — predicate already checked at this PC
            } else {
                let pending = PendingCallback::lightweight(
                    state,
                    CallbackReason::AvoidPredicate { addr: pc },
                );
                return Ok(StepOutcome::NeedCallback(pending));
            }
        }

        // Check avoid addresses (address-based, only when NOT using callable predicate)
        if self.avoid_addrs.contains(&pc) {
            self.push_or_drop_terminal(STASH_AVOID, state);
            return Ok(StepOutcome::Routed);
        }

        // P2 fix: Check if callable find predicate needs Python evaluation
        // When find is a callable (lambda/function), we must return to Python
        // to evaluate it for each state, not just check addresses.
        // Skip if this state was just checked (resume_find_predicate(false)
        // sets skip_find_predicate_state to avoid infinite loop).
        if self.find_needs_python {
            let state_id = state.state_id();
            if self
                .constraint_tracker
                .skip_find_predicate_states
                .remove(&state_id)
            {
                // Fall through to hooks/stepping — predicate already checked
            } else {
                let pending =
                    PendingCallback::lightweight(state, CallbackReason::FindPredicate { addr: pc });
                return Ok(StepOutcome::NeedCallback(pending));
            } // else (not skip_find_predicate_state)
        }

        // Check find addresses (address-based, only when NOT using callable predicate)
        if self.find_addrs.contains(&pc) {
            // Only add to found if the state is satisfiable
            // (UNSAT states reached the address via infeasible paths)
            if self.constraint_solver.lazy_solves || state.satisfiable() {
                self.sm
                    .stashes_mut()
                    .entry(STASH_FOUND.to_string())
                    .or_default()
                    .push_back(state);
            } else {
                log::debug!("State at find address 0x{:x} is UNSAT, pruning", pc);
                self.push_or_drop_terminal(STASH_PRUNED, state);
            }
            return Ok(StepOutcome::Routed);
        }

        // Check hooks (SimProcedures)
        // GAP 6: Stack-based skip tracking for zero-length hooks
        // Clean up expired skip entries before checking
        self.skip_hook_stack
            .retain(|&(_, expiry)| expiry > self.steps);

        // Check if this address is in the skip stack
        let should_skip_hook = self.skip_hook_stack.iter().any(|&(addr, _)| addr == pc);
        if should_skip_hook {
            // Remove this address from the skip stack (consumed)
            self.skip_hook_stack.retain(|&(addr, _)| addr != pc);
            log::debug!(
                "Skipping hook at 0x{:x} (zero-length hook, step {})",
                pc,
                self.steps
            );
        }
        if self.hooks.contains(&pc) && !should_skip_hook {
            // Check if this is a registered SimProcedure
            if let Some((name, num_args, no_return)) = self.simprocedures.get(&pc).cloned() {
                // Skip native for addresses inside the binary (user-placed hooks)
                let is_in_binary = self
                    .environment
                    .binary_regions
                    .iter()
                    .any(|(base, data)| pc >= *base && pc < *base + data.len() as u64);
                // Try native procedure first (only for external/library hooks)
                if !is_in_binary && let Some(native_proc) = self.native_procedures.get(&name) {
                    // Extract arguments from state registers (and stack
                    // when num_args exceeds the register count). On
                    // failure (symbolic SP, unmapped stack slot) skip
                    // the native fast path and let the Python
                    // SimProcedure callback below handle it — handing
                    // the native handler fabricated zeros would mask
                    // the underlying stack-setup bug.
                    match self.extract_procedure_args(&state, num_args) {
                        Err(e) => {
                            log::debug!(
                                "Skipping native procedure {} (arg extraction failed: {:?})",
                                name,
                                e
                            );
                            self.profiling.native_proc_stats.python_fallbacks += 1;
                            *self
                                .profiling
                                .native_proc_stats
                                .other_fallbacks_by_name
                                .entry(name.clone())
                                .or_insert(0) += 1;
                        }
                        Ok(args) => match native_proc.call_ex(&mut state, &args) {
                            // Borrow note: `native_proc` borrows
                            // `self.native_procedures`; that borrow ends at
                            // the `call_ex` call above (NLL), freeing
                            // `&mut self` for `setup_native_subcall` /
                            // `push_to_active_or_drop` below. These arms must
                            // NOT reference `native_proc` again. (Path B's
                            // `handle_simprocedure` uses an explicit
                            // `NativeProcDisposition` enum for the same reason.)
                            Ok(ProcOutcome::Return(ret_val)) => {
                                // Native execution succeeded
                                self.profiling.native_proc_stats.native_calls += 1;
                                *self
                                    .profiling
                                    .native_proc_stats
                                    .call_counts
                                    .entry(name.clone())
                                    .or_insert(0) += 1;

                                // For no-return procedures (exit/abort), skip
                                // the return-address dance and deadend directly.
                                // Setting PC to a stack-derived return address
                                // can produce a spurious successor (e.g. when
                                // exit is called from rejected() in fauxware,
                                // the post-call address happens to overlap
                                // main's start, causing infinite re-entry).
                                if no_return {
                                    self.push_or_drop_terminal(STASH_DEADENDED, state);
                                    return Ok(StepOutcome::Routed);
                                }

                                // Set return value if present
                                if let Some(rv) = ret_val {
                                    let ret_reg =
                                        self.environment.calling_convention.return_register();
                                    state.set_register_by_offset(ret_reg, rv);
                                }

                                // Get return address and set PC. Use the
                                // state's real register file so that LR/X30/$ra
                                // overrides see actual values; passing a blank
                                // RegisterFile here used to make ARM/ARM64/MIPS
                                // read LR=0 and set PC to 0.
                                let ctx = state.solver().borrow();
                                let ret_addr_opt = self
                                    .environment
                                    .calling_convention
                                    .get_return_addr(state.registers(), None, &ctx);
                                let pops_return_addr =
                                    self.environment.calling_convention.pops_return_addr();
                                drop(ctx);
                                if let Some(ret_addr) = ret_addr_opt {
                                    // Only adjust SP for stack-based ABIs
                                    // (x86/AMD64). ARM/ARM64/MIPS keep ret addr
                                    // in a register and leave SP untouched.
                                    if pops_return_addr {
                                        let sp = state.get_sp().as_u64().unwrap_or(0);
                                        let ptr_size = state.arch().bytes() as u64;
                                        state.set_sp(RustBV::concrete(
                                            (sp + ptr_size) as u128,
                                            state.arch().bits(),
                                        ));
                                    }
                                    state.set_pc(ret_addr);
                                } else if pops_return_addr {
                                    // Fallback: read ret addr from [sp] for
                                    // stack-based ABIs (only useful when the
                                    // calling convention's get_return_addr
                                    // declined to read memory itself).
                                    if let Some(sp) = state.get_sp().as_u64()
                                        && let Ok(ret_bv) =
                                            state.memory_load(sp, state.arch().bytes())
                                        && let Some(ret_addr) = ret_bv.as_u64()
                                    {
                                        let ptr_size = state.arch().bytes() as u64;
                                        state.set_sp(RustBV::concrete(
                                            (sp + ptr_size) as u128,
                                            state.arch().bits(),
                                        ));
                                        state.set_pc(ret_addr);
                                    }
                                }

                                self.push_to_active_or_drop(state);
                                return Ok(StepOutcome::Routed);
                            }
                            Ok(ProcOutcome::CallAndResume {
                                target,
                                args: sub_args,
                                resume_tag,
                            }) => {
                                // The proc requested a guest sub-call. Capture
                                // the caller return address from [sp] BEFORE
                                // `setup_native_subcall` overwrites that slot
                                // with the resume sentinel. A symbolic SP /
                                // unmapped slot (None) or a setup failure falls
                                // back to Python (the state is left untouched
                                // by setup on Err). Do NOT pop SP or honour
                                // `no_return` here: the guest's own `ret`
                                // advances SP, and `handle_native_resume`
                                // finishes without re-adjusting it.
                                let setup = match self.get_return_addr(&state) {
                                    Some(caller_return_addr) => self
                                        .setup_native_subcall(
                                            &mut state,
                                            name.clone(),
                                            args,
                                            caller_return_addr,
                                            target,
                                            sub_args,
                                            resume_tag,
                                        )
                                        .map_err(|e| format!("{e:?}")),
                                    None => Err("no concrete return address".to_string()),
                                };
                                match setup {
                                    Ok(()) => {
                                        self.profiling.native_proc_stats.native_calls += 1;
                                        *self
                                            .profiling
                                            .native_proc_stats
                                            .call_counts
                                            .entry(name.clone())
                                            .or_insert(0) += 1;
                                        self.push_to_active_or_drop(state);
                                        return Ok(StepOutcome::Routed);
                                    }
                                    Err(reason) => {
                                        log::debug!(
                                            "native sub-call setup failed ({}); \
                                             falling back to Python for {}",
                                            reason,
                                            name
                                        );
                                        self.profiling.native_proc_stats.python_fallbacks += 1;
                                        *self
                                            .profiling
                                            .native_proc_stats
                                            .other_fallbacks_by_name
                                            .entry(name.clone())
                                            .or_insert(0) += 1;
                                    }
                                }
                            }
                            Err(e) => {
                                // Native execution failed, fall back to Python
                                self.profiling.native_proc_stats.python_fallbacks += 1;
                                let bucket = match e {
                                    ProcedureError::SymbolicArgument(_) => {
                                        &mut self
                                            .profiling
                                            .native_proc_stats
                                            .symbolic_fallbacks_by_name
                                    }
                                    ProcedureError::NotImplemented => {
                                        &mut self
                                            .profiling
                                            .native_proc_stats
                                            .not_implemented_fallbacks_by_name
                                    }
                                    _ => {
                                        &mut self
                                            .profiling
                                            .native_proc_stats
                                            .other_fallbacks_by_name
                                    }
                                };
                                *bucket.entry(name.clone()).or_insert(0) += 1;
                            }
                        },
                    }
                } // if !is_in_binary

                // Fall back to Python for SimProcedure execution
                self.simprocedure_python_fallback_count += 1;
                *self
                    .simprocedure_fallback_by_name
                    .entry(name.clone())
                    .or_insert(0) += 1;
                let return_addr = self.get_return_addr(&state).unwrap_or(0);

                // No deferred forks in run-loop path, so pre_callback_snapshot
                // is unnecessary (it's only used as fork base for deferred forks).
                // Use shared solver (O(1) Rc clone) instead of fork (~3-40ms Z3 clone).
                let solver_ref = state.solver();
                let shared_ctx = RustSolverContext::from_shared_sym_context(solver_ref.clone());

                let pending = PendingCallback::with_context(
                    state,
                    None,
                    CallbackReason::SimProcedure {
                        addr: pc,
                        name: name.clone(),
                        num_args,
                        return_addr,
                    },
                    "Ijk_Call",
                    Some(shared_ctx),
                    Vec::new(),
                    FxHashMap::default(),
                    FxHashMap::default(),
                );

                return Ok(StepOutcome::NeedCallback(pending));
            }
        }

        // Step the state, passing the skip_hook_addr if we just skipped
        let skip_addr_for_step = if should_skip_hook { Some(pc) } else { None };
        match self.step_state_with_skip(callbacks, state, skip_addr_for_step) {
            Ok(successors) => Ok(StepOutcome::Successors(successors)),
            Err(StepError::NeedCallback(pending)) => {
                // Check if the callback address is a find/avoid address
                // (these were added as interpreter hooks to stop execution)
                let callback_addr = match &pending.reason {
                    CallbackReason::SimProcedure { addr, .. } => Some(*addr),
                    _ => None,
                };
                if let Some(addr) = callback_addr
                    && (self.find_addrs.contains(&addr) || self.avoid_addrs.contains(&addr))
                {
                    let is_find = self.find_addrs.contains(&addr);

                    // Process deferred forks BEFORE handling the find/avoid state.
                    // These represent unexplored branches that diverged before
                    // reaching the find/avoid address and must not be dropped.
                    let fork_base = pending
                        .pre_callback_snapshot
                        .unwrap_or_else(|| pending.state.fork());
                    let root_state_id = self.sm.root_or_self(pending.state.state_id());

                    let mut snapshots = pending.fork_snapshots;
                    let cb_fork_start = if self.profiling.profiling_enabled {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    let cb_fork_total = pending.deferred_forks.len() as u64;
                    for fork in pending.deferred_forks {
                        let condition = pending.stored_conditions.get(&fork.condition_id);
                        let reconstructed = if condition.is_none() {
                            if let Some(ref py_ast) = fork.condition_ast {
                                Python::attach(|py| {
                                    let ast = py_ast.bind(py);
                                    let solver_ref = fork_base.solver();
                                    let ctx: &SymContext = &solver_ref.borrow();
                                    claripy_to_rustbv(py, ast, ctx).ok()
                                })
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        if let Some(cond) = condition.or(reconstructed.as_ref()) {
                            // Add the taken-path constraint to the main state
                            // (mirrors the BlockEnd handling at line 3560-3564).
                            if fork.path_taken {
                                pending.state.solver().borrow().assume_true(cond);
                            } else {
                                pending.state.solver().borrow().assume_false(cond);
                            }
                            let fork_op_start = if self.profiling.profiling_enabled {
                                Some(std::time::Instant::now())
                            } else {
                                None
                            };
                            let forked = super::helpers::build_unexplored_fork(
                                &fork_base,
                                &fork,
                                cond,
                                &mut snapshots,
                            );
                            if let Some(start) = fork_op_start {
                                self.profiling.accumulated_stats.solver_fork_time_ns +=
                                    start.elapsed().as_nanos() as u64;
                                self.profiling.accumulated_stats.solver_fork_count += 1;
                            }
                            self.sm.set_root(forked.state_id(), root_state_id);
                            let sat_start = if self.profiling.profiling_enabled {
                                Some(std::time::Instant::now())
                            } else {
                                None
                            };
                            if self.constraint_solver.lazy_solves || forked.satisfiable() {
                                if let Some(start) = sat_start {
                                    self.profiling.accumulated_stats.solver_sat_time_ns +=
                                        start.elapsed().as_nanos() as u64;
                                    self.profiling.accumulated_stats.solver_sat_count += 1;
                                }
                                self.push_to_active_or_drop(forked);
                            } else if let Some(start) = sat_start {
                                self.profiling.accumulated_stats.solver_sat_time_ns +=
                                    start.elapsed().as_nanos() as u64;
                                self.profiling.accumulated_stats.solver_sat_count += 1;
                            }
                        }
                    }
                    if let Some(start) = cb_fork_start {
                        self.profiling.accumulated_stats.deferred_fork_time_ns +=
                            start.elapsed().as_nanos() as u64;
                        self.profiling.accumulated_stats.deferred_fork_count += cb_fork_total;
                    }

                    // Now handle the main state
                    if is_find {
                        if self.constraint_solver.lazy_solves || pending.state.satisfiable() {
                            self.sm
                                .stashes_mut()
                                .entry(STASH_FOUND.to_string())
                                .or_default()
                                .push_back(pending.state);
                        } else {
                            log::debug!("State at find address 0x{:x} is UNSAT, pruning", addr);
                            self.push_or_drop_terminal(STASH_PRUNED, pending.state);
                        }
                    } else {
                        self.push_or_drop_terminal(STASH_AVOID, pending.state);
                    }
                    return Ok(StepOutcome::Routed);
                }

                // Need Python callback — hand the pending back to the driver,
                // which builds the event and stores it.
                Ok(StepOutcome::NeedCallback(pending))
            }
            Err(StepError::Deadended(state)) => {
                Ok(StepOutcome::Terminal(TerminalDisposition::Deadended(state)))
            }
            Err(StepError::Error(state, message)) => {
                let pc = state.pc();
                let state_id = state.state_id();
                Ok(StepOutcome::Terminal(TerminalDisposition::Errored {
                    state,
                    pc,
                    message,
                    state_id,
                }))
            }
            Err(StepError::Unconstrained(state, forks)) => {
                Ok(StepOutcome::Terminal(TerminalDisposition::Unconstrained {
                    state,
                    forks,
                }))
            }
        }
    }

    /// Apply a terminal disposition to the stashes, reproducing each original
    /// per-stash push path (and its side effects) byte-for-byte. Called by the
    /// driver, which then runs post-step bookkeeping.
    pub(crate) fn apply_terminal(&mut self, disposition: TerminalDisposition) {
        match disposition {
            TerminalDisposition::Deadended(state) => {
                self.push_or_drop_terminal(STASH_DEADENDED, state);
            }
            TerminalDisposition::Errored {
                state,
                pc,
                message,
                state_id,
            } => {
                self.errors.push((pc, message, state_id));
                self.sm
                    .stashes_mut()
                    .entry(STASH_ERRORED.to_string())
                    .or_default()
                    .push_back(state);
            }
            TerminalDisposition::Unconstrained { state, forks } => {
                // State has too many symbolic jump targets - move to unconstrained stash
                log::debug!("State {} moved to unconstrained stash", state.state_id());
                self.sm.push_or_drop_terminal(STASH_UNCONSTRAINED, state);
                // angr-027h: loop-exit deferred forks materialized in eager
                // mode at the unconstrained jump. Route them to active (or
                // found/avoid) exactly like normal successors so a
                // find-guided search can reach a target behind the loop.
                for fork in forks {
                    self.route_successor(fork, true);
                }
            }
        }
    }

    /// Build the `ExplorationEvent` for a pending Python callback from its
    /// reason. Single place events are constructed (DRY): absorbs the former
    /// inline predicate/simproc constructions and the post-step match. Takes the
    /// LOCAL `pending` by ref so the `PythonVEXFallback` counter mutations can
    /// touch `self` without a borrow conflict; the driver stores `pending`
    /// afterward.
    pub(crate) fn callback_event(&mut self, pending: &PendingCallback) -> ExplorationEvent {
        let state_id = pending.state.state_id();
        match &pending.reason {
            CallbackReason::SimProcedure {
                addr,
                name,
                num_args,
                return_addr,
            } => ExplorationEvent::need_simprocedure(
                state_id,
                *addr,
                name.clone(),
                *num_args,
                *return_addr,
                self.found_count(),
                self.active_count(),
                self.steps,
            ),
            CallbackReason::Syscall { num } => ExplorationEvent::need_syscall(
                state_id,
                *num,
                self.found_count(),
                self.active_count(),
                self.steps,
            ),
            CallbackReason::FindPredicate { addr } => ExplorationEvent::need_predicate(
                state_id,
                *addr,
                "find_predicate",
                self.found_count(),
                self.active_count(),
                self.steps,
            ),
            CallbackReason::AvoidPredicate { addr } => ExplorationEvent::need_predicate(
                state_id,
                *addr,
                "avoid_predicate",
                self.found_count(),
                self.active_count(),
                self.steps,
            ),
            CallbackReason::SymbolicBranch {
                condition_id,
                true_target,
                false_target,
            } => ExplorationEvent::need_symbolic_branch(
                state_id,
                *condition_id,
                *true_target,
                *false_target,
                self.found_count(),
                self.active_count(),
                self.steps,
            ),
            CallbackReason::PythonVEXFallback { addr, reason } => {
                self.vex_fallback_count += 1;
                self.vex_fallback_addrs
                    .entry(*addr)
                    .or_insert_with(|| reason.clone());
                if reason.contains(DCAS_UNSUPPORTED_REASON) {
                    self.dcas_unsupported_count += 1;
                    if self.dcas_warned_states.insert(state_id) {
                        log::warn!(
                            "DCAS (cmpxchg16b) unsupported in Rust interpreter at \
                             0x{:x} (state {}); falling back to Python VEX engine",
                            addr,
                            state_id
                        );
                    }
                }
                if reason.contains(VECRET_GSPTR_REASON) {
                    self.vecret_gsptr_fallback_count += 1;
                }
                ExplorationEvent::need_python_vex(
                    state_id,
                    *addr,
                    reason,
                    self.found_count(),
                    self.active_count(),
                    self.steps,
                )
            }
            CallbackReason::Error { message } => ExplorationEvent::error(
                message.clone(),
                self.found_count(),
                self.active_count(),
                self.steps,
            ),
        }
    }
}
