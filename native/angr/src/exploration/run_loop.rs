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
//!
//! **Panic policy (CQ .8):** the parallel-driver half of this module locks the
//! `ParallelShared` mutexes (`root_map`, `kind_map`, `counters`, `up_rx`) via
//! `.lock().expect("… poisoned")`. Those poison messages are invariant guards,
//! not error paths: the crate ships with `panic = "abort"`, so a thread can
//! never unwind out of a held `MutexGuard` to poison a lock (it aborts at the
//! panic site first), and the `expect("session live"/"pool set")` sites guard
//! state-machine invariants the driver upholds locally. See the "Panic policy"
//! section of [`scheduler`](super::scheduler) for the full argument — the same
//! reasoning covers every `.expect` in this file, so there is no fallible site
//! to propagate and no Python-exception path to build under this profile.
//!
//! **Enforcement (qwyti.15):** this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` so any *new* fallible
//! unwrap must be justified. The ~35 pre-existing `.expect()` sites are all the
//! poison / session-live / pool-set invariant guards described above; each
//! function that holds them carries a narrow
//! `#[allow(clippy::expect_used, reason = ...)]` pointing back at this Panic
//! policy, and the `mod tests;` decl is exempted (the deny propagates into
//! `#[path]` test submodules — see bd `invariant-clippy-deny-propagates-to-test-submodule`).
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

#[cfg(feature = "vex-engine-z3")]
use std::sync::Mutex;
#[cfg(feature = "vex-engine-z3")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "vex-engine-z3")]
use lru::LruCache;

use super::core_outcome::{
    BounceKind, CoreCtx, CoreReturn, NativeSubcall, ParallelProfiling, PendingBounce,
    PostStepInputs, materialize_bounce_forks, run_post_step_core,
};
use super::native_proc_dispatch::{
    NativeProcCounters, NativeProcDisposition, dispatch_native_proc,
};
#[cfg(feature = "vex-engine-z3")]
use super::scheduler::{
    CancelToken, PersistentPool, ProcessFn, RunSession, TaskOutcome,
    TerminalDisposition as SchedDisposition, TerminalSummary, WaveJob, WorkerUp,
};
#[cfg(feature = "vex-engine-z3")]
use super::step_core::run_interpreter_step_core;
#[cfg(feature = "vex-engine-z3")]
use crate::state::StateMigrationPayload;
#[cfg(feature = "vex-engine-z3")]
use crate::vex::IRSB;
#[cfg(feature = "vex-engine-z3")]
use std::sync::mpsc::{Receiver, RecvTimeoutError};
#[cfg(feature = "vex-engine-z3")]
use std::time::Duration;

/// Materialized-terminal disposition the parallel worker stamps into
/// [`ParallelShared::kind_map`] so the coordinator can route a state recovered
/// across the `thread::scope` join (which arrives as an untagged
/// [`StateMigrationPayload`]). Found states go to `STASH_FOUND`; `Unconstrained`
/// to that stash; `Bounce` carries the [`BounceKind`] the coordinator replays
/// through `dispatch_bounce` (with empty deferred-fork data — the worker already
/// materialized those forks locally).
///
/// A payload with NO `kind_map` entry (`None`) is the fourth case: a still-live
/// frontier state a worker drained back across the cancel boundary (Bug M1), or
/// the injector surplus the coordinator drained after the barrier. Neither ever
/// reached `run_post_step_core`, so neither can be tagged here; the coordinator
/// routes untagged payloads as bare active successors.
#[cfg(feature = "vex-engine-z3")]
#[derive(Clone)]
pub(crate) enum MatKind {
    Found,
    Unconstrained,
    Bounce(BounceKind),
}

// NOTE (angr-nkoct steady-state Phase B): the duplex-protocol types this file
// used to scaffold (`WorkerCtl` / `WorkerUp` / `RunSession`, with their
// Send/Sync compile-time proof) now live in `scheduler.rs`, implemented and
// unit-tested — the scheduler owns the transport; this file owns routing.

/// The coordinator-side handle to a live steady-state parallel session
/// (angr-nkoct). Held on the manager (`parallel_session`) across `run()`
/// returns while workers keep their frontiers resident. Bundles the scheduler
/// session, the coordinator's receiving half of the upstream channel, and the
/// `Arc`-shared routing maps / profiling the `ProcessFn` closure also holds —
/// so the coordinator can route streamed terminals and fold counters without
/// recovering sole ownership (workers keep their clones for the session's
/// life).
#[cfg(feature = "vex-engine-z3")]
pub(crate) struct SteadySession {
    session: Arc<RunSession>,
    /// Wrapped in a `Mutex` only so `SteadySession: Sync` holds — required
    /// because the coordinator recv's inside `py.detach`, whose closure must be
    /// `Send` (`mpsc::Receiver` is itself `!Sync`). Only the single coordinator
    /// thread ever receives, so the lock is uncontended (mirrors
    /// `PersistentPool::done_rx`).
    up_rx: Mutex<Receiver<WorkerUp>>,
    shared: Arc<ParallelShared>,
    prof: Arc<ParallelProfiling>,
    workers: usize,
    /// Worker ids currently parked (Quiesced/Paused), cleared whenever the
    /// coordinator injects new work and re-wakes. `len() == workers` with
    /// `session.pending() == 0` is the quiescence condition.
    parked: Vec<usize>,
}

#[cfg(feature = "vex-engine-z3")]
impl SteadySession {
    /// Cancel the session and wake every worker so it observes the cancel at
    /// its next task boundary. Does NOT wait for the drain — used by the
    /// manager `Drop`, where the pool's own `join` completes teardown.
    pub(crate) fn cancel_and_wake(&self, pool: Option<&PersistentPool>) {
        self.session.cancel();
        if let Some(pool) = pool {
            for w in 0..self.workers {
                pool.wake_worker(w, &self.session);
            }
        }
    }
}

/// What `steady_pump` hands back to the steady coordinator loop.
#[cfg(feature = "vex-engine-z3")]
enum SteadyOutcome {
    /// Bounce terminals to dispatch through `process_parallel_bounce_queue`
    /// (may be empty — a signal to re-check `num_find` at the loop top).
    Bounces(Vec<(RustSimState, BounceKind, u64)>),
    /// The resident frontier is exhausted (all workers parked, nothing
    /// outstanding).
    Quiesced,
    /// This `run()` hit its step budget; yield to Python.
    Budget,
}

/// How long `finalize_steady_session` waits for each steady worker's `Paused`
/// ack before giving up (angr-e4cys).
///
/// Cancellation is task-boundary-only — `scheduler::CancelToken` is
/// deliberately not a mid-solve Z3 interrupt — so a worker that is inside one
/// Z3 solve when the session is cancelled cannot ack until that solve returns
/// or hits its own timeout. Scaling the drain deadline off the configured
/// solver timeout keeps a healthy-but-slow worker (raised
/// `set_solver_timeout`) from being mistaken for a lost wakeup, while the 60s
/// floor preserves the historical deadline at the 30s default.
#[cfg(feature = "vex-engine-z3")]
fn steady_finalize_deadline(solver_timeout_ms: u32) -> Duration {
    Duration::from_millis(u64::from(solver_timeout_ms).saturating_mul(2))
        .max(Duration::from_secs(60))
}

/// The worker ids that never acked `Paused` before the drain deadline
/// (angr-e4cys): every worker in `0..workers` that is absent from `paused`.
///
/// Named for the timeout error `finalize_steady_session` raises: these are the
/// workers whose resident frontier states are lost. Isolated from the live
/// finalize path so the "which workers are stuck" computation is falsifiable
/// without a running session — a regression that inverts the predicate (naming
/// the *acked* workers) or off-by-ones the range trips the unit tests below.
#[cfg(feature = "vex-engine-z3")]
fn stuck_worker_ids(workers: usize, paused: &[usize]) -> Vec<usize> {
    (0..workers).filter(|w| !paused.contains(w)).collect()
}

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
#[cfg(feature = "vex-engine-z3")]
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
    /// Best-effort worker-side early-cancel HINT (seeded with the manager's
    /// current `found_count`). Bumped ONLY by a worker's pre-step find arm
    /// (`parallel_process_state`); reaching `num_find` there trips the pool
    /// cancel one wave/pump-round early. It is deliberately NOT authoritative:
    /// coordinator-routed finds (a bounce whose target is a find address, or an
    /// untagged residual frontier state sitting at a find pc — both routed by
    /// `route_materialized_terminal` → `push_found_capped`) never touch this
    /// hint, so it under-counts them. The found-set *count* stays exactly
    /// `num_find` regardless, because the authoritative cap is
    /// `push_found_capped`'s `found_count() >= num_find` gate plus the loop-top
    /// `found_count() >= num_find` finalize check — those, not this hint, bound
    /// collection (angr-op0dn.13.17). The only effect of the miss is that a run
    /// whose finds all arrive via the coordinator paths cancels a round later
    /// than one whose finds hit the worker pre-step arm.
    worker_found_hint: AtomicUsize,
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

#[cfg(feature = "vex-engine-z3")]
impl ParallelShared {
    /// The single construction site for a wave-scoped OR session-scoped
    /// `ParallelShared`: empty maps/counters, zeroed `stepped`, and the
    /// early-cancel hint seeded from the manager's current found count. Both
    /// coordinators (`run_parallel_loop`'s per-wave rebuild and
    /// `ensure_steady_session`'s once-per-session build) go through here so a
    /// new field cannot be seeded in one and forgotten in the other
    /// (angr-ph300.12).
    fn seeded(found_count: usize, num_find: usize) -> Self {
        Self {
            root_map: Mutex::new(FxHashMap::default()),
            kind_map: Mutex::new(FxHashMap::default()),
            counters: Mutex::new(Vec::new()),
            worker_found_hint: AtomicUsize::new(found_count),
            num_find,
            stepped: AtomicUsize::new(0),
        }
    }
}

/// Take one routed terminal's lineage root + [`MatKind`] tag from a live steady
/// session's shared maps, **removing** both entries (angr-offd5): a routed
/// terminal never re-enters a worker under the same id, so its `root_map` /
/// `kind_map` entries are dead and must not accumulate over a long session (a
/// re-injected bounce gets its root re-stamped by `steady_inject_resumed`, so
/// the removal is safe there too). The root falls back to the id itself when
/// absent, mirroring the wave-path `root_map.get(&id).copied().unwrap_or(id)`.
///
/// Extracted from [`RustExplorationManager::route_steady_terminal`] so the
/// per-terminal pruning contract is unit-falsifiable without a live session —
/// a revert to a non-removing `.get(&id).copied()` leaves the entry behind and
/// trips the test (angr-n0irt.14). Continue successors' entries are NOT touched
/// here; they are freed wholesale when the session drops.
#[allow(
    clippy::expect_used,
    reason = "root_map/kind_map poison guards: the locks poison only on an impossible panic=abort unwind — see the module Panic policy header"
)]
#[cfg(feature = "vex-engine-z3")]
fn take_steady_routing(shared: &ParallelShared, id: u64) -> (u64, Option<MatKind>) {
    let root = shared
        .root_map
        .lock()
        .expect("root_map poisoned")
        .remove(&id)
        .unwrap_or(id);
    let kind = shared
        .kind_map
        .lock()
        .expect("kind_map poisoned")
        .remove(&id);
    (root, kind)
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
///   `Found` (bumping the worker-side `worker_found_hint`, requesting cancel at
///   `num_find` — a best-effort early trip; coordinator-routed finds rely on the
///   `push_found_capped` cap, not this hint);
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
#[allow(
    clippy::expect_used,
    reason = "poison-guard invariants: the ParallelShared root_map/kind_map/counters locks only poison if a thread unwound while holding them, which panic=abort forecloses — see the module Panic policy header"
)]
#[cfg(feature = "vex-engine-z3")]
fn parallel_process_state(
    mut state: RustSimState,
    cancel: &CancelToken,
    block_cache: &mut LruCache<u64, std::sync::Arc<IRSB>>,
    cc: &CoreCtx,
    callbacks: &PythonCallbacks,
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
    if cc.ctx.avoid_addrs.contains(&pc) {
        return TaskOutcome::summarized(vec![TerminalSummary::of(
            &state,
            SchedDisposition::Avoided,
        )]);
    }
    if cc.ctx.find_addrs.contains(&pc) {
        if cc.ctx.lazy_solves || state.satisfiable() {
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
                shared.worker_found_hint.fetch_add(1, Ordering::SeqCst) + 1 >= shared.num_find;
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

    if cancel.preempts_in_flight() {
        // A peer already hit `num_find` (or the session finalized) while this
        // state was being dispatched.
        // Bail WITHOUT stepping, but hand the state back as an untagged terminal
        // (no `kind_map` entry) so the coordinator routes it to `STASH_ACTIVE`:
        // it is an un-explored frontier state, and dropping it here would lose it
        // exactly like the pre-fix cancel path did (Bug M1, angr-op0dn.13.8).
        //
        // A *budget* cancel (angr-9ke6b.52) deliberately does NOT land here: the
        // dispatch is already charged to `run(n)`'s budget, so the step must
        // complete or a small `n` makes no progress at all. See
        // [`CancelToken::preempts_in_flight`].
        return TaskOutcome {
            continue_states: Vec::new(),
            terminal_states: vec![state],
            terminal_summaries: Vec::new(),
            request_cancel: false,
        };
    }

    // --- Step (PC is not a find/avoid address). ---
    // Warm per-worker block cache (angr-vh834 Work Item 3): `block_cache` is
    // owned by the worker thread and threaded in here, so lifted blocks persist
    // across dispatches AND waves. Cache hits avoid a GIL-serialized re-lift; the
    // warm cache returns identical (pure) `Arc<IRSB>` lifts, so the found set and
    // content fingerprints are unchanged.
    let mut step =
        run_interpreter_step_core(cc.ctx, callbacks, &mut state, pc, None, None, block_cache);

    // Fold this step's FULL interpreter stats into the shared profiling
    // accumulator so every sum-typed counter — lift_time_ns, block-cache
    // hit/miss, blocks_executed, load/store/expr timings, ... — surfaces in
    // `mgr.stats()`, mirroring the single-threaded
    // `accumulated_stats.merge(&step.step_stats)` (stepping.rs). Stamp
    // `step_count = 1` first, exactly as the single-threaded path does. The
    // coordinator merges this into `accumulated_stats` via `prof.fold_into`
    // after the wave barrier. Unconditional: timing fields are zero when
    // profiling is off, and cache counters are always-on (angr-qhkye).
    step.step_stats.step_count = 1;
    cc.prof.accumulate_step(&step.step_stats);

    // Restore the warm cache: `run_interpreter_step_core` swapped a fresh empty
    // placeholder into `*block_cache` and handed the now-populated cache back in
    // `updated_block_cache`. Write it back so the newly-lifted blocks persist for
    // this worker's next dispatch (mirrors the single-threaded restore in
    // `step_state_with_skip`: `self.environment.block_cache = step.updated_block_cache`).
    *block_cache = step.updated_block_cache;

    // State-update preamble (shared with step_state_inner via
    // apply_interpreter_step_result). Worker drops the rest of the per-call
    // step_stats (no shared accumulation in the MVP).
    super::stepping::apply_interpreter_step_result(
        &mut state,
        step.recovered_memory,
        step.new_registers,
        step.new_pc,
        step.symbolic_ip_at_exit,
        step.new_call_stack,
        step.new_detailed_history,
        step.new_tsc_counter,
    );

    let inputs = PostStepInputs {
        result: step.result,
        deferred_forks: step.deferred_forks,
        last_condition: step.last_condition,
        stored_conditions: step.stored_conditions,
        fork_snapshots: step.fork_snapshots,
    };
    let outcome = run_post_step_core(cc, state, inputs, root_hint);

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
                state: mut bstate,
                deferred_forks,
                stored_conditions,
                fork_snapshots,
            } = bounce;
            // Materialize the bounce's deferred forks in-thread so no unexplored
            // branch is lost across the (!Send loose-condition) bounce boundary.
            // Done BEFORE the pc stamp below so the forks are derived from the
            // exact same base state the single-threaded path forks from.
            let (forks, pruned2, _ids) = materialize_bounce_forks(
                cc,
                &bstate,
                deferred_forks,
                stored_conditions,
                fork_snapshots,
                root_hint,
            );
            // angr-op0dn.13.11: the state-update preamble above stamped
            // `step.new_pc` — which is 0 for a hook / SimProcedure bounce — so a
            // bounce state leaves the worker with pc=0. `dispatch_bounce`
            // overwrites the pc for exactly these kinds, so the bounce path does
            // not care; but the coordinator can also route this state back to
            // `STASH_ACTIVE` (the untagged-residual arm of
            // `route_materialized_terminal`, or `flush_parked_bounces_to_active`
            // at snapshot time), where pc=0 makes it die at its next lift
            // ("Lift error at 0x0") and takes its whole subtree with it. Stamp
            // the re-enterable entry address here instead: the state is parked AT
            // the call site with the callback not yet run, so resuming from the
            // bounce target is a faithful replay.
            if let Some(addr) = bounce_target_addr(&kind) {
                bstate.set_pc(addr);
            }
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
    Terminal(TerminalStep),
    /// A Python callback is pending. The `PendingCallback` is returned as a value
    /// (NOT yet stored in `self.pending_callback`) so the driver can build the
    /// event from this local — letting the `PythonVEXFallback` counter mutations
    /// touch `self` while only `pending` is borrowed (no E0502) — before storing.
    NeedCallback(PendingCallback),
}

/// The three terminal step outcomes, each carrying the data the driver needs to
/// reproduce the original per-stash push (and its side effects) byte-for-byte.
pub(crate) enum TerminalStep {
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
        // The parallel/steady coordinator paths only exist with the Z3-backed
        // engine (the scheduler transports `StateMigrationPayload`). Without
        // Z3 there is only the single-threaded loop.
        #[cfg(feature = "vex-engine-z3")]
        {
            if !self.must_run_serial() {
                if self.steady_state_eligible() {
                    return self.run_loop_parallel_steady(py, n);
                }
                return self.run_loop_parallel(py, n);
            }
        }
        #[cfg(not(feature = "vex-engine-z3"))]
        let _ = py;
        self.run_loop_single_threaded(n)
    }
}

/// The parallel/steady-state coordinator half of the run loop. Every method
/// here transports work through the work-stealing `scheduler` (which carries
/// `StateMigrationPayload`), so the whole block is gated on the Z3-backed
/// engine — the `vex-engine`-without-`vex-engine-z3` build has only the
/// single-threaded loop below.
#[cfg(feature = "vex-engine-z3")]
impl RustExplorationManager {
    /// Whether `run_loop` must fall back to the single-threaded loop instead of
    /// any parallel path, given the worker count and registered native
    /// techniques.
    ///
    /// Native techniques (LoopBound / Timeout / LengthLimiter) are
    /// coordinator-side and only run BETWEEN waves via `apply_native_techniques`.
    /// A single wave runs its frontier to quiescence with the GIL released, so on
    /// a non-terminating frontier (e.g. a LoopBound meant to cap a loop) the wave
    /// never returns and the technique never prunes — `run()` hangs and ignores
    /// both the technique and the `run(n)` step budget. The Python driver's
    /// `parallel_eligible` gate downgrades to serial for the kwarg path, but the
    /// `RUST_PARALLEL_WORKERS` env path bypasses that gate (the env value always
    /// wins), so guard here at the engine chokepoint too. The single-threaded
    /// loop applies techniques after every step (angr-ph300.6).
    ///
    /// A pending skip-hook entry (`skip_hook_stack`, populated by Python's
    /// `set_skip_hook_addr` across the zero-length/stale-hook recovery callbacks)
    /// also forces serial: `skip_hook_stack` is read and consumed in exactly one
    /// place — `step_one`'s GAP-6 block — which only the single-threaded loop
    /// reaches. `parallel_process_state` calls `run_interpreter_step_core` with
    /// `skip_addr = None`, so a state resumed into a wave/steady session after a
    /// skip was registered would re-register and immediately re-fire the same
    /// zero-length/stale hook, spinning callback→resume→callback with no
    /// path-side break. The stack is populated between `run_loop` calls (a
    /// callback is returned to Python, which sets the skip and calls `run()`
    /// again), so gating at this entry chokepoint routes the very next `run()`
    /// to serial, where `step_one` consumes the entry and the loop breaks; once
    /// the stack drains, subsequent `run()`s go parallel again (angr-04tw3.1).
    pub(crate) fn must_run_serial(&self) -> bool {
        self.parallel_real_workers <= 1
            || !self.native_techniques.is_empty()
            || !self.skip_hook_stack.is_empty()
    }

    /// Whether the steady-state loop (angr-nkoct) engages for this `run()`.
    /// ALL must hold: opt-in env flag; the Python driver's frontier-residency
    /// promise (address-based explore, no `until`, no techniques — nothing
    /// reads the active stash between `run()` calls); and no callable
    /// find/avoid predicates (the wave/single-threaded skip-state tracking has
    /// no steady analogue). Otherwise fall through to the wave loop.
    ///
    /// Stays env-gated (`parallel_steady_env`), NOT auto-armed by the
    /// `parallel_workers=` kwarg: the found over-collection is fixed
    /// (`push_found_capped` makes the found set worker-invariant, angr-op0dn.13.17)
    /// but steady is still net-negative on the CTF corpus, so enabling it by
    /// default would regress `num_find=1` first-find benches (bd memory
    /// `steady-state-loop-opt-in-net-negative-corpus`).
    fn steady_state_eligible(&self) -> bool {
        self.parallel_steady_env
            && self.parallel_frontier_residency
            && !self.find_needs_python
            && !self.avoid_needs_python
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
    /// **`num_find` early-exit preserves the active frontier (Bug M1, fixed in
    /// angr-op0dn.13.8).** When a wave reaches `num_find`, a worker trips the
    /// shared [`CancelToken`] and every worker stops at its next *task boundary*.
    /// Both halves of the un-explored frontier are drained back rather than
    /// dropped: each worker detaches its un-dispatched local states into the
    /// wave's `results` (`scheduler_worker.rs::worker_loop`), and the coordinator
    /// pulls the never-stolen injector surplus after the barrier
    /// (`WaveJob::drain_residual_payloads`). Both arrive UNTAGGED (no `kind_map`
    /// entry), so `route_materialized_terminal` routes them as bare active
    /// successors — post-explore `active_count()` therefore matches the
    /// single-threaded loop's and the frontier is resumable. `residual_drains`
    /// counts every such state; the serde is bounded by the residual frontier,
    /// never the explored set.
    #[allow(
        clippy::expect_used,
        reason = "poison-guard + local state-machine invariants (root_map/kind_map poison, pool-just-created, post-barrier sole ownership of ParallelShared) — see the module Panic policy header"
    )]
    // The seed loop holds `root_map` across all inserts by design (see the
    // block comment below); clippy resolves `significant_drop_tightening` at
    // the item, so the allow lives here.
    #[allow(clippy::significant_drop_tightening)]
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
                // A prior wave may have parked the tail of its bounce queue in
                // `pending_parallel_bounces` (states living in NO stash) before
                // this wave reached `num_find`. Serial exploration leaves the
                // equivalent frontier in STASH_ACTIVE, so flush the parked
                // bounces back to active here for `active_count` parity — else
                // they are stranded until the next `run()` (which never comes
                // once `found` is reported) or a snapshot flush (angr-ph300.8).
                self.flush_parked_bounces_to_active();
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
                // `found_count()` is always < `num_find` here — path (a) at the
                // loop top returns `found` for `>= num_find` BEFORE draining — so
                // signal `active_empty`, not `found`. The found stash already
                // holds any partial solutions; the event only drives loop
                // control. Returning `found` with a partial count spins the
                // Python explore loop forever (angr-q1mwl): its found-break is
                // gated on `found_count >= num_find`, which a partial can never
                // satisfy, and the active stash never refills.
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
            let prof = Arc::new(ParallelProfiling::default());
            let shared = Arc::new(ParallelShared::seeded(self.found_count(), self.num_find));
            let mut seeds: Vec<StateMigrationPayload> = Vec::with_capacity(drained.len());
            // `rm` is held across the whole seed loop by design: relocking per
            // iteration (clippy's `significant_drop_tightening` suggestion)
            // would thrash the mutex. The allow sits on the enclosing fn —
            // clippy resolves this lint's level at the item, not the block.
            {
                let mut rm = shared.root_map.lock().expect("root_map poisoned");
                for state in drained {
                    // Migration drains STASH_ACTIVE directly, bypassing
                    // policy.select — notify explicitly so a memoizing policy
                    // (e.g. LoopHeadRoundRobin's key_cache) doesn't leak an
                    // entry for a state it will never select again (angr-3xk63).
                    self.policy.on_state_removed(state.state_id());
                    let root = self.sm.root_or_self(state.state_id());
                    rm.insert(state.state_id(), root);
                    seeds.push(state.detach_for_migration());
                }
            }

            // The GIL-free per-state processor, living in the `Arc<WaveJob>` the
            // persistent workers share. The callbacks clone happens HERE, on the
            // GIL thread — `Py<T>::clone` needs the GIL, so it must never run
            // inside a worker.
            let process = self.build_parallel_process(callbacks.clone(), &prof, &shared);
            let mut job = WaveJob::new_with_policy(seeds, process, Arc::clone(&self.policy));
            // Mirror the manager's frontier cap onto the wave: in-wave forks
            // never touch `push_to_active_or_drop`, so this is the only place
            // `max_active_states` is enforced under parallel dispatch
            // (angr-9ke6b.48).
            job.set_max_active_states(self.max_active_states);
            // Bound the wave by the `run(n)` budget this call has LEFT
            // (angr-9ke6b.52). The `dispatched_total >= max_steps` check at the
            // bottom of this loop only fires between waves, and a wave runs its
            // frontier to quiescence — so without this a single `run(n)` on a
            // non-terminating frontier executed unboundedly many steps, unlike
            // the single-threaded loop (checks every iteration) and the steady
            // loop (`steady_pump` polls every 50ms). A spent budget trips the
            // wave's `CancelToken`, so the un-dispatched frontier returns
            // untagged and routes back to `STASH_ACTIVE`, exactly as the serial
            // loop leaves it. Enforcement is soft by up to `workers - 1`
            // dispatches (task-boundary check; see
            // `WorkTransport::max_dispatches`).
            job.set_max_dispatches(Some(max_steps.saturating_sub(dispatched_total)));

            // Release the GIL and run the wave to quiescence on the persistent
            // pool. Workers keep live successors thread-local (the f≈0 path) and
            // materialize only found/bounce/unconstrained terminals across the
            // barrier.
            let pool = self
                .parallel_pool
                .as_ref()
                .expect("parallel pool just created");
            // The barrier's own stats snapshot predates the coordinator's
            // injector drain below, so it is re-taken afterwards (`job.stats()`)
            // to include those `residual_drains`.
            let (job, _barrier_stats) = py.detach(|| pool.run_wave(job));

            // ---- Back on the GIL thread: apply deferred mutations. ----

            // Recover the materialized terminals plus (on a cancelled wave) the
            // injector surplus no worker ever stole — the coordinator half of the
            // Bug M1 cancel-drain; the worker-local half already pushed its
            // residual frontier into `results`. Both are untagged and route back
            // to `STASH_ACTIVE` below. Then DROP the wave (and its process
            // closure) to release the closure's `Arc` clones of `prof` / `shared`
            // — required before `Arc::into_inner` can recover sole ownership of
            // `shared`.
            let mut materialized = job.take_results();
            materialized.extend(job.drain_residual_payloads());
            let stats = job.stats();
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
            let mut bounce_queue: Vec<(RustSimState, BounceKind, u64)> = Vec::new();
            let dropped = self.route_materialized_payloads(
                materialized,
                &mut kind_map,
                &root_map,
                &mut bounce_queue,
            );
            if dropped > 0 {
                log::warn!("wave: dropped {dropped} terminal(s) that failed to reattach");
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
            self.parallel_post_cancel_steps += stats.post_cancel_steps as u64;
            self.parallel_residual_drains += stats.residual_drains as u64;
            log::debug!(
                "wave: seeds={} dispatches={} offloaded={} terminals={} serde_ms={:.1} step_ms={:.1}",
                stats.seeds,
                stats.dispatches(),
                stats.surplus_offloaded,
                stats.materialized_terminals,
                stats.serde_ns as f64 / 1e6,
                stats.step_ns as f64 / 1e6,
            );
            self.fold_scheduler_dispatch_stats(&stats);
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

    /// Build the GIL-free per-state processor both parallel coordinators hand to
    /// the scheduler. It owns a fresh [`StepContext`] snapshot and `Arc`-shares
    /// `prof` / `shared` / the two native registries, so it is `'static` and
    /// `Fn + Sync`; its callbacks self-acquire the GIL via `Python::attach`.
    ///
    /// `callbacks` must already be cloned by the caller WHILE HOLDING THE GIL
    /// (`Py<T>::clone` needs it, and this must never run on a worker). The wave
    /// loop rebuilds one per wave, `ensure_steady_session` one per session
    /// (angr-ph300.12 — previously duplicated verbatim in both).
    fn build_parallel_process(
        &self,
        callbacks: PythonCallbacks,
        prof: &Arc<ParallelProfiling>,
        shared: &Arc<ParallelShared>,
    ) -> Box<ProcessFn> {
        let ctx = self.step_context();
        let procs = Arc::clone(&self.native_procedures);
        let syscalls = Arc::clone(&self.native_syscalls);
        let prof = Arc::clone(prof);
        let shared = Arc::clone(shared);
        Box::new(move |state, cancel, block_cache| {
            let cc = CoreCtx {
                ctx: &ctx,
                prof: &prof,
                native_procs: &procs,
                native_syscalls: &syscalls,
                callbacks: Some(&callbacks),
            };
            parallel_process_state(state, cancel, block_cache, &cc, &callbacks, &shared)
        })
    }

    /// Drain the per-dispatch accounting out of a [`ParallelShared`] through the
    /// shared reference: takes the queued `CoreCounters` (lock + `mem::take`) and
    /// swaps the `stepped` counter to zero, returning its value (M2 semantics —
    /// see the `stepped` field doc). Deliberately does NOT require sole ownership
    /// of the `Arc`, so the steady-state coordinator can fold incrementally at
    /// every event return while workers still hold clones; the wave loop calls it
    /// once per wave right before `Arc::into_inner`.
    #[allow(
        clippy::expect_used,
        reason = "counters-mutex poison guard: poison implies a prior unwind under panic=abort, impossible — see the module Panic policy header"
    )]
    fn fold_parallel_shared_counters(&mut self, shared: &ParallelShared) -> u64 {
        let worker_stepped = shared.stepped.swap(0, Ordering::SeqCst) as u64;
        let drained = std::mem::take(&mut *shared.counters.lock().expect("counters poisoned"));
        for counters in drained {
            self.fold_core_counters(counters);
        }
        worker_stepped
    }

    /// Reattach and route every materialized payload recovered from a wave
    /// barrier, returning the number that could **not** be reattached.
    ///
    /// A failed reattach is logged and skipped, never propagated (angr-ph300.11).
    /// By this point the wave's results have already been taken out of the job
    /// and its residual injector drained, so the payload vec is the *only*
    /// remaining handle on those states: aborting on payload `k` of `n` would
    /// silently destroy the remaining `n - k` states — including materialized
    /// founds — with no stash record. That is strictly worse than dropping the
    /// one payload whose bytes are bad, and it diverges from the steady-state
    /// twins ([`Self::route_steady_terminal`] and `finalize_steady_session`),
    /// which both log-and-drop per state.
    fn route_materialized_payloads(
        &mut self,
        materialized: Vec<StateMigrationPayload>,
        kind_map: &mut FxHashMap<u64, MatKind>,
        root_map: &FxHashMap<u64, u64>,
        bounce_queue: &mut Vec<(RustSimState, BounceKind, u64)>,
    ) -> usize {
        let main_ctx = z3::Context::thread_local();
        let mut dropped = 0usize;
        for payload in materialized {
            let state = match payload.reattach(&main_ctx) {
                Ok(s) => s,
                Err(e) => {
                    log::error!("wave: reattach failed, dropping terminal: {e:?}");
                    dropped += 1;
                    continue;
                }
            };
            let id = state.state_id();
            let root = root_map.get(&id).copied().unwrap_or(id);
            let kind = kind_map.remove(&id);
            self.route_materialized_terminal(state, kind, root, bounce_queue);
        }
        dropped
    }

    /// Route one reattached materialized terminal by its [`MatKind`] tag —
    /// the per-payload body of the wave loop's post-barrier routing, extracted
    /// so the steady-state coordinator can route terminals one at a time as they
    /// stream in. The caller looks up the state's `kind_map`/`root_map` entries
    /// and passes them per-state — the steady caller also REMOVES both, since
    /// entries for a routed terminal are dead and must not accumulate over a
    /// long session (the wave caller owns maps that die with the wave, so it
    /// only reads); true bounces are pushed to `bounce_queue` for
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
                self.push_found_capped(state);
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
                            self.push_found_capped(state);
                        } else {
                            log::debug!(
                                "parallel: bounce at find addr 0x{addr:x} is UNSAT, pruning"
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
            None => {
                // A still-live frontier state drained back across the cancel
                // boundary (Bug M1): both the worker-local drain and the
                // coordinator's injector drain emit these UNTAGGED (the worker
                // never ran `run_post_step_core` on them, so there is no
                // `kind_map` entry to stamp). Semantically it
                // is a bare active successor — one the single-threaded loop
                // would have left sitting in `STASH_ACTIVE` — so route it as
                // one, honoring find/avoid addresses exactly as `route_successor`
                // does for a fresh successor. The one deviation is the find gate:
                // a residual sitting at a find pc is capped at `num_find`
                // (angr-op0dn.13.17) so the parallel drain does not over-collect
                // past the serial baseline; surplus stays active + re-findable.
                self.sm.set_root(id, root);
                let spc = state.pc();
                if self.find_addrs.contains(&spc)
                    && (self.constraint_solver.lazy_solves || state.satisfiable())
                {
                    self.push_found_capped(state);
                } else {
                    self.route_successor(state, true);
                }
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
                    self.apply_terminal(TerminalStep::Deadended(s));
                    self.steps += 1;
                }
                Err(StepError::Error(s, message)) => {
                    let pc = s.pc();
                    let state_id = s.state_id();
                    self.apply_terminal(TerminalStep::Errored {
                        state: s,
                        pc,
                        message,
                        state_id,
                    });
                    self.steps += 1;
                }
                Err(StepError::Unconstrained(s, forks)) => {
                    self.apply_terminal(TerminalStep::Unconstrained { state: s, forks });
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

    // =====================================================================
    // Steady-state coordinator (angr-nkoct). One long-lived `SteadySession`
    // spans many `run()` calls: workers keep their frontiers RESIDENT across
    // the Python-callback boundary, so a bounce costs one materialize +
    // re-inject instead of a full-frontier detach/reattach re-seed. Engaged
    // only when `steady_state_eligible()`; the wave loop remains the fallback.
    // =====================================================================

    /// The steady-state parallel run loop. Reuses `parallel_process_state` +
    /// `ParallelShared` (session-scoped instead of wave-scoped) and the shared
    /// `route_materialized_terminal` / `process_parallel_bounce_queue` routing
    /// helpers; the difference from the wave loop is purely the driver:
    /// terminals stream up an mpsc channel and workers stay resident rather
    /// than synchronizing at a per-wave barrier.
    pub(crate) fn run_loop_parallel_steady(
        &mut self,
        py: Python<'_>,
        n: Option<u32>,
    ) -> PyResult<ExplorationEvent> {
        let callbacks = self
            .callbacks
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("callbacks not set"))?
            .clone();
        if !callbacks.is_ready() {
            return Err(PyRuntimeError::new_err("callbacks not ready"));
        }

        let _gil_wall = crate::gil_profile::RunLoopWallGuard::new(self.profiling.profiling_enabled);
        let max_steps = n.unwrap_or(self.max_steps_per_run) as u64;

        // Create the session on first entry; on re-entry (after a need_callback
        // return) it is already live with workers stepping resident frontiers.
        self.ensure_steady_session();
        let dispatched_at_entry = self.steady_dispatched_total();

        loop {
            // I8(a): enough solutions — finalize (drains the resident frontier
            // back to STASH_ACTIVE so the post-run stashes are truthful) and
            // report found.
            if self.found_count() >= self.num_find {
                self.finalize_steady_session(py)?;
                // Flush any bounces a prior run() parked in
                // `pending_parallel_bounces` back to STASH_ACTIVE for
                // `active_count` parity with serial exploration (angr-ph300.8).
                // Runs after finalize so the resident-id guard in
                // `flush_parked_bounces_to_active` sees the drained frontier.
                self.flush_parked_bounces_to_active();
                return Ok(ExplorationEvent::found(
                    self.found_count(),
                    self.active_count(),
                    self.steps,
                ));
            }

            // Dispatch bounces parked from a prior run() (M3) straight through
            // dispatch_bounce; natively-resolved successors land in STASH_ACTIVE
            // and are re-injected below, first-needing-callback returns its
            // event with the session left LIVE.
            if !self.pending_parallel_bounces.is_empty() {
                let queue = std::mem::take(&mut self.pending_parallel_bounces);
                if let Some(event) = self.process_parallel_bounce_queue(&callbacks, queue) {
                    // The session stays LIVE across this return, so fold the
                    // workers' accounting now — otherwise mid-session `stats()`
                    // reports a stale step/counter total until the next
                    // finalize (angr-offd5).
                    self.fold_steady_counters_incrementally();
                    return Ok(event);
                }
            }

            // Feed any freshly-routed active states (initial seeds on first
            // entry; natively-resolved bounce successors on later passes) into
            // the session and wake the workers.
            self.seed_steady_session_from_active();

            // Pump worker reports until something needs the coordinator's
            // attention. The recv waits run under `py.detach` (GIL released) so
            // workers — which self-acquire the GIL for lifts/callbacks — never
            // block on us.
            match self.steady_pump(py, &callbacks, dispatched_at_entry, max_steps)? {
                SteadyOutcome::Bounces(queue) => {
                    if let Some(event) = self.process_parallel_bounce_queue(&callbacks, queue) {
                        // Session left LIVE across the callback — fold now
                        // (angr-offd5).
                        self.fold_steady_counters_incrementally();
                        return Ok(event);
                    }
                    // Natively-resolved bounce successors are now in
                    // STASH_ACTIVE; loop to re-seed and keep pumping.
                }
                SteadyOutcome::Quiesced => {
                    // Frontier exhausted with no bounces outstanding. Finalize
                    // (a no-op drain — workers are already parked empty) and
                    // report.
                    self.finalize_steady_session(py)?;
                    // `found_count()` is always < `num_find` here (path (a)
                    // returns `found` for `>= num_find` before we reach a
                    // Quiesced outcome), so signal `active_empty`, not `found` —
                    // a partial `found` count spins the Python explore loop
                    // forever (angr-q1mwl).
                    return Ok(ExplorationEvent::active_empty(
                        self.found_count(),
                        self.steps,
                    ));
                }
                SteadyOutcome::Budget => {
                    self.parallel_steady_budget_yields += 1;
                    self.finalize_steady_session(py)?;
                    return Ok(ExplorationEvent::step_complete(
                        self.found_count(),
                        self.active_count(),
                        self.steps,
                    ));
                }
            }
        }
    }

    /// Lazily create the persistent pool + steady session. On re-entry with a
    /// live session this is a no-op. Builds one `ParallelShared` + `ProcessFn`
    /// for the session's whole life (unlike the wave loop's per-wave rebuild),
    /// so the resident frontier is stepped against a stable config snapshot —
    /// the `steady_config_guard` finalizes the session before any config
    /// mutation, which is what keeps that snapshot valid.
    #[allow(
        clippy::expect_used,
        reason = "state-machine invariants: callbacks are checked by the caller and the parallel pool is set before this runs — see the module Panic policy header"
    )]
    fn ensure_steady_session(&mut self) {
        if self.parallel_session.is_some() {
            return;
        }
        let workers = self.parallel_real_workers.max(2);
        if self.parallel_pool.is_none() {
            self.parallel_pool = Some(PersistentPool::new(workers));
        }
        let prof = Arc::new(ParallelProfiling::default());
        let shared = Arc::new(ParallelShared::seeded(self.found_count(), self.num_find));
        let proc_callbacks = self
            .callbacks
            .as_ref()
            .expect("callbacks checked by caller")
            .clone();
        let process = self.build_parallel_process(proc_callbacks, &prof, &shared);
        // The frontier cap travels with the session for its whole lifetime —
        // a steady session never rounds its frontier through `STASH_ACTIVE`,
        // so this is the only `max_active_states` enforcement it gets
        // (angr-9ke6b.48).
        let (session, up_rx) =
            RunSession::new_with_policy(process, Arc::clone(&self.policy), self.max_active_states);
        let workers = self.parallel_pool.as_ref().expect("pool set").num_workers();
        self.parallel_pool
            .as_ref()
            .expect("pool set")
            .start_session(&session);
        self.parallel_session = Some(SteadySession {
            session,
            up_rx: Mutex::new(up_rx),
            shared,
            prof,
            workers,
            parked: Vec::new(),
        });
    }

    /// Drain STASH_ACTIVE into the live session's injector (stamping each
    /// state's lineage root into `root_map` so descendants inherit it) and wake
    /// the workers. No-op when the stash is empty (the common re-entry case:
    /// resume feeds the session injector directly).
    #[allow(
        clippy::expect_used,
        reason = "session-live + root_map poison guards: the session is seeded live and the lock only poisons on an impossible panic=abort unwind — see the module Panic policy header"
    )]
    // The seed loop holds `root_map` across all inserts by design; clippy
    // resolves `significant_drop_tightening` at the item, so the allow is here.
    #[allow(clippy::significant_drop_tightening)]
    fn seed_steady_session_from_active(&mut self) {
        let drained: Vec<RustSimState> = match self.sm.get_mut(STASH_ACTIVE) {
            Some(s) if !s.is_empty() => s.drain(..).collect(),
            _ => return,
        };
        let sess = self
            .parallel_session
            .as_mut()
            .expect("session live during seed");
        let mut payloads = Vec::with_capacity(drained.len());
        // `rm` is held across the whole seed loop by design: relocking per
        // iteration would thrash the mutex.
        {
            let mut rm = sess.shared.root_map.lock().expect("root_map poisoned");
            for state in drained {
                // Same drain-without-notify pattern as the wave-mode migration
                // drain above — this seeds a live parallel session at startup,
                // bypassing policy.select (angr-3xk63).
                self.policy.on_state_removed(state.state_id());
                let root = self.sm.root_or_self(state.state_id());
                rm.insert(state.state_id(), root);
                payloads.push(state.detach_for_migration());
            }
        }
        sess.session.inject_seeds(payloads);
        // New work landed: every parked worker must be re-pinged, and our
        // parked-tracking is stale.
        sess.parked.clear();
        if let Some(pool) = &self.parallel_pool {
            for w in 0..sess.workers {
                pool.wake_worker(w, &sess.session);
            }
        }
    }

    /// Inject a batch of `(state_id, root, payload)` resumed successors into the
    /// live session (angr-nkoct): stamp each root into `root_map`, inject as
    /// `resume_reinjects`, and wake the workers. Called by
    /// `route_resume_successors` (resume.rs) after it partitions off find/avoid
    /// successors; no-op on an empty batch or absent session.
    #[cfg(feature = "vex-engine-z3")]
    #[allow(
        clippy::expect_used,
        reason = "root_map poison guard: poison implies a prior unwind under panic=abort, impossible — see the module Panic policy header"
    )]
    pub(crate) fn steady_inject_resumed(&mut self, inject: Vec<(u64, u64, StateMigrationPayload)>) {
        if inject.is_empty() {
            return;
        }
        let Some(sess) = self.parallel_session.as_mut() else {
            return;
        };
        {
            let mut rm = sess.shared.root_map.lock().expect("root_map poisoned");
            for (id, root, _) in &inject {
                rm.insert(*id, *root);
            }
        }
        let payloads: Vec<_> = inject.into_iter().map(|(_, _, p)| p).collect();
        sess.session.inject_resumed(payloads);
        sess.parked.clear();
        if let Some(pool) = &self.parallel_pool {
            for w in 0..sess.workers {
                pool.wake_worker(w, &sess.session);
            }
        }
    }

    /// Fold the live session's worker accounting into the manager WITHOUT
    /// finalizing it (angr-offd5), honouring the incremental-fold contract
    /// documented on [`Self::fold_parallel_shared_counters`].
    ///
    /// `finalize_steady_session` is the only other folder, and a steady session
    /// stays live across every `need_callback` return — so on a bounce-heavy
    /// explore that never quiesces, `stats()['steps']` and every `CoreCounters`
    /// total would otherwise sit stale for the whole session. Both the counter
    /// drain and the `stepped` read are M2 swaps, so folding early and folding
    /// again at finalize cannot double-count.
    #[cfg(feature = "vex-engine-z3")]
    fn fold_steady_counters_incrementally(&mut self) {
        let Some(shared) = self
            .parallel_session
            .as_ref()
            .map(|s| Arc::clone(&s.shared))
        else {
            return;
        };
        let worker_stepped = self.fold_parallel_shared_counters(&shared);
        self.steps += worker_stepped;
    }

    /// Monotonic dispatch count for the live session (0 if none) — the steady
    /// analogue of the wave loop's `dispatched_total`, used for the step budget.
    fn steady_dispatched_total(&self) -> u64 {
        self.parallel_session
            .as_ref()
            .map(|s| s.session.stats().dispatches() as u64)
            .unwrap_or(0)
    }

    /// Pump worker `WorkerUp` reports, routing terminals as they stream in,
    /// until the coordinator must act: bounces to dispatch, quiescence, or the
    /// step budget. Found/unconstrained/residual terminals route immediately;
    /// bounce terminals accumulate and are returned in a batch (so the caller
    /// surfaces at most one Python callback per `run()` via
    /// `process_parallel_bounce_queue`). All recv waits release the GIL.
    #[allow(
        clippy::expect_used,
        reason = "session-live + up_rx poison guards: the pump only runs while the session is live and the lock poisons only on an impossible panic=abort unwind — see the module Panic policy header"
    )]
    fn steady_pump(
        &mut self,
        py: Python<'_>,
        _callbacks: &PythonCallbacks,
        dispatched_at_entry: u64,
        max_steps: u64,
    ) -> PyResult<SteadyOutcome> {
        let mut bounce_queue: Vec<(RustSimState, BounceKind, u64)> = Vec::new();
        loop {
            // Every arm below that can push into `bounce_queue` returns in the
            // same breath, so the queue is necessarily empty at the loop top —
            // the budget yield can never strand a pending bounce (angr-ph300.14).
            debug_assert!(
                bounce_queue.is_empty(),
                "steady_pump reached loop top with {} pending bounces",
                bounce_queue.len()
            );
            // Budget check (approximate, mirrors the wave loop): finalize and
            // yield to Python once this run() has dispatched its allotment.
            if self
                .steady_dispatched_total()
                .saturating_sub(dispatched_at_entry)
                >= max_steps
            {
                return Ok(SteadyOutcome::Budget);
            }

            let recv = {
                let sess = self
                    .parallel_session
                    .as_ref()
                    .expect("session live in pump");
                py.detach(|| {
                    sess.up_rx
                        .lock()
                        .expect("up_rx poisoned")
                        .recv_timeout(Duration::from_millis(50))
                })
            };
            match recv {
                Ok(WorkerUp::Terminal { payload }) => {
                    self.route_steady_terminal(payload, &mut bounce_queue);
                    // Surface accumulated bounces promptly (they block their
                    // lineage on a Python hook); found short-circuits win too.
                    if !bounce_queue.is_empty() {
                        return Ok(SteadyOutcome::Bounces(std::mem::take(&mut bounce_queue)));
                    }
                    if self.found_count() >= self.num_find {
                        return Ok(SteadyOutcome::Bounces(Vec::new())); // caller re-checks num_find
                    }
                }
                Ok(WorkerUp::Quiesced { worker_id }) | Ok(WorkerUp::Paused { worker_id }) => {
                    let sess = self.parallel_session.as_mut().expect("session live");
                    if !sess.parked.contains(&worker_id) {
                        sess.parked.push(worker_id);
                    }
                    if sess.parked.len() >= sess.workers && sess.session.pending() == 0 {
                        // All workers parked and no queued/in-flight work: drain
                        // any straggler terminals, then it's genuine quiescence
                        // (unless bounces are pending — hand those back first).
                        self.drain_steady_stragglers(&mut bounce_queue);
                        if !bounce_queue.is_empty() {
                            return Ok(SteadyOutcome::Bounces(std::mem::take(&mut bounce_queue)));
                        }
                        return Ok(SteadyOutcome::Quiesced);
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    // Periodic re-check of quiescence/budget (guards against a
                    // lost wakeup): loop re-evaluates the conditions above.
                    let sess = self.parallel_session.as_ref().expect("session live");
                    if sess.parked.len() >= sess.workers && sess.session.pending() == 0 {
                        self.drain_steady_stragglers(&mut bounce_queue);
                        if !bounce_queue.is_empty() {
                            return Ok(SteadyOutcome::Bounces(std::mem::take(&mut bounce_queue)));
                        }
                        return Ok(SteadyOutcome::Quiesced);
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // Every worker's Sender dropped — the pool is gone. Nothing
                    // more can arrive; treat as quiescence.
                    return Ok(SteadyOutcome::Quiesced);
                }
            }
        }
    }

    /// Non-blocking drain of any terminals still buffered in the session mpsc
    /// (routes found/unconstrained/residual, accumulates bounces). Called once
    /// all workers have parked, where no further message can be produced.
    #[allow(
        clippy::expect_used,
        reason = "session-live + up_rx poison guards: draining runs only while the session is live and the lock poisons only on an impossible panic=abort unwind — see the module Panic policy header"
    )]
    fn drain_steady_stragglers(&mut self, bounce_queue: &mut Vec<(RustSimState, BounceKind, u64)>) {
        loop {
            let msg = {
                let sess = self.parallel_session.as_ref().expect("session live");
                sess.up_rx.lock().expect("up_rx poisoned").try_recv()
            };
            match msg {
                Ok(WorkerUp::Terminal { payload }) => {
                    self.route_steady_terminal(payload, bounce_queue)
                }
                Ok(WorkerUp::Quiesced { worker_id }) | Ok(WorkerUp::Paused { worker_id }) => {
                    let sess = self.parallel_session.as_mut().expect("session live");
                    if !sess.parked.contains(&worker_id) {
                        sess.parked.push(worker_id);
                    }
                }
                Err(_) => break,
            }
        }
    }

    /// Reattach one streamed terminal into the main Z3 context and route it via
    /// the shared `route_materialized_terminal` helper. The coordinator owns
    /// `kind_map`/`root_map` and removes the routed state's OWN entry from both
    /// as it routes — a terminal never re-enters a worker under the same id, so
    /// its entries are dead (angr-offd5). A bounce that IS re-injected gets its
    /// root re-stamped by `steady_inject_resumed`, so the removal is safe there
    /// too.
    ///
    /// This bounds the maps by the live frontier only in the terminal
    /// dimension: `root_map` still retains one entry per `Continue` successor
    /// for the session's lifetime (those states are still being stepped, and
    /// nothing signals when a subtree finishes), so it is cleared wholesale when
    /// the session is dropped in `finalize_steady_session`.
    ///
    /// Bounce roundtrips are counted here; the returned `bounce_queue` is
    /// dispatched by the caller.
    #[allow(
        clippy::expect_used,
        reason = "session-live + root_map/kind_map poison guards: routing runs only while the session is live and the locks poison only on an impossible panic=abort unwind — see the module Panic policy header"
    )]
    fn route_steady_terminal(
        &mut self,
        payload: StateMigrationPayload,
        bounce_queue: &mut Vec<(RustSimState, BounceKind, u64)>,
    ) {
        let main_ctx = z3::Context::thread_local();
        let state = match payload.reattach(&main_ctx) {
            Ok(s) => s,
            Err(e) => {
                log::error!("steady: reattach failed, dropping terminal: {e:?}");
                return;
            }
        };
        let id = state.state_id();
        let sess = self.parallel_session.as_ref().expect("session live");
        let (root, kind) = take_steady_routing(&sess.shared, id);
        if matches!(kind, Some(MatKind::Bounce(_))) {
            sess.session.count_bounce_roundtrip();
        }
        let before = bounce_queue.len();
        self.route_materialized_terminal(state, kind, root, bounce_queue);
        // A bounce routed to a find/avoid address short-circuits inside the
        // helper (no push); if it landed in FOUND, the outer num_find check
        // will finalize + cancel, stopping the workers.
        let _ = before;
    }
}

impl RustExplorationManager {
    /// Finalize the live steady session (angr-nkoct): cancel it, wake every
    /// worker so it observes the cancel and drains its resident frontier
    /// upstream, route those residuals + the injector surplus back to
    /// STASH_ACTIVE, and fold the session's counters/profiling into the
    /// manager. No-op without a live session. This is the ONLY path that
    /// returns resident frontier states to the stashes, so it must run on
    /// every non-`need_callback` exit and before any config mutation (the
    /// `steady_config_guard`).
    #[cfg(feature = "vex-engine-z3")]
    #[allow(
        clippy::expect_used,
        reason = "up_rx + root_map/kind_map poison guards: finalize consumes the live session and the locks poison only on an impossible panic=abort unwind — see the module Panic policy header"
    )]
    pub(crate) fn finalize_steady_session(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(mut sess) = self.parallel_session.take() else {
            return Ok(());
        };
        sess.session.cancel();
        if let Some(pool) = &self.parallel_pool {
            for w in 0..sess.workers {
                pool.wake_worker(w, &sess.session);
            }
        }

        // Collect every worker's Paused ack, routing residual terminals as they
        // stream in. All recv waits release the GIL (workers may self-acquire
        // it while draining). A generous deadline turns a lost-wakeup bug into a
        // visible error rather than a silent hang.
        //
        // The deadline is derived from the configured solver timeout rather
        // than hardcoded (angr-e4cys): cancellation is task-boundary-only
        // (`scheduler::CancelToken` is deliberately NOT a mid-solve Z3
        // interrupt), so a worker inside one solve cannot ack until that solve
        // returns. A user who raised `set_solver_timeout` past 30s would
        // otherwise trip the drain deadline on a healthy-but-slow worker.
        let deadline = steady_finalize_deadline(self.constraint_solver.solver_timeout_ms);
        let mut paused: Vec<usize> = Vec::new();
        let mut residuals: Vec<StateMigrationPayload> = Vec::new();
        let mut timed_out = false;
        while paused.len() < sess.workers {
            let recv = py.detach(|| {
                sess.up_rx
                    .lock()
                    .expect("up_rx poisoned")
                    .recv_timeout(deadline)
            });
            match recv {
                Ok(WorkerUp::Terminal { payload }) => residuals.push(payload),
                Ok(WorkerUp::Paused { worker_id }) => {
                    if !paused.contains(&worker_id) {
                        paused.push(worker_id);
                    }
                }
                Ok(WorkerUp::Quiesced { worker_id }) => {
                    // A worker that quiesced before observing cancel: the wake
                    // ping re-enters it, it sees cancel, drains, and acks
                    // Paused. Count nothing yet.
                    let _ = worker_id;
                }
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {
                    // Do NOT bail out here (angr-e4cys): an early return drops
                    // `sess` — and with it every residual already drained
                    // upstream plus this session's counters — before the
                    // routing/fold block below runs. Break instead, salvage
                    // what arrived, then surface the error at the end. Only the
                    // still-stuck workers' states are lost, and they are named.
                    timed_out = true;
                    break;
                }
            }
        }
        // Injector surplus (offloaded but never stolen) — safe now that every
        // worker is parked.
        residuals.extend(sess.session.drain_residual_payloads());

        // Route residuals back to STASH_ACTIVE (untagged ⇒ active successor).
        let main_ctx = z3::Context::thread_local();
        let mut discard: Vec<(RustSimState, BounceKind, u64)> = Vec::new();
        for payload in residuals {
            let state = match payload.reattach(&main_ctx) {
                Ok(s) => s,
                Err(e) => {
                    log::error!("steady finalize: reattach failed, dropping residual: {e:?}");
                    continue;
                }
            };
            let id = state.state_id();
            let root = sess
                .shared
                .root_map
                .lock()
                .expect("root_map poisoned")
                .get(&id)
                .copied()
                .unwrap_or(id);
            let kind = sess
                .shared
                .kind_map
                .lock()
                .expect("kind_map poisoned")
                .remove(&id);
            self.route_materialized_terminal(state, kind, root, &mut discard);
        }
        // A residual should never be a bounce (bounces materialize during the
        // pump, not the drain); if one slips through, route it as active so
        // nothing is lost.
        for (state, _kind, root) in discard {
            let id = state.state_id();
            self.sm.set_root(id, root);
            self.route_successor(state, true);
        }

        // Fold session accounting into the manager (counters + profiling).
        sess.parked.clear();
        let stats = sess.session.stats();
        // Same M2 swap + `CoreCounters` drain the wave loop performs — shared so
        // the two paths cannot drift in counter semantics (angr-ph300.12).
        let worker_stepped = self.fold_parallel_shared_counters(&sess.shared);
        self.steps += worker_stepped;
        self.parallel_tasks += stats.dispatches() as u64;
        self.parallel_migrations += (stats.surplus_offloaded + stats.materialized_terminals) as u64;
        self.parallel_reattaches += stats.reattaches as u64;
        self.parallel_bounce_roundtrips += stats.bounce_roundtrips as u64;
        self.parallel_resume_reinjects += stats.resume_reinjects as u64;
        self.parallel_residual_drains += stats.residual_drains as u64;
        self.parallel_post_cancel_steps += stats.post_cancel_steps as u64;
        self.fold_scheduler_dispatch_stats(&stats);
        sess.prof.drain_into(&mut self.profiling.accumulated_stats);

        self.apply_uniqueness_filter();
        self.apply_native_techniques();
        if timed_out {
            let stuck = stuck_worker_ids(sess.workers, &paused);
            return Err(PyRuntimeError::new_err(format!(
                "steady finalize timed out after {:?} waiting for workers {stuck:?} to drain \
                 ({} of {} acked); their resident frontier states are lost. Residuals that did \
                 arrive were routed and counters folded. If a single solve legitimately runs \
                 longer than this, raise set_solver_timeout — the drain deadline scales with it.",
                deadline,
                paused.len(),
                sess.workers,
            )));
        }
        Ok(())
    }

    /// Finalize a live steady session if a config mutation is about to
    /// invalidate its snapshotted `StepContext` (find/avoid addrs, hooks,
    /// solver/memory config) or touch the active stash. GIL-free build stub is
    /// a no-op. Called from the guarded `set_*` / `register_*` pymethods.
    #[cfg(feature = "vex-engine-z3")]
    pub(crate) fn steady_config_guard(&mut self) {
        if self.parallel_session.is_some() {
            // A pymethod may be called without a Python token in hand, but we
            // are always on the GIL thread here (pymethods hold the GIL), so
            // reacquire it to drain the session.
            let _ = Python::attach(|py| self.finalize_steady_session(py));
        }
    }

    #[cfg(not(feature = "vex-engine-z3"))]
    pub(crate) fn steady_config_guard(&mut self) {}

    /// Materialize `pending_parallel_bounces` back into `STASH_ACTIVE` so any
    /// consumer that only reads stashes can see them (angr-op0dn.13.10):
    /// `dump_snapshot_bytes`, `finalize_parallel_session` (explore-end stash
    /// accounting), and `run_loop_single_threaded` (angr-05kiw — the only
    /// route that never drains the queue itself).
    ///
    /// A wave that surfaces one `need_callback` event parks the REST of its
    /// bounce queue in `pending_parallel_bounces` — states that live in NO
    /// stash, dispatched by the next `run()` (see `process_parallel_bounce_queue`).
    /// `dump_snapshot` only serializes stashes, so a snapshot taken while
    /// bounces are parked silently drops those states and every path behind
    /// them: the same frontier resumed in-memory reaches every leaf, while the
    /// snapshot resumes with a truncated one.
    ///
    /// A parked bounce is parked AT its call site with the callback not yet
    /// run, so re-entering it is a faithful replay: restore the pc to the
    /// bounce target and push the state back to active, where the next step
    /// re-lifts the hook and bounces again. Only kinds with a re-enterable
    /// entry address ([`bounce_target_addr`] — `Hook` / `SimProcedurePython`)
    /// can be replayed that way; the worker zeroed the pc of the others
    /// (`state.set_pc(step.new_pc)` in `parallel_process_state`), so they have
    /// no recoverable resume point and stay parked — logged, not silently
    /// dropped. Ids already resident in a stash are skipped so the flush can
    /// never double-insert a state.
    ///
    /// Deliberately `set_pc` WITHOUT `add_to_history`, unlike the sibling
    /// bounce-restore sites (`dispatch_bounce`'s `Hook` /
    /// `SimProcedurePython` arms and `route_materialized_terminal`'s
    /// find/avoid short-circuit), which pair the two. The bounce target is
    /// appended to history exactly once, by whichever site is *last* to touch
    /// the state: those siblings are terminal for this bounce (the callback
    /// dispatch, or a FOUND/AVOID push that is never stepped again), so they
    /// must append it themselves — the worker core's `bounce()` does not.
    /// This path is not terminal: the state goes back to `STASH_ACTIVE` and
    /// its next step re-lifts the hook and re-enters `dispatch_bounce`, which
    /// appends the address then. Appending here too would push it twice for
    /// one visit, since `add_to_history` never dedups.
    pub(crate) fn flush_parked_bounces_to_active(&mut self) {
        if self.pending_parallel_bounces.is_empty() {
            return;
        }
        let resident: std::collections::HashSet<u64> = self
            .sm
            .stashes()
            .values()
            .flat_map(|states| states.iter().map(|s| s.state_id()))
            .collect();
        let parked = std::mem::take(&mut self.pending_parallel_bounces);
        let mut kept = Vec::new();
        for (mut state, kind, root) in parked {
            let id = state.state_id();
            match bounce_target_addr(&kind) {
                Some(addr) if !resident.contains(&id) => {
                    state.set_pc(addr);
                    self.sm.set_root(id, root);
                    self.route_successor(state, true);
                }
                _ => {
                    log::warn!(
                        "parked bounce for state {id} has no re-enterable entry \
                         address (kind={kind:?}); it stays live in this manager but \
                         will NOT appear in any stash (snapshot / stash_counts)"
                    );
                    kept.push((state, kind, root));
                }
            }
        }
        self.pending_parallel_bounces = kept;
    }

    /// Inner body of the pymethods-exposed `run`. See `run` in `mod.rs`.
    ///
    /// Thin driver over `step_one`: owns the loop frame (I8 termination,
    /// state pop, post-step bookkeeping) and routes each `StepOutcome`.
    pub(crate) fn run_loop_single_threaded(
        &mut self,
        n: Option<u32>,
    ) -> PyResult<ExplorationEvent> {
        // A previous wave may have parked the tail of its bounce queue in
        // `pending_parallel_bounces` (states living in NO stash). Only the two
        // parallel loops drain that queue, so a `run()` that routes here
        // instead — worker count dropped to 1, a native technique registered,
        // or a callable find/avoid predicate set between calls — would strand
        // those states permanently and silently (angr-05kiw). Replay them into
        // STASH_ACTIVE first so every route consumes the queue.
        self.flush_parked_bounces_to_active();

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

            // Get next state from active stash via the selection policy
            // (angr-a32jl.1): Fifo=pop_front (BFS), Lifo=pop_back (DFS).
            // `policy` is bound first so the closure captures only that field,
            // leaving `self.sm` free to borrow mutably.
            let policy = &*self.policy;
            let state = match self.sm.get_mut(STASH_ACTIVE).and_then(|s| policy.select(s)) {
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
                    // I8 termination path (b): active stash exhausted. Always
                    // signal `active_empty` — `found_count()` is provably <
                    // `num_find` here (path (a) at the loop top returns `found`
                    // for `>= num_find` BEFORE this drain), so emitting `found`
                    // with a partial count spins the Python explore loop forever
                    // (angr-q1mwl): its found-break needs `found_count >=
                    // num_find`, unreachable for a partial, and active never
                    // refills. The found stash still carries any partial
                    // solutions. See module header.
                    return Ok(ExplorationEvent::active_empty(
                        self.found_count(),
                        self.steps,
                    ));
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
                log::debug!("State at find address 0x{pc:x} is UNSAT, pruning");
                self.push_or_drop_terminal(STASH_PRUNED, state);
            }
            return Ok(StepOutcome::Routed);
        }

        // Check hooks (SimProcedures)
        // GAP 6: stack-based skip tracking for zero-length hooks. Expires
        // stale entries and pops at most one token for `pc`; see
        // `consume_skip_hook`.
        let should_skip_hook = self.consume_skip_hook(pc);
        if self.hooks.contains(&pc) && !should_skip_hook {
            // Check if this is a registered SimProcedure
            if let Some((name, num_args, no_return)) = self.simprocedures.get(&pc).cloned() {
                // Skip native for main-object hooks (user-placed `proj.hook()`
                // overrides); see `execution_env::prefer_native_dispatch`.
                let prefer_native = self.environment.prefer_native_dispatch(pc);
                // Try native procedure first (only for external/library hooks)
                if prefer_native
                    && let Some(native_proc) = self.native_procedures.get(&name).cloned()
                {
                    // Extract arguments from state registers (and stack when
                    // num_args exceeds the register count). On failure (symbolic
                    // SP, unmapped stack slot) `dispatch_native_proc` skips the
                    // native fast path and falls through to the Python
                    // SimProcedure callback below — handing the native handler
                    // fabricated zeros would mask the underlying stack-setup bug.
                    //
                    // `num_args` is the Python SimProcedure's FIXED-arg count
                    // (variadics excluded). Native procs that consume variadic
                    // pointers (scanf family) declare a larger `num_args()`; use
                    // the max so `extract_procedure_args` reads the full window.
                    // Truncating to the Python count made the scanf family a
                    // silent no-op end-to-end (angr-8onrp).
                    //
                    // The registry entry is cloned (cheap `Arc`) so the shared
                    // dispatcher can take `&mut self.profiling` alongside it.
                    let native_num_args = num_args.max(native_proc.num_args());
                    let args = self.extract_procedure_args(&state, native_num_args);
                    let stats = &mut self.profiling.native_proc_stats;
                    let disposition = dispatch_native_proc(
                        native_proc.as_ref(),
                        &mut state,
                        &name,
                        no_return,
                        args,
                        // Unlike the parallel mirror, a strict-page-access fault
                        // still bounces to Python here (angr-ph300.73 tracks
                        // unifying the two).
                        false,
                        &mut NativeProcCounters {
                            native_calls: &mut stats.native_calls,
                            python_fallbacks: &mut stats.python_fallbacks,
                            call_counts: &mut stats.call_counts,
                            symbolic_fallbacks_by_name: &mut stats.symbolic_fallbacks_by_name,
                            not_implemented_fallbacks_by_name: &mut stats
                                .not_implemented_fallbacks_by_name,
                            other_fallbacks_by_name: &mut stats.other_fallbacks_by_name,
                        },
                    );
                    match disposition {
                        NativeProcDisposition::Returned { no_return, ret_val } => {
                            // For no-return procedures (exit/abort), skip the
                            // return-address dance and deadend directly. Setting
                            // PC to a stack-derived return address can produce a
                            // spurious successor (e.g. when exit is called from
                            // rejected() in fauxware, the post-call address
                            // happens to overlap main's start, causing infinite
                            // re-entry).
                            if no_return {
                                self.push_or_drop_terminal(STASH_DEADENDED, state);
                                return Ok(StepOutcome::Routed);
                            }

                            // Set return value if present
                            if let Some(rv) = ret_val {
                                let ret_reg = self.environment.calling_convention.return_register();
                                state.set_register_by_offset(ret_reg, rv);
                            }

                            // Get return address and set PC. Use the state's real
                            // register file so that LR/X30/$ra overrides see
                            // actual values; passing a blank RegisterFile here
                            // used to make ARM/ARM64/MIPS read LR=0 and set PC to
                            // 0.
                            let ctx = state.solver().borrow();
                            let ret_addr_opt = self.environment.calling_convention.get_return_addr(
                                state.registers(),
                                None,
                                &ctx,
                            );
                            let pops_return_addr =
                                self.environment.calling_convention.pops_return_addr();
                            drop(ctx);
                            if let Some(ret_addr) = ret_addr_opt {
                                // Only adjust SP for stack-based ABIs
                                // (x86/AMD64). ARM/ARM64/MIPS keep ret addr in a
                                // register and leave SP untouched.
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
                                // stack-based ABIs (only useful when the calling
                                // convention's get_return_addr declined to read
                                // memory itself).
                                if let Some(sp) = state.get_sp().as_u64()
                                    && let Ok(ret_bv) = state.memory_load(sp, state.arch().bytes())
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
                        NativeProcDisposition::SubCall {
                            proc_name,
                            saved_args,
                            target,
                            sub_args,
                            resume_tag,
                        } => {
                            // The proc requested a guest sub-call. Capture the
                            // caller return address from [sp] BEFORE
                            // `setup_native_subcall` overwrites that slot with the
                            // resume sentinel. A symbolic SP / unmapped slot
                            // (None) or a setup failure falls back to Python (the
                            // state is left untouched by setup on Err). Do NOT pop
                            // SP or honour `no_return` here: the guest's own `ret`
                            // advances SP, and `handle_native_resume` finishes
                            // without re-adjusting it.
                            let setup = match self.get_return_addr(&state) {
                                Some(caller_return_addr) => self
                                    .setup_native_subcall(
                                        &mut state,
                                        NativeSubcall {
                                            proc_name,
                                            saved_args,
                                            caller_return_addr,
                                            target,
                                            sub_args,
                                            resume_tag,
                                        },
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
                                        "native sub-call setup failed ({reason}); \
                                         falling back to Python for {name}"
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
                        NativeProcDisposition::Fallback => {}
                        NativeProcDisposition::Segfault(_) => {
                            unreachable!("mirror_segfault=false never yields Segfault")
                        }
                    }
                } // if prefer_native

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
                    ForkBundle::empty(),
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
                    let profiling_enabled = self.profiling.profiling_enabled;
                    let lazy_solves = self.constraint_solver.lazy_solves;
                    let materialized = super::fork_materialize::materialize_deferred_forks(
                        pending.deferred_forks,
                        super::fork_materialize::MaterializeForkCtx {
                            fork_base: &fork_base,
                            stored_conditions: &pending.stored_conditions,
                            snapshots: &mut snapshots,
                            lazy_solves,
                            // The taken-path guard lands on the find/avoid state
                            // itself (mirrors the BlockEnd handling).
                            guard_sink: Some(&pending.state),
                            stats: profiling_enabled
                                .then_some(&mut self.profiling.accumulated_stats),
                        },
                    );
                    for forked in &materialized.unsat {
                        // Lineage is registered for UNSAT forks too, then the
                        // state is dropped (this path has never had a pruned
                        // stash push).
                        self.sm.set_root(forked.state_id(), root_state_id);
                    }
                    for forked in materialized.sat {
                        self.sm.set_root(forked.state_id(), root_state_id);
                        self.push_to_active_or_drop(forked);
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
                            log::debug!("State at find address 0x{addr:x} is UNSAT, pruning");
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
                Ok(StepOutcome::Terminal(TerminalStep::Deadended(state)))
            }
            Err(StepError::Error(state, message)) => {
                let pc = state.pc();
                let state_id = state.state_id();
                Ok(StepOutcome::Terminal(TerminalStep::Errored {
                    state,
                    pc,
                    message,
                    state_id,
                }))
            }
            Err(StepError::Unconstrained(state, forks)) => {
                Ok(StepOutcome::Terminal(TerminalStep::Unconstrained {
                    state,
                    forks,
                }))
            }
        }
    }

    /// Apply a terminal disposition to the stashes, reproducing each original
    /// per-stash push path (and its side effects) byte-for-byte. Called by the
    /// driver, which then runs post-step bookkeeping.
    pub(crate) fn apply_terminal(&mut self, disposition: TerminalStep) {
        match disposition {
            TerminalStep::Deadended(state) => {
                self.push_or_drop_terminal(STASH_DEADENDED, state);
            }
            TerminalStep::Errored {
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
            TerminalStep::Unconstrained { state, forks } => {
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
                SimProcCall {
                    addr: *addr,
                    name: name.clone(),
                    num_args: *num_args,
                    return_addr: *return_addr,
                },
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
                             0x{addr:x} (state {state_id}); falling back to Python VEX engine"
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

#[cfg(all(test, feature = "vex-engine-z3"))]
#[path = "run_loop_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable"
)]
mod tests;
