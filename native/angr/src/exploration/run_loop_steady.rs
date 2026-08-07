//! The steady-state parallel coordinator (angr-nkoct).
//!
//! One long-lived [`SteadySession`] spans many `run()` calls: workers keep their
//! frontiers RESIDENT across the Python-callback boundary, so a bounce costs one
//! materialize + re-inject instead of a full-frontier detach/reattach re-seed.
//! Engaged only when `steady_state_eligible()`; the per-wave loop in
//! [`run_loop_wave`](super::run_loop_wave) remains the fallback. Both share the
//! GIL-free worker body in [`run_loop_worker`](super::run_loop_worker) and the
//! wave loop's `route_materialized_terminal` / `process_parallel_bounce_queue`
//! routing helpers.
//!
//! Split out of `run_loop.rs` (angr-9ke6b.49) — no behavior change.
//!
//! **Panic policy / lint enforcement:** identical to [`run_loop`](super::run_loop)
//! — the crate ships with `panic = "abort"`, so a `MutexGuard` can never be
//! poisoned by an unwind, and every `.expect()` here is a poison /
//! session-live / pool-set invariant guard carrying a narrow
//! `#[allow(clippy::expect_used, reason = ...)]`. This module re-states the
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` so a *new* fallible
//! unwrap must still be justified.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

use std::sync::Mutex;

use super::core_outcome::{BounceKind, ParallelProfiling};
use super::scheduler::{PersistentPool, RunSession, WorkerUp};
use crate::state::StateMigrationPayload;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

use super::run_loop_worker::{MatKind, ParallelShared};

// NOTE (angr-nkoct steady-state Phase B): the duplex-protocol types this file
// used to scaffold (`WorkerCtl` / `WorkerUp` / `RunSession`, with their
// Send/Sync compile-time proof) now live in `scheduler_pool.rs`, implemented and
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

#[cfg(feature = "vex-engine-z3")]
impl RustExplorationManager {
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
            // Always-on, NOT `debug_assert!` (angr-9ke6b.220): a stranded
            // bounce is a silently lost state, and the release profile leaves
            // `debug-assertions = false`. Cost is a `Vec::is_empty` per pump
            // iteration.
            assert!(
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
        // `bounce_queue` may or may not grow here — a bounce routed to a
        // find/avoid address short-circuits inside the helper (no push); if it
        // landed in FOUND, the outer num_find check will finalize + cancel,
        // stopping the workers. So there is nothing to assert about its length.
        self.route_materialized_terminal(state, kind, root, bounce_queue);
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
    /// solver/memory config) or touch the active stash. Called from the guarded
    /// `set_*` / `register_*` pymethods. The no-Z3 build has no steady session
    /// at all; its no-op stub lives in `run_loop.rs`, because this whole module
    /// is Z3-gated (angr-9ke6b.236).
    pub(crate) fn steady_config_guard(&mut self) {
        if self.parallel_session.is_some() {
            // A pymethod may be called without a Python token in hand, but we
            // are always on the GIL thread here (pymethods hold the GIL), so
            // reacquire it to drain the session.
            let finalized = Python::attach(|py| self.finalize_steady_session(py));
            // SILENT(cat-c): the guard is injected by
            // `#[angr_macros::steady_guarded]` into ~58 pymethods, all but one
            // of which return `()`, so the only Err `finalize_steady_session`
            // can produce (the drain timeout, whose message names the stuck
            // workers and says their resident frontier states are lost) has no
            // `?` to ride out on. Losing states silently would make a later
            // `run()` look merely under-explored, so warn loudly with the full
            // error text instead of discarding it (angr-sqfj8.38). The public
            // `finalize_parallel_session` pymethod stays the propagating path.
            if let Err(e) = finalized {
                log::warn!(
                    "steady_config_guard: finalizing the live steady session for a config \
                     mutation failed; frontier states may have been lost and this run's \
                     results may be incomplete: {e}"
                );
            }
        }
    }
}

#[cfg(all(test, feature = "vex-engine-z3"))]
#[path = "run_loop_steady_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable"
)]
mod tests;
