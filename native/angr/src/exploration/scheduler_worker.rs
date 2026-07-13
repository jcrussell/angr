//! Per-worker dispatch / steal / offload loop for the parallel scheduler
//! (extracted from `scheduler.rs`, angr-nbim4.2).
//!
//! These are the free functions the worker threads run: the wave loop
//! ([`worker_loop`]) and the steady-state session loop ([`worker_session_loop`]),
//! plus their shared dispatch/steal/offload helpers. They construct and read the
//! parent module's per-wave transport types (`WaveJob`, `RunSession`,
//! `WorkTransport`, `SchedulerCounters`) as a descendant module, so all of the
//! parent's private items are in scope via the glob below — no visibility edits
//! on those structs are needed. Only the two loop entry points and the shared
//! `dispatch_next` are re-exported into the parent (`pub(super)`); see the parent
//! module for the transport invariant and panic-policy rationale.

use super::*;

/// The per-worker loop: dispatch a live local state (or steal+reattach one),
/// process it, push live successors locally, shed surplus on imbalance, and
/// collect terminals. Reads all per-wave state from `job`; `ctx` is the worker's
/// persistent Z3 context, `block_cache` its warm per-worker IRSB cache, and
/// `local` its persistent live frontier — all owned by [`worker_thread`] and
/// reused across waves (angr-nkoct increment 1).
///
/// `local` is empty at entry in every reachable case — a wave either runs to
/// quiescence or drains its frontier on cancel (Bug M1, angr-op0dn.13.8) — so the
/// carry-over accounting below is a defensive `fetch_add(0)`. It is kept so that
/// any future retention path stays balanced: a carried-over state's terminal
/// `pending.fetch_sub(1)` must be matched by an add here.
pub(super) fn worker_loop(
    worker_id: usize,
    job: &WaveJob,
    ctx: &Context,
    block_cache: &mut LruCache<u64, Arc<IRSB>>,
    local: &mut VecDeque<RustSimState>,
) {
    let t = &job.transport;

    // Account for any frontier retained across the wave barrier so `pending`
    // (the quiescence detector) covers every state that will be dispatched and
    // `fetch_sub`'d below. No-op when `local` is empty (today's invariant).
    if !local.is_empty() {
        t.pending.fetch_add(local.len(), Ordering::SeqCst);
    }

    loop {
        if t.cancel.is_cancelled() {
            // Bug M1 fix (angr-op0dn.13.8): on cancel (e.g. the run loop hit
            // `num_find`) the worker stops HERE, at a task boundary, and DRAINS
            // its un-dispatched local frontier into `results` as untagged
            // payloads — the coordinator routes those back to `STASH_ACTIVE`, so
            // the post-`num_find` active stash matches the single-threaded loop's
            // and the un-explored frontier stays resumable. The injector surplus
            // is drained by the coordinator after the barrier
            // (`WaveJob::drain_residual_payloads`). This pays serde for states the
            // caller may discard; that is the price of resumability, and it is
            // bounded by the residual frontier (never the explored set).
            drain_local_into_results(job, local);
            return;
        }

        let state = match dispatch_next(worker_id, t, local, ctx) {
            Some(state) => state,
            // No task available and nothing outstanding anywhere (or
            // cancelled): no future task can ever appear this wave.
            None => return,
        };

        let outcome = (job.process)(state, &t.cancel, block_cache);

        // Post-find speculative-waste accounting (angr-1ilq.8); see the twin in
        // `worker_session_loop`. The step's products are drained back on the next
        // iteration's cancel check, but the step itself was still speculative.
        if t.cancel.is_cancelled() && !outcome.request_cancel {
            t.counters.post_cancel_steps.fetch_add(1, Ordering::SeqCst);
        }

        // Continue-states stay LIVE and LOCAL — no serde on the fast path.
        absorb_continues(t, local, outcome.continue_states);

        // Materialized terminals must cross the join, so they are detached here
        // (correct context). This is part of the honest steal fraction.
        if !outcome.terminal_states.is_empty() {
            let mut guard = job.results.lock().expect("results mutex poisoned");
            for terminal in outcome.terminal_states {
                guard.push(terminal.detach_for_migration());
                t.counters
                    .materialized_terminals
                    .fetch_add(1, Ordering::SeqCst);
            }
        }

        // Summaries pay no serde — record and drop the full states in-context.
        if !outcome.terminal_summaries.is_empty() {
            t.counters.record_summaries(&outcome.terminal_summaries);
            let mut guard = job.summaries.lock().expect("summaries mutex poisoned");
            guard.extend(outcome.terminal_summaries);
        }

        // Shed surplus to the injector if a sibling is starving or we are over
        // the high-water mark. This is the only continue-path serde site.
        offload_surplus(local, &t.injector, &t.idle_workers, &t.counters);

        t.pending.fetch_sub(1, Ordering::SeqCst);

        if outcome.request_cancel {
            t.cancel.cancel();
            // Loop back to the cancel check, which drains this worker's residual
            // frontier into `results` before returning (the twin of
            // `worker_session_loop`). Returning straight from here would drop the
            // finder's OWN backlog — the very states Bug M1 is about, and the
            // ones most likely to be live (it just forked them).
        }
    }
}

/// The steady-state per-worker loop (angr-nkoct): same dispatch skeleton as
/// [`worker_loop`], but terminals are STREAMED up the session's mpsc channel as
/// they materialize (no barrier, no shared results vec), and running out of
/// work does not end the session:
///
/// * **Quiescence** (empty local + empty injector + `pending == 0`): send
///   `Quiesced` and return — the caller ([`worker_thread`]) parks on its
///   control channel, and a `WorkerCtl::Run` wake ping re-enters this loop
///   against the same session after the coordinator injects more work.
/// * **Cancel** (the session's finalize/num_find/budget signal): DRAIN the
///   residual local frontier upstream ([`drain_local_upstream`] — the steady
///   path's fix for wave-mode Bug M1), ack `Paused`, and return. Re-entry via a
///   stale wake ping on a cancelled session just re-acks `Paused` (empty
///   drain); the coordinator dedupes acks by `worker_id`.
///
/// All upstream sends ignore errors: the coordinator may have dropped the
/// receiver after finalize (session teardown), and a worker must park cleanly
/// through that race rather than panic.
///
/// The loop returns with `local` empty in EVERY case, so no `!Send` state is
/// parked inside the worker across a coordinator boundary — the resident
/// frontier lives only in BUSY workers.
pub(super) fn worker_session_loop(
    session: &RunSession,
    worker_id: usize,
    ctx: &Context,
    block_cache: &mut LruCache<u64, Arc<IRSB>>,
    local: &mut VecDeque<RustSimState>,
) {
    let t = &session.transport;

    // Defensive: absorb any frontier a prior wave left on `local` into this
    // session's quiescence accounting, so mode-mixing on one pool is safe. Both
    // loops now leave `local` empty on every exit (quiesced or cancel-drained),
    // so this is a no-op in practice.
    if !local.is_empty() {
        t.pending.fetch_add(local.len(), Ordering::SeqCst);
    }

    loop {
        if t.cancel.is_cancelled() {
            drain_local_upstream(session, local);
            let _ = session.up_tx.send(WorkerUp::Paused { worker_id });
            return;
        }

        let state = match dispatch_next(worker_id, t, local, ctx) {
            Some(state) => state,
            None => {
                if t.cancel.is_cancelled() {
                    // Cancel landed while spinning in the injector: drain (the
                    // local queue is already empty here, but keep the single
                    // drain path) and ack.
                    drain_local_upstream(session, local);
                    let _ = session.up_tx.send(WorkerUp::Paused { worker_id });
                } else {
                    // Global quiescence: park empty-handed until a wake ping.
                    let _ = session.up_tx.send(WorkerUp::Quiesced { worker_id });
                }
                return;
            }
        };

        let outcome = (session.process)(state, &t.cancel, block_cache);

        // Post-find speculative-waste accounting (angr-1ilq.8): the top-of-loop
        // guard means we only reach here with `cancel` unset at dispatch time,
        // so a cancel visible NOW that this step did not itself raise was
        // requested by a peer/coordinator while this step was in flight — the
        // work is speculative (its products are drained back on the next
        // iteration's cancel check).
        if t.cancel.is_cancelled() && !outcome.request_cancel {
            t.counters.post_cancel_steps.fetch_add(1, Ordering::SeqCst);
        }

        absorb_continues(t, local, outcome.continue_states);

        // Stream materialized terminals up as they happen — the coordinator
        // routes them (found/unconstrained/bounce) while other states are still
        // being stepped. Detached HERE (correct context), exactly like the wave
        // loop's results push.
        for terminal in outcome.terminal_states {
            let payload = terminal.detach_for_migration();
            t.counters
                .materialized_terminals
                .fetch_add(1, Ordering::SeqCst);
            let _ = session.up_tx.send(WorkerUp::Terminal { payload });
        }

        // Dead paths stay cheap summaries, counted and dropped in-context (the
        // same deadended-content caveat as wave mode).
        t.counters.record_summaries(&outcome.terminal_summaries);

        offload_surplus(local, &t.injector, &t.idle_workers, &t.counters);

        t.pending.fetch_sub(1, Ordering::SeqCst);

        if outcome.request_cancel {
            t.cancel.cancel();
            // Loop back to the cancel check, which drains + acks Paused.
        }
    }
}

/// Detach every residual live state on `local` and hand it to `sink` as an
/// (untagged) migration payload — the coordinator routes untagged payloads back
/// to the active stash. Balances `pending` and counts `residual_drains` for the
/// states it removes. The worker-local half of the Bug M1 cancel-drain, shared
/// by both modes (DRY); the two callers differ only in the transport the payload
/// leaves on (session mpsc vs. the wave's `results` vec).
fn drain_local_with(
    t: &WorkTransport,
    local: &mut VecDeque<RustSimState>,
    mut sink: impl FnMut(StateMigrationPayload),
) {
    if local.is_empty() {
        return;
    }
    let n = local.len();
    for state in local.drain(..) {
        t.counters.residual_drains.fetch_add(1, Ordering::SeqCst);
        sink(state.detach_for_migration());
    }
    t.pending.fetch_sub(n, Ordering::SeqCst);
}

/// Wave-mode residual drain: push the un-dispatched local frontier into the
/// wave's shared `results` vec, where the post-barrier coordinator picks it up
/// alongside the materialized terminals.
fn drain_local_into_results(job: &WaveJob, local: &mut VecDeque<RustSimState>) {
    if local.is_empty() {
        return;
    }
    let mut guard = job.results.lock().expect("results mutex poisoned");
    drain_local_with(&job.transport, local, |payload| guard.push(payload));
}

/// Steady-mode residual drain: stream the un-dispatched local frontier upstream
/// as untagged `Terminal`s. Sends ignore errors (the coordinator may already
/// have dropped the receiver after finalize).
pub(super) fn drain_local_upstream(session: &RunSession, local: &mut VecDeque<RustSimState>) {
    drain_local_with(&session.transport, local, |payload| {
        let _ = session.up_tx.send(WorkerUp::Terminal { payload });
    });
}

/// Pull the next state to process: the live local queue first (LIFO — the
/// freshest child is hottest in cache and the Z3 context; zero serde), else
/// steal + reattach from the shared injector. Returns `None` when no task is
/// available and none can ever appear (quiescence) — or on cancellation.
/// Shared verbatim by [`worker_loop`] and [`worker_session_loop`] (DRY: the
/// dispatch half of the two modes is identical; only terminal transport and
/// end-of-work behavior differ).
pub(super) fn dispatch_next(
    worker_id: usize,
    t: &WorkTransport,
    local: &mut VecDeque<RustSimState>,
    ctx: &Context,
) -> Option<RustSimState> {
    loop {
        // Local frontier pop goes through the selection policy (angr-1ilq.9):
        // `Lifo` (the default) reproduces the pre-seam `pop_back`; a find-aware
        // policy can reorder without touching this skeleton.
        match t.policy.select(local) {
            Some(state) => {
                t.counters.local_dispatches.fetch_add(1, Ordering::SeqCst);
                t.record_dispatch(worker_id);
                return Some(state);
            }
            None => {
                match steal_from_injector(&t.injector, &t.pending, &t.cancel, &t.idle_workers) {
                    Some(payload) => {
                        t.counters
                            .injector_dispatches
                            .fetch_add(1, Ordering::SeqCst);
                        // Rebuild the state in THIS worker's context. The ptr-eq
                        // guard inside `reattach` holds because `ctx` is exactly this
                        // thread's thread-local.
                        match payload.reattach(ctx) {
                            Ok(state) => {
                                // Observability only (angr-vh834 Phase 1): count every
                                // injector-steal reattach. No routing change.
                                t.counters.reattaches.fetch_add(1, Ordering::SeqCst);
                                t.record_dispatch(worker_id);
                                return Some(state);
                            }
                            Err(err) => {
                                // Unreachable in practice (the worker set its own ctx
                                // as thread-local), but never silently keep a phantom
                                // task outstanding.
                                log::error!("scheduler reattach failed, dropping task: {err:?}");
                                t.pending.fetch_sub(1, Ordering::SeqCst);
                                continue;
                            }
                        }
                    }
                    None => return None,
                }
            }
        }
    }
}

/// Push live successors onto the worker's local queue and count them IN before
/// the caller counts the parent task OUT (`pending` must never dip to zero with
/// live descendants queued).
pub(super) fn absorb_continues(
    t: &WorkTransport,
    local: &mut VecDeque<RustSimState>,
    continue_states: Vec<RustSimState>,
) {
    let spawned = continue_states.len();
    for child in continue_states {
        // Fork insertion goes through the selection policy (angr-1ilq.9); both
        // built-ins append at the tail (`push_back`), so the default is
        // identical to the pre-seam open-coded push.
        t.policy.on_fork(local, child);
    }
    if spawned > 0 {
        t.pending.fetch_add(spawned, Ordering::SeqCst);
    }
}

/// Shed surplus live states from `local` to the shared `injector`, detaching
/// each into a `Send` payload. Two triggers:
///
/// * **A — idle-gated (primary):** if any sibling is currently starving
///   (`idle_workers > 0`) and we hold a backlog (`len >= 2`), offload up to half
///   the backlog (hysteresis, never below one local state) so the starving
///   sibling has work to steal. This mirrors the `record_migration_sample`
///   imbalance model (helpers.rs) so the measured steal fraction lines up with
///   the gate's `f_model`.
/// * **B — high-water cap (safety):** if the local queue exceeds [`LOCAL_HWM`],
///   shed down to `HWM/2` regardless of idle siblings, bounding per-worker
///   memory. Rarely fires on a narrow frontier.
///
/// Offload is a relocation of already-counted tasks; it does not touch
/// `pending`.
pub(super) fn offload_surplus(
    local: &mut VecDeque<RustSimState>,
    injector: &Injector<StateMigrationPayload>,
    idle_workers: &AtomicUsize,
    counters: &SchedulerCounters,
) {
    // Trigger A: idle-gated load sharing. Offload the COLDEST states (front),
    // keeping our hot tail; at most half per production event for hysteresis.
    if idle_workers.load(Ordering::SeqCst) > 0 && local.len() >= 2 {
        let to_offload = local.len() / 2;
        for _ in 0..to_offload {
            if local.len() <= 1 {
                break;
            }
            if let Some(state) = local.pop_front() {
                injector.push(state.detach_for_migration());
                counters.surplus_offloaded.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    // Trigger B: hard memory cap.
    if local.len() > LOCAL_HWM {
        while local.len() > LOCAL_HWM / 2 {
            if let Some(state) = local.pop_front() {
                injector.push(state.detach_for_migration());
                counters.surplus_offloaded.fetch_add(1, Ordering::SeqCst);
            } else {
                break;
            }
        }
    }
}

/// Pull one payload from the shared injector, blocking (cooperative yield) while
/// work may still appear. Returns `None` only when the injector is empty AND no
/// task is outstanding anywhere (a task is only created by an in-flight task, so
/// none can ever appear again) — or when cancellation is requested.
///
/// Maintains the `idle_workers` signal across the wait so producers can detect
/// starvation and offload (Trigger A).
pub(super) fn steal_from_injector(
    injector: &Injector<StateMigrationPayload>,
    pending: &AtomicUsize,
    cancel: &CancelToken,
    idle_workers: &AtomicUsize,
) -> Option<StateMigrationPayload> {
    idle_workers.fetch_add(1, Ordering::SeqCst);
    // Bounded backoff: spin (yield) for the first `SPIN_LIMIT` empty polls, then
    // fall back to a short capped sleep so an idle worker does not burn a full
    // core while the productive worker holds a heavy Z3 solve (faorh/8shhe:
    // steal_from_injector previously busy-spun yield_now, starving a
    // core-constrained box). `spins` counts only consecutive empty-with-pending
    // polls; a `Retry` (injector contention) means work is imminent, so we do
    // not back off there.
    const SPIN_LIMIT: u32 = 128;
    const MAX_SLEEP_US: u64 = 200;
    let mut spins: u32 = 0;
    let result = loop {
        if cancel.is_cancelled() {
            break None;
        }
        match injector.steal() {
            Steal::Success(task) => break Some(task),
            Steal::Retry => continue,
            Steal::Empty => {
                // Empty right now. If nothing is outstanding anywhere we are
                // done; otherwise another worker is mid-task and may yet offload
                // surplus — back off and retry.
                if pending.load(Ordering::SeqCst) == 0 {
                    break None;
                }
                if spins < SPIN_LIMIT {
                    spins += 1;
                    std::thread::yield_now();
                } else {
                    // Linearly ramp the sleep past the spin phase, capped so we
                    // stay responsive when the producer finally offloads.
                    let us = ((spins - SPIN_LIMIT + 1) as u64).min(MAX_SLEEP_US);
                    spins = spins.saturating_add(1);
                    std::thread::sleep(std::time::Duration::from_micros(us));
                }
            }
        }
    };
    idle_workers.fetch_sub(1, Ordering::SeqCst);
    result
}
