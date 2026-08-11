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
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212, angr-c7xno.37):**
//! every surviving `.expect` here is a `Mutex::lock()` poison guard on one of
//! the wave transport's own mutexes (`job.results`, `job.summaries`), sitting
//! under a reviewed `#[allow(clippy::expect_used, reason = "...")]` that names
//! the guard. Each is covered verbatim by the "Mutex poisoning cannot happen"
//! bullet of the parent [`scheduler`](super) Panic policy — `panic = "abort"`
//! means no thread can unwind out of a live guard to flag the lock. No literal
//! count is stated on purpose: one of the sites is `#[cfg(test)]`-gated, so a
//! naive grep disagrees with any number written here (see bd memory
//! `avoid-enumerating-expect-sites-in-module-docs`). The parent's
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` already reaches this
//! file (it is a `#[path]` child module); the deny is restated below so the
//! guarantee is visible to anyone reading this file on its own.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use std::time::Instant;

/// The per-worker loop: dispatch a live local state (or steal+reattach one),
/// process it, push live successors locally, shed surplus on imbalance, and
/// collect terminals. Reads all per-wave state from `job`; `ctx` is the worker's
/// persistent Z3 context, `block_cache` its warm per-worker IRSB cache, and
/// `local` its persistent live frontier — all owned by [`worker_thread`](super::pool::worker_thread) and
/// reused across waves (angr-nkoct increment 1).
///
/// `local` is empty at entry in every reachable case — a wave either runs to
/// quiescence or drains its frontier on cancel (Bug M1, angr-op0dn.13.8) — so the
/// carry-over accounting below is a defensive `fetch_add(0)`. It is kept so that
/// any future retention path stays balanced: a carried-over state's terminal
/// `pending.fetch_sub(1)` must be matched by an add here.
// The terminal-drain loop holds `job.results` across all pushes by design (see
// the block comment); clippy resolves `significant_drop_tightening` at the
// item, so the allow lives here.
#[allow(clippy::significant_drop_tightening)]
#[allow(
    clippy::expect_used,
    reason = "`job.results` and (under `#[cfg(test)]`) `job.summaries` poison guards: poison requires a thread to unwind out of a live `MutexGuard`, which `panic = \"abort\"` forecloses — see the parent scheduler module Panic policy"
)]
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
        // Wave step-budget backstop (angr-9ke6b.52). The coordinator can only
        // test `dispatched_total >= max_steps` BETWEEN waves, and a wave runs to
        // quiescence — so on a frontier that never terminates, `run(n)` used to
        // execute unboundedly many steps. Spending the budget trips the same
        // `cancel` a `num_find` hit does, so the un-dispatched frontier takes the
        // residual-drain path below and stays resumable.
        if t.dispatch_budget_exhausted() {
            t.cancel.cancel_for_budget();
        }
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

        let step_start = Instant::now();
        let outcome = (job.process)(state, &t.cancel, block_cache);
        t.counters
            .step_ns
            .fetch_add(step_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Post-find speculative-waste accounting (angr-1ilq.8); see the twin in
        // `worker_session_loop`. The step's products are drained back on the next
        // iteration's cancel check, but the step itself was still speculative.
        // A budget stop (angr-9ke6b.52) is NOT speculative waste — the step was
        // charged to `run(n)` and its products are kept — so it is excluded via
        // `preempts_in_flight`.
        if t.cancel.preempts_in_flight() && !outcome.request_cancel {
            t.counters.post_cancel_steps.fetch_add(1, Ordering::SeqCst);
        }

        // Continue-states stay LIVE and LOCAL — no serde on the fast path.
        absorb_continues(t, local, outcome.continue_states);

        // Materialized terminals must cross the join, so they are detached here
        // (correct context). This is part of the honest steal fraction.
        // `guard` is held across the whole terminal-drain loop by design:
        // relocking per iteration would thrash the mutex. The
        // `significant_drop_tightening` allow sits on the enclosing fn — clippy
        // resolves this lint's level at the item, not the block.
        if !outcome.terminal_states.is_empty() {
            let mut guard = job.results.lock().expect("results mutex poisoned");
            for terminal in outcome.terminal_states {
                guard.push(detach_timed(terminal, &t.counters));
                t.counters
                    .materialized_terminals
                    .fetch_add(1, Ordering::SeqCst);
            }
        }

        // Summaries pay no serde — record and drop the full states in-context.
        // Production reads only the counters (`record_summaries`); the vec-copy
        // into `job.summaries` exists solely for the `#[cfg(test)]`
        // `WaveJob::into_results` reader, so it is gated out of the production
        // wave path (no per-batch mutex lock) — angr-1yge9.8 item 7.
        if !outcome.terminal_summaries.is_empty() {
            t.counters.record_summaries(&outcome.terminal_summaries);
            #[cfg(test)]
            {
                let mut guard = job.summaries.lock().expect("summaries mutex poisoned");
                guard.extend(outcome.terminal_summaries);
            }
        }

        // Shed surplus to the injector if a sibling is starving or we are over
        // the high-water mark. This is the only continue-path serde site.
        offload_surplus(local, &t.injector, &t.idle_workers, &t.counters, &t.policy);

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
///   `Quiesced` and return — the caller ([`worker_thread`](super::pool::worker_thread)) parks on its
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
            session.notify_up(WorkerUp::Paused { worker_id });
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
                    session.notify_up(WorkerUp::Paused { worker_id });
                } else {
                    // Global quiescence: park empty-handed until a wake ping.
                    session.notify_up(WorkerUp::Quiesced { worker_id });
                }
                return;
            }
        };

        let step_start = Instant::now();
        let outcome = (session.process)(state, &t.cancel, block_cache);
        t.counters
            .step_ns
            .fetch_add(step_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Post-find speculative-waste accounting (angr-1ilq.8): the top-of-loop
        // guard means we only reach here with `cancel` unset at dispatch time,
        // so a cancel visible NOW that this step did not itself raise was
        // requested by a peer/coordinator while this step was in flight — the
        // work is speculative (its products are drained back on the next
        // iteration's cancel check).
        //
        // Uses `preempts_in_flight`, not `is_cancelled`, for the same reason as
        // the twin in `worker_loop`: a budget stop charges the step to `run(n)`
        // and keeps its products, so it is not waste. Today that is a no-op
        // here — `set_max_dispatches` lives on `WaveJob` only, so a session
        // transport keeps `max_dispatches == None`, nothing calls
        // `cancel_for_budget`, and a session `CancelToken` never reaches
        // `BUDGET_CANCELLED`. Keeping the two predicates identical means adding
        // budget cancellation to the steady path later cannot silently
        // misclassify charged steps as speculative (angr-sqfj8.47).
        if t.cancel.preempts_in_flight() && !outcome.request_cancel {
            t.counters.post_cancel_steps.fetch_add(1, Ordering::SeqCst);
        }

        absorb_continues(t, local, outcome.continue_states);

        // Stream materialized terminals up as they happen — the coordinator
        // routes them (found/unconstrained/bounce) while other states are still
        // being stepped. Detached HERE (correct context), exactly like the wave
        // loop's results push.
        for terminal in outcome.terminal_states {
            let payload = detach_timed(terminal, &t.counters);
            t.counters
                .materialized_terminals
                .fetch_add(1, Ordering::SeqCst);
            session.notify_up(WorkerUp::Terminal { payload });
        }

        // Dead paths stay cheap summaries, counted and dropped in-context (the
        // same deadended-content caveat as wave mode).
        t.counters.record_summaries(&outcome.terminal_summaries);

        offload_surplus(local, &t.injector, &t.idle_workers, &t.counters, &t.policy);

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
///
/// Like `offload_surplus`, this detaches states from `local` WITHOUT going
/// through `policy.select` — `local.drain(..)` bypasses it entirely. There is
/// no hot/cold end to respect here the way `offload_one` has to: a residual
/// drain takes the whole deque, so `select_for_offload` has nothing to choose
/// between. What it shares with the offload path is the eviction hook, and the
/// coordinator routes the drained states back to `STASH_ACTIVE`, from where a
/// steady session's `seed_steady_session_from_active` -> `inject_seeds` puts
/// them back on the shared injector. A stolen-back state is reattached
/// directly by `dispatch_next`'s steal branch, never re-entering through
/// `on_fork`/`select` — the identical bypass `offload_surplus` has, just on a
/// more routine path (fires on essentially every steady-session pause/finalize,
/// `finalize_steady_session` in `run_loop_steady.rs`). So this drain must call
/// `policy.on_state_removed` per state too, or a memoizing policy's per-state
/// side table leaks the same way (angr-ua7fd).
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
        t.policy.on_state_removed(state.state_id());
        t.counters.residual_drains.fetch_add(1, Ordering::SeqCst);
        sink(detach_timed(state, &t.counters));
    }
    t.pending.fetch_sub(n, Ordering::SeqCst);
}

/// Wave-mode residual drain: push the un-dispatched local frontier into the
/// wave's shared `results` vec, where the post-barrier coordinator picks it up
/// alongside the materialized terminals.
#[allow(
    clippy::expect_used,
    reason = "`job.results` poison guard: poison requires a thread to unwind out of a live `MutexGuard`, which `panic = \"abort\"` forecloses — see the parent scheduler module Panic policy"
)]
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
        session.notify_up(WorkerUp::Terminal { payload });
    });
}

/// Pull the next state to process: the live local queue first (whichever end
/// `policy.select` names — a local state is hot in cache and the Z3 context;
/// zero serde), else
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
        // `Fifo` (the constructor default, see `selection_policy`'s module doc)
        // takes the front and `Lifo` reproduces the pre-seam `pop_back`; a
        // find-aware policy can reorder without touching this skeleton.
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
                        // thread's thread-local. Charged to the serde budget: it is
                        // the other half of the migration `detach_timed` opened.
                        let reattach_start = Instant::now();
                        let reattached = payload.reattach(ctx);
                        t.counters.serde_ns.fetch_add(
                            reattach_start.elapsed().as_nanos() as u64,
                            Ordering::Relaxed,
                        );
                        match reattached {
                            Ok(state) => {
                                // Observability only (angr-vh834 Phase 1): count every
                                // injector-steal reattach. No routing change.
                                t.counters.reattaches.fetch_add(1, Ordering::SeqCst);
                                t.record_dispatch(worker_id);
                                return Some(state);
                            }
                            Err(err) => {
                                // The `ContextMismatch` half is unreachable in practice
                                // (the worker set its own ctx as thread-local), but
                                // `reattach` also propagates whatever
                                // `RustSimState::from_serialized` returns, so this arm
                                // is a real failure surface. Drop the task loudly and
                                // never silently keep a phantom outstanding — the
                                // `fetch_sub` is what makes quiescence reachable.
                                // Covered by
                                // `test_dispatch_next_drops_a_corrupt_payload_and_keeps_stealing`
                                // and `test_dispatch_next_corrupt_only_payload_reaches_quiescence`.
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
///
/// **Frontier cap (angr-9ke6b.48).** When the transport carries a
/// `max_active_states` limit, forks beyond it are pruned HERE rather than
/// queued. The serial loop enforces the same limit in
/// `RustExplorationManager::push_to_active_or_drop` against
/// `sm.active_count()`, but a parallel frontier lives in the workers' local
/// queues + the injector for the whole wave/session and only reaches
/// `STASH_ACTIVE` at a wave boundary — so without this the cap was a no-op the
/// moment `RUST_PARALLEL_WORKERS > 1` (or steady mode) engaged, and the OOM
/// safety valve the option exists for never tripped.
///
/// `pending` (queued + in-flight) is the parallel analogue of `active_count()`,
/// minus one for the parent task still counted in it — the parent is counted
/// OUT by the caller immediately after, exactly as the serial loop's
/// currently-stepping state is already popped out of `STASH_ACTIVE`. With `W`
/// workers the bound is soft by up to `W - 1` (each peer's in-flight parent is
/// still counted), which is the point: it is a runaway-growth backstop, not an
/// exact quota.
///
/// Pruned forks become `Pruned` [`TerminalSummary`] counter entries and are
/// dropped in-context — the same treatment every other worker-side dead path
/// gets. They are NOT recoverable in `STASH_PRUNED` the way the serial path's
/// are; materializing them would pay serde for states the cap exists to
/// discard.
pub(super) fn absorb_continues(
    t: &WorkTransport,
    local: &mut VecDeque<RustSimState>,
    mut continue_states: Vec<RustSimState>,
) {
    if let Some(limit) = t.max_active_states {
        let live = t.pending.load(Ordering::SeqCst).saturating_sub(1);
        let budget = limit.saturating_sub(live);
        if budget < continue_states.len() {
            let pruned: Vec<TerminalSummary> = continue_states[budget..]
                .iter()
                .map(|s| TerminalSummary::of(s, TerminalDisposition::Pruned))
                .collect();
            continue_states.truncate(budget);
            t.counters.record_summaries(&pruned);
        }
    }

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
///   (`idle_workers > 0`) and we hold a backlog (`len >= 2`), offload ONE state
///   per starving sibling (never below one local state) so each starving sibling
///   has work to steal. This mirrors the `record_migration_sample` imbalance
///   model (helpers.rs), which likewise counts ONE steal per dispatch event, so
///   the measured steal fraction lines up with the gate's `f_model`.
///
///   It used to shed HALF the backlog per production event, which is what made
///   CADET_00001_partial collapse ~19x at `W=2` (angr-faorh): a chronically
///   imbalanced divergent frontier round-tripped a large fraction of the growing
///   frontier through full Z3 detach/reattach serde, and serde swamped compute.
///   One-per-idle-sibling is the smallest offload that still un-starves everyone
///   — a woken sibling immediately forks its own children into its own local
///   queue and becomes self-sufficient, so the extra states the halving shed were
///   pure serde cost (angr-8shhe).
/// * **B — high-water cap (safety):** if the local queue exceeds [`LOCAL_HWM`],
///   shed down to `HWM/2` regardless of idle siblings, bounding per-worker
///   memory. Rarely fires on a narrow frontier.
///
/// Offload is a relocation of already-counted tasks; it does not touch
/// `pending`.
///
/// Every `select_for_offload` here detaches a state from the worker's local deque
/// through a path other than `policy.select` — so `policy.on_state_removed`
/// is called alongside it. That is the eviction hook a memoizing policy
/// (`LoopHeadRoundRobin::key_cache`) needs: a state migrated through the
/// injector is later reattached directly by `dispatch_next`'s steal branch,
/// never re-entering `active`/`local` through `on_fork`, so `select` never
/// gets a chance to revisit and evict it itself (angr-ua7fd).
pub(super) fn offload_surplus(
    local: &mut VecDeque<RustSimState>,
    injector: &Injector<StateMigrationPayload>,
    idle_workers: &AtomicUsize,
    counters: &SchedulerCounters,
    policy: &Arc<dyn SelectionPolicy>,
) {
    // Trigger A: idle-gated load sharing. Offload the COLDEST states (whichever
    // end `policy.select_for_offload` names), keeping our hot end; at most one
    // state per starving sibling, so the volume
    // of Z3 serde is bounded by the idle count and not by the frontier width.
    let idle = idle_workers.load(Ordering::SeqCst);
    if idle > 0 && local.len() >= 2 && offload_is_affordable(counters) {
        let to_offload = idle.min(local.len() - 1);
        for _ in 0..to_offload {
            if local.len() <= 1 {
                break;
            }
            if !offload_one(local, injector, policy, counters) {
                break;
            }
        }
    }

    // Trigger B: hard memory cap. NOT budget-gated — it bounds per-worker memory,
    // so it must fire even when serde has blown its budget.
    if local.len() > LOCAL_HWM {
        while local.len() > LOCAL_HWM / 2 {
            if !offload_one(local, injector, policy, counters) {
                break;
            }
        }
    }
}

/// Detach the coldest state from `local` — the end `policy.select_for_offload`
/// nominates, NOT a hardcoded `pop_front` — and push it onto the shared
/// injector, running the `on_state_removed` eviction hook and bumping the
/// `surplus_offloaded` counter. Returns `false` when `local` is empty (nothing
/// to offload).
///
/// Which end is cold is a property of the *policy*, and the constructor default
/// is `Fifo` (front-dispatching), not `Lifo`: hardcoding `pop_front` here made
/// every default-policy offload ship away exactly the state `select` was about
/// to hand back for free, paying a full Z3 detach/reattach for it
/// (angr-03vl4.19).
///
/// This is the single body shared by both `offload_surplus` triggers: keeping
/// the eviction hook + counter bump in one place is what prevents the
/// per-site-fix-missed-in-a-sibling divergence class (angr-04tw3.7, same shape
/// as angr-myzjx.25). Any future invariant added to the offload path lands here
/// once and both triggers inherit it.
fn offload_one(
    local: &mut VecDeque<RustSimState>,
    injector: &Injector<StateMigrationPayload>,
    policy: &Arc<dyn SelectionPolicy>,
    counters: &SchedulerCounters,
) -> bool {
    if let Some(state) = policy.select_for_offload(local) {
        policy.on_state_removed(state.state_id());
        injector.push(detach_timed(state, counters));
        counters.surplus_offloaded.fetch_add(1, Ordering::SeqCst);
        true
    } else {
        false
    }
}

/// Serde budget: migration serde may consume at most `1 / SERDE_BUDGET_DIVISOR`
/// of the useful stepping time all workers have logged so far.
///
/// Both halves of a migration (`detach_for_migration` + `reattach`) are a full Z3
/// AST round-trip through SMT-LIB2, whose cost scales with the state's constraint
/// set — while a step's cost does not. So there is no fixed offload rate that is
/// right for every workload: on a solve-heavy frontier (`fork_solve_trap`) a
/// migration is cheap next to the solve it unblocks, and on a fat-constraint
/// divergent frontier (`CADET_00001_partial`, angr-faorh) it costs an order of
/// magnitude MORE than the step it hands off. Measuring the two and shutting
/// Trigger A off once serde stops paying for itself is what makes one policy fit
/// both, and it is self-tuning — no per-bench constant to hand-fit (angr-8shhe).
const SERDE_BUDGET_DIVISOR: u64 = 4;

/// Whether a Trigger A offload is still worth its Z3 serde cost — i.e. whether
/// accumulated serde is still within [`SERDE_BUDGET_DIVISOR`] of accumulated step
/// time. Both counters start at 0, so the first migrations are always allowed:
/// the budget can only be judged once there is something to measure.
///
/// This is deliberately a POOL-WIDE, monotone check, not a per-state estimate. A
/// worker cannot know what a migration will cost before paying for it, and once a
/// frontier is established as migration-dominated it stays that way — so the
/// cheap, sticky answer is the right one. Trigger B (the memory cap) ignores it.
fn offload_is_affordable(counters: &SchedulerCounters) -> bool {
    let serde = counters.serde_ns.load(Ordering::Relaxed);
    let step = counters.step_ns.load(Ordering::Relaxed);
    serde.saturating_mul(SERDE_BUDGET_DIVISOR) <= step
}

/// `detach_for_migration`, with the Z3-serde cost charged to the pool's serde
/// budget. Every continue-path detach goes through here so the budget sees the
/// whole tax (terminal detaches are unavoidable — they must cross the join — so
/// they are charged but never gated).
pub(super) fn detach_timed(
    state: RustSimState,
    counters: &SchedulerCounters,
) -> StateMigrationPayload {
    let t0 = Instant::now();
    let payload = state.detach_for_migration();
    counters
        .serde_ns
        .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    payload
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

test_submod!("scheduler_worker_tests.rs" => tests);
