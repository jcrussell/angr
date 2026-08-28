//! The per-wave parallel coordinator (angr-vh834 Phase 5).
//!
//! Each wave drains the entire active stash into migration seeds, runs the
//! work-stealing pool to quiescence with the GIL released, then applies every
//! deferred mutation coordinator-side. The worker body it dispatches lives in
//! [`run_loop_worker`]; the terminal-routing helpers
//! defined here (`route_materialized_terminal`,
//! `process_parallel_bounce_queue`) are shared with the steady-state
//! coordinator in [`run_loop_steady`].
//!
//! Split out of `run_loop.rs` (angr-9ke6b.49) — no behavior change.
//!
//! **Panic policy / lint enforcement:** identical to [`run_loop`]
//! — the crate ships with `panic = "abort"`, so a `MutexGuard` can never be
//! poisoned by an unwind, and every `.expect()` here is a poison /
//! session-live / pool-set invariant guard carrying a narrow
//! `#[allow(clippy::expect_used, reason = ...)]`. This module re-states the
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` so a *new* fallible
//! unwrap must still be justified.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

use std::sync::atomic::Ordering;

use super::core_outcome::{BounceKind, CoreCtx, ParallelProfiling, PendingBounce};
use super::scheduler::{PersistentPool, ProcessFn, WaveJob};
use crate::state::StateMigrationPayload;

use super::run_loop::{TerminalStep, bounce_target_addr};
use super::run_loop_worker::{MatKind, ParallelShared, parallel_process_state};

/// The parallel/steady-state coordinator half of the run loop. Every method
/// here transports work through the work-stealing `scheduler` (which carries
/// `StateMigrationPayload`), so the whole block is gated on the Z3-backed
/// engine — the `vex-engine`-without-`vex-engine-z3` build has only the
/// single-threaded loop below.
#[cfg(feature = "vex-engine-z3")]
impl RustExplorationManager {
    /// Parallel coordinator path (angr-vh834 Phase 5): a real work-stealing wave
    /// loop. Each wave drains the entire active stash into migration seeds, runs
    /// the `PersistentPool` (GIL released) to quiescence — workers keep
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
    /// the parallel-favourable case. A `run(n)` budget smaller than the worker
    /// count falls back the same way (angr-9ke6b.221): see the residual-drain
    /// comment on the `max_steps < workers` check below.
    ///
    /// **`num_find` early-exit preserves the active frontier (Bug M1, fixed in
    /// angr-op0dn.13.8).** When a wave reaches `num_find`, a worker trips the
    /// shared [`CancelToken`](crate::exploration::scheduler::CancelToken) and every
    /// worker stops at its next *task boundary*.
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
    pub(crate) fn run_loop_parallel_wave(
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

        // A step budget too small to hand every worker one dispatch cannot pay
        // for a wave (angr-9ke6b.221). `set_max_dispatches` below caps the wave
        // at the remaining budget, so such a wave dispatches a handful of
        // states, trips its `CancelToken`, and then detaches+reattaches the
        // ENTIRE resident frontier through serde to leave it resumable in
        // `STASH_ACTIVE`. Python's `RustExplorationManager.run(n=N)` step /
        // `step_func` mode maps to N native `run(1)` calls, so that is ~one
        // full-frontier drain per useful dispatch, every step.
        //
        // Measured on the 8-leaf pbounce synthetic
        // (`tests/engines/rust/test_parallel_wave.py`), `mgr.run(n=4096)`,
        // same found/steps at every worker count: serial 0.19s / 0 drains vs
        // workers=2 2.92s / 178 drains and workers=4 2.60s / 169 drains — a
        // ~15x step-mode penalty for zero parallelism. The single-threaded loop
        // honors the same budget exactly and leaves the frontier in
        // `STASH_ACTIVE` by construction, so route there instead. It also
        // flushes `pending_parallel_bounces` on entry, so a queue a prior
        // large-budget wave parked is not stranded by the switch.
        if max_steps < workers as u64 {
            return self.run_loop_single_threaded(n);
        }

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
                    self.profiling.accumulated_stats.run_loop_time_ns += crate::elapsed_ns(start);
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
                    self.profiling.accumulated_stats.run_loop_time_ns += crate::elapsed_ns(start);
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
    /// the scheduler. It owns a fresh
    /// [`StepContext`](crate::exploration::step_core::StepContext) snapshot and `Arc`-shares
    /// `prof` / `shared` / the two native registries, so it is `'static` and
    /// `Fn + Sync`; its callbacks self-acquire the GIL via `Python::attach`.
    ///
    /// `callbacks` must already be cloned by the caller WHILE HOLDING THE GIL
    /// (`Py<T>::clone` needs it, and this must never run on a worker). The wave
    /// loop rebuilds one per wave, `ensure_steady_session` one per session
    /// (angr-ph300.12 — previously duplicated verbatim in both).
    pub(crate) fn build_parallel_process(
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
    pub(crate) fn fold_parallel_shared_counters(&mut self, shared: &ParallelShared) -> u64 {
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
    pub(crate) fn route_materialized_payloads(
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
    pub(crate) fn route_materialized_terminal(
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
                self.push_or_drop_terminal(STASH_UNCONSTRAINED, state);
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
                        if state.survives_sat_prune(self.constraint_solver.lazy_solves) {
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
                    && state.survives_sat_prune(self.constraint_solver.lazy_solves)
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
    pub(crate) fn process_parallel_bounce_queue(
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
}

test_submod!(z3 "run_loop_wave_tests.rs" => tests);
