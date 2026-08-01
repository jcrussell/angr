//! The GIL-free worker body shared by both parallel coordinators
//! ([`run_loop_wave`](super::run_loop_wave) and
//! [`run_loop_steady`](super::run_loop_steady)).
//!
//! `parallel_process_state` is the scheduler-thread analogue of
//! [`RustExplorationManager::step_one`](super::run_loop_single): it steps one
//! state without the GIL and without `&mut self`, classifying the result into a
//! [`TaskOutcome`]. [`ParallelShared`] is the per-wave / per-session side-channel
//! it stamps routing decisions into, and [`MatKind`] is the tag the coordinator
//! reads back off a migrated payload.
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
use std::sync::atomic::{AtomicUsize, Ordering};

use lru::LruCache;

use super::core_outcome::{
    BounceKind, CoreCtx, CoreReturn, PendingBounce, PostStepInputs, materialize_bounce_forks,
    run_post_step_core,
};
use super::scheduler::{
    CancelToken, TaskOutcome, TerminalDisposition as SchedDisposition, TerminalSummary,
};
use super::step_core::run_interpreter_step_core;
use crate::vex::IRSB;

use super::run_loop::bounce_target_addr;

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

/// Per-wave state shared (by `&`) across all scheduler workers. Every field is
/// interior-mutable (atomics / `Mutex`) so the `Fn + Sync` worker closure can
/// touch it without `&mut`.
#[cfg(feature = "vex-engine-z3")]
pub(crate) struct ParallelShared {
    /// `state_id -> lineage root`. Seeded with each drained active state's
    /// `root_or_self`; workers insert every successor / materialized terminal so
    /// the coordinator can replay `sm.set_root` and descendants inherit the root
    /// across the local re-dispatch + migration boundary.
    pub(crate) root_map: Mutex<FxHashMap<u64, u64>>,
    /// `state_id -> MatKind` for materialized terminals (found / unconstrained /
    /// bounce), so the coordinator routes each recovered payload correctly.
    pub(crate) kind_map: Mutex<FxHashMap<u64, MatKind>>,
    /// Per-step `CoreCounters` the coordinator folds into the manager after the
    /// wave (native-proc / syscall / simproc fallback tallies).
    pub(crate) counters: Mutex<Vec<super::core_outcome::CoreCounters>>,
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
    pub(crate) worker_found_hint: AtomicUsize,
    pub(crate) num_find: usize,
    /// angr-vh834 Phase 5 (M2): number of dispatches that actually took an
    /// interpreter step AND produced a `Successors`/`Terminal`-equivalent outcome
    /// (`run_post_step_core` returned `Continue`/`Deadended`/`Errored`/
    /// `Unconstrained`). EXCLUDES pre-step find/avoid routes and `NeedsPython`
    /// bounces — exactly the dispatches single-threaded `step_one` counts with
    /// `self.steps += 1` (Successors/Terminal only; Routed/NeedCallback skip it).
    /// The coordinator folds this into `self.steps`, NOT `stats.dispatches()`
    /// (which over-counts by including pre-step routes and bounces).
    pub(crate) stepped: AtomicUsize,
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
    pub(crate) fn seeded(found_count: usize, num_find: usize) -> Self {
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
pub(crate) fn parallel_process_state(
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
