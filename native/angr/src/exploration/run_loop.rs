//! Main exploration run loop: the `run_loop` dispatcher plus the pieces every
//! driver shares.
//!
//! Split into four sibling modules (angr-9ke6b.49), one per concern:
//! [`run_loop_single`] (the always-compiled
//! single-threaded driver), [`run_loop_wave`] and
//! [`run_loop_steady`] (the two Z3-gated parallel
//! coordinators), and [`run_loop_worker`] (the GIL-free
//! worker body both coordinators dispatch). What stays here is what more than
//! one of them needs: the `run_loop` entry point and its `must_run_serial` /
//! `steady_state_eligible` routing predicates, the [`StepOutcome`] /
//! [`TerminalStep`] vocabulary, `bounce_target_addr`, and
//! `flush_parked_bounces_to_active`.
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
//! `self.run_loop(py, n)`; the body is extracted here as `pub(crate)` methods
//! on `RustExplorationManager`, mirroring the `helpers.rs` / `stepping.rs`
//! extension-impl pattern used elsewhere in `exploration/`. That split
//! predates PyO3's `multiple-pymethods` feature, which is now enabled
//! (angr-9ke6b.50, see `invariant-pyo3-multiple-pymethods-enabled`) — it is
//! kept as a style choice, not a constraint.
//!
//! **Invariant I8 (cross-mixin termination, mirror of the
//! `I8. Exploration-loop termination conditions` entry in the
//! `angr/exploration/rust_manager.py` module docstring):** the run loop
//! must terminate on EITHER (a)
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
//! section of [`scheduler`] for the full argument — the same
//! reasoning covers every `.expect` in this file and in the four sibling
//! modules, so there is no fallible site to propagate and no Python-exception
//! path to build under this profile.
//!
//! **Enforcement (qwyti.15):** this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` so any *new* fallible
//! unwrap must be justified (each sibling module re-states the same deny). The
//! ~35 pre-existing `.expect()` sites are all the
//! poison / session-live / pool-set invariant guards described above; each
//! function that holds them carries a narrow
//! `#[allow(clippy::expect_used, reason = ...)]` pointing back at this Panic
//! policy, and the `mod tests;` decl is exempted (the deny propagates into
//! `#[path]` test submodules — see bd `invariant-clippy-deny-propagates-to-test-submodule`).
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

use super::core_outcome::BounceKind;

/// The find/avoid-checkable target address a materialized bounce carries.
///
/// Mirrors single-threaded `step_one`'s NeedCallback special case, which only
/// inspects `CallbackReason::SimProcedure { addr, .. }`. A `Hook` bounce and a
/// `SimProcedurePython` bounce are the two `BounceKind`s `dispatch_bounce` turns
/// into a `SimProcedure` callback reason, so only those are eligible for the
/// find/avoid short-circuit (Bug C1); every other bounce kind returns `None` and
/// always falls through to a real Python bounce, exactly as single-threaded does.
pub(crate) fn bounce_target_addr(kind: &BounceKind) -> Option<u64> {
    match kind {
        BounceKind::Hook { addr } | BounceKind::SimProcedurePython { addr, .. } => Some(*addr),
        _ => None,
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
    /// `errors.push((pc, message, state_id))` then `push_errored(state)`
    /// (errored states are never dropped — hence their own chokepoint rather
    /// than `push_or_drop_terminal`).
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
                return self.run_loop_parallel_wave(py, n);
            }
        }
        #[cfg(not(feature = "vex-engine-z3"))]
        let _ = py;
        self.run_loop_single_threaded(n)
    }

    /// No-Z3 stub for the steady-session config guard the `set_*` /
    /// `register_*` pymethods call unconditionally. The real implementation
    /// lives on `run_loop_steady.rs`, but that whole module is Z3-gated, so a
    /// `#[cfg(not(...))]` twin inside it is unreachable — it has to live here,
    /// in an ungated module (angr-9ke6b.236).
    #[cfg(not(feature = "vex-engine-z3"))]
    pub(crate) fn steady_config_guard(&mut self) {}
}

/// Dispatcher predicates for the parallel/steady coordinator paths, which only
/// exist with the Z3-backed engine (the scheduler transports
/// `StateMigrationPayload`).
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

    /// Retire a persistent worker pool whose size no longer matches what the
    /// next parallel run would ask for, so the lazy
    /// `if self.parallel_pool.is_none()` creation in `run_loop_parallel_wave` /
    /// `ensure_steady_session` rebuilds it at the new count.
    ///
    /// Without this, `set_parallel_workers(N)` was silently inert once a pool
    /// existed (angr-5mnx3.14): both creation sites are guarded on `is_none()`
    /// and `PersistentPool` has no resize path, so the first parallel run's
    /// worker count was pinned for the manager's whole life. That is reachable
    /// in practice — `rust_manager.py::_engage_parallel_workers` calls the
    /// setter on *every* `explore()` to downgrade/restore parallelism across
    /// eligible vs ineligible explores (angr-ph300.77).
    ///
    /// A request of `n <= 1` deliberately KEEPS the pool: `must_run_serial`
    /// routes that count to the single-threaded loop, so the parked threads are
    /// idle rather than wrong, and tearing them down would make the
    /// downgrade/restore cycle above pay a full thread-spawn + Z3-context
    /// rebuild on every ineligible explore.
    ///
    /// Called from `set_parallel_workers` *after* the `#[steady_guarded]`
    /// injection, so any live session has already been finalized; the workers
    /// are therefore parked on their ctl channel and the join is prompt. The
    /// GIL is released across it anyway, for the same reason
    /// `RustExplorationManager::drop` does: a worker that is somehow still
    /// mid-dispatch blocks in `Python::attach`, which would deadlock a
    /// GIL-holding join.
    pub(crate) fn retire_parallel_pool_for_resize(&mut self) {
        if self.parallel_real_workers <= 1 {
            return;
        }
        let wanted = self.parallel_real_workers.max(2);
        if self
            .parallel_pool
            .as_ref()
            .is_none_or(|p| p.num_workers() == wanted)
        {
            return;
        }
        if self.parallel_session.is_some() {
            // SILENT(cat-c): the `#[steady_guarded]` guard finalizes the session
            // before this runs, so a survivor means the drain timed out (it
            // warns separately). Dropping the pool would join workers that are
            // still dispatching — deadlock instead of a wrong worker count — so
            // keep the stale pool and say so.
            log::warn!(
                "set_parallel_workers({wanted}) could not resize the worker pool: a steady session survived the config guard, so the existing pool keeps its worker count"
            );
            return;
        }
        let Some(pool) = self.parallel_pool.take() else {
            return;
        };
        // A pymethod always holds the GIL but need not have a token in hand
        // (same reasoning as `steady_config_guard`), so reacquire to release.
        Python::attach(|py| py.detach(move || drop(pool)));
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
}

impl RustExplorationManager {
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
    /// dropped. Ids already resident in a stash are DISCARDED (not re-parked,
    /// regardless of kind) so the flush can never double-insert a state and
    /// never accumulates a duplicate no later flush could drain.
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
            if resident.contains(&id) {
                // SILENT(cat-b): a stash already holds this id, so the parked
                // copy is a redundant duplicate — re-entering it would
                // double-insert the state. Dropping is not a loss of work: the
                // resident copy carries the path forward. Re-parking it instead
                // (what the old catch-all arm did) leaked the duplicate
                // forever, since every later flush re-ran this same check and
                // re-parked it again, under a "no re-enterable entry address"
                // message that named the wrong reason (angr-03vl4.18).
                log::debug!(
                    "dropping parked bounce for state {id} (kind={kind:?}): the id \
                     is already resident in a stash, so the parked copy is a \
                     duplicate the flush must not re-insert"
                );
                continue;
            }
            match bounce_target_addr(&kind) {
                Some(addr) => {
                    state.set_pc(addr);
                    self.sm.set_root(id, root);
                    self.route_successor(state, true);
                }
                None => {
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

    /// How many parked bounces a
    /// [`flush_parked_bounces_to_active`](Self::flush_parked_bounces_to_active) would
    /// materialize into `STASH_ACTIVE` right now (angr-03vl4.15).
    ///
    /// The read-only census accessors (`active_count`, `stash_counts`) take
    /// `&self`, so unlike `dump_snapshot_bytes` / `finalize_parallel_session`
    /// they cannot flush before reading — yet a steady/wave session can return
    /// to Python with bounces still parked, and those states live in NO stash.
    /// Without this addend a mid-explore `len(mgr)` or progress callback
    /// silently under-reports the frontier. Mirrors the flush's own two filters
    /// so the count equals the post-flush stash population: an id already
    /// resident in a stash is a duplicate the flush discards, and a kind with no
    /// re-enterable entry address ([`bounce_target_addr`]) stays parked forever
    /// and never reaches a stash — counting either would make the census jump
    /// *down* across a flush that lost nothing.
    ///
    /// Attributed wholly to the ACTIVE stash, which is where the flush's
    /// `route_successor` sends a bounce whose target is neither a find nor an
    /// avoid address — the normal case, since the target is a hook /
    /// SimProcedure entry. When a hook address IS also a find address the flush
    /// would land it in FOUND instead, so the per-stash attribution can be off
    /// by that state while the total stays exact. That asymmetry is deliberate:
    /// `found_count` gates control flow (`>= num_find` in all three run loops
    /// and in `push_found_capped`), and a parked bounce is a live frontier state
    /// whose callback has not run yet — not a collected result — so inflating
    /// the found count could end an explore before the state is ever collected.
    ///
    /// Free on the serial path: the queue is empty there, and the early return
    /// skips building the resident-id set (this runs per `run()` return and per
    /// profiling sample, so an unconditional O(frontier) scan would be a real
    /// cost).
    pub(crate) fn parked_bounces_flushable_count(&self) -> usize {
        if self.pending_parallel_bounces.is_empty() {
            return 0;
        }
        let resident: std::collections::HashSet<u64> = self
            .sm
            .stashes()
            .values()
            .flat_map(|states| states.iter().map(|s| s.state_id()))
            .collect();
        self.pending_parallel_bounces
            .iter()
            .filter(|(state, kind, _)| {
                !resident.contains(&state.state_id()) && bounce_target_addr(kind).is_some()
            })
            .count()
    }

    /// The states parked in `pending_parallel_bounces` on their own — for the
    /// bounce-queue-specific reads (`flush_parked_bounces_to_active`'s
    /// residency filter, the test helpers) that want this bucket and not the
    /// other two. A manager-level BROADCAST wants
    /// [`all_live_states`](Self::all_live_states) instead (angr-0jh0j.82).
    ///
    /// `pending_parallel_bounces` is the third bucket of live `RustSimState`s —
    /// alongside the stashes and `pending_callbacks` — and like the second it
    /// lives in NO stash, so a broadcast that only loops `sm.stashes_mut()`
    /// misses it. Nothing re-applies the change when
    /// [`flush_parked_bounces_to_active`](Self::flush_parked_bounces_to_active)
    /// later routes the state back to
    /// `STASH_ACTIVE`, so it would resume on the old configuration forever.
    /// `#[angr_macros::steady_guarded]` does not cover this: the guard only
    /// drains the parallel session's *resident* frontier back into stashes.
    ///
    /// Yields EVERY parked entry, including the ones
    /// [`parked_bounces_flushable_count`](Self::parked_bounces_flushable_count)
    /// excludes. That asymmetry is
    /// deliberate: the census must equal the post-flush stash population, while
    /// a broadcast must reach every state that can still execute. An
    /// unflushable kind stays live in this manager, and a resident duplicate is
    /// dropped only at flush time — until then it is a real state a later step
    /// could observe.
    pub(crate) fn parked_bounce_states(&self) -> impl Iterator<Item = &RustSimState> {
        parked_states(&self.pending_parallel_bounces)
    }

    /// The two buckets of live `RustSimState`s that live in NO stash, chained:
    /// [`pending_states_mut`] (the parked callback's live continuation plus the
    /// pre-branch snapshot its deferred forks are materialized from,
    /// angr-sqfj8.32) and [`parked_states_mut`] (the parked parallel bounce
    /// queue).
    ///
    /// For a broadcast that deliberately covers only *some* stashes and so
    /// cannot use [`all_live_states_mut`](Self::all_live_states_mut) —
    /// `_active_states_map_memory` maps into `STASH_ACTIVE` only, because a
    /// deadended/errored state will never fault on the new region. Same
    /// `fork_snapshots` caveat as
    /// [`all_live_states`](Self::all_live_states).
    pub(crate) fn non_stash_live_states_mut(&mut self) -> impl Iterator<Item = &mut RustSimState> {
        let Self {
            pending_callbacks,
            pending_parallel_bounces,
            ..
        } = self;
        non_stash_states_mut(pending_callbacks, pending_parallel_bounces)
    }

    /// EVERY `RustSimState` this manager can still execute: all three buckets —
    /// all stashes, [`pending_states`], [`parked_states`] — in one iterator.
    ///
    /// This is the helper a manager-wide *read-only* walk should reach for
    /// (`analyze_constraint_sharing` folds each state's solver into a sharing
    /// census; `set_deterministic` reaches through each state to its shared
    /// solver). Hand-rolling the `.chain().chain()` instead is how the
    /// undercount bugs of angr-sqfj8.32 / angr-03vl4.10 / angr-0jh0j.15 each
    /// reappeared in a new call site: the three bucket lines differ only by
    /// method name, so a copy that drops one still compiles (angr-0jh0j.82).
    ///
    /// Walks ALL stashes, not just `STASH_ACTIVE`, so a census built on it is
    /// deterministic across exploration outcomes (no `find`/`avoid` bias).
    ///
    /// Does NOT cover `pending.fork_snapshots` — those carry a raw
    /// `SymContext`/solver and memory sidecar, not a `RustSimState`. A caller
    /// that must reach them (`set_deterministic`,
    /// `_active_states_map_memory`) chains its own `pending_callbacks` loop on
    /// top; see [`pending_states`].
    pub(crate) fn all_live_states(&self) -> impl Iterator<Item = &RustSimState> {
        self.sm
            .stashes()
            .values()
            .flat_map(|stash| stash.iter())
            .chain(pending_states(&self.pending_callbacks))
            .chain(parked_states(&self.pending_parallel_bounces))
    }

    /// `&mut` half of [`all_live_states`](Self::all_live_states), for a
    /// broadcast that mutates each state in place (`set_max_history`) rather
    /// than reaching through it to a shared solver. Same coverage and same
    /// `fork_snapshots` caveat.
    pub(crate) fn all_live_states_mut(&mut self) -> impl Iterator<Item = &mut RustSimState> {
        let Self {
            sm,
            pending_callbacks,
            pending_parallel_bounces,
            ..
        } = self;
        sm.stashes_mut()
            .values_mut()
            .flat_map(|stash| stash.iter_mut())
            .chain(non_stash_states_mut(
                pending_callbacks,
                pending_parallel_bounces,
            ))
    }
}

/// Bucket-2 body, as a free function over the field rather than a `&self`
/// method, so [`RustExplorationManager::all_live_states_mut`] can reach it
/// while separately holding a `&mut` borrow of `sm` (two `&mut self` method
/// calls could not coexist). The `&self`/`&mut self` accessors above delegate
/// here so the "live continuation plus pre-branch snapshot" definition of the
/// bucket lives in exactly one place per mutability.
pub(super) fn pending_states(
    pending_callbacks: &FxHashMap<StateId, PendingCallback>,
) -> impl Iterator<Item = &RustSimState> {
    pending_callbacks.values().flat_map(|pending| {
        std::iter::once(&pending.state).chain(pending.pre_callback_snapshot.as_ref())
    })
}

/// `&mut` half of [`pending_states`].
pub(super) fn pending_states_mut(
    pending_callbacks: &mut FxHashMap<StateId, PendingCallback>,
) -> impl Iterator<Item = &mut RustSimState> {
    pending_callbacks.values_mut().flat_map(|pending| {
        std::iter::once(&mut pending.state).chain(pending.pre_callback_snapshot.as_mut())
    })
}

/// Bucket-3 body; free-function counterpart of [`pending_states`], same
/// borrow-splitting reason.
pub(super) fn parked_states(
    pending_parallel_bounces: &[(RustSimState, BounceKind, u64)],
) -> impl Iterator<Item = &RustSimState> {
    pending_parallel_bounces.iter().map(|(state, _, _)| state)
}

/// `&mut` half of [`parked_states`].
pub(super) fn parked_states_mut(
    pending_parallel_bounces: &mut [(RustSimState, BounceKind, u64)],
) -> impl Iterator<Item = &mut RustSimState> {
    pending_parallel_bounces
        .iter_mut()
        .map(|(state, _, _)| state)
}

/// Buckets 2 and 3 chained, shared by
/// [`RustExplorationManager::non_stash_live_states_mut`] and
/// [`RustExplorationManager::all_live_states_mut`] so the pair of `.chain()`
/// lines this bead exists to de-duplicate is written once.
pub(super) fn non_stash_states_mut<'a>(
    pending_callbacks: &'a mut FxHashMap<StateId, PendingCallback>,
    pending_parallel_bounces: &'a mut [(RustSimState, BounceKind, u64)],
) -> impl Iterator<Item = &'a mut RustSimState> {
    pending_states_mut(pending_callbacks).chain(parked_states_mut(pending_parallel_bounces))
}

test_submod!(z3 "run_loop_tests.rs" => tests);
