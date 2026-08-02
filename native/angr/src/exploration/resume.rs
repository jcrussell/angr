//! Resume callbacks invoked by Python after a callback dispatch.
//!
//! When the Rust run loop pauses to call into Python (SimProcedure, syscall,
//! hook, error, symbolic branch, find/avoid predicate), the Python side
//! finishes the work and calls back into one of the `resume_*` /
//! `deadend_pending_callback` methods to apply state changes, process
//! deferred forks, and route the resulting states into the right stashes.
//!
//! The pyclass-facing thin wrappers live in `mod.rs` and forward to the
//! `pub(crate)` bodies in this module. PyO3 0.27.2 in this project does not
//! enable `multiple-pymethods`, so each pyclass is limited to a single
//! `#[pymethods]` impl block — see `invariant-pyo3-single-pymethods-impl`.
//!
//! This is a Python-boundary module; `unwrap`/`expect` are denied here so a
//! future panic-on-input landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]` (angr-qwyti.11 enforcement layer).
#![deny(clippy::unwrap_used, clippy::expect_used)]
//! This module mirrors the `helpers.rs` / `stepping.rs` / `run_loop.rs`
//! extension-impl pattern used elsewhere in `exploration/`.

use super::*;

impl RustExplorationManager {
    /// Inner body of the pymethods-exposed `resume_after_simprocedure`.
    /// See the wrapper in `mod.rs` for the public Python signature.
    pub(crate) fn _resume_after_simprocedure(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        new_pc: u64,
        register_changes: Option<Vec<(u32, u32, Vec<u8>)>>,
        memory_changes: Option<Vec<(u64, Vec<u8>)>>,
        new_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        crate::gil_profile::park_end();
        let pending = self
            .pending_callbacks
            .remove(&StateId::new(state_id))
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;

        // Apply changes
        let mut changes = StateChanges::new();
        changes.new_pc = Some(new_pc);

        if let Some(reg_changes) = register_changes {
            changes.register_writes = reg_changes;
        }

        if let Some(mem_changes) = memory_changes {
            changes.memory_writes = mem_changes;
        }

        let mut state = pending.state;
        state.apply_changes(&changes);
        state.set_pc(new_pc);

        // Sync constraints from Python back to Rust
        // This ensures constraints added by SimProcedures (e.g., strcmp return conditions)
        // are properly reflected in the Rust solver state
        //
        // P12: Track if state becomes UNSAT after constraint sync
        let mut main_state_unsat = false;
        if let Some(constraints) = new_constraints {
            let is_sat = self.sync_constraints_from_python(py, &state, constraints)?;
            if !is_sat {
                log::debug!(
                    "P12: State {} became UNSAT after constraint sync in resume_after_simprocedure.",
                    state.state_id()
                );
                main_state_unsat = true;
            }
        }

        // NOTE: deferred forks whose condition_id is absent from stored_conditions
        // are NOT dropped — the materialization loop below reconstructs the
        // condition from `fork.condition_ast` (P11) or, failing that, creates a
        // conservative unconstrained fork (P15). A diagnostic loop that warned
        // such forks "will be skipped" was removed (angr-cudgw.16): the claim was
        // false and the loop had no side effect.

        // Process deferred forks that were stored during the step
        // These represent unexplored branches that should be added to active
        //
        // CRITICAL: Deferred forks diverged BEFORE the callback, so they should
        // NOT inherit callback constraints. Use pre_callback_snapshot as fork base.
        // Only create fork_base when there are deferred forks — state.fork() costs ~3ms
        // due to Z3 solver clone, and most callbacks have zero deferred forks.
        //
        // ORDER MATTERS (angr-khpsh): the snapshot-less fallback `state.fork()`
        // must run BEFORE apply_deferred_fork_constraints below, otherwise the
        // fork base inherits the taken-path guard and every unexplored side is
        // trivially UNSAT — a reachable branch silently pruned. The deadend
        // handler avoids this by never applying those constraints up front.
        let has_deferred_forks = !pending.deferred_forks.is_empty();
        let fork_base = if has_deferred_forks {
            Some(
                pending
                    .pre_callback_snapshot
                    .unwrap_or_else(|| state.fork()),
            )
        } else {
            drop(pending.pre_callback_snapshot); // explicitly drop unused snapshot
            None
        };

        // Add taken-path constraints from deferred forks to the main state.
        // Without these, the solver doesn't know which branch was taken,
        // causing incorrect results for subsequent symbolic operations.
        apply_deferred_fork_constraints(
            &state,
            &pending.deferred_forks,
            &pending.stored_conditions,
        );

        // Track root state ID for lineage
        // The root is inherited from the original pending state
        let root_state_id = self.sm.root_or_self(state.state_id());

        // P12: Only add main state if SAT, otherwise add to pruned list
        let (mut successors, mut pruned_states) = if main_state_unsat {
            (Vec::new(), vec![state])
        } else {
            (vec![state], Vec::new())
        };
        let mut snapshots = pending.fork_snapshots;
        if !pending.deferred_forks.is_empty() {
            #[allow(
                clippy::expect_used,
                reason = "internal invariant: fork_base is set earlier in the step whenever deferred_forks is non-empty"
            )]
            let fb = fork_base
                .as_ref()
                .expect("fork_base set before deferred fork processing");
            // DO NOT sync callback constraints to the forked states! These paths
            // diverged before the callback occurred; adding callback constraints
            // would pollute the unexplored branches. The continuing state
            // already carries the taken-path guard, hence `guard_sink: None`.
            let materialized = super::fork_materialize::materialize_deferred_forks(
                pending.deferred_forks,
                super::fork_materialize::MaterializeForkCtx {
                    fork_base: fb,
                    stored_conditions: &pending.stored_conditions,
                    snapshots: &mut snapshots,
                    lazy_solves: self.constraint_solver.lazy_solves,
                    guard_sink: None,
                    stats: None,
                },
            );
            for forked in materialized.sat.iter().chain(materialized.unsat.iter()) {
                self.sm.set_root(forked.state_id(), root_state_id);
            }
            successors.extend(materialized.sat);
            pruned_states.extend(materialized.unsat);
        }

        // Add all successors (original state + forks) to stashes
        // P13: Check satisfiability for each before adding
        // Note: We split the loops to avoid double mutable borrow of self.sm
        let mut final_successors = Vec::new();
        for successor in successors {
            if self.constraint_solver.lazy_solves || successor.satisfiable() {
                final_successors.push(successor);
            } else {
                log::debug!(
                    "P13: Successor state {} is UNSAT, moving to pruned stash",
                    successor.state_id()
                );
                pruned_states.push(successor);
            }
        }

        // Add to active stash, checking find/avoid first. In steady-state mode
        // this re-injects active-bound successors straight into the live
        // session (they re-enter the resident frontier without a STASH_ACTIVE
        // round-trip) and counts them as resume_reinjects; otherwise it routes
        // to STASH_ACTIVE as before.
        self.route_resume_successors(final_successors);

        // Add to pruned stash
        for s in pruned_states {
            self.push_or_drop_terminal(STASH_PRUNED, s);
        }

        // Apply native uniqueness filter if enabled
        self.apply_uniqueness_filter();
        // Apply native techniques (LengthLimiter, Timeout, LoopBound)
        self.apply_native_techniques();

        Ok(())
    }

    /// Route successors produced by a Python resume callback. In steady-state
    /// mode (angr-nkoct) with a live session, active-bound successors are
    /// re-injected into the session's injector (+ `resume_reinjects`) and the
    /// workers woken, so a resumed state re-enters the resident frontier
    /// directly; find/avoid-bound successors still route through
    /// `route_successor`. Without a live session this is exactly
    /// `for s in successors { route_successor(s, false) }`.
    pub(crate) fn route_resume_successors(&mut self, successors: Vec<RustSimState>) {
        #[cfg(feature = "vex-engine-z3")]
        {
            if self.parallel_session.is_some() {
                // Stamp roots + detach active-bound successors (owned pass over
                // `&self.sm`); route find/avoid ones normally. The detached
                // batch is handed to the run-loop's session-injection helper
                // (which owns the private `SteadySession` internals).
                let mut inject: Vec<(u64, u64, crate::state::StateMigrationPayload)> = Vec::new();
                for s in successors {
                    let pc = s.pc();
                    if self.find_addrs.contains(&pc) || self.avoid_addrs.contains(&pc) {
                        self.route_successor(s, false);
                    } else {
                        let id = s.state_id();
                        let root = self.sm.root_or_self(id);
                        inject.push((id, root, s.detach_for_migration()));
                    }
                }
                self.steady_inject_resumed(inject);
                return;
            }
        }
        for s in successors {
            self.route_successor(s, false);
        }
    }

    /// Inner body of the pymethods-exposed `deadend_pending_callback`.
    /// Materialize a terminal-sink pending's deferred forks, returning the main
    /// state (unmoved) so the caller can push it into its terminal stash.
    ///
    /// This is the shared body for the two consumers whose main state is on its
    /// way to a *terminal* stash (`_deadend_pending_callback` → DEADENDED,
    /// `_resume_after_error` → ERRORED). Both must process deferred forks that
    /// diverged EARLIER in the same step BEFORE sinking the main state: those
    /// forks are unexplored branches with nothing to do with why the state is
    /// terminating, so dropping them silently prunes a reachable path (the
    /// deadend variant was angr-ph300.7, the error variant angr-4xaga.5).
    ///
    /// Because the main state sinks to a terminal stash, it is the `guard_sink`
    /// for each taken-path constraint, and UNSAT materialized forks are only
    /// lineage-registered (no pruned-stash push) — identical for both callers.
    /// Centralizing here means a future terminal consumer calls ONE method
    /// instead of re-deriving the fork_base / guard_sink / routing dance, which
    /// is exactly the step that was forgotten in angr-4xaga.5 (angr-qwyti.4).
    /// The contrasting non-terminal consumers (`_resume_after_simprocedure`,
    /// `_resume_after_symbolic_branch`) keep their own materialize blocks: their
    /// main state survives and carries the taken-path guard, so `guard_sink`
    /// and routing differ and must not share this path.
    fn materialize_terminal_deferred_forks(&mut self, pending: PendingCallback) -> RustSimState {
        let PendingCallback {
            state,
            pre_callback_snapshot,
            deferred_forks,
            stored_conditions,
            fork_snapshots,
            ..
        } = pending;

        if !deferred_forks.is_empty() {
            let fork_base = pre_callback_snapshot.unwrap_or_else(|| state.fork());
            let root_state_id = self.sm.root_or_self(state.state_id());

            let mut snapshots = fork_snapshots;
            let materialized = super::fork_materialize::materialize_deferred_forks(
                deferred_forks,
                super::fork_materialize::MaterializeForkCtx {
                    fork_base: &fork_base,
                    stored_conditions: &stored_conditions,
                    snapshots: &mut snapshots,
                    lazy_solves: self.constraint_solver.lazy_solves,
                    guard_sink: Some(&state),
                    stats: None,
                },
            );
            for forked in materialized.unsat {
                // Lineage registered, then dropped: terminal paths have no
                // pruned-stash push for materialized forks.
                self.sm.set_root(forked.state_id(), root_state_id);
            }
            for forked in materialized.sat {
                self.sm.set_root(forked.state_id(), root_state_id);
                self.route_successor(forked, false);
            }
        }

        state
    }

    pub(crate) fn _deadend_pending_callback(&mut self, state_id: u64) -> PyResult<()> {
        crate::gil_profile::park_end();
        let pending = self
            .pending_callbacks
            .remove(&StateId::new(state_id))
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state for deadend"))?;

        // Process deferred forks BEFORE deadending — these represent
        // unexplored branches that diverged before the exit/abort call. The P11
        // `condition_ast` reconstruction matters most here: without it an
        // AST-only deferred fork parked behind a no-return SimProcedure
        // (exit/abort) was silently dropped and its unexplored branch never
        // reached (angr-ph300.7).
        let state = self.materialize_terminal_deferred_forks(pending);
        self.push_or_drop_terminal(STASH_DEADENDED, state);
        Ok(())
    }

    /// Inner body of the pymethods-exposed `resume_after_error`.
    pub(crate) fn _resume_after_error(&mut self, state_id: u64, error_msg: &str) -> PyResult<()> {
        crate::gil_profile::park_end();
        let pending = self
            .pending_callbacks
            .remove(&StateId::new(state_id))
            .ok_or_else(|| {
                PyRuntimeError::new_err("no pending callback state for error handling")
            })?;

        let pc = pending.state.pc();
        let state_id = pending.state.state_id();

        log::warn!(
            "P17: Moving state {state_id} to errored stash after callback error at 0x{pc:x}: {error_msg}"
        );

        // Record the error
        self.errors.push((pc, error_msg.to_string(), state_id));

        // Process deferred forks BEFORE erroring — these represent unexplored
        // branches that diverged EARLIER in the same step, before the callback
        // that raised. They have nothing to do with why the Python handler
        // threw, so dropping them (as this path used to) silently prunes a
        // reachable branch: a find= target behind that untaken side becomes
        // permanently unreachable with no diagnostic beyond the generic
        // callback-error warning (angr-4xaga.5). Shared terminal-sink path with
        // _deadend_pending_callback — pending.state likewise sinks to a terminal
        // stash (STASH_ERRORED here vs STASH_DEADENDED there).
        let state = self.materialize_terminal_deferred_forks(pending);

        // Move to errored stash
        self.push_errored(state);

        Ok(())
    }

    /// Inner body of the pymethods-exposed `resume_after_symbolic_branch`.
    pub(crate) fn _resume_after_symbolic_branch(
        &mut self,
        _py: Python<'_>,
        state_id: u64,
        true_pc: u64,
        false_pc: u64,
        true_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
        false_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        crate::gil_profile::park_end();
        // true_constraints and false_constraints are accepted for API compatibility but
        // the branch condition is sourced from stored_conditions (set by interpreter).
        let _ = (true_constraints, false_constraints);
        let pending = self
            .pending_callbacks
            .remove(&StateId::new(state_id))
            .ok_or_else(|| PyRuntimeError::new_err("no pending symbolic branch callback"))?;

        // Get the branch condition from stored_conditions (set by interpreter)
        let branch_condition = match &pending.reason {
            CallbackReason::SymbolicBranch { condition_id, .. } => {
                pending.stored_conditions.get(condition_id).cloned()
            }
            _ => None,
        };

        // Track root state ID for lineage
        let root_state_id = self.sm.root_or_self(pending.state.state_id());

        // Deferred forks diverged BEFORE this symbolic branch, so their
        // unexplored side must NOT inherit the branch guard (assume_true /
        // assume_false applied to true_state / false_state below), nor the
        // deferred forks' own taken-path constraints (applied just below).
        // Mirror the simprocedure path (_resume_after_simprocedure): build a
        // clean fork base from the pre-callback snapshot (or a fork of the
        // pre-branch state) BEFORE either is applied. Without this, a
        // snapshot-less deferred fork gets built on a guard-polluted solver
        // and a reachable path is falsely pruned as UNSAT (angr-ph300.9 for
        // the branch guard, angr-khpsh for the taken-path constraints). Only
        // pay the ~3ms fork cost when there are deferred forks to materialize.
        let fork_base = if pending.deferred_forks.is_empty() {
            drop(pending.pre_callback_snapshot); // explicitly drop unused snapshot
            None
        } else {
            Some(
                pending
                    .pre_callback_snapshot
                    .unwrap_or_else(|| pending.state.fork()),
            )
        };

        // Add taken-path constraints from deferred forks to the main state
        // BEFORE forking for the symbolic branch. Since fork() creates an
        // independent solver copy, both true_state and false_state will
        // inherit these constraints. Without this, the solver wouldn't know
        // which deferred-fork branch was taken.
        apply_deferred_fork_constraints(
            &pending.state,
            &pending.deferred_forks,
            &pending.stored_conditions,
        );

        // Create the true state (fork of original) and add constraint
        let mut true_state = pending.state.fork();
        self.sm.set_root(true_state.state_id(), root_state_id);
        true_state.set_pc(true_pc);
        if let Some(ref cond) = branch_condition {
            // Add assume_true: guard is true → exit taken
            let solver_ref = true_state.solver();
            solver_ref.borrow().assume_true(cond);
        }

        // Create the false state (use original) and add constraint
        let mut false_state = pending.state;
        false_state.set_pc(false_pc);
        if let Some(ref cond) = branch_condition {
            // Add assume_false: guard is false → fallthrough
            let solver_ref = false_state.solver();
            solver_ref.borrow().assume_false(cond);
        }

        // Process deferred forks that were accumulated before this symbolic branch.
        // These represent unexplored branches from earlier in the step that must
        // not be silently dropped.
        let mut deferred_successors = Vec::new();
        let mut deferred_pruned = Vec::new();
        if !pending.deferred_forks.is_empty() {
            // fork_base is Some whenever deferred_forks is non-empty (set above);
            // it is the guard-free base so unexplored sides don't inherit the
            // branch condition assumed onto true_state / false_state.
            #[allow(
                clippy::expect_used,
                reason = "internal invariant: fork_base is Some whenever deferred_forks is non-empty (set earlier in the step)"
            )]
            let fb = fork_base
                .as_ref()
                .expect("fork_base set when deferred_forks non-empty");
            let mut snapshots = pending.fork_snapshots;
            // `guard_sink: None` — this handler mints both directions itself
            // (true_state / false_state above), so nothing here should receive
            // the taken-path guard.
            let materialized = super::fork_materialize::materialize_deferred_forks(
                pending.deferred_forks,
                super::fork_materialize::MaterializeForkCtx {
                    fork_base: fb,
                    stored_conditions: &pending.stored_conditions,
                    snapshots: &mut snapshots,
                    lazy_solves: self.constraint_solver.lazy_solves,
                    guard_sink: None,
                    stats: None,
                },
            );
            for forked in materialized.sat.iter().chain(materialized.unsat.iter()) {
                self.sm.set_root(forked.state_id(), root_state_id);
            }
            deferred_successors.extend(materialized.sat);
            deferred_pruned.extend(materialized.unsat);
        }

        // Add states to stashes. BOTH branch states need a real satisfiability
        // check: the interpreter's non-deferred symbolic-branch path
        // (`statements.rs`, `IRStmt::Exit` under `!use_deferred_forks`) returns
        // `SymbolicBranch` WITHOUT calling `check_branch_feasibility`, so
        // neither direction is known feasible when we get here — only the
        // guard's *symbolic-ness* was established. This code used to prime the
        // sat cache with `true` on the strength of a feasibility check that
        // never ran, so an UNSAT branch state sailed through every downstream
        // `satisfiable()` gate and landed in the found stash (angr-3ag1l).
        let mut pruned_states = Vec::new();

        for state in [true_state, false_state] {
            if self.constraint_solver.lazy_solves || state.satisfiable() {
                // Satisfiability established (and cached by `is_sat`), so the
                // found gate does not need to re-check it.
                self.route_successor(state, false);
            } else {
                pruned_states.push(state);
            }
        }

        // Deferred fork states were sat-checked above, when they were built.
        for s in deferred_successors {
            self.route_successor(s, false);
        }

        // Add to pruned stash
        pruned_states.extend(deferred_pruned);
        for s in pruned_states {
            self.push_or_drop_terminal(STASH_PRUNED, s);
        }

        log::debug!(
            "Resumed after symbolic branch: true_pc=0x{true_pc:x}, false_pc=0x{false_pc:x}"
        );

        // Apply native uniqueness filter if enabled
        self.apply_uniqueness_filter();
        // Apply native techniques (LengthLimiter, Timeout, LoopBound)
        self.apply_native_techniques();

        Ok(())
    }

    /// Inner body of the pymethods-exposed `resume_find_predicate`.
    pub(crate) fn _resume_find_predicate(&mut self, state_id: u64, matched: bool) -> PyResult<()> {
        let pending = self
            .pending_callbacks
            .remove(&StateId::new(state_id))
            .ok_or_else(|| PyRuntimeError::new_err("no pending find predicate callback"))?;

        // Forcing-function canary (angr-qwyti.4): find/avoid predicate pendings
        // are built via `PendingCallback::lightweight`, which hardcodes an empty
        // `deferred_forks`, so this consumer deliberately does NOT materialize.
        // If a future refactor ever routes a `with_context` pending (carrying
        // deferred forks) here, this trips loudly rather than silently pruning a
        // reachable branch — the exact silent-drop bug class of angr-4xaga.5.
        // Real `assert!` (not `debug_assert!`) because the ralph/CI gate runs
        // `cargo test --release`, where debug assertions are compiled out; this
        // path is a Python-boundary bounce, not a hot loop, so the cost is nil.
        assert!(
            pending.deferred_forks.is_empty(),
            "find-predicate pending carries {} deferred fork(s); lightweight callbacks must not (angr-qwyti.4)",
            pending.deferred_forks.len(),
        );

        if matched {
            log::debug!("Find predicate matched - moving state to found stash");
            let state_id = pending.state.state_id();
            self.sm
                .stashes_mut()
                .entry(STASH_FOUND.to_string())
                .or_default()
                .push_back(pending.state);
            self.sm.index(state_id, STASH_FOUND);
        } else {
            log::debug!("Find predicate did not match - continuing exploration");
            // Mark this state to skip the find predicate check on next pop,
            // preventing infinite loop (state was already checked at this PC).
            let state_id = pending.state.state_id();
            self.constraint_tracker
                .skip_find_predicate_states
                .insert(state_id);
            self.push_to_active_or_drop(pending.state);
        }

        Ok(())
    }

    /// Inner body of the pymethods-exposed `resume_avoid_predicate`.
    pub(crate) fn _resume_avoid_predicate(&mut self, state_id: u64, matched: bool) -> PyResult<()> {
        let pending = self
            .pending_callbacks
            .remove(&StateId::new(state_id))
            .ok_or_else(|| PyRuntimeError::new_err("no pending avoid predicate callback"))?;

        // Forcing-function canary (angr-qwyti.4): see _resume_find_predicate.
        // Avoid-predicate pendings are `lightweight` (empty deferred_forks) and
        // this consumer intentionally skips materialization; trip loudly if a
        // future refactor ever routes a fork-carrying pending here.
        assert!(
            pending.deferred_forks.is_empty(),
            "avoid-predicate pending carries {} deferred fork(s); lightweight callbacks must not (angr-qwyti.4)",
            pending.deferred_forks.len(),
        );

        if matched {
            log::debug!("Avoid predicate matched - moving state to avoid stash");
            self.push_or_drop_terminal(STASH_AVOID, pending.state);
        } else {
            log::debug!("Avoid predicate did not match - continuing exploration");
            let state_id = pending.state.state_id();
            self.constraint_tracker
                .skip_avoid_predicate_states
                .insert(state_id);
            self.push_to_active_or_drop(pending.state);
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "resume_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable"
)]
mod tests;
