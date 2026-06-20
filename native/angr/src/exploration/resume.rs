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
//! This module mirrors the `helpers.rs` / `stepping.rs` / `run_loop.rs`
//! extension-impl pattern used elsewhere in `exploration/`.

use super::*;

impl RustExplorationManager {
    /// Inner body of the pymethods-exposed `resume_after_simprocedure`.
    /// See the wrapper in `mod.rs` for the public Python signature.
    pub(crate) fn _resume_after_simprocedure(
        &mut self,
        py: Python<'_>,
        new_pc: u64,
        register_changes: Option<Vec<(u32, u32, Vec<u8>)>>,
        memory_changes: Option<Vec<(u64, Vec<u8>)>>,
        new_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        let pending = self
            .pending_callback
            .take()
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

        // Add taken-path constraints from deferred forks to the main state.
        // Without these, the solver doesn't know which branch was taken,
        // causing incorrect results for subsequent symbolic operations.
        for fork in &pending.deferred_forks {
            if let Some(cond) = pending.stored_conditions.get(&fork.condition_id) {
                if fork.path_taken {
                    state.solver().borrow().assume_true(cond);
                } else {
                    state.solver().borrow().assume_false(cond);
                }
            }
        }

        // Process deferred forks that were stored during the step
        // These represent unexplored branches that should be added to active
        //
        // CRITICAL: Deferred forks diverged BEFORE the callback, so they should
        // NOT inherit callback constraints. Use pre_callback_snapshot as fork base.
        // Only create fork_base when there are deferred forks — state.fork() costs ~3ms
        // due to Z3 solver clone, and most callbacks have zero deferred forks.
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

        // Track root state ID for lineage
        // The root is inherited from the original pending state
        let original_state_id = state.state_id();
        let root_state_id = self
            .sm
            .roots()
            .get(&original_state_id)
            .copied()
            .unwrap_or(original_state_id);

        // P12: Only add main state if SAT, otherwise add to pruned list
        let (mut successors, mut pruned_states) = if main_state_unsat {
            (Vec::new(), vec![state])
        } else {
            (vec![state], Vec::new())
        };
        let mut snapshots = pending.fork_snapshots;
        for fork in pending.deferred_forks {
            // Look up the condition for this deferred fork
            let condition = pending.stored_conditions.get(&fork.condition_id);

            // P11 fix: If condition not in stored_conditions, try to reconstruct from condition_ast
            let reconstructed_condition = if condition.is_none() {
                if let Some(ref py_ast) = fork.condition_ast {
                    // Try to convert the claripy AST to RustBV
                    Python::attach(|py| {
                        let ast = py_ast.bind(py);
                        let fb = fork_base
                            .as_ref()
                            .expect("fork_base set before deferred fork processing");
                        let solver_ref = fb.solver();
                        let ctx: &SymContext = &solver_ref.borrow();
                        claripy_to_rustbv(py, ast, ctx).ok()
                    })
                } else {
                    None
                }
            } else {
                None
            };

            let effective_condition = condition.or(reconstructed_condition.as_ref());

            if let Some(cond) = effective_condition {
                // Use solver snapshot (from before branch constraint) if available
                let fb = fork_base
                    .as_ref()
                    .expect("fork_base set before deferred fork processing");
                let forked = if let Some(snapshot) = snapshots.remove(&fork.condition_id) {
                    let mut f = fb.fork_from_snapshot(snapshot);
                    if fork.path_taken {
                        f.solver().borrow().assume_false(cond);
                    } else {
                        f.solver().borrow().assume_true(cond);
                    }
                    f.set_pc(fork.unexplored_target);
                    f
                } else if fork.path_taken {
                    let mut f = fb.fork_false(cond);
                    f.set_pc(fork.unexplored_target);
                    f
                } else {
                    let mut f = fb.fork_true(cond);
                    f.set_pc(fork.unexplored_target);
                    f
                };

                // Track root state ID for this forked state
                self.sm.set_root(forked.state_id(), root_state_id);

                // DO NOT sync callback constraints to forked state!
                // These paths diverged before the callback occurred.
                // Adding callback constraints would pollute unexplored branches.

                // P13: Check satisfiability before adding to successors
                if self.constraint_solver.lazy_solves || forked.satisfiable() {
                    successors.push(forked);
                    if reconstructed_condition.is_some() {
                        log::debug!(
                            "P11: Reconstructed condition from condition_ast for fork at 0x{:x}",
                            fork.branch_addr
                        );
                    }
                } else {
                    log::debug!(
                        "P13: Forked state at 0x{:x} is UNSAT, adding to pruned",
                        fork.unexplored_target
                    );
                    pruned_states.push(forked);
                }
            } else {
                // P15: Better handling of missing deferred fork conditions
                // Try harder to get a condition or create a fresh boolean to explore both paths
                log::warn!(
                    "P15: Missing condition for deferred fork at 0x{:x} (condition_id={}). \
                     Creating conservative fork to explore the path.",
                    fork.branch_addr,
                    fork.condition_id
                );
                // Create a fork without additional constraints - this is conservative
                // but ensures we don't lose valid paths
                let mut forked = fork_base
                    .as_ref()
                    .expect("fork_base set before fork")
                    .fork();
                forked.set_pc(fork.unexplored_target);
                self.sm.set_root(forked.state_id(), root_state_id);

                // P13: Still check satisfiability
                if self.constraint_solver.lazy_solves || forked.satisfiable() {
                    successors.push(forked);
                } else {
                    log::debug!(
                        "P13: Unconstrained fork at 0x{:x} is UNSAT, adding to pruned",
                        fork.unexplored_target
                    );
                    pruned_states.push(forked);
                }
            }
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

        // Add to active stash, checking find/avoid first
        for s in final_successors {
            self.route_successor(s, false);
        }

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

    /// Inner body of the pymethods-exposed `deadend_pending_callback`.
    pub(crate) fn _deadend_pending_callback(&mut self) -> PyResult<()> {
        let pending = self
            .pending_callback
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state for deadend"))?;

        // Process deferred forks BEFORE deadending — these represent
        // unexplored branches that diverged before the exit/abort call.
        if !pending.deferred_forks.is_empty() {
            let fork_base = pending
                .pre_callback_snapshot
                .unwrap_or_else(|| pending.state.fork());
            let original_state_id = pending.state.state_id();
            let root_state_id = self
                .sm
                .roots()
                .get(&original_state_id)
                .copied()
                .unwrap_or(original_state_id);

            let mut snapshots = pending.fork_snapshots;
            for fork in pending.deferred_forks {
                let condition = pending.stored_conditions.get(&fork.condition_id);
                if let Some(cond) = condition {
                    if fork.path_taken {
                        pending.state.solver().borrow().assume_true(cond);
                    } else {
                        pending.state.solver().borrow().assume_false(cond);
                    }
                    let forked = if let Some(snapshot) = snapshots.remove(&fork.condition_id) {
                        let mut f = fork_base.fork_from_snapshot(snapshot);
                        if fork.path_taken {
                            f.solver().borrow().assume_false(cond);
                        } else {
                            f.solver().borrow().assume_true(cond);
                        }
                        f.set_pc(fork.unexplored_target);
                        f
                    } else if fork.path_taken {
                        let mut f = fork_base.fork_false(cond);
                        f.set_pc(fork.unexplored_target);
                        f
                    } else {
                        let mut f = fork_base.fork_true(cond);
                        f.set_pc(fork.unexplored_target);
                        f
                    };
                    self.sm.set_root(forked.state_id(), root_state_id);
                    if self.constraint_solver.lazy_solves || forked.satisfiable() {
                        self.route_successor(forked, false);
                    }
                }
            }
        }

        self.push_or_drop_terminal(STASH_DEADENDED, pending.state);
        Ok(())
    }

    /// Inner body of the pymethods-exposed `resume_after_error`.
    pub(crate) fn _resume_after_error(&mut self, error_msg: &str) -> PyResult<()> {
        let pending = self.pending_callback.take().ok_or_else(|| {
            PyRuntimeError::new_err("no pending callback state for error handling")
        })?;

        let pc = pending.state.pc();
        let state_id = pending.state.state_id();

        log::warn!(
            "P17: Moving state {} to errored stash after callback error at 0x{:x}: {}",
            state_id,
            pc,
            error_msg
        );

        // Record the error
        self.errors.push((pc, error_msg.to_string(), state_id));

        // Move to errored stash
        self.sm
            .stashes_mut()
            .entry(STASH_ERRORED.to_string())
            .or_default()
            .push_back(pending.state);

        Ok(())
    }

    /// Inner body of the pymethods-exposed `resume_after_symbolic_branch`.
    pub(crate) fn _resume_after_symbolic_branch(
        &mut self,
        _py: Python<'_>,
        true_pc: u64,
        false_pc: u64,
        true_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
        false_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        // true_constraints and false_constraints are accepted for API compatibility but
        // the branch condition is sourced from stored_conditions (set by interpreter).
        let _ = (true_constraints, false_constraints);
        let pending = self
            .pending_callback
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("no pending symbolic branch callback"))?;

        // Get the branch condition from stored_conditions (set by interpreter)
        let branch_condition = match &pending.reason {
            CallbackReason::SymbolicBranch { condition_id, .. } => {
                pending.stored_conditions.get(condition_id).cloned()
            }
            _ => None,
        };

        // Add taken-path constraints from deferred forks to the main state
        // BEFORE forking for the symbolic branch. Since fork() creates an
        // independent solver copy, both true_state and false_state will
        // inherit these constraints. Without this, the solver wouldn't know
        // which deferred-fork branch was taken.
        for fork in &pending.deferred_forks {
            if let Some(cond) = pending.stored_conditions.get(&fork.condition_id) {
                if fork.path_taken {
                    pending.state.solver().borrow().assume_true(cond);
                } else {
                    pending.state.solver().borrow().assume_false(cond);
                }
            }
        }

        // Track root state ID for lineage
        let original_state_id = pending.state.state_id();
        let root_state_id = self
            .sm
            .roots()
            .get(&original_state_id)
            .copied()
            .unwrap_or(original_state_id);

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
            let mut snapshots = pending.fork_snapshots;
            for fork in pending.deferred_forks {
                let condition = pending.stored_conditions.get(&fork.condition_id);

                // P11 fix: reconstruct from condition_ast if not in stored_conditions
                let reconstructed_condition = if condition.is_none() {
                    if let Some(ref py_ast) = fork.condition_ast {
                        Python::attach(|py| {
                            let ast = py_ast.bind(py);
                            let solver_ref = true_state.solver();
                            let ctx: &crate::symbolic::SymContext = &solver_ref.borrow();
                            crate::claripy_bridge::claripy_to_rustbv(py, ast, ctx).ok()
                        })
                    } else {
                        None
                    }
                } else {
                    None
                };

                let effective_condition = condition.or(reconstructed_condition.as_ref());

                if let Some(cond) = effective_condition {
                    // Use solver snapshot if available (from before branch constraint)
                    let forked = if let Some(snapshot) = snapshots.remove(&fork.condition_id) {
                        let mut f = true_state.fork_from_snapshot(snapshot);
                        if fork.path_taken {
                            f.solver().borrow().assume_false(cond);
                        } else {
                            f.solver().borrow().assume_true(cond);
                        }
                        f.set_pc(fork.unexplored_target);
                        f
                    } else if fork.path_taken {
                        let mut f = true_state.fork_false(cond);
                        f.set_pc(fork.unexplored_target);
                        f
                    } else {
                        let mut f = true_state.fork_true(cond);
                        f.set_pc(fork.unexplored_target);
                        f
                    };

                    self.sm.set_root(forked.state_id(), root_state_id);

                    if self.constraint_solver.lazy_solves || forked.satisfiable() {
                        deferred_successors.push(forked);
                    } else {
                        log::debug!(
                            "Deferred fork at 0x{:x} is UNSAT, adding to pruned",
                            fork.unexplored_target
                        );
                        deferred_pruned.push(forked);
                    }
                } else {
                    // Missing condition: create conservative fork
                    log::warn!(
                        "Missing condition for deferred fork at 0x{:x} in symbolic branch handler",
                        fork.branch_addr
                    );
                    let mut forked = true_state.fork();
                    forked.set_pc(fork.unexplored_target);
                    self.sm.set_root(forked.state_id(), root_state_id);
                    if self.constraint_solver.lazy_solves || forked.satisfiable() {
                        deferred_successors.push(forked);
                    } else {
                        deferred_pruned.push(forked);
                    }
                }
            }
        }

        // Add states to stashes.
        // When branch_condition is present, the interpreter already proved
        // both paths feasible via can_be_true/can_be_false. Prime the sat
        // cache so downstream satisfiable() checks are free (cache hits)
        // instead of doing redundant Z3 check() calls.
        let mut active_states = Vec::new();
        let mut pruned_states = Vec::new();

        if branch_condition.is_some() {
            true_state.set_sat_cache(true);
            false_state.set_sat_cache(true);
            // Check find/avoid on new states before adding to active
            if self.find_addrs.contains(&true_pc) {
                self.sm
                    .stashes_mut()
                    .entry(STASH_FOUND.to_string())
                    .or_default()
                    .push_back(true_state);
            } else if self.avoid_addrs.contains(&true_pc) {
                self.push_or_drop_terminal(STASH_AVOID, true_state);
            } else {
                active_states.push(true_state);
            }
            if self.find_addrs.contains(&false_pc) {
                self.sm
                    .stashes_mut()
                    .entry(STASH_FOUND.to_string())
                    .or_default()
                    .push_back(false_state);
            } else if self.avoid_addrs.contains(&false_pc) {
                self.push_or_drop_terminal(STASH_AVOID, false_state);
            } else {
                active_states.push(false_state);
            }
        } else {
            // Fallback: no stored condition, need actual sat checks
            if self.constraint_solver.lazy_solves || true_state.satisfiable() {
                active_states.push(true_state);
            } else {
                pruned_states.push(true_state);
            }
            if self.constraint_solver.lazy_solves || false_state.satisfiable() {
                active_states.push(false_state);
            } else {
                pruned_states.push(false_state);
            }
        }

        // Add deferred fork states, checking find/avoid
        for s in deferred_successors {
            let spc = s.pc();
            if self.find_addrs.contains(&spc) {
                self.sm
                    .stashes_mut()
                    .entry(STASH_FOUND.to_string())
                    .or_default()
                    .push_back(s);
            } else if self.avoid_addrs.contains(&spc) {
                self.push_or_drop_terminal(STASH_AVOID, s);
            } else {
                active_states.push(s);
            }
        }

        // Add to active stash, respecting max_active_states.
        for s in active_states {
            self.push_to_active_or_drop(s);
        }

        // Add to pruned stash
        pruned_states.extend(deferred_pruned);
        for s in pruned_states {
            self.push_or_drop_terminal(STASH_PRUNED, s);
        }

        log::debug!(
            "Resumed after symbolic branch: true_pc=0x{:x}, false_pc=0x{:x}",
            true_pc,
            false_pc
        );

        // Apply native uniqueness filter if enabled
        self.apply_uniqueness_filter();
        // Apply native techniques (LengthLimiter, Timeout, LoopBound)
        self.apply_native_techniques();

        Ok(())
    }

    /// Inner body of the pymethods-exposed `resume_find_predicate`.
    pub(crate) fn _resume_find_predicate(&mut self, matched: bool) -> PyResult<()> {
        let pending = self
            .pending_callback
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("no pending find predicate callback"))?;

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
    pub(crate) fn _resume_avoid_predicate(&mut self, matched: bool) -> PyResult<()> {
        let pending = self
            .pending_callback
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("no pending avoid predicate callback"))?;

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
