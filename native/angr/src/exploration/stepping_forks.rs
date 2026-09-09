//! Deferred-fork materialization into a successor list.
//!
//! Extracted from `stepping.rs` (angr-5mnx3.71).
//! [`RustExplorationManager::process_deferred_forks_into`] is the
//! single-threaded materializer used by the step-driver arms that do not go
//! through the main MaxBlocks/BlockEnd path (notably the generic skip for
//! unmodeled calls in
//! `stepping_bounce.rs`); `dispatch_fork_inspect` is the `state.inspect`
//! `fork` breakpoint dispatch it fires per forked state.

use super::*;

impl RustExplorationManager {
    /// Process deferred forks and add the resulting forked states to the successor list.
    /// This is used by code paths (like the unmodeled-call generic skip) that don't go through
    /// the main MaxBlocks/BlockEnd deferred fork processing.
    pub(crate) fn process_deferred_forks_into(
        &mut self,
        successors: &mut Vec<RustSimState>,
        deferred_forks: Vec<DeferredFork>,
        stored_conditions: &FxHashMap<u64, RustBV>,
        mut fork_snapshots: FxHashMap<u64, crate::interpreter::BranchSnapshot>,
    ) {
        if deferred_forks.is_empty() {
            return;
        }

        let root_state_id = self.sm.root_or_self(successors[0].state_id());
        // Guards of the forks already materialized. `successors[0]` accumulates
        // them below, but a fork built from a pre-branch *snapshot* does not —
        // see `PriorGuards` (angr-62ar5).
        let mut prior_guards = super::fork_materialize::PriorGuards::new(true);

        for fork in &deferred_forks {
            if let Some(condition) = stored_conditions.get(&fork.condition_id) {
                // Mint the unexplored-path fork, THEN add the taken-path
                // constraint to the main state (which fires the constraints
                // inspect BP around the add — angr-op0dn.14.4.1). The order is
                // load-bearing: see `fork_unexplored_and_guard_base`.
                let forked = super::fork_materialize::fork_unexplored_and_guard_base(
                    self.callbacks.as_ref(),
                    &successors[0],
                    fork,
                    condition,
                    &mut fork_snapshots,
                    &prior_guards,
                );
                prior_guards.record(condition.clone(), fork.path_taken);

                self.sm.set_root(forked.state_id(), root_state_id);

                // state.inspect fork BP — see `dispatch_fork_inspect` for rationale.
                self.dispatch_fork_inspect(forked.state_id());

                if forked.survives_sat_prune(self.constraint_solver.lazy_solves) {
                    successors.push(forked);
                } else {
                    self.push_or_drop_terminal(STASH_PRUNED, forked);
                }
            } else {
                // Conservative fork without condition
                let mut forked = successors[0].fork();
                forked.set_pc(fork.unexplored_target);
                self.sm.set_root(forked.state_id(), root_state_id);

                self.dispatch_fork_inspect(forked.state_id());

                if forked.survives_sat_prune(self.constraint_solver.lazy_solves) {
                    successors.push(forked);
                } else {
                    self.push_or_drop_terminal(STASH_PRUNED, forked);
                }
            }
        }

        self.profiling.accumulated_stats.deferred_fork_count += deferred_forks.len() as u64;
    }

    /// Fire a `state.inspect.fork` BP for the given forked state id.
    /// Bit-gated on `InspectEvent::Fork` (bit 4) — single atomic load in
    /// the common no-BP case. Dispatches `when='after'` with no attrs,
    /// matching Python `SimSuccessors._preprocess_successor`
    /// (`angr/engines/successors.py`), where the BP fires
    /// on the newly-added successor after constraints + ip are applied
    /// but before satisfiability is checked downstream. Errors from the
    /// user's BP action are swallowed (logged at debug) — same MVP
    /// pattern as the other Rust-side inspect dispatchers.
    #[inline]
    pub(crate) fn dispatch_fork_inspect(&self, forked_state_id: u64) {
        let cb = match self.callbacks.as_ref() {
            Some(c) => c,
            None => return,
        };
        // Fork = bit 4 (reserved slot mirrored in
        // `_INSPECT_EVENT_SPECS["fork"]`).
        if !cb.inspect_event_enabled(crate::callbacks::InspectBit::Fork) {
            return;
        }
        // call_inspect_fork self-attaches the GIL (angr-vh834 Phase 4), so no
        // explicit Python::attach wrapper is needed here.
        if let Err(e) = cb.call_inspect_fork(forked_state_id as i64, "after") {
            log::debug!("fork inspect dispatch raised (state {forked_state_id}): {e}");
        }
    }
}
