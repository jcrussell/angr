//! Deferred-fork / branch-condition / fork-snapshot accessors.
use super::*;

impl<'a> VEXInterpreter<'a> {
    /// Take the deferred forks, leaving an empty vector.
    pub(crate) fn take_deferred_forks(&mut self) -> Vec<DeferredFork> {
        std::mem::take(&mut self.deferred_forks)
    }

    /// Clear the deferred forks.
    ///
    /// Production drains the vector with `take_deferred_forks` at the bounce
    /// point; only `execution_tests` uses this accessor (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn clear_deferred_forks(&mut self) {
        self.deferred_forks.clear();
    }

    /// Get the number of deferred forks.
    ///
    /// Production drains the vector with `take_deferred_forks` at the bounce
    /// point; only `execution_tests` uses this accessor (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn num_deferred_forks(&self) -> usize {
        self.deferred_forks.len()
    }

    /// Take the last branch condition, if any.
    ///
    /// This is set when a SymbolicBranch result is created, and can be retrieved
    /// by callers who need to add constraints for forked states.
    /// The condition is cleared after being retrieved.
    pub(crate) fn take_last_branch_condition(&mut self) -> Option<RustBV> {
        self.last_branch_condition.take()
    }

    /// Get a stored condition by ID.
    ///
    /// Returns the branch condition associated with the given condition ID,
    /// if one was stored. This is used for deferred fork handling.
    ///
    /// Production bulk-drains via `take_stored_conditions`; only
    /// `execution_tests` looks up a single id (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn get_stored_condition(&self, condition_id: u64) -> Option<&RustBV> {
        self.stored_conditions.get(&condition_id)
    }

    /// Take all stored conditions.
    ///
    /// Returns all stored conditions as a HashMap. The internal map is cleared.
    /// This is useful for bulk retrieval when processing multiple deferred forks.
    pub(crate) fn take_stored_conditions(&mut self) -> FxHashMap<u64, RustBV> {
        std::mem::take(&mut self.stored_conditions)
    }

    /// Take all branch snapshots for deferred forks.
    ///
    /// Returns full state snapshots captured BEFORE branch constraints were added,
    /// keyed by condition_id. Used for correct alternate-path forking.
    pub(crate) fn take_fork_snapshots(&mut self) -> FxHashMap<u64, BranchSnapshot> {
        std::mem::take(&mut self.fork_snapshots)
    }
}
