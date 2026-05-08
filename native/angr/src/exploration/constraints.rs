//! Constraint-related state for `RustExplorationManager`.
//!
//! Splits the manager's constraint configuration and tracking into two
//! sub-structs:
//!
//! * `ConstraintSolver` — solver configuration knobs (timeout, lazy solves).
//! * `ConstraintTracker` — per-run bookkeeping for the native uniqueness
//!   filter and the find/avoid predicate skip sets that prevent infinite
//!   loops after `resume_*_predicate(false)`.
//!
//! Same pattern as `ProfilingCollector`: `pub(crate)` direct field access by
//! design — callers read/write the inner fields through a thin delegation.

use std::collections::HashSet;

/// Solver configuration knobs propagated to each per-state solver.
#[derive(Debug, Default)]
pub(crate) struct ConstraintSolver {
    /// When true, skip satisfiability checks on forked states (LAZY_SOLVES).
    pub(crate) lazy_solves: bool,
    /// Z3 solver timeout in milliseconds (default: 30000).
    pub(crate) solver_timeout_ms: u32,
}

impl ConstraintSolver {
    pub(crate) fn new() -> Self {
        ConstraintSolver {
            lazy_solves: false,
            solver_timeout_ms: 30000,
        }
    }
}

/// Per-run tracking sets used by uniqueness filtering and the find/avoid
/// predicate skip-list machinery.
#[derive(Debug, Default)]
pub(crate) struct ConstraintTracker {
    /// Native uniqueness filter: register names to check.
    pub(crate) uniqueness_registers: Vec<String>,
    /// Set of seen register tuple hashes for uniqueness checking.
    pub(crate) uniqueness_set: HashSet<u64>,
    /// State IDs to skip the find predicate check for on next pop.
    /// Set after `resume_find_predicate(false)` to prevent infinite loops.
    pub(crate) skip_find_predicate_states: HashSet<u64>,
    /// State IDs to skip the avoid predicate check for on next pop.
    pub(crate) skip_avoid_predicate_states: HashSet<u64>,
}
