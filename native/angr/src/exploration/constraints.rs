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

use crate::symbolic::DEFAULT_SOLVER_TIMEOUT_MS;

/// Solver configuration knobs propagated to each per-state solver.
#[derive(Debug, Default)]
pub(crate) struct ConstraintSolver {
    /// When true, skip satisfiability checks on forked states (LAZY_SOLVES).
    pub(crate) lazy_solves: bool,
    /// Z3 solver timeout in milliseconds (default: [`DEFAULT_SOLVER_TIMEOUT_MS`]).
    pub(crate) solver_timeout_ms: u32,
    /// Strict-deterministic witness selection (angr-op0dn.10.3, M2.3). When
    /// true, every state entering a stash gets
    /// [`SymContext::set_deterministic`](crate::symbolic::SymContext::set_deterministic),
    /// so `eval` returns the unsigned-minimum witness and `eval_upto` the
    /// ascending prefix of the feasible set. Forks inherit from their parent,
    /// so seeding the states that enter the manager covers the lineage.
    pub(crate) deterministic: bool,
}

impl ConstraintSolver {
    pub(crate) fn new() -> Self {
        ConstraintSolver {
            lazy_solves: false,
            solver_timeout_ms: DEFAULT_SOLVER_TIMEOUT_MS,
            deterministic: false,
        }
    }
}

/// Apply strict-deterministic witness selection to one state's solver.
///
/// Single seam for the `deterministic` flag so `_create_state`, `_add_state`
/// and `set_deterministic` share one cfg-gated body: on a non-Z3 build there
/// is no `SymContext::set_deterministic` and the flag is inert.
#[cfg(feature = "vex-engine-z3")]
pub(crate) fn apply_state_deterministic(state: &crate::state::RustSimState, v: bool) {
    state.solver().borrow().set_deterministic(v);
}

#[cfg(not(feature = "vex-engine-z3"))]
pub(crate) fn apply_state_deterministic(_state: &crate::state::RustSimState, _v: bool) {}

/// Whether a state's solver is in strict-deterministic mode. Always false on
/// a non-Z3 build, where the mode does not exist.
#[cfg(feature = "vex-engine-z3")]
pub(crate) fn state_is_deterministic(state: &crate::state::RustSimState) -> bool {
    state.solver().borrow().is_deterministic()
}

#[cfg(not(feature = "vex-engine-z3"))]
pub(crate) fn state_is_deterministic(_state: &crate::state::RustSimState) -> bool {
    false
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
