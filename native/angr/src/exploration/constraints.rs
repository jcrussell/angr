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
//! It also holds the two cfg-gated free functions that are the single seam for
//! the strict-deterministic flag — `apply_state_deterministic` and
//! `state_is_deterministic` — both inert on a non-`vex-engine-z3` build, plus
//! the claripy constraint-import helper.
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

/// Import a Python list of claripy constraints into one solver context.
///
/// Single body shared by `_add_constraints_to_state` (state_api.rs) and
/// `_add_constraints_to_pending` (pending_api.rs), which differ only in how
/// they reach the `SymContext` and in their log wording — a correctness fix to
/// the fast path used to have to land in two places (angr-9ke6b.71).
///
/// Two paths per constraint:
///
/// * **Fast path** (Z3 builds): ask claripy's z3 backend to convert the AST and
///   assert the resulting raw Z3 pointer directly. The RustBV conversion is
///   still attempted *first* so a constraint that has a RustBV form is recorded
///   in the assumed IR rather than double-logged as a residual — see
///   [`SymContext::add_constraint_raw_assumed`](crate::symbolic::SymContext::add_constraint_raw_assumed)
///   (angr-op0dn.14.2).
/// * **Slow path**: convert to a `RustBV` and assume it true (widening a
///   non-boolean value to `bv != 0`).
///
/// `kind` is the noun used in the per-constraint failure log ("initial" /
/// "pending"); the caller emits its own summary line. Returns the number of
/// constraints successfully asserted — unconvertible ones are skipped, matching
/// the pre-existing behavior of both call sites.
pub(crate) fn import_python_constraints(
    py: pyo3::Python<'_>,
    ctx: &crate::symbolic::SymContext,
    constraints: &pyo3::Bound<'_, pyo3::types::PyList>,
    kind: &str,
) -> u32 {
    use crate::claripy_bridge::claripy_to_rustbv;
    use crate::symbolic::RustBV;
    use pyo3::prelude::*;

    // Pre-fetch Z3 backend for the raw-pointer fast path.
    // SILENT(cat-a): probing for claripy's optional z3 backend; a missing
    // backend is expected control flow (the loop below simply takes the
    // generic slow path instead of the typed fast path), so collapsing the
    // error to None here loses no correctness.
    #[cfg(feature = "vex-engine-z3")]
    let z3_backend = py
        .import("claripy")
        .and_then(|c| c.getattr("backends"))
        .and_then(|b| b.getattr("z3"))
        .ok();

    let mut added = 0u32;
    for item in constraints.iter() {
        // Fast path: extract typed Z3 AST handle and assert directly.
        #[cfg(feature = "vex-engine-z3")]
        {
            if let Some(ref backend) = z3_backend
                && let Ok(z3_obj) = backend.call_method1("convert", (&item,))
                && let Ok(ast_ref) = z3_obj.call_method0("as_ast")
                && let Ok(ptr) = ast_ref.getattr("value").and_then(|v| v.extract::<usize>())
            {
                let z3_ctx = z3::Context::thread_local();
                // SAFETY: claripy's z3 backend returned this pointer for a live
                // AST it caches; matches our thread-local Z3 context.
                if let Some(z3_ast) =
                    unsafe { crate::symbolic::Z3AstPtr::from_borrowed_raw(&z3_ctx, ptr) }
                {
                    match claripy_to_rustbv(py, &item, ctx) {
                        Ok(bv) => {
                            ctx.add_constraint_raw_assumed(z3_ast);
                            ctx.assumed_constraints_push(bv, true);
                        }
                        Err(_) => ctx.add_constraint_raw(z3_ast),
                    }
                    added += 1;
                    continue;
                }
            }
        }

        // Slow path: convert via RustBV.
        match claripy_to_rustbv(py, &item, ctx) {
            Ok(bv) => {
                if bv.width() == 1 {
                    ctx.assume_true(&bv);
                } else {
                    let zero = RustBV::concrete(0, bv.width());
                    let neq = bv.ne(&zero, ctx);
                    ctx.assume_true(&neq);
                }
                added += 1;
            }
            Err(e) => {
                log::debug!("Could not convert {kind} constraint: {e}");
            }
        }
    }
    added
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
