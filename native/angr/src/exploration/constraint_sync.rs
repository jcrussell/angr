//! Python→Rust constraint synchronization after a Python callback bounce.
//!
//! Split out of `exploration::helpers` (angr-9ke6b.76). A Python SimProcedure
//! or breakpoint may add constraints to the bounced state; those must land in
//! the Rust solver before execution resumes. The reverse direction is handled
//! by attaching a `RustSolverContext` to the Python state, so nothing is
//! replayed across the FFI — see
//! [`RustExplorationManager::sync_constraints_from_python`].
//!
//! **Panic policy (angr-qwyti.11):** Python-boundary module, so it carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

impl RustExplorationManager {
    /// Sync constraints from Python callbacks back to the Rust state's solver.
    ///
    /// Python SimProcedures may add new constraints (e.g., strcmp conditions);
    /// this method syncs those new constraints BACK to Rust after the callback.
    /// The reverse direction (Rust→Python) is handled by attaching a
    /// `RustSolverContext` to the Python state in
    /// `rust_callback_dispatch._install_rust_solver_on_callback_state`, so
    /// constraints never need to be replayed across the FFI.
    ///
    /// Without this, constraints added by SimProcedures would be lost when
    /// Rust resumes execution, leading to incorrect symbolic evaluation.
    ///
    /// Conversion strategy: the shared seam
    /// [`import_one_constraint`](super::constraints::import_one_constraint),
    /// with [`ConstraintTier::RustBvFirst`](super::constraints::ConstraintTier::RustBvFirst)
    /// — `claripy_to_rustbv` first (which preserves the assumed_constraints
    /// tracking Python re-import depends on), falling back to lossless Z3
    /// pointer extraction when it hits an unsupported op (FP, etc). Both tiers
    /// share the same Z3 context with claripy via the shared backend.
    ///
    /// The order is the mirror of `import_python_constraints`'s — see
    /// [`ConstraintTier`](super::constraints::ConstraintTier) for why either
    /// order is correct. Keeping RustBV first here also keeps this method
    /// exercisable from `cargo test` without installing claripy's `Z3_context`
    /// process-wide: the raw-pointer tier requires that install (see the
    /// `#[ignore]` rationale on `sync_constraints_fp_rescued_via_z3_pointer`),
    /// so a Z3PtrFirst sync would push every tier-1 test behind `--ignored`.
    ///
    /// Returns:
    ///   - Ok(true): Constraints synced and state is SAT (satisfiable)
    ///   - Ok(false): State became UNSAT after syncing - should be pruned
    ///   - Err: Python error during sync
    pub(crate) fn sync_constraints_from_python(
        &self,
        py: Python<'_>,
        state: &RustSimState,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<bool> {
        use pyo3::types::PyListMethods;

        use super::constraints::{
            ConstraintImport, ConstraintTier, import_one_constraint, resolve_claripy_z3_backend,
        };

        let solver_ref = state.solver();
        let sym_ctx = solver_ref.borrow();
        let ctx_ref: &SymContext = &sym_ctx;

        let mut success_count = 0usize;
        let mut z3_ptr_fallback_count = 0usize;
        let mut failed_count = 0usize;

        // Resolved once for the whole list, not per constraint.
        let z3_backend = resolve_claripy_z3_backend(py);

        // Extract list items - we need to convert each to RustBV
        let len = constraints.len();
        for i in 0..len {
            // Use get_item with usize index. An in-bounds `PyList::get_item`
            // essentially never fails, but a failure here must still be
            // counted and logged like every other failure branch below —
            // otherwise the `failed_count + success_count` total logged at the
            // end silently undercounts the list length (angr-sqfj8.57).
            let constraint = match constraints.get_item(i) {
                Ok(c) => c,
                Err(e) => {
                    failed_count += 1;
                    log::warn!(
                        "Constraint {i} could not be read from the Python list: {e}. \
                         Solver state may diverge."
                    );
                    continue;
                }
            };

            match import_one_constraint(
                py,
                ctx_ref,
                &constraint,
                z3_backend.as_ref(),
                ConstraintTier::RustBvFirst,
            ) {
                ConstraintImport::Assumed => success_count += 1,
                ConstraintImport::RawOnly(e) => {
                    z3_ptr_fallback_count += 1;
                    success_count += 1;
                    log::debug!("Constraint {i} fell back to Z3 ptr (claripy_to_rustbv: {e})");
                }
                ConstraintImport::Failed(e) => {
                    failed_count += 1;
                    log::warn!("Constraint {i} conversion failed: {e}. Solver state may diverge.");
                }
            }
        }

        if z3_ptr_fallback_count > 0 {
            log::debug!(
                "sync_constraints_from_python: {z3_ptr_fallback_count} constraints rescued via Z3 ptr fallback"
            );
        }

        if failed_count > 0 {
            log::warn!(
                "sync_constraints_from_python: {}/{} constraints failed to convert",
                failed_count,
                // overflow-ok: both are `usize` tallies over the one Python
                // constraint list this fn walks, so the sum is that list's length.
                failed_count + success_count
            );
        }

        if success_count > 0 {
            log::debug!("Synced {success_count} constraints from Python to Rust");
        }

        // Check satisfiability and return status so callers can prune UNSAT states.
        // Only a *decided* Unsat prunes: an undecided query (Z3 Unknown /
        // timeout) says nothing about feasibility, and dropping the state on it
        // would lose a path that is very likely feasible (angr-03vl4.62,
        // `invariant-z3-unknown-not-unsat`). The state is kept and the caller
        // re-checks it on the next solver query.
        #[cfg(feature = "vex-engine-z3")]
        match sym_ctx.is_sat_checked() {
            Some(false) => {
                log::debug!(
                    "Constraints are UNSAT after syncing {success_count} from Python (failed={failed_count}). \
                     Returning false to trigger pruning."
                );
                return Ok(false);
            }
            None => {
                log::warn!(
                    "Satisfiability undecided (Z3 timeout) after syncing {success_count} constraints \
                     from Python (failed={failed_count}); keeping the state rather than pruning it."
                );
            }
            Some(true) => {}
        }

        // Partial-sync re-check: if any constraints failed to convert, do an
        // explicit SAT check. Failed conversions can leave state in divergent state.
        // Defence-in-depth only: under `vex-engine-z3` the unconditional
        // post-sync SAT check above already rejects everything this gate could
        // catch. Kept for a build where that check is compiled out. `sync_constraints_partial_failure_unsat_is_pruned`
        // (helpers_tests.rs) pins the observable contract — partial sync plus
        // contradictory constraints prunes — not which gate fires.
        #[cfg(feature = "vex-engine-z3")]
        if failed_count > 0 && success_count > 0 {
            // Same decided-Unsat-only rule as the post-sync check above.
            if sym_ctx.is_sat_checked() == Some(false) {
                log::debug!(
                    "State became UNSAT with partial constraint sync ({}/{} failed). Pruning.",
                    failed_count,
                    // overflow-ok: both are `usize` tallies over the one Python
                    // constraint list this fn walks, so the sum is that list's length.
                    failed_count + success_count
                );
                return Ok(false);
            }
        }

        Ok(true)
    }
}

test_submod!("constraint_sync_tests.rs" => tests);
