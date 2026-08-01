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
    /// Conversion strategy: claripy_to_rustbv first (preserves assumed_constraints
    /// tracking that Python re-import depends on), then fall back to lossless Z3
    /// pointer extraction when claripy_to_rustbv hits an unsupported op (FP, etc).
    /// Both paths share the same Z3 context with claripy via the shared backend.
    ///
    /// Returns:
    ///   - Ok(true): Constraints synced and state is SAT (satisfiable)
    ///   - Ok(false): State became UNSAT after syncing - should be pruned (P12)
    ///   - Err: Python error during sync
    pub(crate) fn sync_constraints_from_python(
        &self,
        py: Python<'_>,
        state: &RustSimState,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<bool> {
        use pyo3::types::PyListMethods;

        let solver_ref = state.solver();
        let sym_ctx = solver_ref.borrow();
        let ctx_ref: &SymContext = &sym_ctx;

        let mut success_count = 0usize;
        let mut z3_ptr_fallback_count = 0usize;
        let mut failed_count = 0usize;

        // Lazily resolve claripy.backends.z3 once for the Z3 ptr fallback path.
        // None on first failure so we don't keep paying the import cost.
        #[cfg(feature = "vex-engine-z3")]
        let mut z3_backend: Option<Bound<'_, PyAny>> = None;

        // Extract list items - we need to convert each to RustBV
        let len = constraints.len();
        for i in 0..len {
            // Use get_item with usize index
            if let Ok(constraint) = constraints.get_item(i) {
                // Convert claripy AST to RustBV
                match claripy_to_rustbv(py, &constraint, ctx_ref) {
                    Ok(bv) => {
                        // Add constraint to solver
                        #[cfg(feature = "vex-engine-z3")]
                        {
                            if bv.width() == 1 {
                                sym_ctx.assume_true(&bv);
                                success_count += 1;
                            } else {
                                // For wider values, interpret as "value != 0"
                                let zero = RustBV::concrete(0, bv.width());
                                let neq = bv.ne(&zero, ctx_ref);
                                sym_ctx.assume_true(&neq);
                                success_count += 1;
                            }
                        }
                        #[cfg(not(feature = "vex-engine-z3"))]
                        {
                            // Without Z3, constraints are tracked but not solved
                            success_count += 1;
                        }
                    }
                    Err(e) => {
                        // Fallback: extract the constraint's underlying Z3 AST
                        // pointer via claripy.backends.z3 and assert it on the
                        // solver directly. claripy and the Rust solver share a
                        // Z3 context, so the pointer is valid here. This rescues
                        // constraints that use ops claripy_to_rustbv doesn't
                        // model (e.g. FP), keeping the round-trip lossless.
                        #[cfg(feature = "vex-engine-z3")]
                        {
                            let backend = match z3_backend.as_ref() {
                                Some(b) => Some(b.clone()),
                                None => match Self::resolve_claripy_z3_backend(py) {
                                    Ok(b) => {
                                        z3_backend = Some(b.clone());
                                        Some(b)
                                    }
                                    Err(import_err) => {
                                        log::debug!(
                                            "claripy.backends.z3 unavailable for fallback: {import_err}"
                                        );
                                        None
                                    }
                                },
                            };

                            let z3_ast = backend.as_ref().and_then(|b| {
                                Self::extract_z3_ptr_from_claripy(b, &constraint)
                                    .ok()
                                    .flatten()
                            });

                            if let Some(ast) = z3_ast {
                                sym_ctx.add_constraint_raw(ast);
                                z3_ptr_fallback_count += 1;
                                success_count += 1;
                                log::debug!(
                                    "Constraint {i} fell back to Z3 ptr (claripy_to_rustbv: {e})"
                                );
                                continue;
                            }
                        }

                        failed_count += 1;
                        log::warn!(
                            "Constraint {i} conversion failed: {e}. Solver state may diverge."
                        );
                    }
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
                failed_count + success_count
            );
        }

        if success_count > 0 {
            log::debug!("Synced {success_count} constraints from Python to Rust");
        }

        // P12: Check satisfiability and return status so callers can prune UNSAT states
        #[cfg(feature = "vex-engine-z3")]
        {
            let is_sat = sym_ctx.is_sat();
            if !is_sat {
                log::debug!(
                    "P12: Constraints are UNSAT after syncing {success_count} from Python (failed={failed_count}). \
                     Returning false to trigger pruning."
                );
                return Ok(false);
            }
        }

        // P14: If many constraints failed to convert, do explicit SAT check
        // Failed conversions can leave state in divergent state.
        // Defence-in-depth only: under `vex-engine-z3` the P12 block above
        // SAT-checks unconditionally, so it already rejects everything this
        // gate could catch. Kept for a build where P12's check is compiled
        // out. `sync_constraints_partial_failure_unsat_is_pruned`
        // (helpers_tests.rs) pins the observable contract — partial sync plus
        // contradictory constraints prunes — not which gate fires.
        #[cfg(feature = "vex-engine-z3")]
        if failed_count > 0 && success_count > 0 {
            let is_sat = sym_ctx.is_sat();
            if !is_sat {
                log::debug!(
                    "P14: State became UNSAT with partial constraint sync ({}/{} failed). Pruning.",
                    failed_count,
                    failed_count + success_count
                );
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Resolve `claripy.backends.z3` once per call to `sync_constraints_from_python`.
    /// Returned as a `Bound<PyAny>` because that's what `convert(...)` needs.
    #[cfg(feature = "vex-engine-z3")]
    fn resolve_claripy_z3_backend(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let claripy = py.import("claripy")?;
        let backends = claripy.getattr("backends")?;
        backends.getattr("z3")
    }

    /// Extract a typed [`Z3AstPtr`] from a claripy AST by going through
    /// `claripy.backends.z3.convert(ast).as_ast().value`. Returns `Ok(None)`
    /// if the conversion succeeded but the extracted pointer is null;
    /// returns `Err` if the Python conversion path itself failed.
    ///
    /// The returned handle has its own `Z3_inc_ref` ref; callers can drop
    /// it without affecting claripy's cached AST.
    #[cfg(feature = "vex-engine-z3")]
    fn extract_z3_ptr_from_claripy(
        z3_backend: &Bound<'_, PyAny>,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Z3AstPtr>> {
        let z3_obj = z3_backend.call_method1("convert", (ast,))?;
        let raw_ast = z3_obj.call_method0("as_ast")?;
        let ptr = raw_ast.getattr("value")?.extract::<usize>()?;
        let ctx = z3::Context::thread_local();
        // SAFETY: claripy's z3 backend returned this pointer for a live
        // AST it holds in its own cache; the AST is in the process-global
        // Z3 context, which matches our thread-local context.
        Ok(unsafe { Z3AstPtr::from_borrowed_raw(&ctx, ptr) })
    }
}

#[cfg(test)]
#[path = "constraint_sync_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
