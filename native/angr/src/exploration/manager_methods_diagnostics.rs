//! `#[pymethods]` for [`RustExplorationManager`]: solver/constraint
//! diagnostics — read-only introspection that reports on the engine's own
//! behaviour rather than on the exploration result.
//!
//! One of several `#[pymethods]` blocks for the pyclass; see the
//! `manager_methods` module doc for the split rationale (angr-9ke6b.50).
//!
//! Two groups live here:
//!
//! - **Solver profiling**: `get_solver_stats`, `reset_solver_stats` (static,
//!   process-global Z3 counters).
//! - **Constraint-sharing analysis**: `analyze_constraint_sharing` — walks
//!   every stash's assumed-constraint DAG and reports pointer- vs
//!   structural-identity sharing (angr-zdho).
//!
//! Split out of the former `manager_methods_stats` in angr-03vl4.25; the
//! per-exploration counters a reader looking for "stats" usually wants are in
//! `stats_api`, whose wrappers live in `manager_methods_constraints`
//! (`stats`, `get_fallback_stats`) and `manager_methods_procedures`
//! (`native_procedure_stats`).
//!
//! This is a Python-boundary module; `unwrap`/`expect` are denied here so a
//! future panic-on-input landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]` (angr-qwyti.11 enforcement layer).
#![deny(clippy::unwrap_used, clippy::expect_used)]
use super::*;

#[angr_macros::steady_guard_checked]
#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pymethods]
impl RustExplorationManager {
    /// Get global Z3 solver profiling stats as a dict.
    #[staticmethod]
    pub fn get_solver_stats() -> std::collections::HashMap<String, u64> {
        crate::symbolic::get_solver_stats()
    }

    /// Reset global Z3 solver profiling stats to zero.
    #[staticmethod]
    pub fn reset_solver_stats() {
        crate::symbolic::reset_solver_stats()
    }

    /// Walk every state's assumed-constraint RustBV graph and report
    /// pointer-identity vs structural-identity sharing (angr-zdho).
    ///
    /// Returns a dict with:
    ///   - `total_visits`     — DAG descents, counting Arc re-visits.
    ///   - `unique_pointers`  — distinct RustBV Arc allocations seen. The
    ///     current per-call `to_z3_ast_cached` cache collapses repeat
    ///     visits of the same Arc pointer down to this.
    ///   - `unique_shapes`    — distinct structural shapes. A
    ///     construction-time hash-cons (angr-behq) would dedupe to this.
    ///   - `structural_duplicates` — `unique_pointers - unique_shapes`.
    ///   - `states_analyzed`  — how many states contributed constraints.
    ///   - `constraints_analyzed` — total `(RustBV, bool)` pairs folded in.
    ///
    /// Walks ALL stashes (so it's deterministic across exploration
    /// outcomes — no `find`/`avoid` bias).
    pub fn analyze_constraint_sharing(&self) -> std::collections::HashMap<String, u64> {
        let mut walk = crate::symbolic::ConstraintSharingWalk::new();
        let mut states_analyzed: u64 = 0;
        let mut constraints_analyzed: u64 = 0;
        for stash in self.sm.stashes().values() {
            for state in stash.iter() {
                let ctx = state.solver().borrow();
                let n_constraints = ctx.assumed_constraint_count();
                if n_constraints == 0 {
                    continue;
                }
                ctx.fold_sharing_walk(&mut walk);
                states_analyzed += 1;
                constraints_analyzed = constraints_analyzed.saturating_add(n_constraints as u64);
            }
        }
        let stats = walk.into_stats();
        let mut out = std::collections::HashMap::new();
        out.insert("total_visits".into(), stats.total_visits);
        out.insert("unique_pointers".into(), stats.unique_pointers);
        out.insert("unique_shapes".into(), stats.unique_shapes);
        out.insert(
            "structural_duplicates".into(),
            stats.unique_pointers.saturating_sub(stats.unique_shapes),
        );
        out.insert("states_analyzed".into(), states_analyzed);
        out.insert("constraints_analyzed".into(), constraints_analyzed);
        out
    }
}

test_submod!("manager_methods_diagnostics_tests.rs" => tests);
