//! `#[pymethods]` for [`RustExplorationManager`]: solver profiling stats.
//!
//! One of several `#[pymethods]` blocks for the pyclass; see the
//! `manager_methods` module doc for the split rationale (angr-9ke6b.50).
//!
//! This is a Python-boundary module; `unwrap`/`expect` are denied here so a
//! future panic-on-input landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]` (angr-qwyti.11 enforcement layer).
#![deny(clippy::unwrap_used, clippy::expect_used)]
use super::*;

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

    /// Serialize the full stash manager (all stashes, lineage, counters) to
    /// a versioned byte envelope. Wraps [`StashManager::dump_snapshot`]
    /// (see `stash.rs::STASH_SNAPSHOT_VERSION`). Bucket-D `Py<PyAny>`
    /// overlays (symbolic_pages / hook_symbolic_memory / addr_to_ast) are
    /// NOT captured — the Python wrapper handles those via
    /// `claripy.dumps`/`loads`.
    ///
    /// angr-op0dn.13.6: takes `&mut self` and finalizes a live steady session
    /// first. Under steady mode (angr-nkoct) the frontier lives inside the
    /// worker Z3 contexts and is in NO stash, so dumping `self.sm` mid-session
    /// would silently truncate the search — the snapshot would look valid and
    /// resume with a smaller frontier. Finalize-then-capture is the contract.
    pub fn dump_snapshot_bytes<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        self.steady_config_guard();
        self.flush_parked_bounces_to_active();
        let bytes = self.sm.dump_snapshot();
        PyBytes::new(py, &bytes)
    }

    /// Restore the stash manager from a [`Self::dump_snapshot_bytes`]
    /// envelope. Replaces `self.sm` wholesale; manager-level configuration
    /// (find/avoid addrs, hooks, simprocedures, solver/memory config) is
    /// preserved. An empty envelope or stale version byte raises
    /// `ValueError`.
    ///
    /// angr-ph300.21: mirrors the dump-side finalize contract before the
    /// swap. A manager parked on `need_callback`, or with a live steady
    /// session (`RUST_PARALLEL_STEADY=1`), carries worker-resident states and
    /// pending callback/bounce entries that belong to the PRE-restore world.
    /// Without a teardown, the next `run()` would finalize the OLD steady
    /// session — draining those states into the freshly restored stashes (two
    /// explorations merged) — and a surviving pending callback could resume a
    /// state from the discarded world. We therefore `steady_config_guard()`
    /// first (drains any live session into the soon-to-be-discarded `self.sm`)
    /// and clear all pending single-step / callback / parked-bounce state, so
    /// the restored frontier starts from a clean manager.
    pub fn load_snapshot_bytes(&mut self, bytes: &[u8]) -> PyResult<()> {
        let restored = StashManager::load_snapshot(bytes)
            .map_err(|e| PyValueError::new_err(format!("snapshot load failed: {e}")))?;
        self.steady_config_guard();
        self.pending_callbacks.clear();
        self.pending_parallel_bounces.clear();
        self.current_stepping_state_id = None;
        self.sm = restored;
        Ok(())
    }
}
