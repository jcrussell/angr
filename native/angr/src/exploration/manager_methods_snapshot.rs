//! `#[pymethods]` for [`RustExplorationManager`]: stash snapshot
//! serialize/restore — `dump_snapshot_bytes` / `load_snapshot_bytes`,
//! versioned `StashManager` byte envelopes that both finalize a live steady
//! session first (see their docs for the finalize-then-capture contract).
//!
//! One of several `#[pymethods]` blocks for the pyclass; see the
//! `manager_methods` module doc for the split rationale (angr-9ke6b.50).
//!
//! This file was called `manager_methods_stats.rs` until angr-03vl4.25 and
//! also held the solver-profiling counters and the constraint-sharing walk;
//! those moved to `manager_methods_diagnostics`, since the old name read as
//! the `#[pymethods]` half of `stats_api` (which it never was — `stats_api`'s
//! wrappers live in `manager_methods_constraints` and
//! `manager_methods_procedures`).
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
    #[angr_macros::steady_guarded]
    pub fn dump_snapshot_bytes<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
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
    ///
    /// Deliberately NOT `#[angr_macros::steady_guarded]` (unlike its dump-side
    /// sibling [`Self::dump_snapshot_bytes`]): the guard must run AFTER the
    /// fallible parse below, not as the unconditional first statement the
    /// macro would inject — a malformed envelope should error out without
    /// finalizing a live session that turns out not to be replaced.
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

test_submod!("manager_methods_snapshot_tests.rs" => tests);
