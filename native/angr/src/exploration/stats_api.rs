//! Statistics-export methods.
//!
//! Bodies for the pyclass-exposed methods that aggregate counters and
//! state-population stats into Python dicts — `stats`, `get_fallback_stats`,
//! and `native_procedure_stats`. The pyclass-facing thin wrappers live in
//! `mod.rs` and forward to the `pub(crate)` bodies in this module.
//!
//! PyO3 0.27.2 in this project does not enable `multiple-pymethods`, so each
//! pyclass is limited to a single `#[pymethods]` impl block — see
//! `invariant-pyo3-single-pymethods-impl`. This module mirrors the
//! `helpers.rs` / `stepping.rs` / `run_loop.rs` / `resume.rs` /
//! `pending_api.rs` / `state_api.rs` extension-impl pattern used elsewhere in
//! `exploration/`.

use super::*;

impl RustExplorationManager {
    pub(crate) fn _stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("steps", self.steps)?;
        dict.set_item("active", self.active_count())?;
        dict.set_item("found", self.found_count())?;
        dict.set_item("errors", self.errors.len())?;
        dict.set_item("hooks", self.hooks.len())?;
        dict.set_item("simprocedures", self.simprocedures.len())?;
        dict.set_item("find_addrs", self.find_addrs.len())?;
        dict.set_item("avoid_addrs", self.avoid_addrs.len())?;
        dict.set_item("block_cache_size", self.environment.block_cache.len())?;
        dict.set_item("native_proc_calls", self.profiling.native_proc_stats.native_calls)?;
        dict.set_item("native_proc_fallbacks", self.profiling.native_proc_stats.python_fallbacks)?;
        dict.set_item("avoided_count", self.sm.avoided_count)?;
        dict.set_item("pruned_count", self.sm.pruned_count)?;
        dict.set_item("deadended_count", self.sm.deadended_count)?;
        dict.set_item("drop_terminal_states", self.sm.drop_terminal_states())?;
        dict.set_item("state_roots_size", self.sm.roots().len())?;
        dict.set_item("vex_fallback_count", self.vex_fallback_count)?;
        dict.set_item("vex_fallback_unique_addrs", self.vex_fallback_addrs.len())?;
        dict.set_item("dcas_unsupported_count", self.dcas_unsupported_count)?;
        dict.set_item(
            "simprocedure_python_fallback_count",
            self.simprocedure_python_fallback_count,
        )?;
        dict.set_item(
            "syscall_python_fallback_count",
            self.syscall_python_fallback_count,
        )?;
        Ok(dict)
    }

    pub(crate) fn _get_fallback_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("count", self.vex_fallback_count)?;
        let addrs = PyDict::new(py);
        for (&addr, reason) in &self.vex_fallback_addrs {
            addrs.set_item(format!("0x{:x}", addr), reason)?;
        }
        dict.set_item("addresses", addrs)?;
        dict.set_item("dcas_unsupported_count", self.dcas_unsupported_count)?;
        dict.set_item(
            "simprocedure_python_fallback_count",
            self.simprocedure_python_fallback_count,
        )?;
        dict.set_item(
            "syscall_python_fallback_count",
            self.syscall_python_fallback_count,
        )?;
        Ok(dict)
    }

    pub(crate) fn _native_procedure_stats<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("native_calls", self.profiling.native_proc_stats.native_calls)?;
        dict.set_item("python_fallbacks", self.profiling.native_proc_stats.python_fallbacks)?;

        let call_counts = PyDict::new(py);
        for (name, count) in &self.profiling.native_proc_stats.call_counts {
            call_counts.set_item(name, *count)?;
        }
        dict.set_item("call_counts", call_counts)?;

        Ok(dict)
    }
}
