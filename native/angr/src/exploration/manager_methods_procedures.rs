//! `#[pymethods]` for [`RustExplorationManager`]: native procedure management.
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
    /// Disable all native procedures (always use Python).
    #[angr_macros::steady_guarded]
    pub fn disable_native_procedures(&mut self) {
        Arc::make_mut(&mut self.native_procedures).disable_all();
    }

    /// Enable all native procedures.
    #[angr_macros::steady_guarded]
    pub fn enable_native_procedures(&mut self) {
        Arc::make_mut(&mut self.native_procedures).enable_all();
    }

    /// Check if native procedures are enabled.
    pub fn native_procedures_enabled(&self) -> bool {
        self.native_procedures.is_enabled()
    }

    /// Disable a specific native procedure (fall back to Python).
    #[angr_macros::steady_guarded]
    pub fn disable_native_procedure(&mut self, name: &str) {
        Arc::make_mut(&mut self.native_procedures).disable(name);
    }

    /// Enable a specific native procedure.
    #[angr_macros::steady_guarded]
    pub fn enable_native_procedure(&mut self, name: &str) {
        Arc::make_mut(&mut self.native_procedures).enable(name);
    }

    /// Set a Python override for a procedure.
    ///
    /// When set, the native implementation is never called.
    #[angr_macros::steady_guarded]
    pub fn set_python_override(&mut self, name: &str) {
        Arc::make_mut(&mut self.native_procedures).set_python_override(name);
    }

    /// Remove a Python override.
    #[angr_macros::steady_guarded]
    pub fn remove_python_override(&mut self, name: &str) {
        Arc::make_mut(&mut self.native_procedures).remove_python_override(name);
    }

    /// Get list of available native procedures.
    pub fn list_native_procedures(&self) -> Vec<String> {
        self.native_procedures
            .procedure_names()
            .iter()
            .map(std::string::ToString::to_string)
            .collect()
    }

    /// Check if a procedure has a native implementation.
    pub fn has_native_procedure(&self, name: &str) -> bool {
        self.native_procedures.has_native(name)
    }

    /// Get native procedure statistics.
    /// See `stats_api::_native_procedure_stats` for the body.
    pub fn native_procedure_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self._native_procedure_stats(py)
    }

    /// Register a Python callable as a native procedure.
    ///
    /// The callable receives `list[int]` of concrete arg values and must
    /// return `Optional[int]` for the return value (None = no value).
    /// Symbolic arguments cause an automatic fallback to the regular
    /// Python SimProcedure path; the registered callable is only invoked
    /// when all args are concrete.
    #[pyo3(signature = (name, num_args, no_return, callable))]
    #[angr_macros::steady_guarded]
    pub fn register_python_procedure(
        &mut self,
        name: String,
        num_args: usize,
        no_return: bool,
        callable: Py<PyAny>,
    ) {
        let proc = std::sync::Arc::new(crate::procedures::python_proc::PythonNativeProcedure::new(
            name, num_args, no_return, callable,
        ));
        Arc::make_mut(&mut self.native_procedures).register(proc);
    }

    /// Register native `ReturnUnconstrained` stubs: `(display_name, ret_bits)`.
    ///
    /// A `SimLibrary` hands out `ReturnUnconstrained` for every symbol it has
    /// no model for, so those hooks are keyed on the *binary's* symbol name
    /// rather than a libc one and cannot live in the static registry. Python
    /// (`RustExplorationManager._register_simprocedures`) filters
    /// `project._sim_procedures` down to the plain stubs — no `return_val=`
    /// kwarg, prototype return size known — and passes them here.
    ///
    /// Names that already have a real native procedure, or that carry a zero
    /// return width, are skipped: a genuine implementation always outranks a
    /// "return a fresh symbol" stub.
    #[angr_macros::steady_guarded]
    pub fn register_unconstrained_stubs(&mut self, stubs: Vec<(String, u32)>) {
        let registry = Arc::make_mut(&mut self.native_procedures);
        for (name, ret_bits) in stubs {
            if ret_bits == 0 || registry.has_native(&name) {
                continue;
            }
            registry.register(Arc::new(
                crate::procedures::stub::NativeReturnUnconstrained::new(&name, ret_bits),
            ));
        }
    }
}
