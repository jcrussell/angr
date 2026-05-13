//! Python-callable wrapper for the native SimProcedure registry.
//!
//! `PythonNativeProcedure` lets a Python callable be registered as if it
//! were a Rust-side native procedure. The wrapper extracts concrete u64
//! arguments before invoking the callable; symbolic arguments cause a
//! `SymbolicArgument` error so the dispatcher falls back to Python's
//! regular SimProcedure path. The callable returns `Optional[int]` for
//! the procedure's return value (None = no return value, or pair with
//! `no_return=True` for terminal procedures like exit/abort).
//!
//! Contract:
//! ```python
//! def my_proc(args: list[int]) -> Optional[int]:
//!     # args are concrete u64 values
//!     # return None = no return value
//!     # return int = use as return value (wrapped in arch-bits BV)
//!     return 42
//! ```

use pyo3::prelude::*;

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Wraps a Python callable so it can act as a native SimProcedure.
pub struct PythonNativeProcedure {
    name: String,
    num_args: usize,
    no_return: bool,
    callable: Py<PyAny>,
}

impl PythonNativeProcedure {
    pub fn new(name: String, num_args: usize, no_return: bool, callable: Py<PyAny>) -> Self {
        Self {
            name,
            num_args,
            no_return,
            callable,
        }
    }

    /// Leaked &'static str for the trait's name() method. Stored once at
    /// registration time so successive calls return the same pointer.
    fn leaked_name(&self) -> &'static str {
        // Box the name into a heap allocation that lives for the rest of
        // the process. The registry holds an Arc<Self> for the lifetime
        // of the manager; leaking once per registered procedure is OK.
        Box::leak(self.name.clone().into_boxed_str())
    }
}

impl NativeSimProcedure for PythonNativeProcedure {
    fn name(&self) -> &'static str {
        self.leaked_name()
    }

    fn num_args(&self) -> usize {
        self.num_args
    }

    fn no_return(&self) -> bool {
        self.no_return
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // Extract concrete arg values. Symbolic args fall back to Python.
        let mut concrete_args: Vec<u64> = Vec::with_capacity(args.len());
        for (i, arg) in args.iter().enumerate() {
            let v = extract_concrete_arg(arg, &format!("arg{}", i))?;
            concrete_args.push(v);
        }

        let arch_bits = state.arch().bits();

        Python::attach(|py| -> Result<Option<RustBV>, ProcedureError> {
            let result = self.callable.call1(py, (concrete_args,)).map_err(|e| {
                ProcedureError::Other(format!("python procedure '{}' raised: {}", self.name, e))
            })?;

            if result.is_none(py) {
                return Ok(None);
            }

            let ret_int: u64 = result.extract(py).map_err(|e| {
                ProcedureError::Other(format!(
                    "python procedure '{}' returned non-int: {}",
                    self.name, e
                ))
            })?;

            Ok(Some(RustBV::concrete(ret_int as u128, arch_bits)))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procedures::NativeProcedureRegistry;

    #[test]
    fn test_register_python_procedure() {
        pyo3::prepare_freethreaded_python();
        Python::attach(|py| {
            // Build a Python lambda that returns 42.
            let locals = pyo3::types::PyDict::new(py);
            py.run(
                std::ffi::CString::new("f = lambda args: 42")
                    .unwrap()
                    .as_c_str(),
                None,
                Some(&locals),
            )
            .unwrap();
            let callable: Py<PyAny> = locals.get_item("f").unwrap().unwrap().unbind();

            let proc = PythonNativeProcedure::new("py_meaning".to_string(), 0, false, callable);
            assert_eq!(proc.name(), "py_meaning");
            assert_eq!(proc.num_args(), 0);
            assert!(!proc.no_return());

            let mut state = RustSimState::new("amd64").unwrap();
            let result = proc.call(&mut state, &[]).unwrap();
            assert_eq!(result.unwrap().as_u64(), Some(42));
        });
    }

    #[test]
    fn test_python_procedure_symbolic_falls_back() {
        pyo3::prepare_freethreaded_python();
        Python::attach(|py| {
            let locals = pyo3::types::PyDict::new(py);
            py.run(
                std::ffi::CString::new("f = lambda args: 0")
                    .unwrap()
                    .as_c_str(),
                None,
                Some(&locals),
            )
            .unwrap();
            let callable: Py<PyAny> = locals.get_item("f").unwrap().unwrap().unbind();

            let proc = PythonNativeProcedure::new("py_sym".to_string(), 1, false, callable);

            let mut state = RustSimState::new("amd64").unwrap();
            let ctx = state.solver().borrow();
            let sym = RustBV::symbolic(&ctx, "x", 64);
            drop(ctx);
            let result = proc.call(&mut state, &[sym]);
            assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
        });
    }

    #[test]
    fn test_python_procedure_returns_none() {
        pyo3::prepare_freethreaded_python();
        Python::attach(|py| {
            let locals = pyo3::types::PyDict::new(py);
            py.run(
                std::ffi::CString::new("f = lambda args: None")
                    .unwrap()
                    .as_c_str(),
                None,
                Some(&locals),
            )
            .unwrap();
            let callable: Py<PyAny> = locals.get_item("f").unwrap().unwrap().unbind();

            let proc = PythonNativeProcedure::new("py_void".to_string(), 0, false, callable);

            let mut state = RustSimState::new("amd64").unwrap();
            let result = proc.call(&mut state, &[]).unwrap();
            assert!(result.is_none());
        });
    }

    #[test]
    fn test_register_with_registry() {
        pyo3::prepare_freethreaded_python();
        Python::attach(|py| {
            let locals = pyo3::types::PyDict::new(py);
            py.run(
                std::ffi::CString::new("f = lambda args: args[0] * 2 if args else 0")
                    .unwrap()
                    .as_c_str(),
                None,
                Some(&locals),
            )
            .unwrap();
            let callable: Py<PyAny> = locals.get_item("f").unwrap().unwrap().unbind();

            let mut registry = NativeProcedureRegistry::empty();
            let proc = std::sync::Arc::new(PythonNativeProcedure::new(
                "double".to_string(),
                1,
                false,
                callable,
            ));
            registry.register(proc);

            assert!(registry.has_native("double"));
            let p = registry.get("double").unwrap().clone();

            let mut state = RustSimState::new("amd64").unwrap();
            let result = p.call(&mut state, &[RustBV::concrete(21u128, 64)]).unwrap();
            assert_eq!(result.unwrap().as_u64(), Some(42));
        });
    }
}
