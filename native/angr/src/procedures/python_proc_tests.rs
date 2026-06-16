// Tests for python_proc.rs (PythonNativeProcedure).
// Split from the parent module's #[cfg(test)] block (see angr test-split campaign).

use super::*;
use crate::procedures::NativeProcedureRegistry;

#[test]
fn test_register_python_procedure() {
    Python::initialize();
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
    Python::initialize();
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
    Python::initialize();
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
    Python::initialize();
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
