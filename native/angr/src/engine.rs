//! Python-facing VEX execution engine.
//!
//! This module provides the PyO3 bindings for the Rust VEX execution engine,
//! allowing it to be used as an alternative engine in angr.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Deserialize a pyvex IRSB from JSON bytes into a Rust IRSB.
/// Returns a Python dict with the parsed IRSB fields.
#[pyfunction]
fn deserialize_irsb(py: Python<'_>, json_bytes: &Bound<'_, PyBytes>) -> PyResult<PyObject> {
    let json_str = std::str::from_utf8(json_bytes.as_bytes())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid UTF-8: {e}")))?;
    let irsb = crate::vex::deserialize_irsb(json_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Deserialize error: {e}")))?;

    // Return basic IRSB info as a dict for verification
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("addr", irsb.addr)?;
    dict.set_item("size", irsb.size())?;
    dict.set_item("num_stmts", irsb.statements.len())?;
    dict.set_item("jumpkind", format!("{:?}", irsb.jumpkind))?;
    Ok(dict.into())
}

/// Register the vex_engine submodule and its classes/functions.
pub fn vex_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(deserialize_irsb, m)?)?;
    // More classes will be registered as they are ported:
    // m.add_class::<RustVEXEngine>()?;
    // m.add_class::<RustSolverContext>()?;
    // etc.
    Ok(())
}
