//! Python-facing VEX execution engine.
//!
//! This module provides the PyO3 bindings for the Rust VEX execution engine,
//! allowing it to be used as an alternative engine in angr.

use pyo3::prelude::*;

/// Register the vex_engine submodule and its classes.
pub fn vex_engine(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes will be registered here as they are ported from rust-engine-v2.
    // e.g. m.add_class::<RustSimState>()?;
    Ok(())
}
