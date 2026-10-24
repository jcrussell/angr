//! The PyO3 face of [`crate::state`]: the `RustSimState` and
//! `ExplorationStateSnapshot` Python classes.
//!
//! `state` itself holds no PyO3 code (angr-mx00p.75); everything Python sees
//! of a state is a thin wrapper here, registered in `angr.rustylib.vex_engine`
//! by `engine::vex_engine`.
//!
//! - `pymethods` — [`PyRustSimState`]'s `#[pymethods]`, including the
//!   boolean flag pairs `state::options`'s flag list declares.
//! - `snapshot` — [`ExplorationStateSnapshot`], the pyclass over a
//!   [`StateExport`](crate::state::StateExport).

mod pymethods;
mod snapshot;

pub(crate) use snapshot::ExplorationStateSnapshot;

use pyo3::prelude::*;

use crate::state::RustSimState;

/// Python-facing wrapper for RustSimState.
///
/// This provides the PyO3 interface for creating and manipulating
/// Rust-native simulation states from Python.
#[pyclass(name = "RustSimState", unsendable)]
pub struct PyRustSimState {
    inner: RustSimState,
}

impl PyRustSimState {
    /// Get access to the inner state (for Rust-side use).
    pub fn inner(&self) -> &RustSimState {
        &self.inner
    }
}
