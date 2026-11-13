//! [`ConstraintLog`]: the Python handle for a copy of one state's solver
//! constraints, passed from `export_constraint_log` to
//! `import_constraint_log`.

use pyo3::prelude::*;

/// A copy of one state's constraint log, as `export_constraint_log` returns
/// it. Python can hold it, take its `len()` and hand it to
/// `import_constraint_log`, nothing else: it has no constructor, so every
/// log an import sees was exported by a manager, and its entries are
/// constraints some state already holds. That is what the raw-pointer pair
/// it replaces could not promise (angr-az6ru.2). The entries hold no Z3
/// term; the class stays `unsendable` only because nothing needs it to
/// cross threads.
#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pyclass(unsendable, frozen)]
pub struct ConstraintLog(pub(crate) crate::symbolic::ConstraintLog);

#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pymethods]
impl ConstraintLog {
    fn __len__(&self) -> usize {
        self.0.len()
    }
}
