//! The mid-step [`InspectView`] a BP action reads through, held for the
//! length of one inspect fire.
//!
//! The interpreter publishes the view through
//! `EngineHost::publish_inspect_view` only while `PythonCallbacks`'s
//! `inspect_wants_view` flag is set (a registered BP has an action). It is
//! kept in a thread-local, not on the manager: the manager's PyCell is
//! mutably borrowed for the whole step, so these pyfunctions answer without
//! touching it (angr-mx00p.132.53).

use std::cell::RefCell;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::claripy_bridge::{claripy_module, rustbv_to_claripy};
use crate::interpreter::InspectView;
use crate::solver::RustSolverContext;
use crate::symbolic::RustBV;

thread_local! {
    static INSPECT_VIEW: RefCell<Option<InspectView>> = const { RefCell::new(None) };
}

/// Hold `view` for the fire that follows, or drop it (`None`).
pub(crate) fn publish(view: Option<InspectView>) {
    INSPECT_VIEW.with(|cell| *cell.borrow_mut() = view);
}

/// Run `read` against the published view; `None` when no fire is in flight.
fn with_view<R>(read: impl FnOnce(&InspectView) -> R) -> Option<R> {
    INSPECT_VIEW.with(|cell| cell.borrow().as_ref().map(read))
}

fn to_ast(py: Python<'_>, value: &RustBV) -> PyResult<Py<PyAny>> {
    rustbv_to_claripy(py, value, claripy_module(py)?.as_any())
}

fn no_view() -> PyErr {
    PyRuntimeError::new_err("no inspect view is published: not inside a BP action fired mid-step")
}

/// `(block_addr, insn_addr)` of the published view, or `None` outside a fire.
#[pyfunction]
pub(crate) fn inspect_view_addrs() -> Option<(u64, u64)> {
    with_view(|view| (view.block_addr(), view.insn_addr()))
}

/// The register at `offset`, `size` bytes wide, as a claripy AST.
#[pyfunction]
pub(crate) fn inspect_view_register(py: Python<'_>, offset: u32, size: u32) -> PyResult<Py<PyAny>> {
    let value = with_view(|view| view.register(offset, size)).ok_or_else(no_view)?;
    to_ast(py, &value)
}

/// The `size`-byte value at `addr` as a claripy AST, or `None` when the view
/// cannot serve it whole (`InspectView::load`).
#[pyfunction]
pub(crate) fn inspect_view_load(
    py: Python<'_>,
    addr: u64,
    size: u32,
) -> PyResult<Option<Py<PyAny>>> {
    let value = with_view(|view| view.load(addr, size)).ok_or_else(no_view)?;
    value.map(|v| to_ast(py, &v)).transpose()
}

/// A fork of the stepping state's solver, the context a BP action's
/// `state.solver` queries run against mid-step: the manager's
/// `fork_state_solver` would need the borrowed manager (angr-vmi2l.338).
#[pyfunction]
pub(crate) fn inspect_view_fork_solver() -> PyResult<RustSolverContext> {
    with_view(|view| RustSolverContext::from_sym_context(view.solver().fork())).ok_or_else(no_view)
}

/// The concrete bytes the view holds from `addr` on, at most `max`
/// (`InspectView::concrete_bytes`).
#[pyfunction]
pub(crate) fn inspect_view_bytes(addr: u64, max: usize) -> PyResult<Vec<u8>> {
    with_view(|view| view.concrete_bytes(addr, max)).ok_or_else(no_view)
}

/// Register the view pyfunctions on the `vex_engine` module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(inspect_view_addrs, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(inspect_view_register, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(inspect_view_load, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(inspect_view_fork_solver, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(inspect_view_bytes, m)?)?;
    Ok(())
}
