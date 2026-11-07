//! Bridge between claripy ASTs and RustBV.
//!
//! This module provides bidirectional conversion between Python claripy ASTs
//! and Rust RustBV values, enabling native symbolic execution in Rust.
//!
//! ## Identity Preservation
//!
//! Symbolic identity is structural: a claripy AST's clarirs nodes are what a
//! `RustBV` holds (`RustBV::Ast`), and export hands Python the claripy object
//! over that same node. A leaf is the clarirs variable of its name, so
//! `BVS("x", 32)` imported from Python and the leaf Rust builds from `"x"` are
//! one node. clarirs interning is the only identity table.
//!
//! The import keeps no cache: it walks the claripy AST's clarirs nodes, and
//! repeated imports of a node build the same interned terms (measured as no
//! slower than the former claripy-hash `AST_CACHE`, angr-mx00p.14). The one
//! memo here is `symbolic::z3_lower`'s, which holds a node's Z3 lowering only
//! while the node lives.
//!
//! ## Python-side counterparts
//!
//! The Python `RustExplorationManager` maintains an orthogonal set of
//! caches (`_state_cache`, the per-loader `_loader_pages_cache`).
//! Per-state metadata is NOT among them — it lives on the Rust
//! `RustSimState` and is freed when Rust drops the state. None of these
//! are documented here; see the cross-mixin invariants **I5** and **I6**
//! in the `angr/exploration/rust_manager.py` module docstring for the
//! cross-FFI rules, and the invariants `invariant-state-cache-mirror-id`
//! and `invariant-lazy-region-auto-map` for the full rationale.

use pyo3::prelude::*;

mod export;
mod import;
mod lower;

pub(crate) use export::*;
pub(crate) use import::*;
pub(crate) use lower::lower_claripy_bool;

/// The claripy Python module (clarirs, built into this extension).
///
/// angr has no top-level `claripy` package: claripy is the `claripy` submodule
/// of `angr.rustylib` that `lib.rs::rustylib` registers in `sys.modules` while
/// the extension initializes, and `angr/__init__.py` re-exports it as
/// `angr.claripy`. When it is not registered — a `cargo test` binary never runs
/// that init, and its embedded interpreter cannot import `angr` — build it in
/// place with the same `claripy::claripy` initializer, so every AST this
/// binary exports is one of its own claripy types.
pub(crate) fn claripy_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    const NAME: &str = "angr.rustylib.claripy";
    let modules = py.import("sys")?.getattr("modules")?;
    if let Ok(module) = modules.get_item(NAME) {
        return Ok(module.cast_into::<PyModule>()?);
    }
    let module = PyModule::new(py, NAME)?;
    crate::claripy::claripy(py, &module)?;
    modules.set_item(NAME, &module)?;
    Ok(module)
}

/// Wrap a clarirs AST as a claripy Python object of *this* binary's type.
///
/// Under `cargo test` the embedded interpreter's `angr.rustylib` is the
/// installed extension, whose claripy classes are different type objects from
/// this test binary's — so a test that needs `lower_claripy_ast` to accept its
/// input builds the AST here instead of through `import("angr.rustylib.claripy")`.
#[cfg(test)]
#[allow(clippy::expect_used, reason = "test fixture")]
pub(crate) fn claripy_object<'py>(
    py: Python<'py>,
    ast: clarirs_core::prelude::AstRef<'static>,
) -> Bound<'py, PyAny> {
    crate::claripy::ast::base::Base::from_ast(py, ast)
        .expect("wrap clarirs AST")
        .into_any()
}

/// Error type for claripy bridge operations.
///
/// `#[non_exhaustive]`: minor versions may add new variants (angr-irwe);
/// intra-crate matches must include a wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, thiserror::Error)]
pub enum BridgeError {
    /// Unsupported claripy operation.
    #[error("unsupported claripy op: {0}")]
    UnsupportedOp(String),
    /// Type mismatch.
    #[error("type mismatch: {0}")]
    TypeMismatch(String),
    /// Python error.
    #[error("python error: {0}")]
    PythonError(String),
    /// Invalid arguments.
    #[error("invalid arguments: {0}")]
    InvalidArgs(String),
    /// Recursive AST/Expression-tree descent exceeded the depth guard
    /// (angr-2a3i9). Raised instead of recursing further so a
    /// pathologically deep tree fails with a catchable error rather than
    /// overflowing the native Rust stack (SIGSEGV on the guard page, not
    /// visible to CPython's own recursion-limit counter since
    /// `claripy_to_rustbv` is not `#[pyfunction]`-wrapped).
    #[error("recursion depth limit exceeded: {0}")]
    RecursionLimit(String),
}

impl From<PyErr> for BridgeError {
    fn from(err: PyErr) -> Self {
        BridgeError::PythonError(err.to_string())
    }
}

/// Wording hub for a failed `claripy_to_rustbv` at the PyO3 boundary
/// (angr-12jjk.4).
///
/// `what` names the thing being imported — `"register rax"`,
/// `"memory 0x400080"`, `"content byte 3"` — and is rendered as
/// `AST import failed (<what>): <err>`. Call it from inside the `map_err`
/// closure so the `format!` for a dynamic `what` stays off the success path.
///
/// The `ValueError` class is the one every import site already raised: a
/// bridge failure means the caller handed us an AST this engine cannot
/// represent, which is a bad-argument condition, not an engine fault.
/// Exports use [`ast_export_err`] and `RuntimeError` for the mirror-image
/// reason.
///
/// Both helpers take the error as `impl Display` rather than a concrete type
/// because the two directions disagree: `claripy_to_rustbv` fails with
/// [`BridgeError`], `rustbv_to_claripy` with `PyErr`.
pub(crate) fn ast_import_err(what: &str, e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("AST import failed ({what}): {e}"))
}

/// Wording hub for a failed `rustbv_to_claripy` at the PyO3 boundary
/// (angr-12jjk.4). See [`ast_import_err`] for the `what` convention.
///
/// Raises `RuntimeError`: the caller supplied no AST here, so a failure to
/// build one out of a value the engine itself is holding is an engine fault.
pub(crate) fn ast_export_err(what: &str, e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("AST export failed ({what}): {e}"))
}

test_submod!("../claripy_bridge_tests.rs" => tests);
