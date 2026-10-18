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
//! memo in the bridge is `z3_lower`'s, which holds a node's Z3 lowering only
//! while the node lives.
//!
//! ## Python-side counterparts
//!
//! The Python `RustExplorationManager` maintains an orthogonal set of
//! caches (in-memory `_init_cache`, disk init cache, `_mem_cache`,
//! `_state_cache`). Per-state metadata is NOT among them — it lives on
//! the Rust `RustSimState` and is freed when Rust drops the state (see
//! `RustStateCacheMixin::_cleanup_symbolic_pages_cache`). None of these
//! are documented here; see the cross-mixin invariants **I3**, **I4**,
//! **I5**, **I6** in the `angr/exploration/rust_manager.py` module
//! docstring for the cross-FFI rules and the bd memories
//! `invariant-init-cache-plugin-coverage` (what the init cache does and
//! does not carry; the option allow-list is a *mirror*, not additive),
//! `invariant-apply-state-metadata-option-allowlist` (value-changing
//! SimOptions must be on that list),
//! `invariant-init-cache-concrete-input-digest` (argv/env in the key),
//! `disk-init-cache-symbolic-reg-invariant` (the user-symbolic
//! cache-disable gate), `disk-cache-register-filter`,
//! `invariant-state-cache-mirror-id`, and
//! `invariant-lazy-region-auto-map` for the full rationale.
//!
//! `_mem_cache` has no memory of its own (the former citation here named
//! a key that never existed, angr-sqfj8.16): it is **consume-once**, set
//! by `RustExplorationManager._load_init_from_disk_cache`'s caller and
//! taken — then reset to `None` — by `_try_fast_memory_sync` on the
//! first `_sync_memory_to_rust`, so it is a one-shot hand-off, not a
//! cache that can go stale. `invariant-state-id-cache-epoch-audit`
//! records it under that classification alongside the other Python-side
//! caches that are *not* the stale-serve-after-Rust-steps shape.

use pyo3::prelude::*;

mod export;
mod import;
mod z3_lower;

pub(crate) use export::*;
pub(crate) use import::*;
#[cfg(feature = "vex-engine-z3")]
pub(crate) use z3_lower::{lower_bool, lower_bv, lower_claripy_ast};

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
/// `#[non_exhaustive]` per angr-irwe: minor versions may add new
/// variants; intra-crate matches must include a wildcard arm.
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
    /// visible to CPython's own recursion-limit counter since neither
    /// `claripy_to_rustbv` nor `rustbv_to_claripy_memo` is
    /// `#[pyfunction]`-wrapped).
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

/// Extract an integer value from a Python object.
/// Handles both regular ints and large ints.
fn extract_int_value(obj: Bound<'_, PyAny>) -> Result<u128, BridgeError> {
    // Try extracting as i64 first (fast path)
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(v as u128);
    }

    // Try extracting as u64
    if let Ok(v) = obj.extract::<u64>() {
        return Ok(v as u128);
    }

    // Try extracting as u128
    if let Ok(v) = obj.extract::<u128>() {
        return Ok(v);
    }

    // For larger values, use Python's int.to_bytes. Propagate an extract
    // failure rather than assuming some width: guessing here would either
    // truncate a wide value or spuriously trip the >128 rejection below, and
    // both are the silently-wrong-answer outcome the angr-cxw7 comment exists
    // to prevent (angr-sqfj8.24).
    let bit_length: usize = obj.call_method0("bit_length")?.extract()?;

    // RustBV::Concrete stores its value in a u128, so any integer that needs
    // more than 128 significant bits cannot be represented exactly. Silently
    // masking to the low 128 bits (the old `clamp(1, 16)` behavior) produced a
    // wrong concrete value, e.g. `claripy.BVV(1 << 200, 256)` imported as 0.
    // Error loudly instead so callers don't operate on truncated data
    // (angr-cxw7). Wide *symbolic* BVs still work via the Concat path; this
    // only rejects wide concrete literals.
    if bit_length > 128 {
        return Err(BridgeError::InvalidArgs(format!(
            "concrete integer needs {bit_length} bits but RustBV::Concrete is capped at 128; \
             wide concrete BVV import is unsupported"
        )));
    }
    let byte_length = bit_length.div_ceil(8);
    let byte_length = byte_length.clamp(1, 16);

    let bytes_obj = obj.call_method1("to_bytes", (byte_length, "little"))?;
    let bytes: Vec<u8> = bytes_obj.extract()?;

    // `int.to_bytes(byte_length, ..)` returns exactly `byte_length` bytes and
    // `byte_length <= 16` follows from the `bit_length > 128` rejection above,
    // so this is unreachable for a well-behaved `int`. Check it anyway rather
    // than skipping the excess bytes inside the loop: `obj` is caller-supplied
    // Python, so a `to_bytes` override can hand back a wider buffer, and
    // masking that to the low 128 bits is the silently-wrong-answer outcome
    // the `bit_length > 128` rejection exists to prevent (angr-cxw7).
    if bytes.len() > 16 {
        return Err(BridgeError::InvalidArgs(format!(
            "to_bytes returned {} bytes for a {bit_length}-bit integer; \
             RustBV::Concrete holds at most 16",
            bytes.len()
        )));
    }

    let mut value: u128 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        value |= (b as u128) << (i * 8);
    }

    Ok(value)
}

/// Check if a Python object is a claripy AST.
///
/// SILENT(cat-a): a `hasattr` that itself errors (a `__getattr__` that raises)
/// is treated as "attribute absent", i.e. "not an AST". This is a dispatch
/// gate only — both call sites (`load_from_callback` in `interpreter/mod.rs`
/// and `try_convert_symbolic_value` in `interpreter/expressions_loads.rs`) answer
/// `false` by falling through to a fresh symbolic BV, which is exactly what a
/// failed
/// `claripy_to_rustbv` on such an object would have produced anyway. Nothing
/// downstream can observe the difference, so there is no error to propagate.
pub(crate) fn is_claripy_ast(obj: &Bound<'_, PyAny>) -> bool {
    obj.hasattr("op").unwrap_or(false) && obj.hasattr("args").unwrap_or(false)
}

test_submod!("../claripy_bridge_tests.rs" => tests);
