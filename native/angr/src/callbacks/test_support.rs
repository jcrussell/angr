//! Shared test-support helpers for the `callbacks/` unit-test modules
//! (angr-5mnx3.7).
//!
//! `dispatch_tests.rs` and `inspect_tests.rs` both drive their subject by
//! `py.run`-ing a small Python snippet that defines callback functions plus a
//! recorder list, then handing those objects to `PythonCallbacks` setters and
//! asserting on what the recorder saw. The three helpers below are that
//! plumbing; they were duplicated byte-for-byte in both files until this
//! module existed, so a fix to one copy silently left the other behind.
//!
//! Precedent for the shape: `vex/ops/test_helpers.rs`. Helpers are
//! `pub(super)` — i.e. visible throughout the `callbacks` subtree — because
//! the two consumers sit at *different* depths: `dispatch_tests` is a child
//! of `dispatch`, `inspect_tests` a child of `inspect`, while this module is
//! a direct child of `callbacks`. Both reach it as
//! `crate::callbacks::test_support`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Execute `src` in a fresh globals dict and hand it back so tests can read
/// the recorder lists the snippet defines.
pub(super) fn defs<'py>(py: Python<'py>, src: &std::ffi::CStr) -> Bound<'py, PyDict> {
    let globals = PyDict::new(py);
    py.run(src, Some(&globals), None)
        .expect("define test callbacks");
    globals
}

/// Pull a named object out of a `defs()` globals dict as an owned `Py<PyAny>`.
pub(super) fn obj(globals: &Bound<'_, PyDict>, name: &str) -> Py<PyAny> {
    globals
        .get_item(name)
        .unwrap()
        .unwrap_or_else(|| panic!("{name} not defined"))
        .unbind()
}

/// Read a recorder list defined by a `defs()` snippet.
pub(super) fn recorder<'py>(globals: &Bound<'py, PyDict>, name: &str) -> Bound<'py, PyList> {
    globals
        .get_item(name)
        .unwrap()
        .unwrap_or_else(|| panic!("{name} not defined"))
        .cast_into::<PyList>()
        .expect("recorder must be a list")
}
