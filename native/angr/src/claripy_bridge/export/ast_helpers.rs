//! Python-side helper for the RustBV -> claripy export: concrete-value `BVV`
//! construction, which needs a live interpreter — what separates it from the
//! pure decision seams in [`super::width_decisions`].

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt};

use super::width_decisions::ConcreteBvvEncoding;

/// Export a concrete `u128` value of the given bit `width` as a `claripy.BVV`.
///
/// - `width <= 64`: pass the value as an `i64`.
/// - byte-aligned and `width / 8 <= 16` (fits in the u128's 16 bytes): pass the
///   exact big-endian bytes.
/// - otherwise (non-byte-aligned OR width > 128): pass a Python int, which
///   `BVV(int, width)` zero-pads correctly for any width.
pub(super) fn concrete_value_to_bvv(
    py: Python<'_>,
    claripy_mod: &Bound<'_, PyAny>,
    value: u128,
    width: u32,
) -> PyResult<Py<PyAny>> {
    match ConcreteBvvEncoding::for_width(width) {
        ConcreteBvvEncoding::Int64 => claripy_mod
            .call_method1("BVV", (value as i64, width))
            .map(std::convert::Into::into),
        ConcreteBvvEncoding::Bytes(byte_count) => {
            let bytes = value.to_be_bytes();
            let start = bytes.len().saturating_sub(byte_count);
            let py_bytes = PyBytes::new(py, &bytes[start..]);
            claripy_mod
                .call_method1("BVV", (py_bytes, width))
                .map(std::convert::Into::into)
        }
        ConcreteBvvEncoding::PyIntWide => {
            let py_int = PyInt::new(py, value);
            claripy_mod
                .call_method1("BVV", (py_int, width))
                .map(std::convert::Into::into)
        }
    }
}
