//! Python-side helper for the RustBV -> claripy export: concrete-value `BVV`
//! construction, which needs a live interpreter — what separates it from the
//! pure decision seams in [`super::width_decisions`].

use pyo3::prelude::*;
use pyo3::types::PyInt;

use super::width_decisions::ConcreteBvvEncoding;

/// Export a concrete `u128` value of the given bit `width` as a `claripy.BVV`.
///
/// - `width <= 64`: pass the value as an `i64`.
/// - otherwise: pass a Python int, which `BVV(int, width)` zero-pads correctly
///   for any width.
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
        ConcreteBvvEncoding::PyIntWide => {
            // `BVV` does not reduce a non-negative int mod 2^width, so drop the
            // bits above `width` here (the old `bytes` arm sliced them off).
            let masked = if width < 128 {
                value & ((1u128 << width) - 1)
            } else {
                value
            };
            let py_int = PyInt::new(py, masked);
            claripy_mod
                .call_method1("BVV", (py_int, width))
                .map(std::convert::Into::into)
        }
    }
}
