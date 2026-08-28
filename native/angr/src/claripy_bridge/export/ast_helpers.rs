//! Python-side helpers for the RustBV -> claripy export walk.
//!
//! Split out of `export.rs` (angr-5mnx3.11): the leaf helpers that build or
//! repair a single claripy object — type sniffing, int/bool wrapping, `Bool` ->
//! `BV(1)` coercion, concrete-value `BVV` construction and the sound
//! clz/ctz/popcount encoding. All of them need a live interpreter, which is what
//! separates them from the pure decision seams in [`super::width_decisions`];
//! the recursive `BVOp`-dispatch walk that calls them stays in `export.rs`.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt};

use super::width_decisions::ConcreteBvvEncoding;

/// Name of `obj`'s Python type, or `"unknown"` if the interpreter refuses to
/// answer.
///
/// The type name drives control flow in [`ensure_claripy_ast`] (`"bool"` /
/// `"int"`) and [`coerce_bool_to_bv1`] (`"Bool"`), so the failure fallback has
/// to be a string that matches none of those arms — `"unknown"` is that
/// sentinel, and the callers treat it as "not one of the shapes I handle".
pub(super) fn py_type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map_or_else(|_| "unknown".to_string(), |n| n.to_string())
}

/// Ensure a `Py<PyAny>` is a claripy AST, wrapping ints/bools if needed.
///
/// This is a defensive function to handle cases where a Python int or bool
/// might be returned from cache or operations instead of a proper claripy AST.
/// Operations like Extract require claripy ASTs and will fail with
/// "'int' object has no attribute 'length'" if passed an int.
pub(super) fn ensure_claripy_ast(
    py: Python<'_>,
    obj: &Py<PyAny>,
    claripy_mod: &Bound<'_, PyAny>,
    width_hint: Option<u32>,
) -> PyResult<Py<PyAny>> {
    let bound = obj.bind(py);

    // Check if it's already a claripy AST by checking for 'op' attribute
    let missing_op_attr = match bound.hasattr("op") {
        Ok(true) => {
            return Ok(obj.clone_ref(py));
        }
        Ok(false) => true,
        // SILENT(cat-a): a `hasattr` that itself errors (a `__getattr__` that
        // raises) answers `false` here, but this flag only gates the debug log
        // below — it changes no control flow, and claiming "missing 'op' attr"
        // for an object we could not interrogate would be the misleading half.
        // The object still falls through to the type dispatch and, being
        // neither `bool` nor `int`, lands on the tagged unknown-type fallback
        // at the end of this function, which warns on its own (angr-sqfj8.23).
        Err(e) => {
            log::warn!("ensure_claripy_ast: hasattr('op') failed: {e}");
            false
        }
    };

    // Check the actual Python type to distinguish bool from int.
    // IMPORTANT: `bool` is a subclass of `int` in Python, and `extract::<bool>()`
    // happily succeeds for a plain `int`, so a bare extract cannot tell the two
    // apart. The discrimination is therefore done on the *type name* below —
    // each branch gates on `type_name == "bool"` / `== "int"` first and only
    // then extracts, so the extract is a value unpack, never the type test.
    // The name comparison is exact, so an `int` subclass (name != "int") falls
    // through to the tagged unknown-type fallback at the end rather than being
    // silently wrapped as a BVV.
    let type_name = py_type_name(bound);
    if missing_op_attr {
        log::debug!("ensure_claripy_ast: object {type_name} missing 'op' attr, wrapping");
    }

    // Check if it's exactly a Python bool (not an int that happens to be 0 or 1)
    if type_name == "bool"
        && let Ok(bool_val) = bound.extract::<bool>()
    {
        // If width hint is provided, wrap as BVV (for use in BV operations)
        // Otherwise wrap as BoolV (for use in Bool operations)
        if let Some(w) = width_hint {
            let val: i64 = i64::from(bool_val);
            return claripy_mod
                .call_method1("BVV", (val, w))
                .map(std::convert::Into::into);
        }
        return claripy_mod
            .call_method1("BoolV", (bool_val,))
            .map(std::convert::Into::into);
    }

    // If it's an int, wrap in BVV with the provided width hint
    // Try i128 first for larger values, then fall back to i64
    if type_name == "int" {
        // SILENT(cat-b): a bare Python int carries no width, so an absent
        // `width_hint` is defaulted to 64 rather than refused. Both call sites
        // (`rustbv_to_claripy_memo`'s symbolic-cache-hit arm and its
        // `RustBV::Expression` operand loop) always pass a `Some` derived from
        // the corresponding `RustBV::width()`, so the default is currently
        // unreachable; if a future caller omits the hint, a narrower operand
        // widens to 64 bits instead of erroring (angr-sqfj8.23).
        let width = width_hint.unwrap_or(64);
        // Try to extract as i128 for larger values
        if let Ok(int_val) = bound.extract::<i128>() {
            log::debug!("ensure_claripy_ast: wrapping int {int_val} in BVV with width {width}");
            // For values that fit in i64, use that (more compatible)
            if int_val >= i64::MIN as i128 && int_val <= i64::MAX as i128 {
                return claripy_mod
                    .call_method1("BVV", (int_val as i64, width))
                    .map(std::convert::Into::into);
            } else {
                // For larger values, pass as Python int directly
                return claripy_mod
                    .call_method1("BVV", (&bound, width))
                    .map(std::convert::Into::into);
            }
        }
        // Fallback: pass the Python object directly and let claripy handle it
        log::debug!("ensure_claripy_ast: wrapping large int in BVV with width {width}");
        return claripy_mod
            .call_method1("BVV", (&bound, width))
            .map(std::convert::Into::into);
    }

    // SILENT(cat-c): anything that is neither an AST (no `op` attr) nor a
    // `bool`/`int` we know how to wrap is handed back untouched. Both callers
    // feed the result straight into claripy op construction, so a genuinely
    // wrong object (a `str`, a `float`, `None`) becomes either a `TypeError`
    // or the `NotImplemented` singleton one frame later rather than a raised
    // error here — the wrong-answer-risk class. Not upgraded to an `Err`
    // because the reachable case is the opposite one: an object whose
    // `hasattr('op')` raised above is very likely a real AST, and returning it
    // unchanged is correct. Warns unconditionally so the two are
    // distinguishable in a log (angr-sqfj8.23).
    log::warn!("ensure_claripy_ast: unknown type {type_name}, returning as-is");
    Ok(obj.clone_ref(py))
}

/// Convert a claripy `Bool` AST to a 1-bit BV via `If(cond, BVV(1,1), BVV(0,1))`.
///
/// claripy's `ZeroExt`/`SignExt`/`Extract` and the width-matching binary-op
/// path all require a BV operand, not a Bool. This is the single canonical
/// coercion (previously re-implemented inline four times — angr-c3rd,
/// angr-n0irt.11).
pub(super) fn bool_to_bv1<'py>(
    claripy_mod: &Bound<'py, PyAny>,
    arg: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let one = claripy_mod.call_method1("BVV", (1i64, 1u32))?;
    let zero = claripy_mod.call_method1("BVV", (0i64, 1u32))?;
    claripy_mod.call_method1("If", (arg, one, zero))
}

/// If `arg` is a claripy `Bool`, coerce it to a 1-bit BV; otherwise return it
/// unchanged. Used by the `ZeroExt`/`SignExt`/`Extract` arms, which reject a
/// Bool operand.
pub(super) fn coerce_bool_to_bv1<'py>(
    claripy_mod: &Bound<'py, PyAny>,
    arg: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if py_type_name(arg) == "Bool" {
        bool_to_bv1(claripy_mod, arg)
    } else {
        Ok(arg.clone())
    }
}

/// Export a concrete `u128` value of the given bit `width` as a `claripy.BVV`.
///
/// Shared by the `Concrete` and `Constrained` arms of [`super::rustbv_to_claripy_memo`]
/// so the two stay in lockstep — they drifted once (angr-ph300.52): the
/// `Constrained` arm was missing the `width / 8 <= 16` clause, so a byte-aligned
/// width > 128 (e.g. 192) built a 16-byte `PyBytes` and handed it to
/// `BVV(bytes, 192)`, which raises `ClaripyValueError` for the string/size
/// mismatch, while the `Concrete` twin succeeded.
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

/// angr-acoq: build a sound claripy encoding for a symbolic clz/ctz/popcount
/// whose single operand has already been converted to `operand` (a claripy BV
/// of the same `width`). The result width equals `width`, matching the
/// concrete fast path (`BVV(result, width)`). Only valid for `width <= 64`.
///
/// Encodings (all tied to `operand`, so Python-side eval stays consistent with
/// the Rust engine's value):
///   - clz: nested `If(bit[w-1-i]==1, i, ...)` from LSB to MSB so the MSB test
///     is outermost; default `w` when no bit is set.
///   - ctz: nested `If(bit[i]==1, i, ...)` from MSB to LSB so the LSB test is
///     outermost; default `w` when no bit is set.
///   - popcount: sum of `ZeroExt(w-1, bit[i])` over all `i`.
pub(super) fn build_sound_bitcount(
    _py: Python<'_>,
    claripy_mod: &Bound<'_, PyAny>,
    op: &crate::symbolic::BVOp,
    operand: &Bound<'_, PyAny>,
    width: u32,
) -> PyResult<Py<PyAny>> {
    use crate::symbolic::BVOp;

    let extract_bit = |pos: u32| -> PyResult<Bound<'_, PyAny>> {
        claripy_mod.call_method1("Extract", (pos, pos, operand))
    };

    match op {
        BVOp::Popcount => {
            // sum of zero-extended individual bits; result fits in `width`.
            let mut acc: Bound<'_, PyAny> = if width > 1 {
                claripy_mod.call_method1("ZeroExt", (width - 1, extract_bit(0)?))?
            } else {
                extract_bit(0)?
            };
            for pos in 1..width {
                let ext = claripy_mod.call_method1("ZeroExt", (width - 1, extract_bit(pos)?))?;
                acc = acc.call_method1("__add__", (ext,))?;
            }
            Ok(acc.into())
        }
        BVOp::Clz | BVOp::Ctz => {
            let one = claripy_mod.call_method1("BVV", (1i64, 1u32))?;
            let mut result = claripy_mod.call_method1("BVV", (width as i64, width))?;
            // For clz, iterate positions LSB->MSB so the MSB test is outermost.
            // For ctz, iterate MSB->LSB so the LSB test is outermost.
            let positions: Vec<u32> = match op {
                BVOp::Clz => (0..width).collect(),
                _ => (0..width).rev().collect(),
            };
            for pos in positions {
                let cond = extract_bit(pos)?.call_method1("__eq__", (&one,))?;
                let leading_count = match op {
                    BVOp::Clz => width - 1 - pos,
                    _ => pos, // ctz: trailing zeros == bit index of lowest set bit
                };
                let val = claripy_mod.call_method1("BVV", (leading_count as i64, width))?;
                result = claripy_mod.call_method1("If", (cond, val, result))?;
            }
            Ok(result.into())
        }
        _ => unreachable!("build_sound_bitcount only handles clz/ctz/popcount"),
    }
}
