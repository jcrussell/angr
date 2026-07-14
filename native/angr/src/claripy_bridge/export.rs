//! RustBV -> claripy AST export (the BVOp-dispatch rebuild path).
//!
//! `rustbv_to_claripy` / `rustbv_to_claripy_memo` reconstruct a claripy AST
//! from a `RustBV`, returning the original imported AST verbatim on a cache
//! hit (preserving annotations) and rebuilding from `BVOp + operands`
//! otherwise. `ensure_claripy_ast` and `build_sound_bitcount` are helpers.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt};

use crate::symbolic::RustBV;

use super::cache::{
    get_claripy_ast, get_expression_ast_by_operands, store_claripy_ast_with_info,
    store_expression_ast_by_operands,
};

/// Ensure a Py<PyAny> is a claripy AST, wrapping ints/bools if needed.
///
/// This is a defensive function to handle cases where a Python int or bool
/// might be returned from cache or operations instead of a proper claripy AST.
/// Operations like Extract require claripy ASTs and will fail with
/// "'int' object has no attribute 'length'" if passed an int.
fn ensure_claripy_ast(
    py: Python<'_>,
    obj: &Py<PyAny>,
    claripy_mod: &Bound<'_, PyAny>,
    width_hint: Option<u32>,
) -> PyResult<Py<PyAny>> {
    let bound = obj.bind(py);

    // Check if it's already a claripy AST by checking for 'op' attribute
    match bound.hasattr("op") {
        Ok(true) => {
            return Ok(obj.clone());
        }
        Ok(false) => {
            let type_name = bound
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            log::debug!("ensure_claripy_ast: object {type_name} missing 'op' attr, wrapping");
        }
        Err(e) => {
            log::warn!("ensure_claripy_ast: hasattr('op') failed: {e}");
        }
    }

    // Check the actual Python type to distinguish bool from int
    // IMPORTANT: In Python, bool is a subclass of int, so we must check bool FIRST
    // but use is_instance_of, not extract, because extract::<bool>() succeeds for ints too
    let type_name = bound
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "unknown".to_string());

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

    // Otherwise return as-is and hope for the best
    log::warn!("ensure_claripy_ast: unknown type {type_name}, returning as-is");
    Ok(obj.clone())
}

/// Materialize one `(guard, is_assumed_true)` entry from a `SymContext`'s
/// assumed log as a claripy **boolean** constraint (angr-op0dn.14.4.1).
///
/// The guard is a 1-bit `RustBV`. `rustbv_to_claripy` lowers it to whatever
/// claripy type its op tree implies — a `Bool` for a comparison, but a 1-bit
/// `BV` for e.g. an `Extract`/`Ite`-derived guard. `claripy.Not()` on a BV
/// returns `NotImplemented` rather than raising, so a BV-typed guard used to
/// export as the `NotImplemented` singleton. Compare a BV guard against
/// `BVV(1|0, 1)` instead — the same `guard != 0` semantics the Python engine
/// gives a VEX exit guard — and reserve `Not` for genuinely boolean guards.
///
/// Shared by `_export_state_constraints` and the native `constraints` inspect
/// dispatch so both render an assumed guard identically.
pub fn assumed_guard_to_claripy(
    py: Python<'_>,
    bv: &RustBV,
    claripy_mod: &Bound<'_, PyAny>,
    is_true: bool,
) -> PyResult<Py<PyAny>> {
    let ast = rustbv_to_claripy(py, bv, claripy_mod)?;
    let bound = ast.bind(py);
    // Bool ASTs have `length is None`; BV ASTs carry their bit width.
    let is_bv = bound
        .getattr("length")
        .map(|l| !l.is_none())
        .unwrap_or(false);
    if is_bv {
        let bit = u64::from(is_true);
        let bvv = claripy_mod.call_method1("BVV", (bit, 1u32))?;
        return Ok(bound.call_method1("__eq__", (bvv,))?.unbind());
    }
    if is_true {
        Ok(ast)
    } else {
        Ok(claripy_mod.call_method1("Not", (ast,))?.unbind())
    }
}

/// Convert a RustBV back to a claripy AST.
///
/// This is used when returning symbolic results to Python.
/// For Expression variants, this recursively reconstructs the claripy AST
/// from the operation tree, preserving the original expression structure.
pub fn rustbv_to_claripy(
    py: Python<'_>,
    bv: &RustBV,
    claripy_mod: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    // GIL-work timing (angr-1ilq.7): time the whole export (recursive memo
    // descent + claripy method calls) as one region. This is the non-recursive
    // outer wrapper, so a single guard here covers all of `_memo`; the depth
    // guard prevents double-counting when a dispatch method called us already.
    let _gil =
        crate::gil_profile::GilWorkGuard::enter_as(crate::gil_profile::GilClass::ClaripyExport);

    // Memoize by RustBV pointer identity to dedupe shared subtrees in DAGs.
    // sym-write's symbolic-store ITE chains have ~25 unique Arc-shared
    // subtrees expanded into a 142k-node tree without dedup; converting that
    // takes ~2.7s vs ~tens of ms with memoization.
    let mut memo: HashMap<usize, Py<PyAny>> = HashMap::new();
    rustbv_to_claripy_memo(py, bv, claripy_mod, &mut memo)
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
fn build_sound_bitcount(
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

fn rustbv_to_claripy_memo(
    py: Python<'_>,
    bv: &RustBV,
    claripy_mod: &Bound<'_, PyAny>,
    memo: &mut HashMap<usize, Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    use crate::symbolic::BVOp;

    // Check cache first for Symbolic variants
    // This preserves AST identity across FFI boundary
    if let RustBV::Symbolic { id, width, .. } = bv
        && let Some(cached) = get_claripy_ast(*id)
    {
        // Validate cached value is a claripy AST, not an int
        let cached_valid = ensure_claripy_ast(py, &cached, claripy_mod, Some(*width))?;
        return Ok(cached_valid);
    }

    // For Expression variants imported via claripy_to_rustbv, look up the
    // original AST keyed by the operands Arc pointer. This returns the
    // Python-side AST verbatim, preserving any annotations attached at the
    // Expression level (which are otherwise dropped when we rebuild from
    // BVOp+operands). See angr-ykdq.
    if let RustBV::Expression { operands, .. } = bv {
        let operands_ptr = Arc::as_ptr(operands) as *const () as usize;
        if let Some(cached) = get_expression_ast_by_operands(py, operands_ptr) {
            return Ok(cached);
        }
    }

    // Memoization: only Expression variants are worth caching (the recursive case
    // with potential DAG sharing). For Expression, key by `bv` pointer so two
    // sibling references to the same operand inside a shared `Arc<[RustBV]>` only
    // pay the conversion cost once.
    let memo_key = if matches!(bv, RustBV::Expression { .. }) {
        let k = bv as *const RustBV as usize;
        if let Some(cached) = memo.get(&k) {
            return Ok(cached.clone_ref(py));
        }
        Some(k)
    } else {
        None
    };

    let result: PyResult<Py<PyAny>> = match bv {
        RustBV::Concrete { value, width } => {
            // Create claripy.BVV(value, width)
            if *width <= 64 {
                claripy_mod
                    .call_method1("BVV", (*value as i64, *width))
                    .map(std::convert::Into::into)
            } else if *width % 8 == 0 && *width as usize / 8 <= 16 {
                // Byte-aligned and fits in u128 (16 bytes): use bytes for
                // exact representation. Wider Concrete widths cannot exceed
                // u128 value but can declare a larger bit width — fall
                // through to PyInt which pads correctly.
                let byte_count = *width as usize / 8;
                let bytes = value.to_be_bytes();
                let start = bytes.len().saturating_sub(byte_count);
                let py_bytes = PyBytes::new(py, &bytes[start..]);
                claripy_mod
                    .call_method1("BVV", (py_bytes, *width))
                    .map(std::convert::Into::into)
            } else {
                // Non-byte-aligned OR width > 128: use Python int to avoid
                // string/size mismatch. claripy.BVV(int_value, width) works
                // for any width and zero-pads high bits.
                let py_int = PyInt::new(py, *value);
                claripy_mod
                    .call_method1("BVV", (py_int, *width))
                    .map(std::convert::Into::into)
            }
        }
        RustBV::Symbolic {
            id, name, width, ..
        } => {
            // Cache was already checked above, so this is a symbol minted purely
            // inside Rust (e.g. a `stdin_N_i` byte from the native read proc).
            //
            // Register the fresh claripy BVS against THIS symbol id under the
            // RUST name (angr-izov2). `claripy.BVS(name, w)` without
            // `explicit_name` renames the symbol to `name_<counter>_<w>`, so
            // without a registration neither the hash nor the name+width lookup
            // in `claripy_to_rustbv` can recognise the AST on the way back: the
            // importer mints a brand new, unconstrained Rust symbol and every
            // constraint carried by the original leaf silently stops binding.
            // That is exactly what a Python bounce (SimProcedure hook) does to
            // a symbolic value it returns. Registering by hash also pins the
            // exported AST, so repeated exports of this symbol hand Python the
            // same claripy object.
            let ast: Py<PyAny> = claripy_mod
                .call_method1("BVS", (&**name, *width))
                .map(Py::<PyAny>::from)?;
            let py_hash = ast.bind(py).hash()? as i64;
            store_claripy_ast_with_info(py_hash, *id, name, *width, ast.clone_ref(py));
            Ok(ast)
        }
        RustBV::Constrained { value, width, .. } => {
            // For constrained values, return the concrete value
            if *width <= 64 {
                claripy_mod
                    .call_method1("BVV", (*value as i64, *width))
                    .map(std::convert::Into::into)
            } else if *width % 8 == 0 {
                let byte_count = *width as usize / 8;
                let bytes = value.to_be_bytes();
                let start = bytes.len().saturating_sub(byte_count);
                let py_bytes = PyBytes::new(py, &bytes[start..]);
                claripy_mod
                    .call_method1("BVV", (py_bytes, *width))
                    .map(std::convert::Into::into)
            } else {
                // Non-byte-aligned: use Python int
                let py_int = PyInt::new(py, *value);
                claripy_mod
                    .call_method1("BVV", (py_int, *width))
                    .map(std::convert::Into::into)
            }
        }
        RustBV::Expression { op, operands, .. } => {
            // Recursively convert operands to claripy ASTs
            let raw_args: Vec<Py<PyAny>> = operands
                .iter()
                .map(|operand| rustbv_to_claripy_memo(py, operand, claripy_mod, memo))
                .collect::<Result<_, _>>()?;

            // Validate all args to ensure they're claripy ASTs with correct widths
            let args: Vec<Py<PyAny>> = raw_args
                .iter()
                .enumerate()
                .map(|(i, arg)| {
                    let width = operands.get(i).map(super::super::symbolic::RustBV::width);
                    ensure_claripy_ast(py, arg, claripy_mod, width)
                })
                .collect::<Result<Vec<_>, _>>()?;

            // For binary ops, ensure operand widths match (resize if needed)
            let args = if args.len() == 2 && !matches!(op, BVOp::Extract(_, _) | BVOp::Concat) {
                let a0 = args[0].bind(py);
                let a1 = args[1].bind(py);
                let w0: Option<u32> = a0.getattr("length").ok().and_then(|l| l.extract().ok());
                let w1: Option<u32> = a1.getattr("length").ok().and_then(|l| l.extract().ok());
                // Handle Bool operands (length=None): convert each to BV(1) via
                // If(cond, 1, 0), then fall through to the normal op-dispatch
                // match below so every BVOp (Eq/Ult/Sub/And/...) is rebuilt with
                // its real semantics. (Previously these arms early-returned and
                // either dropped the second operand or rebuilt non-And/Or/Xor ops
                // as __add__ — see bead angr-c3rd.)
                let bool_to_bv1 = |arg: &Py<PyAny>| -> PyResult<Py<PyAny>> {
                    let bv = claripy_mod.call_method1(
                        "If",
                        (
                            arg,
                            claripy_mod.call_method1("BVV", (1i32, 1u32))?,
                            claripy_mod.call_method1("BVV", (0i32, 1u32))?,
                        ),
                    )?;
                    Ok(bv.unbind())
                };
                let (args, w0, w1) = match (w0, w1) {
                    (None, Some(w)) => (vec![bool_to_bv1(&args[0])?, args[1].clone()], 1u32, w),
                    (Some(w), None) => (vec![args[0].clone(), bool_to_bv1(&args[1])?], w, 1u32),
                    (None, None) => (
                        vec![bool_to_bv1(&args[0])?, bool_to_bv1(&args[1])?],
                        1u32,
                        1u32,
                    ),
                    (Some(a), Some(b)) => (args, a, b),
                };
                if w0 != w1 {
                    if w0 < w1 {
                        let extended = claripy_mod.call_method1("ZeroExt", (w1 - w0, &args[0]))?;
                        vec![extended.unbind(), args[1].clone()]
                    } else {
                        let extended = claripy_mod.call_method1("ZeroExt", (w0 - w1, &args[1]))?;
                        vec![args[0].clone(), extended.unbind()]
                    }
                } else {
                    args
                }
            } else {
                args
            };

            // Build the claripy expression based on the operation
            match op {
                // Arithmetic operations (binary, use method on first arg)
                BVOp::Add => {
                    let arg0 = args[0].bind(py);
                    arg0.call_method1("__add__", (&args[1],))
                        .map(std::convert::Into::into)
                }
                BVOp::Sub => {
                    let arg0 = args[0].bind(py);
                    arg0.call_method1("__sub__", (&args[1],))
                        .map(std::convert::Into::into)
                }
                BVOp::Mul => {
                    let arg0 = args[0].bind(py);
                    arg0.call_method1("__mul__", (&args[1],))
                        .map(std::convert::Into::into)
                }
                BVOp::UDiv => claripy_mod
                    .call_method1("UDiv", (&args[0], &args[1]))
                    .map(std::convert::Into::into),
                BVOp::SDiv => claripy_mod
                    .call_method1("SDiv", (&args[0], &args[1]))
                    .map(std::convert::Into::into),
                BVOp::URem => claripy_mod
                    .call_method1("URem", (&args[0], &args[1]))
                    .map(std::convert::Into::into),
                BVOp::SRem => claripy_mod
                    .call_method1("SMod", (&args[0], &args[1]))
                    .map(std::convert::Into::into),
                BVOp::Neg => {
                    let arg0 = args[0].bind(py);
                    arg0.call_method0("__neg__").map(std::convert::Into::into)
                }

                // Bitwise operations
                BVOp::And => {
                    let arg0 = args[0].bind(py);
                    let result = arg0.call_method1("__and__", (&args[1],))?;
                    // Check for NotImplemented (width mismatch etc)
                    if result.is_none()
                        || result
                            .get_type()
                            .name()
                            .is_ok_and(|n| n == "NotImplementedType")
                    {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "__and__ returned NotImplemented",
                        ));
                    }
                    Ok(result.into())
                }
                BVOp::Or => {
                    let arg0 = args[0].bind(py);
                    let result = arg0.call_method1("__or__", (&args[1],))?;
                    if result.is_none()
                        || result
                            .get_type()
                            .name()
                            .is_ok_and(|n| n == "NotImplementedType")
                    {
                        let t0 = args[0]
                            .bind(py)
                            .get_type()
                            .name()
                            .map(|n| n.to_string())
                            .unwrap_or("?".into());
                        let t1 = args[1]
                            .bind(py)
                            .get_type()
                            .name()
                            .map(|n| n.to_string())
                            .unwrap_or("?".into());
                        let w0: String = args[0]
                            .bind(py)
                            .getattr("length")
                            .map(|l| format!("{l}"))
                            .unwrap_or("?".into());
                        let w1: String = args[1]
                            .bind(py)
                            .getattr("length")
                            .map(|l| format!("{l}"))
                            .unwrap_or("?".into());
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "__or__ NotImpl: {t0}(w={w0}) | {t1}(w={w1})"
                        )));
                    }
                    Ok(result.into())
                }
                BVOp::Xor => {
                    let arg0 = args[0].bind(py);
                    let result = arg0.call_method1("__xor__", (&args[1],))?;
                    if result.is_none()
                        || result
                            .get_type()
                            .name()
                            .is_ok_and(|n| n == "NotImplementedType")
                    {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "__xor__ returned NotImplemented",
                        ));
                    }
                    Ok(result.into())
                }
                BVOp::Not => {
                    let arg0 = args[0].bind(py);
                    arg0.call_method0("__invert__")
                        .map(std::convert::Into::into)
                }

                // Shift operations
                BVOp::Shl => {
                    let arg0 = args[0].bind(py);
                    arg0.call_method1("__lshift__", (&args[1],))
                        .map(std::convert::Into::into)
                }
                BVOp::Lshr => claripy_mod
                    .call_method1("LShR", (&args[0], &args[1]))
                    .map(std::convert::Into::into),
                BVOp::Ashr => {
                    let arg0 = args[0].bind(py);
                    arg0.call_method1("__rshift__", (&args[1],))
                        .map(std::convert::Into::into)
                }
                BVOp::RotL => claripy_mod
                    .call_method1("RotateLeft", (&args[0], &args[1]))
                    .map(std::convert::Into::into),
                BVOp::RotR => claripy_mod
                    .call_method1("RotateRight", (&args[0], &args[1]))
                    .map(std::convert::Into::into),

                // Extension operations (args already validated)
                BVOp::ZeroExt(extend_bits) => {
                    let arg0_type = args[0]
                        .bind(py)
                        .get_type()
                        .name()
                        .map(|n| n.to_string())
                        .unwrap_or_else(|_| "unknown".to_string());

                    // If arg0 is a Bool, convert it to a 1-bit BV first
                    // claripy.ZeroExt requires a BV, not a Bool
                    if arg0_type == "Bool" {
                        // Use claripy.If(cond, BVV(1, 1), BVV(0, 1)) to convert Bool to 1-bit BV
                        let one = claripy_mod.call_method1("BVV", (1i64, 1u32))?;
                        let zero = claripy_mod.call_method1("BVV", (0i64, 1u32))?;
                        let bv1 = claripy_mod.call_method1("If", (&args[0], one, zero))?;
                        claripy_mod
                            .call_method1("ZeroExt", (*extend_bits, bv1))
                            .map(std::convert::Into::into)
                    } else {
                        claripy_mod
                            .call_method1("ZeroExt", (*extend_bits, &args[0]))
                            .map(std::convert::Into::into)
                    }
                }
                BVOp::SignExt(extend_bits) => {
                    let arg0_type = args[0]
                        .bind(py)
                        .get_type()
                        .name()
                        .map(|n| n.to_string())
                        .unwrap_or_else(|_| "unknown".to_string());

                    // If arg0 is a Bool, convert it to a 1-bit BV first
                    // claripy.SignExt requires a BV, not a Bool
                    if arg0_type == "Bool" {
                        let one = claripy_mod.call_method1("BVV", (1i64, 1u32))?;
                        let zero = claripy_mod.call_method1("BVV", (0i64, 1u32))?;
                        let bv1 = claripy_mod.call_method1("If", (&args[0], one, zero))?;
                        claripy_mod
                            .call_method1("SignExt", (*extend_bits, bv1))
                            .map(std::convert::Into::into)
                    } else {
                        claripy_mod
                            .call_method1("SignExt", (*extend_bits, &args[0]))
                            .map(std::convert::Into::into)
                    }
                }
                BVOp::Extract(high, low) => {
                    let arg0_type = args[0]
                        .bind(py)
                        .get_type()
                        .name()
                        .map(|n| n.to_string())
                        .unwrap_or_else(|_| "unknown".to_string());

                    // If arg0 is a Bool, convert it to a 1-bit BV first
                    // claripy.Extract requires a BV, not a Bool
                    if arg0_type == "Bool" {
                        let one = claripy_mod.call_method1("BVV", (1i64, 1u32))?;
                        let zero = claripy_mod.call_method1("BVV", (0i64, 1u32))?;
                        let bv1 = claripy_mod.call_method1("If", (&args[0], one, zero))?;
                        claripy_mod
                            .call_method1("Extract", (*high, *low, bv1))
                            .map(std::convert::Into::into)
                    } else {
                        claripy_mod
                            .call_method1("Extract", (*high, *low, &args[0]))
                            .map(std::convert::Into::into)
                    }
                }
                BVOp::Concat => {
                    // Concat takes multiple args (already validated)
                    if args.len() == 2 {
                        claripy_mod
                            .call_method1("Concat", (&args[0], &args[1]))
                            .map(std::convert::Into::into)
                    } else {
                        // For multi-arg concat, build a tuple
                        let args_tuple = pyo3::types::PyTuple::new(py, &args)?;
                        claripy_mod
                            .call_method1("Concat", args_tuple)
                            .map(std::convert::Into::into)
                    }
                }

                // Comparison operations
                // Note: __eq__ and __ne__ on claripy BVV objects may return Python bool,
                // not claripy Bool. We must wrap Python bools to ensure claripy AST output.
                BVOp::Eq => {
                    let arg0 = args[0].bind(py);
                    let result = arg0.call_method1("__eq__", (&args[1],))?;
                    // If result is Python bool/int (concrete comparison result),
                    // wrap it in claripy.BoolV. Use extract::<bool> which works for
                    // both PyBool and PyInt (True/False are ints in Python).
                    if let Ok(bool_val) = result.extract::<bool>() {
                        claripy_mod
                            .call_method1("BoolV", (bool_val,))
                            .map(std::convert::Into::into)
                    } else {
                        Ok(result.into())
                    }
                }
                BVOp::Ne => {
                    let arg0 = args[0].bind(py);
                    let result = arg0.call_method1("__ne__", (&args[1],))?;
                    // If result is Python bool/int (concrete comparison result),
                    // wrap it in claripy.BoolV. Use extract::<bool> which works for
                    // both PyBool and PyInt (True/False are ints in Python).
                    if let Ok(bool_val) = result.extract::<bool>() {
                        claripy_mod
                            .call_method1("BoolV", (bool_val,))
                            .map(std::convert::Into::into)
                    } else {
                        Ok(result.into())
                    }
                }
                BVOp::Ult => claripy_mod
                    .call_method1("ULT", (&args[0], &args[1]))
                    .map(std::convert::Into::into),
                BVOp::Ule => claripy_mod
                    .call_method1("ULE", (&args[0], &args[1]))
                    .map(std::convert::Into::into),
                BVOp::Ugt => claripy_mod
                    .call_method1("UGT", (&args[0], &args[1]))
                    .map(std::convert::Into::into),
                BVOp::Uge => claripy_mod
                    .call_method1("UGE", (&args[0], &args[1]))
                    .map(std::convert::Into::into),
                BVOp::Slt => claripy_mod
                    .call_method1("SLT", (&args[0], &args[1]))
                    .map(std::convert::Into::into),
                BVOp::Sle => claripy_mod
                    .call_method1("SLE", (&args[0], &args[1]))
                    .map(std::convert::Into::into),
                BVOp::Sgt => claripy_mod
                    .call_method1("SGT", (&args[0], &args[1]))
                    .map(std::convert::Into::into),
                BVOp::Sge => claripy_mod
                    .call_method1("SGE", (&args[0], &args[1]))
                    .map(std::convert::Into::into),

                // Conditional
                BVOp::Ite => {
                    // If(cond, then_val, else_val)
                    claripy_mod
                        .call_method1("If", (&args[0], &args[1], &args[2]))
                        .map(std::convert::Into::into)
                }

                // Utility operations
                BVOp::Reverse => claripy_mod
                    .call_method1("Reverse", (&args[0],))
                    .map(std::convert::Into::into),
                BVOp::Clz | BVOp::Ctz | BVOp::Popcount => {
                    let op_name = match op {
                        BVOp::Clz => "clz",
                        BVOp::Ctz => "ctz",
                        BVOp::Popcount => "popcount",
                        _ => unreachable!(),
                    };
                    let width = bv.width();

                    // If operand is concrete, compute actual result
                    if let Some(operand) = operands.first()
                        && let Some(concrete_val) = operand.as_u128()
                    {
                        let result = match op {
                            BVOp::Clz => {
                                // Count leading zeros, adjusting for width
                                if concrete_val == 0 {
                                    width as u128
                                } else {
                                    let leading = concrete_val.leading_zeros();
                                    // Adjust for actual bit width (128 - width)
                                    (leading - (128 - width)) as u128
                                }
                            }
                            BVOp::Ctz => {
                                // Count trailing zeros
                                if concrete_val == 0 {
                                    width as u128
                                } else {
                                    concrete_val.trailing_zeros().min(width) as u128
                                }
                            }
                            BVOp::Popcount => {
                                // Count ones
                                concrete_val.count_ones() as u128
                            }
                            _ => unreachable!(),
                        };
                        return claripy_mod
                            .call_method1("BVV", (result as i64, width))
                            .map(std::convert::Into::into);
                    }

                    // Symbolic input. For width<=64 emit a sound encoding tied
                    // to the operand AST (angr-acoq); otherwise fall back to a
                    // fresh unconstrained BVS and flag it.
                    let operands_ptr = Arc::as_ptr(operands) as *const () as usize;
                    let ast_res: PyResult<Py<PyAny>> = if width <= 64 {
                        crate::symbolic::record_export_sound_clz();
                        build_sound_bitcount(py, claripy_mod, op, args[0].bind(py), width)
                    } else {
                        crate::symbolic::record_export_unconstrained_clz();
                        log::debug!(
                            "Creating unconstrained {op_name} result for symbolic width>64 input (constraint relationship lost)"
                        );
                        claripy_mod
                            .call_method1("BVS", (format!("{op_name}_result"), width))
                            .map(std::convert::Into::into)
                    };
                    // Stabilize identity: repeated exports of the same RustBV
                    // (same operands Arc) return the identical claripy AST via
                    // the EXPRESSION_BY_OPERANDS_PTR lookup at the top of this fn.
                    if let Ok(ref ast) = ast_res {
                        store_expression_ast_by_operands(
                            operands_ptr,
                            bv.clone(),
                            ast.clone_ref(py),
                        );
                    }
                    ast_res
                }
                // Float ops: claripy's fpAdd/fpSub etc. need an FSort argument
                // and rounding mode; round-tripping a Z3 FP expression through
                // claripy is fragile. The Rust engine keeps the Z3 FP
                // constraint internally (via build_fp_z3_ast_cached) — for the
                // Python side we expose a fresh symbolic BV at the result
                // width (varies for FtoI/CmpXxx). Constraint info is lost
                // when the value crosses back to claripy, but the in-engine
                // solver still sees the FP terms.
                BVOp::Float { kind, prec } => {
                    let width = kind.result_bits(*prec);
                    let operands_ptr = Arc::as_ptr(operands) as *const () as usize;
                    crate::symbolic::record_export_unconstrained_fp();
                    let name = format!("fp_{kind:?}_{prec:?}_result");
                    let ast_res: PyResult<Py<PyAny>> = claripy_mod
                        .call_method1("BVS", (name, width))
                        .map(std::convert::Into::into);
                    // Stabilize identity across repeated exports (angr-acoq).
                    if let Ok(ref ast) = ast_res {
                        store_expression_ast_by_operands(
                            operands_ptr,
                            bv.clone(),
                            ast.clone_ref(py),
                        );
                    }
                    ast_res
                }
            }
        }
    };

    match result {
        Ok(ast) => {
            if let Some(k) = memo_key {
                memo.insert(k, ast.clone_ref(py));
            }
            Ok(ast)
        }
        Err(e) => Err(e),
    }
}
