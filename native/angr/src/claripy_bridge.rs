//! Bridge between claripy ASTs and RustBV.
//!
//! This module provides bidirectional conversion between Python claripy ASTs
//! and Rust RustBV values, enabling native symbolic execution in Rust.

use std::cell::RefCell;
use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

use crate::symbolic::{RustBV, SymContext};

/// Thread-local cache for AST conversions.
/// Key is the Python object ID (pointer), value is the converted RustBV.
thread_local! {
    static AST_CACHE: RefCell<HashMap<isize, RustBV>> = RefCell::new(HashMap::with_capacity(256));
}

/// Clear the AST conversion cache.
/// Call this at block boundaries or when the constraint set changes significantly.
pub fn clear_ast_cache() {
    AST_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

/// Error type for claripy bridge operations.
#[derive(Debug, Clone)]
pub enum BridgeError {
    /// Unsupported claripy operation.
    UnsupportedOp(String),
    /// Type mismatch.
    TypeMismatch(String),
    /// Python error.
    PythonError(String),
    /// Invalid arguments.
    InvalidArgs(String),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::UnsupportedOp(op) => write!(f, "unsupported claripy op: {}", op),
            BridgeError::TypeMismatch(msg) => write!(f, "type mismatch: {}", msg),
            BridgeError::PythonError(msg) => write!(f, "python error: {}", msg),
            BridgeError::InvalidArgs(msg) => write!(f, "invalid arguments: {}", msg),
        }
    }
}

impl std::error::Error for BridgeError {}

impl From<PyErr> for BridgeError {
    fn from(err: PyErr) -> Self {
        BridgeError::PythonError(err.to_string())
    }
}

/// Convert a claripy AST to a RustBV.
///
/// This recursively converts the claripy expression tree to RustBV operations.
/// Supports: BVV, BVS, arithmetic, bitwise, comparison, and extension ops.
/// Uses thread-local caching to avoid redundant conversions.
pub fn claripy_to_rustbv(
    py: Python<'_>,
    ast: &Bound<'_, PyAny>,
    ctx: &SymContext,
) -> Result<RustBV, BridgeError> {
    // Check cache first using Python object ID
    let ast_id = ast.as_ptr() as isize;
    let cached = AST_CACHE.with(|cache| {
        cache.borrow().get(&ast_id).cloned()
    });
    if let Some(cached_bv) = cached {
        return Ok(cached_bv);
    }

    // Get the operation name
    let op: String = ast.getattr("op")?.extract()?;
    let args = ast.getattr("args")?;

    let result = match op.as_str() {
        // Concrete bitvector value
        "BVV" => {
            let args_tuple = args
                .downcast::<PyTuple>()
                .map_err(|e| BridgeError::TypeMismatch(e.to_string()))?;
            let value: u128 = extract_int_value(args_tuple.get_item(0)?)?;
            let width: u32 = args_tuple.get_item(1)?.extract()?;
            Ok(RustBV::concrete(value, width))
        }

        // Symbolic bitvector value
        "BVS" => {
            let args_tuple = args
                .downcast::<PyTuple>()
                .map_err(|e| BridgeError::TypeMismatch(e.to_string()))?;
            let name: String = args_tuple.get_item(0)?.extract()?;
            // Width might be in args[1] or in .length attribute
            let width: u32 = if args_tuple.len() > 1 {
                args_tuple.get_item(1)?.extract().unwrap_or_else(|_| {
                    ast.getattr("length").and_then(|l| l.extract()).unwrap_or(64)
                })
            } else {
                ast.getattr("length")?.extract()?
            };
            Ok(RustBV::symbolic(ctx, &name, width))
        }

        // Arithmetic operations
        "__add__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__add__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.add(&right, ctx))
        }

        "__sub__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__sub__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.sub(&right, ctx))
        }

        "__mul__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__mul__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.mul(&right, ctx))
        }

        "__floordiv__" | "SDiv" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("div requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.sdiv(&right, ctx))
        }

        "UDiv" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("UDiv requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.udiv(&right, ctx))
        }

        "__mod__" | "SMod" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("mod requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.srem(&right, ctx))
        }

        "URem" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("URem requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.urem(&right, ctx))
        }

        "__neg__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                return Err(BridgeError::InvalidArgs("__neg__ requires 1 arg".into()));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            Ok(val.neg(ctx))
        }

        // Bitwise operations
        "__and__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__and__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.and(&right, ctx))
        }

        "__or__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__or__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.or(&right, ctx))
        }

        "__xor__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__xor__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.xor(&right, ctx))
        }

        "__invert__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                return Err(BridgeError::InvalidArgs("__invert__ requires 1 arg".into()));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            Ok(val.not(ctx))
        }

        // Shift operations
        "__lshift__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__lshift__ requires 2 args".into()));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let amt = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(val.shl(&amt, ctx))
        }

        "LShR" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("LShR requires 2 args".into()));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let amt = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(val.lshr(&amt, ctx))
        }

        "__rshift__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__rshift__ requires 2 args".into()));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let amt = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(val.ashr(&amt, ctx))
        }

        "RotateLeft" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("RotateLeft requires 2 args".into()));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let amt = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(val.rotl(&amt, ctx))
        }

        "RotateRight" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("RotateRight requires 2 args".into()));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let amt = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(val.rotr(&amt, ctx))
        }

        // Extension operations
        "ZeroExt" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("ZeroExt requires 2 args".into()));
            }
            let extend_bits: u32 = args_list[0].extract()?;
            let val = claripy_to_rustbv(py, &args_list[1], ctx)?;
            let new_width = val.width() + extend_bits;
            Ok(val.zero_extend(new_width, ctx))
        }

        "SignExt" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("SignExt requires 2 args".into()));
            }
            let extend_bits: u32 = args_list[0].extract()?;
            let val = claripy_to_rustbv(py, &args_list[1], ctx)?;
            let new_width = val.width() + extend_bits;
            Ok(val.sign_extend(new_width, ctx))
        }

        "Extract" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 3 {
                return Err(BridgeError::InvalidArgs("Extract requires 3 args".into()));
            }
            let high: u32 = args_list[0].extract()?;
            let low: u32 = args_list[1].extract()?;
            let val = claripy_to_rustbv(py, &args_list[2], ctx)?;
            Ok(val.extract(high, low, ctx))
        }

        "Concat" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                return Err(BridgeError::InvalidArgs("Concat requires at least 1 arg".into()));
            }
            let mut result = claripy_to_rustbv(py, &args_list[0], ctx)?;
            for arg in &args_list[1..] {
                let next = claripy_to_rustbv(py, arg, ctx)?;
                result = result.concat(&next, ctx);
            }
            Ok(result)
        }

        // Comparison operations (return 1-bit result)
        "__eq__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__eq__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.eq(&right, ctx))
        }

        "__ne__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("__ne__ requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.ne(&right, ctx))
        }

        "ULT" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("ULT requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.ult(&right, ctx))
        }

        "ULE" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("ULE requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.ule(&right, ctx))
        }

        "UGT" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("UGT requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.ugt(&right, ctx))
        }

        "UGE" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("UGE requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.uge(&right, ctx))
        }

        "SLT" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("SLT requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.slt(&right, ctx))
        }

        "SLE" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("SLE requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.sle(&right, ctx))
        }

        "SGT" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("SGT requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.sgt(&right, ctx))
        }

        "SGE" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("SGE requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.sge(&right, ctx))
        }

        // If-then-else
        "If" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 3 {
                return Err(BridgeError::InvalidArgs("If requires 3 args".into()));
            }
            let cond = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let then_val = claripy_to_rustbv(py, &args_list[1], ctx)?;
            let else_val = claripy_to_rustbv(py, &args_list[2], ctx)?;
            Ok(cond.ite(&then_val, &else_val, ctx))
        }

        // Reverse bytes
        "Reverse" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                return Err(BridgeError::InvalidArgs("Reverse requires 1 arg".into()));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            // Byte-reverse the value
            reverse_bytes(&val, ctx)
        }

        _ => Err(BridgeError::UnsupportedOp(op)),
    };

    // Cache successful results
    if let Ok(ref bv) = result {
        AST_CACHE.with(|cache| {
            cache.borrow_mut().insert(ast_id, bv.clone());
        });
    }

    result
}

/// Convert a RustBV back to a claripy AST.
///
/// This is used when returning symbolic results to Python.
pub fn rustbv_to_claripy(
    py: Python<'_>,
    bv: &RustBV,
    claripy_mod: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    match bv {
        RustBV::Concrete { value, width } => {
            // Create claripy.BVV(value, width)
            if *width <= 64 {
                claripy_mod
                    .call_method1("BVV", (*value as i64, *width))
                    .map(|obj| obj.into())
            } else {
                // For wider values, pass as bytes
                let bytes = value.to_le_bytes();
                let py_bytes = PyBytes::new(py, &bytes[..(*width as usize / 8)]);
                claripy_mod
                    .call_method1("BVV", (py_bytes, *width))
                    .map(|obj| obj.into())
            }
        }
        RustBV::Symbolic { name, width, .. } => {
            // Create claripy.BVS(name, width)
            claripy_mod
                .call_method1("BVS", (name.as_str(), *width))
                .map(|obj| obj.into())
        }
        RustBV::Constrained { value, width, .. } => {
            // For constrained values, return the concrete value
            if *width <= 64 {
                claripy_mod
                    .call_method1("BVV", (*value as i64, *width))
                    .map(|obj| obj.into())
            } else {
                let bytes = value.to_le_bytes();
                let py_bytes = PyBytes::new(py, &bytes[..(*width as usize / 8)]);
                claripy_mod
                    .call_method1("BVV", (py_bytes, *width))
                    .map(|obj| obj.into())
            }
        }
    }
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

    // For larger values, use Python's int.to_bytes
    let bit_length: usize = obj
        .call_method0("bit_length")?
        .extract()
        .unwrap_or(128);
    let byte_length = (bit_length + 7) / 8;
    let byte_length = byte_length.max(1).min(16); // Clamp to 1-16 bytes

    let bytes_obj = obj.call_method1("to_bytes", (byte_length, "little"))?;
    let bytes: Vec<u8> = bytes_obj.extract()?;

    let mut value: u128 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if i < 16 {
            value |= (b as u128) << (i * 8);
        }
    }

    Ok(value)
}

/// Byte-reverse a RustBV value.
fn reverse_bytes(
    bv: &RustBV,
    ctx: &SymContext,
) -> Result<RustBV, BridgeError> {
    let width = bv.width();
    if width % 8 != 0 {
        return Err(BridgeError::InvalidArgs(
            "Reverse requires byte-aligned width".into(),
        ));
    }

    if let Some(value) = bv.as_u128() {
        // Concrete case: reverse bytes
        let num_bytes = width / 8;
        let mut reversed: u128 = 0;
        for i in 0..num_bytes {
            let byte = (value >> (i * 8)) & 0xFF;
            reversed |= byte << ((num_bytes - 1 - i) * 8);
        }
        Ok(RustBV::concrete(reversed, width))
    } else {
        // Symbolic case: build concatenation of reversed byte extracts
        let num_bytes = width / 8;
        let mut result = bv.extract(7, 0, ctx);
        for i in 1..num_bytes {
            let byte = bv.extract((i + 1) * 8 - 1, i * 8, ctx);
            result = result.concat(&byte, ctx);
        }
        Ok(result)
    }
}

/// Check if a Python object is a claripy AST.
pub fn is_claripy_ast(obj: &Bound<'_, PyAny>) -> bool {
    obj.hasattr("op").unwrap_or(false) && obj.hasattr("args").unwrap_or(false)
}

/// Get the width (in bits) of a claripy AST.
pub fn get_ast_width(ast: &Bound<'_, PyAny>) -> Option<u32> {
    ast.getattr("length").ok()?.extract().ok()
}

/// Check if a claripy AST is concrete (BVV).
pub fn is_concrete_ast(ast: &Bound<'_, PyAny>) -> bool {
    if let Ok(op) = ast.getattr("op") {
        if let Ok(op_str) = op.extract::<String>() {
            return op_str == "BVV";
        }
    }
    false
}

/// Extract the concrete value from a BVV AST.
pub fn extract_concrete_value(ast: &Bound<'_, PyAny>) -> Option<u128> {
    if !is_concrete_ast(ast) {
        return None;
    }

    let args = ast.getattr("args").ok()?;
    let args_tuple = args.downcast::<PyTuple>().ok()?;
    extract_int_value(args_tuple.get_item(0).ok()?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_int_value_small() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let val = 42i64.into_py(py);
            assert_eq!(extract_int_value(val.bind(py).clone()).unwrap(), 42);
        });
    }
}
