//! claripy AST -> RustBV import (the op-dispatch conversion path).
//!
//! Entry points `python_to_rustbv` / `claripy_to_rustbv` walk a claripy
//! expression tree and build the equivalent `RustBV`, with a `RustBVHandle`
//! fast path and thread-local caching via the `super::cache` helpers.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::symbolic::{RustBV, RustBVHandle, RustSymbolTable, SymContext};

use super::cache::{
    AST_CACHE, lookup_symbol_by_hash, lookup_symbol_by_name_and_width, lookup_symbol_name_by_id,
    store_claripy_ast_with_info, store_expression_ast_by_operands,
};
use super::{BridgeError, extract_int_value, is_claripy_ast};

/// Check if a Python object is a RustBVHandle.
///
/// This is a fast check that allows bypassing claripy conversion entirely
/// when the Python side returns a handle instead of a claripy AST.
pub fn is_rust_handle(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<RustBVHandle>()
}

/// Try to extract a RustBV from a RustBVHandle via the symbol table.
///
/// This is the fast path for handle-based operations. If the object is a
/// RustBVHandle, we look up the RustBV directly from the symbol table,
/// completely bypassing claripy AST conversion.
///
/// Returns None if the object is not a handle or the handle ID is not found.
pub fn try_handle_to_rustbv(
    obj: &Bound<'_, PyAny>,
    symbol_table: &RustSymbolTable,
) -> Option<RustBV> {
    if let Ok(handle) = obj.extract::<RustBVHandle>() {
        symbol_table.get(handle.id())
    } else {
        None
    }
}

/// Convert a Python object to RustBV, trying handle first, then claripy.
///
/// This is the primary entry point for Python -> Rust conversion on the hot path.
/// It first checks if the object is a RustBVHandle (fast path), and only falls
/// back to claripy conversion if necessary.
///
/// Returns the RustBV, or an error if conversion fails.
pub fn python_to_rustbv(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    symbol_table: &RustSymbolTable,
    ctx: &SymContext,
) -> Result<RustBV, BridgeError> {
    // Fast path: check for RustBVHandle first
    if let Some(bv) = try_handle_to_rustbv(obj, symbol_table) {
        return Ok(bv);
    }

    // Slow path: claripy AST conversion
    if is_claripy_ast(obj) {
        claripy_to_rustbv(py, obj, ctx)
    } else {
        let type_name = obj
            .get_type()
            .name()
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        Err(BridgeError::TypeMismatch(format!(
            "expected RustBVHandle or claripy AST, got {type_name}"
        )))
    }
}

/// Try to extract a concrete BVV value directly from a claripy AST.
/// Returns Some((value, width)) if the AST is a BVV, None otherwise.
/// This is much cheaper than full claripy_to_rustbv conversion.
#[inline]
pub fn try_extract_bvv(ast: &Bound<'_, PyAny>) -> Option<(u128, u32)> {
    let op: String = ast.getattr("op").ok()?.extract().ok()?;
    if op != "BVV" {
        return None;
    }
    let args = ast.getattr("args").ok()?;
    let args_tuple = args.cast::<PyTuple>().ok()?;
    let value: u128 = extract_int_value(args_tuple.get_item(0).ok()?).ok()?;
    let width: u32 = args_tuple.get_item(1).ok()?.extract().ok()?;
    Some((value, width))
}

/// Convert a claripy AST to a RustBV.
///
/// This recursively converts the claripy expression tree to RustBV operations.
/// Supports: BVV, BVS, arithmetic, bitwise, comparison, and extension ops.
/// Uses thread-local LRU caching with claripy's stable `__hash__` to avoid
/// redundant conversions across constraint additions.
///
/// `py` is part of the public API — 10+ external callers across the
/// interpreter/solver/prefetch modules pass their existing GIL token in.
/// The body itself doesn't reach for `py` directly (the `Bound` carries its
/// own token), but `py` flows through the recursive calls and stays in the
/// signature for caller ergonomics.
#[allow(clippy::only_used_in_recursion)]
pub fn claripy_to_rustbv(
    py: Python<'_>,
    ast: &Bound<'_, PyAny>,
    ctx: &SymContext,
) -> Result<RustBV, BridgeError> {
    // GIL-work timing (angr-1ilq.7): bracket the whole conversion — the Python
    // attribute reads, the hash, and the recursive descent — as one region. The
    // depth guard makes the self-recursion and any caller-nested callback a
    // timing no-op, so each disjoint GIL region is counted exactly once.
    let _gil =
        crate::gil_profile::GilWorkGuard::enter_as(crate::gil_profile::GilClass::ClaripyImport);

    // Get the operation name first to determine caching strategy
    let op: String = ast.getattr("op")?.extract()?;
    let op_str = op.as_str();

    // Cache all immutable AST nodes using claripy's stable __hash__.
    // BVV (concrete) nodes are cheap to create and don't need caching.
    // All symbolic/compound nodes benefit from caching.
    let use_cache = op_str != "BVV";

    // Get claripy's stable hash for cache lookup.
    // Claripy ASTs use their internal _hash attribute which is a consistent identifier.
    // We use ast.hash() which returns Python's Py_hash_t (guaranteed to fit in i64).
    let ast_hash: i64 = if use_cache {
        // Use ast.hash() method from PyAny which properly handles Py_hash_t
        ast.hash()? as i64
    } else {
        0 // Not used
    };

    // Check LRU cache for previously converted AST
    if use_cache {
        let cached = tl_cache!(AST_CACHE, get(&ast_hash).cloned());
        if let Some(cached_bv) = cached {
            // Defensive width check — claripy hashes are content-addressed and
            // already include length, so collisions are exceedingly rare, but
            // returning a wrong-width BV would silently corrupt downstream ops.
            // Bool ASTs have length=None and we represent them as width-1 BVs.
            let expected_width: u32 = ast
                .getattr("length")
                .ok()
                .and_then(|l| l.extract::<u32>().ok())
                .unwrap_or(1);
            if cached_bv.width() == expected_width {
                return Ok(cached_bv);
            }
            // Width mismatch: evict the stale entry and fall through to
            // reconvert. The recomputed BV will be re-cached below.
            tl_cache!(AST_CACHE, pop(&ast_hash));
        }
    }

    let args = ast.getattr("args")?;

    let result = match op_str {
        // Concrete bitvector value
        "BVV" => {
            let args_tuple = args
                .cast::<PyTuple>()
                .map_err(|e| BridgeError::TypeMismatch(e.to_string()))?;
            let value: u128 = extract_int_value(args_tuple.get_item(0)?)?;
            let width: u32 = args_tuple.get_item(1)?.extract()?;
            Ok(RustBV::concrete(value, width))
        }

        // Symbolic bitvector value
        "BVS" => {
            let args_tuple = args
                .cast::<PyTuple>()
                .map_err(|e| BridgeError::TypeMismatch(e.to_string()))?;
            let name: String = args_tuple.get_item(0)?.extract()?;
            // Width might be in args[1] or in .length attribute
            let width: u32 = if args_tuple.len() > 1 {
                args_tuple.get_item(1)?.extract().unwrap_or_else(|_| {
                    ast.getattr("length")
                        .and_then(|l| l.extract())
                        .unwrap_or(64)
                })
            } else {
                ast.getattr("length")?.extract()?
            };

            // CRITICAL: Check global registry first for identity preservation
            // If this symbol was already imported, return the existing RustBV
            // to maintain identity across Python<->Rust boundary
            if let Some(existing_id) = lookup_symbol_by_hash(ast_hash) {
                // Symbol already registered, return a reference to it.
                //
                // Rebuild it under the CANONICAL Rust name, not the name this
                // claripy AST carries: the Z3 constant behind a Symbolic is
                // `BV::new_const(name, width)`, so reusing the id with a
                // different string produces a variable Z3 considers unrelated
                // to the original — the id makes the RustBV/export layer look
                // right while every constraint quietly stops binding
                // (angr-izov2). The names differ whenever the symbol was minted
                // in Rust and exported: `claripy.BVS(name, w)` renames to
                // `name_<counter>_<w>` unless `explicit_name` is set.
                let canonical = lookup_symbol_name_by_id(existing_id);
                let bound_name = canonical.as_deref().unwrap_or(&name);
                return Ok(RustBV::symbolic_with_id(existing_id, bound_name, width));
            }

            // Also check by name+width for cases where the hash changed but name is stable
            // D2 Fix: Use width-qualified lookup to avoid collisions
            if let Some(info) = lookup_symbol_by_name_and_width(&name, width) {
                // Symbol with same name/width exists, return reference
                return Ok(RustBV::symbolic_with_id(info.rust_id, &name, width));
            }

            // Create new symbol and register with full info
            let bv = RustBV::symbolic(ctx, &name, width);
            // Store the original claripy AST so we can return it when converting back
            // This preserves symbol identity for Python's memory model
            if let RustBV::Symbolic { id, .. } = &bv {
                store_claripy_ast_with_info(ast_hash, *id, &name, width, ast.clone().unbind());
            }
            Ok(bv)
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

        "SDiv" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("div requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.sdiv(&right, ctx))
        }

        // claripy BV.__floordiv__ is UNSIGNED division (verified: BVV(0xFFFFFFFE,32)//3
        // == 0x55555554). Only the SDiv op-name maps to signed sdiv.
        "__floordiv__" | "UDiv" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("UDiv requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.udiv(&right, ctx))
        }

        "SMod" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("mod requires 2 args".into()));
            }
            let left = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let right = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(left.srem(&right, ctx))
        }

        // claripy BV.__mod__ is UNSIGNED remainder (verified: BVV(0xFFFFFFFE,32)%3 == 2).
        // Only the SMod op-name maps to signed srem (claripy SMod == z3 bvsrem).
        "__mod__" | "URem" => {
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
                return Err(BridgeError::InvalidArgs(
                    "__lshift__ requires 2 args".into(),
                ));
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
                return Err(BridgeError::InvalidArgs(
                    "__rshift__ requires 2 args".into(),
                ));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let amt = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(val.ashr(&amt, ctx))
        }

        "RotateLeft" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs(
                    "RotateLeft requires 2 args".into(),
                ));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            let amt = claripy_to_rustbv(py, &args_list[1], ctx)?;
            Ok(val.rotl(&amt, ctx))
        }

        "RotateRight" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs(
                    "RotateRight requires 2 args".into(),
                ));
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
            let val_width = val.width();

            // P5 fix: Validate Extract bounds to prevent runtime errors
            if high >= val_width {
                return Err(BridgeError::InvalidArgs(format!(
                    "Extract high={high} >= width={val_width}"
                )));
            }
            if low > high {
                return Err(BridgeError::InvalidArgs(format!(
                    "Extract low={low} > high={high}"
                )));
            }

            Ok(val.extract(high, low, ctx))
        }

        "Concat" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                return Err(BridgeError::InvalidArgs(
                    "Concat requires at least 1 arg".into(),
                ));
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

        // Boolean constant
        "BoolV" => {
            let args_tuple = args
                .cast::<PyTuple>()
                .map_err(|e| BridgeError::TypeMismatch(e.to_string()))?;
            let value: bool = args_tuple.get_item(0)?.extract()?;
            // Return 1-bit BV (1 for true, 0 for false)
            Ok(RustBV::concrete(u128::from(value), 1))
        }

        // Symbolic boolean (angr-q6r1): claripy.BoolS("name") is a 1-bit
        // Bool leaf with args=(name,) and no `.length`. Mirror the BVS
        // identity-preservation path with a fixed width of 1 so an
        // `If(BoolS, ...)` (e.g. posix.fork) survives the FFI boundary
        // instead of failing AST conversion. The reverse-direction
        // `rust_to_claripy` rebuilds Symbolic-width-1 values as a
        // claripy `BVS` of width 1, which composes correctly when an
        // outer claripy op (e.g. `If`) coerces the operand to Bool.
        "BoolS" => {
            let args_tuple = args
                .cast::<PyTuple>()
                .map_err(|e| BridgeError::TypeMismatch(e.to_string()))?;
            let name: String = args_tuple.get_item(0)?.extract()?;
            let width: u32 = 1;

            if let Some(existing_id) = lookup_symbol_by_hash(ast_hash) {
                // Canonical Rust name, per the BVS branch above (angr-izov2).
                let canonical = lookup_symbol_name_by_id(existing_id);
                let bound_name = canonical.as_deref().unwrap_or(&name);
                return Ok(RustBV::symbolic_with_id(existing_id, bound_name, width));
            }
            if let Some(info) = lookup_symbol_by_name_and_width(&name, width) {
                return Ok(RustBV::symbolic_with_id(info.rust_id, &name, width));
            }

            let bv = RustBV::symbolic(ctx, &name, width);
            if let RustBV::Symbolic { id, .. } = &bv {
                store_claripy_ast_with_info(ast_hash, *id, &name, width, ast.clone().unbind());
            }
            Ok(bv)
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

        // Boolean operations (for constraints)
        "And" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                // Empty And is True
                return Ok(RustBV::concrete(1, 1));
            }
            // Boolean And: all 1-bit values must be 1
            let mut result = claripy_to_rustbv(py, &args_list[0], ctx)?;
            for arg in &args_list[1..] {
                let next = claripy_to_rustbv(py, arg, ctx)?;
                result = result.and(&next, ctx);
            }
            Ok(result)
        }

        "Or" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                // Empty Or is False
                return Ok(RustBV::concrete(0, 1));
            }
            // Boolean Or: at least one 1-bit value must be 1
            let mut result = claripy_to_rustbv(py, &args_list[0], ctx)?;
            for arg in &args_list[1..] {
                let next = claripy_to_rustbv(py, arg, ctx)?;
                result = result.or(&next, ctx);
            }
            Ok(result)
        }

        "Not" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                return Err(BridgeError::InvalidArgs("Not requires 1 arg".into()));
            }
            let val = claripy_to_rustbv(py, &args_list[0], ctx)?;
            // Boolean Not: invert 1-bit value
            Ok(val.not(ctx))
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

        _ => Err(BridgeError::UnsupportedOp(op_str.to_string())),
    };

    // Cache all symbolic/compound AST nodes using claripy's stable hash.
    // This dramatically reduces conversion overhead when the same expressions
    // appear in multiple constraints.
    if use_cache && let Ok(ref bv) = result {
        // Forward cache: claripy hash → RustBV
        tl_cache!(AST_CACHE, put(ast_hash, bv.clone()));

        // Reverse cache: for Expression results, key the original claripy
        // AST by the operands Arc pointer so `rustbv_to_claripy_memo` can
        // return it verbatim (preserving annotations attached at the
        // Expression level). This is the only Rust→claripy expression
        // cache — the export path has no way to recompute `ast_hash` from
        // a RustBV, so a hash-keyed cache was structurally unreadable.
        if let RustBV::Expression { operands, .. } = bv {
            let operands_ptr = Arc::as_ptr(operands) as *const () as usize;
            store_expression_ast_by_operands(operands_ptr, bv.clone(), ast.clone().unbind());
        }
    }

    result
}

/// Byte-reverse a RustBV value.
fn reverse_bytes(bv: &RustBV, ctx: &SymContext) -> Result<RustBV, BridgeError> {
    let width = bv.width();
    if !width.is_multiple_of(8) {
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
