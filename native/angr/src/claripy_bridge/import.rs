//! claripy AST -> RustBV import (the op-dispatch conversion path).
//!
//! Entry point `claripy_to_rustbv` walks a claripy expression tree and builds
//! the equivalent `RustBV`, with a `RustBVHandle` fast path
//! (`try_handle_to_rustbv`) and thread-local caching via the `super::cache`
//! helpers.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::symbolic::{RustBV, RustBVHandle, RustSymbolTable, SymContext, SymbolKind};

use super::cache::{
    AST_CACHE, lookup_symbol_by_hash, lookup_symbol_by_name_and_width, lookup_symbol_name_by_id,
    store_claripy_ast_with_info, store_expression_ast_by_operands,
};
use super::{BridgeError, extract_int_value};

/// Try to extract a RustBV from a RustBVHandle via the symbol table.
///
/// This is the fast path for handle-based operations. If the object is a
/// RustBVHandle, we look up the RustBV directly from the symbol table,
/// completely bypassing claripy AST conversion.
///
/// Returns None if the object is not a handle or the handle ID is not found.
pub(crate) fn try_handle_to_rustbv(
    obj: &Bound<'_, PyAny>,
    symbol_table: &RustSymbolTable,
) -> Option<RustBV> {
    if let Ok(handle) = obj.extract::<RustBVHandle>() {
        symbol_table.get(handle.id())
    } else {
        None
    }
}

/// Try to extract a concrete BVV value directly from a claripy AST.
/// Returns Some((value, width)) if the AST is a BVV, None otherwise.
/// This is much cheaper than full claripy_to_rustbv conversion.
#[inline]
pub(crate) fn try_extract_bvv(ast: &Bound<'_, PyAny>) -> Option<(u128, u32)> {
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
///
/// This is a thin public wrapper around [`claripy_to_rustbv_depth`], which
/// carries the recursion-depth guard (angr-2a3i9). Keeping this outer
/// signature stable (no depth parameter) means the 10+ external call sites
/// across interpreter/solver/prefetch never need to know about the guard —
/// they always start a fresh descent at depth 0.
pub(crate) fn claripy_to_rustbv(
    py: Python<'_>,
    ast: &Bound<'_, PyAny>,
    ctx: &SymContext,
) -> Result<RustBV, BridgeError> {
    claripy_to_rustbv_depth(py, ast, ctx, 0)
}

/// Maximum recursion depth for [`claripy_to_rustbv_depth`]'s self-recursion
/// (angr-2a3i9).
///
/// This function is called directly from the main Python thread (via
/// `solver.eval()`, constraint export, etc.), which runs on CPython's
/// default OS stack — 8 MiB (`RLIMIT_STACK`), not covered by the 16 MiB
/// worker-thread stack from the sibling fix (angr-h92bx, commit
/// 4131a8e27) that only applies to `std::thread`-spawned parallel workers.
/// Every recursion level here is a plain native Rust stack frame with no
/// depth counter to catch, so an unbounded-depth AST (e.g. a long
/// non-shared `Concat`/`Add` chain that gets zero benefit from the
/// per-call memo) overflows the guard page as a raw SIGSEGV instead of a
/// catchable Python exception.
///
/// Per-frame budget, from measured (`std::mem::size_of`, release build)
/// sizes of the types actually live in each stack frame:
/// `size_of::<RustBV>() == 64 B`, `size_of::<Result<RustBV, BridgeError>>()
/// == 64 B`, `size_of::<String>() == 24 B`, `size_of::<Bound<PyAny>>() ==
/// 8 B`, `size_of::<Vec<Bound<PyAny>>>() == 24 B`. A typical match arm
/// (e.g. the binary-op arms) holds `op: String` (24 B), `args_list:
/// Vec<Bound<PyAny>>` (24 B), the `ast`/`args` `Bound<PyAny>` handles
/// (8 B each), up to three intermediate `RustBV` locals for the widest
/// arms — `Concat`/`And`/`Or`'s `result` accumulator or `If`'s
/// `cond`/`then_val`/`else_val` (64 B each, 192 B worst case) — plus the
/// function's own `Result<RustBV, BridgeError>` return slot (64 B).
/// Summing the worst case: 24 + 24 + 8 + 8 + 192 + 64 = 320 B of live
/// locals. Doubling that for return address, saved registers, and
/// alignment padding that `size_of` does not capture gives a conservative
/// **640 B/frame** — rounded up to **1024 B/frame** for headroom against
/// compiler/platform variance (debug builds, non-x86_64 targets, etc.).
///
/// Targeting well inside the 8 MiB main-thread stack — not the full 8 MiB,
/// since the interpreter/solver call chain that reaches this function
/// carries its own (non-recursive) frames below it, and per the sibling
/// worker-thread fix's own margin philosophy (angr-h92bx, commit
/// 4131a8e27: 16 MiB "to clear the main thread by 2x rather than fall
/// short of it") — a **4096**-deep limit costs at most
/// `4096 * 1024 B = 4 MiB`, leaving >2x headroom under the 8 MiB budget
/// even at the padded per-frame estimate. 4096 is also far beyond any real
/// (non-adversarial) claripy AST depth: the sym-write benchmark's
/// symbolic-store ITE chains — the deepest known real-world case, a 142k-node
/// tree without dedup — collapse to ~25 shared subtrees via the per-call memo
/// (see `rustbv_to_claripy`'s doc comment), so real trees bottom out at a
/// few dozen levels, not thousands.
const MAX_IMPORT_RECURSION_DEPTH: u32 = 4096;

/// Depth-guarded recursive implementation of [`claripy_to_rustbv`]. See that
/// function's doc comment for the public contract; see
/// [`MAX_IMPORT_RECURSION_DEPTH`] for the depth-guard calibration (angr-2a3i9).
#[allow(clippy::only_used_in_recursion)]
fn claripy_to_rustbv_depth(
    py: Python<'_>,
    ast: &Bound<'_, PyAny>,
    ctx: &SymContext,
    depth: u32,
) -> Result<RustBV, BridgeError> {
    if depth > MAX_IMPORT_RECURSION_DEPTH {
        return Err(BridgeError::RecursionLimit(format!(
            "claripy_to_rustbv: exceeded max recursion depth {MAX_IMPORT_RECURSION_DEPTH} \
             while importing a claripy AST -- the tree is either pathologically deep or a \
             long unshared chain that the per-call memo cannot dedup"
        )));
    }

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
                crate::symbolic::record_claripy_ast_cache(true);
                return Ok(cached_bv);
            }
            // Width mismatch: evict the stale entry and fall through to
            // reconvert. The recomputed BV will be re-cached below.
            tl_cache!(AST_CACHE, pop(&ast_hash));
            crate::symbolic::record_claripy_ast_cache(false);
        } else {
            crate::symbolic::record_claripy_ast_cache(false);
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
            // Width might be in args[1] or in .length attribute. On double
            // failure DO NOT default to 64: a wrong width poisons the
            // name+width registry (lookup_symbol_by_name_and_width) and the Z3
            // const sort for the process lifetime, and every downstream op
            // silently papers over the mismatch with ZeroExt. Failing loud is
            // strictly better than minting a wrong-width symbol (angr-ph300.56).
            let width: u32 = if args_tuple.len() > 1 {
                match args_tuple.get_item(1)?.extract() {
                    Ok(w) => w,
                    Err(_) => ast
                        .getattr("length")
                        .and_then(|l| l.extract())
                        .map_err(|e| {
                            BridgeError::InvalidArgs(format!(
                                "BVS '{name}': args[1] not an int and .length unreadable ({e})"
                            ))
                        })?,
                }
            } else {
                ast.getattr("length")?.extract()?
            };

            Ok(import_symbolic_leaf(
                ast,
                ast_hash,
                &name,
                width,
                SymbolKind::BitVector,
                ctx,
            ))
        }

        // Arithmetic operations
        "__add__" => import_binary_op(py, &args, ctx, depth, "__add__", RustBV::add),
        "__sub__" => import_binary_op(py, &args, ctx, depth, "__sub__", RustBV::sub),
        "__mul__" => import_binary_op(py, &args, ctx, depth, "__mul__", RustBV::mul),
        "SDiv" => import_binary_op(py, &args, ctx, depth, "div", RustBV::sdiv),

        // claripy BV.__floordiv__ is UNSIGNED division (verified: BVV(0xFFFFFFFE,32)//3
        // == 0x55555554). Only the SDiv op-name maps to signed sdiv.
        "__floordiv__" | "UDiv" => import_binary_op(py, &args, ctx, depth, "UDiv", RustBV::udiv),

        "SMod" => import_binary_op(py, &args, ctx, depth, "mod", RustBV::srem),

        // claripy BV.__mod__ is UNSIGNED remainder (verified: BVV(0xFFFFFFFE,32)%3 == 2).
        // Only the SMod op-name maps to signed srem (claripy SMod == z3 bvsrem).
        "__mod__" | "URem" => import_binary_op(py, &args, ctx, depth, "URem", RustBV::urem),

        "__neg__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                return Err(BridgeError::InvalidArgs("__neg__ requires 1 arg".into()));
            }
            let val = claripy_to_rustbv_depth(py, &args_list[0], ctx, depth + 1)?;
            Ok(val.neg(ctx))
        }

        // Bitwise operations
        "__and__" => import_binary_op(py, &args, ctx, depth, "__and__", RustBV::and),
        "__or__" => import_binary_op(py, &args, ctx, depth, "__or__", RustBV::or),
        "__xor__" => import_binary_op(py, &args, ctx, depth, "__xor__", RustBV::xor),

        "__invert__" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                return Err(BridgeError::InvalidArgs("__invert__ requires 1 arg".into()));
            }
            let val = claripy_to_rustbv_depth(py, &args_list[0], ctx, depth + 1)?;
            Ok(val.not(ctx))
        }

        // Shift operations
        "__lshift__" => import_binary_op(py, &args, ctx, depth, "__lshift__", RustBV::shl),
        "LShR" => import_binary_op(py, &args, ctx, depth, "LShR", RustBV::lshr),
        "__rshift__" => import_binary_op(py, &args, ctx, depth, "__rshift__", RustBV::ashr),
        "RotateLeft" => import_binary_op(py, &args, ctx, depth, "RotateLeft", RustBV::rotl),
        "RotateRight" => import_binary_op(py, &args, ctx, depth, "RotateRight", RustBV::rotr),

        // Extension operations
        "ZeroExt" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("ZeroExt requires 2 args".into()));
            }
            let extend_bits: u32 = args_list[0].extract()?;
            let val = claripy_to_rustbv_depth(py, &args_list[1], ctx, depth + 1)?;
            let new_width = val.width() + extend_bits;
            Ok(val.zero_extend(new_width, ctx))
        }

        "SignExt" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("SignExt requires 2 args".into()));
            }
            let extend_bits: u32 = args_list[0].extract()?;
            let val = claripy_to_rustbv_depth(py, &args_list[1], ctx, depth + 1)?;
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
            let val = claripy_to_rustbv_depth(py, &args_list[2], ctx, depth + 1)?;
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
            let mut result = claripy_to_rustbv_depth(py, &args_list[0], ctx, depth + 1)?;
            for arg in &args_list[1..] {
                let next = claripy_to_rustbv_depth(py, arg, ctx, depth + 1)?;
                result = result.concat(&next, ctx);
            }
            Ok(result)
        }

        // Comparison operations (return 1-bit result)
        "__eq__" => import_binary_op(py, &args, ctx, depth, "__eq__", RustBV::eq),
        "__ne__" => import_binary_op(py, &args, ctx, depth, "__ne__", RustBV::ne),
        "ULT" => import_binary_op(py, &args, ctx, depth, "ULT", RustBV::ult),
        "ULE" => import_binary_op(py, &args, ctx, depth, "ULE", RustBV::ule),
        "UGT" => import_binary_op(py, &args, ctx, depth, "UGT", RustBV::ugt),
        "UGE" => import_binary_op(py, &args, ctx, depth, "UGE", RustBV::uge),
        "SLT" => import_binary_op(py, &args, ctx, depth, "SLT", RustBV::slt),
        "SLE" => import_binary_op(py, &args, ctx, depth, "SLE", RustBV::sle),
        "SGT" => import_binary_op(py, &args, ctx, depth, "SGT", RustBV::sgt),
        "SGE" => import_binary_op(py, &args, ctx, depth, "SGE", RustBV::sge),

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
            // Registered under the Bool sort so this does NOT collide with a
            // same-named `BVS(name, 1)` (angr-9ke6b.38): both are width-1
            // Symbolics here, and aliasing them would make export hand back
            // the other one's claripy AST — a BV where a Bool was expected.
            Ok(import_symbolic_leaf(
                ast,
                ast_hash,
                &name,
                1,
                SymbolKind::Bool,
                ctx,
            ))
        }

        // If-then-else
        "If" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 3 {
                return Err(BridgeError::InvalidArgs("If requires 3 args".into()));
            }
            let cond = claripy_to_rustbv_depth(py, &args_list[0], ctx, depth + 1)?;
            let then_val = claripy_to_rustbv_depth(py, &args_list[1], ctx, depth + 1)?;
            let else_val = claripy_to_rustbv_depth(py, &args_list[2], ctx, depth + 1)?;
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
            let mut result = claripy_to_rustbv_depth(py, &args_list[0], ctx, depth + 1)?;
            for arg in &args_list[1..] {
                let next = claripy_to_rustbv_depth(py, arg, ctx, depth + 1)?;
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
            let mut result = claripy_to_rustbv_depth(py, &args_list[0], ctx, depth + 1)?;
            for arg in &args_list[1..] {
                let next = claripy_to_rustbv_depth(py, arg, ctx, depth + 1)?;
                result = result.or(&next, ctx);
            }
            Ok(result)
        }

        "Not" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                return Err(BridgeError::InvalidArgs("Not requires 1 arg".into()));
            }
            let val = claripy_to_rustbv_depth(py, &args_list[0], ctx, depth + 1)?;
            // Boolean Not: invert 1-bit value
            Ok(val.not(ctx))
        }

        // Reverse bytes
        "Reverse" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.is_empty() {
                return Err(BridgeError::InvalidArgs("Reverse requires 1 arg".into()));
            }
            let val = claripy_to_rustbv_depth(py, &args_list[0], ctx, depth + 1)?;
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

/// Shared body for the ~24 plain binary ops in [`claripy_to_rustbv_depth`]'s
/// op-dispatch match (angr-9ke6b.45 collapsed them into this).
///
/// Every one of those arms — arithmetic, bitwise, shift/rotate, comparison —
/// had grown its own verbatim copy of the same shape: extract `args` into a
/// 2-element list, reject any other arity, recurse into both operands at
/// `depth + 1`, then call the matching `RustBV` method. Duplicating it 24
/// times gave an op-specific typo (recursing twice into `args_list[0]`,
/// forgetting `depth + 1`, naming the wrong op in the error) 24 places to
/// hide; there is now exactly one.
///
/// `op_name` is the label used in the arity error and is passed explicitly
/// rather than reusing the dispatched `op` string, because several arms match
/// two claripy op-names at once (`"__floordiv__" | "UDiv"`) and report the
/// canonical one.
///
/// `apply` is a plain `fn` pointer, not an `impl Fn`, so all 24 call sites
/// share a single instantiation of this function instead of monomorphizing
/// one copy each.
fn import_binary_op(
    py: Python<'_>,
    args: &Bound<'_, PyAny>,
    ctx: &SymContext,
    depth: u32,
    op_name: &str,
    apply: fn(&RustBV, &RustBV, &SymContext) -> RustBV,
) -> Result<RustBV, BridgeError> {
    let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
    if args_list.len() != 2 {
        return Err(BridgeError::InvalidArgs(format!(
            "{op_name} requires 2 args"
        )));
    }
    let left = claripy_to_rustbv_depth(py, &args_list[0], ctx, depth + 1)?;
    let right = claripy_to_rustbv_depth(py, &args_list[1], ctx, depth + 1)?;
    Ok(apply(&left, &right, ctx))
}

/// Import a symbolic *leaf* (`BVS` / `BoolS`) while preserving symbol identity
/// across the Python↔Rust boundary (angr-9ke6b.37 extracted this from the two
/// arms that had each grown their own copy).
///
/// Three-step resolution, in order:
///
/// 1. **By claripy hash** — the symbol was already imported from this exact
///    AST, so reuse its id. Rebuild it under the CANONICAL Rust name, not the
///    name this claripy AST carries: the Z3 constant behind a `Symbolic` is
///    `BV::new_const(name, width)`, so reusing the id with a different string
///    produces a variable Z3 considers unrelated to the original — the id
///    makes the RustBV/export layer look right while every constraint quietly
///    stops binding (angr-izov2). The names differ whenever the symbol was
///    minted in Rust and exported: `claripy.BVS(name, w)` renames to
///    `name_<counter>_<w>` unless `explicit_name` is set.
/// 2. **By name+width+sort** — the hash changed but the identity is stable.
///    The key includes `kind` so a `BoolS("x")` and a `BVS("x", 1)` occupy
///    distinct slots (angr-9ke6b.38); dropping it re-aliases them and export
///    hands back the wrong sort's claripy AST.
/// 3. **Mint + register** — first sighting; store the originating claripy AST
///    so the reverse conversion can return it verbatim.
///
/// Every step works on `kind.rust_symbol_name(name)`, never the raw claripy
/// name. A Bool leaf is modelled as a width-1 BV, so the sort tag in the
/// registry key separates the two symbols' *identity* but on its own leaves
/// them sharing one Z3 constant; the tagged name is what `RustBV::from_parts`
/// decodes to build a Bool-sorted constant instead (angr-9ke6b.223). Tagging
/// here — before any registry call — keeps the `name_to_info` key,
/// `rust_id_to_name`, the `RustBV` name and the Z3 term in agreement, which is
/// the precondition step 1 relies on.
fn import_symbolic_leaf(
    ast: &Bound<'_, PyAny>,
    ast_hash: i64,
    name: &str,
    width: u32,
    kind: SymbolKind,
    ctx: &SymContext,
) -> RustBV {
    let rust_name = kind.rust_symbol_name(name);

    if let Some(existing_id) = lookup_symbol_by_hash(ast_hash) {
        let canonical = lookup_symbol_name_by_id(existing_id);
        let bound_name = canonical.as_deref().unwrap_or(&rust_name);
        return RustBV::symbolic_with_id(existing_id, bound_name, width);
    }

    if let Some(info) = lookup_symbol_by_name_and_width(&rust_name, width, kind) {
        return RustBV::symbolic_with_id(info.rust_id, &rust_name, width);
    }

    let bv = RustBV::symbolic(ctx, &rust_name, width);
    if let RustBV::Symbolic { id, .. } = &bv {
        store_claripy_ast_with_info(ast_hash, *id, &rust_name, width, kind, ast.clone().unbind());
    }
    bv
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
