//! claripy AST -> RustBV import (the op-dispatch conversion path).
//!
//! Entry point `claripy_to_rustbv` walks a claripy expression tree and builds
//! the equivalent `RustBV`.
//!
//! claripy is clarirs, so an AST of this extension's own claripy type is
//! walked natively ([`import_clarirs`]): the `AstRef` behind it is read
//! directly instead of through `op` / `args` attribute calls. The attribute
//! walk remains for any other object (`cargo test`'s embedded interpreter
//! imports the installed extension, whose claripy classes are different type
//! objects; some tests hand in duck-typed ASTs).

use std::cell::RefCell;
use std::collections::HashMap;

use clarirs_core::prelude::{AstOp, AstRef};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::claripy::ast::base::Base;
use crate::claripy::ast::opstring::ToOpString;

use crate::symbolic::{RustBV, SymContext, SymbolKind, check_bv_width, check_extract_bounds};

use super::{BridgeError, extract_int_value};

/// Try to extract a concrete BVV value directly from a claripy AST.
/// Returns Some((value, width)) if the AST is a BVV, None otherwise.
/// This is much cheaper than full claripy_to_rustbv conversion.
///
/// A width past [`check_bv_width`]'s cap yields `None` rather than the pair:
/// this is a *fast path* for the `"BVV"` arm of [`claripy_to_rustbv_depth`],
/// and that arm now refuses such a width, so returning the pair here would let
/// the shortcut accept what the general path rejects (and hand
/// `solver::RustSolverContext::eval_batch` an absurd width to build a
/// `RustBV::concrete` from) (angr-0jh0j.8). Callers fall back to the general
/// path, which reports the refusal as a catchable error.
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
    // SILENT(cat-a): every `?` above already means "not a plain BVV, use the
    // general path"; an over-cap width is the same answer for the same reason.
    check_bv_width("BVV", width).ok()?;
    Some((value, width))
}

/// Reject a same-width binary operation whose operands disagree in width.
///
/// `value_ops.rs` only `debug_assert_eq!`s this, and its module doc says that
/// is deliberate *because* "the Python boundary rejects it up front" — this is
/// that boundary, so the check has to be here or it does not exist in the
/// shipped `release` profile. A mismatch reaching `RustBV::add`/`ult`/... in
/// release builds a Z3 term out of two different sorts, which aborts the
/// process under `panic = "abort"` rather than raising into Python, and the
/// same holds for `Z3_mk_ite` with mismatched then/else sorts.
///
/// The "claripy already validates this" assumption does not hold
/// unconditionally: claripy's `length_same_check` is gated behind
/// `_d._DEBUG`, a documented user-facing performance toggle whose audience is
/// exactly the one reaching for this engine. `Extract` and `Concat` in this
/// same file already guard their own width preconditions explicitly
/// (`check_extract_bounds`, `check_bv_width`); the binary-op and `If` arms
/// were the inconsistent case (angr-6cp06.81).
fn check_same_width(op: &str, left: &RustBV, right: &RustBV) -> Result<(), BridgeError> {
    if left.width() != right.width() {
        return Err(BridgeError::InvalidArgs(format!(
            "{op}: operand width mismatch: {}-bit vs {}-bit",
            left.width(),
            right.width()
        )));
    }
    Ok(())
}

/// Width of an op whose result width is *derived* from its operands.
///
/// The `"ZeroExt"` / `"SignExt"` arms of [`claripy_to_rustbv_depth`] compute
/// `source_width + extend_bits` from a caller-chosen `extend_bits`, and the
/// `"Concat"` arm accumulates `result.width() + next.width()` across its
/// operand list. Both halves of that need guarding at this trust boundary: the
/// sum wraps in the shipped `release` profile (no overflow checks), which would
/// make `new_width` come out *smaller* than the source and silently violate the
/// narrowing precondition `value_ops::zero_extend_into` only `debug_assert!`s;
/// and an unwrapped sum can still be an absurd width Z3 would allocate a
/// multi-gigabit sort for. So: `checked_add` to refuse rather than saturate (an
/// identity, not a size), then [`check_bv_width`] on the sum.
fn extended_width(op: &str, source_width: u32, extend_bits: u32) -> Result<u32, BridgeError> {
    let new_width = source_width.checked_add(extend_bits).ok_or_else(|| {
        BridgeError::InvalidArgs(format!(
            "{op}: width {source_width} + {extend_bits} overflows u32"
        ))
    })?;
    check_bv_width(op, new_width).map_err(BridgeError::InvalidArgs)?;
    Ok(new_width)
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
/// signature stable (no depth parameter) means the external call sites
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
/// worker-thread stack from the sibling fix (angr-h92bx) that only applies to `std::thread`-spawned parallel workers.
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
/// worker-thread fix's own margin philosophy (16 MiB "to clear the main
/// thread by 2x rather than fall short of it") (angr-h92bx) — a **4096**-deep
/// limit costs at most
/// `4096 * 1024 B = 4 MiB`, leaving >2x headroom under the 8 MiB budget
/// even at the padded per-frame estimate. 4096 is also far beyond any real
/// (non-adversarial) claripy AST depth: the sym-write benchmark's
/// symbolic-store ITE chains — the deepest known real-world case, a 142k-node
/// tree without dedup — collapse to ~25 shared subtrees via the per-call memo
/// (see `rustbv_to_claripy`'s doc comment), so real trees bottom out at a
/// few dozen levels, not thousands.
///
/// Export has no twin bound: it is a single `Base::from_ast` with no
/// recursion.
pub(super) const MAX_IMPORT_RECURSION_DEPTH: u32 = crate::symbolic::z3_lower::MAX_LOWER_DEPTH;

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
    // attribute reads and the recursive descent — as one region. The
    // depth guard makes the self-recursion and any caller-nested callback a
    // timing no-op, so each disjoint GIL region is counted exactly once.
    let _gil =
        crate::gil_profile::GilWorkGuard::enter_as(crate::gil_profile::GilClass::ClaripyImport);

    if let Ok(base) = ast.cast::<Base>() {
        return import_clarirs(&base.get().ast(), ctx, depth);
    }

    let op: String = ast.getattr("op")?.extract()?;
    let op_str = op.as_str();

    let args = ast.getattr("args")?;

    match op_str {
        // Concrete bitvector value
        "BVV" => {
            let args_tuple = args
                .cast::<PyTuple>()
                .map_err(|e| BridgeError::TypeMismatch(e.to_string()))?;
            let value: u128 = extract_int_value(args_tuple.get_item(0)?)?;
            let width: u32 = args_tuple.get_item(1)?.extract()?;
            // The claripy boundary is a trust boundary just like the handle
            // API: a caller-chosen width reaches Z3 as a sort allocation
            // (angr-0jh0j.8).
            check_bv_width("BVV", width).map_err(BridgeError::InvalidArgs)?;
            Ok(RustBV::concrete(value, width))
        }

        // Symbolic bitvector value
        "BVS" => {
            let args_tuple = args
                .cast::<PyTuple>()
                .map_err(|e| BridgeError::TypeMismatch(e.to_string()))?;
            let name: String = args_tuple.get_item(0)?.extract()?;
            // Width might be in args[1] or in .length attribute. On double
            // failure DO NOT default to 64: a wrong width mints a different
            // variable than the one Python holds, and every downstream op
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
            check_bv_width("BVS", width).map_err(BridgeError::InvalidArgs)?;

            Ok(import_symbolic_leaf(
                &name,
                width,
                SymbolKind::BitVector,
                ctx,
            ))
        }

        // Arithmetic operations
        "__add__" => import_assoc_op(py, &args, ctx, depth, "__add__", RustBV::add),
        "__sub__" => import_binary_op(py, &args, ctx, depth, "__sub__", RustBV::sub),
        "__mul__" => import_assoc_op(py, &args, ctx, depth, "__mul__", RustBV::mul),
        "SDiv" => import_binary_op(py, &args, ctx, depth, "div", RustBV::sdiv),

        // claripy BV.__floordiv__ is UNSIGNED division (verified: BVV(0xFFFFFFFE,32)//3
        // == 0x55555554). Only the SDiv op-name maps to signed sdiv.
        "__floordiv__" | "UDiv" => import_binary_op(py, &args, ctx, depth, "UDiv", RustBV::udiv),

        "SMod" => import_binary_op(py, &args, ctx, depth, "mod", RustBV::srem),

        // claripy BV.__mod__ is UNSIGNED remainder (verified: BVV(0xFFFFFFFE,32)%3 == 2).
        // Only the SMod op-name maps to signed srem (claripy SMod == z3 bvsrem).
        "__mod__" | "URem" => import_binary_op(py, &args, ctx, depth, "URem", RustBV::urem),

        "__neg__" => import_unary_op(py, &args, ctx, depth, "__neg__", RustBV::neg),

        // Bitwise operations
        "__and__" => import_assoc_op(py, &args, ctx, depth, "__and__", RustBV::and),
        "__or__" => import_assoc_op(py, &args, ctx, depth, "__or__", RustBV::or),
        "__xor__" => import_assoc_op(py, &args, ctx, depth, "__xor__", RustBV::xor),

        "__invert__" => import_unary_op(py, &args, ctx, depth, "__invert__", RustBV::not),

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
            let new_width = extended_width("ZeroExt", val.width(), extend_bits)?;
            Ok(val.zero_extend(new_width, ctx))
        }

        "SignExt" => {
            let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
            if args_list.len() != 2 {
                return Err(BridgeError::InvalidArgs("SignExt requires 2 args".into()));
            }
            let extend_bits: u32 = args_list[0].extract()?;
            let val = claripy_to_rustbv_depth(py, &args_list[1], ctx, depth + 1)?;
            let new_width = extended_width("SignExt", val.width(), extend_bits)?;
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
            // Validate Extract bounds to prevent runtime errors. Shared with
            // the handle API's `op_extract`, which guards the same
            // `Z3_mk_extract` precondition at the other trust boundary
            // (angr-c7xno.92).
            check_extract_bounds("Extract", high, low, val.width())
                .map_err(BridgeError::InvalidArgs)?;

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
                // Guard the accumulator at *every* step rather than once at the
                // end: `value_ops::concat_owned` adds the two operand widths
                // with a plain unchecked `u32` add, so an intermediate wrap
                // would leave a `RustBV` whose stored width is small and wrong
                // while its Z3 sort is enormous — and `width()` is what every
                // later extract-bounds / eval / export check trusts.
                extended_width("Concat", result.width(), next.width())?;
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
            Ok(import_symbolic_leaf(&name, 1, SymbolKind::Bool, ctx))
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
            check_same_width("If", &then_val, &else_val)?;
            Ok(cond.ite(&then_val, &else_val, ctx))
        }

        // Boolean operations (for constraints). Both fold their operand list
        // with `import_nary_op`; empty `And` is True, empty `Or` is False.
        "And" => import_nary_op(py, &args, ctx, depth, "And", RustBV::and, 1),

        "Or" => import_nary_op(py, &args, ctx, depth, "Or", RustBV::or, 0),

        // Bool Xor, n-ary like And/Or (the BV form is `__xor__`).
        "Xor" => import_assoc_op(py, &args, ctx, depth, "Xor", RustBV::xor),

        // Boolean Not: invert 1-bit value — same `RustBV::not` as `__invert__`.
        "Not" => import_unary_op(py, &args, ctx, depth, "Not", RustBV::not),

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
    }
}

/// [`claripy_to_rustbv_depth`] over a clarirs `AstRef`, without the Python
/// attribute walk. The value keeps the clarirs nodes it is built from, so it
/// exports as those nodes, annotations included.
///
/// Op semantics and every width / bounds check mirror the attribute walk's
/// arms one for one.
///
/// Each distinct node is imported once per call: a merged state's memory
/// holds `If` trees that share their subterms, so a walk without the memo
/// visits a node once per path to it, exponential in the merge depth
/// (angr-mx00p.89).
fn import_clarirs(
    node: &AstRef<'static>,
    ctx: &SymContext,
    depth: u32,
) -> Result<RustBV, BridgeError> {
    import_clarirs_memo(node, ctx, depth, &RefCell::new(HashMap::new()))
}

/// Imported value per clarirs node, keyed by its hash: clarirs interns nodes,
/// so equal hashes are the same node.
type ImportMemo = RefCell<HashMap<u64, RustBV>>;

fn import_clarirs_memo(
    node: &AstRef<'static>,
    ctx: &SymContext,
    depth: u32,
    memo: &ImportMemo,
) -> Result<RustBV, BridgeError> {
    if let Some(done) = memo.borrow().get(&node.hash()) {
        return Ok(done.clone());
    }
    let value = import_clarirs_node(node, ctx, depth, memo)?;
    memo.borrow_mut().insert(node.hash(), value.clone());
    Ok(value)
}

#[allow(clippy::too_many_lines, reason = "one arm per supported AstOp")]
fn import_clarirs_node(
    node: &AstRef<'static>,
    ctx: &SymContext,
    depth: u32,
    memo: &ImportMemo,
) -> Result<RustBV, BridgeError> {
    if depth > MAX_IMPORT_RECURSION_DEPTH {
        return Err(BridgeError::RecursionLimit(format!(
            "claripy_to_rustbv: exceeded max recursion depth {MAX_IMPORT_RECURSION_DEPTH} \
             while importing a claripy AST"
        )));
    }

    let op_name = || node.to_opstring();
    let children = || node.op().child_iter();
    let import = |child: &AstRef<'static>| import_clarirs_memo(child, ctx, depth + 1, memo);
    let child = |i: usize| -> Result<RustBV, BridgeError> {
        let c = node.op().get_child(i).ok_or_else(|| {
            BridgeError::InvalidArgs(format!("{} is missing operand {i}", node.to_opstring()))
        })?;
        import(&c)
    };
    let binary = |apply: fn(&RustBV, &RustBV, &SymContext) -> RustBV| {
        let (l, r) = (child(0)?, child(1)?);
        check_same_width(&op_name(), &l, &r)?;
        Ok(apply(&l, &r, ctx))
    };
    // Left fold over every operand; `empty` is the identity for an operand-less
    // Bool And/Or, and `None` makes an empty list an error.
    let fold = |apply: fn(&RustBV, &RustBV, &SymContext) -> RustBV, empty: Option<u128>| {
        let mut it = children();
        let Some(first) = it.next() else {
            return empty.map(|v| RustBV::concrete(v, 1)).ok_or_else(|| {
                BridgeError::InvalidArgs(format!("{} requires at least 1 arg", op_name()))
            });
        };
        let mut acc = import(&first)?;
        for c in it {
            let next = import(&c)?;
            check_same_width(&op_name(), &acc, &next)?;
            acc = apply(&acc, &next, ctx);
        }
        Ok(acc)
    };
    let is_bool = node.ast_type().is_bool();
    // A float term, or a predicate or conversion over floats or strings, has
    // no operand-wise form: importing a float child as its IEEE bits would turn
    // `fp.eq` into bit equality. Such a node is taken whole as a
    // `RustBV::Ast` (angr-mx00p.11), as is any op without an arm below.
    // Admitted, so a term with no Z3 form is refused here, at the trust
    // boundary, not at its first query (angr-mx00p.116.5).
    let whole = || {
        crate::symbolic::admit_clarirs(node.clone())
            .map_err(|e| BridgeError::UnsupportedOp(format!("{}: {e}", op_name())))
    };
    // A compound Bool (a comparison, or a connective over Bools) is also
    // taken whole: Python claripy has already simplified it, `RustBV::node`
    // reads it in the shapes an operand-wise import would build, and keeping
    // the node keeps the term Python holds, so it exports back as the same
    // node, annotations included (angr-mx00p.12). Bool leaves and literals
    // still go through the arms below.
    let compound_bool = is_bool && !matches!(node.op(), AstOp::BoolS(_) | AstOp::BoolV(_));
    let opaque = compound_bool
        || node.ast_type().is_float()
        || node.ast_type().is_string()
        || children().any(|c| c.ast_type().is_float() || c.ast_type().is_string());

    match node.op() {
        _ if opaque => whole(),
        AstOp::BVV(value) => {
            let width = value.len();
            check_bv_width("BVV", width).map_err(BridgeError::InvalidArgs)?;
            let big = value.to_biguint();
            // RustBV::Concrete holds a u128; refuse rather than truncate a
            // wider literal, as the attribute walk's extract_int_value does
            // (angr-cxw7).
            let v = u128::try_from(&big).map_err(|_| {
                BridgeError::InvalidArgs(format!(
                    "concrete integer needs {} bits but RustBV::Concrete is capped at 128; \
                     wide concrete BVV import is unsupported",
                    big.bits()
                ))
            })?;
            Ok(RustBV::concrete(v, width))
        }
        // A clarirs leaf is taken whole: interning makes it the one node with
        // that name, the variable Z3 binds, and keeps its annotations
        // (angr-mx00p.12). A BoolS comes back as `ite(b, 1, 0)`.
        AstOp::BVS(_, width) => {
            check_bv_width("BVS", *width).map_err(BridgeError::InvalidArgs)?;
            if *width == 0 {
                // The empty bitvector's only value, as `RustBV::from_parts`
                // degenerates a width-0 leaf (angr-fs8kb.28).
                return Ok(RustBV::concrete(0, 0));
            }
            whole()
        }
        AstOp::BoolS(_) => whole(),
        AstOp::BoolV(b) => Ok(RustBV::concrete(u128::from(*b), 1)),

        AstOp::Add(_) => fold(RustBV::add, None),
        AstOp::Mul(_) => fold(RustBV::mul, None),
        AstOp::And(_) => fold(RustBV::and, None),
        AstOp::Or(_) => fold(RustBV::or, None),
        AstOp::Xor(_) => fold(RustBV::xor, None),
        AstOp::Sub(..) => binary(RustBV::sub),
        AstOp::UDiv(..) => binary(RustBV::udiv),
        AstOp::SDiv(..) => binary(RustBV::sdiv),
        AstOp::URem(..) => binary(RustBV::urem),
        AstOp::SRem(..) => binary(RustBV::srem),
        AstOp::ShL(..) => binary(RustBV::shl),
        AstOp::LShR(..) => binary(RustBV::lshr),
        AstOp::AShR(..) => binary(RustBV::ashr),
        AstOp::RotateLeft(..) => binary(RustBV::rotl),
        AstOp::RotateRight(..) => binary(RustBV::rotr),
        AstOp::Neg(_) => child(0).map(|v| v.neg(ctx)),
        AstOp::Not(_) => child(0).map(|v| v.not(ctx)),
        AstOp::ITE(..) => {
            let (cond, then_val, else_val) = (child(0)?, child(1)?, child(2)?);
            check_same_width("If", &then_val, &else_val)?;
            Ok(cond.ite(&then_val, &else_val, ctx))
        }
        AstOp::ZeroExt(_, bits) => {
            let val = child(0)?;
            let new_width = extended_width("ZeroExt", val.width(), *bits)?;
            Ok(val.zero_extend(new_width, ctx))
        }
        AstOp::SignExt(_, bits) => {
            let val = child(0)?;
            let new_width = extended_width("SignExt", val.width(), *bits)?;
            Ok(val.sign_extend(new_width, ctx))
        }
        AstOp::Extract(_, high, low) => {
            let val = child(0)?;
            check_extract_bounds("Extract", *high, *low, val.width())
                .map_err(BridgeError::InvalidArgs)?;
            Ok(val.extract(*high, *low, ctx))
        }
        AstOp::Concat(_) => {
            let parts = children()
                .map(|c| import(&c))
                .collect::<Result<Vec<_>, _>>()?;
            let Some((first, rest)) = parts.split_first() else {
                return Err(BridgeError::InvalidArgs(
                    "Concat requires at least 1 arg".into(),
                ));
            };
            rest.iter().try_fold(first.width(), |width, next| {
                extended_width("Concat", width, next.width())
            })?;
            // Balanced, not a left fold: each clarirs node holds the union of
            // its operands' variables, so a chain over n symbolic parts would
            // build sets of every size up to n.
            Ok(RustBV::concat_balanced(&parts, ctx))
        }
        AstOp::ByteReverse(_) => reverse_bytes(&child(0)?, ctx),
        _ => whole(),
    }
}

/// Shared body for the ~24 plain binary ops in [`claripy_to_rustbv_depth`]'s
/// op-dispatch match, collapsed into this one helper (angr-9ke6b.45).
///
/// Every one of those arms — arithmetic, bitwise, shift/rotate, comparison —
/// had grown its own verbatim copy of the same shape: extract `args` into a
/// 2-element list, reject any other arity, recurse into both operands at
/// `depth + 1`, then call the matching `RustBV` method. Duplicating it per
/// binary op gave an op-specific typo (recursing twice into `args_list[0]`,
/// forgetting `depth + 1`, naming the wrong op in the error) one place per op
/// to hide; there is now exactly one.
///
/// `op_name` is the label used in the arity error and is passed explicitly
/// rather than reusing the dispatched `op` string, because several arms match
/// two claripy op-names at once (`"__floordiv__" | "UDiv"`) and report the
/// canonical one.
///
/// `apply` is a plain `fn` pointer, not an `impl Fn`, so every call site
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
    check_same_width(op_name, &left, &right)?;
    Ok(apply(&left, &right, ctx))
}

/// N-ary counterpart of [`import_binary_op`] for the associative bitvector
/// ops (`__add__`, `__mul__`, `__and__`, `__or__`, `__xor__`).
///
/// clarirs flattens a chain of these into one node whose `args` holds every
/// operand — `x + y + z` is a single 3-argument `__add__` — so a strict
/// two-argument import refused any longer chain. A refused register or memory
/// import is only logged, so the value silently reached Rust as its default
/// instead. Folds left, width-checking each operand against the accumulator
/// as [`import_nary_op`] does; at least one operand is required.
fn import_assoc_op(
    py: Python<'_>,
    args: &Bound<'_, PyAny>,
    ctx: &SymContext,
    depth: u32,
    op_name: &str,
    apply: fn(&RustBV, &RustBV, &SymContext) -> RustBV,
) -> Result<RustBV, BridgeError> {
    let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
    let Some((first, rest)) = args_list.split_first() else {
        return Err(BridgeError::InvalidArgs(format!(
            "{op_name} requires at least 1 arg"
        )));
    };
    let mut result = claripy_to_rustbv_depth(py, first, ctx, depth + 1)?;
    for arg in rest {
        let next = claripy_to_rustbv_depth(py, arg, ctx, depth + 1)?;
        check_same_width(op_name, &result, &next)?;
        result = apply(&result, &next, ctx);
    }
    Ok(result)
}

/// Arity-1 counterpart of [`import_binary_op`] (angr-12jjk.15).
///
/// `__neg__`, `__invert__` and `Not` had each kept a verbatim copy of the
/// same shape the binary helper already collapsed — extract `args`, reject a
/// missing operand, recurse at `depth + 1`, apply the `RustBV` method — which
/// left the identical typo classes (wrong op-name in the error, a dropped
/// `depth + 1`) three places to hide.
///
/// The arity check is `is_empty()` rather than `len() != 1` because that is
/// what all three arms did: claripy only ever emits one operand for these
/// ops, and extra trailing args are ignored rather than rejected.
///
/// Like the binary helper, `apply` is a plain `fn` pointer so every call site
/// shares one instantiation.
fn import_unary_op(
    py: Python<'_>,
    args: &Bound<'_, PyAny>,
    ctx: &SymContext,
    depth: u32,
    op_name: &str,
    apply: fn(&RustBV, &SymContext) -> RustBV,
) -> Result<RustBV, BridgeError> {
    let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
    if args_list.is_empty() {
        return Err(BridgeError::InvalidArgs(format!(
            "{op_name} requires 1 arg"
        )));
    }
    let val = claripy_to_rustbv_depth(py, &args_list[0], ctx, depth + 1)?;
    Ok(apply(&val, ctx))
}

/// Variadic counterpart of [`import_binary_op`], for the boolean `And` / `Or`
/// arms claripy emits with an arbitrary operand count.
///
/// Both arms had kept a hand-rolled fold that reached `RustBV::and` /
/// `RustBV::or` directly and so skipped the [`check_same_width`] guard every
/// other multi-operand arm performs — `RustBV`'s own width-equality checks are
/// `debug_assert_eq!` and compile out of the shipped release profile, so a
/// mismatched-width operand pair built a mismatched-sort Z3 `bvand`/`bvor` and
/// aborted the process (angr-fs8kb.80). Folding through one helper makes the
/// check structural: every operand is measured against the accumulator before
/// it is applied.
///
/// `empty` is the identity the op returns for a zero-operand list, as a 1-bit
/// concrete: `1` for `And` (empty conjunction is True), `0` for `Or`.
///
/// Like the other two helpers, `apply` is a plain `fn` pointer so both call
/// sites share one instantiation.
fn import_nary_op(
    py: Python<'_>,
    args: &Bound<'_, PyAny>,
    ctx: &SymContext,
    depth: u32,
    op_name: &str,
    apply: fn(&RustBV, &RustBV, &SymContext) -> RustBV,
    empty: u128,
) -> Result<RustBV, BridgeError> {
    let args_list: Vec<Bound<'_, PyAny>> = args.extract()?;
    let Some((first, rest)) = args_list.split_first() else {
        return Ok(RustBV::concrete(empty, 1));
    };
    let mut result = claripy_to_rustbv_depth(py, first, ctx, depth + 1)?;
    for arg in rest {
        let next = claripy_to_rustbv_depth(py, arg, ctx, depth + 1)?;
        check_same_width(op_name, &result, &next)?;
        result = apply(&result, &next, ctx);
    }
    Ok(result)
}

/// Import a symbol leaf from the attribute walk: the leaf of its Rust-side
/// name, the Bool tag applied ([`SymbolKind::rust_symbol_name`]), so a `BoolS`
/// and a same-named 1-bit `BVS` stay two variables (angr-9ke6b.38/.223).
/// A leaf is its name, so re-importing a symbol is the same
/// leaf (angr-mx00p.12).
fn import_symbolic_leaf(name: &str, width: u32, kind: SymbolKind, ctx: &SymContext) -> RustBV {
    RustBV::symbolic(ctx, kind.rust_symbol_name(name), width)
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
        if num_bytes == 0 {
            // The empty bitvector has no bytes to reverse, and is its own
            // reversal. The concrete branch above handles this implicitly (its
            // loop body never runs); mirror that here rather than letting the
            // unconditional `extract(7, 0)` below run out of range on a
            // width-0 value (angr-fs8kb.82).
            return Ok(bv.clone());
        }
        let mut result = bv.extract(7, 0, ctx);
        for i in 1..num_bytes {
            let byte = bv.extract((i + 1) * 8 - 1, i * 8, ctx);
            result = result.concat(&byte, ctx);
        }
        Ok(result)
    }
}

test_submod!("import_tests.rs" => import_tests);
