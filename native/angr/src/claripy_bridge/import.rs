//! claripy AST -> RustBV import.
//!
//! Entry point `claripy_to_rustbv` walks the clarirs nodes behind a claripy
//! AST ([`import_clarirs`]) and builds the equivalent `RustBV`. claripy is
//! clarirs, so the `AstRef` is read directly; there is no `op` / `args`
//! attribute walk. Only this extension's own claripy type is accepted: under
//! `cargo test` the embedded interpreter's `angr.rustylib` is the installed
//! extension, whose claripy classes are different type objects, so tests
//! build their ASTs with `claripy_bridge::claripy_object` /
//! `claripy_module` (angr-vmi2l.222).

use std::cell::RefCell;
use std::collections::HashMap;

use clarirs_core::prelude::{AstOp, AstRef};
use pyo3::prelude::*;

use crate::claripy::ast::base::Base;
use crate::claripy::ast::opstring::ToOpString;

use crate::symbolic::{RustBV, SymContext, check_bv_width, check_extract_bounds};

use super::BridgeError;

/// The clarirs node behind a Python claripy AST: an AST of this extension's
/// own claripy type, or a `SimActionObject` wrapping one in `.ast`
/// (angr-vmi2l.221). `None` for anything else.
pub(crate) fn clarirs_node(ast: &Bound<'_, PyAny>) -> Option<AstRef<'static>> {
    if let Ok(base) = ast.cast::<Base>() {
        return Some(base.get().ast());
    }
    let inner = ast.getattr("ast").ok()?;
    Some(inner.cast::<Base>().ok()?.get().ast())
}

/// Try to extract a concrete BVV value directly from a claripy AST.
/// Returns Some((value, width)) if the AST is a BVV, None otherwise.
/// This is much cheaper than full claripy_to_rustbv conversion.
///
/// A width past [`check_bv_width`]'s cap, or a value wider than a `u128`,
/// yields `None` rather than the pair: this is a *fast path* for the `BVV`
/// arm of [`import_clarirs_node`], which refuses such a width, so returning
/// the pair here would let the shortcut accept what the general path rejects
/// (angr-0jh0j.8). Callers fall back to the general path, which reports the
/// refusal as a catchable error.
#[inline]
pub(crate) fn try_extract_bvv(ast: &Bound<'_, PyAny>) -> Option<(u128, u32)> {
    let node = clarirs_node(ast)?;
    let AstOp::BVV(value) = node.op() else {
        return None;
    };
    let width = value.len();
    // SILENT(cat-a): "not a plain BVV, use the general path"; an over-cap
    // width or a value past 128 bits is the same answer for the same reason.
    check_bv_width("BVV", width).ok()?;
    let value = u128::try_from(&value.to_biguint()).ok()?;
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
/// The `"ZeroExt"` / `"SignExt"` arms of [`import_clarirs_node`] compute
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
/// `ast` must be a claripy AST of this extension's type, or a
/// `SimActionObject` wrapping one ([`clarirs_node`]); anything else is
/// [`BridgeError::TypeMismatch`]. The walk is [`import_clarirs`].
///
/// `py` is part of the public API — the interpreter/solver/prefetch callers
/// pass their existing GIL token in. The body itself doesn't reach for it
/// (the `Bound` carries its own token).
pub(crate) fn claripy_to_rustbv(
    _py: Python<'_>,
    ast: &Bound<'_, PyAny>,
    ctx: &SymContext,
) -> Result<RustBV, BridgeError> {
    // GIL-work timing (angr-1ilq.7): bracket the whole conversion as one
    // region, so a caller-nested callback is a timing no-op and each disjoint
    // GIL region is counted exactly once.
    let _gil =
        crate::gil_profile::GilWorkGuard::enter_as(crate::gil_profile::GilClass::ClaripyImport);

    let node = clarirs_node(ast).ok_or_else(|| {
        let ty = ast
            .get_type()
            .name()
            .map_or_else(|_| "<unknown>".to_string(), |n| n.to_string());
        BridgeError::TypeMismatch(format!("expected a claripy AST, got {ty}"))
    })?;
    import_clarirs(&node, ctx, 0)
}

/// Maximum recursion depth for [`import_clarirs_node`]'s self-recursion
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
pub(super) const MAX_IMPORT_RECURSION_DEPTH: u32 = 4096;

/// [`claripy_to_rustbv`] over a clarirs `AstRef`. The value keeps the
/// clarirs nodes it is built from, so it exports as those nodes, annotations
/// included: a node that carries an annotation is never rebuilt operand-wise.
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
    let binary = |apply: fn(&RustBV, &RustBV) -> RustBV| {
        let (l, r) = (child(0)?, child(1)?);
        check_same_width(&op_name(), &l, &r)?;
        Ok(apply(&l, &r))
    };
    // Left fold over every operand; `empty` is the identity for an operand-less
    // Bool And/Or, and `None` makes an empty list an error.
    let fold = |apply: fn(&RustBV, &RustBV) -> RustBV, empty: Option<u128>| {
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
            acc = apply(&acc, &next);
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

    // An annotated BV node is taken whole too: rebuilding it from `RustBV`
    // ops mints a fresh, unannotated node (a literal would even collapse to
    // `RustBV::Concrete`), so a user's taint annotation would not survive
    // the round trip (angr-vmi2l.219). The node stays `RustBV::Ast` even
    // when it is a literal, since `RustBV::Concrete` has nowhere to keep
    // the annotation. A width-0 leaf keeps its degenerate import below.
    let annotated_bv = !is_bool && node.size() != 0 && !node.annotations().is_empty();
    let whole_annotated = || {
        whole().map(|value| match value {
            RustBV::Concrete { width, .. } => RustBV::Ast {
                width,
                ast: node.clone(),
            },
            kept @ RustBV::Ast { .. } => kept,
        })
    };

    match node.op() {
        _ if opaque => whole(),
        _ if annotated_bv => whole_annotated(),
        AstOp::BVV(value) => {
            let width = value.len();
            check_bv_width("BVV", width).map_err(BridgeError::InvalidArgs)?;
            // RustBV::Concrete holds a u128. A literal whose value needs more
            // is taken whole as its clarirs term, the shape `RustBV::concat`
            // builds for a wide concrete result, never truncated
            // (angr-cxw7). A wide x87 `fpreg` write from the Python VEX
            // fallback is one (angr-vmi2l.170).
            match u128::try_from(&value.to_biguint()) {
                Ok(v) => Ok(RustBV::concrete(v, width)),
                Err(_) => whole(),
            }
        }
        // A clarirs leaf is taken whole: interning makes it the one node with
        // that name, the variable Z3 binds, and keeps its annotations
        // (angr-mx00p.12). A BoolS comes back as `ite(b, 1, 0)`.
        AstOp::BVS(_, width) => {
            check_bv_width("BVS", *width).map_err(BridgeError::InvalidArgs)?;
            if *width == 0 {
                // The empty bitvector's only value, as `RustBV::symbol`
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
        AstOp::Neg(_) => child(0).map(|v| v.neg()),
        AstOp::Not(_) => child(0).map(|v| v.not()),
        AstOp::ITE(..) => {
            let (cond, then_val, else_val) = (child(0)?, child(1)?, child(2)?);
            check_same_width("If", &then_val, &else_val)?;
            Ok(cond.ite(&then_val, &else_val))
        }
        AstOp::ZeroExt(_, bits) => {
            let val = child(0)?;
            let new_width = extended_width("ZeroExt", val.width(), *bits)?;
            Ok(val.zero_extend(new_width))
        }
        AstOp::SignExt(_, bits) => {
            let val = child(0)?;
            let new_width = extended_width("SignExt", val.width(), *bits)?;
            Ok(val.sign_extend(new_width))
        }
        AstOp::Extract(_, high, low) => {
            let val = child(0)?;
            check_extract_bounds("Extract", *high, *low, val.width())
                .map_err(BridgeError::InvalidArgs)?;
            Ok(val.extract(*high, *low))
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
            Ok(RustBV::concat_balanced(&parts))
        }
        AstOp::ByteReverse(_) => reverse_bytes(&child(0)?),
        _ => whole(),
    }
}

/// Byte-reverse a RustBV value.
fn reverse_bytes(bv: &RustBV) -> Result<RustBV, BridgeError> {
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
        let mut result = bv.extract(7, 0);
        for i in 1..num_bytes {
            let byte = bv.extract((i + 1) * 8 - 1, i * 8);
            result = result.concat(&byte);
        }
        Ok(result)
    }
}

test_submod!("import_tests.rs" => import_tests);
