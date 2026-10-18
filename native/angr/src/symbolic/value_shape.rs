//! A representation-independent view of a `RustBV` node (angr-mx00p.12).
//!
//! [`RustBV::node`] gives the op and operands of a compound value whether it
//! is held as a `BVOp` [`RustBV::Expression`] or as a clarirs
//! [`RustBV::Ast`], so construction-time rewrites and the fast paths that
//! inspect operands work on both. An `Expression`'s operands are borrowed; an
//! `Ast`'s children are wrapped as values, a literal that fits a `u128` as
//! [`RustBV::Concrete`] and anything else as an `Ast`.
//!
//! The view reads an `Ast` in the shapes `Expression` already has for the
//! engine's Bool encoding (see `value_clarirs.rs`): `ite(cmp, 1, 0)` is the
//! 1-bit comparison itself, a Bool `And` / `Or` / `Xor` / `Not` under that
//! encoding is the 1-bit bitwise op over its operands' bits, and a Bool
//! operand anywhere else (an `ITE` condition) is presented as its bit.

use std::borrow::Cow;

use clarirs_core::prelude::{AstOp, AstRef};
use num_bigint::BigUint;

use super::value::{BVOp, RustBV};
use super::value_clarirs::bit_of;

type Term = AstRef<'static>;

/// The op and operands of a compound value. See the module doc.
pub(crate) struct Node<'a> {
    pub(crate) op: BVOp,
    pub(crate) operands: Cow<'a, [RustBV]>,
}

impl RustBV {
    /// The op and operands of this value, or `None` for a leaf or constant.
    pub(crate) fn node(&self) -> Option<Node<'_>> {
        match self {
            RustBV::Expression { op, operands, .. } => Some(Node {
                op: op.clone(),
                operands: Cow::Borrowed(&operands[..]),
            }),
            RustBV::Ast { ast, .. } => ast_node(ast).map(|(op, operands)| Node {
                op,
                operands: Cow::Owned(operands),
            }),
            RustBV::Concrete { .. } | RustBV::Symbolic { .. } => None,
        }
    }

    /// The variable a leaf value is, as `(name, width)`: a `Symbolic`, or an
    /// `Ast` that is a `BVS` or a Bool symbol under the bit encoding. A Bool
    /// symbol is named the way a `Symbolic` Bool leaf is, so both arms agree.
    /// Two leaves with the same name and width are one Z3 constant
    /// (`invariant-symbol-identity-is-by-name`).
    #[cfg_attr(not(feature = "vex-engine"), allow(dead_code))]
    pub(crate) fn leaf_var(&self) -> Option<(std::sync::Arc<str>, u32)> {
        use super::registry::SymbolKind;
        use std::sync::Arc;
        match self {
            RustBV::Symbolic { name, width, .. } => Some((name.clone(), *width)),
            RustBV::Ast { ast, width, .. } => match ast.op() {
                AstOp::BVS(name, w) => Some((Arc::from(name.as_str()), *w)),
                AstOp::ITE(c, t, e) if *width == 1 && is_bvv(t, 1) && is_bvv(e, 0) => {
                    match c.op() {
                        AstOp::BoolS(name) => Some((
                            Arc::from(&*SymbolKind::Bool.rust_symbol_name(name.as_str())),
                            1,
                        )),
                        _ => None,
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Wrap a bitvector-sorted term known to lower: a literal that fits a
    /// `u128` as `Concrete`, anything else as an `Ast` lowered on demand. A
    /// subterm of a lowered term qualifies (lowering the parent lowered it),
    /// and so does a term the constructors build over lowerable operands with
    /// the ops `value_clarirs::build_term` emits, so the lazy lowering cannot
    /// fail.
    pub(crate) fn from_clarirs_lowerable(ast: Term) -> RustBV {
        if let AstOp::BVV(value) = ast.op()
            && let Ok(v) = u128::try_from(&value.to_biguint())
        {
            return RustBV::concrete(v, value.len());
        }
        RustBV::Ast {
            width: ast.size(),
            ast,
            memo: Default::default(),
        }
    }
}

fn wrap(ast: &Term) -> RustBV {
    RustBV::from_clarirs_lowerable(ast.clone())
}

/// A Bool operand as the engine's 1-bit encoding.
fn wrap_bool(b: &Term) -> Option<RustBV> {
    bit_of(b.clone()).ok().map(RustBV::from_clarirs_lowerable)
}

fn wrap_all<'a>(children: impl IntoIterator<Item = &'a Term>) -> Vec<RustBV> {
    children.into_iter().map(wrap).collect()
}

fn is_bvv(term: &Term, value: u64) -> bool {
    matches!(term.op(), AstOp::BVV(v) if v.to_biguint() == BigUint::from(value))
}

/// The view of a bitvector-sorted clarirs node.
fn ast_node(ast: &Term) -> Option<(BVOp, Vec<RustBV>)> {
    let pair = |a: &Term, b: &Term| vec![wrap(a), wrap(b)];
    Some(match ast.op() {
        AstOp::Add(args) => (BVOp::Add, wrap_all(args)),
        AstOp::Mul(args) => (BVOp::Mul, wrap_all(args)),
        AstOp::And(args) => (BVOp::And, wrap_all(args)),
        AstOp::Or(args) => (BVOp::Or, wrap_all(args)),
        AstOp::Xor(args) => (BVOp::Xor, wrap_all(args)),
        AstOp::Concat(args) => (BVOp::Concat, wrap_all(args)),
        AstOp::Sub(a, b) => (BVOp::Sub, pair(a, b)),
        AstOp::UDiv(a, b) => (BVOp::UDiv, pair(a, b)),
        AstOp::SDiv(a, b) => (BVOp::SDiv, pair(a, b)),
        AstOp::URem(a, b) => (BVOp::URem, pair(a, b)),
        AstOp::SRem(a, b) => (BVOp::SRem, pair(a, b)),
        AstOp::ShL(a, b) => (BVOp::Shl, pair(a, b)),
        AstOp::LShR(a, b) => (BVOp::Lshr, pair(a, b)),
        AstOp::AShR(a, b) => (BVOp::Ashr, pair(a, b)),
        AstOp::RotateLeft(a, b) => (BVOp::RotL, pair(a, b)),
        AstOp::RotateRight(a, b) => (BVOp::RotR, pair(a, b)),
        AstOp::Neg(a) => (BVOp::Neg, vec![wrap(a)]),
        AstOp::Not(a) => (BVOp::Not, vec![wrap(a)]),
        AstOp::ByteReverse(a) => (BVOp::Reverse, vec![wrap(a)]),
        AstOp::ZeroExt(a, bits) => (BVOp::ZeroExt(*bits), vec![wrap(a)]),
        AstOp::SignExt(a, bits) => (BVOp::SignExt(*bits), vec![wrap(a)]),
        AstOp::Extract(a, high, low) => (BVOp::Extract(*high, *low), vec![wrap(a)]),
        AstOp::ITE(c, t, e) if ast.size() == 1 && is_bvv(t, 1) && is_bvv(e, 0) => bool_node(c)?,
        AstOp::ITE(c, t, e) => (BVOp::Ite, vec![wrap_bool(c)?, wrap(t), wrap(e)]),
        _ => return None,
    })
}

/// The 1-bit view of a Bool `b` held as `ite(b, 1, 0)`. `None` for a Bool
/// leaf or literal, and for a Bool op with no 1-bit `BVOp` form.
fn bool_node(b: &Term) -> Option<(BVOp, Vec<RustBV>)> {
    let cmp = |op: BVOp, l: &Term, r: &Term| Some((op, vec![wrap(l), wrap(r)]));
    let bits = |args: &[Term]| args.iter().map(wrap_bool).collect::<Option<Vec<_>>>();
    match b.op() {
        AstOp::Eq(l, r) if l.ast_type().is_bitvec() => cmp(BVOp::Eq, l, r),
        AstOp::Neq(l, r) if l.ast_type().is_bitvec() => cmp(BVOp::Ne, l, r),
        AstOp::ULT(l, r) => cmp(BVOp::Ult, l, r),
        AstOp::ULE(l, r) => cmp(BVOp::Ule, l, r),
        AstOp::UGT(l, r) => cmp(BVOp::Ugt, l, r),
        AstOp::UGE(l, r) => cmp(BVOp::Uge, l, r),
        AstOp::SLT(l, r) => cmp(BVOp::Slt, l, r),
        AstOp::SLE(l, r) => cmp(BVOp::Sle, l, r),
        AstOp::SGT(l, r) => cmp(BVOp::Sgt, l, r),
        AstOp::SGE(l, r) => cmp(BVOp::Sge, l, r),
        AstOp::And(args) => Some((BVOp::And, bits(args)?)),
        AstOp::Or(args) => Some((BVOp::Or, bits(args)?)),
        AstOp::Xor(args) => Some((BVOp::Xor, bits(args)?)),
        AstOp::Not(a) => Some((BVOp::Not, vec![wrap_bool(a)?])),
        _ => None,
    }
}

test_submod!("value_shape_tests.rs" => tests);
