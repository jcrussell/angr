//! `RustBV::to_clarirs` must denote what the value's own Z3 lowering does:
//! for every `BVOp`, the lowered clarirs term and `to_z3_ast` are asserted
//! unequal and must be unsat. Operands are built as raw `Expression` nodes so
//! the construction-time folds cannot simplify the op away before it is
//! converted.

#![allow(clippy::unwrap_used, reason = "test fixtures")]

use std::sync::Arc;

use super::super::value::{BVOp, RustBV};
use super::{EXPRESSIONS_CONVERTED, bool_of};
use crate::claripy_bridge::lower_bv;
use crate::symbolic::SymContext;

fn raw(op: BVOp, width: u32, operands: Vec<RustBV>) -> RustBV {
    RustBV::Expression {
        id: RustBV::EXPRESSION_ID,
        width,
        op,
        operands: Arc::from(operands),
        memo: Default::default(),
    }
}

/// The clarirs term and the old lowering agree on every assignment.
fn assert_equivalent(value: &RustBV) {
    let old = value.to_z3_ast();
    let new = lower_bv(&value.to_clarirs().unwrap()).unwrap();
    let solver = z3::Solver::new();
    solver.assert(old.eq(&new).not());
    // satresult-exempt: test oracle; Unknown must fail the test, not pass it
    assert_eq!(solver.check(), z3::SatResult::Unsat, "disagreement on {value:?}");
}

#[test]
fn every_integer_op_denotes_its_z3_lowering() {
    let ctx = SymContext::new();
    for w in [8u32, 32, 64, 128, 200] {
        let x = RustBV::symbolic(&ctx, format!("tc_x{w}"), w);
        let y = RustBV::symbolic(&ctx, format!("tc_y{w}"), w);
        let k = RustBV::concrete(0x5a, w);
        for op in [
            BVOp::Add,
            BVOp::Sub,
            BVOp::Mul,
            BVOp::UDiv,
            BVOp::SDiv,
            BVOp::URem,
            BVOp::SRem,
            BVOp::And,
            BVOp::Or,
            BVOp::Xor,
            BVOp::Shl,
            BVOp::Lshr,
            BVOp::Ashr,
            BVOp::RotL,
            BVOp::RotR,
        ] {
            assert_equivalent(&raw(op.clone(), w, vec![x.clone(), y.clone()]));
            assert_equivalent(&raw(op, w, vec![x.clone(), k.clone()]));
        }
        for op in [BVOp::Neg, BVOp::Not, BVOp::Reverse, BVOp::Clz, BVOp::Ctz, BVOp::Popcount] {
            assert_equivalent(&raw(op, w, vec![x.clone()]));
        }
        for op in [
            BVOp::Eq,
            BVOp::Ne,
            BVOp::Ult,
            BVOp::Ule,
            BVOp::Ugt,
            BVOp::Uge,
            BVOp::Slt,
            BVOp::Sle,
            BVOp::Sgt,
            BVOp::Sge,
        ] {
            assert_equivalent(&raw(op, 1, vec![x.clone(), y.clone()]));
        }
        assert_equivalent(&raw(BVOp::ZeroExt(8), w + 8, vec![x.clone()]));
        assert_equivalent(&raw(BVOp::SignExt(8), w + 8, vec![x.clone()]));
        assert_equivalent(&raw(BVOp::Extract(w - 2, 1), w - 2, vec![x.clone()]));
        assert_equivalent(&raw(BVOp::Concat, 2 * w, vec![x.clone(), y.clone()]));
        // `Ite` reads a comparison bit, and any other condition as "nonzero".
        let lt = raw(BVOp::Ult, 1, vec![x.clone(), y.clone()]);
        assert_equivalent(&raw(BVOp::Ite, w, vec![lt, x.clone(), y.clone()]));
        assert_equivalent(&raw(BVOp::Ite, w, vec![x.clone(), x.clone(), y.clone()]));
    }
}

#[test]
fn reverse_of_a_narrow_or_unaligned_width_is_the_identity() {
    let ctx = SymContext::new();
    for w in [4u32, 8, 12] {
        let x = RustBV::symbolic(&ctx, format!("tc_rev{w}"), w);
        let rev = raw(BVOp::Reverse, w, vec![x.clone()]);
        assert_eq!(rev.to_clarirs().unwrap(), x.to_clarirs().unwrap());
        assert_equivalent(&rev);
    }
}

#[test]
fn bool_leaves_and_wide_constants_convert() {
    let ctx = SymContext::new();
    // A `!bool!` leaf is a Bool constant behind the 1-bit encoding, and the
    // condition view peels it back to that constant.
    let flag = RustBV::symbolic(&ctx, "!bool!tc_flag", 1);
    assert_equivalent(&flag);
    let term = flag.to_clarirs().unwrap();
    assert!(bool_of(&term).unwrap().ast_type().is_bool());
    assert!(matches!(
        bool_of(&term).unwrap().op(),
        clarirs_core::prelude::AstOp::BoolS(_)
    ));
    // A concrete above 128 bits keeps its low 128 bits, zero above.
    let x = RustBV::symbolic(&ctx, "tc_wide", 200);
    let wide_k = RustBV::concrete(u128::MAX, 200);
    assert_equivalent(&raw(BVOp::Add, 200, vec![x, wide_k]));
}

#[test]
fn floats_and_zero_width_values_are_refused() {
    use super::super::value::{FloatOpKind, FloatPrec};
    let ctx = SymContext::new();
    let f = RustBV::symbolic(&ctx, "tc_f", 32);
    let float = raw(
        BVOp::Float { kind: FloatOpKind::Add, prec: FloatPrec::F32 },
        32,
        vec![f.clone(), f],
    );
    assert!(float.to_clarirs().is_err());
    assert!(RustBV::concrete(0, 0).to_clarirs().is_err());
}

#[test]
fn an_anonymous_z3_wrapper_is_refused_not_renamed() {
    // `z3_ast_to_eval_bv` wraps an arbitrary Z3 term as a nameless Symbolic.
    // Converting it by name would swap that term for a fresh constant, so a
    // comparison over it must keep the Expression form and lower the real term.
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "tc_anon_x", 32);
    let anon = RustBV::Symbolic {
        id: 0,
        width: 32,
        name: Arc::from(""),
        ast: x.to_z3_ast().bvadd(z3::ast::BV::from_u64(1, 32)),
    };
    assert!(anon.to_clarirs().is_err());
    let cmp = anon.eq(&RustBV::concrete(5, 32), &ctx);
    assert!(matches!(cmp, RustBV::Expression { .. }), "{cmp:?}");
    assert_equivalent_to(&cmp, &x.to_z3_ast().bvadd(z3::ast::BV::from_u64(1, 32)).eq(z3::ast::BV::from_u64(5, 32)));
}

/// `value` lowers to a Bool-valued bit equal to `expected` on every model.
fn assert_equivalent_to(value: &RustBV, expected: &z3::ast::Bool) {
    let solver = z3::Solver::new();
    solver.assert(value.to_z3_bool().eq(expected).not());
    // satresult-exempt: test oracle; Unknown must fail the test, not pass it
    assert_eq!(solver.check(), z3::SatResult::Unsat);
}

#[test]
fn an_expression_converts_once_while_it_lives() {
    // Each op over an `Expression` operand reuses the operand's term: a chain
    // of `n` nodes converts `n` times in total, however often it is wrapped.
    let ctx = SymContext::new();
    let one = RustBV::concrete(1, 32);
    let mut acc = RustBV::symbolic(&ctx, "tc_chain", 32);
    for _ in 0..32 {
        acc = raw(BVOp::Add, 32, vec![acc, one.clone()]);
        let before = EXPRESSIONS_CONVERTED.get();
        acc.to_clarirs().unwrap();
        assert_eq!(EXPRESSIONS_CONVERTED.get() - before, 1);
    }
    let before = EXPRESSIONS_CONVERTED.get();
    acc.clone().to_clarirs().unwrap();
    assert_eq!(EXPRESSIONS_CONVERTED.get(), before);
}

#[test]
fn one_bit_logic_denotes_its_z3_lowering_and_stays_a_bool() {
    // 1-bit And/Or/Xor/Not are built on the Bools their operands stand for,
    // so the term is `ite(c, 1, 0)` with `c` a Bool connective.
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "tc_l_x", 32);
    let y = RustBV::symbolic(&ctx, "tc_l_y", 32);
    let lt = raw(BVOp::Ult, 1, vec![x.clone(), y.clone()]);
    let eq = raw(BVOp::Eq, 1, vec![x.clone(), y]);
    let flag = RustBV::symbolic(&ctx, "tc_l_flag", 1);
    for (a, b) in [(&lt, &eq), (&lt, &flag), (&flag, &flag)] {
        for op in [BVOp::And, BVOp::Or, BVOp::Xor] {
            let v = raw(op, 1, vec![a.clone(), b.clone()]);
            assert_equivalent(&v);
            assert!(super::bool_inside(&v.to_clarirs().unwrap()).is_some(), "{v:?}");
        }
        let not = raw(BVOp::Not, 1, vec![a.clone()]);
        assert_equivalent(&not);
        assert!(super::bool_inside(&not.to_clarirs().unwrap()).is_some(), "{not:?}");
    }
}
