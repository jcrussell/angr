//! `RustBV::node` must read a clarirs `Ast` the way it reads the `Expression`
//! it was converted from, and the construction-time rewrites that inspect
//! operands through it must fire on `Ast` operands too.

#![allow(clippy::unwrap_used, reason = "test fixtures")]

use std::sync::Arc;

use clarirs_core::prelude::AstFactory;

use super::super::value::{BVOp, RustBV};
use crate::claripy::ast::GLOBAL_CONTEXT;
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

/// The same value held as a clarirs term.
fn ast(value: &RustBV) -> RustBV {
    RustBV::from_clarirs_subterm(value.to_clarirs().unwrap())
}

/// Two values denote the same clarirs term.
fn same(a: &RustBV, b: &RustBV) -> bool {
    a.to_clarirs().unwrap() == b.to_clarirs().unwrap()
}

#[test]
fn an_ast_reads_like_the_expression_it_came_from() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "shape_x", 32);
    let y = RustBV::symbolic(&ctx, "shape_y", 32);
    let mut values = vec![];
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
        values.push(raw(op, 32, vec![x.clone(), y.clone()]));
    }
    for op in [BVOp::Neg, BVOp::Not, BVOp::Reverse] {
        values.push(raw(op, 32, vec![x.clone()]));
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
        values.push(raw(op, 1, vec![x.clone(), y.clone()]));
    }
    values.push(raw(BVOp::ZeroExt(8), 40, vec![x.clone()]));
    values.push(raw(BVOp::SignExt(8), 40, vec![x.clone()]));
    values.push(raw(BVOp::Extract(15, 4), 12, vec![x.clone()]));
    values.push(raw(BVOp::Concat, 64, vec![x.clone(), y.clone()]));
    let lt = raw(BVOp::Ult, 1, vec![x.clone(), y.clone()]);
    values.push(raw(BVOp::Ite, 32, vec![lt, x.clone(), y.clone()]));

    for expr in &values {
        let held = ast(expr);
        assert!(matches!(held, RustBV::Ast { .. }), "{expr:?} converted to {held:?}");
        let (want, got) = (expr.node().unwrap(), held.node().unwrap());
        assert_eq!(want.op, got.op, "op of {expr:?}");
        assert_eq!(want.operands.len(), got.operands.len(), "arity of {expr:?}");
        for (w, g) in want.operands.iter().zip(got.operands.iter()) {
            assert!(same(w, g), "operand of {expr:?}: {w:?} vs {g:?}");
        }
    }
}

#[test]
fn literal_operands_of_an_ast_are_concrete() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "shape_lit", 32);
    let sum = ast(&raw(BVOp::Add, 32, vec![x, RustBV::concrete(7, 32)]));
    let node = sum.node().unwrap();
    assert_eq!(node.operands[1].as_u128(), Some(7));
    assert!(RustBV::concrete(3, 8).node().is_none());
}

#[test]
fn a_bool_op_under_the_bit_encoding_reads_as_the_bitwise_op() {
    let gc = &*GLOBAL_CONTEXT;
    let x = gc.bvs("shape_bx", 8).unwrap();
    let y = gc.bvs("shape_by", 8).unwrap();
    let both = gc
        .and2(gc.ult(x.clone(), y.clone()).unwrap(), gc.ugt(x, y).unwrap())
        .unwrap();
    let bit = RustBV::from_clarirs(both).unwrap();
    let node = bit.node().unwrap();
    assert_eq!(node.op, BVOp::And);
    let ops: Vec<_> = node.operands.iter().map(|b| b.node().unwrap().op).collect();
    assert_eq!(ops, [BVOp::Ult, BVOp::Ugt]);
    // A Bool leaf behind the encoding is a leaf.
    let flag = RustBV::from_clarirs(gc.bools("shape_flag").unwrap()).unwrap();
    assert!(flag.node().is_none());
}

#[test]
fn cancelling_rewrites_see_through_an_ast_operand() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "shape_cancel", 32);
    let neg = ast(&raw(BVOp::Neg, 32, vec![x.clone()]));
    assert!(same(&neg.neg(&ctx), &x));
    let not = ast(&raw(BVOp::Not, 32, vec![x.clone()]));
    assert!(same(&not.not(&ctx), &x));
    let rev = ast(&raw(BVOp::Reverse, 32, vec![x.clone()]));
    assert!(same(&rev.reverse(&ctx), &x));
}

#[test]
fn extract_rules_see_through_an_ast_operand() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "shape_ex", 16);
    let y = RustBV::symbolic(&ctx, "shape_ey", 16);
    // Rule 2: extract of a concat lands in one part.
    let cat = ast(&raw(BVOp::Concat, 32, vec![x.clone(), y.clone()]));
    assert!(same(&cat.extract(15, 0, &ctx), &y));
    assert!(same(&cat.extract(31, 16, &ctx), &x));
    // Rule 1: extract of an extract fuses.
    let inner = ast(&raw(BVOp::Extract(11, 4), 8, vec![x.clone()]));
    let fused = inner.extract(3, 0, &ctx);
    assert!(same(&fused, &raw(BVOp::Extract(7, 4), 4, vec![x.clone()])));
    // Rule 4: extract of a zero extension is the inner value, or zero.
    let wide = ast(&raw(BVOp::ZeroExt(16), 32, vec![x.clone()]));
    assert!(same(&wide.extract(15, 0, &ctx), &x));
    assert_eq!(wide.extract(31, 16, &ctx).as_u128(), Some(0));
    // Rule 5: extract of the low bits of a sign extension.
    let sext = ast(&raw(BVOp::SignExt(16), 32, vec![x.clone()]));
    assert!(same(&sext.extract(7, 0, &ctx), &raw(BVOp::Extract(7, 0), 8, vec![x])));
}

#[test]
fn zext_compare_fold_sees_through_an_ast_operand() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "shape_zc", 8);
    let wide = ast(&raw(BVOp::ZeroExt(24), 32, vec![x.clone()]));
    // The constant has bits the extension forces to zero: decided outright.
    assert_eq!(wide.eq(&RustBV::concrete(0x100, 32), &ctx).as_u128(), Some(0));
    // Otherwise the compare narrows to the inner width.
    let narrowed = wide.eq(&RustBV::concrete(0x41, 32), &ctx);
    let node = narrowed.node().unwrap();
    assert_eq!(node.op, BVOp::Eq);
    assert!(same(&node.operands[0], &x));
    assert_eq!(node.operands[1].width(), 8);
}
