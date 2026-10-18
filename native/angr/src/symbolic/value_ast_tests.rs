//! `RustBV::Ast`: clarirs terms held whole (angr-mx00p.11).

use std::sync::Arc;

use clarirs_core::prelude::*;

use super::*;
use crate::claripy::ast::GLOBAL_CONTEXT;

fn ctx() -> &'static Context<'static> {
    &GLOBAL_CONTEXT
}

fn ast_of(bv: &RustBV) -> &AstRef<'static> {
    match bv {
        RustBV::Ast { ast, .. } => ast,
        other => panic!("expected RustBV::Ast, got {other:?}"),
    }
}

/// `x` as an f32, with `fp.eq(x, x)` and its bit-pattern lookalike.
fn nan_probe() -> (AstRef<'static>, AstRef<'static>) {
    let x = ctx().fps("ast_nan_x", FSort::f32()).unwrap();
    (x.clone(), ctx().eq_(&x, &x).unwrap())
}

#[test]
fn from_clarirs_encodes_each_sort_as_a_bitvector() {
    let b = ctx().bools("ast_b").unwrap();
    let as_bv = RustBV::from_clarirs(b.clone()).unwrap();
    assert_eq!(as_bv.width(), 1);
    assert!(matches!(ast_of(&as_bv).op(), AstOp::ITE(c, ..) if Arc::ptr_eq(c, &b)));

    let f = ctx().fps("ast_f", FSort::f64()).unwrap();
    let bits = RustBV::from_clarirs(f.clone()).unwrap();
    assert_eq!(bits.width(), 64);
    assert!(matches!(ast_of(&bits).op(), AstOp::FpToIEEEBV(inner) if Arc::ptr_eq(inner, &f)));

    let literal = ctx().bvv(BitVec::from((0x1234, 16))).unwrap();
    assert_eq!(RustBV::from_clarirs(literal).unwrap(), RustBV::concrete(0x1234, 16));

    assert!(RustBV::from_clarirs(ctx().strings("ast_s").unwrap()).is_err());
}

#[test]
fn from_clarirs_refuses_terms_without_a_z3_form() {
    let x = ctx().bvs("ast_vsa_x", 32).unwrap();
    let union = ctx().union([x.clone(), x]).unwrap();
    assert!(RustBV::from_clarirs(union).is_err());
}

#[test]
fn ast_keeps_float_semantics_in_the_solver() {
    // `fp.eq(x, x)` is false for NaN; a bit-pattern comparison never is.
    let sym = SymContext::new_mock();
    let (_, eq) = nan_probe();
    let cond = RustBV::from_clarirs(eq).unwrap();
    assert!(sym.can_be_false(&cond), "NaN must make fp.eq(x, x) false");
    assert!(sym.can_be_true(&cond));
}

#[test]
fn ast_operands_compose_with_engine_ops() {
    let sym = SymContext::new_mock();
    let (x, _) = nan_probe();
    let bits = RustBV::from_clarirs(x).unwrap();
    // The engine builds an Expression over the Ast leaf; the solver sees both.
    let sign = bits.extract(31, 31, &sym);
    let positive_nan_bits = RustBV::concrete(0x7fc0_0000, 32);
    let cond = bits.eq(&positive_nan_bits, &sym);
    assert!(sym.can_be_true(&cond));
    assert_eq!(sign.width(), 1);
    assert!(sym.can_be_true(&sign.eq(&RustBV::concrete(1, 1), &sym)));
}

#[test]
fn equality_is_node_identity() {
    let (_, eq) = nan_probe();
    let a = RustBV::from_clarirs(eq.clone()).unwrap();
    let b = RustBV::from_clarirs(eq).unwrap();
    assert_eq!(a, b, "the same interned node");
    let other = RustBV::from_clarirs(ctx().bools("ast_other").unwrap()).unwrap();
    assert_ne!(a, other);
}

#[test]
fn serde_round_trip_is_the_same_node() {
    let (_, eq) = nan_probe();
    let bv = RustBV::from_clarirs(eq).unwrap();
    let json = serde_json::to_string(&bv).unwrap();
    let back: RustBV = serde_json::from_str(&json).unwrap();
    assert!(Arc::ptr_eq(ast_of(&bv), ast_of(&back)));
    assert_eq!(back.width(), 1);
    // The decoded value lowers afresh and still solves the same way.
    let sym = SymContext::new_mock();
    assert!(sym.can_be_false(&back));
}

#[test]
fn serde_rejects_a_non_bitvector_root() {
    let json = serde_json::to_string(&RustBVData::Ast {
        dag: clarirs_core::ast::wire::WireDag::encode(&[ctx().bools("ast_root").unwrap()]),
    })
    .unwrap();
    assert!(serde_json::from_str::<RustBV>(&json).is_err());
}

#[test]
fn serde_rejects_a_term_without_a_z3_form() {
    // A crafted payload must not smuggle in what `from_clarirs` refuses: the
    // solver lowers every `RustBV::Ast` on the promise that it was checked.
    let x = ctx().bvs("ast_wire_vsa_x", 32).unwrap();
    let json = serde_json::to_string(&RustBVData::Ast {
        dag: clarirs_core::ast::wire::WireDag::encode(&[ctx().union([x.clone(), x]).unwrap()]),
    })
    .unwrap();
    assert!(serde_json::from_str::<RustBV>(&json).is_err());
}

#[test]
fn a_comparison_bit_lowers_to_the_native_comparison() {
    // `ite(x < y, 1, 0)` must reach Z3 as `x < y` itself, not as a bitvector
    // ite compared with 1: the same Z3 term the engine builds by hand.
    use z3::ast::BV;
    let x = ctx().bvs("ast_cmp_x", 32).unwrap();
    let y = ctx().bvs("ast_cmp_y", 32).unwrap();
    let bit = RustBV::from_clarirs(ctx().ult(x, y).unwrap()).unwrap();
    let native = BV::new_const("ast_cmp_x", 32).bvult(BV::new_const("ast_cmp_y", 32));
    assert_eq!(bit.to_z3_bool(), native);
    // A bit with no comparison under it is `bit != 0`, which still decides.
    let flag = RustBV::from_clarirs(ctx().bools("ast_cmp_flag").unwrap()).unwrap();
    assert_eq!(flag.to_z3_bool(), z3::ast::Bool::new_const("ast_cmp_flag"));
}
