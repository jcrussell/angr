//! `SymBV::Ast`: clarirs terms held whole (angr-mx00p.11).

use std::sync::Arc;

use clarirs_core::prelude::*;

use super::*;
use clarirs_core::GLOBAL_CONTEXT;

fn ctx() -> &'static Context<'static> {
    &GLOBAL_CONTEXT
}

fn ast_of(bv: &SymBV) -> &AstRef<'static> {
    match bv {
        SymBV::Ast { ast, .. } => ast,
        other => panic!("expected SymBV::Ast, got {other:?}"),
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
    let as_bv = SymBV::from_clarirs(b.clone()).unwrap();
    assert_eq!(as_bv.width(), 1);
    assert!(matches!(ast_of(&as_bv).op(), AstOp::ITE(c, ..) if Arc::ptr_eq(c, &b)));

    let f = ctx().fps("ast_f", FSort::f64()).unwrap();
    let bits = SymBV::from_clarirs(f.clone()).unwrap();
    assert_eq!(bits.width(), 64);
    assert!(matches!(ast_of(&bits).op(), AstOp::FpToIEEEBV(inner) if Arc::ptr_eq(inner, &f)));

    let literal = ctx().bvv(BitVec::from((0x1234, 16))).unwrap();
    assert_eq!(
        SymBV::from_clarirs(literal).unwrap(),
        SymBV::concrete(0x1234, 16)
    );

    assert!(SymBV::from_clarirs(ctx().strings("ast_s").unwrap()).is_err());
}

#[test]
fn equality_is_node_identity() {
    let (_, eq) = nan_probe();
    let a = SymBV::from_clarirs(eq.clone()).unwrap();
    let b = SymBV::from_clarirs(eq).unwrap();
    assert_eq!(a, b, "the same interned node");
    let other = SymBV::from_clarirs(ctx().bools("ast_other").unwrap()).unwrap();
    assert_ne!(a, other);
}

#[test]
fn serde_rejects_a_non_bitvector_root() {
    let json = serde_json::to_string(&SymBVData::Ast {
        dag: clarirs_core::ast::wire::WireDag::encode(&[ctx().bools("ast_root").unwrap()]),
    })
    .unwrap();
    assert!(serde_json::from_str::<SymBV>(&json).is_err());
}
