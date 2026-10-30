//! The encodings `value_clarirs::build_term` gives ops and `to_clarirs` gives
//! values. Nodes are built with `RustBV::from_op`, which skips the
//! construction-time folds, so each op is seen as built.

use super::super::value::{BVOp, RustBV};
use super::bool_of;

fn raw(op: BVOp, width: u32, operands: Vec<RustBV>) -> RustBV {
    RustBV::from_op(width, op, operands)
}

#[test]
fn reverse_of_a_narrow_or_unaligned_width_is_the_identity() {
    for w in [4u32, 8, 12] {
        let x = RustBV::symbol(format!("tc_rev{w}"), w);
        let rev = raw(BVOp::Reverse, w, vec![x.clone()]);
        assert_eq!(rev.to_clarirs().unwrap(), x.to_clarirs().unwrap());
    }
}

#[test]
fn bool_leaves_convert() {
    // A `!bool!` leaf is a Bool constant behind the 1-bit encoding, and the
    // condition view peels it back to that constant.
    let flag = RustBV::symbol("!bool!tc_flag", 1);
    let term = flag.to_clarirs().unwrap();
    assert!(bool_of(&term).unwrap().ast_type().is_bool());
    assert!(matches!(
        bool_of(&term).unwrap().op(),
        clarirs_core::prelude::AstOp::BoolS(_)
    ));
}

#[test]
fn zero_width_values_are_refused() {
    assert!(RustBV::concrete(0, 0).to_clarirs().is_err());
}

#[test]
fn one_bit_logic_stays_a_bool() {
    // 1-bit And/Or/Xor/Not are built on the Bools their operands stand for,
    // so the term is `ite(c, 1, 0)` with `c` a Bool connective.
    let x = RustBV::symbol("tc_l_x", 32);
    let y = RustBV::symbol("tc_l_y", 32);
    let lt = raw(BVOp::Ult, 1, vec![x.clone(), y.clone()]);
    let eq = raw(BVOp::Eq, 1, vec![x.clone(), y]);
    let flag = RustBV::symbol("tc_l_flag", 1);
    for (a, b) in [(&lt, &eq), (&lt, &flag), (&flag, &flag)] {
        for op in [BVOp::And, BVOp::Or, BVOp::Xor] {
            let v = raw(op, 1, vec![a.clone(), b.clone()]);
            assert!(
                super::bool_inside(&v.to_clarirs().unwrap()).is_some(),
                "{v:?}"
            );
        }
        let not = raw(BVOp::Not, 1, vec![a.clone()]);
        assert!(
            super::bool_inside(&not.to_clarirs().unwrap()).is_some(),
            "{not:?}"
        );
    }
}
