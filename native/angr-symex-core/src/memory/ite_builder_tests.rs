//! `loads_match` decides when an N-way load collapses to one value; it must
//! recognize equal clarirs terms, and a leaf held in either arm.

#![allow(clippy::unwrap_used, reason = "test fixtures")]

use super::loads_match;
use crate::symbolic::RustBV;

fn as_ast(value: &RustBV) -> RustBV {
    RustBV::from_clarirs_lowerable(value.to_clarirs().unwrap())
}

#[test]
fn equal_clarirs_terms_match_and_different_ones_do_not() {
    let x = RustBV::symbol("ld_x", 32);
    let a = as_ast(&x.add(&RustBV::concrete(4, 32)));
    let b = as_ast(&x.add(&RustBV::concrete(4, 32)));
    let c = as_ast(&x.add(&RustBV::concrete(8, 32)));
    assert!(loads_match(&a, &b));
    assert!(!loads_match(&a, &c));
}

#[test]
fn a_leaf_matches_itself_across_arms() {
    let x = RustBV::symbol("ld_leaf", 32);
    assert!(loads_match(&x, &as_ast(&x)));
    let y = RustBV::symbol("ld_other", 32);
    assert!(!loads_match(&x, &as_ast(&y)));
}
