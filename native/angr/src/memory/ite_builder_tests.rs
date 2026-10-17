//! `loads_match` decides when an N-way load collapses to one value; it must
//! recognize equal clarirs terms, and a leaf held in either arm.

#![allow(clippy::unwrap_used, reason = "test fixtures")]

use super::loads_match;
use crate::symbolic::{RustBV, SymContext};

fn as_ast(value: &RustBV) -> RustBV {
    RustBV::from_clarirs_subterm(value.to_clarirs().unwrap())
}

#[test]
fn equal_clarirs_terms_match_and_different_ones_do_not() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "ld_x", 32);
    let a = as_ast(&x.add(&RustBV::concrete(4, 32), &ctx));
    let b = as_ast(&x.add(&RustBV::concrete(4, 32), &ctx));
    let c = as_ast(&x.add(&RustBV::concrete(8, 32), &ctx));
    assert!(loads_match(&a, &b));
    assert!(!loads_match(&a, &c));
}

#[test]
fn a_leaf_matches_itself_across_arms() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "ld_leaf", 32);
    assert!(loads_match(&x, &as_ast(&x)));
    let y = RustBV::symbolic(&ctx, "ld_other", 32);
    assert!(!loads_match(&x, &as_ast(&y)));
}
