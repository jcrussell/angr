//! Core `SymBV` construction/arithmetic tests: concrete folds, extend /
//! truncate / extract / concat shaping, and comparison ops.
//!
//! Sibling modules cover the rest of `value.rs`'s surface — see
//! `value_simplify_tests.rs`, `value_zext_cmp_tests.rs`,
//! `value_serde_tests.rs`, `value_concrete_arm_tests.rs` and
//! `value_eq_debug_tests.rs`.

use super::value_tests_support::bv_shape;
use super::*;

#[test]
fn test_concrete_add() {
    let a = SymBV::concrete(5, 32);
    let b = SymBV::concrete(3, 32);
    let result = a.add(&b);
    assert_eq!(result.as_u64(), Some(8));
}

#[test]
fn test_concrete_sub() {
    let a = SymBV::concrete(10, 32);
    let b = SymBV::concrete(3, 32);
    let result = a.sub(&b);
    assert_eq!(result.as_u64(), Some(7));
}

#[test]
fn test_concrete_overflow() {
    let a = SymBV::concrete(0xFF, 8);
    let b = SymBV::concrete(1, 8);
    let result = a.add(&b);
    assert_eq!(result.as_u64(), Some(0)); // 8-bit overflow wraps to 0
}

#[test]
fn test_sign_extend() {
    let a = SymBV::concrete(0xFF, 8); // -1 in 8 bits
    let result = a.sign_extend(32);
    assert_eq!(result.as_u64(), Some(0xFFFFFFFF)); // -1 in 32 bits
}

#[test]
fn test_zero_extend() {
    let a = SymBV::concrete(0xFF, 8);
    let result = a.zero_extend(32);
    assert_eq!(result.as_u64(), Some(0xFF));
}

#[test]
fn test_truncate() {
    let a = SymBV::concrete(0x12345678, 32);
    let result = a.truncate(8);
    assert_eq!(result.as_u64(), Some(0x78));
}

#[test]
fn test_extract() {
    let a = SymBV::concrete(0xABCD, 16);
    let result = a.extract(11, 4);
    assert_eq!(result.width(), 8);
    assert_eq!(result.as_u64(), Some(0xBC));
}

#[test]
fn test_extract_high_bits_of_wide_concrete_are_zero() {
    // A Concrete stores its value in a u128, so bits at positions >= 128 are
    // logically zero even for a width-256 value. extract must guard the
    // shift/mask so it returns 0 (not a u128 shift-overflow abort / mod-128
    // wrap) when the requested window lands entirely in those high bits.
    let wide = SymBV::concrete(u128::MAX, 256);
    // Window [143:136] is entirely above bit 128 -> all logically-zero bits.
    let hi = wide.extract(143, 136);
    assert_eq!(hi.width(), 8);
    assert_eq!(hi.as_u128(), Some(0));
    // low >= 128 alone: window [200:193] also fully in the zero region.
    let hi2 = wide.extract(200, 193);
    assert_eq!(hi2.as_u128(), Some(0));
    // Low window still returns the real bits.
    let lo = wide.extract(7, 0);
    assert_eq!(lo.as_u128(), Some(0xFF));
    // A result_width >= 128 window over the low 128 real bits stays exact.
    let full_lo = wide.extract(127, 0);
    assert_eq!(full_lo.width(), 128);
    assert_eq!(full_lo.as_u128(), Some(u128::MAX));
}

#[test]
fn test_extract_canonicalizes_through_drive_extract() {
    // Every Extract entry point routes through drive_extract's rules-1-5 pass
    // (angr-9ke6b.237).
    let a = SymBV::symbol("a", 8);
    let b = SymBV::symbol("b", 8);
    let cat = a.concat(&b);

    // Rule 2: Extract entirely within one Concat part collapses to that part.
    assert_eq!(bv_shape(&cat.extract(7, 0)), bv_shape(&b));
    assert_eq!(bv_shape(&cat.extract(15, 8)), bv_shape(&a));
    assert_eq!(bv_shape(&cat.extract(7, 0)), bv_shape(&cat.extract(7, 0)));

    // Rule 1: Extract-of-Extract folds into a single Extract of the base.
    let wide = SymBV::symbol("w", 32);
    let folded = wide.extract(23, 8).extract(7, 0);
    assert_eq!(bv_shape(&folded), bv_shape(&wide.extract(15, 8)));
    assert_eq!(bv_shape(&folded), "Expr(Extract(15, 8), 8, [Leaf(w, 32)])");

    // Rule 4: Extract above a ZeroExt's original width is the zero constant.
    let top = b.zero_extend(32).extract(31, 24);
    assert_eq!(top, SymBV::zero(8));
}

#[test]
fn test_concat() {
    let hi = SymBV::concrete(0xAB, 8);
    let lo = SymBV::concrete(0xCD, 8);
    let result = hi.concat(&lo);
    assert_eq!(result.width(), 16);
    assert_eq!(result.as_u64(), Some(0xABCD));
}

/// Helper: recursive depth of an expression AST. Concrete/symbolic
/// leaves have depth 0; every op node adds one.
fn ast_depth(bv: &SymBV) -> u32 {
    match bv.node() {
        Some(node) => 1 + node.operands.iter().map(ast_depth).max().unwrap_or(0),
        None => 0,
    }
}

#[test]
fn test_concat_balanced_single() {
    let parts = [SymBV::symbol("a", 8)];
    let result = SymBV::concat_balanced(&parts);
    assert_eq!(result.width(), 8);
}

#[test]
fn test_concat_balanced_pair() {
    let hi = SymBV::concrete(0xAB, 8);
    let lo = SymBV::concrete(0xCD, 8);
    let result = SymBV::concat_balanced(&[hi, lo]);
    assert_eq!(result.width(), 16);
    assert_eq!(result.as_u64(), Some(0xABCD));
}

#[test]
fn test_concat_balanced_concrete_value() {
    // Build 0xDEADBEEF byte-by-byte (high to low) and check the value.
    let parts: Vec<SymBV> = [0xDE, 0xAD, 0xBE, 0xEF]
        .iter()
        .map(|&b| SymBV::concrete(b, 8))
        .collect();
    let result = SymBV::concat_balanced(&parts);
    assert_eq!(result.width(), 32);
    assert_eq!(result.as_u64(), Some(0xDEADBEEF));
}

#[test]
fn test_concat_balanced_depth_is_log() {
    // 8 symbolic bytes → linear chain would have depth 7; balanced
    // tree should be 3 (log2(8)).
    let parts: Vec<SymBV> = (0..8).map(|i| SymBV::symbol(format!("b{i}"), 8)).collect();
    let balanced = SymBV::concat_balanced(&parts);
    assert_eq!(balanced.width(), 64);
    assert_eq!(ast_depth(&balanced), 3);

    // Sanity: the left-fold reference is depth 7.
    let mut linear = parts[0].clone();
    for p in &parts[1..] {
        linear = linear.concat(p);
    }
    assert_eq!(ast_depth(&linear), 7);
}

#[test]
fn test_concat_balanced_odd_length() {
    // Odd length (5) should still produce ceil(log2(5))=3-deep tree.
    let parts: Vec<SymBV> = (0..5).map(|i| SymBV::symbol(format!("o{i}"), 8)).collect();
    let balanced = SymBV::concat_balanced(&parts);
    assert_eq!(balanced.width(), 40);
    assert!(ast_depth(&balanced) <= 3);
}

#[test]
fn test_concat_balanced_matches_linear_value() {
    // For concrete inputs, both balanced and linear concat must
    // produce the same numeric value.
    let bytes = [0x12u128, 0x34, 0x56, 0x78, 0x9A, 0xBC];
    let parts: Vec<SymBV> = bytes.iter().map(|&b| SymBV::concrete(b, 8)).collect();
    let balanced = SymBV::concat_balanced(&parts);
    let mut linear = parts[0].clone();
    for p in &parts[1..] {
        linear = linear.concat(p);
    }
    assert_eq!(balanced.width(), linear.width());
    assert_eq!(balanced.as_u64(), linear.as_u64());
    assert_eq!(balanced.as_u64(), Some(0x123456789ABC));
}

#[test]
fn test_comparison_unsigned() {
    let a = SymBV::concrete(5, 32);
    let b = SymBV::concrete(10, 32);

    assert_eq!(a.ult(&b).as_u64(), Some(1));
    assert_eq!(a.ule(&b).as_u64(), Some(1));
    assert_eq!(a.ugt(&b).as_u64(), Some(0));
    assert_eq!(a.uge(&b).as_u64(), Some(0));
}

#[test]
fn test_comparison_signed() {
    let a = SymBV::concrete(0xFF, 8); // -1 signed
    let b = SymBV::concrete(1, 8);

    assert_eq!(a.slt(&b).as_u64(), Some(1)); // -1 < 1
    assert_eq!(a.ult(&b).as_u64(), Some(0)); // 255 > 1 unsigned
}

/// Width-0 symbolic construction degenerates to `Concrete` instead of aborting.
///
/// Z3 has no 0-width bitvector sort, so `BV::new_const(name, 0)` panicked inside
/// the sort constructor — a SIGABRT under `panic="abort"`, not a catchable
/// error. Every symbolic leaf constructor funnels through `SymBV::symbol`,
/// so all are covered here (angr-fs8kb.28).
#[test]
fn test_zero_width_symbolic_degenerates_to_concrete() {
    for bv in [SymBV::symbol("zw", 0), SymBV::symbol("zw_id", 0)] {
        assert!(bv.is_concrete(), "width-0 symbolic must degenerate: {bv:?}");
        assert_eq!(bv.width(), 0);
        // The empty bitvector has exactly one inhabitant, and `concrete`'s mask
        // collapses to 0 at width 0, so that inhabitant is spelled `0`.
        assert_eq!(bv.as_u64(), Some(0));
    }
}

/// A nonzero width still builds a real `Symbolic` leaf — the width-0 early
/// return must not swallow the ordinary path.
#[test]
fn test_nonzero_width_symbolic_still_symbolic() {
    let bv = SymBV::symbol("nonzero", 8);
    assert!(!bv.is_concrete());
    assert_eq!(bv.leaf_var().map(|(_, w)| w), Some(8));
}

/// `SymBV::from_op` over operands clarirs cannot build a term from (here,
/// mismatched widths, which no constructor produces) returns a fresh
/// variable of the result width rather than a wrong value, and two such
/// failures are distinct variables.
#[test]
fn from_op_that_cannot_build_is_a_fresh_variable() {
    let x = SymBV::symbol("unbuildable_x", 32);
    let y = SymBV::symbol("unbuildable_y", 8);
    let a = SymBV::from_op(32, BVOp::Add, vec![x.clone(), y.clone()]);
    let b = SymBV::from_op(32, BVOp::Add, vec![x, y]);
    let (name, width) = a.leaf_var().expect("a fresh leaf");
    assert!(name.starts_with("unbuildable_op_"), "{name}");
    assert_eq!(width, 32);
    assert_ne!(a, b, "each failure is its own variable");
}
