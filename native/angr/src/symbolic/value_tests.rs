//! Core `RustBV` construction/arithmetic tests: concrete folds, extend /
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
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(5, 32);
    let b = RustBV::concrete(3, 32);
    let result = a.add(&b, &ctx);
    assert_eq!(result.as_u64(), Some(8));
}

#[test]
fn test_concrete_sub() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(10, 32);
    let b = RustBV::concrete(3, 32);
    let result = a.sub(&b, &ctx);
    assert_eq!(result.as_u64(), Some(7));
}

#[test]
fn test_concrete_overflow() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(0xFF, 8);
    let b = RustBV::concrete(1, 8);
    let result = a.add(&b, &ctx);
    assert_eq!(result.as_u64(), Some(0)); // 8-bit overflow wraps to 0
}

#[test]
fn test_sign_extend() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(0xFF, 8); // -1 in 8 bits
    let result = a.sign_extend(32, &ctx);
    assert_eq!(result.as_u64(), Some(0xFFFFFFFF)); // -1 in 32 bits
}

#[test]
fn test_zero_extend() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(0xFF, 8);
    let result = a.zero_extend(32, &ctx);
    assert_eq!(result.as_u64(), Some(0xFF));
}

#[test]
fn test_truncate() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(0x12345678, 32);
    let result = a.truncate(8, &ctx);
    assert_eq!(result.as_u64(), Some(0x78));
}

#[test]
fn test_extract() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(0xABCD, 16);
    let result = a.extract(11, 4, &ctx);
    assert_eq!(result.width(), 8);
    assert_eq!(result.as_u64(), Some(0xBC));
}

#[test]
fn test_extract_no_ctx_high_bits_of_wide_concrete_are_zero() {
    // A Concrete stores its value in a u128, so bits at positions >= 128 are
    // logically zero even for a width-256 value. extract_no_ctx must guard the
    // shift/mask so it returns 0 (not a u128 shift-overflow abort / mod-128
    // wrap) when the requested window lands entirely in those high bits.
    let wide = RustBV::concrete(u128::MAX, 256);
    // Window [143:136] is entirely above bit 128 -> all logically-zero bits.
    let hi = wide.extract_no_ctx(143, 136);
    assert_eq!(hi.width(), 8);
    assert_eq!(hi.as_u128(), Some(0));
    // low >= 128 alone: window [200:193] also fully in the zero region.
    let hi2 = wide.extract_no_ctx(200, 193);
    assert_eq!(hi2.as_u128(), Some(0));
    // Low window still returns the real bits.
    let lo = wide.extract_no_ctx(7, 0);
    assert_eq!(lo.as_u128(), Some(0xFF));
    // A result_width >= 128 window over the low 128 real bits stays exact.
    let full_lo = wide.extract_no_ctx(127, 0);
    assert_eq!(full_lo.width(), 128);
    assert_eq!(full_lo.as_u128(), Some(u128::MAX));
}


#[test]
fn test_extract_no_ctx_canonicalizes_like_extract_into() {
    // extract_no_ctx used to hand-roll a strict subset of the rules-1-5 pass,
    // so a symbolic operand got a different shape depending on which entry
    // point the caller reached for (angr-9ke6b.237). Both now route through
    // drive_extract.
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "a", 8);
    let b = RustBV::symbolic(&ctx, "b", 8);
    let cat = a.concat(&b, &ctx);

    // Rule 2: Extract entirely within one Concat part collapses to that part.
    assert_eq!(bv_shape(&cat.extract_no_ctx(7, 0)), bv_shape(&b));
    assert_eq!(bv_shape(&cat.extract_no_ctx(15, 8)), bv_shape(&a));
    assert_eq!(
        bv_shape(&cat.extract_no_ctx(7, 0)),
        bv_shape(&cat.extract(7, 0, &ctx))
    );

    // Rule 1: Extract-of-Extract folds into a single Extract of the base.
    let wide = RustBV::symbolic(&ctx, "w", 32);
    let folded = wide.extract_no_ctx(23, 8).extract_no_ctx(7, 0);
    assert_eq!(bv_shape(&folded), bv_shape(&wide.extract(15, 8, &ctx)));
    // Debug for a leaf symbol carries its allocated id (angr-sqfj8.99), which
    // is drawn from a process-global counter — build the expectation from the
    // actual id rather than hardcoding one.
    let wide_id = match &wide {
        RustBV::Symbolic { id, .. } => *id,
        other => panic!("expected a leaf symbol, got {other:?}"),
    };
    assert_eq!(
        bv_shape(&folded),
        format!("Expr(Extract(15, 8), 8, [Symbolic(#{wide_id}, w, 32)])")
    );

    // Rule 4: Extract above a ZeroExt's original width is the zero constant.
    let top = b.zero_extend(32, &ctx).extract_no_ctx(31, 24);
    assert_eq!(top, RustBV::zero(8));
}

#[test]
fn test_concat() {
    let ctx = SymContext::new_mock();
    let hi = RustBV::concrete(0xAB, 8);
    let lo = RustBV::concrete(0xCD, 8);
    let result = hi.concat(&lo, &ctx);
    assert_eq!(result.width(), 16);
    assert_eq!(result.as_u64(), Some(0xABCD));
}

/// Helper: recursive depth of an expression AST. Concrete/symbolic
/// leaves have depth 0; every Expression node adds one.
fn ast_depth(bv: &RustBV) -> u32 {
    match bv {
        RustBV::Expression { operands, .. } => {
            1 + operands.iter().map(ast_depth).max().unwrap_or(0)
        }
        _ => 0,
    }
}

#[test]
fn test_concat_balanced_single() {
    let ctx = SymContext::new_mock();
    let parts = [RustBV::symbolic(&ctx, "a", 8)];
    let result = RustBV::concat_balanced(&parts, &ctx);
    assert_eq!(result.width(), 8);
}

#[test]
fn test_concat_balanced_pair() {
    let ctx = SymContext::new_mock();
    let hi = RustBV::concrete(0xAB, 8);
    let lo = RustBV::concrete(0xCD, 8);
    let result = RustBV::concat_balanced(&[hi, lo], &ctx);
    assert_eq!(result.width(), 16);
    assert_eq!(result.as_u64(), Some(0xABCD));
}

#[test]
fn test_concat_balanced_concrete_value() {
    // Build 0xDEADBEEF byte-by-byte (high to low) and check the value.
    let ctx = SymContext::new_mock();
    let parts: Vec<RustBV> = [0xDE, 0xAD, 0xBE, 0xEF]
        .iter()
        .map(|&b| RustBV::concrete(b, 8))
        .collect();
    let result = RustBV::concat_balanced(&parts, &ctx);
    assert_eq!(result.width(), 32);
    assert_eq!(result.as_u64(), Some(0xDEADBEEF));
}

#[test]
fn test_concat_balanced_depth_is_log() {
    // 8 symbolic bytes → linear chain would have depth 7; balanced
    // tree should be 3 (log2(8)).
    let ctx = SymContext::new_mock();
    let parts: Vec<RustBV> = (0..8)
        .map(|i| RustBV::symbolic(&ctx, format!("b{i}"), 8))
        .collect();
    let balanced = RustBV::concat_balanced(&parts, &ctx);
    assert_eq!(balanced.width(), 64);
    assert_eq!(ast_depth(&balanced), 3);

    // Sanity: the left-fold reference is depth 7.
    let mut linear = parts[0].clone();
    for p in &parts[1..] {
        linear = linear.concat(p, &ctx);
    }
    assert_eq!(ast_depth(&linear), 7);
}

#[test]
fn test_concat_balanced_odd_length() {
    // Odd length (5) should still produce ceil(log2(5))=3-deep tree.
    let ctx = SymContext::new_mock();
    let parts: Vec<RustBV> = (0..5)
        .map(|i| RustBV::symbolic(&ctx, format!("o{i}"), 8))
        .collect();
    let balanced = RustBV::concat_balanced(&parts, &ctx);
    assert_eq!(balanced.width(), 40);
    assert!(ast_depth(&balanced) <= 3);
}

#[test]
fn test_concat_balanced_matches_linear_value() {
    // For concrete inputs, both balanced and linear concat must
    // produce the same numeric value.
    let ctx = SymContext::new_mock();
    let bytes = [0x12u128, 0x34, 0x56, 0x78, 0x9A, 0xBC];
    let parts: Vec<RustBV> = bytes.iter().map(|&b| RustBV::concrete(b, 8)).collect();
    let balanced = RustBV::concat_balanced(&parts, &ctx);
    let mut linear = parts[0].clone();
    for p in &parts[1..] {
        linear = linear.concat(p, &ctx);
    }
    assert_eq!(balanced.width(), linear.width());
    assert_eq!(balanced.as_u64(), linear.as_u64());
    assert_eq!(balanced.as_u64(), Some(0x123456789ABC));
}

#[test]
fn test_comparison_unsigned() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(5, 32);
    let b = RustBV::concrete(10, 32);

    assert_eq!(a.ult(&b, &ctx).as_u64(), Some(1));
    assert_eq!(a.ule(&b, &ctx).as_u64(), Some(1));
    assert_eq!(a.ugt(&b, &ctx).as_u64(), Some(0));
    assert_eq!(a.uge(&b, &ctx).as_u64(), Some(0));
}

#[test]
fn test_comparison_signed() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(0xFF, 8); // -1 signed
    let b = RustBV::concrete(1, 8);

    assert_eq!(a.slt(&b, &ctx).as_u64(), Some(1)); // -1 < 1
    assert_eq!(a.ult(&b, &ctx).as_u64(), Some(0)); // 255 > 1 unsigned
}
