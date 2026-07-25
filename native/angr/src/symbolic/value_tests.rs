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

// =========================================================================
// Expression simplification tests
// =========================================================================

#[test]
fn test_add_identity() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let zero = RustBV::zero(32);
    // x + 0 → x (should return symbolic, not expression)
    let r1 = x.add(&zero, &ctx);
    assert!(matches!(r1, RustBV::Symbolic { .. }));
    // 0 + x → x
    let r2 = zero.add(&x, &ctx);
    assert!(matches!(r2, RustBV::Symbolic { .. }));
}

#[test]
fn test_sub_identity() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let zero = RustBV::zero(32);
    // x - 0 → x
    let r = x.sub(&zero, &ctx);
    assert!(matches!(r, RustBV::Symbolic { .. }));
}

#[test]
fn test_mul_identity_and_annihilator() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let zero = RustBV::zero(32);
    let one = RustBV::concrete(1, 32);
    // x * 0 → 0
    assert_eq!(x.mul(&zero, &ctx).as_u128(), Some(0));
    // 0 * x → 0
    assert_eq!(zero.mul(&x, &ctx).as_u128(), Some(0));
    // x * 1 → x
    assert!(matches!(x.mul(&one, &ctx), RustBV::Symbolic { .. }));
    // 1 * x → x
    assert!(matches!(one.mul(&x, &ctx), RustBV::Symbolic { .. }));
}

#[test]
fn test_and_identity_and_annihilator() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zero = RustBV::zero(8);
    let ones = RustBV::ones(8);
    // x & 0 → 0
    assert_eq!(x.and(&zero, &ctx).as_u128(), Some(0));
    // x & 0xFF → x
    assert!(matches!(x.and(&ones, &ctx), RustBV::Symbolic { .. }));
    // 0xFF & x → x
    assert!(matches!(ones.and(&x, &ctx), RustBV::Symbolic { .. }));
}

#[test]
fn test_or_identity_and_annihilator() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zero = RustBV::zero(8);
    let ones = RustBV::ones(8);
    // x | 0 → x
    assert!(matches!(x.or(&zero, &ctx), RustBV::Symbolic { .. }));
    // 0 | x → x
    assert!(matches!(zero.or(&x, &ctx), RustBV::Symbolic { .. }));
    // x | 0xFF → 0xFF
    assert_eq!(x.or(&ones, &ctx).as_u128(), Some(0xFF));
}

#[test]
fn test_xor_identity() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let zero = RustBV::zero(32);
    // x ^ 0 → x
    assert!(matches!(x.xor(&zero, &ctx), RustBV::Symbolic { .. }));
    // 0 ^ x → x
    assert!(matches!(zero.xor(&x, &ctx), RustBV::Symbolic { .. }));
}

#[test]
fn test_not_double_negation() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    // not(not(x)) → x
    let r = x.not(&ctx).not(&ctx);
    assert!(matches!(r, RustBV::Symbolic { .. }));
}

#[test]
fn test_neg_double_negation() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    // neg(neg(x)) → x
    let r = x.neg(&ctx).neg(&ctx);
    assert!(matches!(r, RustBV::Symbolic { .. }));
}

#[test]
fn test_reverse_double() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    // reverse(reverse(x)) → x
    let r = x.reverse(&ctx).reverse(&ctx);
    assert!(matches!(r, RustBV::Symbolic { .. }));
}

#[test]
fn test_shift_by_zero() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let zero = RustBV::zero(32);
    // x << 0 → x
    assert!(matches!(x.shl(&zero, &ctx), RustBV::Symbolic { .. }));
    // x >> 0 → x
    assert!(matches!(x.lshr(&zero, &ctx), RustBV::Symbolic { .. }));
    // x >>> 0 → x
    assert!(matches!(x.ashr(&zero, &ctx), RustBV::Symbolic { .. }));
}

#[test]
fn test_shift_zero_value() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let zero = RustBV::zero(32);
    // 0 << x → 0
    assert_eq!(zero.shl(&x, &ctx).as_u128(), Some(0));
    // 0 >> x → 0
    assert_eq!(zero.lshr(&x, &ctx).as_u128(), Some(0));
}

#[test]
fn test_shl_concrete_amount_rewrites_to_concat() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let four = RustBV::concrete(4, 32);
    // sym << 4 → Concat(Extract(27, 0, sym), 0^4)
    let r = x.shl(&four, &ctx);
    assert_eq!(r.width(), 32);
    match r {
        RustBV::Expression {
            op: BVOp::Concat,
            operands,
            ..
        } => {
            assert_eq!(operands.len(), 2);
            assert_eq!(operands[1].as_u128(), Some(0)); // low bits are zero
            assert_eq!(operands[1].width(), 4);
            assert_eq!(operands[0].width(), 28); // top bits extracted from x
        }
        other => panic!("expected Concat, got {other:?}"),
    }
    // Behavior preserved when LHS happens to be concrete (still constant-folds).
    let v = RustBV::concrete(0x1234, 32);
    assert_eq!(v.shl(&four, &ctx).as_u128(), Some(0x12340));
}

#[test]
fn test_shl_concrete_amount_at_or_above_width_yields_zero() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    // sym << 32 → 0
    let r = x.shl(&RustBV::concrete(32, 32), &ctx);
    assert_eq!(r.as_u128(), Some(0));
    // sym << 999 (oversized amount) → 0
    let r = x.shl(&RustBV::concrete(999, 32), &ctx);
    assert_eq!(r.as_u128(), Some(0));
}

#[test]
fn test_lshr_concrete_amount_rewrites_to_concat() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let eight = RustBV::concrete(8, 32);
    // sym >> 8 → Concat(0^8, Extract(31, 8, sym))
    let r = x.lshr(&eight, &ctx);
    assert_eq!(r.width(), 32);
    match r {
        RustBV::Expression {
            op: BVOp::Concat,
            operands,
            ..
        } => {
            assert_eq!(operands.len(), 2);
            assert_eq!(operands[0].as_u128(), Some(0));
            assert_eq!(operands[0].width(), 8);
            assert_eq!(operands[1].width(), 24);
        }
        other => panic!("expected Concat, got {other:?}"),
    }
    let r = x.lshr(&RustBV::concrete(32, 32), &ctx);
    assert_eq!(r.as_u128(), Some(0));
}

#[test]
fn test_ashr_concrete_amount_rewrites_to_sign_extend() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let four = RustBV::concrete(4, 32);
    // sym >>> 4 → SignExt(4, Extract(31, 4, sym))  [extends 28-bit slice by 4 bits]
    let r = x.ashr(&four, &ctx);
    assert_eq!(r.width(), 32);
    match r {
        RustBV::Expression {
            op: BVOp::SignExt(4),
            operands,
            ..
        } => {
            assert_eq!(operands[0].width(), 28);
        }
        other => panic!("expected SignExt(4), got {other:?}"),
    }
    // Beyond width: SignExt of MSB (1-bit slice extended by 31 bits).
    let r = x.ashr(&RustBV::concrete(64, 32), &ctx);
    assert_eq!(r.width(), 32);
    match r {
        RustBV::Expression {
            op: BVOp::SignExt(31),
            operands,
            ..
        } => {
            assert_eq!(operands[0].width(), 1);
        }
        other => panic!("expected SignExt(31), got {other:?}"),
    }
}

#[test]
fn test_mul_by_power_of_two_rewrites_to_shl_then_concat() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let eight = RustBV::concrete(8, 32);
    // sym * 8 → sym << 3 → Concat(Extract(28, 0, sym), 0^3)
    let r = x.mul(&eight, &ctx);
    assert_eq!(r.width(), 32);
    match r {
        RustBV::Expression {
            op: BVOp::Concat,
            operands,
            ..
        } => {
            assert_eq!(operands[1].as_u128(), Some(0));
            assert_eq!(operands[1].width(), 3);
            assert_eq!(operands[0].width(), 29);
        }
        other => panic!("expected Concat (via Shl), got {other:?}"),
    }
    // Commutative case: 8 * sym → same shape.
    let r = eight.mul(&x, &ctx);
    match r {
        RustBV::Expression {
            op: BVOp::Concat,
            operands,
            ..
        } => {
            assert_eq!(operands[1].as_u128(), Some(0));
            assert_eq!(operands[1].width(), 3);
        }
        other => panic!("expected Concat (commutative), got {other:?}"),
    }
}

#[test]
fn test_mul_by_non_power_of_two_stays_as_mul() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    // sym * 3 → still Mul (no rewrite for non-pow2 constants)
    let r = x.mul(&RustBV::concrete(3, 32), &ctx);
    match r {
        RustBV::Expression { op: BVOp::Mul, .. } => {}
        other => panic!("expected Mul, got {other:?}"),
    }
}

#[test]
fn test_sign_extend_identity() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    // sign_extend to same width → x
    let r = x.sign_extend(32, &ctx);
    assert!(matches!(r, RustBV::Symbolic { .. }));
    assert_eq!(r.width(), 32);
}

#[test]
fn test_extract_zero_ext() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let ext = x.zero_extend(32, &ctx); // 8-bit → 32-bit
    // Extract low 8 bits → original x
    let lo = ext.extract(7, 0, &ctx);
    assert!(matches!(lo, RustBV::Symbolic { .. }));
    assert_eq!(lo.width(), 8);
    // Extract high 8 bits → 0
    let hi = ext.extract(31, 24, &ctx);
    assert_eq!(hi.as_u128(), Some(0));
}

#[test]
fn test_extract_sign_ext_low_bits() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let ext = x.sign_extend(32, &ctx);
    // Extract low 8 bits → original x
    let lo = ext.extract(7, 0, &ctx);
    assert!(matches!(lo, RustBV::Symbolic { .. }));
    assert_eq!(lo.width(), 8);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_reverse_z3_emission_32bit_leaf() {
    // Reverse(x) over a symbolic leaf should produce byte-reversed value
    // through the Z3 emission path.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let rev = x.reverse(&ctx); // Expression { Reverse, [x] }
    // Pin x = 0x11223344; expect Reverse(x) = 0x44332211
    let pinned = x.eq(&RustBV::concrete(0x11223344, 32), &ctx);
    ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    assert_eq!(ctx.eval(&rev), Some(0x44332211));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_reverse_z3_emission_64bit_leaf() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 64);
    let rev = x.reverse(&ctx);
    let pinned = x.eq(&RustBV::concrete(0x0123456789ABCDEF, 64), &ctx);
    ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    assert_eq!(ctx.eval(&rev), Some(0xEFCDAB8967452301));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_reverse_z3_emission_16bit_leaf() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 16);
    let rev = x.reverse(&ctx);
    let pinned = x.eq(&RustBV::concrete(0xAABB, 16), &ctx);
    ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    assert_eq!(ctx.eval(&rev), Some(0xBBAA));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_reverse_z3_emission_concat_of_bytes() {
    // Reverse(Concat(b0, b1, ..., b7)) with independent byte BVSes.
    // Memory loads in angr typically produce this shape; the result should
    // be Concat(b7, b6, ..., b0) — the byte-reversed value.
    let ctx = SymContext::new_mock();
    let bytes: Vec<RustBV> = (0..8u32)
        .map(|i| RustBV::symbolic(&ctx, format!("b{i}"), 8))
        .collect();
    // Build claripy-style Concat(b0, b1, ..., b7) with b0 as high.
    let mut concat = bytes[0].clone();
    for b in &bytes[1..] {
        concat = concat.concat(b, &ctx);
    }
    let rev = concat.reverse(&ctx);
    // Pin each byte to a distinct value and verify byte-reversed result.
    let vals: [u64; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    for (b, v) in bytes.iter().zip(vals.iter()) {
        let pin = b.eq(&RustBV::concrete(*v as u128, 8), &ctx);
        ctx.add_constraint(pin.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    }
    // concat = 0x1122334455667788; reverse → 0x8877665544332211
    assert_eq!(ctx.eval(&rev), Some(0x8877665544332211));
}

// --- Pre-Z3 Extract rewrite pass (angr-p8cz) ---

/// Helper: build a raw Extract Expression node without going through
/// `extract_into`. This simulates Extract nodes that bypass the
/// construction-time rewrite (e.g., via `truncate_into` or
/// `extract_no_ctx`), so we can verify the Z3-emission pass picks them up.
#[cfg(feature = "vex-engine-z3")]
fn raw_extract_node(inner: RustBV, high: u32, low: u32) -> RustBV {
    let result_width = high - low + 1;
    RustBV::Expression {
        id: RustBV::EXPRESSION_ID,
        width: result_width,
        op: BVOp::Extract(high, low),
        operands: std::sync::Arc::<[RustBV]>::from([inner]),
        memo: Default::default(),
    }
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_pre_z3_extract_over_reverse_byte_aligned() {
    // Extract a single byte from Reverse(x). With the rewrite pass the
    // Reverse should be eliminated entirely from the Z3 AST.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let rev = x.reverse(&ctx);
    // Bypass extract_into via the raw-node helper: pretend this Extract was
    // built by truncate_into or extract_no_ctx after the Reverse existed.
    // Reverse(x) byte 0 ([7:0]) is byte 3 ([31:24]) of x.
    let lo_byte = raw_extract_node(rev, 7, 0);
    let pinned = x.eq(&RustBV::concrete(0x11223344, 32), &ctx);
    ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    assert_eq!(ctx.eval(&lo_byte), Some(0x11));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_pre_z3_extract_over_reverse_crossing_byte() {
    // Two-byte extract across a byte boundary on Reverse — still
    // byte-aligned (high % 8 == 7, low % 8 == 0), should be rewritten.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let rev = x.reverse(&ctx);
    // Reverse(x)[15:0] = bytes 0,1 of reverse = bytes 3,2 of x = top half reversed.
    let lower16 = raw_extract_node(rev, 15, 0);
    let pinned = x.eq(&RustBV::concrete(0x11223344, 32), &ctx);
    ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    // Reverse(0x11223344) = 0x44332211; low 16 bits = 0x2211
    assert_eq!(ctx.eval(&lower16), Some(0x2211));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_pre_z3_extract_over_concat_within_low() {
    // Extract entirely within the low (right) part of a Concat — should
    // delegate to that operand alone.
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "a", 16);
    let b = RustBV::symbolic(&ctx, "b", 16);
    let cat = a.concat(&b, &ctx); // a:high, b:low
    // Extract [15:0] of cat = entirely within b.
    let lo = raw_extract_node(cat, 15, 0);
    ctx.add_constraint(
        a.eq(&RustBV::concrete(0xAAAA, 16), &ctx)
            .to_z3_ast()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    ctx.add_constraint(
        b.eq(&RustBV::concrete(0xBBBB, 16), &ctx)
            .to_z3_ast()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    assert_eq!(ctx.eval(&lo), Some(0xBBBB));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_pre_z3_extract_over_concat_within_high() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "a", 16);
    let b = RustBV::symbolic(&ctx, "b", 16);
    let cat = a.concat(&b, &ctx);
    // Extract [31:16] of cat = entirely within a.
    let hi = raw_extract_node(cat, 31, 16);
    ctx.add_constraint(
        a.eq(&RustBV::concrete(0xAAAA, 16), &ctx)
            .to_z3_ast()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    ctx.add_constraint(
        b.eq(&RustBV::concrete(0xBBBB, 16), &ctx)
            .to_z3_ast()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    assert_eq!(ctx.eval(&hi), Some(0xAAAA));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_pre_z3_extract_over_concat_crossing() {
    // Crosses the a/b boundary in the middle — distribute Extract to both.
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "a", 16);
    let b = RustBV::symbolic(&ctx, "b", 16);
    let cat = a.concat(&b, &ctx);
    // Extract [23:8] = high 8 bits of b ([15:8]) concat with low 8 bits of a ([7:0]).
    let mid = raw_extract_node(cat, 23, 8);
    ctx.add_constraint(
        a.eq(&RustBV::concrete(0x1234, 16), &ctx)
            .to_z3_ast()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    ctx.add_constraint(
        b.eq(&RustBV::concrete(0x5678, 16), &ctx)
            .to_z3_ast()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    // cat = 0x12345678; extract [23:8] = 0x3456
    assert_eq!(ctx.eval(&mid), Some(0x3456));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_pre_z3_extract_over_extract_fused() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 64);
    let mid = x.extract(47, 16, &ctx); // width 32, = bits 16..=47 of x
    // Build outer Extract WITHOUT going through extract_into.
    let outer = raw_extract_node(mid, 23, 8); // mid[23:8] = bits [39:24] of x
    ctx.add_constraint(
        x.eq(&RustBV::concrete(0x0011_2233_4455_6677, 64), &ctx)
            .to_z3_ast()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    // x bytes (LSB→MSB): 0x77 0x66 0x55 0x44 0x33 0x22 0x11 0x00.
    // bits [39:24] of x = byte indices 3..=4 = 0x33:0x44 (high:low) = 0x3344.
    assert_eq!(ctx.eval(&outer), Some(0x3344));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_pre_z3_extract_over_zero_ext_low_bits() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(32, &ctx); // width 32
    // Extract [7:0] entirely within original x.
    let lo = raw_extract_node(zx, 7, 0);
    ctx.add_constraint(
        x.eq(&RustBV::concrete(0xAB, 8), &ctx)
            .to_z3_ast()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    assert_eq!(ctx.eval(&lo), Some(0xAB));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_pre_z3_extract_over_zero_ext_extended_bits() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(32, &ctx);
    // Extract [31:24] is entirely in the zero-extended region.
    let top = raw_extract_node(zx, 31, 24);
    ctx.add_constraint(
        x.eq(&RustBV::concrete(0xFF, 8), &ctx)
            .to_z3_ast()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    assert_eq!(ctx.eval(&top), Some(0));
}

/// Regression for the latent bug in extract_into's Rule 3: multi-byte
/// byte-aligned Extract of Reverse(x) used to drop the byte shuffle
/// (returned plain Extract from x), giving the wrong byte order in the
/// constraint tree. Now it returns Reverse(Extract(...)), preserving
/// semantics through every consumer (Z3 round-trip, downstream rewrites).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_extract_over_reverse_multibyte_construction() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let rev = x.reverse(&ctx);
    // Use the normal extract (which now goes through the fixed Rule 3).
    let lower16 = rev.extract(15, 0, &ctx);
    let pinned = x.eq(&RustBV::concrete(0x11223344, 32), &ctx);
    ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    // Reverse(0x11223344) = 0x44332211; low 16 = 0x2211.
    assert_eq!(ctx.eval(&lower16), Some(0x2211));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_extract_over_reverse_single_byte_construction() {
    // Single-byte case: rule reduces to plain Extract from x.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let rev = x.reverse(&ctx);
    // Byte 0 of reverse = byte 3 of x.
    let b0 = rev.extract(7, 0, &ctx);
    ctx.add_constraint(
        x.eq(&RustBV::concrete(0x11223344, 32), &ctx)
            .to_z3_ast()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    assert_eq!(ctx.eval(&b0), Some(0x11));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_pre_z3_extract_over_sign_ext_low_bits() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let sx = x.sign_extend(32, &ctx);
    let lo = raw_extract_node(sx, 7, 0);
    ctx.add_constraint(
        x.eq(&RustBV::concrete(0x80, 8), &ctx)
            .to_z3_ast()
            .eq(z3::ast::BV::from_u64(1, 1)),
    );
    assert_eq!(ctx.eval(&lo), Some(0x80));
}

// =========================================================================
// angr-g7nq: Cmp(ZeroExt(k, x), BVV) trivial-constraint fast path
// =========================================================================

#[test]
fn test_zext_eq_high_bits_nonzero_is_false() {
    // ZeroExt(8, x:8) == 0x100 — high byte nonzero → folds to concrete 0.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let c = RustBV::concrete(0x100, 16);
    let r = zx.eq(&c, &ctx);
    assert_eq!(r.as_u64(), Some(0));
    // Commuted form folds the same way.
    let r_rev = c.eq(&zx, &ctx);
    assert_eq!(r_rev.as_u64(), Some(0));
}

#[test]
fn test_zext_ne_high_bits_nonzero_is_true() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let c = RustBV::concrete(0x100, 16);
    let r = zx.ne(&c, &ctx);
    assert_eq!(r.as_u64(), Some(1));
}

#[test]
fn test_zext_eq_high_bits_zero_collapses() {
    // ZeroExt(8, x:8) == 0x42 — high byte zero → narrows to Eq(x, 0x42),
    // which is still symbolic but should be a width-8 Expression not the
    // width-16 form, evidenced by the operand width.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = zx.eq(&RustBV::concrete(0x42, 16), &ctx);
    // Result is a width-1 Eq expression over width-8 operands.
    match &r {
        RustBV::Expression {
            op,
            operands,
            width,
            ..
        } => {
            assert_eq!(*op, BVOp::Eq);
            assert_eq!(*width, 1);
            assert_eq!(operands.len(), 2);
            assert_eq!(operands[0].width(), 8);
            assert_eq!(operands[1].width(), 8);
            assert_eq!(operands[1].as_u64(), Some(0x42));
        }
        _ => panic!("expected narrowed Eq expression, got {r:?}"),
    }
}

#[test]
fn test_zext_eq_collapse_solver_consistency() {
    // After narrowing, asserting Eq(zext(x), 0x42) must still pin x to 0x42.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let eq = zx.eq(&RustBV::concrete(0x42, 16), &ctx);
    ctx.add_constraint(eq.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    assert_eq!(ctx.eval(&x), Some(0x42));
    // Also verify the wider zext expression evaluates to the constant.
    assert_eq!(ctx.eval(&zx), Some(0x42));
}

#[test]
fn test_zext_ult_high_bits_nonzero_is_true() {
    // ZeroExt(8, x:8) < 0x200 — all zext values < 256, so always < 0x200.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = zx.ult(&RustBV::concrete(0x200, 16), &ctx);
    assert_eq!(r.as_u64(), Some(1));
}

#[test]
fn test_zext_ult_const_zero_is_false() {
    // ZeroExt(8, x:8) < 0 — never true.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = zx.ult(&RustBV::concrete(0, 16), &ctx);
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_zext_ult_swapped_high_bits_nonzero_is_false() {
    // 0x200 < ZeroExt(8, x:8) — never true (RHS < 256).
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = RustBV::concrete(0x200, 16).ult(&zx, &ctx);
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_zext_ule_high_bits_nonzero_is_true() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = zx.ule(&RustBV::concrete(0x200, 16), &ctx);
    assert_eq!(r.as_u64(), Some(1));
}

#[test]
fn test_zext_ule_swapped_const_zero_is_true() {
    // 0 <= ZeroExt(x) — always true.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = RustBV::concrete(0, 16).ule(&zx, &ctx);
    assert_eq!(r.as_u64(), Some(1));
}

#[test]
fn test_zext_ugt_high_bits_nonzero_is_false() {
    // ZeroExt(8, x:8) > 0x200 — never true (LHS < 256 <= 0x200).
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = zx.ugt(&RustBV::concrete(0x200, 16), &ctx);
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_zext_uge_high_bits_nonzero_is_false() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx = x.zero_extend(16, &ctx);
    let r = zx.uge(&RustBV::concrete(0x200, 16), &ctx);
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_zext_cmp_no_fold_when_both_symbolic() {
    // Cmp(ZeroExt(x), ZeroExt(y)) — no const side, no fold.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let y = RustBV::symbolic(&ctx, "y", 8);
    let zx = x.zero_extend(16, &ctx);
    let zy = y.zero_extend(16, &ctx);
    let r = zx.eq(&zy, &ctx);
    // Should be a regular Eq expression at width 16 — no fold.
    match &r {
        RustBV::Expression { op, operands, .. } => {
            assert_eq!(*op, BVOp::Eq);
            assert_eq!(operands[0].width(), 16);
            assert_eq!(operands[1].width(), 16);
        }
        _ => panic!("expected Eq expression, got {r:?}"),
    }
}

#[test]
fn test_zext_cmp_no_fold_for_signed_ops() {
    // SignExt(k, x) is not handled — the comparison should pass through.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let sx = x.sign_extend(16, &ctx); // SignExt, not ZeroExt
    let r = sx.eq(&RustBV::concrete(0x100, 16), &ctx);
    // Should be Eq expression (no fold).
    match &r {
        RustBV::Expression { op, operands, .. } => {
            assert_eq!(*op, BVOp::Eq);
            assert_eq!(operands[0].width(), 16);
        }
        _ => panic!("expected Eq expression, got {r:?}"),
    }
}

#[test]
fn test_zext_cmp_chain_collapse_to_narrowest() {
    // ZeroExt(16, ZeroExt(8, x:8)) == 0xFF — both extends collapse cleanly.
    // The inner zero_extend collapses via constant fold; the outer is what
    // we're testing. After the first call, the inner zx is a ZeroExt(8, x);
    // wrapping in zero_extend(32, ...) produces ZeroExt(16, ZeroExt(8, x))
    // which by the operand structure is treated as ZeroExt(16, <inner>),
    // where <inner> has width 16. The const 0xFF has zero high 16 bits, so
    // we narrow to Eq(<inner-as-zext>, 0xFF:16); that then narrows again
    // recursively. Final: Eq(x, 0xFF:8).
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 8);
    let zx16 = x.zero_extend(16, &ctx);
    let zx32 = zx16.zero_extend(32, &ctx);
    let r = zx32.eq(&RustBV::concrete(0xFF, 32), &ctx);
    match &r {
        RustBV::Expression {
            op,
            operands,
            width,
            ..
        } => {
            assert_eq!(*op, BVOp::Eq);
            assert_eq!(*width, 1);
            // Should have narrowed to width-8 operands.
            assert_eq!(operands[0].width(), 8);
            assert_eq!(operands[1].as_u64(), Some(0xFF));
        }
        _ => panic!("expected narrowed Eq, got {r:?}"),
    }
}

/// Regression guard for the angr-behq finding (2026-05-21):
/// Z3's AST hash-cons already de-dupes structurally-equal RustBV trees
/// at to_z3_ast time. Two structurally-equal Expression trees produce
/// the SAME Z3_ast pointer — so RustBV-level construction hash-cons
/// would NOT reduce Z3 AST node count for the to_z3_ast() output.
///
/// If this test ever fails, the assumption that drove closing
/// angr-behq is broken and that bead should be re-opened.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn z3_already_dedupes_structurally_equal_rustbv_trees() {
    use z3::ast::Ast as Z3AstTrait;
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic_with_id(2001, "x_behq", 32);
    let y = RustBV::symbolic_with_id(2002, "y_behq", 32);

    // Two structurally-identical add(x, 5) trees built independently.
    let p1 = x.clone().add_into(RustBV::concrete(5, 32), &ctx);
    let p2 = x.add_into(RustBV::concrete(5, 32), &ctx);
    let p1_ptr = p1.to_z3_ast().get_z3_ast().as_ptr();
    let p2_ptr = p2.to_z3_ast().get_z3_ast().as_ptr();
    assert_eq!(p1_ptr, p2_ptr, "Z3 should canonicalize add(x,5)");

    // Two structurally-identical mul(add(x,5), y) trees, depth 2.
    let q1 = p1.mul(&y, &ctx);
    let q2 = p2.mul(&y, &ctx);
    let q1_ptr = q1.to_z3_ast().get_z3_ast().as_ptr();
    let q2_ptr = q2.to_z3_ast().get_z3_ast().as_ptr();
    assert_eq!(q1_ptr, q2_ptr, "Z3 should canonicalize mul(add(x,5),y)");
}

/// Regression guard for angr-kkpr (2026-05-21):
/// RustBV constructors for commutative ops (add/mul/and/or/xor/eq/ne)
/// canonicalize operand order via `canonical_sort_key`. After this fix,
/// `add(x, y)` and `add(y, x)` produce the SAME RustBV (and therefore
/// the same Z3 AST). Z3 itself does NOT normalize commutative arg order
/// at mk_bv* time — preprocessing tactics do, but only on assertions —
/// so the canonicalization happens on the Rust side at construction.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn commutative_ops_canonicalize_operand_order() {
    use z3::ast::Ast as Z3AstTrait;
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic_with_id(3001, "x_kkpr", 32);
    let y = RustBV::symbolic_with_id(3002, "y_kkpr", 32);
    let c = RustBV::concrete(7, 32);

    // add(x, y) and add(y, x) → same RustBV → same Z3 AST.
    let r1 = x.clone().add_into(y.clone(), &ctx);
    let r2 = y.clone().add_into(x.clone(), &ctx);
    assert_eq!(
        r1.to_z3_ast().get_z3_ast().as_ptr(),
        r2.to_z3_ast().get_z3_ast().as_ptr(),
        "add(x,y) and add(y,x) should canonicalize to the same Z3 AST"
    );

    // add(x, c) and add(c, x) → same RustBV → same Z3 AST.
    // (concrete should sort to the right, so both become add(x, c).)
    let r3 = x.clone().add_into(c.clone(), &ctx);
    let r4 = c.add_into(x.clone(), &ctx);
    assert_eq!(
        r3.to_z3_ast().get_z3_ast().as_ptr(),
        r4.to_z3_ast().get_z3_ast().as_ptr(),
        "add(x,c) and add(c,x) should canonicalize to the same Z3 AST"
    );
    // Verify concrete is in operand[1] after canonicalization.
    match &r3 {
        RustBV::Expression { operands, .. } => {
            assert!(
                matches!(operands[1], RustBV::Concrete { .. }),
                "concrete should sort to the right of symbolic"
            );
        }
        _ => panic!("expected Expression for add(x, c)"),
    }

    // Same property for mul / and / or / xor / eq / ne.
    type BinOp = fn(RustBV, RustBV, &SymContext) -> RustBV;
    let cases: &[(&str, BinOp)] = &[
        ("mul", |a, b, c| a.mul_into(b, c)),
        ("and", |a, b, c| a.and_into(b, c)),
        ("or", |a, b, c| a.or_into(b, c)),
        ("xor", |a, b, c| a.xor_into(b, c)),
        ("eq", |a, b, c| a.eq_into(b, c)),
        ("ne", |a, b, c| a.ne_into(b, c)),
    ];
    for (name, build) in cases {
        let lhs = build(x.clone(), y.clone(), &ctx);
        let rhs = build(y.clone(), x.clone(), &ctx);
        assert_eq!(
            lhs.to_z3_ast().get_z3_ast().as_ptr(),
            rhs.to_z3_ast().get_z3_ast().as_ptr(),
            "{name}(x,y) and {name}(y,x) should canonicalize to the same Z3 AST"
        );
    }
}

// -----------------------------------------------------------------
// Serde round-trip tests for the op-tree primitives (angr-x04s.1.1).
// Each variant is constructed inside a fresh SymContext, serialized
// to JSON, deserialized back, and structurally compared to the
// original. The Symbolic variant's Z3 AST is reconstructed lazily
// — we verify that round-tripped Symbolic values still produce a
// valid Z3 AST via to_z3_ast().
// -----------------------------------------------------------------

#[test]
fn serde_roundtrip_concrete() {
    let bv = RustBV::concrete(0xdead_beef, 32);
    let json = serde_json::to_string(&bv).expect("serialize");
    let back: RustBV = serde_json::from_str(&json).expect("deserialize");
    match (&bv, &back) {
        (
            RustBV::Concrete {
                value: v1,
                width: w1,
            },
            RustBV::Concrete {
                value: v2,
                width: w2,
            },
        ) => {
            assert_eq!(v1, v2);
            assert_eq!(w1, w2);
        }
        _ => panic!("variant changed across round-trip"),
    }
}

#[test]
fn serde_roundtrip_constrained() {
    let bv = RustBV::Constrained {
        id: 42,
        value: 7,
        width: 64,
    };
    let json = serde_json::to_string(&bv).expect("serialize");
    let back: RustBV = serde_json::from_str(&json).expect("deserialize");
    match back {
        RustBV::Constrained { id, value, width } => {
            assert_eq!(id, 42);
            assert_eq!(value, 7);
            assert_eq!(width, 64);
        }
        _ => panic!("variant changed across round-trip"),
    }
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn serde_roundtrip_symbolic() {
    use z3::ast::Ast as Z3AstTrait;
    let ctx = SymContext::new_mock();
    let bv = RustBV::symbolic(&ctx, "x_serde", 32);
    let (orig_id, orig_width, orig_name) = match &bv {
        RustBV::Symbolic {
            id, width, name, ..
        } => (*id, *width, name.to_string()),
        _ => panic!("expected Symbolic"),
    };
    let json = serde_json::to_string(&bv).expect("serialize");
    // Deserialize under the same context (thread-local is still active).
    let back: RustBV = serde_json::from_str(&json).expect("deserialize");
    match &back {
        RustBV::Symbolic {
            id, width, name, ..
        } => {
            assert_eq!(*id, orig_id);
            assert_eq!(*width, orig_width);
            assert_eq!(name.as_ref(), orig_name.as_str());
        }
        _ => panic!("variant changed across round-trip"),
    }
    // The lazily-rebuilt AST must be usable: it should produce a Z3 AST
    // pointer (sanity check that BV::new_const succeeded under the
    // active thread-local context).
    let _ptr = back.to_z3_ast().get_z3_ast().as_ptr();
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn serde_roundtrip_expression_tree() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x_expr_serde", 32);
    let c = RustBV::concrete(7, 32);
    let expr = x.add(&c, &ctx);
    // expr is Expression(Add, [Symbolic(x), Concrete(7)]) after
    // commutative canonicalization (concrete sorts to the right).
    let json = serde_json::to_string(&expr).expect("serialize");
    let back: RustBV = serde_json::from_str(&json).expect("deserialize");
    match &back {
        RustBV::Expression {
            op,
            operands,
            width,
            ..
        } => {
            assert_eq!(*op, BVOp::Add);
            assert_eq!(*width, 32);
            assert_eq!(operands.len(), 2);
            assert!(matches!(operands[0], RustBV::Symbolic { .. }));
            assert!(matches!(
                operands[1],
                RustBV::Concrete {
                    value: 7,
                    width: 32
                }
            ));
        }
        _ => panic!("variant changed across round-trip"),
    }
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn serde_roundtrip_float_op() {
    // BVOp::Float carries kind + prec; verify it survives JSON
    // round-trip without losing fields (covers FloatOpKind +
    // FloatPrec derive correctness).
    let op = BVOp::Float {
        kind: FloatOpKind::ConvertItoF {
            src_bits: 32,
            signed: true,
        },
        prec: FloatPrec::F64,
    };
    let json = serde_json::to_string(&op).expect("serialize");
    let back: BVOp = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(op, back);
}

/// Regression (angr-ovqja.3): the persistent per-`Expression` Z3 AST memo must
/// not leak a stale-context AST. The memo caches the built `z3::ast::BV` for an
/// `Expression`; if the thread-local Z3 context is swapped (only `with_z3_context`
/// in tests — never production), `to_z3_ast()` must rebuild against the active
/// context rather than return the cached BV from the previous context.
///
/// Uses an Extract-over-concrete node so the rebuilt AST resolves to a fresh
/// context-local constant with NO `Symbolic` leaves (whose own `ast` cache would
/// confound the context check — that staleness predates this memo).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_expression_memo_no_stale_context_leak() {
    use z3::ast::Ast as Z3AstTrait;
    use z3::{Config, Context};

    // `with_z3_context` cannot be used here: its `Send + Sync` closure bound
    // exists precisely to forbid smuggling a Z3-bearing value (our `RustBV`)
    // across the boundary — yet the stale-memo path requires converting the
    // SAME `RustBV` in two contexts. Swap the thread-local context manually
    // and restore it on the way out (single-threaded test, so this is sound).
    let original = Context::thread_local();

    let expr = raw_extract_node(RustBV::concrete(0x11223344, 32), 7, 0);

    // First conversion in the default thread-local context populates the memo.
    let ast1 = expr.to_z3_ast();
    let default_ctx = ast1.get_ctx().get_z3_context().as_ptr() as usize;

    // Second conversion in the SAME context is served from the memo and must
    // still belong to that context.
    let ast1b = expr.to_z3_ast();
    assert_eq!(
        ast1b.get_ctx().get_z3_context().as_ptr() as usize,
        default_ctx,
        "same-context repeat conversion must stay in the default context",
    );

    // Swap to a freshly-allocated context. The memo holds a default-context BV,
    // so the guard must reject it and rebuild against the new context.
    let cfg = Config::new();
    let new_ctx = Context::new(&cfg);
    let new_ctx_ptr = new_ctx.get_z3_context().as_ptr() as usize;
    assert_ne!(
        default_ctx, new_ctx_ptr,
        "test bug: new context equals default; not testing cross-context",
    );

    Context::set_thread_local(&new_ctx);
    let in_new_ctx = expr.to_z3_ast().get_ctx().get_z3_context().as_ptr() as usize;
    // Restore before asserting so a failure does not poison sibling tests.
    Context::set_thread_local(&original);
    assert_eq!(
        in_new_ctx, new_ctx_ptr,
        "stale-context leak: memo returned a BV from the old context",
    );

    // Back in the default context the guard rejects the now-new-context memo
    // and rebuilds against the default context again.
    let ast3 = expr.to_z3_ast();
    assert_eq!(
        ast3.get_ctx().get_z3_context().as_ptr() as usize,
        default_ctx,
        "returning to the default context must rebuild against it",
    );
}

// --- Cross-context AST translation (angr-panhl.2 parallel kill-gate) ---
//
// `translate_into` is the per-task `Z3_translate` primitive for the
// shared-nothing parallel design (Option A). It must move a `RustBV` from
// one thread-local Z3 context to another with every constraint re-checking
// identical, and at a per-node cost in the spike's ballpark.

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_into_cross_context_eval() {
    use z3::ast::Ast as Z3AstTrait;
    use z3::{Config, Context};

    // Source context = default thread-local. Build a memory-load-shaped tree.
    let src = SymContext::new_mock();
    let x = RustBV::symbolic(&src, "x", 32);
    let rev = x.reverse(&src); // Expression { Reverse, [x] }
    let pin = x.eq(&RustBV::concrete(0x11223344, 32), &src);

    // Translate into a freshly-allocated, independent context. `Z3_translate`
    // takes explicit source/dest contexts, so translation does NOT depend on
    // which context is currently thread-local.
    let original = Context::thread_local();
    let cfg = Config::new();
    let target = Context::new(&cfg);
    let target_ptr = target.get_z3_context().as_ptr() as usize;
    assert_ne!(
        original.get_z3_context().as_ptr() as usize,
        target_ptr,
        "test bug: target equals source context",
    );
    let rev_t = rev.translate_into(&target);
    let pin_t = pin.translate_into(&target);

    // Every translated leaf AST must now belong to the target context.
    let leaf_ctx = match &rev_t {
        RustBV::Expression { operands, .. } => match &operands[0] {
            RustBV::Symbolic { ast, .. } => ast.get_ctx().get_z3_context().as_ptr() as usize,
            other => panic!("expected symbolic leaf, got {other:?}"),
        },
        other => panic!("expected expression, got {other:?}"),
    };

    // Evaluate the translated tree under the target context and restore the
    // thread-local before asserting (a failure must not poison sibling tests).
    Context::set_thread_local(&target);
    let ctx_b = SymContext::new_mock();
    ctx_b.add_constraint(pin_t.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    let got = ctx_b.eval(&rev_t);
    Context::set_thread_local(&original);

    assert_eq!(
        leaf_ctx, target_ptr,
        "translated leaf AST must live in the target context",
    );
    assert_eq!(
        got,
        Some(0x44332211),
        "cross-context Reverse(x) eval must match the single-context result",
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_into_roundtrips_through_fresh_context() {
    // The bead's no-op correctness gate: round-trip A -> B -> A through a
    // fresh independent context; every constraint must still re-check equal.
    // (Translating into the *same* context is unsupported — `Z3_translate`
    // returns null — so a genuine round trip must bounce through a distinct
    // context, exactly as a worker hand-off and hand-back would.)
    let ctx = SymContext::new_mock(); // values live in default thread-local A
    let x = RustBV::symbolic(&ctx, "x", 64);
    let rev = x.reverse(&ctx);
    let pin = x.eq(&RustBV::concrete(0x0123456789ABCDEF, 64), &ctx);

    let a = z3::Context::thread_local(); // handle to context A
    let b = z3::Context::new(&z3::Config::new());
    let rev_back = rev.translate_into(&b).translate_into(&a);
    let pin_back = pin.translate_into(&b).translate_into(&a);

    // Thread-local is still A, so the round-tripped values eval directly.
    ctx.add_constraint(pin_back.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    assert_eq!(ctx.eval(&rev_back), Some(0xEFCDAB8967452301));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_into_concrete_identity() {
    // Concrete / Constrained carry no AST: translation is a cheap value clone
    // that stays valid in any context (it never references one).
    let target = z3::Context::new(&z3::Config::new());
    let c = RustBV::concrete(0xDEADBEEF, 32);
    let t = c.translate_into(&target);
    assert_eq!(t.as_u64(), Some(0xDEAD_BEEF));
    assert_eq!(t.width(), 32);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_translate_into_per_node_cost() {
    use std::time::Instant;

    // Mimic a real mid-run memory-load constraint: Reverse(Concat(64 byte
    // BVSes)). Only the 64 Symbolic leaves carry context-bound ASTs that need
    // `Z3_translate`; the Concat/Reverse Expression nodes recurse and reset
    // their lazy memo (rebuilt under whichever context is active), so they are
    // nearly free. This is the favorable shape the kill-gate predicted.
    let ctx = SymContext::new_mock();
    let leaves: Vec<RustBV> = (0..64u32)
        .map(|i| RustBV::symbolic(&ctx, format!("b{i}"), 8))
        .collect();
    let mut acc = leaves[0].clone();
    for b in &leaves[1..] {
        acc = acc.concat(b, &ctx);
    }
    let expr = acc.reverse(&ctx);
    let n_leaves = leaves.len();
    let n_nodes = n_leaves + (n_leaves - 1) + 1; // leaves + concats + reverse

    let target = z3::Context::new(&z3::Config::new());
    let iters = 200u32;
    // Warm up one translation (allocator / first-touch) before timing.
    let _ = expr.translate_into(&target);
    let start = Instant::now();
    for _ in 0..iters {
        let _ = expr.translate_into(&target);
    }
    let elapsed = start.elapsed();
    let per_node = elapsed.as_nanos() as f64 / (iters as f64 * n_nodes as f64);
    let per_leaf = elapsed.as_nanos() as f64 / (iters as f64 * n_leaves as f64);
    eprintln!(
        "translate_into cost: {n_nodes} nodes ({n_leaves} Z3 leaves), {iters} iters in {elapsed:?} \
         => {per_node:.0} ns/node, {per_leaf:.0} ns/leaf-translate",
    );
    // Loose sanity ceiling only — the precise number is the kill-gate
    // measurement (recorded in the bd note), not a tight CI assertion.
    assert!(
        per_node < 50_000.0,
        "translate cost {per_node:.0} ns/node is absurdly high — likely a regression",
    );
}

// =========================================================================
// Concrete shift/rotate amounts >= 2^32 and full-width edge cases (angr-ph300.30)
//
// The concrete arms of shl/lshr/ashr/rotl/rotr must reduce the amount BEFORE
// narrowing it to u32, matching the Z3/symbolic arms. A pre-narrow `a as u32`
// truncated amounts >= 2^32 to their low 32 bits, so e.g. an amount of 2^32
// read as 0. Full-width shifts must not wrap a u128 shift back to a no-op.
// =========================================================================

const HUGE_AMT: u128 = 1u128 << 32; // low 32 bits are all zero → the trap value

#[test]
fn test_shl_amount_at_2pow32_is_zero_not_identity() {
    let ctx = SymContext::new_mock();
    // 2^32 >= width(64) → claripy bvshl yields 0. Pre-fix: (2^32 as u32)=0 → v.
    let r = RustBV::concrete(5, 64).shl(&RustBV::concrete(HUGE_AMT, 64), &ctx);
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_lshr_amount_at_2pow32_is_zero() {
    let ctx = SymContext::new_mock();
    let r = RustBV::concrete(0xDEAD_BEEF, 64).lshr(&RustBV::concrete(HUGE_AMT, 64), &ctx);
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_ashr_amount_at_2pow32_saturates_to_sign() {
    let ctx = SymContext::new_mock();
    // Negative operand → all-sign; positive → 0.
    let neg =
        RustBV::concrete(0x8000_0000_0000_0000, 64).ashr(&RustBV::concrete(HUGE_AMT, 64), &ctx);
    assert_eq!(neg.as_u64(), Some(0xFFFF_FFFF_FFFF_FFFF));
    let pos =
        RustBV::concrete(0x4000_0000_0000_0000, 64).ashr(&RustBV::concrete(HUGE_AMT, 64), &ctx);
    assert_eq!(pos.as_u64(), Some(0));
}

#[test]
fn test_shl_full_width_128_is_zero() {
    let ctx = SymContext::new_mock();
    // amt == width == 128: pre-fix wrapping_shl(128) == shift-by-0 → v.
    let r = RustBV::concrete(0xFF, 128).shl(&RustBV::concrete(128, 128), &ctx);
    assert_eq!(r.as_u128(), Some(0));
}

#[test]
fn test_lshr_full_width_128_is_zero() {
    let ctx = SymContext::new_mock();
    // amt >= width == 128: pre-fix `wrapping_shr(amt as u32)` masked the amount
    // mod 128, so amt==128 shifted by 0 and returned v (angr-g35y0). The clamp
    // must return 0 to match the symbolic `c >= w → 0` arm. Cover the exact
    // boundary (128), a value between w and 2^32 (200), and one past the u32
    // truncation edge (2^32 + 3, whose low 32 bits are 3).
    let v: u128 = 0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210;
    for amt in [128u128, 200, (1u128 << 32) + 3] {
        let r = RustBV::concrete(v, 128).lshr(&RustBV::concrete(amt, 128), &ctx);
        assert_eq!(r.as_u128(), Some(0), "lshr by {amt} should be 0");
    }
}

#[test]
fn test_ashr_full_width_128_saturates_to_sign() {
    let ctx = SymContext::new_mock();
    // amt >= width == 128: pre-fix `signed >> (amt as u32)` either shifted an
    // i128 by 128 (debug abort) or masked mod 128 (angr-g35y0). Must saturate
    // to all sign bits, matching the symbolic `c >= w → SignExt(MSB)` arm.
    let neg: u128 = 1u128 << 127; // MSB set → negative
    let pos: u128 = 1u128 << 100; // MSB clear → positive
    for amt in [128u128, 200, (1u128 << 32) + 3] {
        let rn = RustBV::concrete(neg, 128).ashr(&RustBV::concrete(amt, 128), &ctx);
        assert_eq!(
            rn.as_u128(),
            Some(u128::MAX),
            "ashr(neg) by {amt} → all ones"
        );
        let rp = RustBV::concrete(pos, 128).ashr(&RustBV::concrete(amt, 128), &ctx);
        assert_eq!(rp.as_u128(), Some(0), "ashr(pos) by {amt} → 0");
    }
}

#[test]
fn test_shl_full_width_128_amounts_above_boundary() {
    let ctx = SymContext::new_mock();
    // Complement test_shl_full_width_128_is_zero (amt==128) with amounts above
    // the boundary and past the u32 truncation edge — all must yield 0.
    let v: u128 = 0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210;
    for amt in [200u128, (1u128 << 32) + 3] {
        let r = RustBV::concrete(v, 128).shl(&RustBV::concrete(amt, 128), &ctx);
        assert_eq!(r.as_u128(), Some(0), "shl by {amt} should be 0");
    }
}

#[test]
fn test_rotl_amount_at_2pow32_reduces_mod_width() {
    let ctx = SymContext::new_mock();
    // 2^32 % 48 == 16, so rotl by 2^32 == rotl by 16. Pre-fix rotated by 0.
    let by_huge = RustBV::concrete(0x1, 48).rotl(&RustBV::concrete(HUGE_AMT, 48), &ctx);
    let by_16 = RustBV::concrete(0x1, 48).rotl(&RustBV::concrete(16, 48), &ctx);
    assert_eq!(by_huge.as_u64(), by_16.as_u64());
    assert_eq!(by_huge.as_u64(), Some(1 << 16));
}

#[test]
fn test_rotr_amount_at_2pow32_reduces_mod_width() {
    let ctx = SymContext::new_mock();
    let by_huge = RustBV::concrete(1 << 16, 48).rotr(&RustBV::concrete(HUGE_AMT, 48), &ctx);
    let by_16 = RustBV::concrete(1 << 16, 48).rotr(&RustBV::concrete(16, 48), &ctx);
    assert_eq!(by_huge.as_u64(), by_16.as_u64());
    assert_eq!(by_huge.as_u64(), Some(1));
}

#[test]
fn test_rotl_width_128_zero_effective_amount_no_panic() {
    let ctx = SymContext::new_mock();
    // amt % 128 == 0 → `v >> (w - amt)` would shift u128 by 128 (debug abort
    // under panic=abort). Must return v unchanged.
    let v: u128 = 0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210;
    let r = RustBV::concrete(v, 128).rotl(&RustBV::concrete(256, 128), &ctx);
    assert_eq!(r.as_u128(), Some(v));
    let r2 = RustBV::concrete(v, 128).rotr(&RustBV::concrete(256, 128), &ctx);
    assert_eq!(r2.as_u128(), Some(v));
}

// =========================================================================
// rotl/rotr concrete-arm coverage (angr-ph300.33): before this the only
// direct rotate tests were the 2^32-truncation / 128-zero-amount edges above.
// These pin the ordinary hand-computed folds across widths {8,64} and the
// amount==w-1 boundary, plus the symbolic fallthrough arm and the algebraic
// inverse/mod-width identities.
// =========================================================================

#[test]
fn test_rotl_rotr_concrete_hand_computed() {
    let ctx = SymContext::new_mock();
    // w=8, 0x81 = 1000_0001.
    let x8 = RustBV::concrete(0x81, 8);
    assert_eq!(x8.rotl(&RustBV::concrete(0, 8), &ctx).as_u64(), Some(0x81));
    // rotl by 1 → 0000_0011 = 0x03.
    assert_eq!(x8.rotl(&RustBV::concrete(1, 8), &ctx).as_u64(), Some(0x03));
    // rotl by w-1 (7) → 1100_0000 = 0xC0 (== rotr by 1).
    assert_eq!(x8.rotl(&RustBV::concrete(7, 8), &ctx).as_u64(), Some(0xC0));
    assert_eq!(x8.rotr(&RustBV::concrete(1, 8), &ctx).as_u64(), Some(0xC0));
    // amt == w folds back to identity (a % w == 0).
    assert_eq!(x8.rotl(&RustBV::concrete(8, 8), &ctx).as_u64(), Some(0x81));

    // w=64: rotl(1, 63) sets the top bit; rotr(1, 1) does the same.
    let one64 = RustBV::concrete(1, 64);
    assert_eq!(
        one64.rotl(&RustBV::concrete(63, 64), &ctx).as_u64(),
        Some(0x8000_0000_0000_0000)
    );
    assert_eq!(
        one64.rotr(&RustBV::concrete(1, 64), &ctx).as_u64(),
        Some(0x8000_0000_0000_0000)
    );
}

#[test]
fn test_rotl_rotr_symbolic_builds_rotate_node() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let amt = RustBV::concrete(5, 32);
    // Symbolic operand → no fold, emit the dark RotL/RotR expr arms.
    match x.rotl(&amt, &ctx) {
        RustBV::Expression {
            op: BVOp::RotL,
            operands,
            width,
            ..
        } => {
            assert_eq!(width, 32);
            assert_eq!(operands.len(), 2);
            assert_eq!(operands[1].as_u128(), Some(5));
        }
        other => panic!("expected RotL node, got {other:?}"),
    }
    match x.rotr(&amt, &ctx) {
        RustBV::Expression {
            op: BVOp::RotR,
            operands,
            width,
            ..
        } => {
            assert_eq!(width, 32);
            assert_eq!(operands.len(), 2);
            assert_eq!(operands[1].as_u128(), Some(5));
        }
        other => panic!("expected RotR node, got {other:?}"),
    }
}

#[test]
fn test_rotl_rotr_concrete_identities() {
    let ctx = SymContext::new_mock();
    let v: u128 = 0xDEAD_BEEF_CAFE_1234;
    let x = RustBV::concrete(v, 64);
    // rotl(rotr(x, n), n) == x for an arbitrary n.
    let n = RustBV::concrete(13, 64);
    let round = x.rotr(&n, &ctx).rotl(&n, &ctx);
    assert_eq!(round.as_u128(), Some(v));
    // rotl(x, n) == rotl(x, n mod w): width 48, n = w+5 vs 5.
    let x48 = RustBV::concrete(0x1234_5678_9ABC, 48);
    let by_big = x48.rotl(&RustBV::concrete(48 + 5, 48), &ctx);
    let by_small = x48.rotl(&RustBV::concrete(5, 48), &ctx);
    assert_eq!(by_big.as_u64(), by_small.as_u64());
}

// angr-g2je6: SMT-LIB bvsdiv is total but asymmetric — x / 0 is -1 for x >= 0
// and +1 for x < 0. The concrete fold in `sdiv_into` used to return all-ones
// unconditionally, so sdiv(-5, 0) differed depending on whether the operands
// arrived concrete or symbolic-then-pinned.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_sdiv_by_zero_concrete_matches_z3() {
    for dividend in [5u128, 0, (-5i64) as u64 as u128, 1u128 << 63] {
        let ctx = SymContext::new_mock();
        let concrete = RustBV::concrete(dividend, 64).sdiv(&RustBV::concrete(0, 64), &ctx);
        let folded = concrete.as_u128().expect("concrete sdiv must fold");

        let d = RustBV::symbolic(&ctx, "sdiv_dvd", 64);
        let pinned = d.eq(&RustBV::concrete(dividend, 64), &ctx);
        ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
        let symbolic = d.sdiv(&RustBV::concrete(0, 64), &ctx);
        assert_eq!(
            Some(folded),
            ctx.eval(&symbolic),
            "sdiv({dividend:#x}, 0): concrete fold disagrees with Z3 bvsdiv"
        );
    }
}

// Sibling coverage for the ops that were already Z3-correct, so a future
// "simplification" of the zero-divisor arms cannot silently break them.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_udiv_srem_urem_by_zero_concrete_matches_z3() {
    for dividend in [7u128, 0, (-7i64) as u64 as u128] {
        let ctx = SymContext::new_mock();
        let d = RustBV::symbolic(&ctx, "dvd0", 64);
        let pinned = d.eq(&RustBV::concrete(dividend, 64), &ctx);
        ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
        let zero = RustBV::concrete(0, 64);
        let c = RustBV::concrete(dividend, 64);

        assert_eq!(
            c.udiv(&zero, &ctx).as_u128(),
            ctx.eval(&d.udiv(&zero, &ctx))
        );
        assert_eq!(
            c.urem(&zero, &ctx).as_u128(),
            ctx.eval(&d.urem(&zero, &ctx))
        );
        assert_eq!(
            c.srem(&zero, &ctx).as_u128(),
            ctx.eval(&d.srem(&zero, &ctx))
        );
    }
}

// angr-n0irt.15: the wrapping_div/wrapping_rem in sdiv_into/srem_into exist
// solely to survive the i128::MIN / -1 overflow, which plain `/`,`%` would
// panic on even in release (panic=abort -> SIGABRT). That path is only
// reachable at width 128: sign_extend's i128 result can equal i128::MIN only
// when the operand is 1u128<<127. The by-zero tests above are width-64 and
// never a non-zero divisor at width 128, so a revert to plain `/`,`%` during a
// future dedup pass would go uncaught. This pins the exact overflow case.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_sdiv_srem_min_by_neg_one_width_128() {
    let ctx = SymContext::new_mock();
    let min = 1u128 << 127; // i128::MIN reinterpreted as u128
    let neg_one = u128::MAX; // -1 at width 128
    let a = RustBV::concrete(min, 128);
    let b = RustBV::concrete(neg_one, 128);

    // Concrete folds must not panic and must match Z3 bvsdiv/bvsrem, which
    // wrap: MIN / -1 == MIN, MIN % -1 == 0.
    let sdiv = a.sdiv(&b, &ctx);
    let srem = a.srem(&b, &ctx);
    assert_eq!(
        sdiv.as_u128(),
        Some(min),
        "sdiv(i128::MIN, -1) must wrap to i128::MIN, not panic"
    );
    assert_eq!(
        srem.as_u128(),
        Some(0),
        "srem(i128::MIN, -1) must wrap to 0, not panic"
    );

    // Symbolic-pinned parity: same expression with a symbolic dividend pinned
    // to MIN, evaluated through Z3, must agree with the concrete fold.
    let d = RustBV::symbolic(&ctx, "sdiv128_min", 128);
    let pinned = d.eq(&RustBV::concrete(min, 128), &ctx);
    ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    assert_eq!(
        sdiv.as_u128(),
        ctx.eval(&d.sdiv(&b, &ctx)),
        "concrete sdiv fold disagrees with Z3 bvsdiv at width 128"
    );
    assert_eq!(
        srem.as_u128(),
        ctx.eval(&d.srem(&b, &ctx)),
        "concrete srem fold disagrees with Z3 bvsrem at width 128"
    );
}
