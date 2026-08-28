//! Expression-simplification tests: algebraic identities, the shift/mul →
//! Concat rewrites, and the pre-Z3 `Extract` rewrite pass (angr-p8cz).

use super::value_tests_support::raw_extract_node;
use super::*;

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
fn test_reverse_non_byte_aligned_width_is_identity_not_truncation() {
    // angr-sqfj8.92: a byte reverse is undefined over a partial byte. The old
    // code guarded the precondition with a `debug_assert!` and, in release,
    // swapped only the `w / 8` whole bytes — silently zeroing the high bits of
    // a concrete operand (12-bit 0xABC came back as 0x0BC-swapped garbage).
    // Every profile now returns the operand unchanged instead, so no bit is
    // dropped and the concrete answer matches what the Z3 lowering already
    // produces for a misaligned width.
    let ctx = SymContext::new_mock();
    for (width, value) in [(12u32, 0xABCu128), (9, 0x1FF), (33, 0x1_2345_6789)] {
        let rev = RustBV::concrete(value, width).reverse(&ctx);
        assert_eq!(rev.width(), width, "width {width} must be preserved");
        assert_eq!(
            rev.as_u128(),
            Some(value),
            "width {width}: misaligned reverse must keep every bit"
        );
    }

    // Symbolic operands take the same no-op path, so no `Reverse` node is
    // built for a width whose claripy export would raise.
    let sym = RustBV::symbolic(&ctx, "misaligned", 12);
    let rev = sym.reverse(&ctx);
    assert!(matches!(rev, RustBV::Symbolic { .. }), "{rev:?}");
    assert_eq!(rev.width(), 12);

    // Byte-aligned widths still reverse normally.
    let aligned = RustBV::concrete(0x1122, 16).reverse(&ctx);
    assert_eq!(aligned.as_u128(), Some(0x2211));
}

#[test]
fn test_reverse_wide_concrete_declines_fold_instead_of_overshifting() {
    // angr-03vl4.60: a `Concrete` keeps only its low 128 bits, so a byte
    // reverse at width > 128 has no readable source for the bytes it must move
    // down — and the fold loop's `v >> (i * 8)` / `byte << ((n - 1 - i) * 8)`
    // shifts a u128 by up to `w - 8`, which aborts under `debug_assertions`
    // and silently wraps mod 128 in release. Same resolution as a non-trivial
    // wide rotate (see `define_rotate_pair`): decline the fold, stay symbolic.
    let ctx = SymContext::new_mock();
    for width in [136u32, 192, 256] {
        let rev = RustBV::concrete(0x1122_3344, width).reverse(&ctx);
        assert_eq!(rev.width(), width, "width {width} must be preserved");
        assert!(
            matches!(
                rev,
                RustBV::Expression {
                    op: BVOp::Reverse,
                    ..
                }
            ),
            "width {width}: wide reverse must stay symbolic, got {rev:?}"
        );
    }

    // The boundary width still folds concretely.
    let at_ceiling = RustBV::concrete(0xAA, 128).reverse(&ctx);
    assert_eq!(at_ceiling.as_u128(), Some(0xAA << 120));
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


#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_pre_z3_extract_over_reverse_byte_aligned() {
    // Extract a single byte from Reverse(x). With the rewrite pass the
    // Reverse should be eliminated entirely from the Z3 AST.
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let rev = x.reverse(&ctx);
    // Bypass extract_into via the raw-node helper: pretend this Extract was
    // built by truncate_into after the Reverse existed.
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
