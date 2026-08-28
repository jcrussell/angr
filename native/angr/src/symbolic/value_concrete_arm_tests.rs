//! Tests pinning the *concrete* arms of shift/rotate/divide against their Z3
//! (symbolic) counterparts — the arms most prone to silently disagreeing on
//! amounts >= 2^32, full-width operands, and division by zero.

use super::*;

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
