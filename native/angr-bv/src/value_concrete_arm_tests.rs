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
    // 2^32 >= width(64) → claripy bvshl yields 0. Pre-fix: (2^32 as u32)=0 → v.
    let r = SymBV::concrete(5, 64).shl(&SymBV::concrete(HUGE_AMT, 64));
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_lshr_amount_at_2pow32_is_zero() {
    let r = SymBV::concrete(0xDEAD_BEEF, 64).lshr(&SymBV::concrete(HUGE_AMT, 64));
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_ashr_amount_at_2pow32_saturates_to_sign() {
    // Negative operand → all-sign; positive → 0.
    let neg = SymBV::concrete(0x8000_0000_0000_0000, 64).ashr(&SymBV::concrete(HUGE_AMT, 64));
    assert_eq!(neg.as_u64(), Some(0xFFFF_FFFF_FFFF_FFFF));
    let pos = SymBV::concrete(0x4000_0000_0000_0000, 64).ashr(&SymBV::concrete(HUGE_AMT, 64));
    assert_eq!(pos.as_u64(), Some(0));
}

#[test]
fn test_shl_full_width_128_is_zero() {
    // amt == width == 128: pre-fix wrapping_shl(128) == shift-by-0 → v.
    let r = SymBV::concrete(0xFF, 128).shl(&SymBV::concrete(128, 128));
    assert_eq!(r.as_u128(), Some(0));
}

#[test]
fn test_lshr_full_width_128_is_zero() {
    // amt >= width == 128: pre-fix `wrapping_shr(amt as u32)` masked the amount
    // mod 128, so amt==128 shifted by 0 and returned v (angr-g35y0). The clamp
    // must return 0 to match the symbolic `c >= w → 0` arm. Cover the exact
    // boundary (128), a value between w and 2^32 (200), and one past the u32
    // truncation edge (2^32 + 3, whose low 32 bits are 3).
    let v: u128 = 0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210;
    for amt in [128u128, 200, (1u128 << 32) + 3] {
        let r = SymBV::concrete(v, 128).lshr(&SymBV::concrete(amt, 128));
        assert_eq!(r.as_u128(), Some(0), "lshr by {amt} should be 0");
    }
}

#[test]
fn test_ashr_full_width_128_saturates_to_sign() {
    // amt >= width == 128: pre-fix `signed >> (amt as u32)` either shifted an
    // i128 by 128 (debug abort) or masked mod 128 (angr-g35y0). Must saturate
    // to all sign bits, matching the symbolic `c >= w → SignExt(MSB)` arm.
    let neg: u128 = 1u128 << 127; // MSB set → negative
    let pos: u128 = 1u128 << 100; // MSB clear → positive
    for amt in [128u128, 200, (1u128 << 32) + 3] {
        let rn = SymBV::concrete(neg, 128).ashr(&SymBV::concrete(amt, 128));
        assert_eq!(
            rn.as_u128(),
            Some(u128::MAX),
            "ashr(neg) by {amt} → all ones"
        );
        let rp = SymBV::concrete(pos, 128).ashr(&SymBV::concrete(amt, 128));
        assert_eq!(rp.as_u128(), Some(0), "ashr(pos) by {amt} → 0");
    }
}

#[test]
fn test_shl_full_width_128_amounts_above_boundary() {
    // Complement test_shl_full_width_128_is_zero (amt==128) with amounts above
    // the boundary and past the u32 truncation edge — all must yield 0.
    let v: u128 = 0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210;
    for amt in [200u128, (1u128 << 32) + 3] {
        let r = SymBV::concrete(v, 128).shl(&SymBV::concrete(amt, 128));
        assert_eq!(r.as_u128(), Some(0), "shl by {amt} should be 0");
    }
}

#[test]
fn test_rotl_amount_at_2pow32_reduces_mod_width() {
    // 2^32 % 48 == 16, so rotl by 2^32 == rotl by 16. Pre-fix rotated by 0.
    let by_huge = SymBV::concrete(0x1, 48).rotl(&SymBV::concrete(HUGE_AMT, 48));
    let by_16 = SymBV::concrete(0x1, 48).rotl(&SymBV::concrete(16, 48));
    assert_eq!(by_huge.as_u64(), by_16.as_u64());
    assert_eq!(by_huge.as_u64(), Some(1 << 16));
}

#[test]
fn test_rotr_amount_at_2pow32_reduces_mod_width() {
    let by_huge = SymBV::concrete(1 << 16, 48).rotr(&SymBV::concrete(HUGE_AMT, 48));
    let by_16 = SymBV::concrete(1 << 16, 48).rotr(&SymBV::concrete(16, 48));
    assert_eq!(by_huge.as_u64(), by_16.as_u64());
    assert_eq!(by_huge.as_u64(), Some(1));
}

#[test]
fn test_rotl_width_128_zero_effective_amount_no_panic() {
    // amt % 128 == 0 → `v >> (w - amt)` would shift u128 by 128 (debug abort
    // under panic=abort). Must return v unchanged.
    let v: u128 = 0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210;
    let r = SymBV::concrete(v, 128).rotl(&SymBV::concrete(256, 128));
    assert_eq!(r.as_u128(), Some(v));
    let r2 = SymBV::concrete(v, 128).rotr(&SymBV::concrete(256, 128));
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
    // w=8, 0x81 = 1000_0001.
    let x8 = SymBV::concrete(0x81, 8);
    assert_eq!(x8.rotl(&SymBV::concrete(0, 8)).as_u64(), Some(0x81));
    // rotl by 1 → 0000_0011 = 0x03.
    assert_eq!(x8.rotl(&SymBV::concrete(1, 8)).as_u64(), Some(0x03));
    // rotl by w-1 (7) → 1100_0000 = 0xC0 (== rotr by 1).
    assert_eq!(x8.rotl(&SymBV::concrete(7, 8)).as_u64(), Some(0xC0));
    assert_eq!(x8.rotr(&SymBV::concrete(1, 8)).as_u64(), Some(0xC0));
    // amt == w folds back to identity (a % w == 0).
    assert_eq!(x8.rotl(&SymBV::concrete(8, 8)).as_u64(), Some(0x81));

    // w=64: rotl(1, 63) sets the top bit; rotr(1, 1) does the same.
    let one64 = SymBV::concrete(1, 64);
    assert_eq!(
        one64.rotl(&SymBV::concrete(63, 64)).as_u64(),
        Some(0x8000_0000_0000_0000)
    );
    assert_eq!(
        one64.rotr(&SymBV::concrete(1, 64)).as_u64(),
        Some(0x8000_0000_0000_0000)
    );
}

#[test]
fn test_rotl_rotr_symbolic_builds_rotate_node() {
    let x = SymBV::symbol("x", 32);
    let amt = SymBV::concrete(5, 32);
    // Symbolic operand → no fold, emit the dark RotL/RotR expr arms.
    for (got, op) in [(x.rotl(&amt), BVOp::RotL), (x.rotr(&amt), BVOp::RotR)] {
        assert_eq!(got.width(), 32);
        let node = got
            .node()
            .unwrap_or_else(|| panic!("expected {op:?} node, got {got:?}"));
        assert_eq!(node.op, op);
        assert_eq!(node.operands.len(), 2);
        assert_eq!(node.operands[1].as_u128(), Some(5));
    }
}

#[test]
fn test_rotl_rotr_concrete_identities() {
    let v: u128 = 0xDEAD_BEEF_CAFE_1234;
    let x = SymBV::concrete(v, 64);
    // rotl(rotr(x, n), n) == x for an arbitrary n.
    let n = SymBV::concrete(13, 64);
    let round = x.rotr(&n).rotl(&n);
    assert_eq!(round.as_u128(), Some(v));
    // rotl(x, n) == rotl(x, n mod w): width 48, n = w+5 vs 5.
    let x48 = SymBV::concrete(0x1234_5678_9ABC, 48);
    let by_big = x48.rotl(&SymBV::concrete(48 + 5, 48));
    let by_small = x48.rotl(&SymBV::concrete(5, 48));
    assert_eq!(by_big.as_u64(), by_small.as_u64());
}
