//! Property-based tests for the VEX op-dispatch layer (`VEXOps::binop` /
//! `VEXOps::unop`), added under angr-qwyti.14.
//!
//! `value_ops_property_tests.rs` already covers the `RustBV` concrete-folding
//! primitives directly. These tests instead drive the *guest-facing* VEX op
//! dispatch: they build `IROp` variants and feed concrete operands through
//! `VEXOps::binop`/`unop`, so a regression in the dispatch table (wrong op
//! routed, wrong signedness, wrong width normalisation) is caught even when
//! the underlying `RustBV` primitive is correct. Every op here folds to a
//! `Concrete` node whose `as_u128()` is checked against an INDEPENDENT
//! plain-`u128` reference — never against another VEX op.
//!
//! Style mirrors `value_ops_property_tests.rs` (commit 5a9528e6a): quickcheck
//! properties for random coverage, plus deterministic `#[test]` sweeps over
//! boundary widths/values so a regression fails on every run rather than
//! probabilistically.

use super::*;

use quickcheck_macros::quickcheck;

/// Integer `IRType`s the scalar op dispatch accepts, paired with their width.
/// All are `as_u128`-readable and drive distinct width-normalisation paths.
const INT_TYPES: [IRType; 4] = [IRType::I8, IRType::I16, IRType::I32, IRType::I64];

/// Map a seed byte to one of the four integer types.
fn ty(seed: u8) -> IRType {
    INT_TYPES[(seed % 4) as usize]
}

/// Low-`width` mask (`width` in `1..=64` here — all int types fit a u64).
fn mask(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

/// Concrete BV of `ty`'s width; `concrete()` masks `v` for us.
fn bv(v: u128, t: IRType) -> RustBV {
    RustBV::concrete(v, t.bits())
}

/// Read back a concrete-folded op result; fails the property if the op did not
/// fold to a `Concrete` node (all inputs here are concrete, so it must).
fn val(b: &RustBV) -> u128 {
    b.as_u128()
        .expect("VEX op on concrete operands must fold to Concrete")
}

/// Two's-complement interpretation of the low `width` bits of `v` as i128.
fn to_signed(v: u128, width: u32) -> i128 {
    let m = mask(width);
    let vm = v & m;
    let sign = 1u128 << (width - 1);
    if vm & sign != 0 {
        // set the bits above `width` so the u128 pattern is the negative i128.
        (vm | !m) as i128
    } else {
        vm as i128
    }
}

// ---------------------------------------------------------------------------
// Logical: And / Or / Xor / Not
// ---------------------------------------------------------------------------

#[quickcheck]
fn prop_and_or_xor_match_reference(a: u128, b: u128, ts: u8) -> bool {
    let ctx = SymContext::new_mock();
    let t = ty(ts);
    let (am, bm, m) = (a & mask(t.bits()), b & mask(t.bits()), mask(t.bits()));
    let and = val(&VEXOps::binop(IROp::And(t), bv(a, t), bv(b, t), &ctx).unwrap());
    let or = val(&VEXOps::binop(IROp::Or(t), bv(a, t), bv(b, t), &ctx).unwrap());
    let xor = val(&VEXOps::binop(IROp::Xor(t), bv(a, t), bv(b, t), &ctx).unwrap());
    and == (am & bm) && or == (am | bm) && xor == ((am ^ bm) & m)
}

#[quickcheck]
fn prop_logic_commutative(a: u128, b: u128, ts: u8) -> bool {
    let ctx = SymContext::new_mock();
    let t = ty(ts);
    let ab = val(&VEXOps::binop(IROp::And(t), bv(a, t), bv(b, t), &ctx).unwrap());
    let ba = val(&VEXOps::binop(IROp::And(t), bv(b, t), bv(a, t), &ctx).unwrap());
    let oab = val(&VEXOps::binop(IROp::Or(t), bv(a, t), bv(b, t), &ctx).unwrap());
    let oba = val(&VEXOps::binop(IROp::Or(t), bv(b, t), bv(a, t), &ctx).unwrap());
    let xab = val(&VEXOps::binop(IROp::Xor(t), bv(a, t), bv(b, t), &ctx).unwrap());
    let xba = val(&VEXOps::binop(IROp::Xor(t), bv(b, t), bv(a, t), &ctx).unwrap());
    ab == ba && oab == oba && xab == xba
}

#[quickcheck]
fn prop_not_involution_and_demorgan(a: u128, b: u128, ts: u8) -> bool {
    let ctx = SymContext::new_mock();
    let t = ty(ts);
    let m = mask(t.bits());
    // !!x == x
    let n = VEXOps::unop(IROp::Not(t), bv(a, t), &ctx).unwrap();
    let nn = VEXOps::unop(IROp::Not(t), n, &ctx).unwrap();
    if val(&nn) != (a & m) {
        return false;
    }
    // De Morgan: !(x & y) == (!x) | (!y)
    let and = VEXOps::binop(IROp::And(t), bv(a, t), bv(b, t), &ctx).unwrap();
    let lhs = val(&VEXOps::unop(IROp::Not(t), and, &ctx).unwrap());
    let na = VEXOps::unop(IROp::Not(t), bv(a, t), &ctx).unwrap();
    let nb = VEXOps::unop(IROp::Not(t), bv(b, t), &ctx).unwrap();
    let rhs = val(&VEXOps::binop(IROp::Or(t), na, nb, &ctx).unwrap());
    lhs == rhs
}

// ---------------------------------------------------------------------------
// Shift: Shl / Shr (logical) / Sar (arithmetic)
// ---------------------------------------------------------------------------

/// Build the shift-amount operand at the operand's own width (the dispatch
/// normalises it to `left.width()` regardless). Amount kept in `0..width` so
/// the reference stays simple and no width-truncation of the amount occurs.
#[quickcheck]
fn prop_shifts_match_reference(v: u128, amt_seed: u8, ts: u8) -> bool {
    let ctx = SymContext::new_mock();
    let t = ty(ts);
    let w = t.bits();
    let amt = (amt_seed as u32) % w; // 0..w-1
    let vm = v & mask(w);
    let amtbv = RustBV::concrete(amt as u128, w);

    let shl = val(&VEXOps::binop(IROp::Shl(t), bv(v, t), amtbv.clone(), &ctx).unwrap());
    let shr = val(&VEXOps::binop(IROp::Shr(t), bv(v, t), amtbv.clone(), &ctx).unwrap());
    let sar = val(&VEXOps::binop(IROp::Sar(t), bv(v, t), amtbv, &ctx).unwrap());

    let want_shl = (vm << amt) & mask(w);
    let want_shr = vm >> amt; // logical
    let want_sar = ((to_signed(vm, w) >> amt) as u128) & mask(w); // arithmetic
    shl == want_shl && shr == want_shr && sar == want_sar
}

#[quickcheck]
fn prop_shift_by_zero_is_identity(v: u128, ts: u8) -> bool {
    let ctx = SymContext::new_mock();
    let t = ty(ts);
    let w = t.bits();
    let vm = v & mask(w);
    let z = RustBV::concrete(0, w);
    val(&VEXOps::binop(IROp::Shl(t), bv(v, t), z.clone(), &ctx).unwrap()) == vm
        && val(&VEXOps::binop(IROp::Shr(t), bv(v, t), z.clone(), &ctx).unwrap()) == vm
        && val(&VEXOps::binop(IROp::Sar(t), bv(v, t), z, &ctx).unwrap()) == vm
}

// ---------------------------------------------------------------------------
// Cmp: EQ / NE / LT / LE (signed) / LTU / LEU (unsigned)
// ---------------------------------------------------------------------------

#[quickcheck]
fn prop_cmp_eq_ne_complement(a: u128, b: u128, ts: u8) -> bool {
    let ctx = SymContext::new_mock();
    let t = ty(ts);
    let eq = val(&VEXOps::binop(IROp::CmpEQ(t), bv(a, t), bv(b, t), &ctx).unwrap());
    let ne = val(&VEXOps::binop(IROp::CmpNE(t), bv(a, t), bv(b, t), &ctx).unwrap());
    let want_eq = if (a & mask(t.bits())) == (b & mask(t.bits())) {
        1
    } else {
        0
    };
    // EQ matches reference, NE is its 1-bit complement, and EQ(x,x) == 1.
    eq == want_eq
        && ne == (want_eq ^ 1)
        && val(&VEXOps::binop(IROp::CmpEQ(t), bv(a, t), bv(a, t), &ctx).unwrap()) == 1
}

#[quickcheck]
fn prop_cmp_signed_unsigned_reference(a: u128, b: u128, ts: u8) -> bool {
    let ctx = SymContext::new_mock();
    let t = ty(ts);
    let w = t.bits();
    let (au, bu) = (a & mask(w), b & mask(w));
    let (ai, bi) = (to_signed(a, w), to_signed(b, w));

    let ltu = val(&VEXOps::binop(IROp::CmpLTU(t), bv(a, t), bv(b, t), &ctx).unwrap());
    let leu = val(&VEXOps::binop(IROp::CmpLEU(t), bv(a, t), bv(b, t), &ctx).unwrap());
    let lt = val(&VEXOps::binop(IROp::CmpLT(t), bv(a, t), bv(b, t), &ctx).unwrap());
    let le = val(&VEXOps::binop(IROp::CmpLE(t), bv(a, t), bv(b, t), &ctx).unwrap());

    ltu == u128::from(au < bu)
        && leu == u128::from(au <= bu)
        && lt == u128::from(ai < bi)
        && le == u128::from(ai <= bi)
}

// ---------------------------------------------------------------------------
// Conversion: SignExtend / ZeroExtend / Truncate round-trips
// ---------------------------------------------------------------------------

#[quickcheck]
fn prop_zero_extend_truncate_roundtrip(v: u128, from_seed: u8, to_seed: u8) -> bool {
    let ctx = SymContext::new_mock();
    // pick from <= to among the int types
    let (mut from, mut to) = (ty(from_seed), ty(to_seed));
    if from.bits() > to.bits() {
        core::mem::swap(&mut from, &mut to);
    }
    let vm = v & mask(from.bits());
    let ze = VEXOps::unop(IROp::ZeroExtend { from, to }, bv(v, from), &ctx).unwrap();
    // zero-extend preserves the low value...
    if val(&ze) != vm {
        return false;
    }
    // ...and truncating back recovers it.
    let tr = VEXOps::unop(IROp::Truncate { from: to, to: from }, ze, &ctx).unwrap();
    val(&tr) == vm
}

#[quickcheck]
fn prop_sign_extend_reference(v: u128, from_seed: u8, to_seed: u8) -> bool {
    let ctx = SymContext::new_mock();
    let (mut from, mut to) = (ty(from_seed), ty(to_seed));
    if from.bits() > to.bits() {
        core::mem::swap(&mut from, &mut to);
    }
    let (fw, tw) = (from.bits(), to.bits());
    // Reference: reinterpret the low `fw` bits as signed, mask to `tw`.
    let want = (to_signed(v, fw) as u128) & mask(tw);
    let se = VEXOps::unop(IROp::SignExtend { from, to }, bv(v, from), &ctx).unwrap();
    val(&se) == want
}

// ---------------------------------------------------------------------------
// Deterministic boundary sweeps (mirror value_ops_property_tests.rs)
//
// The quickcheck props above sample widths/amounts uniformly, so the exact
// off-by-one boundaries (shift by width-1, sign bit set, all-ones) are hit
// only probabilistically. These `#[test]` fns cross every int type with a
// fixed adversarial value/amount set so any single-boundary regression fails
// on every run.
// ---------------------------------------------------------------------------

/// Adversarial concrete values for a `width`-bit operand: zero, one, all-ones,
/// sign bit alone, sign clear / low set, and the two alternating patterns.
fn boundary_values(width: u32) -> Vec<u128> {
    let m = mask(width);
    let sign = 1u128 << (width - 1);
    let alt_a = 0xAAAA_AAAA_AAAA_AAAAu128 & m;
    let alt_5 = 0x5555_5555_5555_5555u128 & m;
    vec![0, 1, m, sign, m ^ sign, alt_a, alt_5]
}

#[test]
fn boundary_shifts_match_reference() {
    let ctx = SymContext::new_mock();
    for &t in &INT_TYPES {
        let w = t.bits();
        for &v in &boundary_values(w) {
            for amt in [0u32, 1, w / 2, w - 1] {
                let amtbv = RustBV::concrete(amt as u128, w);
                let vm = v & mask(w);
                let shl = val(&VEXOps::binop(IROp::Shl(t), bv(v, t), amtbv.clone(), &ctx).unwrap());
                let shr = val(&VEXOps::binop(IROp::Shr(t), bv(v, t), amtbv.clone(), &ctx).unwrap());
                let sar = val(&VEXOps::binop(IROp::Sar(t), bv(v, t), amtbv, &ctx).unwrap());
                assert_eq!(shl, (vm << amt) & mask(w), "shl w={w} v={v:#x} amt={amt}");
                assert_eq!(shr, vm >> amt, "shr w={w} v={v:#x} amt={amt}");
                assert_eq!(
                    sar,
                    ((to_signed(vm, w) >> amt) as u128) & mask(w),
                    "sar w={w} v={v:#x} amt={amt}"
                );
            }
        }
    }
}

#[test]
fn boundary_cmp_signed_vs_unsigned_diverge_on_sign_bit() {
    let ctx = SymContext::new_mock();
    for &t in &INT_TYPES {
        let w = t.bits();
        // A value with only the sign bit set is the largest unsigned but the
        // most-negative signed — the classic case that separates LT from LTU.
        let neg = 1u128 << (w - 1); // most-negative signed / large unsigned
        let one = 1u128;
        // signed: neg < 1  => CmpLT == 1 ; unsigned: neg > 1 => CmpLTU == 0
        assert_eq!(
            val(&VEXOps::binop(IROp::CmpLT(t), bv(neg, t), bv(one, t), &ctx).unwrap()),
            1,
            "signed neg<1 w={w}"
        );
        assert_eq!(
            val(&VEXOps::binop(IROp::CmpLTU(t), bv(neg, t), bv(one, t), &ctx).unwrap()),
            0,
            "unsigned neg>1 w={w}"
        );
    }
}

#[test]
fn boundary_sign_vs_zero_extend_diverge_on_negative() {
    let ctx = SymContext::new_mock();
    // from I8 (-1 = 0xFF) to each wider type: sign-extend fills ones, zero
    // fills zeros. Directly checks the two conversion ops do not alias.
    let from = IRType::I8;
    for &to in &INT_TYPES {
        if to.bits() <= from.bits() {
            continue;
        }
        let tw = to.bits();
        let se = val(&VEXOps::unop(IROp::SignExtend { from, to }, bv(0xFF, from), &ctx).unwrap());
        let ze = val(&VEXOps::unop(IROp::ZeroExtend { from, to }, bv(0xFF, from), &ctx).unwrap());
        assert_eq!(se, mask(tw), "sext 0xFF -> {tw} must be all-ones");
        assert_eq!(ze, 0xFF, "zext 0xFF -> {tw} must be 0xFF");
    }
}
