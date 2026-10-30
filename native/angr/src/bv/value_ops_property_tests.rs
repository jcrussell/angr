//! Property-based tests for `value_ops.rs` (RustBV concrete-folding ops)
//! (angr-ph300.1). The `bv_codec` round-trips and the Z3 cross-checks are in
//! `symbolic/value_ops_z3_property_tests.rs`.
//!
//! `quickcheck`/`quickcheck_macros` were already declared in
//! `native/angr/Cargo.toml` `[dev-dependencies]` but had zero uses; these
//! are the first. Every RustBV op here is exercised on *concrete* operands so
//! it constant-folds to a `Concrete` node whose value `as_u128()` reads back —
//! the property then checks the folded result against a plain-`u128` reference
//! computation. This covers the wide op surface (arithmetic / bitwise / shifts
//! / structural) with random inputs the hand-written `value_*_tests.rs` examples
//! cannot.
//!
//! Declared as a child of `value_ops` so `use super::*` reaches the module's
//! private free helpers (`sign_extend`, `sign_extend_to`).

use super::*;

/// The op of a compound value, whichever arm holds it.
fn op_of(bv: &RustBV) -> Option<BVOp> {
    bv.node().map(|n| n.op)
}
use quickcheck_macros::quickcheck;

/// Map an arbitrary seed byte to a legal BV width in `1..=128`.
fn w128(seed: u8) -> u32 {
    (seed % 128) as u32 + 1
}

/// Map an arbitrary seed byte to a "narrow" width in `1..=64` — used where two
/// operands are concatenated (so `wa + wb <= 128` stays `as_u128`-readable).
fn w64(seed: u8) -> u32 {
    (seed % 64) as u32 + 1
}

/// Low-`width` mask (`width` in `1..=128`).
fn mask(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

/// Concrete RustBV; `concrete()` masks `v` to `width` for us.
fn bv(v: u128, width: u32) -> RustBV {
    RustBV::concrete(v, width)
}

/// Read back a concrete-folded op result. Panics (fails the property) if the op
/// unexpectedly did not fold to a concrete node.
fn val(b: &RustBV) -> u128 {
    b.as_u128()
        .expect("op on concrete operands must fold to Concrete")
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

#[quickcheck]
fn prop_add_commutative(a: u128, b: u128, ws: u8) -> bool {
    let w = w128(ws);
    val(&bv(a, w).add(&bv(b, w))) == val(&bv(b, w).add(&bv(a, w)))
}

#[quickcheck]
fn prop_add_zero_identity(a: u128, ws: u8) -> bool {
    let w = w128(ws);
    val(&bv(a, w).add(&bv(0, w))) == (a & mask(w))
}

#[quickcheck]
fn prop_add_matches_wrapping(a: u128, b: u128, ws: u8) -> bool {
    let w = w128(ws);
    let expected = (a & mask(w)).wrapping_add(b & mask(w)) & mask(w);
    val(&bv(a, w).add(&bv(b, w))) == expected
}

#[quickcheck]
fn prop_sub_matches_wrapping(a: u128, b: u128, ws: u8) -> bool {
    let w = w128(ws);
    let expected = (a & mask(w)).wrapping_sub(b & mask(w)) & mask(w);
    val(&bv(a, w).sub(&bv(b, w))) == expected
}

#[quickcheck]
fn prop_sub_self_is_zero(a: u128, ws: u8) -> bool {
    let w = w128(ws);
    val(&bv(a, w).sub(&bv(a, w))) == 0
}

#[quickcheck]
fn prop_mul_commutative(a: u128, b: u128, ws: u8) -> bool {
    let w = w128(ws);
    val(&bv(a, w).mul(&bv(b, w))) == val(&bv(b, w).mul(&bv(a, w)))
}

#[quickcheck]
fn prop_mul_zero_and_one(a: u128, ws: u8) -> bool {
    let w = w128(ws);
    val(&bv(a, w).mul(&bv(0, w))) == 0 && val(&bv(a, w).mul(&bv(1, w))) == (a & mask(w))
}

#[quickcheck]
fn prop_mul_matches_wrapping(a: u128, b: u128, ws: u8) -> bool {
    let w = w128(ws);
    let expected = (a & mask(w)).wrapping_mul(b & mask(w)) & mask(w);
    val(&bv(a, w).mul(&bv(b, w))) == expected
}

#[quickcheck]
fn prop_neg_involution(a: u128, ws: u8) -> bool {
    let w = w128(ws);
    val(&bv(a, w).neg().neg()) == (a & mask(w))
}

#[quickcheck]
fn prop_neg_add_is_zero(a: u128, ws: u8) -> bool {
    let w = w128(ws);
    let x = bv(a, w);
    val(&x.add(&x.neg())) == 0
}

// ---------------------------------------------------------------------------
// Bitwise
// ---------------------------------------------------------------------------

#[quickcheck]
fn prop_and_absorption(a: u128, ws: u8) -> bool {
    let w = w128(ws);
    let x = bv(a, w);
    // x & x == x, x & 0 == 0, x & ones == x
    val(&x.and(&x)) == (a & mask(w))
        && val(&x.and(&bv(0, w))) == 0
        && val(&x.and(&bv(mask(w), w))) == (a & mask(w))
}

#[quickcheck]
fn prop_or_absorption(a: u128, ws: u8) -> bool {
    let w = w128(ws);
    let x = bv(a, w);
    // x | x == x, x | 0 == x, x | ones == ones
    val(&x.or(&x)) == (a & mask(w))
        && val(&x.or(&bv(0, w))) == (a & mask(w))
        && val(&x.or(&bv(mask(w), w))) == mask(w)
}

#[quickcheck]
fn prop_xor_laws(a: u128, b: u128, ws: u8) -> bool {
    let w = w128(ws);
    let x = bv(a, w);
    let y = bv(b, w);
    // x ^ x == 0, x ^ 0 == x, (x ^ y) ^ y == x, commutative
    val(&x.xor(&x)) == 0
        && val(&x.xor(&bv(0, w))) == (a & mask(w))
        && val(&x.xor(&y).xor(&y)) == (a & mask(w))
        && val(&x.xor(&y)) == val(&y.xor(&x))
}

#[quickcheck]
fn prop_not_involution_and_mask(a: u128, ws: u8) -> bool {
    let w = w128(ws);
    let x = bv(a, w);
    // !!x == x, !x == x ^ ones
    val(&x.not().not()) == (a & mask(w)) && val(&x.not()) == ((a & mask(w)) ^ mask(w))
}

// ---------------------------------------------------------------------------
// Shifts
// ---------------------------------------------------------------------------

#[quickcheck]
fn prop_shl_matches_reference(v: u128, amt: u8, ws: u8) -> bool {
    let w = w128(ws);
    // The amount BV is masked to `w` by concrete(), so the effective shift is
    // `amt & mask(w)`; mirror that in the reference.
    let amt_eff = (amt as u128) & mask(w);
    let vm = v & mask(w);
    let expected = if amt_eff >= w as u128 {
        0
    } else {
        vm.wrapping_shl(amt_eff as u32) & mask(w)
    };
    val(&bv(v, w).shl(&bv(amt as u128, w))) == expected
}

#[quickcheck]
fn prop_lshr_matches_reference(v: u128, amt: u8, ws: u8) -> bool {
    let w = w128(ws);
    let amt_eff = (amt as u128) & mask(w);
    let vm = v & mask(w);
    let expected = if amt_eff >= w as u128 {
        0
    } else {
        vm.wrapping_shr(amt_eff as u32) & mask(w)
    };
    val(&bv(v, w).lshr(&bv(amt as u128, w))) == expected
}

#[quickcheck]
fn prop_ashr_matches_reference(v: u128, amt: u8, ws: u8) -> bool {
    let w = w128(ws);
    let amt_eff = (amt as u128) & mask(w);
    // ashr saturates the amount to w-1 (all sign bits).
    let amt_clamped = amt_eff.min((w - 1) as u128) as u32;
    let signed = sign_extend(v & mask(w), w);
    let expected = (signed >> amt_clamped) as u128 & mask(w);
    val(&bv(v, w).ashr(&bv(amt as u128, w))) == expected
}

// ---------------------------------------------------------------------------
// Structural
// ---------------------------------------------------------------------------

#[quickcheck]
fn prop_concat_extract_inverse(a: u128, b: u128, was: u8, wbs: u8) -> bool {
    let (wa, wb) = (w64(was), w64(wbs));
    let hi = bv(a, wa);
    let lo = bv(b, wb);
    let c = hi.concat(&lo); // width wa + wb
    let top = c.extract(wa + wb - 1, wb); // recovers hi
    let bot = c.extract(wb - 1, 0); // recovers lo
    val(&top) == (a & mask(wa)) && val(&bot) == (b & mask(wb))
}

#[quickcheck]
fn prop_zero_extend_truncate_roundtrip(v: u128, ws: u8, extra: u8) -> bool {
    let w = w64(ws);
    let w2 = w + (extra % 64) as u32; // >= w, <= 127
    let x = bv(v, w);
    let ze = x.zero_extend(w2);
    // zero-extend preserves the low value, and truncating back recovers it.
    val(&ze) == (v & mask(w)) && val(&ze.truncate(w)) == (v & mask(w))
}

#[quickcheck]
fn prop_sign_extend_matches_reference(v: u128, ws: u8, extra: u8) -> bool {
    let w = w64(ws);
    let w2 = w + (extra % 64) as u32; // >= w, <= 127
    // w2 <= 127, so the fold never declines (see `sign_extend_to`).
    let expected = sign_extend_to(v & mask(w), w, w2).expect("w2 <= 128 always folds") & mask(w2);
    val(&bv(v, w).sign_extend(w2)) == expected
}

#[quickcheck]
fn prop_eq_matches_reference(a: u128, b: u128, ws: u8) -> bool {
    let w = w128(ws);
    let want = if (a & mask(w)) == (b & mask(w)) { 1 } else { 0 };
    // eq is reflexive and its concrete result is a 1-bit 0/1.
    val(&bv(a, w).eq(&bv(b, w))) == want && val(&bv(a, w).eq(&bv(a, w))) == 1
}

// ---------------------------------------------------------------------------
// Width-boundary exhaustive checks (angr-qwyti.5)
//
// The quickcheck properties above sample widths uniformly via `w128`/`w64`, so
// the specific off-by-one boundaries that produced the width/truncation bug
// class (scanf %hd/%hhd, sprintf h/hh, LoadG widening, extract width>=128,
// shift/rotate cast, make_bv_from_bytes, ...) are hit only probabilistically.
// These `#[test]` fns *deterministically* cross every width in `BOUNDARY_WIDTHS`
// with a fixed set of adversarial values against a plain-`u128` reference, so a
// regression on any single boundary fails on every run rather than 1/128 of the
// time.
// ---------------------------------------------------------------------------

/// The widths actually load-bearing in the codebase: byte/half/word/dword/qword
/// sizes and each ±1 neighbour, plus 65/127/128 for the >64-bit codec paths.
const BOUNDARY_WIDTHS: [u32; 12] = [1, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128];

/// Adversarial concrete values for a `width`-bit BV: all-zeros, all-ones, the
/// sign bit alone, sign bit clear / low bit set, and the two alternating
/// patterns. Each is pre-masked to `width`.
fn boundary_values(width: u32) -> Vec<u128> {
    let m = mask(width);
    let sign = if width == 0 { 0 } else { 1u128 << (width - 1) };
    let alt_a = 0xAAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAAu128 & m;
    let alt_5 = 0x5555_5555_5555_5555_5555_5555_5555_5555u128 & m;
    vec![0, 1, m, sign, m ^ sign, alt_a, alt_5]
}

#[test]
fn boundary_truncate_matches_mask() {
    for &w in &BOUNDARY_WIDTHS {
        for &to in &BOUNDARY_WIDTHS {
            if to > w {
                continue; // truncate only narrows
            }
            for &v in &boundary_values(w) {
                let got = val(&bv(v, w).truncate(to));
                assert_eq!(got, v & mask(to), "truncate({w}->{to}) v={v:#x}");
            }
        }
    }
}

#[test]
fn boundary_zero_extend_preserves_low_and_zeros_high() {
    for &w in &BOUNDARY_WIDTHS {
        for &to in &BOUNDARY_WIDTHS {
            if to < w {
                continue; // zero_extend only widens
            }
            for &v in &boundary_values(w) {
                let ze = bv(v, w).zero_extend(to);
                // Low `w` bits unchanged, high bits zero => value == masked src.
                assert_eq!(val(&ze), v & mask(w), "zext({w}->{to}) v={v:#x}");
                // Round-trip back down recovers the source exactly.
                assert_eq!(
                    val(&ze.truncate(w)),
                    v & mask(w),
                    "zext-trunc({w}->{to}->{w}) v={v:#x}"
                );
            }
        }
    }
}

#[test]
fn boundary_sign_extend_matches_reference() {
    for &w in &BOUNDARY_WIDTHS {
        for &to in &BOUNDARY_WIDTHS {
            if to < w {
                continue; // sign_extend only widens
            }
            for &v in &boundary_values(w) {
                let want =
                    sign_extend_to(v & mask(w), w, to).expect("to <= 128 always folds") & mask(to);
                let got = val(&bv(v, w).sign_extend(to));
                assert_eq!(got, want, "sext({w}->{to}) v={v:#x}");
            }
        }
    }
}

/// Regression for the `sign_extend_to` `1u128 << 128` overflow (angr-qwyti.16):
/// widening a negative value to exactly 128 bits overflowed the mask shift,
/// SIGABRTing under `panic = "abort"` and silently zero-extending in release.
/// `boundary_sign_extend_matches_reference` above cannot catch it — it uses
/// `sign_extend_to` as its own reference, so a wrong `sign_extend_to` matches a
/// wrong `sign_extend`. Here the reference is the *independent* `sign_extend`
/// i128 helper (a distinct code path, correct for every `from_width < 128`).
#[test]
fn sign_extend_to_128_matches_i128_reference() {
    for &w in &BOUNDARY_WIDTHS {
        if w >= 128 {
            continue; // no widening past the u128 ceiling
        }
        for &v in &boundary_values(w) {
            let src = v & mask(w);
            // Reference: sign_extend() returns the i128 two's-complement value;
            // its bit pattern IS the 128-bit sign extension.
            let want = sign_extend(src, w) as u128;
            let got = val(&bv(src, w).sign_extend(128));
            assert_eq!(got, want, "sext({w}->128) v={src:#x}");
            // Spot-check the classic case directly: a negative 64-bit value must
            // gain a solid run of high ones, not zero-extend.
            if w == 64 && src == 0x8000_0000_0000_0000 {
                assert_eq!(
                    got, 0xFFFF_FFFF_FFFF_FFFF_8000_0000_0000_0000,
                    "sext of i64::MIN to 128 must fill the top with ones"
                );
            }
        }
    }
}

#[test]
fn boundary_extract_single_bits_and_slices() {
    for &w in &BOUNDARY_WIDTHS {
        for &v in &boundary_values(w) {
            let x = bv(v, w);
            // Every individual bit position.
            for i in 0..w {
                let got = val(&x.extract(i, i));
                assert_eq!(got, (v >> i) & 1, "extract bit {i} of w={w} v={v:#x}");
            }
            // Low slice [to-1:0] for each boundary width `to <= w`, plus the
            // top-anchored slice [w-1 : w-to].
            for &to in &BOUNDARY_WIDTHS {
                if to > w {
                    continue;
                }
                let low = val(&x.extract(to - 1, 0));
                assert_eq!(low, v & mask(to), "extract low {to} of w={w} v={v:#x}");
                let hi = val(&x.extract(w - 1, w - to));
                assert_eq!(
                    hi,
                    (v >> (w - to)) & mask(to),
                    "extract top {to} of w={w} v={v:#x}"
                );
            }
        }
    }
}

#[test]
fn boundary_concat_recovers_operands() {
    // Restrict to pairs whose sum stays `as_u128`-readable (<= 128).
    for &wa in &BOUNDARY_WIDTHS {
        for &wb in &BOUNDARY_WIDTHS {
            if wa + wb > 128 {
                continue;
            }
            for &a in &boundary_values(wa) {
                for &b in &boundary_values(wb) {
                    let c = bv(a, wa).concat(&bv(b, wb)); // width wa+wb
                    let top = c.extract(wa + wb - 1, wb);
                    let bot = c.extract(wb - 1, 0);
                    assert_eq!(val(&top), a & mask(wa), "concat hi wa={wa} wb={wb}");
                    assert_eq!(val(&bot), b & mask(wb), "concat lo wa={wa} wb={wb}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Wider-than-128 concrete fast-path checks (angr-qwyti.12)
//
// `RustBV::Concrete` stores its value in a u128, but a `Concrete` node may
// legitimately carry a `width > 128` when only the low 128 bits are non-zero
// (e.g. `zero_extend`/`truncate` fast paths return `concrete(v, to_width)` for
// any `to_width`). Every bit at position >= 128 of such a node is *logically
// zero*. The concrete fast paths that internally shift/mask a u128 by a width
// or bit-position (shl/lshr/ashr, sign_extend) previously assumed width <= 128
// and would overflow (`1u128 << 129`, `i128 >> 200`) or wrap (`wrapping_shl(a)`
// with `a mod 128`) at these widths -- the exact `1u128 << 128`-class hazard
// that has been fixed before (assembly-side, slicing-side, sign_extend_to,
// sar_fill_mask) (angr-tk7yv, angr-dondi, angr-qwyti.16, angr-qwyti.17).
// These deterministic checks cross widths > 128 with adversarial values and an
// INDEPENDENT plain-u128 reference so a regression fails on every run, not
// 1/128 of the time.
// ---------------------------------------------------------------------------

/// Widths past the u128 storage ceiling: 128+1, and the AVX/YMM-scale widths a
/// `zero_extend` of a smaller concrete would produce.
const WIDE_WIDTHS: [u32; 4] = [129, 160, 192, 256];

/// Adversarial low-128-bit payloads for a wide `Concrete` (bits >= 128 are
/// always zero by construction). Cannot reuse `boundary_values` -- its
/// `1u128 << (width - 1)` sign term overflows for `width > 128`.
fn wide_payloads() -> [u128; 7] {
    [
        0,
        1,
        u128::MAX,
        1u128 << 127,
        (1u128 << 127) - 1,
        0xAAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAAu128,
        0x5555_5555_5555_5555_5555_5555_5555_5555u128,
    ]
}

/// Shift amounts that straddle the 128-bit storage boundary for each wide
/// width `w`: below, at, just past, and at/over the declared width.
fn wide_shift_amounts(w: u32) -> [u128; 8] {
    [0, 1, 63, 127, 128, 129, (w - 1) as u128, w as u128]
}

/// `true` when `v << a` sets a bit at or past position 128, the ceiling of a
/// `Concrete`'s u128 payload.
///
/// Phrased in terms of `v`'s highest set bit (`127 - leading_zeros`) rather
/// than `shl_into`'s own `checked_mul`, so the oracle below stays an
/// independent statement of the rule instead of a restatement of the code
/// under test (angr-fs8kb.26).
fn shl_escapes_payload(v: u128, a: u128) -> bool {
    v != 0 && a > u128::from(v.leading_zeros())
}

#[test]
fn wide_shl_declines_the_fold_when_bits_escape_the_payload() {
    for &w in &WIDE_WIDTHS {
        for &v in &wide_payloads() {
            for &a in &wide_shift_amounts(w) {
                let got = bv(v, w).shl(&bv(a, w));
                assert_eq!(got.width(), w, "shl width w={w} v={v:#x} a={a}");
                if a >= u128::from(w) {
                    // Every bit leaves the declared width: 0 at any width.
                    assert_eq!(val(&got), 0, "shl w={w} v={v:#x} a={a}");
                } else if shl_escapes_payload(v, a) {
                    // The true result (e.g. `2^127 << 1` = `2^128`, an ordinary
                    // 256-bit number) needs a bit the u128 payload cannot hold,
                    // so the fold must be declined — the same choice
                    // `define_rotate_pair!` and `reverse_owned` make.
                    assert!(
                        got.as_u128().is_none(),
                        "shl must not fold w={w} v={v:#x} a={a}"
                    );
                    assert_eq!(op_of(&got), Some(BVOp::Shl), "shl op w={w} v={v:#x} a={a}");
                } else {
                    // Whole result fits the payload. Oracle by repeated
                    // doubling, which cannot overflow here and shares no code
                    // with `shl_into`'s `wrapping_shl`.
                    let mut want = v;
                    for _ in 0..a {
                        want *= 2;
                    }
                    assert_eq!(val(&got), want, "shl w={w} v={v:#x} a={a}");
                }
            }
        }
    }
}

#[test]
fn wide_lshr_clears_readable_bits_past_128() {
    for &w in &WIDE_WIDTHS {
        for &v in &wide_payloads() {
            for &a in &wide_shift_amounts(w) {
                let want = if a >= 128 {
                    0
                } else {
                    v.wrapping_shr(a as u32)
                };
                let got = val(&bv(v, w).lshr(&bv(a, w)));
                assert_eq!(got, want, "lshr w={w} v={v:#x} a={a}");
            }
        }
    }
}

#[test]
fn wide_ashr_is_logical_because_sign_bit_is_zero() {
    for &w in &WIDE_WIDTHS {
        for &v in &wide_payloads() {
            for &a in &wide_shift_amounts(w) {
                // The true sign bit (position w-1 >= 128) is a logical zero, so
                // arithmetic shift right equals logical shift right -- even when
                // bit 127 of the stored payload is set.
                let want = if a >= 128 {
                    0
                } else {
                    v.wrapping_shr(a as u32)
                };
                let got = val(&bv(v, w).ashr(&bv(a, w)));
                assert_eq!(got, want, "ashr w={w} v={v:#x} a={a}");
            }
        }
    }
}

#[test]
fn wide_sign_extend_is_noop_because_source_is_nonnegative() {
    for &w in &WIDE_WIDTHS {
        for &to in &WIDE_WIDTHS {
            if to <= w {
                continue; // sign_extend only widens
            }
            for &v in &wide_payloads() {
                // Sign bit at position w-1 >= 128 is zero => sign-extend == the
                // value itself. Must NOT overflow `1u128 << (from_width - 1)`.
                let got = val(&bv(v, w).sign_extend(to));
                assert_eq!(got, v, "sext({w}->{to}) v={v:#x}");
            }
        }
    }
}

#[test]
fn wide_zero_extend_and_truncate_roundtrip() {
    for &w in &WIDE_WIDTHS {
        for &v in &wide_payloads() {
            // zero_extend to a larger wide width preserves the payload.
            for &to in &WIDE_WIDTHS {
                if to <= w {
                    continue;
                }
                let ze = bv(v, w).zero_extend(to);
                assert_eq!(val(&ze), v, "zext({w}->{to}) v={v:#x}");
            }
            // truncate down to each boundary width recovers the masked payload.
            for &to in &BOUNDARY_WIDTHS {
                let got = val(&bv(v, w).truncate(to));
                assert_eq!(got, v & mask(to), "trunc({w}->{to}) v={v:#x}");
            }
        }
    }
}

#[test]
fn wide_extract_reads_zero_past_bit_128() {
    for &w in &WIDE_WIDTHS {
        for &v in &wide_payloads() {
            let x = bv(v, w);
            // A slice entirely at/above bit 128 is all zeros.
            let top = x.extract(w - 1, 128);
            assert_eq!(val(&top), 0, "extract top [{}:128] w={w} v={v:#x}", w - 1);
            // The low 128 bits round-trip exactly.
            let low = x.extract(127, 0);
            assert_eq!(val(&low), v, "extract low [127:0] w={w} v={v:#x}");
            // A slice straddling bit 128 keeps only the in-range payload bits.
            // Needs `high < w`, so only for widths comfortably past 128.
            if w >= 136 {
                let straddle = x.extract(135, 120); // 16-bit slice [135:120]
                let want = (v >> 120) & mask(16);
                assert_eq!(val(&straddle), want, "extract [135:120] w={w} v={v:#x}");
            }
        }
    }
}

// angr-03vl4.58: signed concrete fast paths at width > 128.
#[test]
fn wide_signed_compare_degenerates_to_unsigned() {
    for &w in &WIDE_WIDTHS {
        for &a in &wide_payloads() {
            for &b in &wide_payloads() {
                // The true sign bit (position w-1 >= 128) is a logical zero on
                // both sides, so every operand is non-negative and the signed
                // comparisons agree with the unsigned u128 ones -- even when
                // bit 127 of a stored payload is set.
                let (x, y) = (bv(a, w), bv(b, w));
                let checks = [
                    ("slt", val(&x.slt(&y)), a < b),
                    ("sle", val(&x.sle(&y)), a <= b),
                    ("sgt", val(&x.sgt(&y)), a > b),
                    ("sge", val(&x.sge(&y)), a >= b),
                ];
                for (name, got, want) in checks {
                    assert_eq!(got, u128::from(want), "{name} w={w} a={a:#x} b={b:#x}");
                }
            }
        }
    }
}

// angr-03vl4.58: signed concrete fast paths at width > 128.
#[test]
fn wide_sdiv_srem_degenerate_to_unsigned() {
    for &w in &WIDE_WIDTHS {
        for &a in &wide_payloads() {
            for &b in &wide_payloads() {
                let (x, y) = (bv(a, w), bv(b, w));
                // Both operands are non-negative (sign bit past the storage),
                // so bvsdiv/bvsrem equal bvudiv/bvurem. Division by zero stays
                // total (x / 0 == -1 == all-ones for x >= 0), but all-ones is
                // not representable up here, so that one case declines the
                // fold instead — see `wide_div_by_zero_declines_the_concrete_fold`
                // (angr-nknv1). `x % 0 == x` is representable and still folds.
                let want_rem = if b == 0 { a } else { a % b };
                let got_div = x.sdiv(&y);
                match a.checked_div(b) {
                    Some(q) => assert_eq!(val(&got_div), q, "sdiv w={w} a={a:#x} b={b:#x}"),
                    None => assert!(
                        got_div.as_u128().is_none(),
                        "sdiv by zero must not fold w={w} a={a:#x}"
                    ),
                }
                assert_eq!(val(&x.srem(&y)), want_rem, "srem w={w} a={a:#x} b={b:#x}");
            }
        }
    }
}

// angr-03vl4.59: rotl/rotr concrete fast paths at width > 128.
#[test]
fn wide_rotate_declines_the_concrete_fold() {
    for &w in &WIDE_WIDTHS {
        for &v in &wide_payloads() {
            for &a in &wide_shift_amounts(w) {
                for (name, got, op) in [
                    ("rotl", bv(v, w).rotl(&bv(a, w)), BVOp::RotL),
                    ("rotr", bv(v, w).rotr(&bv(a, w)), BVOp::RotR),
                ] {
                    assert_eq!(got.width(), w, "{name} width w={w} v={v:#x} a={a}");
                    if a % (w as u128) == 0 {
                        // Rotation by a multiple of the width is the identity
                        // and stays foldable even above the storage ceiling.
                        assert_eq!(val(&got), v, "{name} w={w} v={v:#x} a={a}");
                    } else {
                        // Every set bit of a `width > 128` `Concrete` lives in
                        // the low 128 bits; rotating any of them lands outside
                        // the u128 payload, so there is no concrete answer to
                        // fold to. The node must stay symbolic rather than
                        // shift a u128 by `amt >= 128` (abort/wrap).
                        assert!(
                            got.as_u128().is_none(),
                            "{name} must not fold w={w} v={v:#x} a={a}"
                        );
                        assert_eq!(
                            op_of(&got),
                            Some(op.clone()),
                            "{name} op w={w} v={v:#x} a={a}"
                        );
                    }
                }
            }
        }
    }
}

// angr-03vl4.60: the byte-reverse concrete fast path at width > 128, same
// unstorable-result shape as the wide rotate above.
#[test]
fn wide_reverse_declines_the_concrete_fold() {
    for &w in &WIDE_WIDTHS {
        for &v in &wide_payloads() {
            let got = bv(v, w).reverse();
            assert_eq!(got.width(), w, "reverse width w={w} v={v:#x}");
            if !w.is_multiple_of(8) {
                // A partial byte has no reverse at any width; the misaligned
                // arm returns the operand unchanged (angr-sqfj8.92).
                assert_eq!(val(&got), v, "misaligned reverse w={w} v={v:#x}");
            } else {
                // A reverse moves byte 0 to bit position `w - 8 >= 128`, which
                // the u128 payload cannot represent, so there is no concrete
                // answer to fold to.
                assert!(
                    got.as_u128().is_none(),
                    "reverse must not fold w={w} v={v:#x}"
                );
                assert_eq!(
                    op_of(&got),
                    Some(BVOp::Reverse),
                    "reverse op w={w} v={v:#x}"
                );
            }
        }
    }
}

/// The `width <= 128` fold is unchanged by the wide guard: check it against an
/// independent masked-`u128` reference across the boundary widths and the
/// amounts that straddle 0 / w / 2^32.
#[test]
fn narrow_rotate_matches_masked_u128_reference() {
    for &w in &BOUNDARY_WIDTHS {
        for &v in &boundary_values(w) {
            for &raw_a in &[0u128, 1, 7, (w - 1) as u128, w as u128, 1u128 << 32] {
                // The amount operand is itself a `width`-bit BV, so `concrete()`
                // masks it; the reference must see the same masked value.
                let a = raw_a & mask(w);
                let amt = (a % w as u128) as u32;
                let want_l = if amt == 0 {
                    v
                } else {
                    ((v << amt) | (v >> (w - amt))) & mask(w)
                };
                let want_r = if amt == 0 {
                    v
                } else {
                    ((v >> amt) | (v << (w - amt))) & mask(w)
                };
                let (x, n) = (bv(v, w), bv(a, w));
                assert_eq!(val(&x.rotl(&n)), want_l, "rotl w={w} v={v:#x} a={a}");
                assert_eq!(val(&x.rotr(&n)), want_r, "rotr w={w} v={v:#x} a={a}");
            }
        }
    }
}

// angr-0jh0j.51: the add/sub/mul concrete fast paths at width > 128. Below the
// storage ceiling a carry out of bit 127 IS the mod-2^width wraparound the width
// mandates; above it the carried bit is an ordinary bit of the result (2^128 is
// a perfectly good 200-bit number) that the u128 payload cannot hold, so the
// fold has to decline instead of emitting a smaller, wrong number.
#[test]
fn wide_arith_declines_the_fold_when_the_result_leaves_the_payload() {
    for &w in &WIDE_WIDTHS {
        for &a in &wide_payloads() {
            for &b in &wide_payloads() {
                for (name, got, op, exact) in [
                    ("add", bv(a, w).add(&bv(b, w)), BVOp::Add, a.checked_add(b)),
                    ("sub", bv(a, w).sub(&bv(b, w)), BVOp::Sub, a.checked_sub(b)),
                    ("mul", bv(a, w).mul(&bv(b, w)), BVOp::Mul, a.checked_mul(b)),
                ] {
                    assert_eq!(got.width(), w, "{name} width w={w} a={a:#x} b={b:#x}");
                    match exact {
                        // Fits the payload: the fold still fires and is exact.
                        Some(v) => assert_eq!(val(&got), v, "{name} w={w} a={a:#x} b={b:#x}"),
                        // Crosses bit 127: must stay symbolic so Z3 computes it
                        // at the declared width.
                        None => {
                            assert!(
                                got.as_u128().is_none(),
                                "{name} must not fold w={w} a={a:#x} b={b:#x}"
                            );
                            assert_eq!(
                                op_of(&got),
                                Some(op.clone()),
                                "{name} op w={w} a={a:#x} b={b:#x}"
                            );
                        }
                    }
                }
            }
        }
    }
}

// angr-0jh0j.52: not/neg at width > 128. A true NOT must set every bit of
// `[128..w)` (they start at the implicit-zero storage floor), and a true
// negation `2^w - v` sets them too for any non-zero v — neither is
// representable, so both decline.
#[test]
fn wide_not_and_neg_decline_the_concrete_fold() {
    for &w in &WIDE_WIDTHS {
        for &v in &wide_payloads() {
            let not = bv(v, w).not();
            assert_eq!(not.width(), w, "not width w={w} v={v:#x}");
            assert!(not.as_u128().is_none(), "not must not fold w={w} v={v:#x}");
            assert_eq!(op_of(&not), Some(BVOp::Not), "not op w={w} v={v:#x}");

            let neg = bv(v, w).neg();
            assert_eq!(neg.width(), w, "neg width w={w} v={v:#x}");
            if v == 0 {
                // -0 is 0 at every width and stays foldable.
                assert_eq!(val(&neg), 0, "neg of zero w={w}");
            } else {
                assert!(neg.as_u128().is_none(), "neg must not fold w={w} v={v:#x}");
                assert_eq!(op_of(&neg), Some(BVOp::Neg), "neg op w={w} v={v:#x}");
            }
        }
    }
}

// angr-0jh0j.53: sign-extending a NEGATIVE narrow source to a target width
// above the storage ceiling. The sign fill has to reach bit `to - 1 >= 128`,
// which `sign_extend_to`'s u128 mask arithmetic cannot express — folding anyway
// turned e.g. a 32-bit -1 widened to 200 bits into 2^128 - 1, a non-negative
// number ~2^72 times too small, that every `sign_bit_beyond_storage`-gated
// signed op would then read as non-negative.
#[test]
fn wide_sign_extend_of_negative_source_declines_the_concrete_fold() {
    for &from in &[1u32, 8, 32, 64, 127, 128] {
        for &to in &WIDE_WIDTHS {
            let neg = 1u128 << (from - 1); // sign bit set => negative source
            let got = bv(neg, from).sign_extend(to);
            assert_eq!(got.width(), to, "sext({from}->{to}) width");
            if from >= 128 {
                // The source's own sign bit is already beyond storage, so the
                // value is logically non-negative and the extension is a no-op.
                assert_eq!(val(&got), neg, "sext({from}->{to}) v={neg:#x}");
            } else {
                assert!(
                    got.as_u128().is_none(),
                    "sext({from}->{to}) of negative v={neg:#x} must not fold"
                );
                assert_eq!(
                    op_of(&got),
                    Some(BVOp::SignExt(to - from)),
                    "sext({from}->{to}) op"
                );
            }
            // A non-negative source is a plain zero-extend and still folds.
            let pos = neg - 1;
            let got_pos = bv(pos, from).sign_extend(to);
            assert_eq!(val(&got_pos), pos, "sext({from}->{to}) v={pos:#x}");
        }
    }
}

// angr-nknv1: `all_ones_mask` saturates to `u128::MAX` above 128 bits, so the
// `x & all_ones -> x` / `x | all_ones -> all_ones` arms used to fire against a
// concrete that is really the ordinary number `2^128 - 1`. Against a SYMBOLIC
// operand both shortcuts are then wrong: the AND must clear bits `[128..w)`
// (returning `x` keeps them) and the OR must keep them (returning
// `Self::ones(w)` = `2^128 - 1` drops them). Both must stay symbolic above the
// ceiling, and both must still fire at or below it.
#[test]
fn wide_and_or_against_saturated_all_ones_decline_the_simplification() {
    for &w in &WIDE_WIDTHS {
        let x = RustBV::symbol("x", w);
        let ones = bv(u128::MAX, w);
        for (name, got, op) in [
            ("x & ones", x.and(&ones), BVOp::And),
            ("ones & x", ones.and(&x), BVOp::And),
            ("x | ones", x.or(&ones), BVOp::Or),
            ("ones | x", ones.or(&x), BVOp::Or),
        ] {
            assert_eq!(got.width(), w, "{name} width w={w}");
            assert!(got.as_u128().is_none(), "{name} must not fold w={w}");
            assert_eq!(op_of(&got), Some(op.clone()), "{name} op w={w}");
        }
    }
    // At or below the ceiling `u128::MAX` masked to `w` really is all-ones, so
    // both simplifications must still fire.
    for &w in &[1u32, 8, 64, 127, 128] {
        let x = RustBV::symbol("x", w);
        let ones = bv(u128::MAX, w);
        for (name, got) in [("x & ones", x.and(&ones)), ("ones & x", ones.and(&x))] {
            assert_eq!(got.width(), w, "{name} width w={w}");
            assert!(op_of(&got).is_none() && got.is_symbolic(), "{name} w={w}");
        }
        assert_eq!(val(&x.or(&ones)), mask(w), "x | ones w={w}");
        assert_eq!(val(&ones.or(&x)), mask(w), "ones | x w={w}");
    }
}

// angr-nknv1: `x / 0` is all-ones, which `RustBV::ones` cannot represent above
// 128 bits (it saturates to `2^128 - 1`), so both division folds must decline
// there and let Z3's equally-total bvudiv/bvsdiv answer at the declared width.
#[test]
fn wide_div_by_zero_declines_the_concrete_fold() {
    for &w in &WIDE_WIDTHS {
        for &a in &wide_payloads() {
            let zero = bv(0, w);
            for (name, got, op) in [
                ("udiv", bv(a, w).udiv(&zero), BVOp::UDiv),
                ("sdiv", bv(a, w).sdiv(&zero), BVOp::SDiv),
            ] {
                assert_eq!(got.width(), w, "{name} width w={w} a={a:#x}");
                assert!(
                    got.as_u128().is_none(),
                    "{name} by zero must not fold w={w} a={a:#x}"
                );
                assert_eq!(op_of(&got), Some(op.clone()), "{name} op w={w} a={a:#x}");
            }
            // A non-zero divisor still folds: both operands are logically
            // non-negative up here, so signed division degenerates to unsigned.
            let d = bv(3, w);
            assert_eq!(val(&bv(a, w).udiv(&d)), a / 3, "udiv w={w} a={a:#x}");
            assert_eq!(val(&bv(a, w).sdiv(&d)), a / 3, "sdiv w={w} a={a:#x}");
        }
    }
    // Below the ceiling all-ones is representable and `x / 0` still folds.
    for &w in &[1u32, 8, 64, 128] {
        assert_eq!(val(&bv(1, w).udiv(&bv(0, w))), mask(w), "udiv w={w}");
        assert_eq!(val(&bv(1, w).sdiv(&bv(0, w))), mask(w), "sdiv w={w}");
    }
}

/// Harness 6: the shared `test_boundary_values::boundary_widths()` table
/// agrees with this file's own, hand-picked `BOUNDARY_WIDTHS`/`WIDE_WIDTHS`
/// sweep above for rotl/rotr/reverse. This is the DRY cross-check the shared
/// helper exists for (the Harness-6 sweep described in
/// `test_boundary_values.rs`), not a replacement for the
/// file-local tables' much finer-grained coverage — and it does add one new
/// data point neither hand-picked list has: `255` (odd, `>128`, not a
/// multiple of 8), which lands in a third code path (declines the rotate
/// fold like `WIDE_WIDTHS`, but — unlike `WIDE_WIDTHS`'s all-multiple-of-8
/// entries — also declines the byte-reverse fold on *alignment* grounds).
///
/// Width `0` is exercised for rotl/rotr too, and is the regression case for
/// angr-309bq: `define_rotate_pair!`'s concrete arm used to compute
/// `amt = a % (w as u128)` unconditionally, and `% 0` is a Rust panic in every
/// profile (not one gated by overflow-checks). No VEX `IROp` has a 0-bit
/// operand, but a width-0 value is a supported degenerate `RustBV`, so the
/// rotate folds to the 0-bit identity instead.
#[test]
fn rotate_and_reverse_boundary_widths() {
    // The widths of the engine's `test_boundary_values::boundary_widths`
    // table, which the value layer cannot import (angr-mx00p.116.7).
    for w in [0u32, 1, 127, 128, 129, 255, 256] {
        for &raw_a in &[0u128, 1, w as u128, 1u128 << 32] {
            let (x, n) = (bv(u128::MAX, w), bv(raw_a, w));
            let rl = x.rotl(&n);
            let rr = x.rotr(&n);
            assert_eq!(rl.width(), w, "rotl width w={w} a={raw_a}");
            assert_eq!(rr.width(), w, "rotr width w={w} a={raw_a}");

            let amt = raw_a & mask(w.min(128));
            if w == 0 {
                // The only 0-bit value is the empty one; rotating it by any
                // amount is the identity, and must not divide by the width.
                assert_eq!(val(&rl), 0, "rotl w=0 a={raw_a}");
                assert_eq!(val(&rr), 0, "rotr w=0 a={raw_a}");
            } else if w > 128 && !amt.is_multiple_of(w as u128) {
                assert!(rl.as_u128().is_none(), "rotl must not fold w={w} amt={amt}");
                assert!(rr.as_u128().is_none(), "rotr must not fold w={w} amt={amt}");
            } else if w <= 128 {
                let v = mask(w);
                let amtn = (amt % w as u128) as u32;
                let want_l = if amtn == 0 {
                    v
                } else {
                    ((v << amtn) | (v >> (w - amtn))) & mask(w)
                };
                let want_r = if amtn == 0 {
                    v
                } else {
                    ((v >> amtn) | (v << (w - amtn))) & mask(w)
                };
                assert_eq!(val(&rl), want_l, "rotl w={w} a={raw_a}");
                assert_eq!(val(&rr), want_r, "rotr w={w} a={raw_a}");
            }
        }

        let rv = bv(u128::MAX, w).reverse();
        assert_eq!(rv.width(), w, "reverse width w={w}");
        if w.is_multiple_of(8) && w <= 128 {
            // Byte-reversing an all-ones value is itself all-ones.
            assert_eq!(val(&rv), mask(w), "reverse w={w}");
        } else if w.is_multiple_of(8) {
            assert!(rv.as_u128().is_none(), "wide reverse must not fold w={w}");
        }
    }
}
