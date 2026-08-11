//! Property-based tests for `value_ops.rs` (RustBV concrete-folding ops) and
//! `bv_codec.rs` (concrete<->Z3 BV round-trips), added under angr-ph300.1.
//!
//! `quickcheck`/`quickcheck_macros` were already declared in
//! `native/angr/Cargo.toml` `[dev-dependencies]` but had zero uses; these
//! are the first. Every RustBV op here is exercised on *concrete* operands so
//! it constant-folds to a `Concrete` node whose value `as_u128()` reads back —
//! the property then checks the folded result against a plain-`u128` reference
//! computation. This covers the wide op surface (arithmetic / bitwise / shifts
//! / structural) with random inputs the hand-written `value_tests.rs` examples
//! cannot.
//!
//! Declared as a child of `value_ops` so `use super::*` reaches the module's
//! private free helpers (`sign_extend`, `sign_extend_to`); `bv_codec`'s
//! `pub(super)` codec fns are reachable via `crate::symbolic::bv_codec`.

use super::*;
use crate::symbolic::bv_codec::{extract_bv_value, extract_bv_value_wide, make_bv_const};
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
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    val(&bv(a, w).add(&bv(b, w), &ctx)) == val(&bv(b, w).add(&bv(a, w), &ctx))
}

#[quickcheck]
fn prop_add_zero_identity(a: u128, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    val(&bv(a, w).add(&bv(0, w), &ctx)) == (a & mask(w))
}

#[quickcheck]
fn prop_add_matches_wrapping(a: u128, b: u128, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    let expected = (a & mask(w)).wrapping_add(b & mask(w)) & mask(w);
    val(&bv(a, w).add(&bv(b, w), &ctx)) == expected
}

#[quickcheck]
fn prop_sub_matches_wrapping(a: u128, b: u128, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    let expected = (a & mask(w)).wrapping_sub(b & mask(w)) & mask(w);
    val(&bv(a, w).sub(&bv(b, w), &ctx)) == expected
}

#[quickcheck]
fn prop_sub_self_is_zero(a: u128, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    val(&bv(a, w).sub(&bv(a, w), &ctx)) == 0
}

#[quickcheck]
fn prop_mul_commutative(a: u128, b: u128, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    val(&bv(a, w).mul(&bv(b, w), &ctx)) == val(&bv(b, w).mul(&bv(a, w), &ctx))
}

#[quickcheck]
fn prop_mul_zero_and_one(a: u128, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    val(&bv(a, w).mul(&bv(0, w), &ctx)) == 0 && val(&bv(a, w).mul(&bv(1, w), &ctx)) == (a & mask(w))
}

#[quickcheck]
fn prop_mul_matches_wrapping(a: u128, b: u128, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    let expected = (a & mask(w)).wrapping_mul(b & mask(w)) & mask(w);
    val(&bv(a, w).mul(&bv(b, w), &ctx)) == expected
}

#[quickcheck]
fn prop_neg_involution(a: u128, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    val(&bv(a, w).neg(&ctx).neg(&ctx)) == (a & mask(w))
}

#[quickcheck]
fn prop_neg_add_is_zero(a: u128, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    let x = bv(a, w);
    val(&x.add(&x.neg(&ctx), &ctx)) == 0
}

// ---------------------------------------------------------------------------
// Division / remainder (udiv / sdiv / urem / srem)
//
// angr-n0irt.16: the div/rem family had two real production bugs land the same
// week this suite was introduced — 169b171a2 (wrapping div/rem VEX fix) and
// 12f946f5c (concrete div-by-zero fold fix) — yet the property suite covered
// none of them. SMT-LIB div/rem are *total* (x/0 and x%0 are defined, bvsdiv
// rounds toward zero, and MIN/-1 wraps rather than overflowing), so a
// hand-rolled `u128` reference would risk re-encoding the very bug it means to
// catch. The ground truth here is therefore Z3 itself: fold the op on concrete
// operands, then build the same op on a symbolic dividend pinned to that value
// and compare against `ctx.eval`. Gated on `vex-engine-z3` (needs the solver).
// ---------------------------------------------------------------------------

/// Fold `op` on concrete `(a, b)` at width `w`, then assert the folded value
/// equals Z3's evaluation of the same op with a symbolic dividend pinned to
/// `a` (divisor stays concrete). Returns `true` on agreement. `name` is only
/// used to label a Z3-eval failure. Mirrors the pinning pattern in
/// `value_tests::test_sdiv_by_zero_concrete_matches_z3`.
#[cfg(feature = "vex-engine-z3")]
fn div_rem_matches_z3(
    a: u128,
    b: u128,
    w: u32,
    op: fn(&RustBV, &RustBV, &SymContext) -> RustBV,
    name: &str,
) -> bool {
    let ctx = SymContext::new_mock();
    let folded = op(&bv(a, w), &bv(b, w), &ctx)
        .as_u128()
        .expect("div/rem on concrete operands must fold to Concrete");

    let d = RustBV::symbolic(&ctx, "dvd", w);
    let pinned = d.eq(&bv(a, w), &ctx);
    ctx.add_constraint(pinned.to_z3_ast().eq(z3::ast::BV::from_u64(1, 1)));
    let sym = op(&d, &bv(b, w), &ctx);
    match ctx.eval(&sym) {
        Some(z) => z == folded,
        None => panic!("{name}: Z3 could not eval pinned {name}({a:#x}, {b:#x}) @w{w}"),
    }
}

#[cfg(feature = "vex-engine-z3")]
#[quickcheck]
fn prop_udiv_matches_z3(a: u128, b: u128, ws: u8) -> bool {
    div_rem_matches_z3(a, b, w128(ws), RustBV::udiv, "udiv")
}

#[cfg(feature = "vex-engine-z3")]
#[quickcheck]
fn prop_sdiv_matches_z3(a: u128, b: u128, ws: u8) -> bool {
    div_rem_matches_z3(a, b, w128(ws), RustBV::sdiv, "sdiv")
}

#[cfg(feature = "vex-engine-z3")]
#[quickcheck]
fn prop_urem_matches_z3(a: u128, b: u128, ws: u8) -> bool {
    div_rem_matches_z3(a, b, w128(ws), RustBV::urem, "urem")
}

#[cfg(feature = "vex-engine-z3")]
#[quickcheck]
fn prop_srem_matches_z3(a: u128, b: u128, ws: u8) -> bool {
    div_rem_matches_z3(a, b, w128(ws), RustBV::srem, "srem")
}

// Random operands almost never hit b == 0, so force the zero-divisor arm for
// all four ops — this is the 12f946f5c fold-vs-Z3 divergence made into a gate.
#[cfg(feature = "vex-engine-z3")]
#[quickcheck]
fn prop_div_rem_by_zero_matches_z3(a: u128, ws: u8) -> bool {
    let w = w128(ws);
    div_rem_matches_z3(a, 0, w, RustBV::udiv, "udiv/0")
        && div_rem_matches_z3(a, 0, w, RustBV::sdiv, "sdiv/0")
        && div_rem_matches_z3(a, 0, w, RustBV::urem, "urem/0")
        && div_rem_matches_z3(a, 0, w, RustBV::srem, "srem/0")
}

// The signed MIN / -1 overflow corner at every width: two's-complement MIN
// divided by all-ones. Native `iN` division would panic here; SMT-LIB bvsdiv
// wraps back to MIN. Pin both operands to the corner and check fold == Z3.
#[cfg(feature = "vex-engine-z3")]
#[quickcheck]
fn prop_sdiv_srem_min_over_neg1_matches_z3(ws: u8) -> bool {
    let w = w128(ws);
    let min = 1u128 << (w - 1); // two's-complement MIN at width w
    let neg1 = mask(w); // -1 at width w
    div_rem_matches_z3(min, neg1, w, RustBV::sdiv, "sdiv MIN/-1")
        && div_rem_matches_z3(min, neg1, w, RustBV::srem, "srem MIN/-1")
}

// ---------------------------------------------------------------------------
// Bitwise
// ---------------------------------------------------------------------------

#[quickcheck]
fn prop_and_absorption(a: u128, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    let x = bv(a, w);
    // x & x == x, x & 0 == 0, x & ones == x
    val(&x.and(&x, &ctx)) == (a & mask(w))
        && val(&x.and(&bv(0, w), &ctx)) == 0
        && val(&x.and(&bv(mask(w), w), &ctx)) == (a & mask(w))
}

#[quickcheck]
fn prop_or_absorption(a: u128, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    let x = bv(a, w);
    // x | x == x, x | 0 == x, x | ones == ones
    val(&x.or(&x, &ctx)) == (a & mask(w))
        && val(&x.or(&bv(0, w), &ctx)) == (a & mask(w))
        && val(&x.or(&bv(mask(w), w), &ctx)) == mask(w)
}

#[quickcheck]
fn prop_xor_laws(a: u128, b: u128, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    let x = bv(a, w);
    let y = bv(b, w);
    // x ^ x == 0, x ^ 0 == x, (x ^ y) ^ y == x, commutative
    val(&x.xor(&x, &ctx)) == 0
        && val(&x.xor(&bv(0, w), &ctx)) == (a & mask(w))
        && val(&x.xor(&y, &ctx).xor(&y, &ctx)) == (a & mask(w))
        && val(&x.xor(&y, &ctx)) == val(&y.xor(&x, &ctx))
}

#[quickcheck]
fn prop_not_involution_and_mask(a: u128, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    let x = bv(a, w);
    // !!x == x, !x == x ^ ones
    val(&x.not(&ctx).not(&ctx)) == (a & mask(w)) && val(&x.not(&ctx)) == ((a & mask(w)) ^ mask(w))
}

// ---------------------------------------------------------------------------
// Shifts
// ---------------------------------------------------------------------------

#[quickcheck]
fn prop_shl_matches_reference(v: u128, amt: u8, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
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
    val(&bv(v, w).shl(&bv(amt as u128, w), &ctx)) == expected
}

#[quickcheck]
fn prop_lshr_matches_reference(v: u128, amt: u8, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    let amt_eff = (amt as u128) & mask(w);
    let vm = v & mask(w);
    let expected = if amt_eff >= w as u128 {
        0
    } else {
        vm.wrapping_shr(amt_eff as u32) & mask(w)
    };
    val(&bv(v, w).lshr(&bv(amt as u128, w), &ctx)) == expected
}

#[quickcheck]
fn prop_ashr_matches_reference(v: u128, amt: u8, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    let amt_eff = (amt as u128) & mask(w);
    // ashr saturates the amount to w-1 (all sign bits).
    let amt_clamped = amt_eff.min((w - 1) as u128) as u32;
    let signed = sign_extend(v & mask(w), w);
    let expected = (signed >> amt_clamped) as u128 & mask(w);
    val(&bv(v, w).ashr(&bv(amt as u128, w), &ctx)) == expected
}

// ---------------------------------------------------------------------------
// Structural
// ---------------------------------------------------------------------------

#[quickcheck]
fn prop_concat_extract_inverse(a: u128, b: u128, was: u8, wbs: u8) -> bool {
    let ctx = SymContext::new_mock();
    let (wa, wb) = (w64(was), w64(wbs));
    let hi = bv(a, wa);
    let lo = bv(b, wb);
    let c = hi.concat(&lo, &ctx); // width wa + wb
    let top = c.extract(wa + wb - 1, wb, &ctx); // recovers hi
    let bot = c.extract(wb - 1, 0, &ctx); // recovers lo
    val(&top) == (a & mask(wa)) && val(&bot) == (b & mask(wb))
}

#[quickcheck]
fn prop_zero_extend_truncate_roundtrip(v: u128, ws: u8, extra: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w64(ws);
    let w2 = w + (extra % 64) as u32; // >= w, <= 127
    let x = bv(v, w);
    let ze = x.zero_extend(w2, &ctx);
    // zero-extend preserves the low value, and truncating back recovers it.
    val(&ze) == (v & mask(w)) && val(&ze.truncate(w, &ctx)) == (v & mask(w))
}

#[quickcheck]
fn prop_sign_extend_matches_reference(v: u128, ws: u8, extra: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w64(ws);
    let w2 = w + (extra % 64) as u32; // >= w, <= 127
    let expected = sign_extend_to(v & mask(w), w, w2) & mask(w2);
    val(&bv(v, w).sign_extend(w2, &ctx)) == expected
}

#[quickcheck]
fn prop_eq_matches_reference(a: u128, b: u128, ws: u8) -> bool {
    let ctx = SymContext::new_mock();
    let w = w128(ws);
    let want = if (a & mask(w)) == (b & mask(w)) { 1 } else { 0 };
    // eq is reflexive and its concrete result is a 1-bit 0/1.
    val(&bv(a, w).eq(&bv(b, w), &ctx)) == want && val(&bv(a, w).eq(&bv(a, w), &ctx)) == 1
}

// ---------------------------------------------------------------------------
// bv_codec: concrete <-> Z3 BV round-trips
// ---------------------------------------------------------------------------

#[quickcheck]
fn prop_make_bv_const_extract_roundtrip(v: u128, ws: u8) -> bool {
    use z3::ast::Ast;
    let w = w128(ws);
    // For width > 64 `make_bv_const` builds a `concat` of two numerals whose
    // `Display` is the expression form, not a `#x`/`#b` numeral — the decode
    // path only ever sees *simplified* model numerals, so simplify first.
    let z = make_bv_const(v, w).simplify();
    // extract_bv_value returns the low 128 bits; for width <= 128 that is the
    // masked value exactly.
    extract_bv_value(&z) == Some(v & mask(w))
}

#[quickcheck]
fn prop_extract_bv_value_wide_roundtrip(v: u128, ws: u8) -> bool {
    use z3::ast::Ast;
    let w = w128(ws);
    let z = make_bv_const(v, w).simplify();
    let vm = v & mask(w);
    // Decode the returned big-endian bytes back to a u128 (width <= 128) and
    // compare to the masked value — robust to the codec's exact byte length /
    // top-byte masking rather than re-deriving it here.
    match extract_bv_value_wide(&z, w) {
        Some(bytes) => {
            let got = bytes.iter().fold(0u128, |acc, &b| (acc << 8) | b as u128);
            got == vm
        }
        None => false,
    }
}

// ---------------------------------------------------------------------------
// try_zext_const_cmp_fold equivalence (angr-ph300.37)
//
// The fold rewrites `Cmp(ZeroExt(k, x), BVV(c))` (and the commuted form) into a
// narrowed comparison or a 1-bit constant, hitting 14 branches across 6 fold
// tags. A wrong trivial-decide still returns a well-formed 1-bit constant, so a
// direction/arm regression is silent at the type level. This property proves,
// per random case, that the folded RustBV result is *semantically identical* to
// the unfolded Z3 encoding: the solver finds no assignment to `x` that makes the
// two disagree. Covers all 6 user-facing unsigned cmps in both operand orders
// (so Ugt/Uge's inverted UltSwapped/UleSwapped mapping is exercised), which
// drives every branch of `try_zext_const_cmp_fold` including the swapped tags.
// ---------------------------------------------------------------------------

/// The six user-facing unsigned comparisons routed through the fold.
#[derive(Copy, Clone)]
enum UCmp {
    Eq,
    Ne,
    Ult,
    Ule,
    Ugt,
    Uge,
}

const UCMPS: [UCmp; 6] = [
    UCmp::Eq,
    UCmp::Ne,
    UCmp::Ult,
    UCmp::Ule,
    UCmp::Ugt,
    UCmp::Uge,
];

/// Apply the RustBV comparison (the folding path) with `zext` on the left when
/// `zext_left`, else on the right.
fn rust_cmp(op: UCmp, zext: &RustBV, c: &RustBV, zext_left: bool, ctx: &SymContext) -> RustBV {
    let (a, b) = if zext_left { (zext, c) } else { (c, zext) };
    match op {
        UCmp::Eq => a.eq(b, ctx),
        UCmp::Ne => a.ne(b, ctx),
        UCmp::Ult => a.ult(b, ctx),
        UCmp::Ule => a.ule(b, ctx),
        UCmp::Ugt => a.ugt(b, ctx),
        UCmp::Uge => a.uge(b, ctx),
    }
}

/// The unfolded Z3 reference for the same comparison.
fn z3_cmp(op: UCmp, a: &z3::ast::BV, b: &z3::ast::BV) -> z3::ast::Bool {
    match op {
        UCmp::Eq => a.eq(b),
        UCmp::Ne => a.eq(b).not(),
        UCmp::Ult => a.bvult(b),
        UCmp::Ule => a.bvule(b),
        UCmp::Ugt => a.bvugt(b),
        UCmp::Uge => a.bvuge(b),
    }
}

#[quickcheck]
fn prop_zext_const_cmp_fold_matches_z3(
    iw_seed: u8,
    k_seed: u8,
    c: u128,
    op_seed: u8,
    zext_left: bool,
) -> bool {
    let inner_width = (iw_seed % 32) as u32 + 1; // 1..=32
    let extend_bits = (k_seed % 32) as u32 + 1; // 1..=32
    let total = inner_width + extend_bits; // 2..=64 — as_u128-readable
    let op = UCMPS[(op_seed % 6) as usize];

    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", inner_width);
    let zx = x.zero_extend(total, &ctx);
    let cbv = RustBV::concrete(c & mask(total), total);

    // Folded RustBV result (goes through try_zext_const_cmp_fold).
    let folded_bv = rust_cmp(op, &zx, &cbv, zext_left, &ctx).to_z3_ast();

    // Unfolded Z3 reference over the same symbolic `x`.
    let zx_z3 = zx.to_z3_ast();
    let c_z3 = cbv.to_z3_ast();
    let (za, zb) = if zext_left {
        (&zx_z3, &c_z3)
    } else {
        (&c_z3, &zx_z3)
    };
    let ref_bv = z3_cmp(op, za, zb).ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1));

    // Equivalent iff no assignment to `x` makes the two 1-bit results differ.
    ctx.add_constraint(folded_bv.eq(&ref_bv).not());
    !ctx.is_sat()
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
    let ctx = SymContext::new_mock();
    for &w in &BOUNDARY_WIDTHS {
        for &to in &BOUNDARY_WIDTHS {
            if to > w {
                continue; // truncate only narrows
            }
            for &v in &boundary_values(w) {
                let got = val(&bv(v, w).truncate(to, &ctx));
                assert_eq!(got, v & mask(to), "truncate({w}->{to}) v={v:#x}");
            }
        }
    }
}

#[test]
fn boundary_zero_extend_preserves_low_and_zeros_high() {
    let ctx = SymContext::new_mock();
    for &w in &BOUNDARY_WIDTHS {
        for &to in &BOUNDARY_WIDTHS {
            if to < w {
                continue; // zero_extend only widens
            }
            for &v in &boundary_values(w) {
                let ze = bv(v, w).zero_extend(to, &ctx);
                // Low `w` bits unchanged, high bits zero => value == masked src.
                assert_eq!(val(&ze), v & mask(w), "zext({w}->{to}) v={v:#x}");
                // Round-trip back down recovers the source exactly.
                assert_eq!(
                    val(&ze.truncate(w, &ctx)),
                    v & mask(w),
                    "zext-trunc({w}->{to}->{w}) v={v:#x}"
                );
            }
        }
    }
}

#[test]
fn boundary_sign_extend_matches_reference() {
    let ctx = SymContext::new_mock();
    for &w in &BOUNDARY_WIDTHS {
        for &to in &BOUNDARY_WIDTHS {
            if to < w {
                continue; // sign_extend only widens
            }
            for &v in &boundary_values(w) {
                let want = sign_extend_to(v & mask(w), w, to) & mask(to);
                let got = val(&bv(v, w).sign_extend(to, &ctx));
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
    let ctx = SymContext::new_mock();
    for &w in &BOUNDARY_WIDTHS {
        if w >= 128 {
            continue; // no widening past the u128 ceiling
        }
        for &v in &boundary_values(w) {
            let src = v & mask(w);
            // Reference: sign_extend() returns the i128 two's-complement value;
            // its bit pattern IS the 128-bit sign extension.
            let want = sign_extend(src, w) as u128;
            let got = val(&bv(src, w).sign_extend(128, &ctx));
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
    let ctx = SymContext::new_mock();
    for &w in &BOUNDARY_WIDTHS {
        for &v in &boundary_values(w) {
            let x = bv(v, w);
            // Every individual bit position.
            for i in 0..w {
                let got = val(&x.extract(i, i, &ctx));
                assert_eq!(got, (v >> i) & 1, "extract bit {i} of w={w} v={v:#x}");
            }
            // Low slice [to-1:0] for each boundary width `to <= w`, plus the
            // top-anchored slice [w-1 : w-to].
            for &to in &BOUNDARY_WIDTHS {
                if to > w {
                    continue;
                }
                let low = val(&x.extract(to - 1, 0, &ctx));
                assert_eq!(low, v & mask(to), "extract low {to} of w={w} v={v:#x}");
                let hi = val(&x.extract(w - 1, w - to, &ctx));
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
    let ctx = SymContext::new_mock();
    // Restrict to pairs whose sum stays `as_u128`-readable (<= 128).
    for &wa in &BOUNDARY_WIDTHS {
        for &wb in &BOUNDARY_WIDTHS {
            if wa + wb > 128 {
                continue;
            }
            for &a in &boundary_values(wa) {
                for &b in &boundary_values(wb) {
                    let c = bv(a, wa).concat(&bv(b, wb), &ctx); // width wa+wb
                    let top = c.extract(wa + wb - 1, wb, &ctx);
                    let bot = c.extract(wb - 1, 0, &ctx);
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
// that was fixed twice before (angr-tk7yv assembly-side, angr-dondi
// slicing-side, angr-qwyti.16 sign_extend_to, angr-qwyti.17 sar_fill_mask).
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

#[test]
fn wide_shl_clears_readable_bits_past_128() {
    let ctx = SymContext::new_mock();
    for &w in &WIDE_WIDTHS {
        for &v in &wide_payloads() {
            for &a in &wide_shift_amounts(w) {
                // low 128 bits of `v << a`: any shift >= 128 pushes every stored
                // bit out of the readable window.
                let want = if a >= 128 {
                    0
                } else {
                    v.wrapping_shl(a as u32)
                };
                let got = val(&bv(v, w).shl(&bv(a, w), &ctx));
                assert_eq!(got, want, "shl w={w} v={v:#x} a={a}");
            }
        }
    }
}

#[test]
fn wide_lshr_clears_readable_bits_past_128() {
    let ctx = SymContext::new_mock();
    for &w in &WIDE_WIDTHS {
        for &v in &wide_payloads() {
            for &a in &wide_shift_amounts(w) {
                let want = if a >= 128 {
                    0
                } else {
                    v.wrapping_shr(a as u32)
                };
                let got = val(&bv(v, w).lshr(&bv(a, w), &ctx));
                assert_eq!(got, want, "lshr w={w} v={v:#x} a={a}");
            }
        }
    }
}

#[test]
fn wide_ashr_is_logical_because_sign_bit_is_zero() {
    let ctx = SymContext::new_mock();
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
                let got = val(&bv(v, w).ashr(&bv(a, w), &ctx));
                assert_eq!(got, want, "ashr w={w} v={v:#x} a={a}");
            }
        }
    }
}

#[test]
fn wide_sign_extend_is_noop_because_source_is_nonnegative() {
    let ctx = SymContext::new_mock();
    for &w in &WIDE_WIDTHS {
        for &to in &WIDE_WIDTHS {
            if to <= w {
                continue; // sign_extend only widens
            }
            for &v in &wide_payloads() {
                // Sign bit at position w-1 >= 128 is zero => sign-extend == the
                // value itself. Must NOT overflow `1u128 << (from_width - 1)`.
                let got = val(&bv(v, w).sign_extend(to, &ctx));
                assert_eq!(got, v, "sext({w}->{to}) v={v:#x}");
            }
        }
    }
}

#[test]
fn wide_zero_extend_and_truncate_roundtrip() {
    let ctx = SymContext::new_mock();
    for &w in &WIDE_WIDTHS {
        for &v in &wide_payloads() {
            // zero_extend to a larger wide width preserves the payload.
            for &to in &WIDE_WIDTHS {
                if to <= w {
                    continue;
                }
                let ze = bv(v, w).zero_extend(to, &ctx);
                assert_eq!(val(&ze), v, "zext({w}->{to}) v={v:#x}");
            }
            // truncate down to each boundary width recovers the masked payload.
            for &to in &BOUNDARY_WIDTHS {
                let got = val(&bv(v, w).truncate(to, &ctx));
                assert_eq!(got, v & mask(to), "trunc({w}->{to}) v={v:#x}");
            }
        }
    }
}

#[test]
fn wide_extract_reads_zero_past_bit_128() {
    let ctx = SymContext::new_mock();
    for &w in &WIDE_WIDTHS {
        for &v in &wide_payloads() {
            let x = bv(v, w);
            // A slice entirely at/above bit 128 is all zeros.
            let top = x.extract(w - 1, 128, &ctx);
            assert_eq!(val(&top), 0, "extract top [{}:128] w={w} v={v:#x}", w - 1);
            // The low 128 bits round-trip exactly.
            let low = x.extract(127, 0, &ctx);
            assert_eq!(val(&low), v, "extract low [127:0] w={w} v={v:#x}");
            // A slice straddling bit 128 keeps only the in-range payload bits.
            // Needs `high < w`, so only for widths comfortably past 128.
            if w >= 136 {
                let straddle = x.extract(135, 120, &ctx); // 16-bit slice [135:120]
                let want = (v >> 120) & mask(16);
                assert_eq!(val(&straddle), want, "extract [135:120] w={w} v={v:#x}");
            }
        }
    }
}

// angr-03vl4.58: signed concrete fast paths at width > 128.
#[test]
fn wide_signed_compare_degenerates_to_unsigned() {
    let ctx = SymContext::new_mock();
    for &w in &WIDE_WIDTHS {
        for &a in &wide_payloads() {
            for &b in &wide_payloads() {
                // The true sign bit (position w-1 >= 128) is a logical zero on
                // both sides, so every operand is non-negative and the signed
                // comparisons agree with the unsigned u128 ones -- even when
                // bit 127 of a stored payload is set.
                let (x, y) = (bv(a, w), bv(b, w));
                let checks = [
                    ("slt", val(&x.slt(&y, &ctx)), a < b),
                    ("sle", val(&x.sle(&y, &ctx)), a <= b),
                    ("sgt", val(&x.sgt(&y, &ctx)), a > b),
                    ("sge", val(&x.sge(&y, &ctx)), a >= b),
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
    let ctx = SymContext::new_mock();
    for &w in &WIDE_WIDTHS {
        for &a in &wide_payloads() {
            for &b in &wide_payloads() {
                let (x, y) = (bv(a, w), bv(b, w));
                // Both operands are non-negative (sign bit past the storage),
                // so bvsdiv/bvsrem equal bvudiv/bvurem. Division by zero stays
                // total: x / 0 == -1 == all-ones for x >= 0, and x % 0 == x.
                let want_div = if b == 0 { u128::MAX } else { a / b };
                let want_rem = if b == 0 { a } else { a % b };
                assert_eq!(
                    val(&x.sdiv(&y, &ctx)),
                    want_div,
                    "sdiv w={w} a={a:#x} b={b:#x}"
                );
                assert_eq!(
                    val(&x.srem(&y, &ctx)),
                    want_rem,
                    "srem w={w} a={a:#x} b={b:#x}"
                );
            }
        }
    }
}
