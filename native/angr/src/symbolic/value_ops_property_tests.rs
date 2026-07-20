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
