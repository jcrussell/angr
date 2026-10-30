//! The solver-facing half of `bv/value_ops_property_tests.rs`: these tests lower or solve, so
//! they stay in `symbolic` (angr-mx00p.116.7).

use crate::bv::*;
// Property-based tests for `value_ops.rs` (RustBV concrete-folding ops) and
// `bv_codec.rs` (concrete<->Z3 BV round-trips) (angr-ph300.1).
//
// `quickcheck`/`quickcheck_macros` were already declared in
// `native/angr/Cargo.toml` `[dev-dependencies]` but had zero uses; these
// are the first. Every RustBV op here is exercised on *concrete* operands so
// it constant-folds to a `Concrete` node whose value `as_u128()` reads back —
// the property then checks the folded result against a plain-`u128` reference
// computation. This covers the wide op surface (arithmetic / bitwise / shifts
// / structural) with random inputs the hand-written `value_*_tests.rs` examples
// cannot.
//
// Declared as a child of `value_ops` so `use super::*` reaches the module's
// private free helpers (`sign_extend`, `sign_extend_to`); `bv_codec`'s
// `pub(super)` codec fns are reachable via `crate::symbolic::bv_codec`.

use crate::symbolic::ToZ3;
use crate::symbolic::SymContext;
use crate::symbolic::bv_codec::{extract_bv_value, extract_bv_value_wide, make_bv_const};
use quickcheck_macros::quickcheck;
/// Map an arbitrary seed byte to a legal BV width in `1..=128`.
fn w128(seed: u8) -> u32 {
    (seed % 128) as u32 + 1
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
// ---------------------------------------------------------------------------
// Division / remainder (udiv / sdiv / urem / srem)
//
// angr-n0irt.16: the div/rem family has had real production bugs (wrapping
// div/rem in VEX, the concrete div-by-zero fold) (angr-n0xru, angr-g2je6).
// SMT-LIB div/rem are *total* (x/0 and x%0 are defined, bvsdiv
// rounds toward zero, and MIN/-1 wraps rather than overflowing), so a
// hand-rolled `u128` reference would risk re-encoding the very bug it means to
// catch. The ground truth here is therefore Z3 itself: fold the op on concrete
// operands, then build the same op on a symbolic dividend pinned to that value
// and compare against `ctx.eval`.
// ---------------------------------------------------------------------------

/// Fold `op` on concrete `(a, b)` at width `w`, then assert the folded value
/// equals Z3's evaluation of the same op with a symbolic dividend pinned to
/// `a` (divisor stays concrete). Returns `true` on agreement. `name` is only
/// used to label a Z3-eval failure. Mirrors the pinning pattern in
/// `value_concrete_arm_tests::test_sdiv_by_zero_concrete_matches_z3`.
fn div_rem_matches_z3(
    a: u128,
    b: u128,
    w: u32,
    op: fn(&RustBV, &RustBV) -> RustBV,
    name: &str,
) -> bool {
    let ctx = SymContext::new_mock();
    let folded = op(&bv(a, w), &bv(b, w))
        .as_u128()
        .expect("div/rem on concrete operands must fold to Concrete");

    let d = RustBV::symbol("dvd", w);
    let pinned = d.eq(&bv(a, w));
    ctx.add_constraint(pinned.to_z3_ast().unwrap().eq(z3::ast::BV::from_u64(1, 1)));
    let sym = op(&d, &bv(b, w));
    match ctx.eval(&sym) {
        Some(z) => z == folded,
        None => panic!("{name}: Z3 could not eval pinned {name}({a:#x}, {b:#x}) @w{w}"),
    }
}
#[quickcheck]
fn prop_udiv_matches_z3(a: u128, b: u128, ws: u8) -> bool {
    div_rem_matches_z3(a, b, w128(ws), RustBV::udiv, "udiv")
}
#[quickcheck]
fn prop_sdiv_matches_z3(a: u128, b: u128, ws: u8) -> bool {
    div_rem_matches_z3(a, b, w128(ws), RustBV::sdiv, "sdiv")
}
#[quickcheck]
fn prop_urem_matches_z3(a: u128, b: u128, ws: u8) -> bool {
    div_rem_matches_z3(a, b, w128(ws), RustBV::urem, "urem")
}
#[quickcheck]
fn prop_srem_matches_z3(a: u128, b: u128, ws: u8) -> bool {
    div_rem_matches_z3(a, b, w128(ws), RustBV::srem, "srem")
}
// Random operands almost never hit b == 0, so force the zero-divisor arm for
// all four ops — this is the fold-vs-Z3 divergence made into a gate (angr-g2je6).
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
#[quickcheck]
fn prop_sdiv_srem_min_over_neg1_matches_z3(ws: u8) -> bool {
    let w = w128(ws);
    let min = 1u128 << (w - 1); // two's-complement MIN at width w
    let neg1 = mask(w); // -1 at width w
    div_rem_matches_z3(min, neg1, w, RustBV::sdiv, "sdiv MIN/-1")
        && div_rem_matches_z3(min, neg1, w, RustBV::srem, "srem MIN/-1")
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
fn rust_cmp(op: UCmp, zext: &RustBV, c: &RustBV, zext_left: bool) -> RustBV {
    let (a, b) = if zext_left { (zext, c) } else { (c, zext) };
    match op {
        UCmp::Eq => a.eq(b),
        UCmp::Ne => a.ne(b),
        UCmp::Ult => a.ult(b),
        UCmp::Ule => a.ule(b),
        UCmp::Ugt => a.ugt(b),
        UCmp::Uge => a.uge(b),
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
    let x = RustBV::symbol("x", inner_width);
    let zx = x.zero_extend(total);
    let cbv = RustBV::concrete(c & mask(total), total);

    // Folded RustBV result (goes through try_zext_const_cmp_fold).
    let folded_bv = rust_cmp(op, &zx, &cbv, zext_left).to_z3_ast().unwrap();

    // Unfolded Z3 reference over the same symbolic `x`.
    let zx_z3 = zx.to_z3_ast().unwrap();
    let c_z3 = cbv.to_z3_ast().unwrap();
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
