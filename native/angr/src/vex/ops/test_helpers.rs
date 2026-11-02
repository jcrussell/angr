// angr-9hleg: shared SIMD lane-test helpers, hoisted out of the former
// monolithic ops_tests.rs so the by-family `ops/tests_<fam>.rs` siblings can
// each `use super::test_helpers::*`. Helpers are `pub(super)` so all
// test sibling modules (children of `ops`) reach them. Behavior-preserving.

use clarirs_core::GLOBAL_CONTEXT;
use clarirs_core::prelude::*;
use clarirs_z3::Z3Solver;
use num_bigint::BigUint;

use super::*;
use crate::symbolic::SymContext;

// =========================================================================
// SIMD lane-test helpers (angr-ec82).
//
// This file declares no tests of its own. Its consumers are the sibling
// `ops/tests_vec_*.rs` modules plus `ops/tests_float_cmp.rs`; they all share
// the same plumbing: pack lane values little-endian into a u128, call
// VEXOps::binop/unop, then unpack and assert per lane. These helpers move
// ONLY that mechanical pack/unpack/assert skeleton out of the individual
// tests. Expected-value arrays stay inline at each call site — they are
// independent reference constants, never derived from the code under test
// (vacuous-test audit), and the helpers must preserve that
// property (they never compute an expected value).
// =========================================================================

/// Pack f32 lanes little-endian into a u128 (lane `i` occupies bits
/// `i*32 .. i*32+32`).
pub(super) fn pack_lanes_f32(lanes: &[f32]) -> u128 {
    let mut v: u128 = 0;
    for (i, &x) in lanes.iter().enumerate() {
        v |= (x.to_bits() as u128) << (i as u32 * 32);
    }
    v
}

/// Pack f64 lanes little-endian into a u128 (lane `i` occupies bits
/// `i*64 .. i*64+64`).
pub(super) fn pack_lanes_f64(lanes: &[f64]) -> u128 {
    let mut v: u128 = 0;
    for (i, &x) in lanes.iter().enumerate() {
        v |= (x.to_bits() as u128) << (i as u32 * 64);
    }
    v
}

/// Pack integer lanes of `lane_bits` width little-endian into a u128. Each
/// value is masked to `lane_bits` before being shifted in, so callers may
/// pass sign-extended values (e.g. `i16 as u16 as u128`) directly.
pub(super) fn pack_lanes_uint(lanes: &[u128], lane_bits: u32) -> u128 {
    let mask = lane_mask(lane_bits);
    let mut v: u128 = 0;
    for (i, &x) in lanes.iter().enumerate() {
        v |= (x & mask) << (i as u32 * lane_bits);
    }
    v
}

/// Low `lane_bits`-bit mask (`u128::MAX` when `lane_bits >= 128`).
pub(super) fn lane_mask(lane_bits: u32) -> u128 {
    if lane_bits >= 128 {
        u128::MAX
    } else {
        (1u128 << lane_bits) - 1
    }
}

/// Extract the raw bits of lane `i` (`lane_bits` wide) from a packed u128.
pub(super) fn unpack_lane(got: u128, i: usize, lane_bits: u32) -> u128 {
    (got >> (i as u32 * lane_bits)) & lane_mask(lane_bits)
}

/// Extract lane `i` as an f32 from a packed u128.
pub(super) fn unpack_lane_f32(got: u128, i: usize) -> f32 {
    f32::from_bits(unpack_lane(got, i, 32) as u32)
}

/// Extract lane `i` as an f64 from a packed u128.
pub(super) fn unpack_lane_f64(got: u128, i: usize) -> f64 {
    f64::from_bits(unpack_lane(got, i, 64) as u64)
}

/// Assert each f32 lane of `got` is within `tol` of the matching `exp`.
pub(super) fn assert_f32_lanes_approx(got: u128, exp: &[f32], tol: f32) {
    for (i, &expected) in exp.iter().enumerate() {
        let lane = unpack_lane_f32(got, i);
        assert!(
            (lane - expected).abs() < tol,
            "lane {i} expected {expected}, got {lane}"
        );
    }
}

/// Assert each f64 lane of `got` is within `tol` of the matching `exp`.
pub(super) fn assert_f64_lanes_approx(got: u128, exp: &[f64], tol: f64) {
    for (i, &expected) in exp.iter().enumerate() {
        let lane = unpack_lane_f64(got, i);
        assert!(
            (lane - expected).abs() < tol,
            "lane {i} expected {expected}, got {lane}"
        );
    }
}

/// Assert each f32 lane of `got` is bit-identical to the matching `exp`
/// (use for sign/NaN-sensitive ops like VFAbs where `==` is too lax).
pub(super) fn assert_f32_lanes_bits(got: u128, exp: &[f32]) {
    for (i, &expected) in exp.iter().enumerate() {
        let lane = unpack_lane_f32(got, i);
        assert_eq!(
            lane.to_bits(),
            expected.to_bits(),
            "lane {i} expected {expected}, got {lane}"
        );
    }
}

/// Assert each integer lane (`lane_bits` wide) of `got` equals the matching
/// `exp` entry. Callers pass masked/sign-extended expected values as u128.
pub(super) fn assert_int_lanes_eq(got: u128, exp: &[u128], lane_bits: u32) {
    let mask = lane_mask(lane_bits);
    for (i, &expected) in exp.iter().enumerate() {
        let lane = unpack_lane(got, i, lane_bits);
        let expected = expected & mask;
        assert_eq!(
            lane, expected,
            "lane {i} expected {expected:#x}, got {lane:#x}"
        );
    }
}

// ---- Newton-Raphson FP estimate / step (angr-iyon) ----
//
// VEX leaves Recip/RSqrt Est precision implementation-defined, and the Step
// ops in angr Python have no dedicated handler — both branches collapse to
// a fresh symbolic per lane. These tests pin the *shape* (lane count, width,
// upper-lane passthrough for the SSE F0x4 variants) rather than the value.
//
// Convenience: turn a width-N RustBV into its u128 representation via the
// solver so the result of a fresh-symbolic-per-lane op is observable.
pub(super) fn eval_v128(ctx: &SymContext, bv: &RustBV) -> u128 {
    assert!(ctx.is_sat(), "expected SAT for eval");
    ctx.eval(bv).expect("eval returned None")
}

pub(super) fn eval_i64(ctx: &SymContext, bv: &RustBV) -> u64 {
    eval_v128(ctx, bv) as u64
}

pub(super) fn eval_i32(ctx: &SymContext, bv: &RustBV) -> u32 {
    eval_v128(ctx, bv) as u32
}

// ---- solver-free evaluation (angr-mx00p.100.7) ----
//
// `SymContext` is the fork's Z3 wrapper; these evaluate through `clarirs_z3`
// directly, as angr-bv's `oracle_tests.rs` does, so the tests using them
// carry no dependency on it.

/// Z3's value for `bv` once every 1-bit `constraint` is asserted to be 1, or
/// `None` when the constraints are unsatisfiable. A symbolic `bv` is bound to
/// a fresh result symbol rather than passed to `Z3Solver::eval`, which runs
/// clarirs's simplify first: callers keep a term's operands symbolic (pinned
/// by a constraint) so that Z3, not the clarirs fold, decides its value.
pub(super) fn z3_eval(bv: &RustBV, constraints: &[RustBV]) -> Option<u128> {
    let ctx = &*GLOBAL_CONTEXT;
    let mut solver = Z3Solver::new(ctx);
    for c in constraints {
        assert_eq!(c.width(), 1, "a constraint is a 1-bit value");
        let holds = ctx
            .eq_(c.to_clarirs().unwrap(), ctx.bvv(BitVec::from((BigUint::from(1u32), 1))).unwrap())
            .unwrap();
        solver.add(&holds).unwrap();
    }
    if !solver.satisfiable().unwrap() {
        return None;
    }
    if let Some(v) = bv.as_u128() {
        return Some(v);
    }
    let result = ctx.bvs("vex_ops_z3_eval_result", bv.width()).unwrap();
    solver
        .add(&ctx.eq_(result.clone(), bv.to_clarirs().unwrap()).unwrap())
        .unwrap();
    match solver.eval(&result).unwrap().op() {
        AstOp::BVV(v) => Some(u128::try_from(v.to_biguint()).expect("at most 128 bits")),
        op => panic!("Z3 returned a non-literal {op:?}"),
    }
}

/// A `width`-bit symbol `name` and the 1-bit constraint pinning it to `v`.
pub(super) fn pinned_symbol(name: &str, v: u128, width: u32) -> (RustBV, RustBV) {
    let sym = RustBV::symbol(name, width);
    let pin = sym.eq(&RustBV::concrete(v, width));
    (sym, pin)
}

// ---- FCmpScalarLane (Iop_Cmp{EQ,LT,LE,UN}{32F0x4,64F0x2}) ----

pub(super) fn make_v128_lane0(lane0: u128, upper96: u128) -> u128 {
    debug_assert!(lane0 <= 0xFFFF_FFFF);
    (upper96 << 32) | lane0
}

pub(super) fn make_v128_lane0_64(lane0: u128, upper64: u128) -> u128 {
    debug_assert!(lane0 <= 0xFFFF_FFFF_FFFF_FFFF);
    (upper64 << 64) | lane0
}

// ---- FCmpVecPacked (Iop_Cmp{EQ,LT,LE,GT,GE,UN}{32Fx2,32Fx4,64Fx2}) ----

/// Pack four f32 values into a single 128-bit vector (lane 0 first).
pub(super) fn pack_4xf32(a: f32, b: f32, c: f32, d: f32) -> u128 {
    pack_lanes_f32(&[a, b, c, d])
}

pub(super) fn pack_2xf64(a: f64, b: f64) -> u128 {
    pack_lanes_f64(&[a, b])
}

/// Pack two f32 lanes into the low 64 bits (lane 0 first).
pub(super) fn pack_2xf32_64(a: f32, b: f32) -> u128 {
    pack_lanes_f32(&[a, b])
}
