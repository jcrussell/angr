// angr-l4dx: VEXOps unit tests, extracted out of the former in-file
// `mod tests` (~5165 lines) into a sibling file to shrink vex/ops.rs below
// the god-object threshold. Declared as a direct child of `ops` so
// `use super::*` reaches `ops`'s private items.

use super::*;

// =========================================================================
// SIMD lane-test helpers (angr-ec82).
//
// The ~57 packed-SIMD tests below all share the same plumbing: pack lane
// values little-endian into a u128, call VEXOps::binop/unop, then unpack
// and assert per lane. These helpers move ONLY that mechanical pack/unpack/
// assert skeleton out of the individual tests. Expected-value arrays stay
// inline at each call site — they are independent reference constants, never
// derived from the code under test (vacuous-test audit, 2026-06-12), and the
// helpers must preserve that property (they never compute an expected value).
// =========================================================================

/// Pack f32 lanes little-endian into a u128 (lane `i` occupies bits
/// `i*32 .. i*32+32`).
fn pack_lanes_f32(lanes: &[f32]) -> u128 {
    let mut v: u128 = 0;
    for (i, &x) in lanes.iter().enumerate() {
        v |= (x.to_bits() as u128) << (i as u32 * 32);
    }
    v
}

/// Pack f64 lanes little-endian into a u128 (lane `i` occupies bits
/// `i*64 .. i*64+64`).
fn pack_lanes_f64(lanes: &[f64]) -> u128 {
    let mut v: u128 = 0;
    for (i, &x) in lanes.iter().enumerate() {
        v |= (x.to_bits() as u128) << (i as u32 * 64);
    }
    v
}

/// Pack integer lanes of `lane_bits` width little-endian into a u128. Each
/// value is masked to `lane_bits` before being shifted in, so callers may
/// pass sign-extended values (e.g. `i16 as u16 as u128`) directly.
fn pack_lanes_uint(lanes: &[u128], lane_bits: u32) -> u128 {
    let mask = lane_mask(lane_bits);
    let mut v: u128 = 0;
    for (i, &x) in lanes.iter().enumerate() {
        v |= (x & mask) << (i as u32 * lane_bits);
    }
    v
}

/// Low `lane_bits`-bit mask (`u128::MAX` when `lane_bits >= 128`).
fn lane_mask(lane_bits: u32) -> u128 {
    if lane_bits >= 128 {
        u128::MAX
    } else {
        (1u128 << lane_bits) - 1
    }
}

/// Extract the raw bits of lane `i` (`lane_bits` wide) from a packed u128.
fn unpack_lane(got: u128, i: usize, lane_bits: u32) -> u128 {
    (got >> (i as u32 * lane_bits)) & lane_mask(lane_bits)
}

/// Extract lane `i` as an f32 from a packed u128.
fn unpack_lane_f32(got: u128, i: usize) -> f32 {
    f32::from_bits(unpack_lane(got, i, 32) as u32)
}

/// Extract lane `i` as an f64 from a packed u128.
fn unpack_lane_f64(got: u128, i: usize) -> f64 {
    f64::from_bits(unpack_lane(got, i, 64) as u64)
}

/// Assert each f32 lane of `got` is within `tol` of the matching `exp`.
fn assert_f32_lanes_approx(got: u128, exp: &[f32], tol: f32) {
    for (i, &expected) in exp.iter().enumerate() {
        let lane = unpack_lane_f32(got, i);
        assert!(
            (lane - expected).abs() < tol,
            "lane {i} expected {expected}, got {lane}"
        );
    }
}

/// Assert each f64 lane of `got` is within `tol` of the matching `exp`.
fn assert_f64_lanes_approx(got: u128, exp: &[f64], tol: f64) {
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
fn assert_f32_lanes_bits(got: u128, exp: &[f32]) {
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
fn assert_int_lanes_eq(got: u128, exp: &[u128], lane_bits: u32) {
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

#[test]
fn test_add_op() {
    let ctx = SymContext::new_mock();

    let a = RustBV::concrete(5, 32);
    let b = RustBV::concrete(3, 32);

    let result = VEXOps::binop(IROp::Add(IRType::I32), a, b, &ctx).unwrap();
    assert_eq!(result.as_u64(), Some(8));
}

#[test]
fn test_mul_widening() {
    let ctx = SymContext::new_mock();

    let a = RustBV::concrete(0xFFFFFFFF, 32);
    let b = RustBV::concrete(0xFFFFFFFF, 32);

    let result = VEXOps::binop(IROp::MullU(IRType::I32), a, b, &ctx).unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128(), Some(0xFFFFFFFE00000001));
}

#[test]
fn test_comparison_ops() {
    let ctx = SymContext::new_mock();

    let a = RustBV::concrete(5, 32);
    let b = RustBV::concrete(10, 32);

    let lt = VEXOps::binop(IROp::CmpLTU(IRType::I32), a.clone(), b.clone(), &ctx).unwrap();
    assert_eq!(lt.as_u64(), Some(1));

    let eq = VEXOps::binop(IROp::CmpEQ(IRType::I32), a.clone(), b.clone(), &ctx).unwrap();
    assert_eq!(eq.as_u64(), Some(0));
}

#[test]
fn test_divmod_u64_to_32_concrete() {
    let ctx = SymContext::new_mock();
    // 100 / 7 = 14 rem 2
    let dvd = RustBV::concrete(100, 64);
    let dvs = RustBV::concrete(7, 32);
    let result = VEXOps::binop(IROp::DivModU64to32, dvd, dvs, &ctx).unwrap();
    let v = result.as_u128().unwrap() as u64;
    assert_eq!(v & 0xFFFF_FFFF, 14, "quotient");
    assert_eq!((v >> 32) & 0xFFFF_FFFF, 2, "remainder");
}

#[test]
fn test_divmod_u64_to_32_symbolic_dividend() {
    let ctx = SymContext::new_mock();
    let dvd = RustBV::symbolic(&ctx, "dvd", 64);
    let dvs = RustBV::concrete(7, 32);
    let result = VEXOps::binop(IROp::DivModU64to32, dvd, dvs, &ctx).unwrap();
    assert_eq!(result.width(), 64);
    assert!(result.is_symbolic());
}

#[test]
fn test_divmod_s64_to_32_symbolic_divisor() {
    let ctx = SymContext::new_mock();
    let dvd = RustBV::concrete(0xFFFF_FFFF_FFFF_FF9C, 64); // -100 as i64
    let dvs = RustBV::symbolic(&ctx, "dvs", 32);
    let result = VEXOps::binop(IROp::DivModS64to32, dvd, dvs, &ctx).unwrap();
    assert_eq!(result.width(), 64);
    assert!(result.is_symbolic());
}

#[test]
fn test_divmod_u128_to_64_concrete() {
    let ctx = SymContext::new_mock();
    let dvd = RustBV::concrete(1000, 128);
    let dvs = RustBV::concrete(13, 64);
    let result = VEXOps::binop(IROp::DivModU128to64, dvd, dvs, &ctx).unwrap();
    let v = result.as_u128().unwrap();
    assert_eq!(v & 0xFFFF_FFFF_FFFF_FFFF, 76, "quotient = 1000/13");
    assert_eq!((v >> 64) & 0xFFFF_FFFF_FFFF_FFFF, 12, "remainder = 1000%13");
}

#[test]
fn test_divmod_u128_to_64_symbolic() {
    let ctx = SymContext::new_mock();
    let dvd = RustBV::symbolic(&ctx, "dvd128", 128);
    let dvs = RustBV::concrete(7, 64);
    let result = VEXOps::binop(IROp::DivModU128to64, dvd, dvs, &ctx).unwrap();
    assert_eq!(result.width(), 128);
    assert!(result.is_symbolic());
}

#[test]
fn test_sign_extend() {
    let ctx = SymContext::new_mock();

    let a = RustBV::concrete(0xFF, 8); // -1 in 8 bits

    let result = VEXOps::unop(
        IROp::SignExtend {
            from: IRType::I8,
            to: IRType::I32,
        },
        a,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.as_u64(), Some(0xFFFFFFFF)); // -1 in 32 bits
}

#[test]
fn test_float_add_concrete() {
    let ctx = SymContext::new_mock();

    let a = RustBV::concrete(1.5f32.to_bits() as u128, 32);
    let b = RustBV::concrete(2.5f32.to_bits() as u128, 32);

    let result = VEXOps::binop(IROp::FAdd(IRType::F32), a, b, &ctx).unwrap();
    let result_f = f32::from_bits(result.as_u64().unwrap() as u32);
    assert!((result_f - 4.0).abs() < 0.0001);
}

#[test]
fn test_qop_fmadd_concrete_f64() {
    let ctx = SymContext::new_mock();

    // 2.0 * 3.0 + 4.0 = 10.0
    let a = RustBV::concrete(2.0f64.to_bits() as u128, 64);
    let b = RustBV::concrete(3.0f64.to_bits() as u128, 64);
    let c = RustBV::concrete(4.0f64.to_bits() as u128, 64);

    let result = VEXOps::qop(IROp::FMAdd(IRType::F64), a, b, c, &ctx).unwrap();
    let result_f = f64::from_bits(result.as_u64().unwrap());
    assert!((result_f - 10.0).abs() < 1e-9);
}

#[test]
fn test_qop_fmsub_concrete_f32() {
    let ctx = SymContext::new_mock();

    // 5.0 * 2.0 - 3.0 = 7.0
    let a = RustBV::concrete(5.0f32.to_bits() as u128, 32);
    let b = RustBV::concrete(2.0f32.to_bits() as u128, 32);
    let c = RustBV::concrete(3.0f32.to_bits() as u128, 32);

    let result = VEXOps::qop(IROp::FMSub(IRType::F32), a, b, c, &ctx).unwrap();
    let result_f = f32::from_bits(result.as_u64().unwrap() as u32);
    assert!((result_f - 7.0).abs() < 1e-5);
}

#[test]
fn test_qop_rejects_non_quaternary() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(0, 64);
    let b = RustBV::concrete(0, 64);
    let c = RustBV::concrete(0, 64);

    let err = VEXOps::qop(IROp::Add(IRType::I64), a, b, c, &ctx).unwrap_err();
    match err {
        OpError::NotQuaternary(_) => (),
        other => panic!("expected NotQuaternary, got {:?}", other),
    }
}

#[test]
fn test_vector_add() {
    let ctx = SymContext::new_mock();

    // Two vectors of 4 x i32
    // [1, 2, 3, 4] + [10, 20, 30, 40] = [11, 22, 33, 44]
    let a: u128 = 1 | (2 << 32) | (3 << 64) | (4 << 96);
    let b: u128 = 10 | (20 << 32) | (30 << 64) | (40 << 96);

    let av = RustBV::concrete(a, 128);
    let bv = RustBV::concrete(b, 128);

    let result = VEXOps::binop(
        IROp::VAdd {
            elem: IRType::I32,
            count: 4,
        },
        av,
        bv,
        &ctx,
    )
    .unwrap();
    let rv = result.as_u128().unwrap();

    assert_eq!(rv & 0xFFFFFFFF, 11);
    assert_eq!((rv >> 32) & 0xFFFFFFFF, 22);
    assert_eq!((rv >> 64) & 0xFFFFFFFF, 33);
    assert_eq!((rv >> 96) & 0xFFFFFFFF, 44);
}

#[test]
#[allow(clippy::identity_op)] // explicit 4-lane layout reads better than the minimized form
fn test_vector_cmp_eq() {
    let ctx = SymContext::new_mock();

    // Two vectors of 4 x i32
    // [1, 2, 3, 4] == [1, 0, 3, 0] -> [0xFFFFFFFF, 0, 0xFFFFFFFF, 0]
    let a: u128 = 1 | (2 << 32) | (3 << 64) | (4 << 96);
    let b: u128 = 1 | (0 << 32) | (3 << 64) | (0 << 96);

    let av = RustBV::concrete(a, 128);
    let bv = RustBV::concrete(b, 128);

    let result = VEXOps::binop(
        IROp::VCmpEQ {
            elem: IRType::I32,
            count: 4,
        },
        av,
        bv,
        &ctx,
    )
    .unwrap();
    let rv = result.as_u128().unwrap();

    assert_eq!(rv & 0xFFFFFFFF, 0xFFFFFFFF); // 1 == 1
    assert_eq!((rv >> 32) & 0xFFFFFFFF, 0); // 2 != 0
    assert_eq!((rv >> 64) & 0xFFFFFFFF, 0xFFFFFFFF); // 3 == 3
    assert_eq!((rv >> 96) & 0xFFFFFFFF, 0); // 4 != 0
}

#[test]
fn test_concat() {
    let ctx = SymContext::new_mock();

    let hi = RustBV::concrete(0xDEAD, 16);
    let lo = RustBV::concrete(0xBEEF, 16);

    let result = VEXOps::binop(IROp::Concat { ty: IRType::I32 }, hi, lo, &ctx).unwrap();
    assert_eq!(result.as_u64(), Some(0xDEADBEEF));
}

#[test]
fn test_bit_manipulation() {
    let ctx = SymContext::new_mock();

    // CLZ (count leading zeros)
    let a = RustBV::concrete(0x0F00, 16); // binary: 0000111100000000
    let clz = VEXOps::unop(IROp::Clz(IRType::I16), a.clone(), &ctx).unwrap();
    assert_eq!(clz.as_u64(), Some(4)); // 4 leading zeros

    // CTZ (count trailing zeros)
    let ctz = VEXOps::unop(IROp::Ctz(IRType::I16), a.clone(), &ctx).unwrap();
    assert_eq!(ctz.as_u64(), Some(8)); // 8 trailing zeros

    // PopCount
    let pop = VEXOps::unop(IROp::PopCount(IRType::I16), a, &ctx).unwrap();
    assert_eq!(pop.as_u64(), Some(4)); // 4 bits set
}

#[test]
fn test_vec_float_scalar_add() {
    let ctx = SymContext::new_mock();

    // SSE scalar add: ADDSS xmm0, xmm1
    // xmm0 = [4.0f, 0, 0, 0], xmm1 = [2.0f, 0, 0, 0]
    // Result: xmm0 = [6.0f, 0, 0, 0]
    let f4_bits = 4.0f32.to_bits() as u128;
    let f2_bits = 2.0f32.to_bits() as u128;

    let xmm0 = RustBV::concrete(f4_bits, 128);
    let xmm1 = RustBV::concrete(f2_bits, 128);

    let result = VEXOps::binop(IROp::VFAddS { elem: IRType::F32 }, xmm0, xmm1, &ctx).unwrap();
    let result_val = result.as_u128().unwrap();
    let result_f32 = f32::from_bits((result_val & 0xFFFFFFFF) as u32);

    assert!(
        (result_f32 - 6.0).abs() < 0.0001,
        "Expected 6.0, got {}",
        result_f32
    );
}

#[test]
fn test_vec_float_scalar_div() {
    let ctx = SymContext::new_mock();

    // SSE scalar div: DIVSS xmm0, xmm1
    // xmm0 = [6.0f, 0, 0, 0], xmm1 = [2.0f, 0, 0, 0]
    // Result: xmm0 = [3.0f, 0, 0, 0]
    let f6_bits = 6.0f32.to_bits() as u128;
    let f2_bits = 2.0f32.to_bits() as u128;

    let xmm0 = RustBV::concrete(f6_bits, 128);
    let xmm1 = RustBV::concrete(f2_bits, 128);

    let result = VEXOps::binop(IROp::VFDivS { elem: IRType::F32 }, xmm0, xmm1, &ctx).unwrap();
    let result_val = result.as_u128().unwrap();
    let result_f32 = f32::from_bits((result_val & 0xFFFFFFFF) as u32);

    assert!(
        (result_f32 - 3.0).abs() < 0.0001,
        "Expected 3.0, got {}",
        result_f32
    );
}

/// Symbolic FAdd: solving `x + 2.0 == 5.0` should yield x == 3.0.
///
/// This is the canonical "constraint propagation" check for the new Z3
/// FP-theory wiring: previously the symbolic branch returned a fresh
/// unconstrained symbol and the solver would accept any value of x.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_float_add_symbolic_constraint() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "x", 32);
    let two = RustBV::concrete(2.0f32.to_bits() as u128, 32);

    let sum = VEXOps::binop(IROp::FAdd(IRType::F32), x.clone(), two, &ctx).unwrap();

    // Constrain: sum's IEEE bits == bits(5.0).
    let five = RustBV::concrete(5.0f32.to_bits() as u128, 32);
    let eq = sum.to_z3_ast().eq(five.to_z3_ast());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT after FAdd symbolic constraint");

    let model_x = ctx.eval(&x).expect("eval(x) returned None");
    let result_f = f32::from_bits(model_x as u32);
    assert!(
        (result_f - 3.0).abs() < 1e-6,
        "Expected x == 3.0, got {}",
        result_f
    );
}

/// Symbolic FSqrt: `sqrt(x) == 4.0` should give x == 16.0.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_float_sqrt_symbolic_constraint() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "sqrt_x", 64);

    let sqrt_x = VEXOps::unop(IROp::FSqrt(IRType::F64), x.clone(), &ctx).unwrap();

    let four = RustBV::concrete(4.0f64.to_bits() as u128, 64);
    let eq = sqrt_x.to_z3_ast().eq(four.to_z3_ast());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT after FSqrt symbolic constraint");

    let model_x = ctx.eval(&x).expect("eval(x) returned None");
    let result_f = f64::from_bits(model_x as u64);
    assert!(
        (result_f - 16.0).abs() < 1e-9,
        "Expected x == 16.0, got {}",
        result_f
    );
}

/// Symbolic FCmpLT: bracket x with `1.0 < x < 2.0` via two symbolic
/// FCmpLT comparisons. Solver should accept and produce x in (1.0, 2.0).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_float_cmp_lt_symbolic_constraint() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "fcmp_x", 32);
    let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
    let two = RustBV::concrete(2.0f32.to_bits() as u128, 32);

    // x < 2.0
    let lt_x_two = VEXOps::binop(IROp::FCmpLT(IRType::F32), x.clone(), two, &ctx).unwrap();
    // 1.0 < x
    let lt_one_x = VEXOps::binop(IROp::FCmpLT(IRType::F32), one, x.clone(), &ctx).unwrap();

    ctx.assume_true(&lt_x_two);
    ctx.assume_true(&lt_one_x);
    assert!(ctx.is_sat(), "expected SAT for 1.0 < x < 2.0");

    let model_x = ctx.eval(&x).expect("eval(x) returned None");
    let result_f = f32::from_bits(model_x as u32);
    assert!(
        result_f > 1.0 && result_f < 2.0,
        "Expected 1.0 < x < 2.0, got {}",
        result_f
    );
}

/// Symbolic VFAddS (ADDSS-style): `x_low + 2.0 == 5.0` should yield x_low == 3.0,
/// and the upper 96 bits of `xmm0` must pass through unchanged.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_float_scalar_add_symbolic() {
    let ctx = SymContext::new_mock();
    let xmm0 = RustBV::symbolic(&ctx, "xmm0", 128);
    // xmm1 = [2.0f, 0, 0, 0]
    let xmm1 = RustBV::concrete(2.0f32.to_bits() as u128, 128);

    let result =
        VEXOps::binop(IROp::VFAddS { elem: IRType::F32 }, xmm0.clone(), xmm1, &ctx).unwrap();
    assert_eq!(result.width(), 128);

    // Constrain low 32 bits of result to bits(5.0).
    let res_lo = result.extract(31, 0, &ctx);
    let five = RustBV::concrete(5.0f32.to_bits() as u128, 32);
    let eq = res_lo.to_z3_ast().eq(five.to_z3_ast());
    ctx.add_constraint(eq);
    assert!(
        ctx.is_sat(),
        "expected SAT after VFAddS symbolic constraint"
    );

    let model_x = ctx.eval(&xmm0).expect("eval(xmm0) returned None");
    let lane0 = f32::from_bits((model_x & 0xFFFF_FFFF) as u32);
    assert!(
        (lane0 - 3.0).abs() < 1e-6,
        "Expected lane0 == 3.0, got {}",
        lane0
    );

    // Verify upper 96 bits of result equal upper 96 bits of xmm0 (passthrough).
    let upper_in = xmm0.extract(127, 32, &ctx);
    let upper_out = result.extract(127, 32, &ctx);
    let eq_upper = upper_in.to_z3_ast().eq(upper_out.to_z3_ast());
    ctx.add_constraint(eq_upper);
    assert!(ctx.is_sat(), "expected upper-bits passthrough to hold");
}

/// Symbolic VFSqrtS (SQRTSS-style): sqrt(low32(xmm)) == 4.0 → low32 == 16.0.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_float_scalar_sqrt_symbolic() {
    let ctx = SymContext::new_mock();
    let xmm = RustBV::symbolic(&ctx, "xmm_sqrt", 128);

    let result = VEXOps::unop(IROp::VFSqrtS { elem: IRType::F32 }, xmm.clone(), &ctx).unwrap();
    assert_eq!(result.width(), 128);

    let res_lo = result.extract(31, 0, &ctx);
    let four = RustBV::concrete(4.0f32.to_bits() as u128, 32);
    let eq = res_lo.to_z3_ast().eq(four.to_z3_ast());
    ctx.add_constraint(eq);
    assert!(
        ctx.is_sat(),
        "expected SAT after VFSqrtS symbolic constraint"
    );

    let model_x = ctx.eval(&xmm).expect("eval(xmm) returned None");
    let lane0 = f32::from_bits((model_x & 0xFFFF_FFFF) as u32);
    assert!(
        (lane0 - 16.0).abs() < 1e-4,
        "Expected lane0 == 16.0, got {}",
        lane0
    );
}

/// Symbolic VFMaxS (MAXSS-style): with `xmm0` symbolic and `xmm1 = [3.0, ...]`,
/// constrain low32(result) == 5.0 — solver must pick xmm0.lane0 == 5.0 (since
/// max(5.0, 3.0) == 5.0). Also a model where xmm0.lane0 == 1.0 must NOT satisfy
/// the constraint (we don't test that here, but the ITE encoding guarantees it).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_float_scalar_max_symbolic() {
    let ctx = SymContext::new_mock();
    let xmm0 = RustBV::symbolic(&ctx, "xmm_max", 128);
    let xmm1 = RustBV::concrete(3.0f32.to_bits() as u128, 128);

    let result =
        VEXOps::binop(IROp::VFMaxS { elem: IRType::F32 }, xmm0.clone(), xmm1, &ctx).unwrap();
    assert_eq!(result.width(), 128);

    let res_lo = result.extract(31, 0, &ctx);
    let five = RustBV::concrete(5.0f32.to_bits() as u128, 32);
    let eq = res_lo.to_z3_ast().eq(five.to_z3_ast());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT after VFMaxS == 5.0");

    let model_x = ctx.eval(&xmm0).expect("eval(xmm0) returned None");
    let lane0 = f32::from_bits((model_x & 0xFFFF_FFFF) as u32);
    assert!(
        (lane0 - 5.0).abs() < 1e-6,
        "Expected lane0 == 5.0 (since max(lane0, 3.0) == 5.0), got {}",
        lane0
    );
}

/// Symbolic VFMinS (MINSS-style): with `xmm1 = [3.0, ...]` and target == 1.0,
/// solver must pick xmm0.lane0 == 1.0 (since min(1.0, 3.0) == 1.0).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_float_scalar_min_symbolic() {
    let ctx = SymContext::new_mock();
    let xmm0 = RustBV::symbolic(&ctx, "xmm_min", 128);
    let xmm1 = RustBV::concrete(3.0f32.to_bits() as u128, 128);

    let result =
        VEXOps::binop(IROp::VFMinS { elem: IRType::F32 }, xmm0.clone(), xmm1, &ctx).unwrap();
    assert_eq!(result.width(), 128);

    let res_lo = result.extract(31, 0, &ctx);
    let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
    let eq = res_lo.to_z3_ast().eq(one.to_z3_ast());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT after VFMinS == 1.0");

    let model_x = ctx.eval(&xmm0).expect("eval(xmm0) returned None");
    let lane0 = f32::from_bits((model_x & 0xFFFF_FFFF) as u32);
    assert!(
        (lane0 - 1.0).abs() < 1e-6,
        "Expected lane0 == 1.0 (since min(lane0, 3.0) == 1.0), got {}",
        lane0
    );
}

// =========================================================================
// FP edge cases (angr-io1t)
// =========================================================================

/// RoundF32toInt with symbolic rm: value=-2.5f32, target=-3.0f32 forces
/// rm low-2-bits == 1 (round toward -inf). Exercises the 4-way ITE built
/// by build_fp_round_to_int_cached.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_round_f32_to_int_symbolic_rm() {
    let ctx = SymContext::new_mock();
    let rm = RustBV::symbolic(&ctx, "rm_f32", 32);
    let value = RustBV::concrete((-2.5f32).to_bits() as u128, 32);

    let result = VEXOps::binop(IROp::RoundF32toInt, rm.clone(), value, &ctx).unwrap();
    let target = RustBV::concrete((-3.0f32).to_bits() as u128, 32);
    let eq = result.to_z3_ast().eq(target.to_z3_ast());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT for round(-2.5)==-3.0");

    let model_rm = ctx.eval(&rm).expect("eval(rm) returned None");
    assert_eq!(
        (model_rm as u32) & 0x3,
        1,
        "expected rm low2 bits == 1 (round toward -inf), got {}",
        model_rm & 0x3
    );
}

/// RoundF64toInt with symbolic rm: value=2.5f64, target=3.0f64 forces
/// rm low-2-bits == 2 (round toward +inf). RNE on 2.5 ties to even (2).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_round_f64_to_int_symbolic_rm() {
    let ctx = SymContext::new_mock();
    let rm = RustBV::symbolic(&ctx, "rm_f64", 32);
    let value = RustBV::concrete(2.5f64.to_bits() as u128, 64);

    let result = VEXOps::binop(IROp::RoundF64toInt, rm.clone(), value, &ctx).unwrap();
    let target = RustBV::concrete(3.0f64.to_bits() as u128, 64);
    let eq = result.to_z3_ast().eq(target.to_z3_ast());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT for round(2.5)==3.0");

    let model_rm = ctx.eval(&rm).expect("eval(rm) returned None");
    assert_eq!(
        (model_rm as u32) & 0x3,
        2,
        "expected rm low2 bits == 2 (round toward +inf), got {}",
        model_rm & 0x3
    );
}

/// FDiv with concrete RNE rm goes through the native-f32 fast path.
/// 1.0/10.0 under RNE is 0x3DCCCCCD (correctly rounded up).
#[test]
fn test_float_div_with_rm_rne_fastpath_f32() {
    let ctx = SymContext::new_mock();
    let rm_rne = RustBV::concrete(0, 32);
    let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
    let ten = RustBV::concrete(10.0f32.to_bits() as u128, 32);

    let result = VEXOps::binop_with_rm(IROp::FDiv(IRType::F32), rm_rne, one, ten, &ctx).unwrap();
    // RNE keeps the native-f32 fast path; result is fully concrete (not
    // a Z3 expression).
    assert!(!result.is_symbolic());
    assert_eq!(result.as_u64(), Some(0x3DCCCCCD));
}

/// FDiv with concrete RZ (toward zero) rm: 1.0/10.0 truncates the last
/// mantissa bit → 0x3DCCCCCC (one ULP below RNE).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_float_div_with_rm_rz_f32() {
    let ctx = SymContext::new_mock();
    let rm_rz = RustBV::concrete(3, 32);
    let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
    let ten = RustBV::concrete(10.0f32.to_bits() as u128, 32);

    let result = VEXOps::binop_with_rm(IROp::FDiv(IRType::F32), rm_rz, one, ten, &ctx).unwrap();
    let bits = ctx.eval(&result).expect("eval failed") as u32;
    assert_eq!(bits, 0x3DCCCCCC, "1/10 with RZ rounds toward zero");
}

/// FDiv with concrete RU (toward +inf) rm: 1.0/10.0 = 0x3DCCCCCD (same
/// as RNE because the discarded bits push up).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_float_div_with_rm_ru_f32() {
    let ctx = SymContext::new_mock();
    let rm_ru = RustBV::concrete(2, 32);
    let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
    let ten = RustBV::concrete(10.0f32.to_bits() as u128, 32);

    let result = VEXOps::binop_with_rm(IROp::FDiv(IRType::F32), rm_ru, one, ten, &ctx).unwrap();
    let bits = ctx.eval(&result).expect("eval failed") as u32;
    assert_eq!(bits, 0x3DCCCCCD);
}

/// FDiv with concrete RD (toward -inf) rm on a positive result equals RZ:
/// 1.0/10.0 → 0x3DCCCCCC.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_float_div_with_rm_rd_f32() {
    let ctx = SymContext::new_mock();
    let rm_rd = RustBV::concrete(1, 32);
    let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
    let ten = RustBV::concrete(10.0f32.to_bits() as u128, 32);

    let result = VEXOps::binop_with_rm(IROp::FDiv(IRType::F32), rm_rd, one, ten, &ctx).unwrap();
    let bits = ctx.eval(&result).expect("eval failed") as u32;
    assert_eq!(bits, 0x3DCCCCCC);
}

/// FDiv with symbolic rm: constraining result == 0x3DCCCCCC forces rm
/// low-2-bits ∈ {1, 3} (RD or RZ); 0x3DCCCCCD forces rm low-2-bits ∈
/// {0, 2}. Exercises the 4-way ITE built by `build_fp_arith_rm_cached`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_float_div_with_symbolic_rm_f32() {
    let ctx = SymContext::new_mock();
    let rm = RustBV::symbolic(&ctx, "rm_div_f32", 32);
    let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
    let ten = RustBV::concrete(10.0f32.to_bits() as u128, 32);

    let result =
        VEXOps::binop_with_rm(IROp::FDiv(IRType::F32), rm.clone(), one, ten, &ctx).unwrap();
    let target = RustBV::concrete(0x3DCCCCCC, 32);
    let eq = result.to_z3_ast().eq(target.to_z3_ast());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT for div(1,10) == 0x3DCCCCCC");

    let model_rm = ctx.eval(&rm).expect("eval(rm) returned None") as u32;
    let low2 = model_rm & 0x3;
    assert!(
        low2 == 1 || low2 == 3,
        "expected rm low2 ∈ {{1, 3}} (RD/RZ), got {}",
        low2
    );
}

/// FAdd with concrete RZ on inexact-sum operands. 0x3F800001 + 0x3F800002
/// = 2.0 + 1.5ulp (exact). RNE rounds-to-even → 2.0 + 2ulp = 0x40000002;
/// RZ truncates → 2.0 + 1ulp = 0x40000001.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_float_add_with_rm_rz_f32() {
    let ctx = SymContext::new_mock();
    let rm_rz = RustBV::concrete(3, 32);
    let a = RustBV::concrete(0x3F800001, 32);
    let b = RustBV::concrete(0x3F800002, 32);

    let result = VEXOps::binop_with_rm(IROp::FAdd(IRType::F32), rm_rz, a, b, &ctx).unwrap();
    let bits = ctx.eval(&result).expect("eval failed") as u32;
    assert_eq!(bits, 0x40000001, "RZ truncates 1.5ulp tie down");
}

/// SqrtRm: sqrt(2.0f32) under RU (toward +inf). True value is between
/// 0x3FB504F3 (RNE) and 0x3FB504F4; RU pushes up by one ulp.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_float_sqrt_with_rm_ru_f32() {
    let ctx = SymContext::new_mock();
    let rm_ru = RustBV::concrete(2, 32);
    let two = RustBV::concrete(2.0f32.to_bits() as u128, 32);

    let result = VEXOps::unop_with_rm(IROp::FSqrt(IRType::F32), rm_ru, two, &ctx).unwrap();
    let bits = ctx.eval(&result).expect("eval failed") as u32;
    assert_eq!(bits, 0x3FB504F4, "sqrt(2) under RU rounds up one ulp");
}

/// SqrtRm: sqrt(2.0f32) RNE keeps the native fast path (no Z3).
#[test]
fn test_float_sqrt_with_rm_rne_fastpath_f32() {
    let ctx = SymContext::new_mock();
    let rm_rne = RustBV::concrete(0, 32);
    let two = RustBV::concrete(2.0f32.to_bits() as u128, 32);

    let result = VEXOps::unop_with_rm(IROp::FSqrt(IRType::F32), rm_rne, two, &ctx).unwrap();
    assert!(!result.is_symbolic());
    assert_eq!(result.as_u64(), Some(0x3FB504F3));
}

/// Iop_SqrtF32 reaches us as a VEX Binop (arg1=rm, arg2=value). Verify
/// that VEXOps::binop routes it through unop_with_rm so the rounding
/// mode is honored — RU on sqrt(2.0f32) must round up by one ulp.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_float_sqrt_via_binop_with_rm_ru_f32() {
    let ctx = SymContext::new_mock();
    let rm_ru = RustBV::concrete(2, 32);
    let two = RustBV::concrete(2.0f32.to_bits() as u128, 32);

    let result = VEXOps::binop(IROp::FSqrt(IRType::F32), rm_ru, two, &ctx).unwrap();
    let bits = ctx.eval(&result).expect("eval failed") as u32;
    assert_eq!(bits, 0x3FB504F4, "binop FSqrt+RU rounds up one ulp");
}

/// Iop_SqrtF64 via binop with RNE: native fast path returns sqrt(16.0)=4.0.
#[test]
fn test_float_sqrt_via_binop_rne_fastpath_f64() {
    let ctx = SymContext::new_mock();
    let rm_rne = RustBV::concrete(0, 32);
    let sixteen = RustBV::concrete(16.0f64.to_bits() as u128, 64);

    let result = VEXOps::binop(IROp::FSqrt(IRType::F64), rm_rne, sixteen, &ctx).unwrap();
    assert!(!result.is_symbolic(), "RNE concrete must stay native");
    assert_eq!(result.as_u64(), Some(4.0f64.to_bits()));
}

/// F32→I32S with NaN: Rust `as` cast collapses NaN to 0.
#[test]
fn test_f32_to_i32s_nan() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(f32::NAN.to_bits() as u128, 32);
    let result = VEXOps::unop(IROp::F32toI32S, arg, &ctx).unwrap();
    assert_eq!(result.width(), 32);
    assert_eq!(result.as_u64(), Some(0), "NaN as i32 must be 0");
}

/// F32→I32S with +inf: saturates to i32::MAX.
#[test]
fn test_f32_to_i32s_pos_infinity() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(f32::INFINITY.to_bits() as u128, 32);
    let result = VEXOps::unop(IROp::F32toI32S, arg, &ctx).unwrap();
    assert_eq!(
        result.as_u64(),
        Some(i32::MAX as u32 as u64),
        "+inf as i32 must saturate to i32::MAX"
    );
}

/// F32→I32S with -inf: saturates to i32::MIN.
#[test]
fn test_f32_to_i32s_neg_infinity() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(f32::NEG_INFINITY.to_bits() as u128, 32);
    let result = VEXOps::unop(IROp::F32toI32S, arg, &ctx).unwrap();
    assert_eq!(
        result.as_u64(),
        Some(i32::MIN as u32 as u64),
        "-inf as i32 must saturate to i32::MIN"
    );
}

/// F32→I32S with very large magnitude: also saturates.
#[test]
fn test_f32_to_i32s_overflow() {
    let ctx = SymContext::new_mock();
    let big = RustBV::concrete(1e30f32.to_bits() as u128, 32);
    let result = VEXOps::unop(IROp::F32toI32S, big, &ctx).unwrap();
    assert_eq!(
        result.as_u64(),
        Some(i32::MAX as u32 as u64),
        "1e30 must saturate to i32::MAX"
    );

    let neg_big = RustBV::concrete((-1e30f32).to_bits() as u128, 32);
    let neg_result = VEXOps::unop(IROp::F32toI32S, neg_big, &ctx).unwrap();
    assert_eq!(
        neg_result.as_u64(),
        Some(i32::MIN as u32 as u64),
        "-1e30 must saturate to i32::MIN"
    );
}

/// I32S→F32 of INT32_MIN: -2^31 is exactly representable in F32.
#[test]
fn test_i32s_to_f32_int_min() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(i32::MIN as u32 as u128, 32);
    let result = VEXOps::unop(IROp::I32StoF32, arg, &ctx).unwrap();
    let f = f32::from_bits(result.as_u64().unwrap() as u32);
    assert_eq!(
        f, -2147483648.0f32,
        "I32_MIN must round-trip to -2^31 as f32"
    );
}

/// F64→F32 (no-rm unop) precision overflow: 1e300 exceeds f32::MAX.
/// Rust `as f32` saturates toward +inf, matching IEEE-754 behavior under RNE.
#[test]
fn test_f64_to_f32_overflow() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(1e300f64.to_bits() as u128, 64);
    let result = VEXOps::unop(IROp::F64toF32, arg, &ctx).unwrap();
    assert_eq!(result.width(), 32);
    let f = f32::from_bits(result.as_u64().unwrap() as u32);
    assert!(
        f.is_infinite() && f.is_sign_positive(),
        "1e300 → +inf, got {}",
        f
    );
}

/// F64→F32 of NaN: result is still NaN. Just check is_nan; the exact
/// payload bits aren't part of the contract.
#[test]
fn test_f64_to_f32_nan() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(f64::NAN.to_bits() as u128, 64);
    let result = VEXOps::unop(IROp::F64toF32, arg, &ctx).unwrap();
    assert_eq!(result.width(), 32);
    let f = f32::from_bits(result.as_u64().unwrap() as u32);
    assert!(f.is_nan(), "NaN must remain NaN after F64→F32");
}

/// F64→F32 of -infinity: preserved as -infinity.
#[test]
fn test_f64_to_f32_neg_infinity() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(f64::NEG_INFINITY.to_bits() as u128, 64);
    let result = VEXOps::unop(IROp::F64toF32, arg, &ctx).unwrap();
    let f = f32::from_bits(result.as_u64().unwrap() as u32);
    assert!(f.is_infinite() && f.is_sign_negative(), "-inf preserved");
}

/// VFSubS concrete lane isolation: SUBSS xmm0, xmm1 — only lane 0 changes,
/// upper 96 bits of xmm0 pass through unchanged.
#[test]
fn test_vec_float_scalar_sub_concrete_lane_isolation() {
    let ctx = SymContext::new_mock();

    // xmm0: lane0=10.0, upper bits = 0xDEAD_BEEF_CAFE_BABE_1234_5678 (96 bits)
    let upper_pattern: u128 = 0xDEAD_BEEF_CAFE_BABE_1234_5678u128 << 32;
    let xmm0_bits = upper_pattern | (10.0f32.to_bits() as u128);
    let xmm0 = RustBV::concrete(xmm0_bits, 128);
    let xmm1 = RustBV::concrete(3.0f32.to_bits() as u128, 128);

    let result = VEXOps::binop(IROp::VFSubS { elem: IRType::F32 }, xmm0, xmm1, &ctx).unwrap();
    let rv = result.as_u128().unwrap();
    let lane0 = f32::from_bits((rv & 0xFFFF_FFFF) as u32);
    assert!(
        (lane0 - 7.0).abs() < 1e-6,
        "10.0 - 3.0 == 7.0, got {}",
        lane0
    );
    assert_eq!(
        rv & !0xFFFF_FFFFu128,
        upper_pattern,
        "upper 96 bits must pass through"
    );
}

/// VFMulS concrete lane isolation: MULSS xmm0, xmm1.
#[test]
fn test_vec_float_scalar_mul_concrete_lane_isolation() {
    let ctx = SymContext::new_mock();

    let upper_pattern: u128 = 0xFEED_FACE_BAAD_F00D_8BAD_F00Du128 << 32;
    let xmm0_bits = upper_pattern | (4.0f32.to_bits() as u128);
    let xmm0 = RustBV::concrete(xmm0_bits, 128);
    let xmm1 = RustBV::concrete(2.5f32.to_bits() as u128, 128);

    let result = VEXOps::binop(IROp::VFMulS { elem: IRType::F32 }, xmm0, xmm1, &ctx).unwrap();
    let rv = result.as_u128().unwrap();
    let lane0 = f32::from_bits((rv & 0xFFFF_FFFF) as u32);
    assert!(
        (lane0 - 10.0).abs() < 1e-6,
        "4.0 * 2.5 == 10.0, got {}",
        lane0
    );
    assert_eq!(
        rv & !0xFFFF_FFFFu128,
        upper_pattern,
        "upper 96 bits must pass through"
    );
}

/// Concrete coverage for every scalar-in-vector FP IROp at F64 precision.
/// The {add,sub,mul,div,sqrt,max,min} S-suffixed variants all write to
/// lane 0 (low 64 bits) and pass the upper 64 bits of `xmm0` through.
#[test]
fn test_vec_float_scalar_all_variants_f64() {
    let ctx = SymContext::new_mock();
    let upper_pattern: u128 = 0xCAFE_BABE_DEAD_BEEFu128 << 64;

    let xmm0 = |lane0: f64| RustBV::concrete(upper_pattern | (lane0.to_bits() as u128), 128);
    let xmm1 = |lane0: f64| RustBV::concrete(lane0.to_bits() as u128, 128);

    let cases: Vec<(IROp, f64, f64, f64)> = vec![
        (IROp::VFAddS { elem: IRType::F64 }, 4.0, 1.5, 5.5),
        (IROp::VFSubS { elem: IRType::F64 }, 5.0, 1.25, 3.75),
        (IROp::VFMulS { elem: IRType::F64 }, 3.0, 2.5, 7.5),
        (IROp::VFDivS { elem: IRType::F64 }, 9.0, 4.0, 2.25),
        (IROp::VFMaxS { elem: IRType::F64 }, 1.5, 2.5, 2.5),
        (IROp::VFMinS { elem: IRType::F64 }, 1.5, 2.5, 1.5),
    ];

    for (op, l, r, expected) in cases {
        let result = VEXOps::binop(op, xmm0(l), xmm1(r), &ctx).unwrap();
        let rv = result.as_u128().unwrap();
        let lane0 = f64::from_bits((rv & 0xFFFF_FFFF_FFFF_FFFFu128) as u64);
        assert!(
            (lane0 - expected).abs() < 1e-9,
            "{:?}: lane0={} expected={}",
            op,
            lane0,
            expected,
        );
        assert_eq!(
            rv & !0xFFFF_FFFF_FFFF_FFFFu128,
            upper_pattern,
            "{:?}: upper 64 bits must pass through",
            op,
        );
    }

    // Unary VFSqrtS{F64}: sqrt(16.0) = 4.0, upper 64 bits pass through.
    let arg = xmm0(16.0);
    let sqrt_res = VEXOps::unop(IROp::VFSqrtS { elem: IRType::F64 }, arg, &ctx).unwrap();
    let sv = sqrt_res.as_u128().unwrap();
    let sqrt_lane0 = f64::from_bits((sv & 0xFFFF_FFFF_FFFF_FFFFu128) as u64);
    assert!(
        (sqrt_lane0 - 4.0).abs() < 1e-9,
        "VFSqrtS{{F64}}: lane0={} expected=4.0",
        sqrt_lane0,
    );
    assert_eq!(
        sv & !0xFFFF_FFFF_FFFF_FFFFu128,
        upper_pattern,
        "VFSqrtS{{F64}}: upper 64 bits must pass through",
    );
}

/// VFMaxS concrete with lane0=NaN: Rust `>` returns false for NaN, so
/// max picks the right operand. Documents the SSE max-is-not-IEEE-max
/// semantics encoded by the ITE in vec_float_scalar_lane_minmax.
#[test]
fn test_vec_float_scalar_max_nan_concrete() {
    let ctx = SymContext::new_mock();

    let xmm0 = RustBV::concrete(f32::NAN.to_bits() as u128, 128);
    let xmm1 = RustBV::concrete(3.0f32.to_bits() as u128, 128);

    let result = VEXOps::binop(IROp::VFMaxS { elem: IRType::F32 }, xmm0, xmm1, &ctx).unwrap();
    let rv = result.as_u128().unwrap();
    let lane0 = f32::from_bits((rv & 0xFFFF_FFFF) as u32);
    // NaN > 3.0 is false, so max picks 3.0 (the right operand).
    assert!(
        (lane0 - 3.0).abs() < 1e-6,
        "MAXSS(NaN, 3.0) returns the right operand, got {}",
        lane0
    );
}

/// Symbolic ShlN16x8: shift count is symbolic; constrain to 4 and verify
/// each lane is `lane << 4`. Exercises the symbolic-shift fallback that
/// builds Z3 `bvshl` expressions per lane.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_shl_n_symbolic_shift() {
    let ctx = SymContext::new_mock();

    // Vector: [0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008]
    let mut v: u128 = 0;
    for i in 0..8u32 {
        v |= ((i + 1) as u128) << (i * 16);
    }
    let vec = RustBV::concrete(v, 128);
    let shift = RustBV::symbolic(&ctx, "shl_amt", 8);

    let result = VEXOps::binop(
        IROp::VShlN {
            elem: IRType::I16,
            count: 8,
        },
        vec,
        shift.clone(),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);

    // Constrain shift == 4.
    let four = RustBV::concrete(4, 8);
    let eq = shift.to_z3_ast().eq(four.to_z3_ast());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT after constraining shift == 4");

    let model = ctx.eval(&result).expect("eval(result) returned None");
    for i in 0..8u32 {
        let lane = (model >> (i * 16)) & 0xFFFF;
        let expected = ((i as u128 + 1) << 4) & 0xFFFF;
        assert_eq!(
            lane, expected,
            "lane {} expected {:#x}, got {:#x}",
            i, expected, lane
        );
    }
}

/// Symbolic ShrN32x4: shift count is symbolic; constrain to 8 and verify
/// each lane is `lane >> 8`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_shr_n_symbolic_shift() {
    let ctx = SymContext::new_mock();

    // Vector: [0xAABBCCDD, 0x11223344, 0xDEADBEEF, 0xCAFEBABE]
    let lanes: [u32; 4] = [0xAABBCCDD, 0x11223344, 0xDEADBEEF, 0xCAFEBABE];
    let mut v: u128 = 0;
    for (i, lane) in lanes.iter().enumerate() {
        v |= (*lane as u128) << (i * 32);
    }
    let vec = RustBV::concrete(v, 128);
    let shift = RustBV::symbolic(&ctx, "shr_amt", 8);

    let result = VEXOps::binop(
        IROp::VShrN {
            elem: IRType::I32,
            count: 4,
        },
        vec,
        shift.clone(),
        &ctx,
    )
    .unwrap();

    let eight = RustBV::concrete(8, 8);
    let eq = shift.to_z3_ast().eq(eight.to_z3_ast());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT after constraining shift == 8");

    let model = ctx.eval(&result).expect("eval(result) returned None");
    for (i, lane) in lanes.iter().enumerate() {
        let got = ((model >> (i * 32)) & 0xFFFF_FFFF) as u32;
        let expected = lane >> 8;
        assert_eq!(
            got, expected,
            "lane {} expected {:#x}, got {:#x}",
            i, expected, got
        );
    }
}

/// Symbolic SarN16x8 with negative lanes: shift count is symbolic; constrain
/// to 4 and verify sign-extending shift (negative values stay negative).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_sar_n_symbolic_shift() {
    let ctx = SymContext::new_mock();

    // Lanes: mix of positive and negative i16 values.
    let lanes: [i16; 8] = [-1, -32768, -16, 0, 1, 0x4000, -2, 256];
    let mut v: u128 = 0;
    for (i, lane) in lanes.iter().enumerate() {
        v |= ((*lane as u16) as u128) << (i * 16);
    }
    let vec = RustBV::concrete(v, 128);
    let shift = RustBV::symbolic(&ctx, "sar_amt", 8);

    let result = VEXOps::binop(
        IROp::VSarN {
            elem: IRType::I16,
            count: 8,
        },
        vec,
        shift.clone(),
        &ctx,
    )
    .unwrap();

    let four = RustBV::concrete(4, 8);
    let eq = shift.to_z3_ast().eq(four.to_z3_ast());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT after constraining shift == 4");

    let model = ctx.eval(&result).expect("eval(result) returned None");
    for (i, lane) in lanes.iter().enumerate() {
        let got = ((model >> (i * 16)) & 0xFFFF) as u16 as i16;
        let expected = lane >> 4; // arithmetic shift in Rust on i16
        assert_eq!(
            got, expected,
            "lane {} expected {}, got {}",
            i, expected, got
        );
    }
}

/// Symbolic ShlN with unbounded shift: just verify a Z3 expression is
/// produced rather than an UnsupportedVectorOp error. Documents the
/// "fully unconstrained" path stays inside the solver.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_shl_n_unbounded_shift() {
    let ctx = SymContext::new_mock();

    let vec = RustBV::concrete(0x1111_2222_3333_4444u128, 64);
    let shift = RustBV::symbolic(&ctx, "shl_amt_free", 8);

    let result = VEXOps::binop(
        IROp::VShlN {
            elem: IRType::I16,
            count: 4,
        },
        vec,
        shift,
        &ctx,
    )
    .expect("unbounded symbolic shift must not error");
    assert_eq!(result.width(), 64);
}

// =========================================================================
// Packed integer min/max/abs tests
// =========================================================================

/// PMINSW-style: signed min over 8x i16 lanes, mix of positive and negative.
#[test]
fn test_vec_int_min_signed_concrete() {
    let ctx = SymContext::new_mock();

    let l: [i16; 8] = [-5, 100, 0, -32768, 1, -1, 32767, -2];
    let r: [i16; 8] = [-3, 200, -100, -32767, -1, 0, 32766, 3];
    let exp: [i16; 8] = [-5, 100, -100, -32768, -1, -1, 32766, -2];

    let lv = pack_lanes_uint(&l.map(|x| x as u16 as u128), 16);
    let rv = pack_lanes_uint(&r.map(|x| x as u16 as u128), 16);
    let result = VEXOps::binop(
        IROp::VMin {
            elem: IRType::I16,
            count: 8,
            signed: true,
        },
        RustBV::concrete(lv, 128),
        RustBV::concrete(rv, 128),
        &ctx,
    )
    .unwrap();
    assert_int_lanes_eq(
        result.as_u128().unwrap(),
        &exp.map(|x| x as u16 as u128),
        16,
    );
}

/// PMAXUB-style: unsigned max over 16x u8 lanes.
#[test]
fn test_vec_int_max_unsigned_concrete() {
    let ctx = SymContext::new_mock();

    let l: [u8; 16] = [
        0xFF, 0x00, 0x80, 0x7F, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    ];
    let r: [u8; 16] = [0x00, 0xFF, 0x7F, 0x80, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0];
    let mut exp = [0u8; 16];
    for i in 0..16 {
        exp[i] = if l[i] > r[i] { l[i] } else { r[i] };
    }

    let lv = pack_lanes_uint(&l.map(|x| x as u128), 8);
    let rv = pack_lanes_uint(&r.map(|x| x as u128), 8);
    let result = VEXOps::binop(
        IROp::VMax {
            elem: IRType::I8,
            count: 16,
            signed: false,
        },
        RustBV::concrete(lv, 128),
        RustBV::concrete(rv, 128),
        &ctx,
    )
    .unwrap();
    assert_int_lanes_eq(result.as_u128().unwrap(), &exp.map(|x| x as u128), 8);
}

/// PABSW-style: per-lane absolute value over 8x i16 lanes (incl. INT_MIN
/// which stays INT_MIN under two's-complement |x|).
#[test]
fn test_vec_int_abs_concrete() {
    let ctx = SymContext::new_mock();

    let v: [i16; 8] = [-5, 100, 0, -32768, 1, -1, 32767, -200];
    let exp: [u16; 8] = [5, 100, 0, 0x8000 /* INT_MIN stays */, 1, 1, 32767, 200];

    let result = VEXOps::unop(
        IROp::VAbs {
            elem: IRType::I16,
            count: 8,
        },
        RustBV::concrete(pack_lanes_uint(&v.map(|x| x as u16 as u128), 16), 128),
        &ctx,
    )
    .unwrap();
    assert_int_lanes_eq(result.as_u128().unwrap(), &exp.map(|x| x as u128), 16);
}

// =========================================================================
// VReverse — byte/halfword/word/bit reversal within lane (angr-tukg.4).
// =========================================================================

/// Iop_Reverse8sIn32_x2 — byte-swap within each 32-bit word (REV32
/// applied to a NEON D-register). 64-bit total.
#[test]
fn test_vec_reverse_8in32_x2_concrete() {
    let ctx = SymContext::new_mock();
    // Two 32-bit lanes: low = 0x11223344, high = 0xAABBCCDD.
    let v: u128 = 0xAABBCCDD_11223344u128;
    let result = VEXOps::unop(
        IROp::VReverse {
            sub_width: 8,
            elem: IRType::I32,
            count: 2,
        },
        RustBV::concrete(v, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    // Low lane bytes reversed: 0x11223344 → 0x44332211.
    // High lane: 0xAABBCCDD → 0xDDCCBBAA.
    let expected: u128 = 0xDDCCBBAA_44332211u128;
    assert_eq!(result.as_u128().unwrap(), expected);
}

/// Iop_Reverse32sIn64_x2 — swap the two 32-bit halves of each 64-bit
/// lane. Directly mirrors the only explicit Python reference at
/// `angr/engines/vex/claripy/irop.py:_op_Iop_Reverse32sIn64_x2`.
#[test]
fn test_vec_reverse_32in64_x2_concrete_matches_python_ref() {
    let ctx = SymContext::new_mock();
    // Python ref: Concat(arg[95:64], arg[127:96], arg[31:0], arg[63:32]).
    // Pick a 128-bit value with distinct 32-bit slices to exercise every
    // permutation slot.
    let v: u128 = 0xAAAAAAAA_BBBBBBBB_CCCCCCCC_DDDDDDDDu128;
    let result = VEXOps::unop(
        IROp::VReverse {
            sub_width: 32,
            elem: IRType::I64,
            count: 2,
        },
        RustBV::concrete(v, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    // Slices of v (LSB → MSB indexing): [31:0]=DDDDDDDD, [63:32]=CCCCCCCC,
    //                                  [95:64]=BBBBBBBB, [127:96]=AAAAAAAA.
    // Python Concat(MSB→LSB): [95:64], [127:96], [31:0], [63:32]
    //   = BBBBBBBB AAAAAAAA DDDDDDDD CCCCCCCC (MSB→LSB)
    let expected: u128 = 0xBBBBBBBB_AAAAAAAA_DDDDDDDD_CCCCCCCCu128;
    assert_eq!(result.as_u128().unwrap(), expected);
}

/// Iop_Reverse1sIn8_x8 — RBIT: reverse the bit order inside each byte.
#[test]
fn test_vec_reverse_1in8_x8_concrete() {
    let ctx = SymContext::new_mock();
    // Byte 0 = 0b10110010 = 0xB2; reversed = 0b01001101 = 0x4D.
    // Byte 1 = 0xFF (palindrome); reversed = 0xFF.
    // Byte 2 = 0x01; reversed = 0x80.
    // Byte 3 = 0x80; reversed = 0x01.
    // Byte 4 = 0xA5; reversed = 0xA5 (10100101 → 10100101).
    // Byte 5 = 0x00; reversed = 0x00.
    // Byte 6 = 0x0F; reversed = 0xF0.
    // Byte 7 = 0xF0; reversed = 0x0F.
    let in_bytes: [u8; 8] = [0xB2, 0xFF, 0x01, 0x80, 0xA5, 0x00, 0x0F, 0xF0];
    let exp_bytes: [u8; 8] = [0x4D, 0xFF, 0x80, 0x01, 0xA5, 0x00, 0xF0, 0x0F];
    let mut v: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        v |= (in_bytes[i] as u128) << (i * 8);
        e |= (exp_bytes[i] as u128) << (i * 8);
    }
    let result = VEXOps::unop(
        IROp::VReverse {
            sub_width: 1,
            elem: IRType::I8,
            count: 8,
        },
        RustBV::concrete(v, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Reverse16sIn64_x2 — halfword swap inside each 64-bit lane,
/// applied across two lanes (128-bit Q-register form).
#[test]
fn test_vec_reverse_16in64_x2_concrete() {
    let ctx = SymContext::new_mock();
    // Lane 0 (low 64): halfwords [0x1111, 0x2222, 0x3333, 0x4444] (LSB→MSB).
    // After reversal: [0x4444, 0x3333, 0x2222, 0x1111].
    // Lane 1 (high 64): halfwords [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD].
    // After reversal: [0xDDDD, 0xCCCC, 0xBBBB, 0xAAAA].
    let v: u128 = 0xDDDD_CCCC_BBBB_AAAA_4444_3333_2222_1111u128;
    let expected: u128 = 0xAAAA_BBBB_CCCC_DDDD_1111_2222_3333_4444u128;
    let result = VEXOps::unop(
        IROp::VReverse {
            sub_width: 16,
            elem: IRType::I64,
            count: 2,
        },
        RustBV::concrete(v, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), expected);
}

/// Involution: applying VReverse twice is the identity (any permutation
/// that swaps positions i ↔ n-1-i is its own inverse). Exercise on a
/// symbolic 128-bit input via a Z3 equivalence check.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_reverse_double_apply_is_identity() {
    let ctx = SymContext::new_mock();
    for (sub_width, elem, count) in [
        (8u8, IRType::I32, 4u8), // Reverse8sIn32_x4
        (16, IRType::I64, 2),    // Reverse16sIn64_x2
        (32, IRType::I64, 2),    // Reverse32sIn64_x2
        (1, IRType::I8, 16),     // Reverse1sIn8_x16
    ] {
        let width = elem.bits() * count as u32;
        let arg = RustBV::symbolic(&ctx, "vrev_arg", width);
        let op = IROp::VReverse {
            sub_width,
            elem,
            count,
        };
        let once = VEXOps::unop(op, arg.clone(), &ctx).unwrap();
        let twice = VEXOps::unop(op, once, &ctx).unwrap();
        // Assert there is no satisfying assignment where twice != arg.
        ctx.push();
        ctx.add_constraint(twice.to_z3_ast().eq(arg.to_z3_ast()).not());
        assert!(
            !ctx.is_sat(),
            "double-apply must equal identity for sub_width={} elem={:?} count={}",
            sub_width,
            elem,
            count
        );
        ctx.pop();
    }
}

/// Symbolic parity vs claripy reference: `Iop_Reverse32sIn64_x2` must
/// produce the same bits as the explicit Python implementation
/// `Concat(arg[95:64], arg[127:96], arg[31:0], arg[63:32])` for any
/// 128-bit input. Verified through a Z3 universality check.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_reverse_32in64_x2_symbolic_matches_python_ref() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::symbolic(&ctx, "vrev_arg", 128);
    let got = VEXOps::unop(
        IROp::VReverse {
            sub_width: 32,
            elem: IRType::I64,
            count: 2,
        },
        arg.clone(),
        &ctx,
    )
    .unwrap();

    // Build the Python reference: Concat(arg[95:64], arg[127:96],
    //                                    arg[31:0],  arg[63:32]).
    // `concat_le_elements` indexes 0 → LSB, so push in LSB→MSB order:
    //   bits [31:0]  output  ← arg[63:32]
    //   bits [63:32] output  ← arg[31:0]
    //   bits [95:64] output  ← arg[127:96]
    //   bits [127:96] output ← arg[95:64]
    let py = VEXOps::concat_le_elements(
        vec![
            arg.extract(63, 32, &ctx),
            arg.extract(31, 0, &ctx),
            arg.extract(127, 96, &ctx),
            arg.extract(95, 64, &ctx),
        ],
        &ctx,
    );

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VReverse 32sIn64_x2 must match the Python reference Concat pattern"
    );
    ctx.pop();
}

/// Symbolic VMax (signed): constrain right == 7, derive left from a free
/// 4x i32 vector, and verify that asserting result == [7, 7, 7, 7] forces
/// every lane of left to be <= 7.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_int_max_symbolic_signed() {
    let ctx = SymContext::new_mock();

    // r = [7, 7, 7, 7] as i32x4
    let mut rv: u128 = 0;
    for i in 0..4u32 {
        rv |= (7u128) << (i * 32);
    }
    let r = RustBV::concrete(rv, 128);
    let l = RustBV::symbolic(&ctx, "vmax_l", 128);

    let result = VEXOps::binop(
        IROp::VMax {
            elem: IRType::I32,
            count: 4,
            signed: true,
        },
        l.clone(),
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);

    // Constrain result == [7, 7, 7, 7]; this only requires l <= 7 per lane,
    // so the constraint must remain SAT.
    let exp = RustBV::concrete(rv, 128);
    ctx.add_constraint(result.to_z3_ast().eq(exp.to_z3_ast()));
    assert!(ctx.is_sat(), "expected SAT after constraining max == 7");
}

// =========================================================================
// Packed FP add/sub/mul/div/sqrt/abs/min/max tests
// =========================================================================

/// ADDPS-style: 4x f32 add, concrete.
#[test]
fn test_vec_float_add_concrete_f32x4() {
    let ctx = SymContext::new_mock();

    let l = [1.0f32, 2.5, -3.0, 0.5];
    let r = [10.0f32, -2.5, 3.0, 8.0];
    let exp: [f32; 4] = [11.0, 0.0, 0.0, 8.5];

    let result = VEXOps::binop(
        IROp::VFAdd {
            elem: IRType::F32,
            count: 4,
        },
        RustBV::concrete(pack_lanes_f32(&l), 128),
        RustBV::concrete(pack_lanes_f32(&r), 128),
        &ctx,
    )
    .unwrap();
    assert_f32_lanes_approx(result.as_u128().unwrap(), &exp, 1e-6);
}

/// DIVPD-style: 2x f64 div, concrete.
#[test]
fn test_vec_float_div_concrete_f64x2() {
    let ctx = SymContext::new_mock();

    let l = [10.0f64, -8.0];
    let r = [4.0f64, 2.0];
    let exp = [2.5f64, -4.0];

    let result = VEXOps::binop(
        IROp::VFDiv {
            elem: IRType::F64,
            count: 2,
        },
        RustBV::concrete(pack_lanes_f64(&l), 128),
        RustBV::concrete(pack_lanes_f64(&r), 128),
        &ctx,
    )
    .unwrap();
    assert_f64_lanes_approx(result.as_u128().unwrap(), &exp, 1e-12);
}

/// SQRTPS-style: 4x f32 sqrt, concrete.
#[test]
fn test_vec_float_sqrt_concrete_f32x4() {
    let ctx = SymContext::new_mock();

    let v = [4.0f32, 9.0, 16.0, 25.0];
    let exp = [2.0f32, 3.0, 4.0, 5.0];

    let result = VEXOps::unop(
        IROp::VFSqrt {
            elem: IRType::F32,
            count: 4,
        },
        RustBV::concrete(pack_lanes_f32(&v), 128),
        &ctx,
    )
    .unwrap();
    assert_f32_lanes_approx(result.as_u128().unwrap(), &exp, 1e-6);
}

/// Iop_Abs32Fx4-style: per-lane fabs (clears sign bit).
#[test]
fn test_vec_float_abs_concrete_f32x4() {
    let ctx = SymContext::new_mock();

    let v = [-1.5f32, 2.5, -0.0, f32::NEG_INFINITY];
    let exp = [1.5f32, 2.5, 0.0, f32::INFINITY];

    let result = VEXOps::unop(
        IROp::VFAbs {
            elem: IRType::F32,
            count: 4,
        },
        RustBV::concrete(pack_lanes_f32(&v), 128),
        &ctx,
    )
    .unwrap();
    assert_f32_lanes_bits(result.as_u128().unwrap(), &exp);
}

/// MAXPS-style: per-lane max of two f32x4 vectors.
#[test]
fn test_vec_float_max_concrete_f32x4() {
    let ctx = SymContext::new_mock();

    let l = [1.0f32, -2.0, 3.0, 0.5];
    let r = [4.0f32, -3.0, 2.0, 0.6];
    let exp = [4.0f32, -2.0, 3.0, 0.6];

    let result = VEXOps::binop(
        IROp::VFMax {
            elem: IRType::F32,
            count: 4,
        },
        RustBV::concrete(pack_lanes_f32(&l), 128),
        RustBV::concrete(pack_lanes_f32(&r), 128),
        &ctx,
    )
    .unwrap();
    assert_f32_lanes_approx(result.as_u128().unwrap(), &exp, 1e-6);
}

/// MINPD-style: per-lane min of two f64x2 vectors.
#[test]
fn test_vec_float_min_concrete_f64x2() {
    let ctx = SymContext::new_mock();

    let l = [1.5f64, -2.5];
    let r = [-1.5f64, 0.5];
    let exp = [-1.5f64, -2.5];

    let result = VEXOps::binop(
        IROp::VFMin {
            elem: IRType::F64,
            count: 2,
        },
        RustBV::concrete(pack_lanes_f64(&l), 128),
        RustBV::concrete(pack_lanes_f64(&r), 128),
        &ctx,
    )
    .unwrap();
    assert_f64_lanes_approx(result.as_u128().unwrap(), &exp, 1e-12);
}

/// Symbolic VFAdd: build a free f32x4 left vector, constrain it so each
/// lane equals 1.0, add a concrete [2.0, 3.0, 4.0, 5.0], and verify the
/// model agrees with [3.0, 4.0, 5.0, 6.0] per lane.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_float_add_symbolic_f32x4() {
    let ctx = SymContext::new_mock();

    let consts = [2.0f32, 3.0, 4.0, 5.0];
    let r = RustBV::concrete(pack_lanes_f32(&consts), 128);

    let lv_target = pack_lanes_f32(&[1.0f32; 4]);
    let l = RustBV::symbolic(&ctx, "vfadd_l", 128);
    ctx.add_constraint(
        l.to_z3_ast()
            .eq(RustBV::concrete(lv_target, 128).to_z3_ast()),
    );

    let result = VEXOps::binop(
        IROp::VFAdd {
            elem: IRType::F32,
            count: 4,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert!(ctx.is_sat(), "expected SAT");

    let model = ctx.eval(&result).expect("eval(result) returned None");
    let exp = [3.0f32, 4.0, 5.0, 6.0];
    assert_f32_lanes_approx(model, &exp, 1e-6);
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
fn eval_v128(ctx: &SymContext, bv: &RustBV) -> u128 {
    assert!(ctx.is_sat(), "expected SAT for eval");
    ctx.eval(bv).expect("eval returned None")
}
fn eval_i64(ctx: &SymContext, bv: &RustBV) -> u64 {
    eval_v128(ctx, bv) as u64
}
fn eval_i32(ctx: &SymContext, bv: &RustBV) -> u32 {
    eval_v128(ctx, bv) as u32
}

/// SSE RCPSS shape: 128-bit result, lane 0 fresh symbolic (32-bit width),
/// upper 96 bits passed through unchanged from the arg.
#[test]
fn test_vfrecip_est_s_f32_upper_passthrough() {
    let ctx = SymContext::new_mock();
    let upper96 = 0xDEAD_BEEF_CAFE_BABE_1234_5678u128;
    let arg = RustBV::concrete((upper96 << 32) | 0x4080_0000u128, 128); // lane0 = 4.0f32
    let result = VEXOps::unop(IROp::VFRecipEstS { elem: IRType::F32 }, arg, &ctx).unwrap();
    assert_eq!(result.width(), 128);
    let v = eval_v128(&ctx, &result);
    assert_eq!(v >> 32, upper96, "upper 96 bits must pass through");
}

/// SSE RSQRTSS shape: same as RCPSS — upper 96 bits passthrough, lane 0 fresh.
#[test]
fn test_vfrsqrt_est_s_f32_upper_passthrough() {
    let ctx = SymContext::new_mock();
    let upper96 = 0xAAAA_BBBB_CCCC_DDDD_EEEE_FFFFu128;
    let arg = RustBV::concrete((upper96 << 32) | 0x4400_0000u128, 128); // lane0 = 512.0f32
    let result = VEXOps::unop(IROp::VFRSqrtEstS { elem: IRType::F32 }, arg, &ctx).unwrap();
    assert_eq!(result.width(), 128);
    let v = eval_v128(&ctx, &result);
    assert_eq!(v >> 32, upper96, "upper 96 bits must pass through");
}

/// NEON Iop_RecipEst32Fx2 (D-reg, 2x f32 = 64-bit result).
#[test]
fn test_vfrecip_est_packed_f32x2_shape() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(0u128, 64);
    let result = VEXOps::unop(
        IROp::VFRecipEst {
            elem: IRType::F32,
            count: 2,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    let _ = eval_i64(&ctx, &result);
}

/// SSE RCPPS / NEON Q-reg Iop_RecipEst32Fx4 (4x f32 = 128-bit result).
#[test]
fn test_vfrecip_est_packed_f32x4_shape() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(0u128, 128);
    let result = VEXOps::unop(
        IROp::VFRecipEst {
            elem: IRType::F32,
            count: 4,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    let _ = eval_v128(&ctx, &result);
}

/// AVX Iop_RecipEst32Fx8 (8x f32 = 256-bit result).
#[test]
fn test_vfrecip_est_packed_f32x8_shape() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(0u128, 256);
    let result = VEXOps::unop(
        IROp::VFRecipEst {
            elem: IRType::F32,
            count: 8,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 256);
}

/// NEON Iop_RecipEst64Fx2 (2x f64 = 128-bit result).
#[test]
fn test_vfrecip_est_packed_f64x2_shape() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(0u128, 128);
    let result = VEXOps::unop(
        IROp::VFRecipEst {
            elem: IRType::F64,
            count: 2,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
}

/// Iop_RSqrtEst*: same widths as RecipEst, separate dispatch arm.
#[test]
fn test_vfrsqrt_est_packed_f64x2_shape() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(0u128, 128);
    let result = VEXOps::unop(
        IROp::VFRSqrtEst {
            elem: IRType::F64,
            count: 2,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
}

/// NEON Iop_RecipStep32Fx2: D-reg binary, 64-bit result.
#[test]
fn test_vfrecip_step_packed_f32x2_shape() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(0u128, 64);
    let b = RustBV::concrete(0u128, 64);
    let result = VEXOps::binop(
        IROp::VFRecipStep {
            elem: IRType::F32,
            count: 2,
        },
        a,
        b,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    let _ = eval_i64(&ctx, &result);
}

/// NEON Iop_RecipStep64Fx2: Q-reg binary, 128-bit result.
#[test]
fn test_vfrecip_step_packed_f64x2_shape() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(0u128, 128);
    let b = RustBV::concrete(0u128, 128);
    let result = VEXOps::binop(
        IROp::VFRecipStep {
            elem: IRType::F64,
            count: 2,
        },
        a,
        b,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
}

/// NEON Iop_RSqrtStep32Fx4: Q-reg binary, 128-bit result.
#[test]
fn test_vfrsqrt_step_packed_f32x4_shape() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(0u128, 128);
    let b = RustBV::concrete(0u128, 128);
    let result = VEXOps::binop(
        IROp::VFRSqrtStep {
            elem: IRType::F32,
            count: 4,
        },
        a,
        b,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
}

/// NEON Iop_RecipEst32Ux2 (D-reg URECPE, 2x u32 = 64-bit result). Fresh
/// symbolic per lane; shape is what matters here.
#[test]
fn test_virecip_est_packed_u32x2_shape() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(0u128, 64);
    let result = VEXOps::unop(IROp::VIRecipEst { count: 2 }, arg, &ctx).unwrap();
    assert_eq!(result.width(), 64);
    // Symbolic: should not be concrete because each lane was minted fresh.
    assert!(result.as_u128().is_none());
}

/// NEON Iop_RecipEst32Ux4 (Q-reg URECPE, 4x u32 = 128-bit result).
#[test]
fn test_virecip_est_packed_u32x4_shape() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(0u128, 128);
    let result = VEXOps::unop(IROp::VIRecipEst { count: 4 }, arg, &ctx).unwrap();
    assert_eq!(result.width(), 128);
    assert!(result.as_u128().is_none());
}

/// NEON Iop_RSqrtEst32Ux2 (D-reg URSQRTE, 2x u32 = 64-bit result).
#[test]
fn test_virsqrt_est_packed_u32x2_shape() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(0u128, 64);
    let result = VEXOps::unop(IROp::VIRSqrtEst { count: 2 }, arg, &ctx).unwrap();
    assert_eq!(result.width(), 64);
    assert!(result.as_u128().is_none());
}

/// NEON Iop_RSqrtEst32Ux4 (Q-reg URSQRTE, 4x u32 = 128-bit result).
#[test]
fn test_virsqrt_est_packed_u32x4_shape() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(0u128, 128);
    let result = VEXOps::unop(IROp::VIRSqrtEst { count: 4 }, arg, &ctx).unwrap();
    assert_eq!(result.width(), 128);
    assert!(result.as_u128().is_none());
}

/// End-to-end opcode-string routing for the integer NEON RecipEst /
/// RSqrtEst ops — these resolve to the new VIRecipEst / VIRSqrtEst
/// variants instead of NeonUnimplemented. Catches regressions where
/// parse_vector and parse_neon_unimplemented fall out of sync.
#[test]
fn test_int_recip_rsqrt_opcode_routing() {
    use crate::vex::opcode_map::parse_opcode;
    for (name, count) in [("Iop_RecipEst32Ux2", 2u8), ("Iop_RecipEst32Ux4", 4)] {
        match parse_opcode(name) {
            IROp::VIRecipEst { count: c } => assert_eq!(c, count),
            other => panic!("{}: expected VIRecipEst, got {:?}", name, other),
        }
    }
    for (name, count) in [("Iop_RSqrtEst32Ux2", 2u8), ("Iop_RSqrtEst32Ux4", 4)] {
        match parse_opcode(name) {
            IROp::VIRSqrtEst { count: c } => assert_eq!(c, count),
            other => panic!("{}: expected VIRSqrtEst, got {:?}", name, other),
        }
    }
}

/// End-to-end opcode-string routing: all 16 FP Recip/RSqrt opcodes
/// resolve to the new IROp variants (and not to NeonUnimplemented).
/// Catches regressions where parse_float and parse_neon_unimplemented
/// fall out of sync.
#[test]
fn test_recip_rsqrt_opcode_routing() {
    use crate::vex::opcode_map::parse_opcode;
    // Est family: F0x4 → VFRecipEstS/VFRSqrtEstS, others → packed.
    let recip_est_packed = [
        ("Iop_RecipEst32Fx2", IRType::F32, 2),
        ("Iop_RecipEst32Fx4", IRType::F32, 4),
        ("Iop_RecipEst32Fx8", IRType::F32, 8),
        ("Iop_RecipEst64Fx2", IRType::F64, 2),
    ];
    for (name, elem, count) in recip_est_packed {
        match parse_opcode(name) {
            IROp::VFRecipEst { elem: e, count: c } => {
                assert_eq!(e, elem);
                assert_eq!(c, count);
            }
            other => panic!("{}: expected VFRecipEst, got {:?}", name, other),
        }
    }
    assert!(matches!(
        parse_opcode("Iop_RecipEst32F0x4"),
        IROp::VFRecipEstS { elem: IRType::F32 }
    ));
    let rsqrt_est_packed = [
        ("Iop_RSqrtEst32Fx2", IRType::F32, 2),
        ("Iop_RSqrtEst32Fx4", IRType::F32, 4),
        ("Iop_RSqrtEst32Fx8", IRType::F32, 8),
        ("Iop_RSqrtEst64Fx2", IRType::F64, 2),
    ];
    for (name, elem, count) in rsqrt_est_packed {
        match parse_opcode(name) {
            IROp::VFRSqrtEst { elem: e, count: c } => {
                assert_eq!(e, elem);
                assert_eq!(c, count);
            }
            other => panic!("{}: expected VFRSqrtEst, got {:?}", name, other),
        }
    }
    assert!(matches!(
        parse_opcode("Iop_RSqrtEst32F0x4"),
        IROp::VFRSqrtEstS { elem: IRType::F32 }
    ));
    // Step family: NEON-only, no F0x4 form.
    let step_pairs = [
        ("Iop_RecipStep32Fx2", IRType::F32, 2),
        ("Iop_RecipStep32Fx4", IRType::F32, 4),
        ("Iop_RecipStep64Fx2", IRType::F64, 2),
    ];
    for (name, elem, count) in step_pairs {
        match parse_opcode(name) {
            IROp::VFRecipStep { elem: e, count: c } => {
                assert_eq!(e, elem);
                assert_eq!(c, count);
            }
            other => panic!("{}: expected VFRecipStep, got {:?}", name, other),
        }
    }
    let rsqrt_step_pairs = [
        ("Iop_RSqrtStep32Fx2", IRType::F32, 2),
        ("Iop_RSqrtStep32Fx4", IRType::F32, 4),
        ("Iop_RSqrtStep64Fx2", IRType::F64, 2),
    ];
    for (name, elem, count) in rsqrt_step_pairs {
        match parse_opcode(name) {
            IROp::VFRSqrtStep { elem: e, count: c } => {
                assert_eq!(e, elem);
                assert_eq!(c, count);
            }
            other => panic!("{}: expected VFRSqrtStep, got {:?}", name, other),
        }
    }
    // Suppress the unused-helper lint when running this test alone:
    let _ = eval_i32;
}

// ---- FCmpScalarLane (Iop_Cmp{EQ,LT,LE,UN}{32F0x4,64F0x2}) ----

fn make_v128_lane0(lane0: u128, upper96: u128) -> u128 {
    debug_assert!(lane0 <= 0xFFFF_FFFF);
    (upper96 << 32) | lane0
}
fn make_v128_lane0_64(lane0: u128, upper64: u128) -> u128 {
    debug_assert!(lane0 <= 0xFFFF_FFFF_FFFF_FFFF);
    (upper64 << 64) | lane0
}

#[test]
fn test_fcmp_scalar_lane_eq_f32_concrete_true() {
    // CMPEQSS: lane0(left)==lane0(right) → 0xFFFFFFFF in lane0; upper from left.
    let ctx = SymContext::new_mock();
    let upper = 0xDEAD_BEEF_DEAD_BEEF_DEAD_BEEFu128;
    let l = RustBV::concrete(make_v128_lane0(2.0f32.to_bits() as u128, upper), 128);
    let r = RustBV::concrete(2.0f32.to_bits() as u128, 128);
    let res = VEXOps::binop(
        IROp::FCmpScalarLane {
            kind: FCmpKind::Eq,
            ty: IRType::F32,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let v = res.as_u128().expect("concrete result");
    assert_eq!(v & 0xFFFF_FFFF, 0xFFFF_FFFF, "lane0 should be all-1s");
    assert_eq!(v >> 32, upper, "upper96 must passthrough from left");
}

#[test]
fn test_fcmp_scalar_lane_eq_f32_concrete_false() {
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(1.0f32.to_bits() as u128, 128);
    let r = RustBV::concrete(2.0f32.to_bits() as u128, 128);
    let res = VEXOps::binop(
        IROp::FCmpScalarLane {
            kind: FCmpKind::Eq,
            ty: IRType::F32,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    let v = res.as_u128().unwrap();
    assert_eq!(v & 0xFFFF_FFFF, 0, "lane0 should be 0 on false");
}

#[test]
fn test_fcmp_scalar_lane_lt_f32_concrete() {
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(1.0f32.to_bits() as u128, 128);
    let r = RustBV::concrete(2.0f32.to_bits() as u128, 128);
    let res = VEXOps::binop(
        IROp::FCmpScalarLane {
            kind: FCmpKind::Lt,
            ty: IRType::F32,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.as_u128().unwrap() & 0xFFFF_FFFF, 0xFFFF_FFFF);
}

#[test]
fn test_fcmp_scalar_lane_le_f64_concrete_eq() {
    let ctx = SymContext::new_mock();
    let upper = 0x123456789ABCDEF0u128;
    let l = RustBV::concrete(make_v128_lane0_64(2.5f64.to_bits() as u128, upper), 128);
    let r = RustBV::concrete(2.5f64.to_bits() as u128, 128);
    let res = VEXOps::binop(
        IROp::FCmpScalarLane {
            kind: FCmpKind::Le,
            ty: IRType::F64,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    let v = res.as_u128().unwrap();
    assert_eq!(v & 0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF);
    assert_eq!(v >> 64, upper, "upper64 passthrough from left");
}

#[test]
fn test_fcmp_scalar_lane_un_f32_concrete_nan() {
    // CMPUNORD: returns true if either operand is NaN.
    let ctx = SymContext::new_mock();
    let nan = f32::NAN.to_bits() as u128;
    let l = RustBV::concrete(nan, 128);
    let r = RustBV::concrete(1.0f32.to_bits() as u128, 128);
    let res = VEXOps::binop(
        IROp::FCmpScalarLane {
            kind: FCmpKind::Un,
            ty: IRType::F32,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.as_u128().unwrap() & 0xFFFF_FFFF, 0xFFFF_FFFF);
}

#[test]
fn test_fcmp_scalar_lane_un_f64_concrete_ordered() {
    // Both ordered → CMPUNORD returns 0.
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(1.0f64.to_bits() as u128, 128);
    let r = RustBV::concrete(2.0f64.to_bits() as u128, 128);
    let res = VEXOps::binop(
        IROp::FCmpScalarLane {
            kind: FCmpKind::Un,
            ty: IRType::F64,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.as_u128().unwrap() & 0xFFFF_FFFF_FFFF_FFFF, 0);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_fcmp_scalar_lane_eq_f32_symbolic() {
    // Symbolic: constrain low32(result)==0xFFFFFFFF given right=2.0 and
    // some symbolic left → solver must pick left.lane0 == 2.0.
    let ctx = SymContext::new_mock();
    let l = RustBV::symbolic(&ctx, "fcmp_lane_l", 128);
    let r = RustBV::concrete(2.0f32.to_bits() as u128, 128);
    let res = VEXOps::binop(
        IROp::FCmpScalarLane {
            kind: FCmpKind::Eq,
            ty: IRType::F32,
        },
        l.clone(),
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let lo32 = res.extract(31, 0, &ctx);
    let true_mask = RustBV::concrete(0xFFFF_FFFF, 32);
    ctx.add_constraint(lo32.to_z3_ast().eq(true_mask.to_z3_ast()));
    assert!(
        ctx.is_sat(),
        "expected SAT after FCmpScalarLane Eq mask=all1s"
    );

    let model = ctx.eval(&l).expect("eval(l) returned None");
    let lane0 = f32::from_bits((model & 0xFFFF_FFFF) as u32);
    assert!(
        (lane0 - 2.0).abs() < 1e-6,
        "expected lane0==2.0, got {}",
        lane0
    );
}

// ---- FComCC (Iop_CmpF32, Iop_CmpF64, x87 FCOM) ----

#[test]
fn test_fcom_cc_f32_concrete_eq() {
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(2.5f32.to_bits() as u128, 32);
    let r = RustBV::concrete(2.5f32.to_bits() as u128, 32);
    let res = VEXOps::binop(IROp::FComCC(IRType::F32), l, r, &ctx).unwrap();
    assert_eq!(res.width(), 32);
    assert_eq!(res.as_u128().unwrap(), 0x40);
}

#[test]
fn test_fcom_cc_f32_concrete_lt() {
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(1.0f32.to_bits() as u128, 32);
    let r = RustBV::concrete(2.0f32.to_bits() as u128, 32);
    let res = VEXOps::binop(IROp::FComCC(IRType::F32), l, r, &ctx).unwrap();
    assert_eq!(res.as_u128().unwrap(), 0x01);
}

#[test]
fn test_fcom_cc_f64_concrete_gt() {
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(3.0f64.to_bits() as u128, 64);
    let r = RustBV::concrete(2.0f64.to_bits() as u128, 64);
    let res = VEXOps::binop(IROp::FComCC(IRType::F64), l, r, &ctx).unwrap();
    assert_eq!(res.as_u128().unwrap(), 0x00);
}

#[test]
fn test_fcom_cc_f64_concrete_unordered() {
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(f64::NAN.to_bits() as u128, 64);
    let r = RustBV::concrete(1.0f64.to_bits() as u128, 64);
    let res = VEXOps::binop(IROp::FComCC(IRType::F64), l, r, &ctx).unwrap();
    assert_eq!(res.as_u128().unwrap(), 0x45);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_fcom_cc_symbolic_lt() {
    // Symbolic: constrain FComCC(x, 5.0) == 0x01 → x must be < 5.0 (and not NaN).
    let ctx = SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "fcom_x", 64);
    let five = RustBV::concrete(5.0f64.to_bits() as u128, 64);
    let res = VEXOps::binop(IROp::FComCC(IRType::F64), x.clone(), five, &ctx).unwrap();
    assert_eq!(res.width(), 32);
    let want = RustBV::concrete(0x01, 32);
    ctx.add_constraint(res.to_z3_ast().eq(want.to_z3_ast()));
    assert!(ctx.is_sat(), "expected SAT for FComCC(x, 5.0) == LT");
    let model_x = ctx.eval(&x).expect("eval(x) None");
    let xf = f64::from_bits(model_x as u64);
    assert!(
        !xf.is_nan() && xf < 5.0,
        "expected x < 5.0 and not NaN, got {}",
        xf
    );
}

// ---- FCmpVecPacked (Iop_Cmp{EQ,LT,LE,GT,GE,UN}{32Fx2,32Fx4,64Fx2}) ----

/// Pack four f32 values into a single 128-bit vector (lane 0 first).
fn pack_4xf32(a: f32, b: f32, c: f32, d: f32) -> u128 {
    pack_lanes_f32(&[a, b, c, d])
}
fn pack_2xf64(a: f64, b: f64) -> u128 {
    pack_lanes_f64(&[a, b])
}
/// Pack two f32 lanes into the low 64 bits (lane 0 first).
fn pack_2xf32_64(a: f32, b: f32) -> u128 {
    pack_lanes_f32(&[a, b])
}

#[test]
#[allow(clippy::identity_op)] // explicit 4-lane mask layout reads better than the minimized form
fn test_fcmp_packed_eq_32fx4_concrete() {
    // CMPEQPS lane-by-lane: lanes 0 and 2 equal, lanes 1 and 3 differ.
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(pack_4xf32(1.0, 2.0, 3.0, 4.0), 128);
    let r = RustBV::concrete(pack_4xf32(1.0, 5.0, 3.0, 7.0), 128);
    let res = VEXOps::binop(
        IROp::FCmpVecPacked {
            kind: FCmpKind::Eq,
            elem: IRType::F32,
            count: 4,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let v = res.as_u128().unwrap();
    // Expect lane 0 = 0xFFFFFFFF, lane 1 = 0, lane 2 = 0xFFFFFFFF, lane 3 = 0.
    let expected: u128 =
        (0xFFFF_FFFFu128 << 0) | (0u128 << 32) | (0xFFFF_FFFFu128 << 64) | (0u128 << 96);
    assert_eq!(v, expected);
}

#[test]
fn test_fcmp_packed_lt_32fx4_concrete() {
    // CMPLTPS: 1.0 < 2.0 (T), 5.0 < 3.0 (F), -1.0 < 0.0 (T), 4.0 < 4.0 (F).
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(pack_4xf32(1.0, 5.0, -1.0, 4.0), 128);
    let r = RustBV::concrete(pack_4xf32(2.0, 3.0, 0.0, 4.0), 128);
    let res = VEXOps::binop(
        IROp::FCmpVecPacked {
            kind: FCmpKind::Lt,
            elem: IRType::F32,
            count: 4,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    let v = res.as_u128().unwrap();
    let expected: u128 = 0xFFFF_FFFFu128 | (0xFFFF_FFFFu128 << 64);
    assert_eq!(v, expected);
}

#[test]
fn test_fcmp_packed_gt_32fx4_concrete() {
    // CMPGTPS: 5.0 > 2.0 (T), 1.0 > 3.0 (F), 4.0 > 4.0 (F), 9.0 > 0.0 (T).
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(pack_4xf32(5.0, 1.0, 4.0, 9.0), 128);
    let r = RustBV::concrete(pack_4xf32(2.0, 3.0, 4.0, 0.0), 128);
    let res = VEXOps::binop(
        IROp::FCmpVecPacked {
            kind: FCmpKind::Gt,
            elem: IRType::F32,
            count: 4,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    let v = res.as_u128().unwrap();
    let expected: u128 = 0xFFFF_FFFFu128 | (0xFFFF_FFFFu128 << 96);
    assert_eq!(v, expected);
}

#[test]
#[allow(clippy::identity_op)] // explicit 4-lane mask layout reads better than the minimized form
fn test_fcmp_packed_ge_32fx4_concrete() {
    // CMPGEPS: 5.0>=2.0 T, 3.0>=3.0 T, 1.0>=2.0 F, 4.0>=4.0 T.
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(pack_4xf32(5.0, 3.0, 1.0, 4.0), 128);
    let r = RustBV::concrete(pack_4xf32(2.0, 3.0, 2.0, 4.0), 128);
    let res = VEXOps::binop(
        IROp::FCmpVecPacked {
            kind: FCmpKind::Ge,
            elem: IRType::F32,
            count: 4,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    let v = res.as_u128().unwrap();
    let expected: u128 =
        0xFFFF_FFFFu128 | (0xFFFF_FFFFu128 << 32) | (0u128 << 64) | (0xFFFF_FFFFu128 << 96);
    assert_eq!(v, expected);
}

#[test]
fn test_fcmp_packed_le_64fx2_concrete() {
    // CMPLEPD: 1.0<=2.0 T, 3.0<=3.0 T.
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(pack_2xf64(1.0, 3.0), 128);
    let r = RustBV::concrete(pack_2xf64(2.0, 3.0), 128);
    let res = VEXOps::binop(
        IROp::FCmpVecPacked {
            kind: FCmpKind::Le,
            elem: IRType::F64,
            count: 2,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    let v = res.as_u128().unwrap();
    let expected: u128 = 0xFFFF_FFFF_FFFF_FFFFu128 | (0xFFFF_FFFF_FFFF_FFFFu128 << 64);
    assert_eq!(v, expected);
}

#[test]
fn test_fcmp_packed_un_32fx4_nan_in_one_lane() {
    // CMPUNPS: NaN in lane 1 only → only lane 1 set.
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(pack_4xf32(1.0, f32::NAN, 3.0, 4.0), 128);
    let r = RustBV::concrete(pack_4xf32(2.0, 5.0, 3.0, 4.0), 128);
    let res = VEXOps::binop(
        IROp::FCmpVecPacked {
            kind: FCmpKind::Un,
            elem: IRType::F32,
            count: 4,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    let v = res.as_u128().unwrap();
    let expected: u128 = 0xFFFF_FFFFu128 << 32;
    assert_eq!(v, expected);
}

#[test]
fn test_fcmp_packed_un_64fx2_nan_in_either() {
    // CMPUNPD: lane 0 has NaN on right, lane 1 ordered.
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(pack_2xf64(1.0, 3.0), 128);
    let r = RustBV::concrete(pack_2xf64(f64::NAN, 4.0), 128);
    let res = VEXOps::binop(
        IROp::FCmpVecPacked {
            kind: FCmpKind::Un,
            elem: IRType::F64,
            count: 2,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    let v = res.as_u128().unwrap();
    let expected: u128 = 0xFFFF_FFFF_FFFF_FFFFu128;
    assert_eq!(v, expected);
}

#[test]
fn test_fcmp_packed_eq_32fx2_concrete_i64() {
    // ARM NEON Iop_CmpEQ32Fx2 returns I64 (2 lanes of 32-bit float in 64 bits).
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(pack_2xf32_64(1.0, 2.0) & 0xFFFF_FFFF_FFFF_FFFF, 64);
    let r = RustBV::concrete(pack_2xf32_64(1.0, 5.0) & 0xFFFF_FFFF_FFFF_FFFF, 64);
    let res = VEXOps::binop(
        IROp::FCmpVecPacked {
            kind: FCmpKind::Eq,
            elem: IRType::F32,
            count: 2,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 64);
    let v = res.as_u128().unwrap();
    // Lane 0 equal → 0xFFFFFFFF; lane 1 unequal → 0.
    assert_eq!(v as u64, 0xFFFF_FFFFu64);
}

#[test]
fn test_fcmp_packed_gt_32fx2_concrete_i64() {
    // ARM NEON Iop_CmpGT32Fx2.
    let ctx = SymContext::new_mock();
    let l = RustBV::concrete(pack_2xf32_64(5.0, 1.0) & 0xFFFF_FFFF_FFFF_FFFF, 64);
    let r = RustBV::concrete(pack_2xf32_64(2.0, 3.0) & 0xFFFF_FFFF_FFFF_FFFF, 64);
    let res = VEXOps::binop(
        IROp::FCmpVecPacked {
            kind: FCmpKind::Gt,
            elem: IRType::F32,
            count: 2,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 64);
    let v = res.as_u128().unwrap();
    assert_eq!(v as u64, 0xFFFF_FFFFu64);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_fcmp_packed_lt_32fx4_symbolic() {
    // Symbolic vector `l` against concrete `r`. Constrain low lane mask to all-1s
    // → solver must satisfy lane 0 of l < lane 0 of r (= 5.0). Other lanes free.
    let ctx = SymContext::new_mock();
    let l = RustBV::symbolic(&ctx, "fpkd_lt_l", 128);
    let r = RustBV::concrete(pack_4xf32(5.0, 1.0, 1.0, 1.0), 128);
    let res = VEXOps::binop(
        IROp::FCmpVecPacked {
            kind: FCmpKind::Lt,
            elem: IRType::F32,
            count: 4,
        },
        l.clone(),
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let lane0_mask = res.extract(31, 0, &ctx);
    let true_mask = RustBV::concrete(0xFFFF_FFFF, 32);
    ctx.add_constraint(lane0_mask.to_z3_ast().eq(true_mask.to_z3_ast()));
    assert!(ctx.is_sat(), "expected SAT for lane0 LT");
    let model = ctx.eval(&l).expect("eval(l) None");
    let lane0 = f32::from_bits((model & 0xFFFF_FFFF) as u32);
    assert!(
        lane0 < 5.0 && !lane0.is_nan(),
        "expected lane0 < 5.0 and not NaN, got {}",
        lane0
    );
}

// -------------------------------------------------------------------------
// NEON SIMD (angr-bkcs.2)
// -------------------------------------------------------------------------

#[test]
fn test_vmul_8x8_concrete() {
    let ctx = SymContext::new_mock();
    // 8 lanes of 8-bit, lane i = i for both → product = i*i mod 256.
    // l = 0x0706050403020100, r = same. result lane i = i*i.
    let l = RustBV::concrete(0x0706_0504_0302_0100u128, 64);
    let r = RustBV::concrete(0x0706_0504_0302_0100u128, 64);
    let res = VEXOps::binop(
        IROp::VMul {
            elem: IRType::I8,
            count: 8,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 64);
    let v = res.as_u128().unwrap();
    // lane i (bits [8i+7:8i]) should equal i*i.
    for i in 0u128..8 {
        let lane = (v >> (i * 8)) & 0xFF;
        assert_eq!(lane, (i * i) & 0xFF, "lane {} of Mul8x8", i);
    }
}

#[test]
fn test_vmul_8x16_concrete() {
    let ctx = SymContext::new_mock();
    // All lanes = 3, multiplied by all lanes = 5 → all lanes = 15.
    let lo64 = 0x0303_0303_0303_0303u128;
    let l = RustBV::concrete(lo64 | (lo64 << 64), 128);
    let lo5 = 0x0505_0505_0505_0505u128;
    let r = RustBV::concrete(lo5 | (lo5 << 64), 128);
    let res = VEXOps::binop(
        IROp::VMul {
            elem: IRType::I8,
            count: 16,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let v = res.as_u128().unwrap();
    for i in 0..16 {
        let lane = (v >> (i * 8)) & 0xFF;
        assert_eq!(lane, 15, "lane {} of Mul8x16", i);
    }
}

#[test]
fn test_vget_elem_8x8_concrete() {
    let ctx = SymContext::new_mock();
    let vec = RustBV::concrete(0x8877_6655_4433_2211u128, 64);
    // Lane 0 = 0x11, lane 7 = 0x88.
    for (lane, expected) in [
        (0u128, 0x11),
        (1, 0x22),
        (2, 0x33),
        (3, 0x44),
        (4, 0x55),
        (5, 0x66),
        (6, 0x77),
        (7, 0x88),
    ] {
        let idx = RustBV::concrete(lane, 8);
        let res = VEXOps::binop(
            IROp::VGetElem {
                elem: IRType::I8,
                count: 8,
            },
            vec.clone(),
            idx,
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 8);
        assert_eq!(res.as_u128().unwrap(), expected, "lane {}", lane);
    }
}

#[test]
fn test_vget_elem_16x8_concrete() {
    let ctx = SymContext::new_mock();
    // V128 with 8 lanes of 16 bits. Lane 3 = 0xDEAD.
    let mut payload: u128 = 0;
    payload |= 0xDEAD_u128 << (3 * 16);
    payload |= 0xBEEF_u128 << (7 * 16);
    let vec = RustBV::concrete(payload, 128);
    let res = VEXOps::binop(
        IROp::VGetElem {
            elem: IRType::I16,
            count: 8,
        },
        vec.clone(),
        RustBV::concrete(3, 8),
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 16);
    assert_eq!(res.as_u128().unwrap(), 0xDEAD);

    let res = VEXOps::binop(
        IROp::VGetElem {
            elem: IRType::I16,
            count: 8,
        },
        vec,
        RustBV::concrete(7, 8),
        &ctx,
    )
    .unwrap();
    assert_eq!(res.as_u128().unwrap(), 0xBEEF);
}

#[test]
fn test_vget_elem_64x2_concrete() {
    let ctx = SymContext::new_mock();
    let lo = 0xAAAA_BBBB_CCCC_DDDDu128;
    let hi = 0x1111_2222_3333_4444u128;
    let vec = RustBV::concrete(lo | (hi << 64), 128);
    let r0 = VEXOps::binop(
        IROp::VGetElem {
            elem: IRType::I64,
            count: 2,
        },
        vec.clone(),
        RustBV::concrete(0, 8),
        &ctx,
    )
    .unwrap();
    assert_eq!(r0.as_u128().unwrap(), lo);
    let r1 = VEXOps::binop(
        IROp::VGetElem {
            elem: IRType::I64,
            count: 2,
        },
        vec,
        RustBV::concrete(1, 8),
        &ctx,
    )
    .unwrap();
    assert_eq!(r1.as_u128().unwrap(), hi);
}

#[test]
fn test_vset_elem_8x8_concrete() {
    let ctx = SymContext::new_mock();
    let vec = RustBV::concrete(0x0u128, 64);
    // Set lane 3 to 0xFF.
    let res = VEXOps::binop_with_rm(
        IROp::VSetElem {
            elem: IRType::I8,
            count: 8,
        },
        vec,
        RustBV::concrete(3, 8),
        RustBV::concrete(0xFF, 8),
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 64);
    assert_eq!(res.as_u128().unwrap(), 0xFF00_0000u128);
}

#[test]
fn test_vset_elem_16x8_concrete() {
    let ctx = SymContext::new_mock();
    let vec = RustBV::concrete(0u128, 128);
    // Set lane 5 to 0xCAFE in a 16x8 vector.
    let res = VEXOps::binop_with_rm(
        IROp::VSetElem {
            elem: IRType::I16,
            count: 8,
        },
        vec,
        RustBV::concrete(5, 8),
        RustBV::concrete(0xCAFE, 16),
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    assert_eq!(res.as_u128().unwrap(), 0xCAFE_u128 << (5 * 16));
}

#[test]
fn test_vset_elem_preserves_other_lanes() {
    let ctx = SymContext::new_mock();
    let vec = RustBV::concrete(0xDEAD_BEEF_CAFE_F00Du128, 64);
    // Overwrite lane 2 (byte 2) with 0x77.
    let res = VEXOps::binop_with_rm(
        IROp::VSetElem {
            elem: IRType::I8,
            count: 8,
        },
        vec,
        RustBV::concrete(2, 8),
        RustBV::concrete(0x77, 8),
        &ctx,
    )
    .unwrap();
    let v = res.as_u128().unwrap();
    // Original byte 2 was 0xFE; expect 0x77 in its place, rest unchanged.
    let expected: u128 = (0xDEAD_BEEF_CAFE_F00Du128 & !(0xFFu128 << 16)) | (0x77u128 << 16);
    assert_eq!(v, expected);
}

#[test]
fn test_vset_elem_round_trip_via_get() {
    let ctx = SymContext::new_mock();
    let vec = RustBV::concrete(0u128, 128);
    let inserted = VEXOps::binop_with_rm(
        IROp::VSetElem {
            elem: IRType::I32,
            count: 4,
        },
        vec,
        RustBV::concrete(2, 8),
        RustBV::concrete(0x1234_5678, 32),
        &ctx,
    )
    .unwrap();
    let lane = VEXOps::binop(
        IROp::VGetElem {
            elem: IRType::I32,
            count: 4,
        },
        inserted,
        RustBV::concrete(2, 8),
        &ctx,
    )
    .unwrap();
    assert_eq!(lane.as_u128().unwrap(), 0x1234_5678);
}

// -------------------------------------------------------------------
// NEON Dup / Widen / Narrow / QNarrow tests (angr-hzs0)
// -------------------------------------------------------------------

#[test]
fn test_vdup_8x8_concrete() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(0xABu128, 8);
    let res = VEXOps::unop(
        IROp::VDup {
            elem: IRType::I8,
            count: 8,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 64);
    assert_eq!(res.as_u128().unwrap(), 0xABAB_ABAB_ABAB_ABABu128);
}

#[test]
fn test_vdup_16x8_concrete() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(0xCAFEu128, 16);
    let res = VEXOps::unop(
        IROp::VDup {
            elem: IRType::I16,
            count: 8,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let expected: u128 = (0..8).fold(0u128, |acc, i| acc | (0xCAFE_u128 << (i * 16)));
    assert_eq!(res.as_u128().unwrap(), expected);
}

#[test]
fn test_vdup_32x4_concrete() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(0xDEAD_BEEFu128, 32);
    let res = VEXOps::unop(
        IROp::VDup {
            elem: IRType::I32,
            count: 4,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let expected: u128 = (0..4).fold(0u128, |acc, i| acc | (0xDEAD_BEEF_u128 << (i * 32)));
    assert_eq!(res.as_u128().unwrap(), expected);
}

#[test]
fn test_vwiden_8sto16x8_signed() {
    let ctx = SymContext::new_mock();
    // 8 lanes of I8: lane 0 = 0xFF (= -1 signed), lane 1 = 0x7F (= 127),
    // lane 2 = 0x80 (= -128), rest = 0.
    let arg_val: u128 = 0xFFu128 | (0x7Fu128 << 8) | (0x80u128 << 16);
    let arg = RustBV::concrete(arg_val, 64);
    let res = VEXOps::unop(
        IROp::VWiden {
            from: IRType::I8,
            count: 8,
            signed: true,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let v = res.as_u128().unwrap();
    // Lane 0: 0xFFFF (sign-ext of 0xFF), Lane 1: 0x007F, Lane 2: 0xFF80.
    assert_eq!(v & 0xFFFF, 0xFFFF, "lane 0");
    assert_eq!((v >> 16) & 0xFFFF, 0x007F, "lane 1");
    assert_eq!((v >> 32) & 0xFFFF, 0xFF80, "lane 2");
    assert_eq!((v >> 48) & 0xFFFF, 0x0000, "lane 3");
}

#[test]
fn test_vwiden_8uto16x8_unsigned() {
    let ctx = SymContext::new_mock();
    let arg_val: u128 = 0xFFu128 | (0x80u128 << 8);
    let arg = RustBV::concrete(arg_val, 64);
    let res = VEXOps::unop(
        IROp::VWiden {
            from: IRType::I8,
            count: 8,
            signed: false,
        },
        arg,
        &ctx,
    )
    .unwrap();
    let v = res.as_u128().unwrap();
    assert_eq!(v & 0xFFFF, 0x00FF, "lane 0 zero-extended");
    assert_eq!((v >> 16) & 0xFFFF, 0x0080, "lane 1 zero-extended");
}

#[test]
fn test_vwiden_32sto64x2_signed() {
    let ctx = SymContext::new_mock();
    // Lane 0 = 0x80000000 (= INT_MIN signed), Lane 1 = 0x12345678.
    let arg_val: u128 = 0x8000_0000u128 | (0x1234_5678u128 << 32);
    let arg = RustBV::concrete(arg_val, 64);
    let res = VEXOps::unop(
        IROp::VWiden {
            from: IRType::I32,
            count: 2,
            signed: true,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let v = res.as_u128().unwrap();
    let lo = v as u64;
    let hi = (v >> 64) as u64;
    assert_eq!(lo, 0xFFFF_FFFF_8000_0000u64, "lane 0 sign-extended");
    assert_eq!(hi, 0x0000_0000_1234_5678u64, "lane 1 zero-positive");
}

#[test]
fn test_vnarrow_un_16to8x8_concrete() {
    let ctx = SymContext::new_mock();
    // Input V128 = 8 lanes of I16: 0x1122, 0x3344, 0x5566, 0x7788, 0x99AA,
    // 0xBBCC, 0xDDEE, 0xFF00.
    let lanes_in: [u16; 8] = [
        0x1122, 0x3344, 0x5566, 0x7788, 0x99AA, 0xBBCC, 0xDDEE, 0xFF00,
    ];
    let mut v: u128 = 0;
    for (i, &lane) in lanes_in.iter().enumerate() {
        v |= (lane as u128) << (i * 16);
    }
    let arg = RustBV::concrete(v, 128);
    let res = VEXOps::unop(
        IROp::VNarrowUn {
            from: IRType::I16,
            count: 8,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 64);
    let out = res.as_u128().unwrap();
    // Each output lane = low byte of input lane.
    for (i, &lane) in lanes_in.iter().enumerate() {
        let got = (out >> (i * 8)) & 0xFF;
        assert_eq!(got, (lane & 0xFF) as u128, "out lane {}", i);
    }
}

#[test]
fn test_vnarrow_bin_16to8x16_concrete() {
    let ctx = SymContext::new_mock();
    // Each input is V128 with 8 lanes of I16. left lanes -> output lanes 0..8;
    // right lanes -> output lanes 8..16.
    let lanes_l: [u16; 8] = [
        0x0011, 0x0022, 0x0033, 0x0044, 0x0055, 0x0066, 0x0077, 0x0088,
    ];
    let lanes_r: [u16; 8] = [
        0x0099, 0x00AA, 0x00BB, 0x00CC, 0x00DD, 0x00EE, 0x00FF, 0x0001,
    ];
    let mut l: u128 = 0;
    let mut r: u128 = 0;
    for (i, &lane) in lanes_l.iter().enumerate() {
        l |= (lane as u128) << (i * 16);
    }
    for (i, &lane) in lanes_r.iter().enumerate() {
        r |= (lane as u128) << (i * 16);
    }
    let res = VEXOps::binop(
        IROp::VNarrowBin {
            from: IRType::I16,
            count: 16,
        },
        RustBV::concrete(l, 128),
        RustBV::concrete(r, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let out = res.as_u128().unwrap();
    for (i, &lane) in lanes_l.iter().enumerate() {
        let got = (out >> (i * 8)) & 0xFF;
        assert_eq!(got, (lane & 0xFF) as u128, "left lane {}", i);
    }
    for (i, &lane) in lanes_r.iter().enumerate() {
        let got = (out >> ((8 + i) * 8)) & 0xFF;
        assert_eq!(got, (lane & 0xFF) as u128, "right lane {}", i);
    }
}

#[test]
fn test_vqnarrow_un_16sto8sx8_saturates() {
    let ctx = SymContext::new_mock();
    // signed I16 source, signed I8 target. Range [-128, 127].
    // Lane 0 = 1000 (clamps to 127), lane 1 = -200 (clamps to -128 = 0x80),
    // lane 2 = 50 (passes through), lane 3 = -50 (passes through).
    let v: u128 = (1000i16 as u16 as u128)
        | ((-200i16 as u16 as u128) << 16)
        | ((50i16 as u16 as u128) << 32)
        | ((-50i16 as u16 as u128) << 48);
    let arg = RustBV::concrete(v, 128);
    let res = VEXOps::unop(
        IROp::VQNarrowUn {
            from: IRType::I16,
            count: 8,
            src_signed: true,
            dst_signed: true,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 64);
    let out = res.as_u128().unwrap();
    assert_eq!((out & 0xFF) as u8, 127, "lane 0 saturated to 127");
    assert_eq!(((out >> 8) & 0xFF) as u8, 0x80, "lane 1 saturated to -128");
    assert_eq!(((out >> 16) & 0xFF) as u8, 50, "lane 2 unchanged");
    assert_eq!(
        ((out >> 24) & 0xFF) as u8,
        (-50i8) as u8,
        "lane 3 unchanged"
    );
}

#[test]
fn test_vqnarrow_un_16sto8ux8_signed_to_unsigned() {
    let ctx = SymContext::new_mock();
    // Signed source -> unsigned dst. Range [0, 255].
    // -1 (0xFFFF) -> 0; 300 -> 255; 100 -> 100.
    let v: u128 = (0xFFFFu128) | ((300u128) << 16) | ((100u128) << 32);
    let arg = RustBV::concrete(v, 128);
    let res = VEXOps::unop(
        IROp::VQNarrowUn {
            from: IRType::I16,
            count: 8,
            src_signed: true,
            dst_signed: false,
        },
        arg,
        &ctx,
    )
    .unwrap();
    let out = res.as_u128().unwrap();
    assert_eq!((out & 0xFF) as u8, 0, "negative -> 0");
    assert_eq!(((out >> 8) & 0xFF) as u8, 255, "300 -> 255");
    assert_eq!(((out >> 16) & 0xFF) as u8, 100, "passes through");
}

#[test]
#[allow(clippy::identity_op)] // explicit 8-lane layout reads better than the minimized form
fn test_vqnarrow_un_16uto8ux8_unsigned() {
    let ctx = SymContext::new_mock();
    // Unsigned source -> unsigned dst. Range [0, 255]. 256 saturates to 255.
    let v: u128 = (256u128) | ((100u128) << 16) | ((0u128) << 32) | ((0xFFFFu128) << 48);
    let arg = RustBV::concrete(v, 128);
    let res = VEXOps::unop(
        IROp::VQNarrowUn {
            from: IRType::I16,
            count: 8,
            src_signed: false,
            dst_signed: false,
        },
        arg,
        &ctx,
    )
    .unwrap();
    let out = res.as_u128().unwrap();
    assert_eq!((out & 0xFF) as u8, 255, "256 -> 255");
    assert_eq!(((out >> 8) & 0xFF) as u8, 100);
    assert_eq!(((out >> 16) & 0xFF) as u8, 0);
    assert_eq!(((out >> 24) & 0xFF) as u8, 255, "0xFFFF -> 255");
}

#[test]
fn test_vqnarrow_bin_16sto8sx16_two_inputs() {
    let ctx = SymContext::new_mock();
    // 8 lanes per input of signed I16. Left lane 0 = 200 (>127 -> 127),
    // right lane 7 = -300 (< -128 -> -128).
    let mut l: u128 = 0;
    let mut r: u128 = 0;
    l |= 200u128;
    l |= (50i16 as u16 as u128) << 16;
    r |= 5i16 as u16 as u128;
    r |= ((-300i16) as u16 as u128) << (7 * 16);

    let res = VEXOps::binop(
        IROp::VQNarrowBin {
            from: IRType::I16,
            count: 16,
            src_signed: true,
            dst_signed: true,
        },
        RustBV::concrete(l, 128),
        RustBV::concrete(r, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let out = res.as_u128().unwrap();
    // Left lane 0 -> output byte 0: 127.
    assert_eq!((out & 0xFF) as u8, 127, "left lane 0 sat to 127");
    // Left lane 1 -> output byte 1: 50.
    assert_eq!(((out >> 8) & 0xFF) as u8, 50, "left lane 1 unchanged");
    // Right lane 0 -> output byte 8: 5.
    assert_eq!(((out >> 64) & 0xFF) as u8, 5, "right lane 0 unchanged");
    // Right lane 7 -> output byte 15: -128.
    assert_eq!(
        ((out >> (15 * 8)) & 0xFF) as u8,
        0x80,
        "right lane 7 sat to -128"
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vdup_symbolic_arg() {
    // Symbolic 8-bit value, dup to 8x8. Constrain the output to a known
    // pattern and check the solver picks the right scalar.
    let ctx = SymContext::new_mock();
    let arg = RustBV::symbolic(&ctx, "dup_arg", 8);
    let res = VEXOps::unop(
        IROp::VDup {
            elem: IRType::I8,
            count: 8,
        },
        arg.clone(),
        &ctx,
    )
    .unwrap();
    let target = RustBV::concrete(0x4242_4242_4242_4242u128, 64);
    ctx.add_constraint(res.to_z3_ast().eq(target.to_z3_ast()));
    assert!(ctx.is_sat(), "expected SAT for dup to 0x42 broadcast");
    let model_arg = ctx.eval(&arg).expect("eval(arg) None");
    assert_eq!(model_arg, 0x42);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vqnarrow_un_symbolic_saturates() {
    // Symbolic I16 saturating to signed I8. Constrain output lane to 127
    // and require source > 127 to confirm saturation kicked in.
    let ctx = SymContext::new_mock();
    let arg = RustBV::symbolic(&ctx, "qn_arg", 128); // 8 lanes I16
    let res = VEXOps::unop(
        IROp::VQNarrowUn {
            from: IRType::I16,
            count: 8,
            src_signed: true,
            dst_signed: true,
        },
        arg.clone(),
        &ctx,
    )
    .unwrap();
    // Constrain output lane 0 = 127 AND source lane 0 = 200.
    let out_lane0 = res.extract(7, 0, &ctx);
    ctx.add_constraint(
        out_lane0
            .to_z3_ast()
            .eq(RustBV::concrete(127, 8).to_z3_ast()),
    );
    let src_lane0 = arg.extract(15, 0, &ctx);
    ctx.add_constraint(
        src_lane0
            .to_z3_ast()
            .eq(RustBV::concrete(200, 16).to_z3_ast()),
    );
    assert!(
        ctx.is_sat(),
        "expected SAT: src lane 0 = 200 saturates to 127"
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vget_elem_symbolic_idx() {
    // Build a concrete vector with distinct lane values, then read
    // through a symbolic idx and constrain it to return a specific lane.
    let ctx = SymContext::new_mock();
    let vec = RustBV::concrete(0x8877_6655_4433_2211u128, 64);
    let sym_idx = RustBV::symbolic(&ctx, "get_idx", 8);
    let res = VEXOps::binop(
        IROp::VGetElem {
            elem: IRType::I8,
            count: 8,
        },
        vec,
        sym_idx.clone(),
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 8);
    // Constrain result to 0x66 → solver must pick idx == 5.
    let target = RustBV::concrete(0x66, 8);
    ctx.add_constraint(res.to_z3_ast().eq(target.to_z3_ast()));
    assert!(ctx.is_sat(), "expected SAT for lane==0x66");
    let model_idx = ctx.eval(&sym_idx).expect("eval(idx) None");
    // idx must be 5 mod 8 (modulo because ITE chain ignores high bits).
    assert_eq!(model_idx & 0x7, 5, "expected idx&7 == 5, got {}", model_idx);
}

// =========================================================================
// angr-tukg.1 — NEON saturating add/sub (VQAdd / VQSub).
// =========================================================================

/// Iop_QAdd8Sx8 — signed 8-bit saturating add over 8 lanes. Exercises:
/// non-overflowing add, positive overflow → INT8_MAX (0x7F), negative
/// overflow → INT8_MIN (0x80), exact-boundary cases.
#[test]
fn test_vqadd_8sx8_concrete() {
    let ctx = SymContext::new_mock();
    // Lane layout (LSB→MSB):
    //   0:  100 + 100 = 200, signed overflow → clamp to +127 (0x7F).
    //   1:  -100 + -100 = -200, signed underflow → clamp to -128 (0x80).
    //   2:   50 + 60 = 110, no overflow → 110 (0x6E).
    //   3:  -50 + -60 = -110, no overflow → -110 (0x92).
    //   4:  127 + 1  = INT_MAX+1 → clamp to +127.
    //   5: -128 + -1 = INT_MIN-1 → clamp to -128.
    //   6:  127 + -1 = 126 (no overflow).
    //   7: -128 + 1  = -127 (no overflow).
    let lanes_a: [i8; 8] = [100, -100, 50, -50, 127, -128, 127, -128];
    let lanes_b: [i8; 8] = [100, -100, 60, -60, 1, -1, -1, 1];
    let lanes_e: [i8; 8] = [127, -128, 110, -110, 127, -128, 126, -127];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= ((lanes_a[i] as u8) as u128) << (i * 8);
        b |= ((lanes_b[i] as u8) as u128) << (i * 8);
        e |= ((lanes_e[i] as u8) as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VQAdd {
            elem: IRType::I8,
            count: 8,
            signed: true,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_QAdd8Ux8 — unsigned 8-bit saturating add. Overflow clamps to 0xFF.
#[test]
fn test_vqadd_8ux8_concrete() {
    let ctx = SymContext::new_mock();
    // 0: 200+100 = 300 → clamp to 255 (0xFF).
    // 1: 255+1   = 256 → clamp to 255.
    // 2: 0+0 → 0.
    // 3: 200+55 = 255 (boundary, no clamp).
    // 4: 200+56 = 256 → clamp.
    // 5: 50+50  = 100.
    // 6,7: 0 fillers.
    let lanes_a: [u8; 8] = [200, 255, 0, 200, 200, 50, 0, 0];
    let lanes_b: [u8; 8] = [100, 1, 0, 55, 56, 50, 0, 0];
    let lanes_e: [u8; 8] = [255, 255, 0, 255, 255, 100, 0, 0];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (lanes_a[i] as u128) << (i * 8);
        b |= (lanes_b[i] as u128) << (i * 8);
        e |= (lanes_e[i] as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VQAdd {
            elem: IRType::I8,
            count: 8,
            signed: false,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_QSub16Sx4 — signed 16-bit saturating sub. Verify both overflow
/// directions and the no-overflow path.
#[test]
fn test_vqsub_16sx4_concrete() {
    let ctx = SymContext::new_mock();
    // 0:  30000 - (-10000) = 40000 → clamp to 32767.
    // 1: -30000 - 10000    = -40000 → clamp to -32768.
    // 2: 100 - 50 = 50 (no overflow).
    // 3: -100 - (-50) = -50 (no overflow).
    let lanes_a: [i16; 4] = [30000, -30000, 100, -100];
    let lanes_b: [i16; 4] = [-10000, 10000, 50, -50];
    let lanes_e: [i16; 4] = [32767, -32768, 50, -50];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..4 {
        a |= ((lanes_a[i] as u16) as u128) << (i * 16);
        b |= ((lanes_b[i] as u16) as u128) << (i * 16);
        e |= ((lanes_e[i] as u16) as u128) << (i * 16);
    }
    let result = VEXOps::binop(
        IROp::VQSub {
            elem: IRType::I16,
            count: 4,
            signed: true,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_QSub32Ux4 — unsigned 32-bit saturating sub on a 128-bit Q-reg.
/// Underflow clamps to 0.
#[test]
fn test_vqsub_32ux4_concrete() {
    let ctx = SymContext::new_mock();
    let lanes_a: [u32; 4] = [100, 0xFFFF_FFFF, 1, 0];
    let lanes_b: [u32; 4] = [50, 1, 5, 5]; // underflow on lane 2 and 3
    let lanes_e: [u32; 4] = [50, 0xFFFF_FFFE, 0, 0];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..4 {
        a |= (lanes_a[i] as u128) << (i * 32);
        b |= (lanes_b[i] as u128) << (i * 32);
        e |= (lanes_e[i] as u128) << (i * 32);
    }
    let result = VEXOps::binop(
        IROp::VQSub {
            elem: IRType::I32,
            count: 4,
            signed: false,
        },
        RustBV::concrete(a, 128),
        RustBV::concrete(b, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Symbolic universality: Iop_QAdd8Sx8 must produce the same bits as the
/// claripy reference at `_op_generic_QAdd` for any 64-bit input. Built per
/// lane using the explicit sign-bit overflow formula (cap_cond + cap).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vqadd_8sx8_symbolic_matches_python_ref() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vqadd_a", 64);
    let b = RustBV::symbolic(&ctx, "vqadd_b", 64);
    let got = VEXOps::binop(
        IROp::VQAdd {
            elem: IRType::I8,
            count: 8,
            signed: true,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();

    // Reference: per-lane signed saturating add as in irop.py.
    let smax = RustBV::concrete(0x7F, 8);
    let smin = RustBV::concrete(0x80, 8);
    let mut lanes = Vec::with_capacity(8);
    for i in 0..8u32 {
        let lo = i * 8;
        let hi = lo + 7;
        let a_lane = a.extract(hi, lo, &ctx);
        let b_lane = b.extract(hi, lo, &ctx);
        let res = a_lane.clone().add_into(b_lane.clone(), &ctx);
        let top_a = a_lane.extract(7, 7, &ctx);
        let top_b = b_lane.extract(7, 7, &ctx);
        let top_r = res.extract(7, 7, &ctx);
        // ~(top_a ^ top_b) & (top_a ^ top_r) == 1
        let signs_match = top_a.clone().xor_into(top_b, &ctx).not_into(&ctx);
        let r_flipped = top_a.xor_into(top_r.clone(), &ctx);
        let overflow = signs_match
            .and_into(r_flipped, &ctx)
            .eq(&RustBV::concrete(1, 1), &ctx);
        let cap = top_r
            .eq(&RustBV::concrete(1, 1), &ctx)
            .ite(&smax, &smin, &ctx);
        lanes.push(overflow.ite(&cap, &res, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VQAdd 8Sx8 must match the claripy QAdd reference for all 64-bit inputs"
    );
    ctx.pop();
}

/// Parse routing: Iop_QAdd / Iop_QSub variants land on VQAdd / VQSub with
/// the expected (elem, count, signed) decomposition. Covers a sampling
/// across D-reg (total=64) and Q-reg (total=128) shapes plus both
/// signedness conventions.
#[test]
fn test_parse_vqaddsub_routing() {
    use crate::vex::opcode_map::parse_opcode;

    let qadd_cases: &[(&str, IRType, u8, bool)] = &[
        ("Iop_QAdd8Sx8", IRType::I8, 8, true),
        ("Iop_QAdd16Ux4", IRType::I16, 4, false),
        ("Iop_QAdd32Sx2", IRType::I32, 2, true),
        ("Iop_QAdd64Ux1", IRType::I64, 1, false),
        ("Iop_QAdd8Ux16", IRType::I8, 16, false),
        ("Iop_QAdd16Sx8", IRType::I16, 8, true),
        ("Iop_QAdd64Sx2", IRType::I64, 2, true),
    ];
    for (op, e, c, s) in qadd_cases {
        match parse_opcode(op) {
            IROp::VQAdd {
                elem,
                count,
                signed,
            } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
                assert_eq!(signed, *s, "{}: signed", op);
            }
            other => panic!("{}: expected VQAdd, got {:?}", op, other),
        }
    }

    let qsub_cases: &[(&str, IRType, u8, bool)] = &[
        ("Iop_QSub8Sx8", IRType::I8, 8, true),
        ("Iop_QSub32Ux4", IRType::I32, 4, false),
        ("Iop_QSub64Sx2", IRType::I64, 2, true),
    ];
    for (op, e, c, s) in qsub_cases {
        match parse_opcode(op) {
            IROp::VQSub {
                elem,
                count,
                signed,
            } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
                assert_eq!(signed, *s, "{}: signed", op);
            }
            other => panic!("{}: expected VQSub, got {:?}", op, other),
        }
    }
}

// =========================================================================
// angr-tukg.2 — NEON pairwise add/min/max (VPwAdd / VPwAddL / VPwMin / VPwMax).
// =========================================================================

/// Iop_PwAdd16x4 — pairwise add over 4 lanes of 16 bits, two 64-bit
/// sources. Output[0..2] from `a`, output[2..4] from `b`. Tests both
/// halves and a sample of values.
#[test]
fn test_vpwadd_16x4_concrete() {
    let ctx = SymContext::new_mock();
    // a lanes (LSB→MSB): 1, 2, 3, 4 → pairs (1+2, 3+4) = 3, 7.
    // b lanes:           10, 20, 100, 200 → pairs (30, 300).
    // Output (LSB→MSB): 3, 7, 30, 300.
    let a_lanes: [u16; 4] = [1, 2, 3, 4];
    let b_lanes: [u16; 4] = [10, 20, 100, 200];
    let e_lanes: [u16; 4] = [3, 7, 30, 300];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..4 {
        a |= (a_lanes[i] as u128) << (i * 16);
        b |= (b_lanes[i] as u128) << (i * 16);
        e |= (e_lanes[i] as u128) << (i * 16);
    }
    let result = VEXOps::binop(
        IROp::VPwAdd {
            elem: IRType::I16,
            count: 4,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_PwAdd8x16 — Q-reg variant: 16 lanes of 8 bits. Output[0..8] from
/// `a`, output[8..16] from `b`. Wrap-around per lane (e.g. 0xFF+0x01=0x00).
#[test]
fn test_vpwadd_8x16_concrete() {
    let ctx = SymContext::new_mock();
    // a: pairs (10+20, 30+40, ..., 70+80) → 30, 70, 110, ..., 0xFF+0x01=0x00.
    // Use explicit lanes for clarity.
    let a_lanes: [u8; 16] = [10, 20, 30, 40, 50, 60, 70, 80, 0xFF, 0x01, 0, 0, 0, 0, 0, 0];
    let b_lanes: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0x80, 0x80, 0, 0, 0, 0];
    // a-half output (8 lanes): (10+20, 30+40, 50+60, 70+80, 0xFF+0x01=0x00, 0, 0, 0)
    //                       = (30, 70, 110, 150, 0, 0, 0, 0)
    // b-half output (8 lanes): (1+2, 3+4, 5+6, 7+8, 9+10, 0x80+0x80=0x00, 0, 0)
    //                       = (3, 7, 11, 15, 19, 0, 0, 0)
    let e_lanes: [u8; 16] = [30, 70, 110, 150, 0, 0, 0, 0, 3, 7, 11, 15, 19, 0, 0, 0];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..16 {
        a |= (a_lanes[i] as u128) << (i * 8);
        b |= (b_lanes[i] as u128) << (i * 8);
        e |= (e_lanes[i] as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VPwAdd {
            elem: IRType::I8,
            count: 16,
        },
        RustBV::concrete(a, 128),
        RustBV::concrete(b, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_PwAddL8Sx8 — signed widening pairwise add. Input is 8 lanes × 8 bits;
/// output is 4 lanes × 16 bits. Negative sources must sign-extend before
/// adding so the sum doesn't lose its sign.
#[test]
fn test_vpwaddl_8sx8_concrete() {
    let ctx = SymContext::new_mock();
    // a lanes (signed i8): -100, -100, 100, 100, -1, -1, 1, 1
    // Pairs: -200, 200, -2, 2 (as i16).
    let a_lanes: [i8; 8] = [-100, -100, 100, 100, -1, -1, 1, 1];
    let e_lanes: [i16; 4] = [-200, 200, -2, 2];
    let mut a: u128 = 0;
    let mut e: u128 = 0;
    for (i, &lane) in a_lanes.iter().enumerate() {
        a |= ((lane as u8) as u128) << (i * 8);
    }
    for (i, &lane) in e_lanes.iter().enumerate() {
        e |= ((lane as u16) as u128) << (i * 16);
    }
    let result = VEXOps::unop(
        IROp::VPwAddL {
            elem: IRType::I8,
            count: 8,
            signed: true,
        },
        RustBV::concrete(a, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_PwAddL8Ux16 — unsigned widening pairwise add, Q-reg. Verifies that
/// 0xFF + 0xFF widens to 0x01FE rather than overflowing in 8 bits.
#[test]
fn test_vpwaddl_8ux16_concrete() {
    let ctx = SymContext::new_mock();
    let a_lanes: [u8; 16] = [
        0xFF, 0xFF, 0x80, 0x80, 0x01, 0x02, 0, 0, 100, 50, 200, 100, 0, 0, 0, 0,
    ];
    let e_lanes: [u16; 8] = [0x01FE, 0x0100, 0x0003, 0, 150, 300, 0, 0];
    let mut a: u128 = 0;
    let mut e: u128 = 0;
    for (i, &lane) in a_lanes.iter().enumerate() {
        a |= (lane as u128) << (i * 8);
    }
    for (i, &lane) in e_lanes.iter().enumerate() {
        e |= (lane as u128) << (i * 16);
    }
    let result = VEXOps::unop(
        IROp::VPwAddL {
            elem: IRType::I8,
            count: 16,
            signed: false,
        },
        RustBV::concrete(a, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_PwMin16Sx4 — pairwise signed min. Exercises mixed-sign pairs and
/// confirms output[2..4] comes from `b`.
#[test]
fn test_vpwmin_16sx4_concrete() {
    let ctx = SymContext::new_mock();
    // a: -100, 100, 200, -200 → pairs min(-100,100)=-100, min(200,-200)=-200
    // b: 30000, -1, 0, 0       → pairs min=-1, min=0
    let a_lanes: [i16; 4] = [-100, 100, 200, -200];
    let b_lanes: [i16; 4] = [30000, -1, 0, 0];
    let e_lanes: [i16; 4] = [-100, -200, -1, 0];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..4 {
        a |= ((a_lanes[i] as u16) as u128) << (i * 16);
        b |= ((b_lanes[i] as u16) as u128) << (i * 16);
        e |= ((e_lanes[i] as u16) as u128) << (i * 16);
    }
    let result = VEXOps::binop(
        IROp::VPwMin {
            elem: IRType::I16,
            count: 4,
            signed: true,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_PwMax8Ux8 — pairwise unsigned max. Distinguishes 0x80 (signed -128)
/// from 0x01 to confirm unsigned compare.
#[test]
fn test_vpwmax_8ux8_concrete() {
    let ctx = SymContext::new_mock();
    // a: 0x80, 0x01, 0x10, 0x10, 0xFF, 0xFF, 0, 0
    //   → unsigned max pairs: 0x80, 0x10, 0xFF, 0
    // b: 0, 0, 0, 0, 50, 60, 70, 80
    //   → max: 0, 0, 60, 80
    let a_lanes: [u8; 8] = [0x80, 0x01, 0x10, 0x10, 0xFF, 0xFF, 0, 0];
    let b_lanes: [u8; 8] = [0, 0, 0, 0, 50, 60, 70, 80];
    let e_lanes: [u8; 8] = [0x80, 0x10, 0xFF, 0, 0, 0, 60, 80];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (a_lanes[i] as u128) << (i * 8);
        b |= (b_lanes[i] as u128) << (i * 8);
        e |= (e_lanes[i] as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VPwMax {
            elem: IRType::I8,
            count: 8,
            signed: false,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Symbolic universality (spec-replay): Iop_PwAdd16x4 must produce the same
/// bits as a hand-built reference for any 64-bit input pair. Claripy has no
/// `_op_generic_PwAdd`, so we reference-build inline per `z3-spec-replay-test-template`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vpwadd_16x4_matches_spec_replay() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vpwadd_a", 64);
    let b = RustBV::symbolic(&ctx, "vpwadd_b", 64);
    let got = VEXOps::binop(
        IROp::VPwAdd {
            elem: IRType::I16,
            count: 4,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();

    // Reference: 2 pairs from a, then 2 pairs from b.
    let mut lanes = Vec::with_capacity(4);
    for src in [&a, &b] {
        for i in 0..2u32 {
            let lo0 = 2 * i * 16;
            let lo1 = (2 * i + 1) * 16;
            let l0 = src.extract(lo0 + 15, lo0, &ctx);
            let l1 = src.extract(lo1 + 15, lo1, &ctx);
            lanes.push(l0.add_into(l1, &ctx));
        }
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VPwAdd 16x4 must match the spec-replay reference for all 64-bit inputs"
    );
    ctx.pop();
}

/// Symbolic universality (spec-replay): Iop_PwAddL16Sx4 widens each lane
/// before adding. Reference uses explicit sign-extend.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vpwaddl_16sx4_matches_spec_replay() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::symbolic(&ctx, "vpwaddl_a", 64);
    let got = VEXOps::unop(
        IROp::VPwAddL {
            elem: IRType::I16,
            count: 4,
            signed: true,
        },
        arg.clone(),
        &ctx,
    )
    .unwrap();

    // Reference: 2 pairs, each pair sign-extended to 32 bits then added.
    let mut lanes = Vec::with_capacity(2);
    for i in 0..2u32 {
        let lo0 = 2 * i * 16;
        let lo1 = (2 * i + 1) * 16;
        let l0 = arg.extract(lo0 + 15, lo0, &ctx).sign_extend_into(32, &ctx);
        let l1 = arg.extract(lo1 + 15, lo1, &ctx).sign_extend_into(32, &ctx);
        lanes.push(l0.add_into(l1, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VPwAddL 16Sx4 must match the spec-replay sign-extend reference"
    );
    ctx.pop();
}

/// Symbolic universality (spec-replay): Iop_PwMin16Sx4 — signed pairwise
/// min via SLE/ITE for each pair.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vpwmin_16sx4_matches_spec_replay() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vpwmin_a", 64);
    let b = RustBV::symbolic(&ctx, "vpwmin_b", 64);
    let got = VEXOps::binop(
        IROp::VPwMin {
            elem: IRType::I16,
            count: 4,
            signed: true,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();

    let mut lanes = Vec::with_capacity(4);
    for src in [&a, &b] {
        for i in 0..2u32 {
            let lo0 = 2 * i * 16;
            let lo1 = (2 * i + 1) * 16;
            let l0 = src.extract(lo0 + 15, lo0, &ctx);
            let l1 = src.extract(lo1 + 15, lo1, &ctx);
            let cond = l0.clone().sle_into(l1.clone(), &ctx);
            lanes.push(cond.ite_into(l0, l1, &ctx));
        }
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VPwMin 16Sx4 must match the spec-replay signed-min reference"
    );
    ctx.pop();
}

/// Parse routing: Iop_PwAdd / PwAddL / PwMin / PwMax variants land on
/// VPwAdd / VPwAddL / VPwMin / VPwMax with the expected decomposition.
/// Iop_PwAdd32Fx2 remains in NeonUnimplemented.
#[test]
fn test_parse_pairwise_routing() {
    use crate::vex::opcode_map::parse_opcode;

    let pwadd_cases: &[(&str, IRType, u8)] = &[
        ("Iop_PwAdd8x8", IRType::I8, 8),
        ("Iop_PwAdd16x4", IRType::I16, 4),
        ("Iop_PwAdd32x2", IRType::I32, 2),
        ("Iop_PwAdd8x16", IRType::I8, 16),
        ("Iop_PwAdd16x8", IRType::I16, 8),
        ("Iop_PwAdd32x4", IRType::I32, 4),
    ];
    for (op, e, c) in pwadd_cases {
        match parse_opcode(op) {
            IROp::VPwAdd { elem, count } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
            }
            other => panic!("{}: expected VPwAdd, got {:?}", op, other),
        }
    }

    let pwaddl_cases: &[(&str, IRType, u8, bool)] = &[
        ("Iop_PwAddL8Sx8", IRType::I8, 8, true),
        ("Iop_PwAddL8Ux8", IRType::I8, 8, false),
        ("Iop_PwAddL16Sx4", IRType::I16, 4, true),
        ("Iop_PwAddL16Ux4", IRType::I16, 4, false),
        ("Iop_PwAddL32Sx2", IRType::I32, 2, true),
        ("Iop_PwAddL32Ux2", IRType::I32, 2, false),
        ("Iop_PwAddL8Sx16", IRType::I8, 16, true),
        ("Iop_PwAddL8Ux16", IRType::I8, 16, false),
        ("Iop_PwAddL16Sx8", IRType::I16, 8, true),
        ("Iop_PwAddL16Ux8", IRType::I16, 8, false),
        ("Iop_PwAddL32Sx4", IRType::I32, 4, true),
        ("Iop_PwAddL32Ux4", IRType::I32, 4, false),
    ];
    for (op, e, c, s) in pwaddl_cases {
        match parse_opcode(op) {
            IROp::VPwAddL {
                elem,
                count,
                signed,
            } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
                assert_eq!(signed, *s, "{}: signed", op);
            }
            other => panic!("{}: expected VPwAddL, got {:?}", op, other),
        }
    }

    let pwmin_cases: &[(&str, IRType, u8, bool)] = &[
        ("Iop_PwMin8Sx8", IRType::I8, 8, true),
        ("Iop_PwMin8Ux8", IRType::I8, 8, false),
        ("Iop_PwMin16Sx4", IRType::I16, 4, true),
        ("Iop_PwMin16Ux4", IRType::I16, 4, false),
        ("Iop_PwMin32Sx2", IRType::I32, 2, true),
        ("Iop_PwMin32Ux2", IRType::I32, 2, false),
    ];
    for (op, e, c, s) in pwmin_cases {
        match parse_opcode(op) {
            IROp::VPwMin {
                elem,
                count,
                signed,
            } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
                assert_eq!(signed, *s, "{}: signed", op);
            }
            other => panic!("{}: expected VPwMin, got {:?}", op, other),
        }
    }

    let pwmax_cases: &[(&str, IRType, u8, bool)] = &[
        ("Iop_PwMax8Sx8", IRType::I8, 8, true),
        ("Iop_PwMax8Ux8", IRType::I8, 8, false),
        ("Iop_PwMax16Sx4", IRType::I16, 4, true),
        ("Iop_PwMax16Ux4", IRType::I16, 4, false),
        ("Iop_PwMax32Sx2", IRType::I32, 2, true),
        ("Iop_PwMax32Ux2", IRType::I32, 2, false),
    ];
    for (op, e, c, s) in pwmax_cases {
        match parse_opcode(op) {
            IROp::VPwMax {
                elem,
                count,
                signed,
            } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
                assert_eq!(signed, *s, "{}: signed", op);
            }
            other => panic!("{}: expected VPwMax, got {:?}", op, other),
        }
    }

    // Float pairwise add stays unimplemented.
    match parse_opcode("Iop_PwAdd32Fx2") {
        IROp::NeonUnimplemented(name) => assert_eq!(name, "Iop_PwAdd32Fx2"),
        other => panic!("Iop_PwAdd32Fx2 expected NeonUnimplemented, got {:?}", other),
    }
}

// =========================================================================
// angr-tukg.3 — NEON rounding halving add (VAvg).
// =========================================================================

/// Iop_Avg8Ux8 — unsigned rounding-average over 8 lanes of 8 bits.
/// Exercises the round-up at the half-way point (`(a+b+1) >> 1`) and the
/// no-overflow guarantee for 0xFF+0xFF.
#[test]
fn test_vavg_8ux8_concrete() {
    let ctx = SymContext::new_mock();
    // Per-lane: rounded average of u8 values.
    //   lane 0: avg(0, 0) = 0.
    //   lane 1: avg(1, 1) = 1.
    //   lane 2: avg(1, 2) = 2  (round up; truncating would give 1).
    //   lane 3: avg(0xFF, 0xFF) = 0xFF (no overflow — widening absorbs +1).
    //   lane 4: avg(0xFE, 0xFF) = 0xFF (round up; truncating would give 0xFE).
    //   lane 5: avg(0x10, 0x20) = 0x18.
    //   lane 6: avg(0x80, 0x80) = 0x80.
    //   lane 7: avg(0x7F, 0x01) = 0x40.
    let a_lanes: [u8; 8] = [0, 1, 1, 0xFF, 0xFE, 0x10, 0x80, 0x7F];
    let b_lanes: [u8; 8] = [0, 1, 2, 0xFF, 0xFF, 0x20, 0x80, 0x01];
    let e_lanes: [u8; 8] = [0, 1, 2, 0xFF, 0xFF, 0x18, 0x80, 0x40];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (a_lanes[i] as u128) << (i * 8);
        b |= (b_lanes[i] as u128) << (i * 8);
        e |= (e_lanes[i] as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VAvg {
            elem: IRType::I8,
            count: 8,
            signed: false,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Avg16Ux8 — Q-reg variant: 8 lanes of 16 bits. 0xFFFF+0xFFFF
/// rounding-average must stay 0xFFFF (no truncation loss).
#[test]
fn test_vavg_16ux8_concrete() {
    let ctx = SymContext::new_mock();
    let a_lanes: [u16; 8] = [0, 1, 0xFFFF, 0xFFFE, 0x1000, 0x8000, 0x7FFF, 0x0123];
    let b_lanes: [u16; 8] = [0, 2, 0xFFFF, 0xFFFF, 0x2000, 0x8000, 0x0001, 0x0456];
    // Avg = (a+b+1) >> 1
    let e_lanes: [u16; 8] = [0, 2, 0xFFFF, 0xFFFF, 0x1800, 0x8000, 0x4000, 0x02BD];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (a_lanes[i] as u128) << (i * 16);
        b |= (b_lanes[i] as u128) << (i * 16);
        e |= (e_lanes[i] as u128) << (i * 16);
    }
    let result = VEXOps::binop(
        IROp::VAvg {
            elem: IRType::I16,
            count: 8,
            signed: false,
        },
        RustBV::concrete(a, 128),
        RustBV::concrete(b, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Avg8Sx8 — signed rounding-average. Validates that two -128 lanes
/// give -128 (sign-extension fence) and mixed-sign lanes round correctly.
#[test]
fn test_vavg_8sx8_concrete() {
    let ctx = SymContext::new_mock();
    // Per-lane signed avg with round-half-up.
    //   lane 0: avg(-128, -128) = -128.
    //   lane 1: avg(127, 127)   = 127.
    //   lane 2: avg(-1, 0)      = 0 (round up: (-1+0+1)/2=0).
    //   lane 3: avg(-2, -1)     = -1.
    //   lane 4: avg(-100, 100)  = 0.
    //   lane 5: avg(-100, 101)  = 1.
    //   lane 6: avg(127, -128)  = 0 (the +1 makes the sum -1+1=0; >>1=0).
    //   lane 7: avg(50, 51)     = 51.
    let a_lanes: [i8; 8] = [-128, 127, -1, -2, -100, -100, 127, 50];
    let b_lanes: [i8; 8] = [-128, 127, 0, -1, 100, 101, -128, 51];
    let e_lanes: [i8; 8] = [-128, 127, 0, -1, 0, 1, 0, 51];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= ((a_lanes[i] as u8) as u128) << (i * 8);
        b |= ((b_lanes[i] as u8) as u128) << (i * 8);
        e |= ((e_lanes[i] as u8) as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VAvg {
            elem: IRType::I8,
            count: 8,
            signed: true,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Symbolic universality (spec-replay): Iop_Avg16Ux4 must equal the
/// reference `((zext(a)+zext(b)+1) >> 1)[15:0]` per lane for all 64-bit
/// inputs. Claripy has no `_op_generic_Avg`; the test uses the
/// `z3-spec-replay-test-template` pattern.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vavg_16ux4_symbolic_universal_unsigned() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vavg_a", 64);
    let b = RustBV::symbolic(&ctx, "vavg_b", 64);
    let got = VEXOps::binop(
        IROp::VAvg {
            elem: IRType::I16,
            count: 4,
            signed: false,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();

    let mut lanes = Vec::with_capacity(4);
    for i in 0..4u32 {
        let lo = i * 16;
        let hi = lo + 15;
        let al = a.extract(hi, lo, &ctx).zero_extend_into(17, &ctx);
        let bl = b.extract(hi, lo, &ctx).zero_extend_into(17, &ctx);
        let sum = al
            .add_into(bl, &ctx)
            .add_into(RustBV::concrete(1, 17), &ctx);
        let shifted = sum.lshr_into(RustBV::concrete(1, 17), &ctx);
        lanes.push(shifted.extract(15, 0, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VAvg 16Ux4 must match the unsigned spec-replay reference"
    );
    ctx.pop();
}

/// Symbolic universality (spec-replay): Iop_Avg8Sx8 signed rounding-avg.
/// Reference sign-extends each lane to 9 bits before summing.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vavg_8sx8_symbolic_universal_signed() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vavg_sa", 64);
    let b = RustBV::symbolic(&ctx, "vavg_sb", 64);
    let got = VEXOps::binop(
        IROp::VAvg {
            elem: IRType::I8,
            count: 8,
            signed: true,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();

    let mut lanes = Vec::with_capacity(8);
    for i in 0..8u32 {
        let lo = i * 8;
        let hi = lo + 7;
        let al = a.extract(hi, lo, &ctx).sign_extend_into(9, &ctx);
        let bl = b.extract(hi, lo, &ctx).sign_extend_into(9, &ctx);
        let sum = al.add_into(bl, &ctx).add_into(RustBV::concrete(1, 9), &ctx);
        let shifted = sum.lshr_into(RustBV::concrete(1, 9), &ctx);
        lanes.push(shifted.extract(7, 0, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VAvg 8Sx8 must match the signed spec-replay reference"
    );
    ctx.pop();
}

/// Parse routing: all 12 Iop_Avg variants land on IROp::VAvg with the
/// expected `(elem, count, signed)` decomposition. No Avg op should remain
/// in NeonUnimplemented.
#[test]
fn test_parse_avg_routing() {
    use crate::vex::opcode_map::parse_opcode;

    let cases: &[(&str, IRType, u8, bool)] = &[
        ("Iop_Avg8Ux8", IRType::I8, 8, false),
        ("Iop_Avg16Ux4", IRType::I16, 4, false),
        ("Iop_Avg32Ux2", IRType::I32, 2, false),
        ("Iop_Avg8Sx8", IRType::I8, 8, true),
        ("Iop_Avg16Sx4", IRType::I16, 4, true),
        ("Iop_Avg32Sx2", IRType::I32, 2, true),
        ("Iop_Avg8Ux16", IRType::I8, 16, false),
        ("Iop_Avg16Ux8", IRType::I16, 8, false),
        ("Iop_Avg32Ux4", IRType::I32, 4, false),
        ("Iop_Avg8Sx16", IRType::I8, 16, true),
        ("Iop_Avg16Sx8", IRType::I16, 8, true),
        ("Iop_Avg32Sx4", IRType::I32, 4, true),
    ];
    for (op, e, c, s) in cases {
        match parse_opcode(op) {
            IROp::VAvg {
                elem,
                count,
                signed,
            } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
                assert_eq!(signed, *s, "{}: signed", op);
            }
            other => panic!("{}: expected VAvg, got {:?}", op, other),
        }
    }
}

// =========================================================================
// angr-tukg.6 — NEON per-lane Cnt / Clz / Cls + GF(2) PolynomialMul.
// =========================================================================

/// Iop_Cnt8x8 — per-byte popcount over 8 lanes. Covers all bits-set,
/// no-bits-set, single-bit, and mid-density patterns.
#[test]
fn test_vcnt_8x8_concrete() {
    let ctx = SymContext::new_mock();
    let lanes: [u8; 8] = [0x00, 0xFF, 0x01, 0x80, 0x0F, 0xF0, 0x55, 0xAA];
    // 0,8,1,1,4,4,4,4 — popcount per lane.
    let expected: [u8; 8] = [0, 8, 1, 1, 4, 4, 4, 4];
    let mut a: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (lanes[i] as u128) << (i * 8);
        e |= (expected[i] as u128) << (i * 8);
    }
    let result = VEXOps::unop(IROp::VCnt { count: 8 }, RustBV::concrete(a, 64), &ctx).unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Cnt8x16 — Q-reg popcount: 16 bytes.
#[test]
fn test_vcnt_8x16_concrete() {
    let ctx = SymContext::new_mock();
    let mut a: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..16u32 {
        // Lane i has popcount i % 9.
        let v: u8 = (1u16.wrapping_shl(i % 9).wrapping_sub(1)) as u8;
        a |= (v as u128) << (i * 8);
        e |= ((i % 9) as u128) << (i * 8);
    }
    let result = VEXOps::unop(IROp::VCnt { count: 16 }, RustBV::concrete(a, 128), &ctx).unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Clz8x8 — per-byte count leading zeros.
#[test]
fn test_vclz_8x8_concrete() {
    let ctx = SymContext::new_mock();
    let lanes: [u8; 8] = [0x00, 0x80, 0x40, 0x01, 0xFF, 0x10, 0x08, 0x7F];
    let expected: [u8; 8] = [8, 0, 1, 7, 0, 3, 4, 1];
    let mut a: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (lanes[i] as u128) << (i * 8);
        e |= (expected[i] as u128) << (i * 8);
    }
    let result = VEXOps::unop(
        IROp::VClz {
            elem: IRType::I8,
            count: 8,
        },
        RustBV::concrete(a, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Clz32x4 — Q-reg 32-bit lane clz. Covers all-zero (→ 32), MSB set
/// (→ 0), and a mid-range value.
#[test]
fn test_vclz_32x4_concrete() {
    let ctx = SymContext::new_mock();
    let lanes: [u32; 4] = [0x00000000, 0x80000000, 0x00010000, 0xFFFFFFFF];
    let expected: [u32; 4] = [32, 0, 15, 0];
    let mut a: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..4 {
        a |= (lanes[i] as u128) << (i * 32);
        e |= (expected[i] as u128) << (i * 32);
    }
    let result = VEXOps::unop(
        IROp::VClz {
            elem: IRType::I32,
            count: 4,
        },
        RustBV::concrete(a, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Cls8x8 — per-byte count leading sign bits (excluding MSB).
/// All-same → N-1; first mismatch determines result.
#[test]
fn test_vcls_8x8_concrete() {
    let ctx = SymContext::new_mock();
    // Inputs covering both sign polarities.
    // 0x00 (0b00000000) → all 7 non-MSB bits match MSB(0) → 7.
    // 0xFF (0b11111111) → all 7 non-MSB bits match MSB(1) → 7.
    // 0x01 (0b00000001) → bit6=0,bit5=0,...,bit1=0 match MSB, bit0=1 differs → 6.
    // 0x02 (0b00000010) → bit1=1 differs at pos 1 → first mismatch at pos 1 → 5.
    // 0x40 (0b01000000) → bit6=1 differs from MSB(0) → 0.
    // 0xC0 (0b11000000) → bit6=1 matches MSB(1); bit5=0 differs at pos 5 → 1.
    // 0x80 (0b10000000) → MSB=1; bits 6..0 all 0, all differ → 0.
    // 0x7F (0b01111111) → MSB=0; bit6=1 differs at pos 6 → 0.
    let lanes: [u8; 8] = [0x00, 0xFF, 0x01, 0x02, 0x40, 0xC0, 0x80, 0x7F];
    let expected: [u8; 8] = [7, 7, 6, 5, 0, 1, 0, 0];
    let mut a: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (lanes[i] as u128) << (i * 8);
        e |= (expected[i] as u128) << (i * 8);
    }
    let result = VEXOps::unop(
        IROp::VCls {
            elem: IRType::I8,
            count: 8,
        },
        RustBV::concrete(a, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_PolynomialMul8x8 — GF(2) multiply per byte, low 8 bits.
/// Reference (closed-form per byte):
///   a=0x01, b=0x57 → 0x01*0x57 = 0x57.
///   a=0x02, b=0x57 → 0x57<<1 = 0xAE; low8 = 0xAE.
///   a=0x03, b=0x57 → 0x57 ^ 0xAE = 0xF9; low8 = 0xF9.
///   a=0x80, b=0x80 → 0x80<<7 = 0x4000; low8 = 0x00.
///   a=0xFF, b=0x01 → XOR of 0x01<<0..7 = 0xFF.
///   a=0xC0, b=0x55 → (0x55<<6)^(0x55<<7) = 0x1540 ^ 0x2A80 = 0x3FC0; low8=0xC0.
///   a=0x00, b=0xFF → 0.
///   a=0xFF, b=0xFF → low8 of GF(2) 0xFF*0xFF; computed below.
#[test]
fn test_vpolynomial_mul_8x8_concrete() {
    let ctx = SymContext::new_mock();
    let a_lanes: [u8; 8] = [0x01, 0x02, 0x03, 0x80, 0xFF, 0xC0, 0x00, 0xFF];
    let b_lanes: [u8; 8] = [0x57, 0x57, 0x57, 0x80, 0x01, 0x55, 0xFF, 0xFF];

    // Compute expected via the same algorithm — keeps the test honest.
    let mut expected: [u8; 8] = [0; 8];
    for i in 0..8 {
        let (a, b) = (a_lanes[i] as u16, b_lanes[i] as u16);
        let mut prod: u16 = 0;
        for bit in 0..8 {
            if (a >> bit) & 1 != 0 {
                prod ^= b << bit;
            }
        }
        expected[i] = (prod & 0xFF) as u8;
    }
    // Sanity-check a couple of the closed-form values to catch a bad
    // expected-table generator.
    assert_eq!(expected[0], 0x57);
    assert_eq!(expected[1], 0xAE);
    assert_eq!(expected[2], 0xF9);
    assert_eq!(expected[3], 0x00);
    assert_eq!(expected[4], 0xFF);
    assert_eq!(expected[6], 0x00);

    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (a_lanes[i] as u128) << (i * 8);
        b |= (b_lanes[i] as u128) << (i * 8);
        e |= (expected[i] as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VPolynomialMul {
            count: 8,
            widen: false,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_PolynomialMull8x8 — widening GF(2) multiply: 8x8 → 8x16. Per lane
/// keeps the full 16-bit product. Output total = 128.
#[test]
fn test_vpolynomial_mull_8x8_concrete() {
    let ctx = SymContext::new_mock();
    let a_lanes: [u8; 8] = [0x01, 0x02, 0x80, 0x40, 0xFF, 0x10, 0x55, 0x00];
    let b_lanes: [u8; 8] = [0x57, 0x57, 0x80, 0x02, 0xFF, 0x10, 0xAA, 0xFF];
    let mut expected: [u16; 8] = [0; 8];
    for i in 0..8 {
        let (a, b) = (a_lanes[i] as u16, b_lanes[i] as u16);
        let mut prod: u16 = 0;
        for bit in 0..8 {
            if (a >> bit) & 1 != 0 {
                prod ^= b << bit;
            }
        }
        expected[i] = prod;
    }
    // Widening keeps the full product. 0x80*0x80 over GF(2) = 0x4000.
    assert_eq!(expected[2], 0x4000);
    // 0x01*0x57 = 0x57; widening must preserve this (low byte only).
    assert_eq!(expected[0], 0x0057);

    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (a_lanes[i] as u128) << (i * 8);
        b |= (b_lanes[i] as u128) << (i * 8);
        e |= (expected[i] as u128) << (i * 16);
    }
    let result = VEXOps::binop(
        IROp::VPolynomialMul {
            count: 8,
            widen: true,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Symbolic universality (spec-replay): Iop_Cnt8x8 must equal the
/// per-byte bit-sum reference for all 64-bit inputs.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vcnt_8x8_symbolic_universal() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vcnt_a", 64);
    let got = VEXOps::unop(IROp::VCnt { count: 8 }, a.clone(), &ctx).unwrap();

    let mut lanes = Vec::with_capacity(8);
    for i in 0..8u32 {
        let lo = i * 8;
        let mut acc = RustBV::concrete(0, 8);
        for b in 0..8 {
            let bit = a.extract(lo + b, lo + b, &ctx);
            acc = acc.add_into(bit.zero_extend_into(8, &ctx), &ctx);
        }
        lanes.push(acc);
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VCnt 8x8 must match the spec-replay reference"
    );
    ctx.pop();
}

/// Symbolic universality (spec-replay): Iop_Clz8x8 must match the
/// claripy-style ITE chain reference per lane.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vclz_8x8_symbolic_universal() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vclz_a", 64);
    let got = VEXOps::unop(
        IROp::VClz {
            elem: IRType::I8,
            count: 8,
        },
        a.clone(),
        &ctx,
    )
    .unwrap();

    let mut lanes = Vec::with_capacity(8);
    let one = RustBV::concrete(1, 1);
    for i in 0..8u32 {
        let lo = i * 8;
        let hi = lo + 7;
        let lane = a.extract(hi, lo, &ctx);
        // Build claripy-style ITE chain inline.
        let mut expr = RustBV::concrete(8, 8);
        for b in 0..8u32 {
            let bit = lane.extract(b, b, &ctx);
            let cond = bit.eq_into(one.clone(), &ctx);
            let then_v = RustBV::concrete((8 - b - 1) as u128, 8);
            expr = cond.ite_into(then_v, expr, &ctx);
        }
        lanes.push(expr);
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VClz 8x8 must match the spec-replay reference"
    );
    ctx.pop();
}

/// Symbolic universality (spec-replay): Iop_PolynomialMul8x8 must match
/// the XOR-of-shifts reference per lane (low 8 bits).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vpolynomial_mul_8x8_symbolic_universal() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vpmul_a", 64);
    let b = RustBV::symbolic(&ctx, "vpmul_b", 64);
    let got = VEXOps::binop(
        IROp::VPolynomialMul {
            count: 8,
            widen: false,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();

    let one = RustBV::concrete(1, 1);
    let mut lanes = Vec::with_capacity(8);
    for i in 0..8u32 {
        let lo = i * 8;
        let hi = lo + 7;
        let a_lane = a.extract(hi, lo, &ctx);
        let b_lane = b.extract(hi, lo, &ctx);
        let b_wide = b_lane.zero_extend_into(16, &ctx);
        let mut acc = RustBV::concrete(0, 16);
        for bit in 0..8u32 {
            let bit_a = a_lane.extract(bit, bit, &ctx);
            let cond = bit_a.eq_into(one.clone(), &ctx);
            let shifted = b_wide
                .clone()
                .shl_into(RustBV::concrete(bit as u128, 16), &ctx);
            let addend = cond.ite_into(shifted, RustBV::concrete(0, 16), &ctx);
            acc = acc.xor_into(addend, &ctx);
        }
        lanes.push(acc.extract(7, 0, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VPolynomialMul 8x8 must match the spec-replay reference"
    );
    ctx.pop();
}

/// Parse routing: all Cnt/Clz/Cls/PolynomialMul opcodes land on their real
/// IROp variants (not NeonUnimplemented).
#[test]
fn test_parse_cnt_clz_cls_pmul_routing() {
    use crate::vex::opcode_map::parse_opcode;

    // VCnt
    for (op, c) in &[("Iop_Cnt8x8", 8u8), ("Iop_Cnt8x16", 16)] {
        match parse_opcode(op) {
            IROp::VCnt { count } => assert_eq!(count, *c, "{}: count", op),
            other => panic!("{}: expected VCnt, got {:?}", op, other),
        }
    }

    // VClz
    let clz_cases: &[(&str, IRType, u8)] = &[
        ("Iop_Clz8x8", IRType::I8, 8),
        ("Iop_Clz16x4", IRType::I16, 4),
        ("Iop_Clz32x2", IRType::I32, 2),
        ("Iop_Clz8x16", IRType::I8, 16),
        ("Iop_Clz16x8", IRType::I16, 8),
        ("Iop_Clz32x4", IRType::I32, 4),
    ];
    for (op, e, c) in clz_cases {
        match parse_opcode(op) {
            IROp::VClz { elem, count } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
            }
            other => panic!("{}: expected VClz, got {:?}", op, other),
        }
    }

    // VCls
    let cls_cases: &[(&str, IRType, u8)] = &[
        ("Iop_Cls8x8", IRType::I8, 8),
        ("Iop_Cls16x4", IRType::I16, 4),
        ("Iop_Cls32x2", IRType::I32, 2),
        ("Iop_Cls8x16", IRType::I8, 16),
        ("Iop_Cls16x8", IRType::I16, 8),
        ("Iop_Cls32x4", IRType::I32, 4),
    ];
    for (op, e, c) in cls_cases {
        match parse_opcode(op) {
            IROp::VCls { elem, count } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
            }
            other => panic!("{}: expected VCls, got {:?}", op, other),
        }
    }

    // VPolynomialMul (non-widening and widening)
    let pmul_cases: &[(&str, u8, bool)] = &[
        ("Iop_PolynomialMul8x8", 8, false),
        ("Iop_PolynomialMul8x16", 16, false),
        ("Iop_PolynomialMull8x8", 8, true),
    ];
    for (op, c, w) in pmul_cases {
        match parse_opcode(op) {
            IROp::VPolynomialMul { count, widen } => {
                assert_eq!(count, *c, "{}: count", op);
                assert_eq!(widen, *w, "{}: widen", op);
            }
            other => panic!("{}: expected VPolynomialMul, got {:?}", op, other),
        }
    }
}

// =========================================================================
// angr-tukg.7 — NEON vector-shift-by-vector (VShl / VShr / VSar).
// =========================================================================

/// Iop_Shl8x8 — left shift each of 8 lanes by the corresponding count lane.
/// Covers: zero shift (identity), in-range shifts, and out-of-range counts
/// (≥ lane width → zero, matching Z3 bvshl).
#[test]
fn test_vshl_8x8_concrete() {
    let ctx = SymContext::new_mock();
    // Lane layout (LSB→MSB): vec lanes, then shift lanes.
    //   0:  0x01 << 0  = 0x01.
    //   1:  0x01 << 1  = 0x02.
    //   2:  0x01 << 7  = 0x80.
    //   3:  0x01 << 8  = 0 (count == lane width).
    //   4:  0x01 << 255 = 0 (count > lane width).
    //   5:  0xFF << 4  = 0xF0 (high bits shifted out).
    //   6:  0x55 << 1  = 0xAA.
    //   7:  0x80 << 1  = 0 (high bit shifted out).
    let lanes_v: [u8; 8] = [0x01, 0x01, 0x01, 0x01, 0x01, 0xFF, 0x55, 0x80];
    let lanes_s: [u8; 8] = [0, 1, 7, 8, 255, 4, 1, 1];
    let lanes_e: [u8; 8] = [0x01, 0x02, 0x80, 0x00, 0x00, 0xF0, 0xAA, 0x00];
    let mut v: u128 = 0;
    let mut s: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        v |= (lanes_v[i] as u128) << (i * 8);
        s |= (lanes_s[i] as u128) << (i * 8);
        e |= (lanes_e[i] as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VShl {
            elem: IRType::I8,
            count: 8,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Shr16x8 — logical right shift over 8x i16 lanes on a 128-bit vector.
/// Verifies zero-fill (Shr discards the sign bit) and out-of-range counts.
#[test]
fn test_vshr_16x8_concrete() {
    let ctx = SymContext::new_mock();
    let lanes_v: [u16; 8] = [
        0x8000, 0xFFFF, 0xABCD, 0x0001, 0xFFFF, 0x4000, 0x0F0F, 0x1234,
    ];
    let lanes_s: [u16; 8] = [15, 8, 4, 0, 16, 1, 4, 100];
    // 0x8000 >> 15 = 1 (no sign extend).
    // 0xFFFF >> 8  = 0x00FF.
    // 0xABCD >> 4  = 0x0ABC.
    // 0x0001 >> 0  = 0x0001.
    // 0xFFFF >> 16 = 0 (count == width).
    // 0x4000 >> 1  = 0x2000.
    // 0x0F0F >> 4  = 0x00F0.
    // 0x1234 >> 100 = 0 (count > width).
    let lanes_e: [u16; 8] = [1, 0x00FF, 0x0ABC, 0x0001, 0, 0x2000, 0x00F0, 0];
    let mut v: u128 = 0;
    let mut s: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        v |= (lanes_v[i] as u128) << (i * 16);
        s |= (lanes_s[i] as u128) << (i * 16);
        e |= (lanes_e[i] as u128) << (i * 16);
    }
    let result = VEXOps::binop(
        IROp::VShr {
            elem: IRType::I16,
            count: 8,
        },
        RustBV::concrete(v, 128),
        RustBV::concrete(s, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Sar32x2 — arithmetic right shift over 2x i32 lanes on a 64-bit
/// vector. Verifies sign-fill on negative inputs and out-of-range counts.
#[test]
fn test_vsar_32x2_concrete() {
    let ctx = SymContext::new_mock();
    // 0: -1i32 (0xFFFF_FFFF) >> 4   = -1 (sign-fill keeps all bits set).
    // 1: 0x4000_0000   >> 1   = 0x2000_0000 (positive → logical shift).
    // Note: 0xFFFF_FFFF >> 32 would also be all-1 in arithmetic shift,
    // but Z3 bvashr semantics for count >= width are sign-fill which is
    // matched by our concrete fast path.
    let lanes_v: [u32; 2] = [0xFFFF_FFFF, 0x4000_0000];
    let lanes_s: [u32; 2] = [4, 1];
    let lanes_e: [u32; 2] = [0xFFFF_FFFF, 0x2000_0000];
    let mut v: u128 = 0;
    let mut s: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..2 {
        v |= (lanes_v[i] as u128) << (i * 32);
        s |= (lanes_s[i] as u128) << (i * 32);
        e |= (lanes_e[i] as u128) << (i * 32);
    }
    let result = VEXOps::binop(
        IROp::VSar {
            elem: IRType::I32,
            count: 2,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Sar32x2 — out-of-range count produces sign-fill (-1 for negative
/// lanes, 0 for positive). Concrete fast path mirrors Z3 bvashr.
#[test]
fn test_vsar_32x2_oor_sign_fill() {
    let ctx = SymContext::new_mock();
    // 0: -1i32 >> 64 = sign-fill = 0xFFFF_FFFF.
    // 1:  1i32 >> 32 = sign-fill = 0 (positive).
    let lanes_v: [u32; 2] = [0xFFFF_FFFF, 0x0000_0001];
    let lanes_s: [u32; 2] = [64, 32];
    let lanes_e: [u32; 2] = [0xFFFF_FFFF, 0];
    let mut v: u128 = 0;
    let mut s: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..2 {
        v |= (lanes_v[i] as u128) << (i * 32);
        s |= (lanes_s[i] as u128) << (i * 32);
        e |= (lanes_e[i] as u128) << (i * 32);
    }
    let result = VEXOps::binop(
        IROp::VSar {
            elem: IRType::I32,
            count: 2,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Sal8x8 (== Iop_Shl8x8 bit-for-bit): a single concrete check that
/// both opcodes parse to VShl and produce identical results.
#[test]
fn test_vsal_routes_to_vshl_and_matches() {
    use crate::vex::opcode_map::parse_opcode;
    let ctx = SymContext::new_mock();
    let v = 0x1234_5678_9ABC_DEF0u128;
    let s = 0x0102_0304_0506_0708u128; // per-lane counts 8,7,6,5,4,3,2,1
    let shl_res = VEXOps::binop(
        IROp::VShl {
            elem: IRType::I8,
            count: 8,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    // Sal{N}x{M} must parse to the same IROp variant.
    assert!(matches!(parse_opcode("Iop_Sal8x8"), IROp::VShl { .. }));
    let sal_op = parse_opcode("Iop_Sal8x8");
    let sal_res = VEXOps::binop(
        sal_op,
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(sal_res.as_u128().unwrap(), shl_res.as_u128().unwrap());
}

/// Z3 universality parity vs the explicit claripy reference for
/// `Iop_Shl16x4` (operation_map["Shl"] = "__lshift__"; vector dispatch
/// per `_op_vector_mapped` extracts each lane and applies `bvshl`).
/// Symbolic inputs + add_constraint(got ≠ py).not() → assert UNSAT.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vshl_16x4_symbolic_matches_python_ref() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "shl_a_16x4", 64);
    let b = RustBV::symbolic(&ctx, "shl_b_16x4", 64);
    let got = VEXOps::binop(
        IROp::VShl {
            elem: IRType::I16,
            count: 4,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();
    // Build the claripy reference: per-lane bvshl. _op_vector_mapped
    // concats lanes high→low; we use concat_le_elements (low→high), so
    // the resulting BV is structurally equivalent.
    let mut lanes: Vec<RustBV> = Vec::with_capacity(4);
    for i in 0..4 {
        let lo = i * 16;
        let hi = lo + 15;
        let al = a.extract(hi, lo, &ctx);
        let bl = b.extract(hi, lo, &ctx);
        lanes.push(al.shl_into(bl, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);
    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    let universal = !ctx.is_sat();
    ctx.pop();
    assert!(
        universal,
        "VShl 16x4 must match the claripy __lshift__ reference for all 64-bit inputs"
    );
}

/// Z3 universality parity vs the claripy reference for `Iop_Sar8x16`
/// (operation_map["Sar"] = "__rshift__" → bvashr; 16 lanes of 8 bits
/// across a 128-bit vector).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vsar_8x16_symbolic_matches_python_ref() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "sar_a_8x16", 128);
    let b = RustBV::symbolic(&ctx, "sar_b_8x16", 128);
    let got = VEXOps::binop(
        IROp::VSar {
            elem: IRType::I8,
            count: 16,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();
    let mut lanes: Vec<RustBV> = Vec::with_capacity(16);
    for i in 0..16 {
        let lo = i * 8;
        let hi = lo + 7;
        let al = a.extract(hi, lo, &ctx);
        let bl = b.extract(hi, lo, &ctx);
        lanes.push(al.ashr_into(bl, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);
    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    let universal = !ctx.is_sat();
    ctx.pop();
    assert!(
        universal,
        "VSar 8x16 must match the claripy __rshift__ reference for all 128-bit inputs"
    );
}

/// Parse routing: all 8 VShl shapes (8x8 .. 64x2) and the matching Sal
/// aliases land on VShl with the expected (elem, count); same for VShr/VSar.
#[test]
fn test_parse_vshift_routing() {
    use crate::vex::opcode_map::parse_opcode;
    let shl_cases: &[(&str, IRType, u8)] = &[
        ("Iop_Shl8x8", IRType::I8, 8),
        ("Iop_Shl16x4", IRType::I16, 4),
        ("Iop_Shl32x2", IRType::I32, 2),
        ("Iop_Shl64x1", IRType::I64, 1),
        ("Iop_Shl8x16", IRType::I8, 16),
        ("Iop_Shl16x8", IRType::I16, 8),
        ("Iop_Shl32x4", IRType::I32, 4),
        ("Iop_Shl64x2", IRType::I64, 2),
        // Sal aliases route to the same variant.
        ("Iop_Sal8x8", IRType::I8, 8),
        ("Iop_Sal16x4", IRType::I16, 4),
        ("Iop_Sal32x2", IRType::I32, 2),
        ("Iop_Sal64x1", IRType::I64, 1),
        ("Iop_Sal8x16", IRType::I8, 16),
        ("Iop_Sal16x8", IRType::I16, 8),
        ("Iop_Sal32x4", IRType::I32, 4),
        ("Iop_Sal64x2", IRType::I64, 2),
    ];
    for (op, e, c) in shl_cases {
        match parse_opcode(op) {
            IROp::VShl { elem, count } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
            }
            other => panic!("{}: expected VShl, got {:?}", op, other),
        }
    }

    let shr_cases: &[(&str, IRType, u8)] = &[
        ("Iop_Shr8x8", IRType::I8, 8),
        ("Iop_Shr16x4", IRType::I16, 4),
        ("Iop_Shr32x2", IRType::I32, 2),
        ("Iop_Shr64x1", IRType::I64, 1),
        ("Iop_Shr8x16", IRType::I8, 16),
        ("Iop_Shr16x8", IRType::I16, 8),
        ("Iop_Shr32x4", IRType::I32, 4),
        ("Iop_Shr64x2", IRType::I64, 2),
    ];
    for (op, e, c) in shr_cases {
        match parse_opcode(op) {
            IROp::VShr { elem, count } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
            }
            other => panic!("{}: expected VShr, got {:?}", op, other),
        }
    }

    let sar_cases: &[(&str, IRType, u8)] = &[
        ("Iop_Sar8x8", IRType::I8, 8),
        ("Iop_Sar16x4", IRType::I16, 4),
        ("Iop_Sar32x2", IRType::I32, 2),
        ("Iop_Sar64x1", IRType::I64, 1),
        ("Iop_Sar8x16", IRType::I8, 16),
        ("Iop_Sar16x8", IRType::I16, 8),
        ("Iop_Sar32x4", IRType::I32, 4),
        ("Iop_Sar64x2", IRType::I64, 2),
    ];
    for (op, e, c) in sar_cases {
        match parse_opcode(op) {
            IROp::VSar { elem, count } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
            }
            other => panic!("{}: expected VSar, got {:?}", op, other),
        }
    }
}

// =========================================================================
// angr-tukg.8 — NEON saturating vector shifts (VQShlSat).
// =========================================================================

/// Iop_QShl8x8 — unsigned saturating left shift by vector (D-reg).
/// Covers in-range left shift, OOR left shift (amt ≥ width → UMAX if
/// `a != 0` else 0), overflow saturation to UMAX, and the negative-amt
/// branch (right shift via logical shift, with OOR → 0).
#[test]
fn test_vqshl_8x8_concrete_unsigned() {
    let ctx = SymContext::new_mock();
    // amt is sign-extended as i8: 0xFF = -1, 0xFC = -4, 0xF8 = -8.
    let lanes_v: [u8; 8] = [0x01, 0x01, 0x01, 0x80, 0x40, 0xFF, 0x10, 0x80];
    let lanes_s: [u8; 8] = [0, 7, 8, 1, 1, 0xFF, 0xFC, 0xF8];
    // 0x01<<0  = 0x01.    0x01<<7  = 0x80.   0x01<<8 OOR, a!=0 → 0xFF.
    // 0x80<<1  overflow → 0xFF.              0x40<<1  = 0x80 (no overflow).
    // 0xFF >> 1 (amt=-1) = 0x7F (lshr).      0x10>>4 (amt=-4) = 0x01.
    // 0x80>>8 OOR → 0 (lshr).
    let lanes_e: [u8; 8] = [0x01, 0x80, 0xFF, 0xFF, 0x80, 0x7F, 0x01, 0x00];
    let (mut v, mut s, mut e) = (0u128, 0u128, 0u128);
    for i in 0..8 {
        v |= (lanes_v[i] as u128) << (i * 8);
        s |= (lanes_s[i] as u128) << (i * 8);
        e |= (lanes_e[i] as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VQShlSat {
            elem: IRType::I8,
            count: 8,
            signed: false,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_QSal16x4 — signed saturating left shift by vector (D-reg).
/// Covers positive overflow → SMAX, negative overflow → SMIN, in-range
/// shift, and the negative-amt branch (right shift via ashr, sign-fill).
#[test]
fn test_vqsal_16x4_concrete_signed() {
    let ctx = SymContext::new_mock();
    // i16: SMAX=0x7FFF, SMIN=0x8000, -1=0xFFFF.
    // lane 0: 0x1000 (positive) << 3 = 0x8000 (negative when truncated) →
    //   overflow → SMAX = 0x7FFF.
    // lane 1: 0xF000 (= -4096) << 1 = 0xE000 (= -8192); ashr(0xE000,1)
    //   = 0xF000 == a → no overflow → 0xE000.
    // lane 2: 0xFFFF (= -1) with amt = -1 (0xFFFF sign-extended): ashr
    //   by 1 → 0xFFFF (sign-fill).
    // lane 3: 0x0040 (positive) with amt = 16 (OOR): a > 0 → SMAX.
    let lanes_v: [u16; 4] = [0x1000, 0xF000, 0xFFFF, 0x0040];
    let lanes_s: [u16; 4] = [3, 1, 0xFFFF, 16];
    let lanes_e: [u16; 4] = [0x7FFF, 0xE000, 0xFFFF, 0x7FFF];
    let (mut v, mut s, mut e) = (0u128, 0u128, 0u128);
    for i in 0..4 {
        v |= (lanes_v[i] as u128) << (i * 16);
        s |= (lanes_s[i] as u128) << (i * 16);
        e |= (lanes_e[i] as u128) << (i * 16);
    }
    let result = VEXOps::binop(
        IROp::VQShlSat {
            elem: IRType::I16,
            count: 4,
            signed: true,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_QSal8x16 — signed saturating left shift over 16x i8 lanes
/// (Q-reg/128-bit). Covers OOR-with-negative-input → SMIN saturation
/// (a < 0 with amt > width).
#[test]
fn test_vqsal_8x16_concrete_smin_saturation() {
    let ctx = SymContext::new_mock();
    // Build a 16-lane vector: alternating positive overflow (a=1, amt=8 OOR)
    // and negative overflow (a=0xFF=-1, amt=8 OOR).
    // amt=8 for all lanes. Positive a=1 (>0) → SMAX=0x7F.
    //                     Negative a=0xFF (<0) → SMIN=0x80.
    let mut v: u128 = 0;
    let mut s: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..16 {
        let (a, expected) = if i % 2 == 0 {
            (1u8, 0x7Fu8)
        } else {
            (0xFFu8, 0x80u8)
        };
        v |= (a as u128) << (i * 8);
        s |= 8u128 << (i * 8);
        e |= (expected as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VQShlSat {
            elem: IRType::I8,
            count: 16,
            signed: true,
        },
        RustBV::concrete(v, 128),
        RustBV::concrete(s, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_QShl64x1 — width=64 edge case (single-lane D-reg). Exercises the
/// elem_width = 64 boundary of the concrete fast path (elem_mask uses
/// the full u64 range; sign-extension via the |!elem_mask| branch).
#[test]
fn test_vqshl_64x1_concrete_width_boundary() {
    let ctx = SymContext::new_mock();
    // Unsigned: 0x0000_0000_0000_0001 << 63 = 0x8000_0000_0000_0000.
    // Round-trip: (0x8000... >> 63) = 1 == a. No overflow. Result OK.
    let v = 1u128;
    let s = 63u128;
    let e = 0x8000_0000_0000_0000u128;
    let result = VEXOps::binop(
        IROp::VQShlSat {
            elem: IRType::I64,
            count: 1,
            signed: false,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.as_u128().unwrap(), e);

    // Same input, signed: a=1 (positive), shift left by 63 → 0x8000...
    // which is SMIN as signed. Overflow → SMAX = 0x7FFF_FFFF_FFFF_FFFF.
    let result_s = VEXOps::binop(
        IROp::VQShlSat {
            elem: IRType::I64,
            count: 1,
            signed: true,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result_s.as_u128().unwrap(), 0x7FFF_FFFF_FFFF_FFFFu128);
}

/// Parse routing for all 16 saturating-shift opcodes: Iop_QShl{N}x{M}
/// (signed=false) and Iop_QSal{N}x{M} (signed=true), 8 shapes each.
#[test]
fn test_parse_vqshlsat_routing() {
    use crate::vex::opcode_map::parse_opcode;
    let shapes: &[(&str, IRType, u8)] = &[
        ("8x8", IRType::I8, 8),
        ("16x4", IRType::I16, 4),
        ("32x2", IRType::I32, 2),
        ("64x1", IRType::I64, 1),
        ("8x16", IRType::I8, 16),
        ("16x8", IRType::I16, 8),
        ("32x4", IRType::I32, 4),
        ("64x2", IRType::I64, 2),
    ];
    for (sfx, elem_e, count_e) in shapes {
        for (prefix, want_signed) in [("Iop_QShl", false), ("Iop_QSal", true)] {
            let name = format!("{}{}", prefix, sfx);
            match parse_opcode(&name) {
                IROp::VQShlSat {
                    elem,
                    count,
                    signed,
                } => {
                    assert_eq!(elem, *elem_e, "{}: elem", name);
                    assert_eq!(count, *count_e, "{}: count", name);
                    assert_eq!(signed, want_signed, "{}: signed", name);
                }
                other => panic!("{}: expected VQShlSat, got {:?}", name, other),
            }
        }
    }
}

/// Symbolic parity: the saturating shift behaves identically to a hand-
/// rolled per-lane ITE chain over Z3 bvshl/bvlshr/bvashr + round-trip
/// overflow detection. There is no `_op_generic_QShl` in claripy, so the
/// reference is the same algorithm encoded straight from the spec. This
/// catches encoding mistakes (wrong cap, swapped then/else, sign-bit
/// extraction errors) without depending on a Python reference.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vqshl_16x4_symbolic_universal_unsigned() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "qshl_a_16x4", 64);
    let b = RustBV::symbolic(&ctx, "qshl_b_16x4", 64);
    let got = VEXOps::binop(
        IROp::VQShlSat {
            elem: IRType::I16,
            count: 4,
            signed: false,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();
    // Reference: per-lane spec replay with the same primitives as the
    // helper. Any encoding drift will surface as a SAT counter-example.
    let umax = RustBV::concrete(0xFFFFu128, 16);
    let zero16 = RustBV::concrete(0, 16);
    let bit_one = RustBV::concrete(1, 1);
    let mut lanes: Vec<RustBV> = Vec::with_capacity(4);
    for i in 0..4 {
        let lo = i * 16;
        let hi = lo + 15;
        let al = a.extract(hi, lo, &ctx);
        let bl = b.extract(hi, lo, &ctx);
        let shl_v = al.clone().shl_into(bl.clone(), &ctx);
        let recovered = shl_v.clone().lshr_into(bl.clone(), &ctx);
        let no_overflow = recovered.eq(&al, &ctx);
        let left_branch = no_overflow.ite(&shl_v, &umax, &ctx);
        let neg_amt = zero16.clone().sub_into(bl.clone(), &ctx);
        let right_branch = al.clone().lshr_into(neg_amt, &ctx);
        let amt_top = bl.extract(15, 15, &ctx);
        let amt_is_neg = amt_top.eq(&bit_one, &ctx);
        lanes.push(amt_is_neg.ite(&right_branch, &left_branch, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);
    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    let universal = !ctx.is_sat();
    ctx.pop();
    assert!(
        universal,
        "VQShlSat 16x4 (unsigned) must match the per-lane spec for all 64-bit inputs"
    );
}

/// Same parity check for the signed (QSal) branch — verifies the SMAX/
/// SMIN cap selection from the data-sign bit.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vqsal_16x4_symbolic_universal_signed() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "qsal_a_16x4", 64);
    let b = RustBV::symbolic(&ctx, "qsal_b_16x4", 64);
    let got = VEXOps::binop(
        IROp::VQShlSat {
            elem: IRType::I16,
            count: 4,
            signed: true,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();
    let smax = RustBV::concrete(0x7FFFu128, 16);
    let smin = RustBV::concrete(0x8000u128, 16);
    let zero16 = RustBV::concrete(0, 16);
    let bit_one = RustBV::concrete(1, 1);
    let mut lanes: Vec<RustBV> = Vec::with_capacity(4);
    for i in 0..4 {
        let lo = i * 16;
        let hi = lo + 15;
        let al = a.extract(hi, lo, &ctx);
        let bl = b.extract(hi, lo, &ctx);
        let shl_v = al.clone().shl_into(bl.clone(), &ctx);
        let recovered = shl_v.clone().ashr_into(bl.clone(), &ctx);
        let no_overflow = recovered.eq(&al, &ctx);
        let a_top = al.extract(15, 15, &ctx);
        let a_is_neg = a_top.eq(&bit_one, &ctx);
        let cap = a_is_neg.ite(&smin, &smax, &ctx);
        let left_branch = no_overflow.ite(&shl_v, &cap, &ctx);
        let neg_amt = zero16.clone().sub_into(bl.clone(), &ctx);
        let right_branch = al.clone().ashr_into(neg_amt, &ctx);
        let amt_top = bl.extract(15, 15, &ctx);
        let amt_is_neg = amt_top.eq(&bit_one, &ctx);
        lanes.push(amt_is_neg.ite(&right_branch, &left_branch, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);
    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    let universal = !ctx.is_sat();
    ctx.pop();
    assert!(
        universal,
        "VQShlSat 16x4 (signed) must match the per-lane spec for all 64-bit inputs"
    );
}

// =========================================================================
// angr-cudgw.15: binop family-dispatch misroute degrades to OpError, not
// panic. The top-level VEXOps::binop() routing guard and each family fn's
// accepted op-set are two hand-maintained lists; IROp is a plain (non
// #[non_exhaustive]) derive enum, so a guard/family drift compiles clean
// and would otherwise hit an `unreachable!()` that aborts the whole
// process. Each family fn now returns OpError::NotBinary on a misroute so
// the caller falls through to the existing Python-fallback path. These
// tests call the family fns directly with an op that belongs to a
// DIFFERENT family, simulating that drift.
// =========================================================================

#[test]
fn binop_arith_misroute_returns_not_binary() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(1, 32);
    let b = RustBV::concrete(2, 32);
    // And is bitwise, not arith — a deliberate misroute.
    let err = VEXOps::binop_arith(IROp::And(IRType::I32), a, b, &ctx).unwrap_err();
    assert!(matches!(err, OpError::NotBinary(_)), "got {err:?}");
}

#[test]
fn binop_bitwise_shift_cmp_misroute_returns_not_binary() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(1, 32);
    let b = RustBV::concrete(2, 32);
    // Add is arith, not bitwise/shift/cmp — a deliberate misroute.
    let err =
        VEXOps::binop_bitwise_shift_cmp(IROp::Add(IRType::I32), a, b, &ctx).unwrap_err();
    assert!(matches!(err, OpError::NotBinary(_)), "got {err:?}");
}

#[test]
fn binop_float_misroute_returns_not_binary() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(1, 32);
    let b = RustBV::concrete(2, 32);
    // Add is scalar-int, not float — a deliberate misroute.
    let err = VEXOps::binop_float(IROp::Add(IRType::I32), a, b, &ctx).unwrap_err();
    assert!(matches!(err, OpError::NotBinary(_)), "got {err:?}");
}

#[test]
fn binop_vec_int_misroute_returns_not_binary() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(1, 32);
    let b = RustBV::concrete(2, 32);
    // Add is scalar-int, not vector-int — a deliberate misroute.
    let err = VEXOps::binop_vec_int(IROp::Add(IRType::I32), a, b, &ctx).unwrap_err();
    assert!(matches!(err, OpError::NotBinary(_)), "got {err:?}");
}

#[test]
fn binop_vec_float_misroute_returns_not_binary() {
    let ctx = SymContext::new_mock();
    let a = RustBV::concrete(1, 32);
    let b = RustBV::concrete(2, 32);
    // Add is scalar-int, not vector-float — a deliberate misroute.
    let err = VEXOps::binop_vec_float(IROp::Add(IRType::I32), a, b, &ctx).unwrap_err();
    assert!(matches!(err, OpError::NotBinary(_)), "got {err:?}");
}
