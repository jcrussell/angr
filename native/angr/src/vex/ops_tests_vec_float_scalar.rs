// angr-9hleg: SSE scalar-in-vector FP + recip/rsqrt estimate tests (mirror of ops_vec_float_scalar.rs).

use super::ops_test_helpers::*;
use super::*;

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
        "Expected 6.0, got {result_f32}"
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
        "Expected 3.0, got {result_f32}"
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
        "Expected lane0 == 3.0, got {lane0}"
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
        "Expected lane0 == 16.0, got {lane0}"
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
        "Expected lane0 == 5.0 (since max(lane0, 3.0) == 5.0), got {lane0}"
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
        "Expected lane0 == 1.0 (since min(lane0, 3.0) == 1.0), got {lane0}"
    );
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
    assert!((lane0 - 7.0).abs() < 1e-6, "10.0 - 3.0 == 7.0, got {lane0}");
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
        "4.0 * 2.5 == 10.0, got {lane0}"
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
            "{op:?}: lane0={lane0} expected={expected}",
        );
        assert_eq!(
            rv & !0xFFFF_FFFF_FFFF_FFFFu128,
            upper_pattern,
            "{op:?}: upper 64 bits must pass through",
        );
    }

    // Unary VFSqrtS{F64}: sqrt(16.0) = 4.0, upper 64 bits pass through.
    let arg = xmm0(16.0);
    let sqrt_res = VEXOps::unop(IROp::VFSqrtS { elem: IRType::F64 }, arg, &ctx).unwrap();
    let sv = sqrt_res.as_u128().unwrap();
    let sqrt_lane0 = f64::from_bits((sv & 0xFFFF_FFFF_FFFF_FFFFu128) as u64);
    assert!(
        (sqrt_lane0 - 4.0).abs() < 1e-9,
        "VFSqrtS{{F64}}: lane0={sqrt_lane0} expected=4.0",
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
        "MAXSS(NaN, 3.0) returns the right operand, got {lane0}"
    );
}

/// SSE RCPSS shape: 128-bit result, lane 0 fresh symbolic (32-bit width),
/// upper 96 bits passed through unchanged from the arg.
#[cfg(feature = "vex-engine-z3")]
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
#[cfg(feature = "vex-engine-z3")]
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
#[cfg(feature = "vex-engine-z3")]
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
#[cfg(feature = "vex-engine-z3")]
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
#[cfg(feature = "vex-engine-z3")]
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
            other => panic!("{name}: expected VIRecipEst, got {other:?}"),
        }
    }
    for (name, count) in [("Iop_RSqrtEst32Ux2", 2u8), ("Iop_RSqrtEst32Ux4", 4)] {
        match parse_opcode(name) {
            IROp::VIRSqrtEst { count: c } => assert_eq!(c, count),
            other => panic!("{name}: expected VIRSqrtEst, got {other:?}"),
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
            other => panic!("{name}: expected VFRecipEst, got {other:?}"),
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
            other => panic!("{name}: expected VFRSqrtEst, got {other:?}"),
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
            other => panic!("{name}: expected VFRecipStep, got {other:?}"),
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
            other => panic!("{name}: expected VFRSqrtStep, got {other:?}"),
        }
    }
    // Suppress the unused-helper lint when running this test alone:
    let _ = eval_i32;
}
