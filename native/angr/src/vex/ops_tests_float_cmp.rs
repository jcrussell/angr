// angr-9hleg: scalar/packed FP compare tests (mirror of ops_float_cmp.rs).

use super::ops_test_helpers::*;
use super::*;
use crate::vex::ir::FCmpKind;

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
        "Expected 1.0 < x < 2.0, got {result_f}"
    );
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
        "expected lane0==2.0, got {lane0}"
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
        "expected x < 5.0 and not NaN, got {xf}"
    );
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
        "expected lane0 < 5.0 and not NaN, got {lane0}"
    );
}
