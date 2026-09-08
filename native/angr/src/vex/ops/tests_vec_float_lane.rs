// angr-9hleg: packed per-lane FP arith tests (mirror of ops/vec_float_lane.rs).

use super::test_helpers::*;
use super::*;
use crate::vex::ir::IRType;

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

/// Iop_Abs32Fx2-style: the NEON D-reg shape (64-bit total), which had no
/// parse arm until angr-sqfj8.115. Only the low two lanes are meaningful.
#[test]
fn test_vec_float_abs_concrete_f32x2_is_64_bit() {
    let ctx = SymContext::new_mock();

    let v = [-1.5f32, -0.0];
    let exp = [1.5f32, 0.0];

    let result = VEXOps::unop(
        IROp::VFAbs {
            elem: IRType::F32,
            count: 2,
        },
        RustBV::concrete(pack_lanes_f32(&v), 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_f32_lanes_bits(result.as_u128().unwrap(), &exp);
}

/// Iop_Neg32Fx4-style: per-lane sign flip. Checked bit-exactly because negate
/// must flip the sign of -0.0 and of infinities, where `==` would not notice.
#[test]
fn test_vec_float_neg_concrete_f32x4() {
    let ctx = SymContext::new_mock();

    let v = [-1.5f32, 2.5, -0.0, f32::NEG_INFINITY];
    let exp = [1.5f32, -2.5, 0.0, f32::INFINITY];

    let result = VEXOps::unop(
        IROp::VFNeg {
            elem: IRType::F32,
            count: 4,
        },
        RustBV::concrete(pack_lanes_f32(&v), 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_f32_lanes_bits(result.as_u128().unwrap(), &exp);
}

/// Iop_Neg64Fx2-style: per-lane sign flip on the f64x2 shape.
#[test]
fn test_vec_float_neg_concrete_f64x2() {
    let ctx = SymContext::new_mock();

    let v = [3.25f64, -0.0];
    let exp = [-3.25f64, 0.0];

    let result = VEXOps::unop(
        IROp::VFNeg {
            elem: IRType::F64,
            count: 2,
        },
        RustBV::concrete(pack_lanes_f64(&v), 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_f64_lanes_approx(result.as_u128().unwrap(), &exp, 1e-12);
}

/// Iop_Neg32Fx2-style: the NEON D-reg shape, symbolic operand. Pins that the
/// Z3 path builds a 64-bit (not 128-bit) result and negates each lane.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_float_neg_symbolic_f32x2() {
    let ctx = SymContext::new_mock();

    let target = pack_lanes_f32(&[1.5f32, -4.0]);
    let v = RustBV::symbolic(&ctx, "vfneg_v", 64);
    ctx.add_constraint(v.to_z3_ast().eq(RustBV::concrete(target, 64).to_z3_ast()));

    let result = VEXOps::unop(
        IROp::VFNeg {
            elem: IRType::F32,
            count: 2,
        },
        v,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert!(ctx.is_sat(), "expected SAT");

    let model = ctx.eval(&result).expect("eval(result) returned None");
    assert_f32_lanes_approx(model, &[-1.5f32, 4.0], 1e-6);
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

// =========================================================================
// Over-arity hardening (angr-j60q0.2) — mirror of tests_vec_int_lane.rs's
// `OverArityLaneOp` / `test_vec_int_lane_over_arity_returns_error`.
// =========================================================================

/// A `FloatLaneOp` whose `arity()` exceeds `FLOAT_LANE_OP_MAX_ARITY`. No such
/// op exists in the dispatch table today; this stands in for a future one
/// (e.g. a fused multiply-add) added without widening the const, which is
/// exactly the case the fixed-size `buf32`/`buf64` arrays in
/// `VEXOps::vec_float_lane_op` cannot index.
struct OverArityFloatLaneOp;

impl FloatLaneOp for OverArityFloatLaneOp {
    fn arity(&self) -> usize {
        FLOAT_LANE_OP_MAX_ARITY + 1
    }
    fn concrete_f32(&self, a: &[f32]) -> f32 {
        a[0]
    }
    fn concrete_f64(&self, a: &[f64]) -> f64 {
        a[0]
    }
    fn symbolic(
        &self,
        args: Vec<RustBV>,
        _prec: crate::symbolic::FloatPrec,
        _ctx: &SymContext,
    ) -> RustBV {
        args[0].clone()
    }
}

/// The concrete fast path must return a typed error rather than index
/// `buf32`/`buf64` out of bounds. `debug_assert!` alone would not catch it:
/// the shipped build (and `cargo test --release`) has debug assertions off, so
/// the guard has to be a real `if`.
#[test]
fn test_vec_float_lane_over_arity_returns_error() {
    let ctx = SymContext::new_mock();
    let args = [
        RustBV::concrete(pack_4xf32(1.0, 2.0, 3.0, 4.0), 128),
        RustBV::concrete(pack_4xf32(1.0, 2.0, 3.0, 4.0), 128),
        RustBV::concrete(pack_4xf32(1.0, 2.0, 3.0, 4.0), 128),
    ];
    let err = VEXOps::vec_float_lane_op(&args, IRType::F32, 4, &OverArityFloatLaneOp, &ctx)
        .expect_err("over-arity op must not be dispatched");
    match err {
        OpError::UnsupportedVectorOp(msg) => {
            assert!(
                msg.contains("vec_float_lane_op arity 3 exceeds max 2"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("expected UnsupportedVectorOp, got {other:?}"),
    }
}
