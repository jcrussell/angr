// angr-9hleg: float/int conversion + round-to-int tests (mirror of ops_conversions.rs).

use super::*;

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
