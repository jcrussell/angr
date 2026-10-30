// angr-9hleg: float/int conversion + round-to-int tests (mirror of ops/conversions.rs).

use crate::symbolic::ToZ3;
use super::*;

// =========================================================================
// FP edge cases (angr-io1t)
// =========================================================================

/// RoundF32toInt with symbolic rm in 0..=4: value=-2.5f32, target=-3.0f32
/// forces rm to 1 (toward -inf) or 4 (ties away from zero). Exercises the
/// `VexRm` ITE built by `value_clarirs::with_vex_rm`.
#[test]
fn test_round_f32_to_int_symbolic_rm() {
    let ctx = SymContext::new_mock();
    let rm = RustBV::symbolic(&ctx, "rm_f32", 32);
    ctx.assume_true(&rm.ule(&RustBV::concrete(4, 32), &ctx));
    let value = RustBV::concrete((-2.5f32).to_bits() as u128, 32);

    let result = VEXOps::binop(IROp::RoundF32toInt, rm.clone(), value, &ctx).unwrap();
    let target = RustBV::concrete((-3.0f32).to_bits() as u128, 32);
    let eq = result.to_z3_ast().unwrap().eq(target.to_z3_ast().unwrap());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT for round(-2.5)==-3.0");

    let model_rm = ctx.eval(&rm).expect("eval(rm) returned None");
    assert!(
        model_rm == 1 || model_rm == 4,
        "expected rm 1 (toward -inf) or 4 (ties away), got {model_rm}"
    );
}

/// RoundF64toInt with symbolic rm in 0..=4: value=2.5f64, target=3.0f64
/// forces rm to 2 (toward +inf) or 4 (ties away). RNE on 2.5 ties to even (2).
#[test]
fn test_round_f64_to_int_symbolic_rm() {
    let ctx = SymContext::new_mock();
    let rm = RustBV::symbolic(&ctx, "rm_f64", 32);
    ctx.assume_true(&rm.ule(&RustBV::concrete(4, 32), &ctx));
    let value = RustBV::concrete(2.5f64.to_bits() as u128, 64);

    let result = VEXOps::binop(IROp::RoundF64toInt, rm.clone(), value, &ctx).unwrap();
    let target = RustBV::concrete(3.0f64.to_bits() as u128, 64);
    let eq = result.to_z3_ast().unwrap().eq(target.to_z3_ast().unwrap());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT for round(2.5)==3.0");

    let model_rm = ctx.eval(&rm).expect("eval(rm) returned None");
    assert!(
        model_rm == 2 || model_rm == 4,
        "expected rm 2 (toward +inf) or 4 (ties away), got {model_rm}"
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
        "1e300 → +inf, got {f}"
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

// =========================================================================
// F64→F32 with an explicit rounding mode (angr-c7xno.85)
//
// The binop form `Iop_F64toF32(rm, value)` is the one the interpreter
// actually reaches (`eval_binop` → `VEXOps::binop`); the unop form above is
// the RNE-implicit stub. The concrete path used to ignore `rm` entirely
// (Rust's `as f32` cast is hard-wired to round-nearest-ties-to-even), so
// every assertion below that names a directed mode is a regression test.
// =========================================================================

/// Drive the real dispatch path: `binop(F64toF32, rm, value)`.
fn f64_to_f32_with_rm(v: f64, rm: u32, ctx: &SymContext) -> f32 {
    let result = VEXOps::binop(
        IROp::F64toF32,
        RustBV::concrete(rm as u128, 32),
        RustBV::concrete(v.to_bits() as u128, 64),
        ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 32);
    f32::from_bits(result.as_u64().unwrap() as u32)
}

/// A value 3/4 of the way from one f32 to its successor: RNE rounds *up* to
/// the successor, so RZ/RD (which must round down) disagree with the cast.
#[test]
fn test_f64_to_f32_rm_straddles_boundary() {
    let ctx = SymContext::new_mock();
    let lo = 1.0f32;
    let hi = lo.next_up();
    let v = f64::from(lo) + 0.75 * (f64::from(hi) - f64::from(lo));

    assert_eq!(f64_to_f32_with_rm(v, 0, &ctx), hi, "RNE picks the nearer hi");
    assert_eq!(f64_to_f32_with_rm(v, 1, &ctx), lo, "toward -inf must give lo");
    assert_eq!(f64_to_f32_with_rm(v, 2, &ctx), hi, "toward +inf must give hi");
    assert_eq!(f64_to_f32_with_rm(v, 3, &ctx), lo, "toward zero must give lo");
}

/// Same boundary, negated: toward-zero now means rounding *up* (toward -lo),
/// which is the opposite ULP step from the positive case.
#[test]
fn test_f64_to_f32_rm_straddles_boundary_negative() {
    let ctx = SymContext::new_mock();
    let lo = 1.0f32;
    let hi = lo.next_up();
    let v = -(f64::from(lo) + 0.75 * (f64::from(hi) - f64::from(lo)));

    assert_eq!(f64_to_f32_with_rm(v, 0, &ctx), -hi, "RNE picks the nearer -hi");
    assert_eq!(f64_to_f32_with_rm(v, 1, &ctx), -hi, "toward -inf must give -hi");
    assert_eq!(f64_to_f32_with_rm(v, 2, &ctx), -lo, "toward +inf must give -lo");
    assert_eq!(f64_to_f32_with_rm(v, 3, &ctx), -lo, "toward zero must give -lo");
}

/// Exactly-representable values are rounding-mode independent.
#[test]
fn test_f64_to_f32_rm_exact_is_mode_independent() {
    let ctx = SymContext::new_mock();
    for rm in 0..4 {
        assert_eq!(f64_to_f32_with_rm(0.5f64, rm, &ctx), 0.5f32, "rm={rm}");
        assert_eq!(f64_to_f32_with_rm(-0.0f64, rm, &ctx), -0.0f32, "rm={rm}");
        assert!(
            f64_to_f32_with_rm(f64::NAN, rm, &ctx).is_nan(),
            "NaN stays NaN under rm={rm}"
        );
        assert_eq!(
            f64_to_f32_with_rm(f64::NEG_INFINITY, rm, &ctx),
            f32::NEG_INFINITY,
            "rm={rm}"
        );
    }
}

/// IEEE-754 overflow: the directed modes that round toward the finite side
/// must saturate at f32::MAX instead of producing infinity.
#[test]
fn test_f64_to_f32_rm_overflow_saturates() {
    let ctx = SymContext::new_mock();
    assert_eq!(f64_to_f32_with_rm(1e300, 0, &ctx), f32::INFINITY);
    assert_eq!(f64_to_f32_with_rm(1e300, 1, &ctx), f32::MAX, "toward -inf");
    assert_eq!(f64_to_f32_with_rm(1e300, 2, &ctx), f32::INFINITY, "toward +inf");
    assert_eq!(f64_to_f32_with_rm(1e300, 3, &ctx), f32::MAX, "toward zero");
    assert_eq!(f64_to_f32_with_rm(-1e300, 3, &ctx), f32::MIN, "toward zero");
    assert_eq!(f64_to_f32_with_rm(-1e300, 2, &ctx), f32::MIN, "toward +inf");
}

/// IEEE-754 underflow: rounding away from zero must reach the smallest
/// subnormal rather than flushing to zero.
#[test]
fn test_f64_to_f32_rm_underflow() {
    let ctx = SymContext::new_mock();
    let tiny = 1e-60f64; // far below f32's smallest subnormal (~1.4e-45)
    let min_subnormal = f32::from_bits(1);
    assert_eq!(f64_to_f32_with_rm(tiny, 0, &ctx), 0.0f32);
    assert_eq!(f64_to_f32_with_rm(tiny, 1, &ctx), 0.0f32, "toward -inf");
    assert_eq!(
        f64_to_f32_with_rm(tiny, 2, &ctx),
        min_subnormal,
        "toward +inf"
    );
    assert_eq!(f64_to_f32_with_rm(tiny, 3, &ctx), 0.0f32, "toward zero");
    assert_eq!(
        f64_to_f32_with_rm(-tiny, 1, &ctx),
        -min_subnormal,
        "toward -inf"
    );
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

// =========================================================================
// Rounding-mode int->float and the vector conversions (angr-mx00p.6)
// =========================================================================

/// 2^53 + 1 is not a double: nearest-even and toward-zero/-inf give 2^53,
/// toward +inf gives 2^53 + 2; the negation mirrors it.
#[test]
fn test_i64s_to_f64_rounding_modes() {
    let ctx = SymContext::new_mock();
    let conv = |v: i64, mode: u128| {
        let r = VEXOps::binop(
            IROp::I64StoF64,
            RustBV::concrete(mode, 32),
            RustBV::concrete(v as u64 as u128, 64),
            &ctx,
        )
        .unwrap();
        f64::from_bits(r.as_u128().unwrap() as u64)
    };
    let x = (1i64 << 53) + 1;
    let two53 = (1u64 << 53) as f64;
    assert_eq!(conv(x, 0), two53);
    assert_eq!(conv(x, 1), two53);
    assert_eq!(conv(x, 2), two53 + 2.0);
    assert_eq!(conv(x, 3), two53);
    assert_eq!(conv(-x, 1), -two53 - 2.0);
    assert_eq!(conv(-x, 2), -two53);
    assert_eq!(conv(-x, 3), -two53);
    // Exact values are unaffected by the mode.
    assert_eq!(conv(-7, 2), -7.0);
}

#[test]
fn test_i64u_to_f32_rounds_up_to_2_pow_64() {
    let ctx = SymContext::new_mock();
    let r = VEXOps::binop(
        IROp::I64UtoF32,
        RustBV::concrete(0, 32),
        RustBV::concrete(u64::MAX as u128, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(f32::from_bits(r.as_u128().unwrap() as u32), 18446744073709551616.0);
}

/// The symbolic `ConvertItoFRm` term agrees with the concrete fold under
/// every VEX rounding mode.
#[test]
fn test_int_to_float_rm_symbolic_matches_concrete() {
    use crate::symbolic::FloatPrec;
    let ctx = SymContext::new_mock();
    let x = ((1i64 << 53) + 1).wrapping_neg() as u64 as u128;
    for mode in 0..4u128 {
        let concrete = VEXOps::binop(
            IROp::I64StoF64,
            RustBV::concrete(mode, 32),
            RustBV::concrete(x, 64),
            &ctx,
        )
        .unwrap();
        let term = build_float_expr(
            FloatOpKind::ConvertItoFRm {
                src_bits: 64,
                signed: true,
            },
            FloatPrec::F64,
            vec![RustBV::concrete(mode, 32), RustBV::concrete(x, 64)],
        );
        assert_eq!(ctx.eval(&term), concrete.as_u128(), "mode {mode}");
    }
}

/// cvtdq2ps: four signed int lanes to F32, lane 0 lowest.
#[test]
fn test_vec_i32s_to_f32x4() {
    let ctx = SymContext::new_mock();
    let lanes: [i32; 4] = [1, -2, 1 << 24 | 1, 7];
    let v = lanes
        .iter()
        .rev()
        .fold(0u128, |acc, l| (acc << 32) | (*l as u32 as u128));
    let r = VEXOps::binop(
        IROp::VI32toF32 {
            count: 4,
            signed: true,
        },
        RustBV::concrete(0, 32),
        RustBV::concrete(v, 128),
        &ctx,
    )
    .unwrap()
    .as_u128()
    .unwrap();
    let lane = |i: u32| f32::from_bits((r >> (32 * i)) as u32);
    assert_eq!(
        [lane(0), lane(1), lane(2), lane(3)],
        [1.0, -2.0, 16777216.0, 7.0]
    );
}

/// cvtps2dq rounds by the passed mode; the `_RZ` unop form truncates.
#[test]
fn test_vec_f32_to_i32s_x4_modes() {
    let ctx = SymContext::new_mock();
    let lanes: [f32; 4] = [1.5, -1.5, 2.5, -0.4];
    let v = lanes
        .iter()
        .rev()
        .fold(0u128, |acc, l| (acc << 32) | (l.to_bits() as u128));
    let op = IROp::VF32toI32 {
        count: 4,
        signed: true,
    };
    let unpack = |r: RustBV| {
        let r = r.as_u128().unwrap();
        [0, 1, 2, 3].map(|i| (r >> (32 * i)) as u32 as i32)
    };
    let nearest = VEXOps::binop(op, RustBV::concrete(0, 32), RustBV::concrete(v, 128), &ctx).unwrap();
    assert_eq!(unpack(nearest), [2, -2, 2, 0]);
    let rz = VEXOps::unop(op, RustBV::concrete(v, 128), &ctx).unwrap();
    assert_eq!(unpack(rz), [1, -1, 2, 0]);
}

#[test]
fn test_vec_conversions_check_width_and_map() {
    let ctx = SymContext::new_mock();
    assert!(
        VEXOps::unop(
            IROp::VI32toF32 {
                count: 4,
                signed: true
            },
            RustBV::concrete(0, 64),
            &ctx
        )
        .is_err()
    );
    assert_eq!(
        crate::vex::parse_opcode("Iop_I32StoF32x4"),
        IROp::VI32toF32 {
            count: 4,
            signed: true
        }
    );
    assert_eq!(
        crate::vex::parse_opcode("Iop_F32toI32Ux2_RZ"),
        IROp::VF32toI32 {
            count: 2,
            signed: false
        }
    );
}
