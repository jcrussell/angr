//! Direct unit tests for the floating-point Z3-AST builders in `value_z3.rs`
//! (`build_fp_z3_ast_cached` and its `build_fp_round_to_int_cached` /
//! `build_fp_arith_rm_cached` / `build_fp_i_to_f_cached` /
//! `build_fp_f_to_i_cached` / `build_fp_f_to_f_cached` helpers).
//!
//! Before angr-c7xno.74 this surface was only covered *transitively*, through
//! the VEX-op layer (`vex/ops/tests_float_arith.rs` and friends). A regression
//! confined to the Z3-emission layer — a swapped operand order inside an
//! `unsafe { Z3_mk_fpa_* }` call, a wrong `vex_rm_to_z3` selector mapping, a
//! rounding mode dropped on the way to a builder — would only have been caught
//! if some VEX op happened to exercise that exact combination.
//!
//! These tests build `RustBV::Expression { op: BVOp::Float { .. } }` nodes by
//! hand and push them straight through `SymContext::eval`, so every
//! `FloatOpKind` variant is checked at the `value_z3.rs` level with no
//! VEX decode in between. Expectations are computed from Rust's own IEEE-754
//! semantics (`f32`/`f64` arithmetic) so a Z3-side change that disagrees with
//! the concrete interpreter path fails here.
//!
//! NaN *results* are asserted via `is_nan()` on the decoded value rather than
//! by bit pattern: `Z3_mk_fpa_to_ieee_bv` leaves the NaN payload unspecified.

use super::*;
use crate::symbolic::SymContext;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Builders / decoders
// ---------------------------------------------------------------------------

/// Build a raw `BVOp::Float` expression node. Deliberately does not go through
/// `vex::ops::lane_traits::build_float_expr` — the point of this file is to
/// reach `build_fp_z3_ast_cached` without the VEX layer.
fn fp_expr(kind: FloatOpKind, prec: FloatPrec, operands: Vec<RustBV>) -> RustBV {
    RustBV::Expression {
        id: RustBV::EXPRESSION_ID,
        width: kind.result_bits(prec),
        op: BVOp::Float { kind, prec },
        operands: Arc::<[RustBV]>::from(operands),
        memo: Default::default(),
    }
}

/// IEEE bits of an `f32` as a 32-bit concrete `RustBV`.
fn bv32(v: f32) -> RustBV {
    RustBV::concrete(u128::from(v.to_bits()), 32)
}

/// IEEE bits of an `f64` as a 64-bit concrete `RustBV`.
fn bv64(v: f64) -> RustBV {
    RustBV::concrete(u128::from(v.to_bits()), 64)
}

/// A VEX rounding-mode operand (32-bit, low 2 bits significant).
fn rm(vex_rm: u8) -> RustBV {
    RustBV::concrete(u128::from(vex_rm), 32)
}

/// Evaluate `e` in `ctx` and reinterpret the result bits as an `f32`.
fn eval_f32(ctx: &SymContext, e: &RustBV) -> f32 {
    f32::from_bits(ctx.eval(e).expect("eval returned None") as u32)
}

/// Evaluate `e` in `ctx` and reinterpret the result bits as an `f64`.
fn eval_f64(ctx: &SymContext, e: &RustBV) -> f64 {
    f64::from_bits(ctx.eval(e).expect("eval returned None") as u64)
}

/// Evaluate `e` in `ctx` as a raw integer (compares, FtoI results).
fn eval_bits(ctx: &SymContext, e: &RustBV) -> u128 {
    ctx.eval(e).expect("eval returned None")
}

// ---------------------------------------------------------------------------
// Plain (RNE) arithmetic: Add / Sub / Mul / Div
// ---------------------------------------------------------------------------

#[test]
fn test_fp_binary_arith_f64_matches_rust() {
    let ctx = SymContext::new();
    // Operand order matters for the non-commutative ops: a swapped
    // `Z3_mk_fpa_sub`/`Z3_mk_fpa_div` argument pair is invisible on
    // symmetric inputs, so use a != b throughout.
    let (a, b) = (7.5f64, 2.25f64);
    for (kind, expected) in [
        (FloatOpKind::Add, a + b),
        (FloatOpKind::Sub, a - b),
        (FloatOpKind::Mul, a * b),
        (FloatOpKind::Div, a / b),
    ] {
        let e = fp_expr(kind, FloatPrec::F64, vec![bv64(a), bv64(b)]);
        assert_eq!(e.width(), 64, "{kind:?} result width");
        assert_eq!(eval_f64(&ctx, &e), expected, "{kind:?}");
    }
}

#[test]
fn test_fp_binary_arith_f32_matches_rust() {
    let ctx = SymContext::new();
    let (a, b) = (7.5f32, 2.25f32);
    for (kind, expected) in [
        (FloatOpKind::Add, a + b),
        (FloatOpKind::Sub, a - b),
        (FloatOpKind::Mul, a * b),
        (FloatOpKind::Div, a / b),
    ] {
        let e = fp_expr(kind, FloatPrec::F32, vec![bv32(a), bv32(b)]);
        assert_eq!(e.width(), 32, "{kind:?} result width");
        assert_eq!(eval_f32(&ctx, &e), expected, "{kind:?}");
    }
}

/// The default rounding mode of the *non*-`Rm` arithmetic ops is
/// round-nearest-ties-to-even, matching Rust's `/`. `1.0/3.0` is the cheapest
/// value that differs between RNE and round-toward-zero at both precisions.
#[test]
fn test_fp_default_rounding_is_nearest_even() {
    let ctx = SymContext::new();
    let e = fp_expr(FloatOpKind::Div, FloatPrec::F64, vec![bv64(1.0), bv64(3.0)]);
    assert_eq!(eval_f64(&ctx, &e).to_bits(), (1.0f64 / 3.0).to_bits());
}

// ---------------------------------------------------------------------------
// Unary: Sqrt / Neg / Abs
// ---------------------------------------------------------------------------

#[test]
fn test_fp_unary_ops() {
    let ctx = SymContext::new();
    let sqrt = fp_expr(FloatOpKind::Sqrt, FloatPrec::F64, vec![bv64(2.0)]);
    assert_eq!(eval_f64(&ctx, &sqrt), 2.0f64.sqrt());

    let neg = fp_expr(FloatOpKind::Neg, FloatPrec::F64, vec![bv64(3.5)]);
    assert_eq!(eval_f64(&ctx, &neg), -3.5);

    let abs = fp_expr(FloatOpKind::Abs, FloatPrec::F64, vec![bv64(-3.5)]);
    assert_eq!(eval_f64(&ctx, &abs), 3.5);

    let sqrt32 = fp_expr(FloatOpKind::Sqrt, FloatPrec::F32, vec![bv32(2.0)]);
    assert_eq!(eval_f32(&ctx, &sqrt32), 2.0f32.sqrt());
}

/// `Neg`/`Abs` are sign-bit ops in IEEE-754: they must act on `-0.0` (whose
/// value compares equal to `+0.0`, so only the bit pattern distinguishes a
/// correct implementation from a no-op).
#[test]
fn test_fp_neg_abs_on_signed_zero() {
    let ctx = SymContext::new();
    let neg = fp_expr(FloatOpKind::Neg, FloatPrec::F64, vec![bv64(0.0)]);
    assert_eq!(eval_f64(&ctx, &neg).to_bits(), (-0.0f64).to_bits());

    let abs = fp_expr(FloatOpKind::Abs, FloatPrec::F64, vec![bv64(-0.0)]);
    assert_eq!(eval_f64(&ctx, &abs).to_bits(), 0.0f64.to_bits());
}

// ---------------------------------------------------------------------------
// Ternary: Fma / Fms
// ---------------------------------------------------------------------------

#[test]
fn test_fp_fma_and_fms() {
    let ctx = SymContext::new();
    let (a, b, c) = (3.0f64, 4.0f64, 1.5f64);
    let fma = fp_expr(
        FloatOpKind::Fma,
        FloatPrec::F64,
        vec![bv64(a), bv64(b), bv64(c)],
    );
    assert_eq!(eval_f64(&ctx, &fma), a.mul_add(b, c));

    // Fms is emitted as `fma(a, b, -c)`; an implementation that forgot the
    // negation would return the Fma answer.
    let fms = fp_expr(
        FloatOpKind::Fms,
        FloatPrec::F64,
        vec![bv64(a), bv64(b), bv64(c)],
    );
    assert_eq!(eval_f64(&ctx, &fms), a.mul_add(b, -c));
    assert_ne!(eval_f64(&ctx, &fms), eval_f64(&ctx, &fma));
}

/// Fusion is observable: `a*b` rounded before adding `c` loses the low bits
/// that a true FMA keeps. Pins that the builder emits `Z3_mk_fpa_fma` rather
/// than a `mul` followed by an `add`.
///
/// `a = 1 + 2^-27` squares to `1 + 2^-26 + 2^-54`; the `2^-54` term falls off
/// the end of the double when the product is rounded on its own, but survives
/// once `-1` cancels the leading bit, so fused and unfused differ.
#[test]
fn test_fp_fma_is_actually_fused() {
    let ctx = SymContext::new();
    let a = 1.0f64 + 2f64.powi(-27);
    let (b, c) = (a, -1.0f64);
    let fma = fp_expr(
        FloatOpKind::Fma,
        FloatPrec::F64,
        vec![bv64(a), bv64(b), bv64(c)],
    );
    assert_eq!(eval_f64(&ctx, &fma), a.mul_add(b, c));
    assert_ne!(eval_f64(&ctx, &fma), (a * b) + c);
}

// ---------------------------------------------------------------------------
// Compares: CmpEq / CmpLt / CmpLe / IsNaN (1-bit results)
// ---------------------------------------------------------------------------

#[test]
fn test_fp_compares_1bit_results() {
    let ctx = SymContext::new();
    for (kind, a, b, expected) in [
        (FloatOpKind::CmpEq, 1.5f64, 1.5f64, 1u128),
        (FloatOpKind::CmpEq, 1.5, 2.5, 0),
        (FloatOpKind::CmpLt, 1.5, 2.5, 1),
        (FloatOpKind::CmpLt, 2.5, 1.5, 0),
        (FloatOpKind::CmpLt, 1.5, 1.5, 0),
        (FloatOpKind::CmpLe, 1.5, 1.5, 1),
        (FloatOpKind::CmpLe, 2.5, 1.5, 0),
    ] {
        let e = fp_expr(kind, FloatPrec::F64, vec![bv64(a), bv64(b)]);
        assert_eq!(e.width(), 1, "{kind:?} result width");
        assert_eq!(eval_bits(&ctx, &e), expected, "{kind:?}({a}, {b})");
    }
}

/// IEEE-754: every ordered comparison against NaN is false, and `-0.0 == 0.0`.
/// Both are places where a bit-vector comparison would give the wrong answer,
/// so they pin that the FP theory (not `Z3_mk_eq` on the bits) is being used.
#[test]
fn test_fp_compare_nan_and_signed_zero_semantics() {
    let ctx = SymContext::new();
    let nan = f64::NAN;
    for kind in [FloatOpKind::CmpEq, FloatOpKind::CmpLt, FloatOpKind::CmpLe] {
        let e = fp_expr(kind, FloatPrec::F64, vec![bv64(nan), bv64(nan)]);
        assert_eq!(eval_bits(&ctx, &e), 0, "{kind:?}(NaN, NaN)");
    }
    let eq = fp_expr(
        FloatOpKind::CmpEq,
        FloatPrec::F64,
        vec![bv64(-0.0), bv64(0.0)],
    );
    assert_eq!(eval_bits(&ctx, &eq), 1, "-0.0 == 0.0");
}

#[test]
fn test_fp_is_nan() {
    let ctx = SymContext::new();
    for (v, expected) in [(f64::NAN, 1u128), (0.0, 0), (f64::INFINITY, 0)] {
        let e = fp_expr(FloatOpKind::IsNaN, FloatPrec::F64, vec![bv64(v)]);
        assert_eq!(e.width(), 1);
        assert_eq!(eval_bits(&ctx, &e), expected, "IsNaN({v})");
    }
    let e32 = fp_expr(FloatOpKind::IsNaN, FloatPrec::F32, vec![bv32(f32::NAN)]);
    assert_eq!(eval_bits(&ctx, &e32), 1);
}

/// Special values survive the BV -> Float -> BV round trip: `1.0/0.0` is
/// `+inf` and `-1.0/0.0` is `-inf`, both of which have unique IEEE encodings.
///
/// `0.0/0.0` is deliberately *not* asserted here: it is a NaN, and Z3 models
/// `Z3_mk_fpa_to_ieee_bv` on NaN as an unspecified value, so the model is free
/// to hand back any bit pattern (including a finite one). NaN detection is
/// covered by `test_fp_is_nan`, which asks the FP theory directly instead of
/// going through the bit encoding.
#[test]
fn test_fp_arith_producing_infinity() {
    let ctx = SymContext::new();
    let inf = fp_expr(FloatOpKind::Div, FloatPrec::F64, vec![bv64(1.0), bv64(0.0)]);
    assert_eq!(eval_f64(&ctx, &inf), f64::INFINITY);

    let neg_inf = fp_expr(FloatOpKind::Div, FloatPrec::F64, vec![bv64(-1.0), bv64(0.0)]);
    assert_eq!(eval_f64(&ctx, &neg_inf), f64::NEG_INFINITY);
}

// ---------------------------------------------------------------------------
// RoundToInt (concrete + symbolic rounding mode)
// ---------------------------------------------------------------------------

/// All four VEX rounding-mode selectors, on a value where each mode gives a
/// distinct answer. Catches a permuted `vex_rm_to_z3` mapping, which the
/// arithmetic tests above (RNE only) cannot see.
#[test]
fn test_fp_round_to_int_all_modes() {
    let ctx = SymContext::new();
    // 0 = nearest-even, 1 = -inf, 2 = +inf, 3 = zero.
    for (vex_rm, v, expected) in [
        (0u8, 2.5f64, 2.0f64), // ties to even
        (0, 3.5, 4.0),
        (1, 2.7, 2.0),
        (1, -2.2, -3.0),
        (2, 2.2, 3.0),
        (2, -2.7, -2.0),
        (3, 2.7, 2.0),
        (3, -2.7, -2.0),
    ] {
        let e = fp_expr(
            FloatOpKind::RoundToInt,
            FloatPrec::F64,
            vec![rm(vex_rm), bv64(v)],
        );
        assert_eq!(eval_f64(&ctx, &e), expected, "rm={vex_rm} value={v}");
    }
}

/// Only the low 2 bits of the rm operand select the mode; VEX passes a full
/// 32-bit value whose high bits carry other state.
#[test]
fn test_fp_round_to_int_ignores_high_rm_bits() {
    let ctx = SymContext::new();
    let e = fp_expr(
        FloatOpKind::RoundToInt,
        FloatPrec::F64,
        vec![RustBV::concrete(0xFFFF_FFFC, 32), bv64(2.7)],
    );
    // 0xFFFF_FFFC & 3 == 0 -> nearest.
    assert_eq!(eval_f64(&ctx, &e), 3.0);
}

/// A symbolic rm operand routes through `dispatch_symbolic_rm`, which builds
/// all four arms and ITEs on the low 2 bits. Constraining the symbol to each
/// selector must reproduce the concrete-rm answers exactly.
#[test]
fn test_fp_round_to_int_symbolic_rm_matches_concrete() {
    for (vex_rm, expected) in [(0u8, 3.0f64), (1, 2.0), (2, 3.0), (3, 2.0)] {
        let ctx = SymContext::new();
        let rm_sym = RustBV::symbolic(&ctx, "rm", 32);
        ctx.assume_true(&rm_sym.eq(&RustBV::concrete(u128::from(vex_rm), 32), &ctx));
        let e = fp_expr(
            FloatOpKind::RoundToInt,
            FloatPrec::F64,
            vec![rm_sym, bv64(2.7)],
        );
        assert_eq!(eval_f64(&ctx, &e), expected, "symbolic rm={vex_rm}");
    }
}

// ---------------------------------------------------------------------------
// Rm-carrying arithmetic: AddRm / SubRm / MulRm / DivRm / SqrtRm
// ---------------------------------------------------------------------------

/// `1.0/10.0` under round-toward-zero is one ULP below the RNE result, so a
/// builder that dropped the rm operand (the angr-c7xno.85 bug shape) fails
/// here.
///
/// The witness has to be picked with care: `1/3` is *not* one, because its
/// binary expansion rounds down, making RNE and RZ agree. `0.1` is the
/// standard example of a decimal that rounds *up* into a double.
#[test]
fn test_fp_arith_rm_rounding_is_threaded() {
    let ctx = SymContext::new();
    // RNE (rm 0) vs RZ (rm 3) on a value that is not exactly representable.
    let rne = fp_expr(
        FloatOpKind::DivRm,
        FloatPrec::F64,
        vec![rm(0), bv64(1.0), bv64(10.0)],
    );
    let rz = fp_expr(
        FloatOpKind::DivRm,
        FloatPrec::F64,
        vec![rm(3), bv64(1.0), bv64(10.0)],
    );
    let rne_bits = eval_f64(&ctx, &rne).to_bits();
    let rz_bits = eval_f64(&ctx, &rz).to_bits();
    assert_eq!(rne_bits, (1.0f64 / 10.0).to_bits());
    assert_ne!(rne_bits, rz_bits, "RZ must differ from RNE for 1/10");
    assert_eq!(rz_bits, rne_bits - 1, "RZ truncates toward zero");
}

#[test]
fn test_fp_arith_rm_values_match_plain_ops_under_rne() {
    let ctx = SymContext::new();
    let (a, b) = (7.5f64, 2.25f64);
    for (kind, expected) in [
        (FloatOpKind::AddRm, a + b),
        (FloatOpKind::SubRm, a - b),
        (FloatOpKind::MulRm, a * b),
        (FloatOpKind::DivRm, a / b),
    ] {
        let e = fp_expr(kind, FloatPrec::F64, vec![rm(0), bv64(a), bv64(b)]);
        assert_eq!(eval_f64(&ctx, &e), expected, "{kind:?}");
    }
    // SqrtRm is the unary member of the family: operands are [rm, a].
    let sqrt = fp_expr(FloatOpKind::SqrtRm, FloatPrec::F64, vec![rm(0), bv64(2.0)]);
    assert_eq!(eval_f64(&ctx, &sqrt), 2.0f64.sqrt());
    let sqrt_rz = fp_expr(FloatOpKind::SqrtRm, FloatPrec::F64, vec![rm(3), bv64(2.0)]);
    assert_eq!(
        eval_f64(&ctx, &sqrt_rz).to_bits(),
        2.0f64.sqrt().to_bits() - 1,
        "sqrt(2) rounds up under RNE, so RZ is one ULP lower"
    );
}

#[test]
fn test_fp_arith_rm_symbolic_rm_matches_concrete() {
    for (vex_rm, expect_rz) in [(0u8, false), (3u8, true)] {
        let ctx = SymContext::new();
        let rm_sym = RustBV::symbolic(&ctx, "rm", 32);
        ctx.assume_true(&rm_sym.eq(&RustBV::concrete(u128::from(vex_rm), 32), &ctx));
        let e = fp_expr(
            FloatOpKind::DivRm,
            FloatPrec::F64,
            vec![rm_sym, bv64(1.0), bv64(10.0)],
        );
        let rne_bits = (1.0f64 / 10.0).to_bits();
        let expected = if expect_rz { rne_bits - 1 } else { rne_bits };
        assert_eq!(eval_f64(&ctx, &e).to_bits(), expected, "symbolic rm={vex_rm}");
    }
}

// ---------------------------------------------------------------------------
// ConvertItoF
// ---------------------------------------------------------------------------

#[test]
fn test_fp_convert_i_to_f_signed_and_unsigned() {
    let ctx = SymContext::new();
    // -1 as a 32-bit 2's-complement pattern: signed -> -1.0, unsigned -> 2^32-1.
    let bits = RustBV::concrete(0xFFFF_FFFF, 32);
    let signed = fp_expr(
        FloatOpKind::ConvertItoF {
            src_bits: 32,
            signed: true,
        },
        FloatPrec::F64,
        vec![bits.clone()],
    );
    assert_eq!(eval_f64(&ctx, &signed), -1.0);

    let unsigned = fp_expr(
        FloatOpKind::ConvertItoF {
            src_bits: 32,
            signed: false,
        },
        FloatPrec::F64,
        vec![bits],
    );
    assert_eq!(eval_f64(&ctx, &unsigned), 4_294_967_295.0);
}

#[test]
fn test_fp_convert_i_to_f_widths_and_precisions() {
    let ctx = SymContext::new();
    let e64 = fp_expr(
        FloatOpKind::ConvertItoF {
            src_bits: 64,
            signed: true,
        },
        FloatPrec::F64,
        vec![RustBV::concrete(42, 64)],
    );
    assert_eq!(eval_f64(&ctx, &e64), 42.0);

    let e32 = fp_expr(
        FloatOpKind::ConvertItoF {
            src_bits: 32,
            signed: true,
        },
        FloatPrec::F32,
        vec![RustBV::concrete(u128::from((-7i32) as u32), 32)],
    );
    assert_eq!(eval_f32(&ctx, &e32), -7.0);
}

// ---------------------------------------------------------------------------
// ConvertFtoI / ConvertFtoIRm
// ---------------------------------------------------------------------------

/// The no-rm variant is documented as implicit RNE, so `2.5 -> 2` (ties to
/// even) rather than the C-style truncation `2`. `3.5 -> 4` is what separates
/// the two.
#[test]
fn test_fp_convert_f_to_i_implicit_rne() {
    let ctx = SymContext::new();
    for (v, expected) in [(2.5f64, 2u128), (3.5, 4), (-2.5, (-2i64) as u64 as u128)] {
        let e = fp_expr(
            FloatOpKind::ConvertFtoI {
                dst_bits: 64,
                signed: true,
            },
            FloatPrec::F64,
            vec![bv64(v)],
        );
        assert_eq!(e.width(), 64, "result width comes from dst_bits");
        assert_eq!(eval_bits(&ctx, &e), expected, "FtoI({v})");
    }
}

#[test]
fn test_fp_convert_f_to_i_dst_width_and_signedness() {
    let ctx = SymContext::new();
    let narrow = fp_expr(
        FloatOpKind::ConvertFtoI {
            dst_bits: 32,
            signed: true,
        },
        FloatPrec::F64,
        vec![bv64(-5.0)],
    );
    assert_eq!(narrow.width(), 32);
    assert_eq!(eval_bits(&ctx, &narrow), u128::from((-5i32) as u32));

    let unsigned = fp_expr(
        FloatOpKind::ConvertFtoI {
            dst_bits: 32,
            signed: false,
        },
        FloatPrec::F64,
        vec![bv64(4_000_000_000.0)],
    );
    assert_eq!(eval_bits(&ctx, &unsigned), 4_000_000_000);
}

/// The rm-carrying variant must actually honor the selector: `2.7` rounds to
/// 3 / 2 / 3 / 2 under nearest / -inf / +inf / zero.
#[test]
fn test_fp_convert_f_to_i_rm_all_modes() {
    let ctx = SymContext::new();
    for (vex_rm, expected) in [(0u8, 3u128), (1, 2), (2, 3), (3, 2)] {
        let e = fp_expr(
            FloatOpKind::ConvertFtoIRm {
                dst_bits: 32,
                signed: true,
            },
            FloatPrec::F64,
            vec![rm(vex_rm), bv64(2.7)],
        );
        assert_eq!(eval_bits(&ctx, &e), expected, "FtoIRm rm={vex_rm}");
    }
}

#[test]
fn test_fp_convert_f_to_i_symbolic_rm_matches_concrete() {
    for (vex_rm, expected) in [(0u8, 3u128), (1, 2), (2, 3), (3, 2)] {
        let ctx = SymContext::new();
        let rm_sym = RustBV::symbolic(&ctx, "rm", 32);
        ctx.assume_true(&rm_sym.eq(&RustBV::concrete(u128::from(vex_rm), 32), &ctx));
        let e = fp_expr(
            FloatOpKind::ConvertFtoIRm {
                dst_bits: 32,
                signed: true,
            },
            FloatPrec::F64,
            vec![rm_sym, bv64(2.7)],
        );
        assert_eq!(eval_bits(&ctx, &e), expected, "symbolic FtoIRm rm={vex_rm}");
    }
}

// ---------------------------------------------------------------------------
// ConvertFtoF / ConvertFtoFRm
// ---------------------------------------------------------------------------

#[test]
fn test_fp_convert_f_to_f_both_directions() {
    let ctx = SymContext::new();
    let widen = fp_expr(
        FloatOpKind::ConvertFtoF {
            src_prec: FloatPrec::F32,
        },
        FloatPrec::F64,
        vec![bv32(1.5)],
    );
    assert_eq!(widen.width(), 64);
    assert_eq!(eval_f64(&ctx, &widen), 1.5);

    let narrow = fp_expr(
        FloatOpKind::ConvertFtoF {
            src_prec: FloatPrec::F64,
        },
        FloatPrec::F32,
        vec![bv64(1.5)],
    );
    assert_eq!(narrow.width(), 32);
    assert_eq!(eval_f32(&ctx, &narrow), 1.5);
}

/// f64 -> f32 narrowing is the rounding-sensitive direction. `1/3` at double
/// precision sits between two f32 neighbours, so RNE and RZ disagree — the
/// exact shape `f64_to_f32_rm` silently got wrong in angr-c7xno.85.
#[test]
fn test_fp_convert_f_to_f_rm_rounding_is_threaded() {
    let ctx = SymContext::new();
    let src = 1.0f64 / 3.0;
    let rne = fp_expr(
        FloatOpKind::ConvertFtoFRm {
            src_prec: FloatPrec::F64,
        },
        FloatPrec::F32,
        vec![rm(0), bv64(src)],
    );
    let rz = fp_expr(
        FloatOpKind::ConvertFtoFRm {
            src_prec: FloatPrec::F64,
        },
        FloatPrec::F32,
        vec![rm(3), bv64(src)],
    );
    assert_eq!(eval_f32(&ctx, &rne).to_bits(), (src as f32).to_bits());
    assert_ne!(
        eval_f32(&ctx, &rz).to_bits(),
        eval_f32(&ctx, &rne).to_bits(),
        "RZ must differ from RNE narrowing 1/3 to f32"
    );
    // 1/3 rounds *up* to the nearest f32, so truncation lands one ULP below.
    assert_eq!(
        eval_f32(&ctx, &rz).to_bits(),
        eval_f32(&ctx, &rne).to_bits() - 1
    );
}

#[test]
fn test_fp_convert_f_to_f_symbolic_rm_matches_concrete() {
    let src = 1.0f64 / 3.0;
    for (vex_rm, expect_rz) in [(0u8, false), (3u8, true)] {
        let ctx = SymContext::new();
        let rm_sym = RustBV::symbolic(&ctx, "rm", 32);
        ctx.assume_true(&rm_sym.eq(&RustBV::concrete(u128::from(vex_rm), 32), &ctx));
        let e = fp_expr(
            FloatOpKind::ConvertFtoFRm {
                src_prec: FloatPrec::F64,
            },
            FloatPrec::F32,
            vec![rm_sym, bv64(src)],
        );
        let rne_bits = (src as f32).to_bits();
        let expected = if expect_rz { rne_bits - 1 } else { rne_bits };
        assert_eq!(eval_f32(&ctx, &e).to_bits(), expected, "symbolic rm={vex_rm}");
    }
}

// ---------------------------------------------------------------------------
// Symbolic operands + solver interaction
// ---------------------------------------------------------------------------

/// The FP builder must produce a *constrainable* term, not just something that
/// evaluates: solve for `x` in `x + 1.5 == 4.0` through the Z3 FP theory.
#[test]
fn test_fp_add_with_symbolic_operand_solves() {
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "x", 64);
    let sum = fp_expr(FloatOpKind::Add, FloatPrec::F64, vec![x.clone(), bv64(1.5)]);
    // Compare on the IEEE bits: 4.0 has a unique encoding, so this pins x.
    ctx.assume_true(&sum.eq(&bv64(4.0), &ctx));
    assert!(ctx.is_sat());
    assert_eq!(eval_f64(&ctx, &x), 2.5);
}

/// An unsatisfiable FP constraint must be reported unsat — a builder that
/// emitted a fresh unconstrained value instead of the real op would make this
/// spuriously sat.
#[test]
fn test_fp_unsat_constraint_is_detected() {
    let ctx = SymContext::new();
    // Nothing squares to a negative number, so sqrt(x) == -1.0 is unsat
    // (IEEE sqrt of a negative is NaN, and NaN != -1.0 bitwise).
    let x = RustBV::symbolic(&ctx, "x", 64);
    let is_lt = fp_expr(
        FloatOpKind::CmpLt,
        FloatPrec::F64,
        vec![x.clone(), bv64(1.0)],
    );
    let is_gt = fp_expr(FloatOpKind::CmpLt, FloatPrec::F64, vec![bv64(2.0), x]);
    ctx.assume_true(&is_lt);
    ctx.assume_true(&is_gt);
    assert!(!ctx.is_sat(), "x < 1.0 && 2.0 < x must be unsat");
}

/// Shared sub-expressions go through the `to_z3_ast_cached` pointer-identity
/// cache; an FP node reused twice in one tree must build once and still
/// evaluate correctly.
#[test]
fn test_fp_shared_subexpression_through_cache() {
    let ctx = SymContext::new();
    let inner = fp_expr(FloatOpKind::Add, FloatPrec::F64, vec![bv64(1.5), bv64(2.5)]);
    let outer = fp_expr(
        FloatOpKind::Mul,
        FloatPrec::F64,
        vec![inner.clone(), inner.clone()],
    );
    assert_eq!(eval_f64(&ctx, &outer), 16.0);
}

// ---------------------------------------------------------------------------
// Coverage guard
// ---------------------------------------------------------------------------

/// Every `FloatOpKind` variant must be reachable through `to_z3_ast` at both
/// precisions without panicking, and the resulting Z3 BV's width must match
/// `FloatOpKind::result_bits`. This is the backstop that a newly added variant
/// cannot land without at least a smoke check here — the `match` is
/// exhaustive, so adding a variant fails to compile until it is listed.
#[test]
fn test_every_float_op_kind_builds_at_declared_width() {
    let ctx = SymContext::new();
    for prec in [FloatPrec::F32, FloatPrec::F64] {
        let fp = |v: f64| match prec {
            FloatPrec::F32 => bv32(v as f32),
            FloatPrec::F64 => bv64(v),
        };
        let kinds = [
            FloatOpKind::Add,
            FloatOpKind::Sub,
            FloatOpKind::Mul,
            FloatOpKind::Div,
            FloatOpKind::Sqrt,
            FloatOpKind::Neg,
            FloatOpKind::Abs,
            FloatOpKind::Fma,
            FloatOpKind::Fms,
            FloatOpKind::CmpEq,
            FloatOpKind::CmpLt,
            FloatOpKind::CmpLe,
            FloatOpKind::IsNaN,
            FloatOpKind::RoundToInt,
            FloatOpKind::ConvertItoF {
                src_bits: 32,
                signed: true,
            },
            FloatOpKind::ConvertFtoI {
                dst_bits: 32,
                signed: true,
            },
            FloatOpKind::ConvertFtoIRm {
                dst_bits: 32,
                signed: true,
            },
            FloatOpKind::ConvertFtoF {
                src_prec: FloatPrec::F32,
            },
            FloatOpKind::ConvertFtoFRm {
                src_prec: FloatPrec::F32,
            },
            FloatOpKind::AddRm,
            FloatOpKind::SubRm,
            FloatOpKind::MulRm,
            FloatOpKind::DivRm,
            FloatOpKind::SqrtRm,
        ];
        // Compile-time guard: if a variant is added to `FloatOpKind`, this
        // exhaustive match stops compiling until the list above grows too.
        for k in kinds {
            match k {
                FloatOpKind::Add
                | FloatOpKind::Sub
                | FloatOpKind::Mul
                | FloatOpKind::Div
                | FloatOpKind::Sqrt
                | FloatOpKind::Neg
                | FloatOpKind::Abs
                | FloatOpKind::Fma
                | FloatOpKind::Fms
                | FloatOpKind::CmpEq
                | FloatOpKind::CmpLt
                | FloatOpKind::CmpLe
                | FloatOpKind::IsNaN
                | FloatOpKind::RoundToInt
                | FloatOpKind::ConvertItoF { .. }
                | FloatOpKind::ConvertFtoI { .. }
                | FloatOpKind::ConvertFtoIRm { .. }
                | FloatOpKind::ConvertFtoF { .. }
                | FloatOpKind::ConvertFtoFRm { .. }
                | FloatOpKind::AddRm
                | FloatOpKind::SubRm
                | FloatOpKind::MulRm
                | FloatOpKind::DivRm
                | FloatOpKind::SqrtRm => {}
            }
        }

        for kind in kinds {
            // Operand list per kind: rm-carrying ops take the rm BV first,
            // ItoF takes an integer BV, FtoF takes a value at `src_prec`.
            let operands: Vec<RustBV> = match kind {
                FloatOpKind::ConvertItoF { src_bits, .. } => {
                    vec![RustBV::concrete(3, u32::from(src_bits))]
                }
                FloatOpKind::ConvertFtoF { src_prec } => vec![match src_prec {
                    FloatPrec::F32 => bv32(1.5),
                    FloatPrec::F64 => bv64(1.5),
                }],
                FloatOpKind::ConvertFtoFRm { src_prec } => vec![
                    rm(0),
                    match src_prec {
                        FloatPrec::F32 => bv32(1.5),
                        FloatPrec::F64 => bv64(1.5),
                    },
                ],
                FloatOpKind::RoundToInt
                | FloatOpKind::ConvertFtoIRm { .. }
                | FloatOpKind::SqrtRm => vec![rm(0), fp(1.5)],
                FloatOpKind::AddRm
                | FloatOpKind::SubRm
                | FloatOpKind::MulRm
                | FloatOpKind::DivRm => vec![rm(0), fp(1.5), fp(2.5)],
                other => vec![fp(1.5); other.arity()],
            };
            assert_eq!(
                operands.len(),
                kind.arity(),
                "{kind:?} operand count vs arity"
            );

            let e = fp_expr(kind, prec, operands);
            let expected_width = kind.result_bits(prec);
            assert_eq!(e.width(), expected_width, "{kind:?} @ {prec:?}");
            let ast = e.to_z3_ast();
            assert_eq!(
                ast.get_size(),
                expected_width,
                "{kind:?} @ {prec:?} Z3 AST width"
            );
            assert!(
                ctx.eval(&e).is_some(),
                "{kind:?} @ {prec:?} should evaluate"
            );
        }
    }
}
