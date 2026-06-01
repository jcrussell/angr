//! Integration tests for x87 log/exp ops (`Iop_Yl2xF64`, `Iop_2xm1F64`).
//!
//! Covers both the concrete libm fast path (already validated in the
//! `vex::transcendentals` unit tests, mirrored here for parity) and the
//! symbolic concretize-and-pin fallback introduced by bd `angr-i5lj.1`.
//! Symbolic tests require Z3 since they exercise `ctx.eval` and
//! `ctx.assume_true`.

#![cfg(feature = "vex-engine-z3")]

use rustylib::symbolic::{RustBV, SymContext};
use rustylib::vex::transcendentals::{
    IOP_2XM1_F64, IOP_YL2X_F64, try_concrete_binop_rm, try_concrete_triop_rm,
    try_concretize_binop_rm, try_concretize_triop_rm,
};

fn bv64(f: f64) -> RustBV {
    RustBV::concrete(f.to_bits() as u128, 64)
}
fn rm() -> RustBV {
    RustBV::concrete(0, 32)
}
fn extract_f64(bv: RustBV) -> f64 {
    f64::from_bits(bv.as_u128().unwrap() as u64)
}
fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps || ((a - b).abs() / b.abs().max(1e-300)) < eps
}

#[test]
fn two_xm1_concrete_via_libm() {
    // 2^0 - 1 = 0
    let r = try_concrete_binop_rm(IOP_2XM1_F64, &rm(), &bv64(0.0)).unwrap();
    assert_eq!(extract_f64(r), 0.0);
    // 2^0.5 - 1 ≈ 0.4142135...
    let r = try_concrete_binop_rm(IOP_2XM1_F64, &rm(), &bv64(0.5)).unwrap();
    assert!(approx_eq(extract_f64(r), 2f64.sqrt() - 1.0, 1e-12));
}

#[test]
fn yl2x_concrete_via_libm() {
    // 3 * log2(2) = 3
    let r = try_concrete_triop_rm(IOP_YL2X_F64, &rm(), &bv64(3.0), &bv64(2.0)).unwrap();
    assert!(approx_eq(extract_f64(r), 3.0, 1e-12));
}

#[test]
fn two_xm1_symbolic_concretize_pins_input() {
    // Build symbolic x constrained to == 1.0; the concretize fallback
    // must eval x to 1.0, compute 2^1 - 1 = 1.0, and pin x to 1.0.
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "x_2xm1", 64);
    let target = bv64(1.0);
    let eq = x.eq(&target, &ctx);
    ctx.assume_true(&eq);

    let r = try_concretize_binop_rm(IOP_2XM1_F64, &rm(), &x, &ctx).unwrap();
    assert!(approx_eq(extract_f64(r), 1.0, 1e-12));

    // After the pin, evaluating x must still give 1.0.
    let xv = ctx.eval(&x).unwrap() as u64;
    assert_eq!(f64::from_bits(xv), 1.0);
}

#[test]
fn two_xm1_symbolic_concretize_unconstrained_returns_some_concrete() {
    // No prior constraints on x — the solver picks any model; the
    // returned BV must still be concrete (lost precision is the
    // documented tradeoff).
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "x_2xm1_uc", 64);
    let r = try_concretize_binop_rm(IOP_2XM1_F64, &rm(), &x, &ctx).unwrap();
    assert!(r.is_concrete(), "concretize must yield a concrete BV");
    // After the call, x is pinned. eval again must match the result.
    let xv = ctx.eval(&x).unwrap() as u64;
    let xf = f64::from_bits(xv);
    let expected = xf.exp2() - 1.0;
    let got = extract_f64(r);
    assert!(
        approx_eq(got, expected, 1e-12) || (expected.is_nan() && got.is_nan()),
        "result must match libm 2^x - 1 on the chosen sample"
    );
}

#[test]
fn yl2x_symbolic_concretize_pins_both_inputs() {
    // y * log2(x) with both inputs symbolic and constrained.
    let ctx = SymContext::new();
    let y = RustBV::symbolic(&ctx, "y_yl2x", 64);
    let x = RustBV::symbolic(&ctx, "x_yl2x", 64);
    let yt = bv64(2.0);
    let xt = bv64(4.0);
    let yeq = y.eq(&yt, &ctx);
    let xeq = x.eq(&xt, &ctx);
    ctx.assume_true(&yeq);
    ctx.assume_true(&xeq);

    let r = try_concretize_triop_rm(IOP_YL2X_F64, &rm(), &y, &x, &ctx).unwrap();
    // 2 * log2(4) = 4
    assert!(approx_eq(extract_f64(r), 4.0, 1e-12));
}

#[test]
fn yl2x_symbolic_concretize_one_side_concrete() {
    // y symbolic (pinned to 1.0), x concrete (8.0). Result: 1 * log2(8) = 3.
    let ctx = SymContext::new();
    let y = RustBV::symbolic(&ctx, "y_yl2x_mix", 64);
    ctx.assume_true(&y.eq(&bv64(1.0), &ctx));

    let r = try_concretize_triop_rm(IOP_YL2X_F64, &rm(), &y, &bv64(8.0), &ctx).unwrap();
    assert!(approx_eq(extract_f64(r), 3.0, 1e-12));
}

#[test]
fn concretize_other_opcode_returns_none() {
    // The concretize fallback is scoped to angr-i5lj.1's two ops only;
    // other transcendentals must return None so the caller can fall
    // through to the existing fresh-symbolic path.
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "sin_in", 64);
    // Iop_SinF64 = 0x14e6 — not in scope for angr-i5lj.1.
    assert!(try_concretize_binop_rm(0x14e6, &rm(), &x, &ctx).is_none());
    // Iop_AtanF64 = 0x14de — not in scope for angr-i5lj.1.
    let y = RustBV::symbolic(&ctx, "atan_y", 64);
    assert!(try_concretize_triop_rm(0x14de, &rm(), &y, &x, &ctx).is_none());
}

#[test]
fn concretize_returns_none_when_unsat() {
    // Constrain x to be a value that contradicts itself → UNSAT → eval
    // returns None → concretize returns None.
    let ctx = SymContext::new();
    let x = RustBV::symbolic(&ctx, "unsat_x", 64);
    ctx.assume_true(&x.eq(&bv64(1.0), &ctx));
    ctx.assume_true(&x.eq(&bv64(2.0), &ctx));
    assert!(!ctx.is_sat());
    assert!(try_concretize_binop_rm(IOP_2XM1_F64, &rm(), &x, &ctx).is_none());
}
