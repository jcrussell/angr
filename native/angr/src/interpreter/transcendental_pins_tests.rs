//! Tests for `transcendental_pins`: the pins themselves, driven the way
//! `eval_binop` / `eval_triop` drive them (pin, then dispatch through
//! `VEXOps` on the pinned values), and end to end through the interpreter.

use super::*;
use crate::vex::ir::IROp;
use crate::vex::ops::VEXOps;
use crate::vex::transcendentals::{
    IOP_2XM1_F64, IOP_ATAN_F64, IOP_COS_F64, IOP_RECPEXP_F64, IOP_SCALE_F64, IOP_SIN_F64,
    IOP_TAN_F64, IOP_YL2X_F64, IOP_YL2XP1_F64, pins_binop, pins_triop,
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

/// Binop `(rm, x)` as `eval_binop` dispatches it: pin a symbolic `x` when
/// `pins_binop` names the op, then `VEXOps::binop`. `None` is a decline.
fn concretize_binop_rm(opcode: u32, rm: &RustBV, x: &RustBV, ctx: &SymContext) -> Option<RustBV> {
    let x = if pins_binop(opcode) && x.is_symbolic() {
        pin_operand(x, ctx).unwrap_or_else(|| x.clone())
    } else {
        x.clone()
    };
    VEXOps::binop(IROp::Raw(opcode), rm.clone(), x).ok()
}

/// Triop `(rm, a, b)` as `eval_triop` dispatches it: pin symbolic operands
/// when `pins_triop` names the op, then `VEXOps::binop_with_rm`.
fn concretize_triop_rm(
    opcode: u32,
    rm: &RustBV,
    a: &RustBV,
    b: &RustBV,
    ctx: &SymContext,
) -> Option<RustBV> {
    let (a, b) = if pins_triop(opcode) && (a.is_symbolic() || b.is_symbolic()) {
        pin_operands(a, b, ctx).unwrap_or_else(|| (a.clone(), b.clone()))
    } else {
        (a.clone(), b.clone())
    };
    VEXOps::binop_with_rm(IROp::Raw(opcode), rm.clone(), a, b).ok()
}

/// angr-z8elx: the two operands must be pinned to a JOINT witness. Under
/// strict-deterministic mode `eval` returns a per-variable minimum, so two
/// independent evals hand back `a=0, b=0` here — individually feasible, but
/// the `a + b == 5` path constraint makes them jointly infeasible, and the
/// pins would turn a SAT context UNSAT (a silently dropped feasible path).
#[test]
fn concretize_triop_rm_pins_a_joint_witness_under_deterministic_mode() {
    let ctx = SymContext::new_mock();
    ctx.set_deterministic(true);
    let a = RustBV::symbol("tz_a", 64);
    let b = RustBV::symbol("tz_b", 64);
    let sum = a.add(&b);
    ctx.assume_true(&sum.eq(&RustBV::concrete(5, 64)));
    assert!(
        ctx.is_sat(),
        "precondition: correlated operands are satisfiable"
    );

    let r = concretize_triop_rm(IOP_YL2X_F64, &rm(), &a, &b, &ctx);
    assert!(r.is_some(), "symbolic operands should concretize");
    assert!(
        ctx.is_sat(),
        "pinning both operands must not contradict the path constraint"
    );
}

/// angr-fs8kb.22: the binop counterpart of
/// `concretize_triop_rm_pins_a_joint_witness_under_deterministic_mode`. One
/// operand means there is no *joint*-witness hazard, but the rest of the
/// contract still holds: the sampled value must be pinned with a hard
/// constraint, and the returned value must be libm applied to the value
/// actually pinned. Strict-deterministic mode makes the witness predictable —
/// `eval` routes through `min`, so an unconstrained operand samples 0.
#[test]
fn concretize_binop_rm_pins_the_sampled_witness() {
    let ctx = SymContext::new_mock();
    ctx.set_deterministic(true);
    let x = RustBV::symbol("tz_x", 64);

    let r = concretize_binop_rm(IOP_COS_F64, &rm(), &x, &ctx)
        .expect("symbolic operand should concretize");
    // Witness 0 is the bit pattern of +0.0, so the result must be cos(0.0).
    assert!(approx_eq(extract_f64(r), 1.0, 1e-12));

    // The pin is a real hard constraint, not just a local sample: `x` is no
    // longer free. Dropping the `assume_true` in `pin_operand`
    // leaves the result concrete but the operand unconstrained, which is the
    // path-inconsistency this asserts against.
    ctx.assume_true(&x.ne(&bv64(0.0)));
    assert!(
        !ctx.is_sat(),
        "the sampled witness must have been pinned onto the operand"
    );
}

/// The other half: pinning must agree with whatever path constraint already
/// holds, so a context that was SAT before the call is still SAT after it.
#[test]
fn concretize_binop_rm_pin_agrees_with_an_existing_path_constraint() {
    let ctx = SymContext::new_mock();
    ctx.set_deterministic(true);
    let x = RustBV::symbol("tz_pc", 64);
    ctx.assume_true(&x.eq(&bv64(0.5)));
    assert!(ctx.is_sat(), "precondition: constrained operand is satisfiable");

    let r = concretize_binop_rm(IOP_SIN_F64, &rm(), &x, &ctx)
        .expect("symbolic operand should concretize");
    assert!(
        ctx.is_sat(),
        "pinning the operand must not contradict the path constraint"
    );
    assert!(approx_eq(extract_f64(r), 0.5f64.sin(), 1e-12));
}

/// The concretization fallback covers a strictly smaller opcode set than the
/// concrete path: `Iop_RecpExp*` is handled by `try_concrete_binop_rm` but is
/// deliberately out of scope here (closed-form, never needs libm — see the
/// module doc). A symbolic operand on one of those must fall through to the
/// caller's fresh-symbolic path rather than silently pinning.
#[test]
fn concretize_binop_rm_rejects_out_of_scope_opcodes() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("tz_oos", 64);
    assert!(concretize_binop_rm(IOP_RECPEXP_F64, &rm(), &x, &ctx).is_none());
    assert!(concretize_binop_rm(0xdead, &rm(), &x, &ctx).is_none());
    // ...and no constraint was assumed on the way out.
    assert!(ctx.is_sat());
}

/// A concrete operand short-circuits to the libm fast path and assumes nothing
/// — `eval_binop` only pins a symbolic operand.
#[test]
fn concretize_binop_rm_passes_concrete_operands_through() {
    let ctx = SymContext::new_mock();
    let r = concretize_binop_rm(IOP_COS_F64, &rm(), &bv64(0.0), &ctx).unwrap();
    assert!(approx_eq(extract_f64(r), 1.0, 1e-12));
}

// Moved from the `tests/vex_ops_x87.rs` integration test, which cannot reach
// the `pub(crate)` interpreter.

#[test]
fn two_xm1_symbolic_concretize_pins_input() {
    // Build symbolic x constrained to == 1.0; the concretize fallback
    // must eval x to 1.0, compute 2^1 - 1 = 1.0, and pin x to 1.0.
    let ctx = SymContext::new();
    let x = RustBV::symbol("x_2xm1", 64);
    let target = bv64(1.0);
    let eq = x.eq(&target);
    ctx.assume_true(&eq);

    let r = concretize_binop_rm(IOP_2XM1_F64, &rm(), &x, &ctx).unwrap();
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
    let x = RustBV::symbol("x_2xm1_uc", 64);
    let r = concretize_binop_rm(IOP_2XM1_F64, &rm(), &x, &ctx).unwrap();
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
    let y = RustBV::symbol("y_yl2x", 64);
    let x = RustBV::symbol("x_yl2x", 64);
    let yt = bv64(2.0);
    let xt = bv64(4.0);
    let yeq = y.eq(&yt);
    let xeq = x.eq(&xt);
    ctx.assume_true(&yeq);
    ctx.assume_true(&xeq);

    let r = concretize_triop_rm(IOP_YL2X_F64, &rm(), &y, &x, &ctx).unwrap();
    // 2 * log2(4) = 4
    assert!(approx_eq(extract_f64(r), 4.0, 1e-12));
}

#[test]
fn yl2x_symbolic_concretize_one_side_concrete() {
    // y symbolic (pinned to 1.0), x concrete (8.0). Result: 1 * log2(8) = 3.
    let ctx = SymContext::new();
    let y = RustBV::symbol("y_yl2x_mix", 64);
    ctx.assume_true(&y.eq(&bv64(1.0)));

    let r = concretize_triop_rm(IOP_YL2X_F64, &rm(), &y, &bv64(8.0), &ctx).unwrap();
    assert!(approx_eq(extract_f64(r), 3.0, 1e-12));
}

#[test]
fn concretize_other_opcode_returns_none() {
    // Out-of-scope opcodes must return None so the caller can fall
    // through to the existing fresh-symbolic path. Iop_PRemF64 (0x14e1)
    // and Iop_RoundF64toInt (0x14eb) are real x87 ops but intentionally
    // not handled by the concretize fallback.
    let ctx = SymContext::new();
    let x = RustBV::symbol("prem_in", 64);
    assert!(concretize_binop_rm(0xdead, &rm(), &x, &ctx).is_none());
    assert!(concretize_binop_rm(0x14e1, &rm(), &x, &ctx).is_none());
    let y = RustBV::symbol("prem_y", 64);
    assert!(concretize_triop_rm(0xdead, &rm(), &y, &x, &ctx).is_none());
    assert!(concretize_triop_rm(0x14e1, &rm(), &y, &x, &ctx).is_none());
}

#[test]
fn sin_cos_tan_symbolic_concretize_pins_input() {
    // Iop_SinF64: pin x to π/6, expect sin(π/6) ≈ 0.5.
    let ctx = SymContext::new();
    let x = RustBV::symbol("x_sin", 64);
    ctx.assume_true(&x.eq(&bv64(std::f64::consts::PI / 6.0)));
    let r = concretize_binop_rm(IOP_SIN_F64, &rm(), &x, &ctx).unwrap();
    assert!(approx_eq(extract_f64(r), 0.5, 1e-12));

    // Iop_CosF64: pin x to 0, expect cos(0) = 1.
    let ctx = SymContext::new();
    let x = RustBV::symbol("x_cos", 64);
    ctx.assume_true(&x.eq(&bv64(0.0)));
    let r = concretize_binop_rm(IOP_COS_F64, &rm(), &x, &ctx).unwrap();
    assert_eq!(extract_f64(r), 1.0);

    // Iop_TanF64: pin x to π/4, expect tan(π/4) ≈ 1.
    let ctx = SymContext::new();
    let x = RustBV::symbol("x_tan", 64);
    ctx.assume_true(&x.eq(&bv64(std::f64::consts::PI / 4.0)));
    let r = concretize_binop_rm(IOP_TAN_F64, &rm(), &x, &ctx).unwrap();
    assert!(approx_eq(extract_f64(r), 1.0, 1e-12));
}

#[test]
fn atan_symbolic_concretize_pins_both_inputs() {
    // atan2(1, 1) = π/4 with both inputs symbolic.
    let ctx = SymContext::new();
    let y = RustBV::symbol("y_atan", 64);
    let x = RustBV::symbol("x_atan", 64);
    ctx.assume_true(&y.eq(&bv64(1.0)));
    ctx.assume_true(&x.eq(&bv64(1.0)));
    let r = concretize_triop_rm(IOP_ATAN_F64, &rm(), &y, &x, &ctx).unwrap();
    assert!(approx_eq(
        extract_f64(r),
        std::f64::consts::FRAC_PI_4,
        1e-12
    ));
}

#[test]
fn yl2xp1_symbolic_concretize_pins_both_inputs() {
    // 1 * log2(7 + 1) = 3 with both inputs symbolic.
    let ctx = SymContext::new();
    let y = RustBV::symbol("y_yl2xp1", 64);
    let x = RustBV::symbol("x_yl2xp1", 64);
    ctx.assume_true(&y.eq(&bv64(1.0)));
    ctx.assume_true(&x.eq(&bv64(7.0)));
    let r = concretize_triop_rm(IOP_YL2XP1_F64, &rm(), &y, &x, &ctx).unwrap();
    assert!(approx_eq(extract_f64(r), 3.0, 1e-12));
}

#[test]
fn scale_symbolic_concretize_one_side_concrete() {
    // 1.5 * 2^trunc(2.7) = 6 with y symbolic, x concrete.
    let ctx = SymContext::new();
    let y = RustBV::symbol("y_scale", 64);
    ctx.assume_true(&y.eq(&bv64(1.5)));
    let r = concretize_triop_rm(IOP_SCALE_F64, &rm(), &y, &bv64(2.7), &ctx).unwrap();
    assert!(approx_eq(extract_f64(r), 6.0, 1e-12));
}

#[test]
fn sin_symbolic_unconstrained_returns_some_concrete() {
    // Unconstrained Iop_SinF64 must still yield a concrete BV; the
    // result must match libm sin() on whatever sample the solver chose.
    let ctx = SymContext::new();
    let x = RustBV::symbol("x_sin_uc", 64);
    let r = concretize_binop_rm(IOP_SIN_F64, &rm(), &x, &ctx).unwrap();
    assert!(r.is_concrete());
    let xv = ctx.eval(&x).unwrap() as u64;
    let xf = f64::from_bits(xv);
    let got = extract_f64(r);
    let expected = xf.sin();
    assert!(
        approx_eq(got, expected, 1e-12) || (expected.is_nan() && got.is_nan()),
        "result must match libm sin() on chosen sample (got={got}, expected={expected})"
    );
}

#[test]
fn concretize_returns_none_when_unsat() {
    // Constrain x to be a value that contradicts itself → UNSAT → eval
    // returns None → concretize returns None.
    let ctx = SymContext::new();
    let x = RustBV::symbol("unsat_x", 64);
    ctx.assume_true(&x.eq(&bv64(1.0)));
    ctx.assume_true(&x.eq(&bv64(2.0)));
    assert!(!ctx.is_sat());
    assert!(concretize_binop_rm(IOP_2XM1_F64, &rm(), &x, &ctx).is_none());
}

#[test]
fn pins_name_exactly_the_libm_transcendentals() {
    for op in [IOP_SIN_F64, IOP_COS_F64, IOP_TAN_F64, IOP_2XM1_F64] {
        assert!(pins_binop(op) && !pins_triop(op), "{op:#x}");
    }
    for op in [IOP_YL2X_F64, IOP_YL2XP1_F64, IOP_SCALE_F64, IOP_ATAN_F64] {
        assert!(pins_triop(op) && !pins_binop(op), "{op:#x}");
    }
    // FRECPX is closed-form: never pinned.
    assert!(!pins_binop(IOP_RECPEXP_F64) && !pins_triop(IOP_RECPEXP_F64));
}

/// A symbolic temp through `eval_binop` / `eval_triop` itself: the result is
/// concrete and the operands are pinned on the path.
#[test]
fn interpreter_pins_symbolic_transcendental_operands() {
    use crate::interpreter::{TestHost, VEXInterpreter};
    use crate::vex::ir::{IRConst, IRExpr, TypeEnv, VexArch};

    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("tp_e2e_x", 64);
    let y = RustBV::symbol("tp_e2e_y", 64);
    ctx.assume_true(&x.eq(&bv64(0.0)));
    ctx.assume_true(&y.eq(&bv64(8.0)));
    let callbacks = TestHost::default();
    let mut interp = VEXInterpreter::new(VexArch::AMD64, &ctx);
    interp.temps = vec![Some(x.clone()), Some(y.clone())];
    let env = TypeEnv::new();
    let rm_expr = IRExpr::Const(IRConst::U32(0));

    let cos = interp
        .eval_binop(&callbacks, IROp::Raw(IOP_COS_F64), &rm_expr, &IRExpr::RdTmp(0), &env)
        .unwrap();
    assert_eq!(extract_f64(cos), 1.0);

    // x * log2(y) = 0 * 3.
    let yl2x = interp
        .eval_triop(
            &callbacks,
            IROp::Raw(IOP_YL2X_F64),
            &rm_expr,
            &IRExpr::RdTmp(0),
            &IRExpr::RdTmp(1),
            &env,
        )
        .unwrap();
    assert_eq!(extract_f64(yl2x), 0.0);

    // The pins are hard constraints: neither operand is free any more.
    assert!(!ctx.can_be_true(&x.ne(&bv64(0.0))));
    assert!(!ctx.can_be_true(&y.ne(&bv64(8.0))));
}

/// A symbolic FRECPX operand is not pinned: the op declines and takes the
/// same Python fallback it took before the pins moved to the interpreter.
#[test]
fn interpreter_leaves_a_symbolic_recp_exp_to_the_fallback() {
    use crate::interpreter::{TestHost, VEXInterpreter};
    use crate::vex::ir::{IRConst, IRExpr, TypeEnv, VexArch};

    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("tp_e2e_recpexp", 64);
    let callbacks = TestHost::default();
    let mut interp = VEXInterpreter::new(VexArch::AMD64, &ctx);
    interp.temps = vec![Some(x.clone())];
    let before = interp.stats.python_vex_binop_fallback_count;
    let r = interp.eval_binop(
        &callbacks,
        IROp::Raw(IOP_RECPEXP_F64),
        &IRExpr::Const(IRConst::U32(0)),
        &IRExpr::RdTmp(0),
        &TypeEnv::new(),
    );
    assert!(r.is_err(), "a symbolic RecpExp routes to Python, got {r:?}");
    assert_eq!(interp.stats.python_vex_binop_fallback_count, before + 1);
    assert!(ctx.can_be_true(&x.ne(&bv64(0.0))), "no pin was assumed");
}
