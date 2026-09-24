// Tests for vex/transcendentals.rs — extracted from the inline `mod tests`.
// See parent module for the transcendental VEX op implementations under test.

use super::*;

fn bv64(f: f64) -> RustBV {
    RustBV::concrete(f.to_bits() as u128, 64)
}
fn bv32(f: f32) -> RustBV {
    RustBV::concrete(f.to_bits() as u128, 32)
}
fn rm() -> RustBV {
    RustBV::concrete(0, 32)
}
fn extract_f64(bv: RustBV) -> f64 {
    f64::from_bits(bv.as_u128().unwrap() as u64)
}
fn extract_f32(bv: RustBV) -> f32 {
    f32::from_bits(bv.as_u128().unwrap() as u32)
}

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps || ((a - b).abs() / b.abs().max(1e-300)) < eps
}

#[test]
fn sin_cos_tan_concrete() {
    let r = try_concrete_binop_rm(IOP_SIN_F64, &rm(), &bv64(std::f64::consts::PI / 6.0)).unwrap();
    assert!(approx_eq(extract_f64(r), 0.5, 1e-12));

    let r = try_concrete_binop_rm(IOP_COS_F64, &rm(), &bv64(0.0)).unwrap();
    assert_eq!(extract_f64(r), 1.0);

    let r = try_concrete_binop_rm(IOP_TAN_F64, &rm(), &bv64(std::f64::consts::PI / 4.0)).unwrap();
    assert!(approx_eq(extract_f64(r), 1.0, 1e-12));
}

#[test]
fn two_xm1_concrete() {
    // 2^0 - 1 = 0
    let r = try_concrete_binop_rm(IOP_2XM1_F64, &rm(), &bv64(0.0)).unwrap();
    assert_eq!(extract_f64(r), 0.0);
    // 2^1 - 1 = 1
    let r = try_concrete_binop_rm(IOP_2XM1_F64, &rm(), &bv64(1.0)).unwrap();
    assert!(approx_eq(extract_f64(r), 1.0, 1e-12));
    // 2^-1 - 1 = -0.5
    let r = try_concrete_binop_rm(IOP_2XM1_F64, &rm(), &bv64(-1.0)).unwrap();
    assert!(approx_eq(extract_f64(r), -0.5, 1e-12));
}

#[test]
fn yl2x_concrete() {
    // 1 * log2(8) = 3
    let r = try_concrete_triop_rm(IOP_YL2X_F64, &rm(), &bv64(1.0), &bv64(8.0)).unwrap();
    assert!(approx_eq(extract_f64(r), 3.0, 1e-12));
    // 2 * log2(4) = 4
    let r = try_concrete_triop_rm(IOP_YL2X_F64, &rm(), &bv64(2.0), &bv64(4.0)).unwrap();
    assert!(approx_eq(extract_f64(r), 4.0, 1e-12));
}

#[test]
fn yl2xp1_concrete() {
    // 1 * log2(7+1) = 3
    let r = try_concrete_triop_rm(IOP_YL2XP1_F64, &rm(), &bv64(1.0), &bv64(7.0)).unwrap();
    assert!(approx_eq(extract_f64(r), 3.0, 1e-12));
}

#[test]
fn scale_concrete() {
    // 1.5 * 2^trunc(2.7) = 1.5 * 4 = 6
    let r = try_concrete_triop_rm(IOP_SCALE_F64, &rm(), &bv64(1.5), &bv64(2.7)).unwrap();
    assert!(approx_eq(extract_f64(r), 6.0, 1e-12));
    // 1.0 * 2^trunc(-3.5) = 1.0 * 0.125 = 0.125
    let r = try_concrete_triop_rm(IOP_SCALE_F64, &rm(), &bv64(1.0), &bv64(-3.5)).unwrap();
    assert_eq!(extract_f64(r), 0.125);
}

#[test]
fn atan_concrete() {
    // atan2(1, 1) = pi/4
    let r = try_concrete_triop_rm(IOP_ATAN_F64, &rm(), &bv64(1.0), &bv64(1.0)).unwrap();
    assert!(approx_eq(
        extract_f64(r),
        std::f64::consts::FRAC_PI_4,
        1e-12
    ));
    // atan2(0, 1) = 0
    let r = try_concrete_triop_rm(IOP_ATAN_F64, &rm(), &bv64(0.0), &bv64(1.0)).unwrap();
    assert_eq!(extract_f64(r), 0.0);
}

#[test]
fn symbolic_returns_none() {
    let ctx = crate::symbolic::SymContext::new_mock();
    let sym = RustBV::symbolic(&ctx, "x", 64);
    assert!(try_concrete_binop_rm(IOP_SIN_F64, &rm(), &sym).is_none());
    assert!(try_concrete_triop_rm(IOP_YL2X_F64, &rm(), &bv64(1.0), &sym).is_none());
}

#[test]
fn unknown_opcode_returns_none() {
    assert!(try_concrete_binop_rm(0xdead, &rm(), &bv64(1.0)).is_none());
    assert!(try_concrete_triop_rm(0xdead, &rm(), &bv64(1.0), &bv64(1.0)).is_none());
}

#[test]
fn recip_exp_specials() {
    // 2.0 → exponent 1 → 2^-1 = 0.5
    let r = try_concrete_binop_rm(IOP_RECPEXP_F64, &rm(), &bv64(2.0)).unwrap();
    assert_eq!(extract_f64(r), 0.5);
    // 4.0 → exponent 2 → 2^-2 = 0.25
    let r = try_concrete_binop_rm(IOP_RECPEXP_F64, &rm(), &bv64(4.0)).unwrap();
    assert_eq!(extract_f64(r), 0.25);
    // 0 → +inf
    let r = try_concrete_binop_rm(IOP_RECPEXP_F64, &rm(), &bv64(0.0)).unwrap();
    assert_eq!(extract_f64(r), f64::INFINITY);
    // f32: 8.0 → exponent 3 → 2^-3 = 0.125
    let r = try_concrete_binop_rm(IOP_RECPEXP_F32, &rm(), &bv32(8.0)).unwrap();
    assert_eq!(extract_f32(r), 0.125);
}

/// angr-z8elx: the two operands must be pinned to a JOINT witness. Under
/// strict-deterministic mode `eval` returns a per-variable minimum, so two
/// independent evals hand back `a=0, b=0` here — individually feasible, but
/// the `a + b == 5` path constraint makes them jointly infeasible, and the
/// pins would turn a SAT context UNSAT (a silently dropped feasible path).
#[test]
fn concretize_triop_rm_pins_a_joint_witness_under_deterministic_mode() {
    let ctx = crate::symbolic::SymContext::new_mock();
    ctx.set_deterministic(true);
    let a = RustBV::symbolic(&ctx, "tz_a", 64);
    let b = RustBV::symbolic(&ctx, "tz_b", 64);
    let sum = a.add(&b, &ctx);
    ctx.assume_true(&sum.eq(&RustBV::concrete(5, 64), &ctx));
    assert!(
        ctx.is_sat(),
        "precondition: correlated operands are satisfiable"
    );

    let r = try_concretize_triop_rm(IOP_YL2X_F64, &rm(), &a, &b, &ctx);
    assert!(r.is_some(), "symbolic operands should concretize");
    assert!(
        ctx.is_sat(),
        "pinning both operands must not contradict the path constraint"
    );
}

/// angr-9ke6b.233 end-to-end: the pyvex opcode *string* must reach the libm
/// fast paths through the same dispatch the interpreter uses
/// (`IRExpr::Binop` → `VEXOps::binop`, `IRExpr::Triop` →
/// `VEXOps::binop_with_rm`). Before .233 nothing constructed `IROp::Raw`,
/// so `parse_opcode` returned `Unmapped` for every op in this module and
/// the whole file was unreachable outside the tests above.
#[test]
fn parse_opcode_dispatch_reaches_libm_fast_paths() {
    use crate::vex::opcode_map::parse_opcode;
    use crate::vex::ops::VEXOps;

    let ctx = crate::symbolic::SymContext::new_mock();

    // Binop form: (rm, x).
    let r = VEXOps::binop(parse_opcode("Iop_CosF64"), rm(), bv64(0.0), &ctx).unwrap();
    assert!(approx_eq(extract_f64(r), 1.0, 1e-12));

    // Triop form: (rm, x, y) — log2(8) * 1.0 == 3.0.
    let r = VEXOps::binop_with_rm(
        parse_opcode("Iop_Yl2xF64"),
        rm(),
        bv64(1.0),
        bv64(8.0),
        &ctx,
    )
    .unwrap();
    assert!(approx_eq(extract_f64(r), 3.0, 1e-12));
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
    let ctx = crate::symbolic::SymContext::new_mock();
    ctx.set_deterministic(true);
    let x = RustBV::symbolic(&ctx, "tz_x", 64);

    let r = try_concretize_binop_rm(IOP_COS_F64, &rm(), &x, &ctx)
        .expect("symbolic operand should concretize");
    // Witness 0 is the bit pattern of +0.0, so the result must be cos(0.0).
    assert!(approx_eq(extract_f64(r), 1.0, 1e-12));

    // The pin is a real hard constraint, not just a local sample: `x` is no
    // longer free. Dropping the `assume_true` in `try_concretize_binop_rm`
    // leaves the result concrete but the operand unconstrained, which is the
    // path-inconsistency this asserts against.
    ctx.assume_true(&x.ne(&bv64(0.0), &ctx));
    assert!(
        !ctx.is_sat(),
        "the sampled witness must have been pinned onto the operand"
    );
}

/// The other half: pinning must agree with whatever path constraint already
/// holds, so a context that was SAT before the call is still SAT after it.
#[test]
fn concretize_binop_rm_pin_agrees_with_an_existing_path_constraint() {
    let ctx = crate::symbolic::SymContext::new_mock();
    ctx.set_deterministic(true);
    let x = RustBV::symbolic(&ctx, "tz_pc", 64);
    ctx.assume_true(&x.eq(&bv64(0.5), &ctx));
    assert!(ctx.is_sat(), "precondition: constrained operand is satisfiable");

    let r = try_concretize_binop_rm(IOP_SIN_F64, &rm(), &x, &ctx)
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
    let ctx = crate::symbolic::SymContext::new_mock();
    let x = RustBV::symbolic(&ctx, "tz_oos", 64);
    assert!(try_concretize_binop_rm(IOP_RECPEXP_F64, &rm(), &x, &ctx).is_none());
    assert!(try_concretize_binop_rm(0xdead, &rm(), &x, &ctx).is_none());
    // ...and no constraint was assumed on the way out.
    assert!(ctx.is_sat());
}

/// A concrete operand short-circuits to the libm fast path and assumes nothing
/// — the `x.is_concrete()` arm of `try_concretize_binop_rm`.
#[test]
fn concretize_binop_rm_passes_concrete_operands_through() {
    let ctx = crate::symbolic::SymContext::new_mock();
    let r = try_concretize_binop_rm(IOP_COS_F64, &rm(), &bv64(0.0), &ctx).unwrap();
    assert!(approx_eq(extract_f64(r), 1.0, 1e-12));
}
