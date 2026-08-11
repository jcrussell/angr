// angr-9hleg: scalar FP arith tests incl. rounding-mode variants (mirror of ops/float_arith.rs).

use super::*;
use crate::vex::ir::IRType;

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
        other => panic!("expected NotQuaternary, got {other:?}"),
    }
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
        "Expected x == 3.0, got {result_f}"
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
        "Expected x == 16.0, got {result_f}"
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
        "expected rm low2 ∈ {{1, 3}} (RD/RZ), got {low2}"
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

/// Symbolic OPERAND under a concrete non-RNE rm must flow through the Z3
/// rm-aware path, NOT collapse to a fresh unconstrained symbolic (the
/// pre-angr-tfjl behavior cudgw.5 tracked). `a` is symbolic; FAdd(a, b)
/// under RZ. Constraining `a == 0x3F800001` must pin the result to the
/// RZ-truncated sum `0x40000001` (a fresh-symbolic result would leave it
/// free, making `result == 0x40000002` satisfiable too).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_float_add_symbolic_operand_with_rm_rz_f32() {
    let ctx = SymContext::new_mock();
    let rm_rz = RustBV::concrete(3, 32);
    let a = RustBV::symbolic(&ctx, "fadd_sym_a", 32);
    let b = RustBV::concrete(0x3F800002, 32);

    let result = VEXOps::binop_with_rm(IROp::FAdd(IRType::F32), rm_rz, a.clone(), b, &ctx).unwrap();
    assert!(
        result.is_symbolic(),
        "symbolic operand must yield a symbolic result, not a constant"
    );

    // Pin a to the inexact-sum operand; the RZ result is fully determined.
    // If `result` were a fresh unconstrained symbolic, eval would not be
    // pinned to the RZ-truncated sum.
    ctx.add_constraint(
        a.to_z3_ast()
            .eq(RustBV::concrete(0x3F800001, 32).to_z3_ast()),
    );
    assert!(ctx.is_sat(), "pinned symbolic operand must keep ctx SAT");

    let bits = ctx.eval(&result).expect("eval failed") as u32;
    assert_eq!(
        bits, 0x40000001,
        "RZ truncates the 1.5ulp tie of the pinned symbolic operand"
    );
}

// ---------------------------------------------------------------------------
// Qop (FMAdd / FMSub) rounding-mode threading — angr-03vl4.30
// ---------------------------------------------------------------------------

/// Witness for FMA rounding: `1.0 * 1.0 + 1.5*2^-53` sits 1.5 ULP above 1.0,
/// so RNE rounds *up* to `1.0 + 2^-52` while RZ truncates back to `1.0`. Any
/// path that drops the Qop's rm operand (the pre-angr-03vl4.30 behaviour)
/// returns the RNE answer for every mode.
fn fma_tie_operands() -> (RustBV, RustBV, RustBV) {
    let c = 3.0f64 * 2.0f64.powi(-54);
    (
        RustBV::concrete(1.0f64.to_bits() as u128, 64),
        RustBV::concrete(1.0f64.to_bits() as u128, 64),
        RustBV::concrete(c.to_bits() as u128, 64),
    )
}

/// RNE rm keeps the native `mul_add` fast path: fully concrete result.
#[test]
fn test_qop_with_rm_rne_fastpath_f64() {
    let ctx = SymContext::new_mock();
    let (a, b, c) = fma_tie_operands();
    let rm_rne = RustBV::concrete(0, 32);

    let result = VEXOps::qop_with_rm(IROp::FMAdd(IRType::F64), rm_rne, a, b, c, &ctx).unwrap();
    assert!(!result.is_symbolic(), "RNE must not route through Z3");
    assert_eq!(result.as_u64(), Some((1.0f64 + f64::EPSILON).to_bits()));
}

/// RZ rm on the same operands truncates toward zero → exactly 1.0.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_qop_with_rm_rz_f64_differs_from_rne() {
    let ctx = SymContext::new_mock();
    let (a, b, c) = fma_tie_operands();
    let rm_rz = RustBV::concrete(3, 32);

    let result = VEXOps::qop_with_rm(IROp::FMAdd(IRType::F64), rm_rz, a, b, c, &ctx).unwrap();
    let bits = ctx.eval(&result).expect("eval failed");
    assert_eq!(
        bits,
        u128::from(1.0f64.to_bits()),
        "RZ truncates the 1.5ulp FMA sum"
    );
    assert_ne!(
        bits,
        u128::from((1.0f64 + f64::EPSILON).to_bits()),
        "must differ from RNE"
    );
}

/// FMSub under RZ: `a*b - c` with a negated `c` reaches the same 1.5 ULP sum,
/// proving both that the rm is threaded and that the `-c` negation survives
/// the rm-carrying path.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_qop_with_rm_fmsub_rz_f64() {
    let ctx = SymContext::new_mock();
    let (a, b, _) = fma_tie_operands();
    let neg_c = -3.0f64 * 2.0f64.powi(-54);
    let c = RustBV::concrete(neg_c.to_bits() as u128, 64);
    let rm_rz = RustBV::concrete(3, 32);

    let result = VEXOps::qop_with_rm(IROp::FMSub(IRType::F64), rm_rz, a, b, c, &ctx).unwrap();
    let bits = ctx.eval(&result).expect("eval failed");
    assert_eq!(bits, u128::from(1.0f64.to_bits()));
}

/// A symbolic rm builds the 4-way ITE: pinning the result to the RZ answer
/// forces the rm low-2-bits away from RNE.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_qop_with_rm_symbolic_rm_f64() {
    let ctx = SymContext::new_mock();
    let (a, b, c) = fma_tie_operands();
    let rm_sym = RustBV::symbolic(&ctx, "fma_rm", 32);

    let result =
        VEXOps::qop_with_rm(IROp::FMAdd(IRType::F64), rm_sym.clone(), a, b, c, &ctx).unwrap();
    assert!(result.is_symbolic(), "symbolic rm must yield a Z3 expression");

    ctx.add_constraint(
        result
            .to_z3_ast()
            .eq(RustBV::concrete(u128::from(1.0f64.to_bits()), 64).to_z3_ast()),
    );
    assert!(ctx.is_sat(), "the RZ/RD answer must be reachable");
    let rm_val = ctx.eval(&rm_sym).expect("eval failed") & 0x3;
    assert!(
        rm_val == 1 || rm_val == 3,
        "only RD/RZ truncate the 1.5ulp sum, got rm={rm_val}"
    );
}

/// Non-FMA Qops carry no rounding mode; `qop_with_rm` delegates to `qop` so
/// the typed error surface is unchanged.
#[test]
fn test_qop_with_rm_rejects_non_quaternary() {
    let ctx = SymContext::new_mock();
    let zero = || RustBV::concrete(0, 64);
    let err = VEXOps::qop_with_rm(IROp::Add(IRType::I64), zero(), zero(), zero(), zero(), &ctx)
        .unwrap_err();
    match err {
        OpError::NotQuaternary(_) => (),
        other => panic!("expected NotQuaternary, got {other:?}"),
    }
}
