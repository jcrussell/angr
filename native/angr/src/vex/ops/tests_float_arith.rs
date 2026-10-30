// angr-9hleg: scalar FP arith tests incl. rounding-mode variants (mirror of ops/float_arith.rs).

use crate::symbolic::ToZ3;
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
#[test]
fn test_float_add_symbolic_constraint() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("x", 32);
    let two = RustBV::concrete(2.0f32.to_bits() as u128, 32);

    let sum = VEXOps::binop(IROp::FAdd(IRType::F32), x.clone(), two, &ctx).unwrap();

    // Constrain: sum's IEEE bits == bits(5.0).
    let five = RustBV::concrete(5.0f32.to_bits() as u128, 32);
    let eq = sum.to_z3_ast().unwrap().eq(five.to_z3_ast().unwrap());
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
#[test]
fn test_float_sqrt_symbolic_constraint() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("sqrt_x", 64);

    let sqrt_x = VEXOps::unop(IROp::FSqrt(IRType::F64), x.clone(), &ctx).unwrap();

    let four = RustBV::concrete(4.0f64.to_bits() as u128, 64);
    let eq = sqrt_x.to_z3_ast().unwrap().eq(four.to_z3_ast().unwrap());
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
/// ∈ {1, 3} (RD or RZ); 0x3DCCCCCD is what RNE, RU and ties-away give.
/// Exercises the `VexRm` ITE built by `value_clarirs::with_vex_rm`.
#[test]
fn test_float_div_with_symbolic_rm_f32() {
    let ctx = SymContext::new_mock();
    let rm = RustBV::symbol("rm_div_f32", 32);
    let one = RustBV::concrete(1.0f32.to_bits() as u128, 32);
    let ten = RustBV::concrete(10.0f32.to_bits() as u128, 32);

    let result =
        VEXOps::binop_with_rm(IROp::FDiv(IRType::F32), rm.clone(), one, ten, &ctx).unwrap();
    let target = RustBV::concrete(0x3DCCCCCC, 32);
    let eq = result.to_z3_ast().unwrap().eq(target.to_z3_ast().unwrap());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT for div(1,10) == 0x3DCCCCCC");

    let model_rm = ctx.eval(&rm).expect("eval(rm) returned None");
    assert!(
        model_rm == 1 || model_rm == 3,
        "expected rm ∈ {{1, 3}} (RD/RZ), got {model_rm}"
    );
}

/// FAdd with concrete RZ on inexact-sum operands. 0x3F800001 + 0x3F800002
/// = 2.0 + 1.5ulp (exact). RNE rounds-to-even → 2.0 + 2ulp = 0x40000002;
/// RZ truncates → 2.0 + 1ulp = 0x40000001.
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
/// rm-aware path, NOT collapse to a fresh unconstrained symbolic
/// (angr-tfjl). `a` is symbolic; FAdd(a, b)
/// under RZ. Constraining `a == 0x3F800001` must pin the result to the
/// RZ-truncated sum `0x40000001` (a fresh-symbolic result would leave it
/// free, making `result == 0x40000002` satisfiable too).
#[test]
fn test_float_add_symbolic_operand_with_rm_rz_f32() {
    let ctx = SymContext::new_mock();
    let rm_rz = RustBV::concrete(3, 32);
    let a = RustBV::symbol("fadd_sym_a", 32);
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
        a.to_z3_ast().unwrap()
            .eq(RustBV::concrete(0x3F800001, 32).to_z3_ast().unwrap()),
    );
    assert!(ctx.is_sat(), "pinned symbolic operand must keep ctx SAT");

    let bits = ctx.eval(&result).expect("eval failed") as u32;
    assert_eq!(
        bits, 0x40000001,
        "RZ truncates the 1.5ulp tie of the pinned symbolic operand"
    );
}

// ---------------------------------------------------------------------------
// Qop (FMAdd / FMSub) rounding-mode threading (angr-03vl4.30)
// ---------------------------------------------------------------------------

/// Witness for FMA rounding: `1.0 * 1.0 + 1.5*2^-53` sits 1.5 ULP above 1.0,
/// so RNE rounds *up* to `1.0 + 2^-52` while RZ truncates back to `1.0`. Any
/// path that drops the Qop's rm operand (angr-03vl4.30)
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

/// A symbolic rm builds the `VexRm` ITE: pinning the result to the RZ answer
/// forces rm to RD or RZ (ties-away rounds this positive tie up).
#[test]
fn test_qop_with_rm_symbolic_rm_f64() {
    let ctx = SymContext::new_mock();
    let (a, b, c) = fma_tie_operands();
    let rm_sym = RustBV::symbol("fma_rm", 32);

    let result =
        VEXOps::qop_with_rm(IROp::FMAdd(IRType::F64), rm_sym.clone(), a, b, c, &ctx).unwrap();
    assert!(result.is_symbolic(), "symbolic rm must yield a Z3 expression");

    ctx.add_constraint(
        result
            .to_z3_ast().unwrap()
            .eq(RustBV::concrete(u128::from(1.0f64.to_bits()), 64).to_z3_ast().unwrap()),
    );
    assert!(ctx.is_sat(), "the RZ/RD answer must be reachable");
    let rm_val = ctx.eval(&rm_sym).expect("eval failed");
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

// ---------------------------------------------------------------------------
// angr-0bh1z: Iop_MaxNumF32/64, Iop_MinNumF32/64 (AArch32 VMAXNM/VMINNM).
// ---------------------------------------------------------------------------

/// Ordinary (no-NaN) pairs behave like plain max/min at both widths.
#[test]
fn test_max_num_min_num_concrete_ordinary() {
    let ctx = SymContext::new_mock();

    let a32 = RustBV::concrete(1.5f32.to_bits() as u128, 32);
    let b32 = RustBV::concrete((-2.5f32).to_bits() as u128, 32);
    let max = VEXOps::binop(IROp::FMaxNum(IRType::F32), a32.clone(), b32.clone(), &ctx).unwrap();
    let min = VEXOps::binop(IROp::FMinNum(IRType::F32), a32, b32, &ctx).unwrap();
    assert_eq!(f32::from_bits(max.as_u64().unwrap() as u32), 1.5);
    assert_eq!(f32::from_bits(min.as_u64().unwrap() as u32), -2.5);

    let a64 = RustBV::concrete(1.5f64.to_bits() as u128, 64);
    let b64 = RustBV::concrete((-2.5f64).to_bits() as u128, 64);
    let max = VEXOps::binop(IROp::FMaxNum(IRType::F64), a64.clone(), b64.clone(), &ctx).unwrap();
    let min = VEXOps::binop(IROp::FMinNum(IRType::F64), a64, b64, &ctx).unwrap();
    assert_eq!(f64::from_bits(max.as_u64().unwrap()), 1.5);
    assert_eq!(f64::from_bits(min.as_u64().unwrap()), -2.5);
}

/// The whole point of maxNum/minNum: with exactly one NaN operand the *other*
/// operand is returned, from either side. The compare-and-select `VFMax`/
/// `VFMin` lane ops deliberately do not do this.
#[test]
fn test_max_num_min_num_propagates_the_non_nan_operand() {
    let ctx = SymContext::new_mock();
    let nan = RustBV::concrete(f32::NAN.to_bits() as u128, 32);
    let num = RustBV::concrete(3.25f32.to_bits() as u128, 32);

    for op in [IROp::FMaxNum(IRType::F32), IROp::FMinNum(IRType::F32)] {
        let nan_left = VEXOps::binop(op, nan.clone(), num.clone(), &ctx).unwrap();
        let nan_right = VEXOps::binop(op, num.clone(), nan.clone(), &ctx).unwrap();
        assert_eq!(
            f32::from_bits(nan_left.as_u64().unwrap() as u32),
            3.25,
            "{op:?} with NaN on the left"
        );
        assert_eq!(
            f32::from_bits(nan_right.as_u64().unwrap() as u32),
            3.25,
            "{op:?} with NaN on the right"
        );
    }

    // Both NaN -> NaN.
    let both = VEXOps::binop(IROp::FMaxNum(IRType::F32), nan.clone(), nan, &ctx).unwrap();
    assert!(f32::from_bits(both.as_u64().unwrap() as u32).is_nan());
}

/// `maxNum(+0,-0) = +0` and `minNum(+0,-0) = -0`, in both operand orders —
/// the ARM ARM FPMax/FPMin contract that `f32::max` explicitly does not pin.
#[test]
fn test_max_num_min_num_signed_zero() {
    let ctx = SymContext::new_mock();
    let pos = RustBV::concrete(0.0f32.to_bits() as u128, 32);
    let neg = RustBV::concrete((-0.0f32).to_bits() as u128, 32);

    let bits = |bv: RustBV| bv.as_u64().unwrap() as u32;
    for (l, r) in [(pos.clone(), neg.clone()), (neg.clone(), pos.clone())] {
        let max = VEXOps::binop(IROp::FMaxNum(IRType::F32), l.clone(), r.clone(), &ctx).unwrap();
        let min = VEXOps::binop(IROp::FMinNum(IRType::F32), l, r, &ctx).unwrap();
        assert_eq!(bits(max), 0.0f32.to_bits(), "maxNum(+0,-0) must be +0");
        assert_eq!(bits(min), (-0.0f32).to_bits(), "minNum(+0,-0) must be -0");
    }
}

/// Symbolic maxNum: constraining `maxNum(x, 2.0) == 5.0` forces x == 5.0, so
/// the Z3 `fp.max` wiring actually propagates rather than minting a fresh
/// unconstrained symbol.
#[test]
fn test_max_num_symbolic_constraint() {
    let ctx = SymContext::new_mock();
    let x = RustBV::symbol("x", 32);
    let two = RustBV::concrete(2.0f32.to_bits() as u128, 32);

    let m = VEXOps::binop(IROp::FMaxNum(IRType::F32), x.clone(), two, &ctx).unwrap();
    let five = RustBV::concrete(5.0f32.to_bits() as u128, 32);
    ctx.add_constraint(m.to_z3_ast().unwrap().eq(five.to_z3_ast().unwrap()));
    assert!(ctx.is_sat(), "expected SAT after FMaxNum symbolic constraint");

    let model_x = ctx.eval(&x).expect("eval(x) returned None");
    let result_f = f32::from_bits(model_x as u32);
    assert!(
        (result_f - 5.0).abs() < 1e-6,
        "Expected x == 5.0, got {result_f}"
    );
}

// angr-mx00p.31: `apply_rounding` mode 0 is std `round_ties_even`. Pin the
// edge cases the hand-rolled RNE it replaced was written around, against
// literal expected values rather than the std function itself.

/// RNE ties, the ±0.5 sign of zero, values too large to carry a fraction,
/// and the largest ties each precision can represent.
#[test]
fn test_apply_rounding_nearest_edge_cases() {
    use super::float_arith::apply_rounding;
    use crate::symbolic::VexRm;
    let f64_cases: &[(f64, f64)] = &[
        (0.5, 0.0),
        (1.5, 2.0),
        (2.5, 2.0),
        (3.5, 4.0),
        (-0.5, -0.0),
        (-2.5, -2.0),
        (-3.5, -4.0),
        (2.4999999999999996, 2.0),
        (-0.0, -0.0),
        (1e300, 1e300),
        (-1e300, -1e300),
        // 2^52 - 0.5 is the largest f64 tie; 2^52 - 1 is odd, so it rounds up.
        (4_503_599_627_370_495.5, 4_503_599_627_370_496.0),
        (f64::MAX, f64::MAX),
        (f64::INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, f64::NEG_INFINITY),
    ];
    for &(input, want) in f64_cases {
        let got = apply_rounding(input, VexRm::NearestEven);
        assert_eq!(got.to_bits(), want.to_bits(), "f64 RNE({input:e})");
    }
    let f32_cases: &[(f32, f32)] = &[
        (2.5, 2.0),
        (3.5, 4.0),
        (-0.5, -0.0),
        (-3.5, -4.0),
        // 2^23 - 0.5 is the largest f32 tie; 2^23 - 1 is odd, so it rounds up.
        (8_388_607.5, 8_388_608.0),
        (1e38, 1e38),
        (f32::INFINITY, f32::INFINITY),
        (f32::NEG_INFINITY, f32::NEG_INFINITY),
    ];
    for &(input, want) in f32_cases {
        let got = apply_rounding(input, VexRm::NearestEven);
        assert_eq!(got.to_bits(), want.to_bits(), "f32 RNE({input:e})");
    }
    // A quiet NaN comes back a NaN with its payload intact.
    let qnan = f64::from_bits(0x7ff8_0000_0000_1234);
    assert_eq!(apply_rounding(qnan, VexRm::NearestEven).to_bits(), qnan.to_bits());
    let qnan32 = f32::from_bits(0x7fc0_1234);
    assert_eq!(apply_rounding(qnan32, VexRm::NearestEven).to_bits(), qnan32.to_bits());
    // A signaling NaN comes back quieted, payload kept: the hand-rolled RNE
    // fell through to `round` for NaN, and std `f32::round_ties_even` alone
    // would return the f32 sNaN unchanged.
    assert_eq!(apply_rounding(f32::from_bits(0x7f80_0001), VexRm::NearestEven).to_bits(), 0x7fc0_0001);
    assert_eq!(
        apply_rounding(f64::from_bits(0x7ff0_0000_0000_0001), VexRm::NearestEven).to_bits(),
        0x7ff8_0000_0000_0001
    );
}

/// `VexRm::NearestAway` (libVEX mode 4, AArch64 FCVTAS/FCVTAU) rounds ties
/// away from zero and everything else to nearest, keeping the sign of zero
/// (angr-vmi2l.56).
#[test]
fn test_apply_rounding_nearest_away() {
    use super::float_arith::apply_rounding;
    use crate::symbolic::VexRm;
    for &(input, want) in &[
        (0.5f64, 1.0f64),
        (1.5, 2.0),
        (2.5, 3.0),
        (-0.5, -1.0),
        (-2.5, -3.0),
        (2.4999999999999996, 2.0),
        (-0.3, -0.0),
        (-0.0, -0.0),
        (4_503_599_627_370_495.5, 4_503_599_627_370_496.0),
        (f64::INFINITY, f64::INFINITY),
    ] {
        let got = apply_rounding(input, VexRm::NearestAway);
        assert_eq!(got.to_bits(), want.to_bits(), "f64 RNA({input:e})");
    }
    for &(input, want) in &[(2.5f32, 3.0f32), (-2.5, -3.0), (0.5, 1.0), (8_388_606.5, 8_388_607.0)] {
        let got = apply_rounding(input, VexRm::NearestAway);
        assert_eq!(got.to_bits(), want.to_bits(), "f32 RNA({input:e})");
    }
}

/// The implicit-RNE float-to-int conversions saturate out-of-range results
/// and map NaN to 0 after rounding (Rust `as` semantics).
#[test]
fn test_float_to_int_rne_saturates_and_maps_nan_to_zero() {
    let ctx = SymContext::new_mock();
    let conv = |op: IROp, bits: u128, width: u32| {
        VEXOps::unop(op, RustBV::concrete(bits, width), &ctx)
            .unwrap()
            .as_u128()
            .unwrap()
    };
    let f64b = |f: f64| f.to_bits() as u128;
    assert_eq!(conv(IROp::F64toI32S, f64b(2.5), 64), 2);
    assert_eq!(conv(IROp::F64toI32S, f64b(-2.5), 64), (-2i32) as u32 as u128);
    assert_eq!(conv(IROp::F64toI32S, f64b(1e300), 64), i32::MAX as u32 as u128);
    assert_eq!(conv(IROp::F64toI32S, f64b(-1e300), 64), i32::MIN as u32 as u128);
    assert_eq!(conv(IROp::F64toI32S, f64b(f64::NAN), 64), 0);
    assert_eq!(conv(IROp::F64toI64U, f64b(-0.5), 64), 0);
    assert_eq!(conv(IROp::F64toI64U, f64b(f64::INFINITY), 64), u64::MAX as u128);
}
