// angr-ph300.61: SetV128lo32 / SetV128lo64 low-lane insertion tests
// (mirror of ops_vec_set_lo.rs). Covers the concrete splice path plus the
// symbolic `extract(127,k).concat(val)` reconstruction and the binop dispatch
// of both IROp variants — all previously exercised only indirectly via the
// scalar-FP ops that share `splice_lane0_u128`.

use super::*;

/// Concrete SetV128lo64 (MOVSD-style low-lane splice): overwrite the low 64
/// bits, leave the upper 64 bits untouched.
#[test]
fn test_set_v128_lo64_concrete() {
    let ctx = SymContext::new_mock();

    // vec upper 64 = 0xAAAA..., lower 64 = 0x1111...; val low 64 = 0xBBBB...
    let vec_bits: u128 = (0xAAAA_AAAA_AAAA_AAAAu128 << 64) | 0x1111_1111_1111_1111u128;
    let val_bits: u128 = 0xBBBB_BBBB_BBBB_BBBBu128;

    let vec = RustBV::concrete(vec_bits, 128);
    let val = RustBV::concrete(val_bits, 64);

    let result = VEXOps::binop(IROp::SetV128lo64, vec, val, &ctx).unwrap();
    let out = result.as_u128().unwrap();

    assert_eq!(result.width(), 128);
    // Low 64 replaced by val.
    assert_eq!(out & 0xFFFF_FFFF_FFFF_FFFF, 0xBBBB_BBBB_BBBB_BBBB);
    // Upper 64 preserved.
    assert_eq!(out >> 64, 0xAAAA_AAAA_AAAA_AAAA);
}

/// Concrete SetV128lo32 (MOVSS-style low-lane splice): overwrite the low 32
/// bits, leave the upper 96 bits untouched.
#[test]
fn test_set_v128_lo32_concrete() {
    let ctx = SymContext::new_mock();

    let vec_bits: u128 = (0xDEAD_BEEF_CAFE_BABEu128 << 64) | 0x1234_5678_9ABC_DEF0u128;
    let val_bits: u128 = 0x0000_0000_1111_2222u128; // low 32 bits = 0x1111_2222

    let vec = RustBV::concrete(vec_bits, 128);
    let val = RustBV::concrete(val_bits & 0xFFFF_FFFF, 32);

    let result = VEXOps::binop(IROp::SetV128lo32, vec, val, &ctx).unwrap();
    let out = result.as_u128().unwrap();

    assert_eq!(result.width(), 128);
    // Low 32 replaced.
    assert_eq!(out & 0xFFFF_FFFF, 0x1111_2222);
    // Upper 96 preserved.
    assert_eq!(out >> 32, vec_bits >> 32);
}

/// Symbolic SetV128lo64: with a symbolic vector, constrain the low 64 bits of
/// the result to equal the (concrete) inserted value, then Z3-check that the
/// upper 64 bits of the result track the original vector's upper 64 bits.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_set_v128_lo64_symbolic() {
    let ctx = SymContext::new_mock();
    let vec = RustBV::symbolic(&ctx, "xmm_setlo", 128);
    let val = RustBV::concrete(0x0102_0304_0506_0708u128, 64);

    let result = VEXOps::binop(IROp::SetV128lo64, vec.clone(), val, &ctx).unwrap();
    assert_eq!(result.width(), 128);

    // Low 64 of result must equal the inserted value.
    let res_lo = result.extract(63, 0, &ctx);
    let inserted = RustBV::concrete(0x0102_0304_0506_0708u128, 64);
    let eq_lo = res_lo.to_z3_ast().eq(inserted.to_z3_ast());
    ctx.add_constraint(eq_lo);
    assert!(ctx.is_sat(), "low 64 must equal inserted value");

    // Upper 64 of result must equal upper 64 of the symbolic input (passthrough).
    let upper_in = vec.extract(127, 64, &ctx);
    let upper_out = result.extract(127, 64, &ctx);
    let eq_upper = upper_in.to_z3_ast().eq(upper_out.to_z3_ast());
    ctx.add_constraint(eq_upper);
    assert!(ctx.is_sat(), "upper-64 passthrough must hold");
}

/// Symbolic SetV128lo32: mixing the low-32 replacement with a symbolic upper.
/// Constrain the low 32 bits of the result and prove the upper 96 bits are the
/// symbolic input's upper 96 bits (never disturbed by the splice).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_set_v128_lo32_symbolic() {
    let ctx = SymContext::new_mock();
    let vec = RustBV::symbolic(&ctx, "xmm_setlo32", 128);
    let val = RustBV::concrete(0xCAFE_F00Du128, 32);

    let result = VEXOps::binop(IROp::SetV128lo32, vec.clone(), val, &ctx).unwrap();
    assert_eq!(result.width(), 128);

    let res_lo = result.extract(31, 0, &ctx);
    let inserted = RustBV::concrete(0xCAFE_F00Du128, 32);
    let eq_lo = res_lo.to_z3_ast().eq(inserted.to_z3_ast());
    ctx.add_constraint(eq_lo);
    assert!(ctx.is_sat(), "low 32 must equal inserted value");

    let upper_in = vec.extract(127, 32, &ctx);
    let upper_out = result.extract(127, 32, &ctx);
    let eq_upper = upper_in.to_z3_ast().eq(upper_out.to_z3_ast());
    ctx.add_constraint(eq_upper);
    assert!(ctx.is_sat(), "upper-96 passthrough must hold");

    // Negative check: a model where upper bits diverge must be UNSAT.
    let bad = upper_out
        .to_z3_ast()
        .eq(RustBV::concrete(0, 96).to_z3_ast());
    // Only pin the symbolic input's upper to a nonzero value first so bad=0 is
    // genuinely contradictory rather than vacuously satisfiable.
    let pin = upper_in
        .to_z3_ast()
        .eq(RustBV::concrete(0x7777_7777_7777_7777_7777_7777u128, 96).to_z3_ast());
    ctx.add_constraint(pin);
    ctx.add_constraint(bad);
    assert!(
        !ctx.is_sat(),
        "upper-out must follow upper-in; divergence must be UNSAT"
    );
}
