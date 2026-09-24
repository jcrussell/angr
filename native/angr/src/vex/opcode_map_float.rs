//! Floating-point and transcendental opcode parsing for
//! [`super::opcode_map`].
//!
//! Holds the two `parse_*` families that `parse_opcode` consults between the
//! integer conversions and the vector families: `parse_float` (scalar and
//! packed FP arithmetic, comparisons, conversions and the NEON
//! reciprocal/rsqrt estimate steps) and `parse_transcendental` (the libm-backed
//! x87 / AArch64 ops that ride in `IROp::Raw`).
//!
//! Siblings: `super::opcode_map` (dispatcher, macros, integer families,
//! `parse_special`, type/endness/jumpkind parsing) and
//! `super::opcode_map_vector` (SIMD families).
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::ir::{FCmpKind, IROp, IRType};
use super::opcode_map::{
    cast_arms, fcmp_scalar_arms, fcmp_vec_arms, scalar_arms, tuple_arms, vec_arms,
};

/// Parse floating point operations
pub(super) fn parse_float(op_str: &str) -> Option<IROp> {
    // Basic FP arithmetic + unary + fused multiply-add/sub.
    tuple_arms!(op_str; "Iop_Add"  => FAdd  { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_Sub"  => FSub  { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_Mul"  => FMul  { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_Div"  => FDiv  { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_Neg"  => FNeg  { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_Abs"  => FAbs  { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_Sqrt" => FSqrt { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_MAdd" => FMAdd { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_MSub" => FMSub { "F32" => F32, "F64" => F64 });

    // IEEE-754-2008 max-number / min-number (AArch32 VMAXNM / VMINNM, emitted
    // by guest_arm_toIR.c). Scalar `F32`/`F64` shapes only — VEX declares no
    // vector `Iop_MaxNum*x*`; the packed ARM forms lift to `Iop_Max32Fx4` &c.
    // These must be matched *before* the `Iop_Max`/`Iop_Min` vector arms below
    // would ever see them, which the exact-suffix `tuple_arms!` match gives
    // for free ("Iop_MaxNumF32" never equals "Iop_Max" + a vector suffix).
    tuple_arms!(op_str; "Iop_MaxNum" => FMaxNum { "F32" => F32, "F64" => F64 });
    tuple_arms!(op_str; "Iop_MinNum" => FMinNum { "F32" => F32, "F64" => F64 });

    // Scalar-in-vector float ops (SSE scalar: ADDSS, SUBSS, MULSS, DIVSS, etc.).
    scalar_arms!(op_str; "Iop_Add"  => VFAddS  { "32F0x4" => F32, "64F0x2" => F64 });
    scalar_arms!(op_str; "Iop_Sub"  => VFSubS  { "32F0x4" => F32, "64F0x2" => F64 });
    scalar_arms!(op_str; "Iop_Mul"  => VFMulS  { "32F0x4" => F32, "64F0x2" => F64 });
    scalar_arms!(op_str; "Iop_Div"  => VFDivS  { "32F0x4" => F32, "64F0x2" => F64 });
    scalar_arms!(op_str; "Iop_Sqrt" => VFSqrtS { "32F0x4" => F32, "64F0x2" => F64 });
    scalar_arms!(op_str; "Iop_Max"  => VFMaxS  { "32F0x4" => F32, "64F0x2" => F64 });
    scalar_arms!(op_str; "Iop_Min"  => VFMinS  { "32F0x4" => F32, "64F0x2" => F64 });

    // Packed (whole-vector) float ops — SSE / AVX / NEON.
    // The 32Fx2 shape is the ARM NEON D-reg 2-lane form (VADD.F32 &c.). It was
    // missing from Add/Sub/Mul/Min/Max until angr-sqfj8.114 even though the
    // evaluator is generic over (elem, count). Div and Sqrt have no 32Fx2 arm
    // because VEX declares no `Iop_Div32Fx2`/`Iop_Sqrt32Fx2`.
    //
    // Abs/Neg have their own shape set (angr-sqfj8.115): the header declares
    // exactly `Iop_Abs32Fx{2,4}` / `Iop_Abs64Fx2` and `Iop_Neg32Fx{2,4}` /
    // `Iop_Neg64Fx2` — no 256-bit form of either, so unlike Add/Sub/Mul/Min/Max
    // they get no `32Fx8`/`64Fx4` arms (Abs had two dead ones until .115).
    vec_arms!(op_str; "Iop_Add"  => VFAdd  { "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "64Fx2" => (F64, 2), "32Fx8" => (F32, 8), "64Fx4" => (F64, 4) });
    vec_arms!(op_str; "Iop_Sub"  => VFSub  { "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "64Fx2" => (F64, 2), "32Fx8" => (F32, 8), "64Fx4" => (F64, 4) });
    vec_arms!(op_str; "Iop_Mul"  => VFMul  { "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "64Fx2" => (F64, 2), "32Fx8" => (F32, 8), "64Fx4" => (F64, 4) });
    vec_arms!(op_str; "Iop_Div"  => VFDiv  { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2), "32Fx8" => (F32, 8), "64Fx4" => (F64, 4) });
    vec_arms!(op_str; "Iop_Sqrt" => VFSqrt { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2), "32Fx8" => (F32, 8), "64Fx4" => (F64, 4) });
    vec_arms!(op_str; "Iop_Abs"  => VFAbs  { "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "64Fx2" => (F64, 2) });
    vec_arms!(op_str; "Iop_Neg"  => VFNeg  { "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "64Fx2" => (F64, 2) });
    vec_arms!(op_str; "Iop_Min"  => VFMin  { "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "64Fx2" => (F64, 2), "32Fx8" => (F32, 8), "64Fx4" => (F64, 4) });
    vec_arms!(op_str; "Iop_Max"  => VFMax  { "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "64Fx2" => (F64, 2), "32Fx8" => (F32, 8), "64Fx4" => (F64, 4) });

    // NEON pairwise FP add/max/min — `Iop_PwAdd32Fx2` (ARM VPADD.F32, D-reg),
    // `Iop_PwMax32Fx{2,4}` / `Iop_PwMin32Fx{2,4}` (VPMAX.F32 / VPMIN.F32,
    // AArch64 FMAXP / FMINP). These are the FP variants of the `Pw*` family;
    // the integer `Iop_Pw{Add,Min,Max}{N}{S/U}x{M}` are routed in parse_vector
    // to VPwAdd / VPwMin / VPwMax and never reach these arms (the `F` in the
    // suffix makes the two sets disjoint). Matched here because parse_float
    // runs before both parse_vector and parse_neon_unimplemented in
    // parse_opcode. VEX declares no `Iop_PwAdd32Fx4`, hence the single add arm.
    vec_arms!(op_str; "Iop_PwAdd" => VFPwAdd { "32Fx2" => (F32, 2) });
    vec_arms!(op_str; "Iop_PwMax" => VFPwMax { "32Fx2" => (F32, 2), "32Fx4" => (F32, 4) });
    vec_arms!(op_str; "Iop_PwMin" => VFPwMin { "32Fx2" => (F32, 2), "32Fx4" => (F32, 4) });

    // FP reciprocal estimate (1/x) — RCPPS / NEON FRECPE. F0x4 = SSE scalar.
    scalar_arms!(op_str; "Iop_RecipEst" => VFRecipEstS { "32F0x4" => F32 });
    vec_arms!(op_str; "Iop_RecipEst" => VFRecipEst {
        "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "32Fx8" => (F32, 8), "64Fx2" => (F64, 2),
    });
    // FP Newton-Raphson reciprocal step — NEON FRECPS.
    vec_arms!(op_str; "Iop_RecipStep" => VFRecipStep {
        "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "64Fx2" => (F64, 2),
    });
    // FP reciprocal-sqrt estimate (1/sqrt(x)) — RSQRTPS / NEON FRSQRTE.
    scalar_arms!(op_str; "Iop_RSqrtEst" => VFRSqrtEstS { "32F0x4" => F32 });
    vec_arms!(op_str; "Iop_RSqrtEst" => VFRSqrtEst {
        "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "32Fx8" => (F32, 8), "64Fx2" => (F64, 2),
    });
    // FP Newton-Raphson reciprocal-sqrt step — NEON FRSQRTS.
    vec_arms!(op_str; "Iop_RSqrtStep" => VFRSqrtStep {
        "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "64Fx2" => (F64, 2),
    });

    // FP comparisons (scalar I1 result — used internally and for ccall lifts).
    tuple_arms!(op_str; "Iop_CmpF" => FComCC { "32" => F32, "64" => F64 });

    // SSE scalar-lane compares: lane 0 → all-1s/0 mask, upper lanes pass-through.
    fcmp_scalar_arms!(op_str; "Iop_CmpEQ" => Eq { "32F0x4" => F32, "64F0x2" => F64 });
    fcmp_scalar_arms!(op_str; "Iop_CmpLT" => Lt { "32F0x4" => F32, "64F0x2" => F64 });
    fcmp_scalar_arms!(op_str; "Iop_CmpLE" => Le { "32F0x4" => F32, "64F0x2" => F64 });
    fcmp_scalar_arms!(op_str; "Iop_CmpUN" => Un { "32F0x4" => F32, "64F0x2" => F64 });

    // Packed FP compares (SSE cmpps/cmppd, ARM NEON 32Fx2). Per-lane mask:
    // each lane independently produces all-1s (true) or 0 (false).
    // 32Fx2 returns I64 (ARM NEON), 32Fx4 / 64Fx2 return V128 (SSE).
    fcmp_vec_arms!(op_str; "Iop_CmpEQ" => Eq { "32Fx2" => (F32, 2), "32Fx4" => (F32, 4), "64Fx2" => (F64, 2) });
    fcmp_vec_arms!(op_str; "Iop_CmpLT" => Lt { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2) });
    fcmp_vec_arms!(op_str; "Iop_CmpLE" => Le { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2) });
    fcmp_vec_arms!(op_str; "Iop_CmpGT" => Gt { "32Fx2" => (F32, 2), "32Fx4" => (F32, 4) });
    fcmp_vec_arms!(op_str; "Iop_CmpGE" => Ge { "32Fx2" => (F32, 2), "32Fx4" => (F32, 4) });
    fcmp_vec_arms!(op_str; "Iop_CmpUN" => Un { "32Fx4" => (F32, 4), "64Fx2" => (F64, 2) });

    // Reinterpret as different type.
    cast_arms!(op_str; "Iop_Reinterp" => Reinterpret {
        "F32asI32" => (F32, I32), "I32asF32" => (I32, F32),
        "F64asI64" => (F64, I64), "I64asF64" => (I64, F64),
    });

    match op_str {
        // SetV128lo operations
        "Iop_SetV128lo32" => Some(IROp::SetV128lo32),
        "Iop_SetV128lo64" => Some(IROp::SetV128lo64),

        // FP conversions (these are nullary tuple variants — no IRType inside).
        "Iop_F32toF64" => Some(IROp::F32toF64),
        "Iop_F64toF32" => Some(IROp::F64toF32),
        "Iop_I32StoF32" => Some(IROp::I32StoF32),
        "Iop_I32StoF64" => Some(IROp::I32StoF64),
        "Iop_I64StoF32" => Some(IROp::I64StoF32),
        "Iop_I64StoF64" => Some(IROp::I64StoF64),
        "Iop_I32UtoF32" => Some(IROp::I32UtoF32),
        "Iop_I32UtoF64" => Some(IROp::I32UtoF64),
        "Iop_I64UtoF32" => Some(IROp::I64UtoF32),
        "Iop_I64UtoF64" => Some(IROp::I64UtoF64),
        "Iop_F32toI32S" => Some(IROp::F32toI32S),
        "Iop_F64toI32S" => Some(IROp::F64toI32S),
        "Iop_F32toI64S" => Some(IROp::F32toI64S),
        "Iop_F64toI64S" => Some(IROp::F64toI64S),
        "Iop_F32toI32U" => Some(IROp::F32toI32U),
        "Iop_F64toI32U" => Some(IROp::F64toI32U),
        "Iop_F32toI64U" => Some(IROp::F32toI64U),
        "Iop_F64toI64U" => Some(IROp::F64toI64U),

        // Rounding
        "Iop_RoundF32toInt" => Some(IROp::RoundF32toInt),
        "Iop_RoundF64toInt" => Some(IROp::RoundF64toInt),

        _ => None,
    }
}

/// Parse the x87 / AArch64 transcendentals that have no dedicated `IROp`
/// variant and are evaluated by the libm fast paths in
/// [`crate::vex::transcendentals`].
///
/// These map to `IROp::Raw(tag)`, which is the *only* producer of that
/// variant — the `IROp::Raw` arms of `VEXOps::binop` (in its private
/// `binop_misc` helper) and `VEXOps::binop_with_rm` are its only
/// consumers, and both route straight into `transcendentals`. The tag
/// values are the libVEX `Iop_*` discriminants, but nothing compares them
/// against libVEX any more: they are an internal token that only has to
/// agree with the `IOP_*` consts in `transcendentals.rs`.
///
/// Between angr-h0ur (which deleted the native-lift feature, the last
/// caller of the numeric `parse_opcode_from_u32`) and angr-9ke6b.233,
/// nothing constructed `IROp::Raw` at all, so every one of these opcodes
/// fell through to `IROp::Unmapped` and the libm paths were dead outside
/// their own unit tests.
pub(super) fn parse_transcendental(op_str: &str) -> Option<IROp> {
    use super::transcendentals as tr;
    let tag = match op_str {
        // Binop (rm, x) -> F64
        "Iop_SinF64" => tr::IOP_SIN_F64,
        "Iop_CosF64" => tr::IOP_COS_F64,
        "Iop_TanF64" => tr::IOP_TAN_F64,
        "Iop_2xm1F64" => tr::IOP_2XM1_F64,
        // Binop (rm, x) -> F32/F64: AArch64 FRECPX, closed-form.
        "Iop_RecpExpF64" => tr::IOP_RECPEXP_F64,
        "Iop_RecpExpF32" => tr::IOP_RECPEXP_F32,
        // Triop (rm, x, y) -> F64
        "Iop_AtanF64" => tr::IOP_ATAN_F64,
        "Iop_Yl2xF64" => tr::IOP_YL2X_F64,
        "Iop_Yl2xp1F64" => tr::IOP_YL2XP1_F64,
        "Iop_ScaleF64" => tr::IOP_SCALE_F64,
        _ => return None,
    };
    Some(IROp::Raw(tag))
}
