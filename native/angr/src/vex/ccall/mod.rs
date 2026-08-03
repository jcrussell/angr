//! Clean call (CCall) implementations for VEX IR.
//!
//! This module implements the x86/AMD64 helper functions that VEX IR uses for
//! condition code calculations. The production entry point is
//! `handle_ccall_with_ctx`, which dispatches to the appropriate helper based on
//! the callee name; the interpreter calls it directly with `Some(ctx)` so
//! symbolic condition codes resolve against the live solver context.
//! `handle_ccall` is a thin ctx-less convenience wrapper (used only by tests).

use crate::symbolic::RustBV;
use crate::symbolic::SymContext;

mod arm32;
mod arm64;
mod x86_concrete;
mod x86_symbolic;

use arm32::*;
use arm64::*;
use x86_concrete::*;
use x86_symbolic::*;

/// CC_OP values for AMD64 (from VEX's libvex_guest_amd64.h)
mod amd64_cc_op {
    pub(crate) const G_CC_OP_COPY: u64 = 0;
    pub(crate) const G_CC_OP_ADDB: u64 = 1;
    pub(crate) const G_CC_OP_ADDW: u64 = 2;
    pub(crate) const G_CC_OP_ADDL: u64 = 3;
    pub(crate) const G_CC_OP_ADDQ: u64 = 4;
    pub(crate) const G_CC_OP_SUBB: u64 = 5;
    pub(crate) const G_CC_OP_SUBW: u64 = 6;
    pub(crate) const G_CC_OP_SUBL: u64 = 7;
    pub(crate) const G_CC_OP_SUBQ: u64 = 8;
    pub(crate) const G_CC_OP_ADCB: u64 = 9;
    pub(crate) const G_CC_OP_ADCW: u64 = 10;
    pub(crate) const G_CC_OP_ADCL: u64 = 11;
    pub(crate) const G_CC_OP_ADCQ: u64 = 12;
    pub(crate) const G_CC_OP_SBBB: u64 = 13;
    pub(crate) const G_CC_OP_SBBW: u64 = 14;
    pub(crate) const G_CC_OP_SBBL: u64 = 15;
    pub(crate) const G_CC_OP_SBBQ: u64 = 16;
    pub(crate) const G_CC_OP_LOGICB: u64 = 17;
    pub(crate) const G_CC_OP_LOGICW: u64 = 18;
    pub(crate) const G_CC_OP_LOGICL: u64 = 19;
    pub(crate) const G_CC_OP_LOGICQ: u64 = 20;
    pub(crate) const G_CC_OP_INCB: u64 = 21;
    pub(crate) const G_CC_OP_INCW: u64 = 22;
    pub(crate) const G_CC_OP_INCL: u64 = 23;
    pub(crate) const G_CC_OP_INCQ: u64 = 24;
    pub(crate) const G_CC_OP_DECB: u64 = 25;
    pub(crate) const G_CC_OP_DECW: u64 = 26;
    pub(crate) const G_CC_OP_DECL: u64 = 27;
    pub(crate) const G_CC_OP_DECQ: u64 = 28;
    pub(crate) const G_CC_OP_SHLB: u64 = 29;
    pub(crate) const G_CC_OP_SHLW: u64 = 30;
    pub(crate) const G_CC_OP_SHLL: u64 = 31;
    pub(crate) const G_CC_OP_SHLQ: u64 = 32;
    pub(crate) const G_CC_OP_SHRB: u64 = 33;
    pub(crate) const G_CC_OP_SHRW: u64 = 34;
    pub(crate) const G_CC_OP_SHRL: u64 = 35;
    pub(crate) const G_CC_OP_SHRQ: u64 = 36;
    pub(crate) const G_CC_OP_ROLB: u64 = 37;
    pub(crate) const G_CC_OP_ROLW: u64 = 38;
    pub(crate) const G_CC_OP_ROLL: u64 = 39;
    pub(crate) const G_CC_OP_ROLQ: u64 = 40;
    pub(crate) const G_CC_OP_RORB: u64 = 41;
    pub(crate) const G_CC_OP_RORW: u64 = 42;
    pub(crate) const G_CC_OP_RORL: u64 = 43;
    pub(crate) const G_CC_OP_RORQ: u64 = 44;
    pub(crate) const G_CC_OP_UMULB: u64 = 45;
    pub(crate) const G_CC_OP_UMULW: u64 = 46;
    pub(crate) const G_CC_OP_UMULL: u64 = 47;
    pub(crate) const G_CC_OP_UMULQ: u64 = 48;
    pub(crate) const G_CC_OP_SMULB: u64 = 49;
    pub(crate) const G_CC_OP_SMULW: u64 = 50;
    pub(crate) const G_CC_OP_SMULL: u64 = 51;
    pub(crate) const G_CC_OP_SMULQ: u64 = 52;
}

/// CC_OP values for X86 (from VEX's libvex_guest_x86.h)
mod x86_cc_op {
    pub(crate) const G_CC_OP_COPY: u64 = 0;
    pub(crate) const G_CC_OP_ADDB: u64 = 1;
    pub(crate) const G_CC_OP_ADDW: u64 = 2;
    pub(crate) const G_CC_OP_ADDL: u64 = 3;
    pub(crate) const G_CC_OP_SUBB: u64 = 4;
    pub(crate) const G_CC_OP_SUBW: u64 = 5;
    pub(crate) const G_CC_OP_SUBL: u64 = 6;
    pub(crate) const G_CC_OP_ADCB: u64 = 7;
    pub(crate) const G_CC_OP_ADCW: u64 = 8;
    pub(crate) const G_CC_OP_ADCL: u64 = 9;
    pub(crate) const G_CC_OP_SBBB: u64 = 10;
    pub(crate) const G_CC_OP_SBBW: u64 = 11;
    pub(crate) const G_CC_OP_SBBL: u64 = 12;
    pub(crate) const G_CC_OP_LOGICB: u64 = 13;
    pub(crate) const G_CC_OP_LOGICW: u64 = 14;
    pub(crate) const G_CC_OP_LOGICL: u64 = 15;
    pub(crate) const G_CC_OP_INCB: u64 = 16;
    pub(crate) const G_CC_OP_INCW: u64 = 17;
    pub(crate) const G_CC_OP_INCL: u64 = 18;
    pub(crate) const G_CC_OP_DECB: u64 = 19;
    pub(crate) const G_CC_OP_DECW: u64 = 20;
    pub(crate) const G_CC_OP_DECL: u64 = 21;
    pub(crate) const G_CC_OP_SHLB: u64 = 22;
    pub(crate) const G_CC_OP_SHLW: u64 = 23;
    pub(crate) const G_CC_OP_SHLL: u64 = 24;
    pub(crate) const G_CC_OP_SHRB: u64 = 25;
    pub(crate) const G_CC_OP_SHRW: u64 = 26;
    pub(crate) const G_CC_OP_SHRL: u64 = 27;
    pub(crate) const G_CC_OP_ROLB: u64 = 28;
    pub(crate) const G_CC_OP_ROLW: u64 = 29;
    pub(crate) const G_CC_OP_ROLL: u64 = 30;
    pub(crate) const G_CC_OP_RORB: u64 = 31;
    pub(crate) const G_CC_OP_RORW: u64 = 32;
    pub(crate) const G_CC_OP_RORL: u64 = 33;
    pub(crate) const G_CC_OP_UMULB: u64 = 34;
    pub(crate) const G_CC_OP_UMULW: u64 = 35;
    pub(crate) const G_CC_OP_UMULL: u64 = 36;
    pub(crate) const G_CC_OP_SMULB: u64 = 37;
    pub(crate) const G_CC_OP_SMULW: u64 = 38;
    pub(crate) const G_CC_OP_SMULL: u64 = 39;
}

/// Condition types (same for x86 and AMD64)
mod cond_type {
    pub(crate) const COND_O: u64 = 0; // Overflow
    pub(crate) const COND_NO: u64 = 1; // Not overflow
    pub(crate) const COND_B: u64 = 2; // Below (CF=1)
    pub(crate) const COND_NB: u64 = 3; // Not below (CF=0)
    pub(crate) const COND_Z: u64 = 4; // Zero (ZF=1)
    pub(crate) const COND_NZ: u64 = 5; // Not zero (ZF=0)
    pub(crate) const COND_BE: u64 = 6; // Below or equal (CF=1 or ZF=1)
    pub(crate) const COND_NBE: u64 = 7; // Not below or equal (CF=0 and ZF=0)
    pub(crate) const COND_S: u64 = 8; // Sign (SF=1)
    pub(crate) const COND_NS: u64 = 9; // Not sign (SF=0)
    pub(crate) const COND_P: u64 = 10; // Parity even (PF=1)
    pub(crate) const COND_NP: u64 = 11; // Parity odd (PF=0)
    pub(crate) const COND_L: u64 = 12; // Less (SF != OF)
    pub(crate) const COND_NL: u64 = 13; // Not less (SF == OF)
    pub(crate) const COND_LE: u64 = 14; // Less or equal (ZF=1 or SF != OF)
    pub(crate) const COND_NLE: u64 = 15; // Not less or equal (ZF=0 and SF == OF)
}

/// Flag bit offsets in EFLAGS
mod flag_shift {
    pub(crate) const G_CC_SHIFT_O: u32 = 11;
    pub(crate) const G_CC_SHIFT_S: u32 = 7;
    pub(crate) const G_CC_SHIFT_Z: u32 = 6;
    pub(crate) const G_CC_SHIFT_A: u32 = 4;
    pub(crate) const G_CC_SHIFT_C: u32 = 0;
    pub(crate) const G_CC_SHIFT_P: u32 = 2;
}

/// Flag bit masks
mod flag_mask {
    pub(crate) const G_CC_MASK_O: u64 = 1 << super::flag_shift::G_CC_SHIFT_O;
    pub(crate) const G_CC_MASK_S: u64 = 1 << super::flag_shift::G_CC_SHIFT_S;
    pub(crate) const G_CC_MASK_Z: u64 = 1 << super::flag_shift::G_CC_SHIFT_Z;
    pub(crate) const G_CC_MASK_A: u64 = 1 << super::flag_shift::G_CC_SHIFT_A;
    pub(crate) const G_CC_MASK_C: u64 = 1 << super::flag_shift::G_CC_SHIFT_C;
    pub(crate) const G_CC_MASK_P: u64 = 1 << super::flag_shift::G_CC_SHIFT_P;
}

/// Operation category implied by a CC_OP value.
#[derive(Debug, Clone, Copy, PartialEq)]
enum OpCategory {
    Copy,
    Add,
    Sub,
    Adc,
    Sbb,
    Logic,
    Inc,
    Dec,
    Shl,
    Shr,
    Rol,
    Ror,
    Umul,
    Smul,
}

/// Decoded metadata for a CC_OP value: operand width and category.
#[derive(Debug, Clone, Copy)]
struct CcOpInfo {
    nbits: u32,
    category: OpCategory,
}

/// Which x86/amd64 dialect a CCall is calling into.
#[derive(Debug, Clone, Copy, PartialEq)]
enum CcArch {
    Amd64,
    X86,
}

impl CcArch {
    /// Pick the dialect from a CCall callee name (`"amd64g_..."` vs `"x86g_..."`).
    fn from_ccall_name(name: &str) -> Self {
        if name.starts_with("amd64g") {
            CcArch::Amd64
        } else {
            CcArch::X86
        }
    }
}

/// Build the body of a per-arch `cc_op_info` decoder. Expands to an explicit
/// `match cc_op { ... }`, so the compiled dispatch is identical to a hand-
/// written match (jump-table or sorted compare per rustc's strategy) — no
/// runtime table or virtual dispatch is introduced. The macro just collapses
/// the per-family B/W/L[/Q] repetition at the source level.
///
/// `widths: [(bits, suffix-token-list)...]` groups widths under a category;
/// e.g. amd64 uses `[(8, B), (16, W), (32, L), (64, Q)]` while x86 omits Q.
/// Constants are still named explicitly per family
/// (`avoid-arithmetic-cc-op-decoder`: no cc_op-value arithmetic — explicit
/// names survive any VEX enum reordering).
macro_rules! cc_op_match {
    (
        $cc_op:expr;
        copy = $copy_path:path => $copy_bits:expr;
        $(
            $cat:ident { $( $bits:expr => $pat:path ),+ $(,)? }
        ),+ $(,)?
    ) => {{
        let (nbits, category) = match $cc_op {
            $copy_path => ($copy_bits, OpCategory::Copy),
            $($(
                $pat => ($bits, OpCategory::$cat),
            )+)+
            _ => return None,
        };
        Some(CcOpInfo { nbits, category })
    }};
}

/// Decode an AMD64 CC_OP into (nbits, category). Returns `None` for unknown values.
fn amd64_cc_op_info(cc_op: u64) -> Option<CcOpInfo> {
    use amd64_cc_op::*;
    cc_op_match!(cc_op;
        copy = G_CC_OP_COPY => 64;
        Add   { 8 => G_CC_OP_ADDB,   16 => G_CC_OP_ADDW,   32 => G_CC_OP_ADDL,   64 => G_CC_OP_ADDQ },
        Sub   { 8 => G_CC_OP_SUBB,   16 => G_CC_OP_SUBW,   32 => G_CC_OP_SUBL,   64 => G_CC_OP_SUBQ },
        Adc   { 8 => G_CC_OP_ADCB,   16 => G_CC_OP_ADCW,   32 => G_CC_OP_ADCL,   64 => G_CC_OP_ADCQ },
        Sbb   { 8 => G_CC_OP_SBBB,   16 => G_CC_OP_SBBW,   32 => G_CC_OP_SBBL,   64 => G_CC_OP_SBBQ },
        Logic { 8 => G_CC_OP_LOGICB, 16 => G_CC_OP_LOGICW, 32 => G_CC_OP_LOGICL, 64 => G_CC_OP_LOGICQ },
        Inc   { 8 => G_CC_OP_INCB,   16 => G_CC_OP_INCW,   32 => G_CC_OP_INCL,   64 => G_CC_OP_INCQ },
        Dec   { 8 => G_CC_OP_DECB,   16 => G_CC_OP_DECW,   32 => G_CC_OP_DECL,   64 => G_CC_OP_DECQ },
        Shl   { 8 => G_CC_OP_SHLB,   16 => G_CC_OP_SHLW,   32 => G_CC_OP_SHLL,   64 => G_CC_OP_SHLQ },
        Shr   { 8 => G_CC_OP_SHRB,   16 => G_CC_OP_SHRW,   32 => G_CC_OP_SHRL,   64 => G_CC_OP_SHRQ },
        Rol   { 8 => G_CC_OP_ROLB,   16 => G_CC_OP_ROLW,   32 => G_CC_OP_ROLL,   64 => G_CC_OP_ROLQ },
        Ror   { 8 => G_CC_OP_RORB,   16 => G_CC_OP_RORW,   32 => G_CC_OP_RORL,   64 => G_CC_OP_RORQ },
        Umul  { 8 => G_CC_OP_UMULB,  16 => G_CC_OP_UMULW,  32 => G_CC_OP_UMULL,  64 => G_CC_OP_UMULQ },
        Smul  { 8 => G_CC_OP_SMULB,  16 => G_CC_OP_SMULW,  32 => G_CC_OP_SMULL,  64 => G_CC_OP_SMULQ },
    )
}

/// Decode an X86 CC_OP into (nbits, category). Returns `None` for unknown values.
fn x86_cc_op_info(cc_op: u64) -> Option<CcOpInfo> {
    use x86_cc_op::*;
    cc_op_match!(cc_op;
        copy = G_CC_OP_COPY => 32;
        Add   { 8 => G_CC_OP_ADDB,   16 => G_CC_OP_ADDW,   32 => G_CC_OP_ADDL },
        Sub   { 8 => G_CC_OP_SUBB,   16 => G_CC_OP_SUBW,   32 => G_CC_OP_SUBL },
        Adc   { 8 => G_CC_OP_ADCB,   16 => G_CC_OP_ADCW,   32 => G_CC_OP_ADCL },
        Sbb   { 8 => G_CC_OP_SBBB,   16 => G_CC_OP_SBBW,   32 => G_CC_OP_SBBL },
        Logic { 8 => G_CC_OP_LOGICB, 16 => G_CC_OP_LOGICW, 32 => G_CC_OP_LOGICL },
        Inc   { 8 => G_CC_OP_INCB,   16 => G_CC_OP_INCW,   32 => G_CC_OP_INCL },
        Dec   { 8 => G_CC_OP_DECB,   16 => G_CC_OP_DECW,   32 => G_CC_OP_DECL },
        Shl   { 8 => G_CC_OP_SHLB,   16 => G_CC_OP_SHLW,   32 => G_CC_OP_SHLL },
        Shr   { 8 => G_CC_OP_SHRB,   16 => G_CC_OP_SHRW,   32 => G_CC_OP_SHRL },
        Rol   { 8 => G_CC_OP_ROLB,   16 => G_CC_OP_ROLW,   32 => G_CC_OP_ROLL },
        Ror   { 8 => G_CC_OP_RORB,   16 => G_CC_OP_RORW,   32 => G_CC_OP_RORL },
        Umul  { 8 => G_CC_OP_UMULB,  16 => G_CC_OP_UMULW,  32 => G_CC_OP_UMULL },
        Smul  { 8 => G_CC_OP_SMULB,  16 => G_CC_OP_SMULW,  32 => G_CC_OP_SMULL },
    )
}

/// Arch-dispatched decoder.
fn cc_op_info(arch: CcArch, cc_op: u64) -> Option<CcOpInfo> {
    match arch {
        CcArch::Amd64 => amd64_cc_op_info(cc_op),
        CcArch::X86 => x86_cc_op_info(cc_op),
    }
}

/// Compute concrete EFLAGS for a non-Copy category. The caller MUST handle
/// `OpCategory::Copy` before calling this — Copy isn't a real flag-producing
/// operation (it just stores already-computed flags in `cc_dep1`).
fn compute_flags_from_category(
    category: OpCategory,
    nbits: u32,
    cc_dep1: u64,
    cc_dep2: u64,
    cc_ndep: u64,
) -> Flags {
    match category {
        OpCategory::Copy => unreachable!("Copy must be handled by caller"),
        OpCategory::Add => calc_flags_add(nbits, cc_dep1, cc_dep2),
        OpCategory::Sub => calc_flags_sub(nbits, cc_dep1, cc_dep2),
        OpCategory::Adc => calc_flags_adc(nbits, cc_dep1, cc_dep2, cc_ndep),
        OpCategory::Sbb => calc_flags_sbb(nbits, cc_dep1, cc_dep2, cc_ndep),
        OpCategory::Logic => calc_flags_logic(nbits, cc_dep1),
        OpCategory::Inc => calc_flags_inc(nbits, cc_dep1, cc_ndep),
        OpCategory::Dec => calc_flags_dec(nbits, cc_dep1, cc_ndep),
        OpCategory::Shl => calc_flags_shl(nbits, cc_dep1, cc_dep2),
        OpCategory::Shr => calc_flags_shr(nbits, cc_dep1, cc_dep2),
        OpCategory::Rol => calc_flags_rol(nbits, cc_dep1, cc_ndep),
        OpCategory::Ror => calc_flags_ror(nbits, cc_dep1, cc_ndep),
        OpCategory::Umul => calc_flags_umul(nbits, cc_dep1, cc_dep2),
        OpCategory::Smul => calc_flags_smul(nbits, cc_dep1, cc_dep2),
    }
}

/// Evaluate a condition based on flags.
///
/// Returns `None` for unknown condition codes, mirroring
/// `x86_symbolic::eval_sym_condition`; callers fall through to Python rather
/// than reporting a definite (and wrong) "condition false".
fn eval_condition(cond: u64, flags: &Flags) -> Option<u64> {
    use cond_type::*;

    let inv = (cond & 1) as u8;

    let result = match cond {
        COND_O | COND_NO => inv ^ flags.of,
        COND_B | COND_NB => inv ^ flags.cf,
        COND_Z | COND_NZ => inv ^ flags.zf,
        COND_BE | COND_NBE => inv ^ (flags.cf | flags.zf),
        COND_S | COND_NS => inv ^ flags.sf,
        COND_P | COND_NP => inv ^ flags.pf,
        COND_L | COND_NL => inv ^ (flags.sf ^ flags.of),
        COND_LE | COND_NLE => inv ^ ((flags.sf ^ flags.of) | flags.zf),
        // SILENT(cat-a): unrecognized cond -> defer to the Python ccall
        // implementation, same as the symbolic path.
        _ => return None,
    };

    Some((result & 1) as u64)
}

/// Evaluate condition from COPY operation (flags in cc_dep1)
fn eval_condition_from_copy(cond: u64, cc_dep1: u64) -> Option<u64> {
    let cf = ((cc_dep1 >> flag_shift::G_CC_SHIFT_C) & 1) as u8;
    let pf = ((cc_dep1 >> flag_shift::G_CC_SHIFT_P) & 1) as u8;
    let zf = ((cc_dep1 >> flag_shift::G_CC_SHIFT_Z) & 1) as u8;
    let sf = ((cc_dep1 >> flag_shift::G_CC_SHIFT_S) & 1) as u8;
    let of = ((cc_dep1 >> flag_shift::G_CC_SHIFT_O) & 1) as u8;

    let flags = Flags { cf, pf, zf, sf, of };
    eval_condition(cond, &flags)
}

/// Calculate condition for AMD64 architecture.
///
/// Arguments:
/// - cond: Condition type (CondO, CondZ, etc.)
/// - cc_op: Operation type (G_CC_OP_SUBB, G_CC_OP_LOGICL, etc.)
/// - cc_dep1: First operand (or result for some ops)
/// - cc_dep2: Second operand (or shifted bits for shift ops)
/// - cc_ndep: Non-dependent value (preserved flags for INC/DEC/ROL/ROR)
pub fn amd64g_calculate_condition(
    cond: u64,
    cc_op: u64,
    cc_dep1: u64,
    cc_dep2: u64,
    cc_ndep: u64,
) -> Option<u64> {
    calculate_condition(CcArch::Amd64, cond, cc_op, cc_dep1, cc_dep2, cc_ndep)
}

/// Calculate condition for X86 architecture.
pub fn x86g_calculate_condition(
    cond: u64,
    cc_op: u64,
    cc_dep1: u64,
    cc_dep2: u64,
    cc_ndep: u64,
) -> Option<u64> {
    calculate_condition(CcArch::X86, cond, cc_op, cc_dep1, cc_dep2, cc_ndep)
}

fn calculate_condition(
    arch: CcArch,
    cond: u64,
    cc_op: u64,
    cc_dep1: u64,
    cc_dep2: u64,
    cc_ndep: u64,
) -> Option<u64> {
    let info = cc_op_info(arch, cc_op)?;
    if info.category == OpCategory::Copy {
        return eval_condition_from_copy(cond, cc_dep1);
    }
    let flags = compute_flags_from_category(info.category, info.nbits, cc_dep1, cc_dep2, cc_ndep);
    eval_condition(cond, &flags)
}

/// Calculate the carry flag (CF) for the given cc_op.
fn calculate_eflags_c_amd64(cc_op: u64, cc_dep1: u64, cc_dep2: u64, cc_ndep: u64) -> Option<u64> {
    calculate_eflags_c(CcArch::Amd64, cc_op, cc_dep1, cc_dep2, cc_ndep)
}

fn calculate_eflags_c_x86(cc_op: u64, cc_dep1: u64, cc_dep2: u64, cc_ndep: u64) -> Option<u64> {
    calculate_eflags_c(CcArch::X86, cc_op, cc_dep1, cc_dep2, cc_ndep)
}

fn calculate_eflags_c(
    arch: CcArch,
    cc_op: u64,
    cc_dep1: u64,
    cc_dep2: u64,
    cc_ndep: u64,
) -> Option<u64> {
    let info = cc_op_info(arch, cc_op)?;
    if info.category == OpCategory::Copy {
        return Some((cc_dep1 >> flag_shift::G_CC_SHIFT_C) & 1);
    }
    let flags = compute_flags_from_category(info.category, info.nbits, cc_dep1, cc_dep2, cc_ndep);
    Some(flags.cf as u64)
}

/// Pack flags into the standard EFLAGS format.
fn pack_eflags(flags: &Flags) -> u64 {
    ((flags.of as u64) << flag_shift::G_CC_SHIFT_O)
        | ((flags.sf as u64) << flag_shift::G_CC_SHIFT_S)
        | ((flags.zf as u64) << flag_shift::G_CC_SHIFT_Z)
        | ((flags.pf as u64) << flag_shift::G_CC_SHIFT_P)
        | ((flags.cf as u64) << flag_shift::G_CC_SHIFT_C)
}

/// Calculate all eflags for AMD64.
fn calculate_eflags_all_amd64(cc_op: u64, cc_dep1: u64, cc_dep2: u64, cc_ndep: u64) -> Option<u64> {
    calculate_eflags_all(CcArch::Amd64, cc_op, cc_dep1, cc_dep2, cc_ndep)
}

/// Calculate all eflags for X86.
fn calculate_eflags_all_x86(cc_op: u64, cc_dep1: u64, cc_dep2: u64, cc_ndep: u64) -> Option<u64> {
    calculate_eflags_all(CcArch::X86, cc_op, cc_dep1, cc_dep2, cc_ndep)
}

fn calculate_eflags_all(
    arch: CcArch,
    cc_op: u64,
    cc_dep1: u64,
    cc_dep2: u64,
    cc_ndep: u64,
) -> Option<u64> {
    let info = cc_op_info(arch, cc_op)?;
    if info.category == OpCategory::Copy {
        // For COPY, cc_dep1 already contains the flags
        return Some(
            cc_dep1
                & (flag_mask::G_CC_MASK_O
                    | flag_mask::G_CC_MASK_S
                    | flag_mask::G_CC_MASK_Z
                    | flag_mask::G_CC_MASK_P
                    | flag_mask::G_CC_MASK_C
                    | flag_mask::G_CC_MASK_A),
        );
    }
    let flags = compute_flags_from_category(info.category, info.nbits, cc_dep1, cc_dep2, cc_ndep);
    Some(pack_eflags(&flags))
}

/// Handle a CCall expression (ctx-less convenience wrapper).
///
/// Delegates to [`handle_ccall_with_ctx`] with `ctx = None`. Production code
/// (the interpreter) calls `handle_ccall_with_ctx` directly with a live
/// `SymContext`; this wrapper has no production callers and is retained only
/// for tests that exercise the concrete path (see angr-36vvn.4), so it is
/// compiled under `cfg(test)` and scoped to `pub(super)` — the only consumer
/// is the `ccall_tests` child module.
///
/// Returns Some(result) if the call was handled, None if not supported.
#[cfg(test)]
pub(super) fn handle_ccall(name: &str, args: &[RustBV], ret_bits: u32) -> Option<RustBV> {
    handle_ccall_with_ctx(name, args, ret_bits, None)
}

/// Handle a CCall with optional symbolic context for symbolic condition codes.
pub fn handle_ccall_with_ctx(
    name: &str,
    args: &[RustBV],
    ret_bits: u32,
    ctx: Option<&crate::symbolic::SymContext>,
) -> Option<RustBV> {
    // Check for x86g_calculate_condition or amd64g_calculate_condition
    if name == "amd64g_calculate_condition" || name == "x86g_calculate_condition" {
        // Args: cond, cc_op, cc_dep1, cc_dep2, cc_ndep
        if args.len() < 5 {
            return None;
        }

        // Try concrete path first
        if let (Some(cond), Some(cc_op), Some(cc_dep1), Some(cc_dep2), Some(cc_ndep)) = (
            args[0].as_u64(),
            args[1].as_u64(),
            args[2].as_u64(),
            args[3].as_u64(),
            args[4].as_u64(),
        ) {
            let result = if name == "amd64g_calculate_condition" {
                amd64g_calculate_condition(cond, cc_op, cc_dep1, cc_dep2, cc_ndep)?
            } else {
                x86g_calculate_condition(cond, cc_op, cc_dep1, cc_dep2, cc_ndep)?
            };
            return Some(RustBV::concrete(result as u128, ret_bits));
        }

        // Symbolic path: build SymFlags for the cc_op category, then evaluate
        // the condition. Covers SUB/ADD/LOGIC/INC/DEC and all standard
        // condition codes (O/B/Z/BE/S/P/L/LE plus inverses).
        if let (Some(cond), Some(cc_op), Some(sym_ctx)) = (args[0].as_u64(), args[1].as_u64(), ctx)
        {
            let arch = CcArch::from_ccall_name(name);
            if let Some(info) = cc_op_info(arch, cc_op) {
                let flags = if info.category == OpCategory::Copy {
                    sym_flags_from_copy(&args[2], sym_ctx)
                } else {
                    match sym_flags_for_category(
                        info.category,
                        info.nbits,
                        &args[2],
                        &args[3],
                        &args[4],
                        sym_ctx,
                    ) {
                        Ok(f) => f,
                        // Unreachable: the surrounding `if` handles Copy first.
                        Err(SymFlagsError::CopyHandledByCaller) => return None,
                    }
                };
                if let Some(bit) = eval_sym_condition(cond, &flags, sym_ctx) {
                    return Some(bit.zero_extend(ret_bits, sym_ctx));
                }
            }
        }

        return None;
    }

    // Check for eflags_c / rflags_c CCall.
    // Handle both "eflags" and "rflags" naming variants.
    if name == "amd64g_calculate_eflags_c"
        || name == "amd64g_calculate_rflags_c"
        || name == "x86g_calculate_eflags_c"
        || name == "x86g_calculate_rflags_c"
    {
        // Args: cc_op, cc_dep1, cc_dep2, cc_ndep
        if args.len() < 4 {
            return None;
        }

        // Try concrete path first
        if let (Some(cc_op), Some(cc_dep1), Some(cc_dep2), Some(cc_ndep)) = (
            args[0].as_u64(),
            args[1].as_u64(),
            args[2].as_u64(),
            args[3].as_u64(),
        ) {
            let is_amd64 = name.starts_with("amd64g");
            let result = if is_amd64 {
                calculate_eflags_c_amd64(cc_op, cc_dep1, cc_dep2, cc_ndep)?
            } else {
                calculate_eflags_c_x86(cc_op, cc_dep1, cc_dep2, cc_ndep)?
            };
            return Some(RustBV::concrete(result as u128, ret_bits));
        }

        // Symbolic path for carry flag
        if let (Some(cc_op), Some(sym_ctx)) = (args[0].as_u64(), ctx) {
            let arch = CcArch::from_ccall_name(name);
            if let Some(info) = cc_op_info(arch, cc_op) {
                let nb = info.nbits;
                let cf = match info.category {
                    OpCategory::Copy => {
                        // CF = bit G_CC_SHIFT_C of dep1. `extract(shift, shift)`
                        // is exactly `(dep1 >> shift) & 1` as a 1-bit BV — reuse
                        // the shared helper (as the Inc/Dec arm below does)
                        // instead of hand-building the lshr/and/extract chain,
                        // so the flag-extraction idiom has one source of truth.
                        Some(sym_extract_flag(
                            &args[1],
                            flag_shift::G_CC_SHIFT_C,
                            sym_ctx,
                        ))
                    }
                    OpCategory::Sub => {
                        let d1 = extract_to_nbits(&args[1], nb, sym_ctx);
                        let d2 = extract_to_nbits(&args[2], nb, sym_ctx);
                        Some(d1.ult(&d2, sym_ctx))
                    }
                    OpCategory::Add => {
                        let d1 = extract_to_nbits(&args[1], nb, sym_ctx);
                        let d2 = extract_to_nbits(&args[2], nb, sym_ctx);
                        let result = d1.add(&d2, sym_ctx);
                        Some(result.ult(&d1, sym_ctx))
                    }
                    OpCategory::Logic => Some(RustBV::concrete(0, 1)),
                    // INC/DEC do not modify CF; VEX preserves it in cc_ndep
                    // (args[3]). CF = (cc_ndep >> SHIFT_C) & 1 — matches the
                    // concrete calc_flags_inc/calc_flags_dec path. Without this,
                    // a symbolic cc_ndep (e.g. blank_state's uninitialized
                    // flags before any flag-setting op) forced the whole ccall
                    // to fall back to a fresh unconstrained symbolic carry,
                    // poisoning later branch guards (angr-g6dg).
                    OpCategory::Inc | OpCategory::Dec => Some(sym_extract_flag(
                        &args[3],
                        flag_shift::G_CC_SHIFT_C,
                        sym_ctx,
                    )),
                    // Everything else: reuse the shared `SymFlags` builder and
                    // take its CF rather than re-deriving each formula here.
                    // ADC/SBB carry is oldC-dependent (angr-9ke6b.88); the
                    // shift/rotate/multiply carries read cc_dep2 / cc_ndep in
                    // ways that are equally easy to get subtly wrong
                    // (angr-9ke6b.219). Listed exhaustively so a new
                    // `OpCategory` fails to compile instead of falling back to
                    // Python (30x wall-clock, measured on angr-9ke6b.88).
                    OpCategory::Adc
                    | OpCategory::Sbb
                    | OpCategory::Shl
                    | OpCategory::Shr
                    | OpCategory::Rol
                    | OpCategory::Ror
                    | OpCategory::Umul
                    | OpCategory::Smul => sym_flags_for_category(
                        info.category,
                        nb,
                        &args[1],
                        &args[2],
                        &args[3],
                        sym_ctx,
                    )
                    .ok()
                    .map(|f| f.cf),
                };
                if let Some(c) = cf {
                    return Some(c.zero_extend(ret_bits, sym_ctx));
                }
            }
        }

        return None;
    }

    // Check for eflags_all / rflags_all CCall.
    // VEX emits both "amd64g_calculate_rflags_all" and "amd64g_calculate_eflags_all"
    // depending on the context. We need to handle both names.
    if name == "amd64g_calculate_eflags_all"
        || name == "amd64g_calculate_rflags_all"
        || name == "x86g_calculate_eflags_all"
        || name == "x86g_calculate_rflags_all"
    {
        // Args: cc_op, cc_dep1, cc_dep2, cc_ndep
        if args.len() < 4 {
            return None;
        }

        // Try concrete path first
        if let (Some(cc_op), Some(cc_dep1), Some(cc_dep2), Some(cc_ndep)) = (
            args[0].as_u64(),
            args[1].as_u64(),
            args[2].as_u64(),
            args[3].as_u64(),
        ) {
            let is_amd64 = name.starts_with("amd64g");
            let result = if is_amd64 {
                calculate_eflags_all_amd64(cc_op, cc_dep1, cc_dep2, cc_ndep)?
            } else {
                calculate_eflags_all_x86(cc_op, cc_dep1, cc_dep2, cc_ndep)?
            };
            return Some(RustBV::concrete(result as u128, ret_bits));
        }

        // Symbolic path: handle various cc_ops with symbolic deps
        if let (Some(cc_op), Some(sym_ctx)) = (args[0].as_u64(), ctx) {
            if cc_op == 0 {
                // CC_OP_COPY: result = cc_dep1 & flags_mask
                // Mirror the concrete calculate_eflags_all COPY path: mask in
                // O|S|Z|P|C|A via the same named constants (hand-deriving the
                // literal twice is how 0xD5 drifted, dropping the O bit — see
                // angr-36vvn.1).
                let flags_mask: u128 = (flag_mask::G_CC_MASK_O
                    | flag_mask::G_CC_MASK_S
                    | flag_mask::G_CC_MASK_Z
                    | flag_mask::G_CC_MASK_P
                    | flag_mask::G_CC_MASK_C
                    | flag_mask::G_CC_MASK_A) as u128;
                let mask = RustBV::concrete(flags_mask, args[1].width());
                let result = args[1].and(&mask, sym_ctx);
                if result.width() < ret_bits {
                    return Some(result.zero_extend(ret_bits, sym_ctx));
                } else if result.width() > ret_bits {
                    return Some(result.extract(ret_bits - 1, 0, sym_ctx));
                }
                return Some(result);
            }

            // Symbolic SUB/ADD/LOGIC eflags computation
            let arch = CcArch::from_ccall_name(name);
            if let Some(info) = cc_op_info(arch, cc_op) {
                let nb = info.nbits;
                let result = match info.category {
                    OpCategory::Sub => Some(symbolic_eflags_sub(
                        nb, &args[1], &args[2], sym_ctx, ret_bits,
                    )),
                    OpCategory::Add => Some(symbolic_eflags_add(
                        nb, &args[1], &args[2], sym_ctx, ret_bits,
                    )),
                    OpCategory::Logic => {
                        Some(symbolic_eflags_logic(nb, &args[1], sym_ctx, ret_bits))
                    }
                    _ => None,
                };
                if result.is_some() {
                    return result;
                }
            }
        }

        // Unsupported symbolic cc_ops: return None (falls through to fallback)
        return None;
    }

    // x86g_use_seg_selector: linearize a segmented address.
    // Args: [ldt, gdt, seg_selector, virtual_addr]
    // Returns 64-bit value: lower 32 bits = linear address, upper 32 bits = error flag.
    // Fast path: when the relevant descriptor table (LDT or GDT, chosen by tiBit) is concretely
    // zero, treat as flat addressing — this is the common Linux-glibc-TLS case
    // (e.g. mov %gs:0x14, %eax for stack canary reads).
    if name == "x86g_use_seg_selector" {
        if args.len() < 4 {
            return None;
        }
        if let (Some(ldt_val), Some(gdt_val), Some(ss_val), Some(va_val)) = (
            args[0].as_u64(),
            args[1].as_u64(),
            args[2].as_u64(),
            args[3].as_u64(),
        ) {
            // Bad selector: high bits set above 16. Match Python's bad() return.
            if ss_val & !0xFFFFu64 != 0 {
                return Some(RustBV::concrete(1u128 << 32, ret_bits));
            }
            // Pick the descriptor table (tiBit = bit 2 of seg_selector).
            let ti_bit = (ss_val >> 2) & 1;
            let table_empty = if ti_bit == 0 {
                gdt_val == 0
            } else {
                ldt_val == 0
            };
            if table_empty {
                // The flat-addressing sum is 32-bit in Python's
                // `x86g_use_seg_selector` (`(seg_selector << 16) + virtual_addr`
                // over 32-bit BVs, then `.zero_extend(32)`), so it wraps mod
                // 2^32. Masking here is load-bearing, not cosmetic: without it a
                // carry out of bit 31 lands on bit 32, which this ccall's ABI
                // reserves for the error flag. Reachable with any negative
                // displacement off a segment register (`mov %gs:-0x4, %eax` →
                // va = 0xFFFFFFFC), which would otherwise report a bogus
                // bad-selector error. See test_use_seg_selector_gdt_empty_wraps_mod_2_32.
                let linear = ((ss_val & 0xFFFF) << 16).wrapping_add(va_val & 0xFFFFFFFF);
                return Some(RustBV::concrete((linear & 0xFFFF_FFFF) as u128, ret_bits));
            }
        }
        return None;
    }

    // ARM: armg_calculate_condition
    // Args: cond_n_op, cc_dep1, cc_dep2, cc_ndep (cc_dep3 in Python naming)
    if name == "armg_calculate_condition" {
        if args.len() < 4 {
            return None;
        }

        // Concrete path
        if let (Some(cond_n_op), Some(dep1), Some(dep2), Some(ndep)) = (
            args[0].as_u64(),
            args[1].as_u64(),
            args[2].as_u64(),
            args[3].as_u64(),
        ) {
            let result = armg_calculate_condition(cond_n_op, dep1, dep2, ndep)?;
            return Some(RustBV::concrete(result as u128, ret_bits));
        }

        // Symbolic path: concrete cond_n_op with (possibly) symbolic deps.
        // Routes through arm_sym_calculate_condition which covers all 8
        // cc_ops (COPY/ADD/SUB/ADC/SBB/LOGIC/MUL/MULL) and the standard
        // condition codes (EQ/HS/MI/VS/HI/GE/GT plus inverses, AL, NV).
        if let (Some(cond_n_op), Some(sym_ctx)) = (args[0].as_u64(), ctx) {
            let cond = (cond_n_op >> 4) & 0xF;
            let cc_op = cond_n_op & 0xF;
            if let Some(bit) =
                arm_sym_calculate_condition(cond, cc_op, &args[1], &args[2], &args[3], sym_ctx)
            {
                return Some(bit.zero_extend(ret_bits, sym_ctx));
            }
        }

        return None;
    }

    // ARM: armg_calculate_flags_nzcv
    if name == "armg_calculate_flags_nzcv" {
        if args.len() < 4 {
            return None;
        }

        if let (Some(cc_op), Some(dep1), Some(dep2), Some(ndep)) = (
            args[0].as_u64(),
            args[1].as_u64(),
            args[2].as_u64(),
            args[3].as_u64(),
        ) {
            let result = armg_calculate_flags_nzcv(cc_op, dep1, dep2, ndep)?;
            return Some(RustBV::concrete(result as u128, ret_bits));
        }

        return None;
    }

    // ARM: individual flag calculations
    if name == "armg_calculate_flag_n"
        || name == "armg_calculate_flag_z"
        || name == "armg_calculate_flag_c"
        || name == "armg_calculate_flag_v"
    {
        if args.len() < 4 {
            return None;
        }

        if let (Some(cc_op), Some(dep1), Some(dep2), Some(ndep)) = (
            args[0].as_u64(),
            args[1].as_u64(),
            args[2].as_u64(),
            args[3].as_u64(),
        ) {
            let result = match name {
                "armg_calculate_flag_n" => armg_calc_flag_n(cc_op, dep1, dep2, ndep)?,
                "armg_calculate_flag_z" => armg_calc_flag_z(cc_op, dep1, dep2, ndep)?,
                "armg_calculate_flag_c" => armg_calc_flag_c(cc_op, dep1, dep2, ndep)?,
                "armg_calculate_flag_v" => armg_calc_flag_v(cc_op, dep1, dep2, ndep)?,
                _ => return None,
            };
            return Some(RustBV::concrete(result as u128, ret_bits));
        }

        return None;
    }

    // AArch64: arm64g_calculate_condition
    // Args: cond_n_op, cc_dep1, cc_dep2, cc_dep3
    if name == "arm64g_calculate_condition" {
        if args.len() < 4 {
            return None;
        }

        // Concrete path
        if let (Some(cond_n_op), Some(d1), Some(d2), Some(d3)) = (
            args[0].as_u64(),
            args[1].as_u64(),
            args[2].as_u64(),
            args[3].as_u64(),
        ) {
            let result = arm64g_calculate_condition(cond_n_op, d1, d2, d3)?;
            return Some(RustBV::concrete(result as u128, ret_bits));
        }

        // Symbolic path: concrete cond_n_op with (possibly) symbolic deps.
        if let (Some(cond_n_op), Some(sym_ctx)) = (args[0].as_u64(), ctx) {
            let cond = (cond_n_op >> 4) & 0xF;
            let cc_op = cond_n_op & 0xF;
            if let Some(bit) =
                arm64_sym_calculate_condition(cond, cc_op, &args[1], &args[2], &args[3], sym_ctx)
            {
                return Some(bit.zero_extend(ret_bits, sym_ctx));
            }
        }

        return None;
    }

    // AArch64: arm64g_calculate_flags_nzcv
    if name == "arm64g_calculate_flags_nzcv" {
        if args.len() < 4 {
            return None;
        }
        if let (Some(cc_op), Some(d1), Some(d2), Some(d3)) = (
            args[0].as_u64(),
            args[1].as_u64(),
            args[2].as_u64(),
            args[3].as_u64(),
        ) {
            let result = arm64g_calculate_flags_nzcv(cc_op, d1, d2, d3)?;
            return Some(RustBV::concrete(result as u128, ret_bits));
        }
        return None;
    }

    // AArch64: individual flag calculations
    if name == "arm64g_calculate_flag_n"
        || name == "arm64g_calculate_flag_z"
        || name == "arm64g_calculate_flag_c"
        || name == "arm64g_calculate_flag_v"
    {
        if args.len() < 4 {
            return None;
        }

        // Concrete path
        if let (Some(cc_op), Some(d1), Some(d2), Some(d3)) = (
            args[0].as_u64(),
            args[1].as_u64(),
            args[2].as_u64(),
            args[3].as_u64(),
        ) {
            let result = match name {
                "arm64g_calculate_flag_n" => arm64g_calc_flag_n(cc_op, d1, d2, d3)?,
                "arm64g_calculate_flag_z" => arm64g_calc_flag_z(cc_op, d1, d2, d3)?,
                "arm64g_calculate_flag_c" => arm64g_calc_flag_c(cc_op, d1, d2, d3)?,
                "arm64g_calculate_flag_v" => arm64g_calc_flag_v(cc_op, d1, d2, d3)?,
                _ => return None,
            };
            return Some(RustBV::concrete(result as u128, ret_bits));
        }

        // Symbolic path: concrete cc_op with (possibly) symbolic deps.
        if let (Some(cc_op), Some(sym_ctx)) = (args[0].as_u64(), ctx) {
            let bit = match name {
                "arm64g_calculate_flag_n" => {
                    arm64_sym_flag_n(cc_op, &args[1], &args[2], &args[3], sym_ctx)
                }
                "arm64g_calculate_flag_z" => {
                    arm64_sym_flag_z(cc_op, &args[1], &args[2], &args[3], sym_ctx)
                }
                "arm64g_calculate_flag_c" => {
                    arm64_sym_flag_c(cc_op, &args[1], &args[2], &args[3], sym_ctx)
                }
                "arm64g_calculate_flag_v" => {
                    arm64_sym_flag_v(cc_op, &args[1], &args[2], &args[3], sym_ctx)
                }
                _ => None,
            };
            if let Some(bit) = bit {
                return Some(bit.zero_extend(ret_bits, sym_ctx));
            }
        }

        return None;
    }

    // Not a supported CCall
    None
}

#[cfg(test)]
#[path = "../ccall_tests.rs"]
mod ccall_tests;
