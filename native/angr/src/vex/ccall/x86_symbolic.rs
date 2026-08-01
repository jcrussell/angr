//! x86/amd64 symbolic flag computation: `SymFlags` plus the `sym_*` /
//! `symbolic_*` builders that construct 1-bit symbolic flag BVs.
use super::*;

/// Symbolic equivalent of `Flags`: each field is a 1-bit BV.
pub(super) struct SymFlags {
    pub(super) cf: RustBV,
    pub(super) pf: RustBV,
    pub(super) zf: RustBV,
    pub(super) sf: RustBV,
    pub(super) of: RustBV,
}

// ============================================================
// Symbolic flag computation helpers
// ============================================================

/// Extract the low `nbits` from a value that may be wider (e.g. 64-bit AMD64 args for 8/16/32-bit ops).
pub(super) fn extract_to_nbits(val: &RustBV, nbits: u32, ctx: &SymContext) -> RustBV {
    if val.width() == nbits {
        val.clone()
    } else if val.width() > nbits {
        val.extract(nbits - 1, 0, ctx)
    } else {
        val.zero_extend(nbits, ctx)
    }
}

/// Compute parity flag symbolically: PF = 1 if even number of 1-bits in low byte.
/// PF = NOT(b0 XOR b1 XOR b2 XOR b3 XOR b4 XOR b5 XOR b6 XOR b7)
pub(super) fn symbolic_parity(result: &RustBV, ctx: &SymContext) -> RustBV {
    let b0 = result.extract(0, 0, ctx);
    let b1 = result.extract(1, 1, ctx);
    let b2 = result.extract(2, 2, ctx);
    let b3 = result.extract(3, 3, ctx);
    let b4 = result.extract(4, 4, ctx);
    let b5 = result.extract(5, 5, ctx);
    let b6 = result.extract(6, 6, ctx);
    let b7 = result.extract(7, 7, ctx);
    let xor_all = b0
        .xor(&b1, ctx)
        .xor(&b2, ctx)
        .xor(&b3, ctx)
        .xor(&b4, ctx)
        .xor(&b5, ctx)
        .xor(&b6, ctx)
        .xor(&b7, ctx);
    // PF=1 means even parity (even number of set bits), so NOT the XOR
    xor_all.not(ctx)
}

/// Pack individual 1-bit flags into EFLAGS format bitvector.
/// Bit positions: OF@11, SF@7, ZF@6, PF@2, CF@0
pub(super) fn symbolic_pack_eflags(
    of: &RustBV,
    sf: &RustBV,
    zf: &RustBV,
    pf: &RustBV,
    cf: &RustBV,
    ret_bits: u32,
    ctx: &SymContext,
) -> RustBV {
    let of_ext = of.zero_extend(ret_bits, ctx);
    let sf_ext = sf.zero_extend(ret_bits, ctx);
    let zf_ext = zf.zero_extend(ret_bits, ctx);
    let pf_ext = pf.zero_extend(ret_bits, ctx);
    let cf_ext = cf.zero_extend(ret_bits, ctx);

    let shift_11 = RustBV::concrete(11, ret_bits);
    let shift_7 = RustBV::concrete(7, ret_bits);
    let shift_6 = RustBV::concrete(6, ret_bits);
    let shift_2 = RustBV::concrete(2, ret_bits);

    of_ext
        .shl(&shift_11, ctx)
        .or(&sf_ext.shl(&shift_7, ctx), ctx)
        .or(&zf_ext.shl(&shift_6, ctx), ctx)
        .or(&pf_ext.shl(&shift_2, ctx), ctx)
        .or(&cf_ext, ctx)
}

/// SUB / CMP: symbolic flags from dep1 - dep2 at the given width.
pub(super) fn sym_flags_sub(
    nbits: u32,
    dep1: &RustBV,
    dep2: &RustBV,
    ctx: &SymContext,
) -> SymFlags {
    let d1 = extract_to_nbits(dep1, nbits, ctx);
    let d2 = extract_to_nbits(dep2, nbits, ctx);
    let result = d1.sub(&d2, ctx);
    let zero = RustBV::concrete(0, nbits);

    let zf = result.eq(&zero, ctx);
    let sf = result.extract(nbits - 1, nbits - 1, ctx);
    // CF = unsigned borrow (dep1 < dep2)
    let cf = d1.ult(&d2, ctx);
    // OF = ((dep1 ^ dep2) & (dep1 ^ result))[msb]
    let of = d1
        .xor(&d2, ctx)
        .and(&d1.xor(&result, ctx), ctx)
        .extract(nbits - 1, nbits - 1, ctx);
    let pf = symbolic_parity(&result, ctx);
    SymFlags { cf, pf, zf, sf, of }
}

/// ADD: symbolic flags from dep1 + dep2 at the given width.
pub(super) fn sym_flags_add(
    nbits: u32,
    dep1: &RustBV,
    dep2: &RustBV,
    ctx: &SymContext,
) -> SymFlags {
    let d1 = extract_to_nbits(dep1, nbits, ctx);
    let d2 = extract_to_nbits(dep2, nbits, ctx);
    let result = d1.add(&d2, ctx);
    let zero = RustBV::concrete(0, nbits);

    let zf = result.eq(&zero, ctx);
    let sf = result.extract(nbits - 1, nbits - 1, ctx);
    // CF = unsigned carry out (result < dep1)
    let cf = result.ult(&d1, ctx);
    // OF = (~(dep1 ^ dep2) & (dep1 ^ result))[msb]
    let of = d1
        .xor(&d2, ctx)
        .not(ctx)
        .and(&d1.xor(&result, ctx), ctx)
        .extract(nbits - 1, nbits - 1, ctx);
    let pf = symbolic_parity(&result, ctx);
    SymFlags { cf, pf, zf, sf, of }
}

/// ADC: symbolic flags for `dep1 + (dep2 ^ oldC) + oldC` at the given width.
///
/// Mirrors `calc_flags_adc` (the concrete path) exactly, including VEX's
/// `argR = dep2 ^ oldC` encoding and the oldC-dependent carry test
/// (`res <= argL` with an incoming carry, `res < argL` without). Before
/// angr-9ke6b.88 this category had no symbolic builder, so any ADC/SBB with a
/// symbolic operand fell out of `sym_flags_for_category` as `Unsupported`.
pub(super) fn sym_flags_adc(
    nbits: u32,
    dep1: &RustBV,
    dep2: &RustBV,
    ndep: &RustBV,
    ctx: &SymContext,
) -> SymFlags {
    let old_c = sym_extract_flag(ndep, flag_shift::G_CC_SHIFT_C, ctx);
    let old_c_n = old_c.zero_extend(nbits, ctx);
    let arg_l = extract_to_nbits(dep1, nbits, ctx);
    let arg_r = extract_to_nbits(dep2, nbits, ctx).xor(&old_c_n, ctx);
    let result = arg_l.add(&arg_r, ctx).add(&old_c_n, ctx);
    let zero = RustBV::concrete(0, nbits);

    let zf = result.eq(&zero, ctx);
    let sf = result.extract(nbits - 1, nbits - 1, ctx);
    // CF = oldC ? (result <= argL) : (result < argL)
    let cf = old_c.ite(&result.ule(&arg_l, ctx), &result.ult(&arg_l, ctx), ctx);
    // OF = (~(argL ^ argR) & (argL ^ result))[msb]
    let of = arg_l
        .xor(&arg_r, ctx)
        .not(ctx)
        .and(&arg_l.xor(&result, ctx), ctx)
        .extract(nbits - 1, nbits - 1, ctx);
    let pf = symbolic_parity(&result, ctx);
    SymFlags { cf, pf, zf, sf, of }
}

/// SBB: symbolic flags for `dep1 - (dep2 ^ oldC) - oldC` at the given width.
///
/// Mirrors `calc_flags_sbb`; see [`sym_flags_adc`] for the shared oldC
/// encoding. The borrow test compares the *operands* (`argL` vs `argR`), not
/// the result, which is why it cannot reuse `sym_flags_sub`.
pub(super) fn sym_flags_sbb(
    nbits: u32,
    dep1: &RustBV,
    dep2: &RustBV,
    ndep: &RustBV,
    ctx: &SymContext,
) -> SymFlags {
    let old_c = sym_extract_flag(ndep, flag_shift::G_CC_SHIFT_C, ctx);
    let old_c_n = old_c.zero_extend(nbits, ctx);
    let arg_l = extract_to_nbits(dep1, nbits, ctx);
    let arg_r = extract_to_nbits(dep2, nbits, ctx).xor(&old_c_n, ctx);
    let result = arg_l.sub(&arg_r, ctx).sub(&old_c_n, ctx);
    let zero = RustBV::concrete(0, nbits);

    let zf = result.eq(&zero, ctx);
    let sf = result.extract(nbits - 1, nbits - 1, ctx);
    // CF = oldC ? (argL <= argR) : (argL < argR)
    let cf = old_c.ite(&arg_l.ule(&arg_r, ctx), &arg_l.ult(&arg_r, ctx), ctx);
    // OF = ((argL ^ argR) & (argL ^ result))[msb]
    let of = arg_l
        .xor(&arg_r, ctx)
        .and(&arg_l.xor(&result, ctx), ctx)
        .extract(nbits - 1, nbits - 1, ctx);
    let pf = symbolic_parity(&result, ctx);
    SymFlags { cf, pf, zf, sf, of }
}

/// LOGIC (AND/OR/XOR/TEST): result is in dep1; CF=0, OF=0.
pub(super) fn sym_flags_logic(nbits: u32, dep1: &RustBV, ctx: &SymContext) -> SymFlags {
    let result = extract_to_nbits(dep1, nbits, ctx);
    let zero = RustBV::concrete(0, nbits);

    let zf = result.eq(&zero, ctx);
    let sf = result.extract(nbits - 1, nbits - 1, ctx);
    let cf = RustBV::concrete(0, 1);
    let of = RustBV::concrete(0, 1);
    let pf = symbolic_parity(&result, ctx);
    SymFlags { cf, pf, zf, sf, of }
}

/// INC: dep1 = result; CF preserved from ndep; OF = (result == sign_bit).
pub(super) fn sym_flags_inc(
    nbits: u32,
    dep1: &RustBV,
    ndep: &RustBV,
    ctx: &SymContext,
) -> SymFlags {
    let result = extract_to_nbits(dep1, nbits, ctx);
    let zero = RustBV::concrete(0, nbits);
    let sign_bit_val = RustBV::concrete(1u128 << (nbits - 1), nbits);

    let zf = result.eq(&zero, ctx);
    let sf = result.extract(nbits - 1, nbits - 1, ctx);
    // CF preserved from ndep (the saved EFLAGS). ndep width may exceed 1.
    let cf = sym_extract_flag(ndep, flag_shift::G_CC_SHIFT_C, ctx);
    // OF = (result == 0x80...0) — overflow on INC happens when 0x7F..F was incremented.
    let of = result.eq(&sign_bit_val, ctx);
    let pf = symbolic_parity(&result, ctx);
    SymFlags { cf, pf, zf, sf, of }
}

/// DEC: dep1 = result; CF preserved from ndep; OF = (result == sign_bit - 1).
pub(super) fn sym_flags_dec(
    nbits: u32,
    dep1: &RustBV,
    ndep: &RustBV,
    ctx: &SymContext,
) -> SymFlags {
    let result = extract_to_nbits(dep1, nbits, ctx);
    let zero = RustBV::concrete(0, nbits);
    // result == sign_bit - 1 == max signed (0x7F..F)
    let max_signed = RustBV::concrete((1u128 << (nbits - 1)) - 1, nbits);

    let zf = result.eq(&zero, ctx);
    let sf = result.extract(nbits - 1, nbits - 1, ctx);
    let cf = sym_extract_flag(ndep, flag_shift::G_CC_SHIFT_C, ctx);
    let of = result.eq(&max_signed, ctx);
    let pf = symbolic_parity(&result, ctx);
    SymFlags { cf, pf, zf, sf, of }
}

/// Extract a single flag bit at the given shift from a packed-EFLAGS BV.
pub(super) fn sym_extract_flag(packed: &RustBV, shift: u32, ctx: &SymContext) -> RustBV {
    packed.extract(shift, shift, ctx)
}

/// Why `sym_flags_for_category` may decline to build a `SymFlags`.
///
/// Distinguishing these lets callers fall back to the Python ccall path
/// deliberately (and lets future profiling code attribute the fallback to a
/// specific category) instead of swallowing an unannotated `None`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum SymFlagsError {
    /// `OpCategory::Copy` was passed. Copy's flags live directly in `dep1`;
    /// callers must invoke `sym_flags_from_copy` instead.
    CopyHandledByCaller,
    /// No symbolic builder yet for this category — rare in branch flags but
    /// hit by Adc/Sbb/Shl/Shr/Rol/Ror/Umul/Smul. Caller should fall back to
    /// the Python ccall implementation.
    Unsupported(OpCategory),
}

/// Dispatch table: build `SymFlags` for a non-Copy category.
///
/// Callers MUST handle `OpCategory::Copy` before invoking this — Copy is
/// reported as `Err(SymFlagsError::CopyHandledByCaller)` rather than handled
/// in-place because Copy's flags live in `dep1` and need a different extraction.
pub(super) fn sym_flags_for_category(
    category: OpCategory,
    nbits: u32,
    dep1: &RustBV,
    dep2: &RustBV,
    ndep: &RustBV,
    ctx: &SymContext,
) -> Result<SymFlags, SymFlagsError> {
    match category {
        OpCategory::Copy => Err(SymFlagsError::CopyHandledByCaller),
        OpCategory::Sub => Ok(sym_flags_sub(nbits, dep1, dep2, ctx)),
        OpCategory::Add => Ok(sym_flags_add(nbits, dep1, dep2, ctx)),
        OpCategory::Logic => Ok(sym_flags_logic(nbits, dep1, ctx)),
        OpCategory::Inc => Ok(sym_flags_inc(nbits, dep1, ndep, ctx)),
        OpCategory::Dec => Ok(sym_flags_dec(nbits, dep1, ndep, ctx)),
        OpCategory::Adc => Ok(sym_flags_adc(nbits, dep1, dep2, ndep, ctx)),
        OpCategory::Sbb => Ok(sym_flags_sbb(nbits, dep1, dep2, ndep, ctx)),
        OpCategory::Shl
        | OpCategory::Shr
        | OpCategory::Rol
        | OpCategory::Ror
        | OpCategory::Umul
        | OpCategory::Smul => Err(SymFlagsError::Unsupported(category)),
    }
}

/// Extract `SymFlags` from a Copy operation where `dep1` already packs them.
pub(super) fn sym_flags_from_copy(dep1: &RustBV, ctx: &SymContext) -> SymFlags {
    SymFlags {
        cf: sym_extract_flag(dep1, flag_shift::G_CC_SHIFT_C, ctx),
        pf: sym_extract_flag(dep1, flag_shift::G_CC_SHIFT_P, ctx),
        zf: sym_extract_flag(dep1, flag_shift::G_CC_SHIFT_Z, ctx),
        sf: sym_extract_flag(dep1, flag_shift::G_CC_SHIFT_S, ctx),
        of: sym_extract_flag(dep1, flag_shift::G_CC_SHIFT_O, ctx),
    }
}

/// Symbolic equivalent of `eval_condition`: return a 1-bit BV for condition `cond`.
///
/// Returns `None` for unknown condition codes; callers fall through to Python.
pub(super) fn eval_sym_condition(cond: u64, flags: &SymFlags, ctx: &SymContext) -> Option<RustBV> {
    use cond_type::*;
    let inv = (cond & 1) != 0;
    let bit = match cond & !1 {
        COND_O => flags.of.clone(),
        COND_B => flags.cf.clone(),
        COND_Z => flags.zf.clone(),
        COND_BE => flags.cf.or(&flags.zf, ctx),
        COND_S => flags.sf.clone(),
        COND_P => flags.pf.clone(),
        COND_L => flags.sf.xor(&flags.of, ctx),
        COND_LE => flags.sf.xor(&flags.of, ctx).or(&flags.zf, ctx),
        _ => return None,
    };
    Some(if inv { bit.not(ctx) } else { bit })
}

/// Pack symbolic flags into the standard EFLAGS layout at `ret_bits` width.
pub(super) fn sym_pack_eflags(flags: &SymFlags, ret_bits: u32, ctx: &SymContext) -> RustBV {
    symbolic_pack_eflags(
        &flags.of, &flags.sf, &flags.zf, &flags.pf, &flags.cf, ret_bits, ctx,
    )
}

/// Symbolic eflags computation for SUB/CMP: flags from dep1 - dep2 packed at ret_bits.
pub(super) fn symbolic_eflags_sub(
    nbits: u32,
    dep1: &RustBV,
    dep2: &RustBV,
    ctx: &SymContext,
    ret_bits: u32,
) -> RustBV {
    sym_pack_eflags(&sym_flags_sub(nbits, dep1, dep2, ctx), ret_bits, ctx)
}

/// Symbolic eflags computation for ADD: flags from dep1 + dep2 packed at ret_bits.
pub(super) fn symbolic_eflags_add(
    nbits: u32,
    dep1: &RustBV,
    dep2: &RustBV,
    ctx: &SymContext,
    ret_bits: u32,
) -> RustBV {
    sym_pack_eflags(&sym_flags_add(nbits, dep1, dep2, ctx), ret_bits, ctx)
}

/// Symbolic eflags computation for LOGIC: flags from result in dep1 packed at ret_bits.
pub(super) fn symbolic_eflags_logic(
    nbits: u32,
    dep1: &RustBV,
    ctx: &SymContext,
    ret_bits: u32,
) -> RustBV {
    sym_pack_eflags(&sym_flags_logic(nbits, dep1, ctx), ret_bits, ctx)
}

// ============================================================
// Concrete flag computation
// ============================================================
