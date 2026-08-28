//! x86/amd64 symbolic flag computation: `SymFlags` plus the `sym_*` /
//! `symbolic_*` builders that construct 1-bit symbolic flag BVs.
use super::*;

/// Symbolic equivalent of `Flags`: each field is a 1-bit BV.
pub(super) struct SymFlags {
    pub(super) cf: RustBV,
    pub(super) pf: RustBV,
    pub(super) af: RustBV,
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

/// Symbolic twin of `calc_af`: AF = bit `G_CC_SHIFT_A` of
/// `res ^ arg_l ^ arg_r`, the carry out of bit 3.
///
/// All three operands must already be `nbits` wide (`nbits >= 8`, so bit 4
/// always exists).
pub(super) fn sym_calc_af(
    res: &RustBV,
    arg_l: &RustBV,
    arg_r: &RustBV,
    ctx: &SymContext,
) -> RustBV {
    res.xor(arg_l, ctx).xor(arg_r, ctx).extract(
        flag_shift::G_CC_SHIFT_A,
        flag_shift::G_CC_SHIFT_A,
        ctx,
    )
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
    // AF = (result ^ dep1 ^ dep2)[4]
    let af = sym_calc_af(&result, &d1, &d2, ctx);
    SymFlags {
        cf,
        pf,
        af,
        zf,
        sf,
        of,
    }
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
    // AF = (result ^ dep1 ^ dep2)[4]
    let af = sym_calc_af(&result, &d1, &d2, ctx);
    SymFlags {
        cf,
        pf,
        af,
        zf,
        sf,
        of,
    }
}

/// ADC: symbolic flags for `dep1 + (dep2 ^ oldC) + oldC` at the given width.
///
/// Mirrors `calc_flags_adc` (the concrete path) exactly, including VEX's
/// `argR = dep2 ^ oldC` encoding and the oldC-dependent carry test
/// (`res <= argL` with an incoming carry, `res < argL` without). Before
/// angr-9ke6b.88 this category had no symbolic builder, so `sym_flags_for_category`
/// declined it and any ADC/SBB with a symbolic operand fell back to Python.
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
    // AF = (result ^ argL ^ argR)[4]
    let af = sym_calc_af(&result, &arg_l, &arg_r, ctx);
    SymFlags {
        cf,
        pf,
        af,
        zf,
        sf,
        of,
    }
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
    // AF = (result ^ argL ^ argR)[4]
    let af = sym_calc_af(&result, &arg_l, &arg_r, ctx);
    SymFlags {
        cf,
        pf,
        af,
        zf,
        sf,
        of,
    }
}

/// LOGIC (AND/OR/XOR/TEST): result is in dep1; CF=0, OF=0.
pub(super) fn sym_flags_logic(nbits: u32, dep1: &RustBV, ctx: &SymContext) -> SymFlags {
    let result = extract_to_nbits(dep1, nbits, ctx);
    let zero = RustBV::concrete(0, nbits);

    let zf = result.eq(&zero, ctx);
    let sf = result.extract(nbits - 1, nbits - 1, ctx);
    let cf = RustBV::concrete(0, 1);
    let of = RustBV::concrete(0, 1);
    let af = RustBV::concrete(0, 1);
    let pf = symbolic_parity(&result, ctx);
    SymFlags {
        cf,
        pf,
        af,
        zf,
        sf,
        of,
    }
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
    // AF = (result ^ (result - 1) ^ 1)[4]
    let one_n = RustBV::concrete(1, nbits);
    let af = sym_calc_af(&result, &result.sub(&one_n, ctx), &one_n, ctx);
    SymFlags {
        cf,
        pf,
        af,
        zf,
        sf,
        of,
    }
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
    // AF = (result ^ (result + 1) ^ 1)[4]
    let one_n = RustBV::concrete(1, nbits);
    let af = sym_calc_af(&result, &result.add(&one_n, ctx), &one_n, ctx);
    SymFlags {
        cf,
        pf,
        af,
        zf,
        sf,
        of,
    }
}

/// SHL: `dep1` is the post-shift result, `dep2` the pre-shift value rotated so
/// its MSB is the last bit shifted out.
///
/// Mirrors `calc_flags_shl`. VEX only records enough state for a shift-by-1
/// OF (`CF ^ SF`), which is what the concrete path computes too.
pub(super) fn sym_flags_shl(
    nbits: u32,
    dep1: &RustBV,
    dep2: &RustBV,
    ctx: &SymContext,
) -> SymFlags {
    let remaining = extract_to_nbits(dep1, nbits, ctx);
    let shifted = extract_to_nbits(dep2, nbits, ctx);
    let zero = RustBV::concrete(0, nbits);

    // CF = last bit shifted out = MSB of the shifted-out value.
    let cf = shifted.extract(nbits - 1, nbits - 1, ctx);
    let zf = remaining.eq(&zero, ctx);
    let sf = remaining.extract(nbits - 1, nbits - 1, ctx);
    let of = cf.xor(&sf, ctx);
    let pf = symbolic_parity(&remaining, ctx);
    // AF is architecturally undefined after a shift; VEX reports 0.
    let af = RustBV::concrete(0, 1);
    SymFlags {
        cf,
        pf,
        af,
        zf,
        sf,
        of,
    }
}

/// SHR / SAR: `dep1` is the post-shift result, `dep2` the pre-shift value
/// rotated so its LSB is the last bit shifted out.
///
/// Mirrors `calc_flags_shr`; OF is the XOR of the two values' MSBs.
pub(super) fn sym_flags_shr(
    nbits: u32,
    dep1: &RustBV,
    dep2: &RustBV,
    ctx: &SymContext,
) -> SymFlags {
    let remaining = extract_to_nbits(dep1, nbits, ctx);
    let shifted = extract_to_nbits(dep2, nbits, ctx);
    let zero = RustBV::concrete(0, nbits);

    // CF = last bit shifted out = LSB of the shifted-out value.
    let cf = shifted.extract(0, 0, ctx);
    let zf = remaining.eq(&zero, ctx);
    let sf = remaining.extract(nbits - 1, nbits - 1, ctx);
    let of = shifted.extract(nbits - 1, nbits - 1, ctx).xor(&sf, ctx);
    let pf = symbolic_parity(&remaining, ctx);
    // AF is architecturally undefined after a shift; VEX reports 0.
    let af = RustBV::concrete(0, 1);
    SymFlags {
        cf,
        pf,
        af,
        zf,
        sf,
        of,
    }
}

/// ROL: `dep1` is the rotate result; PF/ZF/SF are *preserved* from `ndep`.
///
/// Mirrors `calc_flags_rol`. Rotates on x86 touch only CF and OF, so the other
/// three flags come out of the saved EFLAGS in `cc_ndep` rather than from the
/// result — the same preservation `sym_flags_inc`/`sym_flags_dec` do for CF.
pub(super) fn sym_flags_rol(
    nbits: u32,
    dep1: &RustBV,
    ndep: &RustBV,
    ctx: &SymContext,
) -> SymFlags {
    let res = extract_to_nbits(dep1, nbits, ctx);

    // CF = LSB of the result (the bit rotated around from the top).
    let cf = res.extract(0, 0, ctx);
    let pf = sym_extract_flag(ndep, flag_shift::G_CC_SHIFT_P, ctx);
    let af = sym_extract_flag(ndep, flag_shift::G_CC_SHIFT_A, ctx);
    let zf = sym_extract_flag(ndep, flag_shift::G_CC_SHIFT_Z, ctx);
    let sf = sym_extract_flag(ndep, flag_shift::G_CC_SHIFT_S, ctx);
    // OF = MSB ^ LSB of the result.
    let of = res.extract(nbits - 1, nbits - 1, ctx).xor(&cf, ctx);
    SymFlags {
        cf,
        pf,
        af,
        zf,
        sf,
        of,
    }
}

/// ROR: `dep1` is the rotate result; PF/ZF/SF are *preserved* from `ndep`.
///
/// Mirrors `calc_flags_ror`; see [`sym_flags_rol`] for the preservation rule.
pub(super) fn sym_flags_ror(
    nbits: u32,
    dep1: &RustBV,
    ndep: &RustBV,
    ctx: &SymContext,
) -> SymFlags {
    let res = extract_to_nbits(dep1, nbits, ctx);

    // CF = MSB of the result (the bit rotated around from the bottom).
    let cf = res.extract(nbits - 1, nbits - 1, ctx);
    let pf = sym_extract_flag(ndep, flag_shift::G_CC_SHIFT_P, ctx);
    let af = sym_extract_flag(ndep, flag_shift::G_CC_SHIFT_A, ctx);
    let zf = sym_extract_flag(ndep, flag_shift::G_CC_SHIFT_Z, ctx);
    let sf = sym_extract_flag(ndep, flag_shift::G_CC_SHIFT_S, ctx);
    // OF = XOR of the two top bits of the result.
    let of = cf.xor(&res.extract(nbits - 2, nbits - 2, ctx), ctx);
    SymFlags {
        cf,
        pf,
        af,
        zf,
        sf,
        of,
    }
}

/// UMUL: unsigned `dep1 * dep2`; CF = OF = "the product does not fit in nbits".
///
/// Mirrors `calc_flags_umul`. Both operands are zero-extended to `2 * nbits`
/// so the full product is exact, then split — that is the symbolic equivalent
/// of the concrete path's `u128` widening for the 64-bit case.
pub(super) fn sym_flags_umul(
    nbits: u32,
    dep1: &RustBV,
    dep2: &RustBV,
    ctx: &SymContext,
) -> SymFlags {
    let wide = nbits * 2;
    let a = extract_to_nbits(dep1, nbits, ctx).zero_extend(wide, ctx);
    let b = extract_to_nbits(dep2, nbits, ctx).zero_extend(wide, ctx);
    let product = a.mul(&b, ctx);
    let lo = product.extract(nbits - 1, 0, ctx);
    let hi = product.extract(wide - 1, nbits, ctx);
    let zero = RustBV::concrete(0, nbits);

    // CF = OF = high half non-zero (result truncated).
    let cf = hi.ne(&zero, ctx);
    let of = cf.clone();
    // ZF/SF/PF are architecturally undefined after MUL; VEX computes them
    // anyway, so mirror that rather than inventing a different answer. AF is
    // undefined too and VEX reports 0.
    let zf = lo.eq(&zero, ctx);
    let sf = lo.extract(nbits - 1, nbits - 1, ctx);
    let pf = symbolic_parity(&lo, ctx);
    let af = RustBV::concrete(0, 1);
    SymFlags {
        cf,
        pf,
        af,
        zf,
        sf,
        of,
    }
}

/// SMUL: signed `dep1 * dep2`; CF = OF = "the high half is not the sign
/// extension of the low half".
///
/// Mirrors `calc_flags_smul`; see [`sym_flags_umul`] for the widening scheme.
pub(super) fn sym_flags_smul(
    nbits: u32,
    dep1: &RustBV,
    dep2: &RustBV,
    ctx: &SymContext,
) -> SymFlags {
    let wide = nbits * 2;
    let a = extract_to_nbits(dep1, nbits, ctx).sign_extend(wide, ctx);
    let b = extract_to_nbits(dep2, nbits, ctx).sign_extend(wide, ctx);
    let product = a.mul(&b, ctx);
    let lo = product.extract(nbits - 1, 0, ctx);
    let hi = product.extract(wide - 1, nbits, ctx);
    let zero = RustBV::concrete(0, nbits);

    let sf = lo.extract(nbits - 1, nbits - 1, ctx);
    // Sign extension of `lo`: all-ones when its MSB is set, all-zeros otherwise.
    let lo_sign_ext = sf.sign_extend(nbits, ctx);
    let cf = hi.ne(&lo_sign_ext, ctx);
    let of = cf.clone();
    let zf = lo.eq(&zero, ctx);
    let pf = symbolic_parity(&lo, ctx);
    // AF is undefined after a multiply; VEX reports 0.
    let af = RustBV::concrete(0, 1);
    SymFlags {
        cf,
        pf,
        af,
        zf,
        sf,
        of,
    }
}

/// Extract a single flag bit at the given shift from a packed-EFLAGS BV.
pub(super) fn sym_extract_flag(packed: &RustBV, shift: u32, ctx: &SymContext) -> RustBV {
    packed.extract(shift, shift, ctx)
}

/// Why `sym_flags_for_category` may decline to build a `SymFlags`.
///
/// Since angr-9ke6b.219 every non-Copy category has a symbolic builder, so
/// `Copy` is the only reason left. The error type stays because callers still
/// need to distinguish "declined" from "built", and a future category added to
/// [`OpCategory`] would land here rather than silently degrading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum SymFlagsError {
    /// `OpCategory::Copy` was passed. Copy's flags live directly in `dep1`;
    /// callers must invoke `sym_flags_from_copy` instead.
    CopyHandledByCaller,
}

/// Dispatch table: build `SymFlags` for a non-Copy category.
///
/// Callers MUST handle `OpCategory::Copy` before invoking this — Copy is
/// reported as `Err(SymFlagsError::CopyHandledByCaller)` rather than handled
/// in-place because Copy's flags live in `dep1` and need a different extraction.
///
/// The match is deliberately exhaustive (no `_` arm): a new `OpCategory` must
/// be given a symbolic builder, not silently routed to the Python ccall path —
/// angr-9ke6b.88 measured that fallback at 30x wall-clock on a real bench.
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
        OpCategory::Shl => Ok(sym_flags_shl(nbits, dep1, dep2, ctx)),
        OpCategory::Shr => Ok(sym_flags_shr(nbits, dep1, dep2, ctx)),
        OpCategory::Rol => Ok(sym_flags_rol(nbits, dep1, ndep, ctx)),
        OpCategory::Ror => Ok(sym_flags_ror(nbits, dep1, ndep, ctx)),
        OpCategory::Umul => Ok(sym_flags_umul(nbits, dep1, dep2, ctx)),
        OpCategory::Smul => Ok(sym_flags_smul(nbits, dep1, dep2, ctx)),
    }
}

/// Extract `SymFlags` from a Copy operation where `dep1` already packs them.
pub(super) fn sym_flags_from_copy(dep1: &RustBV, ctx: &SymContext) -> SymFlags {
    SymFlags {
        cf: sym_extract_flag(dep1, flag_shift::G_CC_SHIFT_C, ctx),
        pf: sym_extract_flag(dep1, flag_shift::G_CC_SHIFT_P, ctx),
        af: sym_extract_flag(dep1, flag_shift::G_CC_SHIFT_A, ctx),
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
///
/// Bit positions come from the same `flag_shift` constants the concrete
/// `pack_eflags` uses rather than hand-written literals, per bd memory
/// `invariant-ccall-flag-mask-named-constants`: OF@11, SF@7, ZF@6, AF@4,
/// PF@2, CF@0. Both packers must cover every flag the Copy path passes
/// through, or the omitted bit is structurally always 0 for every computed
/// category (angr-5mnx3.59).
pub(super) fn sym_pack_eflags(flags: &SymFlags, ret_bits: u32, ctx: &SymContext) -> RustBV {
    let shifted = |bit: &RustBV, shift: u32| {
        let ext = bit.zero_extend(ret_bits, ctx);
        if shift == 0 {
            ext
        } else {
            ext.shl(&RustBV::concrete(u128::from(shift), ret_bits), ctx)
        }
    };

    shifted(&flags.of, flag_shift::G_CC_SHIFT_O)
        .or(&shifted(&flags.sf, flag_shift::G_CC_SHIFT_S), ctx)
        .or(&shifted(&flags.zf, flag_shift::G_CC_SHIFT_Z), ctx)
        .or(&shifted(&flags.af, flag_shift::G_CC_SHIFT_A), ctx)
        .or(&shifted(&flags.pf, flag_shift::G_CC_SHIFT_P), ctx)
        .or(&shifted(&flags.cf, flag_shift::G_CC_SHIFT_C), ctx)
}
