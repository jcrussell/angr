//! ARM (32-bit) condition-code support: cc_op/cond constants and the
//! concrete + symbolic flag calculators.
use super::*;

// ============================================================
// ARM condition code support
// ============================================================

/// ARM CC_OP values (from VEX's libvex_guest_arm.h)
pub mod arm_cc_op {
    pub const ARMG_CC_OP_COPY: u64 = 0; // DEP1 = NZCV in 31:28
    pub const ARMG_CC_OP_ADD: u64 = 1; // DEP1 = argL, DEP2 = argR
    pub const ARMG_CC_OP_SUB: u64 = 2; // DEP1 = argL, DEP2 = argR
    pub const ARMG_CC_OP_ADC: u64 = 3; // DEP1 = argL, DEP2 = argR, NDEP = oldC
    pub const ARMG_CC_OP_SBB: u64 = 4; // DEP1 = argL, DEP2 = argR, NDEP = oldC
    pub const ARMG_CC_OP_LOGIC: u64 = 5; // DEP1 = result, DEP2 = shifter_carry_out, NDEP = oldV
    pub const ARMG_CC_OP_MUL: u64 = 6; // DEP1 = result, NDEP = oldC:oldV
    pub const ARMG_CC_OP_MULL: u64 = 7; // DEP1 = resLO32, DEP2 = resHI32, NDEP = oldC:oldV
}

/// ARM condition codes
// Full 0-15 encoding table. Production `armg_calculate_condition` matches on
// `cond & !1`, so it only names the even variants plus AL/NV; the odd
// inverse-condition variants (NE, LO, PL, VC, LS, LT, LE) are referenced only
// by the ccall test suite. Kept complete for the arch-BE campaign (angr-ig3o)
// and as a readable ABI reference; allow(dead_code) applies to non-test builds.
#[cfg_attr(not(test), allow(dead_code))]
pub mod arm_cond {
    pub const ARM_COND_EQ: u64 = 0; // Z=1
    pub const ARM_COND_NE: u64 = 1; // Z=0
    pub const ARM_COND_HS: u64 = 2; // C=1
    pub const ARM_COND_LO: u64 = 3; // C=0
    pub const ARM_COND_MI: u64 = 4; // N=1
    pub const ARM_COND_PL: u64 = 5; // N=0
    pub const ARM_COND_VS: u64 = 6; // V=1
    pub const ARM_COND_VC: u64 = 7; // V=0
    pub const ARM_COND_HI: u64 = 8; // C=1 && Z=0
    pub const ARM_COND_LS: u64 = 9; // C=0 || Z=1
    pub const ARM_COND_GE: u64 = 10; // N=V
    pub const ARM_COND_LT: u64 = 11; // N!=V
    pub const ARM_COND_GT: u64 = 12; // Z=0 && N=V
    pub const ARM_COND_LE: u64 = 13; // Z=1 || N!=V
    pub const ARM_COND_AL: u64 = 14; // always
    pub const ARM_COND_NV: u64 = 15; // never
}

/// ARM NZCV flag bit positions
pub(super) mod arm_flag_shift {
    pub const SHIFT_N: u32 = 31;
    pub const SHIFT_Z: u32 = 30;
    pub const SHIFT_C: u32 = 29;
    pub const SHIFT_V: u32 = 28;
}

// ============================================================
// ARM symbolic flag computation
// ============================================================

/// Extract `dep` to a 32-bit BV (ARM operands are 32-bit; VEX may hand us wider temps).
pub(super) fn arm_extract32(val: &RustBV, ctx: &SymContext) -> RustBV {
    extract_to_nbits(val, 32, ctx)
}

/// Symbolic ARM N flag (bit 31 of result). Returns 1-bit BV.
pub(super) fn arm_sym_flag_n(
    cc_op: u64,
    dep1: &RustBV,
    dep2: &RustBV,
    ndep: &RustBV,
    ctx: &SymContext,
) -> Option<RustBV> {
    use arm_cc_op::*;
    let d1 = arm_extract32(dep1, ctx);
    let d2 = arm_extract32(dep2, ctx);
    let nd = arm_extract32(ndep, ctx);
    let res = match cc_op {
        ARMG_CC_OP_COPY => {
            return Some(d1.extract(arm_flag_shift::SHIFT_N, arm_flag_shift::SHIFT_N, ctx));
        }
        ARMG_CC_OP_ADD => d1.add(&d2, ctx),
        ARMG_CC_OP_SUB => d1.sub(&d2, ctx),
        ARMG_CC_OP_ADC => d1.add(&d2, ctx).add(&nd, ctx),
        ARMG_CC_OP_SBB => {
            let one = RustBV::concrete(1, 32);
            d1.sub(&d2, ctx).sub(&nd.xor(&one, ctx), ctx)
        }
        ARMG_CC_OP_LOGIC | ARMG_CC_OP_MUL => d1,
        ARMG_CC_OP_MULL => d2,
        _ => return None,
    };
    Some(res.extract(31, 31, ctx))
}

/// Symbolic ARM Z flag (result == 0). Returns 1-bit BV.
pub(super) fn arm_sym_flag_z(
    cc_op: u64,
    dep1: &RustBV,
    dep2: &RustBV,
    ndep: &RustBV,
    ctx: &SymContext,
) -> Option<RustBV> {
    use arm_cc_op::*;
    let d1 = arm_extract32(dep1, ctx);
    let d2 = arm_extract32(dep2, ctx);
    let nd = arm_extract32(ndep, ctx);
    let zero = RustBV::concrete(0, 32);
    let res = match cc_op {
        ARMG_CC_OP_COPY => {
            return Some(d1.extract(arm_flag_shift::SHIFT_Z, arm_flag_shift::SHIFT_Z, ctx));
        }
        ARMG_CC_OP_ADD => d1.add(&d2, ctx),
        ARMG_CC_OP_SUB => d1.sub(&d2, ctx),
        ARMG_CC_OP_ADC => d1.add(&d2, ctx).add(&nd, ctx),
        ARMG_CC_OP_SBB => {
            let one = RustBV::concrete(1, 32);
            d1.sub(&d2, ctx).sub(&nd.xor(&one, ctx), ctx)
        }
        ARMG_CC_OP_LOGIC | ARMG_CC_OP_MUL => d1,
        // MULL: Z = (resLO | resHI) == 0
        ARMG_CC_OP_MULL => d1.or(&d2, ctx),
        _ => return None,
    };
    Some(res.eq(&zero, ctx))
}

/// Symbolic ARM C flag. ARM C for SUB is the *complement* of x86 borrow:
/// C=1 iff dep1 >= dep2 (no borrow). Returns 1-bit BV.
pub(super) fn arm_sym_flag_c(
    cc_op: u64,
    dep1: &RustBV,
    dep2: &RustBV,
    ndep: &RustBV,
    ctx: &SymContext,
) -> Option<RustBV> {
    use arm_cc_op::*;
    let d1 = arm_extract32(dep1, ctx);
    let d2 = arm_extract32(dep2, ctx);
    let nd = arm_extract32(ndep, ctx);
    match cc_op {
        ARMG_CC_OP_COPY => Some(d1.extract(arm_flag_shift::SHIFT_C, arm_flag_shift::SHIFT_C, ctx)),
        ARMG_CC_OP_ADD => {
            let res = d1.add(&d2, ctx);
            Some(res.ult(&d1, ctx))
        }
        ARMG_CC_OP_SUB => Some(d1.uge(&d2, ctx)),
        ARMG_CC_OP_ADC => {
            // C: if oldC then res<=dep1 else res<dep1
            let res = d1.add(&d2, ctx).add(&nd, ctx);
            let zero = RustBV::concrete(0, 32);
            let nd_nz = nd.eq(&zero, ctx).not(ctx);
            let when_old_c = res.ule(&d1, ctx);
            let when_no_old_c = res.ult(&d1, ctx);
            // ITE: nd_nz ? when_old_c : when_no_old_c == (nd_nz & when_old_c) | (~nd_nz & when_no_old_c)
            Some(
                nd_nz
                    .and(&when_old_c, ctx)
                    .or(&nd_nz.not(ctx).and(&when_no_old_c, ctx), ctx),
            )
        }
        ARMG_CC_OP_SBB => {
            let zero = RustBV::concrete(0, 32);
            let nd_nz = nd.eq(&zero, ctx).not(ctx);
            let when_old_c = d1.uge(&d2, ctx);
            let when_no_old_c = d1.ugt(&d2, ctx);
            Some(
                nd_nz
                    .and(&when_old_c, ctx)
                    .or(&nd_nz.not(ctx).and(&when_no_old_c, ctx), ctx),
            )
        }
        ARMG_CC_OP_LOGIC => {
            // shifter_carry_out lives in dep2[0]
            Some(d2.extract(0, 0, ctx))
        }
        ARMG_CC_OP_MUL | ARMG_CC_OP_MULL => {
            // (ndep >> 1) & 1
            Some(nd.extract(1, 1, ctx))
        }
        _ => None,
    }
}

/// Symbolic ARM V flag. Returns 1-bit BV.
pub(super) fn arm_sym_flag_v(
    cc_op: u64,
    dep1: &RustBV,
    dep2: &RustBV,
    ndep: &RustBV,
    ctx: &SymContext,
) -> Option<RustBV> {
    use arm_cc_op::*;
    let d1 = arm_extract32(dep1, ctx);
    let d2 = arm_extract32(dep2, ctx);
    let nd = arm_extract32(ndep, ctx);
    match cc_op {
        ARMG_CC_OP_COPY => Some(d1.extract(arm_flag_shift::SHIFT_V, arm_flag_shift::SHIFT_V, ctx)),
        ARMG_CC_OP_ADD => {
            // V = ((res ^ d1) & (res ^ d2))[31]
            let res = d1.add(&d2, ctx);
            Some(
                res.xor(&d1, ctx)
                    .and(&res.xor(&d2, ctx), ctx)
                    .extract(31, 31, ctx),
            )
        }
        ARMG_CC_OP_SUB => {
            // V = ((d1 ^ d2) & (d1 ^ res))[31]
            let res = d1.sub(&d2, ctx);
            Some(
                d1.xor(&d2, ctx)
                    .and(&d1.xor(&res, ctx), ctx)
                    .extract(31, 31, ctx),
            )
        }
        ARMG_CC_OP_ADC => {
            let res = d1.add(&d2, ctx).add(&nd, ctx);
            Some(
                res.xor(&d1, ctx)
                    .and(&res.xor(&d2, ctx), ctx)
                    .extract(31, 31, ctx),
            )
        }
        ARMG_CC_OP_SBB => {
            let one = RustBV::concrete(1, 32);
            let res = d1.sub(&d2, ctx).sub(&nd.xor(&one, ctx), ctx);
            Some(
                d1.xor(&d2, ctx)
                    .and(&d1.xor(&res, ctx), ctx)
                    .extract(31, 31, ctx),
            )
        }
        ARMG_CC_OP_LOGIC => Some(nd.extract(0, 0, ctx)), // old V
        ARMG_CC_OP_MUL | ARMG_CC_OP_MULL => Some(nd.extract(0, 0, ctx)),
        _ => None,
    }
}

/// Symbolic ARM condition evaluation.
///
/// Compute the four NZCV flags for the cc_op, then combine per cond.
/// Returns a 1-bit BV (or `None` if `cond` is unknown or `cc_op` is unsupported).
pub(super) fn arm_sym_calculate_condition(
    cond: u64,
    cc_op: u64,
    dep1: &RustBV,
    dep2: &RustBV,
    ndep: &RustBV,
    ctx: &SymContext,
) -> Option<RustBV> {
    use arm_cond::*;
    let inv = (cond & 1) != 0;

    if cond == ARM_COND_AL {
        return Some(RustBV::concrete(1, 1));
    }
    if cond == ARM_COND_NV {
        return Some(RustBV::concrete(0, 1));
    }

    let flag = match cond & !1 {
        ARM_COND_EQ => arm_sym_flag_z(cc_op, dep1, dep2, ndep, ctx)?,
        ARM_COND_HS => arm_sym_flag_c(cc_op, dep1, dep2, ndep, ctx)?,
        ARM_COND_MI => arm_sym_flag_n(cc_op, dep1, dep2, ndep, ctx)?,
        ARM_COND_VS => arm_sym_flag_v(cc_op, dep1, dep2, ndep, ctx)?,
        ARM_COND_HI => {
            let cf = arm_sym_flag_c(cc_op, dep1, dep2, ndep, ctx)?;
            let zf = arm_sym_flag_z(cc_op, dep1, dep2, ndep, ctx)?;
            cf.and(&zf.not(ctx), ctx)
        }
        ARM_COND_GE => {
            let nf = arm_sym_flag_n(cc_op, dep1, dep2, ndep, ctx)?;
            let vf = arm_sym_flag_v(cc_op, dep1, dep2, ndep, ctx)?;
            nf.xor(&vf, ctx).not(ctx)
        }
        ARM_COND_GT => {
            let nf = arm_sym_flag_n(cc_op, dep1, dep2, ndep, ctx)?;
            let vf = arm_sym_flag_v(cc_op, dep1, dep2, ndep, ctx)?;
            let zf = arm_sym_flag_z(cc_op, dep1, dep2, ndep, ctx)?;
            zf.or(&nf.xor(&vf, ctx), ctx).not(ctx)
        }
        _ => return None,
    };
    Some(if inv { flag.not(ctx) } else { flag })
}

/// Compute ARM N (negative) flag for a given cc_op.
pub(super) fn armg_calc_flag_n(cc_op: u64, dep1: u64, dep2: u64, ndep: u64) -> Option<u64> {
    use arm_cc_op::*;
    match cc_op {
        ARMG_CC_OP_COPY => Some((dep1 >> arm_flag_shift::SHIFT_N) & 1),
        ARMG_CC_OP_ADD => Some((dep1.wrapping_add(dep2)) >> 31),
        ARMG_CC_OP_SUB => Some((dep1.wrapping_sub(dep2)) >> 31),
        ARMG_CC_OP_ADC => Some((dep1.wrapping_add(dep2).wrapping_add(ndep)) >> 31),
        ARMG_CC_OP_SBB => Some((dep1.wrapping_sub(dep2).wrapping_sub(ndep ^ 1)) >> 31),
        ARMG_CC_OP_LOGIC | ARMG_CC_OP_MUL => Some(dep1 >> 31),
        ARMG_CC_OP_MULL => Some(dep2 >> 31),
        _ => None,
    }
}

/// Compute ARM Z (zero) flag for a given cc_op.
pub(super) fn armg_calc_flag_z(cc_op: u64, dep1: u64, dep2: u64, ndep: u64) -> Option<u64> {
    use arm_cc_op::*;
    let dep1 = dep1 & 0xFFFFFFFF;
    let dep2 = dep2 & 0xFFFFFFFF;
    let ndep = ndep & 0xFFFFFFFF;
    match cc_op {
        ARMG_CC_OP_COPY => Some((dep1 >> arm_flag_shift::SHIFT_Z) & 1),
        ARMG_CC_OP_ADD => {
            let res = dep1.wrapping_add(dep2) & 0xFFFFFFFF;
            Some(if res == 0 { 1 } else { 0 })
        }
        ARMG_CC_OP_SUB => {
            let res = dep1.wrapping_sub(dep2) & 0xFFFFFFFF;
            Some(if res == 0 { 1 } else { 0 })
        }
        ARMG_CC_OP_ADC => {
            let res = dep1.wrapping_add(dep2).wrapping_add(ndep) & 0xFFFFFFFF;
            Some(if res == 0 { 1 } else { 0 })
        }
        ARMG_CC_OP_SBB => {
            let res = dep1.wrapping_sub(dep2).wrapping_sub(ndep ^ 1) & 0xFFFFFFFF;
            Some(if res == 0 { 1 } else { 0 })
        }
        ARMG_CC_OP_LOGIC | ARMG_CC_OP_MUL => Some(if (dep1 & 0xFFFFFFFF) == 0 { 1 } else { 0 }),
        ARMG_CC_OP_MULL => Some(if (dep1 | dep2) & 0xFFFFFFFF == 0 {
            1
        } else {
            0
        }),
        _ => None,
    }
}

/// Compute ARM C (carry) flag for a given cc_op.
pub(super) fn armg_calc_flag_c(cc_op: u64, dep1: u64, dep2: u64, ndep: u64) -> Option<u64> {
    use arm_cc_op::*;
    let dep1 = dep1 & 0xFFFFFFFF;
    let dep2 = dep2 & 0xFFFFFFFF;
    let ndep = ndep & 0xFFFFFFFF;
    match cc_op {
        ARMG_CC_OP_COPY => Some((dep1 >> arm_flag_shift::SHIFT_C) & 1),
        ARMG_CC_OP_ADD => {
            let res = dep1.wrapping_add(dep2) & 0xFFFFFFFF;
            Some(if res < dep1 { 1 } else { 0 })
        }
        ARMG_CC_OP_SUB => Some(if dep1 >= dep2 { 1 } else { 0 }),
        ARMG_CC_OP_ADC => {
            let res = dep1.wrapping_add(dep2).wrapping_add(ndep) & 0xFFFFFFFF;
            if ndep != 0 {
                Some(if res <= dep1 { 1 } else { 0 })
            } else {
                Some(if res < dep1 { 1 } else { 0 })
            }
        }
        ARMG_CC_OP_SBB => {
            if ndep != 0 {
                Some(if dep1 >= dep2 { 1 } else { 0 })
            } else {
                Some(if dep1 > dep2 { 1 } else { 0 })
            }
        }
        ARMG_CC_OP_LOGIC => Some(dep2 & 1), // shifter_carry_out
        ARMG_CC_OP_MUL | ARMG_CC_OP_MULL => Some((ndep >> 1) & 1),
        _ => None,
    }
}

/// Compute ARM V (overflow) flag for a given cc_op.
pub(super) fn armg_calc_flag_v(cc_op: u64, dep1: u64, dep2: u64, ndep: u64) -> Option<u64> {
    use arm_cc_op::*;
    let dep1 = dep1 & 0xFFFFFFFF;
    let dep2 = dep2 & 0xFFFFFFFF;
    let ndep = ndep & 0xFFFFFFFF;
    match cc_op {
        ARMG_CC_OP_COPY => Some((dep1 >> arm_flag_shift::SHIFT_V) & 1),
        ARMG_CC_OP_ADD => {
            let res = dep1.wrapping_add(dep2) & 0xFFFFFFFF;
            Some(((res ^ dep1) & (res ^ dep2)) >> 31)
        }
        ARMG_CC_OP_SUB => {
            let res = dep1.wrapping_sub(dep2) & 0xFFFFFFFF;
            Some(((dep1 ^ dep2) & (dep1 ^ res)) >> 31)
        }
        ARMG_CC_OP_ADC => {
            let res = dep1.wrapping_add(dep2).wrapping_add(ndep) & 0xFFFFFFFF;
            Some(((res ^ dep1) & (res ^ dep2)) >> 31)
        }
        ARMG_CC_OP_SBB => {
            let res = dep1.wrapping_sub(dep2).wrapping_sub(ndep ^ 1) & 0xFFFFFFFF;
            Some(((dep1 ^ dep2) & (dep1 ^ res)) >> 31)
        }
        ARMG_CC_OP_LOGIC => Some(ndep & 1), // old V flag
        ARMG_CC_OP_MUL | ARMG_CC_OP_MULL => Some(ndep & 1),
        _ => None,
    }
}

/// Concrete ARM condition evaluation.
///
/// `cond_n_op` encodes: cond in bits \[7:4\], cc_op in bits \[3:0\].
/// Returns 1 if condition is true, 0 if false, None if unsupported.
pub fn armg_calculate_condition(cond_n_op: u64, dep1: u64, dep2: u64, ndep: u64) -> Option<u64> {
    let cond = (cond_n_op >> 4) & 0xF;
    let cc_op = cond_n_op & 0xF;
    let inv = cond & 1;

    use arm_cond::*;

    let flag = match cond & !1 {
        ARM_COND_EQ => {
            // EQ/NE: test Z flag
            armg_calc_flag_z(cc_op, dep1, dep2, ndep)?
        }
        ARM_COND_HS => {
            // HS/LO: test C flag
            armg_calc_flag_c(cc_op, dep1, dep2, ndep)?
        }
        ARM_COND_MI => {
            // MI/PL: test N flag
            armg_calc_flag_n(cc_op, dep1, dep2, ndep)?
        }
        ARM_COND_VS => {
            // VS/VC: test V flag
            armg_calc_flag_v(cc_op, dep1, dep2, ndep)?
        }
        ARM_COND_HI => {
            // HI/LS: C=1 && Z=0 / C=0 || Z=1
            let cf = armg_calc_flag_c(cc_op, dep1, dep2, ndep)?;
            let zf = armg_calc_flag_z(cc_op, dep1, dep2, ndep)?;
            cf & (!zf & 1)
        }
        ARM_COND_GE => {
            // GE/LT: N==V / N!=V
            let nf = armg_calc_flag_n(cc_op, dep1, dep2, ndep)?;
            let vf = armg_calc_flag_v(cc_op, dep1, dep2, ndep)?;
            1 & !(nf ^ vf)
        }
        ARM_COND_GT => {
            // GT/LE: Z=0 && N==V / Z=1 || N!=V
            let nf = armg_calc_flag_n(cc_op, dep1, dep2, ndep)?;
            let vf = armg_calc_flag_v(cc_op, dep1, dep2, ndep)?;
            let zf = armg_calc_flag_z(cc_op, dep1, dep2, ndep)?;
            1 & !(zf | (nf ^ vf))
        }
        ARM_COND_AL => return Some(1),
        _ => return None,
    };

    Some(inv ^ (flag & 1))
}

/// Compute all ARM NZCV flags and pack into bits \[31:28\].
pub fn armg_calculate_flags_nzcv(cc_op: u64, dep1: u64, dep2: u64, ndep: u64) -> Option<u64> {
    let n = armg_calc_flag_n(cc_op, dep1, dep2, ndep)?;
    let z = armg_calc_flag_z(cc_op, dep1, dep2, ndep)?;
    let c = armg_calc_flag_c(cc_op, dep1, dep2, ndep)?;
    let v = armg_calc_flag_v(cc_op, dep1, dep2, ndep)?;
    Some(
        ((n & 1) << arm_flag_shift::SHIFT_N)
            | ((z & 1) << arm_flag_shift::SHIFT_Z)
            | ((c & 1) << arm_flag_shift::SHIFT_C)
            | ((v & 1) << arm_flag_shift::SHIFT_V),
    )
}
