//! AArch64 (arm64g) condition-code support: a parallel section reusing
//! `arm_cond` / `arm_flag_shift` from the arm32 module.
use super::*;

// ============================================================
// AArch64 (arm64g) condition code support
// ============================================================
//
// Mirrors angr's Python `arm64g_calculate_*` (engines/vex/claripy/ccall.py).
// The condition encoding is identical to 32-bit ARM (ARM64Cond* == ARMCond*),
// so we reuse `arm_cond` and `arm_flag_shift`. The cc_op set differs: each
// arithmetic op has explicit 32- and 64-bit variants.
//
// NOTE: where the Python reference and VEX hardware semantics diverge (the
// ADC/SBC carry-in is keyed off `cc_dep2 != 0` in Python rather than the
// carry dep), we follow Python so the diff-fuzz against the Python engine
// matches.

/// AArch64 CC_OP values (from VEX's libvex_guest_arm64.h / angr ccall.py).
pub(super) mod arm64_cc_op {
    pub(crate) const ARM64G_CC_OP_COPY: u64 = 0;
    pub(crate) const ARM64G_CC_OP_ADD32: u64 = 1;
    pub(crate) const ARM64G_CC_OP_ADD64: u64 = 2;
    pub(crate) const ARM64G_CC_OP_SUB32: u64 = 3;
    pub(crate) const ARM64G_CC_OP_SUB64: u64 = 4;
    pub(crate) const ARM64G_CC_OP_ADC32: u64 = 5;
    pub(crate) const ARM64G_CC_OP_ADC64: u64 = 6;
    pub(crate) const ARM64G_CC_OP_SBC32: u64 = 7;
    pub(crate) const ARM64G_CC_OP_SBC64: u64 = 8;
    pub(crate) const ARM64G_CC_OP_LOGIC32: u64 = 9;
    pub(crate) const ARM64G_CC_OP_LOGIC64: u64 = 10;
}

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Arm64Op {
    Add,
    Sub,
    Adc,
    Sbc,
    Logic,
}

/// Decode an arm64 cc_op (other than COPY) into (operation, operand width).
pub(super) fn arm64_decode(cc_op: u64) -> Option<(Arm64Op, u32)> {
    use arm64_cc_op::*;
    Some(match cc_op {
        ARM64G_CC_OP_ADD32 => (Arm64Op::Add, 32),
        ARM64G_CC_OP_ADD64 => (Arm64Op::Add, 64),
        ARM64G_CC_OP_SUB32 => (Arm64Op::Sub, 32),
        ARM64G_CC_OP_SUB64 => (Arm64Op::Sub, 64),
        ARM64G_CC_OP_ADC32 => (Arm64Op::Adc, 32),
        ARM64G_CC_OP_ADC64 => (Arm64Op::Adc, 64),
        ARM64G_CC_OP_SBC32 => (Arm64Op::Sbc, 32),
        ARM64G_CC_OP_SBC64 => (Arm64Op::Sbc, 64),
        ARM64G_CC_OP_LOGIC32 => (Arm64Op::Logic, 32),
        ARM64G_CC_OP_LOGIC64 => (Arm64Op::Logic, 64),
        _ => return None,
    })
}

/// Concrete arm64 arithmetic result, masked to `nb` bits.
pub(super) fn arm64_res(op: Arm64Op, nb: u32, d1: u64, d2: u64, d3: u64) -> u64 {
    let mask = if nb == 64 { u64::MAX } else { (1u64 << nb) - 1 };
    let r = match op {
        Arm64Op::Add => d1.wrapping_add(d2),
        Arm64Op::Sub => d1.wrapping_sub(d2),
        Arm64Op::Adc => d1.wrapping_add(d2).wrapping_add(d3),
        Arm64Op::Sbc => d1.wrapping_sub(d2).wrapping_sub(d3 ^ 1),
        Arm64Op::Logic => d1,
    };
    r & mask
}

pub(super) fn arm64g_calc_flag_n(cc_op: u64, d1: u64, d2: u64, d3: u64) -> Option<u64> {
    if cc_op == arm64_cc_op::ARM64G_CC_OP_COPY {
        return Some((d1 >> arm_flag_shift::SHIFT_N) & 1);
    }
    let (op, nb) = arm64_decode(cc_op)?;
    let res = arm64_res(op, nb, d1, d2, d3);
    Some((res >> (nb - 1)) & 1)
}

pub(super) fn arm64g_calc_flag_z(cc_op: u64, d1: u64, d2: u64, d3: u64) -> Option<u64> {
    if cc_op == arm64_cc_op::ARM64G_CC_OP_COPY {
        return Some((d1 >> arm_flag_shift::SHIFT_Z) & 1);
    }
    let (op, nb) = arm64_decode(cc_op)?;
    let res = arm64_res(op, nb, d1, d2, d3);
    Some(u64::from(res == 0))
}

pub(super) fn arm64g_calc_flag_c(cc_op: u64, d1: u64, d2: u64, d3: u64) -> Option<u64> {
    if cc_op == arm64_cc_op::ARM64G_CC_OP_COPY {
        return Some((d1 >> arm_flag_shift::SHIFT_C) & 1);
    }
    let (op, nb) = arm64_decode(cc_op)?;
    let mask = if nb == 64 { u64::MAX } else { (1u64 << nb) - 1 };
    let a = d1 & mask;
    let b = d2 & mask;
    Some(match op {
        Arm64Op::Add => {
            let res = a.wrapping_add(b) & mask;
            u64::from(res < a)
        }
        Arm64Op::Sub => u64::from(a >= b),
        Arm64Op::Adc => {
            let res = a.wrapping_add(b).wrapping_add(d3) & mask;
            // Python keys the comparison off cc_dep2 (b), not the carry-in.
            if b != 0 {
                u64::from(res <= a)
            } else {
                u64::from(res < a)
            }
        }
        Arm64Op::Sbc => {
            if b != 0 {
                u64::from(a >= b)
            } else {
                u64::from(a > b)
            }
        }
        Arm64Op::Logic => 0,
    })
}

pub(super) fn arm64g_calc_flag_v(cc_op: u64, d1: u64, d2: u64, d3: u64) -> Option<u64> {
    if cc_op == arm64_cc_op::ARM64G_CC_OP_COPY {
        return Some((d1 >> arm_flag_shift::SHIFT_V) & 1);
    }
    let (op, nb) = arm64_decode(cc_op)?;
    let mask = if nb == 64 { u64::MAX } else { (1u64 << nb) - 1 };
    let top = nb - 1;
    let a = d1 & mask;
    let b = d2 & mask;
    Some(match op {
        Arm64Op::Add => {
            let res = a.wrapping_add(b) & mask;
            ((res ^ a) & (res ^ b)) >> top
        }
        Arm64Op::Sub => {
            let res = a.wrapping_sub(b) & mask;
            ((a ^ b) & (a ^ res)) >> top
        }
        Arm64Op::Adc => {
            let res = a.wrapping_add(b).wrapping_add(d3) & mask;
            ((res ^ a) & (res ^ b)) >> top
        }
        Arm64Op::Sbc => {
            let res = a.wrapping_sub(b).wrapping_sub(d3 ^ 1) & mask;
            ((a ^ b) & (a ^ res)) >> top
        }
        Arm64Op::Logic => 0,
    })
}

/// Concrete arm64 condition evaluation.
/// `cond_n_op` encodes cond in bits \[7:4\], cc_op in bits \[3:0\].
pub(super) fn arm64g_calculate_condition(cond_n_op: u64, d1: u64, d2: u64, d3: u64) -> Option<u64> {
    let cond = (cond_n_op >> 4) & 0xF;
    let cc_op = cond_n_op & 0xF;
    let inv = cond & 1;

    use arm_cond::*;
    let flag = match cond & !1 {
        ARM_COND_EQ => arm64g_calc_flag_z(cc_op, d1, d2, d3)?,
        ARM_COND_HS => arm64g_calc_flag_c(cc_op, d1, d2, d3)?,
        ARM_COND_MI => arm64g_calc_flag_n(cc_op, d1, d2, d3)?,
        ARM_COND_VS => arm64g_calc_flag_v(cc_op, d1, d2, d3)?,
        ARM_COND_HI => {
            let cf = arm64g_calc_flag_c(cc_op, d1, d2, d3)?;
            let zf = arm64g_calc_flag_z(cc_op, d1, d2, d3)?;
            cf & (!zf & 1)
        }
        ARM_COND_GE => {
            let nf = arm64g_calc_flag_n(cc_op, d1, d2, d3)?;
            let vf = arm64g_calc_flag_v(cc_op, d1, d2, d3)?;
            1 & !(nf ^ vf)
        }
        ARM_COND_GT => {
            let nf = arm64g_calc_flag_n(cc_op, d1, d2, d3)?;
            let vf = arm64g_calc_flag_v(cc_op, d1, d2, d3)?;
            let zf = arm64g_calc_flag_z(cc_op, d1, d2, d3)?;
            1 & !(zf | (nf ^ vf))
        }
        // AL (14) and NV (15) are both unconditional on AArch64.
        ARM_COND_AL => return Some(1),
        _ => return None,
    };
    Some(inv ^ (flag & 1))
}

/// Pack arm64 NZCV into bits \[31:28\].
pub(super) fn arm64g_calculate_flags_nzcv(cc_op: u64, d1: u64, d2: u64, d3: u64) -> Option<u64> {
    let n = arm64g_calc_flag_n(cc_op, d1, d2, d3)?;
    let z = arm64g_calc_flag_z(cc_op, d1, d2, d3)?;
    let c = arm64g_calc_flag_c(cc_op, d1, d2, d3)?;
    let v = arm64g_calc_flag_v(cc_op, d1, d2, d3)?;
    Some(
        ((n & 1) << arm_flag_shift::SHIFT_N)
            | ((z & 1) << arm_flag_shift::SHIFT_Z)
            | ((c & 1) << arm_flag_shift::SHIFT_C)
            | ((v & 1) << arm_flag_shift::SHIFT_V),
    )
}

/// Symbolic counterpart of `arm64g_calculate_flags_nzcv`: computes the four
/// NZCV bits with `arm64_sym_flag_*` and packs them via the shared
/// `sym_pack_nzcv` (arm32 module — same bit positions).
pub(super) fn arm64_sym_flags_nzcv(
    cc_op: u64,
    d1: &RustBV,
    d2: &RustBV,
    d3: &RustBV,
    ret_bits: u32,
    ctx: &SymContext,
) -> Option<RustBV> {
    let n = arm64_sym_flag_n(cc_op, d1, d2, d3, ctx)?;
    let z = arm64_sym_flag_z(cc_op, d1, d2, d3, ctx)?;
    let c = arm64_sym_flag_c(cc_op, d1, d2, d3, ctx)?;
    let v = arm64_sym_flag_v(cc_op, d1, d2, d3, ctx)?;
    sym_pack_nzcv(&n, &z, &c, &v, ret_bits, ctx)
}

/// Symbolic arm64 arithmetic result. `d1`/`d2`/`d3` must already be `nb`-wide.
pub(super) fn arm64_sym_res(
    op: Arm64Op,
    d1: &RustBV,
    d2: &RustBV,
    d3: &RustBV,
    ctx: &SymContext,
) -> RustBV {
    match op {
        Arm64Op::Add => d1.add(d2, ctx),
        Arm64Op::Sub => d1.sub(d2, ctx),
        Arm64Op::Adc => d1.add(d2, ctx).add(d3, ctx),
        Arm64Op::Sbc => {
            let one = RustBV::concrete(1, d1.width());
            d1.sub(d2, ctx).sub(&d3.xor(&one, ctx), ctx)
        }
        Arm64Op::Logic => d1.clone(),
    }
}

pub(super) fn arm64_sym_flag_n(
    cc_op: u64,
    dep1: &RustBV,
    dep2: &RustBV,
    dep3: &RustBV,
    ctx: &SymContext,
) -> Option<RustBV> {
    if cc_op == arm64_cc_op::ARM64G_CC_OP_COPY {
        let d1 = extract_to_nbits(dep1, 64, ctx);
        return Some(d1.extract(arm_flag_shift::SHIFT_N, arm_flag_shift::SHIFT_N, ctx));
    }
    let (op, nb) = arm64_decode(cc_op)?;
    let d1 = extract_to_nbits(dep1, nb, ctx);
    let d2 = extract_to_nbits(dep2, nb, ctx);
    let d3 = extract_to_nbits(dep3, nb, ctx);
    let res = arm64_sym_res(op, &d1, &d2, &d3, ctx);
    Some(res.extract(nb - 1, nb - 1, ctx))
}

pub(super) fn arm64_sym_flag_z(
    cc_op: u64,
    dep1: &RustBV,
    dep2: &RustBV,
    dep3: &RustBV,
    ctx: &SymContext,
) -> Option<RustBV> {
    if cc_op == arm64_cc_op::ARM64G_CC_OP_COPY {
        let d1 = extract_to_nbits(dep1, 64, ctx);
        return Some(d1.extract(arm_flag_shift::SHIFT_Z, arm_flag_shift::SHIFT_Z, ctx));
    }
    let (op, nb) = arm64_decode(cc_op)?;
    let d1 = extract_to_nbits(dep1, nb, ctx);
    let d2 = extract_to_nbits(dep2, nb, ctx);
    let d3 = extract_to_nbits(dep3, nb, ctx);
    let res = arm64_sym_res(op, &d1, &d2, &d3, ctx);
    let zero = RustBV::concrete(0, nb);
    Some(res.eq(&zero, ctx))
}

pub(super) fn arm64_sym_flag_c(
    cc_op: u64,
    dep1: &RustBV,
    dep2: &RustBV,
    dep3: &RustBV,
    ctx: &SymContext,
) -> Option<RustBV> {
    if cc_op == arm64_cc_op::ARM64G_CC_OP_COPY {
        let d1 = extract_to_nbits(dep1, 64, ctx);
        return Some(d1.extract(arm_flag_shift::SHIFT_C, arm_flag_shift::SHIFT_C, ctx));
    }
    let (op, nb) = arm64_decode(cc_op)?;
    let d1 = extract_to_nbits(dep1, nb, ctx);
    let d2 = extract_to_nbits(dep2, nb, ctx);
    let zero = RustBV::concrete(0, nb);
    Some(match op {
        Arm64Op::Add => {
            let res = d1.add(&d2, ctx);
            res.ult(&d1, ctx)
        }
        Arm64Op::Sub => d1.uge(&d2, ctx),
        Arm64Op::Adc => {
            let d3 = extract_to_nbits(dep3, nb, ctx);
            let res = d1.add(&d2, ctx).add(&d3, ctx);
            // Python keys this off cc_dep2 (d2), not the carry-in.
            let d2_nz = d2.eq(&zero, ctx).not(ctx);
            let when_nz = res.ule(&d1, ctx);
            let when_z = res.ult(&d1, ctx);
            d2_nz
                .and(&when_nz, ctx)
                .or(&d2_nz.not(ctx).and(&when_z, ctx), ctx)
        }
        Arm64Op::Sbc => {
            let d2_nz = d2.eq(&zero, ctx).not(ctx);
            let when_nz = d1.uge(&d2, ctx);
            let when_z = d1.ugt(&d2, ctx);
            d2_nz
                .and(&when_nz, ctx)
                .or(&d2_nz.not(ctx).and(&when_z, ctx), ctx)
        }
        Arm64Op::Logic => RustBV::concrete(0, 1),
    })
}

pub(super) fn arm64_sym_flag_v(
    cc_op: u64,
    dep1: &RustBV,
    dep2: &RustBV,
    dep3: &RustBV,
    ctx: &SymContext,
) -> Option<RustBV> {
    if cc_op == arm64_cc_op::ARM64G_CC_OP_COPY {
        let d1 = extract_to_nbits(dep1, 64, ctx);
        return Some(d1.extract(arm_flag_shift::SHIFT_V, arm_flag_shift::SHIFT_V, ctx));
    }
    let (op, nb) = arm64_decode(cc_op)?;
    let d1 = extract_to_nbits(dep1, nb, ctx);
    let d2 = extract_to_nbits(dep2, nb, ctx);
    let top = nb - 1;
    Some(match op {
        Arm64Op::Add => {
            let res = d1.add(&d2, ctx);
            res.xor(&d1, ctx)
                .and(&res.xor(&d2, ctx), ctx)
                .extract(top, top, ctx)
        }
        Arm64Op::Sub => {
            let res = d1.sub(&d2, ctx);
            d1.xor(&d2, ctx)
                .and(&d1.xor(&res, ctx), ctx)
                .extract(top, top, ctx)
        }
        Arm64Op::Adc => {
            let d3 = extract_to_nbits(dep3, nb, ctx);
            let res = d1.add(&d2, ctx).add(&d3, ctx);
            res.xor(&d1, ctx)
                .and(&res.xor(&d2, ctx), ctx)
                .extract(top, top, ctx)
        }
        Arm64Op::Sbc => {
            let one = RustBV::concrete(1, nb);
            let d3 = extract_to_nbits(dep3, nb, ctx);
            let res = d1.sub(&d2, ctx).sub(&d3.xor(&one, ctx), ctx);
            d1.xor(&d2, ctx)
                .and(&d1.xor(&res, ctx), ctx)
                .extract(top, top, ctx)
        }
        Arm64Op::Logic => RustBV::concrete(0, 1),
    })
}

/// Symbolic arm64 condition evaluation. Returns a 1-bit BV.
pub(super) fn arm64_sym_calculate_condition(
    cond: u64,
    cc_op: u64,
    dep1: &RustBV,
    dep2: &RustBV,
    dep3: &RustBV,
    ctx: &SymContext,
) -> Option<RustBV> {
    use arm_cond::*;
    let inv = (cond & 1) != 0;

    // AL (14) and NV (15) are both unconditional-true on AArch64.
    if cond & !1 == ARM_COND_AL {
        return Some(RustBV::concrete(1, 1));
    }

    let flag = match cond & !1 {
        ARM_COND_EQ => arm64_sym_flag_z(cc_op, dep1, dep2, dep3, ctx)?,
        ARM_COND_HS => arm64_sym_flag_c(cc_op, dep1, dep2, dep3, ctx)?,
        ARM_COND_MI => arm64_sym_flag_n(cc_op, dep1, dep2, dep3, ctx)?,
        ARM_COND_VS => arm64_sym_flag_v(cc_op, dep1, dep2, dep3, ctx)?,
        ARM_COND_HI => {
            let cf = arm64_sym_flag_c(cc_op, dep1, dep2, dep3, ctx)?;
            let zf = arm64_sym_flag_z(cc_op, dep1, dep2, dep3, ctx)?;
            cf.and(&zf.not(ctx), ctx)
        }
        ARM_COND_GE => {
            let nf = arm64_sym_flag_n(cc_op, dep1, dep2, dep3, ctx)?;
            let vf = arm64_sym_flag_v(cc_op, dep1, dep2, dep3, ctx)?;
            nf.xor(&vf, ctx).not(ctx)
        }
        ARM_COND_GT => {
            let nf = arm64_sym_flag_n(cc_op, dep1, dep2, dep3, ctx)?;
            let vf = arm64_sym_flag_v(cc_op, dep1, dep2, dep3, ctx)?;
            let zf = arm64_sym_flag_z(cc_op, dep1, dep2, dep3, ctx)?;
            zf.or(&nf.xor(&vf, ctx), ctx).not(ctx)
        }
        _ => return None,
    };
    Some(if inv { flag.not(ctx) } else { flag })
}
