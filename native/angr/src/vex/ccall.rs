//! Clean call (CCall) implementations for VEX IR.
//!
//! This module implements the x86/AMD64 helper functions that VEX IR uses for
//! condition code calculations. The main entry point is `handle_ccall` which
//! dispatches to the appropriate helper based on the callee name.

use crate::symbolic::RustBV;
use crate::symbolic::SymContext;

/// CC_OP values for AMD64 (from VEX's libvex_guest_amd64.h)
pub mod amd64_cc_op {
    pub const G_CC_OP_COPY: u64 = 0;
    pub const G_CC_OP_ADDB: u64 = 1;
    pub const G_CC_OP_ADDW: u64 = 2;
    pub const G_CC_OP_ADDL: u64 = 3;
    pub const G_CC_OP_ADDQ: u64 = 4;
    pub const G_CC_OP_SUBB: u64 = 5;
    pub const G_CC_OP_SUBW: u64 = 6;
    pub const G_CC_OP_SUBL: u64 = 7;
    pub const G_CC_OP_SUBQ: u64 = 8;
    pub const G_CC_OP_ADCB: u64 = 9;
    pub const G_CC_OP_ADCW: u64 = 10;
    pub const G_CC_OP_ADCL: u64 = 11;
    pub const G_CC_OP_ADCQ: u64 = 12;
    pub const G_CC_OP_SBBB: u64 = 13;
    pub const G_CC_OP_SBBW: u64 = 14;
    pub const G_CC_OP_SBBL: u64 = 15;
    pub const G_CC_OP_SBBQ: u64 = 16;
    pub const G_CC_OP_LOGICB: u64 = 17;
    pub const G_CC_OP_LOGICW: u64 = 18;
    pub const G_CC_OP_LOGICL: u64 = 19;
    pub const G_CC_OP_LOGICQ: u64 = 20;
    pub const G_CC_OP_INCB: u64 = 21;
    pub const G_CC_OP_INCW: u64 = 22;
    pub const G_CC_OP_INCL: u64 = 23;
    pub const G_CC_OP_INCQ: u64 = 24;
    pub const G_CC_OP_DECB: u64 = 25;
    pub const G_CC_OP_DECW: u64 = 26;
    pub const G_CC_OP_DECL: u64 = 27;
    pub const G_CC_OP_DECQ: u64 = 28;
    pub const G_CC_OP_SHLB: u64 = 29;
    pub const G_CC_OP_SHLW: u64 = 30;
    pub const G_CC_OP_SHLL: u64 = 31;
    pub const G_CC_OP_SHLQ: u64 = 32;
    pub const G_CC_OP_SHRB: u64 = 33;
    pub const G_CC_OP_SHRW: u64 = 34;
    pub const G_CC_OP_SHRL: u64 = 35;
    pub const G_CC_OP_SHRQ: u64 = 36;
    pub const G_CC_OP_ROLB: u64 = 37;
    pub const G_CC_OP_ROLW: u64 = 38;
    pub const G_CC_OP_ROLL: u64 = 39;
    pub const G_CC_OP_ROLQ: u64 = 40;
    pub const G_CC_OP_RORB: u64 = 41;
    pub const G_CC_OP_RORW: u64 = 42;
    pub const G_CC_OP_RORL: u64 = 43;
    pub const G_CC_OP_RORQ: u64 = 44;
    pub const G_CC_OP_UMULB: u64 = 45;
    pub const G_CC_OP_UMULW: u64 = 46;
    pub const G_CC_OP_UMULL: u64 = 47;
    pub const G_CC_OP_UMULQ: u64 = 48;
    pub const G_CC_OP_SMULB: u64 = 49;
    pub const G_CC_OP_SMULW: u64 = 50;
    pub const G_CC_OP_SMULL: u64 = 51;
    pub const G_CC_OP_SMULQ: u64 = 52;
}

/// CC_OP values for X86 (from VEX's libvex_guest_x86.h)
pub mod x86_cc_op {
    pub const G_CC_OP_COPY: u64 = 0;
    pub const G_CC_OP_ADDB: u64 = 1;
    pub const G_CC_OP_ADDW: u64 = 2;
    pub const G_CC_OP_ADDL: u64 = 3;
    pub const G_CC_OP_SUBB: u64 = 4;
    pub const G_CC_OP_SUBW: u64 = 5;
    pub const G_CC_OP_SUBL: u64 = 6;
    pub const G_CC_OP_ADCB: u64 = 7;
    pub const G_CC_OP_ADCW: u64 = 8;
    pub const G_CC_OP_ADCL: u64 = 9;
    pub const G_CC_OP_SBBB: u64 = 10;
    pub const G_CC_OP_SBBW: u64 = 11;
    pub const G_CC_OP_SBBL: u64 = 12;
    pub const G_CC_OP_LOGICB: u64 = 13;
    pub const G_CC_OP_LOGICW: u64 = 14;
    pub const G_CC_OP_LOGICL: u64 = 15;
    pub const G_CC_OP_INCB: u64 = 16;
    pub const G_CC_OP_INCW: u64 = 17;
    pub const G_CC_OP_INCL: u64 = 18;
    pub const G_CC_OP_DECB: u64 = 19;
    pub const G_CC_OP_DECW: u64 = 20;
    pub const G_CC_OP_DECL: u64 = 21;
    pub const G_CC_OP_SHLB: u64 = 22;
    pub const G_CC_OP_SHLW: u64 = 23;
    pub const G_CC_OP_SHLL: u64 = 24;
    pub const G_CC_OP_SHRB: u64 = 25;
    pub const G_CC_OP_SHRW: u64 = 26;
    pub const G_CC_OP_SHRL: u64 = 27;
    pub const G_CC_OP_ROLB: u64 = 28;
    pub const G_CC_OP_ROLW: u64 = 29;
    pub const G_CC_OP_ROLL: u64 = 30;
    pub const G_CC_OP_RORB: u64 = 31;
    pub const G_CC_OP_RORW: u64 = 32;
    pub const G_CC_OP_RORL: u64 = 33;
    pub const G_CC_OP_UMULB: u64 = 34;
    pub const G_CC_OP_UMULW: u64 = 35;
    pub const G_CC_OP_UMULL: u64 = 36;
    pub const G_CC_OP_SMULB: u64 = 37;
    pub const G_CC_OP_SMULW: u64 = 38;
    pub const G_CC_OP_SMULL: u64 = 39;
}

/// Condition types (same for x86 and AMD64)
pub mod cond_type {
    pub const COND_O: u64 = 0; // Overflow
    pub const COND_NO: u64 = 1; // Not overflow
    pub const COND_B: u64 = 2; // Below (CF=1)
    pub const COND_NB: u64 = 3; // Not below (CF=0)
    pub const COND_Z: u64 = 4; // Zero (ZF=1)
    pub const COND_NZ: u64 = 5; // Not zero (ZF=0)
    pub const COND_BE: u64 = 6; // Below or equal (CF=1 or ZF=1)
    pub const COND_NBE: u64 = 7; // Not below or equal (CF=0 and ZF=0)
    pub const COND_S: u64 = 8; // Sign (SF=1)
    pub const COND_NS: u64 = 9; // Not sign (SF=0)
    pub const COND_P: u64 = 10; // Parity even (PF=1)
    pub const COND_NP: u64 = 11; // Parity odd (PF=0)
    pub const COND_L: u64 = 12; // Less (SF != OF)
    pub const COND_NL: u64 = 13; // Not less (SF == OF)
    pub const COND_LE: u64 = 14; // Less or equal (ZF=1 or SF != OF)
    pub const COND_NLE: u64 = 15; // Not less or equal (ZF=0 and SF == OF)
}

/// Flag bit offsets in EFLAGS
pub mod flag_shift {
    pub const G_CC_SHIFT_O: u32 = 11;
    pub const G_CC_SHIFT_S: u32 = 7;
    pub const G_CC_SHIFT_Z: u32 = 6;
    pub const G_CC_SHIFT_A: u32 = 4;
    pub const G_CC_SHIFT_C: u32 = 0;
    pub const G_CC_SHIFT_P: u32 = 2;
}

/// Flag bit masks
pub mod flag_mask {
    pub const G_CC_MASK_O: u64 = 1 << super::flag_shift::G_CC_SHIFT_O;
    pub const G_CC_MASK_S: u64 = 1 << super::flag_shift::G_CC_SHIFT_S;
    pub const G_CC_MASK_Z: u64 = 1 << super::flag_shift::G_CC_SHIFT_Z;
    pub const G_CC_MASK_A: u64 = 1 << super::flag_shift::G_CC_SHIFT_A;
    pub const G_CC_MASK_C: u64 = 1 << super::flag_shift::G_CC_SHIFT_C;
    pub const G_CC_MASK_P: u64 = 1 << super::flag_shift::G_CC_SHIFT_P;
}

/// Computed flags from an operation
#[derive(Debug, Clone, Copy)]
struct Flags {
    cf: u8, // Carry flag
    pf: u8, // Parity flag
    zf: u8, // Zero flag
    sf: u8, // Sign flag
    of: u8, // Overflow flag
}

// ============================================================
// Symbolic flag computation helpers
// ============================================================

/// Extract the low `nbits` from a value that may be wider (e.g. 64-bit AMD64 args for 8/16/32-bit ops).
fn extract_to_nbits(val: &RustBV, nbits: u32, ctx: &SymContext) -> RustBV {
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
fn symbolic_parity(result: &RustBV, ctx: &SymContext) -> RustBV {
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
fn symbolic_pack_eflags(
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

/// Symbolic eflags computation for SUB/CMP: flags from dep1 - dep2
fn symbolic_eflags_sub(
    nbits: u32,
    dep1: &RustBV,
    dep2: &RustBV,
    ctx: &SymContext,
    ret_bits: u32,
) -> RustBV {
    let d1 = extract_to_nbits(dep1, nbits, ctx);
    let d2 = extract_to_nbits(dep2, nbits, ctx);
    let result = d1.sub(&d2, ctx);
    let zero = RustBV::concrete(0, nbits);

    // ZF = (result == 0)
    let zf = result.eq(&zero, ctx);
    // SF = result[msb]
    let sf = result.extract(nbits - 1, nbits - 1, ctx);
    // CF = (dep1 < dep2) unsigned — borrow
    let cf = d1.ult(&d2, ctx);
    // OF = ((dep1 ^ dep2) & (dep1 ^ result))[msb] — different signs & result sign differs from dep1
    let of = d1
        .xor(&d2, ctx)
        .and(&d1.xor(&result, ctx), ctx)
        .extract(nbits - 1, nbits - 1, ctx);
    // PF = parity of low byte of result
    let pf = symbolic_parity(&result, ctx);

    symbolic_pack_eflags(&of, &sf, &zf, &pf, &cf, ret_bits, ctx)
}

/// Symbolic eflags computation for ADD: flags from dep1 + dep2
fn symbolic_eflags_add(
    nbits: u32,
    dep1: &RustBV,
    dep2: &RustBV,
    ctx: &SymContext,
    ret_bits: u32,
) -> RustBV {
    let d1 = extract_to_nbits(dep1, nbits, ctx);
    let d2 = extract_to_nbits(dep2, nbits, ctx);
    let result = d1.add(&d2, ctx);
    let zero = RustBV::concrete(0, nbits);

    let zf = result.eq(&zero, ctx);
    let sf = result.extract(nbits - 1, nbits - 1, ctx);
    // CF = (result < dep1) unsigned — carry out
    let cf = result.ult(&d1, ctx);
    // OF = (~(dep1 ^ dep2) & (dep1 ^ result))[msb] — same sign operands, different sign result
    let of = d1
        .xor(&d2, ctx)
        .not(ctx)
        .and(&d1.xor(&result, ctx), ctx)
        .extract(nbits - 1, nbits - 1, ctx);
    let pf = symbolic_parity(&result, ctx);

    symbolic_pack_eflags(&of, &sf, &zf, &pf, &cf, ret_bits, ctx)
}

/// Symbolic eflags computation for LOGIC (AND/OR/XOR): flags from result in dep1
fn symbolic_eflags_logic(nbits: u32, dep1: &RustBV, ctx: &SymContext, ret_bits: u32) -> RustBV {
    let result = extract_to_nbits(dep1, nbits, ctx);
    let zero = RustBV::concrete(0, nbits);

    let zf = result.eq(&zero, ctx);
    let sf = result.extract(nbits - 1, nbits - 1, ctx);
    // CF = 0, OF = 0 for logic ops
    let cf = RustBV::concrete(0, 1);
    let of = RustBV::concrete(0, 1);
    let pf = symbolic_parity(&result, ctx);

    symbolic_pack_eflags(&of, &sf, &zf, &pf, &cf, ret_bits, ctx)
}

// ============================================================
// Concrete flag computation
// ============================================================

/// Calculate parity bit (1 if even parity in low 8 bits)
fn calc_parity(val: u64) -> u8 {
    let byte = val as u8;
    // Count 1 bits in the byte, return 1 if even (even parity)
    if byte.count_ones() % 2 == 0 { 1 } else { 0 }
}

/// Get bitmask for an n-bit value (e.g., nbits=32 -> 0xFFFFFFFF).
#[inline]
fn get_mask(nbits: u32) -> u64 {
    if nbits == 64 {
        u64::MAX
    } else {
        (1u64 << nbits) - 1
    }
}

/// Get the sign bit for an n-bit value (e.g., nbits=32 -> 0x80000000).
#[inline]
fn get_sign_bit(nbits: u32) -> u64 {
    1u64 << (nbits - 1)
}

/// Calculate flags for SUB operation (CMP uses this)
fn calc_flags_sub(nbits: u32, arg_l: u64, arg_r: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);

    let res = arg_l.wrapping_sub(arg_r) & mask;

    // CF: set if borrow (unsigned: arg_l < arg_r)
    let cf = if arg_l < arg_r { 1 } else { 0 };

    // ZF: set if result is zero
    let zf = if res == 0 { 1 } else { 0 };

    // SF: set if result is negative (sign bit set)
    let sf = if (res & sign_bit) != 0 { 1 } else { 0 };

    // OF: set if signed overflow
    // Overflow occurs if: (arg_l ^ arg_r) & (arg_l ^ res) has sign bit set
    let of = if ((arg_l ^ arg_r) & (arg_l ^ res) & sign_bit) != 0 {
        1
    } else {
        0
    };

    // PF: parity of low 8 bits
    let pf = calc_parity(res);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for ADD operation
fn calc_flags_add(nbits: u32, arg_l: u64, arg_r: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);

    let res = arg_l.wrapping_add(arg_r) & mask;

    // CF: set if carry (unsigned overflow: res < arg_l)
    let cf = if res < arg_l { 1 } else { 0 };

    // ZF: set if result is zero
    let zf = if res == 0 { 1 } else { 0 };

    // SF: set if result is negative
    let sf = if (res & sign_bit) != 0 { 1 } else { 0 };

    // OF: set if signed overflow
    // For addition: overflow if both operands have same sign and result has different sign
    // OF = ((arg_l ^ arg_r ^ mask) & (arg_l ^ res)) has sign bit set
    // Simplified: same sign operands, different sign result
    let of = if ((!(arg_l ^ arg_r)) & (arg_l ^ res) & sign_bit) != 0 {
        1
    } else {
        0
    };

    // PF: parity of low 8 bits
    let pf = calc_parity(res);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for LOGIC operation (AND, OR, XOR, TEST)
fn calc_flags_logic(nbits: u32, result: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);
    let res = result & mask;

    // CF and OF are always 0 for logic ops
    let cf = 0;
    let of = 0;

    // ZF: set if result is zero
    let zf = if res == 0 { 1 } else { 0 };

    // SF: set if result is negative
    let sf = if (res & sign_bit) != 0 { 1 } else { 0 };

    // PF: parity of low 8 bits
    let pf = calc_parity(res);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for INC operation
fn calc_flags_inc(nbits: u32, res: u64, cc_ndep: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);
    let res = res & mask;

    // CF is preserved from cc_ndep
    let cf = ((cc_ndep >> flag_shift::G_CC_SHIFT_C) & 1) as u8;

    // ZF: set if result is zero
    let zf = if res == 0 { 1 } else { 0 };

    // SF: set if result is negative
    let sf = if (res & sign_bit) != 0 { 1 } else { 0 };

    // OF: set if res == 0x80...0 (incremented from 0x7F...F)
    let of = if res == sign_bit { 1 } else { 0 };

    // PF: parity of low 8 bits
    let pf = calc_parity(res);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for DEC operation
fn calc_flags_dec(nbits: u32, res: u64, cc_ndep: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);
    let res = res & mask;

    // CF is preserved from cc_ndep
    let cf = ((cc_ndep >> flag_shift::G_CC_SHIFT_C) & 1) as u8;

    // ZF: set if result is zero
    let zf = if res == 0 { 1 } else { 0 };

    // SF: set if result is negative
    let sf = if (res & sign_bit) != 0 { 1 } else { 0 };

    // OF: set if res == 0x7F...F (decremented from 0x80...0)
    let of = if res == (sign_bit - 1) { 1 } else { 0 };

    // PF: parity of low 8 bits
    let pf = calc_parity(res);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for SHL (shift left) operation
fn calc_flags_shl(nbits: u32, remaining: u64, shifted: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);
    let remaining = remaining & mask;
    let shifted = shifted & mask;

    // CF: last bit shifted out (MSB of shifted value)
    let cf = ((shifted >> (nbits - 1)) & 1) as u8;

    // ZF: set if result is zero
    let zf = if remaining == 0 { 1 } else { 0 };

    // SF: set if result is negative
    let sf = if (remaining & sign_bit) != 0 { 1 } else { 0 };

    // OF: XOR of CF and SF (for shift by 1)
    let of = cf ^ sf;

    // PF: parity of low 8 bits
    let pf = calc_parity(remaining);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for SHR (shift right) operation
fn calc_flags_shr(nbits: u32, remaining: u64, shifted: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);
    let remaining = remaining & mask;
    let shifted = shifted & mask;

    // CF: last bit shifted out (LSB of original, i.e., bit 0 of shifted)
    let cf = (shifted & 1) as u8;

    // ZF: set if result is zero
    let zf = if remaining == 0 { 1 } else { 0 };

    // SF: set if result is negative
    let sf = if (remaining & sign_bit) != 0 { 1 } else { 0 };

    // OF: MSB of original value (for shift by 1)
    let of = ((shifted >> (nbits - 1)) ^ (remaining >> (nbits - 1))) as u8 & 1;

    // PF: parity of low 8 bits
    let pf = calc_parity(remaining);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for ROL (rotate left) operation
fn calc_flags_rol(nbits: u32, res: u64, cc_ndep: u64) -> Flags {
    let mask = get_mask(nbits);
    let res = res & mask;

    // CF: LSB of result
    let cf = (res & 1) as u8;

    // PF, ZF, SF are preserved from cc_ndep
    let pf = ((cc_ndep >> flag_shift::G_CC_SHIFT_P) & 1) as u8;
    let zf = ((cc_ndep >> flag_shift::G_CC_SHIFT_Z) & 1) as u8;
    let sf = ((cc_ndep >> flag_shift::G_CC_SHIFT_S) & 1) as u8;

    // OF: MSB XOR LSB of result
    let of = (((res >> (nbits - 1)) ^ res) & 1) as u8;

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for ROR (rotate right) operation
fn calc_flags_ror(nbits: u32, res: u64, cc_ndep: u64) -> Flags {
    let mask = get_mask(nbits);
    let res = res & mask;

    // CF: MSB of result
    let cf = ((res >> (nbits - 1)) & 1) as u8;

    // PF, ZF, SF are preserved from cc_ndep
    let pf = ((cc_ndep >> flag_shift::G_CC_SHIFT_P) & 1) as u8;
    let zf = ((cc_ndep >> flag_shift::G_CC_SHIFT_Z) & 1) as u8;
    let sf = ((cc_ndep >> flag_shift::G_CC_SHIFT_S) & 1) as u8;

    // OF: XOR of two MSBs of result
    let of = (((res >> (nbits - 1)) ^ (res >> (nbits - 2))) & 1) as u8;

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for ADC (add with carry) operation
fn calc_flags_adc(nbits: u32, cc_dep1: u64, cc_dep2: u64, cc_ndep: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);

    let old_c = (cc_ndep >> flag_shift::G_CC_SHIFT_C) & 1;
    let arg_l = cc_dep1 & mask;
    let arg_r = (cc_dep2 ^ old_c) & mask;
    let res = (arg_l.wrapping_add(arg_r).wrapping_add(old_c)) & mask;

    // CF: carry out
    let cf = if old_c != 0 {
        if res <= arg_l { 1 } else { 0 }
    } else {
        if res < arg_l { 1 } else { 0 }
    };

    // ZF: set if result is zero
    let zf = if res == 0 { 1 } else { 0 };

    // SF: set if result is negative
    let sf = if (res & sign_bit) != 0 { 1 } else { 0 };

    // OF: signed overflow
    let of = if ((!(arg_l ^ arg_r)) & (arg_l ^ res) & sign_bit) != 0 {
        1
    } else {
        0
    };

    // PF: parity of low 8 bits
    let pf = calc_parity(res);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for SBB (subtract with borrow) operation
fn calc_flags_sbb(nbits: u32, cc_dep1: u64, cc_dep2: u64, cc_ndep: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);

    let old_c = (cc_ndep >> flag_shift::G_CC_SHIFT_C) & 1;
    let arg_l = cc_dep1 & mask;
    let arg_r = (cc_dep2 ^ old_c) & mask;
    let res = (arg_l.wrapping_sub(arg_r).wrapping_sub(old_c)) & mask;

    // CF: borrow out
    let cf = if old_c != 0 {
        if arg_l <= arg_r { 1 } else { 0 }
    } else {
        if arg_l < arg_r { 1 } else { 0 }
    };

    // ZF: set if result is zero
    let zf = if res == 0 { 1 } else { 0 };

    // SF: set if result is negative
    let sf = if (res & sign_bit) != 0 { 1 } else { 0 };

    // OF: signed overflow
    let of = if ((arg_l ^ arg_r) & (arg_l ^ res) & sign_bit) != 0 {
        1
    } else {
        0
    };

    // PF: parity of low 8 bits
    let pf = calc_parity(res);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for UMUL (unsigned multiply) operation
fn calc_flags_umul(nbits: u32, cc_dep1: u64, cc_dep2: u64) -> Flags {
    let mask = get_mask(nbits);

    let lo = (cc_dep1.wrapping_mul(cc_dep2)) & mask;

    // For proper overflow detection, we need to check if the result
    // fits in nbits. For simplicity, approximate using 128-bit math for 64-bit.
    let hi = if nbits == 64 {
        ((cc_dep1 as u128 * cc_dep2 as u128) >> 64) as u64
    } else {
        (cc_dep1.wrapping_mul(cc_dep2) >> nbits) & mask
    };

    // CF/OF: set if high part is non-zero
    let cf = if hi != 0 { 1 } else { 0 };
    let of = cf;

    // ZF, SF, PF are undefined but we compute them anyway
    let zf = if lo == 0 { 1 } else { 0 };
    let sf = ((lo >> (nbits - 1)) & 1) as u8;
    let pf = calc_parity(lo);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for SMUL (signed multiply) operation
fn calc_flags_smul(nbits: u32, cc_dep1: u64, cc_dep2: u64) -> Flags {
    let mask = get_mask(nbits);

    // Sign-extend operands
    let arg1_signed = if nbits == 64 {
        cc_dep1 as i64
    } else {
        let sign_bit = get_sign_bit(nbits);
        if (cc_dep1 & sign_bit) != 0 {
            (cc_dep1 | !mask) as i64
        } else {
            cc_dep1 as i64
        }
    };

    let arg2_signed = if nbits == 64 {
        cc_dep2 as i64
    } else {
        let sign_bit = get_sign_bit(nbits);
        if (cc_dep2 & sign_bit) != 0 {
            (cc_dep2 | !mask) as i64
        } else {
            cc_dep2 as i64
        }
    };

    let result = (arg1_signed as i128) * (arg2_signed as i128);
    let lo = (result as u64) & mask;
    let hi = ((result >> nbits) as u64) & mask;

    // Sign-extend lo to compare with hi
    let lo_sign_ext = if nbits == 64 {
        if (lo as i64) < 0 { u64::MAX } else { 0 }
    } else {
        let sign_bit = get_sign_bit(nbits);
        if (lo & sign_bit) != 0 { mask } else { 0 }
    };

    // CF/OF: set if hi != sign extension of lo
    let cf = if hi != lo_sign_ext { 1 } else { 0 };
    let of = cf;

    // ZF, SF, PF
    let zf = if lo == 0 { 1 } else { 0 };
    let sf = ((lo >> (nbits - 1)) & 1) as u8;
    let pf = calc_parity(lo);

    Flags { cf, pf, zf, sf, of }
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

/// Decode an AMD64 CC_OP into (nbits, category). Returns `None` for unknown values.
fn amd64_cc_op_info(cc_op: u64) -> Option<CcOpInfo> {
    use OpCategory::*;
    use amd64_cc_op::*;
    let (nbits, category) = match cc_op {
        G_CC_OP_COPY => (64, Copy),
        G_CC_OP_ADDB => (8, Add),
        G_CC_OP_ADDW => (16, Add),
        G_CC_OP_ADDL => (32, Add),
        G_CC_OP_ADDQ => (64, Add),
        G_CC_OP_SUBB => (8, Sub),
        G_CC_OP_SUBW => (16, Sub),
        G_CC_OP_SUBL => (32, Sub),
        G_CC_OP_SUBQ => (64, Sub),
        G_CC_OP_ADCB => (8, Adc),
        G_CC_OP_ADCW => (16, Adc),
        G_CC_OP_ADCL => (32, Adc),
        G_CC_OP_ADCQ => (64, Adc),
        G_CC_OP_SBBB => (8, Sbb),
        G_CC_OP_SBBW => (16, Sbb),
        G_CC_OP_SBBL => (32, Sbb),
        G_CC_OP_SBBQ => (64, Sbb),
        G_CC_OP_LOGICB => (8, Logic),
        G_CC_OP_LOGICW => (16, Logic),
        G_CC_OP_LOGICL => (32, Logic),
        G_CC_OP_LOGICQ => (64, Logic),
        G_CC_OP_INCB => (8, Inc),
        G_CC_OP_INCW => (16, Inc),
        G_CC_OP_INCL => (32, Inc),
        G_CC_OP_INCQ => (64, Inc),
        G_CC_OP_DECB => (8, Dec),
        G_CC_OP_DECW => (16, Dec),
        G_CC_OP_DECL => (32, Dec),
        G_CC_OP_DECQ => (64, Dec),
        G_CC_OP_SHLB => (8, Shl),
        G_CC_OP_SHLW => (16, Shl),
        G_CC_OP_SHLL => (32, Shl),
        G_CC_OP_SHLQ => (64, Shl),
        G_CC_OP_SHRB => (8, Shr),
        G_CC_OP_SHRW => (16, Shr),
        G_CC_OP_SHRL => (32, Shr),
        G_CC_OP_SHRQ => (64, Shr),
        G_CC_OP_ROLB => (8, Rol),
        G_CC_OP_ROLW => (16, Rol),
        G_CC_OP_ROLL => (32, Rol),
        G_CC_OP_ROLQ => (64, Rol),
        G_CC_OP_RORB => (8, Ror),
        G_CC_OP_RORW => (16, Ror),
        G_CC_OP_RORL => (32, Ror),
        G_CC_OP_RORQ => (64, Ror),
        G_CC_OP_UMULB => (8, Umul),
        G_CC_OP_UMULW => (16, Umul),
        G_CC_OP_UMULL => (32, Umul),
        G_CC_OP_UMULQ => (64, Umul),
        G_CC_OP_SMULB => (8, Smul),
        G_CC_OP_SMULW => (16, Smul),
        G_CC_OP_SMULL => (32, Smul),
        G_CC_OP_SMULQ => (64, Smul),
        _ => return None,
    };
    Some(CcOpInfo { nbits, category })
}

/// Decode an X86 CC_OP into (nbits, category). Returns `None` for unknown values.
fn x86_cc_op_info(cc_op: u64) -> Option<CcOpInfo> {
    use OpCategory::*;
    use x86_cc_op::*;
    let (nbits, category) = match cc_op {
        G_CC_OP_COPY => (32, Copy),
        G_CC_OP_ADDB => (8, Add),
        G_CC_OP_ADDW => (16, Add),
        G_CC_OP_ADDL => (32, Add),
        G_CC_OP_SUBB => (8, Sub),
        G_CC_OP_SUBW => (16, Sub),
        G_CC_OP_SUBL => (32, Sub),
        G_CC_OP_ADCB => (8, Adc),
        G_CC_OP_ADCW => (16, Adc),
        G_CC_OP_ADCL => (32, Adc),
        G_CC_OP_SBBB => (8, Sbb),
        G_CC_OP_SBBW => (16, Sbb),
        G_CC_OP_SBBL => (32, Sbb),
        G_CC_OP_LOGICB => (8, Logic),
        G_CC_OP_LOGICW => (16, Logic),
        G_CC_OP_LOGICL => (32, Logic),
        G_CC_OP_INCB => (8, Inc),
        G_CC_OP_INCW => (16, Inc),
        G_CC_OP_INCL => (32, Inc),
        G_CC_OP_DECB => (8, Dec),
        G_CC_OP_DECW => (16, Dec),
        G_CC_OP_DECL => (32, Dec),
        G_CC_OP_SHLB => (8, Shl),
        G_CC_OP_SHLW => (16, Shl),
        G_CC_OP_SHLL => (32, Shl),
        G_CC_OP_SHRB => (8, Shr),
        G_CC_OP_SHRW => (16, Shr),
        G_CC_OP_SHRL => (32, Shr),
        G_CC_OP_ROLB => (8, Rol),
        G_CC_OP_ROLW => (16, Rol),
        G_CC_OP_ROLL => (32, Rol),
        G_CC_OP_RORB => (8, Ror),
        G_CC_OP_RORW => (16, Ror),
        G_CC_OP_RORL => (32, Ror),
        G_CC_OP_UMULB => (8, Umul),
        G_CC_OP_UMULW => (16, Umul),
        G_CC_OP_UMULL => (32, Umul),
        G_CC_OP_SMULB => (8, Smul),
        G_CC_OP_SMULW => (16, Smul),
        G_CC_OP_SMULL => (32, Smul),
        _ => return None,
    };
    Some(CcOpInfo { nbits, category })
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

/// Evaluate a condition based on flags
fn eval_condition(cond: u64, flags: &Flags) -> u64 {
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
        _ => 0, // Unknown condition
    };

    (result & 1) as u64
}

/// Evaluate condition from COPY operation (flags in cc_dep1)
fn eval_condition_from_copy(cond: u64, cc_dep1: u64) -> u64 {
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
        return Some(eval_condition_from_copy(cond, cc_dep1));
    }
    let flags = compute_flags_from_category(info.category, info.nbits, cc_dep1, cc_dep2, cc_ndep);
    Some(eval_condition(cond, &flags))
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
mod arm_flag_shift {
    pub const SHIFT_N: u32 = 31;
    pub const SHIFT_Z: u32 = 30;
    pub const SHIFT_C: u32 = 29;
    pub const SHIFT_V: u32 = 28;
}

/// Compute ARM N (negative) flag for a given cc_op.
fn armg_calc_flag_n(cc_op: u64, dep1: u64, dep2: u64, ndep: u64) -> Option<u64> {
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
fn armg_calc_flag_z(cc_op: u64, dep1: u64, dep2: u64, ndep: u64) -> Option<u64> {
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
fn armg_calc_flag_c(cc_op: u64, dep1: u64, dep2: u64, ndep: u64) -> Option<u64> {
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
fn armg_calc_flag_v(cc_op: u64, dep1: u64, dep2: u64, ndep: u64) -> Option<u64> {
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
/// `cond_n_op` encodes: cond in bits [7:4], cc_op in bits [3:0].
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

/// Compute all ARM NZCV flags and pack into bits [31:28].
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

/// Handle a CCall expression.
///
/// Returns Some(result) if the call was handled, None if not supported.
pub fn handle_ccall(name: &str, args: &[RustBV], ret_bits: u32) -> Option<RustBV> {
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

        // Symbolic path: handle SUB/LOGIC/ADD with symbolic deps.
        // This enables symbolic branch detection for comparisons.
        if let (Some(cond), Some(cc_op), Some(sym_ctx)) = (args[0].as_u64(), args[1].as_u64(), ctx)
        {
            let dep1 = &args[2];
            let dep2 = &args[3];
            let arch = CcArch::from_ccall_name(name);
            if let Some(info) = cc_op_info(arch, cc_op) {
                use cond_type::*;
                let inv = (cond & 1) != 0;
                let nb = info.nbits;
                let result_flag = match info.category {
                    OpCategory::Sub => {
                        // For SUB: must extract to nbits first (64-bit temps for 8/16/32-bit ops).
                        let d1 = extract_to_nbits(dep1, nb, sym_ctx);
                        let d2 = extract_to_nbits(dep2, nb, sym_ctx);
                        match cond & !1 {
                            COND_Z => Some(d1.eq(&d2, sym_ctx)),
                            COND_B => Some(d1.ult(&d2, sym_ctx)),
                            COND_BE => Some(d1.ule(&d2, sym_ctx)),
                            COND_L => Some(d1.slt(&d2, sym_ctx)),
                            COND_LE => Some(d1.sle(&d2, sym_ctx)),
                            _ => None,
                        }
                    }
                    OpCategory::Logic => {
                        let d1 = extract_to_nbits(dep1, nb, sym_ctx);
                        match cond & !1 {
                            COND_Z => {
                                let zero = RustBV::concrete(0, nb);
                                Some(d1.eq(&zero, sym_ctx))
                            }
                            COND_S => Some(d1.extract(nb - 1, nb - 1, sym_ctx)),
                            _ => None,
                        }
                    }
                    OpCategory::Add => {
                        let d1 = extract_to_nbits(dep1, nb, sym_ctx);
                        let d2 = extract_to_nbits(dep2, nb, sym_ctx);
                        let result = d1.add(&d2, sym_ctx);
                        match cond & !1 {
                            COND_Z => {
                                let zero = RustBV::concrete(0, nb);
                                Some(result.eq(&zero, sym_ctx))
                            }
                            // CF = result < dep1 (unsigned overflow)
                            COND_B => Some(result.ult(&d1, sym_ctx)),
                            COND_S => Some(result.extract(nb - 1, nb - 1, sym_ctx)),
                            _ => None,
                        }
                    }
                    _ => None, // Other categories: fall through to None.
                };
                if let Some(flag) = result_flag {
                    let r = if inv { flag.not(sym_ctx) } else { flag };
                    return Some(r.zero_extend(ret_bits, sym_ctx));
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
                        // CF = (dep1 >> SHIFT_C) & 1
                        let shift =
                            RustBV::concrete(flag_shift::G_CC_SHIFT_C as u128, args[1].width());
                        let one = RustBV::concrete(1, args[1].width());
                        Some(
                            args[1]
                                .lshr(&shift, sym_ctx)
                                .and(&one, sym_ctx)
                                .extract(0, 0, sym_ctx),
                        )
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
                    _ => None,
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
                let flags_mask: u128 = 0xD5; // O|S|Z|A|P|C flags
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
                let linear = ((ss_val & 0xFFFF) << 16).wrapping_add(va_val & 0xFFFFFFFF);
                return Some(RustBV::concrete(linear as u128, ret_bits));
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

        // Symbolic path: handle concrete cond_n_op with symbolic deps
        if let (Some(cond_n_op), Some(sym_ctx)) = (args[0].as_u64(), ctx) {
            let cond = (cond_n_op >> 4) & 0xF;
            let cc_op = cond_n_op & 0xF;
            let inv = (cond & 1) != 0;

            use arm_cc_op::*;
            use arm_cond::*;

            // AL: always true
            if cond == ARM_COND_AL {
                return Some(RustBV::concrete(1, ret_bits));
            }
            // NV: always false
            if cond == ARM_COND_NV {
                return Some(RustBV::concrete(0, ret_bits));
            }

            let dep1 = &args[1];
            let dep2 = &args[2];

            // For SUB cc_op, we can compute symbolic comparisons directly
            if cc_op == ARMG_CC_OP_SUB {
                let result_flag = match cond & !1 {
                    ARM_COND_EQ => {
                        // Z flag: dep1 == dep2
                        Some(dep1.eq(dep2, sym_ctx))
                    }
                    ARM_COND_HS => {
                        // C flag: dep1 >= dep2 (unsigned)
                        Some(dep1.uge(dep2, sym_ctx))
                    }
                    ARM_COND_MI => {
                        // N flag: (dep1 - dep2)[31]
                        let res = dep1.sub(dep2, sym_ctx);
                        Some(res.extract(31, 31, sym_ctx))
                    }
                    ARM_COND_HI => {
                        // C && !Z: dep1 > dep2 (unsigned)
                        Some(dep1.ugt(dep2, sym_ctx))
                    }
                    ARM_COND_GE => {
                        // N == V: signed >=
                        Some(dep1.sge(dep2, sym_ctx))
                    }
                    ARM_COND_GT => {
                        // !Z && N==V: signed >
                        Some(dep1.sgt(dep2, sym_ctx))
                    }
                    _ => None,
                };
                if let Some(flag) = result_flag {
                    let r = if inv { flag.not(sym_ctx) } else { flag };
                    return Some(r.zero_extend(ret_bits, sym_ctx));
                }
            }

            // For ADD cc_op
            if cc_op == ARMG_CC_OP_ADD {
                let result_flag = match cond & !1 {
                    ARM_COND_EQ => {
                        // Z: result == 0
                        let res = dep1.add(dep2, sym_ctx);
                        let zero = RustBV::concrete(0, 32);
                        let res32 = extract_to_nbits(&res, 32, sym_ctx);
                        Some(res32.eq(&zero, sym_ctx))
                    }
                    ARM_COND_MI => {
                        // N: result[31]
                        let res = dep1.add(dep2, sym_ctx);
                        let res32 = extract_to_nbits(&res, 32, sym_ctx);
                        Some(res32.extract(31, 31, sym_ctx))
                    }
                    _ => None,
                };
                if let Some(flag) = result_flag {
                    let r = if inv { flag.not(sym_ctx) } else { flag };
                    return Some(r.zero_extend(ret_bits, sym_ctx));
                }
            }

            // For LOGIC cc_op
            if cc_op == ARMG_CC_OP_LOGIC {
                let result_flag = match cond & !1 {
                    ARM_COND_EQ => {
                        // Z: dep1 == 0
                        let zero = RustBV::concrete(0, dep1.width());
                        Some(dep1.eq(&zero, sym_ctx))
                    }
                    ARM_COND_MI => {
                        // N: dep1[31]
                        let d1 = extract_to_nbits(dep1, 32, sym_ctx);
                        Some(d1.extract(31, 31, sym_ctx))
                    }
                    _ => None,
                };
                if let Some(flag) = result_flag {
                    let r = if inv { flag.not(sym_ctx) } else { flag };
                    return Some(r.zero_extend(ret_bits, sym_ctx));
                }
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

    // Not a supported CCall
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_parity() {
        assert_eq!(calc_parity(0x00), 1); // 0 ones = even
        assert_eq!(calc_parity(0x01), 0); // 1 one = odd
        assert_eq!(calc_parity(0x03), 1); // 2 ones = even
        assert_eq!(calc_parity(0xFF), 1); // 8 ones = even
        assert_eq!(calc_parity(0xFE), 0); // 7 ones = odd
    }

    #[test]
    fn test_sub_flags_zero() {
        // 5 - 5 = 0, should set ZF
        let flags = calc_flags_sub(32, 5, 5);
        assert_eq!(flags.zf, 1);
        assert_eq!(flags.cf, 0);
        assert_eq!(flags.sf, 0);
        assert_eq!(flags.of, 0);
    }

    #[test]
    fn test_sub_flags_negative() {
        // 3 - 5 = -2, should set SF and CF
        let flags = calc_flags_sub(32, 3, 5);
        assert_eq!(flags.zf, 0);
        assert_eq!(flags.cf, 1); // borrow
        assert_eq!(flags.sf, 1); // negative result
        assert_eq!(flags.of, 0); // no signed overflow
    }

    #[test]
    fn test_sub_flags_overflow() {
        // 0x80000000 - 1 = 0x7FFFFFFF, signed overflow (MIN_INT - 1)
        let flags = calc_flags_sub(32, 0x80000000, 1);
        assert_eq!(flags.of, 1); // signed overflow
        assert_eq!(flags.sf, 0); // positive result
    }

    #[test]
    fn test_logic_flags_zero() {
        // AND result = 0, should set ZF
        let flags = calc_flags_logic(32, 0);
        assert_eq!(flags.zf, 1);
        assert_eq!(flags.cf, 0);
        assert_eq!(flags.sf, 0);
        assert_eq!(flags.of, 0);
    }

    #[test]
    fn test_logic_flags_negative() {
        // TEST result with sign bit set
        let flags = calc_flags_logic(32, 0x80000000);
        assert_eq!(flags.zf, 0);
        assert_eq!(flags.sf, 1);
    }

    #[test]
    fn test_amd64_condition_setz_after_cmp() {
        // CMP 5, 5 (SUB 5, 5 = 0) then SETZ
        // Should return 1 (ZF is set)
        use amd64_cc_op::G_CC_OP_SUBL;
        use cond_type::COND_Z;

        let result = amd64g_calculate_condition(COND_Z, G_CC_OP_SUBL, 5, 5, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_amd64_condition_setnz_after_cmp() {
        // CMP 5, 3 (SUB 5, 3 = 2) then SETNZ
        // Should return 1 (ZF is clear)
        use amd64_cc_op::G_CC_OP_SUBL;
        use cond_type::COND_NZ;

        let result = amd64g_calculate_condition(COND_NZ, G_CC_OP_SUBL, 5, 3, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_amd64_condition_setb_after_cmp() {
        // CMP 3, 5 (SUB 3, 5 = borrow) then SETB
        // Should return 1 (CF is set)
        use amd64_cc_op::G_CC_OP_SUBL;
        use cond_type::COND_B;

        let result = amd64g_calculate_condition(COND_B, G_CC_OP_SUBL, 3, 5, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_amd64_condition_seta_after_cmp() {
        // CMP 5, 3 (SUB 5, 3 = no borrow, non-zero) then SETA (NBE)
        // Should return 1 (CF=0 and ZF=0)
        use amd64_cc_op::G_CC_OP_SUBL;
        use cond_type::COND_NBE;

        let result = amd64g_calculate_condition(COND_NBE, G_CC_OP_SUBL, 5, 3, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_amd64_condition_setz_after_test() {
        // TEST 0, 0 (LOGIC 0) then SETZ
        // Should return 1 (ZF is set)
        use amd64_cc_op::G_CC_OP_LOGICL;
        use cond_type::COND_Z;

        let result = amd64g_calculate_condition(COND_Z, G_CC_OP_LOGICL, 0, 0, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_amd64_condition_sets_after_test() {
        // TEST with negative result then SETS
        use amd64_cc_op::G_CC_OP_LOGICL;
        use cond_type::COND_S;

        let result = amd64g_calculate_condition(COND_S, G_CC_OP_LOGICL, 0x80000000, 0, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_amd64_condition_setl_signed() {
        // CMP -1, 1 then SETL (signed less than)
        // -1 < 1 is true
        use amd64_cc_op::G_CC_OP_SUBL;
        use cond_type::COND_L;

        let result = amd64g_calculate_condition(COND_L, G_CC_OP_SUBL, 0xFFFFFFFF, 1, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_x86_condition_setz() {
        use cond_type::COND_Z;
        use x86_cc_op::G_CC_OP_SUBL;

        let result = x86g_calculate_condition(COND_Z, G_CC_OP_SUBL, 5, 5, 0);
        assert_eq!(result, Some(1));
    }

    // ARM condition code tests

    /// Helper: encode ARM cond_n_op from condition and cc_op
    fn arm_cond_n_op(cond: u64, cc_op: u64) -> u64 {
        (cond << 4) | cc_op
    }

    #[test]
    fn test_arm_cond_eq_sub_equal() {
        // CMP 5, 5 (SUB) → EQ should be 1
        use arm_cc_op::*;
        use arm_cond::*;
        let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_EQ, ARMG_CC_OP_SUB), 5, 5, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_arm_cond_ne_sub_equal() {
        // CMP 5, 5 (SUB) → NE should be 0
        use arm_cc_op::*;
        use arm_cond::*;
        let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_NE, ARMG_CC_OP_SUB), 5, 5, 0);
        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_arm_cond_ne_sub_different() {
        // CMP 5, 3 (SUB) → NE should be 1
        use arm_cc_op::*;
        use arm_cond::*;
        let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_NE, ARMG_CC_OP_SUB), 5, 3, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_arm_cond_hs_sub() {
        // CMP 5, 3 → HS (unsigned >=) should be 1
        use arm_cc_op::*;
        use arm_cond::*;
        let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_HS, ARMG_CC_OP_SUB), 5, 3, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_arm_cond_lo_sub() {
        // CMP 3, 5 → LO (unsigned <) should be 1
        use arm_cc_op::*;
        use arm_cond::*;
        let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_LO, ARMG_CC_OP_SUB), 3, 5, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_arm_cond_mi_sub() {
        // CMP 3, 5 → MI (negative) should be 1 (3-5 = -2, N set)
        use arm_cc_op::*;
        use arm_cond::*;
        let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_MI, ARMG_CC_OP_SUB), 3, 5, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_arm_cond_pl_sub() {
        // CMP 5, 3 → PL (positive) should be 1 (5-3 = 2, N clear)
        use arm_cc_op::*;
        use arm_cond::*;
        let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_PL, ARMG_CC_OP_SUB), 5, 3, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_arm_cond_gt_sub() {
        // CMP 5, 3 → GT (signed >) should be 1
        use arm_cc_op::*;
        use arm_cond::*;
        let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_GT, ARMG_CC_OP_SUB), 5, 3, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_arm_cond_le_sub() {
        // CMP 3, 5 → LE (signed <=) should be 1
        use arm_cc_op::*;
        use arm_cond::*;
        let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_LE, ARMG_CC_OP_SUB), 3, 5, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_arm_cond_ge_sub_equal() {
        // CMP 5, 5 → GE (signed >=) should be 1
        use arm_cc_op::*;
        use arm_cond::*;
        let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_GE, ARMG_CC_OP_SUB), 5, 5, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_arm_cond_lt_sub_signed() {
        // CMP -1 (0xFFFFFFFF), 1 → LT (signed <) should be 1
        use arm_cc_op::*;
        use arm_cond::*;
        let result =
            armg_calculate_condition(arm_cond_n_op(ARM_COND_LT, ARMG_CC_OP_SUB), 0xFFFFFFFF, 1, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_arm_cond_hi_sub() {
        // CMP 5, 3 → HI (unsigned >) should be 1
        use arm_cc_op::*;
        use arm_cond::*;
        let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_HI, ARMG_CC_OP_SUB), 5, 3, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_arm_cond_ls_sub() {
        // CMP 3, 5 → LS (unsigned <=) should be 1
        use arm_cc_op::*;
        use arm_cond::*;
        let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_LS, ARMG_CC_OP_SUB), 3, 5, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_arm_cond_al() {
        // AL (always) → 1
        use arm_cc_op::*;
        use arm_cond::*;
        let result = armg_calculate_condition(arm_cond_n_op(ARM_COND_AL, ARMG_CC_OP_SUB), 0, 0, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_arm_cond_add_eq() {
        // ADD 5+(-5) = 0, EQ should be 1
        use arm_cc_op::*;
        use arm_cond::*;
        let result = armg_calculate_condition(
            arm_cond_n_op(ARM_COND_EQ, ARMG_CC_OP_ADD),
            5,
            0xFFFFFFFB,
            0, // 5 + (-5) = 0
        );
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_arm_cond_logic_eq() {
        // LOGIC result=0, EQ should be 1
        use arm_cc_op::*;
        use arm_cond::*;
        let result =
            armg_calculate_condition(arm_cond_n_op(ARM_COND_EQ, ARMG_CC_OP_LOGIC), 0, 0, 0);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_arm_flags_nzcv_sub_zero() {
        // SUB 5 - 5 = 0: N=0, Z=1, C=1 (no borrow), V=0
        use arm_cc_op::*;
        let nzcv = armg_calculate_flags_nzcv(ARMG_CC_OP_SUB, 5, 5, 0).unwrap();
        assert_eq!((nzcv >> 31) & 1, 0); // N
        assert_eq!((nzcv >> 30) & 1, 1); // Z
        assert_eq!((nzcv >> 29) & 1, 1); // C (no borrow on ARM means C=1)
        assert_eq!((nzcv >> 28) & 1, 0); // V
    }

    #[test]
    fn test_arm_flags_nzcv_copy() {
        // COPY with NZCV = 0xA0000000 (N=1, Z=0, C=1, V=0)
        use arm_cc_op::*;
        let nzcv = armg_calculate_flags_nzcv(ARMG_CC_OP_COPY, 0xA0000000, 0, 0).unwrap();
        assert_eq!((nzcv >> 31) & 1, 1); // N
        assert_eq!((nzcv >> 30) & 1, 0); // Z
        assert_eq!((nzcv >> 29) & 1, 1); // C
        assert_eq!((nzcv >> 28) & 1, 0); // V
    }

    #[test]
    fn test_arm_handle_ccall_concrete() {
        // Test via handle_ccall dispatch
        use arm_cc_op::*;
        use arm_cond::*;
        let cond_n_op = arm_cond_n_op(ARM_COND_EQ, ARMG_CC_OP_SUB);
        let args = vec![
            RustBV::concrete(cond_n_op as u128, 32),
            RustBV::concrete(5, 32),
            RustBV::concrete(5, 32),
            RustBV::concrete(0, 32),
        ];
        let result = handle_ccall("armg_calculate_condition", &args, 32);
        assert!(result.is_some());
        assert_eq!(result.unwrap().as_u64(), Some(1));
    }
}
