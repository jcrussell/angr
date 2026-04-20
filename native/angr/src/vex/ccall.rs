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
    pub const COND_O: u64 = 0;   // Overflow
    pub const COND_NO: u64 = 1;  // Not overflow
    pub const COND_B: u64 = 2;   // Below (CF=1)
    pub const COND_NB: u64 = 3;  // Not below (CF=0)
    pub const COND_Z: u64 = 4;   // Zero (ZF=1)
    pub const COND_NZ: u64 = 5;  // Not zero (ZF=0)
    pub const COND_BE: u64 = 6;  // Below or equal (CF=1 or ZF=1)
    pub const COND_NBE: u64 = 7; // Not below or equal (CF=0 and ZF=0)
    pub const COND_S: u64 = 8;   // Sign (SF=1)
    pub const COND_NS: u64 = 9;  // Not sign (SF=0)
    pub const COND_P: u64 = 10;  // Parity even (PF=1)
    pub const COND_NP: u64 = 11; // Parity odd (PF=0)
    pub const COND_L: u64 = 12;  // Less (SF != OF)
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
    let xor_all = b0.xor(&b1, ctx).xor(&b2, ctx).xor(&b3, ctx)
        .xor(&b4, ctx).xor(&b5, ctx).xor(&b6, ctx).xor(&b7, ctx);
    // PF=1 means even parity (even number of set bits), so NOT the XOR
    xor_all.not(ctx)
}

/// Pack individual 1-bit flags into EFLAGS format bitvector.
/// Bit positions: OF@11, SF@7, ZF@6, PF@2, CF@0
fn symbolic_pack_eflags(
    of: &RustBV, sf: &RustBV, zf: &RustBV, pf: &RustBV, cf: &RustBV,
    ret_bits: u32, ctx: &SymContext,
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

    of_ext.shl(&shift_11, ctx)
        .or(&sf_ext.shl(&shift_7, ctx), ctx)
        .or(&zf_ext.shl(&shift_6, ctx), ctx)
        .or(&pf_ext.shl(&shift_2, ctx), ctx)
        .or(&cf_ext, ctx)
}

/// Symbolic eflags computation for SUB/CMP: flags from dep1 - dep2
fn symbolic_eflags_sub(nbits: u32, dep1: &RustBV, dep2: &RustBV, ctx: &SymContext, ret_bits: u32) -> RustBV {
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
    let of = d1.xor(&d2, ctx).and(&d1.xor(&result, ctx), ctx).extract(nbits - 1, nbits - 1, ctx);
    // PF = parity of low byte of result
    let pf = symbolic_parity(&result, ctx);

    symbolic_pack_eflags(&of, &sf, &zf, &pf, &cf, ret_bits, ctx)
}

/// Symbolic eflags computation for ADD: flags from dep1 + dep2
fn symbolic_eflags_add(nbits: u32, dep1: &RustBV, dep2: &RustBV, ctx: &SymContext, ret_bits: u32) -> RustBV {
    let d1 = extract_to_nbits(dep1, nbits, ctx);
    let d2 = extract_to_nbits(dep2, nbits, ctx);
    let result = d1.add(&d2, ctx);
    let zero = RustBV::concrete(0, nbits);

    let zf = result.eq(&zero, ctx);
    let sf = result.extract(nbits - 1, nbits - 1, ctx);
    // CF = (result < dep1) unsigned — carry out
    let cf = result.ult(&d1, ctx);
    // OF = (~(dep1 ^ dep2) & (dep1 ^ result))[msb] — same sign operands, different sign result
    let of = d1.xor(&d2, ctx).not(ctx).and(&d1.xor(&result, ctx), ctx).extract(nbits - 1, nbits - 1, ctx);
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
    if nbits == 64 { u64::MAX } else { (1u64 << nbits) - 1 }
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
    let of = if ((arg_l ^ arg_r) & (arg_l ^ res) & sign_bit) != 0 { 1 } else { 0 };

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
    let of = if ((!(arg_l ^ arg_r)) & (arg_l ^ res) & sign_bit) != 0 { 1 } else { 0 };

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
    let of = if ((!(arg_l ^ arg_r)) & (arg_l ^ res) & sign_bit) != 0 { 1 } else { 0 };

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
    let of = if ((arg_l ^ arg_r) & (arg_l ^ res) & sign_bit) != 0 { 1 } else { 0 };

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

/// Get the operand size in bits for an AMD64 CC_OP
fn amd64_op_to_nbits(cc_op: u64) -> Option<u32> {
    use amd64_cc_op::*;
    match cc_op {
        G_CC_OP_ADDB | G_CC_OP_SUBB | G_CC_OP_ADCB | G_CC_OP_SBBB | G_CC_OP_LOGICB |
        G_CC_OP_INCB | G_CC_OP_DECB | G_CC_OP_SHLB | G_CC_OP_SHRB | G_CC_OP_ROLB |
        G_CC_OP_RORB | G_CC_OP_UMULB | G_CC_OP_SMULB => Some(8),

        G_CC_OP_ADDW | G_CC_OP_SUBW | G_CC_OP_ADCW | G_CC_OP_SBBW | G_CC_OP_LOGICW |
        G_CC_OP_INCW | G_CC_OP_DECW | G_CC_OP_SHLW | G_CC_OP_SHRW | G_CC_OP_ROLW |
        G_CC_OP_RORW | G_CC_OP_UMULW | G_CC_OP_SMULW => Some(16),

        G_CC_OP_ADDL | G_CC_OP_SUBL | G_CC_OP_ADCL | G_CC_OP_SBBL | G_CC_OP_LOGICL |
        G_CC_OP_INCL | G_CC_OP_DECL | G_CC_OP_SHLL | G_CC_OP_SHRL | G_CC_OP_ROLL |
        G_CC_OP_RORL | G_CC_OP_UMULL | G_CC_OP_SMULL => Some(32),

        G_CC_OP_ADDQ | G_CC_OP_SUBQ | G_CC_OP_ADCQ | G_CC_OP_SBBQ | G_CC_OP_LOGICQ |
        G_CC_OP_INCQ | G_CC_OP_DECQ | G_CC_OP_SHLQ | G_CC_OP_SHRQ | G_CC_OP_ROLQ |
        G_CC_OP_RORQ | G_CC_OP_UMULQ | G_CC_OP_SMULQ => Some(64),

        G_CC_OP_COPY => Some(64), // COPY uses native size

        _ => None,
    }
}

/// Get the operand size in bits for an X86 CC_OP
fn x86_op_to_nbits(cc_op: u64) -> Option<u32> {
    use x86_cc_op::*;
    match cc_op {
        G_CC_OP_ADDB | G_CC_OP_SUBB | G_CC_OP_ADCB | G_CC_OP_SBBB | G_CC_OP_LOGICB |
        G_CC_OP_INCB | G_CC_OP_DECB | G_CC_OP_SHLB | G_CC_OP_SHRB | G_CC_OP_ROLB |
        G_CC_OP_RORB | G_CC_OP_UMULB | G_CC_OP_SMULB => Some(8),

        G_CC_OP_ADDW | G_CC_OP_SUBW | G_CC_OP_ADCW | G_CC_OP_SBBW | G_CC_OP_LOGICW |
        G_CC_OP_INCW | G_CC_OP_DECW | G_CC_OP_SHLW | G_CC_OP_SHRW | G_CC_OP_ROLW |
        G_CC_OP_RORW | G_CC_OP_UMULW | G_CC_OP_SMULW => Some(16),

        G_CC_OP_ADDL | G_CC_OP_SUBL | G_CC_OP_ADCL | G_CC_OP_SBBL | G_CC_OP_LOGICL |
        G_CC_OP_INCL | G_CC_OP_DECL | G_CC_OP_SHLL | G_CC_OP_SHRL | G_CC_OP_ROLL |
        G_CC_OP_RORL | G_CC_OP_UMULL | G_CC_OP_SMULL => Some(32),

        G_CC_OP_COPY => Some(32), // COPY uses native size

        _ => None,
    }
}

/// Get the operation category for an AMD64 CC_OP
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

fn amd64_op_to_category(cc_op: u64) -> Option<OpCategory> {
    use amd64_cc_op::*;
    match cc_op {
        G_CC_OP_COPY => Some(OpCategory::Copy),
        G_CC_OP_ADDB | G_CC_OP_ADDW | G_CC_OP_ADDL | G_CC_OP_ADDQ => Some(OpCategory::Add),
        G_CC_OP_SUBB | G_CC_OP_SUBW | G_CC_OP_SUBL | G_CC_OP_SUBQ => Some(OpCategory::Sub),
        G_CC_OP_ADCB | G_CC_OP_ADCW | G_CC_OP_ADCL | G_CC_OP_ADCQ => Some(OpCategory::Adc),
        G_CC_OP_SBBB | G_CC_OP_SBBW | G_CC_OP_SBBL | G_CC_OP_SBBQ => Some(OpCategory::Sbb),
        G_CC_OP_LOGICB | G_CC_OP_LOGICW | G_CC_OP_LOGICL | G_CC_OP_LOGICQ => Some(OpCategory::Logic),
        G_CC_OP_INCB | G_CC_OP_INCW | G_CC_OP_INCL | G_CC_OP_INCQ => Some(OpCategory::Inc),
        G_CC_OP_DECB | G_CC_OP_DECW | G_CC_OP_DECL | G_CC_OP_DECQ => Some(OpCategory::Dec),
        G_CC_OP_SHLB | G_CC_OP_SHLW | G_CC_OP_SHLL | G_CC_OP_SHLQ => Some(OpCategory::Shl),
        G_CC_OP_SHRB | G_CC_OP_SHRW | G_CC_OP_SHRL | G_CC_OP_SHRQ => Some(OpCategory::Shr),
        G_CC_OP_ROLB | G_CC_OP_ROLW | G_CC_OP_ROLL | G_CC_OP_ROLQ => Some(OpCategory::Rol),
        G_CC_OP_RORB | G_CC_OP_RORW | G_CC_OP_RORL | G_CC_OP_RORQ => Some(OpCategory::Ror),
        G_CC_OP_UMULB | G_CC_OP_UMULW | G_CC_OP_UMULL | G_CC_OP_UMULQ => Some(OpCategory::Umul),
        G_CC_OP_SMULB | G_CC_OP_SMULW | G_CC_OP_SMULL | G_CC_OP_SMULQ => Some(OpCategory::Smul),
        _ => None,
    }
}

fn x86_op_to_category(cc_op: u64) -> Option<OpCategory> {
    use x86_cc_op::*;
    match cc_op {
        G_CC_OP_COPY => Some(OpCategory::Copy),
        G_CC_OP_ADDB | G_CC_OP_ADDW | G_CC_OP_ADDL => Some(OpCategory::Add),
        G_CC_OP_SUBB | G_CC_OP_SUBW | G_CC_OP_SUBL => Some(OpCategory::Sub),
        G_CC_OP_ADCB | G_CC_OP_ADCW | G_CC_OP_ADCL => Some(OpCategory::Adc),
        G_CC_OP_SBBB | G_CC_OP_SBBW | G_CC_OP_SBBL => Some(OpCategory::Sbb),
        G_CC_OP_LOGICB | G_CC_OP_LOGICW | G_CC_OP_LOGICL => Some(OpCategory::Logic),
        G_CC_OP_INCB | G_CC_OP_INCW | G_CC_OP_INCL => Some(OpCategory::Inc),
        G_CC_OP_DECB | G_CC_OP_DECW | G_CC_OP_DECL => Some(OpCategory::Dec),
        G_CC_OP_SHLB | G_CC_OP_SHLW | G_CC_OP_SHLL => Some(OpCategory::Shl),
        G_CC_OP_SHRB | G_CC_OP_SHRW | G_CC_OP_SHRL => Some(OpCategory::Shr),
        G_CC_OP_ROLB | G_CC_OP_ROLW | G_CC_OP_ROLL => Some(OpCategory::Rol),
        G_CC_OP_RORB | G_CC_OP_RORW | G_CC_OP_RORL => Some(OpCategory::Ror),
        G_CC_OP_UMULB | G_CC_OP_UMULW | G_CC_OP_UMULL => Some(OpCategory::Umul),
        G_CC_OP_SMULB | G_CC_OP_SMULW | G_CC_OP_SMULL => Some(OpCategory::Smul),
        _ => None,
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
    let nbits = amd64_op_to_nbits(cc_op)?;
    let category = amd64_op_to_category(cc_op)?;

    let flags = match category {
        OpCategory::Copy => return Some(eval_condition_from_copy(cond, cc_dep1)),
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
    };

    Some(eval_condition(cond, &flags))
}

/// Calculate condition for X86 architecture.
pub fn x86g_calculate_condition(
    cond: u64,
    cc_op: u64,
    cc_dep1: u64,
    cc_dep2: u64,
    cc_ndep: u64,
) -> Option<u64> {
    let nbits = x86_op_to_nbits(cc_op)?;
    let category = x86_op_to_category(cc_op)?;

    let flags = match category {
        OpCategory::Copy => return Some(eval_condition_from_copy(cond, cc_dep1)),
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
    };

    Some(eval_condition(cond, &flags))
}

/// Calculate the carry flag (CF) for the given cc_op.
fn calculate_eflags_c_amd64(cc_op: u64, cc_dep1: u64, cc_dep2: u64, cc_ndep: u64) -> Option<u64> {
    let nbits = amd64_op_to_nbits(cc_op)?;
    let category = amd64_op_to_category(cc_op)?;

    let flags = match category {
        OpCategory::Copy => {
            return Some((cc_dep1 >> flag_shift::G_CC_SHIFT_C) & 1);
        }
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
    };

    Some(flags.cf as u64)
}

fn calculate_eflags_c_x86(cc_op: u64, cc_dep1: u64, cc_dep2: u64, cc_ndep: u64) -> Option<u64> {
    let nbits = x86_op_to_nbits(cc_op)?;
    let category = x86_op_to_category(cc_op)?;

    let flags = match category {
        OpCategory::Copy => {
            return Some((cc_dep1 >> flag_shift::G_CC_SHIFT_C) & 1);
        }
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
    };

    Some(flags.cf as u64)
}

/// Pack flags into the standard EFLAGS format.
fn pack_eflags(flags: &Flags) -> u64 {
    ((flags.of as u64) << flag_shift::G_CC_SHIFT_O) |
    ((flags.sf as u64) << flag_shift::G_CC_SHIFT_S) |
    ((flags.zf as u64) << flag_shift::G_CC_SHIFT_Z) |
    ((flags.pf as u64) << flag_shift::G_CC_SHIFT_P) |
    ((flags.cf as u64) << flag_shift::G_CC_SHIFT_C)
}

/// Calculate all eflags for AMD64.
fn calculate_eflags_all_amd64(cc_op: u64, cc_dep1: u64, cc_dep2: u64, cc_ndep: u64) -> Option<u64> {
    let nbits = amd64_op_to_nbits(cc_op)?;
    let category = amd64_op_to_category(cc_op)?;

    if category == OpCategory::Copy {
        // For COPY, cc_dep1 already contains the flags
        return Some(cc_dep1 & (flag_mask::G_CC_MASK_O | flag_mask::G_CC_MASK_S |
                               flag_mask::G_CC_MASK_Z | flag_mask::G_CC_MASK_P |
                               flag_mask::G_CC_MASK_C | flag_mask::G_CC_MASK_A));
    }

    let flags = match category {
        OpCategory::Copy => unreachable!(),
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
    };

    Some(pack_eflags(&flags))
}

/// Calculate all eflags for X86.
fn calculate_eflags_all_x86(cc_op: u64, cc_dep1: u64, cc_dep2: u64, cc_ndep: u64) -> Option<u64> {
    let nbits = x86_op_to_nbits(cc_op)?;
    let category = x86_op_to_category(cc_op)?;

    if category == OpCategory::Copy {
        return Some(cc_dep1 & (flag_mask::G_CC_MASK_O | flag_mask::G_CC_MASK_S |
                               flag_mask::G_CC_MASK_Z | flag_mask::G_CC_MASK_P |
                               flag_mask::G_CC_MASK_C | flag_mask::G_CC_MASK_A));
    }

    let flags = match category {
        OpCategory::Copy => unreachable!(),
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
    };

    Some(pack_eflags(&flags))
}

/// Handle a CCall expression.
///
/// Returns Some(result) if the call was handled, None if not supported.
pub fn handle_ccall(
    name: &str,
    args: &[RustBV],
    ret_bits: u32,
) -> Option<RustBV> {
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
            args[0].as_u64(), args[1].as_u64(), args[2].as_u64(),
            args[3].as_u64(), args[4].as_u64(),
        ) {
            let result = if name == "amd64g_calculate_condition" {
                amd64g_calculate_condition(cond, cc_op, cc_dep1, cc_dep2, cc_ndep)?
            } else {
                x86g_calculate_condition(cond, cc_op, cc_dep1, cc_dep2, cc_ndep)?
            };
            return Some(RustBV::concrete(result as u128, ret_bits));
        }

        // Symbolic path: handle SUB/LOGIC with symbolic deps
        // This enables symbolic branch detection for comparisons
        if let (Some(cond), Some(cc_op), Some(sym_ctx)) = (args[0].as_u64(), args[1].as_u64(), ctx) {
            let dep1 = &args[2];
            let dep2 = &args[3];
            let category = if name == "amd64g_calculate_condition" {
                amd64_op_to_category(cc_op)
            } else {
                x86_op_to_category(cc_op)
            };

            if let Some(cat) = category {
                match cat {
                    OpCategory::Sub => {
                        // For SUB: must extract to nbits first (64-bit temps for 8/16/32-bit ops)
                        use cond_type::*;
                        let inv = (cond & 1) != 0;
                        let nbits = if name == "amd64g_calculate_condition" {
                            amd64_op_to_nbits(cc_op)
                        } else {
                            x86_op_to_nbits(cc_op)
                        };
                        if let Some(nb) = nbits {
                            let d1 = extract_to_nbits(dep1, nb, sym_ctx);
                            let d2 = extract_to_nbits(dep2, nb, sym_ctx);
                            match cond & !1 {
                                COND_Z => {
                                    let eq = d1.eq(&d2, sym_ctx);
                                    let r = if inv { eq.not(sym_ctx) } else { eq };
                                    return Some(r.zero_extend(ret_bits, sym_ctx));
                                }
                                COND_B => {
                                    let lt = d1.ult(&d2, sym_ctx);
                                    let r = if inv { lt.not(sym_ctx) } else { lt };
                                    return Some(r.zero_extend(ret_bits, sym_ctx));
                                }
                                COND_BE => {
                                    let le = d1.ule(&d2, sym_ctx);
                                    let r = if inv { le.not(sym_ctx) } else { le };
                                    return Some(r.zero_extend(ret_bits, sym_ctx));
                                }
                                COND_L => {
                                    let lt = d1.slt(&d2, sym_ctx);
                                    let r = if inv { lt.not(sym_ctx) } else { lt };
                                    return Some(r.zero_extend(ret_bits, sym_ctx));
                                }
                                COND_LE => {
                                    let le = d1.sle(&d2, sym_ctx);
                                    let r = if inv { le.not(sym_ctx) } else { le };
                                    return Some(r.zero_extend(ret_bits, sym_ctx));
                                }
                                _ => {}
                            }
                        }
                    }
                    OpCategory::Logic => {
                        use cond_type::*;
                        let inv = (cond & 1) != 0;
                        let nbits = if name == "amd64g_calculate_condition" {
                            amd64_op_to_nbits(cc_op)
                        } else {
                            x86_op_to_nbits(cc_op)
                        };
                        if let Some(nb) = nbits {
                            let d1 = extract_to_nbits(dep1, nb, sym_ctx);
                            match cond & !1 {
                                COND_Z => {
                                    let zero = RustBV::concrete(0, nb);
                                    let eq = d1.eq(&zero, sym_ctx);
                                    let r = if inv { eq.not(sym_ctx) } else { eq };
                                    return Some(r.zero_extend(ret_bits, sym_ctx));
                                }
                                COND_S => {
                                    let sf = d1.extract(nb - 1, nb - 1, sym_ctx);
                                    let r = if inv { sf.not(sym_ctx) } else { sf };
                                    return Some(r.zero_extend(ret_bits, sym_ctx));
                                }
                                _ => {}
                            }
                        }
                    }
                    OpCategory::Add => {
                        use cond_type::*;
                        let inv = (cond & 1) != 0;
                        let nbits = if name == "amd64g_calculate_condition" {
                            amd64_op_to_nbits(cc_op)
                        } else {
                            x86_op_to_nbits(cc_op)
                        };
                        if let Some(nb) = nbits {
                            let d1 = extract_to_nbits(dep1, nb, sym_ctx);
                            let d2 = extract_to_nbits(dep2, nb, sym_ctx);
                            let result = d1.add(&d2, sym_ctx);
                            match cond & !1 {
                                COND_Z => {
                                    let zero = RustBV::concrete(0, nb);
                                    let eq = result.eq(&zero, sym_ctx);
                                    let r = if inv { eq.not(sym_ctx) } else { eq };
                                    return Some(r.zero_extend(ret_bits, sym_ctx));
                                }
                                COND_B => {
                                    // CF = result < dep1 (unsigned overflow)
                                    let cf = result.ult(&d1, sym_ctx);
                                    let r = if inv { cf.not(sym_ctx) } else { cf };
                                    return Some(r.zero_extend(ret_bits, sym_ctx));
                                }
                                COND_S => {
                                    let sf = result.extract(nb - 1, nb - 1, sym_ctx);
                                    let r = if inv { sf.not(sym_ctx) } else { sf };
                                    return Some(r.zero_extend(ret_bits, sym_ctx));
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {} // Other categories: fall through to None
                }
            }
        }

        return None;
    }

    // Check for eflags_c / rflags_c CCall.
    // Handle both "eflags" and "rflags" naming variants.
    if name == "amd64g_calculate_eflags_c" || name == "amd64g_calculate_rflags_c"
        || name == "x86g_calculate_eflags_c" || name == "x86g_calculate_rflags_c" {
        // Args: cc_op, cc_dep1, cc_dep2, cc_ndep
        if args.len() < 4 {
            return None;
        }

        // Try concrete path first
        if let (Some(cc_op), Some(cc_dep1), Some(cc_dep2), Some(cc_ndep)) = (
            args[0].as_u64(), args[1].as_u64(), args[2].as_u64(), args[3].as_u64(),
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
            let is_amd64 = name.starts_with("amd64g");
            let category = if is_amd64 { amd64_op_to_category(cc_op) } else { x86_op_to_category(cc_op) };
            let nbits = if is_amd64 { amd64_op_to_nbits(cc_op) } else { x86_op_to_nbits(cc_op) };
            if let (Some(cat), Some(nb)) = (category, nbits) {
                let cf = match cat {
                    OpCategory::Copy => {
                        // CF = (dep1 >> SHIFT_C) & 1
                        let shift = RustBV::concrete(flag_shift::G_CC_SHIFT_C as u128, args[1].width());
                        let one = RustBV::concrete(1, args[1].width());
                        Some(args[1].lshr(&shift, sym_ctx).and(&one, sym_ctx).extract(0, 0, sym_ctx))
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
    if name == "amd64g_calculate_eflags_all" || name == "amd64g_calculate_rflags_all"
        || name == "x86g_calculate_eflags_all" || name == "x86g_calculate_rflags_all" {
        // Args: cc_op, cc_dep1, cc_dep2, cc_ndep
        if args.len() < 4 {
            return None;
        }

        // Try concrete path first
        if let (Some(cc_op), Some(cc_dep1), Some(cc_dep2), Some(cc_ndep)) = (
            args[0].as_u64(), args[1].as_u64(), args[2].as_u64(), args[3].as_u64(),
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
            let is_amd64 = name.starts_with("amd64g");
            let category = if is_amd64 { amd64_op_to_category(cc_op) } else { x86_op_to_category(cc_op) };
            let nbits = if is_amd64 { amd64_op_to_nbits(cc_op) } else { x86_op_to_nbits(cc_op) };
            if let (Some(cat), Some(nb)) = (category, nbits) {
                let result = match cat {
                    OpCategory::Sub => Some(symbolic_eflags_sub(nb, &args[1], &args[2], sym_ctx, ret_bits)),
                    OpCategory::Add => Some(symbolic_eflags_add(nb, &args[1], &args[2], sym_ctx, ret_bits)),
                    OpCategory::Logic => Some(symbolic_eflags_logic(nb, &args[1], sym_ctx, ret_bits)),
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
        use x86_cc_op::G_CC_OP_SUBL;
        use cond_type::COND_Z;

        let result = x86g_calculate_condition(COND_Z, G_CC_OP_SUBL, 5, 5, 0);
        assert_eq!(result, Some(1));
    }
}
