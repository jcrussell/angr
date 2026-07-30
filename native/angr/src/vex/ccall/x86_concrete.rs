//! x86/amd64 concrete flag computation: the `Flags` struct and the
//! u64-arithmetic `calc_flags_*` helpers feeding the dispatcher.
use super::*;

/// Computed flags from an operation
#[derive(Debug, Clone, Copy)]
pub(super) struct Flags {
    pub(super) cf: u8, // Carry flag
    pub(super) pf: u8, // Parity flag
    pub(super) zf: u8, // Zero flag
    pub(super) sf: u8, // Sign flag
    pub(super) of: u8, // Overflow flag
}

/// Calculate parity bit (1 if even parity in low 8 bits)
pub(super) fn calc_parity(val: u64) -> u8 {
    let byte = val as u8;
    // Count 1 bits in the byte, return 1 if even (even parity)
    u8::from(byte.count_ones().is_multiple_of(2))
}

/// Get bitmask for an n-bit value (e.g., nbits=32 -> 0xFFFFFFFF).
#[inline]
pub(super) fn get_mask(nbits: u32) -> u64 {
    if nbits == 64 {
        u64::MAX
    } else {
        (1u64 << nbits) - 1
    }
}

/// Get the sign bit for an n-bit value (e.g., nbits=32 -> 0x80000000).
#[inline]
pub(super) fn get_sign_bit(nbits: u32) -> u64 {
    1u64 << (nbits - 1)
}

/// Sign-extend the low `nbits` of `val` to a full i64.
/// `nbits` must be one of {8, 16, 32, 64}.
#[inline]
pub(super) fn sign_extend_to_i64(val: u64, nbits: u32) -> i64 {
    if nbits == 64 {
        val as i64
    } else {
        let sign_bit = get_sign_bit(nbits);
        let mask = get_mask(nbits);
        if (val & sign_bit) != 0 {
            (val | !mask) as i64
        } else {
            val as i64
        }
    }
}

/// Calculate flags for SUB operation (CMP uses this)
pub(super) fn calc_flags_sub(nbits: u32, arg_l: u64, arg_r: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);

    // Defensively mask operands to the operand width before any comparison, matching
    // calc_flags_adc/calc_flags_sbb and the symbolic sym_flags_sub (which extract_to_nbits
    // both operands). Callers today always pre-mask, but relying on that caller-side
    // invariant silently produced wrong CF/OF for garbage high bits (angr-36vvn.3).
    let arg_l = arg_l & mask;
    let arg_r = arg_r & mask;

    let res = arg_l.wrapping_sub(arg_r) & mask;

    // CF: set if borrow (unsigned: arg_l < arg_r)
    let cf = u8::from(arg_l < arg_r);

    // ZF: set if result is zero
    let zf = u8::from(res == 0);

    // SF: set if result is negative (sign bit set)
    let sf = u8::from((res & sign_bit) != 0);

    // OF: set if signed overflow
    // Overflow occurs if: (arg_l ^ arg_r) & (arg_l ^ res) has sign bit set
    let of = u8::from(((arg_l ^ arg_r) & (arg_l ^ res) & sign_bit) != 0);

    // PF: parity of low 8 bits
    let pf = calc_parity(res);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for ADD operation
pub(super) fn calc_flags_add(nbits: u32, arg_l: u64, arg_r: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);

    // Defensively mask operands to the operand width before any comparison, matching
    // calc_flags_adc/calc_flags_sbb and the symbolic sym_flags_add. See angr-36vvn.3.
    let arg_l = arg_l & mask;
    let arg_r = arg_r & mask;

    let res = arg_l.wrapping_add(arg_r) & mask;

    // CF: set if carry (unsigned overflow: res < arg_l)
    let cf = u8::from(res < arg_l);

    // ZF: set if result is zero
    let zf = u8::from(res == 0);

    // SF: set if result is negative
    let sf = u8::from((res & sign_bit) != 0);

    // OF: set if signed overflow
    // For addition: overflow if both operands have same sign and result has different sign
    // OF = ((arg_l ^ arg_r ^ mask) & (arg_l ^ res)) has sign bit set
    // Simplified: same sign operands, different sign result
    let of = u8::from(((!(arg_l ^ arg_r)) & (arg_l ^ res) & sign_bit) != 0);

    // PF: parity of low 8 bits
    let pf = calc_parity(res);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for LOGIC operation (AND, OR, XOR, TEST)
pub(super) fn calc_flags_logic(nbits: u32, result: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);
    let res = result & mask;

    // CF and OF are always 0 for logic ops
    let cf = 0;
    let of = 0;

    // ZF: set if result is zero
    let zf = u8::from(res == 0);

    // SF: set if result is negative
    let sf = u8::from((res & sign_bit) != 0);

    // PF: parity of low 8 bits
    let pf = calc_parity(res);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for INC operation
pub(super) fn calc_flags_inc(nbits: u32, res: u64, cc_ndep: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);
    let res = res & mask;

    // CF is preserved from cc_ndep
    let cf = ((cc_ndep >> flag_shift::G_CC_SHIFT_C) & 1) as u8;

    // ZF: set if result is zero
    let zf = u8::from(res == 0);

    // SF: set if result is negative
    let sf = u8::from((res & sign_bit) != 0);

    // OF: set if res == 0x80...0 (incremented from 0x7F...F)
    let of = u8::from(res == sign_bit);

    // PF: parity of low 8 bits
    let pf = calc_parity(res);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for DEC operation
pub(super) fn calc_flags_dec(nbits: u32, res: u64, cc_ndep: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);
    let res = res & mask;

    // CF is preserved from cc_ndep
    let cf = ((cc_ndep >> flag_shift::G_CC_SHIFT_C) & 1) as u8;

    // ZF: set if result is zero
    let zf = u8::from(res == 0);

    // SF: set if result is negative
    let sf = u8::from((res & sign_bit) != 0);

    // OF: set if res == 0x7F...F (decremented from 0x80...0)
    let of = u8::from(res == (sign_bit - 1));

    // PF: parity of low 8 bits
    let pf = calc_parity(res);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for SHL (shift left) operation
pub(super) fn calc_flags_shl(nbits: u32, remaining: u64, shifted: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);
    let remaining = remaining & mask;
    let shifted = shifted & mask;

    // CF: last bit shifted out (MSB of shifted value)
    let cf = ((shifted >> (nbits - 1)) & 1) as u8;

    // ZF: set if result is zero
    let zf = u8::from(remaining == 0);

    // SF: set if result is negative
    let sf = u8::from((remaining & sign_bit) != 0);

    // OF: XOR of CF and SF (for shift by 1)
    let of = cf ^ sf;

    // PF: parity of low 8 bits
    let pf = calc_parity(remaining);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for SHR (shift right) operation
pub(super) fn calc_flags_shr(nbits: u32, remaining: u64, shifted: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);
    let remaining = remaining & mask;
    let shifted = shifted & mask;

    // CF: last bit shifted out (LSB of original, i.e., bit 0 of shifted)
    let cf = (shifted & 1) as u8;

    // ZF: set if result is zero
    let zf = u8::from(remaining == 0);

    // SF: set if result is negative
    let sf = u8::from((remaining & sign_bit) != 0);

    // OF: MSB of original value (for shift by 1)
    let of = ((shifted >> (nbits - 1)) ^ (remaining >> (nbits - 1))) as u8 & 1;

    // PF: parity of low 8 bits
    let pf = calc_parity(remaining);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for ROL (rotate left) operation
pub(super) fn calc_flags_rol(nbits: u32, res: u64, cc_ndep: u64) -> Flags {
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
pub(super) fn calc_flags_ror(nbits: u32, res: u64, cc_ndep: u64) -> Flags {
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
pub(super) fn calc_flags_adc(nbits: u32, cc_dep1: u64, cc_dep2: u64, cc_ndep: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);

    let old_c = (cc_ndep >> flag_shift::G_CC_SHIFT_C) & 1;
    let arg_l = cc_dep1 & mask;
    let arg_r = (cc_dep2 ^ old_c) & mask;
    let res = (arg_l.wrapping_add(arg_r).wrapping_add(old_c)) & mask;

    // CF: carry out
    let cf = if old_c != 0 {
        u8::from(res <= arg_l)
    } else {
        u8::from(res < arg_l)
    };

    // ZF: set if result is zero
    let zf = u8::from(res == 0);

    // SF: set if result is negative
    let sf = u8::from((res & sign_bit) != 0);

    // OF: signed overflow
    let of = u8::from(((!(arg_l ^ arg_r)) & (arg_l ^ res) & sign_bit) != 0);

    // PF: parity of low 8 bits
    let pf = calc_parity(res);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for SBB (subtract with borrow) operation
pub(super) fn calc_flags_sbb(nbits: u32, cc_dep1: u64, cc_dep2: u64, cc_ndep: u64) -> Flags {
    let mask = get_mask(nbits);
    let sign_bit = get_sign_bit(nbits);

    let old_c = (cc_ndep >> flag_shift::G_CC_SHIFT_C) & 1;
    let arg_l = cc_dep1 & mask;
    let arg_r = (cc_dep2 ^ old_c) & mask;
    let res = (arg_l.wrapping_sub(arg_r).wrapping_sub(old_c)) & mask;

    // CF: borrow out
    let cf = if old_c != 0 {
        u8::from(arg_l <= arg_r)
    } else {
        u8::from(arg_l < arg_r)
    };

    // ZF: set if result is zero
    let zf = u8::from(res == 0);

    // SF: set if result is negative
    let sf = u8::from((res & sign_bit) != 0);

    // OF: signed overflow
    let of = u8::from(((arg_l ^ arg_r) & (arg_l ^ res) & sign_bit) != 0);

    // PF: parity of low 8 bits
    let pf = calc_parity(res);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for UMUL (unsigned multiply) operation
pub(super) fn calc_flags_umul(nbits: u32, cc_dep1: u64, cc_dep2: u64) -> Flags {
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
    let cf = u8::from(hi != 0);
    let of = cf;

    // ZF, SF, PF are undefined but we compute them anyway
    let zf = u8::from(lo == 0);
    let sf = ((lo >> (nbits - 1)) & 1) as u8;
    let pf = calc_parity(lo);

    Flags { cf, pf, zf, sf, of }
}

/// Calculate flags for SMUL (signed multiply) operation
pub(super) fn calc_flags_smul(nbits: u32, cc_dep1: u64, cc_dep2: u64) -> Flags {
    let mask = get_mask(nbits);

    // Sign-extend operands
    let arg1_signed = sign_extend_to_i64(cc_dep1, nbits);
    let arg2_signed = sign_extend_to_i64(cc_dep2, nbits);

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
    let cf = u8::from(hi != lo_sign_ext);
    let of = cf;

    // ZF, SF, PF
    let zf = u8::from(lo == 0);
    let sf = ((lo >> (nbits - 1)) & 1) as u8;
    let pf = calc_parity(lo);

    Flags { cf, pf, zf, sf, of }
}
