//! NEON/SSE packed-integer arithmetic VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via `#[path]` in
//! ops.rs), so these `pub(super)` methods stay callable from the unop/binop
//! dispatch in `ops`, and the shared sibling they reference
//! (`Self::concat_le_elements`, which stays in ops.rs) stays visible via the
//! descendant rule.
//!
//! Covers per-lane signed/unsigned min-max (Iop_Min/Max{U,S}{N}x{M}), GF(2)
//! carry-less polynomial multiply (Iop_PolynomialMul/Mull8x{8,16}) and packed
//! absolute value (Iop_Abs{N}x{M} / PABS*).

use super::{OpError, VEXOps};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ir::IRType;

impl VEXOps {
    /// Packed integer per-lane min or max. Handles signed/unsigned via the
    /// `signed` flag and min/max via `is_max`.
    pub(super) fn vec_int_minmax(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        signed: bool,
        is_max: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);

        // Concrete fast path (only when total fits in u128).
        if total_width <= 128
            && let (Some(l), Some(r)) = (left.as_u128(), right.as_u128())
        {
            let mut result: u128 = 0;
            let elem_mask: u128 = Self::low_bit_mask_u128(elem_width);

            for i in 0..count {
                let lo = (i as u32) * elem_width;
                let l_elem = (l >> lo) & elem_mask;
                let r_elem = (r >> lo) & elem_mask;

                let pick_left = if signed {
                    // Sign-extend each lane to i128 for comparison.
                    let l_signed = Self::sign_extend_low_to_i128(l_elem, elem_width);
                    let r_signed = Self::sign_extend_low_to_i128(r_elem, elem_width);
                    if is_max {
                        l_signed >= r_signed
                    } else {
                        l_signed <= r_signed
                    }
                } else {
                    if is_max {
                        l_elem >= r_elem
                    } else {
                        l_elem <= r_elem
                    }
                };

                let chosen = if pick_left { l_elem } else { r_elem };
                result |= (chosen & elem_mask) << lo;
            }
            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic per-lane fallback.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let l_elem = left.extract(hi, lo, ctx);
            let r_elem = right.extract(hi, lo, ctx);

            // For max: ITE(l >= r, l, r); for min: ITE(l <= r, l, r).
            let cond = match (signed, is_max) {
                (true, true) => l_elem.clone().sge_into(r_elem.clone(), ctx),
                (true, false) => l_elem.clone().sle_into(r_elem.clone(), ctx),
                (false, true) => l_elem.clone().uge_into(r_elem.clone(), ctx),
                (false, false) => l_elem.clone().ule_into(r_elem.clone(), ctx),
            };
            elements.push(cond.ite_into(l_elem, r_elem, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON GF(2) polynomial multiply — `Iop_PolynomialMul8x{8,16}` (non-
    /// widening) and `Iop_PolynomialMull8x8` (widening). Per-lane carry-less
    /// multiply over GF(2): the product is the XOR of shifted copies of `b`
    /// selected by the bits of `a`. The non-widening result keeps the low
    /// 8 bits; the widening result keeps all 16.
    pub(super) fn vec_polynomial_mul(
        left: RustBV,
        right: RustBV,
        count: u8,
        widen: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let in_total = 8u32 * count as u32;
        debug_assert_eq!(left.width(), in_total);
        debug_assert_eq!(right.width(), in_total);
        let out_elem: u32 = if widen { 16 } else { 8 };

        // Concrete fast path.
        if let (Some(a_all), Some(b_all)) = (left.as_u128(), right.as_u128()) {
            let mut result: u128 = 0;
            for i in 0..count as u32 {
                let a = ((a_all >> (i * 8)) as u8) as u16;
                let b = ((b_all >> (i * 8)) as u8) as u16;
                // Carry-less mul into 16-bit accumulator.
                let mut prod: u16 = 0;
                for bit in 0..8 {
                    if (a >> bit) & 1 != 0 {
                        prod ^= b << bit;
                    }
                }
                let lane_val = if widen {
                    prod as u128
                } else {
                    (prod & 0xFF) as u128
                };
                result |= lane_val << (i * out_elem);
            }
            let out_total = out_elem * count as u32;
            return Ok(RustBV::concrete(result, out_total));
        }

        // Symbolic: build per-lane polynomial mul as XOR of conditional shifts
        // of `b`. Work in 16 bits to capture the full product; truncate to 8
        // for the non-widening case.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        let one = RustBV::concrete(1, 1);
        let zero16 = RustBV::concrete(0, 16);
        for i in 0..count as u32 {
            let lo = i * 8;
            let hi = lo + 7;
            let a_lane = left.extract(hi, lo, ctx);
            let b_lane = right.extract(hi, lo, ctx);
            // Widen b to 16 bits for shifting (max shift is 7).
            let b_wide = b_lane.zero_extend_into(16, ctx);
            let mut acc = zero16.clone();
            for bit in 0..8u32 {
                let bit_a = a_lane.extract(bit, bit, ctx);
                let cond = bit_a.eq_into(one.clone(), ctx);
                let shifted = b_wide
                    .clone()
                    .shl_into(RustBV::concrete(bit as u128, 16), ctx);
                let addend = cond.ite_into(shifted, zero16.clone(), ctx);
                acc = acc.xor_into(addend, ctx);
            }
            let lane_out = if widen { acc } else { acc.extract(7, 0, ctx) };
            elements.push(lane_out);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Packed integer per-lane absolute value. Returns the unsigned
    /// representation of `|signed_lane|`. INT_MIN stays INT_MIN (matches the
    /// PABS* hardware behavior).
    pub(super) fn vec_int_abs(
        arg: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(arg.width(), total_width);

        // Concrete fast path.
        if total_width <= 128
            && let Some(v) = arg.as_u128()
        {
            let mut result: u128 = 0;
            let elem_mask: u128 = Self::low_bit_mask_u128(elem_width);
            let sign_bit: u128 = 1u128 << (elem_width - 1);

            for i in 0..count {
                let lo = (i as u32) * elem_width;
                let elem_val = (v >> lo) & elem_mask;
                // |x| = (x ^ -1) + 1 when x is negative (two's complement),
                // otherwise x. Simulated within elem_width bits.
                let abs_val = if elem_val & sign_bit != 0 {
                    // -x in elem_width bits = (~x + 1) & mask
                    ((!elem_val).wrapping_add(1)) & elem_mask
                } else {
                    elem_val
                };
                result |= abs_val << lo;
            }
            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic per-lane fallback: ITE(elem < 0, -elem, elem).
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        let zero = RustBV::concrete(0, elem_width);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let elem_val = arg.extract(hi, lo, ctx);
            let neg = elem_val.clone().neg_into(ctx);
            let is_neg = elem_val.clone().slt_into(zero.clone(), ctx);
            elements.push(is_neg.ite_into(neg, elem_val, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }
}
