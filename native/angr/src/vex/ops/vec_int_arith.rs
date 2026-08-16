//! NEON/SSE packed-integer arithmetic VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via a plain `mod` decl in
//! `ops/mod.rs`), so these `pub(super)` methods stay callable from the unop/binop
//! dispatch in `ops`, and the shared sibling they reference
//! (`Self::concat_le_elements`, which stays in `ops/mod.rs`) stays visible via the
//! descendant rule.
//!
//! Covers GF(2) carry-less polynomial multiply (Iop_PolynomialMul/Mull8x{8,16}).
//! The per-lane signed/unsigned min-max (Iop_Min/Max{U,S}{N}x{M}) and packed
//! absolute value (Iop_Abs{N}x{M} / PABS*) families moved to the generic
//! `vec_int_lane_op` driver (`IMinMax`/`IAbs` in `ops/mod.rs`).

use super::{OpError, VEXOps};
use crate::symbolic::{RustBV, SymContext};

impl VEXOps {
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
        let out_total = out_elem * count as u32;

        // Concrete fast path (fits in u128 — `out_total <= 128` covers all
        // currently-mapped PolynomialMul shapes, and subsumes the input check
        // since `out_elem >= 8` makes `out_total >= in_total`). `RustBV::Concrete`
        // holds its value in a u128, so a wider declared width could not
        // round-trip through `as_u128`; fall through to the symbolic path.
        if out_total <= 128
            && let (Some(a_all), Some(b_all)) = (left.as_u128(), right.as_u128())
        {
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
}
