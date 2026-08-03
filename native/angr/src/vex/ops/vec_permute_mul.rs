//! Vector sub-unit reversal VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via a plain `mod` decl in
//! `ops/mod.rs`), so these `pub(super)` methods stay callable from the unop/binop
//! dispatch in `ops`, and the shared siblings they reference
//! (`Self::concat_le_elements`, which stays in `ops/mod.rs`) stay visible via
//! super/the descendant rule.
//!
//! Covers the ARM REV*/RBIT sub-unit reversal family
//! (Iop_Reverse{n}sIn{m}_x{k}).

use super::{OpError, VEXOps};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ir::IRType;

impl VEXOps {
    /// NEON byte/halfword/word/bit reversal within each lane —
    /// `Iop_Reverse{sub_width}sIn{elem.bits()}_x{count}`. Reverses the
    /// `elem.bits() / sub_width` sub-units of width `sub_width` inside each
    /// `elem`-wide lane (`count` lanes total). Total width is preserved at
    /// `elem.bits() * count`.
    ///
    /// Encodes the ARM AArch64 REV*/RBIT semantics that pyvex lifts from
    /// VRBIT (sub_width=1), VREV16 (8-in-16), VREV32 (8/16-in-32), and
    /// VREV64 (8/16/32-in-64) — see ARM DDI 0487 C7.2.297-300 (REV*) and
    /// C7.2.288 (RBIT). Matches angr Python's only explicit reference,
    /// `_op_Iop_Reverse32sIn64_x2` in
    /// `angr/engines/vex/claripy/irop.py:599`, generalised to the full
    /// family of `Iop_Reverse{n}sIn{m}_x{k}` opcodes that the Python engine
    /// otherwise marks unsupported.
    pub(super) fn vec_reverse(
        arg: RustBV,
        sub_width: u8,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let sub_width_u32 = sub_width as u32;
        let total = elem_width * count as u32;
        debug_assert_eq!(arg.width(), total);
        debug_assert!(sub_width_u32 > 0 && sub_width_u32 <= elem_width);
        debug_assert_eq!(elem_width % sub_width_u32, 0);
        let sub_per_elem = elem_width / sub_width_u32;

        // Concrete fast path: shuffle bits within each lane using integer ops.
        if total <= 128
            && let Some(v) = arg.as_u128()
        {
            let elem_mask: u128 = Self::low_bit_mask_u128(elem_width);
            let sub_mask: u128 = Self::low_bit_mask_u128(sub_width_u32);
            let mut result: u128 = 0;
            for i in 0..count as u32 {
                let lane_lo = i * elem_width;
                let lane = (v >> lane_lo) & elem_mask;
                let mut reversed_lane: u128 = 0;
                for j in 0..sub_per_elem {
                    let src_lo = j * sub_width_u32;
                    let dst_lo = (sub_per_elem - 1 - j) * sub_width_u32;
                    let sub = (lane >> src_lo) & sub_mask;
                    reversed_lane |= sub << dst_lo;
                }
                result |= reversed_lane << lane_lo;
            }
            return Ok(RustBV::concrete(result, total));
        }

        // Symbolic: extract each sub-unit, place at the mirrored position
        // inside its lane, concat back together.
        let n_subs = (count as u32) * sub_per_elem;
        let mut elements: Vec<RustBV> = Vec::with_capacity(n_subs as usize);
        // concat_le_elements puts elements[0] at LSB, elements[n-1] at MSB.
        // Walk output sub-units from LSB to MSB; within each lane, output
        // sub-unit `j` pulls from input sub-unit `sub_per_elem - 1 - j`.
        for lane in 0..count as u32 {
            let lane_lo = lane * elem_width;
            for j in 0..sub_per_elem {
                let src_lo = lane_lo + (sub_per_elem - 1 - j) * sub_width_u32;
                let src_hi = src_lo + sub_width_u32 - 1;
                elements.push(arg.extract(src_hi, src_lo, ctx));
            }
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Widening vector multiply (`Iop_Mull{N}{S,U}x{M}` full-lane, NEON VMULL;
    /// `Iop_MullEven{N}{S,U}x{M}` even-lane, SSE PMULDQ/PMULUDQ).
    ///
    /// Each contributing input lane (all `count` lanes when `even` is false;
    /// only the even-indexed lanes 0,2,…,count-2 when `even` is true) is
    /// sign- or zero-extended from `elem.bits()` to `2*elem.bits()`, multiplied
    /// with the same-indexed lane of the other operand, and the low
    /// `2*elem.bits()` bits form one output lane. Output lanes are packed
    /// low-to-high, giving a V128 result in every mapped case.
    pub(super) fn vec_mull(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        signed: bool,
        even: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let in_width = elem.bits();
        let out_width = in_width * 2;
        // Contributing input lanes step by 2 for even-lane, else 1.
        let step: u32 = if even { 2 } else { 1 };
        let out_lanes = count as u32 / step;
        let out_total = out_width * out_lanes;

        // Concrete fast path: sign/zero-extend within i128, multiply, mask.
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let in_mask = Self::low_bit_mask_u128(in_width);
            let out_mask = Self::low_bit_mask_u128(out_width);
            let sign_bit: u128 = 1u128 << (in_width - 1);
            let widen = |lane: u128| -> i128 {
                if signed && (lane & sign_bit != 0) {
                    // Two's-complement fill of the upper bits, reinterpreted.
                    (lane | !in_mask) as i128
                } else {
                    lane as i128
                }
            };
            let mut result: u128 = 0;
            for j in 0..out_lanes {
                let lo = j * step * in_width;
                let la = widen((l >> lo) & in_mask);
                let ra = widen((r >> lo) & in_mask);
                let prod = (la.wrapping_mul(ra) as u128) & out_mask;
                result |= prod << (j * out_width);
            }
            return Ok(RustBV::concrete(result, out_total));
        }

        // Symbolic: extract each contributing lane, extend, multiply, concat.
        let mut elements: Vec<RustBV> = Vec::with_capacity(out_lanes as usize);
        for j in 0..out_lanes {
            let lo = j * step * in_width;
            let hi = lo + in_width - 1;
            let la = left
                .extract(hi, lo, ctx)
                .extend_into(out_width, signed, ctx);
            let ra = right
                .extract(hi, lo, ctx)
                .extend_into(out_width, signed, ctx);
            elements.push(la.mul(&ra, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Signed doubling saturating widening multiply — Iop_QDMull{N}Sx{M}
    /// ((I64,I64)->V128, NEON VQDMULL). Full-lane, always signed: each
    /// `elem`-wide lane pair is sign-widened, multiplied, doubled, then clamped
    /// into the signed `2*elem`-bit output lane. Shares the lane layout and
    /// concat ordering with `vec_mull` (even=false); the extra work is the
    /// `*2` and the signed saturation, reusing `saturate_lane_symbolic`.
    pub(super) fn vec_qdmull(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let in_width = elem.bits();
        let out_width = in_width * 2;
        let out_total = out_width * count as u32;

        // Concrete fast path: sign-extend within i128, multiply, double, clamp.
        // in_width <= 32 (only 16Sx4 / 32Sx2 exist) so `2 * la * ra` cannot
        // overflow i128, and the signed 2N-bit range fits in i128.
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let out_mask = Self::low_bit_mask_u128(out_width);
            let sat_max: i128 = (1i128 << (out_width - 1)) - 1;
            let sat_min: i128 = -(1i128 << (out_width - 1));
            let mut result: u128 = 0;
            for j in 0..count as u32 {
                let lo = j * in_width;
                let la = Self::sign_extend_low_to_i128(l >> lo, in_width);
                let ra = Self::sign_extend_low_to_i128(r >> lo, in_width);
                let doubled = la.wrapping_mul(ra).wrapping_mul(2);
                let clamped = doubled.clamp(sat_min, sat_max);
                result |= ((clamped as u128) & out_mask) << (j * out_width);
            }
            return Ok(RustBV::concrete(result, out_total));
        }

        // Symbolic: widen with two spare bits so the doubled product (magnitude
        // up to 2^(2N-1), needing 2N+1 bits with sign) cannot overflow before
        // the clamp, then saturate each lane into the signed out_width.
        let mul_width = out_width + 2;
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for j in 0..count as u32 {
            let lo = j * in_width;
            let hi = lo + in_width - 1;
            let la = left.extract(hi, lo, ctx).extend_into(mul_width, true, ctx);
            let ra = right.extract(hi, lo, ctx).extend_into(mul_width, true, ctx);
            let prod = la.mul(&ra, ctx);
            let doubled = prod.add(&prod, ctx);
            elements.push(Self::saturate_lane_symbolic(
                doubled, mul_width, out_width, /*src_signed=*/ true,
                /*dst_signed=*/ true, ctx,
            ));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }
}
