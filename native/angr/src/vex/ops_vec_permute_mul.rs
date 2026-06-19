//! Vector sub-unit reversal and low-half packed multiply VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via `#[path]` in
//! ops.rs), so these `pub(super)` methods stay callable from the unop/binop
//! dispatch in `ops`, and the shared siblings they reference
//! (`Self::concat_le_elements`, `Self::sign_extend_low_to_u64`, which stay in
//! ops.rs) stay visible via super/the descendant rule.
//!
//! Covers the ARM REV*/RBIT sub-unit reversal family
//! (Iop_Reverse{n}sIn{m}_x{k}) and the low-half packed multiply (PMULLD,
//! Iop_Mul{N}x{M}).

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
            let elem_mask: u128 = if elem_width == 128 {
                u128::MAX
            } else {
                (1u128 << elem_width) - 1
            };
            let sub_mask: u128 = (1u128 << sub_width_u32) - 1;
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

    /// Vector multiply keeping low half (PMULLD).
    /// Performs signed widening multiply on each element pair, keeping only the low bits.
    pub(super) fn vec_mul_lo(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;

        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);

        // For concrete values, compute directly
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let mut result: u128 = 0;
            let mask = (1u128 << elem_width) - 1;

            for i in 0..count {
                let lo = (i as u32) * elem_width;

                let l_elem = (l >> lo) & mask;
                let r_elem = (r >> lo) & mask;

                // Sign-extend each lane to 64 bits, then multiply at u64 and
                // mask to the lane width — high bits beyond `2 * elem_width`
                // are discarded by the mask, so any 64-bit representation
                // matching the lane's signed value in the low bits suffices.
                let l_signed = Self::sign_extend_low_to_u64(l_elem, elem_width);
                let r_signed = Self::sign_extend_low_to_u64(r_elem, elem_width);

                // Multiply and keep low bits
                let product = l_signed.wrapping_mul(r_signed);
                let res_elem = (product as u128) & mask;

                result |= res_elem << lo;
            }

            return Ok(RustBV::concrete(result, total_width));
        }

        // For symbolic values, fall back to element-wise
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);

        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;

            let l_elem = left.extract(hi, lo, ctx);
            let r_elem = right.extract(hi, lo, ctx);

            // For symbolic, just do regular multiply (low bits are the same for signed/unsigned)
            let res_elem = l_elem.mul_into(r_elem, ctx);
            elements.push(res_elem);
        }

        Ok(Self::concat_le_elements(elements, ctx))
    }
}
