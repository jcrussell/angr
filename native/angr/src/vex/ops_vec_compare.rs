//! Vector element-wise comparison and interleave VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via `#[path]` in
//! ops.rs), so these `pub(super)` methods stay callable from the binop
//! dispatch in `ops`, and the shared sibling they reference
//! (`Self::concat_le_elements`, which stays in ops.rs) stays visible via
//! super/the descendant rule.
//!
//! Covers element-wise compare (Iop_Cmp{EQ,GT}{N}x{M}) and the
//! low/high interleave (unpack) ops (Iop_Interleave{LO,HI}{N}x{M}).

use super::{OpError, VEXOps};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ir::IRType;

impl VEXOps {
    /// Vector element-wise comparison.
    pub(super) fn vec_cmp(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        op: &str,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;

        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);

        // For concrete values
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let mut result: u128 = 0;
            let elem_mask = Self::low_bit_mask_u128(elem_width);
            let all_ones = elem_mask;

            for i in 0..count {
                let lo = (i as u32) * elem_width;
                let l_elem = (l >> lo) & elem_mask;
                let r_elem = (r >> lo) & elem_mask;

                let cmp_result = match op {
                    "eq" => l_elem == r_elem,
                    "gt" => {
                        // Signed comparison
                        let l_signed = Self::sign_extend_low_to_i128(l_elem, elem_width);
                        let r_signed = Self::sign_extend_low_to_i128(r_elem, elem_width);
                        l_signed > r_signed
                    }
                    _ => return Err(OpError::UnsupportedVectorOp(op.to_string())),
                };

                // Result is all 1s if true, all 0s if false
                if cmp_result {
                    result |= all_ones << lo;
                }
            }

            return Ok(RustBV::concrete(result, total_width));
        }

        // For symbolic, fall back to element-wise
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);

        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;

            let l_elem = left.extract(hi, lo, ctx);
            let r_elem = right.extract(hi, lo, ctx);

            let cmp_result = match op {
                "eq" => l_elem.eq_into(r_elem, ctx),
                "gt" => l_elem.sgt_into(r_elem, ctx),
                _ => return Err(OpError::UnsupportedVectorOp(op.to_string())),
            };

            // Extend the 1-bit result to full element width (all 1s or all 0s)
            let extended = cmp_result.sign_extend_into(elem_width, ctx);
            elements.push(extended);
        }

        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Vector interleave low halves.
    pub(super) fn vec_interleave_lo(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::vec_interleave(left, right, elem, false, ctx)
    }

    /// Vector interleave high halves.
    pub(super) fn vec_interleave_hi(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::vec_interleave(left, right, elem, true, ctx)
    }

    /// Shared interleave implementation. `high` selects which half of each
    /// operand is read: the low half (`false`, InterleaveLO) or the high half
    /// (`true`, InterleaveHI). In both cases right goes to even output
    /// positions and left to odd.
    fn vec_interleave(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        high: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = left.width();
        let count = total_width / elem_width;
        let half_count = count / 2;
        let base = if high { half_count } else { 0 };

        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let mut result: u128 = 0;
            let elem_mask = Self::low_bit_mask_u128(elem_width);

            for i in 0..half_count {
                let src_lo = (base + i) * elem_width;
                let dst_lo = i * 2 * elem_width;

                let l_elem = (l >> src_lo) & elem_mask;
                let r_elem = (r >> src_lo) & elem_mask;

                // right goes to even positions, left to odd
                result |= r_elem << dst_lo;
                result |= l_elem << (dst_lo + elem_width);
            }

            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic case
        let mut elements: Vec<RustBV> = Vec::new();

        for i in 0..half_count {
            let src_lo = (base + i) * elem_width;
            let src_hi = src_lo + elem_width - 1;

            let l_elem = left.extract(src_hi, src_lo, ctx);
            let r_elem = right.extract(src_hi, src_lo, ctx);

            // right goes to even positions, left to odd
            elements.push(r_elem);
            elements.push(l_elem);
        }

        Ok(Self::concat_le_elements(elements, ctx))
    }
}
