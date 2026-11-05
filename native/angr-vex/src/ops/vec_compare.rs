//! Vector element-wise comparison and interleave VEX op helpers.
//!
//! Covers the low/high interleave (unpack) ops
//! (Iop_Interleave{LO,HI}{N}x{M}) and the odd/even-lane deinterleave ops
//! (Iop_Cat{Even,Odd}Lanes{N}x{M}). The element-wise compare family
//! (Iop_Cmp{EQ,GT}{N}x{M}) moved to the generic `vec_int_lane_op` driver
//! (`ICmpEq`/`ICmpGt` in `ops/mod.rs`).

use super::{OpError, VEXOps};
use crate::ir::IRType;
use angr_bv::SymBV as RustBV;

impl VEXOps {
    /// Vector interleave low halves.
    pub(super) fn vec_interleave_lo(
        left: RustBV,
        right: RustBV,
        elem: IRType,
    ) -> Result<RustBV, OpError> {
        Self::vec_interleave(left, right, elem, false)
    }

    /// Vector interleave high halves.
    pub(super) fn vec_interleave_hi(
        left: RustBV,
        right: RustBV,
        elem: IRType,
    ) -> Result<RustBV, OpError> {
        Self::vec_interleave(left, right, elem, true)
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
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = left.width();
        let count = total_width / elem_width;
        // Both operands must be the same shape: the extract offsets below are
        // derived from `left`'s width but applied to `right` as well. Matches
        // the operand-width checks in `vec_pairwise_binop` / `vec_int_saturating`.
        Self::require_operand_width("vec_interleave right", right.width(), total_width)?;
        let half_count = count / 2;
        let base = if high { half_count } else { 0 };

        // Concrete fast path (fits in u128 — total_width <= 128 covers all
        // currently-mapped interleave shapes). `RustBV::Concrete` holds its
        // value in a u128, so a wider declared width could not round-trip
        // through `as_u128`; fall through to the symbolic path instead.
        if total_width <= 128
            && let (Some(l), Some(r)) = (left.as_u128(), right.as_u128())
        {
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

            let l_elem = left.extract(src_hi, src_lo);
            let r_elem = right.extract(src_hi, src_lo);

            // right goes to even positions, left to odd
            elements.push(r_elem);
            elements.push(l_elem);
        }

        Ok(Self::concat_le_elements(elements))
    }

    /// Odd/even-lane deinterleave (`Iop_Cat{Even,Odd}Lanes{N}x{M}`). Lane 0
    /// is the least significant; `odd` picks lanes 1, 3, … instead of 0, 2, ….
    /// The right operand's picked lanes fill the low half of the result and
    /// the left's the high half, as irop.py's `_op_generic_CatEvenLanes` /
    /// `_op_generic_CatOddLanes` build it.
    pub(super) fn vec_cat_lanes(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        odd: bool,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = left.width();
        Self::require_operand_width("vec_cat_lanes right", right.width(), total_width)?;
        let count = total_width / elem_width;
        let first = u32::from(odd);
        let picked = || (first..count).step_by(2);

        // Concrete fast path, under the same u128 bound as `vec_interleave`.
        if total_width <= 128
            && let (Some(l), Some(r)) = (left.as_u128(), right.as_u128())
        {
            let elem_mask = Self::low_bit_mask_u128(elem_width);
            let mut result: u128 = 0;
            for (dst, src) in picked().enumerate() {
                let dst_lo = dst as u32 * elem_width;
                let src_lo = src * elem_width;
                result |= ((r >> src_lo) & elem_mask) << dst_lo;
                result |= ((l >> src_lo) & elem_mask) << (dst_lo + total_width / 2);
            }
            return Ok(RustBV::concrete(result, total_width));
        }

        let lanes = |v: &RustBV| -> Vec<RustBV> {
            picked()
                .map(|i| v.extract(i * elem_width + elem_width - 1, i * elem_width))
                .collect()
        };
        let mut elements = lanes(&right);
        elements.extend(lanes(&left));
        Ok(Self::concat_le_elements(elements))
    }
}
