//! Vector element-wise comparison and interleave VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via a plain `mod` decl in
//! `ops/mod.rs`), so these `pub(super)` methods stay callable from the binop
//! dispatch in `ops`, and the shared sibling they reference
//! (`Self::concat_le_elements`, which stays in `ops/mod.rs`) stays visible via
//! super/the descendant rule.
//!
//! Covers the low/high interleave (unpack) ops
//! (Iop_Interleave{LO,HI}{N}x{M}). The element-wise compare family
//! (Iop_Cmp{EQ,GT}{N}x{M}) moved to the generic `vec_int_lane_op` driver
//! (`ICmpEq`/`ICmpGt` in `ops/mod.rs`).

use super::{OpError, VEXOps};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ir::IRType;

impl VEXOps {
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

            let l_elem = left.extract(src_hi, src_lo, ctx);
            let r_elem = right.extract(src_hi, src_lo, ctx);

            // right goes to even positions, left to odd
            elements.push(r_elem);
            elements.push(l_elem);
        }

        Ok(Self::concat_le_elements(elements, ctx))
    }
}
