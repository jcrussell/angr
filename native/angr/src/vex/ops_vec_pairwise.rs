//! NEON/SIMD pairwise vector VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via `#[path]` in
//! ops.rs), so these `pub(super)` methods stay callable from the binop/unop
//! dispatch in `ops`, and the shared sibling methods they call
//! (`Self::concat_le_elements`, `Self::vec_float_lane_op`) stay visible by the
//! descendant-module rule. `PwOp` (the per-pair combiner kind) and the `FAdd`
//! `FloatLaneOp` marker both live in `ops` and are reached via `super::`.
//!
//! Covers the pairwise widening add (Iop_PwAddL), the integer binary pairwise
//! family (Iop_PwAdd/PwMin/PwMax), the FP pairwise add (Iop_PwAdd32Fx2), and
//! the rounding halving add (Iop_Avg).

use super::{FAdd, OpError, PwOp, VEXOps};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ir::IRType;

impl VEXOps {
    /// NEON pairwise widening add — `Iop_PwAddL{N}{S/U}x{M}`. Unary.
    /// Output element i (width `2*elem`) = sext_or_zext(a[2i]) +
    /// sext_or_zext(a[2i+1]). Output lane count = `count / 2`; output total
    /// width = input total width.
    pub(super) fn vec_pairwise_add_long(
        arg: RustBV,
        elem: IRType,
        count: u8,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(arg.width(), total_width);
        debug_assert!(count >= 2 && count.is_multiple_of(2));
        let out_pairs = count / 2;
        let out_elem_width = elem_width * 2;

        // Per-lane symbolic build — RustBV ops short-circuit when concrete.
        let mut elements: Vec<RustBV> = Vec::with_capacity(out_pairs as usize);
        for i in 0..out_pairs {
            let lo_a = (2 * i as u32) * elem_width;
            let hi_a = lo_a + elem_width - 1;
            let lo_b = (2 * i as u32 + 1) * elem_width;
            let hi_b = lo_b + elem_width - 1;
            let a_lane = arg.extract(hi_a, lo_a, ctx);
            let b_lane = arg.extract(hi_b, lo_b, ctx);
            let a_wide = a_lane.extend_into(out_elem_width, signed, ctx);
            let b_wide = b_lane.extend_into(out_elem_width, signed, ctx);
            elements.push(a_wide.add_into(b_wide, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON binary pairwise op — `Iop_PwAdd{N}x{M}` / `Iop_PwMin{N}{S/U}x{M}` /
    /// `Iop_PwMax{N}{S/U}x{M}`. Output lane shape matches the inputs. Per-lane:
    ///   * result[i]           = op(a[2i],   a[2i+1])              for i < count/2
    ///   * result[count/2 + i] = op(b[2i],   b[2i+1])              for i < count/2
    pub(super) fn vec_pairwise_binop(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        op: PwOp,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);
        debug_assert!(count >= 2 && count.is_multiple_of(2));
        let half = count / 2;

        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        // First half from `left`, then second half from `right` (same
        // interleave order as the FP `vec_float_pairwise_add`).
        for src in [&left, &right] {
            for i in 0..half {
                let lo_a = (2 * i as u32) * elem_width;
                let hi_a = lo_a + elem_width - 1;
                let lo_b = (2 * i as u32 + 1) * elem_width;
                let hi_b = lo_b + elem_width - 1;
                let a = src.extract(hi_a, lo_a, ctx);
                let b = src.extract(hi_b, lo_b, ctx);
                elements.push(Self::pw_combine(a, b, op, ctx));
            }
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    #[inline]
    fn pw_combine(a: RustBV, b: RustBV, op: PwOp, ctx: &SymContext) -> RustBV {
        match op {
            PwOp::Add => a.add_into(b, ctx),
            PwOp::MinS => {
                let cond = a.clone().sle_into(b.clone(), ctx);
                cond.ite_into(a, b, ctx)
            }
            PwOp::MinU => {
                let cond = a.clone().ule_into(b.clone(), ctx);
                cond.ite_into(a, b, ctx)
            }
            PwOp::MaxS => {
                let cond = a.clone().sge_into(b.clone(), ctx);
                cond.ite_into(a, b, ctx)
            }
            PwOp::MaxU => {
                let cond = a.clone().uge_into(b.clone(), ctx);
                cond.ite_into(a, b, ctx)
            }
        }
    }

    /// NEON pairwise FP add — `Iop_PwAdd32Fx2` (ARM VPADD.F32). Binary; the FP
    /// analogue of `vec_pairwise_binop` with `PwOp::Add`, but the per-pair
    /// combine is an FP add (via the `FAdd` `FloatLaneOp` so both the concrete
    /// and symbolic branches stay in lockstep). Output lane shape matches the
    /// inputs: first half from `left`, second half from `right`. For the only
    /// VEX-emitted shape (`32Fx2`) this yields `[a0+a1, b0+b1]`.
    pub(super) fn vec_float_pairwise_add(
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
        debug_assert!(count >= 2 && count.is_multiple_of(2));
        let half = count / 2;

        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        // First half from `left`, then second half from `right` (same
        // interleave order as the integer `vec_pairwise_binop`).
        for src in [&left, &right] {
            for i in 0..half {
                let lo_a = (2 * i as u32) * elem_width;
                let hi_a = lo_a + elem_width - 1;
                let lo_b = (2 * i as u32 + 1) * elem_width;
                let hi_b = lo_b + elem_width - 1;
                let a = src.extract(hi_a, lo_a, ctx);
                let b = src.extract(hi_b, lo_b, ctx);
                // Reuse the single-lane FP-add path (count=1) so the concrete
                // and symbolic branches match the packed `VFAdd`.
                elements.push(Self::vec_float_lane_op(&[a, b], elem, 1, &FAdd, ctx)?);
            }
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON rounding halving add — `Iop_Avg{N}{S/U}x{M}`. Per-lane:
    ///   `((a[i] + b[i] + 1) >> 1)` truncated to `elem` bits.
    /// Each lane is sign- or zero-extended to `elem+1` bits to absorb the carry
    /// from `+1`, summed, shifted right by 1 (logical — the high bit of the
    /// widened sum carries the rounding bit for both signedness conventions),
    /// then truncated back to `elem` bits.
    pub(super) fn vec_rounding_avg(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);
        let wide = elem_width + 1;
        let one_wide = RustBV::concrete(1, wide);

        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count as u32 {
            let lo = i * elem_width;
            let hi = lo + elem_width - 1;
            let a_lane = left.extract(hi, lo, ctx);
            let b_lane = right.extract(hi, lo, ctx);
            let a_wide = a_lane.extend_into(wide, signed, ctx);
            let b_wide = b_lane.extend_into(wide, signed, ctx);
            let sum = a_wide.add_into(b_wide, ctx).add_into(one_wide.clone(), ctx);
            let shifted = sum.lshr_into(RustBV::concrete(1, wide), ctx);
            // Truncate to elem bits — for signed, the (elem)th bit of `shifted`
            // is the original sign bit by construction, so the low `elem` bits
            // are the correct two's-complement representation.
            elements.push(shifted.extract(elem_width - 1, 0, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }
}
