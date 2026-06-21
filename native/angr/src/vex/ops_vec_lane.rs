//! NEON/SIMD vector lane access and width-conversion VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via `#[path]` in
//! ops.rs), so these `pub(super)` methods stay callable from the unop/binop
//! dispatch in `ops`, and the shared sibling method they call
//! (`Self::concat_le_elements`) stays visible by the descendant-module rule.
//!
//! Covers lane extract/insert (Iop_GetElem/SetElem), broadcast (Iop_Dup),
//! and per-lane widen/narrow (Iop_Widen*, Iop_NarrowUn/NarrowBin). The
//! saturating narrow (Iop_QNarrow*) family and `saturate_lane` live in the
//! sibling `ops_vec_saturate` module alongside the rest of the saturation
//! helpers.

use super::{OpError, VEXOps};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ir::IRType;

impl VEXOps {
    /// NEON lane extract (Iop_GetElem{N}x{M}).
    ///
    /// `vec` is the source vector (width = elem.bits() * count), `idx` is
    /// Ity_I8 (the lane index — 0 = lowest lane). Result is one lane.
    ///
    /// For concrete `idx < count`, this is a simple bit-slice. Concrete
    /// out-of-range indices saturate at the highest valid lane (matches
    /// pyvex's behavior of treating `idx % count` as the effective lane).
    /// Symbolic `idx` builds an ITE chain over all `count` lanes.
    pub(super) fn vec_get_elem(
        vec: RustBV,
        idx: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(vec.width(), total_width);
        debug_assert_eq!(idx.width(), 8);

        // Concrete fast path
        if let (Some(v), Some(i)) = (vec.as_u128(), idx.as_u128()) {
            let lane = (i as u8) % count;
            let lo = lane as u32 * elem_width;
            let mask = Self::low_bit_mask_u128(elem_width);
            let result = (v >> lo) & mask;
            return Ok(RustBV::concrete(result, elem_width));
        }

        // Concrete idx, symbolic vec: bit-slice
        if let Some(i) = idx.as_u128() {
            let lane = (i as u8) % count;
            let lo = lane as u32 * elem_width;
            let hi = lo + elem_width - 1;
            return Ok(vec.extract(hi, lo, ctx));
        }

        // Symbolic idx: ITE chain over all lanes. count is small (<= 16).
        let mut result = vec.extract(elem_width - 1, 0, ctx);
        for lane in 1..count {
            let lo = lane as u32 * elem_width;
            let hi = lo + elem_width - 1;
            let lane_val = vec.extract(hi, lo, ctx);
            let lane_idx = RustBV::concrete(lane as u128, 8);
            let cond = idx.eq(&lane_idx, ctx);
            result = cond.ite(&lane_val, &result, ctx);
        }
        Ok(result)
    }

    /// NEON lane insert (Iop_SetElem{N}x{M}).
    ///
    /// `vec` is the source vector, `idx` (Ity_I8) selects the lane, `val`
    /// (width = elem.bits()) is the replacement. Returns the modified
    /// vector. Concrete out-of-range indices wrap (matches pyvex).
    /// Symbolic `idx` builds an ITE chain.
    pub(super) fn vec_set_elem(
        vec: RustBV,
        idx: RustBV,
        val: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(vec.width(), total_width);
        debug_assert_eq!(idx.width(), 8);
        debug_assert_eq!(val.width(), elem_width);

        // Concrete vec + val + idx: bit-twiddle.
        if let (Some(v), Some(i), Some(x)) = (vec.as_u128(), idx.as_u128(), val.as_u128()) {
            let lane = (i as u8) % count;
            let lo = lane as u32 * elem_width;
            let mask_elem = Self::low_bit_mask_u128(elem_width);
            let shifted_mask = mask_elem << lo;
            let cleared = v & !shifted_mask;
            let new_val = cleared | ((x & mask_elem) << lo);
            return Ok(RustBV::concrete(new_val, total_width));
        }

        // Concrete idx, symbolic vec/val: rebuild from element slices.
        if let Some(i) = idx.as_u128() {
            let lane = (i as u8) % count;
            let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
            for slot in 0..count {
                if slot == lane {
                    elements.push(val.clone());
                } else {
                    let lo = slot as u32 * elem_width;
                    let hi = lo + elem_width - 1;
                    elements.push(vec.extract(hi, lo, ctx));
                }
            }
            return Ok(Self::concat_le_elements(elements, ctx));
        }

        // Symbolic idx: ITE chain — for each lane, select val if idx==lane
        // else the original lane bits, then rebuild the vector.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for slot in 0..count {
            let lo = slot as u32 * elem_width;
            let hi = lo + elem_width - 1;
            let orig_lane = vec.extract(hi, lo, ctx);
            let slot_idx = RustBV::concrete(slot as u128, 8);
            let cond = idx.eq(&slot_idx, ctx);
            let chosen = cond.ite(&val, &orig_lane, ctx);
            elements.push(chosen);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON broadcast scalar to vector (Iop_Dup{N}x{M}). Replicates `arg` into
    /// `count` lanes of width `elem.bits()`.
    pub(super) fn vec_dup(
        arg: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(arg.width(), elem_width);

        // Concrete fast path.
        if total_width <= 128
            && let Some(v) = arg.as_u128()
        {
            let mask: u128 = Self::low_bit_mask_u128(elem_width);
            let lane = v & mask;
            let mut result: u128 = 0;
            for i in 0..count {
                let lo = (i as u32) * elem_width;
                result |= lane << lo;
            }
            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic: concat the same value `count` times.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for _ in 0..count {
            elements.push(arg.clone());
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON widen each lane (Iop_Widen{N}{S/U}to{2N}x{M}). Sign- or
    /// zero-extends each lane from `from.bits()` to `from.bits()*2`.
    pub(super) fn vec_widen(
        arg: RustBV,
        from: IRType,
        count: u8,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let from_width = from.bits();
        let to_width = from_width * 2;
        let in_total = from_width * count as u32;
        debug_assert_eq!(arg.width(), in_total);

        // Concrete fast path.
        if in_total <= 128
            && (to_width * count as u32) <= 128
            && let Some(v) = arg.as_u128()
        {
            let in_mask: u128 = Self::low_bit_mask_u128(from_width);
            let out_mask: u128 = Self::low_bit_mask_u128(to_width);
            let sign_bit: u128 = 1u128 << (from_width - 1);
            let mut result: u128 = 0;
            for i in 0..count {
                let lo = (i as u32) * from_width;
                let lane = (v >> lo) & in_mask;
                let widened = if signed && (lane & sign_bit != 0) {
                    // Sign-extend: fill upper (to_width - from_width) bits with 1s.
                    (lane | !in_mask) & out_mask
                } else {
                    lane
                };
                let out_lo = (i as u32) * to_width;
                result |= widened << out_lo;
            }
            return Ok(RustBV::concrete(result, to_width * count as u32));
        }

        // Symbolic: extract each lane, extend, concat.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * from_width;
            let hi = lo + from_width - 1;
            let lane = arg.extract(hi, lo, ctx);
            let widened = lane.extend_into(to_width, signed, ctx);
            elements.push(widened);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON unary narrow (Iop_NarrowUn{N}to{N/2}x{M}). Truncates each lane
    /// from `from.bits()` to `from.bits()/2`.
    pub(super) fn vec_narrow_un(
        arg: RustBV,
        from: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let from_width = from.bits();
        let to_width = from_width / 2;
        let in_total = from_width * count as u32;
        debug_assert_eq!(arg.width(), in_total);

        // Concrete fast path.
        if in_total <= 128
            && let Some(v) = arg.as_u128()
        {
            let to_mask: u128 = Self::low_bit_mask_u128(to_width);
            let mut result: u128 = 0;
            for i in 0..count {
                let lo = (i as u32) * from_width;
                let lane = (v >> lo) & to_mask;
                let out_lo = (i as u32) * to_width;
                result |= lane << out_lo;
            }
            return Ok(RustBV::concrete(result, to_width * count as u32));
        }

        // Symbolic: extract each low half-lane, concat.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * from_width;
            let hi = lo + to_width - 1;
            elements.push(arg.extract(hi, lo, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON binary narrow (Iop_NarrowBin{N}to{N/2}x{M}). Each input has
    /// `count/2` lanes of width `from`; result has `count` lanes of width
    /// `from/2` with `left` providing the low half and `right` the high half.
    pub(super) fn vec_narrow_bin(
        left: RustBV,
        right: RustBV,
        from: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let from_width = from.bits();
        let to_width = from_width / 2;
        let per_input = (count / 2) as u32;
        let in_total = from_width * per_input;
        debug_assert_eq!(left.width(), in_total);
        debug_assert_eq!(right.width(), in_total);

        // Concrete fast path.
        if in_total <= 128
            && (to_width * count as u32) <= 128
            && let (Some(l), Some(r)) = (left.as_u128(), right.as_u128())
        {
            let to_mask: u128 = Self::low_bit_mask_u128(to_width);
            let mut result: u128 = 0;
            for i in 0..per_input {
                let lo = i * from_width;
                let lane_l = (l >> lo) & to_mask;
                let lane_r = (r >> lo) & to_mask;
                let out_lo_l = i * to_width;
                let out_lo_r = (per_input + i) * to_width;
                result |= lane_l << out_lo_l;
                result |= lane_r << out_lo_r;
            }
            return Ok(RustBV::concrete(result, to_width * count as u32));
        }

        // Symbolic: extract each low half-lane from both operands, concat.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..per_input {
            let lo = i * from_width;
            let hi = lo + to_width - 1;
            elements.push(left.extract(hi, lo, ctx));
        }
        for i in 0..per_input {
            let lo = i * from_width;
            let hi = lo + to_width - 1;
            elements.push(right.extract(hi, lo, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }
}
