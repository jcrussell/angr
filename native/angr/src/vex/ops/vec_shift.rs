//! NEON/SIMD vector shift VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via a plain `mod` decl in
//! `ops/mod.rs`), so these `pub(super)` methods stay callable from the binop
//! dispatch in `ops`, and the shared sibling method they call
//! (`Self::concat_le_elements`) stays visible by the descendant-module rule.
//! `VecShiftKind` likewise lives in `ops` and is reached via `super::`.
//!
//! Covers shift-by-immediate (Iop_ShlN/ShrN/SarN) and shift-by-vector
//! (Iop_Shl/Shr/Sar{N}x{M}, the NEON USHL/SSHL family). The saturating
//! shift (Iop_QShl*) deliberately stays in `ops` alongside the rest of the
//! saturation helpers, and the scalar shift normaliser
//! (`normalize_shift_amount`) stays with the scalar shift dispatch.
//!
//! The concrete fast paths in `vec_shl_n` / `vec_shr_n` / `vec_sar_n` are
//! gated on `total_width <= 128`, matching `vec_int_lane_op`. A `RustBV`
//! stores concrete values in a `u128` (see `RustBV::concrete`), so a wider
//! vector can only arrive as an expression; folding one anyway would shift a
//! `u128` by up to 192 and either panic (debug) or wrap mod 128 (release).
//! The guard became reachable with the AVX2 256-bit shift-by-immediate arms
//! added in angr-sqfj8.118.

use super::{OpError, VEXOps, VecShiftKind};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ir::IRType;

impl VEXOps {
    /// Vector shift left by immediate.
    pub(super) fn vec_shl_n(
        vec: RustBV,
        shift_amt: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;

        Self::require_operand_width("vec_shl_n vec", vec.width(), total_width)?;

        // Concrete shift amount: keep the existing fast paths.
        if total_width <= 128
            && let Some(s) = shift_amt.as_u128()
        {
            let shift = s as u32;

            if shift >= elem_width {
                return Ok(RustBV::concrete(0, total_width));
            }

            if let Some(v) = vec.as_u128() {
                let mut result: u128 = 0;
                let elem_mask = Self::low_bit_mask_u128(elem_width);

                for i in 0..count {
                    let lo = (i as u32) * elem_width;
                    let elem_val = (v >> lo) & elem_mask;
                    let shifted = (elem_val << shift) & elem_mask;
                    result |= shifted << lo;
                }

                return Ok(RustBV::concrete(result, total_width));
            }
        }

        // Symbolic shift amount (or symbolic vector with concrete shift):
        // resize the count to the lane width and apply per-lane.  Z3 bvshl
        // returns 0 when the shift count is >= the operand width, matching
        // the concrete semantics above.
        let resized_shift = Self::resize_vec_shift_amount(shift_amt, elem_width, ctx)?;
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let elem_val = vec.extract(hi, lo, ctx);
            let shifted = elem_val.shl_into(resized_shift.clone(), ctx);
            elements.push(shifted);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Sign-fill mask for an arithmetic right shift of an `elem_width`-bit lane
    /// by `shift` (`shift < elem_width`, guaranteed by every caller): the top
    /// `shift` bits of the lane set to 1. `elem_mask` is
    /// `low_bit_mask_u128(elem_width)`.
    ///
    /// `elem_width - shift` reaches `elem_width`, so a 128-bit lane with
    /// `shift == 0` would left-shift a u128 by 128 — a panic under
    /// `panic=abort`, a silent mod-128 wrap in release. Saturate to 0 there: a
    /// shift-by-0 fills no bits, which is exactly what the reachable `< 128`
    /// path already yields (`(elem_mask << elem_width) & elem_mask == 0`).
    /// Latent today (per-lane arithmetic shifts route only NEON lanes of
    /// <= 64 bits) but mirrors the `sign_extend_to` / `low_bit_mask_u128`
    /// >=128 guards (angr-qwyti.16/.17).
    #[inline]
    pub(super) fn sar_fill_mask(elem_mask: u128, elem_width: u32, shift: u32) -> u128 {
        let fill_bits = elem_width - shift;
        if fill_bits >= 128 {
            0
        } else {
            (elem_mask << fill_bits) & elem_mask
        }
    }

    /// Resize a vector shift amount to the lane width.
    ///
    /// VEX `ShlN` / `ShrN` / `SarN` take an I8 count. When the lane is wider,
    /// zero-extend so Z3's shift ops see matching widths. A wider count would
    /// require an ITE on the high bits — not seen in real VEX, so reject.
    fn resize_vec_shift_amount(
        shift_amt: RustBV,
        elem_width: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        match shift_amt.width().cmp(&elem_width) {
            std::cmp::Ordering::Equal => Ok(shift_amt),
            std::cmp::Ordering::Less => Ok(shift_amt.zero_extend_into(elem_width, ctx)),
            std::cmp::Ordering::Greater => Err(OpError::UnsupportedVectorOp(
                "vector shift amount wider than lane".to_string(),
            )),
        }
    }

    /// Vector shift right logical by immediate.
    pub(super) fn vec_shr_n(
        vec: RustBV,
        shift_amt: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;

        Self::require_operand_width("vec_shr_n vec", vec.width(), total_width)?;

        if total_width <= 128
            && let Some(s) = shift_amt.as_u128()
        {
            let shift = s as u32;

            if shift >= elem_width {
                return Ok(RustBV::concrete(0, total_width));
            }

            if let Some(v) = vec.as_u128() {
                let mut result: u128 = 0;
                let elem_mask = Self::low_bit_mask_u128(elem_width);

                for i in 0..count {
                    let lo = (i as u32) * elem_width;
                    let elem_val = (v >> lo) & elem_mask;
                    let shifted = elem_val >> shift;
                    result |= shifted << lo;
                }

                return Ok(RustBV::concrete(result, total_width));
            }
        }

        let resized_shift = Self::resize_vec_shift_amount(shift_amt, elem_width, ctx)?;
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let elem_val = vec.extract(hi, lo, ctx);
            let shifted = elem_val.lshr_into(resized_shift.clone(), ctx);
            elements.push(shifted);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Vector shift right arithmetic by immediate.
    pub(super) fn vec_sar_n(
        vec: RustBV,
        shift_amt: RustBV,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;

        Self::require_operand_width("vec_sar_n vec", vec.width(), total_width)?;

        if total_width <= 128
            && let Some(s) = shift_amt.as_u128()
        {
            let shift = s as u32;

            if let Some(v) = vec.as_u128() {
                let mut result: u128 = 0;
                let elem_mask = Self::low_bit_mask_u128(elem_width);
                let sign_bit = 1u128 << (elem_width - 1);

                for i in 0..count {
                    let lo = (i as u32) * elem_width;
                    let elem_val = (v >> lo) & elem_mask;

                    // Arithmetic shift - preserve sign
                    let shifted = if shift >= elem_width {
                        // Shift >= width: result is all sign bits
                        if elem_val & sign_bit != 0 {
                            elem_mask // All 1s
                        } else {
                            0 // All 0s
                        }
                    } else {
                        // Check if negative (sign bit set)
                        if elem_val & sign_bit != 0 {
                            // Negative: shift and fill with 1s
                            let shifted_val = elem_val >> shift;
                            let fill_mask = Self::sar_fill_mask(elem_mask, elem_width, shift);
                            (shifted_val | fill_mask) & elem_mask
                        } else {
                            // Positive: simple logical shift
                            elem_val >> shift
                        }
                    };

                    result |= shifted << lo;
                }

                return Ok(RustBV::concrete(result, total_width));
            }
        }

        // Symbolic shift amount (or symbolic vector with concrete shift):
        // Z3 bvashr replicates the sign bit when the shift count is >= the
        // operand width, matching the concrete sign-fill semantics above.
        let resized_shift = Self::resize_vec_shift_amount(shift_amt, elem_width, ctx)?;
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let elem_val = vec.extract(hi, lo, ctx);
            let shifted = elem_val.ashr_into(resized_shift.clone(), ctx);
            elements.push(shifted);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON vector shift by *vector*: lane `i` of result = `lane_a[i] OP lane_b[i]`
    /// where `OP` is logical-left / logical-right / arithmetic-right per `kind`.
    /// Maps to `Iop_Shl{N}x{M}` / `Iop_Sal{N}x{M}` (left), `Iop_Shr{N}x{M}` (lshr),
    /// and `Iop_Sar{N}x{M}` (ashr); see ARM USHL/SSHL (DDI 0487 C7.2.310, C7.2.291).
    /// Z3 `bvshl`/`bvlshr`/`bvashr` semantics handle out-of-range counts the
    /// same way the concrete fast path does (≥ lane width → 0 for shl/lshr,
    /// sign-fill for ashr).
    ///
    /// ## The shift amount is *unsigned* here, deliberately (angr-0jh0j.64)
    ///
    /// libVEX's own header carries a FIXME above the `Iop_Shl8x16`/`Iop_Sal…`
    /// declarations: the ARM32 front/back ends read this operand as an **8-bit
    /// signed** count — negative means "shift the other way" — while every
    /// other target reads it as unsigned. That is not hypothetical; the ARM32
    /// front end lifts `vshl.s8 d0, d1, d2` to `Sar8x8(d1, Sub8x8(0, d2))`,
    /// which only makes sense if a negative count means "shift left". So a
    /// positive ARM32 `vshl` amount reaches us as a count ≥ lane width and we
    /// return sign-fill/zero where hardware would left-shift.
    ///
    /// We match Python anyway: `angr/engines/vex/claripy/irop.py`'s
    /// `_op_vector_mapped` chops lanes and applies
    /// `__lshift__`/`LShR`/`__rshift__` per lane with the same unsigned
    /// reading, and the two engines interleave within a single exploration —
    /// a Rust-only "fix" would make the answer depend on which engine ran the
    /// block. Fixing it is an upstream angr change that has to land on both
    /// sides at once. `vec_qshl_sat`'s signed amount is *not* a precedent for
    /// changing this: the FIXME covers only this non-saturating family, and
    /// the saturating `Iop_QShl`/`Iop_QSal` block is declared below it with no
    /// such caveat.
    ///
    /// Pinned by `test_arm32_neon_vector_shift_matches_python_engine`
    /// (`tests/engines/rust/test_multiarch.py`), which drives both engines
    /// over the divergent lanes and records the hardware answer alongside.
    pub(super) fn vec_shift_vec(
        vec: RustBV,
        amts: RustBV,
        elem: IRType,
        count: u8,
        kind: VecShiftKind,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        Self::require_operand_width("vec_shift_vec vec", vec.width(), total_width)?;
        Self::require_operand_width("vec_shift_vec amts", amts.width(), total_width)?;

        // Concrete fast path: both operands fit in u128 (covers every NEON
        // shape we route here — 64- and 128-bit vectors).
        if total_width <= 128
            && let (Some(v), Some(s)) = (vec.as_u128(), amts.as_u128())
        {
            let mut result: u128 = 0;
            let elem_mask: u128 = Self::low_bit_mask_u128(elem_width);
            let sign_bit: u128 = 1u128 << (elem_width - 1);

            for i in 0..count {
                let lo = (i as u32) * elem_width;
                let a = (v >> lo) & elem_mask;
                // Use the full elem_width-wide count as an unsigned int
                // — matches Z3 bvshl/bvlshr/bvashr semantics (count ≥
                // operand width collapses to 0 or sign-fill).
                let amt = (s >> lo) & elem_mask;

                let shifted: u128 = match kind {
                    VecShiftKind::Shl => {
                        if amt >= elem_width as u128 {
                            0
                        } else {
                            (a << amt) & elem_mask
                        }
                    }
                    VecShiftKind::Shr => {
                        if amt >= elem_width as u128 {
                            0
                        } else {
                            a >> amt
                        }
                    }
                    VecShiftKind::Sar => {
                        let neg = a & sign_bit != 0;
                        if amt >= elem_width as u128 {
                            if neg { elem_mask } else { 0 }
                        } else if neg {
                            let shifted_val = a >> amt;
                            let fill_mask = Self::sar_fill_mask(elem_mask, elem_width, amt as u32);
                            (shifted_val | fill_mask) & elem_mask
                        } else {
                            a >> amt
                        }
                    }
                };

                result |= (shifted & elem_mask) << lo;
            }
            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic per-lane fallback. Both operands are split into lane-width
        // slices; Z3 bvshl/bvlshr/bvashr produce the matching out-of-range
        // behaviour, so no extra width guards are needed.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let a = vec.extract(hi, lo, ctx);
            let b = amts.extract(hi, lo, ctx);
            let shifted = match kind {
                VecShiftKind::Shl => a.shl_into(b, ctx),
                VecShiftKind::Shr => a.lshr_into(b, ctx),
                VecShiftKind::Sar => a.ashr_into(b, ctx),
            };
            elements.push(shifted);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }
}
