//! NEON/SIMD per-lane bit-counting VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via a plain `mod` decl in
//! `ops/mod.rs`), so these `pub(super)` methods stay callable from the unop
//! dispatch in `ops`, and the shared siblings they reference
//! (`Self::concat_le_elements`, the `LaneCountKind` enum — both stay in
//! `ops/mod.rs`) stay visible via super/the descendant rule. The per-lane
//! ITE-chain builders (`clz_chain`, `cls_chain`) are private to this module.
//!
//! Covers per-byte popcount (Iop_Cnt8x{8,16}) and per-lane count-leading-
//! zeros / count-leading-sign-bits (Iop_Clz{N}x{M} / Iop_Cls{N}x{M}).

use super::{LaneCountKind, OpError, VEXOps};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ir::IRType;

impl VEXOps {
    /// NEON per-byte popcount — `Iop_Cnt8x{8,16}` (ARM CNT). Each 8-bit lane
    /// is replaced by the count of set bits in that lane (0..=8). Result
    /// width equals input width (8 * count bits). Symbolic fast-path: each
    /// bit of the lane is zero-extended to 8 bits and summed.
    pub(super) fn vec_cnt(arg: RustBV, count: u8, ctx: &SymContext) -> Result<RustBV, OpError> {
        let total = 8u32 * count as u32;
        Self::require_operand_width("vec_cnt arg", arg.width(), total)?;

        // Concrete fast path: iterate bytes, count_ones each.
        if let Some(v) = arg.as_u128() {
            let mut result: u128 = 0;
            for i in 0..count as u32 {
                let lane = ((v >> (i * 8)) as u8) as u32;
                let popcnt = lane.count_ones() as u128;
                result |= popcnt << (i * 8);
            }
            return Ok(RustBV::concrete(result, total));
        }

        // Symbolic: per byte, sum the 8 bits as 8-bit zero-extended adds.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count as u32 {
            let lane_lo = i * 8;
            let mut acc = RustBV::concrete(0, 8);
            for b in 0..8 {
                let bit = arg.extract(lane_lo + b, lane_lo + b, ctx);
                let bit_ext = bit.zero_extend_into(8, ctx);
                acc = acc.add_into(bit_ext, ctx);
            }
            elements.push(acc);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// SSE byte-mask extract — `Iop_GetMSBs8x{8,16}` (x86 PMOVMSKB). Reduces a
    /// vector of `count` bytes (8 * count bits) to a `count`-bit integer whose
    /// bit `i` is the most-significant bit (bit 7) of input byte `i`. Used by
    /// glibc's SSE strlen/memchr; see angr-75mc.
    pub(super) fn vec_get_msbs(
        arg: RustBV,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let total = 8u32 * count as u32;
        Self::require_operand_width("vec_get_msbs arg", arg.width(), total)?;

        // Concrete fast path: gather bit 7 of each byte into the result.
        if let Some(v) = arg.as_u128() {
            let mut result: u128 = 0;
            for i in 0..count as u32 {
                let msb = (v >> (i * 8 + 7)) & 1;
                result |= msb << i;
            }
            return Ok(RustBV::concrete(result, count as u32));
        }

        // Symbolic: extract each byte's MSB as a 1-bit lane, concat little-end
        // so byte 0's MSB lands at result bit 0.
        let mut bits: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count as u32 {
            let msb_pos = i * 8 + 7;
            bits.push(arg.extract(msb_pos, msb_pos, ctx));
        }
        Ok(Self::concat_le_elements(bits, ctx))
    }

    /// NEON per-lane Clz/Cls — `Iop_Clz{N}x{M}` / `Iop_Cls{N}x{M}`. Each
    /// `elem`-wide lane is replaced by:
    ///   * `Clz`: number of leading-zero bits, in `[0, N]` (all-zero → N).
    ///   * `Cls`: number of consecutive bits below the MSB that equal the
    ///     MSB, in `[0, N-1]` (all-same → N-1).
    ///
    /// Implementation mirrors claripy's `_op_generic_Clz` ITE-chain pattern
    /// (irop.py L700) but applied per-lane. For Cls, the chain runs over the
    /// non-MSB bits and the comparison is against the lane's MSB.
    pub(super) fn vec_lane_count(
        arg: RustBV,
        elem: IRType,
        count: u8,
        kind: LaneCountKind,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total = elem_width * count as u32;
        Self::require_operand_width("vec_lane_count arg", arg.width(), total)?;

        // Concrete fast path. Lane-generic up to `elem_width == 64` (the
        // widest shape libVEX emits, Iop_Clz64x2): every correction below is
        // written against `elem_width`, and `as_u128()` covers the whole
        // vector because total width stays ≤ 128.
        if let Some(v) = arg.as_u128() {
            let mask: u128 = Self::low_bit_mask_u128(elem_width);
            let mut result: u128 = 0;
            for i in 0..count as u32 {
                let lane = (v >> (i * elem_width)) & mask;
                let res_lane = match kind {
                    LaneCountKind::Clz => {
                        if lane == 0 {
                            elem_width as u128
                        } else {
                            // u128 leading zeros minus the padding above elem_width.
                            (lane.leading_zeros() - (128 - elem_width)) as u128
                        }
                    }
                    LaneCountKind::Cls => {
                        let sign = (lane >> (elem_width - 1)) & 1;
                        // Flip if sign is 1 so we count leading zeros of the
                        // result; then strip the MSB and clz it.
                        let flipped = if sign == 1 { (!lane) & mask } else { lane };
                        // Clear the MSB and look at the bits below.
                        let body = flipped & (mask >> 1);
                        if body == 0 {
                            // All bits below MSB matched sign → result = N-1.
                            (elem_width - 1) as u128
                        } else {
                            // leading_zeros of body within (elem_width-1) bits.
                            // body has its MSB-1 bit at position elem_width-2.
                            (body.leading_zeros() - (128 - (elem_width - 1))) as u128
                        }
                    }
                };
                result |= res_lane << (i * elem_width);
            }
            return Ok(RustBV::concrete(result, total));
        }

        // Symbolic ITE-chain per lane (mirrors claripy's _op_generic_Clz shape).
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count as u32 {
            let lane_lo = i * elem_width;
            let lane_hi = lane_lo + elem_width - 1;
            let lane = arg.extract(lane_hi, lane_lo, ctx);
            let res = match kind {
                LaneCountKind::Clz => Self::clz_chain(lane, elem_width, ctx),
                LaneCountKind::Cls => Self::cls_chain(lane, elem_width, ctx),
            };
            elements.push(res);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Build an ITE chain that returns clz of an `n`-bit lane in `n` bits.
    /// Mirrors `_op_generic_Clz` in `angr/engines/vex/claripy/irop.py`.
    fn clz_chain(lane: RustBV, n: u32, ctx: &SymContext) -> RustBV {
        let mut expr = RustBV::concrete(n as u128, n);
        let one = RustBV::concrete(1, 1);
        for a in 0..n {
            let bit = lane.extract(a, a, ctx);
            let cond = bit.eq_into(one.clone(), ctx);
            let then_v = RustBV::concrete((n - a - 1) as u128, n);
            expr = cond.ite_into(then_v, expr, ctx);
        }
        expr
    }

    /// Build an ITE chain that returns cls (count leading sign bits, excluding
    /// the MSB) of an `n`-bit lane in `n` bits. Default value is `n-1` (all
    /// bits below the MSB match the MSB); the chain returns `n - 2 - a` for
    /// the highest non-MSB position `a` whose bit differs from the MSB.
    fn cls_chain(lane: RustBV, n: u32, ctx: &SymContext) -> RustBV {
        let sign = lane.extract(n - 1, n - 1, ctx);
        let mut expr = RustBV::concrete((n - 1) as u128, n);
        for a in 0..(n - 1) {
            let bit = lane.extract(a, a, ctx);
            let cond = bit.ne_into(sign.clone(), ctx);
            let then_v = RustBV::concrete((n - 2 - a) as u128, n);
            expr = cond.ite_into(then_v, expr, ctx);
        }
        expr
    }
}
