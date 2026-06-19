//! NEON/SIMD integer saturation VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via `#[path]` in
//! ops.rs), so these `pub(super)` methods stay callable from the unop/binop
//! dispatch in `ops`, and the shared sibling methods they call
//! (`Self::sign_extend_low_to_i128`, `Self::concat_le_elements`) stay visible
//! by the descendant-module rule. The lane-level clamp helpers
//! (`saturate_lane`, `saturate_lane_symbolic`) are private to this module.
//!
//! Covers the saturating narrow family (Iop_QNarrowUn/QNarrowBin), per-lane
//! saturating add/sub (Iop_QAdd/QSub), and saturating shift-left
//! (Iop_QShl/QSal).

use super::{OpError, VEXOps};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ir::IRType;

impl VEXOps {
    /// Saturate one concrete `from_width`-bit lane into a `to_width`-bit
    /// result. `src_signed` interprets the input; `dst_signed` selects the
    /// output range. Returns the truncated unsigned bit pattern.
    #[inline]
    fn saturate_lane(
        lane: u128,
        from_width: u32,
        to_width: u32,
        src_signed: bool,
        dst_signed: bool,
    ) -> u128 {
        let to_mask: u128 = (1u128 << to_width) - 1;
        // Reinterpret the source lane as i128.
        let val_i: i128 = if src_signed {
            Self::sign_extend_low_to_i128(lane, from_width)
        } else {
            // unsigned source — masking width-bits into i128 keeps it
            // non-negative because from_width <= 64 in all NEON QNarrow ops.
            (lane
                & if from_width == 128 {
                    u128::MAX
                } else {
                    (1u128 << from_width) - 1
                }) as i128
        };
        let (min_i, max_i): (i128, i128) = if dst_signed {
            let half = 1i128 << (to_width - 1);
            (-half, half - 1)
        } else {
            (0i128, ((1u128 << to_width) - 1) as i128)
        };
        let clamped = val_i.clamp(min_i, max_i);
        (clamped as u128) & to_mask
    }

    /// NEON unary saturating narrow (Iop_QNarrowUn{N}{S/U}to{N/2}{S/U}x{M}).
    pub(super) fn vec_qnarrow_un(
        arg: RustBV,
        from: IRType,
        count: u8,
        src_signed: bool,
        dst_signed: bool,
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
            let in_mask: u128 = if from_width == 128 {
                u128::MAX
            } else {
                (1u128 << from_width) - 1
            };
            let mut result: u128 = 0;
            for i in 0..count {
                let lo = (i as u32) * from_width;
                let lane = (v >> lo) & in_mask;
                let sat = Self::saturate_lane(lane, from_width, to_width, src_signed, dst_signed);
                let out_lo = (i as u32) * to_width;
                result |= sat << out_lo;
            }
            return Ok(RustBV::concrete(result, to_width * count as u32));
        }

        // Symbolic: per-lane ITE clamp.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * from_width;
            let hi = lo + from_width - 1;
            let lane = arg.extract(hi, lo, ctx);
            elements.push(Self::saturate_lane_symbolic(
                lane, from_width, to_width, src_signed, dst_signed, ctx,
            ));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON binary saturating narrow (Iop_QNarrowBin{N}{S/U}to{N/2}{S/U}x{M}).
    pub(super) fn vec_qnarrow_bin(
        left: RustBV,
        right: RustBV,
        from: IRType,
        count: u8,
        src_signed: bool,
        dst_signed: bool,
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
            let in_mask: u128 = if from_width == 128 {
                u128::MAX
            } else {
                (1u128 << from_width) - 1
            };
            let mut result: u128 = 0;
            for i in 0..per_input {
                let lo = i * from_width;
                let lane_l = (l >> lo) & in_mask;
                let lane_r = (r >> lo) & in_mask;
                let sat_l =
                    Self::saturate_lane(lane_l, from_width, to_width, src_signed, dst_signed);
                let sat_r =
                    Self::saturate_lane(lane_r, from_width, to_width, src_signed, dst_signed);
                let out_lo_l = i * to_width;
                let out_lo_r = (per_input + i) * to_width;
                result |= sat_l << out_lo_l;
                result |= sat_r << out_lo_r;
            }
            return Ok(RustBV::concrete(result, to_width * count as u32));
        }

        // Symbolic: per-lane clamp from each operand, concatenate.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..per_input {
            let lo = i * from_width;
            let hi = lo + from_width - 1;
            let lane = left.extract(hi, lo, ctx);
            elements.push(Self::saturate_lane_symbolic(
                lane, from_width, to_width, src_signed, dst_signed, ctx,
            ));
        }
        for i in 0..per_input {
            let lo = i * from_width;
            let hi = lo + from_width - 1;
            let lane = right.extract(hi, lo, ctx);
            elements.push(Self::saturate_lane_symbolic(
                lane, from_width, to_width, src_signed, dst_signed, ctx,
            ));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// Symbolic clamp of one `from_width`-bit lane into `to_width` bits.
    /// Builds an ITE: `if lane > max -> max; else if lane < min -> min; else lane[low to_width]`.
    fn saturate_lane_symbolic(
        lane: RustBV,
        from_width: u32,
        to_width: u32,
        src_signed: bool,
        dst_signed: bool,
        ctx: &SymContext,
    ) -> RustBV {
        debug_assert_eq!(lane.width(), from_width);

        // Build the saturation range constants in `from_width` bits so the
        // comparison ops have matching widths.
        let (max_val, min_val): (u128, u128) = if dst_signed {
            let half = 1u128 << (to_width - 1);
            // max = 2^(to_width-1) - 1, min = -2^(to_width-1)
            // In `from_width` bits (two's complement): min = (-half) & mask
            let from_mask = if from_width == 128 {
                u128::MAX
            } else {
                (1u128 << from_width) - 1
            };
            let max = half - 1;
            let min = (!(half - 1) + 1) & from_mask; // -half in from_width bits
            (max, min)
        } else {
            // Unsigned dst: [0, 2^to_width - 1]
            let max = (1u128 << to_width) - 1;
            (max, 0)
        };

        let max_bv = RustBV::concrete(max_val, from_width);
        let min_bv = RustBV::concrete(min_val, from_width);

        // gt_max: compare with src_signed semantics
        let gt_max = if src_signed {
            lane.sgt(&max_bv, ctx)
        } else {
            lane.ugt(&max_bv, ctx)
        };

        // lt_min: only meaningful when min could be < lane. If src is unsigned
        // and dst is unsigned, min == 0 so lt_min is always false; skip to
        // truncate the upper-clamped value.
        let truncated = lane.extract(to_width - 1, 0, ctx);
        let max_truncated = max_bv.extract(to_width - 1, 0, ctx);

        let upper_clamped = gt_max.ite(&max_truncated, &truncated, ctx);

        if !src_signed && !dst_signed {
            return upper_clamped;
        }

        let lt_min = if src_signed {
            lane.slt(&min_bv, ctx)
        } else {
            lane.ult(&min_bv, ctx)
        };
        let min_truncated = min_bv.extract(to_width - 1, 0, ctx);

        lt_min.ite(&min_truncated, &upper_clamped, ctx)
    }

    /// NEON per-lane saturating integer add/sub —
    /// Iop_QAdd{N}{S/U}x{M} / Iop_QSub{N}{S/U}x{M}.
    ///
    /// Mirrors the claripy reference at
    /// `angr/engines/vex/claripy/irop.py::_op_generic_QAdd` (signed):
    ///   * Detect overflow with sign-bit algebra:
    ///     QAdd: `(~(top_a ^ top_b)) & (top_a ^ top_r)` — both inputs same
    ///     sign, result flips → overflow.
    ///     QSub: `( (top_a ^ top_b)) & (top_a ^ top_r)` — inputs differ,
    ///     result's sign differs from minuend → overflow.
    ///   * Saturated cap: `INT_MAX + ~top_r` — yields INT_MAX when the result
    ///     would be "too positive" (top_r=0 → +1 wraps to INT_MIN) and INT_MIN
    ///     when "too negative" (top_r=1 → +0 keeps INT_MAX). Actually:
    ///     top_r=1 (negative result, meaning positive overflow) → ~top_r=0
    ///     → cap = INT_MAX.
    ///     top_r=0 (positive result, meaning negative overflow) → ~top_r=1
    ///     → cap = INT_MAX + 1 = INT_MIN (two's complement wrap).
    ///
    /// Unsigned QAdd: overflow iff `res < a` (carry); cap = UINT_MAX.
    /// Unsigned QSub: overflow iff `res > a` (borrow); cap = 0.
    pub(super) fn vec_int_saturating(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        signed: bool,
        is_sub: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);

        // Concrete fast path (fits in u128 — total_width <= 128 covers all
        // currently-mapped NEON shapes).
        if total_width <= 128
            && let (Some(l), Some(r)) = (left.as_u128(), right.as_u128())
        {
            let mut result: u128 = 0;
            let elem_mask: u128 = if elem_width == 128 {
                u128::MAX
            } else {
                (1u128 << elem_width) - 1
            };
            let sign_bit: u128 = 1u128 << (elem_width - 1);
            let smax: u128 = sign_bit - 1; // 0x7F... in elem_width bits
            let smin: u128 = sign_bit; // 0x80...
            let umax: u128 = elem_mask;

            for i in 0..count {
                let lo = (i as u32) * elem_width;
                let a = (l >> lo) & elem_mask;
                let b = (r >> lo) & elem_mask;

                let sat = if signed {
                    // Sign-extend each lane to i128 to compute the true
                    // arithmetic result, then clamp into [-smax-1, smax].
                    let a_signed = if a & sign_bit != 0 {
                        (a | !elem_mask) as i128
                    } else {
                        a as i128
                    };
                    let b_signed = if b & sign_bit != 0 {
                        (b | !elem_mask) as i128
                    } else {
                        b as i128
                    };
                    let raw: i128 = if is_sub {
                        a_signed - b_signed
                    } else {
                        a_signed + b_signed
                    };
                    let max_signed = smax as i128;
                    let min_signed = -(sign_bit as i128);
                    if raw > max_signed {
                        smax
                    } else if raw < min_signed {
                        smin
                    } else {
                        (raw as u128) & elem_mask
                    }
                } else if is_sub {
                    // Unsigned subtract: clamp underflow to 0.
                    if a >= b { (a - b) & elem_mask } else { 0 }
                } else {
                    // Unsigned add: clamp overflow to UINT_MAX.
                    let raw = a + b; // both < 2^N, sum < 2^(N+1) ≤ 2^128
                    if raw > umax { umax } else { raw }
                };

                result |= (sat & elem_mask) << lo;
            }
            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic per-lane fallback. Translates the claripy algorithm:
        //   top_x = lane[N-1]; res = a ± b; top_r = res[N-1].
        //   signed:   cap_cond = (xor_sign_match ^ (top_a XOR top_r)) == 1.
        //             cap      = (-1)/2 + ~top_r   (signed semantics).
        //   unsigned add: cap_cond = ULT(res, a); cap = -1.
        //   unsigned sub: cap_cond = UGT(res, a); cap =  0.
        let smax_bv = RustBV::concrete(
            ((1u128 << (elem_width - 1)) - 1) & ((!0u128) >> (128 - elem_width)),
            elem_width,
        );
        let smin_bv = RustBV::concrete(
            (1u128 << (elem_width - 1)) & ((!0u128) >> (128 - elem_width)),
            elem_width,
        );
        let umax_bv = RustBV::concrete((!0u128) >> (128 - elem_width), elem_width);
        let zero_bv = RustBV::concrete(0, elem_width);

        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let a = left.extract(hi, lo, ctx);
            let b = right.extract(hi, lo, ctx);

            let res = if is_sub {
                a.clone().sub_into(b.clone(), ctx)
            } else {
                a.clone().add_into(b.clone(), ctx)
            };

            let lane = if signed {
                // Sign-bit relationships dictate the cap and the overflow flag.
                let top_a = a.extract(elem_width - 1, elem_width - 1, ctx);
                let top_b = b.extract(elem_width - 1, elem_width - 1, ctx);
                let top_r = res.extract(elem_width - 1, elem_width - 1, ctx);

                // QAdd: ~(a ^ b) & (a ^ r); QSub: (a ^ b) & (a ^ r).
                let a_xor_b = top_a.clone().xor_into(top_b, ctx);
                let a_xor_r = top_a.xor_into(top_r.clone(), ctx);
                let lhs = if is_sub {
                    a_xor_b
                } else {
                    a_xor_b.not_into(ctx)
                };
                let overflow_flag = lhs.and_into(a_xor_r, ctx);
                let cap_cond = overflow_flag.eq(&RustBV::concrete(1, 1), ctx);

                // cap = INT_MAX when top_r = 1 (positive overflow → clamp high),
                // cap = INT_MIN when top_r = 0 (negative overflow → clamp low).
                let top_r_one = top_r.eq(&RustBV::concrete(1, 1), ctx);
                let cap = top_r_one.ite(&smax_bv, &smin_bv, ctx);

                cap_cond.ite(&cap, &res, ctx)
            } else if is_sub {
                let cap_cond = res.ugt(&a, ctx);
                cap_cond.ite(&zero_bv, &res, ctx)
            } else {
                let cap_cond = res.ult(&a, ctx);
                cap_cond.ite(&umax_bv, &res, ctx)
            };

            elements.push(lane);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// NEON saturating shift-left by vector — `Iop_QShl{N}x{M}` (unsigned,
    /// `signed=false`) / `Iop_QSal{N}x{M}` (signed, `signed=true`). Maps to
    /// ARM UQSHL / SQSHL (DDI 0487 C7.2.327 / C7.2.298). Per-lane semantics
    /// (`amt` = shift-amount lane, sign-extended; `a` = data lane):
    ///   * `amt >= 0`: left shift by `amt`; saturate to UMAX (unsigned) or
    ///     SMAX/SMIN (signed, based on sign of `a`). Counts ≥ lane width
    ///     saturate unless `a == 0`.
    ///   * `amt < 0`: right shift by `-amt`; logical (unsigned) or arithmetic
    ///     (signed). Counts ≥ lane width collapse to 0 / sign-fill.
    ///
    /// Derived from libVEX `host_generic_simd*` h_generic_calc_QShl* helpers;
    /// claripy has no `_op_generic_QShl` / `_op_generic_QSal` reference.
    pub(super) fn vec_qshl_sat(
        vec: RustBV,
        amts: RustBV,
        elem: IRType,
        count: u8,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(vec.width(), total_width);
        debug_assert_eq!(amts.width(), total_width);

        // Concrete fast path: both operands fit in u128 (covers every NEON
        // shape we route here — 64- and 128-bit vectors).
        if total_width <= 128
            && let (Some(v), Some(s)) = (vec.as_u128(), amts.as_u128())
        {
            let mut result: u128 = 0;
            let elem_mask: u128 = if elem_width == 128 {
                u128::MAX
            } else {
                (1u128 << elem_width) - 1
            };
            let sign_bit: u128 = 1u128 << (elem_width - 1);
            let smax: u128 = sign_bit - 1; // 0x7F...
            let smin: u128 = sign_bit; // 0x80...
            let umax: u128 = elem_mask;

            for i in 0..count {
                let lo = (i as u32) * elem_width;
                let a = (v >> lo) & elem_mask;
                let amt_raw = (s >> lo) & elem_mask;
                // Sign-extend the amt lane: ARM SQSHL/UQSHL treat the
                // shift-amount lane as signed (negative → right shift).
                let amt_signed: i128 = if amt_raw & sign_bit != 0 {
                    (amt_raw | !elem_mask) as i128
                } else {
                    amt_raw as i128
                };

                let sat = if amt_signed >= 0 {
                    let shift_amt = amt_signed as u32;
                    if signed {
                        // Sal: signed left shift with overflow → SMAX/SMIN.
                        let a_signed = if a & sign_bit != 0 {
                            (a | !elem_mask) as i128
                        } else {
                            a as i128
                        };
                        let smax_i = smax as i128;
                        let smin_i = -(sign_bit as i128);
                        if shift_amt >= elem_width {
                            // Out-of-range left shift: any nonzero a → saturate.
                            if a_signed > 0 {
                                smax
                            } else if a_signed < 0 {
                                smin
                            } else {
                                0
                            }
                        } else {
                            // i128 shift never overflows for our widths
                            // (max elem_width = 64, shift_amt < 64, so
                            // |a_signed| < 2^63 → |raw| < 2^127).
                            let raw = a_signed << shift_amt;
                            if raw > smax_i {
                                smax
                            } else if raw < smin_i {
                                smin
                            } else {
                                (raw as u128) & elem_mask
                            }
                        }
                    } else {
                        // Shl: unsigned left shift with overflow → UMAX.
                        if shift_amt >= elem_width {
                            if a != 0 { umax } else { 0 }
                        } else {
                            // a < 2^elem_width and shift_amt < elem_width,
                            // so raw fits in u128 (elem_width ≤ 64).
                            let raw = a << shift_amt;
                            if (raw & !elem_mask) != 0 {
                                umax
                            } else {
                                raw & elem_mask
                            }
                        }
                    }
                } else {
                    // amt < 0 → right shift by -amt.
                    let r_amt = (-amt_signed) as u32;
                    if signed {
                        // Ashr: out-of-range → sign-fill.
                        let neg = a & sign_bit != 0;
                        if r_amt >= elem_width {
                            if neg { elem_mask } else { 0 }
                        } else if neg {
                            let shifted_val = a >> r_amt;
                            let fill_mask =
                                (elem_mask << (elem_width as u128 - r_amt as u128)) & elem_mask;
                            (shifted_val | fill_mask) & elem_mask
                        } else {
                            a >> r_amt
                        }
                    } else {
                        // Lshr: out-of-range → 0.
                        if r_amt >= elem_width { 0 } else { a >> r_amt }
                    }
                };

                result |= (sat & elem_mask) << lo;
            }
            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic per-lane fallback. The Z3 bvshl/bvlshr/bvashr semantics
        // (count ≥ width → 0 or sign-fill) match the "out of range" branches
        // of QShl/QSal, so no explicit width guards are needed; overflow is
        // detected by the `(shl >> amt) != a` round-trip check.
        let smax_bv = RustBV::concrete(
            ((1u128 << (elem_width - 1)) - 1) & ((!0u128) >> (128 - elem_width)),
            elem_width,
        );
        let smin_bv = RustBV::concrete(
            (1u128 << (elem_width - 1)) & ((!0u128) >> (128 - elem_width)),
            elem_width,
        );
        let umax_bv = RustBV::concrete((!0u128) >> (128 - elem_width), elem_width);
        let zero_bv = RustBV::concrete(0, elem_width);
        let bit_one = RustBV::concrete(1, 1);

        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let a = vec.extract(hi, lo, ctx);
            let b = amts.extract(hi, lo, ctx);

            // Left-shift branch (amt ≥ 0). Both signed and unsigned saturate
            // when the round-trip `(a << amt) >> amt` differs from `a`.
            let shl_res = a.clone().shl_into(b.clone(), ctx);

            let left_result = if signed {
                // Ashr round-trip detects signed overflow (and OOR shifts).
                let recovered = shl_res.clone().ashr_into(b.clone(), ctx);
                let no_overflow = recovered.eq(&a, ctx);
                let a_top = a.extract(elem_width - 1, elem_width - 1, ctx);
                let a_is_neg = a_top.eq(&bit_one, ctx);
                let cap = a_is_neg.ite(&smin_bv, &smax_bv, ctx);
                no_overflow.ite(&shl_res, &cap, ctx)
            } else {
                // Lshr round-trip detects unsigned overflow.
                let recovered = shl_res.clone().lshr_into(b.clone(), ctx);
                let no_overflow = recovered.eq(&a, ctx);
                no_overflow.ite(&shl_res, &umax_bv, ctx)
            };

            // Right-shift branch (amt < 0). Use -b as the count; Z3 handles
            // OOR (count ≥ width → 0 or sign-fill) natively.
            let neg_amt = zero_bv.clone().sub_into(b.clone(), ctx);
            let right_result = if signed {
                a.clone().ashr_into(neg_amt, ctx)
            } else {
                a.clone().lshr_into(neg_amt, ctx)
            };

            // Dispatch on the sign of amt (top bit).
            let amt_top = b.extract(elem_width - 1, elem_width - 1, ctx);
            let amt_is_neg = amt_top.eq(&bit_one, ctx);
            let lane = amt_is_neg.ite(&right_result, &left_result, ctx);

            elements.push(lane);
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }
}
