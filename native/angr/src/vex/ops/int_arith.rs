//! Integer widening-multiply and divmod VEX op helpers.

use super::{OpError, VEXOps};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ir::IRType;

impl VEXOps {
    /// Widening multiply: both operands must be `ty`-width; the result is
    /// double that width.
    #[inline]
    pub(super) fn widening_mul(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let in_width = ty.bits();
        Self::require_operand_width("widening_mul left", left.width(), in_width)?;
        Self::require_operand_width("widening_mul right", right.width(), in_width)?;
        let out_width = in_width * 2;

        // Extend both operands
        let left_ext = left.extend_into(out_width, signed, ctx);
        let right_ext = right.extend_into(out_width, signed, ctx);

        Ok(left_ext.mul_into(right_ext, ctx))
    }

    /// DivMod: 64-bit dividend / 32-bit divisor -> 64-bit result.
    /// Low 32 bits = quotient, High 32 bits = remainder.
    pub(super) fn divmod_64_to_32(
        dividend: RustBV,
        divisor: RustBV,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::require_operand_width("divmod_64_to_32 dividend", dividend.width(), 64)?;
        Self::require_operand_width("divmod_64_to_32 divisor", divisor.width(), 32)?;
        Self::divmod_double_to_single(dividend, divisor, signed, ctx)
    }

    /// DivMod: 128-bit dividend / 64-bit divisor -> 128-bit result.
    /// Low 64 bits = quotient, High 64 bits = remainder.
    /// `Iop_DivMod{U,S}{32to32,64to64}`: both operands are `ty` wide, the
    /// result is `Concat(rem, quot)` at twice that. angr's Python `_op_divmod`
    /// divides at the operand width; extending the dividend (zero- or
    /// sign-, by `signed`) to double width and dividing by the `ty`-wide
    /// divisor gives the same low halves, division by zero and `MIN / -1`
    /// included, so this reuses [`Self::divmod_double_to_single`].
    pub(super) fn divmod_same_width(
        dividend: RustBV,
        divisor: RustBV,
        ty: IRType,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let bits = ty.bits();
        Self::require_operand_width("divmod_same_width dividend", dividend.width(), bits)?;
        Self::require_operand_width("divmod_same_width divisor", divisor.width(), bits)?;
        let wide = 2 * bits; // overflow-ok: an IRType width, 64 at most here
        let dividend = if signed {
            dividend.sign_extend_into(wide, ctx)
        } else {
            dividend.zero_extend_into(wide, ctx)
        };
        Self::divmod_double_to_single(dividend, divisor, signed, ctx)
    }

    pub(super) fn divmod_128_to_64(
        dividend: RustBV,
        divisor: RustBV,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::require_operand_width("divmod_128_to_64 dividend", dividend.width(), 128)?;
        Self::require_operand_width("divmod_128_to_64 divisor", divisor.width(), 64)?;
        Self::divmod_double_to_single(dividend, divisor, signed, ctx)
    }

    /// Generic DivMod: dividend (width = `2 * divisor.width()`) divided by
    /// divisor; returns a value of the dividend's width packed as
    /// `low half = quotient, high half = remainder`. Both halves are the
    /// divisor's width.
    ///
    /// Z3 defines div/mod by zero totally (udiv→all-ones, urem→dividend,
    /// sdiv→±1, srem→dividend), matching claripy, so the symbolic path
    /// needs no explicit zero guard. The concrete path reproduces those
    /// same totals rather than trapping, so a zero divisor yields the same
    /// value whether the operands arrived concrete or symbolic.
    fn divmod_double_to_single(
        dividend: RustBV,
        divisor: RustBV,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let dividend_w = dividend.width();
        let divisor_w = divisor.width();
        Self::require_operand_width(
            "divmod_double_to_single dividend",
            dividend_w,
            divisor_w * 2,
        )?;

        if let (Some(dvd), Some(dvs)) = (dividend.as_u128(), divisor.as_u128()) {
            let dvd = dvd & Self::low_bit_mask_u128(dividend_w);
            let dvs = dvs & Self::low_bit_mask_u128(divisor_w);

            let half_mask = Self::low_bit_mask_u128(divisor_w);

            if dvs == 0 {
                // Mirror the symbolic arm below, which divides the full-width
                // dividend by a zero-extended zero divisor and takes the low
                // `divisor_w` bits of each Z3 total: quotient = all-ones
                // (unsigned, or signed with a non-negative dividend) / +1
                // (signed, negative dividend); remainder = the dividend.
                let quotient = if signed && Self::sign_extend_low_to_i128(dvd, dividend_w) < 0 {
                    1
                } else {
                    half_mask
                };
                let remainder = dvd & half_mask;
                return Ok(RustBV::concrete(
                    quotient | (remainder << divisor_w),
                    dividend_w,
                ));
            }

            let (quotient, remainder) = if signed {
                let dvd_i = Self::sign_extend_low_to_i128(dvd, dividend_w);
                let dvs_i = Self::sign_extend_low_to_i128(dvs, divisor_w);
                // `i128::MIN / -1` overflows and Rust's checked division panics
                // even in release; with `panic = "abort"` that SIGABRTs the whole
                // process. `wrapping_div`/`wrapping_rem` return the two's-complement
                // wrap (i128::MIN, 0) matching Z3 bvsdiv/bvsrem semantics.
                let q = dvd_i.wrapping_div(dvs_i) as u128 & half_mask;
                let r = dvd_i.wrapping_rem(dvs_i) as u128 & half_mask;
                (q, r)
            } else {
                // libVEX guards only a zero divisor, so the quotient can
                // overflow `divisor_w` (amd64 `div` would raise #DE). Keep
                // its low half, like the symbolic arm's extract and the
                // Python engine, so it cannot OR into the remainder half
                // (angr-vmi2l.58).
                ((dvd / dvs) & half_mask, dvd % dvs)
            };

            let result = quotient | (remainder << divisor_w);
            return Ok(RustBV::concrete(result, dividend_w));
        }

        let divisor_full = divisor.extend_into(dividend_w, signed, ctx);
        let quotient_full = if signed {
            dividend.sdiv(&divisor_full, ctx)
        } else {
            dividend.udiv(&divisor_full, ctx)
        };
        let remainder_full = if signed {
            dividend.srem(&divisor_full, ctx)
        } else {
            dividend.urem(&divisor_full, ctx)
        };
        let quotient_half = quotient_full.extract_into(divisor_w - 1, 0, ctx);
        let remainder_half = remainder_full.extract_into(divisor_w - 1, 0, ctx);
        Ok(remainder_half.concat_into(quotient_half, ctx))
    }

    /// Mask covering the low `width` bits of a u128.
    #[inline]
    pub(super) fn low_bit_mask_u128(width: u32) -> u128 {
        if width >= 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        }
    }
}
