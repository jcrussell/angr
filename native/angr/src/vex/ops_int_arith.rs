//! Integer widening-multiply and divmod VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via `#[path]` in
//! ops.rs), so these `pub(super)` methods stay callable from the binop
//! dispatch in `ops`, and the shared helpers they call that remain in
//! `ops` (`Self::sign_extend_low_to_i128`) stay visible by the
//! descendant-module rule.

use super::{OpError, VEXOps};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ir::IRType;

impl VEXOps {
    /// Widening multiply.
    #[inline]
    pub(super) fn widening_mul(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let in_width = ty.bits();
        let out_width = in_width * 2;

        // Extend both operands
        let left_ext = left.extend_into(out_width, signed, ctx);
        let right_ext = right.extend_into(out_width, signed, ctx);

        Ok(left_ext.mul_into(right_ext, ctx))
    }

    /// High half of multiplication.
    #[inline]
    pub(super) fn mul_hi(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let width = ty.bits();
        let double_width = width * 2;

        // Extend and multiply
        let left_ext = left.extend_into(double_width, signed, ctx);
        let right_ext = right.extend_into(double_width, signed, ctx);

        let product = left_ext.mul_into(right_ext, ctx);

        // Extract high half
        Ok(product.extract_into(double_width - 1, width, ctx))
    }

    /// DivMod: 64-bit dividend / 32-bit divisor -> 64-bit result.
    /// Low 32 bits = quotient, High 32 bits = remainder.
    pub(super) fn divmod_64_to_32(
        dividend: RustBV,
        divisor: RustBV,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(dividend.width(), 64);
        debug_assert_eq!(divisor.width(), 32);
        Self::divmod_double_to_single(dividend, divisor, signed, ctx)
    }

    /// DivMod: 128-bit dividend / 64-bit divisor -> 128-bit result.
    /// Low 64 bits = quotient, High 64 bits = remainder.
    pub(super) fn divmod_128_to_64(
        dividend: RustBV,
        divisor: RustBV,
        signed: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(dividend.width(), 128);
        debug_assert_eq!(divisor.width(), 64);
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
        debug_assert_eq!(dividend_w, divisor_w * 2);

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
                (dvd / dvs, dvd % dvs)
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
