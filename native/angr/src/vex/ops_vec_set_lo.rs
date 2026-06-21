//! V128 low-lane insertion VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via `#[path]` in
//! ops.rs), so these `pub(super)` methods stay callable from the `binop`
//! dispatch in `ops`.
//!
//! Covers the SSE scalar-into-V128 insertion family that overwrites the low
//! 32 or 64 bits of a 128-bit vector while preserving the upper lanes
//! (Iop_SetV128lo32 / Iop_SetV128lo64).

use super::{OpError, VEXOps};
use crate::symbolic::{RustBV, SymContext};

impl VEXOps {
    /// Splice the low `lane_bits` of `lane` into lane 0 of the packed 128-bit
    /// vector `orig`, preserving the upper `128 - lane_bits` bits. Shared
    /// concrete path for every SSE scalar-in-V128 op that overwrites lane 0 and
    /// passes the rest through (SetV128lo32/64, ADDSS/SUBSS/MULSS/DIVSS + F64,
    /// SQRTSS/SQRTSD, MAXSS/MINSS). `lane`'s significant bits are assumed to fit
    /// in `lane_bits`; the `& mask` makes that explicit and harmless.
    #[inline]
    pub(super) fn splice_lane0_u128(orig: u128, lane: u128, lane_bits: u32) -> u128 {
        let mask = Self::low_bit_mask_u128(lane_bits);
        (orig & !mask) | (lane & mask)
    }

    /// Insert a `val.width()`-bit scalar into the low lane of a 128-bit vector,
    /// preserving the upper `128 - val.width()` bits. Backs Iop_SetV128lo32
    /// (`val` 32-bit) and Iop_SetV128lo64 (`val` 64-bit).
    fn set_v128_lo(vec: RustBV, val: RustBV, ctx: &SymContext) -> Result<RustBV, OpError> {
        debug_assert_eq!(vec.width(), 128);
        let k = val.width();

        if let (Some(v), Some(lo)) = (vec.as_u128(), val.as_u128()) {
            let result = Self::splice_lane0_u128(v, lo, k);
            return Ok(RustBV::concrete(result, 128));
        }
        // For symbolic, concatenate the preserved upper bits with the value.
        let upper = vec.extract(127, k, ctx);
        Ok(upper.concat(&val, ctx))
    }

    /// Set low 32 bits of V128.
    pub(super) fn set_v128_lo32(
        vec: RustBV,
        val: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(val.width(), 32);
        Self::set_v128_lo(vec, val, ctx)
    }

    /// Set low 64 bits of V128.
    pub(super) fn set_v128_lo64(
        vec: RustBV,
        val: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(val.width(), 64);
        Self::set_v128_lo(vec, val, ctx)
    }
}
