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
    /// Insert a `val.width()`-bit scalar into the low lane of a 128-bit vector,
    /// preserving the upper `128 - val.width()` bits. Backs Iop_SetV128lo32
    /// (`val` 32-bit) and Iop_SetV128lo64 (`val` 64-bit).
    fn set_v128_lo(vec: RustBV, val: RustBV, ctx: &SymContext) -> Result<RustBV, OpError> {
        debug_assert_eq!(vec.width(), 128);
        let k = val.width();
        let mask: u128 = if k >= 128 {
            u128::MAX
        } else {
            (1u128 << k) - 1
        };

        if let (Some(v), Some(lo)) = (vec.as_u128(), val.as_u128()) {
            let result = (v & !mask) | (lo & mask);
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
