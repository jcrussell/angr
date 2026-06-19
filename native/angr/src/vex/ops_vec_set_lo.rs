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
    /// Set low 32 bits of V128.
    pub(super) fn set_v128_lo32(
        vec: RustBV,
        val: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(vec.width(), 128);
        debug_assert_eq!(val.width(), 32);

        if let (Some(v), Some(lo)) = (vec.as_u128(), val.as_u128()) {
            let upper = v & !0xFFFFFFFFu128;
            let result = upper | (lo & 0xFFFFFFFF);
            return Ok(RustBV::concrete(result, 128));
        }
        // For symbolic, concatenate upper 96 bits with the value
        let upper = vec.extract(127, 32, ctx);
        let result = upper.concat(&val, ctx);
        Ok(result)
    }

    /// Set low 64 bits of V128.
    pub(super) fn set_v128_lo64(
        vec: RustBV,
        val: RustBV,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(vec.width(), 128);
        debug_assert_eq!(val.width(), 64);

        if let (Some(v), Some(lo)) = (vec.as_u128(), val.as_u128()) {
            let upper = v & !0xFFFFFFFFFFFFFFFFu128;
            let result = upper | (lo & 0xFFFFFFFFFFFFFFFF);
            return Ok(RustBV::concrete(result, 128));
        }
        // For symbolic, concatenate upper 64 bits with the value
        let upper = vec.extract(127, 64, ctx);
        let result = upper.concat(&val, ctx);
        Ok(result)
    }
}
