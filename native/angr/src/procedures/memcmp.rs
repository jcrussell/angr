//! Native memcmp implementation.
//!
//! memcmp compares n bytes of two memory regions.
//!
//! # Behavior
//!
//! - Concrete addresses are required (symbolic addresses fall back to Python).
//! - Concrete `n` is required (symbolic n falls back).
//! - Concrete bytes scan with short-circuit on first mismatch (matches libc).
//! - Symbolic bytes produce a 32-bit ITE chain via `compare_bytes` shared
//!   with strcmp/strncmp (with stop_at_null=false).

use super::ProcedureError;
use super::strcmp::{MAX_STRCMP_LEN, compare_bytes};
use crate::symbolic::RustBV;

crate::declare_proc! {
    /// Native memcmp: `int memcmp(const void *s1, const void *s2, size_t n)`.
    ///
    /// Returns < 0, 0, or > 0 per byte-wise comparison.
    name = "memcmp",
    struct = NativeMemcmp,
    args = [s1: concrete, s2: concrete, n: concrete],
    call |state| {
        if n == 0 {
            return Ok(Some(RustBV::zero(32)));
        }
        if n > MAX_STRCMP_LEN as u64 {
            return Err(ProcedureError::MaxIterations(MAX_STRCMP_LEN));
        }
        compare_bytes(state, s1, s2, n,
                      /*stop_at_null=*/false, /*case_insensitive=*/false)
    }
}

#[cfg(test)]
#[path = "memcmp_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod memcmp_tests;
