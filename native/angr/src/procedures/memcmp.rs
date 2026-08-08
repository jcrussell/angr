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

use super::check_max;
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
        check_max(n, MAX_STRCMP_LEN)?;
        compare_bytes(state, s1, s2, n,
                      /*stop_at_null=*/false, /*case_insensitive=*/false)
    }
}

test_submod!("memcmp_tests.rs" => memcmp_tests);
