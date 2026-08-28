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

use super::strcmp::compare_bytes;

crate::declare_proc! {
    /// Native memcmp: `int memcmp(const void *s1, const void *s2, size_t n)`.
    ///
    /// Returns < 0, 0, or > 0 per byte-wise comparison.
    name = "memcmp",
    struct = NativeMemcmp,
    args = [s1: concrete, s2: concrete, n: concrete],
    call |state| {
        // n == 0 and the MAX_STRCMP_LEN cap are both handled by compare_bytes.
        compare_bytes(state, s1, s2, n,
                      /*stop_at_null=*/false, /*case_insensitive=*/false)
    }
}

test_submod!("memcmp_tests.rs" => memcmp_tests);
