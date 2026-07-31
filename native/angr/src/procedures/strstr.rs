//! Native strstr implementation.
//!
//! strstr finds the first occurrence of a substring in a string.
//!
//! Symbolic arguments fall back to Python.

use super::extract_concrete_arg;
use super::strings::{MAX_STRING_SCAN, scan_concrete_predicate, scan_concrete_until_null};
use crate::symbolic::RustBV;

crate::declare_proc! {
    /// strstr: find substring in string.
    ///
    /// ```c
    /// char *strstr(const char *haystack, const char *needle);
    /// ```
    ///
    /// Returns pointer to first occurrence of needle in haystack, or NULL.
    name = "strstr",
    struct = NativeStrstr,
    args = [haystack_addr: concrete, needle_addr: concrete],
    call |state| {
        let bits = state.arch().bits();

        let needle = scan_concrete_until_null(state, needle_addr, MAX_STRING_SCAN, "needle")?;

        // Empty needle: return haystack
        if needle.is_empty() {
            return Ok(Some(RustBV::concrete(haystack_addr as u128, bits)));
        }

        // Scan haystack. The shared helper handles the load/null/bound
        // boilerplate; the closure only encodes the needle match (which peeks
        // ahead via `state`) and the end-of-haystack NULL result.
        let hit = scan_concrete_predicate(
            state,
            haystack_addr,
            MAX_STRING_SCAN,
            "haystack",
            |state, first, i, h_addr| {
                // End of haystack: needle not found.
                if first == 0 {
                    return Ok(Some(0u128));
                }
                // Try to match needle at this position.
                if first == needle[0] {
                    for (j, needle_byte) in needle.iter().enumerate().skip(1) {
                        let val = state.memory_load(h_addr.wrapping_add(j as u64), 1)?;
                        let byte =
                            extract_concrete_arg(&val, &format!("haystack[{}]", i as usize + j))?
                                as u8;
                        if byte != *needle_byte {
                            // SILENT(cat-a): mismatch at this haystack position -- expected
                            // control flow, scan_concrete_predicate advances to the next
                            // position and retries, not a degraded final answer.
                            return Ok(None);
                        }
                    }
                    return Ok(Some(h_addr as u128));
                }
                Ok(None)
            },
        )?;

        Ok(Some(RustBV::concrete(hit, bits)))
    }
}

#[cfg(test)]
#[path = "strstr_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
