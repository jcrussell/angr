//! Native strstr implementation.
//!
//! strstr finds the first occurrence of a substring in a string.
//!
//! Symbolic arguments fall back to Python.

use super::strings::scan_concrete_until_null;
use super::{ProcedureError, extract_concrete_arg};
use crate::symbolic::RustBV;

const MAX_SCAN: usize = 4096;

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

        let needle = scan_concrete_until_null(state, needle_addr, MAX_SCAN, "needle")?;

        // Empty needle: return haystack
        if needle.is_empty() {
            return Ok(Some(RustBV::concrete(haystack_addr as u128, bits)));
        }

        // Scan haystack
        for i in 0..MAX_SCAN as u64 {
            let h_addr = haystack_addr.wrapping_add(i);

            // Check first byte of haystack at this position
            let first_val = state.memory_load(h_addr, 1)?;
            let first = extract_concrete_arg(&first_val, &format!("haystack[{i}]"))? as u8;

            // End of haystack
            if first == 0 {
                return Ok(Some(RustBV::concrete(0u128, bits)));
            }

            // Try to match needle at this position
            if first == needle[0] {
                let mut matched = true;
                for (j, needle_byte) in needle.iter().enumerate().skip(1) {
                    let h_byte_addr = h_addr.wrapping_add(j as u64);
                    let val = state.memory_load(h_byte_addr, 1)?;
                    let byte =
                        extract_concrete_arg(&val, &format!("haystack[{}]", i as usize + j))? as u8;
                    if byte != *needle_byte {
                        matched = false;
                        break;
                    }
                }
                if matched {
                    return Ok(Some(RustBV::concrete(h_addr as u128, bits)));
                }
            }
        }

        Err(ProcedureError::MaxIterations(MAX_SCAN))
    }
}

#[cfg(test)]
#[path = "strstr_tests.rs"]
mod tests;
