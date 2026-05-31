//! Native strstr implementation.
//!
//! strstr finds the first occurrence of a substring in a string.
//!
//! Symbolic arguments fall back to Python.

use super::strings::scan_concrete_until_null;
use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_SCAN: usize = 4096;

/// strstr: find substring in string.
///
/// ```c
/// char *strstr(const char *haystack, const char *needle);
/// ```
///
/// Returns pointer to first occurrence of needle in haystack, or NULL.
pub struct NativeStrstr;

impl NativeSimProcedure for NativeStrstr {
    fn name(&self) -> &'static str {
        "strstr"
    }
    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let haystack_addr = extract_concrete_arg(&args[0], "haystack")?;
        let needle_addr = extract_concrete_arg(&args[1], "needle")?;

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
            let first_val = state
                .memory_load(h_addr, 1)
                ?;
            let first = extract_concrete_arg(&first_val, &format!("haystack[{}]", i))? as u8;

            // End of haystack
            if first == 0 {
                return Ok(Some(RustBV::concrete(0u128, bits)));
            }

            // Try to match needle at this position
            if first == needle[0] {
                let mut matched = true;
                for j in 1..needle.len() {
                    let h_byte_addr = h_addr.wrapping_add(j as u64);
                    let val = state
                        .memory_load(h_byte_addr, 1)
                        ?;
                    let byte =
                        extract_concrete_arg(&val, &format!("haystack[{}]", i as usize + j))? as u8;
                    if byte != needle[j] {
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
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_strstr_found() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"world\x00", Permission::RWX);

        let p = NativeStrstr;
        let result = p
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(0x1006)); // "world" starts at offset 6
    }

    #[test]
    fn test_strstr_not_found() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"xyz\x00", Permission::RWX);

        let p = NativeStrstr;
        let result = p
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(0)); // NULL
    }

    #[test]
    fn test_strstr_empty_needle() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"\x00", Permission::RWX);

        let p = NativeStrstr;
        let result = p
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(0x1000)); // Return haystack
    }

    #[test]
    fn test_strstr_at_start() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"hello\x00", Permission::RWX);

        let p = NativeStrstr;
        let result = p
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(0x1000));
    }
}
