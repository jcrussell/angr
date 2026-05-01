//! Native strcmp/strncmp implementations.
//!
//! strcmp compares two null-terminated strings lexicographically.
//! strncmp compares at most n characters.
//!
//! # Behavior
//!
//! - If addresses are symbolic, falls back to Python
//! - If any compared byte is symbolic, falls back to Python
//! - Maximum string length is 4096 bytes (configurable)

use crate::state::RustSimState;
use crate::symbolic::RustBV;
use super::{extract_concrete_arg, NativeSimProcedure, ProcedureError};

/// Maximum string length before falling back to Python.
const MAX_STRCMP_LEN: usize = 4096;

/// Native strcmp implementation.
///
/// ```c
/// int strcmp(const char *s1, const char *s2);
/// ```
///
/// Returns:
/// - < 0 if s1 < s2
/// - 0 if s1 == s2
/// - > 0 if s1 > s2
pub struct NativeStrcmp;

impl NativeSimProcedure for NativeStrcmp {
    fn name(&self) -> &'static str {
        "strcmp"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let s1_addr = extract_concrete_arg(&args[0], "s1")?;
        let s2_addr = extract_concrete_arg(&args[1], "s2")?;

        // Compare byte by byte
        for i in 0..MAX_STRCMP_LEN as u64 {
            let c1_addr = s1_addr.wrapping_add(i);
            let c2_addr = s2_addr.wrapping_add(i);

            let c1_val = state.memory_load(c1_addr, 1)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
            let c2_val = state.memory_load(c2_addr, 1)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;

            let c1 = extract_concrete_arg(&c1_val, &format!("s1[{}]", i))? as u8;
            let c2 = extract_concrete_arg(&c2_val, &format!("s2[{}]", i))? as u8;

            if c1 != c2 {
                let diff = (c1 as i32) - (c2 as i32);
                return Ok(Some(RustBV::concrete(diff as u128, 32)));
            }

            if c1 == 0 {
                return Ok(Some(RustBV::zero(32)));
            }
        }

        Err(ProcedureError::MaxIterations(MAX_STRCMP_LEN))
    }
}

/// Native strncmp implementation.
///
/// ```c
/// int strncmp(const char *s1, const char *s2, size_t n);
/// ```
///
/// Like strcmp, but compares at most n characters.
pub struct NativeStrncmp;

impl NativeSimProcedure for NativeStrncmp {
    fn name(&self) -> &'static str {
        "strncmp"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let s1_addr = extract_concrete_arg(&args[0], "s1")?;
        let s2_addr = extract_concrete_arg(&args[1], "s2")?;
        let n = extract_concrete_arg(&args[2], "n")? as usize;

        // Handle zero-length comparison
        if n == 0 {
            return Ok(Some(RustBV::zero(32)));
        }

        // Check bounds
        let max_n = n.min(MAX_STRCMP_LEN);

        // Compare byte by byte
        for i in 0..max_n {
            let c1_addr = s1_addr.wrapping_add(i as u64);
            let c2_addr = s2_addr.wrapping_add(i as u64);

            // Load bytes
            let c1_val = state.memory_load(c1_addr, 1)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
            let c2_val = state.memory_load(c2_addr, 1)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;

            // Get concrete values
            let c1 = extract_concrete_arg(&c1_val, &format!("s1[{}]", i))? as u8;
            let c2 = extract_concrete_arg(&c2_val, &format!("s2[{}]", i))? as u8;

            // Compare
            if c1 != c2 {
                let diff = (c1 as i32) - (c2 as i32);
                return Ok(Some(RustBV::concrete(diff as u128, 32)));
            }

            // Check for end of both strings
            if c1 == 0 {
                return Ok(Some(RustBV::zero(32)));
            }
        }

        // Compared n characters, all equal
        if n > MAX_STRCMP_LEN {
            Err(ProcedureError::MaxIterations(MAX_STRCMP_LEN))
        } else {
            Ok(Some(RustBV::zero(32)))
        }
    }
}

/// Native strcasecmp implementation (case-insensitive strcmp).
///
/// ```c
/// int strcasecmp(const char *s1, const char *s2);
/// ```
pub struct NativeStrcasecmp;

impl NativeSimProcedure for NativeStrcasecmp {
    fn name(&self) -> &'static str {
        "strcasecmp"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let s1_addr = extract_concrete_arg(&args[0], "s1")?;
        let s2_addr = extract_concrete_arg(&args[1], "s2")?;

        for i in 0..MAX_STRCMP_LEN as u64 {
            let c1_addr = s1_addr.wrapping_add(i);
            let c2_addr = s2_addr.wrapping_add(i);

            let c1_val = state.memory_load(c1_addr, 1)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
            let c2_val = state.memory_load(c2_addr, 1)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;

            let c1 = extract_concrete_arg(&c1_val, &format!("s1[{}]", i))? as u8;
            let c2 = extract_concrete_arg(&c2_val, &format!("s2[{}]", i))? as u8;

            // Convert to lowercase for comparison
            let c1_lower = to_lower(c1);
            let c2_lower = to_lower(c2);

            if c1_lower != c2_lower {
                let diff = (c1_lower as i32) - (c2_lower as i32);
                return Ok(Some(RustBV::concrete(diff as u128, 32)));
            }

            if c1 == 0 {
                return Ok(Some(RustBV::zero(32)));
            }
        }

        Err(ProcedureError::MaxIterations(MAX_STRCMP_LEN))
    }
}

/// Convert ASCII uppercase to lowercase.
#[inline]
fn to_lower(c: u8) -> u8 {
    if c >= b'A' && c <= b'Z' {
        c + 32
    } else {
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_strcmp_equal() {
        let mut state = RustSimState::new("amd64").unwrap();

        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"hello\x00", Permission::RWX);

        let proc = NativeStrcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        ).unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_strcmp_less_than() {
        let mut state = RustSimState::new("amd64").unwrap();

        state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"abd\x00", Permission::RWX);

        let proc = NativeStrcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        ).unwrap();

        // 'c' - 'd' = -1
        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val < 0);
    }

    #[test]
    fn test_strcmp_greater_than() {
        let mut state = RustSimState::new("amd64").unwrap();

        state.map_memory_data(0x1000, b"abd\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"abc\x00", Permission::RWX);

        let proc = NativeStrcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        ).unwrap();

        // 'd' - 'c' = 1
        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val > 0);
    }

    #[test]
    fn test_strcmp_prefix() {
        let mut state = RustSimState::new("amd64").unwrap();

        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"hello world\x00", Permission::RWX);

        let proc = NativeStrcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        ).unwrap();

        // '\0' - ' ' < 0
        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val < 0);
    }

    #[test]
    fn test_strncmp_limit() {
        let mut state = RustSimState::new("amd64").unwrap();

        state.map_memory_data(0x1000, b"hello1\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"hello2\x00", Permission::RWX);

        let proc = NativeStrncmp;

        // Compare only first 5 chars (should be equal)
        let result = proc.call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(5, 64),
            ],
        ).unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(0));

        // Compare 6 chars (should differ)
        let result = proc.call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(6, 64),
            ],
        ).unwrap();

        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val != 0);
    }

    #[test]
    fn test_strcasecmp() {
        let mut state = RustSimState::new("amd64").unwrap();

        state.map_memory_data(0x1000, b"Hello\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"hello\x00", Permission::RWX);

        let proc = NativeStrcasecmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        ).unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(0));
    }
}
