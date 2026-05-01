//! Native memcmp implementation.
//!
//! memcmp compares n bytes of two memory regions.
//!
//! # Behavior
//!
//! - If addresses or n are symbolic, falls back to Python
//! - If any compared byte is symbolic, falls back to Python

use crate::state::RustSimState;
use crate::symbolic::RustBV;
use super::{extract_concrete_arg, NativeSimProcedure, ProcedureError};

/// Maximum comparison length before falling back to Python.
const MAX_MEMCMP_LEN: usize = 4096;

/// Native memcmp implementation.
///
/// ```c
/// int memcmp(const void *s1, const void *s2, size_t n);
/// ```
///
/// Returns:
/// - < 0 if s1 < s2
/// - 0 if s1 == s2
/// - > 0 if s1 > s2
pub struct NativeMemcmp;

impl NativeSimProcedure for NativeMemcmp {
    fn name(&self) -> &'static str {
        "memcmp"
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

        if n == 0 {
            return Ok(Some(RustBV::zero(32)));
        }

        if n > MAX_MEMCMP_LEN {
            return Err(ProcedureError::MaxIterations(MAX_MEMCMP_LEN));
        }

        for i in 0..n {
            let c1_addr = s1_addr.wrapping_add(i as u64);
            let c2_addr = s2_addr.wrapping_add(i as u64);

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
        }

        Ok(Some(RustBV::zero(32)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_memcmp_equal() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello", Permission::RWX);
        state.map_memory_data(0x2000, b"hello", Permission::RWX);

        let proc = NativeMemcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64), RustBV::concrete(5, 64)],
        ).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_memcmp_less_than() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x01\x02\x03", Permission::RWX);
        state.map_memory_data(0x2000, b"\x01\x02\x04", Permission::RWX);

        let proc = NativeMemcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64), RustBV::concrete(3, 64)],
        ).unwrap();
        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val < 0);
    }

    #[test]
    fn test_memcmp_greater_than() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x01\x02\x04", Permission::RWX);
        state.map_memory_data(0x2000, b"\x01\x02\x03", Permission::RWX);

        let proc = NativeMemcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64), RustBV::concrete(3, 64)],
        ).unwrap();
        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val > 0);
    }

    #[test]
    fn test_memcmp_zero_length() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"abc", Permission::RWX);
        state.map_memory_data(0x2000, b"xyz", Permission::RWX);

        let proc = NativeMemcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
        ).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_memcmp_partial() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello1", Permission::RWX);
        state.map_memory_data(0x2000, b"hello2", Permission::RWX);

        let proc = NativeMemcmp;
        // Compare only first 5 bytes (should be equal)
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64), RustBV::concrete(5, 64)],
        ).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(0));

        // Compare 6 bytes (should differ)
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64), RustBV::concrete(6, 64)],
        ).unwrap();
        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val != 0);
    }

    #[test]
    fn test_memcmp_with_nulls() {
        let mut state = RustSimState::new("amd64").unwrap();
        // memcmp does NOT stop at null bytes (unlike strcmp)
        state.map_memory_data(0x1000, b"ab\x00cd", Permission::RWX);
        state.map_memory_data(0x2000, b"ab\x00ce", Permission::RWX);

        let proc = NativeMemcmp;
        let result = proc.call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64), RustBV::concrete(5, 64)],
        ).unwrap();
        let val = result.unwrap().as_u128().unwrap() as i32;
        assert!(val < 0); // 'd' < 'e'
    }
}
