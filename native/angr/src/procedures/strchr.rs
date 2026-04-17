//! Native strchr and memchr implementations.
//!
//! strchr: find first occurrence of a character in a null-terminated string.
//! memchr: find first occurrence of a byte in a memory region.
//!
//! Symbolic arguments fall back to Python.

use crate::state::RustSimState;
use crate::symbolic::RustBV;
use super::{NativeSimProcedure, ProcedureError};

const MAX_SCAN: usize = 4096;

/// strchr: find character in string.
///
/// ```c
/// char *strchr(const char *s, int c);
/// ```
///
/// Returns pointer to first occurrence of c in s, or NULL if not found.
pub struct NativeStrchr;

impl NativeSimProcedure for NativeStrchr {
    fn name(&self) -> &'static str { "strchr" }
    fn num_args(&self) -> usize { 2 }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let addr = args[0].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("s".to_string())
        })?;
        let target = args[1].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("c".to_string())
        })? as u8;

        let bits = state.arch().bits();

        for i in 0..MAX_SCAN as u64 {
            let byte_addr = addr.wrapping_add(i);
            let byte_val = state.memory_load(byte_addr, 1)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
            let byte = byte_val.as_u64().ok_or_else(|| {
                ProcedureError::SymbolicArgument(format!("memory byte at 0x{:x}", byte_addr))
            })? as u8;

            if byte == target {
                return Ok(Some(RustBV::concrete(byte_addr as u128, bits)));
            }
            if byte == 0 {
                // Null terminator — also check if we're looking for '\0'
                return Ok(Some(RustBV::concrete(0u128, bits)));
            }
        }
        Err(ProcedureError::MaxIterations(MAX_SCAN))
    }
}

/// memchr: find byte in memory region.
///
/// ```c
/// void *memchr(const void *s, int c, size_t n);
/// ```
///
/// Returns pointer to first occurrence of c in the first n bytes of s, or NULL.
pub struct NativeMemchr;

impl NativeSimProcedure for NativeMemchr {
    fn name(&self) -> &'static str { "memchr" }
    fn num_args(&self) -> usize { 3 }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let addr = args[0].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("s".to_string())
        })?;
        let target = args[1].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("c".to_string())
        })? as u8;
        let n = args[2].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("n".to_string())
        })?;

        let bits = state.arch().bits();
        let scan_len = n.min(MAX_SCAN as u64);

        for i in 0..scan_len {
            let byte_addr = addr.wrapping_add(i);
            let byte_val = state.memory_load(byte_addr, 1)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
            let byte = byte_val.as_u64().ok_or_else(|| {
                ProcedureError::SymbolicArgument(format!("memory byte at 0x{:x}", byte_addr))
            })? as u8;

            if byte == target {
                return Ok(Some(RustBV::concrete(byte_addr as u128, bits)));
            }
        }
        // Not found — return NULL
        Ok(Some(RustBV::concrete(0u128, bits)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_strchr_found() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

        let p = NativeStrchr;
        let result = p.call(&mut state, &[
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(b'l' as u128, 64),
        ]).unwrap().unwrap();
        assert_eq!(result.as_u64(), Some(0x1002));
    }

    #[test]
    fn test_strchr_not_found() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

        let p = NativeStrchr;
        let result = p.call(&mut state, &[
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(b'z' as u128, 64),
        ]).unwrap().unwrap();
        assert_eq!(result.as_u64(), Some(0)); // NULL
    }

    #[test]
    fn test_memchr_found() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x01\x02\x03\x04", Permission::RWX);

        let p = NativeMemchr;
        let result = p.call(&mut state, &[
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(3, 64),
            RustBV::concrete(4, 64),
        ]).unwrap().unwrap();
        assert_eq!(result.as_u64(), Some(0x1002));
    }

    #[test]
    fn test_memchr_not_found() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x01\x02\x03\x04", Permission::RWX);

        let p = NativeMemchr;
        let result = p.call(&mut state, &[
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0xFF, 64),
            RustBV::concrete(4, 64),
        ]).unwrap().unwrap();
        assert_eq!(result.as_u64(), Some(0)); // NULL
    }
}
