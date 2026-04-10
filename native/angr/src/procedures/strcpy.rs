//! Native strcpy/strncpy implementation.
//!
//! # Behavior
//!
//! - If any address is symbolic, falls back to Python
//! - If any source byte is symbolic, falls back to Python
//! - Maximum string length is 4096 bytes

use crate::state::RustSimState;
use crate::symbolic::RustBV;
use super::{NativeSimProcedure, ProcedureError};

const MAX_STRLEN: usize = 4096;

/// Native strcpy implementation.
///
/// ```c
/// char *strcpy(char *dest, const char *src);
/// ```
pub struct NativeStrcpy;

impl NativeSimProcedure for NativeStrcpy {
    fn name(&self) -> &'static str {
        "strcpy"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let dest = args[0].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("dest".to_string())
        })?;
        let src = args[1].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("src".to_string())
        })?;

        // Read source string until null terminator
        let mut buf = Vec::with_capacity(256);
        for i in 0..MAX_STRLEN as u64 {
            let byte_val = state.memory_load(src.wrapping_add(i), 1)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
            let byte = byte_val.as_u64().ok_or_else(|| {
                ProcedureError::SymbolicArgument(format!("src byte at offset {}", i))
            })? as u8;
            buf.push(byte);
            if byte == 0 {
                break;
            }
            if i == MAX_STRLEN as u64 - 1 {
                return Err(ProcedureError::MaxIterations(MAX_STRLEN));
            }
        }

        // Write to destination byte-by-byte (including null terminator)
        for (i, &byte) in buf.iter().enumerate() {
            let bv = RustBV::concrete(byte as u128, 8);
            state.memory_store(dest.wrapping_add(i as u64), bv)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
        }

        Ok(Some(args[0].clone()))
    }
}

/// Native strncpy implementation.
///
/// ```c
/// char *strncpy(char *dest, const char *src, size_t n);
/// ```
pub struct NativeStrncpy;

impl NativeSimProcedure for NativeStrncpy {
    fn name(&self) -> &'static str {
        "strncpy"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let dest = args[0].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("dest".to_string())
        })?;
        let src = args[1].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("src".to_string())
        })?;
        let n = args[2].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("n".to_string())
        })?;

        if n > MAX_STRLEN as u64 {
            return Err(ProcedureError::MaxIterations(n as usize));
        }

        // Read up to n bytes from source, stopping at null
        let mut buf = Vec::with_capacity(n as usize);
        let mut null_found = false;
        for i in 0..n {
            if null_found {
                buf.push(0);
            } else {
                let byte_val = state.memory_load(src.wrapping_add(i), 1)
                    .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
                let byte = byte_val.as_u64().ok_or_else(|| {
                    ProcedureError::SymbolicArgument(format!("src byte at offset {}", i))
                })? as u8;
                buf.push(byte);
                if byte == 0 {
                    null_found = true;
                }
            }
        }

        // Write to destination byte-by-byte
        for (i, &byte) in buf.iter().enumerate() {
            let bv = RustBV::concrete(byte as u128, 8);
            state.memory_store(dest.wrapping_add(i as u64), bv)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
        }

        Ok(Some(args[0].clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_strcpy_basic() {
        let mut state = RustSimState::new("amd64").unwrap();

        // Source string
        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
        // Destination buffer
        state.map_memory_data(0x2000, &vec![0u8; 16], Permission::RWX);

        let proc = NativeStrcpy;
        let result = proc.call(&mut state, &[
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(0x1000, 64),
        ]).unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(0x2000));

        // Verify "hello\0" was copied
        let loaded = state.memory_load(0x2000, 6).unwrap();
        // "hello\0" in little-endian u48
        let expected = u64::from_le_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);
        assert_eq!(loaded.as_u64(), Some(expected));
    }
}
