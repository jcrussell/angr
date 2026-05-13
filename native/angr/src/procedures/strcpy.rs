//! Native strcpy/strncpy implementation.
//!
//! # Behavior
//!
//! - If any address is symbolic, falls back to Python
//! - If any source byte is symbolic, falls back to Python
//! - Maximum string length is 4096 bytes

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

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
        let dest = extract_concrete_arg(&args[0], "dest")?;
        let src = extract_concrete_arg(&args[1], "src")?;

        // Read source string until null terminator
        let mut buf = Vec::with_capacity(256);
        for i in 0..MAX_STRLEN as u64 {
            let byte_val = state
                .memory_load(src.wrapping_add(i), 1)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
            let byte = extract_concrete_arg(&byte_val, &format!("src byte at offset {}", i))? as u8;
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
            state
                .memory_store(dest.wrapping_add(i as u64), bv)
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
        let dest = extract_concrete_arg(&args[0], "dest")?;
        let src = extract_concrete_arg(&args[1], "src")?;
        let n = extract_concrete_arg(&args[2], "n")?;

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
                let byte_val = state
                    .memory_load(src.wrapping_add(i), 1)
                    .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
                let byte =
                    extract_concrete_arg(&byte_val, &format!("src byte at offset {}", i))? as u8;
                buf.push(byte);
                if byte == 0 {
                    null_found = true;
                }
            }
        }

        // Write to destination byte-by-byte
        for (i, &byte) in buf.iter().enumerate() {
            let bv = RustBV::concrete(byte as u128, 8);
            state
                .memory_store(dest.wrapping_add(i as u64), bv)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
        }

        Ok(Some(args[0].clone()))
    }
}

/// Native strdup implementation.
///
/// ```c
/// char *strdup(const char *s);
/// ```
///
/// Allocates a new string via heap_alloc, copies the source string
/// (including null terminator), and returns pointer to the new string.
pub struct NativeStrdup;

impl NativeSimProcedure for NativeStrdup {
    fn name(&self) -> &'static str {
        "strdup"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let src = extract_concrete_arg(&args[0], "s")?;

        // Read source string until null terminator
        let mut buf = Vec::with_capacity(256);
        for i in 0..MAX_STRLEN as u64 {
            let byte_val = state
                .memory_load(src.wrapping_add(i), 1)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
            let byte = extract_concrete_arg(&byte_val, &format!("src byte at offset {}", i))? as u8;
            buf.push(byte);
            if byte == 0 {
                break;
            }
            if i == MAX_STRLEN as u64 - 1 {
                return Err(ProcedureError::MaxIterations(MAX_STRLEN));
            }
        }

        // Allocate new buffer (strlen + 1 for null terminator)
        let new_addr = state.heap_alloc(buf.len() as u64);

        // Copy bytes to new allocation
        for (i, &byte) in buf.iter().enumerate() {
            let bv = RustBV::concrete(byte as u128, 8);
            state
                .memory_store(new_addr.wrapping_add(i as u64), bv)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
        }

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(new_addr as u128, bits)))
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
        let result = proc
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x1000, 64)],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(0x2000));

        // Verify "hello\0" was copied
        let loaded = state.memory_load(0x2000, 6).unwrap();
        // "hello\0" in little-endian u48
        let expected = u64::from_le_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);
        assert_eq!(loaded.as_u64(), Some(expected));
    }

    #[test]
    fn test_strcpy_empty() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x00", Permission::RWX);
        state.map_memory_data(0x2000, &[0xFFu8; 8], Permission::RWX);

        NativeStrcpy
            .call(
                &mut state,
                &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x1000, 64)],
            )
            .unwrap();

        // First byte should be null
        let first = state.memory_load(0x2000, 1).unwrap();
        assert_eq!(first.as_u64(), Some(0));
    }

    #[test]
    fn test_strncpy_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);
        state.map_memory_data(0x2000, &[0xFFu8; 16], Permission::RWX);

        NativeStrncpy
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(5, 64),
                ],
            )
            .unwrap();

        // Should copy exactly 5 bytes: "hello"
        for (i, &expected) in b"hello".iter().enumerate() {
            let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
            assert_eq!(byte.as_u64(), Some(expected as u64));
        }
    }

    #[test]
    fn test_strncpy_pads_with_null() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hi\x00", Permission::RWX);
        state.map_memory_data(0x2000, &[0xFFu8; 8], Permission::RWX);

        NativeStrncpy
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(6, 64),
                ],
            )
            .unwrap();

        // Bytes after null should also be null-padded
        for i in 2..6u64 {
            let byte = state.memory_load(0x2000 + i, 1).unwrap();
            assert_eq!(byte.as_u64(), Some(0));
        }
    }

    #[test]
    fn test_strdup_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory(0xC000_0000, 0x10000, Permission::RWX);
        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

        let result = NativeStrdup
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        let new_addr = result.unwrap().as_u64().unwrap();
        assert!(new_addr >= 0xC000_0000);

        // Verify "hello\0" was copied
        for (i, &expected) in b"hello\x00".iter().enumerate() {
            let byte = state.memory_load(new_addr + i as u64, 1).unwrap();
            assert_eq!(byte.as_u64(), Some(expected as u64));
        }

        // Verify heap metadata
        assert!(state.heap_metadata().is_allocated(new_addr));
        assert_eq!(state.heap_metadata().alloc_size(new_addr), Some(6));
    }

    #[test]
    fn test_strdup_empty_string() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory(0xC000_0000, 0x10000, Permission::RWX);
        state.map_memory_data(0x1000, b"\x00", Permission::RWX);

        let result = NativeStrdup
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        let new_addr = result.unwrap().as_u64().unwrap();

        // Should allocate 1 byte for null terminator
        let byte = state.memory_load(new_addr, 1).unwrap();
        assert_eq!(byte.as_u64(), Some(0));
        assert_eq!(state.heap_metadata().alloc_size(new_addr), Some(1));
    }

    #[test]
    fn test_strdup_symbolic_arg() {
        let mut state = RustSimState::new("amd64").unwrap();
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "ptr", 64);
        drop(ctx);
        let result = NativeStrdup.call(&mut state, &[sym]);
        assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
    }
}
