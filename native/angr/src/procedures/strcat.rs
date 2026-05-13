//! Native strcat and strncat implementations.
//!
//! Concrete string concatenation. Symbolic arguments fall back to Python.

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_STRLEN: usize = 4096;

/// Find null terminator position in concrete string.
fn find_null(state: &mut RustSimState, addr: u64) -> Result<u64, ProcedureError> {
    for i in 0..MAX_STRLEN as u64 {
        let byte_addr = addr.wrapping_add(i);
        let byte_val = state
            .memory_load(byte_addr, 1)
            .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
        let byte =
            extract_concrete_arg(&byte_val, &format!("memory byte at 0x{:x}", byte_addr))? as u8;
        if byte == 0 {
            return Ok(byte_addr);
        }
    }
    Err(ProcedureError::MaxIterations(MAX_STRLEN))
}

/// strcat: append src string to dest.
pub struct NativeStrcat;

impl NativeSimProcedure for NativeStrcat {
    fn name(&self) -> &'static str {
        "strcat"
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

        // Find end of dest
        let dest_end = find_null(state, dest)?;

        // Copy src to dest_end (including null terminator)
        for i in 0..MAX_STRLEN as u64 {
            let src_addr = src.wrapping_add(i);
            let byte_val = state
                .memory_load(src_addr, 1)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
            let byte =
                extract_concrete_arg(&byte_val, &format!("memory byte at 0x{:x}", src_addr))? as u8;

            let dst_addr = dest_end.wrapping_add(i);
            state
                .memory_store(dst_addr, RustBV::concrete(byte as u128, 8))
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;

            if byte == 0 {
                break;
            }
        }

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(dest as u128, bits)))
    }
}

/// strncat: append at most n bytes from src to dest.
pub struct NativeStrncat;

impl NativeSimProcedure for NativeStrncat {
    fn name(&self) -> &'static str {
        "strncat"
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

        let dest_end = find_null(state, dest)?;
        let max_copy = n.min(MAX_STRLEN as u64);

        let mut copied = 0u64;
        for i in 0..max_copy {
            let src_addr = src.wrapping_add(i);
            let byte_val = state
                .memory_load(src_addr, 1)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
            let byte =
                extract_concrete_arg(&byte_val, &format!("memory byte at 0x{:x}", src_addr))? as u8;

            if byte == 0 {
                break;
            }

            let dst_addr = dest_end.wrapping_add(i);
            state
                .memory_store(dst_addr, RustBV::concrete(byte as u128, 8))
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
            copied += 1;
        }

        // Null-terminate
        let null_addr = dest_end.wrapping_add(copied);
        state
            .memory_store(null_addr, RustBV::concrete(0u128, 8))
            .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(dest as u128, bits)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_strcat() {
        let mut state = RustSimState::new("amd64").unwrap();
        // dest: "hello\0" with room for more
        let mut buf = vec![0u8; 32];
        buf[..6].copy_from_slice(b"hello\x00");
        state.map_memory_data(0x1000, &buf, Permission::RWX);
        // src: " world\0"
        state.map_memory_data(0x2000, b" world\x00", Permission::RWX);

        let p = NativeStrcat;
        let result = p
            .call(
                &mut state,
                &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(0x1000));

        // Verify concatenated string
        for (i, &expected) in b"hello world\x00".iter().enumerate() {
            let byte = state
                .memory_load(0x1000 + i as u64, 1)
                .unwrap()
                .as_u64()
                .unwrap() as u8;
            assert_eq!(byte, expected, "byte {} mismatch", i);
        }
    }

    #[test]
    fn test_strncat() {
        let mut state = RustSimState::new("amd64").unwrap();
        let mut buf = vec![0u8; 32];
        buf[..3].copy_from_slice(b"hi\x00");
        state.map_memory_data(0x1000, &buf, Permission::RWX);
        state.map_memory_data(0x2000, b"there\x00", Permission::RWX);

        let p = NativeStrncat;
        let result = p
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(3, 64), // only copy 3 bytes
                ],
            )
            .unwrap()
            .unwrap();
        assert_eq!(result.as_u64(), Some(0x1000));

        // Should be "hithe\0" (3 bytes from "there")
        for (i, &expected) in b"hithe\x00".iter().enumerate() {
            let byte = state
                .memory_load(0x1000 + i as u64, 1)
                .unwrap()
                .as_u64()
                .unwrap() as u8;
            assert_eq!(byte, expected, "byte {} mismatch", i);
        }
    }
}
