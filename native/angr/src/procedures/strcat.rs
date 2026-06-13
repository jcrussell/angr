//! Native strcat and strncat implementations.
//!
//! Concrete string concatenation. Symbolic arguments fall back to Python.

use super::strings::{find_null_addr, scan_concrete_bounded, scan_concrete_until_null};
use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_STRLEN: usize = 4096;

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

        let dest_end = find_null_addr(state, dest, MAX_STRLEN, "dest")?;
        let buf = scan_concrete_until_null(state, src, MAX_STRLEN, "src")?;

        for (i, &byte) in buf.iter().enumerate() {
            state.memory_store(
                dest_end.wrapping_add(i as u64),
                RustBV::concrete(byte as u128, 8),
            )?;
        }
        state.memory_store(
            dest_end.wrapping_add(buf.len() as u64),
            RustBV::concrete(0u128, 8),
        )?;

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

        let dest_end = find_null_addr(state, dest, MAX_STRLEN, "dest")?;
        let max_copy = n.min(MAX_STRLEN as u64);

        // Copy at most `max_copy` non-null bytes from src.
        let (buf, _null_found) = scan_concrete_bounded(state, src, max_copy as usize, "src")?;
        for (i, &byte) in buf.iter().enumerate() {
            state.memory_store(
                dest_end.wrapping_add(i as u64),
                RustBV::concrete(byte as u128, 8),
            )?;
        }

        // Always null-terminate after the copied bytes.
        state.memory_store(
            dest_end.wrapping_add(buf.len() as u64),
            RustBV::concrete(0u128, 8),
        )?;

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
