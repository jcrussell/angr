//! Native memset implementation.
//!
//! memset fills a memory region with a constant byte value.
//!
//! # Behavior
//!
//! - If the address or value is symbolic, falls back to Python
//! - If the size is symbolic, falls back to Python
//! - Maximum size is 1MB (configurable)

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Maximum memset size before falling back to Python.
const MAX_MEMSET_SIZE: u64 = 1024 * 1024;

/// Native memset implementation.
///
/// ```c
/// void *memset(void *s, int c, size_t n);
/// ```
///
/// Fills n bytes at s with byte value c. Returns s.
pub struct NativeMemset;

impl NativeSimProcedure for NativeMemset {
    fn name(&self) -> &'static str {
        "memset"
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
        let value = extract_concrete_arg(&args[1], "value")?;
        let byte_val = (value & 0xFF) as u8;
        let size = extract_concrete_arg(&args[2], "size")?;

        if size > MAX_MEMSET_SIZE {
            return Err(ProcedureError::Other(format!(
                "memset size {} exceeds maximum {}",
                size, MAX_MEMSET_SIZE
            )));
        }

        if size == 0 {
            return Ok(Some(args[0].clone()));
        }

        // Fill memory byte-by-byte with the value
        // Use 8-byte chunks for efficiency
        let fill_8 = {
            let mut val: u64 = 0;
            for i in 0..8 {
                val |= (byte_val as u64) << (i * 8);
            }
            val as u128
        };

        let mut offset: u64 = 0;
        while offset + 8 <= size {
            let bv = RustBV::concrete(fill_8, 64);
            state
                .memory_store(dest.wrapping_add(offset), bv)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
            offset += 8;
        }
        // Handle remaining bytes
        while offset < size {
            let bv = RustBV::concrete(byte_val as u128, 8);
            state
                .memory_store(dest.wrapping_add(offset), bv)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
            offset += 1;
        }

        // Return dest pointer
        Ok(Some(args[0].clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_memset_basic() {
        let mut state = RustSimState::new("amd64").unwrap();

        // Map writable memory
        let data = vec![0u8; 16];
        state.map_memory_data(0x1000, &data, Permission::RWX);

        let proc = NativeMemset;
        let result = proc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0x41, 64), // 'A'
                    RustBV::concrete(4, 64),
                ],
            )
            .unwrap();

        // Should return dest
        assert_eq!(result.unwrap().as_u64(), Some(0x1000));

        // Verify memory was filled
        let loaded = state.memory_load(0x1000, 4).unwrap();
        assert_eq!(loaded.as_u64(), Some(0x41414141));
    }

    #[test]
    fn test_memset_zero() {
        let mut state = RustSimState::new("amd64").unwrap();

        let data = vec![0xFFu8; 16];
        state.map_memory_data(0x1000, &data, Permission::RWX);

        let proc = NativeMemset;
        proc.call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(8, 64),
            ],
        )
        .unwrap();

        let loaded = state.memory_load(0x1000, 8).unwrap();
        assert_eq!(loaded.as_u64(), Some(0));
    }
}
