//! Native memcpy implementation.
//!
//! memcpy copies n bytes from source to destination and returns the
//! destination pointer.
//!
//! # Behavior
//!
//! - If any argument is symbolic, falls back to Python
//! - Copies data byte-by-byte, preserving symbolic values
//! - Maximum copy size is 1MB (configurable)

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Maximum copy size before falling back to Python.
const MAX_COPY_SIZE: usize = 1024 * 1024; // 1MB

/// Copy `size` bytes forward from `src` to `dst` using 8-byte chunks.
fn copy_forward(
    state: &mut RustSimState,
    src: u64,
    dst: u64,
    size: usize,
) -> Result<(), ProcedureError> {
    let mut offset: usize = 0;

    // Copy in 8-byte chunks where possible
    while offset + 8 <= size {
        let value = state
            .memory_load(src.wrapping_add(offset as u64), 8)
            .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
        state
            .memory_store(dst.wrapping_add(offset as u64), value)
            .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
        offset += 8;
    }

    // Copy remaining bytes
    while offset < size {
        let value = state
            .memory_load(src.wrapping_add(offset as u64), 1)
            .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
        state
            .memory_store(dst.wrapping_add(offset as u64), value)
            .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
        offset += 1;
    }
    Ok(())
}

/// Native memcpy implementation.
///
/// ```c
/// void *memcpy(void *dest, const void *src, size_t n);
/// ```
///
/// Copies n bytes from src to dest. Returns dest.
pub struct NativeMemcpy;

impl NativeSimProcedure for NativeMemcpy {
    fn name(&self) -> &'static str {
        "memcpy"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let dst = extract_concrete_arg(&args[0], "dst")?;
        let src = extract_concrete_arg(&args[1], "src")?;
        let size = extract_concrete_arg(&args[2], "size")? as usize;

        // Check size limit
        if size > MAX_COPY_SIZE {
            return Err(ProcedureError::MaxIterations(MAX_COPY_SIZE));
        }

        // Handle zero-size copy
        if size == 0 {
            return Ok(Some(args[0].clone()));
        }

        copy_forward(state, src, dst, size)?;
        Ok(Some(args[0].clone()))
    }
}

/// Native memmove implementation.
///
/// ```c
/// void *memmove(void *dest, const void *src, size_t n);
/// ```
///
/// Like memcpy, but handles overlapping regions correctly.
pub struct NativeMemmove;

impl NativeSimProcedure for NativeMemmove {
    fn name(&self) -> &'static str {
        "memmove"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let dst = extract_concrete_arg(&args[0], "dst")?;
        let src = extract_concrete_arg(&args[1], "src")?;
        let size = extract_concrete_arg(&args[2], "size")? as usize;

        // Check size limit
        if size > MAX_COPY_SIZE {
            return Err(ProcedureError::MaxIterations(MAX_COPY_SIZE));
        }

        // Handle zero-size copy
        if size == 0 {
            return Ok(Some(args[0].clone()));
        }

        // For overlapping regions, we need to copy to a temporary buffer
        // or copy in reverse order if dst > src
        if dst > src && dst < src + size as u64 {
            // Overlapping: copy backwards
            for i in (0..size).rev() {
                let src_addr = src.wrapping_add(i as u64);
                let dst_addr = dst.wrapping_add(i as u64);

                let value = state
                    .memory_load(src_addr, 1)
                    .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;

                state
                    .memory_store(dst_addr, value)
                    .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
            }
        } else {
            // Non-overlapping or dst < src: copy forwards
            copy_forward(state, src, dst, size)?;
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
    fn test_memcpy_basic() {
        let mut state = RustSimState::new("amd64").unwrap();

        // Map source with data
        let data = b"hello world!";
        state.map_memory_data(0x1000, data, Permission::RWX);

        // Map destination
        state.map_memory(0x2000, 0x1000, Permission::RWX);

        let proc = NativeMemcpy;
        let result = proc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64), // dst
                    RustBV::concrete(0x1000, 64), // src
                    RustBV::concrete(12, 64),     // size
                ],
            )
            .unwrap();

        // Result should be dst pointer
        assert_eq!(result.unwrap().as_u64(), Some(0x2000));

        // Verify data was copied
        for i in 0..12u64 {
            let val = state.memory_load(0x2000 + i, 1).unwrap();
            assert_eq!(val.as_u64(), Some(data[i as usize] as u64));
        }
    }

    #[test]
    fn test_memcpy_zero_size() {
        let mut state = RustSimState::new("amd64").unwrap();

        // Map memory
        state.map_memory(0x1000, 0x2000, Permission::RWX);

        let proc = NativeMemcpy;
        let result = proc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(0x1000, 64),
                    RustBV::concrete(0, 64), // zero size
                ],
            )
            .unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(0x2000));
    }

    #[test]
    fn test_memcpy_symbolic_dst() {
        let mut state = RustSimState::new("amd64").unwrap();

        let ctx = state.solver().borrow();
        let sym_dst = RustBV::symbolic(&ctx, "dst", 64);
        drop(ctx);

        let proc = NativeMemcpy;
        let result = proc.call(
            &mut state,
            &[
                sym_dst,
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(10, 64),
            ],
        );

        assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
    }

    #[test]
    fn test_memmove_overlapping() {
        let mut state = RustSimState::new("amd64").unwrap();

        // Map memory with "abcdefgh"
        let data = b"abcdefgh\x00\x00\x00\x00\x00\x00\x00\x00";
        state.map_memory_data(0x1000, data, Permission::RWX);

        // Move from offset 2 to offset 0 (overlapping, should work)
        let proc = NativeMemmove;
        let _result = proc
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x1000, 64), // dst
                    RustBV::concrete(0x1002, 64), // src (overlapping)
                    RustBV::concrete(6, 64),      // size
                ],
            )
            .unwrap();

        // Should have "cdefghgh" now
        let val = state.memory_load(0x1000, 1).unwrap();
        assert_eq!(val.as_u64(), Some(b'c' as u64));

        let val = state.memory_load(0x1001, 1).unwrap();
        assert_eq!(val.as_u64(), Some(b'd' as u64));
    }
}
