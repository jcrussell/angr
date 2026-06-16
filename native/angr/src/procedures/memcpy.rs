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
        let value = state.memory_load(src.wrapping_add(offset as u64), 8)?;
        state.memory_store(dst.wrapping_add(offset as u64), value)?;
        offset += 8;
    }

    // Copy remaining bytes
    while offset < size {
        let value = state.memory_load(src.wrapping_add(offset as u64), 1)?;
        state.memory_store(dst.wrapping_add(offset as u64), value)?;
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

                let value = state.memory_load(src_addr, 1)?;

                state.memory_store(dst_addr, value)?;
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
#[path = "memcpy_tests.rs"]
mod tests;
