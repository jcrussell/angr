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

use super::ProcedureError;
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

crate::declare_proc! {
    /// Native memcpy: `void *memcpy(void *dest, const void *src, size_t n)`.
    ///
    /// Copies n bytes from src to dest. Returns dest.
    name = "memcpy",
    struct = NativeMemcpy,
    args = [dst: concrete, src: concrete, size: concrete],
    call |state| {
        let size = size as usize;
        let arch_bits = state.arch().bits();
        // `dst` is required concrete, so the returned dest pointer is the
        // value-equivalent concrete BV (the manual impl returned args[0]).
        let ret = RustBV::concrete(dst as u128, arch_bits);

        // Check size limit
        if size > MAX_COPY_SIZE {
            return Err(ProcedureError::MaxIterations(MAX_COPY_SIZE));
        }

        // Handle zero-size copy
        if size == 0 {
            return Ok(Some(ret));
        }

        copy_forward(state, src, dst, size)?;
        Ok(Some(ret))
    }
}

crate::declare_proc! {
    /// Native memmove: `void *memmove(void *dest, const void *src, size_t n)`.
    ///
    /// Like memcpy, but handles overlapping regions correctly.
    name = "memmove",
    struct = NativeMemmove,
    args = [dst: concrete, src: concrete, size: concrete],
    call |state| {
        let size = size as usize;
        let arch_bits = state.arch().bits();
        // `dst` is required concrete, so the returned dest pointer is the
        // value-equivalent concrete BV (the manual impl returned args[0]).
        let ret = RustBV::concrete(dst as u128, arch_bits);

        // Check size limit
        if size > MAX_COPY_SIZE {
            return Err(ProcedureError::MaxIterations(MAX_COPY_SIZE));
        }

        // Handle zero-size copy
        if size == 0 {
            return Ok(Some(ret));
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
        Ok(Some(ret))
    }
}

#[cfg(test)]
#[path = "memcpy_tests.rs"]
mod tests;
