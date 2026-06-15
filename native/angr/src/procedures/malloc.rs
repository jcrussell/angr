//! Native malloc/free/calloc/realloc implementations.
//!
//! Uses a simple bump allocator (matching angr's SimHeapBrk).
//! Symbolic sizes fall back to Python.

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Native malloc implementation.
///
/// ```c
/// void *malloc(size_t size);
/// ```
pub struct NativeMalloc;

impl NativeSimProcedure for NativeMalloc {
    fn name(&self) -> &'static str {
        "malloc"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let size = extract_concrete_arg(&args[0], "size")?;

        let addr = state.heap_alloc(size);
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(addr as u128, bits)))
    }
}

/// Native free implementation (no-op for bump allocator).
///
/// ```c
/// void free(void *ptr);
/// ```
pub struct NativeFree;

impl NativeSimProcedure for NativeFree {
    fn name(&self) -> &'static str {
        "free"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let ptr = extract_concrete_arg(&args[0], "ptr")?;
        // Track the free (bump allocator doesn't reclaim memory)
        state.heap_free(ptr);
        Ok(None)
    }
}

/// Native calloc implementation.
///
/// ```c
/// void *calloc(size_t nmemb, size_t size);
/// ```
pub struct NativeCalloc;

impl NativeSimProcedure for NativeCalloc {
    fn name(&self) -> &'static str {
        "calloc"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let nmemb = extract_concrete_arg(&args[0], "nmemb")?;
        let size = extract_concrete_arg(&args[1], "size")?;

        let total = nmemb
            .checked_mul(size)
            .ok_or_else(|| ProcedureError::Other("calloc overflow".to_string()))?;

        if total > 1024 * 1024 {
            return Err(ProcedureError::Other(format!(
                "calloc size {} exceeds 1MB limit",
                total
            )));
        }

        let addr = state.heap_alloc(total);

        // Zero-fill the allocated memory
        if total > 0 {
            let mut offset = 0u64;
            while offset + 8 <= total {
                let bv = RustBV::concrete(0, 64);
                state.memory_store(addr.wrapping_add(offset), bv)?;
                offset += 8;
            }
            while offset < total {
                let bv = RustBV::concrete(0, 8);
                state.memory_store(addr.wrapping_add(offset), bv)?;
                offset += 1;
            }
        }

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(addr as u128, bits)))
    }
}

/// Native realloc implementation.
///
/// ```c
/// void *realloc(void *ptr, size_t size);
/// ```
///
/// Simplified: just allocates new block and copies (no real freeing).
pub struct NativeRealloc;

impl NativeSimProcedure for NativeRealloc {
    fn name(&self) -> &'static str {
        "realloc"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let ptr = extract_concrete_arg(&args[0], "ptr")?;
        let size = extract_concrete_arg(&args[1], "size")?;

        if size > 1024 * 1024 {
            return Err(ProcedureError::Other(format!(
                "realloc size {} exceeds 1MB limit",
                size
            )));
        }

        // Get old allocation size before freeing (for copy length)
        let old_size = state.heap_metadata().alloc_size(ptr);
        let new_addr = state.heap_alloc(size);

        // Copy old data if ptr != NULL
        if ptr != 0 && size > 0 {
            // Free the old allocation (metadata only, bump allocator doesn't reclaim)
            state.heap_free(ptr);
            // Copy min(size, old_size) bytes
            let copy_len = old_size.map_or(size, |os| size.min(os));
            let mut offset = 0u64;
            while offset < copy_len {
                let chunk = std::cmp::min(copy_len - offset, 8);
                match state.memory_load(ptr.wrapping_add(offset), chunk as u32) {
                    Ok(val) => {
                        let _ = state.memory_store(new_addr.wrapping_add(offset), val);
                    }
                    Err(_) => break, // Stop copying on unmapped memory
                }
                offset += chunk;
            }
        }

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(new_addr as u128, bits)))
    }
}

/// Native memalign implementation.
///
/// ```c
/// void *memalign(size_t alignment, size_t size);
/// ```
///
/// glibc semantics: returns a pointer to `size` bytes aligned to `alignment`.
/// `alignment` must be a power of two; if it is 0 or 1 we treat the call
/// as a plain `malloc` to match the bump-allocator behavior used elsewhere.
pub struct NativeMemalign;

impl NativeSimProcedure for NativeMemalign {
    fn name(&self) -> &'static str {
        "memalign"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let alignment = extract_concrete_arg(&args[0], "alignment")?;
        let size = extract_concrete_arg(&args[1], "size")?;

        if alignment > 1 && !alignment.is_power_of_two() {
            return Err(ProcedureError::Other(format!(
                "memalign alignment {} is not a power of two",
                alignment
            )));
        }
        if size > 1024 * 1024 {
            return Err(ProcedureError::Other(format!(
                "memalign size {} exceeds 1MB limit",
                size
            )));
        }

        let addr = state.heap_alloc_aligned(size, alignment);
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(addr as u128, bits)))
    }
}

/// Native posix_memalign implementation.
///
/// ```c
/// int posix_memalign(void **memptr, size_t alignment, size_t size);
/// ```
///
/// On success, stores the allocated pointer at `*memptr` and returns 0.
/// On invalid alignment (non-power-of-2, or not a multiple of `sizeof(void*)`),
/// returns `EINVAL` (22) without touching `*memptr`. The return value is the
/// errno code itself — glibc does NOT set the global `errno`.
pub struct NativePosixMemalign;

impl NativeSimProcedure for NativePosixMemalign {
    fn name(&self) -> &'static str {
        "posix_memalign"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let memptr = extract_concrete_arg(&args[0], "memptr")?;
        let alignment = extract_concrete_arg(&args[1], "alignment")?;
        let size = extract_concrete_arg(&args[2], "size")?;

        let bits = state.arch().bits();
        let ptr_bytes = (bits / 8) as u64;

        let einval = 22u64;
        if alignment < ptr_bytes || !alignment.is_power_of_two() || alignment % ptr_bytes != 0 {
            return Ok(Some(RustBV::concrete(einval as u128, 32)));
        }
        if size > 1024 * 1024 {
            return Err(ProcedureError::Other(format!(
                "posix_memalign size {} exceeds 1MB limit",
                size
            )));
        }

        let addr = state.heap_alloc_aligned(size, alignment);
        // Store the allocated pointer at *memptr.
        let ptr_bv = RustBV::concrete(addr as u128, bits);
        state.memory_store(memptr, ptr_bv)?;

        Ok(Some(RustBV::concrete(0, 32)))
    }
}

#[cfg(test)]
#[path = "malloc_tests.rs"]
mod malloc_tests;
