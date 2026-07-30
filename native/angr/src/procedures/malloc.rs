//! Native malloc/free/calloc/realloc implementations.
//!
//! Uses a simple bump allocator (matching angr's SimHeapBrk).
//! Symbolic sizes fall back to Python.

use super::ProcedureError;
use crate::symbolic::RustBV;

/// Maximum allocation size accepted by the native allocator procedures
/// (calloc/realloc/memalign/posix_memalign). Beyond this we defer to Python
/// rather than growing the bump heap. Mirrors memcpy.rs's `MAX_COPY_SIZE`.
const MAX_ALLOC_SIZE: u64 = 1024 * 1024; // 1MB

crate::declare_proc! {
    /// Native malloc: `void *malloc(size_t size)`.
    name = "malloc",
    struct = NativeMalloc,
    args = [size: concrete],
    call |state| {
        let addr = state.heap_alloc(size);
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(addr as u128, bits)))
    }
}

crate::declare_proc! {
    /// Native free (no-op for bump allocator): `void free(void *ptr)`.
    name = "free",
    struct = NativeFree,
    args = [ptr: concrete],
    call |state| {
        // Track the free (bump allocator doesn't reclaim memory)
        state.heap_free(ptr);
        Ok(None)
    }
}

crate::declare_proc! {
    /// Native calloc: `void *calloc(size_t nmemb, size_t size)`.
    name = "calloc",
    struct = NativeCalloc,
    args = [nmemb: concrete, size: concrete],
    call |state| {
        let total = nmemb
            .checked_mul(size)
            .ok_or_else(|| ProcedureError::Other("calloc overflow".to_string()))?;

        if total > MAX_ALLOC_SIZE {
            return Err(ProcedureError::Other(format!(
                "calloc size {total} exceeds 1MB limit"
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

crate::declare_proc! {
    /// Native realloc: `void *realloc(void *ptr, size_t size)`.
    ///
    /// Simplified: just allocates new block and copies (no real freeing).
    name = "realloc",
    struct = NativeRealloc,
    args = [ptr: concrete, size: concrete],
    call |state| {
        if size > MAX_ALLOC_SIZE {
            return Err(ProcedureError::Other(format!(
                "realloc size {size} exceeds 1MB limit"
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

crate::declare_proc! {
    /// Native memalign: `void *memalign(size_t alignment, size_t size)`.
    ///
    /// glibc semantics: returns a pointer to `size` bytes aligned to
    /// `alignment`. `alignment` must be a power of two; if it is 0 or 1 we
    /// treat the call as a plain `malloc` to match the bump-allocator
    /// behavior used elsewhere.
    name = "memalign",
    struct = NativeMemalign,
    args = [alignment: concrete, size: concrete],
    call |state| {
        if alignment > 1 && !alignment.is_power_of_two() {
            return Err(ProcedureError::Other(format!(
                "memalign alignment {alignment} is not a power of two"
            )));
        }
        if size > MAX_ALLOC_SIZE {
            return Err(ProcedureError::Other(format!(
                "memalign size {size} exceeds 1MB limit"
            )));
        }

        let addr = state.heap_alloc_aligned(size, alignment);
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(addr as u128, bits)))
    }
}

crate::declare_proc! {
    /// Native posix_memalign:
    /// `int posix_memalign(void **memptr, size_t alignment, size_t size)`.
    ///
    /// On success, stores the allocated pointer at `*memptr` and returns 0.
    /// On invalid alignment (non-power-of-2, or not a multiple of
    /// `sizeof(void*)`), returns `EINVAL` (22) without touching `*memptr`.
    /// The return value is the errno code itself — glibc does NOT set the
    /// global `errno`.
    name = "posix_memalign",
    struct = NativePosixMemalign,
    args = [memptr: concrete, alignment: concrete, size: concrete],
    call |state| {
        let bits = state.arch().bits();
        let ptr_bytes = (bits / 8) as u64;

        let einval = 22u64;
        if alignment < ptr_bytes || !alignment.is_power_of_two() || alignment % ptr_bytes != 0 {
            return Ok(Some(RustBV::concrete(einval as u128, 32)));
        }
        if size > MAX_ALLOC_SIZE {
            return Err(ProcedureError::Other(format!(
                "posix_memalign size {size} exceeds 1MB limit"
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
