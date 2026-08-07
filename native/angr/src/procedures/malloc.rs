//! Native malloc/free/calloc/realloc implementations.
//!
//! Uses a simple bump allocator (matching angr's SimHeapBrk).
//! Symbolic sizes fall back to Python.

use super::ProcedureError;
use super::check_max;
use crate::symbolic::RustBV;

/// Maximum allocation size accepted by the native allocator procedures
/// (calloc/realloc/memalign/posix_memalign). Beyond this we defer to Python
/// rather than growing the bump heap. Mirrors memcpy.rs's `MAX_COPY_SIZE`.
const MAX_ALLOC_SIZE: usize = 1024 * 1024; // 1MB

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

        check_max(total, MAX_ALLOC_SIZE)?;

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
    /// Simplified: allocates a new block and copies. The bump allocator never
    /// reclaims memory, so "freeing" the old block is metadata-only
    /// (`RustSimState::heap_free`) — but it happens for *every* non-NULL `ptr`,
    /// including `realloc(ptr, 0)`, which POSIX/glibc treat as a free of the
    /// original allocation. Gating the free on `size > 0` would leave `ptr`
    /// marked live in `HeapMetadata` forever.
    name = "realloc",
    struct = NativeRealloc,
    args = [ptr: concrete, size: concrete],
    call |state| {
        check_max(size, MAX_ALLOC_SIZE)?;

        // Get old allocation size before freeing (for copy length)
        let old_size = state.heap_metadata().alloc_size(ptr);
        let new_addr = state.heap_alloc(size);

        // Copy old data if ptr != NULL and the new block can hold anything.
        // angr-sqfj8.82: this used to `break` on a failed load and drop every
        // store's `Result`, so a source region with an unmapped page partway
        // through returned a buffer whose tail was silently whatever
        // `heap_alloc` left behind. `copy_forward` propagates both, so such a
        // realloc now falls back to Python, which reads unmapped memory under
        // angr's own fill semantics instead of guessing.
        if ptr != 0 && size > 0 {
            // Copy min(size, old_size) bytes
            let copy_len = old_size.map_or(size, |os| size.min(os));
            super::memcpy::copy_forward(state, ptr, new_addr, copy_len as usize)?;
        }

        // Free the old allocation (metadata only, bump allocator doesn't
        // reclaim). Unconditional on ptr != 0: realloc(ptr, 0) frees too.
        // Runs *after* the copy so a copy failure leaves `ptr` still live in
        // HeapMetadata for the Python fallback, which re-runs the whole
        // realloc. (The new bump allocation above is leaked on that path —
        // unavoidable without a second read pass, and the bump heap never
        // reclaims anyway.)
        if ptr != 0 {
            state.heap_free(ptr);
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
        check_max(size, MAX_ALLOC_SIZE)?;

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
        check_max(size, MAX_ALLOC_SIZE)?;

        let addr = state.heap_alloc_aligned(size, alignment);
        // Store the allocated pointer at *memptr.
        let ptr_bv = RustBV::concrete(addr as u128, bits);
        state.memory_store(memptr, ptr_bv)?;

        Ok(Some(RustBV::concrete(0, 32)))
    }
}

#[cfg(test)]
#[path = "malloc_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod malloc_tests;
