//! Native malloc/free/calloc/realloc implementations.
//!
//! Uses a simple bump allocator (matching angr's SimHeapBrk).
//! Symbolic sizes fall back to Python.

use crate::state::RustSimState;
use crate::symbolic::RustBV;
use super::{NativeSimProcedure, ProcedureError};

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
        let size = args[0].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("size".to_string())
        })?;

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
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // SimHeapBrk's free is a no-op (returns unconstrained, but we skip that)
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
        let nmemb = args[0].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("nmemb".to_string())
        })?;
        let size = args[1].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("size".to_string())
        })?;

        let total = nmemb.checked_mul(size).ok_or_else(|| {
            ProcedureError::Other("calloc overflow".to_string())
        })?;

        if total > 1024 * 1024 {
            return Err(ProcedureError::Other(format!(
                "calloc size {} exceeds 1MB limit", total
            )));
        }

        let addr = state.heap_alloc(total);

        // Zero-fill the allocated memory
        if total > 0 {
            let mut offset = 0u64;
            while offset + 8 <= total {
                let bv = RustBV::concrete(0, 64);
                state.memory_store(addr.wrapping_add(offset), bv)
                    .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
                offset += 8;
            }
            while offset < total {
                let bv = RustBV::concrete(0, 8);
                state.memory_store(addr.wrapping_add(offset), bv)
                    .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;
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
        let ptr = args[0].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("ptr".to_string())
        })?;
        let size = args[1].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("size".to_string())
        })?;

        if size > 1024 * 1024 {
            return Err(ProcedureError::Other(format!(
                "realloc size {} exceeds 1MB limit", size
            )));
        }

        let new_addr = state.heap_alloc(size);

        // Copy old data if ptr != NULL
        if ptr != 0 && size > 0 {
            // Copy min(size, old_size) bytes — we don't track old_size,
            // so copy `size` bytes (may read garbage, which is fine for symbolic execution)
            let mut offset = 0u64;
            while offset < size {
                let chunk = std::cmp::min(size - offset, 8);
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
