//! Native memcpy implementation.
//!
//! memcpy copies n bytes from source to destination and returns the
//! destination pointer.
//!
//! # Behavior
//!
//! - If `dst` or `src` is symbolic (a symbolic address), falls back to Python.
//! - A concrete `size` takes the fast 8-byte-chunk path.
//! - A symbolic `size` is handled natively via bounded conditional stores: byte
//!   `i` of `dst` is set to `ITE(i < n, src[i], dst[i])` for `i` up to the
//!   solver's upper bound on `n` (mirrors the memset symbolic-size path). Falls
//!   back to Python when that bound is unknown or exceeds
//!   `MAX_SYMBOLIC_COPY_SIZE`.
//! - Copies data byte-by-byte, preserving symbolic values
//! - Maximum concrete copy size is 1MB (configurable)

use super::ProcedureError;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Maximum copy size before falling back to Python.
const MAX_COPY_SIZE: usize = 1024 * 1024; // 1MB

/// Maximum symbolic copy size before falling back to Python. The symbolic path
/// emits per-byte `memory_load` + `ITE` + `memory_store`, far costlier than the
/// concrete chunked path, so it is capped much lower (matches memset).
const MAX_SYMBOLIC_COPY_SIZE: u64 = 4096;

/// Copy a symbolic-length region from `src` to `dst` via bounded conditional
/// stores. Byte `i` of `dst` becomes `ITE(i < n, src[i], dst[i])`, so positions
/// past the (symbolic) length keep their prior contents.
///
/// All source bytes in `[0, max_bound)` are read *before* any store so the copy
/// stays correct when `src` and `dst` overlap (the memmove contract): dst bytes
/// are then read and written once each in increasing order, never re-reading a
/// position already stored.
///
/// Returns `Err(SymbolicArgument)` when the solver has no usable upper bound on
/// `size_bv` or that bound exceeds `MAX_SYMBOLIC_COPY_SIZE`, triggering the
/// Python fallback.
fn copy_symbolic_size(
    state: &mut RustSimState,
    src: u64,
    dst: u64,
    size_bv: &RustBV,
) -> Result<(), ProcedureError> {
    let max_bound = {
        let ctx = state.solver().borrow();
        ctx.max(size_bv, false)
    };
    let max_bound = match max_bound {
        Some(m) if m <= MAX_SYMBOLIC_COPY_SIZE as u128 => m as u64,
        _ => return Err(ProcedureError::SymbolicArgument("size".to_string())),
    };
    if max_bound == 0 {
        return Ok(());
    }

    // Snapshot every source byte that could be copied before mutating dst.
    let mut src_bytes = Vec::with_capacity(max_bound as usize);
    for i in 0..max_bound {
        src_bytes.push(state.memory_load(src.wrapping_add(i), 1)?);
    }

    let width = size_bv.width();
    for i in 0..max_bound {
        let orig = state.memory_load(dst.wrapping_add(i), 1)?;
        let stored = {
            let ctx = state.solver().borrow();
            let cond = RustBV::concrete(i as u128, width).ult(size_bv, &ctx);
            cond.ite(&src_bytes[i as usize], &orig, &ctx)
        };
        state.memory_store(dst.wrapping_add(i), stored)?;
    }
    Ok(())
}

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
    args = [dst: concrete, src: concrete, size_bv: bv],
    call |state| {
        let arch_bits = state.arch().bits();
        // `dst` is required concrete, so the returned dest pointer is the
        // value-equivalent concrete BV (the manual impl returned args[0]).
        let ret = RustBV::concrete(dst as u128, arch_bits);

        // --- Symbolic size: bounded conditional stores. ---
        let Some(size) = size_bv.as_u64() else {
            copy_symbolic_size(state, src, dst, &size_bv)?;
            return Ok(Some(ret));
        };
        let size = size as usize;

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
    args = [dst: concrete, src: concrete, size_bv: bv],
    call |state| {
        let arch_bits = state.arch().bits();
        // `dst` is required concrete, so the returned dest pointer is the
        // value-equivalent concrete BV (the manual impl returned args[0]).
        let ret = RustBV::concrete(dst as u128, arch_bits);

        // --- Symbolic size: bounded conditional stores. ---
        //
        // `copy_symbolic_size` snapshots all source bytes before storing, so it
        // handles overlapping regions correctly without the backward-copy
        // special case the concrete path uses below.
        let Some(size) = size_bv.as_u64() else {
            copy_symbolic_size(state, src, dst, &size_bv)?;
            return Ok(Some(ret));
        };
        let size = size as usize;

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
