//! Native memcpy implementation.
//!
//! memcpy copies n bytes from source to destination and returns the
//! destination pointer.
//!
//! # Behavior
//!
//! - If `dst` and/or `src` is a symbolic address and `size` is concrete, the
//!   native path enumerates the bounded candidate sets for both and emits, per
//!   `(dst_candidate, src_candidate)` pair, conditional stores
//!   `ITE(dst == d && src == s, src_byte, original)`. Falls back to Python when
//!   the size is symbolic, either candidate set is unbounded, or the
//!   `|dst| * |src| * size` store budget is exceeded.
//! - A concrete `size` takes the fast 8-byte-chunk path.
//! - A symbolic `size` is handled natively via bounded conditional stores: byte
//!   `i` of `dst` is set to `ITE(i < n, src[i], dst[i])` for `i` up to the
//!   solver's upper bound on `n` (mirrors the memset symbolic-size path). Falls
//!   back to Python when that bound is unknown or exceeds
//!   `MAX_SYMBOLIC_BYTEWISE_SIZE`.
//! - Copies data byte-by-byte, preserving symbolic values
//! - Maximum concrete copy size is 1MB (configurable)

use super::mem_common::{
    MAX_SYMBOLIC_ADDR_STORES, bounded_symbolic_size, check_symbolic_addr_size,
    enumerate_addr_candidates, symbolic_size_conditional_store,
};
use super::{ProcedureError, check_max, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use std::collections::HashMap;

/// Maximum copy size before falling back to Python.
const MAX_COPY_SIZE: usize = 1024 * 1024; // 1MB

/// True when a `memmove` of `size` bytes from `src` to `dst` must copy
/// *backwards* — i.e. `dst` lies strictly inside the source region
/// `[src, src + size)`, so a forward copy would clobber source bytes it has not
/// read yet.
///
/// The distance is computed with `wrapping_sub` so the classification stays
/// correct when the source region wraps past `u64::MAX`. `dst.wrapping_sub(src)`
/// is exactly the number of bytes from `src` forward (modulo 2^64) to `dst`, so
/// `0 < delta < size` is the wrap-safe spelling of `dst > src && dst < src +
/// size`. Every address computation in this file wraps (see the
/// `wrapping_add` calls in `copy_forward` and `NativeMemmove::call`); a plain
/// `src + size` here would panic in debug builds and silently misclassify the
/// overlap — picking the wrong copy direction — in release.
fn memmove_copies_backward(dst: u64, src: u64, size: u64) -> bool {
    let delta = dst.wrapping_sub(src);
    delta != 0 && delta < size
}

/// Copy `size` bytes from a SYMBOLIC `src` and/or `dst` address.
///
/// Both pointers may be symbolic. Each is resolved to its (bounded) set of
/// concrete solutions; then for every `(dst_candidate d, src_candidate s)` pair
/// and byte `i` the destination byte at `d + i` is set to
/// `ITE(dst == d && src == s, src_byte[s + i], original)`. Distinct pairs
/// compose correctly because `dst`/`src` can each equal at most one candidate,
/// so at most one guard is ever true at any physical address.
///
/// All source bytes are snapshotted *before* any store, so overlapping `src`/
/// `dst` regions copy pre-store values (the memmove contract) — both `memcpy`
/// and `memmove` share this helper for that reason.
///
/// Falls back to Python (`Err`) when the size is symbolic, either candidate set
/// is empty/unbounded, or the `|dst| * |src| * size` store budget is exceeded.
fn copy_symbolic_addr(
    state: &mut RustSimState,
    dst_bv: &RustBV,
    src_bv: &RustBV,
    size_bv: &RustBV,
) -> Result<Option<RustBV>, ProcedureError> {
    // Symbolic address + symbolic size is out of scope; require a concrete size.
    let size = match check_symbolic_addr_size(size_bv)? {
        None => return Ok(Some(dst_bv.clone())),
        Some(s) => s,
    };

    // Enumerate candidate addresses under a cap (unbounded pointers bail).
    let dst_cands = enumerate_addr_candidates(state, dst_bv, "dst")?;
    let src_cands = enumerate_addr_candidates(state, src_bv, "src")?;
    // Bound total work: |dst| * |src| * size conditional stores.
    let pairs = dst_cands.len() as u64 * src_cands.len() as u64;
    if pairs.saturating_mul(size) > MAX_SYMBOLIC_ADDR_STORES {
        return Err(ProcedureError::SymbolicArgument("dst".to_string()));
    }

    // Snapshot every source byte that could be read, BEFORE any store, so
    // overlapping src/dst regions copy pre-store values (memmove contract).
    let mut src_bytes: HashMap<u64, RustBV> = HashMap::new();
    for &s in &src_cands {
        for i in 0..size {
            let p = s.wrapping_add(i);
            if let std::collections::hash_map::Entry::Vacant(e) = src_bytes.entry(p) {
                e.insert(state.memory_load(p, 1)?);
            }
        }
    }

    let dwidth = dst_bv.width();
    let swidth = src_bv.width();
    for &d in &dst_cands {
        let dcond = {
            let ctx = state.solver().borrow();
            dst_bv.eq(&RustBV::concrete(d as u128, dwidth), &ctx)
        };
        for &s in &src_cands {
            let guard = {
                let ctx = state.solver().borrow();
                let scond = src_bv.eq(&RustBV::concrete(s as u128, swidth), &ctx);
                dcond.and(&scond, &ctx)
            };
            for i in 0..size {
                let dp = d.wrapping_add(i);
                let src_byte = &src_bytes[&s.wrapping_add(i)];
                let orig = state.memory_load(dp, 1)?;
                let stored = {
                    let ctx = state.solver().borrow();
                    guard.ite(src_byte, &orig, &ctx)
                };
                state.memory_store(dp, stored)?;
            }
        }
    }
    Ok(Some(dst_bv.clone()))
}

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
/// `size_bv` or that bound exceeds `MAX_SYMBOLIC_BYTEWISE_SIZE`, triggering the
/// Python fallback.
fn copy_symbolic_size(
    state: &mut RustSimState,
    src: u64,
    dst: u64,
    size_bv: &RustBV,
) -> Result<(), ProcedureError> {
    let Some(max_bound) = bounded_symbolic_size(state, size_bv)? else {
        return Ok(());
    };

    // Snapshot every source byte that could be copied before mutating dst, so
    // overlapping src/dst regions observe pre-store values (memmove contract).
    let mut src_bytes = Vec::with_capacity(max_bound as usize);
    for i in 0..max_bound {
        src_bytes.push(state.memory_load(src.wrapping_add(i), 1)?);
    }

    symbolic_size_conditional_store(state, dst, size_bv, max_bound, &src_bytes)
}

/// Copy `size` bytes forward from `src` to `dst` using 8-byte chunks.
///
/// Both the loads and the stores propagate their errors, so a caller that
/// hands this an unreadable source region (or an unwritable destination) gets
/// a `ProcedureError` and falls back to Python rather than a half-copied
/// buffer. `NativeRealloc` reuses it for exactly that reason (angr-sqfj8.82).
pub(super) fn copy_forward(
    state: &mut RustSimState,
    src: u64,
    dst: u64,
    size: usize,
) -> Result<(), ProcedureError> {
    let mut offset: usize = 0;

    // Copy in 8-byte chunks where possible
    // overflow-ok: offset is bounded by `size`, a natively-copied length
    // capped well below usize::MAX (native memcpy/memmove size limits).
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

/// Outcome of `resolve_copy_args` — the argument-resolution preamble shared by
/// `NativeMemcpy` and `NativeMemmove`.
enum CopyPlan {
    /// Nothing is left for the caller to do: the copy was either already
    /// performed by one of the symbolic paths or was a zero-byte no-op. The
    /// payload is the procedure's return value.
    Done(Option<RustBV>),
    /// All three arguments resolved to concrete values and the (non-empty) copy
    /// still has to be made — the one step where `memcpy` and `memmove` differ.
    Concrete { dst: u64, src: u64, size: usize },
}

/// Resolve `memcpy`/`memmove`'s three arguments, handling every case the two
/// procedures share and leaving only the direction-specific concrete copy to
/// the caller.
///
/// The cases handled here — symbolic `dst`/`src` (via `copy_symbolic_addr`),
/// symbolic `size` (via `copy_symbolic_size`), the `MAX_COPY_SIZE` cap and the
/// zero-size short-circuit — are identical for both procedures: both symbolic
/// helpers snapshot every source byte before storing, so they already satisfy
/// the memmove overlap contract and need no backward-copy special case. Only
/// the concrete-size copy differs, which is why that is the one variant handed
/// back rather than performed here.
fn resolve_copy_args(
    state: &mut RustSimState,
    dst_bv: &RustBV,
    src_bv: &RustBV,
    size_bv: &RustBV,
) -> Result<CopyPlan, ProcedureError> {
    // A symbolic `dst` and/or `src` takes the bounded candidate-pair path.
    let (dst, src) = match (
        extract_concrete_arg(dst_bv, "dst"),
        extract_concrete_arg(src_bv, "src"),
    ) {
        (Ok(d), Ok(s)) => (d, s),
        _ => return copy_symbolic_addr(state, dst_bv, src_bv, size_bv).map(CopyPlan::Done),
    };
    // `dst` is concrete here, so the returned dest pointer is the
    // value-equivalent BV (the manual impl returned args[0]).
    let ret = dst_bv.clone();

    // --- Symbolic size: bounded conditional stores. ---
    let Some(size) = size_bv.as_u64() else {
        copy_symbolic_size(state, src, dst, size_bv)?;
        return Ok(CopyPlan::Done(Some(ret)));
    };
    let size = size as usize;

    check_max(size as u64, MAX_COPY_SIZE)?;

    // Handle zero-size copy
    if size == 0 {
        return Ok(CopyPlan::Done(Some(ret)));
    }

    Ok(CopyPlan::Concrete { dst, src, size })
}

crate::declare_proc! {
    /// Native memcpy: `void *memcpy(void *dest, const void *src, size_t n)`.
    ///
    /// Copies n bytes from src to dest. Returns dest. Everything but the final
    /// copy is shared with `NativeMemmove` via `resolve_copy_args`.
    name = "memcpy",
    struct = NativeMemcpy,
    args = [dst_bv: bv, src_bv: bv, size_bv: bv],
    call |state| {
        match resolve_copy_args(state, &dst_bv, &src_bv, &size_bv)? {
            CopyPlan::Done(ret) => Ok(ret),
            CopyPlan::Concrete { dst, src, size } => {
                copy_forward(state, src, dst, size)?;
                Ok(Some(dst_bv))
            }
        }
    }
}

crate::declare_proc! {
    /// Native memmove: `void *memmove(void *dest, const void *src, size_t n)`.
    ///
    /// Like memcpy, but handles overlapping regions correctly. Only that final
    /// concrete copy differs; the argument preamble is shared with
    /// `NativeMemcpy` via `resolve_copy_args` (whose symbolic paths are already
    /// overlap-safe, so they need no backward-copy special case).
    name = "memmove",
    struct = NativeMemmove,
    args = [dst_bv: bv, src_bv: bv, size_bv: bv],
    call |state| {
        match resolve_copy_args(state, &dst_bv, &src_bv, &size_bv)? {
            CopyPlan::Done(ret) => Ok(ret),
            CopyPlan::Concrete { dst, src, size } => {
                if memmove_copies_backward(dst, src, size as u64) {
                    // Overlapping with dst inside [src, src + size): copy
                    // backwards so we never clobber a source byte before
                    // reading it.
                    for i in (0..size).rev() {
                        let value = state.memory_load(src.wrapping_add(i as u64), 1)?;
                        state.memory_store(dst.wrapping_add(i as u64), value)?;
                    }
                } else {
                    // Non-overlapping or dst < src: copy forwards
                    copy_forward(state, src, dst, size)?;
                }
                // Return dest pointer
                Ok(Some(dst_bv))
            }
        }
    }
}

crate::declare_proc! {
    /// Native mempcpy: `void *mempcpy(void *dest, const void *src, size_t n)`.
    ///
    /// Like `memcpy`, but returns `dest + n` instead of `dest`. Reuses the
    /// native `memcpy` copy logic (DRY) and only adjusts the return value;
    /// matches Python `mempcpy` (`return dst_addr + limit`). When the underlying
    /// `memcpy` defers to Python (symbolic/oversized), the `?` propagates so the
    /// whole call falls back, preserving parity.
    name = "mempcpy",
    struct = NativeMempcpy,
    args = [dst_bv: bv, src_bv: bv, size_bv: bv],
    call |state| {
        NativeMemcpy.call(state, &[dst_bv.clone(), src_bv, size_bv.clone()])?;
        // mempcpy returns dst + n (size_t n is the same width as the pointer).
        let ret = {
            let ctx = state.solver().borrow();
            dst_bv.add(&size_bv, &ctx)
        };
        Ok(Some(ret))
    }
}

test_submod!("memcpy_tests.rs" => tests);
