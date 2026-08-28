//! Shared helpers for the symbolic-destination-address store paths in the
//! `mem*` procedures (`memset`, `memcpy`/`memmove`).
//!
//! Both procedures resolve a symbolic pointer to a bounded set of concrete
//! address candidates and then emit per-byte `ITE(ptr == candidate, ...)`
//! conditional stores. The size validation and candidate-enumeration preambles
//! were byte-for-byte identical (memcpy's comments literally said "matches
//! memset"); centralizing them keeps the caps and rejection logic in one place
//! so the two paths cannot silently drift.
//!
//! How the *unit count* is derived stays caller-side because it genuinely
//! differs between the two procedures (`candidates` for memset,
//! `|dst| * |src|` for memcpy), but both feed it to [`check_store_budget`] so
//! the overflow-safe multiply and the `MAX_SYMBOLIC_ADDR_STORES` comparison
//! live in one place.

use super::ProcedureError;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Maximum number of concrete address solutions for a symbolic pointer before
/// falling back to Python. `eval_upto` is asked for one more than this so an
/// unbounded pointer is detected and rejected.
pub(crate) const MAX_SYMBOLIC_ADDR_CANDIDATES: usize = 64;

/// Maximum size for a symbolic-address store. The per-byte ITE path runs
/// `candidates * size` (or `|dst| * |src| * size`) conditional stores, so the
/// size is capped tightly.
pub(crate) const MAX_SYMBOLIC_ADDR_SIZE: u64 = 256;

/// Total conditional-store budget for the symbolic-address path. Exceeding it
/// falls back to Python. The caller computes the unit count and passes it to
/// [`check_store_budget`].
pub(crate) const MAX_SYMBOLIC_ADDR_STORES: u64 = 4096;

/// Reject a symbolic-address store whose `units * size` conditional-store count
/// exceeds [`MAX_SYMBOLIC_ADDR_STORES`], so the caller falls back to Python.
///
/// `units` is the caller-computed candidate count (`|candidates|` for memset,
/// `|dst| * |src|` for memcpy); `size` is the validated per-candidate byte
/// count from [`check_symbolic_addr_size`]. The multiply saturates rather than
/// wrapping: a product too large to represent is unambiguously over budget, so
/// saturation refuses where a bare `*` would wrap to a small in-budget value
/// (see the `invariant-rust-concrete-arith-must-wrap` bd memory for why a size
/// must refuse rather than wrap). Both current callers are bounded well below
/// `u64::MAX`, so this is a shape guard for the next one.
pub(crate) fn check_store_budget(units: u64, size: u64, name: &str) -> Result<(), ProcedureError> {
    if units.saturating_mul(size) > MAX_SYMBOLIC_ADDR_STORES {
        return Err(ProcedureError::SymbolicArgument(name.to_string()));
    }
    Ok(())
}

/// Validate a (possibly symbolic) size for a symbolic-address store path.
///
/// - `Ok(None)` when `size == 0` (caller should no-op and return the pointer).
/// - `Ok(Some(size))` for a valid in-range concrete size.
/// - `Err(SymbolicArgument("size"))` when the size is symbolic or exceeds
///   [`MAX_SYMBOLIC_ADDR_SIZE`].
pub(crate) fn check_symbolic_addr_size(size_bv: &RustBV) -> Result<Option<u64>, ProcedureError> {
    let size = size_bv
        .as_u64()
        .ok_or_else(|| ProcedureError::SymbolicArgument("size".to_string()))?;
    if size == 0 {
        return Ok(None);
    }
    if size > MAX_SYMBOLIC_ADDR_SIZE {
        return Err(ProcedureError::SymbolicArgument("size".to_string()));
    }
    Ok(Some(size))
}

/// Enumerate the bounded set of concrete addresses a (possibly symbolic)
/// pointer can take.
///
/// Asks the solver for one more than the cap so an unbounded pointer is
/// detected and rejected. A concrete pointer yields a single-element set.
/// Returns `Err(SymbolicArgument(name))` when the candidate set is empty or
/// exceeds [`MAX_SYMBOLIC_ADDR_CANDIDATES`].
pub(crate) fn enumerate_addr_candidates(
    state: &RustSimState,
    ptr: &RustBV,
    name: &str,
) -> Result<Vec<u64>, ProcedureError> {
    let candidates = {
        let ctx = state.solver().borrow();
        ctx.eval_upto(ptr, MAX_SYMBOLIC_ADDR_CANDIDATES + 1)
    };
    if candidates.is_empty() || candidates.len() > MAX_SYMBOLIC_ADDR_CANDIDATES {
        return Err(ProcedureError::SymbolicArgument(name.to_string()));
    }
    Ok(candidates.into_iter().map(|a| a as u64).collect())
}

/// Maximum size for a native symbolic-*length* bytewise store loop before
/// falling back to Python. The loop emits one `memory_load` + `ITE` +
/// `memory_store` per byte up to the solver's upper bound on the length, so the
/// bound is capped to keep the generated formula tractable. Shared by the
/// `memset` (fill-byte) and `memcpy`/`memmove` (source-byte) symbolic-size
/// paths, which previously kept two separate `MAX_SYMBOLIC_*_SIZE = 4096`
/// constants.
pub(crate) const MAX_SYMBOLIC_BYTEWISE_SIZE: u64 = 4096;

/// Resolve a solver upper bound for a symbolic length, capped at
/// [`MAX_SYMBOLIC_BYTEWISE_SIZE`].
///
/// - `Ok(None)` when the bound is `0` (caller should no-op).
/// - `Ok(Some(max))` for a valid in-range bound.
/// - `Err(SymbolicArgument("size"))` when the solver cannot bound the length or
///   the bound exceeds the cap (triggering the Python fallback).
///
/// Split out from [`symbolic_size_conditional_store`] so callers that must
/// pre-snapshot source bytes (memcpy's memmove contract) can size their
/// snapshot to the bound before any store happens.
pub(crate) fn bounded_symbolic_size(
    state: &RustSimState,
    size_bv: &RustBV,
) -> Result<Option<u64>, ProcedureError> {
    let max_size = {
        let ctx = state.solver().borrow();
        ctx.max(size_bv, false)
    };
    match max_size {
        Some(0) => Ok(None),
        Some(m) if m <= MAX_SYMBOLIC_BYTEWISE_SIZE as u128 => Ok(Some(m as u64)),
        _ => Err(ProcedureError::SymbolicArgument("size".to_string())),
    }
}

/// Emit the bounded conditional-store loop for a symbolic *length*. Byte `i` of
/// `dest` becomes `ITE(i < n, values[i], original)`, so positions past the
/// (symbolic) length `n = size_bv` keep their prior contents.
///
/// `max_size` is the bound from [`bounded_symbolic_size`]; `values` must be at
/// least `max_size` long (memset passes a repeated fill byte, memcpy passes its
/// pre-snapshotted source bytes). The caller pre-snapshots any source bytes
/// that alias `dest` so overlapping copies observe pre-store values.
pub(crate) fn symbolic_size_conditional_store(
    state: &mut RustSimState,
    dest: u64,
    size_bv: &RustBV,
    max_size: u64,
    values: &[RustBV],
) -> Result<(), ProcedureError> {
    let width = size_bv.width();
    for i in 0..max_size {
        let orig = state.memory_load(dest.wrapping_add(i), 1)?;
        let stored = {
            let ctx = state.solver().borrow();
            let cond = RustBV::concrete(i as u128, width).ult(size_bv, &ctx);
            cond.ite(&values[i as usize], &orig, &ctx)
        };
        state.memory_store(dest.wrapping_add(i), stored)?;
    }
    Ok(())
}

test_submod!("mem_common_tests.rs" => tests);
