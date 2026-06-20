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
//! The per-pointer arity and the `candidates * size` (memset) vs
//! `|dst| * |src| * size` (memcpy) store-budget guard stay caller-side because
//! they genuinely differ between the two procedures.

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
/// falls back to Python. The caller computes the candidate product.
pub(crate) const MAX_SYMBOLIC_ADDR_STORES: u64 = 4096;

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
