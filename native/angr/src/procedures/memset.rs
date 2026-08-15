//! Native memset implementation.
//!
//! memset fills a memory region with a constant byte value.
//!
//! # Behavior
//!
//! - A concrete address takes the paths below. A symbolic address is handled
//!   natively when its set of concrete solutions is small: each candidate
//!   address `a` gets per-byte conditional stores `ITE(dest == a, fill,
//!   original)`. Falls back to Python when the candidate set is unbounded, the
//!   size is symbolic, or the total store budget is exceeded.
//! - A concrete size takes the fast chunked path (8-byte stores).
//! - A symbolic size is handled natively via bounded conditional stores: byte
//!   `i` is set to `ITE(i < n, fill, original_byte)` for `i` up to the solver's
//!   upper bound on `n`. Falls back to Python when that bound is unknown or
//!   exceeds `MAX_SYMBOLIC_BYTEWISE_SIZE` (the per-byte ITE path is far costlier
//!   than the concrete chunked path, so its cap is much smaller).
//! - A symbolic byte `value` is handled natively: the low 8 bits are stored
//!   (symbolically) into every byte of the region — no Python fallback.
//! - Maximum concrete size is 1MB (configurable)

use super::mem_common::{
    MAX_SYMBOLIC_ADDR_STORES, bounded_symbolic_size, check_symbolic_addr_size,
    enumerate_addr_candidates, symbolic_size_conditional_store,
};
use super::{ProcedureError, extract_concrete_arg};
use crate::symbolic::RustBV;

/// Maximum concrete memset size before falling back to Python.
const MAX_MEMSET_SIZE: u64 = 1024 * 1024;

crate::declare_proc! {
    /// Native memset implementation.
    ///
    /// ```c
    /// void *memset(void *s, int c, size_t n);
    /// ```
    ///
    /// Fills n bytes at s with byte value c. Returns s. `dest` is declared
    /// `bv` (not `concrete`) so the original pointer BV can be returned
    /// verbatim; it is extracted to a concrete u64 in the body. `value` is
    /// also `bv`: a concrete byte takes the fast 8-byte-chunk path, while a
    /// symbolic byte is stored (low 8 bits) into every byte natively. `size`
    /// is `bv` too: a concrete size takes the chunked path, a symbolic size
    /// takes the bounded conditional-store path (or falls back to Python).
    name = "memset",
    struct = NativeMemset,
    args = [dest_bv: bv, value_bv: bv, size_bv: bv],
    call |state| {
        // Build the 8-bit fill byte. For a concrete value it is the low byte;
        // for a symbolic value it is the low 8 bits (`extract`).
        let byte_bv = {
            let ctx = state.solver().borrow();
            match value_bv.as_u64() {
                Some(value) => RustBV::concrete((value & 0xFF) as u128, 8),
                None => value_bv.extract(7, 0, &ctx),
            }
        };

        // Resolve the destination. A concrete dest takes the fast/symbolic-size
        // paths below; a symbolic dest takes the bounded multi-candidate
        // conditional-store path.
        let dest = match extract_concrete_arg(&dest_bv, "dest") {
            Ok(d) => d,
            Err(_) => return memset_symbolic_addr(state, &dest_bv, &byte_bv, &size_bv),
        };

        // --- Concrete size: fast chunked path (8-byte stores). ---
        if let Some(size) = size_bv.as_u64() {
            if size > MAX_MEMSET_SIZE {
                return Err(ProcedureError::Other(format!(
                    "memset size {size} exceeds maximum {MAX_MEMSET_SIZE}"
                )));
            }
            if size == 0 {
                return Ok(Some(dest_bv));
            }

            // 64-bit chunk: the fill byte repeated 8 times. For a concrete
            // byte this is a precomputed u64; for a symbolic byte it
            // concatenates 8 copies (endianness-agnostic — every byte is
            // identical).
            let chunk_bv = match byte_bv.as_u64() {
                Some(byte_val) => {
                    let byte_val = byte_val as u8;
                    let mut fill_8: u64 = 0;
                    for i in 0..8 {
                        fill_8 |= (byte_val as u64) << (i * 8);
                    }
                    RustBV::concrete(fill_8 as u128, 64)
                }
                None => {
                    let ctx = state.solver().borrow();
                    let parts: [RustBV; 8] = core::array::from_fn(|_| byte_bv.clone());
                    RustBV::concat_balanced(&parts, &ctx)
                }
            };

            // Fill memory using 8-byte chunks where possible.
            let mut offset: u64 = 0;
            // overflow-ok: offset is bounded by `size`, a natively-serviced
            // length capped well below u64::MAX.
            while offset + 8 <= size {
                state.memory_store(dest.wrapping_add(offset), chunk_bv.clone())?;
                offset += 8;
            }
            // Handle remaining bytes
            while offset < size {
                state.memory_store(dest.wrapping_add(offset), byte_bv.clone())?;
                offset += 1;
            }

            return Ok(Some(dest_bv));
        }

        // --- Symbolic size: bounded conditional stores. ---
        //
        // Determine an upper bound on `n`. The conditional-store loop must run
        // for every byte that *could* be filled, so the bound has to be the
        // true solver max; an unknown or too-large bound falls back to Python.
        let Some(max_size) = bounded_symbolic_size(state, &size_bv)? else {
            return Ok(Some(dest_bv));
        };

        // Byte `i` is set to `ITE(i < n, fill, original)`: positions past the
        // (symbolic) length keep their prior contents. The fill byte is the
        // same for every position.
        let values = vec![byte_bv; max_size as usize];
        symbolic_size_conditional_store(state, dest, &size_bv, max_size, &values)?;

        Ok(Some(dest_bv))
    }
}

/// memset with a SYMBOLIC destination address.
///
/// Enumerates the (bounded) set of concrete addresses the pointer can take and,
/// for each candidate `a`, emits per-byte conditional stores
/// `ITE(dest == a, fill, original)`. Distinct candidates compose correctly
/// because `dest` can equal at most one of them: a later candidate's store at a
/// shared position layers another guarded ITE over the earlier one, and only
/// one guard is ever true.
///
/// Falls back to Python (`Err`) when the size is symbolic, the candidate set is
/// empty/unbounded, or the `candidates * size` store budget is exceeded.
fn memset_symbolic_addr(
    state: &mut crate::state::RustSimState,
    dest_bv: &RustBV,
    byte_bv: &RustBV,
    size_bv: &RustBV,
) -> Result<Option<RustBV>, ProcedureError> {
    // Symbolic address + symbolic size is out of scope; require a concrete size.
    let size = match check_symbolic_addr_size(size_bv)? {
        None => return Ok(Some(dest_bv.clone())),
        Some(s) => s,
    };

    // Enumerate candidate addresses under a cap (unbounded pointers bail).
    let candidates = enumerate_addr_candidates(state, dest_bv, "dest")?;
    // Bound total work: candidates * size conditional stores.
    if candidates.len() as u64 * size > MAX_SYMBOLIC_ADDR_STORES {
        return Err(ProcedureError::SymbolicArgument("dest".to_string()));
    }

    let width = dest_bv.width();
    for a in candidates {
        let cond = {
            let ctx = state.solver().borrow();
            dest_bv.eq(&RustBV::concrete(a as u128, width), &ctx)
        };
        for i in 0..size {
            let p = a.wrapping_add(i);
            let orig = state.memory_load(p, 1)?;
            let stored = {
                let ctx = state.solver().borrow();
                cond.ite(byte_bv, &orig, &ctx)
            };
            state.memory_store(p, stored)?;
        }
    }
    Ok(Some(dest_bv.clone()))
}

crate::declare_proc! {
    /// `void bzero(void *s, size_t n)` — zero `n` bytes at `s`.
    ///
    /// Matches Python angr (`procedures/posix/bzero.py`), which subclasses
    /// `memset` and forwards `memset(s, 0, n)`. Reuses native `memset` (DRY,
    /// same pattern as `__memset_chk`): builds an 8-bit zero fill byte and
    /// delegates to [`NativeMemset`]. The C return type is `void`; memset's
    /// dest-pointer return is harmless and ignored by callers.
    name = "bzero",
    struct = NativeBzero,
    args = [dest_bv: bv, size_bv: bv],
    call |state| {
        NativeMemset.call(state, &[dest_bv, RustBV::concrete(0, 8), size_bv])
    }
}

test_submod!("memset_tests.rs" => tests);
