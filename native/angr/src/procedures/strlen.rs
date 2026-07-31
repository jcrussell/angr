//! Native strlen / strnlen implementations.
//!
//! strlen returns the length of a null-terminated string, not including the
//! null terminator. strnlen does the same but bounded by maxlen.
//!
//! # Behavior
//!
//! - Concrete address required (symbolic addresses fall back to Python).
//! - Concrete fast path scans byte-by-byte.
//! - When a symbolic byte is encountered, we build an ITE chain expressing
//!   the symbolic null position:
//!   result = ITE(b_i == 0, i, result_next)
//!   built right-to-left up to MAX_STRLEN (or a concrete null). The chain's
//!   initial right-most value is the upper bound (MAX_STRLEN for strlen,
//!   maxlen for strnlen). This is an approximation: if no null is ever
//!   reachable in MAX_STRLEN bytes the result will saturate at MAX_STRLEN.
//! - Maximum string length is 4096 bytes (configurable).

use super::ProcedureError;
use super::strings::{
    ScanOutcome, build_strlen_chain, null_exists_constraint, scan_for_null_symbolic,
};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Maximum string length before falling back to Python.
const MAX_STRLEN: usize = 4096;

/// Shared scan for strlen / strnlen. `max_scan` is the upper bound on the
/// number of positions inspected (MAX_STRLEN for strlen, min(maxlen, MAX) for
/// strnlen). `require_null` asserts a terminator exists in the window (strlen
/// only — strnlen legitimately saturates at `maxlen`). Returns
/// Ok(Some(length_bv)) on success.
fn scan_for_null(
    state: &mut RustSimState,
    addr: u64,
    max_scan: u64,
    require_null: bool,
) -> Result<Option<RustBV>, ProcedureError> {
    let arch_bits = state.arch().bits();
    if max_scan == 0 {
        return Ok(Some(RustBV::concrete(0u128, arch_bits)));
    }

    match scan_for_null_symbolic(state, addr, max_scan)? {
        ScanOutcome::AllConcrete { length } => {
            // Hit a concrete null mid-scan → that's the length.
            // Ran the full `max_scan` without finding null → for strnlen the
            // natural answer is `max_scan`; for strlen we ran past MAX_STRLEN
            // and should error out (preserves prior behavior).
            if length == max_scan && max_scan >= MAX_STRLEN as u64 {
                return Err(ProcedureError::MaxIterations(MAX_STRLEN));
            }
            Ok(Some(RustBV::concrete(length as u128, arch_bits)))
        }
        ScanOutcome::Symbolic { bytes } => {
            let (chain, pruning) = {
                let ctx = state.solver().borrow();
                let chain = build_strlen_chain(&bytes, arch_bits, max_scan, &ctx);
                let pruning = if require_null {
                    null_exists_constraint(&bytes, &ctx)
                } else {
                    None
                };
                (chain, pruning)
            };
            if let Some(c) = pruning {
                state.add_constraint(c);
            }
            Ok(Some(chain))
        }
    }
}

crate::declare_proc! {
    /// Native strlen: `size_t strlen(const char *s)`.
    ///
    /// Returns the number of bytes before the first null byte.
    name = "strlen",
    struct = NativeStrlen,
    args = [addr: concrete],
    call |state| {
        scan_for_null(state, addr, MAX_STRLEN as u64, /*require_null=*/true)
    }
}

crate::declare_proc! {
    /// Native strnlen: `size_t strnlen(const char *s, size_t maxlen)`.
    ///
    /// Returns the lesser of the string length and maxlen.
    name = "strnlen",
    struct = NativeStrnlen,
    args = [s: concrete, maxlen: concrete],
    call |state| {
        if maxlen > MAX_STRLEN as u64 {
            return Err(ProcedureError::MaxIterations(maxlen as usize));
        }
        scan_for_null(state, s, maxlen, /*require_null=*/false)
    }
}

#[cfg(test)]
#[path = "strlen_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod strlen_tests;
