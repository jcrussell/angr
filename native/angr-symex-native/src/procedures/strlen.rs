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
//!   built right-to-left over the collected window, which ends at a concrete
//!   null, at the scan bound, or once
//!   [`MAX_SYMBOLIC_SCAN_BYTES`]
//!   symbolic bytes have been collected. strlen also asserts
//!   [`null_exists_constraint`], so
//!   the length lies inside that window and the chain's default is never
//!   taken; this prunes the states where every collected byte is non-null, as
//!   Python's strlen does. strnlen asserts nothing: its default is `maxlen`,
//!   so it defers to Python when the symbolic budget is spent before `maxlen`.
//! - A concrete scan with no null in MAX_STRLEN bytes defers to Python.
//! - Maximum string length is the str-family-wide
//!   [`strings::MAX_STRING_SCAN`](super::strings::MAX_STRING_SCAN), aliased
//!   here as `MAX_STRLEN`.

use super::ProcedureError;
use super::check_max;
use super::strings::{
    MAX_STRING_SCAN as MAX_STRLEN, MAX_SYMBOLIC_SCAN_BYTES, ScanOutcome, build_strlen_chain,
    null_exists_constraint, scan_for_null_symbolic,
};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

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
        ScanOutcome::Symbolic {
            bytes,
            budget_spent,
        } => {
            // strnlen asserts no terminator, so past a spent budget its
            // `max_scan` default would stand for bytes nobody read
            // (angr-vmi2l.99). strlen's `null_exists_constraint` (always
            // `Some` here: the window holds symbolic bytes and no concrete
            // null) forces the length into the prefix, as Python's does.
            if budget_spent && !require_null {
                return Err(ProcedureError::MaxIterations(MAX_SYMBOLIC_SCAN_BYTES));
            }
            let (chain, pruning) = {
                let chain = build_strlen_chain(&bytes, arch_bits, max_scan);
                let pruning = if require_null {
                    null_exists_constraint(&bytes)
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
        check_max(maxlen, MAX_STRLEN)?;
        scan_for_null(state, s, maxlen, /*require_null=*/false)
    }
}

test_submod!("strlen_tests.rs" => strlen_tests);
