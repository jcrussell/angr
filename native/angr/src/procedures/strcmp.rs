//! Native strcmp/strncmp/strcasecmp implementations.
//!
//! strcmp compares two null-terminated strings lexicographically.
//! strncmp compares at most n characters.
//! strcasecmp compares case-insensitively.
//!
//! # Behavior
//!
//! - Concrete addresses are required (symbolic addresses fall back to Python).
//! - Concrete fast path scans byte-by-byte and short-circuits on the first
//!   mismatch or null terminator (mirroring libc behavior).
//! - When the scan encounters a symbolic byte (or for strncmp when n is
//!   symbolic — currently unsupported), we switch to building a 32-bit ITE
//!   chain expressing the byte-wise diff:
//!   result = ITE(c1_i != c2_i, sext(c1_i) - sext(c2_i),
//!   ITE(c1_i == 0, 0, result_next))   -- strcmp/strncmp
//!   result = ITE(c1_i != c2_i, sext(c1_i) - sext(c2_i), result_next) -- memcmp
//! - Maximum compare length is 4096 bytes (configurable).

use super::ProcedureError;
use super::strings::{ConcreteStep, ScanResult, scan_concrete_then_collect};
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

/// Maximum string length before falling back to Python.
pub(super) const MAX_STRCMP_LEN: usize = 4096;

/// Per-position case-folding option used by strcasecmp.
fn case_fold_byte(byte: &RustBV, ctx: &SymContext) -> RustBV {
    // result = ITE(byte in [A, Z], byte + 32, byte) over 8 bits
    let lo = RustBV::concrete(b'A' as u128, 8);
    let hi = RustBV::concrete(b'Z' as u128, 8);
    let in_range = byte.uge(&lo, ctx).and(&byte.ule(&hi, ctx), ctx);
    let delta = RustBV::concrete(32u128, 8);
    let lowered = byte.add(&delta, ctx);
    in_range.ite(&lowered, byte, ctx)
}

/// Build the ITE chain from a list of (c1_i, c2_i) pairs (each 8-bit BVs).
fn build_diff_chain(
    pairs: &[(RustBV, RustBV)],
    stop_at_null: bool,
    case_insensitive: bool,
    ctx: &SymContext,
) -> RustBV {
    let zero32 = RustBV::concrete(0u128, 32);
    let zero8 = RustBV::concrete(0u128, 8);
    let mut result = zero32.clone();
    for (c1, c2) in pairs.iter().rev() {
        let (lhs, rhs) = if case_insensitive {
            (case_fold_byte(c1, ctx), case_fold_byte(c2, ctx))
        } else {
            (c1.clone(), c2.clone())
        };
        let diff = lhs.zero_extend(32, ctx).sub(&rhs.zero_extend(32, ctx), ctx);
        let mismatch = lhs.ne(&rhs, ctx);
        if stop_at_null {
            // result = ITE(c1 != c2, diff, ITE(c1 == 0, 0, result_next))
            let null_cond = c1.eq(&zero8, ctx);
            let inner = null_cond.ite(&zero32, &result, ctx);
            result = mismatch.ite(&diff, &inner, ctx);
        } else {
            // memcmp: result = ITE(c1 != c2, diff, result_next)
            result = mismatch.ite(&diff, &result, ctx);
        }
    }
    result
}

/// Shared scan body for strcmp / strncmp / strcasecmp / memcmp.
pub(super) fn compare_bytes(
    state: &mut RustSimState,
    s1_addr: u64,
    s2_addr: u64,
    max_len: u64,
    stop_at_null: bool,
    case_insensitive: bool,
) -> Result<Option<RustBV>, ProcedureError> {
    if max_len == 0 {
        return Ok(Some(RustBV::zero(32)));
    }
    if max_len > MAX_STRCMP_LEN as u64 {
        return Err(ProcedureError::MaxIterations(MAX_STRCMP_LEN));
    }

    let result = scan_concrete_then_collect(
        state,
        max_len,
        |st, i| {
            let c1 = st.memory_load(s1_addr.wrapping_add(i), 1)?;
            let c2 = st.memory_load(s2_addr.wrapping_add(i), 1)?;
            Ok((c1, c2))
        },
        |(c1_val, c2_val), _i| match (c1_val.as_u64(), c2_val.as_u64()) {
            (Some(b1), Some(b2)) => {
                let mut a = b1 as u8;
                let mut b = b2 as u8;
                if case_insensitive {
                    if a.is_ascii_uppercase() {
                        a += 32;
                    }
                    if b.is_ascii_uppercase() {
                        b += 32;
                    }
                }
                if a != b {
                    let diff = (a as i32) - (b as i32);
                    ConcreteStep::Stop(RustBV::concrete(diff as u128, 32))
                } else if stop_at_null && a == 0 {
                    ConcreteStep::Stop(RustBV::zero(32))
                } else {
                    ConcreteStep::Continue
                }
            }
            // At least one byte is symbolic — switch to ITE-chain collection.
            _ => ConcreteStep::BeginCollect,
        },
        // Symbolic-mode collection stops when c1 is concretely null (positions
        // past the null cannot affect the result for strcmp).
        |(c1_val, _c2_val)| {
            stop_at_null && c1_val.as_u64().map(|b| (b as u8) == 0).unwrap_or(false)
        },
    )?;

    Ok(Some(match result {
        ScanResult::Stopped(r) => r,
        ScanResult::Exhausted => {
            // Walked through max_len with all-concrete bytes and no
            // mismatch / null hit. For strcmp/strncmp this means we ran out
            // of room — error out (matches the prior MaxIterations behavior).
            // For memcmp, equal-up-to-limit is the natural "0" return.
            if stop_at_null && max_len >= MAX_STRCMP_LEN as u64 {
                return Err(ProcedureError::MaxIterations(MAX_STRCMP_LEN));
            }
            RustBV::zero(32)
        }
        ScanResult::Collected(collected) => {
            let pairs: Vec<(RustBV, RustBV)> =
                collected.into_iter().map(|(_, pair)| pair).collect();
            let ctx = state.solver().borrow();
            build_diff_chain(&pairs, stop_at_null, case_insensitive, &ctx)
        }
    }))
}

crate::declare_proc! {
    /// Native strcmp: `int strcmp(const char *s1, const char *s2)`.
    ///
    /// Returns < 0, 0, or > 0 per lexicographic comparison.
    name = "strcmp",
    struct = NativeStrcmp,
    args = [s1: concrete, s2: concrete],
    call |state| {
        compare_bytes(state, s1, s2, MAX_STRCMP_LEN as u64,
                      /*stop_at_null=*/true, /*case_insensitive=*/false)
    }
}

crate::declare_proc! {
    /// Native strncmp: `int strncmp(const char *s1, const char *s2, size_t n)`.
    ///
    /// Like strcmp, but compares at most `n` characters.
    name = "strncmp",
    struct = NativeStrncmp,
    args = [s1: concrete, s2: concrete, n: concrete],
    call |state| {
        let max_len = n.min(MAX_STRCMP_LEN as u64);
        compare_bytes(state, s1, s2, max_len,
                      /*stop_at_null=*/true, /*case_insensitive=*/false)
    }
}

crate::declare_proc! {
    /// Native strcasecmp (case-insensitive strcmp).
    name = "strcasecmp",
    struct = NativeStrcasecmp,
    args = [s1: concrete, s2: concrete],
    call |state| {
        compare_bytes(state, s1, s2, MAX_STRCMP_LEN as u64,
                      /*stop_at_null=*/true, /*case_insensitive=*/true)
    }
}

#[cfg(test)]
#[path = "strcmp_tests.rs"]
mod strcmp_tests;
