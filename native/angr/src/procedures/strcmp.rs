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
//!   mismatch or null terminator (mirroring libc behavior). The mismatch
//!   result is the SIGN (-1 / 1), matching Python's `strncmp`/`memcmp`
//!   SimProcedures rather than glibc's implementation-defined raw byte
//!   difference (angr-e71o4).
//! - When the scan encounters a symbolic byte (or for strncmp when n is
//!   symbolic — currently unsupported), we switch to building a 32-bit ITE
//!   chain over `sign_i = ITE(c1_i <u c2_i, -1, 1)`:
//!   result = ITE(c1_i != c2_i, sign_i,
//!   ITE(c1_i == 0, 0, result_next))   -- strcmp/strncmp
//!   result = ITE(c1_i != c2_i, sign_i, result_next) -- memcmp
//!
//!   The symbolic path returns the same `-1 / 0 / 1` alphabet as the concrete
//!   path above (angr-u8gm8); it previously emitted the raw
//!   `zext(c1,32) - zext(c2,32)` difference, which disagreed both with our own
//!   concrete path and with Python's `memcmp.py` (`ite_cases` over
//!   `BVV(-1)/BVV(0)/BVV(1)`). Python's *non-static* `strncmp.py` constrains
//!   its return to `0`/`1` and never produces a negative value at all, so no
//!   convention can match every Python proc; we match `memcmp.py` and stay
//!   internally consistent, which is what exact-value consumers
//!   (`res == -1`, table indexing) need. The clamp is also cheaper for Z3
//!   than a 32-bit subtraction per position.
//! - Maximum compare length is 4096 bytes (configurable).

use super::ProcedureError;
use super::check_max;
use super::ctype;
use super::strings::{
    ConcreteStep, ScanResult, null_exists_constraint, scan_concrete_then_collect,
};
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

/// Maximum string length before falling back to Python.
pub(super) const MAX_STRCMP_LEN: usize = 4096;

/// Per-position case-folding option used by strcasecmp: `ITE(byte in [A, Z],
/// byte + 32, byte)` over 8 bits.
///
/// Delegates to [`ctype::case_shift_bv`] — the same helper `tolower` is built
/// on — so `strcasecmp`'s symbolic fold and `tolower`/`toupper` cannot drift
/// apart (angr-6cp06.3). [`case_fold_byte_concrete`] does the same for the
/// concrete path (angr-fs8kb.38).
fn case_fold_byte(byte: &RustBV, ctx: &SymContext) -> RustBV {
    ctype::case_shift_bv(byte, b'A', b'Z', 32, ctx)
}

/// Concrete counterpart of [`case_fold_byte`], for `compare_bytes`'s
/// all-concrete fast path.
///
/// Delegates to [`ctype::case_shift_concrete`] — the concrete twin of the
/// helper [`case_fold_byte`] uses — so the fold rule is not spelled a third
/// time (angr-fs8kb.38). A fold of a byte is still a byte ([A, Z] + 32 = [a, z]),
/// so narrowing the `u64` result back is lossless.
fn case_fold_byte_concrete(byte: u8) -> u8 {
    ctype::case_shift_concrete(u64::from(byte), b'A', b'Z', 32) as u8
}

/// Build the ITE chain from a list of (c1_i, c2_i) pairs (each 8-bit BVs).
fn build_diff_chain(
    pairs: &[(RustBV, RustBV)],
    stop_at_null: bool,
    case_insensitive: bool,
    ctx: &SymContext,
) -> RustBV {
    let zero32 = RustBV::concrete(0u128, 32);
    let one32 = RustBV::concrete(1u128, 32);
    let neg_one32 = RustBV::concrete(u32::MAX as u128, 32);
    let zero8 = RustBV::concrete(0u128, 8);
    let mut result = zero32.clone();
    for (c1, c2) in pairs.iter().rev() {
        let (lhs, rhs) = if case_insensitive {
            (case_fold_byte(c1, ctx), case_fold_byte(c2, ctx))
        } else {
            (c1.clone(), c2.clone())
        };
        // Sign, not raw difference — see the module doc (angr-u8gm8).
        let diff = lhs.ult(&rhs, ctx).ite(&neg_one32, &one32, ctx);
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
///
/// `unbounded` says where `max_len` came from, and must NOT be re-derived from
/// `max_len >= MAX_STRCMP_LEN`: `true` means the window is this module's own
/// cap (strcmp/strcasecmp, which compare until a terminator), `false` means it
/// is the caller's own `n` (strncmp/strncasecmp/memcmp). The distinction
/// decides both whether exhausting the window is an answer or a shortfall and
/// whether s1 may be forced to terminate inside it — see the two
/// `ScanResult` arms below (angr-fs8kb.36). It mirrors `strlen::scan_for_null`'s
/// explicit `require_null`, which is why strnlen never had this bug.
pub(super) fn compare_bytes(
    state: &mut RustSimState,
    s1_addr: u64,
    s2_addr: u64,
    max_len: u64,
    unbounded: bool,
    stop_at_null: bool,
    case_insensitive: bool,
) -> Result<Option<RustBV>, ProcedureError> {
    if max_len == 0 {
        return Ok(Some(RustBV::zero(32)));
    }
    check_max(max_len, MAX_STRCMP_LEN)?;

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
                let (a, b) = if case_insensitive {
                    (
                        case_fold_byte_concrete(b1 as u8),
                        case_fold_byte_concrete(b2 as u8),
                    )
                } else {
                    (b1 as u8, b2 as u8)
                };
                if a != b {
                    // Return the SIGN, not the raw byte difference: Python's
                    // strncmp/memcmp SimProcedures return exactly -1 or 1 on a
                    // concrete mismatch, and glibc's magnitude is
                    // implementation-defined anyway. Matching keeps rax
                    // identical across engines for exact-value consumers
                    // (`res == -1`, table indexing) — see angr-e71o4.
                    let signed = if a < b { -1i32 } else { 1i32 };
                    ConcreteStep::Stop(RustBV::concrete(signed as u32 as u128, 32))
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
            // mismatch / null hit. For strcmp/strcasecmp that means we ran out
            // of room before finding the terminator the compare is defined in
            // terms of — error out so Python retries with its own window.
            // For a caller-bounded compare (strncmp/strncasecmp/memcmp) the
            // window IS the whole compare, so equal-up-to-n is the answer: 0.
            if stop_at_null && unbounded {
                return Err(ProcedureError::MaxIterations(MAX_STRCMP_LEN));
            }
            RustBV::zero(32)
        }
        ScanResult::Collected(collected) => {
            // Only the unbounded compares (strcmp/strcasecmp) may assert that
            // s1 is terminated inside the window: a caller-bounded
            // strncmp(a, b, n) can legitimately compare n non-null bytes, at
            // ANY n — C does not require termination for a bounded compare,
            // and Python's strncmp.py lets the null sit beyond the compared
            // range. Deriving this from `max_len >= MAX_STRCMP_LEN` instead
            // pruned exactly those states for n >= 4096 (angr-fs8kb.36).
            // See strings::null_exists_constraint (angr-sgcye).
            let require_null = stop_at_null && unbounded;
            let s1_bytes: Vec<(u64, RustBV)> = collected
                .iter()
                .map(|(i, (c1, _))| (*i, c1.clone()))
                .collect();
            let pairs: Vec<(RustBV, RustBV)> =
                collected.into_iter().map(|(_, pair)| pair).collect();
            let (chain, pruning) = {
                let ctx = state.solver().borrow();
                let chain = build_diff_chain(&pairs, stop_at_null, case_insensitive, &ctx);
                let pruning = if require_null {
                    null_exists_constraint(&s1_bytes, &ctx)
                } else {
                    None
                };
                (chain, pruning)
            };
            if let Some(c) = pruning {
                state.add_constraint(c);
            }
            chain
        }
    }))
}

crate::declare_proc! {
    /// Native strcmp: `int strcmp(const char *s1, const char *s2)`.
    ///
    /// Returns < 0, 0, or > 0 per lexicographic comparison.
    ///
    /// `strcoll` is aliased here: in the C/POSIX locale (angr's default,
    /// matching `procedures/libc/strcoll.py` which inline-calls strcmp) the
    /// locale-aware comparison degenerates to a plain lexicographic strcmp.
    name = "strcmp",
    struct = NativeStrcmp,
    args = [s1: concrete, s2: concrete],
    aliases = ["strcoll"],
    call |state| {
        compare_bytes(state, s1, s2, MAX_STRCMP_LEN as u64, /*unbounded=*/true,
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
        // Deliberately NOT clamped to MAX_STRCMP_LEN: a clamped window would
        // answer for the first 4096 bytes while claiming to answer for `n`.
        // compare_bytes's own check_max defers n > MAX to Python, which sizes
        // its window off strlen instead (angr-fs8kb.36) — same shape as
        // strnlen's check_max.
        compare_bytes(state, s1, s2, n, /*unbounded=*/false,
                      /*stop_at_null=*/true, /*case_insensitive=*/false)
    }
}

crate::declare_proc! {
    /// Native strcasecmp (case-insensitive strcmp).
    name = "strcasecmp",
    struct = NativeStrcasecmp,
    args = [s1: concrete, s2: concrete],
    call |state| {
        compare_bytes(state, s1, s2, MAX_STRCMP_LEN as u64, /*unbounded=*/true,
                      /*stop_at_null=*/true, /*case_insensitive=*/true)
    }
}

crate::declare_proc! {
    /// Native strncasecmp: `int strncasecmp(const char *s1, const char *s2, size_t n)`.
    ///
    /// Like strcasecmp, but compares at most `n` characters (mirrors
    /// `procedures/libc/strncasecmp.py`, which is strncmp with `ignore_case=True`).
    name = "strncasecmp",
    struct = NativeStrncasecmp,
    args = [s1: concrete, s2: concrete, n: concrete],
    call |state| {
        // Unclamped `n`, for the reason spelled out on strncmp above.
        compare_bytes(state, s1, s2, n, /*unbounded=*/false,
                      /*stop_at_null=*/true, /*case_insensitive=*/true)
    }
}

test_submod!("strcmp_tests.rs" => strcmp_tests);
