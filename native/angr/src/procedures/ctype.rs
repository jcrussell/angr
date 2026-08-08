//! Native character classification functions (ctype.h).
//!
//! isdigit, isalpha, isspace, isalnum, isupper, islower, isxdigit, isprint,
//! isascii, isblank, iscntrl, isgraph, ispunct, tolower, toupper.
//!
//! Each takes a single int argument and returns 0 or non-zero.
//! Symbolic arguments are handled by emitting a constraint-shaped result that
//! mirrors the concrete predicate on bits\[7:0\] of the argument; the operand
//! pattern matches the underlying `as u8` truncation in the concrete path.
//!
//! **Panic policy (angr-9ke6b.212):** the guest-supplied argument reaches these
//! procedures as a `RustBV` and is never unwrapped — a symbolic argument takes
//! the predicate path, a concrete one the fast path. The two `expect`s fold a
//! predicate accumulator whose emptiness depends only on the `ranges` /
//! `members` table the *caller* passes, and every caller passes a
//! `const`-shaped slice literal.
//!
//! **Enforcement (angr-qwyti.11):** this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::ProcedureError;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

/// C-locale `isspace()`: `' '` plus the `0x09..=0x0d` run (`\t \n \v \f \r`).
///
/// The single source of truth for "is this byte whitespace" across the native
/// procedures — `NativeIsSpace` below plus the leading-whitespace skips in
/// `strtod::floating_prefix_len` / `strtod`'s endptr rescan and
/// `strtol::parse_concrete_prefix`. Deliberately *not* Rust's
/// `is_ascii_whitespace`, which omits `\v` (0x0b); see
/// `invariant-ctype-mirror-python-not-rust-std`.
pub(crate) fn is_c_space(c: u8) -> bool {
    c == b' ' || (0x09..=0x0d).contains(&c)
}

/// Truncate an argument to its low 8 bits, matching the concrete `as u8` path.
fn arg_byte(arg: &RustBV, ctx: &SymContext) -> RustBV {
    arg.extract(7, 0, ctx)
}

/// Build `byte >= lo && byte <= hi` as a 1-bit BV.
fn byte_in_range(byte: &RustBV, lo: u8, hi: u8, ctx: &SymContext) -> RustBV {
    let lo_bv = RustBV::concrete(lo as u128, 8);
    let hi_bv = RustBV::concrete(hi as u128, 8);
    byte.uge(&lo_bv, ctx).and(&byte.ule(&hi_bv, ctx), ctx)
}

/// Zero-extend a 1-bit predicate to arch.bits() and return it.
fn lift_predicate(state: &RustSimState, pred: RustBV, ctx: &SymContext) -> RustBV {
    let bits = state.arch().bits();
    pred.zero_extend(bits, ctx)
}

/// tolower/toupper share the same pattern: if byte is in [lo, hi], shift by `delta`.
fn case_shift(
    state: &RustSimState,
    arg: &RustBV,
    lo: u8,
    hi: u8,
    delta: i8,
) -> Result<Option<RustBV>, ProcedureError> {
    let bits = state.arch().bits();
    if let Some(c) = arg.as_u64() {
        let b = c as u8;
        let result = if b >= lo && b <= hi {
            ((b as i16) + delta as i16) as u8
        } else {
            b
        };
        return Ok(Some(RustBV::concrete(result as u128, bits)));
    }
    let ctx = state.solver().borrow();
    let byte = arg_byte(arg, &ctx);
    let in_range = byte_in_range(&byte, lo, hi, &ctx);
    let delta_bv = RustBV::concrete(delta as u8 as u128, 8);
    let shifted = byte.add(&delta_bv, &ctx);
    let new_byte = in_range.ite(&shifted, &byte, &ctx);
    Ok(Some(new_byte.zero_extend(bits, &ctx)))
}

/// Build a symbolic ctype predicate from a list of inclusive ranges.
#[allow(
    clippy::expect_used,
    reason = "`pred` is `Some` after the loop iff `ranges` is non-empty, and `ranges` is a slice literal fixed at each call site (isdigit/isalpha/... tables), never guest data"
)]
fn ranges_predicate(
    state: &RustSimState,
    arg: &RustBV,
    ranges: &[(u8, u8)],
    concrete_check: fn(u8) -> bool,
) -> Result<Option<RustBV>, ProcedureError> {
    let bits = state.arch().bits();
    if let Some(c) = arg.as_u64() {
        return Ok(Some(RustBV::concrete(
            u128::from(concrete_check(c as u8)),
            bits,
        )));
    }
    let ctx = state.solver().borrow();
    let byte = arg_byte(arg, &ctx);
    let mut pred: Option<RustBV> = None;
    for &(lo, hi) in ranges {
        let r = byte_in_range(&byte, lo, hi, &ctx);
        pred = Some(match pred {
            None => r,
            Some(prev) => prev.or(&r, &ctx),
        });
    }
    let pred = pred.expect("ranges must be non-empty");
    Ok(Some(lift_predicate(state, pred, &ctx)))
}

/// Build a symbolic ctype predicate from an explicit set of bytes.
#[allow(
    clippy::expect_used,
    reason = "`pred` is `Some` after the loop iff `members` is non-empty, and `members` is a slice literal fixed at each call site (isspace/isblank/... tables), never guest data"
)]
fn set_predicate(
    state: &RustSimState,
    arg: &RustBV,
    members: &[u8],
    concrete_check: fn(u8) -> bool,
) -> Result<Option<RustBV>, ProcedureError> {
    let bits = state.arch().bits();
    if let Some(c) = arg.as_u64() {
        return Ok(Some(RustBV::concrete(
            u128::from(concrete_check(c as u8)),
            bits,
        )));
    }
    let ctx = state.solver().borrow();
    let byte = arg_byte(arg, &ctx);
    let mut pred: Option<RustBV> = None;
    for &m in members {
        let m_bv = RustBV::concrete(m as u128, 8);
        let eq = byte.eq(&m_bv, &ctx);
        pred = Some(match pred {
            None => eq,
            Some(prev) => prev.or(&eq, &ctx),
        });
    }
    let pred = pred.expect("set must be non-empty");
    Ok(Some(lift_predicate(state, pred, &ctx)))
}

/// Return a locale ctype table pointer as an arch-width concrete BV, or an
/// `Err` (→ Python fallback) when the table was never built. The three glibc
/// `__ctype_*_loc` accessors all delegate here; the table itself is malloc'd
/// and populated by Python's `__libc_start_main` init pass and the pointer is
/// pushed into Rust at seed-state creation (see [`crate::state::CtypeLocPtrs`]).
fn ctype_loc_ptr(
    state: &RustSimState,
    ptr: Option<u64>,
    name: &str,
) -> Result<Option<RustBV>, ProcedureError> {
    let addr = ptr.ok_or_else(|| ProcedureError::Other(format!("{name} table not initialized")))?;
    Ok(Some(RustBV::concrete(addr as u128, state.arch().bits())))
}

crate::declare_proc! {
    /// `const unsigned short **__ctype_b_loc(void)` — pointer to the locale
    /// character-classification table (returns Python's
    /// `state.libc.ctype_b_loc_table_ptr` verbatim).
    name = "__ctype_b_loc",
    struct = NativeCtypeBLoc,
    args = [],
    call |state| {
        ctype_loc_ptr(state, state.ctype_loc().b, "__ctype_b_loc")
    }
}

crate::declare_proc! {
    /// `const int32_t **__ctype_tolower_loc(void)` — pointer to the locale
    /// lowercase-conversion table.
    name = "__ctype_tolower_loc",
    struct = NativeCtypeToLowerLoc,
    args = [],
    call |state| {
        ctype_loc_ptr(state, state.ctype_loc().tolower, "__ctype_tolower_loc")
    }
}

crate::declare_proc! {
    /// `const int32_t **__ctype_toupper_loc(void)` — pointer to the locale
    /// uppercase-conversion table.
    name = "__ctype_toupper_loc",
    struct = NativeCtypeToUpperLoc,
    args = [],
    call |state| {
        ctype_loc_ptr(state, state.ctype_loc().toupper, "__ctype_toupper_loc")
    }
}

crate::declare_proc! {
    /// `int isdigit(int c)` — returns nonzero if `c` is an ASCII digit.
    name = "isdigit",
    struct = NativeIsDigit,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(b'0', b'9')], |c| c.is_ascii_digit())
    }
}

crate::declare_proc! {
    /// `int isalpha(int c)`.
    name = "isalpha",
    struct = NativeIsAlpha,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(b'A', b'Z'), (b'a', b'z')], |c| c.is_ascii_alphabetic())
    }
}

crate::declare_proc! {
    /// `int isspace(int c)`.
    /// C locale whitespace: ' ' plus the 0x09..=0x0d run (`\t \n \v \f \r`).
    /// Note this deliberately does *not* use Rust's `is_ascii_whitespace`,
    /// which omits `\v` (0x0b) — the Python `isspace` SimProcedure matches on
    /// `c == 32 || (9 <= c <= 13)`, and so does real libc, so excluding `\v`
    /// would make a guest tokenizer take a different branch under the native
    /// fast path than under Python.
    name = "isspace",
    struct = NativeIsSpace,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(0x09, 0x0d), (b' ', b' ')], is_c_space)
    }
}

crate::declare_proc! {
    /// `int isalnum(int c)`.
    name = "isalnum",
    struct = NativeIsAlnum,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(b'0', b'9'), (b'A', b'Z'), (b'a', b'z')], |c| c.is_ascii_alphanumeric())
    }
}

crate::declare_proc! {
    /// `int isupper(int c)`.
    name = "isupper",
    struct = NativeIsUpper,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(b'A', b'Z')], |c| c.is_ascii_uppercase())
    }
}

crate::declare_proc! {
    /// `int islower(int c)`.
    name = "islower",
    struct = NativeIsLower,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(b'a', b'z')], |c| c.is_ascii_lowercase())
    }
}

crate::declare_proc! {
    /// `int isxdigit(int c)`.
    name = "isxdigit",
    struct = NativeIsXdigit,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(b'0', b'9'), (b'A', b'F'), (b'a', b'f')], |c| c.is_ascii_hexdigit())
    }
}

crate::declare_proc! {
    /// `int isprint(int c)`.
    name = "isprint",
    struct = NativeIsPrint,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(0x20, 0x7e)], |c| (0x20..=0x7e).contains(&c))
    }
}

crate::declare_proc! {
    /// `int isascii(int c)` — nonzero if `c` is a 7-bit ASCII value [0, 127].
    name = "isascii",
    struct = NativeIsAscii,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(0x00, 0x7f)], |c| c <= 0x7f)
    }
}

crate::declare_proc! {
    /// `int isblank(int c)` — nonzero for space (0x20) or tab (0x09).
    name = "isblank",
    struct = NativeIsBlank,
    args = [c: bv],
    call |state| {
        set_predicate(state, &c, b" \t", |c| c == b' ' || c == b'\t')
    }
}

crate::declare_proc! {
    /// `int iscntrl(int c)` — nonzero for control chars [0, 31] or DEL (127).
    name = "iscntrl",
    struct = NativeIsCntrl,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(0x00, 0x1f), (0x7f, 0x7f)], |c| c <= 0x1f || c == 0x7f)
    }
}

crate::declare_proc! {
    /// `int isgraph(int c)` — nonzero for printable non-space chars [33, 126].
    name = "isgraph",
    struct = NativeIsGraph,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(0x21, 0x7e)], |c| (0x21..=0x7e).contains(&c))
    }
}

crate::declare_proc! {
    /// `int ispunct(int c)` — nonzero for punctuation: `[33,47]`, `[58,64]`,
    /// `[91,96]`, `[123,126]` (matches Python's ispunct ranges).
    name = "ispunct",
    struct = NativeIsPunct,
    args = [c: bv],
    call |state| {
        ranges_predicate(
            state,
            &c,
            &[(0x21, 0x2f), (0x3a, 0x40), (0x5b, 0x60), (0x7b, 0x7e)],
            |c| (0x21..=0x2f).contains(&c)
                || (0x3a..=0x40).contains(&c)
                || (0x5b..=0x60).contains(&c)
                || (0x7b..=0x7e).contains(&c),
        )
    }
}

crate::declare_proc! {
    /// `int tolower(int c)` — convert uppercase to lowercase.
    name = "tolower",
    struct = NativeToLower,
    args = [c: bv],
    call |state| {
        case_shift(state, &c, b'A', b'Z', 32)
    }
}

crate::declare_proc! {
    /// `int toupper(int c)` — convert lowercase to uppercase.
    name = "toupper",
    struct = NativeToUpper,
    args = [c: bv],
    call |state| {
        case_shift(state, &c, b'a', b'z', -32)
    }
}

#[cfg(test)]
#[path = "ctype_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod tests;
