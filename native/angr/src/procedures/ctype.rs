//! Native character classification functions (ctype.h).
//!
//! isdigit, isalpha, isspace, isalnum, isupper, islower, isxdigit, isprint,
//! isascii, isblank, iscntrl, isgraph, ispunct, tolower, toupper.
//!
//! Each takes a single int argument and returns 0 or non-zero.
//!
//! **Full-width comparison (angr-6cp06.1).** The argument is compared at its
//! own width (the arch word — native procs receive pointer-sized register
//! values), *not* truncated to bits\[7:0\]. The Python siblings this module
//! mirrors (`angr/procedures/libc/isdigit.py`, `tolower.py`, ...) all compare
//! the full `c` unsigned, so truncating diverged: `isdigit(304)` was true
//! natively (`304 & 0xff == b'0'`) and false in Python, and `tolower(EOF)`
//! returned `255` instead of preserving `-1` — breaking the ubiquitous
//! `while ((c = getchar()) != EOF) putchar(tolower(c));` idiom. The symbolic
//! path shares the concrete path's range tables, so the two cannot drift.
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

use super::{ProcedureError, arch_word};
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

/// C-locale whitespace: `' '` plus the `0x09..=0x0d` run (`\t \n \v \f \r`).
///
/// Deliberately *not* Rust's `is_ascii_whitespace`, which omits `\v` (0x0b);
/// see `invariant-ctype-mirror-python-not-rust-std`.
pub(crate) const C_SPACE_RANGES: &[(u8, u8)] = &[(0x09, 0x0d), (b' ', b' ')];

/// C-locale `isspace()` over a single byte.
///
/// The single source of truth for "is this byte whitespace" across the native
/// procedures — `NativeIsSpace` below (via [`C_SPACE_RANGES`], which this
/// delegates to so the two cannot drift) plus the leading-whitespace skips in
/// `strtod::floating_prefix_len` / `strtod`'s endptr rescan and
/// `strtol::parse_concrete_prefix`.
pub(crate) fn is_c_space(c: u8) -> bool {
    in_ranges(u64::from(c), C_SPACE_RANGES)
}

/// Concrete twin of [`ranges_predicate`]'s symbolic disjunction.
fn in_ranges(v: u64, ranges: &[(u8, u8)]) -> bool {
    ranges
        .iter()
        .any(|&(lo, hi)| v >= u64::from(lo) && v <= u64::from(hi))
}

/// Concrete twin of [`set_predicate`]'s symbolic disjunction.
fn in_set(v: u64, members: &[u8]) -> bool {
    members.iter().any(|&m| v == u64::from(m))
}

/// Build `arg >= lo && arg <= hi` (unsigned, at `arg`'s own width) as a 1-bit BV.
fn arg_in_range(arg: &RustBV, lo: u8, hi: u8, ctx: &SymContext) -> RustBV {
    let width = arg.width();
    let lo_bv = RustBV::concrete(lo as u128, width);
    let hi_bv = RustBV::concrete(hi as u128, width);
    arg.uge(&lo_bv, ctx).and(&arg.ule(&hi_bv, ctx), ctx)
}

/// Zero-extend a 1-bit predicate to arch.bits() and return it.
fn lift_predicate(state: &RustSimState, pred: RustBV, ctx: &SymContext) -> RustBV {
    let bits = state.arch().bits();
    pred.zero_extend(bits, ctx)
}

/// tolower/toupper share the same pattern: if `arg` is in [lo, hi], shift by
/// `delta`; otherwise return `arg` **unchanged and untruncated**, exactly as
/// `claripy.If(And(c >= lo, c <= hi), c + delta, c)` does in `tolower.py` /
/// `toupper.py`. Returning the low byte instead is what corrupted `EOF` into
/// `255` (angr-6cp06.1).
fn case_shift(
    state: &RustSimState,
    arg: &RustBV,
    lo: u8,
    hi: u8,
    delta: i8,
) -> Result<Option<RustBV>, ProcedureError> {
    let width = arg.width();
    if let Some(c) = arg.as_u64() {
        let result = if in_ranges(c, &[(lo, hi)]) {
            // `c` is inside [lo, hi] ⊆ [0, 255] on this arm and `delta` is
            // ±32, so the sum lands back inside [0, 255]; `wrapping_add` is
            // just how a negative `delta` is spelled against a `u64`.
            c.wrapping_add(delta as i64 as u64)
        } else {
            c
        };
        return Ok(Some(RustBV::concrete(u128::from(result), width)));
    }
    let ctx = state.solver().borrow();
    let in_range = arg_in_range(arg, lo, hi, &ctx);
    let delta_bv = RustBV::concrete(delta as u8 as u128, 8).sign_extend(width, &ctx);
    let shifted = arg.add(&delta_bv, &ctx);
    Ok(Some(in_range.ite(&shifted, arg, &ctx)))
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
) -> Result<Option<RustBV>, ProcedureError> {
    let bits = state.arch().bits();
    if let Some(c) = arg.as_u64() {
        return Ok(Some(RustBV::concrete(
            u128::from(in_ranges(c, ranges)),
            bits,
        )));
    }
    let ctx = state.solver().borrow();
    let mut pred: Option<RustBV> = None;
    for &(lo, hi) in ranges {
        let r = arg_in_range(arg, lo, hi, &ctx);
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
) -> Result<Option<RustBV>, ProcedureError> {
    let bits = state.arch().bits();
    if let Some(c) = arg.as_u64() {
        return Ok(Some(RustBV::concrete(u128::from(in_set(c, members)), bits)));
    }
    let ctx = state.solver().borrow();
    let mut pred: Option<RustBV> = None;
    for &m in members {
        let m_bv = RustBV::concrete(m as u128, arg.width());
        let eq = arg.eq(&m_bv, &ctx);
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
    Ok(Some(arch_word(state, addr)))
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
        ranges_predicate(state, &c, &[(b'0', b'9')])
    }
}

crate::declare_proc! {
    /// `int isalpha(int c)`.
    name = "isalpha",
    struct = NativeIsAlpha,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(b'A', b'Z'), (b'a', b'z')])
    }
}

crate::declare_proc! {
    /// `int isspace(int c)`.
    /// C locale whitespace: the [`C_SPACE_RANGES`] table, shared with
    /// [`is_c_space`]. Note this deliberately does *not* use Rust's
    /// `is_ascii_whitespace`, which omits `\v` (0x0b) — the Python `isspace`
    /// SimProcedure matches on `c == 32 || (9 <= c <= 13)`, and so does real
    /// libc, so excluding `\v` would make a guest tokenizer take a different
    /// branch under the native fast path than under Python.
    name = "isspace",
    struct = NativeIsSpace,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, C_SPACE_RANGES)
    }
}

crate::declare_proc! {
    /// `int isalnum(int c)`.
    name = "isalnum",
    struct = NativeIsAlnum,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(b'0', b'9'), (b'A', b'Z'), (b'a', b'z')])
    }
}

crate::declare_proc! {
    /// `int isupper(int c)`.
    name = "isupper",
    struct = NativeIsUpper,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(b'A', b'Z')])
    }
}

crate::declare_proc! {
    /// `int islower(int c)`.
    name = "islower",
    struct = NativeIsLower,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(b'a', b'z')])
    }
}

crate::declare_proc! {
    /// `int isxdigit(int c)`.
    name = "isxdigit",
    struct = NativeIsXdigit,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(b'0', b'9'), (b'A', b'F'), (b'a', b'f')])
    }
}

crate::declare_proc! {
    /// `int isprint(int c)`.
    name = "isprint",
    struct = NativeIsPrint,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(0x20, 0x7e)])
    }
}

crate::declare_proc! {
    /// `int isascii(int c)` — nonzero if `c` is a 7-bit ASCII value [0, 127].
    name = "isascii",
    struct = NativeIsAscii,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(0x00, 0x7f)])
    }
}

crate::declare_proc! {
    /// `int isblank(int c)` — nonzero for space (0x20) or tab (0x09).
    name = "isblank",
    struct = NativeIsBlank,
    args = [c: bv],
    call |state| {
        set_predicate(state, &c, b" \t")
    }
}

crate::declare_proc! {
    /// `int iscntrl(int c)` — nonzero for control chars [0, 31] or DEL (127).
    name = "iscntrl",
    struct = NativeIsCntrl,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(0x00, 0x1f), (0x7f, 0x7f)])
    }
}

crate::declare_proc! {
    /// `int isgraph(int c)` — nonzero for printable non-space chars [33, 126].
    name = "isgraph",
    struct = NativeIsGraph,
    args = [c: bv],
    call |state| {
        ranges_predicate(state, &c, &[(0x21, 0x7e)])
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

test_submod!("ctype_tests.rs" => tests);
