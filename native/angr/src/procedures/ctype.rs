//! Native character classification functions (ctype.h).
//!
//! isdigit, isalpha, isspace, isalnum, isupper, islower, isxdigit, isprint,
//! tolower, toupper.
//!
//! Each takes a single int argument and returns 0 or non-zero.
//! Symbolic arguments are handled by emitting a constraint-shaped result that
//! mirrors the concrete predicate on bits\[7:0\] of the argument; the operand
//! pattern matches the underlying `as u8` truncation in the concrete path.

use super::ProcedureError;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

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
    state: &mut RustSimState,
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
fn ranges_predicate(
    state: &mut RustSimState,
    arg: &RustBV,
    ranges: &[(u8, u8)],
    concrete_check: fn(u8) -> bool,
) -> Result<Option<RustBV>, ProcedureError> {
    let bits = state.arch().bits();
    if let Some(c) = arg.as_u64() {
        return Ok(Some(RustBV::concrete(
            if concrete_check(c as u8) { 1 } else { 0 },
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
fn set_predicate(
    state: &mut RustSimState,
    arg: &RustBV,
    members: &[u8],
    concrete_check: fn(u8) -> bool,
) -> Result<Option<RustBV>, ProcedureError> {
    let bits = state.arch().bits();
    if let Some(c) = arg.as_u64() {
        return Ok(Some(RustBV::concrete(
            if concrete_check(c as u8) { 1 } else { 0 },
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
    /// ASCII whitespace per is_ascii_whitespace: ' ', '\t', '\n', '\x0c', '\r'.
    /// (No '\x0b' — matches Rust's definition; kept consistent with the
    /// pre-existing concrete path so symbolic and concrete agree.)
    name = "isspace",
    struct = NativeIsSpace,
    args = [c: bv],
    call |state| {
        set_predicate(state, &c, &[b' ', b'\t', b'\n', 0x0c, b'\r'], |c| c.is_ascii_whitespace())
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
mod tests;
