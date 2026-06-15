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
mod tests {
    use super::*;
    use crate::procedures::NativeSimProcedure;

    fn make_state() -> RustSimState {
        RustSimState::new("amd64").unwrap()
    }

    fn call_with(proc: &dyn NativeSimProcedure, state: &mut RustSimState, c: u8) -> u64 {
        let args = [RustBV::concrete(c as u128, 64)];
        proc.call(state, &args).unwrap().unwrap().as_u64().unwrap()
    }

    #[test]
    fn test_isdigit() {
        let mut s = make_state();
        let p = NativeIsDigit;
        assert_eq!(call_with(&p, &mut s, b'0'), 1);
        assert_eq!(call_with(&p, &mut s, b'9'), 1);
        assert_eq!(call_with(&p, &mut s, b'a'), 0);
    }

    #[test]
    fn test_isalpha() {
        let mut s = make_state();
        let p = NativeIsAlpha;
        assert_eq!(call_with(&p, &mut s, b'A'), 1);
        assert_eq!(call_with(&p, &mut s, b'z'), 1);
        assert_eq!(call_with(&p, &mut s, b'5'), 0);
    }

    #[test]
    fn test_isspace() {
        let mut s = make_state();
        let p = NativeIsSpace;
        assert_eq!(call_with(&p, &mut s, b' '), 1);
        assert_eq!(call_with(&p, &mut s, b'\t'), 1);
        assert_eq!(call_with(&p, &mut s, b'\n'), 1);
        assert_eq!(call_with(&p, &mut s, b'a'), 0);
    }

    #[test]
    fn test_tolower() {
        let mut s = make_state();
        let p = NativeToLower;
        assert_eq!(call_with(&p, &mut s, b'A'), b'a' as u64);
        assert_eq!(call_with(&p, &mut s, b'z'), b'z' as u64);
        assert_eq!(call_with(&p, &mut s, b'5'), b'5' as u64);
    }

    #[test]
    fn test_toupper() {
        let mut s = make_state();
        let p = NativeToUpper;
        assert_eq!(call_with(&p, &mut s, b'a'), b'A' as u64);
        assert_eq!(call_with(&p, &mut s, b'Z'), b'Z' as u64);
        assert_eq!(call_with(&p, &mut s, b'5'), b'5' as u64);
    }

    #[test]
    fn test_symbolic_isdigit_returns_constrained_bv() {
        // Symbolic input should now produce a symbolic result, not a fallback error.
        let mut s = make_state();
        let ctx = s.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "c", 64);
        drop(ctx);
        let p = NativeIsDigit;
        let result = p.call(&mut s, std::slice::from_ref(&sym)).unwrap().unwrap();
        // Result should be a 64-bit BV (arch.bits()) and *not* concrete.
        assert_eq!(result.width(), 64);
        assert!(result.as_u64().is_none(), "expected symbolic, got concrete");
    }

    #[test]
    fn test_symbolic_isdigit_solver_evaluation() {
        // Constrain the symbolic byte to '5' and check the result solves to 1.
        let mut s = make_state();
        let ctx = s.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "c", 64);
        let target = RustBV::concrete(b'5' as u128, 64);
        let eq = sym.eq(&target, &ctx);
        drop(ctx);
        let p = NativeIsDigit;
        let result = p.call(&mut s, std::slice::from_ref(&sym)).unwrap().unwrap();
        s.add_constraint(eq);
        let ctx = s.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(1));
        assert_eq!(ctx.max(&result, false), Some(1));
    }

    #[test]
    fn test_symbolic_tolower_returns_symbolic() {
        let mut s = make_state();
        let ctx = s.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "c", 64);
        let target = RustBV::concrete(b'A' as u128, 64);
        let eq = sym.eq(&target, &ctx);
        drop(ctx);
        let p = NativeToLower;
        let result = p.call(&mut s, std::slice::from_ref(&sym)).unwrap().unwrap();
        assert_eq!(result.width(), 64);
        assert!(result.as_u64().is_none());

        s.add_constraint(eq);
        let ctx = s.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(b'a' as u128));
        assert_eq!(ctx.max(&result, false), Some(b'a' as u128));
    }
}
