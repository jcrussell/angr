//! Native character classification functions (ctype.h).
//!
//! isdigit, isalpha, isspace, isalnum, isupper, islower, isxdigit, isprint,
//! tolower, toupper.
//!
//! Each takes a single int argument and returns 0 or non-zero.
//! Symbolic arguments are handled by emitting a constraint-shaped result that
//! mirrors the concrete predicate on bits[7:0] of the argument; the operand
//! pattern matches the underlying `as u8` truncation in the concrete path.

use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};
use super::{NativeSimProcedure, ProcedureError};

/// Truncate an argument to its low 8 bits, matching the concrete `as u8` path.
fn arg_byte(args: &[RustBV], ctx: &SymContext) -> RustBV {
    args[0].extract(7, 0, ctx)
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
    args: &[RustBV],
    lo: u8,
    hi: u8,
    delta: i8,
) -> Result<Option<RustBV>, ProcedureError> {
    let bits = state.arch().bits();
    if let Some(c) = args[0].as_u64() {
        let b = c as u8;
        let result = if b >= lo && b <= hi {
            ((b as i16) + delta as i16) as u8
        } else {
            b
        };
        return Ok(Some(RustBV::concrete(result as u128, bits)));
    }
    let ctx = state.solver().borrow();
    let byte = arg_byte(args, &ctx);
    let in_range = byte_in_range(&byte, lo, hi, &ctx);
    let delta_bv = RustBV::concrete(delta as u8 as u128, 8);
    let shifted = byte.add(&delta_bv, &ctx);
    let new_byte = in_range.ite(&shifted, &byte, &ctx);
    Ok(Some(new_byte.zero_extend(bits, &ctx)))
}

/// Build a symbolic ctype predicate from a list of inclusive ranges.
fn ranges_predicate(
    state: &mut RustSimState,
    args: &[RustBV],
    ranges: &[(u8, u8)],
    concrete_check: fn(u8) -> bool,
) -> Result<Option<RustBV>, ProcedureError> {
    let bits = state.arch().bits();
    if let Some(c) = args[0].as_u64() {
        return Ok(Some(RustBV::concrete(
            if concrete_check(c as u8) { 1 } else { 0 },
            bits,
        )));
    }
    let ctx = state.solver().borrow();
    let byte = arg_byte(args, &ctx);
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
    args: &[RustBV],
    members: &[u8],
    concrete_check: fn(u8) -> bool,
) -> Result<Option<RustBV>, ProcedureError> {
    let bits = state.arch().bits();
    if let Some(c) = args[0].as_u64() {
        return Ok(Some(RustBV::concrete(
            if concrete_check(c as u8) { 1 } else { 0 },
            bits,
        )));
    }
    let ctx = state.solver().borrow();
    let byte = arg_byte(args, &ctx);
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

pub struct NativeIsDigit;
impl NativeSimProcedure for NativeIsDigit {
    fn name(&self) -> &'static str { "isdigit" }
    fn num_args(&self) -> usize { 1 }
    fn call(&self, state: &mut RustSimState, args: &[RustBV]) -> Result<Option<RustBV>, ProcedureError> {
        ranges_predicate(state, args, &[(b'0', b'9')], |c| c.is_ascii_digit())
    }
}

pub struct NativeIsAlpha;
impl NativeSimProcedure for NativeIsAlpha {
    fn name(&self) -> &'static str { "isalpha" }
    fn num_args(&self) -> usize { 1 }
    fn call(&self, state: &mut RustSimState, args: &[RustBV]) -> Result<Option<RustBV>, ProcedureError> {
        ranges_predicate(state, args, &[(b'A', b'Z'), (b'a', b'z')], |c| c.is_ascii_alphabetic())
    }
}

pub struct NativeIsSpace;
impl NativeSimProcedure for NativeIsSpace {
    fn name(&self) -> &'static str { "isspace" }
    fn num_args(&self) -> usize { 1 }
    fn call(&self, state: &mut RustSimState, args: &[RustBV]) -> Result<Option<RustBV>, ProcedureError> {
        // ASCII whitespace per is_ascii_whitespace: ' ', '\t', '\n', '\x0c', '\r'.
        // Note: matches Rust's definition (no '\x0b'); kept consistent with the
        // pre-existing concrete path so symbolic and concrete agree.
        set_predicate(state, args, &[b' ', b'\t', b'\n', 0x0c, b'\r'], |c| c.is_ascii_whitespace())
    }
}

pub struct NativeIsAlnum;
impl NativeSimProcedure for NativeIsAlnum {
    fn name(&self) -> &'static str { "isalnum" }
    fn num_args(&self) -> usize { 1 }
    fn call(&self, state: &mut RustSimState, args: &[RustBV]) -> Result<Option<RustBV>, ProcedureError> {
        ranges_predicate(state, args, &[(b'0', b'9'), (b'A', b'Z'), (b'a', b'z')], |c| c.is_ascii_alphanumeric())
    }
}

pub struct NativeIsUpper;
impl NativeSimProcedure for NativeIsUpper {
    fn name(&self) -> &'static str { "isupper" }
    fn num_args(&self) -> usize { 1 }
    fn call(&self, state: &mut RustSimState, args: &[RustBV]) -> Result<Option<RustBV>, ProcedureError> {
        ranges_predicate(state, args, &[(b'A', b'Z')], |c| c.is_ascii_uppercase())
    }
}

pub struct NativeIsLower;
impl NativeSimProcedure for NativeIsLower {
    fn name(&self) -> &'static str { "islower" }
    fn num_args(&self) -> usize { 1 }
    fn call(&self, state: &mut RustSimState, args: &[RustBV]) -> Result<Option<RustBV>, ProcedureError> {
        ranges_predicate(state, args, &[(b'a', b'z')], |c| c.is_ascii_lowercase())
    }
}

pub struct NativeIsXdigit;
impl NativeSimProcedure for NativeIsXdigit {
    fn name(&self) -> &'static str { "isxdigit" }
    fn num_args(&self) -> usize { 1 }
    fn call(&self, state: &mut RustSimState, args: &[RustBV]) -> Result<Option<RustBV>, ProcedureError> {
        ranges_predicate(state, args, &[(b'0', b'9'), (b'A', b'F'), (b'a', b'f')], |c| c.is_ascii_hexdigit())
    }
}

pub struct NativeIsPrint;
impl NativeSimProcedure for NativeIsPrint {
    fn name(&self) -> &'static str { "isprint" }
    fn num_args(&self) -> usize { 1 }
    fn call(&self, state: &mut RustSimState, args: &[RustBV]) -> Result<Option<RustBV>, ProcedureError> {
        ranges_predicate(state, args, &[(0x20, 0x7e)], |c| (0x20..=0x7e).contains(&c))
    }
}

/// tolower: convert uppercase to lowercase.
pub struct NativeToLower;
impl NativeSimProcedure for NativeToLower {
    fn name(&self) -> &'static str { "tolower" }
    fn num_args(&self) -> usize { 1 }
    fn call(&self, state: &mut RustSimState, args: &[RustBV]) -> Result<Option<RustBV>, ProcedureError> {
        case_shift(state, args, b'A', b'Z', 32)
    }
}

/// toupper: convert lowercase to uppercase.
pub struct NativeToUpper;
impl NativeSimProcedure for NativeToUpper {
    fn name(&self) -> &'static str { "toupper" }
    fn num_args(&self) -> usize { 1 }
    fn call(&self, state: &mut RustSimState, args: &[RustBV]) -> Result<Option<RustBV>, ProcedureError> {
        case_shift(state, args, b'a', b'z', -32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let result = p.call(&mut s, &[sym.clone()]).unwrap().unwrap();
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
        let result = p.call(&mut s, &[sym.clone()]).unwrap().unwrap();
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
        let result = p.call(&mut s, &[sym.clone()]).unwrap().unwrap();
        assert_eq!(result.width(), 64);
        assert!(result.as_u64().is_none());

        s.add_constraint(eq);
        let ctx = s.solver().borrow();
        assert_eq!(ctx.min(&result, false), Some(b'a' as u128));
        assert_eq!(ctx.max(&result, false), Some(b'a' as u128));
    }
}
