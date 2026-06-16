//! Tests for ctype.rs character-classification SimProcedures.
//! Extracted from the `mod tests` block (see angr-yg2m sibling-extraction campaign).

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
