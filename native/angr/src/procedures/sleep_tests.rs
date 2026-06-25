// Tests for sleep.rs (NativeSleep / NativeUsleep).
// Extracted from the parent module; see the `#[path]` attr in sleep.rs.
use super::*;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;

#[test]
fn test_sleep_returns_zero_word_sized() {
    let mut state = RustSimState::new("amd64").unwrap();
    let seconds = RustBV::concrete(5, 64);
    let result = NativeSleep.call(&mut state, &[seconds]).unwrap().unwrap();
    assert_eq!(result.as_u64(), Some(0));
    assert_eq!(result.width(), 64);
}

#[test]
fn test_usleep_returns_zero_word_sized() {
    let mut state = RustSimState::new("amd64").unwrap();
    let usec = RustBV::concrete(100, 64);
    let result = NativeUsleep.call(&mut state, &[usec]).unwrap().unwrap();
    assert_eq!(result.as_u64(), Some(0));
    assert_eq!(result.width(), 64);
}

#[test]
fn test_return_width_tracks_arch_32bit() {
    let mut state = RustSimState::new("x86").unwrap();
    let seconds = RustBV::concrete(1, 32);
    let result = NativeSleep.call(&mut state, &[seconds]).unwrap().unwrap();
    assert_eq!(result.as_u64(), Some(0));
    assert_eq!(result.width(), 32);
}

#[test]
fn test_symbolic_arg_still_returns_zero_no_fallback() {
    // Python ignores the duration; a symbolic count must NOT fall back to
    // Python (the `bv` arg kind keeps the raw BV without concretizing).
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "dur", 64);
    drop(ctx);
    let result = NativeSleep
        .call(&mut state, std::slice::from_ref(&sym))
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0));
}

#[test]
fn test_names() {
    assert_eq!(NativeSleep.name(), "sleep");
    assert_eq!(NativeUsleep.name(), "usleep");
}
