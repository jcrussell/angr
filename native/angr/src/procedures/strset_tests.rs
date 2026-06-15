//! Tests for the strset SimProcedure (extracted from strset.rs).
use super::*;
use crate::memory::Permission;

fn setup_state(s: &[u8], set: &[u8]) -> RustSimState {
    let mut state = crate::procedures::test_util::amd64_state();
    state.map_memory_data(0x1000, s, Permission::RWX);
    state.map_memory_data(0x2000, set, Permission::RWX);
    state
}

// --- strpbrk ---

#[test]
fn test_strpbrk_first_match() {
    let mut state = setup_state(b"hello world\x00", b"aeiou\x00");
    let r = NativeStrpbrk
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    // 'e' at 0x1001 is the first vowel.
    assert_eq!(r.as_u64(), Some(0x1001));
}

#[test]
fn test_strpbrk_no_match() {
    let mut state = setup_state(b"xyz\x00", b"abc\x00");
    let r = NativeStrpbrk
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_strpbrk_empty_set() {
    // Empty accept set: nothing can match, return NULL.
    let mut state = setup_state(b"abc\x00", b"\x00");
    let r = NativeStrpbrk
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_strpbrk_empty_haystack() {
    let mut state = setup_state(b"\x00", b"abc\x00");
    let r = NativeStrpbrk
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_strpbrk_symbolic_addr_falls_back() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "s", 64);
    drop(ctx);
    let r = NativeStrpbrk.call(&mut state, &[sym, RustBV::concrete(0x2000, 64)]);
    assert!(matches!(r, Err(ProcedureError::SymbolicArgument(_))));
}

// --- strspn ---

#[test]
fn test_strspn_full_prefix() {
    let mut state = setup_state(b"aaabbb\x00", b"ab\x00");
    let r = NativeStrspn
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    // entire string consists of ab.
    assert_eq!(r.as_u64(), Some(6));
}

#[test]
fn test_strspn_partial() {
    let mut state = setup_state(b"abc123\x00", b"abc\x00");
    let r = NativeStrspn
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(r.as_u64(), Some(3));
}

#[test]
fn test_strspn_zero() {
    let mut state = setup_state(b"xyz\x00", b"abc\x00");
    let r = NativeStrspn
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(r.as_u64(), Some(0));
}

// --- strcspn ---

#[test]
fn test_strcspn_partial() {
    let mut state = setup_state(b"abc,def\x00", b",;\x00");
    let r = NativeStrcspn
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(r.as_u64(), Some(3));
}

#[test]
fn test_strcspn_no_reject_byte() {
    // No byte from reject in s → return strlen(s).
    let mut state = setup_state(b"hello\x00", b",;\x00");
    let r = NativeStrcspn
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(r.as_u64(), Some(5));
}

#[test]
fn test_strcspn_immediate_reject() {
    let mut state = setup_state(b",hello\x00", b",;\x00");
    let r = NativeStrcspn
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(r.as_u64(), Some(0));
}

#[test]
fn test_strspn_symbolic_byte_falls_back() {
    // Symbolic byte in s → SymbolicArgument.
    let mut state = setup_state(b"ab\x00", b"ab\x00");
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "b1", 8);
    drop(ctx);
    state.memory_store(0x1001, sym).unwrap();
    let r = NativeStrspn.call(
        &mut state,
        &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
    );
    assert!(matches!(r, Err(ProcedureError::SymbolicArgument(_))));
}
