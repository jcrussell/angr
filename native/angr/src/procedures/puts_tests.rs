// Tests for puts.rs (NativePuts / NativePutchar / NativeFputc).
// Extracted from the parent module; see the `#[path]` attr in puts.rs.
use super::*;
use crate::memory::Permission;

#[test]
fn test_puts_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

    let result = NativePuts
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(6)); // 5 + newline
    assert_eq!(state.stdout_buffer(), b"hello\n");
}

#[test]
fn test_puts_empty_string() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"\x00", Permission::RWX);

    let result = NativePuts
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(1)); // just newline
    assert_eq!(state.stdout_buffer(), b"\n");
}

#[test]
fn test_puts_multiple_calls() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"def\x00", Permission::RWX);

    NativePuts
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    NativePuts
        .call(&mut state, &[RustBV::concrete(0x2000, 64)])
        .unwrap();
    assert_eq!(state.stdout_buffer(), b"abc\ndef\n");
}

#[test]
fn test_puts_symbolic_addr() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "addr", 64);
    drop(ctx);
    let result = NativePuts.call(&mut state, &[sym]);
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_putchar_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativePutchar
        .call(&mut state, &[RustBV::concrete(b'A' as u128, 32)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(b'A' as u64));
    assert_eq!(state.stdout_buffer(), b"A");
}

#[test]
fn test_putchar_multiple() {
    let mut state = RustSimState::new("amd64").unwrap();
    NativePutchar
        .call(&mut state, &[RustBV::concrete(b'H' as u128, 32)])
        .unwrap();
    NativePutchar
        .call(&mut state, &[RustBV::concrete(b'i' as u128, 32)])
        .unwrap();
    assert_eq!(state.stdout_buffer(), b"Hi");
}

#[test]
fn test_putchar_symbolic() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "c", 32);
    drop(ctx);
    let result = NativePutchar.call(&mut state, &[sym]);
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_fputc_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeFputc
        .call(
            &mut state,
            &[RustBV::concrete(b'X' as u128, 32), RustBV::concrete(0, 64)],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(b'X' as u64));
    assert_eq!(state.stdout_buffer(), b"X");
}
