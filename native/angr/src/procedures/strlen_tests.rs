//! Tests for the strlen SimProcedure (extracted from strlen.rs).
use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;

#[test]
fn test_strlen_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"test\x00", Permission::RWX);
    let proc = NativeStrlen;
    let result = proc
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(4));
}

#[test]
fn test_strlen_empty() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"\x00", Permission::RWX);
    let proc = NativeStrlen;
    let result = proc
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_strlen_longer_string() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);
    let proc = NativeStrlen;
    let result = proc
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(11));
}

#[test]
fn test_strnlen_within_limit() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
    let proc = NativeStrnlen;
    let result = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(10, 64)],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(5));
}

#[test]
fn test_strnlen_at_limit() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);
    let proc = NativeStrnlen;
    let result = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(5, 64)],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(5));
}

#[test]
fn test_strnlen_zero_maxlen() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
    let proc = NativeStrnlen;
    let result = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_strnlen_empty_string() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"\x00rest", Permission::RWX);
    let proc = NativeStrnlen;
    let result = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(10, 64)],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_strlen_symbolic_addr() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym_addr = RustBV::symbolic(&ctx, "addr", 64);
    drop(ctx);
    let proc = NativeStrlen;
    let result = proc.call(&mut state, &[sym_addr]);
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

/// Insert a fully-symbolic byte at `addr` (page must be pre-mapped).
fn place_symbolic_byte(state: &mut RustSimState, addr: u64, name: &str) -> RustBV {
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, name, 8);
    drop(ctx);
    state.memory_store(addr, sym.clone()).unwrap();
    sym
}

#[test]
fn test_strlen_symbolic_byte_returns_symbolic() {
    let mut state = RustSimState::new("amd64").unwrap();
    // Buffer: [a, ?, c, \0, ...]
    state.map_memory_data(0x1000, b"a_c\x00", Permission::RWX);
    let _sym = place_symbolic_byte(&mut state, 0x1001, "b1");

    let proc = NativeStrlen;
    let result = proc
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap()
        .unwrap();
    assert_eq!(result.width(), 64);
    assert!(result.as_u64().is_none(), "expected symbolic length");
}

#[test]
fn test_strlen_symbolic_byte_zero_solution() {
    // Constraining the symbolic byte to 0 should yield length 1.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"a_c\x00", Permission::RWX);
    let sym = place_symbolic_byte(&mut state, 0x1001, "b1");

    let ctx = state.solver().borrow();
    let zero = RustBV::concrete(0u128, 8);
    let eq = sym.eq(&zero, &ctx);
    drop(ctx);

    let proc = NativeStrlen;
    let result = proc
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap()
        .unwrap();
    state.add_constraint(eq);
    let ctx = state.solver().borrow();
    assert_eq!(ctx.min(&result, false), Some(1));
    assert_eq!(ctx.max(&result, false), Some(1));
}

#[test]
fn test_strlen_symbolic_byte_nonzero_solution() {
    // Constraining the symbolic byte to 'b' should yield length 3.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"a_c\x00", Permission::RWX);
    let sym = place_symbolic_byte(&mut state, 0x1001, "b1");

    let ctx = state.solver().borrow();
    let b = RustBV::concrete(b'b' as u128, 8);
    let eq = sym.eq(&b, &ctx);
    drop(ctx);

    let proc = NativeStrlen;
    let result = proc
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap()
        .unwrap();
    state.add_constraint(eq);
    let ctx = state.solver().borrow();
    assert_eq!(ctx.min(&result, false), Some(3));
    assert_eq!(ctx.max(&result, false), Some(3));
}

#[test]
fn test_strnlen_symbolic_byte_capped_at_maxlen() {
    // Buffer of all symbolic bytes; with maxlen=2 result is bounded by 2.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, &[1u8; 16], Permission::RWX);
    let _s0 = place_symbolic_byte(&mut state, 0x1000, "b0");
    let _s1 = place_symbolic_byte(&mut state, 0x1001, "b1");

    let proc = NativeStrnlen;
    let result = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(2, 64)],
        )
        .unwrap()
        .unwrap();
    let ctx = state.solver().borrow();
    // result must be in [0, 2].
    let min = ctx.min(&result, false).unwrap();
    let max = ctx.max(&result, false).unwrap();
    assert!(min <= 2);
    assert!(max <= 2);
}
