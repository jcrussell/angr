use super::*;
use crate::memory::Permission;
use crate::procedures::{NativeSimProcedure, ProcedureError};
use crate::state::RustSimState;

#[test]
fn test_printf_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);

    let result = NativePrintf
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(11));
    assert_eq!(state.stdout_buffer(), b"hello world");
}

#[test]
fn test_printf_empty_format() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"\x00", Permission::RWX);

    let result = NativePrintf
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    // Returns 1 for empty format
    assert_eq!(result.unwrap().as_u64(), Some(1));
    assert_eq!(state.stdout_buffer(), b"");
}

#[test]
fn test_printf_symbolic_addr() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "fmt", 64);
    drop(ctx);
    let result = NativePrintf.call(&mut state, &[sym]);
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}
