// Tests for fgets.rs (NativeFgets/Fgetc/Getchar/Getc procedures).
// Split from the inline `#[cfg(test)] mod tests` block; see rust-mod-tests-sibling-extraction.

use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;

#[test]
fn test_fgets_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let result = NativeFgets
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // buf
                RustBV::concrete(10, 64),     // size
                RustBV::concrete(0, 64),      // stream (stdin)
            ],
        )
        .unwrap();

    // Should return buf address
    assert_eq!(result.unwrap().as_u64(), Some(0x2000));

    // First 9 bytes should be symbolic
    for i in 0..9u64 {
        let byte = state.memory_load(0x2000 + i, 1).unwrap();
        assert!(byte.as_u64().is_none(), "byte {} should be symbolic", i);
    }

    // Byte 9 should be NUL terminator
    let nul = state.memory_load(0x2009, 1).unwrap();
    assert_eq!(nul.as_u64(), Some(0));
}

#[test]
fn test_fgets_size_1() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let result = NativeFgets
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(1, 64), // size=1 means only NUL
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0x2000));

    // Only NUL should be written
    let nul = state.memory_load(0x2000, 1).unwrap();
    assert_eq!(nul.as_u64(), Some(0));
}

#[test]
fn test_fgets_size_0() {
    let mut state = RustSimState::new("amd64").unwrap();

    let result = NativeFgets
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64), // size=0 returns NULL
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0)); // NULL
}

#[test]
fn test_fgets_symbolic_buf() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "buf", 64);
    drop(ctx);

    let result = NativeFgets.call(
        &mut state,
        &[sym, RustBV::concrete(10, 64), RustBV::concrete(0, 64)],
    );
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_fgetc_basic() {
    let mut state = RustSimState::new("amd64").unwrap();

    let result = NativeFgetc
        .call(
            &mut state,
            &[RustBV::concrete(0, 64)], // stream (stdin)
        )
        .unwrap();

    let val = result.unwrap();
    // Should be symbolic (can't get concrete value)
    assert!(val.as_u64().is_none());
}

#[test]
fn test_getchar_basic() {
    let mut state = RustSimState::new("amd64").unwrap();

    let result = NativeGetchar.call(&mut state, &[]).unwrap();
    let val = result.unwrap();
    // Should be symbolic
    assert!(val.as_u64().is_none());
}

#[test]
fn test_getc_basic() {
    let mut state = RustSimState::new("amd64").unwrap();

    let result = NativeGetc
        .call(&mut state, &[RustBV::concrete(0, 64)])
        .unwrap();
    let val = result.unwrap();
    assert!(val.as_u64().is_none());
}
