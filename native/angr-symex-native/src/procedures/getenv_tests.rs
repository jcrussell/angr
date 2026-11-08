//! Tests for the getenv/setenv/unsetenv/clearenv/putenv SimProcedures (extracted from getenv.rs).
use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::symbolic::RustBV;

fn setup_state() -> RustSimState {
    // Two scratch pages plus the heap region.
    crate::procedures::test_util::amd64_state_with_regions(&[
        (0x1000, 0x1000),
        (0x2000, 0x1000),
        (0xC000_0000, 0x10000),
    ])
}

#[test]
fn test_getenv_not_found() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"PATH\x00", Permission::RWX);

    let result = NativeGetenv
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();

    // Should return NULL
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_setenv_then_getenv() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"HOME\x00", Permission::RWX);
    state.map_memory_data(0x1100, b"/root\x00", Permission::RWX);

    // setenv("HOME", "/root", 1)
    let result = NativeSetenv
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x1100, 64),
                RustBV::concrete(1, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));

    // getenv("HOME")
    let result = NativeGetenv
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();

    let ptr = result.unwrap().as_u64().unwrap();
    assert_ne!(ptr, 0, "getenv should return non-NULL");

    // Read back the value
    let mut val = Vec::new();
    for i in 0..5u64 {
        let byte = state.memory_load(ptr + i, 1).unwrap();
        val.push(byte.as_u64().unwrap() as u8);
    }
    assert_eq!(&val, b"/root");

    // NUL terminator
    let nul = state.memory_load(ptr + 5, 1).unwrap();
    assert_eq!(nul.as_u64(), Some(0));
}

#[test]
fn test_setenv_no_overwrite() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"KEY\x00", Permission::RWX);
    state.map_memory_data(0x1100, b"val1\x00", Permission::RWX);
    state.map_memory_data(0x1200, b"val2\x00", Permission::RWX);

    // setenv("KEY", "val1", 1)
    NativeSetenv
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x1100, 64),
                RustBV::concrete(1, 64),
            ],
        )
        .unwrap();

    // setenv("KEY", "val2", 0) — should NOT overwrite
    NativeSetenv
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x1200, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    // getenv should still return "val1"
    let result = NativeGetenv
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    let ptr = result.unwrap().as_u64().unwrap();
    let mut val = Vec::new();
    for i in 0..4u64 {
        let byte = state.memory_load(ptr + i, 1).unwrap();
        val.push(byte.as_u64().unwrap() as u8);
    }
    assert_eq!(&val, b"val1");
}

#[test]
fn test_putenv() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"LANG=en_US\x00", Permission::RWX);

    let result = NativePutenv
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));

    // Verify via state's environment
    assert_eq!(state.getenv(b"LANG"), Some(b"en_US".as_slice()));
}

#[test]
fn test_putenv_no_equals() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"NOEQUALS\x00", Permission::RWX);

    // Should succeed but not add anything
    let result = NativePutenv
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_getenv_symbolic_name() {
    let mut state = setup_state();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbol("name", 64);
    drop(ctx);

    let result = NativeGetenv.call(&mut state, &[sym]);
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_setenv_symbolic_value_falls_back() {
    // The current native setenv reads the value byte-by-byte from
    // memory. If the value pointer is symbolic, the address-extraction
    // returns SymbolicArgument before any memory access happens.
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"KEY\x00", Permission::RWX);
    let sym_value_ptr = {
        RustBV::symbol("value_ptr", 64)
    };

    let result = NativeSetenv.call(
        &mut state,
        &[
            RustBV::concrete(0x1000, 64),
            sym_value_ptr,
            RustBV::concrete(1, 64),
        ],
    );
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_putenv_symbolic_arg_falls_back() {
    let mut state = setup_state();
    let sym = {
        RustBV::symbol("str_ptr", 64)
    };

    let result = NativePutenv.call(&mut state, &[sym]);
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_unsetenv_removes_key() {
    let mut state = setup_state();
    state.setenv(b"FOO".to_vec(), b"bar".to_vec());
    state.map_memory_data(0x1000, b"FOO\x00", Permission::RWX);

    // unsetenv("FOO") returns 0
    let result = NativeUnsetenv
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));

    // gone
    assert_eq!(state.getenv(b"FOO"), None);
}

#[test]
fn test_unsetenv_missing_key_returns_zero() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"MISSING\x00", Permission::RWX);

    let result = NativeUnsetenv
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_unsetenv_symbolic_name() {
    let mut state = setup_state();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbol("name", 64);
    drop(ctx);

    let result = NativeUnsetenv.call(&mut state, &[sym]);
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_unsetenv_then_getenv_returns_null() {
    let mut state = setup_state();
    state.setenv(b"KEY".to_vec(), b"value".to_vec());
    state.map_memory_data(0x1000, b"KEY\x00", Permission::RWX);

    NativeUnsetenv
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();

    let result = NativeGetenv
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_clearenv_removes_all() {
    let mut state = setup_state();
    state.setenv(b"A".to_vec(), b"1".to_vec());
    state.setenv(b"B".to_vec(), b"2".to_vec());

    let result = NativeClearenv.call(&mut state, &[]).unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));

    assert_eq!(state.getenv(b"A"), None);
    assert_eq!(state.getenv(b"B"), None);
    assert!(state.environment().is_empty());
}

#[test]
fn test_clearenv_on_empty_state() {
    let mut state = setup_state();
    let result = NativeClearenv.call(&mut state, &[]).unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_setenv_unsetenv_chain_visible_to_getenv() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"VAR\x00", Permission::RWX);
    state.map_memory_data(0x1100, b"hello\x00", Permission::RWX);

    // setenv("VAR", "hello", 1)
    NativeSetenv
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x1100, 64),
                RustBV::concrete(1, 64),
            ],
        )
        .unwrap();
    assert_eq!(state.getenv(b"VAR"), Some(b"hello".as_slice()));

    // unsetenv("VAR")
    NativeUnsetenv
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(state.getenv(b"VAR"), None);

    // setenv("VAR", "hello", 1) again — overwrite-flag irrelevant since cleared
    NativeSetenv
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x1100, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();
    assert_eq!(state.getenv(b"VAR"), Some(b"hello".as_slice()));

    // clearenv() wipes everything
    NativeClearenv.call(&mut state, &[]).unwrap();
    assert!(state.environment().is_empty());
}

#[test]
fn test_getenv_env_preserved_on_fork() {
    let mut state = setup_state();
    state.setenv(b"KEY".to_vec(), b"value".to_vec());

    let forked = state.fork();
    assert_eq!(forked.getenv(b"KEY"), Some(b"value".as_slice()));
}

fn getenv_ptr(state: &mut RustSimState, name_addr: u64) -> u64 {
    NativeGetenv
        .call(state, &[RustBV::concrete(u128::from(name_addr), 64)])
        .unwrap()
        .unwrap()
        .as_u64()
        .unwrap()
}

fn store_byte(state: &mut RustSimState, addr: u64, byte: u8) {
    state
        .memory_store(addr, RustBV::concrete(u128::from(byte), 8))
        .unwrap();
}

fn load_byte(state: &mut RustSimState, addr: u64) -> u64 {
    state.memory_load(addr, 1).unwrap().as_u64().unwrap()
}

/// angr-vmi2l.129: a `setenv` value is copied to the heap once; the next
/// `getenv` returns the same pointer and sees a guest write through it.
#[test]
fn test_getenv_setenv_value_pointer_is_stable() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"HOME\x00", Permission::RWX);
    state.setenv(b"HOME".to_vec(), b"/root".to_vec());

    let brk_before = state.heap_brk();
    let first = getenv_ptr(&mut state, 0x1000);
    let brk_after = state.heap_brk();
    store_byte(&mut state, first, b'Z');
    let second = getenv_ptr(&mut state, 0x1000);

    assert_eq!(second, first, "two getenv calls must return one pointer");
    assert_eq!(state.heap_brk(), brk_after, "the second getenv must not allocate");
    assert_ne!(brk_after, brk_before);
    assert_eq!(load_byte(&mut state, second), u64::from(b'Z'));

    // A new setenv value is a new string.
    state.setenv(b"HOME".to_vec(), b"/srv".to_vec());
    assert_ne!(getenv_ptr(&mut state, 0x1000), first);
}

/// angr-vmi2l.129: a seeded variable's `getenv` points into its `envp`
/// string, as Python's `getenv` does, and allocates nothing.
#[test]
fn test_getenv_seeded_value_points_into_envp_string() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"HOME\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"HOME=/seeded\x00", Permission::RWX);
    state.seed_environment(vec![(
        b"HOME".to_vec(),
        b"/seeded".to_vec(),
        Some(0x2005),
    )]);

    let brk_before = state.heap_brk();
    assert_eq!(getenv_ptr(&mut state, 0x1000), 0x2005);
    assert_eq!(getenv_ptr(&mut state, 0x1000), 0x2005);
    assert_eq!(state.heap_brk(), brk_before);
}

/// angr-vmi2l.129: `putenv` makes the caller's string part of the
/// environment, so `getenv` points into it.
#[test]
fn test_getenv_after_putenv_points_into_caller_string() {
    let mut state = setup_state();
    state.map_memory_data(0x1000, b"LANG\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"LANG=en_US\x00", Permission::RWX);
    NativePutenv
        .call(&mut state, &[RustBV::concrete(0x2000, 64)])
        .unwrap();

    assert_eq!(getenv_ptr(&mut state, 0x1000), 0x2005);
}
