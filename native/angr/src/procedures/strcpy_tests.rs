// Unit tests for strcpy.rs (NativeStrcpy / NativeStrncpy / NativeStrdup).
// Split out per rust-mod-tests-sibling-extraction; included via #[cfg(test)] #[path].
use super::*;
use crate::memory::Permission;
use crate::procedures::{NativeSimProcedure, ProcedureError};
use crate::state::RustSimState;

#[test]
fn test_strcpy_basic() {
    let mut state = RustSimState::new("amd64").unwrap();

    // Source string
    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
    // Destination buffer
    state.map_memory_data(0x2000, &[0u8; 16], Permission::RWX);

    let proc = NativeStrcpy;
    let result = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x1000, 64)],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0x2000));

    // Verify "hello\0" was copied
    let loaded = state.memory_load(0x2000, 6).unwrap();
    // "hello\0" in little-endian u48
    let expected = u64::from_le_bytes([b'h', b'e', b'l', b'l', b'o', 0, 0, 0]);
    assert_eq!(loaded.as_u64(), Some(expected));
}

#[test]
fn test_strcpy_empty() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"\x00", Permission::RWX);
    state.map_memory_data(0x2000, &[0xFFu8; 8], Permission::RWX);

    NativeStrcpy
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0x1000, 64)],
        )
        .unwrap();

    // First byte should be null
    let first = state.memory_load(0x2000, 1).unwrap();
    assert_eq!(first.as_u64(), Some(0));
}

#[test]
fn test_strncpy_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);
    state.map_memory_data(0x2000, &[0xFFu8; 16], Permission::RWX);

    NativeStrncpy
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(5, 64),
            ],
        )
        .unwrap();

    // Should copy exactly 5 bytes: "hello"
    for (i, &expected) in b"hello".iter().enumerate() {
        let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
        assert_eq!(byte.as_u64(), Some(expected as u64));
    }
}

#[test]
fn test_strncpy_pads_with_null() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hi\x00", Permission::RWX);
    state.map_memory_data(0x2000, &[0xFFu8; 8], Permission::RWX);

    NativeStrncpy
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(6, 64),
            ],
        )
        .unwrap();

    // Bytes after null should also be null-padded
    for i in 2..6u64 {
        let byte = state.memory_load(0x2000 + i, 1).unwrap();
        assert_eq!(byte.as_u64(), Some(0));
    }
}

#[test]
fn test_strdup_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0xC000_0000, 0x10000, Permission::RWX);
    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

    let result = NativeStrdup
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    let new_addr = result.unwrap().as_u64().unwrap();
    assert!(new_addr >= 0xC000_0000);

    // Verify "hello\0" was copied
    for (i, &expected) in b"hello\x00".iter().enumerate() {
        let byte = state.memory_load(new_addr + i as u64, 1).unwrap();
        assert_eq!(byte.as_u64(), Some(expected as u64));
    }

    // Verify heap metadata
    assert!(state.heap_metadata().is_allocated(new_addr));
    assert_eq!(state.heap_metadata().alloc_size(new_addr), Some(6));
}

#[test]
fn test_strdup_empty_string() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0xC000_0000, 0x10000, Permission::RWX);
    state.map_memory_data(0x1000, b"\x00", Permission::RWX);

    let result = NativeStrdup
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    let new_addr = result.unwrap().as_u64().unwrap();

    // Should allocate 1 byte for null terminator
    let byte = state.memory_load(new_addr, 1).unwrap();
    assert_eq!(byte.as_u64(), Some(0));
    assert_eq!(state.heap_metadata().alloc_size(new_addr), Some(1));
}

#[test]
fn test_strdup_symbolic_arg() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "ptr", 64);
    drop(ctx);
    let result = NativeStrdup.call(&mut state, &[sym]);
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}
