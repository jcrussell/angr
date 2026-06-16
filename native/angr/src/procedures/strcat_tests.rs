//! Tests for strcat.rs (extracted from inline `mod tests`).
//! See parent module for the SimProcedure implementations under test.

use super::*;
use crate::memory::Permission;

#[test]
fn test_strcat() {
    let mut state = RustSimState::new("amd64").unwrap();
    // dest: "hello\0" with room for more
    let mut buf = vec![0u8; 32];
    buf[..6].copy_from_slice(b"hello\x00");
    state.map_memory_data(0x1000, &buf, Permission::RWX);
    // src: " world\0"
    state.map_memory_data(0x2000, b" world\x00", Permission::RWX);

    let p = NativeStrcat;
    let result = p
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0x1000));

    // Verify concatenated string
    for (i, &expected) in b"hello world\x00".iter().enumerate() {
        let byte = state
            .memory_load(0x1000 + i as u64, 1)
            .unwrap()
            .as_u64()
            .unwrap() as u8;
        assert_eq!(byte, expected, "byte {} mismatch", i);
    }
}

#[test]
fn test_strncat() {
    let mut state = RustSimState::new("amd64").unwrap();
    let mut buf = vec![0u8; 32];
    buf[..3].copy_from_slice(b"hi\x00");
    state.map_memory_data(0x1000, &buf, Permission::RWX);
    state.map_memory_data(0x2000, b"there\x00", Permission::RWX);

    let p = NativeStrncat;
    let result = p
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(3, 64), // only copy 3 bytes
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0x1000));

    // Should be "hithe\0" (3 bytes from "there")
    for (i, &expected) in b"hithe\x00".iter().enumerate() {
        let byte = state
            .memory_load(0x1000 + i as u64, 1)
            .unwrap()
            .as_u64()
            .unwrap() as u8;
        assert_eq!(byte, expected, "byte {} mismatch", i);
    }
}
