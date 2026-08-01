//! Tests for strcat.rs (extracted from inline `mod tests`).
//! See parent module for the SimProcedure implementations under test.

use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;

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
        assert_eq!(byte, expected, "byte {i} mismatch");
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
        assert_eq!(byte, expected, "byte {i} mismatch");
    }
}

/// `n` past `MAX_STRING_SCAN` with no terminator inside the cap must defer to
/// Python instead of silently copying a truncated 4096 bytes (angr-9ke6b.109),
/// matching the sibling strncpy's `MaxIterations` bail-out.
#[test]
fn test_strncat_n_over_scan_cap_without_null_defers_to_python() {
    let mut state = RustSimState::new("amd64").unwrap();
    let mut dest = vec![0u8; 64];
    dest[..3].copy_from_slice(b"hi\x00");
    state.map_memory_data(0x1000, &dest, Permission::RWX);
    // src: MAX_STRING_SCAN + 1 non-null bytes, so the bounded scan never sees
    // a terminator.
    state.map_memory_data(0x2000, &vec![b'A'; MAX_STRING_SCAN + 1], Permission::RWX);

    let p = NativeStrncat;
    let err = p
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(MAX_STRING_SCAN as u128 + 1, 64),
            ],
        )
        .unwrap_err();
    assert!(
        matches!(err, ProcedureError::MaxIterations(n) if n == MAX_STRING_SCAN + 1),
        "expected MaxIterations fallback, got {err:?}"
    );

    // dest must be untouched — no partial concatenation before the bail-out.
    for (i, &expected) in b"hi\x00".iter().enumerate() {
        let byte = state
            .memory_load(0x1000 + i as u64, 1)
            .unwrap()
            .as_u64()
            .unwrap() as u8;
        assert_eq!(byte, expected, "dest byte {i} was modified");
    }
}

/// An oversized `n` is still servable natively when src's terminator lands
/// inside the cap: the copy is then identical to the unbounded one.
#[test]
fn test_strncat_n_over_scan_cap_with_early_null_serves_natively() {
    let mut state = RustSimState::new("amd64").unwrap();
    let mut dest = vec![0u8; 64];
    dest[..3].copy_from_slice(b"hi\x00");
    state.map_memory_data(0x1000, &dest, Permission::RWX);
    state.map_memory_data(0x2000, b"there\x00", Permission::RWX);

    let p = NativeStrncat;
    let result = p
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(u64::MAX as u128, 64),
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0x1000));

    for (i, &expected) in b"hithere\x00".iter().enumerate() {
        let byte = state
            .memory_load(0x1000 + i as u64, 1)
            .unwrap()
            .as_u64()
            .unwrap() as u8;
        assert_eq!(byte, expected, "byte {i} mismatch");
    }
}
