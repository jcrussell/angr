// Tests for strstr SimProcedure (extracted from strstr.rs mod tests).
// See rust-mod-tests-sibling-extraction for the split recipe.
use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;

#[test]
fn test_strstr_found() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"world\x00", Permission::RWX);

    let p = NativeStrstr;
    let result = p
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0x1006)); // "world" starts at offset 6
}

#[test]
fn test_strstr_not_found() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"xyz\x00", Permission::RWX);

    let p = NativeStrstr;
    let result = p
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0)); // NULL
}

#[test]
fn test_strstr_empty_needle() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"\x00", Permission::RWX);

    let p = NativeStrstr;
    let result = p
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0x1000)); // Return haystack
}

#[test]
fn test_strstr_at_start() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"hello\x00", Permission::RWX);

    let p = NativeStrstr;
    let result = p
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(result.as_u64(), Some(0x1000));
}
