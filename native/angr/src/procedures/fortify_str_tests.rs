// Tests for procedures/fortify_str.rs (the string-family `_chk` wrappers).
// Each wrapper must behave identically to its base proc while dropping the
// trailing `destlen` arg; __stpcpy_chk additionally returns dest + strlen(src).
use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

fn load_cstr(state: &mut RustSimState, addr: u64, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| {
            state
                .memory_load(addr + i as u64, 1)
                .unwrap()
                .as_u64()
                .unwrap() as u8
        })
        .collect()
}

#[test]
fn test_strcpy_chk_copies_and_ignores_destlen() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello\0", Permission::RWX);
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let result = NativeStrcpyChk
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dest
                RustBV::concrete(0x1000, 64), // src
                RustBV::concrete(2, 64),      // destlen (deliberately < strlen; ignored)
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0x2000));
    assert_eq!(&load_cstr(&mut state, 0x2000, 6), b"hello\0");
}

#[test]
fn test_strncpy_chk_pads_and_ignores_destlen() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hi\0", Permission::RWX);
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let result = NativeStrncpyChk
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dest
                RustBV::concrete(0x1000, 64), // src
                RustBV::concrete(5, 64),      // n
                RustBV::concrete(1, 64),      // destlen ignored
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0x2000));
    // strncpy null-pads the remainder of the n-byte window.
    assert_eq!(&load_cstr(&mut state, 0x2000, 5), b"hi\0\0\0");
}

#[test]
fn test_strcat_chk_appends_and_ignores_destlen() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x2000, b"AB\0", Permission::RWX);
    state.map_memory_data(0x1000, b"CD\0", Permission::RWX);

    let result = NativeStrcatChk
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dest
                RustBV::concrete(0x1000, 64), // src
                RustBV::concrete(1, 64),      // destlen ignored
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0x2000));
    assert_eq!(&load_cstr(&mut state, 0x2000, 5), b"ABCD\0");
}

#[test]
fn test_strncat_chk_appends_bounded() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x2000, b"AB\0", Permission::RWX);
    state.map_memory_data(0x1000, b"CDEF\0", Permission::RWX);

    let result = NativeStrncatChk
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dest
                RustBV::concrete(0x1000, 64), // src
                RustBV::concrete(2, 64),      // n: only "CD"
                RustBV::concrete(1, 64),      // destlen ignored
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0x2000));
    assert_eq!(&load_cstr(&mut state, 0x2000, 5), b"ABCD\0");
}

#[test]
fn test_stpcpy_chk_returns_dest_plus_strlen() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello\0", Permission::RWX);
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let result = NativeStpcpyChk
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dest
                RustBV::concrete(0x1000, 64), // src
                RustBV::concrete(1, 64),      // destlen ignored
            ],
        )
        .unwrap();

    // stpcpy returns a pointer to the written NUL: dest + strlen(src).
    assert_eq!(result.unwrap().as_u64(), Some(0x2000 + 5));
    assert_eq!(&load_cstr(&mut state, 0x2000, 6), b"hello\0");
}

#[test]
fn test_chk_num_args() {
    assert_eq!(NativeStrcpyChk.num_args(), 3);
    assert_eq!(NativeStrncpyChk.num_args(), 4);
    assert_eq!(NativeStrcatChk.num_args(), 3);
    assert_eq!(NativeStrncatChk.num_args(), 4);
    assert_eq!(NativeStpcpyChk.num_args(), 3);
}

#[test]
fn test_chk_registered_in_registry() {
    let registry = crate::procedures::NativeProcedureRegistry::new();
    for name in [
        "__strcpy_chk",
        "__strncpy_chk",
        "__strcat_chk",
        "__strncat_chk",
        "__stpcpy_chk",
    ] {
        assert!(
            registry.get(name).is_some(),
            "{name} should be registered natively"
        );
    }
}
