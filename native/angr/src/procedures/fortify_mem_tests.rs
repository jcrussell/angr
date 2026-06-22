// Tests for procedures/fortify_mem.rs (the mem-family `_chk` wrappers).
// Each wrapper must behave identically to its base proc while dropping the
// trailing `destlen` arg; __mempcpy_chk additionally returns dst + n.
use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

#[test]
fn test_memcpy_chk_copies_and_ignores_destlen() {
    let mut state = RustSimState::new("amd64").unwrap();
    let data = b"hello world!";
    state.map_memory_data(0x1000, data, Permission::RWX);
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let result = NativeMemcpyChk
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dst
                RustBV::concrete(0x1000, 64), // src
                RustBV::concrete(12, 64),     // n
                RustBV::concrete(4, 64),      // destlen (deliberately < n; ignored)
            ],
        )
        .unwrap();

    // Returns dst, copies all n bytes regardless of the (smaller) destlen.
    assert_eq!(result.unwrap().as_u64(), Some(0x2000));
    for i in 0..12u64 {
        let val = state.memory_load(0x2000 + i, 1).unwrap();
        assert_eq!(val.as_u64(), Some(data[i as usize] as u64));
    }
}

#[test]
fn test_memmove_chk_overlapping() {
    let mut state = RustSimState::new("amd64").unwrap();
    let data = b"ABCDEFGH";
    state.map_memory_data(0x1000, data, Permission::RWX);

    // Overlapping forward move: dst = src + 2.
    let result = NativeMemmoveChk
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1002, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(6, 64),
                RustBV::concrete(0x100, 64), // destlen ignored
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0x1002));
    // Bytes [0x1002..0x1008) should be the pre-move ABCDEF.
    for (i, b) in b"ABCDEF".iter().enumerate() {
        let val = state.memory_load(0x1002 + i as u64, 1).unwrap();
        assert_eq!(val.as_u64(), Some(*b as u64));
    }
}

#[test]
fn test_memset_chk_fills_and_ignores_destlen() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let result = NativeMemsetChk
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dst
                RustBV::concrete(0x41, 32),   // c = 'A'
                RustBV::concrete(8, 64),      // n
                RustBV::concrete(2, 64),      // destlen ignored
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0x2000));
    for i in 0..8u64 {
        let val = state.memory_load(0x2000 + i, 1).unwrap();
        assert_eq!(val.as_u64(), Some(0x41));
    }
}

#[test]
fn test_mempcpy_chk_returns_dst_plus_n() {
    let mut state = RustSimState::new("amd64").unwrap();
    let data = b"0123456789";
    state.map_memory_data(0x1000, data, Permission::RWX);
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let result = NativeMempcpyChk
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dst
                RustBV::concrete(0x1000, 64), // src
                RustBV::concrete(10, 64),     // n
                RustBV::concrete(0x10, 64),   // destlen ignored
            ],
        )
        .unwrap();

    // mempcpy returns dst + n, NOT dst.
    assert_eq!(result.unwrap().as_u64(), Some(0x2000 + 10));
    for i in 0..10u64 {
        let val = state.memory_load(0x2000 + i, 1).unwrap();
        assert_eq!(val.as_u64(), Some(data[i as usize] as u64));
    }
}

#[test]
fn test_chk_num_args_is_four() {
    assert_eq!(NativeMemcpyChk.num_args(), 4);
    assert_eq!(NativeMemmoveChk.num_args(), 4);
    assert_eq!(NativeMemsetChk.num_args(), 4);
    assert_eq!(NativeMempcpyChk.num_args(), 4);
}

#[test]
fn test_chk_registered_in_registry() {
    let registry = crate::procedures::NativeProcedureRegistry::new();
    for name in [
        "__memcpy_chk",
        "__memmove_chk",
        "__memset_chk",
        "__mempcpy_chk",
    ] {
        assert!(
            registry.get(name).is_some(),
            "{name} should be registered natively"
        );
    }
}
