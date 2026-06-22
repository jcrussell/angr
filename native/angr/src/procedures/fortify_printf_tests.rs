// Tests for procedures/fortify_printf.rs (the printf-family `_chk` wrappers).
// Each wrapper must behave identically to its base proc while dropping the
// injected `flag`/`slen` args glibc passes in `_FORTIFY_SOURCE` builds.
use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

#[test]
fn test_printf_chk_drops_flag_and_writes_stdout() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);

    let result = NativePrintfChk
        .call(
            &mut state,
            &[
                RustBV::concrete(1, 64),      // flag (dropped)
                RustBV::concrete(0x1000, 64), // format
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(11));
    assert_eq!(state.stdout_buffer(), b"hello world");
}

#[test]
fn test_sprintf_chk_drops_flag_slen_and_formats() {
    let mut state = crate::procedures::test_util::amd64_state_with_regions(&[(0x2000, 0x1000)]);
    state.map_memory_data(0x1000, b"val=%d\x00", Permission::RWX);

    let result = NativeSprintfChk
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dest
                RustBV::concrete(1, 64),      // flag (dropped)
                RustBV::concrete(64, 64),     // slen (dropped)
                RustBV::concrete(0x1000, 64), // format
                RustBV::concrete(42, 64),     // %d arg
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(6)); // "val=42"
    for (i, &expected) in b"val=42\x00".iter().enumerate() {
        let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
        assert_eq!(byte.as_u64().unwrap() as u8, expected);
    }
}

#[test]
fn test_snprintf_chk_drops_flag_slen_respects_maxlen() {
    let mut state = crate::procedures::test_util::amd64_state_with_regions(&[(0x2000, 0x1000)]);
    state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);

    let result = NativeSnprintfChk
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dest
                RustBV::concrete(4, 64),      // maxlen: only 3 chars + NUL
                RustBV::concrete(1, 64),      // flag (dropped)
                RustBV::concrete(64, 64),     // slen (dropped)
                RustBV::concrete(0x1000, 64), // format
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    // snprintf returns the would-have-been length, not the truncated one.
    assert_eq!(result.unwrap().as_u64(), Some(11));
    // dest holds "hel\0" (maxlen-1 bytes + NUL).
    for (i, &expected) in b"hel\x00".iter().enumerate() {
        let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
        assert_eq!(byte.as_u64().unwrap() as u8, expected);
    }
}

#[test]
fn test_chk_num_args() {
    assert_eq!(NativePrintfChk.num_args(), 2);
    assert_eq!(NativeSprintfChk.num_args(), 10);
    assert_eq!(NativeSnprintfChk.num_args(), 11);
}

#[test]
fn test_chk_registered_in_registry() {
    let registry = crate::procedures::NativeProcedureRegistry::new();
    for name in ["__printf_chk", "__sprintf_chk", "__snprintf_chk"] {
        assert!(
            registry.get(name).is_some(),
            "{name} should be registered natively"
        );
    }
}
