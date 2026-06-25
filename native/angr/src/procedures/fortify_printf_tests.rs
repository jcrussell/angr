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

/// Map a FILE struct at `file_ptr` whose AMD64 `_IO_FILE._fileno` field (byte
/// offset 112) holds `fd`. Mirrors the helper in `printf_tests.rs`.
fn setup_file_struct(state: &mut RustSimState, file_ptr: u64, fd: i32) {
    const AMD64_FD_OFFSET: u64 = 112;
    state.map_memory_data(file_ptr & !0xfff, &vec![0u8; 0x4000], Permission::RWX);
    let fd_bv = RustBV::concrete(fd as u32 as u128, 32);
    state
        .memory_store(file_ptr + AMD64_FD_OFFSET, fd_bv)
        .unwrap();
}

#[test]
fn test_fprintf_chk_drops_flag_and_writes_fd() {
    // __fprintf_chk(stderr, flag, fmt) writes the raw format to fd 2.
    let mut state = RustSimState::new("amd64").unwrap();
    let file_ptr = 0x5000;
    setup_file_struct(&mut state, file_ptr, 2);
    state.map_memory_data(0x1000, b"error: %d\x00", Permission::RWX);

    let result = NativeFprintfChk
        .call(
            &mut state,
            &[
                RustBV::concrete(file_ptr as u128, 64), // stream
                RustBV::concrete(1, 64),                // flag (dropped)
                RustBV::concrete(0x1000, 64),           // format
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(9));
    assert_eq!(state.fd_buffer(2), b"error: %d");
    assert!(state.stdout_buffer().is_empty());
}

#[test]
fn test_vsnprintf_chk_drops_flag_slen_forwards_to_stub() {
    // __vsnprintf_chk(dest, maxlen, flag, slen, fmt, ap) forwards to the
    // degenerate vsnprintf stub: maxlen>0 → single NUL at dest, returns 1.
    let mut state = crate::procedures::test_util::amd64_state_with_regions(&[(0x2000, 0x1000)]);
    state.map_memory_data(0x1000, b"val=%d\x00", Permission::RWX);
    state
        .memory_store(0x2000, RustBV::concrete(0x41, 8))
        .unwrap(); // pre-seed 'A'

    let result = NativeVsnprintfChk
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dest
                RustBV::concrete(4, 64),      // maxlen
                RustBV::concrete(1, 64),      // flag (dropped)
                RustBV::concrete(64, 64),     // slen (dropped)
                RustBV::concrete(0x1000, 64), // format
                RustBV::concrete(0, 64),      // va_list
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(1));
    let byte = state.memory_load(0x2000, 1).unwrap();
    assert_eq!(byte.as_u64().unwrap() as u8, 0); // NUL written over the 'A'
}

#[test]
fn test_vsnprintf_chk_zero_maxlen_returns_zero() {
    let mut state = crate::procedures::test_util::amd64_state_with_regions(&[(0x2000, 0x1000)]);
    state.map_memory_data(0x1000, b"x\x00", Permission::RWX);

    let result = NativeVsnprintfChk
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dest
                RustBV::concrete(0, 64),      // maxlen == 0
                RustBV::concrete(1, 64),      // flag (dropped)
                RustBV::concrete(64, 64),     // slen (dropped)
                RustBV::concrete(0x1000, 64), // format
                RustBV::concrete(0, 64),      // va_list
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_chk_num_args() {
    assert_eq!(NativePrintfChk.num_args(), 2);
    assert_eq!(NativeSprintfChk.num_args(), 10);
    assert_eq!(NativeSnprintfChk.num_args(), 11);
    assert_eq!(NativeFprintfChk.num_args(), 3);
    assert_eq!(NativeVsnprintfChk.num_args(), 6);
}

#[test]
fn test_chk_registered_in_registry() {
    let registry = crate::procedures::NativeProcedureRegistry::new();
    for name in [
        "__printf_chk",
        "__sprintf_chk",
        "__snprintf_chk",
        "__fprintf_chk",
        "__vsnprintf_chk",
    ] {
        assert!(
            registry.get(name).is_some(),
            "{name} should be registered natively"
        );
    }
}
