// angr-fs8kb.48: the `FILE *`-based half of the former `fileops_tests.rs`,
// split out alongside its subject module. Declared as a direct child of
// `stream_ops` so `use super::*` reaches its private items.

use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::procedures::test_util::{RET_INT_BITS, arch_ret_bits, assert_ret_width};

/// Set up an amd64 state with the heap region mapped so heap_alloc-backed
/// writes can land. Mirrors the pattern used in malloc tests.
fn setup_amd64_state() -> RustSimState {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0xC000_0000, 0x10000, Permission::RWX);
    state
}

#[test]
fn test_fopen_reads_path_and_writes_fileno() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x1000, b"data.bin\0", Permission::RWX);
    state.file_system().register_known_path("data.bin".to_string());
    state.map_memory_data(0x2000, b"r\0", Permission::RWX);

    let result = NativeFopen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    let file_ptr = result.as_u64().unwrap();
    assert_ne!(file_ptr, 0);

    // FILE._fileno at offset 112 on amd64 should be fd=3 (first user fd).
    let fd_bv = state.memory_load(file_ptr + 112, 4).unwrap();
    assert_eq!(fd_bv.as_u64(), Some(3));
    assert!(state.file_system_ref().is_open(3));
    assert_eq!(state.file_system_ref().fd_info(3).unwrap().0, "data.bin");
    // r → ReadOnly (0)
    assert_eq!(state.file_system_ref().fd_info(3).unwrap().2, 0);
}

/// angr-mx00p.132.13 (test_multi_open_file): `fclose` frees the number, so
/// a later `open` of the same path gets fd 3 again, as on the Python engine.
#[test]
fn test_open_after_fclose_reuses_the_fd() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x1000, b"/tmp/x\0", Permission::RWX);
    state.map_memory_data(0x2000, b"w\0", Permission::RWX);
    let path_mode = [RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)];
    let fp = NativeFopen.call(&mut state, &path_mode).unwrap().unwrap();
    let closed = NativeFclose.call(&mut state, &[fp]).unwrap().unwrap();
    assert_eq!(closed.as_u64(), Some(0));

    let fd = crate::procedures::fileops::NativeOpen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(fd.as_u64(), Some(3));
    assert!(state.file_system_ref().is_open(3));
}

#[test]
fn test_fopen_write_mode_flags_writeonly() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x1000, b"out.txt\0", Permission::RWX);
    state.map_memory_data(0x2000, b"w\0", Permission::RWX);

    NativeFopen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();
    // w → WriteOnly (1)
    assert_eq!(state.file_system_ref().fd_info(3).unwrap().2, 1);
}

#[test]
fn test_fopen_rw_plus_mode() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x1000, b"rw.txt\0", Permission::RWX);
    state.map_memory_data(0x2000, b"r+\0", Permission::RWX);

    NativeFopen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();
    // r+ → ReadWrite (2)
    assert_eq!(state.file_system_ref().fd_info(3).unwrap().2, 2);
}

#[test]
fn test_fopen_binary_suffix_ignored() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x1000, b"bin.bin\0", Permission::RWX);
    state.file_system().register_known_path("bin.bin".to_string());
    state.map_memory_data(0x2000, b"rb\0", Permission::RWX);

    NativeFopen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();
    // rb → r → ReadOnly (0)
    assert_eq!(state.file_system_ref().fd_info(3).unwrap().2, 0);
}

#[test]
fn test_fopen_rw_plus_after_binary_order_independent() {
    // glibc accepts the `+` and `b` flags in any order: "rb+" is read/write
    // binary, identical to "r+b". The positional parser used to miss this.
    let mut state = setup_amd64_state();
    state.map_memory_data(0x1000, b"rwb.txt\0", Permission::RWX);
    state.map_memory_data(0x2000, b"rb+\0", Permission::RWX);

    NativeFopen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();
    // rb+ → ReadWrite (2)
    assert_eq!(state.file_system_ref().fd_info(3).unwrap().2, 2);
}

#[test]
fn test_fopen_append_mode_writes_at_eof() {
    // angr-fs8kb.47: "a" used to be byte-for-byte identical to "w" — same
    // FdFlags, same position 0, no O_APPEND anywhere — so a seek-then-write
    // clobbered earlier bytes instead of appending.
    let mut state = setup_amd64_state();
    state.map_memory_data(0x1000, b"log.txt\0", Permission::RWX);
    state.map_memory_data(0x2000, b"a\0", Permission::RWX);

    NativeFopen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();
    // a → WriteOnly (1), same access mode as "w".
    assert_eq!(state.file_system_ref().fd_info(3).unwrap().2, 1);

    let fs = state.file_system();
    assert!(fs.write(3, b"one"));
    assert_eq!(fs.seek(3, 0, 0), Some(0));
    assert!(fs.write(3, b"two"));
    assert_eq!(fs.fd_content(3), b"onetwo");
}

#[test]
fn test_fopen_write_mode_is_not_append() {
    // The other half of the pair above: "w" must keep overwriting after a
    // seek, so the append bit is what distinguishes the two modes.
    let mut state = setup_amd64_state();
    state.map_memory_data(0x1000, b"log.txt\0", Permission::RWX);
    state.map_memory_data(0x2000, b"w\0", Permission::RWX);

    NativeFopen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();

    let fs = state.file_system();
    assert!(fs.write(3, b"one"));
    assert_eq!(fs.seek(3, 0, 0), Some(0));
    assert!(fs.write(3, b"two"));
    assert_eq!(fs.fd_content(3), b"two");
}

#[test]
fn test_fopen_a_plus_is_readwrite_and_appends() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x1000, b"log.txt\0", Permission::RWX);
    state.map_memory_data(0x2000, b"a+\0", Permission::RWX);

    NativeFopen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap();
    // a+ → ReadWrite (2)
    assert_eq!(state.file_system_ref().fd_info(3).unwrap().2, 2);

    let fs = state.file_system();
    assert!(fs.write(3, b"one"));
    // "a+" reads from the start — append governs writes only.
    assert_eq!(fs.seek(3, 0, 0), Some(0));
    assert_eq!(fs.read(3, 3), b"one");
    assert_eq!(fs.seek(3, 0, 0), Some(0));
    assert!(fs.write(3, b"two"));
    assert_eq!(fs.fd_content(3), b"onetwo");
}

#[test]
fn test_fopen_unknown_mode_falls_back() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x1000, b"x.txt\0", Permission::RWX);
    state.map_memory_data(0x2000, b"xyz\0", Permission::RWX);

    let result = NativeFopen.call(
        &mut state,
        &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
    );
    assert!(result.is_err());
}

#[test]
fn test_fopen_unterminated_path_errors() {
    let mut state = setup_amd64_state();
    // No NUL within mapped region — read_cstring_strict will hit a memory error
    // before reaching MAX_PATH (page boundary triggers Unmapped).
    state.map_memory_data(0x1000, &vec![b'A'; 0x1000], Permission::RWX);
    state.map_memory_data(0x2000, b"r\0", Permission::RWX);

    let result = NativeFopen.call(
        &mut state,
        &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
    );
    assert!(result.is_err());
}

#[test]
fn test_fclose_releases_fd() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x1000, b"f.txt\0", Permission::RWX);
    state.file_system().register_known_path("f.txt".to_string());
    state.map_memory_data(0x2000, b"r\0", Permission::RWX);

    let file_bv = NativeFopen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    let file_ptr = file_bv.as_u64().unwrap();
    assert!(state.file_system_ref().is_open(3));

    let ret = NativeFclose
        .call(&mut state, &[RustBV::concrete(file_ptr as u128, 64)])
        .unwrap()
        .unwrap();
    assert_eq!(ret.as_u64(), Some(0));
    // `int fclose`: a 32-bit return, `FILE *fopen`: a word (angr-vmi2l.116).
    assert_ret_width(&ret, RET_INT_BITS, "fclose");
    assert_ret_width(&file_bv, arch_ret_bits(&state), "fopen");
    assert!(!state.file_system_ref().is_open(3));
}

#[test]
fn test_fclose_of_an_fd_not_open_here_falls_back_to_python() {
    let mut state = setup_amd64_state();
    // A FILE struct whose fd Rust never opened: a bounced proc may have
    // opened it in Python (angr-mx00p.132.15).
    state.map_memory_data(0x10000, &vec![0u8; 0x1000], Permission::RWX);
    state
        .memory_store(0x10000 + 112, RustBV::concrete(42, 32))
        .unwrap();

    let ret = NativeFclose.call(&mut state, &[RustBV::concrete(0x10000, 64)]);
    assert!(matches!(ret, Err(ProcedureError::Other(_))));
}

#[test]
fn test_fopen_read_only_unknown_path_falls_back_to_python() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x1000, b"data.bin\0", Permission::RWX);
    state.map_memory_data(0x2000, b"r\0", Permission::RWX);

    let result = NativeFopen.call(
        &mut state,
        &[RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)],
    );
    assert!(matches!(result, Err(ProcedureError::Other(_))));
    assert!(!state.file_system_ref().is_open(3));
}

#[test]
fn test_fclose_negative_fileno_returns_minus_one() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x10000, &vec![0u8; 0x1000], Permission::RWX);
    state
        .memory_store(0x10000 + 112, RustBV::concrete((-1i32 as u32) as u128, 32))
        .unwrap();

    let ret = NativeFclose
        .call(&mut state, &[RustBV::concrete(0x10000, 64)])
        .unwrap()
        .unwrap();
    assert_eq!(ret.as_u64(), Some(u64::from(u32::MAX)));
}

#[test]
fn test_fseek_set_then_ftell() {
    let mut state = setup_amd64_state();
    state
        .file_system()
        .open_with_content("data".to_string(), FdFlags::ReadOnly, vec![0u8; 100]).expect("fd space is not exhausted in tests");
    // Manually build a FILE struct pointing at fd=3.
    state.map_memory_data(0x20000, &vec![0u8; 0x1000], Permission::RWX);
    state
        .memory_store(0x20000 + 112, RustBV::concrete(3, 32))
        .unwrap();

    // fseek(fp, 42, SEEK_SET)
    let ret = NativeFseek
        .call(
            &mut state,
            &[
                RustBV::concrete(0x20000, 64),
                RustBV::concrete(42, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(ret.as_u64(), Some(0));

    // ftell(fp) == 42
    let pos = NativeFtell
        .call(&mut state, &[RustBV::concrete(0x20000, 64)])
        .unwrap()
        .unwrap();
    assert_eq!(pos.as_u64(), Some(42));
    // `int fseek` returns 32 bits, `long ftell` a word (angr-vmi2l.116).
    assert_ret_width(&ret, RET_INT_BITS, "fseek");
    assert_ret_width(&pos, arch_ret_bits(&state), "ftell");
}

#[test]
fn test_fseek_invalid_whence_returns_minus_one() {
    let mut state = setup_amd64_state();
    state
        .file_system()
        .open_with_content("data".to_string(), FdFlags::ReadOnly, vec![0u8; 100]).expect("fd space is not exhausted in tests");
    state.map_memory_data(0x20000, &vec![0u8; 0x1000], Permission::RWX);
    state
        .memory_store(0x20000 + 112, RustBV::concrete(3, 32))
        .unwrap();

    // whence=99 is invalid → -1
    let ret = NativeFseek
        .call(
            &mut state,
            &[
                RustBV::concrete(0x20000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(99, 64),
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(ret.as_u64(), Some(u64::from(u32::MAX)));
}

#[test]
fn test_fseek_unknown_fd_returns_minus_one() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x20000, &vec![0u8; 0x1000], Permission::RWX);
    state
        .memory_store(0x20000 + 112, RustBV::concrete(99, 32))
        .unwrap();

    let ret = NativeFseek
        .call(
            &mut state,
            &[
                RustBV::concrete(0x20000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(ret.as_u64(), Some(u64::from(u32::MAX)));
}

#[test]
fn test_ftell_unknown_fd_returns_minus_one() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x20000, &vec![0u8; 0x1000], Permission::RWX);
    // fd=77 never registered.
    state
        .memory_store(0x20000 + 112, RustBV::concrete(77, 32))
        .unwrap();

    let pos = NativeFtell
        .call(&mut state, &[RustBV::concrete(0x20000, 64)])
        .unwrap()
        .unwrap();
    assert_eq!(pos.as_u64(), Some(u64::MAX));
}

#[test]
fn test_rewind_returns_none_and_resets_position() {
    let mut state = setup_amd64_state();
    state
        .file_system()
        .open_with_content("data".to_string(), FdFlags::ReadOnly, vec![0u8; 100]).expect("fd space is not exhausted in tests");
    state.map_memory_data(0x20000, &vec![0u8; 0x1000], Permission::RWX);
    state
        .memory_store(0x20000 + 112, RustBV::concrete(3, 32))
        .unwrap();

    // Seek to a non-zero position first.
    NativeFseek
        .call(
            &mut state,
            &[
                RustBV::concrete(0x20000, 64),
                RustBV::concrete(50, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();
    assert_eq!(state.file_system_ref().fd_info(3).unwrap().1, 50);

    // rewind returns void (None) and resets position to 0.
    let ret = NativeRewind
        .call(&mut state, &[RustBV::concrete(0x20000, 64)])
        .unwrap();
    assert!(ret.is_none());
    assert_eq!(state.file_system_ref().fd_info(3).unwrap().1, 0);
}

#[test]
fn test_rewind_unknown_fd_is_a_silent_noop() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x20000, &vec![0u8; 0x1000], Permission::RWX);
    // fd=88 never registered, so seek() fails. rewind is void, so unlike
    // NativeFseek there is no -1 to report it with — see the SILENT(cat-a)
    // tag in NativeRewind. It must still succeed and touch nothing.
    state
        .memory_store(0x20000 + 112, RustBV::concrete(88, 32))
        .unwrap();

    let ret = NativeRewind
        .call(&mut state, &[RustBV::concrete(0x20000, 64)])
        .unwrap();
    assert!(ret.is_none());
    assert!(state.file_system_ref().fd_info(88).is_none());
}

#[test]
fn test_rewind_negative_fd_is_a_silent_noop() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x20000, &vec![0u8; 0x1000], Permission::RWX);
    // _fileno = -1: the fd < 0 guard skips the seek entirely.
    state
        .memory_store(0x20000 + 112, RustBV::concrete(0xFFFF_FFFFu64 as u128, 32))
        .unwrap();

    let ret = NativeRewind
        .call(&mut state, &[RustBV::concrete(0x20000, 64)])
        .unwrap();
    assert!(ret.is_none());
}

#[test]
fn test_fdopen_existing_fd() {
    let mut state = setup_amd64_state();
    // Open via FileSystem directly so fd 3 is known to be open.
    state
        .file_system()
        .open("foo".to_string(), FdFlags::ReadOnly).expect("fd space is not exhausted in tests");
    state.map_memory_data(0x2000, b"r\0", Permission::RWX);

    let fp = NativeFdopen
        .call(
            &mut state,
            &[RustBV::concrete(3, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    let file_ptr = fp.as_u64().unwrap();
    assert_ne!(file_ptr, 0);

    let fd_bv = state.memory_load(file_ptr + 112, 4).unwrap();
    assert_eq!(fd_bv.as_u64(), Some(3));
}

#[test]
fn test_fdopen_fd_not_open_here_falls_back_to_python() {
    // fd 42 is not registered in Rust; a bounced proc may have opened it in
    // Python, which returns NULL for every mode when it did not
    // (angr-mx00p.132.15).
    for mode in [&b"r\0"[..], &b"w\0"[..]] {
        let mut state = setup_amd64_state();
        state.map_memory_data(0x2000, mode, Permission::RWX);
        let fp = NativeFdopen.call(
            &mut state,
            &[RustBV::concrete(42, 64), RustBV::concrete(0x2000, 64)],
        );
        assert!(matches!(fp, Err(ProcedureError::Other(_))));
    }
}

#[test]
fn test_fdopen_negative_fd_returns_null() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x2000, b"r\0", Permission::RWX);

    let fp = NativeFdopen
        .call(
            &mut state,
            &[
                RustBV::concrete((-1i64 as u64) as u128, 64),
                RustBV::concrete(0x2000, 64),
            ],
        )
        .unwrap()
        .unwrap();
    assert_eq!(fp.as_u64(), Some(0));
}


/// angr-vmi2l.113: on x86 `long` is 32 bits, so `fseek(fp, -2, SEEK_END)`
/// lands 2 bytes before EOF rather than at +0xfffffffe, and a negative target
/// returns -1 with the position unchanged.
#[test]
fn test_fseek_negative_offset_x86_sign_extends_and_rejects_negative_target() {
    let mut state = RustSimState::new("x86").unwrap();
    let fd = state
        .file_system()
        .open_with_content("data".to_string(), FdFlags::ReadOnly, b"hello".to_vec())
        .expect("fd space is not exhausted in tests");
    state.map_memory_data(0x20000, &vec![0u8; 0x1000], Permission::RWX);
    state
        .memory_store(0x20000 + 56, RustBV::concrete(u128::from(fd), 32))
        .unwrap();
    let fseek = |state: &mut RustSimState, offset: i32, whence: u128| {
        NativeFseek
            .call(
                state,
                &[
                    RustBV::concrete(0x20000, 32),
                    RustBV::concrete(u128::from(offset as u32), 32),
                    RustBV::concrete(whence, 32),
                ],
            )
            .unwrap()
            .unwrap()
            .as_u64()
    };

    assert_eq!(fseek(&mut state, -2, 2), Some(0));
    assert_eq!(state.file_system_ref().fd_info(fd).unwrap().1, 3);
    assert_eq!(fseek(&mut state, -5, 0), Some(0xffff_ffff));
    assert_eq!(fseek(&mut state, -6, 2), Some(0xffff_ffff));
    assert_eq!(state.file_system_ref().fd_info(fd).unwrap().1, 3);
}

/// angr-vmi2l.115: Python's `SimFileDescriptor.seek` fails past EOF where
/// glibc succeeds, so fseek defers there.
#[test]
fn test_fseek_past_eof_falls_back() {
    let mut state = setup_amd64_state();
    state
        .file_system()
        .open_with_content("data".to_string(), FdFlags::ReadOnly, vec![0u8; 100]).expect("fd space is not exhausted in tests");
    state.map_memory_data(0x20000, &vec![0u8; 0x1000], Permission::RWX);
    state
        .memory_store(0x20000 + 112, RustBV::concrete(3, 32))
        .unwrap();
    let fseek = |state: &mut RustSimState, offset: u128| {
        NativeFseek.call(
            state,
            &[RustBV::concrete(0x20000, 64), RustBV::concrete(offset, 64), RustBV::concrete(0, 64)],
        )
    };
    assert!(fseek(&mut state, 101).is_err());
    assert_eq!(fseek(&mut state, 100).unwrap().unwrap().as_u64(), Some(0));
}
