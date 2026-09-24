// angr-fs8kb.48: the `FILE *`-based half of the former `fileops_tests.rs`,
// split out alongside its subject module. Declared as a direct child of
// `stream_ops` so `use super::*` reaches its private items.

use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;

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
    assert!(!state.file_system_ref().is_open(3));
}

#[test]
fn test_fclose_returns_minus_one_for_closed_fd() {
    let mut state = setup_amd64_state();
    // Manually construct a FILE struct with a stale fd.
    state.map_memory_data(0x10000, &vec![0u8; 0x1000], Permission::RWX);
    state
        .memory_store(0x10000 + 112, RustBV::concrete(42, 32))
        .unwrap();

    let ret = NativeFclose
        .call(&mut state, &[RustBV::concrete(0x10000, 64)])
        .unwrap()
        .unwrap();
    assert_eq!(ret.as_u64(), Some(u64::MAX));
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
    assert_eq!(ret.as_u64(), Some(u64::MAX));
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
    assert_eq!(ret.as_u64(), Some(u64::MAX));
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
    assert_eq!(ret.as_u64(), Some(u64::MAX));
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
fn test_fdopen_unknown_fd_returns_null() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x2000, b"r\0", Permission::RWX);

    // fd 42 is not registered → fdopen returns NULL.
    let fp = NativeFdopen
        .call(
            &mut state,
            &[RustBV::concrete(42, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(fp.as_u64(), Some(0));
}

#[test]
fn test_fdopen_unknown_fd_returns_null_for_creating_mode() {
    let mut state = setup_amd64_state();
    state.map_memory_data(0x2000, b"w\0", Permission::RWX);

    // A creating mode (`w`) on a not-open fd still returns NULL, same as `r`:
    // Python's create_file=True only materializes a file for a multi-solution
    // symbolic fd, which the `concrete` arg spec excludes here (angr-j0dp3).
    let fp = NativeFdopen
        .call(
            &mut state,
            &[RustBV::concrete(42, 64), RustBV::concrete(0x2000, 64)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(fp.as_u64(), Some(0));
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

