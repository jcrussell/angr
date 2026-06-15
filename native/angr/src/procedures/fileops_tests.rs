// angr-oa4o pattern: file-operation SimProcedure unit tests, extracted out
// of the former in-file `mod tests` (~729 lines) into a sibling file to
// shrink procedures/fileops.rs below the god-object threshold. Declared as a
// direct child of `fileops` so `use super::*` reaches its private items.

use super::*;
use crate::memory::Permission;

#[test]
fn test_open_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    // Write pathname "test.txt\0" to memory
    state.map_memory_data(0x1000, b"test.txt\0", Permission::RWX);

    let result = NativeOpen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();

    // Should return fd 3 (first user fd after stdin/stdout/stderr)
    assert_eq!(result.unwrap().as_u64(), Some(3));

    // Verify fd is tracked
    assert!(state.file_system_ref().is_open(3));
    let info = state.file_system_ref().fd_info(3).unwrap();
    assert_eq!(info.0, "test.txt");
}

#[test]
fn test_open_multiple() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
    state.map_memory_data(0x2000, b"b.txt\0", Permission::RWX);

    let r1 = NativeOpen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();
    let r2 = NativeOpen
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();

    assert_eq!(r1.unwrap().as_u64(), Some(3));
    assert_eq!(r2.unwrap().as_u64(), Some(4));
}

#[test]
fn test_close_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"test.txt\0", Permission::RWX);

    // Open a file
    NativeOpen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();

    // Close it
    let result = NativeClose
        .call(&mut state, &[RustBV::concrete(3, 64)])
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0)); // success
    assert!(!state.file_system_ref().is_open(3));
}

#[test]
fn test_close_not_open() {
    let mut state = RustSimState::new("amd64").unwrap();

    let result = NativeClose
        .call(&mut state, &[RustBV::concrete(99, 64)])
        .unwrap();

    // Should return -1 (not open)
    let val = result.unwrap().as_u64().unwrap();
    assert_eq!(val, u64::MAX); // -1 as u64
}

#[test]
fn test_lseek_set() {
    let mut state = RustSimState::new("amd64").unwrap();
    // Open and write some content
    state.file_system().open_with_content(
        "test.txt".to_string(),
        crate::state::FdFlags::ReadOnly,
        vec![0u8; 100],
    );

    // SEEK_SET to position 42
    let result = NativeLseek
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(42, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(42));
}

#[test]
fn test_lseek_cur() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.file_system().open_with_content(
        "test.txt".to_string(),
        crate::state::FdFlags::ReadOnly,
        vec![0u8; 100],
    );

    // Seek to 10
    NativeLseek
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(10, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    // SEEK_CUR +5
    let result = NativeLseek
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(5, 64),
                RustBV::concrete(1, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(15));
}

#[test]
fn test_lseek_end() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.file_system().open_with_content(
        "test.txt".to_string(),
        crate::state::FdFlags::ReadOnly,
        vec![0u8; 100],
    );

    // SEEK_END + 0
    let result = NativeLseek
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(2, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(100));
}

#[test]
fn test_dup_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
    // Open fd=3
    NativeOpen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();

    // dup(3) should allocate fd=4
    let result = NativeDup
        .call(&mut state, &[RustBV::concrete(3, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(4));

    // Both fds should be open and refer to the same name.
    assert!(state.file_system_ref().is_open(3));
    assert!(state.file_system_ref().is_open(4));
    assert_eq!(state.file_system_ref().fd_info(4).unwrap().0, "a.txt");
}

#[test]
fn test_dup_closed_fd() {
    let mut state = RustSimState::new("amd64").unwrap();
    // dup of a never-opened fd returns -1.
    let result = NativeDup
        .call(&mut state, &[RustBV::concrete(99, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(u64::MAX));
}

#[test]
fn test_dup_after_close() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
    NativeOpen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();
    NativeClose
        .call(&mut state, &[RustBV::concrete(3, 64)])
        .unwrap();

    // dup of a closed fd returns -1.
    let result = NativeDup
        .call(&mut state, &[RustBV::concrete(3, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(u64::MAX));
}

#[test]
fn test_dup2_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
    NativeOpen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();

    // dup2(3, 7) — newfd=7 was not open
    let result = NativeDup2
        .call(
            &mut state,
            &[RustBV::concrete(3, 64), RustBV::concrete(7, 64)],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(7));
    assert!(state.file_system_ref().is_open(7));
    assert_eq!(state.file_system_ref().fd_info(7).unwrap().0, "a.txt");

    // next_fd should now be past 7 so subsequent open doesn't collide.
    assert!(state.file_system_ref().next_fd() > 7);
}

#[test]
fn test_dup2_closes_target() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
    state.map_memory_data(0x2000, b"b.txt\0", Permission::RWX);
    // Open two fds: 3=a.txt, 4=b.txt
    NativeOpen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();
    NativeOpen
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();

    // dup2(3, 4) — overwrite fd=4 with a duplicate of fd=3
    let result = NativeDup2
        .call(
            &mut state,
            &[RustBV::concrete(3, 64), RustBV::concrete(4, 64)],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(4));
    // fd=4 now refers to a.txt (the old b.txt is overwritten).
    assert_eq!(state.file_system_ref().fd_info(4).unwrap().0, "a.txt");
}

#[test]
fn test_dup2_same_fd() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
    NativeOpen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();

    // dup2(3, 3) is a no-op when oldfd is open; returns 3.
    let result = NativeDup2
        .call(
            &mut state,
            &[RustBV::concrete(3, 64), RustBV::concrete(3, 64)],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(3));
    assert!(state.file_system_ref().is_open(3));
}

#[test]
fn test_dup2_oldfd_not_open() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeDup2
        .call(
            &mut state,
            &[RustBV::concrete(99, 64), RustBV::concrete(7, 64)],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(u64::MAX));
    assert!(!state.file_system_ref().is_open(7));
}

#[test]
fn test_pipe_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, &[0u8; 16], Permission::RWX);

    // pipe(pipefd) — should return 0 and allocate two fds at 3 and 4.
    let result = NativePipe
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));

    // Read pipefd[0] (4 bytes at 0x1000) and pipefd[1] (4 bytes at 0x1004).
    let read_fd_bv = state.memory_load(0x1000, 4).unwrap();
    let write_fd_bv = state.memory_load(0x1004, 4).unwrap();
    assert_eq!(read_fd_bv.as_u64(), Some(3));
    assert_eq!(write_fd_bv.as_u64(), Some(4));

    // Both fds open with correct flags.
    assert!(state.file_system_ref().is_open(3));
    assert!(state.file_system_ref().is_open(4));
    let info_r = state.file_system_ref().fd_info(3).unwrap();
    let info_w = state.file_system_ref().fd_info(4).unwrap();
    // ReadOnly = 0, WriteOnly = 1
    assert_eq!(info_r.2, 0);
    assert_eq!(info_w.2, 1);
}

#[test]
fn test_pipe_then_dup2_to_stdin() {
    // Realistic pattern: pipe(p); dup2(p[0], 0) — redirects stdin to read end.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, &[0u8; 16], Permission::RWX);

    NativePipe
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    // pipefd[0] = 3, pipefd[1] = 4

    // dup2(3, 0): redirect stdin.
    let result = NativeDup2
        .call(
            &mut state,
            &[RustBV::concrete(3, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));

    // fd=0 is now the pipe read end, not the original stdin.
    assert_eq!(state.file_system_ref().fd_info(0).unwrap().0, "<pipe:r>");
    // fd=3 is still open (dup2 doesn't close oldfd).
    assert!(state.file_system_ref().is_open(3));
    assert!(state.file_system_ref().is_open(4));
}

// --- fopen / fdopen / fclose / fseek / ftell / rewind ---

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
    // No NUL within mapped region — read_cstring will hit a memory error
    // before reaching MAX_FOPEN_PATH_LEN (page boundary triggers Unmapped).
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
        .open_with_content("data".to_string(), FdFlags::ReadOnly, vec![0u8; 100]);
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
        .open_with_content("data".to_string(), FdFlags::ReadOnly, vec![0u8; 100]);
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
        .open_with_content("data".to_string(), FdFlags::ReadOnly, vec![0u8; 100]);
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
fn test_fdopen_existing_fd() {
    let mut state = setup_amd64_state();
    // Open via FileSystem directly so fd 3 is known to be open.
    state
        .file_system()
        .open("foo".to_string(), FdFlags::ReadOnly);
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
