// angr-oa4o pattern: file-operation SimProcedure unit tests, extracted out
// of the former in-file `mod tests` (~729 lines) into a sibling file to
// shrink procedures/fileops.rs below the god-object threshold. Declared as a
// direct child of `fileops` so `use super::*` reaches its private items.
//
// angr-fs8kb.48 moved the `FILE *`-based half (fopen/fdopen/fclose/fseek/
// ftell/rewind) to `stream_ops_tests.rs`; what remains covers the fd-table
// primitives open/close/lseek/dup/dup2/pipe.

use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;

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
    ).expect("fd space is not exhausted in tests");

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
    ).expect("fd space is not exhausted in tests");

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
    ).expect("fd space is not exhausted in tests");

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
fn test_dup2_huge_newfd_returns_ebadf_without_wrapping_next_fd() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"a.txt\0", Permission::RWX);
    NativeOpen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();
    let before = state.file_system_ref().next_fd();

    // No syscall layer sits above this proc to apply NEWFD_LIMIT, so these
    // used to reach FileSystem::dup2's raw `next_fd = newfd + 1` bump
    // (angr-03vl4.52). 0xFFFF_FFFF wraps the u32 counter; 0x1_0000_0000 is
    // additionally a `as u32` truncation to 0 == stdin.
    for newfd in [0xFFFF_FFFFu128, 0x1_0000_0000, u128::from(u64::MAX)] {
        let result = NativeDup2
            .call(
                &mut state,
                &[RustBV::concrete(3, 64), RustBV::concrete(newfd, 64)],
            )
            .unwrap();
        assert_eq!(
            result.unwrap().as_u64(),
            Some(u64::MAX),
            "dup2(3, {newfd:#x}) must fail"
        );
    }
    assert_eq!(state.file_system_ref().next_fd(), before);
    // stdin must still be stdin, not an alias of a.txt.
    assert_eq!(state.file_system_ref().fd_info(0).unwrap().0, "/dev/stdin");
}

#[test]
fn test_dup_oldfd_wider_than_u32_is_not_truncated() {
    let mut state = RustSimState::new("amd64").unwrap();
    // 0x1_0000_0000 truncates to 0 (stdin, which IS open), so an `as u32`
    // cast would happily duplicate stdin here (angr-03vl4.52).
    let result = NativeDup
        .call(&mut state, &[RustBV::concrete(0x1_0000_0000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(u64::MAX));
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
fn test_pipe_honors_big_endian_state_override() {
    // Regression (angr-9ke6b.1): `pipe` encoded the fd bytes off
    // `Arch::is_little_endian()`, which is hardcoded true on every arch --
    // so a genuinely big-endian ARM state got LE-encoded fds that then read
    // back byte-reversed through the BE memory load. The fd byte order must
    // follow the state's configured endness, not the arch default.
    let mut state = RustSimState::new_with_endian("ARM", Some(false)).unwrap();
    assert!(!state.is_little_endian());
    state.map_memory_data(0x1000, &[0u8; 16], Permission::RWX);

    let result = NativePipe
        .call(&mut state, &[RustBV::concrete(0x1000, 32)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));

    // Loads go through the same BE memory, so a matching encoding round-trips.
    assert_eq!(state.memory_load(0x1000, 4).unwrap().as_u64(), Some(3));
    assert_eq!(state.memory_load(0x1004, 4).unwrap().as_u64(), Some(4));

    // Byte-level check: BE puts the fd in the *last* byte of each word.
    assert_eq!(state.memory_load(0x1000, 1).unwrap().as_u64(), Some(0));
    assert_eq!(state.memory_load(0x1003, 1).unwrap().as_u64(), Some(3));
    assert_eq!(state.memory_load(0x1007, 1).unwrap().as_u64(), Some(4));
}

#[test]
fn test_pipe_honors_little_endian_state_override() {
    // The mirror of the BE case: an explicitly-LE ARM state keeps LE encoding,
    // with the fd in the *first* byte of each word.
    let mut state = RustSimState::new_with_endian("ARM", Some(true)).unwrap();
    state.map_memory_data(0x1000, &[0u8; 16], Permission::RWX);

    NativePipe
        .call(&mut state, &[RustBV::concrete(0x1000, 32)])
        .unwrap();

    assert_eq!(state.memory_load(0x1000, 4).unwrap().as_u64(), Some(3));
    assert_eq!(state.memory_load(0x1000, 1).unwrap().as_u64(), Some(3));
    assert_eq!(state.memory_load(0x1004, 1).unwrap().as_u64(), Some(4));
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


#[test]
fn test_open_o_append_flag_appends() {
    // The `open(2)` half of the same wiring: O_APPEND sits outside the two
    // access-mode bits `FdFlags::from_posix` reads.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"log.txt\0", Permission::RWX);

    NativeOpen
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(u128::from(crate::state::O_APPEND | 1), 64),
            ],
        )
        .unwrap();
    assert_eq!(state.file_system_ref().fd_info(3).unwrap().2, 1);

    let fs = state.file_system();
    assert!(fs.write(3, b"one"));
    assert_eq!(fs.seek(3, 0, 0), Some(0));
    assert!(fs.write(3, b"two"));
    assert_eq!(fs.fd_content(3), b"onetwo");
}


#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_open_symbolic_pathname_is_concretized() {
    // A symbolic byte in the pathname used to bounce the call to Python
    // (angr-gorvf.13). Python's open concretizes the path with solver.eval;
    // so do we — the call must be served, and the resulting name must be a
    // model of the symbolic buffer, not a decline.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"a\x00c.txt\x00", Permission::RWX);
    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "path_1", 8)
    };
    state.memory_store(0x1001, sym.clone()).unwrap();
    // Pin the symbolic byte so the eval'd path is deterministic.
    let eq = {
        let ctx = state.solver().borrow();
        sym.eq(&RustBV::concrete(b'b' as u128, 8), &ctx)
    };
    state.add_constraint(eq);

    let result = NativeOpen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .expect("symbolic pathname byte must be concretized, not bounced");

    assert_eq!(result.unwrap().as_u64(), Some(3));
    let info = state.file_system_ref().fd_info(3).unwrap();
    assert_eq!(info.0, "abc.txt");
}

#[test]
fn test_open_overlong_pathname_errors_instead_of_truncating() {
    // A concrete pathname with no NUL inside the MAX_PATH window used to be
    // returned as its 256-byte prefix, silently opening a *different* path
    // than the program asked for. Python's open uses an unbounded strlen, so
    // the only correct native answer is to decline (angr-sqfj8.80).
    let mut state = RustSimState::new("amd64").unwrap();
    let mut path = vec![b'a'; MAX_PATH as usize + 44];
    path.push(0);
    state.map_memory_data(0x1000, &path, Permission::RWX);

    let err = NativeOpen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .expect_err("overlong pathname must decline, not truncate");
    assert!(
        matches!(err, ProcedureError::MaxIterations(n) if n == MAX_PATH as usize),
        "expected MaxIterations({MAX_PATH}), got {err:?}"
    );
    // Nothing was opened: the fallback must see an untouched fd table.
    assert!(!state.file_system_ref().is_open(3));
}

#[test]
fn test_open_pathname_filling_the_window_exactly_is_served() {
    // Boundary: the NUL sits at the last scannable index (MAX_PATH - 1), so
    // the scan still finds it and the call is served natively.
    let mut state = RustSimState::new("amd64").unwrap();
    let mut path = vec![b'a'; MAX_PATH as usize - 1];
    path.push(0);
    state.map_memory_data(0x1000, &path, Permission::RWX);

    let result = NativeOpen
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), RustBV::concrete(0, 64)],
        )
        .expect("a pathname that exactly fills the window must still be served");
    assert_eq!(result.unwrap().as_u64(), Some(3));
    let info = state.file_system_ref().fd_info(3).unwrap();
    assert_eq!(info.0.len(), MAX_PATH as usize - 1);
}
