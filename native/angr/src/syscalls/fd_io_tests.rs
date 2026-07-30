//! Tests for fd_io syscall handlers (readv/writev/etc., extracted from fd_io.rs).

use super::*;
use crate::memory::Permission;
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::{RustBV, SymContext};

fn fresh_state() -> RustSimState {
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state.map_memory(0x2000, 0x2000, Permission::RWX);
    state
}

/// Lay out a `struct iovec[]` at `iov_addr` pointing at `segments`
/// `(base, len)` pairs (amd64: 8-byte words).
fn write_iovec_array(state: &mut RustSimState, iov_addr: u64, segments: &[(u64, u64)]) {
    for (idx, (base, len)) in segments.iter().enumerate() {
        let elem = iov_addr + (idx as u64) * 16;
        state
            .memory_store(elem, RustBV::concrete(*base as u128, 64))
            .unwrap();
        state
            .memory_store(elem + 8, RustBV::concrete(*len as u128, 64))
            .unwrap();
    }
}

#[test]
fn metadata() {
    assert_eq!(NativeLseekSyscall.name(), "lseek");
    assert_eq!(NativeLseekSyscall.num_args(), 3);
    assert_eq!(NativeWritevSyscall.name(), "writev");
    assert_eq!(NativeReadvSyscall.name(), "readv");
}

#[test]
fn lseek_set_returns_new_position() {
    let mut state = fresh_state();
    state.file_system().open_with_content(
        "f.bin".to_string(),
        FdFlags::ReadOnly,
        b"abcdefgh".to_vec(),
    );
    // SEEK_SET to 3.
    let out = NativeLseekSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(3, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("ok");
    match out {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 3),
        _ => panic!("expected Continue"),
    }
    assert_eq!(state.file_system_ref().fd_info(3).unwrap().1, 3);
}

#[test]
fn lseek_bad_whence_returns_neg1() {
    let mut state = fresh_state();
    state.file_system().open("f".to_string(), FdFlags::ReadOnly);
    let out = NativeLseekSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(99, 64),
            ],
        )
        .expect("ok");
    match out {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, (-1i64) as u64),
        _ => panic!("expected Continue"),
    }
}

#[test]
fn lseek_unknown_fd_falls_back() {
    let mut state = fresh_state();
    let err = NativeLseekSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(7, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("fallback");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn lseek_symbolic_fd_falls_back() {
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let err = NativeLseekSyscall
        .call(
            &mut state,
            &[
                RustBV::symbolic(&ctx, "fd", 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("fallback");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

#[test]
fn writev_concatenates_segments_to_stdout() {
    let mut state = fresh_state();
    state.map_memory_data(0x3000, b"hello", Permission::RWX);
    state.map_memory_data(0x3100, b" world", Permission::RWX);
    write_iovec_array(&mut state, 0x2000, &[(0x3000, 5), (0x3100, 6)]);

    let out = NativeWritevSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(1, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(2, 64),
            ],
        )
        .expect("ok");
    match out {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 11),
        _ => panic!("expected Continue"),
    }
    assert_eq!(state.stdout_buffer(), b"hello world");
}

#[test]
fn writev_symbolic_byte_leaves_fd_untouched() {
    let mut state = fresh_state();
    state.map_memory_data(0x3000, b"ok", Permission::RWX);
    // Second segment has a symbolic byte.
    let ctx = SymContext::new();
    state
        .memory_store(0x3100, RustBV::symbolic(&ctx, "b", 8))
        .unwrap();
    write_iovec_array(&mut state, 0x2000, &[(0x3000, 2), (0x3100, 1)]);

    let err = NativeWritevSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(1, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(2, 64),
            ],
        )
        .expect_err("fallback");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
    // No partial write to stdout.
    assert_eq!(state.stdout_buffer(), b"");
}

#[test]
fn writev_stdin_falls_back() {
    let mut state = fresh_state();
    write_iovec_array(&mut state, 0x2000, &[(0x3000, 1)]);
    let err = NativeWritevSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(1, 64),
            ],
        )
        .expect_err("fallback");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn writev_user_fd_appends() {
    let mut state = fresh_state();
    state
        .file_system()
        .open("out".to_string(), FdFlags::WriteOnly);
    state.map_memory_data(0x3000, b"abc", Permission::RWX);
    write_iovec_array(&mut state, 0x2000, &[(0x3000, 3)]);
    NativeWritevSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(1, 64),
            ],
        )
        .expect("ok");
    assert_eq!(state.file_system_ref().fd_content(3), b"abc");
}

#[test]
fn readv_user_fd_scatters_content() {
    let mut state = fresh_state();
    state
        .file_system()
        .open_with_content("in".to_string(), FdFlags::ReadOnly, b"abcdef".to_vec());
    write_iovec_array(&mut state, 0x2000, &[(0x3000, 3), (0x3100, 3)]);
    let out = NativeReadvSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(2, 64),
            ],
        )
        .expect("ok");
    match out {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 6),
        _ => panic!("expected Continue"),
    }
    for (i, &b) in b"abc".iter().enumerate() {
        assert_eq!(
            state.memory_load(0x3000 + i as u64, 1).unwrap().as_u64(),
            Some(b as u64)
        );
    }
    for (i, &b) in b"def".iter().enumerate() {
        assert_eq!(
            state.memory_load(0x3100 + i as u64, 1).unwrap().as_u64(),
            Some(b as u64)
        );
    }
}

#[test]
fn readv_stdin_writes_symbolic() {
    let mut state = fresh_state();
    write_iovec_array(&mut state, 0x2000, &[(0x3000, 4)]);
    let out = NativeReadvSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(1, 64),
            ],
        )
        .expect("ok");
    match out {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 4),
        _ => panic!("expected Continue"),
    }
    for i in 0..4u64 {
        assert!(
            state.memory_load(0x3000 + i, 1).unwrap().as_u64().is_none(),
            "byte {i} should be symbolic"
        );
    }
}

#[test]
fn readv_symbolic_fd_scatters_symbolic_bytes_natively() {
    // A symbolic-stream fd (empty content, flagged symbolic via open_symbolic)
    // scatters fresh symbolic bytes natively instead of falling back to
    // Python — matches `read`'s is_symbolic fast path (angr-myzjx.12).
    let mut state = fresh_state();
    let fd = state
        .file_system()
        .open_symbolic("sym.in".to_string(), FdFlags::ReadOnly);
    assert!(state.file_system_ref().is_symbolic(fd));
    write_iovec_array(&mut state, 0x2000, &[(0x3000, 4), (0x3010, 2)]);
    let out = NativeReadvSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(2, 64),
            ],
        )
        .expect("served natively");
    match out {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 6),
        _ => panic!("expected Continue"),
    }
    for i in 0..4u64 {
        assert!(
            state.memory_load(0x3000 + i, 1).unwrap().as_u64().is_none(),
            "seg0 byte {i} should be symbolic"
        );
    }
    for i in 0..2u64 {
        assert!(
            state.memory_load(0x3010 + i, 1).unwrap().as_u64().is_none(),
            "seg1 byte {i} should be symbolic"
        );
    }
}

#[test]
fn readv_symbolic_fd_never_hits_eof() {
    // Unlike a concrete fd, a symbolic stream keeps producing bytes — a
    // second readv still returns the full requested total (no EOF).
    let mut state = fresh_state();
    let fd = state
        .file_system()
        .open_symbolic("sym.in".to_string(), FdFlags::ReadOnly);
    write_iovec_array(&mut state, 0x2000, &[(0x3000, 3)]);
    for _ in 0..2 {
        let out = NativeReadvSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(fd as u128, 64),
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(1, 64),
                ],
            )
            .expect("served natively");
        match out {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 3),
            _ => panic!("expected Continue"),
        }
    }
}

#[test]
fn readv_unknown_fd_falls_back() {
    let mut state = fresh_state();
    write_iovec_array(&mut state, 0x2000, &[(0x3000, 4)]);
    let err = NativeReadvSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(9, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(1, 64),
            ],
        )
        .expect_err("fallback");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn pread64_reads_at_offset_without_moving_position() {
    let mut state = fresh_state();
    state.file_system().open_with_content(
        "in".to_string(),
        FdFlags::ReadOnly,
        b"abcdefgh".to_vec(),
    );
    // pread64(fd=3, buf=0x3000, nbyte=3, offset=2) -> "cde"
    let out = NativePread64Syscall
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x3000, 64),
                RustBV::concrete(3, 64),
                RustBV::concrete(2, 64),
            ],
        )
        .expect("ok");
    match out {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 3),
        _ => panic!("expected Continue"),
    }
    for (i, &b) in b"cde".iter().enumerate() {
        assert_eq!(
            state.memory_load(0x3000 + i as u64, 1).unwrap().as_u64(),
            Some(b as u64)
        );
    }
    // Position must be untouched: a subsequent read starts at byte 0.
    assert_eq!(state.file_system().read(3, 3), b"abc");
}

#[test]
fn pread64_symbolic_offset_falls_back() {
    let mut state = fresh_state();
    state
        .file_system()
        .open_with_content("in".to_string(), FdFlags::ReadOnly, b"abcd".to_vec());
    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "off", 64)
    };
    let err = NativePread64Syscall
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x3000, 64),
                RustBV::concrete(2, 64),
                sym,
            ],
        )
        .expect_err("fallback");
    assert!(matches!(
        err,
        SyscallError::SymbolicArgument(_) | SyscallError::Other(_)
    ));
}

#[test]
fn pwrite64_overwrites_at_offset_without_moving_position() {
    let mut state = fresh_state();
    state.file_system().open_with_content(
        "out".to_string(),
        FdFlags::ReadWrite,
        b"AAAAAA".to_vec(),
    );
    state.map_memory_data(0x3000, b"xy", Permission::RWX);
    // pwrite64(fd=3, buf=0x3000, nbyte=2, offset=2) -> "AAxyAA"
    let out = NativePwrite64Syscall
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x3000, 64),
                RustBV::concrete(2, 64),
                RustBV::concrete(2, 64),
            ],
        )
        .expect("ok");
    match out {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 2),
        _ => panic!("expected Continue"),
    }
    assert_eq!(state.file_system_ref().fd_content(3), b"AAxyAA");
    // Position untouched: a read still starts at byte 0.
    assert_eq!(state.file_system().read(3, 2), b"AA");
}

#[test]
fn pwrite64_extends_past_eof() {
    let mut state = fresh_state();
    state
        .file_system()
        .open_with_content("out".to_string(), FdFlags::ReadWrite, b"ab".to_vec());
    state.map_memory_data(0x3000, b"Z", Permission::RWX);
    // offset 4 is past EOF (len 2): zero-fill gap, write 'Z' at index 4.
    NativePwrite64Syscall
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x3000, 64),
                RustBV::concrete(1, 64),
                RustBV::concrete(4, 64),
            ],
        )
        .expect("ok");
    assert_eq!(state.file_system_ref().fd_content(3), b"ab\0\0Z");
}

#[test]
fn pwrite64_unknown_fd_falls_back() {
    let mut state = fresh_state();
    state.map_memory_data(0x3000, b"x", Permission::RWX);
    let err = NativePwrite64Syscall
        .call(
            &mut state,
            &[
                RustBV::concrete(9, 64),
                RustBV::concrete(0x3000, 64),
                RustBV::concrete(1, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("fallback");
    assert!(matches!(err, SyscallError::Other(_)));
}

/// Register `n` symbolic bytes for `path` and open it. Returns the fd.
fn open_registered_sym_file(state: &mut RustSimState, path: &str, n: usize) -> u32 {
    let bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        (0..n)
            .map(|i| RustBV::symbolic(&ctx, format!("fdiofile_{i}"), 8))
            .collect()
    };
    state.file_system().register_file_content(path, bytes);
    state
        .file_system()
        .open(path.to_string(), FdFlags::ReadOnly)
}

#[test]
fn readv_content_sym_scatters_bvs() {
    // angr-0xyq2 Phase 2: 3 symbolic bytes into two 2-byte segments — the
    // first fills, the second gets the 1-byte remainder (EOF stops there).
    let mut state = fresh_state();
    let fd = open_registered_sym_file(&mut state, "/tmp/flag", 3);
    write_iovec_array(&mut state, 0x2100, &[(0x2200, 2), (0x2300, 2)]);

    let outcome = NativeReadvSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x2100, 64),
                RustBV::concrete(2, 64),
            ],
        )
        .expect("served natively, no fallback");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 3),
        other => panic!("expected Continue, got {other:?}"),
    }
    for addr in [0x2200u64, 0x2201, 0x2300] {
        let byte = state.memory_load(addr, 1).expect("loaded");
        assert!(
            byte.as_u64().is_none(),
            "byte at {addr:#x} should be symbolic"
        );
    }
    assert_eq!(state.file_system_ref().fd_info(fd).unwrap().1, 3);
}

#[test]
fn pread64_content_sym_reads_at_offset_without_moving_position() {
    let mut state = fresh_state();
    let fd = open_registered_sym_file(&mut state, "/tmp/flag", 4);

    let outcome = NativePread64Syscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x2200, 64),
                RustBV::concrete(8, 64), // clamped to the 3 bytes after offset 1
                RustBV::concrete(1, 64),
            ],
        )
        .expect("served natively, no fallback");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 3),
        other => panic!("expected Continue, got {other:?}"),
    }
    for i in 0..3u64 {
        let byte = state.memory_load(0x2200 + i, 1).expect("loaded");
        assert!(byte.as_u64().is_none(), "byte {i} should be symbolic");
    }
    assert_eq!(
        state.file_system_ref().fd_info(fd).unwrap().1,
        0,
        "pread does not move the position"
    );
}

#[test]
fn writev_content_sym_demotes_and_falls_back() {
    // angr-0xyq2 Phase 2 write-demotion, writev leg.
    let mut state = fresh_state();
    let fd = open_registered_sym_file(&mut state, "/tmp/flag", 3);
    state.map_memory_data(0x3000, b"hi", Permission::RWX);
    write_iovec_array(&mut state, 0x2100, &[(0x3000, 2)]);

    let err = NativeWritevSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x2100, 64),
                RustBV::concrete(1, 64),
            ],
        )
        .expect_err("demoted write must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
    let fs = state.file_system_ref();
    assert!(fs.fd_content_sym(fd).is_none(), "content_sym cleared");
    assert!(
        fs.file_content_for_path("/tmp/flag").is_none(),
        "registry gone"
    );
    assert_eq!(fs.fd_content(fd), b"", "nothing written natively");
}

#[test]
fn pwrite64_content_sym_demotes_and_falls_back() {
    // angr-0xyq2 Phase 2 write-demotion, pwrite64 leg.
    let mut state = fresh_state();
    let fd = open_registered_sym_file(&mut state, "/tmp/flag", 3);
    state.map_memory_data(0x3000, b"hi", Permission::RWX);

    let err = NativePwrite64Syscall
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x3000, 64),
                RustBV::concrete(2, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("demoted write must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
    let fs = state.file_system_ref();
    assert!(fs.fd_content_sym(fd).is_none(), "content_sym cleared");
    assert!(
        fs.file_content_for_path("/tmp/flag").is_none(),
        "registry gone"
    );
    assert_eq!(fs.fd_content(fd), b"", "nothing written natively");
}

/// angr-0xyq2 B3: readv bumps `symfile_reads_native` once per guest call,
/// not once per iovec segment (read_sym runs per segment). The counter is
/// process-global and other tests may bump it concurrently, so a mismatch
/// is retried on a fresh state — concurrent noise passes on a later
/// attempt, while a deterministic per-segment double-bump fails every
/// attempt.
#[test]
fn readv_content_sym_bumps_counter_once_per_call() {
    let mut observed = Vec::new();
    for attempt in 0..3 {
        let mut state = fresh_state();
        let fd = open_registered_sym_file(&mut state, "/tmp/flag", 4);
        write_iovec_array(&mut state, 0x2100, &[(0x2200, 2), (0x2300, 2)]);

        let before = crate::symbolic::get_solver_stats()["symfile_reads_native"];
        let outcome = NativeReadvSyscall
            .call(
                &mut state,
                &[
                    RustBV::concrete(fd as u128, 64),
                    RustBV::concrete(0x2100, 64),
                    RustBV::concrete(2, 64),
                ],
            )
            .expect("served natively");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 4, "both segments served"),
            other => panic!("expected Continue, got {other:?}"),
        }
        let delta = crate::symbolic::get_solver_stats()["symfile_reads_native"] - before;
        if delta == 1 {
            return; // exactly one bump for a two-segment readv
        }
        observed.push((attempt, delta));
    }
    panic!("readv must bump symfile_reads_native once per call; observed deltas {observed:?}");
}
