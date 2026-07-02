//! Unit tests for [`super`] — the native `write`/`writev` syscall handlers.
//!
//! Split out of `write.rs` to keep the handler module focused; see the
//! `rust-mod-tests-sibling-extraction` pattern.

use super::*;
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

#[test]
fn handler_metadata() {
    let h = NativeWriteSyscall;
    assert_eq!(h.name(), "write");
    assert_eq!(h.num_args(), 3);
}

#[test]
fn write_stdout_appends_and_returns_count() {
    let h = NativeWriteSyscall;
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state.map_memory_data(0x1000, b"hello", Permission::RWX);

    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(1, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(5, 64),
            ],
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 5),
        _ => panic!("expected Continue"),
    }
    assert_eq!(state.stdout_buffer(), b"hello");
}

#[test]
fn write_stderr_appends_and_returns_count() {
    let h = NativeWriteSyscall;
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state.map_memory_data(0x1000, b"err", Permission::RWX);

    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(2, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(3, 64),
            ],
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 3),
        _ => panic!("expected Continue"),
    }
    assert_eq!(state.fd_buffer(2), b"err");
}

#[test]
fn write_unknown_fd_falls_back() {
    // fd=3 is not open in the FileSystem → Native falls back.
    let h = NativeWriteSyscall;
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state.map_memory_data(0x1000, b"x", Permission::RWX);
    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(1, 64),
            ],
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
    assert_eq!(state.stdout_buffer(), b"");
}

#[test]
fn write_stdin_falls_back() {
    let h = NativeWriteSyscall;
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state.map_memory_data(0x1000, b"x", Permission::RWX);
    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(1, 64),
            ],
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
    // stdin content must be untouched.
    assert_eq!(state.file_system_ref().fd_content(0), b"");
}

#[test]
fn write_user_fd_appends_to_filesystem() {
    let h = NativeWriteSyscall;
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state
        .file_system()
        .open("out.bin".to_string(), crate::state::FdFlags::WriteOnly);
    state.map_memory_data(0x1000, b"hello", Permission::RWX);

    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(5, 64),
            ],
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 5),
        _ => panic!("expected Continue"),
    }
    assert_eq!(state.file_system_ref().fd_content(3), b"hello");
    // stdout untouched.
    assert_eq!(state.stdout_buffer(), b"");
}

#[test]
fn write_closed_user_fd_falls_back() {
    let h = NativeWriteSyscall;
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state
        .file_system()
        .open("out.bin".to_string(), crate::state::FdFlags::WriteOnly);
    assert!(state.file_system().close(3));
    state.map_memory_data(0x1000, b"x", Permission::RWX);

    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(1, 64),
            ],
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn write_too_large_falls_back() {
    let h = NativeWriteSyscall;
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(1, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(5000, 64),
            ],
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn write_symbolic_fd_falls_back() {
    let h = NativeWriteSyscall;
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    let ctx = SymContext::new();
    let sym_fd = RustBV::symbolic(&ctx, "fd", 64);
    let err = h
        .call(
            &mut state,
            &[
                sym_fd,
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(1, 64),
            ],
        )
        .expect_err("must fall back");
    match err {
        SyscallError::SymbolicArgument(msg) => {
            assert!(msg.contains("fd"), "message should name fd, got {msg:?}",)
        }
        other => panic!("expected SymbolicArgument, got {other:?}"),
    }
}

#[test]
fn write_symbolic_byte_falls_back() {
    let h = NativeWriteSyscall;
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    // Map some memory and write a symbolic byte at 0x1000.
    state.map_memory(0x1000, 0x1000, Permission::RWX);
    let ctx = SymContext::new();
    let sym_byte = RustBV::symbolic(&ctx, "b", 8);
    state.memory_store(0x1000, sym_byte).expect("ok");

    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(1, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(1, 64),
            ],
        )
        .expect_err("must fall back");
    match err {
        SyscallError::SymbolicArgument(msg) => assert!(
            msg.contains("buf+0"),
            "message should name the symbolic offset, got {msg:?}",
        ),
        other => panic!("expected SymbolicArgument, got {other:?}"),
    }
    // No partial write to stdout must have occurred.
    assert_eq!(state.stdout_buffer(), b"");
}

#[test]
fn write_content_sym_demotes_and_falls_back() {
    // angr-0xyq2 Phase 2 write-demotion — syscall twin of
    // procedures/write_tests.rs::test_write_content_sym_demotes_and_falls_back.
    let h = NativeWriteSyscall;
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state.map_memory_data(0x1000, b"hi", Permission::RWX);
    let bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        (0..3)
            .map(|i| RustBV::symbolic(&ctx, format!("syswdem_{i}"), 8))
            .collect()
    };
    state
        .file_system()
        .register_file_content("/tmp/flag", bytes);
    let fd = state
        .file_system()
        .open("/tmp/flag".to_string(), crate::state::FdFlags::ReadWrite);

    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(2, 64),
            ],
        )
        .expect_err("demoted write must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
    let fs = state.file_system_ref();
    assert!(fs.fd_content_sym(fd).is_none(), "content_sym cleared");
    assert!(
        fs.file_content_for_path("/tmp/flag").is_none(),
        "registry entry gone"
    );
    assert_eq!(fs.fd_content(fd), b"", "nothing written natively");
}
