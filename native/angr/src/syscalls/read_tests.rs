//! Tests for `read.rs` — the `read(2)` syscall handler.

use super::*;
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

fn fresh_state_with_buf() -> RustSimState {
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    state
}

#[test]
fn handler_metadata() {
    let h = NativeReadSyscall;
    assert_eq!(h.name(), "read");
    assert_eq!(h.num_args(), 3);
}

#[test]
fn read_stdin_returns_count_and_writes_symbolic_bytes() {
    let h = NativeReadSyscall;
    let mut state = fresh_state_with_buf();

    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(4, 64),
            ],
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 4),
        _ => panic!("expected Continue"),
    }
    for i in 0..4u64 {
        let byte = state.memory_load(0x2000 + i, 1).expect("loaded");
        assert!(byte.as_u64().is_none(), "byte {} should be symbolic", i);
    }
}

#[test]
fn read_zero_count_returns_zero_without_writing() {
    let h = NativeReadSyscall;
    let mut state = fresh_state_with_buf();
    // Pre-fill so we can detect any write.
    state.map_memory_data(0x2000, b"abcd", Permission::RWX);

    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        _ => panic!("expected Continue"),
    }
    // Pre-fill must be intact.
    for (i, &b) in b"abcd".iter().enumerate() {
        let byte = state.memory_load(0x2000 + i as u64, 1).expect("loaded");
        assert_eq!(byte.as_u64(), Some(b as u64));
    }
}

#[test]
fn read_unknown_fd_falls_back() {
    // fd=3 is not open in the FileSystem → fall back.
    let h = NativeReadSyscall;
    let mut state = fresh_state_with_buf();
    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(4, 64),
            ],
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn read_user_fd_with_content_serves_natively() {
    // Open fd=3 with concrete content; native read should serve from FS.
    let h = NativeReadSyscall;
    let mut state = fresh_state_with_buf();
    state.file_system().open_with_content(
        "in.bin".to_string(),
        crate::state::FdFlags::ReadOnly,
        b"abcdef".to_vec(),
    );

    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(3, 64),
            ],
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 3),
        _ => panic!("expected Continue"),
    }
    for (i, &want) in b"abc".iter().enumerate() {
        let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
        assert_eq!(byte.as_u64(), Some(want as u64));
    }
    let info = state.file_system_ref().fd_info(3).unwrap();
    assert_eq!(info.1, 3);
}

#[test]
fn read_user_fd_eof_returns_zero() {
    let h = NativeReadSyscall;
    let mut state = fresh_state_with_buf();
    state.file_system().open_with_content(
        "in.bin".to_string(),
        crate::state::FdFlags::ReadOnly,
        b"ab".to_vec(),
    );

    // Drain 2 bytes.
    h.call(
        &mut state,
        &[
            RustBV::concrete(3, 64),
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(2, 64),
        ],
    )
    .expect("ok");

    // Subsequent read returns 0 (EOF), without falling back.
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x2010, 64),
                RustBV::concrete(4, 64),
            ],
        )
        .expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        _ => panic!("expected Continue"),
    }
}

#[test]
fn read_empty_content_fd_falls_back() {
    // Open without content → defer to Python so the symbolic-file model
    // can supply bytes.
    let h = NativeReadSyscall;
    let mut state = fresh_state_with_buf();
    state
        .file_system()
        .open("in.bin".to_string(), crate::state::FdFlags::ReadOnly);
    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(4, 64),
            ],
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn read_closed_fd_falls_back() {
    let h = NativeReadSyscall;
    let mut state = fresh_state_with_buf();
    state.file_system().open_with_content(
        "in.bin".to_string(),
        crate::state::FdFlags::ReadOnly,
        b"x".to_vec(),
    );
    assert!(state.file_system().close(3));
    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(1, 64),
            ],
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn read_too_large_falls_back() {
    let h = NativeReadSyscall;
    let mut state = fresh_state_with_buf();
    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(5000, 64),
            ],
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn symbolic_fd_falls_back() {
    let h = NativeReadSyscall;
    let mut state = fresh_state_with_buf();
    let ctx = SymContext::new();
    let sym_fd = RustBV::symbolic(&ctx, "fd", 64);
    let err = h
        .call(
            &mut state,
            &[
                sym_fd,
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(4, 64),
            ],
        )
        .expect_err("must fall back");
    match err {
        SyscallError::SymbolicArgument(msg) => {
            assert!(msg.contains("fd"), "message should name fd, got {msg:?}");
        }
        other => panic!("expected SymbolicArgument, got {other:?}"),
    }
}

#[test]
fn symbolic_buf_falls_back() {
    let h = NativeReadSyscall;
    let mut state = fresh_state_with_buf();
    let ctx = SymContext::new();
    let sym_buf = RustBV::symbolic(&ctx, "buf", 64);
    let err = h
        .call(
            &mut state,
            &[RustBV::concrete(0, 64), sym_buf, RustBV::concrete(4, 64)],
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

#[test]
fn symbolic_count_falls_back() {
    let h = NativeReadSyscall;
    let mut state = fresh_state_with_buf();
    let ctx = SymContext::new();
    let sym_count = RustBV::symbolic(&ctx, "count", 64);
    let err = h
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(0x2000, 64),
                sym_count,
            ],
        )
        .expect_err("must fall back");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}
