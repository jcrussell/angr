//! The filesystem surface that only exists at the [`RustSimState`] level: the
//! fs's participation in `fork()`, and the `write_stdout`/`fd_buffer`
//! backward-compat wrappers that predate `state.file_system()`.
//!
//! Everything reachable from a bare `FileSystem::default()` lives in
//! `state/filesystem/tests.rs` instead — this file was carved out of the
//! monolithic `state_tests.rs` as `filesystem_basic` (angr-c7xno.69) and
//! carried a duplicate of that file's CRUD/fd-table coverage until
//! angr-03vl4.56 folded the duplicates back and renamed it for the boundary
//! it actually holds.

use super::super::*;

#[test]
fn test_filesystem_fork_isolation() {
    let mut state = RustSimState::new("amd64").unwrap();
    state
        .file_system()
        .open("test.txt".to_string(), FdFlags::ReadOnly)
        .expect("fd space is not exhausted in tests");
    assert!(state.file_system_ref().is_open(3));

    let mut forked = state.fork();
    // Close in forked state
    forked.file_system().close(3);
    assert!(!forked.file_system_ref().is_open(3));
    // Original should still be open
    assert!(state.file_system_ref().is_open(3));
}

#[test]
fn test_filesystem_backward_compat() {
    // fd_buffer/write_fd should still work through FileSystem
    let mut state = RustSimState::new("amd64").unwrap();
    assert!(state.write_stdout(b"hello"));
    assert_eq!(state.stdout_buffer(), b"hello");
    assert!(state.has_stdout());

    assert!(state.write_fd(2, b"err"));
    assert_eq!(state.fd_buffer(2), b"err");
}
