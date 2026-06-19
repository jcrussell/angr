// Tests extracted from write.rs (see #[path] attr in parent module).
// Split per rust-mod-tests-sibling-extraction recipe.
use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;

#[test]
fn test_write_stdout() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello", Permission::RWX);

    let result = NativeWrite
        .call(
            &mut state,
            &[
                RustBV::concrete(1, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(5, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(5));
    assert_eq!(state.stdout_buffer(), b"hello");
}

#[test]
fn test_write_stderr() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"err", Permission::RWX);

    let result = NativeWrite
        .call(
            &mut state,
            &[
                RustBV::concrete(2, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(3, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(3));
    assert_eq!(state.fd_buffer(2), b"err");
}

#[test]
fn test_write_unknown_fd_falls_back() {
    // fd=3 is not open in the FileSystem → Native falls back.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"x", Permission::RWX);
    let result = NativeWrite.call(
        &mut state,
        &[
            RustBV::concrete(3, 64),
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(1, 64),
        ],
    );
    assert!(result.is_err());
}

#[test]
fn test_write_stdin_falls_back() {
    // fd=0 is open (stdin) but we refuse to write — fall back so the
    // Python posix model can produce EBADF.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"x", Permission::RWX);
    let result = NativeWrite.call(
        &mut state,
        &[
            RustBV::concrete(0, 64),
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(1, 64),
        ],
    );
    assert!(result.is_err());
    // The native handler must not have appended to stdin's content.
    assert_eq!(state.file_system_ref().fd_content(0), b"");
}

#[test]
fn test_write_user_fd_appends_to_filesystem() {
    // Open fd=3 via NativeOpen path; native write should append into the
    // file's content buffer without entering Python.
    let mut state = RustSimState::new("amd64").unwrap();
    state
        .file_system()
        .open("out.bin".to_string(), crate::state::FdFlags::WriteOnly);
    state.map_memory_data(0x1000, b"hello", Permission::RWX);

    let result = NativeWrite
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(5, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(5));
    assert_eq!(state.file_system_ref().fd_content(3), b"hello");
    // stdout untouched.
    assert_eq!(state.stdout_buffer(), b"");
}

#[test]
fn test_write_closed_fd_falls_back() {
    // open + close → is_open false → fall back.
    let mut state = RustSimState::new("amd64").unwrap();
    state
        .file_system()
        .open("out.bin".to_string(), crate::state::FdFlags::WriteOnly);
    assert!(state.file_system().close(3));
    state.map_memory_data(0x1000, b"x", Permission::RWX);

    let result = NativeWrite.call(
        &mut state,
        &[
            RustBV::concrete(3, 64),
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(1, 64),
        ],
    );
    assert!(result.is_err());
}

#[test]
fn test_write_too_large() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeWrite.call(
        &mut state,
        &[
            RustBV::concrete(1, 64),
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(5000, 64),
        ],
    );
    assert!(result.is_err());
}
