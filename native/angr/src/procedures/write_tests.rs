// Tests extracted from write.rs (see #[path] attr in parent module).
// Split per rust-mod-tests-sibling-extraction recipe.
use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

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

#[test]
fn test_write_content_sym_demotes_and_falls_back() {
    // angr-0xyq2 Phase 2 write-demotion: a native write to a bounded
    // symbolic file clears content_sym on ALL fds for the path, drops the
    // registry entry, and bounces to Python — the write itself and all
    // subsequent I/O on the file are Python-owned.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hi", Permission::RWX);
    let bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        (0..3)
            .map(|i| RustBV::symbolic(&ctx, format!("wdem_{i}"), 8))
            .collect()
    };
    state
        .file_system()
        .register_file_content("/tmp/flag", bytes);
    let fd1 = state
        .file_system()
        .open("/tmp/flag".to_string(), crate::state::FdFlags::ReadWrite);
    let fd2 = state
        .file_system()
        .open("/tmp/flag".to_string(), crate::state::FdFlags::ReadOnly);

    let result = NativeWrite.call(
        &mut state,
        &[
            RustBV::concrete(fd1 as u128, 64),
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(2, 64),
        ],
    );
    assert!(matches!(result, Err(ProcedureError::Other(_))));
    {
        let fs = state.file_system_ref();
        assert!(fs.fd_content_sym(fd1).is_none(), "written fd demoted");
        assert!(fs.fd_content_sym(fd2).is_none(), "sibling fd demoted");
        assert!(
            fs.file_content_for_path("/tmp/flag").is_none(),
            "registry entry gone"
        );
        assert_eq!(fs.fd_content(fd1), b"", "nothing written natively");
    }

    // Subsequent read on the demoted fd stays a Python fallback: the file is
    // Python-owned from here on, so the contentless-fd mint of angr-gorvf.15
    // must NOT kick in and invent bytes over Python's written content
    // (angr-8kk32).
    let read_result = crate::procedures::read::NativeRead.call(
        &mut state,
        &[
            RustBV::concrete(fd2 as u128, 64),
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(2, 64),
        ],
    );
    assert!(
        read_result.is_err(),
        "demoted file reads fall back to Python"
    );
}

/// angr-0xyq2 A3: write(fd, buf, 0) is a POSIX no-op — it must return 0
/// natively WITHOUT demoting the fd's bounded symbolic content.
#[test]
fn test_zero_length_write_does_not_demote() {
    let mut state = RustSimState::new("amd64").unwrap();
    let bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        (0..3)
            .map(|i| RustBV::symbolic(&ctx, format!("wzero_{i}"), 8))
            .collect()
    };
    state
        .file_system()
        .register_file_content("/tmp/flag", bytes);
    let fd = state
        .file_system()
        .open("/tmp/flag".to_string(), crate::state::FdFlags::ReadWrite);

    let result = NativeWrite
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x1000, 64), // buf may even be unmapped
                RustBV::concrete(0, 64),
            ],
        )
        .expect("zero-length write served natively");
    assert_eq!(result.unwrap().as_u64(), Some(0));
    let fs = state.file_system_ref();
    assert!(fs.fd_content_sym(fd).is_some(), "content_sym intact");
    assert!(
        fs.file_content_for_path("/tmp/flag").is_some(),
        "registry intact"
    );
}
