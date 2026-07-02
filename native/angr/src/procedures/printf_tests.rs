use super::*;
use crate::memory::Permission;
use crate::procedures::{NativeSimProcedure, ProcedureError};
use crate::state::RustSimState;

/// Map a FILE struct at `file_ptr` whose AMD64 `_IO_FILE._fileno` field (byte
/// offset 112) holds `fd`. Mirrors the helper in `puts_tests.rs`.
fn setup_file_struct(state: &mut RustSimState, file_ptr: u64, fd: i32) {
    const AMD64_FD_OFFSET: u64 = 112;
    state.map_memory_data(file_ptr & !0xfff, &vec![0u8; 0x4000], Permission::RWX);
    let fd_bv = RustBV::concrete(fd as u32 as u128, 32);
    state
        .memory_store(file_ptr + AMD64_FD_OFFSET, fd_bv)
        .unwrap();
}

#[test]
fn test_printf_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello world\x00", Permission::RWX);

    let result = NativePrintf
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(11));
    assert_eq!(state.stdout_buffer(), b"hello world");
}

#[test]
fn test_printf_empty_format() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"\x00", Permission::RWX);

    let result = NativePrintf
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    // Returns 1 for empty format
    assert_eq!(result.unwrap().as_u64(), Some(1));
    assert_eq!(state.stdout_buffer(), b"");
}

#[test]
fn test_printf_symbolic_addr() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "fmt", 64);
    drop(ctx);
    let result = NativePrintf.call(&mut state, &[sym]);
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_fprintf_writes_raw_format_to_fd() {
    // fprintf(stderr, fmt) writes the raw format string to fd 2, not stdout.
    let mut state = RustSimState::new("amd64").unwrap();
    let file_ptr = 0x5000;
    setup_file_struct(&mut state, file_ptr, 2);
    state.map_memory_data(0x1000, b"error: %d\x00", Permission::RWX);

    let result = NativeFprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(file_ptr as u128, 64),
                RustBV::concrete(0x1000, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(9));
    assert_eq!(state.fd_buffer(2), b"error: %d");
    assert!(state.stdout_buffer().is_empty());
}

#[test]
fn test_fprintf_to_stdout_fd() {
    let mut state = RustSimState::new("amd64").unwrap();
    let file_ptr = 0x5000;
    setup_file_struct(&mut state, file_ptr, 1);
    state.map_memory_data(0x1000, b"hi\x00", Permission::RWX);

    NativeFprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(file_ptr as u128, 64),
                RustBV::concrete(0x1000, 64),
            ],
        )
        .unwrap();
    assert_eq!(state.stdout_buffer(), b"hi");
}

#[test]
fn test_fprintf_negative_fd_returns_minus_one() {
    let mut state = RustSimState::new("amd64").unwrap();
    let file_ptr = 0x5000;
    setup_file_struct(&mut state, file_ptr, -1);
    state.map_memory_data(0x1000, b"x\x00", Permission::RWX);

    let result = NativeFprintf
        .call(
            &mut state,
            &[
                RustBV::concrete(file_ptr as u128, 64),
                RustBV::concrete(0x1000, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0xFFFFFFFF));
}

#[test]
fn test_fprintf_symbolic_stream_falls_back() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "stream", 64);
    drop(ctx);
    let result = NativeFprintf.call(&mut state, &[sym, RustBV::concrete(0x1000, 64)]);
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

/// angr-0xyq2 Phase 2 (A2): fprintf had no per-site demote guard — the
/// write choke point (`FileSystem::write`) must demote the fd's bounded
/// symbolic content and bounce to Python, writing nothing natively.
#[test]
fn test_fprintf_content_sym_fd_demotes_and_falls_back() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"log line\0", Permission::RWX);
    let bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        (0..3)
            .map(|i| RustBV::symbolic(&ctx, format!("fprintffile_{i}"), 8))
            .collect()
    };
    state
        .file_system()
        .register_file_content("/tmp/flag", bytes);
    let fd = state
        .file_system()
        .open("/tmp/flag".to_string(), crate::state::FdFlags::ReadWrite);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, fd as i32);

    let result = NativeFprintf.call(
        &mut state,
        &[
            RustBV::concrete(file_ptr as u128, 64),
            RustBV::concrete(0x1000, 64),
        ],
    );
    assert!(matches!(result, Err(ProcedureError::Other(_))));
    let fs = state.file_system_ref();
    assert!(fs.fd_content_sym(fd).is_none(), "content_sym cleared");
    assert!(
        fs.file_content_for_path("/tmp/flag").is_none(),
        "registry gone"
    );
    assert_eq!(fs.fd_content(fd), b"", "no bytes written natively");
}
