// Tests for puts.rs (NativePuts / NativePutchar / NativeFputc).
// Extracted from the parent module; see the `#[path]` attr in puts.rs.
use super::*;
use crate::memory::Permission;
use crate::procedures::{NativeSimProcedure, ProcedureError};
use crate::state::RustSimState;

#[test]
fn test_puts_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

    let result = NativePuts
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(6)); // 5 + newline
    assert_eq!(state.stdout_buffer(), b"hello\n");
}

#[test]
fn test_puts_empty_string() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"\x00", Permission::RWX);

    let result = NativePuts
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(1)); // just newline
    assert_eq!(state.stdout_buffer(), b"\n");
}

#[test]
fn test_puts_multiple_calls() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
    state.map_memory_data(0x2000, b"def\x00", Permission::RWX);

    NativePuts
        .call(&mut state, &[RustBV::concrete(0x1000, 64)])
        .unwrap();
    NativePuts
        .call(&mut state, &[RustBV::concrete(0x2000, 64)])
        .unwrap();
    assert_eq!(state.stdout_buffer(), b"abc\ndef\n");
}

#[test]
fn test_puts_symbolic_addr() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "addr", 64);
    drop(ctx);
    let result = NativePuts.call(&mut state, &[sym]);
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_putchar_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativePutchar
        .call(&mut state, &[RustBV::concrete(b'A' as u128, 32)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(b'A' as u64));
    assert_eq!(state.stdout_buffer(), b"A");
}

#[test]
fn test_putchar_multiple() {
    let mut state = RustSimState::new("amd64").unwrap();
    NativePutchar
        .call(&mut state, &[RustBV::concrete(b'H' as u128, 32)])
        .unwrap();
    NativePutchar
        .call(&mut state, &[RustBV::concrete(b'i' as u128, 32)])
        .unwrap();
    assert_eq!(state.stdout_buffer(), b"Hi");
}

#[test]
fn test_putchar_symbolic() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "c", 32);
    drop(ctx);
    let result = NativePutchar.call(&mut state, &[sym]);
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

/// Map a FILE struct at `file_ptr` whose `_fileno` field holds `fd`. The
/// AMD64 `_IO_FILE._fileno` byte offset is 112 (see `fileops::io_file_for_arch`).
fn setup_file_struct(state: &mut RustSimState, file_ptr: u64, fd: i32) {
    const AMD64_FD_OFFSET: u64 = 112;
    state.map_memory_data(file_ptr & !0xfff, &vec![0u8; 0x4000], Permission::RWX);
    let fd_bv = RustBV::concrete(fd as u32 as u128, 32);
    state
        .memory_store(file_ptr + AMD64_FD_OFFSET, fd_bv)
        .unwrap();
}

#[test]
fn test_fputc_basic() {
    // FILE* with _fileno=1 (stdout) — fputc resolves the fd, not assumes stdout.
    let mut state = RustSimState::new("amd64").unwrap();
    let file_ptr = 0x5000;
    setup_file_struct(&mut state, file_ptr, 1);
    let result = NativeFputc
        .call(
            &mut state,
            &[
                RustBV::concrete(b'X' as u128, 32),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(b'X' as u64));
    assert_eq!(state.stdout_buffer(), b"X");
}

#[test]
fn test_fputc_writes_to_stderr_fd() {
    // fputc(c, stderr) must land on fd 2, not stdout — the bug this fixes.
    let mut state = RustSimState::new("amd64").unwrap();
    let file_ptr = 0x5000;
    setup_file_struct(&mut state, file_ptr, 2);
    NativeFputc
        .call(
            &mut state,
            &[
                RustBV::concrete(b'E' as u128, 32),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .unwrap();
    assert_eq!(state.fd_buffer(2), b"E");
    assert!(state.stdout_buffer().is_empty());
}

#[test]
fn test_putc_writes_to_arbitrary_fd() {
    // `putc` has no struct of its own — it is an alias of NativeFputc — so go
    // through the registry to prove the dispatch name still reaches the impl.
    let registry = crate::procedures::NativeProcedureRegistry::new();
    let putc = registry.get("putc").expect("putc should resolve").clone();
    let mut state = RustSimState::new("amd64").unwrap();
    let file_ptr = 0x5000;
    setup_file_struct(&mut state, file_ptr, 7);
    putc.call(
        &mut state,
        &[
            RustBV::concrete(b'q' as u128, 32),
            RustBV::concrete(file_ptr as u128, 64),
        ],
    )
    .unwrap();
    assert_eq!(state.fd_buffer(7), b"q");
}

#[test]
fn test_fputc_negative_fd_returns_minus_one() {
    let mut state = RustSimState::new("amd64").unwrap();
    let file_ptr = 0x5000;
    setup_file_struct(&mut state, file_ptr, -1);
    let result = NativeFputc
        .call(
            &mut state,
            &[
                RustBV::concrete(b'Z' as u128, 32),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0xFFFFFFFF));
    assert!(state.stdout_buffer().is_empty());
}

#[test]
fn test_fputc_symbolic_stream_falls_back() {
    // A symbolic FILE* can't be resolved to an fd — must error so the Python
    // proc handles it (was previously serviced as stdout via the `bv` arg).
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "stream", 64);
    drop(ctx);
    let result = NativeFputc.call(&mut state, &[RustBV::concrete(b'A' as u128, 32), sym]);
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

/// angr-0xyq2 Phase 2 (A2): fputc had no per-site demote guard — the write
/// choke point (`FileSystem::write`) must demote the fd's bounded symbolic
/// content and bounce to Python, writing nothing natively.
#[test]
fn test_fputc_content_sym_fd_demotes_and_falls_back() {
    let mut state = RustSimState::new("amd64").unwrap();
    let bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        (0..3)
            .map(|i| RustBV::symbolic(&ctx, format!("fputcfile_{i}"), 8))
            .collect()
    };
    state
        .file_system()
        .register_file_content("/tmp/flag", bytes);
    let fd = state
        .file_system()
        .open("/tmp/flag".to_string(), crate::state::FdFlags::ReadWrite);
    let file_ptr = 0x5000;
    setup_file_struct(&mut state, file_ptr, fd as i32);

    let result = NativeFputc.call(
        &mut state,
        &[
            RustBV::concrete(b'X' as u128, 32),
            RustBV::concrete(file_ptr as u128, 64),
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
