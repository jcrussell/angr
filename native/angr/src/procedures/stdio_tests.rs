//! Tests for the fflush / setvbuf / setbuf / feof / ferror / fputs stdio
//! SimProcedures (extracted from stdio.rs). fwrite's live in `fwrite_tests.rs`.
use super::*;
use crate::memory::Permission;
use crate::procedures::test_util::{open_registered_sym_file, setup_file_struct};

#[test]
fn test_fflush_returns_zero() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeFflush
        .call(&mut state, &[RustBV::concrete(0x12345678, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_setvbuf_returns_zero() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeSetvbuf
        .call(
            &mut state,
            &[
                RustBV::concrete(0x10000, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_setbuf_returns_void_no_register() {
    // setbuf is a void no-op: Ok(None) means no return register is written.
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeSetbuf
        .call(
            &mut state,
            &[RustBV::concrete(0x10000, 64), RustBV::concrete(0, 64)],
        )
        .unwrap();
    assert!(result.is_none(), "setbuf must not set a return value");
}

#[test]
fn test_setbuf_symbolic_args_no_fallback() {
    // Python ignores both operands; symbolic stream/buf must NOT fall back.
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let stream = RustBV::symbolic(&ctx, "stream", 64);
    let buf = RustBV::symbolic(&ctx, "buf", 64);
    drop(ctx);
    let result = NativeSetbuf.call(&mut state, &[stream, buf]).unwrap();
    assert!(result.is_none());
}

#[test]
fn test_setbuf_name_and_arity() {
    assert_eq!(NativeSetbuf.name(), "setbuf");
    assert_eq!(NativeSetbuf.num_args(), 2);
}

// --- feof / ferror / fputs ---

#[test]
fn test_feof_at_start_of_empty_fd_is_eof() {
    // An open-but-empty fd: position 0 >= content_len 0 → EOF (1).
    let mut state = RustSimState::new("amd64").unwrap();
    let fd = state
        .file_system()
        .open("empty.txt".to_string(), crate::state::FdFlags::ReadOnly).expect("fd space is not exhausted in tests");
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, fd as i32);

    let result = NativeFeof
        .call(&mut state, &[RustBV::concrete(file_ptr as u128, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(1));
}

#[test]
fn test_feof_with_content_available_returns_zero() {
    // Position 0 with non-empty content → not EOF (0).
    let mut state = RustSimState::new("amd64").unwrap();
    let fd = state.file_system().open_with_content(
        "data.txt".to_string(),
        crate::state::FdFlags::ReadOnly,
        b"hello".to_vec(),
    ).expect("fd space is not exhausted in tests");
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, fd as i32);

    let result = NativeFeof
        .call(&mut state, &[RustBV::concrete(file_ptr as u128, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_feof_after_consuming_all_bytes_is_eof() {
    // Read everything, position == content_len → EOF.
    let mut state = RustSimState::new("amd64").unwrap();
    let fd = state.file_system().open_with_content(
        "data.txt".to_string(),
        crate::state::FdFlags::ReadOnly,
        b"abc".to_vec(),
    ).expect("fd space is not exhausted in tests");
    let _ = state.file_system().read(fd, 3);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, fd as i32);

    let result = NativeFeof
        .call(&mut state, &[RustBV::concrete(file_ptr as u128, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(1));
}

#[test]
fn test_feof_content_sym_uses_effective_len() {
    // angr-0xyq2 Phase 2: a bounded symbolic file (empty concrete buffer,
    // 3-byte content_sym) is not at EOF until the position reaches the
    // symbolic length.
    let mut state = RustSimState::new("amd64").unwrap();
    let fd = open_registered_sym_file(&mut state, "/tmp/flag", 3);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, fd as i32);

    let result = NativeFeof
        .call(&mut state, &[RustBV::concrete(file_ptr as u128, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0), "pos 0 < 3: not EOF");

    // Consume all symbolic bytes; pos == symbolic length → EOF.
    let served = state.file_system().read_sym(fd, 3).unwrap();
    assert_eq!(served.len(), 3);
    let result = NativeFeof
        .call(&mut state, &[RustBV::concrete(file_ptr as u128, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(1), "pos 3 >= 3: EOF");
}

#[test]
fn test_fputs_content_sym_demotes_and_falls_back() {
    // fputs shares NativeFwrite's demote-then-fallback discipline.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hi\0", Permission::RWX);
    let fd = open_registered_sym_file(&mut state, "/tmp/flag", 3);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, fd as i32);

    let result = NativeFputs.call(
        &mut state,
        &[
            RustBV::concrete(0x1000, 64),
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

/// angr-0xyq2 A3: fputs("") writes nothing — success without demotion (the
/// choke point's empty-write no-op; there is deliberately no pre-scan gate
/// in fputs).
#[test]
fn test_fputs_empty_string_does_not_demote() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"\0", Permission::RWX);
    let fd = open_registered_sym_file(&mut state, "/tmp/flag", 3);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, fd as i32);

    let result = NativeFputs
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .expect("empty fputs served natively");
    assert_eq!(result.unwrap().as_u64(), Some(1));
    let fs = state.file_system_ref();
    assert!(fs.fd_content_sym(fd).is_some(), "content_sym intact");
    assert!(
        fs.file_content_for_path("/tmp/flag").is_some(),
        "registry intact"
    );
}

#[test]
fn test_feof_negative_fd_returns_zero() {
    let mut state = RustSimState::new("amd64").unwrap();
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, -1);
    let result = NativeFeof
        .call(&mut state, &[RustBV::concrete(file_ptr as u128, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_ferror_always_zero() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeFerror
        .call(&mut state, &[RustBV::concrete(0xdeadbeef, 64)])
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_fputs_writes_to_stdout() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello\0", Permission::RWX);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, 1);

    let result = NativeFputs
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(1));
    assert_eq!(state.stdout_buffer(), b"hello");
}

#[test]
fn test_fputs_writes_to_stderr() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"oops\0", Permission::RWX);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, 2);

    let result = NativeFputs
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(1));
    assert_eq!(state.fd_buffer(2), b"oops");
}

#[test]
fn test_fputs_negative_fd_returns_minus_one() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abc\0", Permission::RWX);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, -1);

    let result = NativeFputs
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .unwrap();
    // -1 as size_t on amd64
    assert_eq!(result.unwrap().as_u64(), Some(u64::MAX));
}

#[test]
fn test_fputs_empty_string_writes_nothing_returns_one() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"\0", Permission::RWX);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, 1);

    let result = NativeFputs
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(1));
    assert_eq!(state.stdout_buffer(), b"");
}
