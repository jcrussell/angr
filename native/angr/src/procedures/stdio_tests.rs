//! Tests for the fwrite/fflush/setvbuf stdio SimProcedures (extracted from stdio.rs).
use super::*;
use crate::memory::Permission;

fn setup_file_struct(state: &mut RustSimState, file_ptr: u64, fd: i32) {
    let arch_name = state.arch().name();
    let (off, _size) =
        crate::procedures::fileops::io_file_for_arch(arch_name).expect("test arch supported");
    // Map enough room for the FILE struct + buf.
    state.map_memory_data(file_ptr & !0xfff, &vec![0u8; 0x4000], Permission::RWX);
    let fd_bv = RustBV::concrete(fd as u32 as u128, 32);
    state.memory_store(file_ptr + off, fd_bv).unwrap();
}

#[test]
fn test_fwrite_stdout() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello world", Permission::RWX);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, 1);

    let result = NativeFwrite
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(1, 64),
                RustBV::concrete(5, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(5));
    assert_eq!(state.stdout_buffer(), b"hello");
}

#[test]
fn test_fwrite_stderr_nmemb_times_size() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abcdefgh", Permission::RWX);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, 2);

    let result = NativeFwrite
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(2, 64),
                RustBV::concrete(3, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(6));
    assert_eq!(state.fd_buffer(2), b"abcdef");
}

#[test]
fn test_fwrite_arbitrary_fd_writes_to_buffer() {
    // fd 5 (a non-stdout/stderr fd) is serviced inline via write_fd, matching
    // NativeFputs and Python fwrite's `simfd.write` for an arbitrary fd —
    // no Python fallback.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abc", Permission::RWX);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, 5);

    let result = NativeFwrite
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(1, 64),
                RustBV::concrete(3, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(3));
    assert_eq!(state.fd_buffer(5), b"abc");
}

#[test]
fn test_fwrite_negative_fd_returns_minus_one() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abc", Permission::RWX);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, -1);

    let result = NativeFwrite
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(1, 64),
                RustBV::concrete(3, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .unwrap();

    // -1 as size_t on amd64
    assert_eq!(result.unwrap().as_u64(), Some(u64::MAX));
}

#[test]
fn test_fwrite_too_large() {
    let mut state = RustSimState::new("amd64").unwrap();
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, 1);

    let result = NativeFwrite.call(
        &mut state,
        &[
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(1, 64),
            RustBV::concrete(8192, 64),
            RustBV::concrete(file_ptr as u128, 64),
        ],
    );
    assert!(result.is_err());
}

/// `size * nmemb` is guest-controlled, so both of `NativeFwrite`'s
/// multiplications (the zero-total no-op gate and the byte-count `total`)
/// saturate. `size = 2^63, nmemb = 2` is the shape that distinguishes them
/// from a plain `*`: the wrapping product is exactly 0, which would take the
/// zero-total early return and report a *successful* no-op write for a
/// request of 2^64 bytes. Saturation pins the product at `u64::MAX` so the
/// oversize gate rejects to Python instead (and, under
/// `--profile release-checked`, a plain `*` would panic here).
#[test]
fn test_fwrite_size_times_nmemb_saturates_instead_of_wrapping() {
    let mut state = RustSimState::new("amd64").unwrap();
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, 1);

    let result = NativeFwrite.call(
        &mut state,
        &[
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(1u128 << 63, 64), // size
            RustBV::concrete(2, 64),           // nmemb
            RustBV::concrete(file_ptr as u128, 64),
        ],
    );
    match result {
        Err(ProcedureError::Other(msg)) => assert!(
            msg.contains(&u64::MAX.to_string()),
            "saturated total should appear in the rejection: {msg}"
        ),
        other => panic!("expected oversize rejection, got {other:?}"),
    }
    assert!(
        state.stdout_buffer().is_empty(),
        "nothing written for a rejected fwrite"
    );
}

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

/// Register `n` symbolic bytes for `path` and open it. Returns the fd.
fn open_registered_sym_file(state: &mut RustSimState, path: &str, n: usize) -> u32 {
    let bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        (0..n)
            .map(|i| RustBV::symbolic(&ctx, format!("stdiofile_{i}"), 8))
            .collect()
    };
    state.file_system().register_file_content(path, bytes);
    state
        .file_system()
        .open(path.to_string(), crate::state::FdFlags::ReadOnly).expect("fd space is not exhausted in tests")
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
fn test_fwrite_content_sym_demotes_and_falls_back() {
    // angr-0xyq2 Phase 2 write-demotion: fwrite to a bounded symbolic file
    // drops the content (fd + registry) and bounces to Python.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello", Permission::RWX);
    let fd = open_registered_sym_file(&mut state, "/tmp/flag", 3);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, fd as i32);

    let result = NativeFwrite.call(
        &mut state,
        &[
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(1, 64),
            RustBV::concrete(5, 64),
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

/// angr-0xyq2 A3: fwrite(p, 0, 0, f) (or any size*nmemb == 0) is a POSIX
/// no-op — 0 is returned natively WITHOUT demoting the fd's bounded
/// symbolic content.
#[test]
fn test_fwrite_zero_total_does_not_demote() {
    let mut state = RustSimState::new("amd64").unwrap();
    let fd = open_registered_sym_file(&mut state, "/tmp/flag", 3);
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, fd as i32);

    let result = NativeFwrite
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64), // src may even be unmapped
                RustBV::concrete(0, 64),
                RustBV::concrete(7, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .expect("zero-length fwrite served natively");
    assert_eq!(result.unwrap().as_u64(), Some(0));
    let fs = state.file_system_ref();
    assert!(fs.fd_content_sym(fd).is_some(), "content_sym intact");
    assert!(
        fs.file_content_for_path("/tmp/flag").is_some(),
        "registry intact"
    );
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
