//! Tests for the fwrite / fwrite_unlocked SimProcedure (split out of
//! `stdio_tests.rs` alongside `fwrite.rs`).
use super::*;
use crate::memory::Permission;
use crate::procedures::test_util::{assert_over_limit, open_registered_sym_file, setup_file_struct};

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
fn fwrite_total_over_limit_falls_back() {
    let mut state = RustSimState::new("amd64").unwrap();
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, 1);

    let err = NativeFwrite
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(1, 64),
                RustBV::concrete(MAX_FWRITE_SIZE as u128 + 1, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .expect_err("must fall back");
    assert_over_limit(&err, "fwrite byte count");
    // The cap sits before the gather, so nothing reached stdout and Python's
    // fallback fwrite stays the single authoritative one.
    assert!(state.stdout_buffer().is_empty());
}

#[test]
fn fwrite_total_at_limit_is_native() {
    // Companion to `fwrite_total_over_limit_falls_back`: this is what pins the
    // check as `>` rather than `>=`. The over-limit half alone passes either
    // way.
    let mut state = RustSimState::new("amd64").unwrap();
    let file_ptr = 0x10000u64;
    setup_file_struct(&mut state, file_ptr, 1);
    state.map_memory_data(0x1000, &vec![b'x'; MAX_FWRITE_SIZE as usize], Permission::RWX);

    let ret = NativeFwrite
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(1, 64),
                RustBV::concrete(MAX_FWRITE_SIZE as u128, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .expect("a byte count exactly at the limit stays native");
    assert_eq!(ret.unwrap().as_u64(), Some(MAX_FWRITE_SIZE));
    assert_eq!(state.stdout_buffer().len(), MAX_FWRITE_SIZE as usize);
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
