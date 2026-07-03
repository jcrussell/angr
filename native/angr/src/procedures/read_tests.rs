//! Tests for the read SimProcedure (extracted from read.rs).
use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;

#[test]
fn test_read_stdin() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let result = NativeRead
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(4, 64),
            ],
        )
        .unwrap();

    // Should return count
    assert_eq!(result.unwrap().as_u64(), Some(4));

    // Each byte should be symbolic
    for i in 0..4u64 {
        let byte = state.memory_load(0x2000 + i, 1).unwrap();
        assert!(byte.as_u64().is_none()); // symbolic
    }
}

#[test]
fn test_read_records_stdin_symbols() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    assert!(!state.has_stdin_symbols());
    let _ = NativeRead
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(3, 64),
            ],
        )
        .unwrap();

    let symbols = state.stdin_symbols();
    assert_eq!(symbols.len(), 3);
    for (_, bits) in symbols {
        assert_eq!(*bits, 8);
    }
}

#[test]
fn test_read_zero_count() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeRead
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_read_unknown_fd_falls_back() {
    // fd=3 is not open in the FileSystem → fall back.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, crate::memory::Permission::RWX);
    let result = NativeRead.call(
        &mut state,
        &[
            RustBV::concrete(3, 64),
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(4, 64),
        ],
    );
    assert!(result.is_err());
}

#[test]
fn test_read_user_fd_with_content_serves_natively() {
    // Open fd=3 with concrete content; native read should serve from FS.
    let mut state = RustSimState::new("amd64").unwrap();
    state.file_system().open_with_content(
        "in.bin".to_string(),
        crate::state::FdFlags::ReadOnly,
        b"abcdef".to_vec(),
    );
    state.map_memory(0x2000, 0x1000, crate::memory::Permission::RWX);

    // Read 3 bytes; should return 3 and copy "abc" to memory.
    let result = NativeRead
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(3, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(3));
    for (i, &want) in b"abc".iter().enumerate() {
        let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
        assert_eq!(byte.as_u64(), Some(want as u64));
    }
    // Position advanced.
    let info = state.file_system_ref().fd_info(3).unwrap();
    assert_eq!(info.1, 3);
}

#[test]
fn test_read_user_fd_eof_returns_zero() {
    // Read past the end of the content returns 0 (Linux EOF semantics).
    let mut state = RustSimState::new("amd64").unwrap();
    state.file_system().open_with_content(
        "in.bin".to_string(),
        crate::state::FdFlags::ReadOnly,
        b"ab".to_vec(),
    );
    state.map_memory(0x2000, 0x1000, crate::memory::Permission::RWX);

    // Drain 2 bytes.
    NativeRead
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(2, 64),
            ],
        )
        .unwrap();
    // Next read: should return 0.
    let result = NativeRead
        .call(
            &mut state,
            &[
                RustBV::concrete(3, 64),
                RustBV::concrete(0x2010, 64),
                RustBV::concrete(4, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_read_empty_content_fd_falls_back() {
    // fd open but no content → defer to Python (symbolic-file model).
    let mut state = RustSimState::new("amd64").unwrap();
    state
        .file_system()
        .open("in.bin".to_string(), crate::state::FdFlags::ReadOnly);
    state.map_memory(0x2000, 0x1000, crate::memory::Permission::RWX);
    let result = NativeRead.call(
        &mut state,
        &[
            RustBV::concrete(3, 64),
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(4, 64),
        ],
    );
    assert!(result.is_err());
}

/// Register mixed bounded symbolic content for `path` (index 1 is a
/// concrete 0x41 entry, the rest symbolic) and open it. Returns the fd.
fn open_registered_sym_file(state: &mut RustSimState, path: &str, n: usize) -> u32 {
    let bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        (0..n)
            .map(|i| {
                if i == 1 {
                    RustBV::concrete(0x41, 8)
                } else {
                    RustBV::symbolic(&ctx, format!("symfile_{i}"), 8)
                }
            })
            .collect()
    };
    state.file_system().register_file_content(path, bytes);
    state
        .file_system()
        .open(path.to_string(), crate::state::FdFlags::ReadOnly)
}

#[test]
fn test_read_content_sym_serves_natively() {
    // angr-0xyq2 Phase 2 counterpart of test_read_empty_content_fd_falls_back:
    // registered symbolic content is served natively (Ok, not a fallback Err).
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, crate::memory::Permission::RWX);
    let fd = open_registered_sym_file(&mut state, "/tmp/flag", 4);

    let result = NativeRead
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(3, 64),
            ],
        )
        .expect("served natively, no fallback");
    assert_eq!(result.unwrap().as_u64(), Some(3));

    // Destination buffer carries the served BVs: symbolic where symbolic,
    // the concrete 0x41 entry at index 1 (mixed content).
    assert!(state.memory_load(0x2000, 1).unwrap().as_u64().is_none());
    assert_eq!(state.memory_load(0x2001, 1).unwrap().as_u64(), Some(0x41));
    assert!(state.memory_load(0x2002, 1).unwrap().as_u64().is_none());
    // Position advanced.
    assert_eq!(state.file_system_ref().fd_info(fd).unwrap().1, 3);
}

#[test]
fn test_read_content_sym_eof_returns_zero() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, crate::memory::Permission::RWX);
    let fd = open_registered_sym_file(&mut state, "/tmp/flag", 2);

    // Drain both bytes, then read again: 0 at EOF (no fallback).
    for (want, buf) in [(2u64, 0x2000u64), (0, 0x2010)] {
        let result = NativeRead
            .call(
                &mut state,
                &[
                    RustBV::concrete(fd as u128, 64),
                    RustBV::concrete(buf as u128, 64),
                    RustBV::concrete(4, 64),
                ],
            )
            .expect("served natively");
        assert_eq!(result.unwrap().as_u64(), Some(want));
    }
}

#[test]
fn test_read_closed_fd_falls_back() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.file_system().open_with_content(
        "in.bin".to_string(),
        crate::state::FdFlags::ReadOnly,
        b"x".to_vec(),
    );
    assert!(state.file_system().close(3));
    state.map_memory(0x2000, 0x1000, crate::memory::Permission::RWX);
    let result = NativeRead.call(
        &mut state,
        &[
            RustBV::concrete(3, 64),
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(1, 64),
        ],
    );
    assert!(result.is_err());
}

#[test]
fn test_read_stdin_short_reads_returns_symbolic_size() {
    // With SHORT_READS, native stdin read returns a symbolic real_size in
    // [0, count] (the lone fork source) rather than the concrete count,
    // mirroring Python's storage/file.py SimPacket short-read path (angr-kf0uy).
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_option("SHORT_READS", true);
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let result = NativeRead
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(4, 64),
            ],
        )
        .unwrap()
        .unwrap();

    assert!(
        result.as_u64().is_none(),
        "short-read stdin read returns symbolic real_size, not the concrete count"
    );
    let ctx = state.solver().borrow();
    assert!(
        ctx.solution(&result, 0),
        "a zero-length short read must be reachable"
    );
    assert!(ctx.solution(&result, 4), "the full read must be reachable");
    assert!(!ctx.solution(&result, 5), "real_size cannot exceed count");
}

#[test]
fn test_read_stdin_default_returns_concrete_count() {
    // Default path (SHORT_READS off): stdin read returns the full concrete
    // count, byte-identical and non-forking.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let result = NativeRead
        .call(
            &mut state,
            &[
                RustBV::concrete(0, 64),
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(4, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(4));
}

#[test]
fn test_read_too_large() {
    let mut state = RustSimState::new("amd64").unwrap();
    let result = NativeRead.call(
        &mut state,
        &[
            RustBV::concrete(0, 64),
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(5000, 64),
        ],
    );
    assert!(result.is_err());
}

/// angr-0xyq2 B1 (revised at Phase 3 review): a single read whose count
/// exceeds the old 4096 clamp on a content_sym fd serves the WHOLE
/// remaining content in one call — matching Python `SimFile.read`'s
/// `min(count, size - pos)`, which never produces a 4096-byte short read.
/// Served natively, never bounced: a Python fallback would split the
/// position cursor (natively-minted fds are not mirrored, angr-8j16).
#[test]
fn test_read_content_sym_over_4096_serves_whole_file_like_python() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x10000, 0x2000, crate::memory::Permission::RWX);
    // 4100 bytes of registered content (concrete entries keep the test
    // fast; mixed files are a supported representation).
    let bytes: Vec<RustBV> = (0..4100u32)
        .map(|i| RustBV::concrete((i & 0xff) as u128, 8))
        .collect();
    state.file_system().register_file_content("/tmp/big", bytes);
    let fd = state
        .file_system()
        .open("/tmp/big".to_string(), crate::state::FdFlags::ReadOnly);

    // count=5000 > old 4096 clamp: serve all 4100 bytes in one call.
    let result = NativeRead
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x10000, 64),
                RustBV::concrete(5000, 64),
            ],
        )
        .expect("served natively, no fallback");
    assert_eq!(result.unwrap().as_u64(), Some(4100));
    assert_eq!(
        state.file_system_ref().fd_info(fd).unwrap().1,
        4100,
        "position advances by the full served byte count"
    );
    // Spot-check both ends of the buffer landed (no 4096 truncation).
    assert_eq!(state.memory_load(0x10000, 1).unwrap().as_u64(), Some(0));
    assert_eq!(
        state.memory_load(0x10000 + 4099, 1).unwrap().as_u64(),
        Some(4099 & 0xff)
    );
    // The next read is at EOF: 0, still native.
    let result = NativeRead
        .call(
            &mut state,
            &[
                RustBV::concrete(fd as u128, 64),
                RustBV::concrete(0x11000, 64),
                RustBV::concrete(5000, 64),
            ],
        )
        .expect("served natively");
    assert_eq!(result.unwrap().as_u64(), Some(0));
}
