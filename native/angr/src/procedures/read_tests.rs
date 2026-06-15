//! Tests for the read SimProcedure (extracted from read.rs).
use super::*;
use crate::memory::Permission;

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
