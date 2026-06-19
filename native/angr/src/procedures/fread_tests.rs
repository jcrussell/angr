// Tests for fread.rs (NativeFread / NativeFreadUnlocked).

use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::{FdFlags, RustSimState};

// AMD64 _IO_FILE._fileno offset (mirrors io_file_for_arch in fileops.rs).
const AMD64_FILENO_OFF: u64 = 112;

/// Build a FILE struct at `file_ptr` whose `_fileno` field holds `fd`.
fn write_file_struct(state: &mut RustSimState, file_ptr: u64, fd: u32) {
    state
        .memory_store(
            file_ptr + AMD64_FILENO_OFF,
            RustBV::concrete(fd as u128, 32),
        )
        .unwrap();
}

#[test]
fn test_fread_concrete_content() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX); // dst + FILE struct

    let fd = state.file_system().open_with_content(
        "a.txt".to_string(),
        FdFlags::ReadOnly,
        b"hello".to_vec(),
    );
    let file_ptr = 0x2800;
    write_file_struct(&mut state, file_ptr, fd);

    let result = NativeFread
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),           // dst
                RustBV::concrete(1, 64),                // size
                RustBV::concrete(5, 64),                // nmemb
                RustBV::concrete(file_ptr as u128, 64), // stream
            ],
        )
        .unwrap();

    // 5 items of size 1 read.
    assert_eq!(result.unwrap().as_u64(), Some(5));
    for (i, b) in b"hello".iter().enumerate() {
        let byte = state.memory_load(0x2000 + i as u64, 1).unwrap();
        assert_eq!(byte.as_u64(), Some(*b as u64));
    }
}

#[test]
fn test_fread_item_count_rounds_down() {
    // size=4, content "hello" (5 bytes) -> only 1 complete item.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let fd = state.file_system().open_with_content(
        "a.txt".to_string(),
        FdFlags::ReadOnly,
        b"hello".to_vec(),
    );
    let file_ptr = 0x2800;
    write_file_struct(&mut state, file_ptr, fd);

    let result = NativeFread
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(4, 64), // size
                RustBV::concrete(2, 64), // nmemb (total 8, only 5 available)
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(1)); // 5 bytes / 4 = 1 item
}

#[test]
fn test_fread_empty_content_falls_back() {
    // An fd open in the Rust FS with no concrete content (a symbolic SimFile)
    // defers to Python rather than synthesizing fresh bytes.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let fd =
        state
            .file_system()
            .open_with_content("sym.txt".to_string(), FdFlags::ReadOnly, Vec::new());
    let file_ptr = 0x2800;
    write_file_struct(&mut state, file_ptr, fd);

    let result = NativeFread.call(
        &mut state,
        &[
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(1, 64),
            RustBV::concrete(4, 64), // 4 items
            RustBV::concrete(file_ptr as u128, 64),
        ],
    );
    assert!(matches!(result, Err(ProcedureError::Other(_))));
}

#[test]
fn test_fread_zero_count() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let result = NativeFread
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64), // size 0
                RustBV::concrete(10, 64),
                RustBV::concrete(0x2800, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0));
}

#[test]
fn test_fread_unopened_fd_falls_back() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    // FILE struct points at fd 42, which is not open in the Rust FS.
    let file_ptr = 0x2800;
    write_file_struct(&mut state, file_ptr, 42);

    let result = NativeFread.call(
        &mut state,
        &[
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(1, 64),
            RustBV::concrete(4, 64),
            RustBV::concrete(file_ptr as u128, 64),
        ],
    );
    assert!(matches!(result, Err(ProcedureError::Other(_))));
}

#[test]
fn test_fread_symbolic_stream_falls_back() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "stream", 64);
    drop(ctx);

    let result = NativeFread.call(
        &mut state,
        &[
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(1, 64),
            RustBV::concrete(4, 64),
            sym,
        ],
    );
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_fread_unlocked_matches_fread() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let fd = state.file_system().open_with_content(
        "a.txt".to_string(),
        FdFlags::ReadOnly,
        b"abcd".to_vec(),
    );
    let file_ptr = 0x2800;
    write_file_struct(&mut state, file_ptr, fd);

    let result = NativeFreadUnlocked
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(1, 64),
                RustBV::concrete(4, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(4));
}
