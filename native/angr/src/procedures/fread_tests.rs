// Tests for fread.rs (NativeFread / NativeFreadUnlocked).

use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::{FdFlags, RustSimState};
use crate::symbolic::RustBV;

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
    ).expect("fd space is not exhausted in tests");
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
    ).expect("fd space is not exhausted in tests");
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
            .open_with_content("sym.txt".to_string(), FdFlags::ReadOnly, Vec::new()).expect("fd space is not exhausted in tests");
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

/// Register `n` symbolic bytes for `path` and open it. Returns the fd.
fn open_registered_sym_file(state: &mut RustSimState, path: &str, n: usize) -> u32 {
    let bytes: Vec<RustBV> = {
        let ctx = state.solver().borrow();
        (0..n)
            .map(|i| RustBV::symbolic(&ctx, format!("freadfile_{i}"), 8))
            .collect()
    };
    state.file_system().register_file_content(path, bytes);
    state
        .file_system()
        .open(path.to_string(), FdFlags::ReadOnly).expect("fd space is not exhausted in tests")
}

#[test]
fn test_fread_content_sym_serves_natively() {
    // angr-0xyq2 Phase 2 counterpart of test_fread_empty_content_falls_back:
    // registered symbolic content is served natively (Ok, not a fallback Err).
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let fd = open_registered_sym_file(&mut state, "/tmp/flag", 5);
    let file_ptr = 0x2800;
    write_file_struct(&mut state, file_ptr, fd);

    let result = NativeFread
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(1, 64), // size
                RustBV::concrete(5, 64), // nmemb
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .expect("served natively, no fallback");
    assert_eq!(result.unwrap().as_u64(), Some(5));
    for i in 0..5u64 {
        let byte = state.memory_load(0x2000 + i, 1).unwrap();
        assert!(byte.as_u64().is_none(), "byte {i} should be symbolic");
    }
    assert_eq!(state.file_system_ref().fd_info(fd).unwrap().1, 5);
}

#[test]
fn test_fread_content_sym_item_rounding_matches_python() {
    // 10 symbolic bytes left, size=4, nmemb=3 (total 12): Python fread does
    // simfd.read(dst, 12) -> 10 bytes (position advances by ALL bytes read,
    // not items*size) and returns ret // size = 2 complete items.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let fd = open_registered_sym_file(&mut state, "/tmp/flag", 10);
    let file_ptr = 0x2800;
    write_file_struct(&mut state, file_ptr, fd);

    let result = NativeFread
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(4, 64), // size
                RustBV::concrete(3, 64), // nmemb
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .expect("served natively");
    assert_eq!(result.unwrap().as_u64(), Some(2), "10 bytes / 4 = 2 items");
    assert_eq!(
        state.file_system_ref().fd_info(fd).unwrap().1,
        10,
        "position advances by bytes read (Python simfd.read semantics)"
    );
}

/// angr-0xyq2 Phase 3 review: fread of ONE item larger than the old 4096
/// clamp must return 1 item, not 0. Under the old `MAX_FREAD_SIZE` clamp,
/// `fread(buf, 8192, 1, f)` on an 8192-byte file served 4096 bytes and
/// returned `4096 / 8192 = 0` items — forever (the cursor half-advanced,
/// so the retry also returned 0). Python (`ret // size` after
/// `simfd.read(dst, size*nmemb)`) returns 1.
#[test]
fn test_fread_content_sym_item_over_4096_returns_full_item() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x10000, 0x3000, Permission::RWX);
    let bytes: Vec<RustBV> = (0..8192u32)
        .map(|i| RustBV::concrete((i & 0xff) as u128, 8))
        .collect();
    state.file_system().register_file_content("/tmp/big", bytes);
    let fd = state
        .file_system()
        .open("/tmp/big".to_string(), FdFlags::ReadOnly).expect("fd space is not exhausted in tests");
    let file_ptr = 0x12800;
    write_file_struct(&mut state, file_ptr, fd);

    let result = NativeFread
        .call(
            &mut state,
            &[
                RustBV::concrete(0x10000, 64),
                RustBV::concrete(8192, 64), // size (one item > old 4096 clamp)
                RustBV::concrete(1, 64),    // nmemb
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .expect("served natively, no fallback");
    assert_eq!(result.unwrap().as_u64(), Some(1), "one complete 8192B item");
    assert_eq!(state.file_system_ref().fd_info(fd).unwrap().1, 8192);
}

#[test]
fn test_fread_content_sym_eof_returns_zero_items() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let fd = open_registered_sym_file(&mut state, "/tmp/flag", 3);
    let file_ptr = 0x2800;
    write_file_struct(&mut state, file_ptr, fd);

    // Drain, then fread again: 0 items at EOF, still no fallback.
    for want in [3u64, 0] {
        let result = NativeFread
            .call(
                &mut state,
                &[
                    RustBV::concrete(0x2000, 64),
                    RustBV::concrete(1, 64),
                    RustBV::concrete(4, 64),
                    RustBV::concrete(file_ptr as u128, 64),
                ],
            )
            .expect("served natively");
        assert_eq!(result.unwrap().as_u64(), Some(want));
    }
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
    ).expect("fd space is not exhausted in tests");
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
