// Tests for fgets.rs (NativeFgets/Fgetc/Getchar/Getc procedures).
// Split from the inline `#[cfg(test)] mod tests` block; see rust-mod-tests-sibling-extraction.

use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;

/// AMD64 `_IO_FILE._fileno` byte offset (see `io_file_for_arch` in fileops.rs).
const AMD64_FD_OFFSET: u64 = 112;

/// Map a FILE struct at `file_ptr` whose `_fileno` field holds `fd`. fgets/fgetc
/// resolve the fd from this field; only fd 0 (stdin) is served natively.
fn setup_file_struct(state: &mut RustSimState, file_ptr: u64, fd: i32) {
    let fd_bv = RustBV::concrete((fd as u32) as u128, 32);
    state.map_memory_data(file_ptr & !0xfff, &vec![0u8; 0x4000], Permission::RWX);
    state
        .memory_store(file_ptr + AMD64_FD_OFFSET, fd_bv)
        .unwrap();
}

#[test]
fn test_fgets_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let stdin: u64 = 0x5000;
    setup_file_struct(&mut state, stdin, 0);

    let result = NativeFgets
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),        // buf
                RustBV::concrete(10, 64),            // size
                RustBV::concrete(stdin as u128, 64), // stream (stdin, _fileno=0)
            ],
        )
        .unwrap();

    // Should return buf address
    assert_eq!(result.unwrap().as_u64(), Some(0x2000));

    // First 9 bytes should be symbolic
    for i in 0..9u64 {
        let byte = state.memory_load(0x2000 + i, 1).unwrap();
        assert!(byte.as_u64().is_none(), "byte {} should be symbolic", i);
    }

    // Byte 9 should be NUL terminator
    let nul = state.memory_load(0x2009, 1).unwrap();
    assert_eq!(nul.as_u64(), Some(0));
}

#[test]
fn test_fgets_size_1() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let stdin: u64 = 0x5000;
    setup_file_struct(&mut state, stdin, 0);

    let result = NativeFgets
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(1, 64),             // size=1 means only NUL
                RustBV::concrete(stdin as u128, 64), // stdin
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0x2000));

    // Only NUL should be written
    let nul = state.memory_load(0x2000, 1).unwrap();
    assert_eq!(nul.as_u64(), Some(0));
}

#[test]
fn test_fgets_size_0() {
    let mut state = RustSimState::new("amd64").unwrap();

    let result = NativeFgets
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0, 64), // size=0 returns NULL
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0)); // NULL
}

#[test]
fn test_fgets_symbolic_buf() {
    let mut state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "buf", 64);
    drop(ctx);

    let result = NativeFgets.call(
        &mut state,
        &[sym, RustBV::concrete(10, 64), RustBV::concrete(0, 64)],
    );
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_fgetc_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    let stdin: u64 = 0x5000;
    setup_file_struct(&mut state, stdin, 0);

    let result = NativeFgetc
        .call(
            &mut state,
            &[RustBV::concrete(stdin as u128, 64)], // stream (stdin, _fileno=0)
        )
        .unwrap();

    let val = result.unwrap();
    // Should be symbolic (can't get concrete value)
    assert!(val.as_u64().is_none());
}

#[test]
fn test_getchar_basic() {
    let mut state = RustSimState::new("amd64").unwrap();

    let result = NativeGetchar.call(&mut state, &[]).unwrap();
    let val = result.unwrap();
    // Should be symbolic
    assert!(val.as_u64().is_none());
}

#[test]
fn test_getc_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    let stdin: u64 = 0x5000;
    setup_file_struct(&mut state, stdin, 0);

    let result = NativeGetc
        .call(&mut state, &[RustBV::concrete(stdin as u128, 64)])
        .unwrap();
    let val = result.unwrap();
    assert!(val.as_u64().is_none());
}

#[test]
fn test_fgets_nonstdin_falls_back() {
    // A real-file stream (fd > 0) must defer to Python's SimFile model, not
    // synthesize stdin bytes.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let file_ptr: u64 = 0x5000;
    setup_file_struct(&mut state, file_ptr, 3);

    let result = NativeFgets.call(
        &mut state,
        &[
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(10, 64),
            RustBV::concrete(file_ptr as u128, 64),
        ],
    );
    assert!(matches!(result, Err(ProcedureError::Other(_))));
}

#[test]
fn test_fgets_invalid_fd_returns_minus1() {
    // _fileno < 0 (no backing descriptor): Python fgets returns -1.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let file_ptr: u64 = 0x5000;
    setup_file_struct(&mut state, file_ptr, -1);

    let result = NativeFgets
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(10, 64),
                RustBV::concrete(file_ptr as u128, 64),
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0xFFFFFFFFFFFFFFFF));
}

#[test]
fn test_fgets_symbolic_stream_falls_back() {
    // A symbolic FILE* can't resolve to an fd — must error so Python handles it.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(0x2000, 0x1000, Permission::RWX);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "stream", 64);
    drop(ctx);

    let result = NativeFgets.call(
        &mut state,
        &[RustBV::concrete(0x2000, 64), RustBV::concrete(10, 64), sym],
    );
    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_fgetc_nonstdin_falls_back() {
    let mut state = RustSimState::new("amd64").unwrap();
    let file_ptr: u64 = 0x5000;
    setup_file_struct(&mut state, file_ptr, 3);

    let result = NativeFgetc.call(&mut state, &[RustBV::concrete(file_ptr as u128, 64)]);
    assert!(matches!(result, Err(ProcedureError::Other(_))));
}
