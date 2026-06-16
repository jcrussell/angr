// Tests for procedures/memcpy.rs (NativeMemcpy / NativeMemmove).
// Extracted from the parent module's #[cfg(test)] block; see memcpy.rs.
use super::*;
use crate::memory::Permission;

#[test]
fn test_memcpy_basic() {
    let mut state = RustSimState::new("amd64").unwrap();

    // Map source with data
    let data = b"hello world!";
    state.map_memory_data(0x1000, data, Permission::RWX);

    // Map destination
    state.map_memory(0x2000, 0x1000, Permission::RWX);

    let proc = NativeMemcpy;
    let result = proc
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64), // dst
                RustBV::concrete(0x1000, 64), // src
                RustBV::concrete(12, 64),     // size
            ],
        )
        .unwrap();

    // Result should be dst pointer
    assert_eq!(result.unwrap().as_u64(), Some(0x2000));

    // Verify data was copied
    for i in 0..12u64 {
        let val = state.memory_load(0x2000 + i, 1).unwrap();
        assert_eq!(val.as_u64(), Some(data[i as usize] as u64));
    }
}

#[test]
fn test_memcpy_zero_size() {
    let mut state = RustSimState::new("amd64").unwrap();

    // Map memory
    state.map_memory(0x1000, 0x2000, Permission::RWX);

    let proc = NativeMemcpy;
    let result = proc
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0, 64), // zero size
            ],
        )
        .unwrap();

    assert_eq!(result.unwrap().as_u64(), Some(0x2000));
}

#[test]
fn test_memcpy_symbolic_dst() {
    let mut state = RustSimState::new("amd64").unwrap();

    let ctx = state.solver().borrow();
    let sym_dst = RustBV::symbolic(&ctx, "dst", 64);
    drop(ctx);

    let proc = NativeMemcpy;
    let result = proc.call(
        &mut state,
        &[
            sym_dst,
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(10, 64),
        ],
    );

    assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_memmove_overlapping() {
    let mut state = RustSimState::new("amd64").unwrap();

    // Map memory with "abcdefgh"
    let data = b"abcdefgh\x00\x00\x00\x00\x00\x00\x00\x00";
    state.map_memory_data(0x1000, data, Permission::RWX);

    // Move from offset 2 to offset 0 (overlapping, should work)
    let proc = NativeMemmove;
    let _result = proc
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64), // dst
                RustBV::concrete(0x1002, 64), // src (overlapping)
                RustBV::concrete(6, 64),      // size
            ],
        )
        .unwrap();

    // Should have "cdefghgh" now
    let val = state.memory_load(0x1000, 1).unwrap();
    assert_eq!(val.as_u64(), Some(b'c' as u64));

    let val = state.memory_load(0x1001, 1).unwrap();
    assert_eq!(val.as_u64(), Some(b'd' as u64));
}
