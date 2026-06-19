use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;
use crate::state::RustSimState;

#[test]
fn test_memset_basic() {
    let mut state = RustSimState::new("amd64").unwrap();

    // Map writable memory
    let data = vec![0u8; 16];
    state.map_memory_data(0x1000, &data, Permission::RWX);

    let proc = NativeMemset;
    let result = proc
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x41, 64), // 'A'
                RustBV::concrete(4, 64),
            ],
        )
        .unwrap();

    // Should return dest
    assert_eq!(result.unwrap().as_u64(), Some(0x1000));

    // Verify memory was filled
    let loaded = state.memory_load(0x1000, 4).unwrap();
    assert_eq!(loaded.as_u64(), Some(0x41414141));
}

#[test]
fn test_memset_zero() {
    let mut state = RustSimState::new("amd64").unwrap();

    let data = vec![0xFFu8; 16];
    state.map_memory_data(0x1000, &data, Permission::RWX);

    let proc = NativeMemset;
    proc.call(
        &mut state,
        &[
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0, 64),
            RustBV::concrete(8, 64),
        ],
    )
    .unwrap();

    let loaded = state.memory_load(0x1000, 8).unwrap();
    assert_eq!(loaded.as_u64(), Some(0));
}

#[test]
fn test_memset_symbolic_value() {
    // A symbolic byte value must NOT fall back to Python: each filled byte
    // should be (a copy of) the low 8 bits of the symbolic value.
    let mut state = RustSimState::new("amd64").unwrap();
    let data = vec![0u8; 16];
    state.map_memory_data(0x1000, &data, Permission::RWX);

    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "c", 64)
    };

    let proc = NativeMemset;
    let result = proc
        .call(
            &mut state,
            &[RustBV::concrete(0x1000, 64), sym, RustBV::concrete(10, 64)],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0x1000));

    // Every byte in [0x1000, 0x100A) is symbolic (no concrete value).
    for i in 0..10u64 {
        let b = state.memory_load(0x1000 + i, 1).unwrap();
        assert_eq!(b.width(), 8);
        assert!(b.as_u64().is_none(), "byte {i} should be symbolic");
    }
    // The byte past the region is untouched (still concrete zero).
    let after = state.memory_load(0x100A, 1).unwrap();
    assert_eq!(after.as_u64(), Some(0));
}
