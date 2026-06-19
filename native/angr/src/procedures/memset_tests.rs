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

#[test]
fn test_memset_symbolic_size_pinned() {
    // A symbolic size constrained to exactly 3 must fill exactly the first 3
    // bytes with the fill value and leave the rest untouched — no Python
    // fallback.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, &[0u8; 16], Permission::RWX);

    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "n", 64)
    };
    // Constrain before the call so ctx.max(size) == 3 inside the proc.
    let eq = {
        let ctx = state.solver().borrow();
        sym.eq(&RustBV::concrete(3, 64), &ctx)
    };
    state.add_constraint(eq);

    let proc = NativeMemset;
    let result = proc
        .call(
            &mut state,
            &[
                RustBV::concrete(0x1000, 64),
                RustBV::concrete(0x41, 64),
                sym,
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0x1000));

    let ctx = state.solver().borrow();
    // Bytes [0,3) are 0x41 (only feasible value under n == 3).
    for i in 0..3u64 {
        let b = state.memory_load(0x1000 + i, 1).unwrap();
        assert_eq!(ctx.min(&b, false), Some(0x41));
        assert_eq!(ctx.max(&b, false), Some(0x41));
    }
    // Bytes at/after the length keep their original zero.
    for i in 3..6u64 {
        let b = state.memory_load(0x1000 + i, 1).unwrap();
        assert_eq!(b.as_u64(), Some(0));
    }
}

#[test]
fn test_memset_symbolic_size_conditional() {
    // A symbolic size with a small upper bound fills conditionally: each byte
    // in the bounded range can take either the fill value or its original
    // contents, depending on the (still-symbolic) length.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, &[0xFFu8; 16], Permission::RWX);

    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "n", 64)
    };
    // n <= 4  → solver max bound is 4, loop runs over [0, 4).
    let le = {
        let ctx = state.solver().borrow();
        sym.ule(&RustBV::concrete(4, 64), &ctx)
    };
    state.add_constraint(le);

    let proc = NativeMemset;
    proc.call(
        &mut state,
        &[
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0x41, 64),
            sym,
        ],
    )
    .unwrap();

    let ctx = state.solver().borrow();
    // Each byte in [0,4) is ITE(i < n, 0x41, 0xFF): both branches feasible
    // because n ranges over [0, 4].
    for i in 0..4u64 {
        let b = state.memory_load(0x1000 + i, 1).unwrap();
        assert_eq!(ctx.min(&b, false), Some(0x41), "byte {i} fill branch");
        assert_eq!(ctx.max(&b, false), Some(0xFF), "byte {i} preserve branch");
    }
    // Byte 5 is beyond the bound and never touched.
    let b = state.memory_load(0x1005, 1).unwrap();
    assert_eq!(b.as_u64(), Some(0xFF));
}

#[test]
fn test_memset_symbolic_size_unbounded_fallback() {
    // An unconstrained symbolic size has no useful upper bound, so memset must
    // fall back to Python (returns Err).
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, &[0u8; 16], Permission::RWX);

    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "n", 64)
    };
    let proc = NativeMemset;
    let result = proc.call(
        &mut state,
        &[
            RustBV::concrete(0x1000, 64),
            RustBV::concrete(0x41, 64),
            sym,
        ],
    );
    assert!(result.is_err(), "unbounded symbolic size should fall back");
}
