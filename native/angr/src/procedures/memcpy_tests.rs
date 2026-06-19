// Tests for procedures/memcpy.rs (NativeMemcpy / NativeMemmove).
// Extracted from the parent module's #[cfg(test)] block; see memcpy.rs.
use super::*;
use crate::memory::Permission;
use crate::procedures::NativeSimProcedure;

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

#[test]
fn test_memcpy_symbolic_size_pinned() {
    // A symbolic size constrained to exactly 4 must copy exactly the first 4
    // source bytes and leave the rest of dst untouched — no Python fallback.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abcdefgh", Permission::RWX); // src
    state.map_memory_data(0x2000, &[0u8; 8], Permission::RWX); // dst

    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "n", 64)
    };
    let eq = {
        let ctx = state.solver().borrow();
        sym.eq(&RustBV::concrete(4, 64), &ctx)
    };
    state.add_constraint(eq);

    let proc = NativeMemcpy;
    let result = proc
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(0x1000, 64),
                sym,
            ],
        )
        .unwrap();
    assert_eq!(result.unwrap().as_u64(), Some(0x2000));

    let ctx = state.solver().borrow();
    // dst[0..4) == "abcd" (only feasible value under n == 4).
    for (i, &c) in b"abcd".iter().enumerate() {
        let b = state.memory_load(0x2000 + i as u64, 1).unwrap();
        assert_eq!(ctx.min(&b, false), Some(c as u128));
        assert_eq!(ctx.max(&b, false), Some(c as u128));
    }
    // dst[4..8) keep their original zero.
    for i in 4..8u64 {
        let b = state.memory_load(0x2000 + i, 1).unwrap();
        assert_eq!(b.as_u64(), Some(0));
    }
}

#[test]
fn test_memcpy_symbolic_size_conditional() {
    // A symbolic size with a small upper bound copies conditionally: each byte
    // in the bounded range can take either the source byte or its original dst
    // contents, depending on the (still-symbolic) length.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, &[0x41u8; 8], Permission::RWX); // src = 'A'
    state.map_memory_data(0x2000, &[0xFFu8; 8], Permission::RWX); // dst

    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "n", 64)
    };
    let le = {
        let ctx = state.solver().borrow();
        sym.ule(&RustBV::concrete(4, 64), &ctx)
    };
    state.add_constraint(le);

    let proc = NativeMemcpy;
    proc.call(
        &mut state,
        &[
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(0x1000, 64),
            sym,
        ],
    )
    .unwrap();

    let ctx = state.solver().borrow();
    // Each byte in [0,4) is ITE(i < n, 0x41, 0xFF): both branches feasible.
    for i in 0..4u64 {
        let b = state.memory_load(0x2000 + i, 1).unwrap();
        assert_eq!(ctx.min(&b, false), Some(0x41), "byte {i} copy branch");
        assert_eq!(ctx.max(&b, false), Some(0xFF), "byte {i} preserve branch");
    }
    // Byte 5 is beyond the bound and never touched.
    let b = state.memory_load(0x2005, 1).unwrap();
    assert_eq!(b.as_u64(), Some(0xFF));
}

#[test]
fn test_memcpy_symbolic_size_unbounded_fallback() {
    // An unconstrained symbolic size has no useful upper bound, so memcpy must
    // fall back to Python (returns Err).
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, &[0u8; 8], Permission::RWX);
    state.map_memory_data(0x2000, &[0u8; 8], Permission::RWX);

    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "n", 64)
    };
    let proc = NativeMemcpy;
    let result = proc.call(
        &mut state,
        &[
            RustBV::concrete(0x2000, 64),
            RustBV::concrete(0x1000, 64),
            sym,
        ],
    );
    assert!(result.is_err(), "unbounded symbolic size should fall back");
}

#[test]
fn test_memmove_symbolic_size_overlap() {
    // memmove with a symbolic size constrained to a single concrete value must
    // copy correctly even when src and dst overlap (snapshots source first).
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(
        0x1000,
        b"abcdefgh\x00\x00\x00\x00\x00\x00\x00\x00",
        Permission::RWX,
    );

    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "n", 64)
    };
    let eq = {
        let ctx = state.solver().borrow();
        sym.eq(&RustBV::concrete(6, 64), &ctx)
    };
    state.add_constraint(eq);

    let proc = NativeMemmove;
    proc.call(
        &mut state,
        &[
            RustBV::concrete(0x1000, 64), // dst
            RustBV::concrete(0x1002, 64), // src (overlapping)
            sym,
        ],
    )
    .unwrap();

    // Should have "cdefgh" at the front (same as concrete-size overlap test).
    let ctx = state.solver().borrow();
    for (i, &c) in b"cdefgh".iter().enumerate() {
        let b = state.memory_load(0x1000 + i as u64, 1).unwrap();
        assert_eq!(ctx.min(&b, false), Some(c as u128), "byte {i}");
        assert_eq!(ctx.max(&b, false), Some(c as u128), "byte {i}");
    }
}
