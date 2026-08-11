// Tests for procedures/mem_common.rs (the shared symbolic-address /
// symbolic-length store helpers behind memset and memcpy/memmove).
// The helpers were only ever exercised indirectly through the two procedures'
// own test files (angr-03vl4.48); these pin the cap and rejection edges
// directly, so a cap change shows up here rather than as a behaviour shift in
// whichever caller happens to cover it.
use super::*;
use crate::memory::Permission;

#[test]
fn test_check_symbolic_addr_size_zero_is_noop() {
    // size == 0 is Ok(None): the caller no-ops and returns the pointer.
    let res = check_symbolic_addr_size(&RustBV::concrete(0, 64)).unwrap();
    assert_eq!(res, None);
}

#[test]
fn test_check_symbolic_addr_size_in_range() {
    let res = check_symbolic_addr_size(&RustBV::concrete(16, 64)).unwrap();
    assert_eq!(res, Some(16));
    // The cap itself is inclusive.
    let at_cap = RustBV::concrete(u128::from(MAX_SYMBOLIC_ADDR_SIZE), 64);
    assert_eq!(
        check_symbolic_addr_size(&at_cap).unwrap(),
        Some(MAX_SYMBOLIC_ADDR_SIZE)
    );
}

#[test]
fn test_check_symbolic_addr_size_over_cap_rejects() {
    let over = RustBV::concrete(u128::from(MAX_SYMBOLIC_ADDR_SIZE) + 1, 64);
    assert!(matches!(
        check_symbolic_addr_size(&over),
        Err(ProcedureError::SymbolicArgument(ref a)) if a == "size"
    ));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_check_symbolic_addr_size_symbolic_rejects() {
    // A symbolic size on the symbolic-address path always falls back, even
    // when the solver could bound it — the per-byte ITE product is only
    // tractable with a concrete size.
    let state = RustSimState::new("amd64").unwrap();
    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "n", 64)
    };
    assert!(matches!(
        check_symbolic_addr_size(&sym),
        Err(ProcedureError::SymbolicArgument(ref a)) if a == "size"
    ));
}

#[test]
fn test_enumerate_addr_candidates_concrete_is_singleton() {
    let state = RustSimState::new("amd64").unwrap();
    let cands = enumerate_addr_candidates(&state, &RustBV::concrete(0x1000, 64), "dst").unwrap();
    assert_eq!(cands, vec![0x1000]);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_enumerate_addr_candidates_pinned_pair() {
    let state = RustSimState::new("amd64").unwrap();
    let ptr = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "p", 64)
    };
    let constraint = {
        let ctx = state.solver().borrow();
        let a = ptr.eq(&RustBV::concrete(0x1000, 64), &ctx);
        let b = ptr.eq(&RustBV::concrete(0x2000, 64), &ctx);
        a.or(&b, &ctx)
    };
    state.add_constraint(constraint);

    let mut cands = enumerate_addr_candidates(&state, &ptr, "dst").unwrap();
    cands.sort_unstable();
    assert_eq!(cands, vec![0x1000, 0x2000]);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_enumerate_addr_candidates_unbounded_rejects() {
    // An unconstrained pointer blows past MAX_SYMBOLIC_ADDR_CANDIDATES; the
    // helper asks for cap+1 solutions precisely so this is detectable.
    let state = RustSimState::new("amd64").unwrap();
    let ptr = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "p", 64)
    };
    assert!(matches!(
        enumerate_addr_candidates(&state, &ptr, "dst"),
        Err(ProcedureError::SymbolicArgument(ref a)) if a == "dst"
    ));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_enumerate_addr_candidates_at_cap_accepted() {
    // Exactly MAX_SYMBOLIC_ADDR_CANDIDATES solutions is in range — the cap is
    // inclusive, and only the cap+1st solution triggers the fallback.
    let state = RustSimState::new("amd64").unwrap();
    let ptr = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "p", 64)
    };
    let bound = {
        let ctx = state.solver().borrow();
        ptr.ult(
            &RustBV::concrete(MAX_SYMBOLIC_ADDR_CANDIDATES as u128, 64),
            &ctx,
        )
    };
    state.add_constraint(bound);

    let cands = enumerate_addr_candidates(&state, &ptr, "dst").unwrap();
    assert_eq!(cands.len(), MAX_SYMBOLIC_ADDR_CANDIDATES);
}

#[test]
fn test_bounded_symbolic_size_concrete() {
    let state = RustSimState::new("amd64").unwrap();
    assert_eq!(
        bounded_symbolic_size(&state, &RustBV::concrete(0, 64)).unwrap(),
        None
    );
    assert_eq!(
        bounded_symbolic_size(&state, &RustBV::concrete(8, 64)).unwrap(),
        Some(8)
    );
    let at_cap = RustBV::concrete(u128::from(MAX_SYMBOLIC_BYTEWISE_SIZE), 64);
    assert_eq!(
        bounded_symbolic_size(&state, &at_cap).unwrap(),
        Some(MAX_SYMBOLIC_BYTEWISE_SIZE)
    );
}

#[test]
fn test_bounded_symbolic_size_over_cap_rejects() {
    let state = RustSimState::new("amd64").unwrap();
    let over = RustBV::concrete(u128::from(MAX_SYMBOLIC_BYTEWISE_SIZE) + 1, 64);
    assert!(matches!(
        bounded_symbolic_size(&state, &over),
        Err(ProcedureError::SymbolicArgument(ref a)) if a == "size"
    ));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_bounded_symbolic_size_uses_solver_upper_bound() {
    // The bound is the solver's max, not the symbol's width: n <= 4 yields 4.
    let state = RustSimState::new("amd64").unwrap();
    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "n", 64)
    };
    let le = {
        let ctx = state.solver().borrow();
        sym.ule(&RustBV::concrete(4, 64), &ctx)
    };
    state.add_constraint(le);
    assert_eq!(bounded_symbolic_size(&state, &sym).unwrap(), Some(4));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_bounded_symbolic_size_unbounded_rejects() {
    let state = RustSimState::new("amd64").unwrap();
    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "n", 64)
    };
    assert!(matches!(
        bounded_symbolic_size(&state, &sym),
        Err(ProcedureError::SymbolicArgument(ref a)) if a == "size"
    ));
}

#[test]
fn test_symbolic_size_conditional_store_concrete_length() {
    // With a concrete length the ITE conditions fold: bytes below the length
    // take the new value, bytes at/after it keep their originals.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, &[0xFFu8; 8], Permission::RWX);

    let size = RustBV::concrete(2, 64);
    let values: Vec<RustBV> = (0..4).map(|_| RustBV::concrete(0x41, 8)).collect();
    symbolic_size_conditional_store(&mut state, 0x1000, &size, 4, &values).unwrap();

    for i in 0..2u64 {
        let b = state.memory_load(0x1000 + i, 1).unwrap();
        assert_eq!(b.as_u64(), Some(0x41), "byte {i} written");
    }
    for i in 2..4u64 {
        let b = state.memory_load(0x1000 + i, 1).unwrap();
        assert_eq!(b.as_u64(), Some(0xFF), "byte {i} preserved");
    }
    // Past max_size the loop never ran at all.
    let b = state.memory_load(0x1004, 1).unwrap();
    assert_eq!(b.as_u64(), Some(0xFF));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_symbolic_size_conditional_store_symbolic_length() {
    // Every byte in [0, max_size) becomes ITE(i < n, value, original), so both
    // branches stay feasible while n ranges over the bound.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, &[0xFFu8; 8], Permission::RWX);

    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "n", 64)
    };
    let le = {
        let ctx = state.solver().borrow();
        sym.ule(&RustBV::concrete(3, 64), &ctx)
    };
    state.add_constraint(le);

    let max_size = bounded_symbolic_size(&state, &sym).unwrap().unwrap();
    assert_eq!(max_size, 3);
    let values: Vec<RustBV> = (0..max_size).map(|_| RustBV::concrete(0x41, 8)).collect();
    symbolic_size_conditional_store(&mut state, 0x1000, &sym, max_size, &values).unwrap();

    let ctx = state.solver().borrow();
    for i in 0..max_size {
        let b = state.memory_load(0x1000 + i, 1).unwrap();
        assert_eq!(ctx.min(&b, false), Some(0x41), "byte {i} store branch");
        assert_eq!(ctx.max(&b, false), Some(0xFF), "byte {i} preserve branch");
    }
    drop(ctx);
    let b = state.memory_load(0x1003, 1).unwrap();
    assert_eq!(b.as_u64(), Some(0xFF), "byte past the bound untouched");
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_symbolic_size_conditional_store_writes_per_index_values() {
    // `values` is indexed per byte (memcpy passes its pre-snapshotted source
    // bytes, not a repeated fill), so index i must land at dest + i.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, &[0u8; 8], Permission::RWX);

    let sym = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "n", 64)
    };
    // Pin n == 3 so the ITEs fold to the stored values.
    let eq = {
        let ctx = state.solver().borrow();
        sym.eq(&RustBV::concrete(3, 64), &ctx)
    };
    state.add_constraint(eq);

    let values: Vec<RustBV> = [0x11u128, 0x22, 0x33]
        .iter()
        .map(|v| RustBV::concrete(*v, 8))
        .collect();
    symbolic_size_conditional_store(&mut state, 0x1000, &sym, 3, &values).unwrap();

    let ctx = state.solver().borrow();
    for (i, want) in [0x11u128, 0x22, 0x33].iter().enumerate() {
        let b = state.memory_load(0x1000 + i as u64, 1).unwrap();
        assert_eq!(ctx.min(&b, false), Some(*want), "byte {i}");
        assert_eq!(ctx.max(&b, false), Some(*want), "byte {i}");
    }
}
