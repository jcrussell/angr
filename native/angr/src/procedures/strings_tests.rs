// Tests for procedures/strings.rs (string-scanning helpers).
// Extracted from the inline `#[cfg(test)] mod tests` block; see rust-mod-tests-sibling-extraction.

use super::*;
use crate::memory::Permission;

#[test]
fn test_scan_concrete_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hello\x00rest", Permission::RWX);
    let buf = scan_concrete_until_null(&mut state, 0x1000, 4096, "s").unwrap();
    assert_eq!(buf, b"hello");
}

#[test]
fn test_scan_concrete_empty() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"\x00rest", Permission::RWX);
    let buf = scan_concrete_until_null(&mut state, 0x1000, 4096, "s").unwrap();
    assert!(buf.is_empty());
}

#[test]
fn test_scan_concrete_symbolic_byte() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"ab\x00", Permission::RWX);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "b", 8);
    drop(ctx);
    state.memory_store(0x1001, sym).unwrap();
    let res = scan_concrete_until_null(&mut state, 0x1000, 4096, "s");
    assert!(matches!(res, Err(ProcedureError::SymbolicArgument(_))));
}

#[test]
fn test_scan_concrete_max_iters() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, &[b'a'; 8], Permission::RWX);
    let res = scan_concrete_until_null(&mut state, 0x1000, 4, "s");
    assert!(matches!(res, Err(ProcedureError::MaxIterations(4))));
}

#[test]
fn test_scan_concrete_bounded_null_found() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"hi\x00xyz", Permission::RWX);
    let (buf, found) = scan_concrete_bounded(&mut state, 0x1000, 10, "s").unwrap();
    assert_eq!(buf, b"hi");
    assert!(found);
}

#[test]
fn test_scan_concrete_bounded_max_no_null() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abcdef", Permission::RWX);
    let (buf, found) = scan_concrete_bounded(&mut state, 0x1000, 4, "s").unwrap();
    assert_eq!(buf, b"abcd");
    assert!(!found);
}

#[test]
fn test_find_null_addr_basic() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
    assert_eq!(
        find_null_addr(&mut state, 0x1000, 4096, "p").unwrap(),
        0x1003
    );
}

#[test]
fn test_find_null_addr_at_start() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"\x00rest", Permission::RWX);
    assert_eq!(
        find_null_addr(&mut state, 0x1000, 4096, "p").unwrap(),
        0x1000
    );
}

#[test]
fn test_scan_for_null_symbolic_all_concrete() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
    match scan_for_null_symbolic(&mut state, 0x1000, 4096).unwrap() {
        ScanOutcome::AllConcrete { length } => assert_eq!(length, 3),
        ScanOutcome::Symbolic { .. } => panic!("expected concrete"),
    }
}

#[test]
fn test_scan_for_null_symbolic_with_sym_byte() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x1000, b"a_c\x00", Permission::RWX);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "b1", 8);
    drop(ctx);
    state.memory_store(0x1001, sym).unwrap();
    match scan_for_null_symbolic(&mut state, 0x1000, 4096).unwrap() {
        ScanOutcome::Symbolic { bytes } => {
            // Position 0 was concrete 'a' (skipped), positions 1..3 collected.
            assert!(bytes.iter().any(|(p, _)| *p == 1));
            // Concrete '\x00' at position 3 stops the scan.
            let last = bytes.last().unwrap();
            assert_eq!(last.0, 3);
        }
        ScanOutcome::AllConcrete { .. } => panic!("expected symbolic"),
    }
}

#[test]
fn test_scan_for_null_symbolic_caps_symbolic_bytes() {
    // A fully-symbolic, unterminated buffer must not collect more than
    // MAX_SYMBOLIC_SCAN_BYTES positions, even with a far larger `max` window —
    // that cap is what keeps the ITE chain and the null_exists_constraint
    // disjunction Python-sized instead of thousands of terms wide.
    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory_data(0x2000, &vec![b'a'; 512], Permission::RWX);
    let ctx = state.solver().borrow();
    let syms: Vec<RustBV> = (0..512)
        .map(|i| RustBV::symbolic(&ctx, format!("c{i}"), 8))
        .collect();
    drop(ctx);
    for (i, sym) in syms.into_iter().enumerate() {
        state.memory_store(0x2000 + i as u64, sym).unwrap();
    }
    match scan_for_null_symbolic(&mut state, 0x2000, 4096).unwrap() {
        ScanOutcome::Symbolic { bytes } => {
            assert_eq!(bytes.len(), MAX_SYMBOLIC_SCAN_BYTES);
            assert_eq!(bytes.last().unwrap().0, MAX_SYMBOLIC_SCAN_BYTES as u64 - 1);
        }
        ScanOutcome::AllConcrete { .. } => panic!("expected symbolic"),
    }
}

#[test]
fn test_scan_for_null_symbolic_concrete_bytes_do_not_draw_down_budget() {
    // Python charges only *symbolic* characters against buf_symbolic_bytes, so
    // a buffer with one symbolic byte followed by concrete filler must still
    // scan all the way to its terminator.
    let mut state = RustSimState::new("amd64").unwrap();
    let mut data = vec![b'a'; 200];
    data.push(0);
    state.map_memory_data(0x3000, &data, Permission::RWX);
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "head", 8);
    drop(ctx);
    state.memory_store(0x3000, sym).unwrap();
    match scan_for_null_symbolic(&mut state, 0x3000, 4096).unwrap() {
        ScanOutcome::Symbolic { bytes } => {
            assert_eq!(bytes.len(), 201);
            assert_eq!(bytes.last().unwrap().0, 200);
        }
        ScanOutcome::AllConcrete { .. } => panic!("expected symbolic"),
    }
}

#[test]
fn test_build_strlen_chain_concrete() {
    let state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    // Bytes: [(0, 'a'), (1, '\0')]. Chain should reduce to 1.
    let bytes = vec![
        (0u64, RustBV::concrete(b'a' as u128, 8)),
        (1u64, RustBV::concrete(0u128, 8)),
    ];
    let result = build_strlen_chain(&bytes, 64, 4096, &ctx);
    assert_eq!(result.as_u64(), Some(1));
}

#[test]
fn test_null_exists_constraint_trivially_true_with_concrete_null() {
    let state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let sym = RustBV::symbolic(&ctx, "b0", 8);
    let bytes = vec![(0u64, sym), (1u64, RustBV::concrete(0u128, 8))];
    // A concrete null in the window already proves termination.
    assert!(null_exists_constraint(&bytes, &ctx).is_none());
}

#[test]
fn test_null_exists_constraint_none_when_all_concrete_nonnull() {
    let state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let bytes = vec![
        (0u64, RustBV::concrete(b'a' as u128, 8)),
        (1u64, RustBV::concrete(b'b' as u128, 8)),
    ];
    // No symbolic byte can be null -> the assertion would be trivially false
    // and would wrongly kill the state; the helper declines instead.
    assert!(null_exists_constraint(&bytes, &ctx).is_none());
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_null_exists_constraint_prunes_unterminated_symbolic_window() {
    let state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let b0 = RustBV::symbolic(&ctx, "s0", 8);
    let b1 = RustBV::symbolic(&ctx, "s1", 8);
    let bytes = vec![(0u64, b0.clone()), (1u64, b1.clone())];
    let cons = null_exists_constraint(&bytes, &ctx).expect("expected a constraint");
    ctx.assume_true(&cons);
    assert!(ctx.is_sat());
    // With both bytes forced non-null the window has no terminator: unsat.
    let nonnull = RustBV::concrete(b'x' as u128, 8);
    ctx.assume_true(&b0.eq(&nonnull, &ctx));
    ctx.assume_true(&b1.eq(&nonnull, &ctx));
    assert!(!ctx.is_sat());
}
