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
