// Tests for procedures/strings.rs (string-scanning helpers).
// Extracted from the inline `#[cfg(test)] mod tests` block; see rust-module-root-split-recipe.

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
    let sym = RustBV::symbol("b", 8);
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
    let sym = RustBV::symbol("b1", 8);
    drop(ctx);
    state.memory_store(0x1001, sym).unwrap();
    match scan_for_null_symbolic(&mut state, 0x1000, 4096).unwrap() {
        ScanOutcome::Symbolic { bytes, budget_spent } => {
            assert!(!budget_spent);
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
        .map(|i| RustBV::symbol(format!("c{i}"), 8))
        .collect();
    drop(ctx);
    for (i, sym) in syms.into_iter().enumerate() {
        state.memory_store(0x2000 + i as u64, sym).unwrap();
    }
    match scan_for_null_symbolic(&mut state, 0x2000, 4096).unwrap() {
        ScanOutcome::Symbolic { bytes, budget_spent } => {
            assert!(budget_spent);
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
    let sym = RustBV::symbol("head", 8);
    drop(ctx);
    state.memory_store(0x3000, sym).unwrap();
    match scan_for_null_symbolic(&mut state, 0x3000, 4096).unwrap() {
        ScanOutcome::Symbolic { bytes, budget_spent } => {
            assert!(!budget_spent);
            assert_eq!(bytes.len(), 201);
            assert_eq!(bytes.last().unwrap().0, 200);
        }
        ScanOutcome::AllConcrete { .. } => panic!("expected symbolic"),
    }
}

#[test]
fn test_build_strlen_chain_concrete() {
    // Bytes: [(0, 'a'), (1, '\0')]. Chain should reduce to 1.
    let bytes = vec![
        (0u64, RustBV::concrete(b'a' as u128, 8)),
        (1u64, RustBV::concrete(0u128, 8)),
    ];
    let result = build_strlen_chain(&bytes, 64, 4096);
    assert_eq!(result.as_u64(), Some(1));
}

#[test]
fn test_null_exists_constraint_trivially_true_with_concrete_null() {
    let sym = RustBV::symbol("b0", 8);
    let bytes = vec![(0u64, sym), (1u64, RustBV::concrete(0u128, 8))];
    // A concrete null in the window already proves termination.
    assert!(null_exists_constraint(&bytes).is_none());
}

#[test]
fn test_null_exists_constraint_none_when_all_concrete_nonnull() {
    let bytes = vec![
        (0u64, RustBV::concrete(b'a' as u128, 8)),
        (1u64, RustBV::concrete(b'b' as u128, 8)),
    ];
    // No symbolic byte can be null -> the assertion would be trivially false
    // and would wrongly kill the state; the helper declines instead.
    assert!(null_exists_constraint(&bytes).is_none());
}

#[test]
fn test_null_exists_constraint_prunes_unterminated_symbolic_window() {
    let state = RustSimState::new("amd64").unwrap();
    let ctx = state.solver().borrow();
    let b0 = RustBV::symbol("s0", 8);
    let b1 = RustBV::symbol("s1", 8);
    let bytes = vec![(0u64, b0.clone()), (1u64, b1.clone())];
    let cons = null_exists_constraint(&bytes).expect("expected a constraint");
    ctx.assume_true(&cons);
    assert!(ctx.is_sat());
    // With both bytes forced non-null the window has no terminator: unsat.
    let nonnull = RustBV::concrete(b'x' as u128, 8);
    ctx.assume_true(&b0.eq(&nonnull));
    ctx.assume_true(&b1.eq(&nonnull));
    assert!(!ctx.is_sat());
}

/// Address of [`budget_spending_buffer`]'s fully symbolic input.
const SYM_BUF: u64 = 0x2000;
/// Address of its concrete, NUL-terminated `'P'` counterpart.
const CONC_BUF: u64 = 0x3000;
/// Length of both buffers: past [`MAX_SYMBOLIC_SCAN_BYTES`], so the decisive
/// bytes at 60..100 lie beyond the budget.
const BUF_LEN: u64 = 100;

/// State with `BUF_LEN` fresh symbolic bytes at `SYM_BUF` and `BUF_LEN`
/// concrete `'P'` bytes plus a NUL at `CONC_BUF` (the a10_h `pwmemcmp`
/// shape).
fn budget_spending_buffer() -> RustSimState {
    let mut state = RustSimState::new("amd64").unwrap();
    let mut conc = vec![b'P'; BUF_LEN as usize];
    conc.push(0);
    state.map_memory_data(CONC_BUF, &conc, Permission::RWX);
    state.map_memory_data(SYM_BUF, &[b'a'; BUF_LEN as usize + 1], Permission::RWX);
    for i in 0..BUF_LEN {
        let sym = RustBV::symbol(format!("in{i}"), 8);
        state.memory_store(SYM_BUF + i, sym).unwrap();
    }
    state.map_memory_data(0x4000, &[0u8; BUF_LEN as usize], Permission::RWX);
    state
}

/// Call `proc` on a fresh [`budget_spending_buffer`] with `args`.
fn call_on_budget_buffer(
    proc: &dyn crate::procedures::NativeSimProcedure,
    args: &[u64],
) -> Result<Option<RustBV>, ProcedureError> {
    let mut state = budget_spending_buffer();
    let args: Vec<RustBV> = args
        .iter()
        .map(|a| RustBV::concrete(u128::from(*a), 64))
        .collect();
    proc.call(&mut state, &args)
}

#[test]
fn test_spent_symbolic_budget_defers_without_pruning_constraint() {
    // Past 60 symbolic positions the collected window is only a prefix. With
    // no constraint forcing the answer into it, the chain's default (0 for a
    // compare, n for strnlen/strncpy, NULL or the last in-budget match for
    // the memchr family) would answer for bytes nobody read, so each of these
    // defers to Python (angr-vmi2l.99).
    use crate::procedures::strchr::{
        NativeMemchr, NativeMemrchr, NativeRawmemchr, NativeStrchr, NativeStrrchr,
    };
    use crate::procedures::strcmp::{NativeStrcmp, NativeStrncasecmp, NativeStrncmp};
    use crate::procedures::strcpy::NativeStrncpy;
    use crate::procedures::strlen::NativeStrnlen;
    use crate::procedures::memcmp::NativeMemcmp;
    let x = u64::from(b'x');
    let cases: &[(&str, &dyn crate::procedures::NativeSimProcedure, Vec<u64>)] = &[
        ("memcmp", &NativeMemcmp, vec![SYM_BUF, CONC_BUF, BUF_LEN]),
        ("strncmp", &NativeStrncmp, vec![SYM_BUF, CONC_BUF, BUF_LEN]),
        ("strncasecmp", &NativeStrncasecmp, vec![SYM_BUF, CONC_BUF, BUF_LEN]),
        // Concrete s1: `null_exists_constraint` has no symbolic s1 byte.
        ("strcmp_conc_s1", &NativeStrcmp, vec![CONC_BUF, SYM_BUF]),
        ("strnlen", &NativeStrnlen, vec![SYM_BUF, BUF_LEN]),
        ("strncpy", &NativeStrncpy, vec![0x4000, SYM_BUF, BUF_LEN]),
        ("memchr", &NativeMemchr, vec![SYM_BUF, x, BUF_LEN]),
        ("memrchr", &NativeMemrchr, vec![SYM_BUF, x, BUF_LEN]),
        ("rawmemchr", &NativeRawmemchr, vec![SYM_BUF, x]),
        ("strchr", &NativeStrchr, vec![SYM_BUF, x]),
        ("strrchr", &NativeStrrchr, vec![SYM_BUF, x]),
        ("strchr_sym_target", &NativeStrchr, vec![SYM_BUF, x]),
    ];
    let mut served = Vec::new();
    for (name, proc, args) in cases.iter() {
        let mut state = budget_spending_buffer();
        let mut args: Vec<RustBV> = args
            .iter()
            .map(|a| RustBV::concrete(u128::from(*a), 64))
            .collect();
        if *name == "strchr_sym_target" {
            args[1] = RustBV::symbol("target", 64);
        }
        let result = proc.call(&mut state, &args);
        if !matches!(result, Err(ProcedureError::MaxIterations(MAX_SYMBOLIC_SCAN_BYTES))) {
            served.push(*name);
        }
    }
    assert!(served.is_empty(), "served instead of deferring: {served:?}");
}

#[test]
fn test_spent_symbolic_budget_served_under_pruning_constraint() {
    // strlen and strcmp with a symbolic s1 assert a terminator inside the
    // collected prefix (as Python's inline strlen does), so their chain's
    // default is unreachable and they stay native.
    use crate::procedures::strcmp::NativeStrcmp;
    use crate::procedures::strlen::NativeStrlen;
    let strlen = call_on_budget_buffer(&NativeStrlen, &[SYM_BUF]);
    assert!(matches!(strlen, Ok(Some(_))), "strlen: {strlen:?}");
    let strcmp = call_on_budget_buffer(&NativeStrcmp, &[SYM_BUF, CONC_BUF]);
    assert!(matches!(strcmp, Ok(Some(_))), "strcmp: {strcmp:?}");
}
