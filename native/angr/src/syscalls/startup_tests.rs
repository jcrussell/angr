//! Unit tests for [`super`] — process/startup syscalls.
//!
//! Split out of `startup.rs` to keep the implementation module lean
//! (see bd memory `rust-mod-tests-sibling-extraction`).

use super::*;
use crate::memory::Permission;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

fn fresh_state() -> RustSimState {
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    state.map_memory(0x2000, 0x2000, Permission::RWX);
    state
}

#[test]
fn metadata() {
    assert_eq!(NativeUnameSyscall.name(), "uname");
    assert_eq!(NativeUnameSyscall.num_args(), 1);
    assert_eq!(NativeSetTidAddressSyscall.name(), "set_tid_address");
    assert_eq!(NativeSetRobustListSyscall.num_args(), 2);
    assert_eq!(NativeGetrandomSyscall.name(), "getrandom");
}

#[test]
fn uname_writes_sysname_and_machine() {
    let mut state = fresh_state();
    let out = NativeUnameSyscall
        .call(&mut state, &[RustBV::concrete(0x2000, 64)])
        .expect("ok");
    match out {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        _ => panic!("expected Continue"),
    }
    // sysname field.
    for (i, &b) in b"Linux".iter().enumerate() {
        assert_eq!(
            state.memory_load(0x2000 + i as u64, 1).unwrap().as_u64(),
            Some(b as u64)
        );
    }
    // NUL padding right after "Linux".
    assert_eq!(state.memory_load(0x2005, 1).unwrap().as_u64(), Some(0));
    // machine field (amd64 → "x86_64") at offset 4*65.
    let m = 4 * UTSNAME_FIELD;
    for (i, &b) in b"x86_64".iter().enumerate() {
        assert_eq!(
            state
                .memory_load(0x2000 + m + i as u64, 1)
                .unwrap()
                .as_u64(),
            Some(b as u64)
        );
    }
}

#[test]
fn uname_symbolic_buf_falls_back() {
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let err = NativeUnameSyscall
        .call(&mut state, &[RustBV::symbolic(&ctx, "buf", 64)])
        .expect_err("fallback");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

#[test]
fn set_tid_address_returns_one() {
    let mut state = fresh_state();
    let out = NativeSetTidAddressSyscall
        .call(&mut state, &[RustBV::concrete(0x2000, 64)])
        .expect("ok");
    match out {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 1),
        _ => panic!("expected Continue"),
    }
}

#[test]
fn set_robust_list_returns_zero() {
    let mut state = fresh_state();
    let out = NativeSetRobustListSyscall
        .call(
            &mut state,
            &[RustBV::concrete(0x2000, 64), RustBV::concrete(24, 64)],
        )
        .expect("ok");
    match out {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        _ => panic!("expected Continue"),
    }
}

#[test]
fn getrandom_fills_symbolic_and_returns_buflen() {
    let mut state = fresh_state();
    let out = NativeGetrandomSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(8, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect("ok");
    match out {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 8),
        _ => panic!("expected Continue"),
    }
    for i in 0..8u64 {
        assert!(
            state.memory_load(0x2000 + i, 1).unwrap().as_u64().is_none(),
            "byte {i} should be symbolic"
        );
    }
}

#[test]
fn getrandom_two_calls_mint_distinct_bytes() {
    let mut state = fresh_state();
    NativeGetrandomSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(1, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();
    NativeGetrandomSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2100, 64),
                RustBV::concrete(1, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .unwrap();
    let a = state.memory_load(0x2000, 1).unwrap();
    let b = state.memory_load(0x2100, 1).unwrap();
    // Distinct names → not solver-equal.
    assert!(a.as_u64().is_none() && b.as_u64().is_none());
}

#[test]
fn getrandom_oversize_falls_back() {
    let mut state = fresh_state();
    let err = NativeGetrandomSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::concrete(9000, 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("fallback");
    assert!(matches!(err, SyscallError::Other(_)));
}

#[test]
fn getrandom_symbolic_buflen_falls_back() {
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let err = NativeGetrandomSyscall
        .call(
            &mut state,
            &[
                RustBV::concrete(0x2000, 64),
                RustBV::symbolic(&ctx, "n", 64),
                RustBV::concrete(0, 64),
            ],
        )
        .expect_err("fallback");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}
