// Tests for the rt_sigaction syscall handler (NativeRtSigactionSyscall).
// Split out of sigaction.rs to keep the handler module focused (angr-2rl59).
use super::*;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

fn fresh_state() -> RustSimState {
    RustSimState::new("amd64").expect("amd64 state")
}

fn mk_args(signum: u64) -> Vec<RustBV> {
    vec![
        RustBV::concrete(signum as u128, 64),
        RustBV::concrete(0, 64),
        RustBV::concrete(0, 64),
        RustBV::concrete(8, 64),
    ]
}

#[test]
fn ordinary_signal_returns_zero() {
    let h = NativeRtSigactionSyscall;
    let mut state = fresh_state();
    let outcome = h.call(&mut state, &mk_args(11 /* SIGSEGV */)).expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        _ => panic!("expected Continue"),
    }
}

#[test]
fn signum_33_returns_neg_einval() {
    let h = NativeRtSigactionSyscall;
    let mut state = fresh_state();
    let outcome = h.call(&mut state, &mk_args(33)).expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_EINVAL),
        _ => panic!("expected Continue"),
    }
    // Sanity: -22 in i64 is 0xFFFFFFFFFFFFFFEA.
    assert_eq!(NEG_EINVAL, 0xFFFF_FFFF_FFFF_FFEA);
}

#[test]
fn symbolic_signum_falls_back() {
    let h = NativeRtSigactionSyscall;
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let sym = RustBV::symbolic(&ctx, "signum", 64);
    let err = h
        .call(
            &mut state,
            &[
                sym,
                RustBV::concrete(0, 64),
                RustBV::concrete(0, 64),
                RustBV::concrete(8, 64),
            ],
        )
        .expect_err("must fall back");
    match err {
        SyscallError::SymbolicArgument(msg) => assert!(
            msg.contains("signum"),
            "should name the symbolic arg, got {msg:?}"
        ),
        other => panic!("expected SymbolicArgument, got {other:?}"),
    }
}

/// signum is the only arg we read — symbolic act/oldact/sigsetsize
/// must NOT force a fallback (Python ignores them anyway).
#[test]
fn symbolic_other_args_do_not_force_fallback() {
    let h = NativeRtSigactionSyscall;
    let mut state = fresh_state();
    let ctx = SymContext::new();
    let sym_act = RustBV::symbolic(&ctx, "act", 64);
    let outcome = h
        .call(
            &mut state,
            &[
                RustBV::concrete(11, 64),
                sym_act,
                RustBV::concrete(0, 64),
                RustBV::concrete(8, 64),
            ],
        )
        .expect("symbolic act/oldact must not block native fast path");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        _ => panic!("expected Continue"),
    }
}

#[test]
fn handler_metadata() {
    let h = NativeRtSigactionSyscall;
    assert_eq!(h.name(), "rt_sigaction");
    assert_eq!(h.num_args(), 4);
}
