// Tests for syscalls/concurrency.rs (futex and related concurrency syscall stubs).
// Extracted from the inline `mod tests` block; see that module's siblings.
use super::*;
use crate::symbolic::SymContext;

fn fresh_state() -> RustSimState {
    RustSimState::new("amd64").expect("amd64 state")
}

// ---- futex --------------------------------------------------------

#[test]
fn futex_wake_returns_zero_concretely() {
    let h = NativeFutexSyscall;
    let mut state = fresh_state();
    let bits = state.arch().bits();
    let args: Vec<RustBV> = (0..6)
        .map(|i| RustBV::concrete(u128::from(i == 1), bits)) // op=1 = FUTEX_WAKE
        .collect();
    let outcome = h.call(&mut state, &args).expect("ok");
    match outcome {
        SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
        other => panic!("expected Continue, got {other:?}"),
    }
}

#[test]
fn futex_wake_private_also_returns_zero() {
    let h = NativeFutexSyscall;
    let mut state = fresh_state();
    let bits = state.arch().bits();
    // FUTEX_WAKE | FUTEX_PRIVATE_FLAG = 1 | 128 = 129. Python's
    // `op & 1` matches; we must too.
    let args: Vec<RustBV> = (0..6)
        .map(|i| RustBV::concrete(if i == 1 { 129 } else { 0 }, bits))
        .collect();
    let outcome = h.call(&mut state, &args).expect("ok");
    assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
}

#[test]
fn futex_wait_returns_fresh_symbolic() {
    let h = NativeFutexSyscall;
    let mut state = fresh_state();
    let bits = state.arch().bits();
    // FUTEX_WAIT = 0, low bit clear → fall through to symbolic.
    let args: Vec<RustBV> = (0..6).map(|_| RustBV::concrete(0, bits)).collect();
    let outcome = h.call(&mut state, &args).expect("ok");
    let ret = match outcome {
        SyscallOutcome::ContinueSymbolic { ret } => ret,
        other => panic!("expected ContinueSymbolic, got {other:?}"),
    };
    assert_eq!(ret.width(), bits);
    assert!(ret.as_u64().is_none(), "futex WAIT must be symbolic");
}

#[test]
fn futex_symbolic_op_returns_error() {
    let h = NativeFutexSyscall;
    let mut state = fresh_state();
    let bits = state.arch().bits();
    let ctx = SymContext::new();
    let mut args: Vec<RustBV> = (0..6).map(|_| RustBV::concrete(0, bits)).collect();
    args[1] = RustBV::symbolic(&ctx, "op_sym", bits);
    let err = h.call(&mut state, &args).expect_err("must error");
    assert!(matches!(err, SyscallError::SymbolicArgument(_)));
}

/// The guard threshold must match the declared `num_args()` (6), not the
/// weaker `< 2` it used to check — a short arg list is a caller bug, and
/// the error message has always claimed 6.
#[test]
fn futex_rejects_short_arg_list() {
    let h = NativeFutexSyscall;
    let mut state = fresh_state();
    let bits = state.arch().bits();
    let args: Vec<RustBV> = (0..5).map(|_| RustBV::concrete(0, bits)).collect();
    let err = h.call(&mut state, &args).expect_err("must error");
    match err {
        SyscallError::Other(msg) => assert!(msg.contains("expected 6 args, got 5"), "{msg}"),
        other => panic!("expected Other, got {other:?}"),
    }
}

// ---- stub family -------------------------------------------------

/// All seven stub handlers return a fresh symbolic of width
/// `arch().bits()` on every supported arch, and successive calls
/// produce distinct `RustBV::Symbolic.id`.
#[test]
fn stub_handlers_return_fresh_symbolic_on_all_arches() {
    let cases: &[(&'static dyn NativeSyscall, &str, usize)] = &[
        (&NativeEventfdSyscall, "eventfd", 1),
        (&NativeEventfd2Syscall, "eventfd2", 2),
        (&NativeEpollCreateSyscall, "epoll_create", 1),
        (&NativeEpollCreate1Syscall, "epoll_create1", 1),
        (&NativeEpollCtlSyscall, "epoll_ctl", 4),
        (&NativeEpollWaitSyscall, "epoll_wait", 4),
        (&NativeEpollPwaitSyscall, "epoll_pwait", 6),
    ];

    for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        for &(handler, label, nargs) in cases {
            assert_eq!(handler.name(), label);
            assert_eq!(handler.num_args(), nargs, "{label} arity");
            let args: Vec<RustBV> = (0..nargs).map(|_| RustBV::concrete(0, bits)).collect();
            let outcome = handler
                .call(&mut state, &args)
                .unwrap_or_else(|e| panic!("{arch} {label}: {e:?}"));
            let ret = match outcome {
                SyscallOutcome::ContinueSymbolic { ret } => ret,
                other => panic!("{arch} {label} expected ContinueSymbolic, got {other:?}"),
            };
            assert_eq!(ret.width(), bits);
            assert!(ret.as_u64().is_none(), "{arch} {label} must be symbolic");

            let outcome2 = handler.call(&mut state, &args).unwrap();
            let ret2 = match outcome2 {
                SyscallOutcome::ContinueSymbolic { ret } => ret,
                _ => unreachable!(),
            };
            let (id1, id2) = match (&ret, &ret2) {
                (RustBV::Symbolic { id: a, .. }, RustBV::Symbolic { id: b, .. }) => (*a, *b),
                _ => panic!("{arch} {label}: expected Symbolic"),
            };
            assert_ne!(
                id1, id2,
                "{arch} {label} successive calls must yield distinct symbols"
            );

            // Solver-distinctness (regression guard for angr-8o7w): the two
            // returns must be *satisfiably unequal*, not merely distinct
            // RustBV ids. A fixed Z3 name minted twice aliases to the same
            // `new_const`, making `ret1 != ret2` unsatisfiable even though
            // the RustBV ids differ. fresh_symbolic appends symbol_counter
            // so each mint is a distinct Z3 term.

            let solver = z3::Solver::new();
            solver.assert(ret.to_z3_ast().eq(ret2.to_z3_ast()).not());
            assert_eq!(
                solver.check(),
                z3::SatResult::Sat,
                "{arch} {label}: successive returns must be solver-distinct"
            );
        }
    }
}
