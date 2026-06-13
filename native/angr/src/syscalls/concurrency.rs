//! Concurrency primitives and event-fd syscall handlers.
//!
//! Covers `futex`, `eventfd`, `eventfd2`, `epoll_create`,
//! `epoll_create1`, `epoll_ctl`, `epoll_wait`, and `epoll_pwait`.
//! Mirrors angr's Python behavior:
//!
//! * `futex` has a dedicated `procedures/linux_kernel/futex.py`. It
//!   evaluates `futex_op` concretely, returns `0` when `op & 1`
//!   (`FUTEX_WAKE` — the low bit covers both private and shared
//!   variants), otherwise returns a fresh symbolic int. Symbolic
//!   `futex_op` falls back to Python so the same `solver.eval` path
//!   runs there.
//! * `eventfd` / `eventfd2` / `epoll_create` / `epoll_create1` /
//!   `epoll_ctl` / `epoll_wait` / `epoll_pwait` have no Python `SimProcedure` — they
//!   fall through to `procedures/stubs/syscall_stub.py::syscall`,
//!   which emits `state.solver.Unconstrained("syscall_stub_<name>",
//!   returnty.size, ...)`. The native handlers mirror that via
//!   `SyscallOutcome::ContinueSymbolic` with a fresh `RustBV::symbolic`
//!   of width `arch().bits()` (the C `long` width on every supported
//!   Linux arch).
//!
//! ## Concurrency-not-modeled note
//!
//! angr is a single-threaded symex engine: it does not model real
//! scheduling or blocking. `futex(FUTEX_WAIT)` and `epoll_wait`
//! therefore return a fresh symbolic rather than ever blocking, which
//! is exactly what the existing Python path does. Binaries that block
//! on these as a synchronization primitive will still race past them;
//! that limitation is upstream of these handlers and applies equally
//! to the Python `syscall_stub` fallback.

use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg, fresh_symbolic, stub_syscall};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// `futex(uaddr, futex_op, val, timeout, uaddr2, val3)` — mirrors
/// `procedures/linux_kernel/futex.py`. Concretely evaluates `futex_op`;
/// `op & 1` (any FUTEX_WAKE variant) → return 0, else → fresh symbolic.
pub struct NativeFutexSyscall;

impl NativeSyscall for NativeFutexSyscall {
    fn name(&self) -> &'static str {
        "futex"
    }

    fn num_args(&self) -> usize {
        6
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        if args.len() < 2 {
            return Err(SyscallError::Other(format!(
                "futex expected 6 args, got {}",
                args.len()
            )));
        }
        let op = extract_concrete_arg(&args[1], "futex op")?;
        // FUTEX_WAKE = 1, FUTEX_WAKE_PRIVATE = 1 | 128, etc. Python
        // matches `op & 1`, covering both the bare and the
        // PRIVATE/CLOCK_REALTIME-flagged variants.
        if op & 1 == 1 {
            return Ok(SyscallOutcome::Continue { ret: 0 });
        }
        let bits = state.arch().bits();
        let ret = {
            let ctx = state.solver().borrow();
            fresh_symbolic(&ctx, "futex", bits)
        };
        Ok(SyscallOutcome::ContinueSymbolic { ret })
    }
}

// eventfd(count) → int
stub_syscall!(NativeEventfdSyscall, "eventfd", "syscall_stub_eventfd", 1);
// eventfd2(count, flags) → int
stub_syscall!(NativeEventfd2Syscall, "eventfd2", "syscall_stub_eventfd2", 2);
// epoll_create(size) → int
stub_syscall!(
    NativeEpollCreateSyscall,
    "epoll_create",
    "syscall_stub_epoll_create",
    1
);
// epoll_create1(flags) → int
stub_syscall!(
    NativeEpollCreate1Syscall,
    "epoll_create1",
    "syscall_stub_epoll_create1",
    1
);
// epoll_ctl(epfd, op, fd, event*) → int
stub_syscall!(
    NativeEpollCtlSyscall,
    "epoll_ctl",
    "syscall_stub_epoll_ctl",
    4
);
// epoll_wait(epfd, events*, maxevents, timeout) → int
stub_syscall!(
    NativeEpollWaitSyscall,
    "epoll_wait",
    "syscall_stub_epoll_wait",
    4
);
// epoll_pwait(epfd, events*, maxevents, timeout, sigmask*, sigsetsize) → int
// AArch64 asm-generic omits legacy `epoll_wait` (binaries call epoll_pwait
// at syscall 22 instead). Registered on every supported arch.
stub_syscall!(
    NativeEpollPwaitSyscall,
    "epoll_pwait",
    "syscall_stub_epoll_pwait",
    6
);

#[cfg(test)]
mod tests {
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
            .map(|i| RustBV::concrete(if i == 1 { 1 } else { 0 }, bits)) // op=1 = FUTEX_WAKE
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
                let args: Vec<RustBV> =
                    (0..nargs).map(|_| RustBV::concrete(0, bits)).collect();
                let outcome = handler
                    .call(&mut state, &args)
                    .unwrap_or_else(|e| panic!("{arch} {label}: {e:?}"));
                let ret = match outcome {
                    SyscallOutcome::ContinueSymbolic { ret } => ret,
                    other => panic!(
                        "{arch} {label} expected ContinueSymbolic, got {other:?}"
                    ),
                };
                assert_eq!(ret.width(), bits);
                assert!(ret.as_u64().is_none(), "{arch} {label} must be symbolic");

                let outcome2 = handler.call(&mut state, &args).unwrap();
                let ret2 = match outcome2 {
                    SyscallOutcome::ContinueSymbolic { ret } => ret,
                    _ => unreachable!(),
                };
                let (id1, id2) = match (&ret, &ret2) {
                    (
                        RustBV::Symbolic { id: a, .. },
                        RustBV::Symbolic { id: b, .. },
                    ) => (*a, *b),
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
                use z3::ast::Ast as _;
                let solver = z3::Solver::new();
                solver.assert(&ret.to_z3_ast()._eq(&ret2.to_z3_ast()).not());
                assert_eq!(
                    solver.check(),
                    z3::SatResult::Sat,
                    "{arch} {label}: successive returns must be solver-distinct"
                );
            }
        }
    }
}
