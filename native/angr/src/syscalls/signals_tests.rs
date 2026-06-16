// Tests for syscalls/signals.rs (extracted from inline mod tests).
// Covers rt_sigaction/rt_sigprocmask stub handlers and tgkill across arches.
use super::*;

/// Sweep all five stub handlers across every supported arch and
/// verify they return a fresh `RustBV::symbolic` of width
/// `arch().bits()`. Successive invocations must yield distinct
/// fresh symbols (different `RustBV::Symbolic.id`), matching
/// `syscall_stub.py::syscall` semantics where each call gets a
/// new `Unconstrained` BV.
#[test]
fn stub_handlers_return_fresh_symbolic_on_all_arches() {
    // (handler, expected name, arity)
    let cases: &[(&'static dyn NativeSyscall, &str, usize)] = &[
        (&NativeKillSyscall, "kill", 2),
        (&NativeRtSigreturnSyscall, "rt_sigreturn", 0),
        (&NativePauseSyscall, "pause", 0),
        (&NativeAlarmSyscall, "alarm", 1),
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
                .unwrap_or_else(|e| panic!("{arch} {label} errored: {e:?}"));
            let ret = match outcome {
                SyscallOutcome::ContinueSymbolic { ret } => ret,
                other => panic!("{arch} {label} expected ContinueSymbolic, got {other:?}"),
            };
            assert_eq!(ret.width(), bits, "{arch} {label} width");
            assert!(!ret.is_concrete(), "{arch} {label} should be symbolic");

            // Second call: must produce a *distinct* fresh symbol.
            let outcome2 = handler
                .call(&mut state, &args)
                .unwrap_or_else(|e| panic!("{arch} {label} 2nd call errored: {e:?}"));
            let ret2 = match outcome2 {
                SyscallOutcome::ContinueSymbolic { ret } => ret,
                other => panic!("{arch} {label} 2nd: expected ContinueSymbolic, got {other:?}"),
            };
            let (id1, id2) = match (&ret, &ret2) {
                (RustBV::Symbolic { id: a, .. }, RustBV::Symbolic { id: b, .. }) => (*a, *b),
                _ => panic!("{arch} {label} returns must be Symbolic"),
            };
            assert_ne!(
                id1, id2,
                "{arch} {label} successive calls must yield distinct fresh symbols",
            );
        }
    }
}

/// `tgkill` returns the concrete constant 0 across every arch.
#[test]
fn tgkill_returns_zero_on_all_arches() {
    let h = NativeTgkillSyscall;
    assert_eq!(h.name(), "tgkill");
    assert_eq!(h.num_args(), 3);

    for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
        let mut state = RustSimState::new(arch).expect("state");
        let bits = state.arch().bits();
        let args = vec![
            RustBV::concrete(1234, bits),
            RustBV::concrete(5678, bits),
            RustBV::concrete(11 /* SIGSEGV */, bits),
        ];
        let outcome = h.call(&mut state, &args).expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0, "{arch} tgkill"),
            other => panic!("{arch} tgkill expected Continue, got {other:?}"),
        }
    }
}
