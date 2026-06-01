//! Signals + process-control syscall handlers (angr-0hif.6).
//!
//! Covers `kill`, `tgkill`, `rt_sigreturn`, `pause`, `alarm` — all five
//! mirror Python angr's behavior bit-for-bit:
//!
//! * `kill`, `rt_sigreturn`, `pause`, `alarm` — no dedicated
//!   `SimProcedure`; fall through to
//!   `procedures/stubs/syscall_stub.py::syscall`, which returns a
//!   fresh `state.solver.Unconstrained` BV. Matched here by emitting
//!   a `RustBV::symbolic` of width `arch().bits()` and routing through
//!   `SyscallOutcome::ContinueSymbolic`.
//! * `tgkill` — Python returns concrete `claripy.BVV(0, sizeof(int))`
//!   (see `procedures/linux_kernel/tgkill.py`). Matched here with
//!   `SyscallOutcome::Continue { ret: 0 }` — the return register is
//!   written as a 64-bit zero, which is bit-identical to a 32-bit
//!   zero zero-extended on every supported arch.
//!
//! Notes on what is *not* covered here:
//!
//! * `rt_sigaction` — already covered by
//!   `syscalls::sigaction::NativeRtSigactionSyscall` (baseline). The
//!   bd description for angr-0hif.6 lists it for completeness; no new
//!   handler is needed.
//! * `rt_sigprocmask` — Python's
//!   `procedures/linux_kernel/sigprocmask.py` mutates
//!   `state.posix.sigmask` and stores at the `oldset` pointer.
//!   `RustSimState` does not currently carry a posix plugin (per
//!   `identity.rs`'s pid hardcoding), so exact native parity would
//!   require plumbing sigmask into the Rust state. Deferred — the
//!   unhandled-syscall path continues to dispatch to the Python
//!   `_handle_syscall_callback`, preserving full sigmask semantics.

use super::{NativeSyscall, SyscallError, SyscallOutcome, stub_syscall};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

// kill(pid, sig) → long
stub_syscall!(NativeKillSyscall, "kill", "syscall_stub_kill", 2);
// rt_sigreturn() → long
stub_syscall!(NativeRtSigreturnSyscall, "rt_sigreturn", "syscall_stub_rt_sigreturn", 0);
// pause() → long
stub_syscall!(NativePauseSyscall, "pause", "syscall_stub_pause", 0);
// alarm(seconds) → long
stub_syscall!(NativeAlarmSyscall, "alarm", "syscall_stub_alarm", 1);

/// `tgkill(tgid, tid, sig)` — mirrors Python's
/// `procedures/linux_kernel/tgkill.py`, which returns
/// `claripy.BVV(0, self.arch.sizeof["int"])` regardless of args.
pub struct NativeTgkillSyscall;

impl NativeSyscall for NativeTgkillSyscall {
    fn name(&self) -> &'static str {
        "tgkill"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

#[cfg(test)]
mod tests {
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

                let args: Vec<RustBV> =
                    (0..nargs).map(|_| RustBV::concrete(0, bits)).collect();
                let outcome = handler
                    .call(&mut state, &args)
                    .unwrap_or_else(|e| panic!("{arch} {label} errored: {e:?}"));
                let ret = match outcome {
                    SyscallOutcome::ContinueSymbolic { ret } => ret,
                    other => panic!(
                        "{arch} {label} expected ContinueSymbolic, got {other:?}"
                    ),
                };
                assert_eq!(ret.width(), bits, "{arch} {label} width");
                assert!(
                    !ret.is_concrete(),
                    "{arch} {label} should be symbolic"
                );

                // Second call: must produce a *distinct* fresh symbol.
                let outcome2 = handler
                    .call(&mut state, &args)
                    .unwrap_or_else(|e| panic!("{arch} {label} 2nd call errored: {e:?}"));
                let ret2 = match outcome2 {
                    SyscallOutcome::ContinueSymbolic { ret } => ret,
                    other => panic!("{arch} {label} 2nd: expected ContinueSymbolic, got {other:?}"),
                };
                let (id1, id2) = match (&ret, &ret2) {
                    (
                        RustBV::Symbolic { id: a, .. },
                        RustBV::Symbolic { id: b, .. },
                    ) => (*a, *b),
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
}
