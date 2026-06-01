//! Process / user / group identity syscall handlers.
//!
//! Mirrors `procedures/linux_kernel/{getpid,getppid,gettid,getuid,getgid,
//! geteuid,getegid}.py`. All seven are constant-return getters in angr's
//! Python implementation:
//!
//! * `getpid`, `gettid` — return `state.posix.pid` (default 1337).
//! * `getppid`            — return `state.posix.ppid` (default 1336).
//! * `getuid`, `geteuid`, `getgid`, `getegid` — return the literal 1000
//!   (matches `Sim*Procedure.run`).
//!
//! RustSimState does not currently carry a posix plugin, so the pid/ppid
//! defaults are hardcoded here to match Python's defaults
//! (`angr/state_plugins/posix.py:159`).
//!
//! Note on `setuid` / `setgid`: angr has no Python `SimProcedure` for
//! these — the unhandled syscall falls through to `procedures/stubs/
//! syscall_stub.py::syscall`, which returns
//! `state.solver.Unconstrained("syscall_stub_<name>", returnty.size, ...)`.
//! `NativeSetuidSyscall` / `NativeSetgidSyscall` mirror that exactly:
//! ignore the single uid_t/gid_t argument and emit a fresh
//! `RustBV::symbolic` sized to `arch().bits()` (the C `long` return
//! width on every supported arch). The dispatcher routes this through
//! `SyscallOutcome::ContinueSymbolic`, matching the stub semantics
//! bit-for-bit so binaries can fork on the return value.
//!
//! No-argument syscalls: handlers advertise `num_args() == 0`, so the
//! dispatcher passes an empty slice. No symbolic-argument fallback is
//! needed.

use super::{NativeSyscall, SyscallError, SyscallOutcome};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Angr's default `state.posix.pid` (see `state_plugins/posix.py:159`).
const DEFAULT_PID: u64 = 1337;
/// Angr's default `state.posix.ppid` (see `state_plugins/posix.py:160`).
const DEFAULT_PPID: u64 = 1336;
/// Angr's default `getuid` / `getgid` / `geteuid` / `getegid` return value
/// (see `procedures/linux_kernel/getuid.py` etc.).
const DEFAULT_UID_GID: u64 = 1000;

/// Macro to declare a zero-arg constant-return syscall handler.
///
/// Cuts the boilerplate for `name() / num_args() == 0 / call() -> Continue { ret }`
/// for the seven identity getters. Each handler is a unit struct so the
/// registry can key on a unique type per name.
macro_rules! constant_syscall {
    ($ty:ident, $label:expr, $ret:expr) => {
        pub struct $ty;

        impl NativeSyscall for $ty {
            fn name(&self) -> &'static str {
                $label
            }

            fn num_args(&self) -> usize {
                0
            }

            fn call(
                &self,
                _state: &mut RustSimState,
                _args: &[RustBV],
            ) -> Result<SyscallOutcome, SyscallError> {
                Ok(SyscallOutcome::Continue { ret: $ret })
            }
        }
    };
}

constant_syscall!(NativeGetpidSyscall, "getpid", DEFAULT_PID);
constant_syscall!(NativeGettidSyscall, "gettid", DEFAULT_PID);
constant_syscall!(NativeGetppidSyscall, "getppid", DEFAULT_PPID);
constant_syscall!(NativeGetuidSyscall, "getuid", DEFAULT_UID_GID);
constant_syscall!(NativeGeteuidSyscall, "geteuid", DEFAULT_UID_GID);
constant_syscall!(NativeGetgidSyscall, "getgid", DEFAULT_UID_GID);
constant_syscall!(NativeGetegidSyscall, "getegid", DEFAULT_UID_GID);

/// Macro to declare a 1-arg syscall handler that returns a fresh symbolic
/// value (matches Python `syscall_stub.py::syscall` ReturnUnconstrained
/// semantics for syscalls with no dedicated `SimProcedure`).
///
/// Used for `setuid` / `setgid`: the single argument is intentionally
/// ignored (no semantic effect on simulated process state), and the
/// return is a fresh `RustBV::symbolic` sized to the arch's `long` width
/// (`arch().bits()` for every supported arch). The symbol name mirrors
/// Python's `f"syscall_stub_{display_name}"`.
macro_rules! stub_syscall_1arg {
    ($ty:ident, $label:expr, $sym_name:expr) => {
        pub struct $ty;

        impl NativeSyscall for $ty {
            fn name(&self) -> &'static str {
                $label
            }

            fn num_args(&self) -> usize {
                1
            }

            fn call(
                &self,
                state: &mut RustSimState,
                _args: &[RustBV],
            ) -> Result<SyscallOutcome, SyscallError> {
                let bits = state.arch().bits();
                let ret = {
                    let ctx = state.solver().borrow();
                    RustBV::symbolic(&ctx, $sym_name, bits)
                };
                Ok(SyscallOutcome::ContinueSymbolic { ret })
            }
        }
    };
}

stub_syscall_1arg!(NativeSetuidSyscall, "setuid", "syscall_stub_setuid");
stub_syscall_1arg!(NativeSetgidSyscall, "setgid", "syscall_stub_setgid");

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_continue(outcome: SyscallOutcome, expected: u64) {
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, expected),
            other => panic!("expected Continue({expected}), got {other:?}"),
        }
    }

    #[test]
    fn getpid_returns_default_pid() {
        let h = NativeGetpidSyscall;
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        assert_eq!(h.name(), "getpid");
        assert_eq!(h.num_args(), 0);
        let outcome = h.call(&mut state, &[]).expect("getpid never errs");
        assert_continue(outcome, DEFAULT_PID);
    }

    #[test]
    fn gettid_returns_default_pid() {
        // gettid shares pid in single-threaded angr.
        let h = NativeGettidSyscall;
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        assert_eq!(h.name(), "gettid");
        let outcome = h.call(&mut state, &[]).expect("gettid never errs");
        assert_continue(outcome, DEFAULT_PID);
    }

    #[test]
    fn getppid_returns_default_ppid() {
        let h = NativeGetppidSyscall;
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        assert_eq!(h.name(), "getppid");
        let outcome = h.call(&mut state, &[]).expect("getppid never errs");
        assert_continue(outcome, DEFAULT_PPID);
    }

    #[test]
    fn setuid_setgid_return_fresh_symbolic() {
        // setuid / setgid have no Python SimProcedure; the stub returns
        // a fresh unconstrained symbol. The native handler must do the
        // same — every invocation yields a distinct symbol (distinct
        // RustBV::Symbolic.id), and the BV must be sized to the arch's
        // `long` (== arch().bits()) on every supported arch.
        for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();

            for handler in [
                &NativeSetuidSyscall as &dyn NativeSyscall,
                &NativeSetgidSyscall as &dyn NativeSyscall,
            ] {
                assert_eq!(handler.num_args(), 1);
                let arg = RustBV::concrete(0, bits);
                let outcome = handler
                    .call(&mut state, &[arg])
                    .expect("setuid/setgid never errs");
                let ret = match outcome {
                    SyscallOutcome::ContinueSymbolic { ret } => ret,
                    other => panic!(
                        "{arch} {} expected ContinueSymbolic, got {other:?}",
                        handler.name()
                    ),
                };
                assert_eq!(
                    ret.width(),
                    bits,
                    "{arch} {} return width should match arch().bits()",
                    handler.name(),
                );
                assert!(
                    ret.as_u64().is_none(),
                    "{arch} {} return must be symbolic (not concrete)",
                    handler.name(),
                );
                // Fresh symbol on every call: confirm by id inequality
                // across a second invocation.
                let arg2 = RustBV::concrete(0, bits);
                let outcome2 = handler.call(&mut state, &[arg2]).unwrap();
                let ret2 = match outcome2 {
                    SyscallOutcome::ContinueSymbolic { ret } => ret,
                    _ => unreachable!(),
                };
                let (id1, id2) = match (&ret, &ret2) {
                    (
                        RustBV::Symbolic { id: a, .. },
                        RustBV::Symbolic { id: b, .. },
                    ) => (*a, *b),
                    _ => panic!("{arch} {} both returns should be Symbolic", handler.name()),
                };
                assert_ne!(
                    id1, id2,
                    "{arch} {} successive calls must yield distinct fresh symbols",
                    handler.name(),
                );
            }
        }
    }

    #[test]
    fn uid_gid_getters_return_1000() {
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        for (h, label) in [
            (
                &NativeGetuidSyscall as &dyn NativeSyscall,
                "getuid",
            ),
            (&NativeGeteuidSyscall as &dyn NativeSyscall, "geteuid"),
            (&NativeGetgidSyscall as &dyn NativeSyscall, "getgid"),
            (&NativeGetegidSyscall as &dyn NativeSyscall, "getegid"),
        ] {
            assert_eq!(h.name(), label);
            assert_eq!(h.num_args(), 0);
            let outcome = h.call(&mut state, &[]).expect("getter never errs");
            assert_continue(outcome, DEFAULT_UID_GID);
        }
    }
}
