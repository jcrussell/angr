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
//! these. The unhandled syscall falls through to the generic stub which
//! emits a fresh symbolic return — meaningfully different from a
//! constant-success return. To preserve parity, those two syscalls are
//! intentionally NOT registered here; they continue through the Python
//! callback path.
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
