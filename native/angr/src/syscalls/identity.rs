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

use super::{NativeSyscall, SyscallError, SyscallOutcome, stub_syscall};
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
        pub(crate) struct $ty;

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

// setuid / setgid: 1-arg stubs (no dedicated Python SimProcedure).
stub_syscall!(NativeSetuidSyscall, "setuid", "syscall_stub_setuid", 1);
stub_syscall!(NativeSetgidSyscall, "setgid", "syscall_stub_setgid", 1);

#[cfg(test)]
#[path = "identity_tests.rs"]
mod tests;
