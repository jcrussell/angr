//! Process / user / group identity syscall handlers.
//!
//! Mirrors `procedures/linux_kernel/{getpid,getppid,gettid,getuid,getgid,
//! geteuid,getegid}.py`. All seven are constant-return getters in angr's
//! Python implementation:
//!
//! * `getpid`, `gettid` — return `state.posix.pid`.
//! * `getppid`            — return `state.posix.ppid`.
//! * `getuid`, `geteuid`, `getgid`, `getegid` — return the literal 1000
//!   (matches `Sim*Procedure.run`).
//!
//! The pid/ppid come from `RustSimState::process_ids`, which the manager
//! seeds from `state.posix` (angr-vmi2l.149).
//!
//! Note on `setuid` / `setgid`: angr has no Python `SimProcedure` for
//! these — the unhandled syscall falls through to `procedures/stubs/
//! syscall_stub.py::syscall`, which returns
//! `state.solver.Unconstrained("syscall_stub_<name>", returnty.size, ...)`.
//! `NativeSetuidSyscall` / `NativeSetgidSyscall` mirror that exactly:
//! ignore the single uid_t/gid_t argument and emit a fresh
//! `RustBV::symbol` sized to `arch().bits()` (the C `long` return
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

/// Angr's default `getuid` / `getgid` / `geteuid` / `getegid` return value
/// (see `procedures/linux_kernel/getuid.py` etc.).
///
/// `pub(crate)` because the same constant backs the PLT-libc-call side of
/// the uid/gid getters: `procedures/getid.rs` imports it rather than keeping a
/// second copy in sync (angr-5mnx3.35).
pub(crate) const DEFAULT_UID_GID: u64 = 1000;

/// Macro to declare a zero-arg syscall handler whose return value is
/// `$ret` evaluated against the state (bound to `$state`).
///
/// Cuts the boilerplate for `name() / num_args() == 0 / call() -> Continue { ret }`
/// for the identity getters. Each handler is a unit struct so the
/// registry can key on a unique type per name.
macro_rules! constant_syscall {
    ($ty:ident, $label:expr, |$state:ident| $ret:expr) => {
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
                $state: &mut RustSimState,
                _args: &[RustBV],
            ) -> Result<SyscallOutcome, SyscallError> {
                Ok(SyscallOutcome::Continue { ret: $ret })
            }
        }
    };
}

constant_syscall!(NativeGetpidSyscall, "getpid", |state| state
    .process_ids()
    .pid);
constant_syscall!(NativeGettidSyscall, "gettid", |state| state
    .process_ids()
    .pid);
constant_syscall!(NativeGetppidSyscall, "getppid", |state| state
    .process_ids()
    .ppid);
constant_syscall!(NativeGetuidSyscall, "getuid", |_state| DEFAULT_UID_GID);
constant_syscall!(NativeGeteuidSyscall, "geteuid", |_state| DEFAULT_UID_GID);
constant_syscall!(NativeGetgidSyscall, "getgid", |_state| DEFAULT_UID_GID);
constant_syscall!(NativeGetegidSyscall, "getegid", |_state| DEFAULT_UID_GID);

// setuid / setgid: 1-arg stubs (no dedicated Python SimProcedure).
stub_syscall!(NativeSetuidSyscall, "setuid", "syscall_stub_setuid", 1);
stub_syscall!(NativeSetgidSyscall, "setgid", "syscall_stub_setgid", 1);

test_submod!("identity_tests.rs" => identity_tests);
