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
stub_syscall!(
    NativeRtSigreturnSyscall,
    "rt_sigreturn",
    "syscall_stub_rt_sigreturn",
    0
);
// pause() → long
stub_syscall!(NativePauseSyscall, "pause", "syscall_stub_pause", 0);
// alarm(seconds) → long
stub_syscall!(NativeAlarmSyscall, "alarm", "syscall_stub_alarm", 1);

/// `tgkill(tgid, tid, sig)` — mirrors Python's
/// `procedures/linux_kernel/tgkill.py`, which returns
/// `claripy.BVV(0, self.arch.sizeof["int"])` regardless of args.
pub(crate) struct NativeTgkillSyscall;

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

test_submod!("signals_tests.rs" => signals_tests);
