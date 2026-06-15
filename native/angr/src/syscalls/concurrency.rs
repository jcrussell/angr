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

use super::{
    NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg, fresh_symbolic, stub_syscall,
};
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
stub_syscall!(
    NativeEventfd2Syscall,
    "eventfd2",
    "syscall_stub_eventfd2",
    2
);
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
#[path = "concurrency_tests.rs"]
mod tests;
