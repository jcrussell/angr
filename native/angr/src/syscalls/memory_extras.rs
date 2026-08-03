//! Memory-advisory syscall handlers.
//!
//! Mirrors angr's behavior for `madvise`, `mremap`, `msync`, `mlock`,
//! `munlock`, `mlockall`, and `munlockall`. None of these have a
//! dedicated Python `SimProcedure` — the unhandled-syscall path falls
//! through to `procedures/stubs/syscall_stub.py::syscall`, which returns
//! `state.solver.Unconstrained("syscall_stub_<name>", returnty.size, ...)`.
//!
//! The native handlers below match that exactly: ignore the args (no
//! semantic effect on simulated process state) and emit a fresh
//! `RustBV::symbolic` sized to `arch().bits()` (the C `long` return
//! width on every supported Linux arch). The dispatcher routes this
//! through `SyscallOutcome::ContinueSymbolic`.
//!
//! Notes:
//! * `mremap` is registered as a stub for parity with Python; angr's
//!   `syscall_stub` does NOT update page tables, so this matches.
//!   Full semantic mremap (move/resize mappings + page-table coordination
//!   with the existing mmap/mprotect/munmap handlers) is tracked
//!   separately so the campaign close here is a pure parity win.
//! * Advisory ops (`mlock`, `munlock`, `mlockall`, `munlockall`,
//!   `madvise`, `msync`) are inherently no-ops in the symbolic VM —
//!   the symbolic return surfaces any binary that branches on it.

use super::stub_syscall;

// madvise(start, len, behavior) → long
stub_syscall!(NativeMadviseSyscall, "madvise", "syscall_stub_madvise", 3);
// mremap(addr, old_len, new_len, flags, new_addr) → long
stub_syscall!(NativeMremapSyscall, "mremap", "syscall_stub_mremap", 5);
// msync(start, len, flags) → long
stub_syscall!(NativeMsyncSyscall, "msync", "syscall_stub_msync", 3);
// mlock(start, len) → long
stub_syscall!(NativeMlockSyscall, "mlock", "syscall_stub_mlock", 2);
// munlock(start, len) → long
stub_syscall!(NativeMunlockSyscall, "munlock", "syscall_stub_munlock", 2);
// mlockall(flags) → long
stub_syscall!(
    NativeMlockallSyscall,
    "mlockall",
    "syscall_stub_mlockall",
    1
);
// munlockall() → long
stub_syscall!(
    NativeMunlockallSyscall,
    "munlockall",
    "syscall_stub_munlockall",
    0
);

#[cfg(test)]
#[path = "memory_extras_tests.rs"]
mod memory_extras_tests;
