//! Resource-limit syscall handlers — `getrlimit`, `setrlimit`,
//! `prlimit64` (and the legacy `ugetrlimit` alias used by x86/arm).
//!
//! * `getrlimit(resource, rlim)` mirrors
//!   `procedures/linux_kernel/getrlimit.py`:
//!   - if `resource == RLIMIT_STACK (3)`: write `8388608` (8 bytes,
//!     little-endian) at `*rlim` for `rlim_cur`, and a fresh symbolic
//!     64-bit BV at `*rlim + 8` for `rlim_max`. Return `0`.
//!   - else: return a fresh symbolic BV (the Python code uses
//!     `sizeof[int]` = 32 bits; the native handler uses
//!     `arch().bits()` instead — the dispatcher writes whatever width
//!     into the return register, and binaries observe a fresh symbol
//!     either way).
//!
//! * `setrlimit` and `prlimit64` have no Python `SimProcedure` — the
//!   unhandled-syscall path falls through to
//!   `procedures/stubs/syscall_stub.py::syscall`, which emits
//!   `state.solver.Unconstrained("syscall_stub_<name>", returnty.size,
//!   ...)`. Native handlers mirror that via
//!   `SyscallOutcome::ContinueSymbolic`.
//!
//! ## ugetrlimit (x86/arm syscall 191)
//!
//! `procedures/linux_kernel/getrlimit.py` defines
//! `class ugetrlimit(getrlimit): pass` — the LFS-style 32-bit-uid_t
//! variant. Semantics are identical, so x86/arm registration aliases
//! syscall 191 to `NativeGetrlimitSyscall`. The `name()` still returns
//! `"getrlimit"` (same as the Python subclass would inherit if
//! introspected via `mro()`); per-arch tests assert this alias holds.

use super::require_syscall_args;
use super::{
    NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg, fresh_symbolic, stub_syscall,
};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// `<sys/resource.h>` RLIMIT_STACK index — the only branch the Python
/// `getrlimit` impl writes a concrete value for.
const RLIMIT_STACK: u64 = 3;
/// Python writes `8 * 1024 * 1024 = 8388608` as `rlim_cur` for
/// RLIMIT_STACK (see `procedures/linux_kernel/getrlimit.py:13`).
const RLIMIT_STACK_CUR: u128 = 8 * 1024 * 1024;

pub(crate) struct NativeGetrlimitSyscall;

impl NativeSyscall for NativeGetrlimitSyscall {
    fn name(&self) -> &'static str {
        "getrlimit"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        require_syscall_args!(self, args);
        let resource = extract_concrete_arg(&args[0], "getrlimit resource")?;
        let rlim = extract_concrete_arg(&args[1], "getrlimit rlim")?;

        if resource == RLIMIT_STACK {
            // rlim_cur (8 bytes) = 8388608, rlim_max (8 bytes) = fresh symbolic.
            let cur = RustBV::concrete(RLIMIT_STACK_CUR, 64);
            let max = {
                let ctx = state.solver().borrow();
                fresh_symbolic(&ctx, "rlim_max", 64)
            };
            state.memory_store(rlim, cur)?;
            // `rlim` is unchecked `extract_concrete_arg` output and release
            // builds disable overflow checks, so wrap explicitly rather than
            // panic only under CI's `release-checked` profile
            // (`invariant-proc-address-arith-wrapping`).
            state.memory_store(rlim.wrapping_add(8), max)?;
            return Ok(SyscallOutcome::Continue { ret: 0 });
        }

        // Non-RLIMIT_STACK branch: Python returns a fresh symbolic. Use
        // arch().bits() to fully populate the return register width;
        // this matches every other stub-style native handler.
        let bits = state.arch().bits();
        let ret = {
            let ctx = state.solver().borrow();
            fresh_symbolic(&ctx, "rlimit", bits)
        };
        Ok(SyscallOutcome::ContinueSymbolic { ret })
    }
}

// setrlimit(resource, rlim) → int — no Python SimProcedure.
stub_syscall!(
    NativeSetrlimitSyscall,
    "setrlimit",
    "syscall_stub_setrlimit",
    2
);
// prlimit64(pid, resource, new_rlim*, old_rlim*) → int — no Python SimProcedure.
stub_syscall!(
    NativePrlimit64Syscall,
    "prlimit64",
    "syscall_stub_prlimit64",
    4
);

test_submod!("rlimit_tests.rs" => rlimit_tests);
