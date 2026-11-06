//! Resource-limit syscall handlers — `getrlimit`, `setrlimit`,
//! `prlimit64` (and the legacy `ugetrlimit` alias used by x86/arm).
//!
//! * `getrlimit(resource, rlim)` mirrors
//!   `procedures/linux_kernel/getrlimit.py`:
//!   - if `resource == RLIMIT_STACK (3)`: write `8388608` at `*rlim` for
//!     `rlim_cur`, and a fresh symbolic BV at `*rlim + bits/8` for
//!     `rlim_max`. `struct rlimit` is two `unsigned long`s, so both
//!     fields are `arch().bits()` wide and stored in the arch's memory
//!     endianness (angr-vmi2l.152). Return `0`.
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
    NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg, stub_syscall,
    symbolic_outcome,
};
use crate::procedures::fresh_tracked_symbol;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// `<sys/resource.h>` RLIMIT_STACK index — the only branch the Python
/// `getrlimit` impl writes a concrete value for.
const RLIMIT_STACK: u64 = 3;
/// Python writes `8 * 1024 * 1024 = 8388608` as `rlim_cur` for
/// RLIMIT_STACK (see `getrlimit.run` in `procedures/linux_kernel/getrlimit.py`).
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
            // rlim_cur = 8388608, rlim_max = fresh symbolic, each an
            // `unsigned long` (angr-vmi2l.152).
            let bits = state.arch().bits();
            let cur = RustBV::concrete(RLIMIT_STACK_CUR, bits);
            let max = fresh_tracked_symbol(state, "rlim_max", bits, &["api", "rlimit", "stack"]);
            let stride = u64::from(bits / 8);
            state.memory_store(rlim, cur)?;
            // `rlim` is unchecked `extract_concrete_arg` output and release
            // builds disable overflow checks, so wrap explicitly rather than
            // panic only under CI's `release-checked` profile
            // (`invariant-proc-address-arith-wrapping`).
            state.memory_store(rlim.wrapping_add(stride), max)?;
            return Ok(SyscallOutcome::Continue { ret: 0 });
        }

        // Non-RLIMIT_STACK branch: Python returns a fresh symbolic. Sharing
        // `symbolic_outcome` is what keeps the width at `arch().bits()`, so
        // the return register is fully populated like every other stub-style
        // native handler.
        Ok(symbolic_outcome(
            state,
            "rlimit",
            &["api", "rlimit", "other"],
        ))
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
