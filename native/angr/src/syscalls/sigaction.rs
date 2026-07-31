//! amd64 rt_sigaction syscall handler (13).
//!
//! Mirrors `procedures/linux_kernel/sigaction.py::rt_sigaction`:
//!   * Pure no-op: ignore `act` / `oldact` / `sigsetsize`; return 0.
//!   * Special case: `signum == 33` → return -EINVAL (-22). The Python
//!     procedure routes this through `state.libc.ret_errno("EINVAL")`,
//!     which (for syscalls) returns `-EINVAL`. Match that bit-for-bit by
//!     storing `(-22 as i64) as u64` in rax.
//!
//! Symbolic `signum` falls back to Python so it can fork on the
//! `signum == 33` constraint. The other three args are intentionally
//! ignored; they may be symbolic without forcing fallback.

use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// `-EINVAL` as a 64-bit two's-complement value (rax bit pattern).
const NEG_EINVAL: u64 = (-22_i64) as u64;

pub(crate) struct NativeRtSigactionSyscall;

impl NativeSyscall for NativeRtSigactionSyscall {
    fn name(&self) -> &'static str {
        "rt_sigaction"
    }

    fn num_args(&self) -> usize {
        // Python signature: signum, act, oldact, sigsetsize. We only inspect
        // signum, but the dispatcher must extract all four to honor the
        // syscall ABI contract (`extract_syscall_args` count).
        4
    }

    fn call(
        &self,
        _state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        if args.is_empty() {
            return Err(SyscallError::Other(
                "rt_sigaction expected ≥1 arg, got 0".into(),
            ));
        }
        let signum = extract_concrete_arg(&args[0], "rt_sigaction signum")?;
        if signum == 33 {
            return Ok(SyscallOutcome::Continue { ret: NEG_EINVAL });
        }
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

#[cfg(test)]
#[path = "sigaction_tests.rs"]
mod tests;
