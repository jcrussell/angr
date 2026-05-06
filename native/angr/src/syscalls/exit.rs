//! exit / exit_group syscall handler.
//!
//! Mirrors `procedures/exit.rs::NativeExit`: signals the dispatcher to
//! deadend the state instead of advancing PC. The exit code (in rdi) is
//! intentionally not extracted — angr's `libc.exit` SimProcedure ignores
//! it for control flow as well, so we follow that contract.

use super::{NativeSyscall, SyscallError, SyscallOutcome};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

pub struct NativeExitSyscall {
    pub name: &'static str,
}

impl NativeSyscall for NativeExitSyscall {
    fn name(&self) -> &'static str {
        self.name
    }

    fn num_args(&self) -> usize {
        0
    }

    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        Ok(SyscallOutcome::Exit)
    }
}
