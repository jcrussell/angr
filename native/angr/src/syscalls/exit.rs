//! exit / exit_group syscall handler.
//!
//! Signals the dispatcher to deadend the state instead of advancing PC, and
//! ends it as Python's `linux_kernel/exit.py` does through
//! `SimProcedure.exit`: an `Ijk_Exit` history entry and the status for the
//! export's `terminate` event (`RustSimState::record_exit`,
//! angr-mx00p.132.7). The entry's address is the syscall's return pc, since
//! Rust never sees the kernel-object stub Python reports.
//!
//! Unit struct: exit and exit_group share identical semantics here, so
//! the registry keys both syscall numbers to the same handler instead of
//! carrying a `name` field to distinguish them.

use super::{NativeSyscall, SyscallError, SyscallOutcome};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

pub(crate) struct NativeExitSyscall;

impl NativeSyscall for NativeExitSyscall {
    fn name(&self) -> &'static str {
        "exit"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        state.record_exit(state.pc(), args.first().cloned());
        Ok(SyscallOutcome::Exit)
    }
}

test_submod!("exit_tests.rs" => exit_tests);
