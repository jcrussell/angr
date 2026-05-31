//! exit / exit_group syscall handler.
//!
//! Mirrors `procedures/exit.rs::NativeExit`: signals the dispatcher to
//! deadend the state instead of advancing PC. The exit code (in rdi) is
//! intentionally not extracted — angr's `libc.exit` SimProcedure ignores
//! it for control flow as well, so we follow that contract.
//!
//! Unit struct: exit and exit_group share identical semantics here, so
//! the registry keys both syscall numbers to the same handler instead of
//! carrying a `name` field to distinguish them.

use super::{NativeSyscall, SyscallError, SyscallOutcome};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

pub struct NativeExitSyscall;

impl NativeSyscall for NativeExitSyscall {
    fn name(&self) -> &'static str {
        "exit"
    }

    fn num_args(&self) -> usize {
        0
    }

    /// `_state` and `_args` are required by the trait but unused: this
    /// handler signals deadend by returning `SyscallOutcome::Exit` without
    /// inspecting any state or argument. `num_args() == 0` also guarantees
    /// the dispatcher passes an empty `args` slice.
    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        Ok(SyscallOutcome::Exit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::RustSimState;
    use crate::symbolic::{RustBV, SymContext};

    /// Handler advertises 0 args so the dispatcher passes an empty
    /// slice. Verifies the SymbolicArgument path is unreachable for
    /// exit-style syscalls — even a hand-constructed symbolic arg
    /// is ignored and the outcome is still Exit.
    #[test]
    fn symbolic_arg_is_unreachable_and_ignored() {
        let h = NativeExitSyscall;
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        let ctx = SymContext::new();
        let sym = RustBV::symbolic(&ctx, "exit_status", 64);
        let outcome = h
            .call(&mut state, &[sym])
            .expect("exit handler ignores args, never errs");
        assert!(matches!(outcome, SyscallOutcome::Exit));
        assert_eq!(h.num_args(), 0, "extract_procedure_args will pass 0 args");
    }
}
