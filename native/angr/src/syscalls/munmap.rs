//! amd64 munmap syscall handler.
//!
//! Mirrors `procedures/linux_kernel/munmap.py`, which is intentionally a
//! no-op that always returns 0. Real `munmap(2)` would unmap pages, but
//! angr's symbolic execution treats memory as monotonically growing —
//! freeing pages would lose constraints on data already in those pages.
//! Matching the Python no-op semantics keeps native dispatch identical
//! to the fallback path.
//!
//! Symbolic args are still accepted (concrete and symbolic both return
//! 0), since the Python implementation does not look at them either.

use super::{NativeSyscall, SyscallError, SyscallOutcome};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

pub struct NativeMunmapSyscall;

impl NativeSyscall for NativeMunmapSyscall {
    fn name(&self) -> &'static str {
        "munmap"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::SymContext;

    #[test]
    fn returns_zero_for_concrete_args() {
        let h = NativeMunmapSyscall;
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        let args = vec![RustBV::concrete(0x1000, 64), RustBV::concrete(0x1000, 64)];
        let outcome = h.call(&mut state, &args).expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
            _ => panic!("expected Continue {{ ret: 0 }}"),
        }
    }

    #[test]
    fn returns_zero_for_symbolic_args() {
        // Python munmap is a no-op that ignores its args, so symbolic args
        // should produce the same result as concrete ones — no fallback.
        let h = NativeMunmapSyscall;
        let mut state = RustSimState::new("amd64").expect("amd64 state");
        let ctx = SymContext::new();
        let args = vec![
            RustBV::symbolic(&ctx, "addr", 64),
            RustBV::symbolic(&ctx, "length", 64),
        ];
        let outcome = h.call(&mut state, &args).expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
            _ => panic!("expected Continue {{ ret: 0 }}"),
        }
    }

    #[test]
    fn handler_metadata() {
        let h = NativeMunmapSyscall;
        assert_eq!(h.name(), "munmap");
        assert_eq!(h.num_args(), 2);
    }
}
