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

pub(crate) struct NativeMunmapSyscall;

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

test_submod!("munmap_tests.rs" => munmap_tests);
