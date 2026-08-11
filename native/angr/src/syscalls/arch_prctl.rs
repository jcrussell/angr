//! amd64 arch_prctl syscall handler (158).
//!
//! Mirrors `procedures/linux_kernel/arch_prctl.py`:
//!   * `code` (rdi) must be concrete; symbolic falls back to Python (which
//!     itself raises `SimValueError`).
//!   * `0x1001` ARCH_SET_GS → `gs_const = addr`, return 0.
//!   * `0x1002` ARCH_SET_FS → `fs_const = addr`, return 0.
//!   * `0x1003` ARCH_GET_FS → store `fs_const` (8B) at `*addr`, return 0.
//!   * `0x1004` ARCH_GET_GS → store `gs_const` (8B) at `*addr`, return 0.
//!   * Anything else → return 22 (EINVAL).
//!
//! Hit early during glibc's TLS initialization (the "frequently called
//! during early init" rationale in angr-4e3q).

use super::require_syscall_args;
use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const ARCH_SET_GS: u64 = 0x1001;
const ARCH_SET_FS: u64 = 0x1002;
const ARCH_GET_FS: u64 = 0x1003;
const ARCH_GET_GS: u64 = 0x1004;
const EINVAL: u64 = 22;

pub(crate) struct NativeArchPrctlSyscall;

impl NativeSyscall for NativeArchPrctlSyscall {
    fn name(&self) -> &'static str {
        "arch_prctl"
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
        let code = extract_concrete_arg(&args[0], "arch_prctl code")?;

        match code {
            ARCH_SET_FS => {
                if !state.set_register("fs_const", args[1].clone()) {
                    return Err(SyscallError::Other(
                        "arch_prctl: fs_const not writable".into(),
                    ));
                }
                Ok(SyscallOutcome::Continue { ret: 0 })
            }
            ARCH_SET_GS => {
                if !state.set_register("gs_const", args[1].clone()) {
                    return Err(SyscallError::Other(
                        "arch_prctl: gs_const not writable".into(),
                    ));
                }
                Ok(SyscallOutcome::Continue { ret: 0 })
            }
            ARCH_GET_FS | ARCH_GET_GS => {
                let reg = if code == ARCH_GET_FS {
                    "fs_const"
                } else {
                    "gs_const"
                };
                let addr = extract_concrete_arg(&args[1], "arch_prctl addr")?;
                let value = state
                    .get_register(reg)
                    .ok_or_else(|| SyscallError::Other(format!("arch_prctl: {reg} unreadable")))?;
                state.memory_store(addr, value)?;
                Ok(SyscallOutcome::Continue { ret: 0 })
            }
            _ => Ok(SyscallOutcome::Continue { ret: EINVAL }),
        }
    }
}

test_submod!("arch_prctl_tests.rs" => arch_prctl_tests);
