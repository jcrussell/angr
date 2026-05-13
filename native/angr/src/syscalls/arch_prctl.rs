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

use super::{NativeSyscall, SyscallError, SyscallOutcome};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const ARCH_SET_GS: u64 = 0x1001;
const ARCH_SET_FS: u64 = 0x1002;
const ARCH_GET_FS: u64 = 0x1003;
const ARCH_GET_GS: u64 = 0x1004;
const EINVAL: u64 = 22;

pub struct NativeArchPrctlSyscall;

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
        if args.len() < 2 {
            return Err(SyscallError::Other(format!(
                "arch_prctl expected 2 args, got {}",
                args.len()
            )));
        }
        let code = args[0]
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument("arch_prctl code".into()))?;

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
                let addr = args[1]
                    .as_u64()
                    .ok_or_else(|| SyscallError::SymbolicArgument("arch_prctl addr".into()))?;
                let value = state
                    .get_register(reg)
                    .ok_or_else(|| SyscallError::Other(format!("arch_prctl: {reg} unreadable")))?;
                state
                    .memory_store(addr, value)
                    .map_err(|e| SyscallError::Other(format!("arch_prctl store: {e:?}")))?;
                Ok(SyscallOutcome::Continue { ret: 0 })
            }
            _ => Ok(SyscallOutcome::Continue { ret: EINVAL }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::state::RustSimState;
    use crate::symbolic::{RustBV, SymContext};

    fn fresh_state() -> RustSimState {
        RustSimState::new("amd64").expect("amd64 state")
    }

    #[test]
    fn arch_set_fs_writes_fs_const() {
        let h = NativeArchPrctlSyscall;
        let mut state = fresh_state();
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(ARCH_SET_FS as u128, 64),
                    RustBV::concrete(0xDEAD_BEEF, 64),
                ],
            )
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
            _ => panic!("expected Continue"),
        }
        let fs = state.get_register("fs_const").expect("fs_const readable");
        assert_eq!(fs.as_u64(), Some(0xDEAD_BEEF));
    }

    #[test]
    fn arch_set_gs_writes_gs_const() {
        let h = NativeArchPrctlSyscall;
        let mut state = fresh_state();
        h.call(
            &mut state,
            &[
                RustBV::concrete(ARCH_SET_GS as u128, 64),
                RustBV::concrete(0x1234_5678, 64),
            ],
        )
        .expect("ok");
        let gs = state.get_register("gs_const").expect("gs_const readable");
        assert_eq!(gs.as_u64(), Some(0x1234_5678));
    }

    #[test]
    fn arch_get_fs_stores_fs_const_at_addr() {
        let h = NativeArchPrctlSyscall;
        let mut state = fresh_state();
        // Pre-set fs_const so we have a known value to read back.
        state.set_register("fs_const", RustBV::concrete(0xCAFE_BABE, 64));
        // Map the destination page so memory_store succeeds.
        state.map_memory(0x4000, 0x1000, Permission::RW);

        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(ARCH_GET_FS as u128, 64),
                    RustBV::concrete(0x4000, 64),
                ],
            )
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
            _ => panic!("expected Continue"),
        }
        let stored = state.memory_load(0x4000, 8).expect("loadable");
        assert_eq!(stored.as_u64(), Some(0xCAFE_BABE));
    }

    #[test]
    fn arch_get_gs_stores_gs_const_at_addr() {
        let h = NativeArchPrctlSyscall;
        let mut state = fresh_state();
        state.set_register("gs_const", RustBV::concrete(0xFEED_FACE, 64));
        state.map_memory(0x5000, 0x1000, Permission::RW);

        h.call(
            &mut state,
            &[
                RustBV::concrete(ARCH_GET_GS as u128, 64),
                RustBV::concrete(0x5000, 64),
            ],
        )
        .expect("ok");
        let stored = state.memory_load(0x5000, 8).expect("loadable");
        assert_eq!(stored.as_u64(), Some(0xFEED_FACE));
    }

    #[test]
    fn unknown_code_returns_einval() {
        let h = NativeArchPrctlSyscall;
        let mut state = fresh_state();
        let outcome = h
            .call(
                &mut state,
                &[RustBV::concrete(0x9999, 64), RustBV::concrete(0, 64)],
            )
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, EINVAL),
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn symbolic_code_falls_back() {
        let h = NativeArchPrctlSyscall;
        let mut state = fresh_state();
        let ctx = SymContext::new();
        let sym = RustBV::symbolic(&ctx, "code", 64);
        let err = h
            .call(&mut state, &[sym, RustBV::concrete(0, 64)])
            .expect_err("must fall back");
        match err {
            SyscallError::SymbolicArgument(msg) => assert!(
                msg.contains("code"),
                "should name the symbolic arg, got {msg:?}"
            ),
            other => panic!("expected SymbolicArgument, got {other:?}"),
        }
    }

    #[test]
    fn get_with_symbolic_addr_falls_back() {
        let h = NativeArchPrctlSyscall;
        let mut state = fresh_state();
        let ctx = SymContext::new();
        let sym_addr = RustBV::symbolic(&ctx, "addr", 64);
        let err = h
            .call(
                &mut state,
                &[RustBV::concrete(ARCH_GET_FS as u128, 64), sym_addr],
            )
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::SymbolicArgument(_)));
    }

    #[test]
    fn get_unmapped_addr_falls_back() {
        let h = NativeArchPrctlSyscall;
        let mut state = fresh_state();
        // No page mapped at 0x9000_0000 → store fails.
        let err = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(ARCH_GET_FS as u128, 64),
                    RustBV::concrete(0x9000_0000, 64),
                ],
            )
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::Other(_)));
    }

    #[test]
    fn handler_metadata() {
        let h = NativeArchPrctlSyscall;
        assert_eq!(h.name(), "arch_prctl");
        assert_eq!(h.num_args(), 2);
    }
}
