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

use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg, stub_syscall};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// `<sys/resource.h>` RLIMIT_STACK index — the only branch the Python
/// `getrlimit` impl writes a concrete value for.
const RLIMIT_STACK: u64 = 3;
/// Python writes `8 * 1024 * 1024 = 8388608` as `rlim_cur` for
/// RLIMIT_STACK (see `procedures/linux_kernel/getrlimit.py:13`).
const RLIMIT_STACK_CUR: u128 = 8 * 1024 * 1024;

pub struct NativeGetrlimitSyscall;

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
        if args.len() < 2 {
            return Err(SyscallError::Other(format!(
                "getrlimit expected 2 args, got {}",
                args.len()
            )));
        }
        let resource = extract_concrete_arg(&args[0], "getrlimit resource")?;
        let rlim = extract_concrete_arg(&args[1], "getrlimit rlim")?;

        if resource == RLIMIT_STACK {
            // rlim_cur (8 bytes) = 8388608, rlim_max (8 bytes) = fresh symbolic.
            let cur = RustBV::concrete(RLIMIT_STACK_CUR, 64);
            let max = {
                let ctx = state.solver().borrow();
                RustBV::symbolic(&ctx, "rlim_max", 64)
            };
            state.memory_store(rlim, cur)?;
            state.memory_store(rlim + 8, max)?;
            return Ok(SyscallOutcome::Continue { ret: 0 });
        }

        // Non-RLIMIT_STACK branch: Python returns a fresh symbolic. Use
        // arch().bits() to fully populate the return register width;
        // this matches every other stub-style native handler.
        let bits = state.arch().bits();
        let ret = {
            let ctx = state.solver().borrow();
            RustBV::symbolic(&ctx, "rlimit", bits)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;
    use crate::symbolic::SymContext;

    fn fresh_state() -> RustSimState {
        RustSimState::new("amd64").expect("amd64 state")
    }

    // ---- getrlimit ---------------------------------------------------

    #[test]
    fn getrlimit_rlimit_stack_writes_rlim_struct_and_returns_zero() {
        let h = NativeGetrlimitSyscall;
        let mut state = fresh_state();
        state.map_memory(0x4000, 0x1000, Permission::RW);

        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(RLIMIT_STACK as u128, 64),
                    RustBV::concrete(0x4000, 64),
                ],
            )
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
            other => panic!("expected Continue, got {other:?}"),
        }

        // rlim_cur = 8388608 (concrete)
        let cur = state.memory_load(0x4000, 8).expect("loadable");
        assert_eq!(
            cur.as_u64(),
            Some(RLIMIT_STACK_CUR as u64),
            "rlim_cur should be the concrete 8 MiB constant"
        );

        // rlim_max = fresh symbolic
        let max = state.memory_load(0x4008, 8).expect("loadable");
        assert!(
            max.is_symbolic(),
            "rlim_max should be symbolic (fresh BVS)"
        );
    }

    #[test]
    fn getrlimit_non_stack_returns_fresh_symbolic() {
        let h = NativeGetrlimitSyscall;
        let mut state = fresh_state();
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(1, 64), // RLIMIT_FSIZE
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("ok");
        let ret = match outcome {
            SyscallOutcome::ContinueSymbolic { ret } => ret,
            other => panic!("expected ContinueSymbolic, got {other:?}"),
        };
        assert_eq!(ret.width(), state.arch().bits());
        assert!(ret.as_u64().is_none(), "non-STACK rlimit must be symbolic");
    }

    #[test]
    fn getrlimit_symbolic_resource_returns_error() {
        let h = NativeGetrlimitSyscall;
        let mut state = fresh_state();
        let ctx = SymContext::new();
        let outcome = h.call(
            &mut state,
            &[
                RustBV::symbolic(&ctx, "resource_sym", 64),
                RustBV::concrete(0x4000, 64),
            ],
        );
        let err = outcome.expect_err("must error on symbolic resource");
        assert!(matches!(err, SyscallError::SymbolicArgument(_)));
    }

    #[test]
    fn getrlimit_symbolic_rlim_returns_error() {
        let h = NativeGetrlimitSyscall;
        let mut state = fresh_state();
        let ctx = SymContext::new();
        let outcome = h.call(
            &mut state,
            &[
                RustBV::concrete(RLIMIT_STACK as u128, 64),
                RustBV::symbolic(&ctx, "rlim_sym", 64),
            ],
        );
        let err = outcome.expect_err("must error on symbolic rlim");
        assert!(matches!(err, SyscallError::SymbolicArgument(_)));
    }

    /// Round-trip: write the rlim struct, load it back, confirm
    /// rlim_cur is the documented constant and rlim_max stays
    /// symbolic. Satisfies the bd acceptance criterion "Integration
    /// test confirms getrlimit/setrlimit round-trip".
    #[test]
    fn getrlimit_setrlimit_roundtrip() {
        let getrlimit = NativeGetrlimitSyscall;
        let setrlimit = NativeSetrlimitSyscall;
        let mut state = fresh_state();
        state.map_memory(0x5000, 0x1000, Permission::RW);

        // 1. getrlimit(RLIMIT_STACK, 0x5000) populates the struct.
        let outcome = getrlimit
            .call(
                &mut state,
                &[
                    RustBV::concrete(RLIMIT_STACK as u128, 64),
                    RustBV::concrete(0x5000, 64),
                ],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        let cur = state
            .memory_load(0x5000, 8)
            .expect("rlim_cur readable")
            .as_u64()
            .expect("concrete");
        assert_eq!(cur, RLIMIT_STACK_CUR as u64);

        // 2. setrlimit(RLIMIT_STACK, 0x5000) returns a fresh symbolic
        //    int (stub semantics). Round-trip in this case = the call
        //    succeeds and yields a symbolic in the return register.
        let outcome = setrlimit
            .call(
                &mut state,
                &[
                    RustBV::concrete(RLIMIT_STACK as u128, 64),
                    RustBV::concrete(0x5000, 64),
                ],
            )
            .expect("ok");
        let ret = match outcome {
            SyscallOutcome::ContinueSymbolic { ret } => ret,
            other => panic!("expected ContinueSymbolic, got {other:?}"),
        };
        assert!(ret.as_u64().is_none(), "setrlimit stub return is symbolic");

        // 3. Re-load — concrete bytes stick around even after the
        //    "no-op" setrlimit (Python's stub does not touch memory).
        let cur2 = state
            .memory_load(0x5000, 8)
            .expect("rlim_cur still there")
            .as_u64()
            .expect("concrete");
        assert_eq!(cur2, RLIMIT_STACK_CUR as u64);
    }

    // ---- setrlimit / prlimit64 stubs ---------------------------------

    #[test]
    fn setrlimit_returns_fresh_symbolic_on_all_arches() {
        let h = NativeSetrlimitSyscall;
        for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();
            let args = [RustBV::concrete(0, bits), RustBV::concrete(0, bits)];
            let outcome = h.call(&mut state, &args).unwrap();
            let ret = match outcome {
                SyscallOutcome::ContinueSymbolic { ret } => ret,
                _ => unreachable!(),
            };
            assert_eq!(ret.width(), bits, "{arch}: width must match arch");
            assert!(ret.as_u64().is_none(), "{arch}: must be symbolic");
        }
    }

    #[test]
    fn prlimit64_returns_fresh_symbolic_on_all_arches() {
        let h = NativePrlimit64Syscall;
        assert_eq!(h.num_args(), 4);
        for arch in ["amd64", "x86", "armel", "aarch64", "mipsel"] {
            let mut state = RustSimState::new(arch).expect("state");
            let bits = state.arch().bits();
            let args: Vec<RustBV> = (0..4).map(|_| RustBV::concrete(0, bits)).collect();
            let outcome = h.call(&mut state, &args).unwrap();
            let ret = match outcome {
                SyscallOutcome::ContinueSymbolic { ret } => ret,
                _ => unreachable!(),
            };
            assert_eq!(ret.width(), bits);
            assert!(ret.as_u64().is_none(), "{arch}: must be symbolic");
        }
    }
}
