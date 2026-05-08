//! amd64 gettimeofday (96) and clock_gettime (228) syscall handlers.
//!
//! Mirror `procedures/posix/sim_time.py`. Each writes a fresh symbolic
//! `struct timeval` / `struct timespec` to the user-supplied pointer:
//!
//! * `gettimeofday(tv, tz)`:
//!   - `tv == 0` → return -1.
//!   - else write `tv_sec` (8B) + `tv_usec` (8B) at `*tv`, return 0.
//!   - `tz` is intentionally ignored (matches the Python procedure).
//!
//! * `clock_gettime(which_clock, ts)`:
//!   - `which_clock != 0` (CLOCK_REALTIME) → fall back to Python (its
//!     procedure raises `SimProcedureError`).
//!   - `ts == 0` → return -1.
//!   - else write `tv_sec` (8B) + `tv_nsec` (8B) at `*ts`, return 0.
//!
//! Notes:
//! * Python uses `state.solver.BVS(name, bits, key=("api", ...))`. The
//!   `key` enables Python's fresh-bvs deduplication when called from the
//!   *same* SimProcedure instance — angr generates a fresh symbol per
//!   call site otherwise. Native handlers don't have that machinery, so
//!   we always create a fresh `RustBV::symbolic(...)` per invocation.
//!   This mildly differs from Python's intra-procedure deduplication but
//!   matches its inter-call behavior, which is what binaries observe.
//! * `USE_SYSTEM_TIMES`: Python's path that returns `int(time.time())` is
//!   not implemented natively; the SimOption set is currently held on
//!   the Python proxy (see angr-t3l3) and not visible here. We always
//!   use the symbolic path — same default as `auto_load_libs=False`
//!   exploration today.
//! * Symbolic args (`tv`, `ts`, `which_clock`) fall back to Python.
//! * Unmapped destination pages cause `state.memory_store` to error,
//!   which we propagate as `SyscallError::Other` so Python (which
//!   auto-faults pages via the default plugin) can handle the store.

use super::{NativeSyscall, SyscallError, SyscallOutcome};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Linux: -1 as a 64-bit two's-complement value (sentinel "error" rax).
const NEG_ONE: u64 = u64::MAX;

const CLOCK_REALTIME: u64 = 0;

pub struct NativeGettimeofdaySyscall;

impl NativeSyscall for NativeGettimeofdaySyscall {
    fn name(&self) -> &'static str {
        "gettimeofday"
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
                "gettimeofday expected 2 args, got {}",
                args.len()
            )));
        }
        let tv = args[0]
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument("gettimeofday tv".into()))?;
        // tz is intentionally not extracted; Python ignores it too.

        if tv == 0 {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }

        let bits = state.arch().bits();
        // Bind the borrow to a local so it ends before we hit memory_store
        // (which takes a mutable borrow of the solver internally).
        let (tv_sec, tv_usec) = {
            let ctx = state.solver().borrow();
            (
                RustBV::symbolic(&ctx, "tv_sec", bits),
                RustBV::symbolic(&ctx, "tv_usec", bits),
            )
        };
        let stride = (bits / 8) as u64;
        state
            .memory_store(tv, tv_sec)
            .map_err(|e| SyscallError::Other(format!("gettimeofday tv_sec store: {e:?}")))?;
        state
            .memory_store(tv + stride, tv_usec)
            .map_err(|e| SyscallError::Other(format!("gettimeofday tv_usec store: {e:?}")))?;
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

pub struct NativeClockGettimeSyscall;

impl NativeSyscall for NativeClockGettimeSyscall {
    fn name(&self) -> &'static str {
        "clock_gettime"
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
                "clock_gettime expected 2 args, got {}",
                args.len()
            )));
        }
        let which_clock = args[0]
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument("clock_gettime which_clock".into()))?;
        if which_clock != CLOCK_REALTIME {
            // Python raises SimProcedureError for non-REALTIME clocks; let
            // it run so the same diagnostic path fires.
            return Err(SyscallError::Other(format!(
                "clock_gettime: unsupported clock {which_clock}"
            )));
        }
        let ts = args[1]
            .as_u64()
            .ok_or_else(|| SyscallError::SymbolicArgument("clock_gettime ts".into()))?;
        if ts == 0 {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }

        let bits = state.arch().bits();
        let (tv_sec, tv_nsec) = {
            let ctx = state.solver().borrow();
            (
                RustBV::symbolic(&ctx, "tv_sec", bits),
                RustBV::symbolic(&ctx, "tv_nsec", bits),
            )
        };
        let stride = (bits / 8) as u64;
        state
            .memory_store(ts, tv_sec)
            .map_err(|e| SyscallError::Other(format!("clock_gettime tv_sec store: {e:?}")))?;
        state
            .memory_store(ts + stride, tv_nsec)
            .map_err(|e| SyscallError::Other(format!("clock_gettime tv_nsec store: {e:?}")))?;
        Ok(SyscallOutcome::Continue { ret: 0 })
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

    // ---- gettimeofday ------------------------------------------------

    #[test]
    fn gettimeofday_null_tv_returns_neg_one() {
        let h = NativeGettimeofdaySyscall;
        let mut state = fresh_state();
        let outcome = h
            .call(
                &mut state,
                &[RustBV::concrete(0, 64), RustBV::concrete(0, 64)],
            )
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_ONE),
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn gettimeofday_writes_symbolic_struct_and_returns_zero() {
        let h = NativeGettimeofdaySyscall;
        let mut state = fresh_state();
        state.map_memory(0x4000, 0x1000, Permission::RW);

        let outcome = h
            .call(
                &mut state,
                &[RustBV::concrete(0x4000, 64), RustBV::concrete(0, 64)],
            )
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, 0),
            _ => panic!("expected Continue"),
        }
        // Both 8-byte slots must be symbolic (a fresh BVS per call), not
        // the zero-fill of a freshly mapped page.
        let tv_sec = state.memory_load(0x4000, 8).expect("loadable");
        let tv_usec = state.memory_load(0x4008, 8).expect("loadable");
        assert!(tv_sec.is_symbolic(), "tv_sec must be symbolic");
        assert!(tv_usec.is_symbolic(), "tv_usec must be symbolic");
    }

    #[test]
    fn gettimeofday_tz_may_be_symbolic() {
        let h = NativeGettimeofdaySyscall;
        let mut state = fresh_state();
        state.map_memory(0x4000, 0x1000, Permission::RW);
        let ctx = SymContext::new();
        let sym_tz = RustBV::symbolic(&ctx, "tz", 64);
        let outcome = h
            .call(&mut state, &[RustBV::concrete(0x4000, 64), sym_tz])
            .expect("symbolic tz must not block fast path");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
    }

    #[test]
    fn gettimeofday_symbolic_tv_falls_back() {
        let h = NativeGettimeofdaySyscall;
        let mut state = fresh_state();
        let ctx = SymContext::new();
        let sym_tv = RustBV::symbolic(&ctx, "tv", 64);
        let err = h
            .call(&mut state, &[sym_tv, RustBV::concrete(0, 64)])
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::SymbolicArgument(_)));
    }

    #[test]
    fn gettimeofday_unmapped_dest_falls_back() {
        let h = NativeGettimeofdaySyscall;
        let mut state = fresh_state();
        // No page mapped at 0xDEAD_0000.
        let err = h
            .call(
                &mut state,
                &[RustBV::concrete(0xDEAD_0000, 64), RustBV::concrete(0, 64)],
            )
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::Other(_)));
    }

    // ---- clock_gettime ----------------------------------------------

    #[test]
    fn clock_gettime_null_ts_returns_neg_one() {
        let h = NativeClockGettimeSyscall;
        let mut state = fresh_state();
        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(CLOCK_REALTIME as u128, 64),
                    RustBV::concrete(0, 64),
                ],
            )
            .expect("ok");
        match outcome {
            SyscallOutcome::Continue { ret } => assert_eq!(ret, NEG_ONE),
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn clock_gettime_writes_symbolic_struct_and_returns_zero() {
        let h = NativeClockGettimeSyscall;
        let mut state = fresh_state();
        state.map_memory(0x4000, 0x1000, Permission::RW);

        let outcome = h
            .call(
                &mut state,
                &[
                    RustBV::concrete(CLOCK_REALTIME as u128, 64),
                    RustBV::concrete(0x4000, 64),
                ],
            )
            .expect("ok");
        assert!(matches!(outcome, SyscallOutcome::Continue { ret: 0 }));
        let tv_sec = state.memory_load(0x4000, 8).expect("loadable");
        let tv_nsec = state.memory_load(0x4008, 8).expect("loadable");
        assert!(tv_sec.is_symbolic());
        assert!(tv_nsec.is_symbolic());
    }

    #[test]
    fn clock_gettime_non_realtime_falls_back() {
        let h = NativeClockGettimeSyscall;
        let mut state = fresh_state();
        // CLOCK_MONOTONIC = 1; native handler must defer to Python so its
        // SimProcedureError fires (matching Python behavior).
        let err = h
            .call(
                &mut state,
                &[RustBV::concrete(1, 64), RustBV::concrete(0x4000, 64)],
            )
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::Other(_)));
    }

    #[test]
    fn clock_gettime_symbolic_clock_falls_back() {
        let h = NativeClockGettimeSyscall;
        let mut state = fresh_state();
        let ctx = SymContext::new();
        let sym = RustBV::symbolic(&ctx, "which_clock", 64);
        let err = h
            .call(&mut state, &[sym, RustBV::concrete(0x4000, 64)])
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::SymbolicArgument(_)));
    }

    #[test]
    fn handler_metadata() {
        let g = NativeGettimeofdaySyscall;
        assert_eq!(g.name(), "gettimeofday");
        assert_eq!(g.num_args(), 2);
        let c = NativeClockGettimeSyscall;
        assert_eq!(c.name(), "clock_gettime");
        assert_eq!(c.num_args(), 2);
    }
}
