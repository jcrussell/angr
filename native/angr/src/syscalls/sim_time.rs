//! amd64 time (201), gettimeofday (96), and clock_gettime (228) syscall handlers.
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

use super::{NativeSyscall, SyscallError, SyscallOutcome, extract_concrete_arg, fresh_symbolic};
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
        let tv = extract_concrete_arg(&args[0], "gettimeofday tv")?;
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
                fresh_symbolic(&ctx, "tv_sec", bits),
                fresh_symbolic(&ctx, "tv_usec", bits),
            )
        };
        let stride = (bits / 8) as u64;
        state.memory_store(tv, tv_sec)?;
        state.memory_store(tv + stride, tv_usec)?;
        Ok(SyscallOutcome::Continue { ret: 0 })
    }
}

/// time(pointer) — return a fresh symbolic time_t in rax, optionally
/// store it at *pointer. Mirrors `procedures/linux_kernel/time.py`:
///
/// * `result := BVS("sys_time", arch.bits)`
/// * if `state.last_time` is `Some(prev)`: constrain `result.SGE(prev)`
/// * else: constrain `result.SGE(0)`
/// * `state.last_time := result`
/// * if `pointer != 0` (concrete): store `result` at `*pointer`
/// * return `result` via rax (ContinueSymbolic)
///
/// Symbolic `pointer`: fall back to Python so its `condition=(pointer != 0)`
/// store logic runs. (Most binaries pass NULL or a concrete stack address.)
pub struct NativeTimeSyscall;

impl NativeSyscall for NativeTimeSyscall {
    fn name(&self) -> &'static str {
        "time"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<SyscallOutcome, SyscallError> {
        if args.is_empty() {
            return Err(SyscallError::Other(format!(
                "time expected 1 arg, got {}",
                args.len()
            )));
        }
        let pointer = extract_concrete_arg(&args[0], "time pointer")?;

        let bits = state.arch().bits();
        let (sys_time, monotonic_constraint) = {
            let ctx = state.solver().borrow();
            let sys_time = fresh_symbolic(&ctx, "sys_time", bits);
            let zero = RustBV::concrete(0, bits);
            // Monotonic constraint: sys_time >= last_time (or >= 0 first call).
            let lower = state.last_time().cloned().unwrap_or(zero);
            let cmp = sys_time.sge(&lower, &ctx);
            (sys_time, cmp)
        };
        state.add_constraint(monotonic_constraint);

        if pointer != 0 {
            state.memory_store(pointer, sys_time.clone())?;
        }

        state.set_last_time(sys_time.clone());
        Ok(SyscallOutcome::ContinueSymbolic { ret: sys_time })
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
        let which_clock = extract_concrete_arg(&args[0], "clock_gettime which_clock")?;
        if which_clock != CLOCK_REALTIME {
            // Python raises SimProcedureError for non-REALTIME clocks; let
            // it run so the same diagnostic path fires.
            return Err(SyscallError::Other(format!(
                "clock_gettime: unsupported clock {which_clock}"
            )));
        }
        let ts = extract_concrete_arg(&args[1], "clock_gettime ts")?;
        if ts == 0 {
            return Ok(SyscallOutcome::Continue { ret: NEG_ONE });
        }

        let bits = state.arch().bits();
        let (tv_sec, tv_nsec) = {
            let ctx = state.solver().borrow();
            (
                fresh_symbolic(&ctx, "tv_sec", bits),
                fresh_symbolic(&ctx, "tv_nsec", bits),
            )
        };
        let stride = (bits / 8) as u64;
        state.memory_store(ts, tv_sec)?;
        state.memory_store(ts + stride, tv_nsec)?;
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
        assert!(matches!(err, SyscallError::Memory(_)));
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
        let t = NativeTimeSyscall;
        assert_eq!(t.name(), "time");
        assert_eq!(t.num_args(), 1);
    }

    // ---- time --------------------------------------------------------

    #[test]
    fn time_null_pointer_returns_symbolic_and_does_not_store() {
        let h = NativeTimeSyscall;
        let mut state = fresh_state();
        let outcome = h.call(&mut state, &[RustBV::concrete(0, 64)]).expect("ok");
        match outcome {
            SyscallOutcome::ContinueSymbolic { ret } => {
                assert!(ret.is_symbolic(), "time return must be symbolic");
                assert_eq!(ret.width(), 64);
            }
            _ => panic!("expected ContinueSymbolic"),
        }
        // last_time updated.
        assert!(state.last_time().is_some());
    }

    #[test]
    fn time_two_calls_are_solver_distinct() {
        // Regression guard for angr-8o7w: two time() returns on one path must
        // be satisfiably unequal. A fixed Z3 name would alias both to the same
        // `new_const`, making `ret1 != ret2` unsatisfiable — fresh_symbolic
        // appends symbol_counter so each return is a distinct Z3 term.
        let h = NativeTimeSyscall;
        let mut state = fresh_state();
        let r1 = match h.call(&mut state, &[RustBV::concrete(0, 64)]).expect("ok") {
            SyscallOutcome::ContinueSymbolic { ret } => ret,
            _ => panic!("expected ContinueSymbolic"),
        };
        let r2 = match h.call(&mut state, &[RustBV::concrete(0, 64)]).expect("ok") {
            SyscallOutcome::ContinueSymbolic { ret } => ret,
            _ => panic!("expected ContinueSymbolic"),
        };
        use z3::ast::Ast as _;
        let solver = z3::Solver::new();
        solver.assert(&r1.to_z3_ast()._eq(&r2.to_z3_ast()).not());
        assert_eq!(
            solver.check(),
            z3::SatResult::Sat,
            "two time() returns must be solver-distinct"
        );
    }

    #[test]
    fn time_writes_symbolic_to_pointer() {
        let h = NativeTimeSyscall;
        let mut state = fresh_state();
        state.map_memory(0x4000, 0x1000, Permission::RW);
        let outcome = h
            .call(&mut state, &[RustBV::concrete(0x4000, 64)])
            .expect("ok");
        let ret = match outcome {
            SyscallOutcome::ContinueSymbolic { ret } => ret,
            _ => panic!("expected ContinueSymbolic"),
        };
        let stored = state.memory_load(0x4000, 8).expect("loadable");
        assert!(stored.is_symbolic(), "stored value must be symbolic");
        // The stored value is the same BV that was returned.
        assert_eq!(ret.width(), stored.width());
    }

    #[test]
    fn time_first_call_constrains_nonnegative() {
        let h = NativeTimeSyscall;
        let mut state = fresh_state();
        let outcome = h.call(&mut state, &[RustBV::concrete(0, 64)]).expect("ok");
        let ret = match outcome {
            SyscallOutcome::ContinueSymbolic { ret } => ret,
            _ => panic!("expected ContinueSymbolic"),
        };
        // Min should be >= 0 (signed).
        let min = state.min(&ret, true).expect("min computable");
        assert!(min as i64 >= 0, "first time() must be SGE 0; got min={min}");
    }

    #[test]
    fn time_consecutive_calls_are_monotonic() {
        let h = NativeTimeSyscall;
        let mut state = fresh_state();
        let first = match h.call(&mut state, &[RustBV::concrete(0, 64)]).expect("ok") {
            SyscallOutcome::ContinueSymbolic { ret } => ret,
            _ => panic!("expected ContinueSymbolic"),
        };
        // Pin the first call to a concrete value to make the monotonicity
        // constraint testable: first == 100.
        let pin = {
            let ctx = state.solver().borrow();
            first.eq(&RustBV::concrete(100, 64), &ctx)
        };
        state.add_constraint(pin);

        let second = match h.call(&mut state, &[RustBV::concrete(0, 64)]).expect("ok") {
            SyscallOutcome::ContinueSymbolic { ret } => ret,
            _ => panic!("expected ContinueSymbolic"),
        };
        // second >= first, and first is pinned to 100, so second >= 100.
        let min = state.min(&second, true).expect("min computable");
        assert!(
            min as i64 >= 100,
            "second time() must be >= first; got min={min}"
        );
    }

    #[test]
    fn time_symbolic_pointer_falls_back() {
        let h = NativeTimeSyscall;
        let mut state = fresh_state();
        let ctx = SymContext::new();
        let sym = RustBV::symbolic(&ctx, "ptr", 64);
        let err = h.call(&mut state, &[sym]).expect_err("must fall back");
        assert!(matches!(err, SyscallError::SymbolicArgument(_)));
    }

    #[test]
    fn time_unmapped_pointer_falls_back() {
        let h = NativeTimeSyscall;
        let mut state = fresh_state();
        let err = h
            .call(&mut state, &[RustBV::concrete(0xDEAD_0000, 64)])
            .expect_err("must fall back");
        assert!(matches!(err, SyscallError::Memory(_)));
    }
}
