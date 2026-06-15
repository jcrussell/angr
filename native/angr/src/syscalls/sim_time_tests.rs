//! Tests for `sim_time.rs` — gettimeofday / clock_gettime / time syscall handlers.

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

    let solver = z3::Solver::new();
    solver.assert(r1.to_z3_ast().eq(r2.to_z3_ast()).not());
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
