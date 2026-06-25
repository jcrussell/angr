// Tests for time.rs (NativeTime libc procedure).
// Extracted from the parent module; see the `#[path]` attr in time.rs.
//
// The proc forwards to the same `fresh_monotonic_time` model as the time
// syscall, so these mirror the syscall coverage in
// `syscalls::sim_time_tests` but assert the *procedure* return convention
// (`Ok(Some(symbolic))`).
use super::*;
use crate::memory::Permission;
use crate::procedures::{NativeSimProcedure, ProcedureError};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

fn fresh_state() -> RustSimState {
    RustSimState::new("amd64").expect("amd64 state")
}

#[test]
fn test_name_and_arity() {
    assert_eq!(NativeTime.name(), "time");
    assert_eq!(NativeTime.num_args(), 1);
}

#[test]
fn test_null_pointer_returns_symbolic_and_updates_last_time() {
    let mut state = fresh_state();
    let ret = NativeTime
        .call(&mut state, &[RustBV::concrete(0, 64)])
        .expect("ok")
        .expect("time returns a value");
    assert!(ret.is_symbolic(), "time() return must be symbolic");
    assert!(state.last_time().is_some(), "last_time must be recorded");
}

#[test]
fn test_first_call_constrains_nonnegative() {
    let mut state = fresh_state();
    let ret = NativeTime
        .call(&mut state, &[RustBV::concrete(0, 64)])
        .expect("ok")
        .expect("value");
    let min = state.min(&ret, true).expect("min computable");
    assert!(min as i64 >= 0, "first time() must be SGE 0; got min={min}");
}

#[test]
fn test_writes_symbolic_to_pointer() {
    let mut state = fresh_state();
    state.map_memory(0x4000, 0x1000, Permission::RW);
    let ret = NativeTime
        .call(&mut state, &[RustBV::concrete(0x4000, 64)])
        .expect("ok")
        .expect("value");
    let stored = state.memory_load(0x4000, 8).expect("loadable");
    assert!(stored.is_symbolic(), "stored *tloc must be symbolic");
    assert_eq!(ret.width(), stored.width());
}

#[test]
fn test_two_calls_are_solver_distinct() {
    // Mirror the syscall regression guard (angr-8o7w): two time() returns on
    // one path must be satisfiably unequal (fresh_symbolic appends a counter).
    let mut state = fresh_state();
    let r1 = NativeTime
        .call(&mut state, &[RustBV::concrete(0, 64)])
        .expect("ok")
        .expect("value");
    let r2 = NativeTime
        .call(&mut state, &[RustBV::concrete(0, 64)])
        .expect("ok")
        .expect("value");
    let solver = z3::Solver::new();
    solver.assert(r1.to_z3_ast().eq(r2.to_z3_ast()).not());
    assert_eq!(
        solver.check(),
        z3::SatResult::Sat,
        "two time() returns must be solver-distinct"
    );
}

#[test]
fn test_symbolic_pointer_falls_back_to_python() {
    // `pointer: concrete` => a symbolic pointer yields ProcedureError so the
    // run loop defers to Python (matching the syscall's extract_concrete_arg).
    let mut state = fresh_state();
    let ptr = {
        let ctx = state.solver().borrow();
        RustBV::symbolic(&ctx, "tloc", 64)
    };
    let result = NativeTime.call(&mut state, &[ptr]);
    assert!(
        matches!(result, Err(ProcedureError::SymbolicArgument(_))),
        "symbolic *tloc must fall back to Python"
    );
}
