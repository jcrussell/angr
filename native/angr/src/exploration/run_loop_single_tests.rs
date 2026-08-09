//! Unit tests for `step_one`'s pre-step routing — the single-threaded
//! stepping decision point (angr-c7xno.34).
//!
//! Scope is deliberately the **lift-free prefix** of `step_one`: every arm
//! that reaches a verdict before `step_state_with_skip` is called. That is the
//! ordered chain of callable-predicate bounces, address-based find/avoid
//! routing, and the Python SimProcedure fallback — all of which return a
//! `StepOutcome` without ever lifting a block, so they need no binary, no
//! Python callbacks that actually fire, and no live project.
//!
//! Everything past that point (the interpreter step itself, successor
//! classification, the deferred-fork materialization inside the
//! `NeedCallback` arm) needs a real lift and stays covered by the Python e2e
//! suite in `tests/engines/rust/`. The parallel mirrors of these same routing
//! decisions live in `run_loop_worker_tests.rs` / `run_loop_wave_tests.rs`.

use super::*;

use crate::callbacks::PythonCallbacks;
use crate::stash::{STASH_AVOID, STASH_FOUND, STASH_PRUNED};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// A fresh manager plus a SAT state parked at `pc`, registered in no stash —
/// exactly what the driver hands `step_one` after popping it from ACTIVE.
fn mgr_and_state(pc: u64) -> (RustExplorationManager, RustSimState, u64) {
    let mgr = RustExplorationManager::new("amd64", None).unwrap();
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_pc(pc);
    let id = state.state_id();
    (mgr, state, id)
}

/// Same, but the state carries two mutually exclusive constraints so
/// `satisfiable()` is false — the fixture the find-address UNSAT prune needs.
fn unsat_state_at(pc: u64) -> RustSimState {
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_pc(pc);
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "single_unsat_x", 64)
    };
    state.set_register("rax", x.clone());
    for witness in [1u128, 2u128] {
        let c = {
            let s = state.solver().borrow();
            x.eq(&RustBV::concrete(witness, 64), &s)
        };
        state.add_constraint(c);
    }
    assert!(!state.satisfiable(), "fixture must be UNSAT");
    state
}

/// `StepOutcome` carries a `RustSimState` / `PendingCallback` and so is not
/// `Debug`; name the variant by hand for assertion messages.
fn variant_of(outcome: &StepOutcome) -> &'static str {
    match outcome {
        StepOutcome::Routed => "Routed",
        StepOutcome::NeedCallback(_) => "NeedCallback",
        StepOutcome::Successors(_) => "Successors",
        StepOutcome::Terminal(_) => "Terminal",
    }
}

fn assert_routed(outcome: StepOutcome) {
    assert!(
        matches!(outcome, StepOutcome::Routed),
        "expected StepOutcome::Routed, got {}",
        variant_of(&outcome)
    );
}

fn expect_callback(outcome: StepOutcome) -> PendingCallback {
    let named = variant_of(&outcome);
    match outcome {
        StepOutcome::NeedCallback(pending) => pending,
        _ => panic!("expected StepOutcome::NeedCallback, got {named}"),
    }
}

#[test]
fn avoid_address_routes_to_avoid_stash() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, _id) = mgr_and_state(0x40_1000);
        mgr.set_avoid_addrs(vec![0x40_1000]);
        let cb = PythonCallbacks::new();

        assert_routed(mgr.step_one(&cb, state).unwrap());

        assert_eq!(mgr.stash_count(STASH_AVOID), 1, "avoid pc parked in AVOID");
        assert_eq!(mgr.found_count(), 0);
        assert_eq!(mgr.active_count(), 0, "avoided state never re-enters ACTIVE");
    });
}

#[test]
fn find_address_routes_satisfiable_state_to_found_stash() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, _id) = mgr_and_state(0x40_2000);
        mgr.set_find_addrs(vec![0x40_2000]);
        let cb = PythonCallbacks::new();

        assert_routed(mgr.step_one(&cb, state).unwrap());

        assert_eq!(mgr.stash_count(STASH_FOUND), 1);
        assert_eq!(mgr.found_count(), 1);
    });
}

/// A state can reach a find address down an infeasible path; `step_one` must
/// prune it rather than report a bogus solution. Mirrors
/// `run_loop_wave_tests.rs::unsat_successor_at_find_pc_routes_to_pruned` on
/// the single-threaded side.
#[test]
fn unsat_state_at_find_address_is_pruned_not_found() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_find_addrs(vec![0x40_3000]);
        let cb = PythonCallbacks::new();

        assert_routed(mgr.step_one(&cb, unsat_state_at(0x40_3000)).unwrap());

        assert_eq!(mgr.found_count(), 0, "UNSAT state is not a solution");
        assert_eq!(mgr.stash_count(STASH_FOUND), 0);
        assert_eq!(mgr.stash_count(STASH_PRUNED), 1);
    });
}

/// `lazy_solves` trades the satisfiability check for speed, so the same UNSAT
/// state is accepted as found. Pins that the skip is the *only* difference.
#[test]
fn lazy_solves_accepts_unsat_state_at_find_address() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_find_addrs(vec![0x40_3000]);
        mgr.constraint_solver.lazy_solves = true;
        let cb = PythonCallbacks::new();

        assert_routed(mgr.step_one(&cb, unsat_state_at(0x40_3000)).unwrap());

        assert_eq!(mgr.found_count(), 1);
        assert_eq!(mgr.stash_count(STASH_PRUNED), 0);
    });
}

/// The avoid check runs first, so an address in both sets is avoided, not
/// found. Ordering is load-bearing: the reverse would report solutions the
/// caller explicitly excluded.
#[test]
fn avoid_wins_over_find_at_the_same_address() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, _id) = mgr_and_state(0x40_4000);
        mgr.set_find_addrs(vec![0x40_4000]);
        mgr.set_avoid_addrs(vec![0x40_4000]);
        let cb = PythonCallbacks::new();

        assert_routed(mgr.step_one(&cb, state).unwrap());

        assert_eq!(mgr.stash_count(STASH_AVOID), 1);
        assert_eq!(mgr.found_count(), 0, "avoid precedes find");
    });
}

/// A callable `avoid=` predicate cannot be evaluated in Rust, so the state
/// bounces to Python. `resume_avoid_predicate(false)` answers by seeding
/// `skip_avoid_predicate_states`; that token must be consumed on the next
/// visit, otherwise the state bounces forever.
#[test]
fn callable_avoid_predicate_bounces_once_then_skip_token_is_consumed() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, id) = mgr_and_state(0x40_5000);
        // Set find first: `set_find_addrs` clears `find_needs_python`, and the
        // find address gives the post-skip fall-through an observable verdict.
        mgr.set_find_addrs(vec![0x40_5000]);
        mgr.set_avoid_needs_python(true);
        let cb = PythonCallbacks::new();

        let pending = expect_callback(mgr.step_one(&cb, state).unwrap());
        match pending.reason {
            CallbackReason::AvoidPredicate { addr } => assert_eq!(addr, 0x40_5000),
            other => panic!("expected AvoidPredicate, got {other:?}"),
        }
        assert_eq!(mgr.found_count(), 0, "bounced before any find routing");

        // Python answered "not avoided": seed the skip token and re-enter.
        mgr.constraint_tracker.skip_avoid_predicate_states.insert(id);
        assert_routed(mgr.step_one(&cb, pending.state).unwrap());

        assert_eq!(mgr.found_count(), 1, "fell through to the find check");
        assert!(
            !mgr.constraint_tracker
                .skip_avoid_predicate_states
                .contains(&id),
            "skip token is one-shot"
        );
    });
}

/// Same contract on the find side. `find_needs_python` is set after
/// `set_find_addrs` on purpose — the setter clears it, and the address list is
/// what makes the post-skip fall-through observable.
#[test]
fn callable_find_predicate_bounces_once_then_skip_token_is_consumed() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, id) = mgr_and_state(0x40_6000);
        mgr.set_find_addrs(vec![0x40_6000]);
        mgr.set_find_needs_python(true);
        let cb = PythonCallbacks::new();

        let pending = expect_callback(mgr.step_one(&cb, state).unwrap());
        match pending.reason {
            CallbackReason::FindPredicate { addr } => assert_eq!(addr, 0x40_6000),
            other => panic!("expected FindPredicate, got {other:?}"),
        }
        assert_eq!(mgr.found_count(), 0);

        mgr.constraint_tracker.skip_find_predicate_states.insert(id);
        assert_routed(mgr.step_one(&cb, pending.state).unwrap());

        assert_eq!(mgr.found_count(), 1);
        assert!(
            !mgr.constraint_tracker
                .skip_find_predicate_states
                .contains(&id),
            "skip token is one-shot"
        );
    });
}

/// An address-based avoid still beats a pending callable *find* predicate:
/// the avoid arms both run before either find arm.
#[test]
fn avoid_address_beats_pending_callable_find_predicate() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, _id) = mgr_and_state(0x40_7000);
        mgr.set_avoid_addrs(vec![0x40_7000]);
        mgr.set_find_needs_python(true);
        let cb = PythonCallbacks::new();

        assert_routed(mgr.step_one(&cb, state).unwrap());

        assert_eq!(mgr.stash_count(STASH_AVOID), 1);
    });
}

/// A hooked address whose SimProcedure has no native implementation bounces to
/// Python with the registered name / arg count, and is accounted in both
/// fallback counters. `__ralph_absent_native_proc` is deliberately not a name
/// the native registry knows, so `dispatch_native_proc` is never reached.
#[test]
fn unimplemented_simprocedure_falls_back_to_python_and_counts() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, _id) = mgr_and_state(0x40_8000);
        let name = "__ralph_absent_native_proc".to_string();
        assert!(
            mgr.native_procedures.get(&name).is_none(),
            "fixture name must not resolve to a native proc"
        );
        mgr.register_simprocedure(0x40_8000, name.clone(), 3, false);
        let cb = PythonCallbacks::new();

        let pending = expect_callback(mgr.step_one(&cb, state).unwrap());
        match &pending.reason {
            CallbackReason::SimProcedure {
                addr,
                name: cb_name,
                num_args,
                ..
            } => {
                assert_eq!(*addr, 0x40_8000);
                assert_eq!(cb_name, &name);
                assert_eq!(*num_args, 3, "the registered fixed-arg count is echoed");
            }
            other => panic!("expected SimProcedure, got {other:?}"),
        }

        assert_eq!(mgr.simprocedure_python_fallback_count, 1);
        assert_eq!(mgr.simprocedure_fallback_by_name.get(&name), Some(&1));
    });
}

/// `register_simprocedure` hooks the address, but a find address at the same pc
/// is decided first — the hook block is never consulted, so no fallback is
/// counted.
#[test]
fn find_address_short_circuits_a_hook_at_the_same_address() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, _id) = mgr_and_state(0x40_9000);
        mgr.register_simprocedure(0x40_9000, "__ralph_absent_native_proc".to_string(), 1, false);
        mgr.set_find_addrs(vec![0x40_9000]);
        let cb = PythonCallbacks::new();

        assert_routed(mgr.step_one(&cb, state).unwrap());

        assert_eq!(mgr.found_count(), 1);
        assert_eq!(
            mgr.simprocedure_python_fallback_count, 0,
            "hook dispatch never ran"
        );
    });
}
