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

/// The find check runs first, so an address in both sets is found, not avoided
/// — angr's Python `Explorer` documents that default (`avoid_priority=False`,
/// which the Rust engine has no knob to flip). Ordering is load-bearing: the
/// reverse silently disagreed with `route_successor`, so the very same config
/// routed a state to FOUND or AVOID purely by which side of a step boundary it
/// was observed on (angr-03vl4.14).
#[test]
fn find_wins_over_avoid_at_the_same_address() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, _id) = mgr_and_state(0x40_4000);
        mgr.set_find_addrs(vec![0x40_4000]);
        mgr.set_avoid_addrs(vec![0x40_4000]);
        let cb = PythonCallbacks::new();

        assert_routed(mgr.step_one(&cb, state).unwrap());

        assert_eq!(mgr.found_count(), 1, "find precedes avoid");
        assert_eq!(mgr.stash_count(STASH_AVOID), 0);
    });
}

/// The regression proper: `step_one` (pre-step, popped from ACTIVE) and
/// `route_successor` (post-step, a fresh successor) must land an
/// overlapping-address state in the SAME stash. Before angr-03vl4.14 the two
/// disagreed — AVOID vs FOUND — with nothing user-visible selecting between
/// them.
#[test]
fn pre_step_and_post_step_routing_agree_on_an_overlapping_address() {
    Python::initialize();
    Python::attach(|_py| {
        let addr = 0x40_4100;
        let cb = PythonCallbacks::new();

        let (mut pre, state, _id) = mgr_and_state(addr);
        pre.set_find_addrs(vec![addr]);
        pre.set_avoid_addrs(vec![addr]);
        assert_routed(pre.step_one(&cb, state).unwrap());

        let (mut post, successor, _id) = mgr_and_state(addr);
        post.set_find_addrs(vec![addr]);
        post.set_avoid_addrs(vec![addr]);
        post.route_successor(successor, true);

        assert_eq!(
            (pre.found_count(), pre.stash_count(STASH_AVOID)),
            (post.found_count(), post.stash_count(STASH_AVOID)),
            "pre-step and post-step routing must not disagree"
        );
        assert_eq!(post.found_count(), 1, "and both must be find-first");
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
        // Set the avoid address first: `set_avoid_addrs` clears
        // `avoid_needs_python`, and the address gives the post-skip
        // fall-through an observable verdict. It has to be the *avoid* address
        // (not a find one) because the find arms now run first, and a find
        // address would route the state before the predicate is ever asked.
        mgr.set_avoid_addrs(vec![0x40_5000]);
        mgr.set_avoid_needs_python(true);
        let cb = PythonCallbacks::new();

        let pending = expect_callback(mgr.step_one(&cb, state).unwrap());
        match pending.reason {
            CallbackReason::AvoidPredicate { addr } => assert_eq!(addr, 0x40_5000),
            other => panic!("expected AvoidPredicate, got {other:?}"),
        }
        assert_eq!(mgr.stash_count(STASH_AVOID), 0, "bounced, not yet routed");

        // Python answered "not avoided": seed the skip token and re-enter.
        mgr.constraint_tracker.skip_avoid_predicate_states.insert(id);
        assert_routed(mgr.step_one(&cb, pending.state).unwrap());

        assert_eq!(
            mgr.stash_count(STASH_AVOID),
            1,
            "fell through to the avoid-address check"
        );
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

/// With BOTH predicates callable and both answering "no", the state must reach
/// the interpreter on its third visit instead of alternating find/avoid bounces
/// forever (angr-03vl4.87). Consuming each token at its own gate used to leave
/// the third visit's find gate token-less, so pass 3 re-bounced to find, pass 4
/// to avoid, and the PC never advanced.
#[test]
fn both_callable_predicates_answering_false_stop_bouncing_on_the_third_visit() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, id) = mgr_and_state(0x40_9000);
        // Avoid address first: `set_avoid_addrs` clears `avoid_needs_python`.
        // It is what makes the third visit's fall-through observable without
        // driving the interpreter, and it sits *after* the avoid gate, so it
        // cannot preempt either bounce.
        mgr.set_avoid_addrs(vec![0x40_9000]);
        mgr.set_find_needs_python(true);
        mgr.set_avoid_needs_python(true);
        let cb = PythonCallbacks::new();

        // Visit 1: find has no token, so it bounces. Python answers "no match".
        let pending = expect_callback(mgr.step_one(&cb, state).unwrap());
        match pending.reason {
            CallbackReason::FindPredicate { addr } => assert_eq!(addr, 0x40_9000),
            other => panic!("expected FindPredicate, got {other:?}"),
        }
        mgr.constraint_tracker.skip_find_predicate_states.insert(id);

        // Visit 2: find falls through on its token; avoid has none, so it
        // bounces. Python answers "not avoided".
        let pending = expect_callback(mgr.step_one(&cb, pending.state).unwrap());
        match pending.reason {
            CallbackReason::AvoidPredicate { addr } => assert_eq!(addr, 0x40_9000),
            other => panic!("expected AvoidPredicate, got {other:?}"),
        }
        mgr.constraint_tracker.skip_avoid_predicate_states.insert(id);

        // Visit 3: the find token must STILL be honored — this is the assertion
        // the old consume-at-the-gate code failed.
        assert_routed(mgr.step_one(&cb, pending.state).unwrap());
        assert_eq!(
            mgr.stash_count(STASH_AVOID),
            1,
            "cleared both gates and hit the avoid address"
        );
        assert!(
            !mgr.constraint_tracker
                .skip_find_predicate_states
                .contains(&id)
                && !mgr
                    .constraint_tracker
                    .skip_avoid_predicate_states
                    .contains(&id),
            "both tokens retire together once the state clears both gates"
        );
    });
}

/// A pending callable *find* predicate beats an address-based avoid: both find
/// arms run before either avoid arm, so the state bounces to Python for the
/// find verdict rather than being routed to AVOID unasked. Only if Python
/// answers "no match" does the avoid address get its say.
#[test]
fn pending_callable_find_predicate_beats_avoid_address() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, id) = mgr_and_state(0x40_7000);
        mgr.set_avoid_addrs(vec![0x40_7000]);
        mgr.set_find_needs_python(true);
        let cb = PythonCallbacks::new();

        let pending = expect_callback(mgr.step_one(&cb, state).unwrap());
        match pending.reason {
            CallbackReason::FindPredicate { addr } => assert_eq!(addr, 0x40_7000),
            other => panic!("expected FindPredicate, got {other:?}"),
        }
        assert_eq!(mgr.stash_count(STASH_AVOID), 0, "avoid did not preempt it");

        // Python answered "no match": now the avoid address routes it.
        mgr.constraint_tracker.skip_find_predicate_states.insert(id);
        assert_routed(mgr.step_one(&cb, pending.state).unwrap());

        assert_eq!(mgr.stash_count(STASH_AVOID), 1);
        assert_eq!(mgr.found_count(), 0);
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
