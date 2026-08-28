//! Shared state/manager fixtures for the `exploration/` test modules
//! (angr-5mnx3.15).
//!
//! `run_loop_tests`, `run_loop_single_tests`, `run_loop_wave_tests`,
//! `run_loop_worker_tests` and `selection_policy_tests` each hang off a
//! different parent (`run_loop.rs`, `run_loop_single.rs`, …), so none of them
//! can reach a sibling's helpers through `super`. Before this module they each
//! carried a verbatim copy of the same state builder, and the three UNSAT
//! builders had additionally drifted apart in name (`unsat_state_at` /
//! `unsat_state`) and in the symbol they mint. This module is the one copy; it
//! lives at the `exploration/` level for the same reason
//! `callbacks::test_support` lives one level above its two consumers.
//!
//! Not everything shared-looking belongs here: `run_loop_steady_tests`'s
//! `SpyPolicy` documents in place *why* it is redefined rather than shared,
//! and `run_loop_worker_tests`'s `Harness` borrows parent-private items a
//! neutral module cannot name.

use crate::exploration::RustExplorationManager;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// A fresh amd64 state parked at `pc`, registered in no stash.
pub(crate) fn state_at(pc: u64) -> RustSimState {
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_pc(pc);
    state
}

/// A fresh manager plus a [`state_at`] `pc` and that state's id. The state is
/// registered in no stash yet — exactly the precondition both
/// `route_materialized_terminal` and `step_one` assume (the caller owns it;
/// the coordinator is about to place it).
pub(crate) fn mgr_and_state(pc: u64) -> (RustExplorationManager, RustSimState, u64) {
    let mgr = RustExplorationManager::new("amd64", None).unwrap();
    let state = state_at(pc);
    let id = state.state_id();
    (mgr, state, id)
}

/// Same as [`state_at`], but the state carries two mutually exclusive
/// constraints so `satisfiable()` is false — the fixture every "an infeasible
/// path reaching a find address is pruned, not found" test needs.
///
/// The symbol name is fixed rather than per-caller: `RustSimState::new` mints
/// a fresh solver context per state, so two fixtures never share a namespace.
pub(crate) fn unsat_state_at(pc: u64) -> RustSimState {
    let mut state = state_at(pc);
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "unsat_x", 64)
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
