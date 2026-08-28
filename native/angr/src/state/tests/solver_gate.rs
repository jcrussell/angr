//! The exploration loop's satisfiability prune gate
//! (`RustSimState::survives_sat_prune`) and its undecided-query contract.
//!
//! Every `exploration/` site that *drops* a state on an unsatisfiable verdict
//! routes through this gate, so the difference between "proven contradictory"
//! and "Z3 gave up" is the difference between a correct prune and a silently
//! deleted feasible path (angr-03vl4.86).

use super::super::*;

/// Pin the solver's `rlimit` (resource budget) so a check aborts to Unknown
/// deterministically. Mirrors `pin_rlimit` in `symbolic/context_tests/solver_queries.rs`
/// — see `test_pin_rlimit_reaches_the_solver` there for why an rlimit beats a
/// wall-clock timeout in a test.
#[cfg(feature = "vex-engine-z3")]
fn pin_rlimit(state: &RustSimState, rlimit: u32) {
    state.solver().borrow().pin_rlimit_for_test(rlimit);
}

/// A *proven* Unsat still prunes — the gate must not be a blanket "keep".
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_survives_sat_prune_drops_a_proven_unsat_state() {
    let state = RustSimState::new("amd64").unwrap();
    {
        let ctx = state.solver().borrow();
        let x = RustBV::symbolic(&ctx, "prune_gate_unsat_x", 32);
        let five = RustBV::concrete(5, 32);
        ctx.assume_true(&x.eq(&five, &ctx));
        ctx.assume_true(&x.eq(&five, &ctx).not(&ctx));
    }
    assert!(
        !state.survives_sat_prune(false),
        "a decided Unsat must still be pruned"
    );
}

/// An *undecided* query (Z3 Unknown / timeout) keeps the state. Rig: the easy
/// `10 < x < 20` set under `rlimit=1`, so the check aborts immediately and
/// lifting the budget afterwards decides it instantly — the test cannot hang
/// in either direction.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_survives_sat_prune_keeps_an_undecided_state() {
    let state = RustSimState::new("amd64").unwrap();
    {
        let ctx = state.solver().borrow();
        let x = RustBV::symbolic(&ctx, "prune_gate_undecided_x", 32);
        ctx.assume_true(&x.ugt(&RustBV::concrete(10, 32), &ctx));
        ctx.assume_true(&x.ult(&RustBV::concrete(20, 32), &ctx));
    }
    pin_rlimit(&state, 1);

    assert_eq!(
        state.satisfiable_checked(),
        None,
        "rlimit=1 must abort the check to Unknown — without that the rest of \
         this test proves nothing"
    );
    assert!(
        !state.satisfiable(),
        "the lenient form is what makes the gate necessary: it reads the same \
         Unknown as 'unsatisfiable'"
    );
    assert!(
        state.survives_sat_prune(false),
        "an undecided satisfiability query must keep the state, not prune it"
    );

    // Nothing was poisoned: with the budget restored the same state decides SAT.
    pin_rlimit(&state, 0);
    assert_eq!(state.satisfiable_checked(), Some(true));
    assert!(state.survives_sat_prune(false));
}

/// `lazy_solves` short-circuits the gate entirely — no Z3 query runs, so even
/// a contradictory state is kept (that is the whole point of the mode).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_survives_sat_prune_short_circuits_under_lazy_solves() {
    let state = RustSimState::new("amd64").unwrap();
    {
        let ctx = state.solver().borrow();
        let x = RustBV::symbolic(&ctx, "prune_gate_lazy_x", 32);
        let five = RustBV::concrete(5, 32);
        ctx.assume_true(&x.eq(&five, &ctx));
        ctx.assume_true(&x.eq(&five, &ctx).not(&ctx));
    }
    assert!(
        state.survives_sat_prune(true),
        "lazy_solves must skip the query and keep the state"
    );
    assert_eq!(
        state.satisfiable_checked(),
        Some(false),
        "the state really is contradictory — the gate kept it only because \
         lazy_solves skipped the query"
    );
}
