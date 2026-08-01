// Tests for exploration/fork_materialize.rs (split out of helpers_tests.rs
// alongside the source split, angr-9ke6b.76).

use super::*;
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};

/// Build a state whose solver context already knows the symbol `x`, plus the
/// `x == 0` condition expressed in that same context.
fn state_with_x_eq_zero() -> (RustSimState, RustBV) {
    let mut state = RustSimState::new("amd64").expect("state");
    state.set_pc(0x40_1000);
    let cond = {
        let ctx_ref = state.solver().borrow();
        let ctx: &SymContext = &ctx_ref;
        let x = RustBV::symbolic(ctx, "x", 64);
        x.eq(&RustBV::zero(64), ctx)
    };
    (state, cond)
}

fn deferred_fork(condition_id: u64, path_taken: bool) -> crate::callbacks::DeferredFork {
    crate::callbacks::DeferredFork {
        branch_addr: 0x40_0500,
        path_taken,
        unexplored_target: 0x40_2000,
        condition_id,
        push_level: 0,
        condition_ast: None,
    }
}

/// The happy path of the shared materializer (angr-ph300.10): a stored
/// condition yields one SAT fork parked at `unexplored_target`, and the
/// `guard_sink` — the state that keeps executing — picks up the *taken*-path
/// guard so it can no longer satisfy the opposite side.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn materialize_deferred_forks_sat_fork_and_guard_sink() {
    Python::initialize();
    let (base, cond) = state_with_x_eq_zero();
    let sink = base.fork();
    let mut stored = FxHashMap::default();
    stored.insert(7u64, cond.clone());
    let mut snapshots = FxHashMap::default();

    let out = materialize_deferred_forks(
        vec![deferred_fork(7, false)], // took the false side; unexplored is x == 0
        MaterializeForkCtx {
            fork_base: &base,
            stored_conditions: &stored,
            snapshots: &mut snapshots,
            lazy_solves: false,
            guard_sink: Some(&sink),
            stats: None,
        },
    );

    assert_eq!(out.sat.len(), 1, "x == 0 is satisfiable on a clean base");
    assert!(out.unsat.is_empty());
    assert_eq!(out.sat[0].pc(), 0x40_2000);

    // path_taken == false -> assume_false(x == 0) on the sink, so a probe that
    // re-asserts x == 0 must come back UNSAT.
    let probe = sink.fork();
    probe.solver().borrow().assume_true(&cond);
    assert!(!probe.satisfiable(), "guard_sink did not receive the guard");
}

/// An unexplored side that contradicts the base lands in `unsat`, not `sat` —
/// and `lazy_solves` short-circuits that check, which is why every caller reads
/// the split rather than assuming SAT.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn materialize_deferred_forks_unsat_split_and_lazy_solves() {
    Python::initialize();
    let (base, cond) = state_with_x_eq_zero();
    base.solver().borrow().assume_true(&cond); // base pins x == 0
    let mut stored = FxHashMap::default();
    stored.insert(7u64, cond);
    let mut snapshots = FxHashMap::default();

    // path_taken == true -> unexplored side asserts x != 0, contradicting base.
    let out = materialize_deferred_forks(
        vec![deferred_fork(7, true)],
        MaterializeForkCtx {
            fork_base: &base,
            stored_conditions: &stored,
            snapshots: &mut snapshots,
            lazy_solves: false,
            guard_sink: None,
            stats: None,
        },
    );
    assert!(out.sat.is_empty());
    assert_eq!(out.unsat.len(), 1, "contradicting fork must be pruned");

    let mut snapshots = FxHashMap::default();
    let lazy = materialize_deferred_forks(
        vec![deferred_fork(7, true)],
        MaterializeForkCtx {
            fork_base: &base,
            stored_conditions: &stored,
            snapshots: &mut snapshots,
            lazy_solves: true,
            guard_sink: None,
            stats: None,
        },
    );
    assert_eq!(lazy.sat.len(), 1, "lazy_solves skips the SAT check");
    assert!(lazy.unsat.is_empty());
}

/// P15: a fork whose condition is in neither `stored_conditions` nor
/// `condition_ast` still yields a conservative, unconstrained fork at the
/// unexplored target — dropping it would lose a reachable path (angr-ph300.7
/// was exactly that drop, in one copy of this loop).
#[test]
fn materialize_deferred_forks_p15_conservative_fork() {
    Python::initialize();
    let (base, _cond) = state_with_x_eq_zero();
    let stored = FxHashMap::default();
    let mut snapshots = FxHashMap::default();

    let out = materialize_deferred_forks(
        vec![deferred_fork(99, true)], // condition_id 99 is not in `stored`
        MaterializeForkCtx {
            fork_base: &base,
            stored_conditions: &stored,
            snapshots: &mut snapshots,
            lazy_solves: false,
            guard_sink: None,
            stats: None,
        },
    );

    assert_eq!(out.sat.len(), 1, "P15 must not drop the fork");
    assert!(out.unsat.is_empty());
    assert_eq!(out.sat[0].pc(), 0x40_2000);
}

/// Profiling is opt-in: `stats: None` costs nothing, `Some` accumulates the
/// deferred-fork / fork-op / SAT counters the run loop reports. An empty fork
/// list still charges the batch *timer* (deliberately — that is the liveness
/// signal `test_fork_counters_exposed_and_non_summable` reads) but bumps no
/// count.
#[test]
fn materialize_deferred_forks_stats_are_opt_in() {
    Python::initialize();
    let (base, cond) = state_with_x_eq_zero();
    let mut stored = FxHashMap::default();
    stored.insert(7u64, cond);
    let mut snapshots = FxHashMap::default();
    let mut stats = ExecutionStats::default();

    let empty = materialize_deferred_forks(
        Vec::new(),
        MaterializeForkCtx {
            fork_base: &base,
            stored_conditions: &stored,
            snapshots: &mut snapshots,
            lazy_solves: false,
            guard_sink: None,
            stats: Some(&mut stats),
        },
    );
    assert!(empty.sat.is_empty() && empty.unsat.is_empty());
    assert_eq!(stats.deferred_fork_count, 0);
    assert!(
        stats.deferred_fork_time_ns > 0,
        "empty batch still charges the timer (profiling-gate liveness signal)"
    );
    assert_eq!(stats.solver_fork_count, 0);

    let out = materialize_deferred_forks(
        vec![deferred_fork(7, false), deferred_fork(7, true)],
        MaterializeForkCtx {
            fork_base: &base,
            stored_conditions: &stored,
            snapshots: &mut snapshots,
            lazy_solves: false,
            guard_sink: None,
            stats: Some(&mut stats),
        },
    );
    assert_eq!(out.sat.len() + out.unsat.len(), 2);
    assert_eq!(stats.deferred_fork_count, 2);
    assert_eq!(stats.solver_fork_count, 2);
    assert_eq!(stats.solver_sat_count, 2);
}

/// Two symbolic vars, so a fork of the *second* branch can be checked for the
/// *first* branch's taken-path guard.
fn state_with_two_conds() -> (RustSimState, RustBV, RustBV) {
    let mut state = RustSimState::new("amd64").expect("state");
    state.set_pc(0x40_1000);
    let (c1, c2) = {
        let ctx_ref = state.solver().borrow();
        let ctx: &SymContext = &ctx_ref;
        let x = RustBV::symbolic(ctx, "x", 64);
        let y = RustBV::symbolic(ctx, "y", 64);
        (x.eq(&RustBV::zero(64), ctx), y.eq(&RustBV::zero(64), ctx))
    };
    (state, c1, c2)
}

/// angr-62ar5: a fork materialized for branch *i* must inherit the taken-path
/// guards of branches `0..i` in the same step. `materialize_deferred_forks`
/// forks off a fixed, guard-free `fork_base` (the pre-callback snapshot), so
/// without an explicit replay the second fork came back under-constrained —
/// free to pick a model that contradicts a decision its own path already made.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn materialize_deferred_forks_replays_earlier_guards_onto_later_forks() {
    Python::initialize();
    let (base, c1, c2) = state_with_two_conds();
    let mut stored = FxHashMap::default();
    stored.insert(1u64, c1.clone());
    stored.insert(2u64, c2.clone());
    let mut snapshots = FxHashMap::default();

    let out = materialize_deferred_forks(
        // Branch 1: took the true side (x == 0). Branch 2: took the false side.
        vec![deferred_fork(1, true), deferred_fork(2, false)],
        MaterializeForkCtx {
            fork_base: &base,
            stored_conditions: &stored,
            snapshots: &mut snapshots,
            lazy_solves: false,
            guard_sink: None,
            stats: None,
        },
    );

    assert_eq!(out.sat.len(), 2);
    assert!(out.unsat.is_empty());

    // Fork 0 is the unexplored side of branch 1 — no priors, so x != 0 only.
    let probe = out.sat[0].fork();
    probe.solver().borrow().assume_true(&c1);
    assert!(!probe.satisfiable(), "fork 0 lost its own inverted guard");

    // Fork 1 is the unexplored side of branch 2. Its path went THROUGH branch
    // 1's true side, so x == 0 must still hold on it.
    let probe = out.sat[1].fork();
    probe.solver().borrow().assume_false(&c1);
    assert!(
        !probe.satisfiable(),
        "fork 1 did not inherit branch 1's taken-path guard (x == 0)"
    );
    // ...and it carries the inverted guard of its own branch (y == 0).
    let probe = out.sat[1].fork();
    probe.solver().borrow().assume_false(&c2);
    assert!(!probe.satisfiable(), "fork 1 lost its own inverted guard");
}

/// Same invariant on the pre-branch-*snapshot* path, which is the one the VEX
/// interpreter actually takes: the snapshot is a clone of the block solver,
/// which by design carries NO taken-path guards at all, so the replay is the
/// only thing putting branch 1's decision onto branch 2's fork.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn snapshot_built_fork_replays_earlier_guards() {
    Python::initialize();
    let (base, c1, c2) = state_with_two_conds();
    let mut stored = FxHashMap::default();
    stored.insert(1u64, c1.clone());
    stored.insert(2u64, c2.clone());
    let mut snapshots = FxHashMap::default();
    snapshots.insert(
        2u64,
        crate::interpreter::BranchSnapshot {
            solver: base.solver().borrow().fork(),
            registers: base.registers().fork(),
            memory: None,
        },
    );

    let out = materialize_deferred_forks(
        vec![deferred_fork(1, true), deferred_fork(2, false)],
        MaterializeForkCtx {
            fork_base: &base,
            stored_conditions: &stored,
            snapshots: &mut snapshots,
            lazy_solves: false,
            guard_sink: None,
            stats: None,
        },
    );

    assert_eq!(out.sat.len(), 2);
    let probe = out.sat[1].fork();
    probe.solver().borrow().assume_false(&c1);
    assert!(
        !probe.satisfiable(),
        "snapshot-built fork did not inherit branch 1's taken-path guard"
    );
}

/// `reconstruct_deferred_fork_condition` (the shared P11 helper extracted in
/// angr-ph300.76) short-circuits without ever touching Python in its two
/// non-reconstruction branches: when the condition is already present in
/// `stored_conditions`, and when the fork carries no `condition_ast`. Both
/// return `None` (nothing was reconstructed), leaving `condition.or(...)`
/// intact at the call sites.
#[test]
fn reconstruct_deferred_fork_condition_early_returns_none() {
    let fork_base = RustSimState::new("amd64").expect("base state");

    let ctx = SymContext::new();
    let stored = RustBV::symbolic(&ctx, "c", 1);
    let fork_with_ast = crate::callbacks::DeferredFork {
        branch_addr: 0x40_0500,
        path_taken: true,
        unexplored_target: 0x40_2000,
        condition_id: 7,
        push_level: 0,
        condition_ast: None,
    };

    // Branch 1: stored condition already present -> nothing to reconstruct,
    // even if an AST were also present. No Python attach happens.
    assert!(
        reconstruct_deferred_fork_condition(Some(&stored), &fork_with_ast, &fork_base).is_none(),
        "present stored condition must short-circuit to None"
    );

    // Branch 2: no stored condition and no condition_ast -> P11 cannot supply
    // a condition, so the P15 conservative arm must take over at the call site.
    assert!(
        reconstruct_deferred_fork_condition(None, &fork_with_ast, &fork_base).is_none(),
        "absent condition_ast must yield None"
    );
}
