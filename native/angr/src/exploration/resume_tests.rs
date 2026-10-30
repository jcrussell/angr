// Tests for exploration/resume.rs — the three resume entry points that
// materialize deferred forks: `_resume_after_simprocedure`,
// `_deadend_pending_callback`, `_resume_after_symbolic_branch`.
//
// All three hand the same `PendingCallback.deferred_forks` to the shared
// `fork_materialize::materialize_deferred_forks`, so an unexplored branch must survive
// identically no matter which callback the step happened to park on. The
// materializer itself is unit-tested in `helpers_tests.rs`; what this file
// pins is the *plumbing* around it — which fork base each entry point builds
// and where the resulting states are routed.
//
// `.7` / `.9` carry their own regressions (AST-only forks behind a no-return
// SimProcedure; guard-polluted fork bases); those are not repeated here.
use super::*;
use crate::exploration::{CallbackReason, PendingCallback};
use crate::stash::{STASH_ACTIVE, STASH_DEADENDED, STASH_PRUNED};
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use rustc_hash::FxHashMap;

const BRANCH_ADDR: u64 = 0x40_0500;
const UNEXPLORED: u64 = 0x40_2000;
const COND_ID: u64 = 7;

/// A state parked at a callback plus the branch condition `x == 0` that an
/// earlier deferred fork in the same step diverged on.
fn state_and_condition() -> (RustSimState, RustBV) {
    let mut state = RustSimState::new("amd64").expect("state");
    state.set_pc(0x40_1000);
    let cond = {
        let x = RustBV::symbol("x", 64);
        x.eq(&RustBV::zero(64))
    };
    (state, cond)
}

/// One deferred fork whose unexplored side is `x == 0` (the step took the
/// `false` side), with the condition resolvable from `stored_conditions`.
fn pending_with_one_fork(
    state: RustSimState,
    cond: &RustBV,
    reason: CallbackReason,
    snapshot: Option<RustSimState>,
) -> PendingCallback {
    let mut stored = FxHashMap::default();
    stored.insert(COND_ID, cond.clone());
    PendingCallback {
        state,
        pre_callback_snapshot: snapshot,
        reason,
        jumpkind: None,
        deferred_forks: vec![crate::interpreter::DeferredFork {
            branch_addr: BRANCH_ADDR,
            path_taken: false,
            unexplored_target: UNEXPLORED,
            condition_id: COND_ID,
        }],
        stored_conditions: stored,
        fork_snapshots: FxHashMap::default(),
    }
}

/// Ids of states parked at the unexplored target across every stash, paired
/// with the stash holding them — the fork-materialization output as a
/// caller-visible fact.
fn fork_placements(mgr: &RustExplorationManager) -> Vec<(String, u64)> {
    let mut out: Vec<(String, u64)> = mgr
        .sm
        .stashes()
        .iter()
        .flat_map(|(name, deque)| {
            deque
                .iter()
                .filter(|s| s.pc() == UNEXPLORED)
                .map(move |s| (name.clone(), s.state_id()))
        })
        .collect();
    out.sort();
    out
}

/// The parking state's own id, so `fork_placements` can be read without it.
fn park(mgr: &mut RustExplorationManager, pending: PendingCallback) -> u64 {
    let sid = pending.state.state_id();
    mgr.pending_callbacks.insert(StateId::new(sid), pending);
    sid
}

/// Parity: with a pre-callback snapshot present — the shape the interpreter
/// produces whenever it parks a callback mid-step — all three entry points
/// materialize the one deferred fork into a live active state at the
/// unexplored target, with the lineage root stamped to the parking state.
///
/// This is the property the shared materializer exists to guarantee: which
/// callback the step happened to bounce on must not decide whether an
/// unexplored branch is ever explored.
#[test]
fn all_three_entry_points_materialize_the_same_deferred_fork() {
    Python::initialize();

    let mut observed: Vec<(&str, usize, u64)> = Vec::new();

    // 1. SimProcedure resume.
    {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let (state, cond) = state_and_condition();
        let snapshot = state.fork();
        let reason = CallbackReason::SimProcedure {
            addr: BRANCH_ADDR,
            name: "puts".to_string(),
            num_args: 1,
            return_addr: 0x40_1004,
        };
        let sid = park(
            &mut mgr,
            pending_with_one_fork(state, &cond, reason, Some(snapshot)),
        );
        Python::attach(|py| {
            mgr._resume_after_simprocedure(py, sid, 0x40_1004, None, None, None)
                .expect("resume simprocedure");
        });
        let forks = fork_placements(&mgr);
        assert_eq!(forks.len(), 1, "simprocedure: one materialized fork");
        assert_eq!(forks[0].0, STASH_ACTIVE, "simprocedure: fork is live");
        assert_eq!(mgr.get_state_root(forks[0].1), Some(sid));
        assert_eq!(mgr.stash_count(STASH_PRUNED), 0);
        observed.push(("simprocedure", forks.len(), forks[0].1));
    }

    // 2. Deadend (no-return SimProcedure / exit).
    {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let (state, cond) = state_and_condition();
        let snapshot = state.fork();
        let reason = CallbackReason::Error {
            message: "exit".to_string(),
        };
        let sid = park(
            &mut mgr,
            pending_with_one_fork(state, &cond, reason, Some(snapshot)),
        );
        mgr._deadend_pending_callback(sid).expect("deadend");

        let forks = fork_placements(&mgr);
        assert_eq!(forks.len(), 1, "deadend: one materialized fork");
        assert_eq!(forks[0].0, STASH_ACTIVE, "deadend: fork is live");
        assert_eq!(mgr.get_state_root(forks[0].1), Some(sid));
        // The parking state itself deadended rather than continuing.
        assert_eq!(mgr.stash_count(STASH_DEADENDED), 1);
        observed.push(("deadend", forks.len(), forks[0].1));
    }

    // 3. Symbolic branch (Python forked both directions for us).
    {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let (state, cond) = state_and_condition();
        let snapshot = state.fork();
        let reason = CallbackReason::SymbolicBranch {
            condition_id: COND_ID,
            true_target: 0x40_3000,
            false_target: 0x40_4000,
        };
        let sid = park(
            &mut mgr,
            pending_with_one_fork(state, &cond, reason, Some(snapshot)),
        );
        Python::attach(|py| {
            mgr._resume_after_symbolic_branch(py, sid, 0x40_3000, 0x40_4000, None, None)
                .expect("resume symbolic branch");
        });
        let forks = fork_placements(&mgr);
        assert_eq!(forks.len(), 1, "symbolic branch: one materialized fork");
        assert_eq!(forks[0].0, STASH_ACTIVE, "symbolic branch: fork is live");
        assert_eq!(mgr.get_state_root(forks[0].1), Some(sid));
        observed.push(("symbolic-branch", forks.len(), forks[0].1));
    }

    // Same count, same stash, in every entry point (ids differ per manager).
    assert_eq!(observed.len(), 3);
    assert!(
        observed.iter().all(|(_, n, _)| *n == 1),
        "entry points disagree on fork materialization: {observed:?}",
    );
}

/// Regression (angr-khpsh): without a `pre_callback_snapshot` all three entry
/// points must STILL keep the unexplored side live.
///
/// The snapshot-less fallback base is `state.fork()`, and it has to be taken
/// *before* `apply_deferred_fork_constraints` stamps the taken-path guard onto
/// the parking state — otherwise the base inherits that guard, the unexplored
/// side (`x == 0` when the step took `x != 0`) is trivially UNSAT, and a
/// reachable branch is silently pruned. `_deadend_pending_callback` was always
/// correct here (it never applies those constraints up front, passing the
/// parking state as `guard_sink` instead); the other two used to prune, which
/// this test now forbids.
#[test]
fn snapshotless_resume_keeps_deferred_fork_live_in_every_entry_point() {
    Python::initialize();

    // 1. Deadend (was already correct).
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let (state, cond) = state_and_condition();
    let reason = CallbackReason::Error {
        message: "exit".to_string(),
    };
    let sid = park(&mut mgr, pending_with_one_fork(state, &cond, reason, None));
    mgr._deadend_pending_callback(sid).expect("deadend");
    let deadend_forks = fork_placements(&mgr);
    assert_eq!(deadend_forks.len(), 1, "deadend keeps the unexplored side");
    assert_eq!(deadend_forks[0].0, STASH_ACTIVE);

    // 2. SimProcedure resume.
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let (state, cond) = state_and_condition();
    let reason = CallbackReason::SimProcedure {
        addr: BRANCH_ADDR,
        name: "puts".to_string(),
        num_args: 1,
        return_addr: 0x40_1004,
    };
    let sid = park(&mut mgr, pending_with_one_fork(state, &cond, reason, None));
    Python::attach(|py| {
        mgr._resume_after_simprocedure(py, sid, 0x40_1004, None, None, None)
            .expect("resume simprocedure");
    });
    let simproc_forks = fork_placements(&mgr);
    assert_eq!(simproc_forks.len(), 1);
    assert_eq!(
        simproc_forks[0].0, STASH_ACTIVE,
        "snapshot-less simprocedure resume must keep the unexplored side live",
    );
    assert_eq!(mgr.stash_count(STASH_PRUNED), 0);

    // 3. Symbolic branch resume.
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let (state, cond) = state_and_condition();
    let reason = CallbackReason::SymbolicBranch {
        condition_id: COND_ID,
        true_target: 0x40_3000,
        false_target: 0x40_4000,
    };
    let sid = park(&mut mgr, pending_with_one_fork(state, &cond, reason, None));
    Python::attach(|py| {
        mgr._resume_after_symbolic_branch(py, sid, 0x40_3000, 0x40_4000, None, None)
            .expect("resume symbolic branch");
    });
    let branch_forks = fork_placements(&mgr);
    assert_eq!(branch_forks.len(), 1);
    assert_eq!(
        branch_forks[0].0, STASH_ACTIVE,
        "snapshot-less symbolic-branch resume must keep the unexplored side live",
    );
}

/// `route_resume_successors` without a parallel session is exactly
/// `route_successor` per state: find-bound successors reach the found stash,
/// everything else the active frontier.
#[test]
fn route_resume_successors_routes_find_and_plain_states() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    mgr.set_find_addrs(vec![0x40_9000]);

    let mut hit = RustSimState::new("amd64").expect("state");
    hit.set_pc(0x40_9000);
    let hit_id = hit.state_id();
    let mut plain = RustSimState::new("amd64").expect("state");
    plain.set_pc(0x40_1000);
    let plain_id = plain.state_id();

    mgr.route_resume_successors(vec![hit, plain]);

    assert_eq!(mgr.get_state_ids(crate::stash::STASH_FOUND), vec![hit_id]);
    assert_eq!(mgr.get_state_ids(STASH_ACTIVE), vec![plain_id]);
}

/// A parked SimProcedure callback with no deferred forks, for the
/// copy-on-write successor entry points (angr-mx00p.112).
fn plain_pending(state: RustSimState) -> PendingCallback {
    PendingCallback {
        state,
        pre_callback_snapshot: None,
        reason: CallbackReason::SimProcedure {
            addr: BRANCH_ADDR,
            name: "hook".to_string(),
            num_args: 0,
            return_addr: 0x40_1004,
        },
        jumpkind: None,
        deferred_forks: Vec::new(),
        stored_conditions: FxHashMap::default(),
        fork_snapshots: FxHashMap::default(),
    }
}

const COW_STASH: &str = "_cb_forks";

/// angr-mx00p.112: a first successor that owns a fork resumes as that fork.
/// The pending callback keeps its key, the fork leaves the holding stash, the
/// state routed to active is the fork (carrying the fork's own register
/// write, not the pending state's), and the replaced pending state is gone.
#[test]
fn adopted_fork_is_the_state_the_pending_callback_resumes() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let mut state = RustSimState::new("amd64").expect("state");
    state.set_pc(BRANCH_ADDR);
    assert!(state.set_register("rax", RustBV::concrete(0x111, 64)));
    let sid = park(&mut mgr, plain_pending(state));

    let fork_id = mgr._fork_state_to_stash(sid, COW_STASH).expect("fork");
    {
        let fork = mgr.find_state_mut(fork_id).expect("fork is findable");
        assert!(fork.set_register("rax", RustBV::concrete(0x222, 64)));
    }
    mgr._adopt_pending_callback_state(sid, fork_id, COW_STASH)
        .expect("adopt");
    assert_eq!(mgr.stash_count(COW_STASH), 0, "fork left the holding stash");

    Python::attach(|py| {
        mgr._resume_after_simprocedure(py, sid, 0x40_1004, None, None, None)
            .expect("resume");
    });
    assert!(mgr.pending_callbacks.is_empty());
    let active = mgr.sm.get(STASH_ACTIVE).expect("active stash");
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].state_id(), fork_id);
    assert_eq!(active[0].pc(), 0x40_1004);
    assert_eq!(
        active[0].get_register("rax").and_then(|v| v.as_u64()),
        Some(0x222)
    );
    assert!(mgr.find_state(sid).is_none(), "the replaced state was dropped");
}

/// Adopting needs both halves: an unknown pending id or a fork outside the
/// named stash is an error, never a silent no-op.
#[test]
fn adopt_rejects_unknown_pending_or_fork() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let state = RustSimState::new("amd64").expect("state");
    let sid = park(&mut mgr, plain_pending(state));
    let fork_id = mgr._fork_state_to_stash(sid, COW_STASH).expect("fork");

    assert!(mgr._adopt_pending_callback_state(sid + 1_000_000, fork_id, COW_STASH).is_err());
    assert!(mgr._adopt_pending_callback_state(sid, fork_id, "elsewhere").is_err());
    // Neither failed attempt consumed anything.
    assert_eq!(mgr.stash_count(COW_STASH), 1);
    assert!(mgr.pending_callbacks.contains_key(&StateId::new(sid)));
}

/// angr-mx00p.112 / angr-vmi2l.294: an extra successor is finished from its
/// own fork. It lands in active at its own pc with the fork's writes and the
/// pending state's lineage root, and leaves the pending callback untouched.
/// A successor whose constraints are unsat goes to pruned, not active.
#[test]
fn forked_successor_routes_from_its_own_fork() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let (mut state, cond) = state_and_condition();
    state.set_pc(BRANCH_ADDR);
    let sid = park(&mut mgr, plain_pending(state));

    let live = mgr._fork_state_to_stash(sid, COW_STASH).expect("fork");
    let dead = mgr._fork_state_to_stash(sid, COW_STASH).expect("fork");
    {
        let fork = mgr.find_state_mut(live).expect("live fork");
        assert!(fork.set_register("rax", RustBV::concrete(0x333, 64)));
    }
    {
        // x == 0 and x != 0: this successor's guard contradicts the path.
        let fork = mgr.find_state(dead).expect("dead fork");
        let solver = fork.solver();
        let ctx = solver.borrow();
        ctx.assume_true(&cond);
        ctx.assume_false(&cond);
    }

    Python::attach(|py| {
        mgr._resume_forked_successor(py, live, COW_STASH, 0x40_3000, None, None, None)
            .expect("resume live fork");
        mgr._resume_forked_successor(py, dead, COW_STASH, 0x40_4000, None, None, None)
            .expect("resume dead fork");
    });

    assert_eq!(mgr.stash_count(COW_STASH), 0);
    let active = mgr.sm.get(STASH_ACTIVE).expect("active stash");
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].state_id(), live);
    assert_eq!(active[0].pc(), 0x40_3000);
    assert_eq!(
        active[0].get_register("rax").and_then(|v| v.as_u64()),
        Some(0x333)
    );
    assert_eq!(mgr.get_state_root(live), Some(sid));
    assert_eq!(mgr.stash_count(STASH_PRUNED), 1, "unsat successor is pruned");
    assert!(
        mgr.pending_callbacks.contains_key(&StateId::new(sid)),
        "the pending callback is not consumed by an extra successor"
    );
    Python::attach(|py| {
        assert!(
            mgr._resume_forked_successor(py, live, COW_STASH, 0, None, None, None)
                .is_err(),
            "a fork already routed is not in the holding stash any more"
        );
    });
}
