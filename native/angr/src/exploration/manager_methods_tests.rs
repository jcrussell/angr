// Tests for exploration/manager_methods.rs — the stash-accounting corner of
// the `#[pymethods]` surface (`move_state`, `drop_state_from_stash`,
// `clear_stash`, `stash_counts`, `state_stash`, `get_state_ids`).
//
// The invariant under test everywhere below: a live state is reachable from
// EXACTLY one stash, and every index (`state_index` -> stash name,
// `state_roots` -> lineage root) agrees with that placement. A stale index
// entry is silent — `get_state_ids` still looks right while `state_stash`
// points at a stash the state left — so each test asserts the deque view and
// the index view together.
//
// `_move_states` / `_reset_for_stage` are covered by `state_lifecycle_tests.rs`;
// this file deliberately does not duplicate them.
use super::*;
use crate::stash::{STASH_ACTIVE, STASH_DEADENDED, STASH_FOUND};
use crate::state::RustSimState;
use std::collections::VecDeque;

/// Push `n` fresh states into `stash`, returning their ids in push order.
fn push_states(mgr: &mut RustExplorationManager, stash: &str, n: usize) -> Vec<u64> {
    (0..n)
        .map(|i| {
            let mut s = RustSimState::new("amd64").expect("state");
            s.set_pc(0x40_0000 + (i as u64) * 0x10);
            let id = s.state_id();
            mgr.sm.push(stash, s);
            id
        })
        .collect()
}

/// Total number of states held across every stash.
fn live_total(mgr: &RustExplorationManager) -> usize {
    mgr.sm.stashes().values().map(VecDeque::len).sum()
}

/// Assert `id` is visible from exactly one stash, and that the index agrees.
fn assert_in_exactly(mgr: &RustExplorationManager, id: u64, expected: &str) {
    let holders: Vec<&str> = mgr
        .sm
        .stashes()
        .iter()
        .filter(|(_, deque)| deque.iter().any(|s| s.state_id() == id))
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(
        holders,
        vec![expected],
        "state {id} held by exactly one stash"
    );
    assert_eq!(
        mgr.sm.stash_of(id),
        Some(expected),
        "index entry for {id} agrees with the deque holding it",
    );
    assert!(
        mgr.get_state_ids(expected).contains(&id),
        "get_state_ids({expected}) reports {id}",
    );
}

/// A single-state move must relocate the deque entry AND repoint the index —
/// the state may not linger in the source stash, and `state_stash` must not
/// keep naming it.
#[test]
fn move_state_leaves_state_in_exactly_one_stash() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let ids = push_states(&mut mgr, STASH_ACTIVE, 3);

    assert!(
        mgr.move_state(ids[1], STASH_ACTIVE, STASH_FOUND)
            .expect("move active->found"),
        "moving a state that is present reports success",
    );

    assert_in_exactly(&mgr, ids[1], STASH_FOUND);
    assert_in_exactly(&mgr, ids[0], STASH_ACTIVE);
    assert_in_exactly(&mgr, ids[2], STASH_ACTIVE);
    assert_eq!(mgr.stash_count(STASH_ACTIVE), 2);
    assert_eq!(mgr.stash_count(STASH_FOUND), 1);
    assert_eq!(live_total(&mgr), 3, "a move never changes the live total");
    assert_eq!(mgr.state_stash(ids[1]).as_deref(), Some(STASH_FOUND));
}

/// Moving from a stash the state is not in must be a reported no-op, not a
/// silent relocation from wherever it actually lives.
#[test]
fn move_state_from_wrong_stash_is_reported_noop() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let ids = push_states(&mut mgr, STASH_ACTIVE, 1);

    assert!(
        !mgr.move_state(ids[0], STASH_DEADENDED, STASH_FOUND)
            .expect("move from empty stash"),
        "state is not in the source stash",
    );
    assert_in_exactly(&mgr, ids[0], STASH_ACTIVE);
    assert_eq!(live_total(&mgr), 1);
}

/// Dropping backs `RustStateProxy.__del__`: the state must leave the deque and
/// BOTH indexes, so a later id lookup cannot resurrect a freed handle.
#[test]
fn drop_state_from_stash_clears_deque_and_indexes() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let keep = push_states(&mut mgr, STASH_ACTIVE, 1)[0];
    let ids = push_states(&mut mgr, "_copies", 2);
    mgr.sm.set_root(ids[0], keep);

    assert_eq!(mgr.get_state_root(ids[0]), Some(keep));
    assert!(mgr.drop_state_from_stash(ids[0], "_copies"));

    assert_eq!(mgr.sm.stash_of(ids[0]), None, "index entry dropped");
    assert_eq!(mgr.state_stash(ids[0]), None);
    assert_eq!(mgr.get_state_root(ids[0]), None, "lineage root dropped");
    assert!(!mgr.get_state_ids("_copies").contains(&ids[0]));
    assert_eq!(live_total(&mgr), 2, "exactly one state reclaimed");

    // Idempotent: a second drop of the same id finds nothing.
    assert!(!mgr.drop_state_from_stash(ids[0], "_copies"));
    // Its sibling and the unrelated active state are untouched.
    assert_in_exactly(&mgr, ids[1], "_copies");
    assert_in_exactly(&mgr, keep, STASH_ACTIVE);
}

/// Naming the wrong stash must not drop the state from the one it is in — the
/// caller picks the stash, so a mismatch is a caller bug, not a licence to
/// free the state.
#[test]
fn drop_state_from_wrong_stash_is_noop() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let ids = push_states(&mut mgr, STASH_ACTIVE, 1);

    assert!(!mgr.drop_state_from_stash(ids[0], "_copies"));
    assert_in_exactly(&mgr, ids[0], STASH_ACTIVE);
    assert_eq!(live_total(&mgr), 1);
}

/// `clear_stash` frees a whole stash; every id it held must vanish from the
/// indexes too, and no other stash may be touched.
#[test]
fn clear_stash_unindexes_every_state_it_held() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let active = push_states(&mut mgr, STASH_ACTIVE, 3);
    let found = push_states(&mut mgr, STASH_FOUND, 1);
    for id in &active {
        mgr.sm.set_root(*id, active[0]);
    }

    mgr.clear_stash(STASH_ACTIVE);

    assert_eq!(mgr.stash_count(STASH_ACTIVE), 0);
    for id in &active {
        assert_eq!(mgr.sm.stash_of(*id), None, "{id} unindexed");
        assert_eq!(mgr.get_state_root(*id), None, "{id} root dropped");
    }
    assert_in_exactly(&mgr, found[0], STASH_FOUND);
    assert_eq!(live_total(&mgr), 1);
}

/// `stash_counts` is the dict Python's `stash_counts()` returns; it must be a
/// faithful per-stash census — equal to `get_state_ids(stash).len()` for every
/// key and summing to the live total — across a move/drop/clear sequence.
///
/// A state parked in `pending_callbacks` lives in NO stash by design, so it is
/// invisible here; `pending_callback_ids` is the only view of it.
#[test]
fn stash_counts_censuses_every_stash_after_move_drop_clear() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let active = push_states(&mut mgr, STASH_ACTIVE, 4);
    push_states(&mut mgr, "_copies", 2);

    mgr.move_state(active[0], STASH_ACTIVE, STASH_FOUND)
        .expect("move");
    mgr.move_state(active[1], STASH_ACTIVE, STASH_DEADENDED)
        .expect("move");
    assert!(mgr.drop_state_from_stash(active[2], STASH_ACTIVE));
    mgr.clear_stash("_copies");

    Python::attach(|py| {
        let counts = mgr.stash_counts(py).expect("stash_counts");
        let mut sum = 0usize;
        for (name, count) in counts.iter() {
            let name: String = name.extract().expect("stash name");
            let count: usize = count.extract().expect("count");
            assert_eq!(
                count,
                mgr.get_state_ids(&name).len(),
                "stash_counts[{name}] agrees with get_state_ids",
            );
            assert_eq!(
                count,
                mgr.stash_count(&name),
                "stash_counts[{name}] agrees with stash_count",
            );
            sum += count;
        }
        assert_eq!(sum, live_total(&mgr), "counts sum to the live state total");
        assert_eq!(sum, 3, "4 + 2 pushed, 1 dropped, 2 cleared");
    });

    assert_in_exactly(&mgr, active[0], STASH_FOUND);
    assert_in_exactly(&mgr, active[1], STASH_DEADENDED);
    assert_in_exactly(&mgr, active[3], STASH_ACTIVE);
    assert_eq!(mgr.active_count(), 1);
    assert_eq!(mgr.found_count(), 1);
}

// --- set_deterministic reach (angr-sqfj8.32) -----------------------------

/// `set_deterministic` propagates to states parked in `pending_callbacks`, not
/// just to the stashes.
///
/// A parked state is in NO stash (see `stash_counts_censuses_every_stash_...`
/// above), so the stash loop cannot reach it. Before angr-sqfj8.32 a flip made
/// while a SimProcedure/syscall callback was outstanding left that state — and
/// every deferred fork later materialized from its `pre_callback_snapshot` /
/// `fork_snapshots` — on the old witness-selection mode, silently and forever
/// (nothing re-applies the flag on resume).
///
/// Asserted in both directions: the flag has to be a genuine propagation, not
/// a one-way `true` latch that a fresh `SymContext` would satisfy by accident.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn set_deterministic_reaches_parked_pending_callback_states() {
    use crate::exploration::{CallbackReason, PendingCallback};
    use rustc_hash::FxHashMap;

    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let stashed = push_states(&mut mgr, STASH_ACTIVE, 1)[0];

    let state = RustSimState::new("amd64").expect("state");
    let sid = state.state_id();
    let mut fork_snapshots = FxHashMap::default();
    fork_snapshots.insert(
        7u64,
        crate::interpreter::BranchSnapshot {
            solver: state.solver().borrow().fork(),
            registers: state.registers().fork(),
            memory: None,
        },
    );
    let pre_callback_snapshot = Some(state.fork());
    mgr.pending_callbacks.insert(
        StateId::new(sid),
        PendingCallback {
            state,
            pre_callback_snapshot,
            reason: CallbackReason::Syscall { num: Some(60) },
            jumpkind: None,
            solver_ctx: None,
            deferred_forks: Vec::new(),
            stored_conditions: FxHashMap::default(),
            fork_snapshots,
        },
    );

    for v in [true, false, true] {
        mgr.set_deterministic(v);
        assert_eq!(
            mgr.state_is_deterministic(stashed).expect("stashed state"),
            v,
            "stashed state follows set_deterministic({v})",
        );
        let pending = mgr
            .pending_callbacks
            .get(&StateId::new(sid))
            .expect("callback still parked");
        assert_eq!(
            constraints::state_is_deterministic(&pending.state),
            v,
            "parked state follows set_deterministic({v})",
        );
        assert_eq!(
            constraints::state_is_deterministic(
                pending
                    .pre_callback_snapshot
                    .as_ref()
                    .expect("pre-callback snapshot"),
            ),
            v,
            "pre-callback snapshot (the deferred forks' fork_base) follows \
             set_deterministic({v})",
        );
        assert_eq!(
            pending.fork_snapshots[&7].solver.is_deterministic(),
            v,
            "pre-branch fork snapshot follows set_deterministic({v})",
        );
    }
}
