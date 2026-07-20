// Tests for exploration/state_lifecycle.rs — the _move_states / _move_state
// stash-shuffling surface.
use super::*;
use crate::stash::STASH_ACTIVE;
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Regression for angr-ph300.17: `_move_states` with `from_stash == to_stash`
/// used to `remove()` the source deque, re-append every state into the
/// recreated key, then overwrite that key with an empty `VecDeque` — silently
/// destroying every state in the stash while reporting them as moved. The
/// same-stash case must now be a no-op that preserves all states and their
/// index entries.
#[test]
fn move_states_same_stash_is_noop() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let mut ids = Vec::new();
    for v in [0x10u128, 0x20, 0x30] {
        let mut s = RustSimState::new("amd64").expect("state");
        s.set_register("rax", RustBV::concrete(v, 64));
        ids.push(s.state_id());
        mgr.sm.push(STASH_ACTIVE, s);
    }

    let count = mgr
        ._move_states(STASH_ACTIVE, STASH_ACTIVE, None)
        .expect("move active->active");

    // No state is actually relocated, so the reported count is 0.
    assert_eq!(count, 0, "same-stash move reports zero relocations");

    // Every state must survive in the source/destination stash.
    let active = mgr.sm.get(STASH_ACTIVE).expect("active stash");
    assert_eq!(
        active.len(),
        3,
        "all states preserved after same-stash move"
    );

    // Index entries must still resolve to the active stash (no dangling).
    for id in ids {
        assert_eq!(
            mgr.sm.stash_of(id),
            Some(STASH_ACTIVE),
            "index entry for {id} still points at active",
        );
    }
}

/// A genuine cross-stash move still relocates every state and reports the
/// correct count — the same-stash guard must not affect the normal path.
#[test]
fn move_states_cross_stash_relocates_all() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    for v in [0x10u128, 0x20] {
        let mut s = RustSimState::new("amd64").expect("state");
        s.set_register("rax", RustBV::concrete(v, 64));
        mgr.sm.push(STASH_ACTIVE, s);
    }

    let count = mgr
        ._move_states(STASH_ACTIVE, "found", None)
        .expect("move active->found");

    assert_eq!(count, 2, "both states relocated");
    assert!(
        mgr.sm.get(STASH_ACTIVE).map(|d| d.len()).unwrap_or(0) == 0,
        "source stash emptied",
    );
    let found = mgr.sm.get("found").expect("found stash");
    assert_eq!(found.len(), 2, "destination stash received both states");
}
