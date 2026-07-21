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

/// Regression for angr-ph300.26: `_reset_for_stage` cleared only a hardcoded
/// five-name stash list, so `pruned` and every technique stash
/// (cut/spinning/timeout/…) survived the reset with their states — and their
/// Z3 solver clones — alive for the remainder of the session.
#[test]
fn reset_for_stage_clears_pruned_and_technique_stashes() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let mut found_id = 0u64;
    for stash in [
        "found",
        "pruned",
        "cut",
        "spinning",
        "timeout",
        "not_unique",
        "merge_waiting_0",
        "deadended",
    ] {
        let mut s = RustSimState::new("amd64").expect("state");
        s.set_register("rax", RustBV::concrete(0x10, 64));
        if stash == "found" {
            found_id = s.state_id();
        }
        mgr.sm.push(stash, s);
    }

    let kept = mgr._reset_for_stage(found_id).expect("reset for stage");
    assert_eq!(kept, found_id, "reset returns the retained state id");

    for stash in [
        "found",
        "pruned",
        "cut",
        "spinning",
        "timeout",
        "not_unique",
        "merge_waiting_0",
        "deadended",
    ] {
        assert_eq!(mgr.sm.count(stash), 0, "{stash} emptied by reset_for_stage");
    }
    assert_eq!(
        mgr.sm.count(STASH_ACTIVE),
        1,
        "only the found state remains"
    );
    assert_eq!(
        mgr.sm.stash_of(found_id),
        Some(STASH_ACTIVE),
        "found state re-indexed onto active",
    );
}

/// Regression for angr-ph300.26: the `active.retain(...)` that drops the
/// non-found states used to bypass the bookkeeping maps entirely, so
/// `state_stash(dropped_id)` kept answering `active` and `state_roots` grew
/// without bound across stages.
#[test]
fn reset_for_stage_unindexes_dropped_active_states() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let mut found = RustSimState::new("amd64").expect("state");
    found.set_register("rax", RustBV::concrete(0x1, 64));
    let found_id = found.state_id();
    mgr.sm.push("found", found);
    mgr.sm.set_root(found_id, found_id);

    let mut dropped_ids = Vec::new();
    for v in [0x20u128, 0x30, 0x40] {
        let mut s = RustSimState::new("amd64").expect("state");
        s.set_register("rax", RustBV::concrete(v, 64));
        let id = s.state_id();
        dropped_ids.push(id);
        mgr.sm.push(STASH_ACTIVE, s);
        mgr.sm.set_root(id, id);
    }

    mgr._reset_for_stage(found_id).expect("reset for stage");

    for id in &dropped_ids {
        assert_eq!(
            mgr.sm.stash_of(*id),
            None,
            "dropped active state {id} unindexed",
        );
        assert_eq!(
            mgr.sm.get_root(*id),
            None,
            "dropped active state {id} root removed",
        );
    }
    // The retained state keeps both its index entry and its lineage root.
    assert_eq!(mgr.sm.stash_of(found_id), Some(STASH_ACTIVE));
    assert_eq!(mgr.sm.get_root(found_id), Some(found_id));
}

/// The found state must survive even when it is the sole occupant of the
/// `found` stash and `active` is empty — the clear-all sweep must skip
/// `active` *after* the move, not clear the state it just placed there.
#[test]
fn reset_for_stage_keeps_moved_state_when_active_was_empty() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let mut found = RustSimState::new("amd64").expect("state");
    found.set_register("rax", RustBV::concrete(0x99, 64));
    let found_id = found.state_id();
    mgr.sm.push("found", found);

    mgr._reset_for_stage(found_id).expect("reset for stage");

    assert_eq!(mgr.sm.count(STASH_ACTIVE), 1);
    assert_eq!(mgr.sm.count("found"), 0);
    assert_eq!(mgr.sm.stash_of(found_id), Some(STASH_ACTIVE));
}

/// A missing found state is still a hard error, and must not disturb the
/// existing stashes on the way out.
#[test]
fn reset_for_stage_errors_on_unknown_state() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let mut s = RustSimState::new("amd64").expect("state");
    s.set_register("rax", RustBV::concrete(0x5, 64));
    mgr.sm.push("pruned", s);

    assert!(mgr._reset_for_stage(999_999).is_err(), "unknown id errors");
    assert_eq!(
        mgr.sm.count("pruned"),
        1,
        "stashes untouched when the move fails",
    );
}
