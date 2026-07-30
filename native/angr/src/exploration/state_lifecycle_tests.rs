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

// ---------------------------------------------------------------------------
// on_state_removed wiring for the Python move/reset APIs (angr-myzjx.25)
//
// LoopHeadRoundRobin memoizes a per-state bucket key in its key_cache and only
// evicts on `on_state_removed`. The move/reset APIs relocate states out of
// STASH_ACTIVE outside `policy.select`, so each must notify or the memo leaks
// for the life of the policy. These pin the notify calls (and their
// `from_stash == STASH_ACTIVE` guards) using a spy policy that records every
// notified id.
// ---------------------------------------------------------------------------

/// Records every `state_id` passed to `on_state_removed`; forwards the deque
/// operations to `Lifo` so nothing else changes.
#[derive(Default)]
struct SpyPolicy {
    removed: std::sync::Mutex<Vec<u64>>,
}

impl selection_policy::SelectionPolicy for SpyPolicy {
    fn select(
        &self,
        active: &mut std::collections::VecDeque<RustSimState>,
    ) -> Option<RustSimState> {
        selection_policy::Lifo.select(active)
    }

    fn on_fork(&self, active: &mut std::collections::VecDeque<RustSimState>, state: RustSimState) {
        selection_policy::Lifo.on_fork(active, state);
    }

    fn name(&self) -> &'static str {
        "spy"
    }

    fn on_state_removed(&self, state_id: u64) {
        self.removed.lock().expect("spy poisoned").push(state_id);
    }
}

fn spy_mgr() -> (RustExplorationManager, std::sync::Arc<SpyPolicy>) {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let spy = std::sync::Arc::new(SpyPolicy::default());
    mgr.policy = spy.clone();
    (mgr, spy)
}

fn sorted_removed(spy: &SpyPolicy) -> Vec<u64> {
    let mut v = spy.removed.lock().expect("spy poisoned").clone();
    v.sort_unstable();
    v
}

/// `_move_states` (no-filter branch) draining STASH_ACTIVE must notify for
/// every relocated state.
#[test]
fn move_states_all_from_active_notifies_policy() {
    let (mut mgr, spy) = spy_mgr();
    let mut ids = Vec::new();
    for v in [0x1u128, 0x2, 0x3] {
        let mut s = RustSimState::new("amd64").expect("state");
        s.set_register("rax", RustBV::concrete(v, 64));
        ids.push(s.state_id());
        mgr.sm.push(STASH_ACTIVE, s);
    }
    ids.sort_unstable();

    mgr._move_states(STASH_ACTIVE, "found", None)
        .expect("move active->found");

    assert_eq!(
        sorted_removed(&spy),
        ids,
        "every state drained from STASH_ACTIVE must be notified",
    );
}

/// `_move_states` moving out of a *non*-active stash must NOT notify — the
/// LoopHead key_cache never held entries for those states.
#[test]
fn move_states_from_non_active_does_not_notify() {
    let (mut mgr, spy) = spy_mgr();
    for v in [0x1u128, 0x2] {
        let mut s = RustSimState::new("amd64").expect("state");
        s.set_register("rax", RustBV::concrete(v, 64));
        mgr.sm.push("found", s);
    }

    mgr._move_states("found", "active", None)
        .expect("move found->active");

    assert!(
        sorted_removed(&spy).is_empty(),
        "non-active source must not trigger on_state_removed",
    );
}

/// `_move_states` filtered branch draining STASH_ACTIVE must notify only the
/// states that actually pass the filter and move.
#[test]
fn move_states_filtered_from_active_notifies_moved_only() {
    let (mut mgr, spy) = spy_mgr();
    let mut all = Vec::new();
    for v in [0x10u128, 0x20, 0x30] {
        let mut s = RustSimState::new("amd64").expect("state");
        s.set_register("rax", RustBV::concrete(v, 64));
        all.push(s.state_id());
        mgr.sm.push(STASH_ACTIVE, s);
    }
    // Move exactly the first two ids.
    let keep: Vec<u64> = all[..2].to_vec();
    let mut expected = keep.clone();
    expected.sort_unstable();

    Python::initialize();
    Python::attach(|py| {
        let keep_set = keep.clone();
        let filter = pyo3::types::PyCFunction::new_closure(
            py,
            None,
            None,
            move |args, _kwargs| -> pyo3::PyResult<bool> {
                let id: u64 = args.get_item(0)?.extract()?;
                Ok(keep_set.contains(&id))
            },
        )
        .expect("closure")
        .into_any()
        .unbind();

        mgr._move_states(STASH_ACTIVE, "found", Some(filter))
            .expect("filtered move");
    });

    assert_eq!(
        sorted_removed(&spy),
        expected,
        "only the filtered-out (moved) active states are notified",
    );
}

/// `_move_state` (single) out of STASH_ACTIVE notifies; out of another stash
/// does not.
#[test]
fn move_state_single_active_guarded() {
    let (mut mgr, spy) = spy_mgr();

    let mut active = RustSimState::new("amd64").expect("state");
    active.set_register("rax", RustBV::concrete(0xaa, 64));
    let active_id = active.state_id();
    mgr.sm.push(STASH_ACTIVE, active);

    let mut parked = RustSimState::new("amd64").expect("state");
    parked.set_register("rax", RustBV::concrete(0xbb, 64));
    let parked_id = parked.state_id();
    mgr.sm.push("found", parked);

    mgr._move_state(active_id, STASH_ACTIVE, "found")
        .expect("active move");
    mgr._move_state(parked_id, "found", "pruned")
        .expect("non-active move");

    assert_eq!(
        sorted_removed(&spy),
        vec![active_id],
        "only the STASH_ACTIVE departure is notified",
    );
}

/// `_move_state` onto a state's own stash deliberately reorders-to-back rather
/// than no-opping (angr-04tw3.4): _StashDict.__setitem__ relies on this to
/// rebuild a stash in caller-specified order (angr-wxuo). The move of a
/// non-STASH_ACTIVE state must also not notify the policy.
#[test]
fn move_state_same_stash_reorders_to_back() {
    let (mut mgr, spy) = spy_mgr();

    let mut first = RustSimState::new("amd64").expect("state");
    first.set_register("rax", RustBV::concrete(0x11, 64));
    let first_id = first.state_id();
    mgr.sm.push("found", first);

    let mut second = RustSimState::new("amd64").expect("state");
    second.set_register("rax", RustBV::concrete(0x22, 64));
    let second_id = second.state_id();
    mgr.sm.push("found", second);

    // Same-stash move of the *front* state sends it to the back — this is the
    // reorder primitive stash-assignment builds on.
    let moved = mgr
        ._move_state(first_id, "found", "found")
        .expect("same-stash move");
    assert!(moved, "state present in its own stash reports moved=true");
    assert_eq!(
        mgr.sm.state_ids("found"),
        vec![second_id, first_id],
        "same-stash move reorders the moved state to the back",
    );

    // A state absent from the named stash reports false.
    let absent = mgr
        ._move_state(0xdead_beef, "found", "found")
        .expect("absent same-stash move");
    assert!(!absent, "state absent from its stash reports moved=false");

    // Neither move touched STASH_ACTIVE, so the policy is never notified.
    assert!(
        sorted_removed(&spy).is_empty(),
        "same-stash move out of a non-active stash must not notify the policy",
    );
}

/// `_reset_for_stage` drops every other active state — each must be notified.
#[test]
fn reset_for_stage_notifies_dropped_active() {
    let (mut mgr, spy) = spy_mgr();

    let mut dropped_ids = Vec::new();
    for v in [0x1u128, 0x2] {
        let mut s = RustSimState::new("amd64").expect("state");
        s.set_register("rax", RustBV::concrete(v, 64));
        dropped_ids.push(s.state_id());
        mgr.sm.push(STASH_ACTIVE, s);
    }
    dropped_ids.sort_unstable();

    let mut found = RustSimState::new("amd64").expect("state");
    found.set_register("rax", RustBV::concrete(0x99, 64));
    let found_id = found.state_id();
    mgr.sm.push("found", found);

    mgr._reset_for_stage(found_id).expect("reset");

    // The found state is *added* to active (not removed), so it must not be
    // notified; the two prior active states are dropped and must be.
    assert_eq!(
        sorted_removed(&spy),
        dropped_ids,
        "the dropped active states are notified, the promoted found state is not",
    );
}

/// `drop_state_from_stash` on "_copies" (its only real caller) removes a
/// non-active state and must not notify.
#[test]
fn drop_state_from_copies_does_not_notify() {
    let (mut mgr, spy) = spy_mgr();

    let mut s = RustSimState::new("amd64").expect("state");
    s.set_register("rax", RustBV::concrete(0x7, 64));
    let id = s.state_id();
    mgr.sm.push("_copies", s);

    assert!(mgr.drop_state_from_stash(id, "_copies"));
    assert!(
        sorted_removed(&spy).is_empty(),
        "dropping a non-active copy must not notify the policy",
    );
}
