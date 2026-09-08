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

/// Records every `state_id` passed to `on_state_removed` (and, for
/// angr-1o7i0, to `on_fork`); forwards the deque operations to `Lifo` so
/// nothing else changes.
#[derive(Default)]
struct SpyPolicy {
    removed: std::sync::Mutex<Vec<u64>>,
    forked: std::sync::Mutex<Vec<u64>>,
}

impl selection_policy::SelectionPolicy for SpyPolicy {
    fn select(
        &self,
        active: &mut std::collections::VecDeque<RustSimState>,
    ) -> Option<RustSimState> {
        selection_policy::Lifo.select(active)
    }

    fn on_fork(&self, active: &mut std::collections::VecDeque<RustSimState>, state: RustSimState) {
        self.forked
            .lock()
            .expect("spy poisoned")
            .push(state.state_id());
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

// ---------------------------------------------------------------------------
// on_fork wiring for the Python insert/move APIs (angr-1o7i0)
//
// The mirror of the block above: `on_state_removed` covers every departure
// from STASH_ACTIVE outside `policy.select`, and `policy.on_fork` must cover
// every *arrival* outside the run loop. Each of these entry points takes a
// caller-supplied destination stash that can be "active", and each used to
// insert with a raw `index_state` + `ensure_stash().push_back`. Stash contents
// are byte-identical either way, so the spy's `forked` log is the only
// observable — same test-design constraint as
// `merge_point_release_and_merge_go_through_on_fork` in
// `native_technique_tests.rs`.
// ---------------------------------------------------------------------------

fn sorted_forked(spy: &SpyPolicy) -> Vec<u64> {
    let mut v = spy.forked.lock().expect("spy poisoned").clone();
    v.sort_unstable();
    v
}

/// `_move_states` (no-filter branch) draining into STASH_ACTIVE must route
/// every relocated state through `on_fork`, and must preserve source order
/// (the built-in policies all append, so per-state pushes land back-to-back in
/// drain order — the bulk `append` this replaced had the same order).
#[test]
fn move_states_all_into_active_go_through_on_fork() {
    let (mut mgr, spy) = spy_mgr();
    let mut ids = Vec::new();
    for v in [0x1u128, 0x2, 0x3] {
        let mut s = RustSimState::new("amd64").expect("state");
        s.set_register("rax", RustBV::concrete(v, 64));
        ids.push(s.state_id());
        mgr.sm.push("found", s);
    }

    mgr._move_states("found", STASH_ACTIVE, None)
        .expect("move found->active");

    let mut expected = ids.clone();
    expected.sort_unstable();
    assert_eq!(
        sorted_forked(&spy),
        expected,
        "every state moved into STASH_ACTIVE must enter through on_fork",
    );
    assert_eq!(
        mgr.sm.state_ids(STASH_ACTIVE),
        ids,
        "source order is preserved across the per-state pushes",
    );
}

/// `_move_states` filtered branch moving into STASH_ACTIVE: only the states
/// that pass the filter reach `on_fork`, still in source order.
#[test]
fn move_states_filtered_into_active_go_through_on_fork() {
    let (mut mgr, spy) = spy_mgr();
    let mut all = Vec::new();
    for v in [0x10u128, 0x20, 0x30] {
        let mut s = RustSimState::new("amd64").expect("state");
        s.set_register("rax", RustBV::concrete(v, 64));
        all.push(s.state_id());
        mgr.sm.push("found", s);
    }
    let keep: Vec<u64> = all[..2].to_vec();

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

        mgr._move_states("found", STASH_ACTIVE, Some(filter))
            .expect("filtered move");
    });

    let mut expected = keep.clone();
    expected.sort_unstable();
    assert_eq!(
        sorted_forked(&spy),
        expected,
        "only the states that actually moved into active reach on_fork",
    );
    assert_eq!(
        mgr.sm.state_ids(STASH_ACTIVE),
        keep,
        "the filtered moves land in source order",
    );
}

/// The single-state entry points — `_create_state`, `_fork_state_to_stash`,
/// `_move_state` — must all route an `active` destination through `on_fork`
/// too. A non-active destination must NOT (`push_to_stash`'s other arm is a
/// plain indexed push).
#[test]
fn single_state_inserts_into_active_go_through_on_fork() {
    let (mut mgr, spy) = spy_mgr();

    let created = mgr._create_state(STASH_ACTIVE).expect("create active");
    let forked = mgr
        ._fork_state_to_stash(created, STASH_ACTIVE)
        .expect("fork into active");
    // Parked elsewhere, then moved back in: a move is not a fork, but it is
    // the exact mirror of the on_state_removed the reverse move fires.
    let parked = mgr._create_state("found").expect("create found");
    assert!(
        mgr._move_state(parked, "found", STASH_ACTIVE)
            .expect("move found->active"),
    );
    // Destination that is not active: no on_fork.
    let elsewhere = mgr._create_state("deferred").expect("create deferred");

    let mut expected = vec![created, forked, parked];
    expected.sort_unstable();
    assert_eq!(
        sorted_forked(&spy),
        expected,
        "create/fork/move into active all hit on_fork; the 'deferred' \
         create ({elsewhere}) and the 'found' create must not",
    );
    assert_eq!(
        mgr.sm.state_ids(STASH_ACTIVE),
        vec![created, forked, parked],
        "arrival order is preserved",
    );
    assert_eq!(
        mgr.sm.find_state(elsewhere).map(RustSimState::state_id),
        Some(elsewhere),
        "the non-active insert is still indexed by push_to_stash's other arm",
    );
}

/// angr-sqfj8.31: a `filter_fn` returning a *truthy non-bool* (the Python
/// predicate convention) must count as a match. The old
/// `extract::<bool>(py).unwrap_or(false)` read a non-empty string as "no
/// match" and moved nothing.
#[test]
fn move_states_filter_accepts_truthy_non_bool() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    for v in [0x10u128, 0x20] {
        let mut s = RustSimState::new("amd64").expect("state");
        s.set_register("rax", RustBV::concrete(v, 64));
        mgr.sm.push(STASH_ACTIVE, s);
    }

    Python::initialize();
    let moved = Python::attach(|py| {
        // Returns a non-empty `str` — truthy, but not a `bool` or an `int`.
        let filter = pyo3::types::PyCFunction::new_closure(
            py,
            None,
            None,
            |_args, _kwargs| -> pyo3::PyResult<String> { Ok("yes".to_owned()) },
        )
        .expect("closure")
        .into_any()
        .unbind();
        mgr._move_states(STASH_ACTIVE, "found", Some(filter))
            .expect("filtered move")
    });

    assert_eq!(moved, 2, "truthy non-bool filter results count as matches");
    assert_eq!(mgr.sm.stashes().get("found").map(VecDeque::len), Some(2));
    assert_eq!(
        mgr.sm.stashes().get(STASH_ACTIVE).map(VecDeque::len),
        Some(0)
    );
}

/// angr-sqfj8.31: an exception raised while evaluating the filter's return
/// value for truth must surface as `Err`, not be swallowed into "no match".
#[test]
fn move_states_filter_bool_error_propagates() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let mut s = RustSimState::new("amd64").expect("state");
    s.set_register("rax", RustBV::concrete(0x10, 64));
    mgr.sm.push(STASH_ACTIVE, s);

    Python::initialize();
    Python::attach(|py| {
        // `__bool__` raises, so `call1` succeeds but the truth test does not.
        let module = pyo3::types::PyModule::from_code(
            py,
            &std::ffi::CString::new(
                "class Boom:\n    def __bool__(self):\n        raise ValueError('boom')\n\ndef f(sid):\n    return Boom()\n",
            )
            .expect("cstring"),
            &std::ffi::CString::new("boom.py").expect("cstring"),
            &std::ffi::CString::new("boom").expect("cstring"),
        )
        .expect("module");
        let filter = module.getattr("f").expect("f").unbind();

        let err = mgr
            ._move_states(STASH_ACTIVE, "found", Some(filter))
            .expect_err("__bool__ raising must propagate");
        assert!(err.to_string().contains("boom"), "got: {err}");
    });

    // Nothing moved, and the source stash is untouched.
    assert_eq!(
        mgr.sm.stashes().get(STASH_ACTIVE).map(VecDeque::len),
        Some(1)
    );
    assert_eq!(
        mgr.sm.stashes().get("found").map_or(0, VecDeque::len),
        0,
        "no state reached the destination stash",
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

// =============================================================================
// _merge_states — removal-tombstone survival through the manager-level path
// =============================================================================
//
// `RustSimState::merge` grew removal tombstones in the angr-9ke6b.121 follow-up
// so one branch's explicit `remove_hook` / `set_option(_, false)` / `unsetenv`
// can't be resurrected by a sibling that never touched the item, and
// `state/tests/merge_config.rs` covers that on `merge` directly. `_merge_states` is the
// entry point `register_merge_point`'s `NativeTechnique::MergePoint` actually
// reaches in a real run; it takes its own copy of each input first, and taking
// that copy with `fork` (which resets the tombstones by design) silently undid
// the whole fix on that path — hence `clone_for_merge` (angr-sqfj8.33).
//
// These drive `_merge_states` rather than a full `register_merge_point`
// exploration: the merge-point technique's only merge action *is* this call, so
// a run would add scheduler noise without covering anything more.

/// Seed a state carrying a hook, a sim option and an env var into `active`,
/// returning `(mgr, ancestor_id)`.
fn merge_tombstone_mgr() -> (RustExplorationManager, u64) {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let mut ancestor = RustSimState::new("amd64").expect("state");
    ancestor.add_hook(0x400000);
    ancestor.set_option("SHORT_READS", true);
    ancestor.setenv(b"PATH".to_vec(), b"/a".to_vec());
    let ancestor_id = ancestor.state_id();
    mgr.sm.push(STASH_ACTIVE, ancestor);
    mgr.rebuild_state_index();

    (mgr, ancestor_id)
}

/// Regression for angr-sqfj8.33: a branch that explicitly removed a hook /
/// option / env var must keep it removed after `_merge_states`, even though a
/// sibling branch still has it live. `_merge_states` forking its inputs reset
/// the tombstones before `merge` could consult them, so every removal was
/// resurrected on this path.
#[test]
fn merge_states_removal_wins_over_untouched_sibling() {
    let (mut mgr, ancestor_id) = merge_tombstone_mgr();

    let a_id = mgr
        ._fork_state_to_stash(ancestor_id, STASH_ACTIVE)
        .expect("fork a");
    let b_id = mgr
        ._fork_state_to_stash(ancestor_id, STASH_ACTIVE)
        .expect("fork b");

    // `a` removes all three; `b` never touches any of them.
    {
        let a = mgr.find_state_mut(a_id).expect("state a");
        a.remove_hook(0x400000);
        a.set_option("SHORT_READS", false);
        a.unsetenv(b"PATH");
    }

    let merged_id = mgr
        ._merge_states(vec![a_id, b_id], STASH_ACTIVE)
        .expect("merge a+b");
    let merged = mgr.find_state(merged_id).expect("merged state");

    assert!(
        !merged.is_hooked(0x400000),
        "a's remove_hook must survive _merge_states, not be resurrected by b"
    );
    assert!(
        !merged.has_option("SHORT_READS"),
        "a's set_option(false) must survive _merge_states"
    );
    assert!(
        merged.environment().get(b"PATH".as_slice()).is_none(),
        "a's unsetenv must survive _merge_states"
    );
}

/// The tombstone carry must not regress the addition direction: an item only
/// one branch added is still unioned into the `_merge_states` result.
#[test]
fn merge_states_still_unions_additions_alongside_removals() {
    let (mut mgr, ancestor_id) = merge_tombstone_mgr();

    let a_id = mgr
        ._fork_state_to_stash(ancestor_id, STASH_ACTIVE)
        .expect("fork a");
    let b_id = mgr
        ._fork_state_to_stash(ancestor_id, STASH_ACTIVE)
        .expect("fork b");

    {
        let a = mgr.find_state_mut(a_id).expect("state a");
        a.remove_hook(0x400000);
    }
    {
        let b = mgr.find_state_mut(b_id).expect("state b");
        b.add_hook(0x500000);
        b.set_option("OTHER_ONLY", true);
        b.setenv(b"HOME".to_vec(), b"/root".to_vec());
    }

    let merged_id = mgr
        ._merge_states(vec![a_id, b_id], STASH_ACTIVE)
        .expect("merge a+b");
    let merged = mgr.find_state(merged_id).expect("merged state");

    assert!(
        !merged.is_hooked(0x400000),
        "a's removal still wins alongside an unrelated addition from b"
    );
    assert!(
        merged.is_hooked(0x500000),
        "b's brand-new hook is still unioned in"
    );
    assert!(
        merged.has_option("OTHER_ONLY"),
        "b's brand-new option is still unioned in"
    );
    assert_eq!(
        merged
            .environment()
            .get(b"HOME".as_slice())
            .map(|v| v.as_slice()),
        Some(b"/root".as_slice()),
        "b's brand-new env var is still unioned in"
    );
}

/// A branch that removes and then re-adds on its own before the merge must end
/// up with the item live: `clone_for_merge` carries the tombstone *set*, and
/// the re-add has to have already cleared its entry.
#[test]
fn merge_states_reinstated_item_survives_own_tombstone() {
    let (mut mgr, ancestor_id) = merge_tombstone_mgr();

    let a_id = mgr
        ._fork_state_to_stash(ancestor_id, STASH_ACTIVE)
        .expect("fork a");
    let b_id = mgr
        ._fork_state_to_stash(ancestor_id, STASH_ACTIVE)
        .expect("fork b");

    {
        let a = mgr.find_state_mut(a_id).expect("state a");
        a.remove_hook(0x400000);
        a.add_hook(0x400000);
        a.set_option("SHORT_READS", false);
        a.set_option("SHORT_READS", true);
        a.unsetenv(b"PATH");
        a.setenv(b"PATH".to_vec(), b"/a2".to_vec());
    }

    let merged_id = mgr
        ._merge_states(vec![a_id, b_id], STASH_ACTIVE)
        .expect("merge a+b");
    let merged = mgr.find_state(merged_id).expect("merged state");

    assert!(
        merged.is_hooked(0x400000),
        "re-added hook must be live after _merge_states"
    );
    assert!(
        merged.has_option("SHORT_READS"),
        "re-enabled option must be live after _merge_states"
    );
    assert_eq!(
        merged
            .environment()
            .get(b"PATH".as_slice())
            .map(|v| v.as_slice()),
        Some(b"/a2".as_slice()),
        "re-setenv value must win after _merge_states"
    );
}

/// Regression for angr-6cp06.27: `_reset_for_stage` swept the stashes but not
/// the two non-stash buckets a live state can sit in, so a state parked in
/// `pending_callbacks` (mid-Python-callback) or `pending_parallel_bounces` (a
/// wave's undispatched bounce) survived the stage transition and leaked its
/// stage-1 constraints/history into the next stage's frontier. Both buckets
/// must be emptied, and each discarded id unindexed + un-rooted the same way
/// the dropped actives are.
#[test]
fn reset_for_stage_clears_pending_callbacks_and_parked_bounces() {
    use crate::exploration::callback_types::{CallbackReason, PendingCallback};
    use crate::exploration::core_outcome::BounceKind;
    use crate::exploration::state_id::StateId;

    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let mut found = RustSimState::new("amd64").expect("state");
    found.set_pc(0x40_1000);
    let found_id = found.state_id();
    mgr.sm.push("found", found);
    mgr.sm.set_root(found_id, found_id);

    // A stage-1 state parked across a Python callback: owned by
    // `pending_callbacks`, resident in no stash, but still indexed/rooted
    // from before `policy.select` popped it off active.
    let mut parked = RustSimState::new("amd64").expect("state");
    parked.set_pc(0x40_2000);
    let parked_id = parked.state_id();
    mgr.sm.set_root(parked_id, parked_id);
    mgr.pending_callbacks.insert(
        StateId::new(parked_id),
        PendingCallback::lightweight(
            parked,
            CallbackReason::Error {
                message: "parked".to_string(),
            },
        ),
    );

    // A stage-1 bounce a parallel wave could not dispatch.
    let mut bounced = RustSimState::new("amd64").expect("state");
    bounced.set_pc(0x40_3000);
    let bounced_id = bounced.state_id();
    mgr.sm.set_root(bounced_id, bounced_id);
    mgr.pending_parallel_bounces
        .push((bounced, BounceKind::Hook { addr: 0x40_5000 }, bounced_id));

    let kept = mgr._reset_for_stage(found_id).expect("reset for stage");
    assert_eq!(kept, found_id);

    assert!(
        mgr.pending_callbacks.is_empty(),
        "parked callback state discarded by reset_for_stage",
    );
    assert!(
        mgr.pending_parallel_bounces.is_empty(),
        "parked bounce discarded by reset_for_stage",
    );
    // `find_state` walks all three buckets — neither stage-1 state is
    // reachable through any of them any more.
    assert!(mgr.find_state(parked_id).is_none(), "parked state gone");
    assert!(mgr.find_state(bounced_id).is_none(), "bounced state gone");

    for id in [parked_id, bounced_id] {
        assert_eq!(mgr.sm.stash_of(id), None, "discarded state {id} unindexed");
        assert_eq!(mgr.sm.get_root(id), None, "discarded state {id} un-rooted");
    }

    // The retained state is untouched: still the sole active, still rooted.
    assert_eq!(mgr.sm.count(STASH_ACTIVE), 1);
    assert_eq!(mgr.sm.stash_of(found_id), Some(STASH_ACTIVE));
    assert_eq!(mgr.sm.get_root(found_id), Some(found_id));
}
