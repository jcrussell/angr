// Tests for exploration/native_technique.rs — the in-Rust exploration
// techniques applied by `apply_native_techniques` (split out of
// helpers_tests.rs alongside the source split, angr-9ke6b.76).

use super::*;
use crate::stash::STASH_ACTIVE;
use crate::state::RustSimState;

// --- MergePoint native technique (native_technique.rs::apply_merge_point) ---
// ManualMergepoint parity (angr-op0dn.11.5). These drive apply_native_techniques
// directly, constructing states at the merge address (with per-state callstacks
// via push_call_frame) so the group/merge/release logic is asserted without a
// full binary.

/// Push a fresh amd64 state at `pc` with a synthetic single-frame callstack
/// whose return address is `ret` (the merge grouping key), then park it active.
fn push_active_at(mgr: &mut RustExplorationManager, pc: u64, ret: u64) -> u64 {
    let mut s = RustSimState::new("amd64").expect("state");
    s.set_pc(pc);
    // return_addr is the field merge_waiters_by_callstack keys on.
    s.push_call(0xdead, 0xbeef, ret, 0x7fff_0000);
    let sid = s.state_id();
    mgr.sm.push(STASH_ACTIVE, s);
    sid
}

/// Three same-callstack states at the merge address (active otherwise empty)
/// collapse to one merged state; states_merged_native counts all three.
#[test]
fn merge_point_merges_same_callstack_when_active_drains() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    for _ in 0..3 {
        push_active_at(&mut mgr, 0x1000, 0xAAAA);
    }
    mgr.register_merge_point(0x1000, 10);
    assert_eq!(mgr.native_technique_count(), 1);

    mgr.apply_native_techniques();

    let active = mgr.sm.get(STASH_ACTIVE).expect("active");
    assert_eq!(active.len(), 1, "3 same-callstack waiters merge to 1");
    assert_eq!(active[0].pc(), 0x1000, "merged state sits at the merge pc");
    let wait = mgr.sm.get("merge_waiting_0x1000").expect("wait stash");
    assert!(wait.is_empty(), "waiters consumed by the merge");
    assert_eq!(mgr.states_merged_native, 3);
}

/// Waiters with two distinct callstacks form two groups -> two merged states.
#[test]
fn merge_point_groups_by_callstack() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    // Two per callstack so each group actually merges (>=2).
    push_active_at(&mut mgr, 0x2000, 0xA);
    push_active_at(&mut mgr, 0x2000, 0xA);
    push_active_at(&mut mgr, 0x2000, 0xB);
    push_active_at(&mut mgr, 0x2000, 0xB);
    mgr.register_merge_point(0x2000, 10);

    mgr.apply_native_techniques();

    let active = mgr.sm.get(STASH_ACTIVE).expect("active");
    assert_eq!(active.len(), 2, "two callstack groups -> two merged states");
    assert!(mgr.sm.get("merge_waiting_0x2000").unwrap().is_empty());
    assert_eq!(mgr.states_merged_native, 4);
}

/// A lone waiter is released back to active unmerged: count preserved, counter
/// untouched.
#[test]
fn merge_point_single_waiter_released_unmerged() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let sid = push_active_at(&mut mgr, 0x3000, 0xC);
    mgr.register_merge_point(0x3000, 10);

    mgr.apply_native_techniques();

    let active = mgr.sm.get(STASH_ACTIVE).expect("active");
    assert_eq!(active.len(), 1, "lone waiter released back to active");
    assert_eq!(active[0].state_id(), sid, "same state, not a merge product");
    assert!(mgr.sm.get("merge_waiting_0x3000").unwrap().is_empty());
    assert_eq!(mgr.states_merged_native, 0, "nothing merged");
}

/// While the active frontier is non-empty and under the round limit the merge
/// is deferred; once `wait_counter_limit` post-step rounds elapse it fires even
/// with a live non-merge state still active.
#[test]
fn merge_point_defers_then_fires_on_counter() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    push_active_at(&mut mgr, 0x4000, 0xD);
    push_active_at(&mut mgr, 0x4000, 0xD);
    // A live non-merge state keeps the frontier non-empty across rounds.
    let live = push_active_at(&mut mgr, 0x5000, 0xE);
    mgr.register_merge_point(0x4000, 2);

    // Round 1: two waiters parked, but active still holds the live state and
    // counter (1) < limit (2) -> no merge yet.
    mgr.apply_native_techniques();
    assert_eq!(mgr.sm.get("merge_waiting_0x4000").unwrap().len(), 2);
    assert_eq!(mgr.states_merged_native, 0, "merge deferred round 1");
    assert_eq!(
        mgr.sm.get(STASH_ACTIVE).unwrap().len(),
        1,
        "live state stays"
    );

    // Round 2: nothing new arrives, counter (2) reaches the limit -> merge
    // fires even though the live state is still active.
    mgr.apply_native_techniques();
    assert_eq!(mgr.states_merged_native, 2, "counter forced the merge");
    assert!(mgr.sm.get("merge_waiting_0x4000").unwrap().is_empty());
    let active_ids = mgr.sm.state_ids(STASH_ACTIVE);
    assert!(active_ids.contains(&live), "live state untouched");
    assert_eq!(active_ids.len(), 2, "live state + merged product");
}

// --- Timeout / LengthLimiter / LoopBound native techniques (angr-9ke6b.77) -
// The other three `apply_native_techniques` arms. LengthLimiter and LoopBound
// share the "scan active, collect indices, remove in reverse, re-home the
// removals" shape, so every test here pins BOTH which states moved and which
// survivors stayed (by id, in order): a forward-iterating index removal
// shifts the tail and takes the wrong states, which a count-only assertion
// would happily accept.

/// Push an amd64 state into the active stash whose block history is `blocks`
/// (the input LengthLimiter and LoopBound scan). Returns its state id.
fn push_active_with_history(mgr: &mut RustExplorationManager, blocks: &[u64]) -> u64 {
    let mut s = RustSimState::new("amd64").expect("state");
    for &b in blocks {
        s.add_to_history(b);
    }
    let sid = s.state_id();
    mgr.sm.push(STASH_ACTIVE, s);
    sid
}

/// Backdate every registered Timeout's start so the deadline is unambiguously
/// in the past. The arm arms `start_time` lazily on its first apply, so a
/// freshly-registered 0.0s timeout would otherwise race the clock's
/// resolution (`elapsed() > 0.0` on a coarse clock can read false).
fn expire_timeouts(mgr: &mut RustExplorationManager) {
    for tech in &mut mgr.native_techniques {
        if let NativeTechnique::Timeout { start_time, .. } = tech {
            *start_time =
                std::time::Instant::now().checked_sub(std::time::Duration::from_secs(3600));
        }
    }
}

/// Before the deadline the Timeout arm is inert: active is untouched and the
/// run is not reported complete.
#[test]
fn timeout_before_deadline_leaves_active_untouched() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let a = push_active_with_history(&mut mgr, &[0x1000]);
    let b = push_active_with_history(&mut mgr, &[0x2000]);
    mgr.register_timeout(3600.0);

    assert!(
        !mgr.apply_native_techniques(),
        "deadline not reached -> run not complete"
    );
    assert_eq!(mgr.sm.state_ids(STASH_ACTIVE), vec![a, b]);
    assert!(
        mgr.sm
            .get("timeout")
            .expect("register_timeout materializes the stash")
            .is_empty()
    );
}

/// Past the deadline the whole active frontier drains into "timeout" (order
/// preserved) and the arm reports the run complete.
#[test]
fn timeout_after_deadline_drains_active_into_timeout_stash() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let a = push_active_with_history(&mut mgr, &[0x1000]);
    let b = push_active_with_history(&mut mgr, &[0x2000]);
    mgr.register_timeout(0.0);
    expire_timeouts(&mut mgr);

    assert!(
        mgr.apply_native_techniques(),
        "expired timeout completes the run"
    );
    assert!(mgr.sm.get(STASH_ACTIVE).expect("active").is_empty());
    assert_eq!(mgr.sm.state_ids("timeout"), vec![a, b]);
}

/// An expired timeout with nothing active still reports complete (the run
/// ends on the deadline, not on having states to move).
#[test]
fn timeout_after_deadline_completes_with_empty_active() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    mgr.register_timeout(0.0);
    expire_timeouts(&mut mgr);

    assert!(mgr.apply_native_techniques());
    assert!(mgr.sm.get("timeout").expect("timeout stash").is_empty());
}

/// LengthLimiter with `drop=false` moves only the over-limit states to "cut"
/// and leaves the survivors in their original active order.
#[test]
fn length_limiter_cuts_over_limit_states_and_keeps_survivors_in_order() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    // History lengths 4, 2, 5, 1, 6 against max_length 3 -> active indices
    // 0, 2 and 4 are over the limit (non-adjacent, so reverse removal matters).
    let a = push_active_with_history(&mut mgr, &[1, 2, 3, 4]);
    let b = push_active_with_history(&mut mgr, &[1, 2]);
    let c = push_active_with_history(&mut mgr, &[1, 2, 3, 4, 5]);
    let d = push_active_with_history(&mut mgr, &[1]);
    let e = push_active_with_history(&mut mgr, &[1, 2, 3, 4, 5, 6]);
    mgr.register_length_limiter(3, false);

    assert!(
        !mgr.apply_native_techniques(),
        "LengthLimiter never completes the run"
    );
    assert_eq!(
        mgr.sm.state_ids(STASH_ACTIVE),
        vec![b, d],
        "exactly the under-limit states survive, in order"
    );
    let mut cut = mgr.sm.state_ids("cut");
    cut.sort_unstable();
    let mut expected = vec![a, c, e];
    expected.sort_unstable();
    assert_eq!(cut, expected);
}

/// `drop=true` discards the over-limit states outright — and never
/// materializes the "cut" stash in the first place.
#[test]
fn length_limiter_drop_discards_instead_of_cutting() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    push_active_with_history(&mut mgr, &[1, 2, 3]);
    let survivor = push_active_with_history(&mut mgr, &[1]);
    push_active_with_history(&mut mgr, &[1, 2, 3, 4]);
    mgr.register_length_limiter(1, true);

    mgr.apply_native_techniques();

    assert_eq!(mgr.sm.state_ids(STASH_ACTIVE), vec![survivor]);
    assert!(
        mgr.sm.get("cut").is_none(),
        "drop=true must not create the cut stash"
    );
}

/// The limit is strictly-greater-than: a history of exactly `max_length`
/// blocks survives, one block more is cut.
#[test]
fn length_limiter_boundary_is_strictly_greater_than_max() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let at_limit = push_active_with_history(&mut mgr, &[1, 2, 3]);
    let over_limit = push_active_with_history(&mut mgr, &[1, 2, 3, 4]);
    mgr.register_length_limiter(3, false);

    mgr.apply_native_techniques();

    assert_eq!(mgr.sm.state_ids(STASH_ACTIVE), vec![at_limit]);
    assert_eq!(mgr.sm.state_ids("cut"), vec![over_limit]);
}

/// LoopBound moves the spinning states to the discard stash; a state whose
/// hottest address appears exactly `bound` times survives (the bound is
/// strictly-greater-than, same as LengthLimiter's).
#[test]
fn loop_bound_moves_spinning_states_to_discard_stash() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    // Spinners at active indices 1 and 3 (0x10 seen 3x > bound 2).
    let at_bound = push_active_with_history(&mut mgr, &[0x10, 0x20, 0x10]);
    let spin_a = push_active_with_history(&mut mgr, &[0x10, 0x20, 0x10, 0x30, 0x10]);
    let straight = push_active_with_history(&mut mgr, &[0x10, 0x20, 0x30, 0x40]);
    let spin_b = push_active_with_history(&mut mgr, &[0x99, 0x99, 0x99]);
    mgr.register_loop_bound(2, "spinning");

    assert!(
        !mgr.apply_native_techniques(),
        "LoopBound never completes the run"
    );
    assert_eq!(
        mgr.sm.state_ids(STASH_ACTIVE),
        vec![at_bound, straight],
        "exactly-at-bound and non-looping states survive, in order"
    );
    let mut spun = mgr.sm.state_ids("spinning");
    spun.sort_unstable();
    let mut expected = vec![spin_a, spin_b];
    expected.sort_unstable();
    assert_eq!(spun, expected);
}

/// The discard stash name is honored: nothing lands in the "spinning" default.
#[test]
fn loop_bound_honors_custom_discard_stash_name() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let spinner = push_active_with_history(&mut mgr, &[0x40, 0x40]);
    mgr.register_loop_bound(1, "my_spin");

    mgr.apply_native_techniques();

    assert!(mgr.sm.get(STASH_ACTIVE).expect("active").is_empty());
    assert_eq!(mgr.sm.state_ids("my_spin"), vec![spinner]);
    assert!(
        mgr.sm.get("spinning").is_none(),
        "the default stash name must not be created"
    );
}

/// Under `drop_terminal_states` the spinners are dropped rather than filed:
/// they leave active, and the (pre-created) discard stash stays empty.
#[test]
fn loop_bound_drops_spinners_when_drop_terminal_states_is_set() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    push_active_with_history(&mut mgr, &[0x40, 0x40]);
    let survivor = push_active_with_history(&mut mgr, &[0x40]);
    mgr.register_loop_bound(1, "spinning");
    mgr.set_drop_terminal_states(true);

    mgr.apply_native_techniques();

    assert_eq!(mgr.sm.state_ids(STASH_ACTIVE), vec![survivor]);
    assert!(
        mgr.sm
            .get("spinning")
            .expect("register_loop_bound pre-creates the stash")
            .is_empty(),
        "drop_terminal_states discards instead of filing"
    );
}
