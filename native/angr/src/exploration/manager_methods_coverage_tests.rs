//! Bucket x method coverage table for the "misses states parked outside a
//! stash" bug class (angr-sqfj8.32, angr-c7xno.21/.97, angr-03vl4.10/.15/.23
//! — bd category `state-tracking-coverage-gap`).
//!
//! `manager_methods_tests.rs` and `pending_api_tests.rs` each pin specific
//! instances of this bug by hand: one `#[test]` per (method, bucket) pair
//! that was found broken by an audit. That shape does not scale — a NEW
//! bucket (there have been three so far: the stashes, `pending_callbacks`,
//! and `pending_parallel_bounces` — see `pending_api.rs::_parent_of`'s
//! bucket census and `run_loop.rs`'s "third bucket" doc comments) or a NEW
//! "applies to every state" method means writing another bespoke test from
//! scratch, and nothing forces that to happen.
//!
//! This file is the 2D generalization: [`Bucket`] enumerates every place a
//! state can be parked, [`park`] and [`find_anywhere`] are the shared
//! park-in-bucket-X / find-it-again-wherever-it-ended-up helper pair, and
//! each column below (`set_deterministic`, `set_max_history`,
//! `_active_states_map_memory`, the census accessors) is one small shared
//! assertion helper reused by a compact grid of one-line `#[test]` wrappers,
//! one per (bucket, column) cell that method's contract actually covers.
//! Adding a fourth bucket is one new [`Bucket`] variant, one new [`park`]
//! arm, and one new one-line `#[test]` per existing column; adding a new
//! "applies to every state" method is one new assertion helper plus one
//! one-line `#[test]` per bucket its contract covers — never a new
//! hand-rolled test body.
//!
//! Not every cell is in every column's contract — `_active_states_map_memory`
//! is deliberately scoped to `STASH_ACTIVE` only (not every stash) and the
//! census accessors deliberately exclude `pending_callbacks` (a parked
//! callback is in-flight, not a collected result) — see each assertion
//! helper's doc comment for the exact domain. Those out-of-contract cells are
//! still asserted, just against "not reached" / "not counted" instead of
//! "reached" / "counted", so a future accidental widening of the scope is
//! caught too, not just a narrowing.
use super::*;
use crate::exploration::core_outcome::BounceKind;
use crate::stash::{STASH_ACTIVE, STASH_FOUND};
use crate::state::RustSimState;

/// Every place a `RustSimState` can be parked outside a plain "no longer
/// tracked" state. `ActiveStash` and `OtherStash` are both plain stash
/// entries — `OtherStash` (`STASH_FOUND`) stands in for "any non-active
/// stash", since every column here that reaches stashes at all does so via a
/// name-agnostic loop over `self.sm.stashes_mut()` and never special-cases a
/// particular non-active name.
#[derive(Clone, Copy, Debug)]
enum Bucket {
    ActiveStash,
    OtherStash,
    PendingCallback,
    ParallelBounce,
}

/// Push one fresh, uniquely-addressed state into `bucket` and return its id.
/// The shared "park" half of the park/find helper pair every test below
/// builds on.
fn park(mgr: &mut RustExplorationManager, bucket: Bucket) -> u64 {
    let mut state = RustSimState::new("amd64").expect("state");
    match bucket {
        Bucket::ActiveStash => {
            state.set_pc(0x40_1000);
            let id = state.state_id();
            mgr.sm.push(STASH_ACTIVE, state);
            id
        }
        Bucket::OtherStash => {
            state.set_pc(0x40_2000);
            let id = state.state_id();
            mgr.sm.push(STASH_FOUND, state);
            id
        }
        Bucket::PendingCallback => {
            state.set_pc(0x40_3000);
            let id = state.state_id();
            mgr.pending_callbacks.insert(
                StateId::new(id),
                PendingCallback::lightweight(state, CallbackReason::Syscall { num: Some(60) }),
            );
            id
        }
        Bucket::ParallelBounce => {
            state.set_pc(0x40_4000);
            let id = state.state_id();
            // A Hook bounce is replayable (`bounce_target_addr` returns
            // `Some`), so it also counts toward the census columns below —
            // the harder-to-satisfy case, per `park_two_bounces` in
            // `manager_methods_tests.rs`.
            mgr.pending_parallel_bounces
                .push((state, BounceKind::Hook { addr: 0x40_9000 }, id));
            id
        }
    }
}

/// Find a parked state by id wherever it lives — the shared "find" half of
/// the pair, searching all three known storage locations. Deliberately
/// independent of [`RustExplorationManager::find_state`] (which searches the
/// same three since angr-eukuf) so a regression in that lookup shows up as a
/// failure of the `with_state` column below, not as every other column
/// silently losing its state.
fn find_anywhere(mgr: &RustExplorationManager, id: u64) -> Option<&RustSimState> {
    if let Some(state) = mgr.sm.stashes().values().flatten().find(|s| s.state_id() == id) {
        return Some(state);
    }
    if let Some(pending) = mgr.pending_callbacks.get(&StateId::new(id)) {
        return Some(&pending.state);
    }
    mgr.parked_bounce_states().find(|s| s.state_id() == id)
}

// ===========================================================================
// Column: set_deterministic (angr-sqfj8.32 / angr-03vl4.10)
// ===========================================================================

/// `set_deterministic`'s contract covers every bucket: it loops every named
/// stash generically, then `pending_callbacks`, then `parked_bounce_states`.
/// Asserted over a true/false/true sequence so a one-way latch cannot pass.
#[cfg(feature = "vex-engine-z3")]
fn assert_set_deterministic_reaches(bucket: Bucket) {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let id = park(&mut mgr, bucket);

    for v in [true, false, true] {
        mgr.set_deterministic(v);
        let state = find_anywhere(&mgr, id)
            .unwrap_or_else(|| panic!("{bucket:?} state {id} unreachable after set({v})"));
        assert_eq!(
            constraints::state_is_deterministic(state),
            v,
            "{bucket:?} follows set_deterministic({v})",
        );
    }
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn set_deterministic_reaches_active_stash() {
    assert_set_deterministic_reaches(Bucket::ActiveStash);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn set_deterministic_reaches_other_stash() {
    assert_set_deterministic_reaches(Bucket::OtherStash);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn set_deterministic_reaches_pending_callback() {
    assert_set_deterministic_reaches(Bucket::PendingCallback);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn set_deterministic_reaches_parallel_bounce() {
    assert_set_deterministic_reaches(Bucket::ParallelBounce);
}

// ===========================================================================
// Column: set_max_history (angr-c7xno.21 / angr-03vl4.10)
// ===========================================================================

/// `set_max_history`'s contract covers every bucket, mirroring
/// `set_deterministic` above. Swept over several caps, including the
/// unlimited `0`, so neither a default nor a one-way latch can pass.
fn assert_set_max_history_reaches(bucket: Bucket) {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let id = park(&mut mgr, bucket);

    for max in [7usize, 3, 0, 42] {
        mgr.set_max_history(max);
        let state = find_anywhere(&mgr, id)
            .unwrap_or_else(|| panic!("{bucket:?} state {id} unreachable after set({max})"));
        assert_eq!(
            state.max_history(),
            max,
            "{bucket:?} follows set_max_history({max})",
        );
    }
}

#[test]
fn set_max_history_reaches_active_stash() {
    assert_set_max_history_reaches(Bucket::ActiveStash);
}

#[test]
fn set_max_history_reaches_other_stash() {
    assert_set_max_history_reaches(Bucket::OtherStash);
}

#[test]
fn set_max_history_reaches_pending_callback() {
    assert_set_max_history_reaches(Bucket::PendingCallback);
}

#[test]
fn set_max_history_reaches_parallel_bounce() {
    assert_set_max_history_reaches(Bucket::ParallelBounce);
}

// ===========================================================================
// Column: _active_states_map_memory (angr-03vl4.23 / angr-03vl4.10)
// ===========================================================================

/// `_active_states_map_memory`'s contract is narrower than the two config
/// broadcasts above: "every state that can still execute" means
/// `STASH_ACTIVE` specifically (not every stash), plus `pending_callbacks`
/// and `pending_parallel_bounces` (both of which re-enter `STASH_ACTIVE` on
/// resume/flush). `expect_reach == false` asserts the deliberate exclusion —
/// a non-active stash must NOT see the mapping — holds too, so a future
/// accidental widening (or narrowing) is caught either direction.
fn assert_active_states_map_memory(bucket: Bucket, expect_reach: bool) {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let id = park(&mut mgr, bucket);

    const ADDR: u64 = 0x44_000;
    let data = [0xde_u8, 0xad, 0xbe, 0xef];
    mgr._active_states_map_memory(ADDR, &data, 5);

    let state = find_anywhere(&mgr, id)
        .unwrap_or_else(|| panic!("{bucket:?} state {id} unreachable after map_memory"));
    assert_eq!(
        state.memory().is_mapped(ADDR),
        expect_reach,
        "{bucket:?} reached by _active_states_map_memory == {expect_reach}",
    );
    if expect_reach {
        // amd64 is little-endian, so the mapped byte sequence reads back
        // reversed.
        let expected = 0xefbe_adde_u64;
        let value = state
            .memory()
            .load_concrete(ADDR, 4, &state.solver().borrow())
            .expect("load mapped bytes")
            .as_u64()
            .expect("concrete bytes");
        assert_eq!(value, expected, "{bucket:?} sees the mapped bytes");
    }
}

#[test]
fn active_states_map_memory_reaches_active_stash() {
    assert_active_states_map_memory(Bucket::ActiveStash, true);
}

#[test]
fn active_states_map_memory_excludes_other_stash() {
    assert_active_states_map_memory(Bucket::OtherStash, false);
}

#[test]
fn active_states_map_memory_reaches_pending_callback() {
    assert_active_states_map_memory(Bucket::PendingCallback, true);
}

#[test]
fn active_states_map_memory_reaches_parallel_bounce() {
    assert_active_states_map_memory(Bucket::ParallelBounce, true);
}

// ===========================================================================
// Column: census (active_count / found_count / stash_counts, angr-03vl4.15)
// ===========================================================================

/// The census accessors' contract differs per bucket, unlike the three
/// broadcast columns above — this is real domain information, not
/// boilerplate, so each arm is spelled out rather than parametrized away:
///
/// * `ActiveStash` / `OtherStash` — a plain stash entry always counts
///   towards its own `stash_counts` key, and towards `active_count` /
///   `found_count` when that key is `active` / `found`.
/// * `PendingCallback` — deliberately invisible to every census accessor: a
///   parked callback is in-flight, not a collected result
///   (`stash_counts_censuses_every_stash_after_move_drop_clear` in
///   `manager_methods_tests.rs`).
/// * `ParallelBounce` — a replayable bounce folds into `active_count` and the
///   `active` `stash_counts` entry (not `found_count`), so a mid-explore
///   census does not depend on whether a flush happened to run yet.
fn assert_census_reflects(bucket: Bucket) {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let baseline_active = mgr.active_count();
    let baseline_found = mgr.found_count();

    park(&mut mgr, bucket);

    let (want_active_delta, want_found_delta) = match bucket {
        Bucket::ActiveStash | Bucket::ParallelBounce => (1, 0),
        Bucket::OtherStash => (0, 1),
        Bucket::PendingCallback => (0, 0),
    };
    assert_eq!(
        mgr.active_count(),
        baseline_active + want_active_delta,
        "{bucket:?} active_count delta",
    );
    assert_eq!(
        mgr.found_count(),
        baseline_found + want_found_delta,
        "{bucket:?} found_count delta",
    );

    Python::attach(|py| {
        let counts = mgr.stash_counts(py).expect("stash_counts");
        let total: usize = counts
            .iter()
            .map(|(_, v)| v.extract::<usize>().expect("count"))
            .sum();
        assert_eq!(
            total,
            want_active_delta + want_found_delta,
            "{bucket:?} stash_counts total delta",
        );
    });
}

#[test]
fn census_reflects_active_stash() {
    assert_census_reflects(Bucket::ActiveStash);
}

#[test]
fn census_reflects_other_stash() {
    assert_census_reflects(Bucket::OtherStash);
}

#[test]
fn census_excludes_pending_callback() {
    assert_census_reflects(Bucket::PendingCallback);
}

#[test]
fn census_reflects_parallel_bounce() {
    assert_census_reflects(Bucket::ParallelBounce);
}

// ===========================================================================
// Column: with_state / with_state_mut (find_state lookup, angr-eukuf)
// ===========================================================================

/// The read path's contract covers every bucket: `find_state` /
/// `find_state_mut` must reach any state the manager still holds, since
/// `with_state`/`with_state_mut` are how ~54 Python-facing methods (the
/// write-through proxy shims included) address a state by id. A miss is not a
/// wrong answer but a `state N not found` `PyValueError` — or, for the
/// internal `_parent_of` walk, a silently truncated ancestry.
///
/// The read leg pins the *identity* of what was found (its pc, unique per
/// bucket in [`park`]) rather than mere `is_some()`, and the write leg mutates
/// through `with_state_mut` and re-reads via the independent
/// [`find_anywhere`], so a lookup that hands back some other bucket's state,
/// or a copy the mutation does not stick to, fails too.
fn assert_with_state_reaches(bucket: Bucket) {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let id = park(&mut mgr, bucket);
    let pc = find_anywhere(&mgr, id).expect("parked state").pc();

    let seen = mgr
        .with_state(id, |state| Ok(state.pc()))
        .unwrap_or_else(|err| panic!("{bucket:?} with_state({id}) failed: {err}"));
    assert_eq!(seen, pc, "{bucket:?} with_state reaches the parked state");

    const NEW_PC: u64 = 0x40_8000;
    mgr.with_state_mut(id, |state| {
        state.set_pc(NEW_PC);
        Ok(())
    })
    .unwrap_or_else(|err| panic!("{bucket:?} with_state_mut({id}) failed: {err}"));
    assert_eq!(
        find_anywhere(&mgr, id).expect("parked state").pc(),
        NEW_PC,
        "{bucket:?} with_state_mut writes through to the parked state",
    );
}

#[test]
fn with_state_reaches_active_stash() {
    assert_with_state_reaches(Bucket::ActiveStash);
}

#[test]
fn with_state_reaches_other_stash() {
    assert_with_state_reaches(Bucket::OtherStash);
}

#[test]
fn with_state_reaches_pending_callback() {
    assert_with_state_reaches(Bucket::PendingCallback);
}

#[test]
fn with_state_reaches_parallel_bounce() {
    assert_with_state_reaches(Bucket::ParallelBounce);
}

/// An id can be resident in a stash *and* parked in
/// `pending_parallel_bounces` at once; `flush_parked_bounces_to_active` drops
/// the parked copy as the redundant duplicate in that case, so the stash copy
/// is the one carrying the path forward and `find_state` must return it — the
/// parked-bounce leg is a fallback, not an override.
#[test]
fn find_state_prefers_the_stash_copy_over_a_duplicate_parked_bounce() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    const RESIDENT_PC: u64 = 0x40_1000;
    let mut state = RustSimState::new("amd64").expect("state");
    state.set_pc(RESIDENT_PC);
    let id = state.state_id();

    // A same-id parked copy sitting at a different pc — snapshot round-trip
    // preserves the id, as `flush_drops_parked_bounce_whose_id_is_already_resident`
    // in `run_loop_tests.rs` relies on.
    let mut duplicate = RustSimState::from_snapshot(state.to_snapshot()).expect("round-trip");
    duplicate.set_pc(0x40_7000);
    mgr.sm.push(STASH_ACTIVE, state);
    mgr.pending_parallel_bounces
        .push((duplicate, BounceKind::Hook { addr: 0x40_9000 }, id));
    let resident_pc = RESIDENT_PC;

    assert_eq!(
        mgr.find_state(id).expect("state").pc(),
        resident_pc,
        "the stash copy wins over the duplicate parked bounce",
    );
    assert_eq!(
        mgr.find_state_mut(id).expect("state").pc(),
        resident_pc,
        "find_state_mut agrees with find_state on precedence",
    );
}
