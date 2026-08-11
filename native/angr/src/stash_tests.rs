// Unit tests for stash.rs (StashManager). Extracted as a `#[path]` sibling
// to keep the parent module focused on production code.

use super::*;

#[test]
fn test_new_has_default_stashes() {
    let mgr = StashManager::new();
    assert!(mgr.get(STASH_ACTIVE).is_some());
    assert!(mgr.get(STASH_FOUND).is_some());
    assert!(mgr.get(STASH_AVOID).is_some());
    assert!(mgr.get(STASH_DEADENDED).is_some());
    assert!(mgr.get(STASH_ERRORED).is_some());
    assert!(mgr.get(STASH_PRUNED).is_some());
    assert!(mgr.get(STASH_UNCONSTRAINED).is_some());
    assert_eq!(mgr.active_count(), 0);
    assert_eq!(mgr.count(STASH_FOUND), 0);
}

#[test]
fn test_counts() {
    let mgr = StashManager::new();
    let counts = mgr.counts();
    assert_eq!(counts.len(), 7);
    for &count in counts.values() {
        assert_eq!(count, 0);
    }
}

/// Regression guard for angr-9ke6b.197: every terminal counter must be
/// driven by the `STASH_*` constant, not by a copy of its string value.
///
/// The table below is keyed on the constants themselves, so if a future
/// change edits a constant's value the test still drives the same code
/// path — and it fails loudly if `push_or_drop_terminal`'s match arms are
/// ever re-hardcoded to literals that drift from the constants.
///
/// `STASH_ERRORED` is deliberately absent: it now panics here (angr-sqfj8.136)
/// and is covered by `test_push_errored_counts_indexes_and_never_drops` plus
/// `test_push_or_drop_terminal_rejects_errored`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_push_or_drop_terminal_counters_track_stash_constants() {
    type CounterFn = fn(&StashManager) -> u64;
    let cases: [(&str, CounterFn); 4] = [
        (STASH_AVOID, |m| m.avoided_count),
        (STASH_PRUNED, |m| m.pruned_count),
        (STASH_DEADENDED, |m| m.deadended_count),
        (STASH_UNCONSTRAINED, |m| m.unconstrained_count),
    ];
    for (stash, counter) in cases {
        let mut mgr = StashManager::new();
        assert_eq!(counter(&mgr), 0);
        mgr.push_or_drop_terminal(stash, RustSimState::new("amd64").unwrap());
        assert_eq!(
            counter(&mgr),
            1,
            "counter for terminal stash {stash:?} did not increment"
        );
        assert_eq!(mgr.count(stash), 1, "state not stored in {stash:?}");
    }
}

/// angr-9ke6b.231: `push_errored` counts and indexes the state, and — unlike
/// `push_or_drop_terminal` — keeps it even under `drop_terminal_states`.
///
/// The serial run loop used to open-code the errored push, which skipped both
/// the counter and the index while the parallel loop folded
/// `stats.summarized_errored` into the same field.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_push_errored_counts_indexes_and_never_drops() {
    for drop_terminal in [false, true] {
        let mut mgr = StashManager::new();
        mgr.set_drop_terminal_states(drop_terminal);
        let state = RustSimState::new("amd64").unwrap();
        let id = state.state_id();

        mgr.push_errored(state);

        assert_eq!(mgr.errored_count, 1, "drop_terminal_states={drop_terminal}");
        assert_eq!(
            mgr.count(STASH_ERRORED),
            1,
            "errored states must survive drop_terminal_states={drop_terminal}"
        );
        assert_eq!(mgr.stash_of(id), Some(STASH_ERRORED));
    }
}

/// angr-sqfj8.136: routing an errored state through `push_or_drop_terminal`
/// is a programming error, not a silently-counted alternative — under
/// `drop_terminal_states` it would discard the state, breaking invariant I6.
#[cfg(feature = "vex-engine-z3")]
#[test]
#[should_panic(expected = "push_errored")]
fn test_push_or_drop_terminal_rejects_errored() {
    let mut mgr = StashManager::new();
    mgr.push_or_drop_terminal(STASH_ERRORED, RustSimState::new("amd64").unwrap());
}

/// Round-trip a non-empty StashManager via the per-state codec.
///
/// Pushes two states into `active` and one into `found`, then dumps via
/// `dump_snapshot` → `load_snapshot` and asserts (a) stash counts
/// match, (b) state_ids survive in their respective stashes, (c) the
/// `state_index` is rebuilt so `stash_of` answers consistently.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_stash_manager_dump_load_round_trip() {
    let mut mgr = StashManager::new();
    let s1 = RustSimState::new("amd64").unwrap();
    let s2 = RustSimState::new("amd64").unwrap();
    let s3 = RustSimState::new("amd64").unwrap();
    let id1 = s1.state_id();
    let id2 = s2.state_id();
    let id3 = s3.state_id();

    mgr.push(STASH_ACTIVE, s1);
    mgr.push(STASH_ACTIVE, s2);
    mgr.push(STASH_FOUND, s3);
    mgr.set_root(id2, id1);
    mgr.avoided_count = 4;
    mgr.deadended_count = 7;

    let bytes = mgr.dump_snapshot();
    assert_eq!(
        bytes[0], STASH_SNAPSHOT_VERSION,
        "stash envelope must carry version byte"
    );
    let restored = StashManager::load_snapshot(&bytes).expect("load_snapshot");

    assert_eq!(restored.count(STASH_ACTIVE), 2);
    assert_eq!(restored.count(STASH_FOUND), 1);
    assert_eq!(restored.avoided_count, 4);
    assert_eq!(restored.deadended_count, 7);

    // state_index rebuilt: stash_of() answers for every id.
    assert_eq!(restored.stash_of(id1), Some(STASH_ACTIVE));
    assert_eq!(restored.stash_of(id2), Some(STASH_ACTIVE));
    assert_eq!(restored.stash_of(id3), Some(STASH_FOUND));

    // state_roots survived.
    assert_eq!(restored.get_root(id2), Some(id1));
}

/// Bad envelope handling: empty bytes and mismatched version byte both
/// route through the typed `SnapshotError` variants.
#[test]
fn test_stash_manager_load_snapshot_errors() {
    match StashManager::load_snapshot(&[]) {
        Err(crate::state::SnapshotError::EmptyEnvelope) => {}
        Err(other) => panic!("expected EmptyEnvelope, got {other:?}"),
        Ok(_) => panic!("empty envelope must fail"),
    }

    let mut mgr = StashManager::new();
    let mut bytes = mgr.dump_snapshot();
    bytes[0] = STASH_SNAPSHOT_VERSION.wrapping_add(7);
    match StashManager::load_snapshot(&bytes) {
        Err(crate::state::SnapshotError::VersionMismatch { found, expected }) => {
            assert_eq!(expected, STASH_SNAPSHOT_VERSION);
            assert_eq!(found, STASH_SNAPSHOT_VERSION.wrapping_add(7));
        }
        Err(other) => panic!("expected VersionMismatch, got {other:?}"),
        Ok(_) => panic!("bad version must fail"),
    }
}

/// angr-euw28: an envelope written by a FOREIGN process carries symbol ids
/// minted by an allocator that also started at 0, so they alias ids this
/// process already handed out. `load_snapshot` must rebase the whole envelope's
/// id space above the local watermark — keeping names (and therefore Z3
/// identity) intact — while an envelope written by THIS process keeps its ids
/// verbatim, since those ids are still ours.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_foreign_envelope_rebases_symbol_ids() {
    use crate::symbolic::{RustBV, symbol_id_watermark};

    /// First `Symbolic` leaf under `bv`, as (id, name).
    fn first_leaf(bv: &RustBV) -> Option<(u64, String)> {
        match bv {
            RustBV::Symbolic { id, name, .. } => Some((*id, name.to_string())),
            RustBV::Expression { operands, .. } => operands.iter().find_map(first_leaf),
            _ => None,
        }
    }

    fn restored_leaf(bytes: &[u8]) -> (u64, String) {
        let restored = StashManager::load_snapshot(bytes).expect("load_snapshot");
        let state = restored
            .get(STASH_ACTIVE)
            .and_then(std::collections::VecDeque::front)
            .expect("restored active state");
        let solver = state.solver();
        let ctx = solver.borrow();
        let assumed = ctx.get_assumed_constraints();
        let (bv, _) = assumed.first().expect("restored assumed constraint");
        first_leaf(bv).expect("restored symbolic leaf")
    }

    let mut mgr = StashManager::new();
    let state = RustSimState::new("amd64").unwrap();
    let source_id = {
        let solver = state.solver();
        let ctx = solver.borrow();
        let leaf = RustBV::symbolic(&ctx, "euw28_leaf", 64);
        let eq = leaf.eq(&RustBV::concrete(7, 64), &ctx);
        ctx.assume_true(&eq);
        first_leaf(&leaf).expect("minted leaf").0
    };
    mgr.push(STASH_ACTIVE, state);
    let bytes = mgr.dump_snapshot();

    // (a) Same process: the ids in the envelope are still ours — keep them.
    let (same_id, same_name) = restored_leaf(&bytes);
    assert_eq!(same_id, source_id, "in-process load must not rebase ids");
    assert_eq!(same_name, "euw28_leaf");

    // (b) Foreign process: flip the origin token in the header. The leaf must
    // land above the watermark (so it cannot alias a locally-minted symbol),
    // under its original name.
    let mut foreign = bytes.clone();
    foreign[1] ^= 0xFF;
    let watermark = symbol_id_watermark();
    let (foreign_id, foreign_name) = restored_leaf(&foreign);
    assert_eq!(
        foreign_name, "euw28_leaf",
        "rebasing must not touch the symbol NAME — Z3 identity is by name"
    );
    assert!(
        foreign_id >= watermark,
        "foreign leaf id {foreign_id} aliases a locally-minted id (watermark {watermark})"
    );
    assert!(
        symbol_id_watermark() > foreign_id,
        "the allocator must be reserved past every rebased leaf"
    );
}

/// `take_state_from` single-sources the find-index-then-remove dance
/// (angr-ph300.27): it resolves via the given stash, unindexes on success,
/// leaves roots untouched, and refuses to relocate a state living elsewhere.
#[test]
fn test_take_state_from_stash_scoped() {
    let mut mgr = StashManager::new();
    let s1 = RustSimState::new("amd64").unwrap();
    let s2 = RustSimState::new("amd64").unwrap();
    let id1 = s1.state_id();
    let id2 = s2.state_id();
    mgr.push(STASH_ACTIVE, s1);
    mgr.push(STASH_FOUND, s2);
    mgr.set_root(id1, id1);

    // Wrong stash: state is NOT relocated, index/root untouched.
    assert!(mgr.take_state_from(id1, STASH_FOUND).is_none());
    assert_eq!(mgr.stash_of(id1), Some(STASH_ACTIVE));
    assert_eq!(mgr.count(STASH_ACTIVE), 1);

    // Correct stash: removed + unindexed, but the root survives (caller's job).
    let taken = mgr.take_state_from(id1, STASH_ACTIVE).expect("state taken");
    assert_eq!(taken.state_id(), id1);
    assert_eq!(mgr.count(STASH_ACTIVE), 0);
    assert_eq!(mgr.stash_of(id1), None);
    assert_eq!(
        mgr.get_root(id1),
        Some(id1),
        "take_state_from must not drop roots"
    );

    // Unknown id in a valid stash yields None without disturbing the stash.
    assert!(mgr.take_state_from(9_999, STASH_FOUND).is_none());
    assert_eq!(mgr.count(STASH_FOUND), 1);
    assert_eq!(mgr.stash_of(id2), Some(STASH_FOUND));
}

/// angr-c7xno.97: a **present-but-stale** `state_index` entry (index says
/// stash A, the state actually lives in stash B) must self-heal for all three
/// lookups, not just `find_state`. Before the fix `find_state_mut` /
/// `take_state` trusted a present entry and returned `None` for a live state,
/// while `find_state` fell back to a full scan — an observable read-vs-write
/// divergence for any state a raw stash move re-homed without `index()`.
#[test]
fn stale_index_entry_self_heals_for_all_three_lookups() {
    let mut mgr = StashManager::new();
    let state = RustSimState::new("amd64").unwrap();
    let sid = state.state_id();
    mgr.push(STASH_FOUND, state);
    // Simulate a raw move that forgot to re-index: the state is in
    // STASH_FOUND, the index still claims STASH_ACTIVE.
    mgr.index(sid, STASH_ACTIVE);
    assert_eq!(mgr.stash_of(sid), Some(STASH_ACTIVE), "index is stale");

    assert!(mgr.find_state(sid).is_some(), "find_state already self-healed");
    assert!(
        mgr.find_state_mut(sid).is_some(),
        "find_state_mut must fall back to the scan its doc comment promises"
    );
    let taken = mgr.take_state(sid).expect("take_state must find the state");
    assert_eq!(taken.state_id(), sid);
    assert_eq!(mgr.count(STASH_FOUND), 0, "taken from its real stash");
    assert_eq!(mgr.stash_of(sid), None, "stale entry cleared on take");
}

/// The stale-index fallback must not resurrect a *deleted* state: an index
/// entry with no backing state anywhere still resolves to `None`.
#[test]
fn stale_index_entry_for_absent_state_still_yields_none() {
    let mut mgr = StashManager::new();
    mgr.index(4_242, STASH_ACTIVE);

    assert!(mgr.find_state(4_242).is_none());
    assert!(mgr.find_state_mut(4_242).is_none());
    assert!(mgr.take_state(4_242).is_none());
}

/// angr-03vl4.80: `insert` is an I6 chokepoint — replacing a stash's contents
/// must re-index the incoming states and drop the index/root entries of the
/// states it evicts, so `state_index` / `state_roots` never outlive the
/// stashes they describe.
#[test]
fn test_insert_maintains_index_and_roots() {
    let mut mgr = StashManager::new();
    let evicted = RustSimState::new("amd64").unwrap();
    let evicted_id = evicted.state_id();
    mgr.push(STASH_ACTIVE, evicted);
    mgr.set_root(evicted_id, evicted_id);
    assert_eq!(mgr.stash_of(evicted_id), Some(STASH_ACTIVE));

    let incoming = RustSimState::new("amd64").unwrap();
    let incoming_id = incoming.state_id();
    let mut deque = VecDeque::new();
    deque.push_back(incoming);
    mgr.insert(STASH_ACTIVE, deque);

    // Evicted state is gone from both maps, not just from the stash.
    assert_eq!(mgr.stash_of(evicted_id), None, "evicted index entry leaked");
    assert_eq!(mgr.get_root(evicted_id), None, "evicted root entry leaked");
    // Incoming state is indexed at the stash it was inserted into.
    assert_eq!(mgr.count(STASH_ACTIVE), 1);
    assert_eq!(mgr.stash_of(incoming_id), Some(STASH_ACTIVE));
    assert!(mgr.find_state(incoming_id).is_some());
}

/// The live `_move_states` shape (`remove` then `insert` an empty deque into
/// the same name) must stay a no-op for the index: the states were already
/// re-indexed to the destination stash by hand.
#[test]
fn test_insert_empty_after_remove_preserves_moved_index() {
    let mut mgr = StashManager::new();
    let state = RustSimState::new("amd64").unwrap();
    let sid = state.state_id();
    mgr.push(STASH_ACTIVE, state);

    let mut from = mgr.remove(STASH_ACTIVE).expect("stash exists");
    mgr.index(sid, STASH_FOUND);
    mgr.ensure_stash(STASH_FOUND).append(&mut from);
    mgr.insert(STASH_ACTIVE, VecDeque::new());

    assert_eq!(mgr.stash_of(sid), Some(STASH_FOUND));
    assert_eq!(mgr.count(STASH_FOUND), 1);
    assert_eq!(mgr.count(STASH_ACTIVE), 0);
}
