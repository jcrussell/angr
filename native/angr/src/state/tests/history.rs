//! History-cap behaviour: `set_detailed_history`'s cap and unlimited mode, and
//! `set_max_history`'s retroactive trim (including the zero-means-no-trim case).
//!
//! Carved out of the former monolithic `state_tests.rs` (angr-c7xno.69).

use super::super::*;

#[test]
fn test_set_detailed_history_honors_cap() {
    // set_detailed_history (called once per step from interpreter results)
    // must drain the oldest entries when the incoming buffer exceeds the
    // configured cap. Otherwise long blocks bypass max_history.
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_max_history(3);
    let entries: Vec<HistoryEntry> = (0..10)
        .map(|i| HistoryEntry {
            addr: 0x1000 + i,
            jumpkind: 0,
            jump_target: 0,
        })
        .collect();
    state.set_detailed_history(entries);
    let kept = state.detailed_history();
    assert_eq!(kept.len(), 3);
    // FIFO eviction: should retain the most-recent 3 entries.
    assert_eq!(kept[0].addr, 0x1007);
    assert_eq!(kept[2].addr, 0x1009);
}

#[test]
fn test_set_max_history_trims_retroactively() {
    // Lowering max_history on a state that already exceeds the new cap
    // must FIFO-evict the oldest entries down to the cap immediately.
    // Without this, add_to_history (which only removes one entry per
    // push when over cap) never converges and the buffer stays bloated.
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_max_history(0); // unlimited
    for i in 0..20u64 {
        state.add_to_history(0x3000 + i);
        state.add_history_entry(0x3000 + i, 0, 0);
    }
    assert_eq!(state.history().len(), 20);
    assert_eq!(state.detailed_history().len(), 20);

    // Retroactively cap to 4 — both buffers shrink to the most-recent 4.
    state.set_max_history(4);
    let kept = state.history();
    assert_eq!(kept.len(), 4);
    assert_eq!(
        kept.iter().copied().collect::<Vec<_>>(),
        vec![0x3010, 0x3011, 0x3012, 0x3013]
    );
    let kept_detailed = state.detailed_history();
    assert_eq!(kept_detailed.len(), 4);
    assert_eq!(kept_detailed[0].addr, 0x3010);
    assert_eq!(kept_detailed[3].addr, 0x3013);
}

#[test]
fn test_set_max_history_zero_no_trim() {
    // Switching to max=0 (unlimited) must NOT trim — existing entries stay.
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_max_history(5);
    for i in 0..5u64 {
        state.add_to_history(0x4000 + i);
    }
    state.set_max_history(0);
    assert_eq!(state.history().len(), 5);
}

#[test]
fn test_set_detailed_history_unlimited() {
    // max_history = 0 means no cap (legacy behavior).
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_max_history(0);
    let entries: Vec<HistoryEntry> = (0..50)
        .map(|i| HistoryEntry {
            addr: 0x2000 + i,
            jumpkind: 0,
            jump_target: 0,
        })
        .collect();
    state.set_detailed_history(entries);
    assert_eq!(state.detailed_history().len(), 50);
}

/// `record_exit` is what an export reads back as Python's `SimProcedure.exit`
/// step (angr-mx00p.132.8): an `Ijk_Exit` entry at the procedure and the
/// status, which fork and the serialized snapshot both carry.
#[test]
fn test_record_exit_survives_fork_and_serialization() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.record_exit(0x10_0000, Some(RustBV::concrete(0xC000_0409, 32)));
    let mut forked = state.fork();
    let restored = RustSimState::from_serialized(&forked.to_serialized()).expect("from_serialized");
    for s in [&state, &forked, &restored] {
        let last = s.detailed_history().back().expect("exit entry");
        assert_eq!(last.addr, 0x10_0000);
        assert_eq!(HistoryEntry::jumpkind_str(last.jumpkind), "Ijk_Exit");
        assert_eq!(s.exit_code(), Some(&RustBV::concrete(0xC000_0409, 32)));
    }
}

/// `defer_exit` is a granular step's no-return procedure (angr-mx00p.136.30):
/// the `Ijk_Exit` entry, no status, and the pending flag the next step
/// deadends on, carried by fork and the serialized snapshot.
#[test]
fn test_defer_exit_survives_fork_and_serialization() {
    let mut state = RustSimState::new("amd64").unwrap();
    assert!(!state.exit_pending());
    state.defer_exit(0x70_0030);
    let mut forked = state.fork();
    let restored = RustSimState::from_serialized(&forked.to_serialized()).expect("from_serialized");
    for s in [&state, &forked, &restored] {
        assert!(s.exit_pending());
        let last = s.detailed_history().back().expect("exit entry");
        assert_eq!(last.addr, 0x70_0030);
        assert_eq!(HistoryEntry::jumpkind_str(last.jumpkind), "Ijk_Exit");
        assert_eq!(s.exit_code(), None);
    }
}

#[test]
fn test_block_count_and_evicted_visits_outlive_the_cap() {
    // `block_count` and `evicted_visits` keep what the `max_history` window
    // drops, through a retroactive trim, a fork and a snapshot round trip
    // (angr-vmi2l.31).
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_max_history(0);
    for i in 0..6u64 {
        state.add_to_history(0x4000 + (i % 2));
    }
    state.set_max_history(2);
    for _ in 0..3 {
        state.add_to_history(0x4000);
    }
    assert_eq!(state.history().len(), 2);
    assert_eq!(state.block_count(), 9);
    // Trim evicted 0x4000 x2 and 0x4001 x2; the pushes evicted 0x4000, 0x4001, 0x4000.
    assert_eq!(state.evicted_visits().get(&0x4000), Some(&4));
    assert_eq!(state.evicted_visits().get(&0x4001), Some(&3));

    let child = state.fork();
    assert_eq!(child.block_count(), 9);
    assert_eq!(child.evicted_visits(), state.evicted_visits());

    let restored = RustSimState::from_snapshot(state.to_snapshot()).unwrap();
    assert_eq!(restored.block_count(), 9);
    assert_eq!(restored.evicted_visits(), state.evicted_visits());

    // Counting does not depend on history tracking, as in Python.
    let mut untracked = RustSimState::new("amd64").unwrap();
    untracked.set_track_history(false);
    untracked.add_to_history(0x4000);
    assert_eq!(untracked.block_count(), 1);
    assert!(untracked.history().is_empty());
}
