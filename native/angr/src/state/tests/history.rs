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
