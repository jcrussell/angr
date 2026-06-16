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
    assert_eq!(mgr.found_count(), 0);
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

    let mgr = StashManager::new();
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
