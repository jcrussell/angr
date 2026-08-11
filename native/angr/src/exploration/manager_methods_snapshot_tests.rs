// Tests for exploration/manager_methods_snapshot.rs (see
// rust-mod-tests-sibling-extraction for why these live in a sibling file
// rather than an inline `mod tests`).
//
// `StashManager::dump_snapshot`/`load_snapshot` are covered at the stash level
// by `stash_tests.rs`; what is exercised here is the *manager*-level contract
// layered on top of them (angr-c7xno.22): dump flushes parked bounces first,
// load clears the pre-restore pending world, and a rejected envelope leaves
// the manager entirely untouched.
use super::*;
use crate::exploration::{CallbackReason, PendingCallback};
use crate::exploration::core_outcome::BounceKind;
use crate::stash::{STASH_ACTIVE, STASH_FOUND, STASH_SNAPSHOT_VERSION};
use crate::state::RustSimState;
use rustc_hash::FxHashMap;

/// Minimal Error-reason pending callback around an already-built state — the
/// same shape `pending_api_tests.rs` uses, and enough to observe whether
/// `load_snapshot_bytes` cleared the pending map.
fn pending_for(state: RustSimState) -> PendingCallback {
    PendingCallback {
        state,
        pre_callback_snapshot: None,
        reason: CallbackReason::Error {
            message: "pending".to_string(),
        },
        jumpkind: None,
        solver_ctx: None,
        deferred_forks: Vec::new(),
        stored_conditions: FxHashMap::default(),
        fork_snapshots: FxHashMap::default(),
    }
}

/// Full manager-level round trip: dump, scribble over the live manager, then
/// restore. Stash populations, lineage and terminal counters must come back,
/// and the manager-level configuration the envelope does NOT carry (find
/// addresses) must survive the wholesale `self.sm` swap rather than being
/// reset with it.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn snapshot_round_trip_restores_stashes_and_keeps_manager_config() {
    Python::initialize();
    Python::attach(|py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        mgr.set_find_addrs(vec![0x40_1000]);

        let s1 = RustSimState::new("amd64").expect("state");
        let s2 = RustSimState::new("amd64").expect("state");
        let s3 = RustSimState::new("amd64").expect("state");
        let (id1, id2, id3) = (s1.state_id(), s2.state_id(), s3.state_id());
        mgr.sm.push(STASH_ACTIVE, s1);
        mgr.sm.push(STASH_ACTIVE, s2);
        mgr.sm.push(STASH_FOUND, s3);
        mgr.sm.set_root(id2, id1);
        mgr.sm.deadended_count = 7;

        let bytes = mgr.dump_snapshot_bytes(py).as_bytes().to_vec();
        assert_eq!(
            bytes[0], STASH_SNAPSHOT_VERSION,
            "manager envelope must carry the stash version byte"
        );

        // Diverge: a third exploration's worth of state the restore must erase.
        mgr.sm
            .push(STASH_ACTIVE, RustSimState::new("amd64").expect("state"));
        mgr.sm.deadended_count = 999;

        mgr.load_snapshot_bytes(&bytes).expect("load");

        assert_eq!(mgr.sm.count(STASH_ACTIVE), 2);
        assert_eq!(mgr.sm.count(STASH_FOUND), 1);
        assert_eq!(mgr.sm.deadended_count, 7);
        assert_eq!(mgr.sm.stash_of(id3), Some(STASH_FOUND));
        assert_eq!(mgr.sm.get_root(id2), Some(id1), "lineage survives the swap");
        assert_eq!(
            mgr.find_addrs.len(),
            1,
            "manager config is not part of the envelope and must be preserved"
        );
    });
}

/// The angr-ph300.21 half of the contract: everything belonging to the
/// PRE-restore world is dropped. A surviving pending callback could resume a
/// state from the discarded exploration, and a surviving parked bounce would
/// be flushed into the freshly restored stashes by the next `run()`.
#[test]
fn load_snapshot_clears_the_pre_restore_pending_world() {
    Python::initialize();
    Python::attach(|py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let empty = mgr.dump_snapshot_bytes(py).as_bytes().to_vec();

        let pending = RustSimState::new("amd64").expect("state");
        let pending_id = pending.state_id();
        mgr.pending_callbacks
            .insert(StateId::new(pending_id), pending_for(pending));
        let bounced = RustSimState::new("amd64").expect("state");
        let bounced_id = bounced.state_id();
        mgr.pending_parallel_bounces.push((
            bounced,
            BounceKind::Hook { addr: 0x40_3000 },
            bounced_id,
        ));
        mgr.current_stepping_state_id = Some(StateId::new(pending_id));

        mgr.load_snapshot_bytes(&empty).expect("load");

        assert!(mgr.pending_callbacks.is_empty(), "pending callbacks cleared");
        assert!(
            mgr.pending_parallel_bounces.is_empty(),
            "parked bounces cleared"
        );
        assert_eq!(mgr.current_stepping_state_id, None);
    });
}

/// `load_snapshot_bytes` is deliberately NOT `#[steady_guarded]`: the guard
/// and the pending-state clear must run AFTER the fallible parse, so a
/// malformed envelope errors out without tearing down a manager that turns out
/// not to be replaced.
#[test]
fn rejected_envelope_leaves_the_manager_untouched() {
    Python::initialize();
    Python::attach(|py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let mut bad = mgr.dump_snapshot_bytes(py).as_bytes().to_vec();
        bad[0] = STASH_SNAPSHOT_VERSION.wrapping_add(7);

        let pending = RustSimState::new("amd64").expect("state");
        let pending_id = pending.state_id();
        mgr.pending_callbacks
            .insert(StateId::new(pending_id), pending_for(pending));
        mgr.current_stepping_state_id = Some(StateId::new(pending_id));
        mgr.sm
            .push(STASH_ACTIVE, RustSimState::new("amd64").expect("state"));

        assert!(
            mgr.load_snapshot_bytes(&bad).is_err(),
            "stale version byte must be rejected"
        );
        assert!(mgr.load_snapshot_bytes(&[]).is_err(), "empty envelope");

        assert_eq!(mgr.pending_callbacks.len(), 1, "pending world preserved");
        assert_eq!(mgr.current_stepping_state_id, Some(StateId::new(pending_id)));
        assert_eq!(mgr.sm.count(STASH_ACTIVE), 1, "stashes not swapped out");
    });
}

/// angr-op0dn.13.6's finalize-then-capture contract, serial half: a state
/// parked as a pending parallel bounce lives in NO stash, so dumping without
/// flushing first would silently truncate the frontier. `dump_snapshot_bytes`
/// flushes into ACTIVE, and the flushed state must therefore appear in the
/// envelope.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn dump_flushes_parked_bounces_into_the_envelope() {
    Python::initialize();
    Python::attach(|py| {
        let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
        let bounced = RustSimState::new("amd64").expect("state");
        let bounced_id = bounced.state_id();
        mgr.pending_parallel_bounces.push((
            bounced,
            BounceKind::Hook { addr: 0x40_3000 },
            bounced_id,
        ));
        assert_eq!(mgr.sm.count(STASH_ACTIVE), 0, "parked, so in no stash");

        let bytes = mgr.dump_snapshot_bytes(py).as_bytes().to_vec();

        assert_eq!(mgr.sm.count(STASH_ACTIVE), 1, "flushed into ACTIVE by dump");
        let restored = StashManager::load_snapshot(&bytes).expect("load");
        assert_eq!(
            restored.stash_of(bounced_id),
            Some(STASH_ACTIVE),
            "the parked state must not be missing from the snapshot"
        );
    });
}
