// Tests for exploration/pending_api.rs.
use super::*;
use crate::exploration::{CallbackReason, PendingCallback};
use crate::stash::STASH_ACTIVE;
use crate::state::RustSimState;
use rustc_hash::FxHashMap;

/// Wrap an already-built state in a minimal Error-reason PendingCallback
/// (no solver fork / deferred forks), so lineage walks can be exercised
/// purely at the Rust unit-test level.
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

/// A >=3-level fork chain A->B->C->D must report its INTERMEDIATE ancestors,
/// not just [D, C, A] (angr-ph300.25). Python's cache walks in
/// rust_callback_dispatch.py key on exactly these ids; dropping B made a
/// callback on D fall back to root A's stale snapshot.
///
/// Also pins the two halves of the walk: an ancestor held in a stash (C, A)
/// and one held as another pending callback (B) are both followed.
#[test]
fn pending_ancestry_includes_intermediate_ancestors() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let a = RustSimState::new("amd64").expect("state");
    let b = a.fork();
    let c = b.fork();
    let d = c.fork();
    let (a_id, b_id, c_id, d_id) = (a.state_id(), b.state_id(), c.state_id(), d.state_id());

    mgr.sm.push(STASH_ACTIVE, a);
    mgr.pending_callbacks
        .insert(StateId::new(b_id), pending_for(b));
    mgr.sm.push(STASH_ACTIVE, c);
    mgr.pending_callbacks
        .insert(StateId::new(d_id), pending_for(d));
    mgr.sm.set_root(d_id, a_id);

    let ancestry = mgr._get_pending_ancestry(d_id).expect("ancestry");
    assert_eq!(
        ancestry,
        vec![d_id, c_id, b_id, a_id],
        "full chain expected, with no duplicate root append"
    );
}

/// The walk is best-effort: each parent id it learns is recorded, but it stops
/// once the ancestor it would have to read next is no longer held (forks
/// consume their parent). The lineage root is still appended, so the result is
/// never shorter than the old [state, parent, root] behaviour.
#[test]
fn pending_ancestry_stops_at_dropped_ancestor_but_keeps_root() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let a = RustSimState::new("amd64").expect("state");
    let b = a.fork();
    let c = b.fork();
    let d = c.fork();
    let (a_id, c_id, d_id) = (a.state_id(), c.state_id(), d.state_id());

    // C is gone, so C's id is still recorded (D names it) but B is unreachable.
    mgr.sm.push(STASH_ACTIVE, a);
    drop(b);
    drop(c);
    mgr.pending_callbacks
        .insert(StateId::new(d_id), pending_for(d));
    mgr.sm.set_root(d_id, a_id);

    let ancestry = mgr._get_pending_ancestry(d_id).expect("ancestry");
    assert_eq!(ancestry, vec![d_id, c_id, a_id]);
}

/// With no lineage root recorded and no ancestors held, the result degrades to
/// [state, parent] — the pre-existing single-hop shape.
#[test]
fn pending_ancestry_without_root_is_state_plus_parent() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let a = RustSimState::new("amd64").expect("state");
    let b = a.fork();
    let (a_id, b_id) = (a.state_id(), b.state_id());
    drop(a);

    mgr.pending_callbacks
        .insert(StateId::new(b_id), pending_for(b));

    let ancestry = mgr._get_pending_ancestry(b_id).expect("ancestry");
    assert_eq!(ancestry, vec![b_id, a_id]);
}
