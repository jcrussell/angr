//! Unit tests for the parallel-driver routing half of `run_loop.rs`
//! (angr-ph300.3.1). These drive `route_materialized_terminal` — the single
//! choke point where a worker-materialized terminal (or an untagged residual
//! frontier state) lands in a coordinator stash — and assert per-`MatKind` arm
//! parity with the single-threaded loop, without a live parallel session.
//!
//! `route_materialized_terminal` is a plain `&mut self` method reachable from a
//! freshly constructed manager, so the fixture is just a manager + a
//! `RustSimState` with a set pc; no worker pool, scheduler, or Python callback
//! is required. The full byte-identical proof stays the Python
//! `tests/engines/rust/` parallel suite; this pins the routing table so a
//! future refactor of the arms fails fast in `cargo test`.

use super::*;

use crate::exploration::core_outcome::BounceKind;
use crate::stash::{STASH_ACTIVE, STASH_AVOID, STASH_FOUND, STASH_UNCONSTRAINED};
use crate::state::RustSimState;

/// A fresh manager plus a state parked at `pc`. The state is registered in no
/// stash yet — exactly the precondition `route_materialized_terminal` assumes
/// (the worker owns it; the coordinator is about to place it).
fn mgr_and_state(pc: u64) -> (RustExplorationManager, RustSimState, u64) {
    let mgr = RustExplorationManager::new("amd64", None).unwrap();
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_pc(pc);
    let id = state.state_id();
    (mgr, state, id)
}

#[test]
fn found_kind_routes_to_found_stash_and_sets_root() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, id) = mgr_and_state(0x40_1000);
        let root = 0xdead_beef;
        let mut bounce_queue = Vec::new();

        mgr.route_materialized_terminal(state, Some(MatKind::Found), root, &mut bounce_queue);

        assert_eq!(mgr.stash_count(STASH_FOUND), 1, "found terminal collected");
        assert_eq!(mgr.found_count(), 1);
        assert!(bounce_queue.is_empty(), "found never queues a bounce");
        assert_eq!(mgr.sm.get_root(id), Some(root), "lineage root replayed");
    });
}

#[test]
fn unconstrained_kind_routes_to_unconstrained_stash() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, id) = mgr_and_state(0x40_2000);
        let root = 0x1234;
        let mut bounce_queue = Vec::new();

        mgr.route_materialized_terminal(
            state,
            Some(MatKind::Unconstrained),
            root,
            &mut bounce_queue,
        );

        assert_eq!(mgr.stash_count(STASH_UNCONSTRAINED), 1);
        assert_eq!(mgr.stash_count(STASH_FOUND), 0);
        assert!(bounce_queue.is_empty());
        assert_eq!(mgr.sm.get_root(id), Some(root));
    });
}

#[test]
fn bounce_at_find_addr_routes_to_found_not_queue() {
    Python::initialize();
    Python::attach(|_py| {
        let find = 0x40_3000;
        let (mut mgr, state, id) = mgr_and_state(0x40_0000);
        mgr.set_find_addrs(vec![find]);
        let mut bounce_queue = Vec::new();

        // A Hook bounce whose target is a find ADDRESS short-circuits to FOUND
        // (Bug C1) rather than replaying a spurious Python bounce.
        let kind = MatKind::Bounce(BounceKind::Hook { addr: find });
        mgr.route_materialized_terminal(state, Some(kind), 7, &mut bounce_queue);

        assert!(bounce_queue.is_empty(), "find-addr bounce is not queued");
        assert_eq!(mgr.stash_count(STASH_FOUND), 1);
        // The routed state is re-pointed at the find target (mirrors dispatch_bounce).
        let found_ids = mgr.get_state_ids(STASH_FOUND);
        assert_eq!(found_ids, vec![id]);
    });
}

#[test]
fn bounce_at_avoid_addr_routes_to_avoid_not_queue() {
    Python::initialize();
    Python::attach(|_py| {
        let avoid = 0x40_4000;
        let (mut mgr, state, _id) = mgr_and_state(0x40_0000);
        mgr.set_avoid_addrs(vec![avoid]);
        let mut bounce_queue = Vec::new();

        let kind = MatKind::Bounce(BounceKind::Hook { addr: avoid });
        mgr.route_materialized_terminal(state, Some(kind), 7, &mut bounce_queue);

        assert!(bounce_queue.is_empty());
        assert_eq!(mgr.stash_count(STASH_AVOID), 1);
        assert_eq!(mgr.stash_count(STASH_FOUND), 0);
    });
}

#[test]
fn bounce_at_neutral_addr_is_queued_for_dispatch() {
    Python::initialize();
    Python::attach(|_py| {
        // No find/avoid registered, so a Hook bounce falls through to the
        // real Python-bounce queue rather than any terminal stash.
        let (mut mgr, state, id) = mgr_and_state(0x40_0000);
        let root = 0x99;
        let mut bounce_queue = Vec::new();

        let kind = MatKind::Bounce(BounceKind::Hook { addr: 0x40_5000 });
        mgr.route_materialized_terminal(state, Some(kind), root, &mut bounce_queue);

        assert_eq!(bounce_queue.len(), 1, "neutral bounce parked for dispatch");
        assert_eq!(bounce_queue[0].0.state_id(), id);
        assert_eq!(bounce_queue[0].2, root, "root carried alongside the bounce");
        assert_eq!(mgr.stash_count(STASH_FOUND), 0);
        assert_eq!(mgr.stash_count(STASH_AVOID), 0);
    });
}

#[test]
fn untagged_residual_at_find_pc_routes_to_found() {
    Python::initialize();
    Python::attach(|_py| {
        let find = 0x40_6000;
        // An untagged (`None`) residual sitting AT a find pc is a still-live
        // frontier state the worker never ran through run_post_step_core; the
        // coordinator honors the find gate exactly as route_successor would.
        let (mut mgr, state, id) = mgr_and_state(find);
        mgr.set_find_addrs(vec![find]);
        let mut bounce_queue = Vec::new();

        mgr.route_materialized_terminal(state, None, 5, &mut bounce_queue);

        assert!(bounce_queue.is_empty());
        assert_eq!(mgr.stash_count(STASH_FOUND), 1);
        assert_eq!(mgr.sm.get_root(id), Some(5));
    });
}

#[test]
fn untagged_residual_at_neutral_pc_routes_to_active() {
    Python::initialize();
    Python::attach(|_py| {
        // A `None` residual NOT at a find/avoid pc is a bare active successor —
        // the frontier state the single-threaded loop would leave in ACTIVE.
        let (mut mgr, state, _id) = mgr_and_state(0x40_7000);
        mgr.set_find_addrs(vec![0x40_6000]);
        let mut bounce_queue = Vec::new();

        mgr.route_materialized_terminal(state, None, 5, &mut bounce_queue);

        assert!(bounce_queue.is_empty());
        assert_eq!(mgr.stash_count(STASH_ACTIVE), 1);
        assert_eq!(mgr.stash_count(STASH_FOUND), 0);
    });
}
