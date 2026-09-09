//! Unit tests for the run-loop dispatcher itself: `must_run_serial` (the
//! parallel-vs-serial routing guard) and `flush_parked_bounces_to_active` (the
//! parked-bounce replay every non-parallel exit depends on). Split out of the
//! former monolithic `run_loop_tests.rs` (angr-9ke6b.49).

use super::*;

use crate::exploration::core_outcome::BounceKind;
use crate::exploration::test_support::mgr_and_state;
use crate::stash::STASH_ACTIVE;
use crate::state::RustSimState;

/// The other half of the same contract: `flush_parked_bounces_to_active` is NOT
/// terminal — it hands the state back to STASH_ACTIVE, where the next step
/// re-lifts the hook and `dispatch_bounce` appends the target. So the flush must
/// restore the pc WITHOUT touching history; appending here too would record one
/// visit twice (`add_to_history` never dedups).
#[test]
fn flush_parked_bounce_leaves_history_for_the_replay_to_append() {
    Python::initialize();
    Python::attach(|_py| {
        const BOUNCE_ADDR: u64 = 0x40_5000;
        // No find/avoid registered, so the flush routes to STASH_ACTIVE — the
        // replayed, non-terminal leg.
        let (mut mgr, mut state, id) = mgr_and_state(0x40_0000);
        state.add_to_history(0x40_0000);
        mgr.pending_parallel_bounces
            .push((state, BounceKind::Hook { addr: BOUNCE_ADDR }, id));

        mgr.flush_parked_bounces_to_active();

        let active = mgr.sm.get(STASH_ACTIVE).expect("active stash exists");
        assert_eq!(active[0].pc(), BOUNCE_ADDR, "pc restored for the replay");
        assert_eq!(
            active[0].history().iter().copied().collect::<Vec<_>>(),
            vec![0x40_0000],
            "flush leaves the bounce target for dispatch_bounce to append"
        );
    });
}

/// A parked bounce whose id is ALREADY resident in a stash is a duplicate the
/// flush must not re-insert — and must not re-park either. Re-parking it (the
/// old catch-all arm, which also mislabelled the reason as "no re-enterable
/// entry address") leaked it forever: every later flush re-ran the same check
/// and put it straight back (angr-03vl4.18).
///
/// A snapshot round-trip is the only way to mint a second `RustSimState`
/// carrying an existing id — `fork`/`Clone` always allocate a fresh one
/// (`state-id-never-reused`, asserted in `fork_with`).
#[test]
fn flush_drops_parked_bounce_whose_id_is_already_resident() {
    Python::initialize();
    Python::attach(|_py| {
        const BOUNCE_ADDR: u64 = 0x40_5000;
        let (mut mgr, mut state, id) = mgr_and_state(0x40_0000);
        let resident = RustSimState::from_snapshot(state.to_snapshot()).expect("round-trip");
        assert_eq!(resident.state_id(), id, "snapshot restore preserves the id");
        mgr.sm.push(STASH_ACTIVE, resident);
        mgr.pending_parallel_bounces
            .push((state, BounceKind::Hook { addr: BOUNCE_ADDR }, id));

        mgr.flush_parked_bounces_to_active();

        assert!(
            mgr.pending_parallel_bounces.is_empty(),
            "a resident duplicate is dropped, not re-parked for the next flush"
        );
        let active = mgr.sm.get(STASH_ACTIVE).expect("active stash exists");
        assert_eq!(active.len(), 1, "the flush must not double-insert the id");
        assert_eq!(
            active[0].pc(),
            0x40_0000,
            "the resident copy is untouched — the parked duplicate is discarded"
        );
    });
}

/// The other half of the same split: a kind with no re-enterable entry address
/// and a NON-resident id still parks (logged, never silently dropped), which is
/// what keeps the state live for a later `_pending_*` API call.
#[test]
fn flush_keeps_parked_bounce_with_no_reenterable_address() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, id) = mgr_and_state(0x40_0000);
        mgr.pending_parallel_bounces
            .push((state, BounceKind::SyscallPython { num: Some(60) }, id));

        mgr.flush_parked_bounces_to_active();

        assert_eq!(
            mgr.pending_parallel_bounces.len(),
            1,
            "no re-enterable address and not resident → stays parked"
        );
        assert!(
            mgr.sm.get(STASH_ACTIVE).is_none_or(|s| s.is_empty()),
            "an unreplayable bounce must not reach a stash"
        );
    });
}

// --- must_run_serial: the run_loop parallel-vs-serial routing guard
// (angr-ph300.6). Native techniques are coordinator-side and only run between
// waves, so a wave on a non-terminating frontier would never quiesce and the
// technique/`run(n)` budget would be ignored. The guard forces serial whenever
// techniques are registered, regardless of how the worker count was engaged
// (kwarg gate OR `RUST_PARALLEL_WORKERS` env). These pin that decision without
// spinning a live wave.

#[test]
fn must_run_serial_true_when_single_worker() {
    Python::initialize();
    Python::attach(|_py| {
        let mgr = RustExplorationManager::new("amd64", None).unwrap();
        // Default construction (no env) leaves one worker → always serial.
        assert!(mgr.must_run_serial());
    });
}

#[test]
fn must_run_serial_false_with_workers_and_no_techniques() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_parallel_workers(2);
        assert!(mgr.native_techniques.is_empty());
        assert!(
            !mgr.must_run_serial(),
            "workers>1 with no techniques takes a parallel path"
        );
    });
}

#[test]
fn must_run_serial_true_when_native_technique_registered() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_parallel_workers(2);
        // A LoopBound is coordinator-side; the wave can't apply it mid-flight.
        mgr.register_loop_bound(5, "deadended");
        assert!(
            mgr.must_run_serial(),
            "a registered native technique forces the serial loop even at workers>1"
        );
    });
}

#[test]
fn must_run_serial_true_when_skip_hook_pending() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_parallel_workers(4);
        assert!(
            !mgr.must_run_serial(),
            "workers>1 with no skip pending takes a parallel path"
        );
        // A pending skip-hook entry has no parallel/steady analogue (only
        // step_one's consume_skip_hook call consumes it), so it must force serial until
        // drained (angr-04tw3.1).
        mgr._set_skip_hook_addr(0x400123);
        assert!(
            mgr.must_run_serial(),
            "a pending skip-hook entry forces the serial loop even at workers>1"
        );
        // Once the stack drains, subsequent run()s go parallel again.
        mgr._clear_skip_hook_addr();
        assert!(
            !mgr.must_run_serial(),
            "clearing the skip-hook stack re-enables the parallel path"
        );
    });
}

/// angr-5mnx3.14: `set_parallel_workers(N)` used to be silently inert once a
/// pool existed — both creation sites are guarded on `parallel_pool.is_none()`
/// and `PersistentPool` has no resize path, so the first parallel run pinned
/// the worker count for the manager's life. `retire_parallel_pool_for_resize`
/// drops the mismatched pool so the next run rebuilds it at the new size.
///
/// Uses the real setter (not the helper directly) so the wiring, not just the
/// helper, is pinned.
#[test]
fn set_parallel_workers_retires_a_pool_of_the_wrong_size() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_parallel_workers(2);
        // Stand in for the lazy creation in `run_loop_parallel_wave` /
        // `ensure_steady_session`, which both build at `workers.max(2)`.
        mgr.parallel_pool = Some(crate::exploration::scheduler::PersistentPool::new(2));

        mgr.set_parallel_workers(4);
        assert!(
            mgr.parallel_pool.is_none(),
            "a worker-count change must retire the stale pool so the next \
             parallel run rebuilds it at the new size"
        );
        assert_eq!(mgr.parallel_real_workers, 4);
    });
}

/// Same setter, same count: the pool is already right, so it must survive —
/// otherwise `_engage_parallel_workers` (which calls the setter on every
/// `explore()`) would pay a thread-spawn + Z3-context rebuild per explore.
#[test]
fn set_parallel_workers_keeps_a_pool_of_the_right_size() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_parallel_workers(3);
        mgr.parallel_pool = Some(crate::exploration::scheduler::PersistentPool::new(3));

        mgr.set_parallel_workers(3);
        assert!(
            mgr.parallel_pool.as_ref().is_some_and(|p| p.num_workers() == 3),
            "an unchanged worker count must not churn the pool"
        );
    });
}

/// The downgrade/restore cycle `rust_manager.py::_engage_parallel_workers`
/// drives across eligible vs ineligible explores (angr-ph300.77): a downgrade
/// to serial KEEPS the pool (`must_run_serial` never reaches it, so idle
/// threads are cheaper than a rebuild), and the restore to the same count finds
/// it intact.
#[test]
fn set_parallel_workers_downgrade_to_serial_keeps_the_pool() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_parallel_workers(2);
        mgr.parallel_pool = Some(crate::exploration::scheduler::PersistentPool::new(2));

        mgr.set_parallel_workers(1);
        assert!(mgr.must_run_serial(), "workers=1 routes to the serial loop");
        assert!(
            mgr.parallel_pool.is_some(),
            "a downgrade to serial leaves the pool parked for the restore"
        );

        mgr.set_parallel_workers(2);
        assert!(
            mgr.parallel_pool.as_ref().is_some_and(|p| p.num_workers() == 2),
            "restoring the original count reuses the parked pool"
        );
    });
}

#[test]
fn must_run_serial_true_when_timeout_registered() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_parallel_workers(4);
        mgr.register_timeout(1.0);
        assert!(
            mgr.must_run_serial(),
            "a registered timeout technique forces the serial loop"
        );
    });
}

/// Helper: park a re-enterable `SimProcedurePython` bounce (a state living in
/// NO stash) at `pc`, the shape a wave leaves behind when it surfaces one
/// `need_callback` and defers the rest of its bounce queue.
fn park_reenterable_bounce(mgr: &mut RustExplorationManager, pc: u64) -> u64 {
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_pc(0xdead);
    let id = state.state_id();
    mgr.pending_parallel_bounces.push((
        state,
        BounceKind::SimProcedurePython {
            addr: pc,
            name: "sp".to_string(),
            num_args: 0,
            return_addr: pc + 0x10,
        },
        id,
    ));
    id
}

/// angr-05kiw: only the two parallel loops drain `pending_parallel_bounces`.
/// A `run()` that routes to the single-threaded loop instead — worker count
/// dropped to 1, a native technique registered, or a callable find/avoid
/// predicate set between calls — used to strand the parked queue permanently
/// and silently. The drain now happens at the top of the loop, ahead of the
/// callbacks check, so even the error exit consumes the queue.
#[test]
fn single_threaded_loop_drains_parked_parallel_bounces() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        const BOUNCE_ADDR: u64 = 0x40_2000;
        let id = park_reenterable_bounce(&mut mgr, BOUNCE_ADDR);
        // `active_count` counts the parked bounce by design (angr-03vl4.15);
        // `stash_count` is the stash-only view that shows it is in none yet.
        assert_eq!(
            mgr.stash_count(STASH_ACTIVE),
            0,
            "parked bounce is in NO stash"
        );

        // No callbacks configured, so the loop bails right after the drain.
        assert!(mgr.run_loop_single_threaded(Some(1)).is_err());

        assert!(
            mgr.pending_parallel_bounces.is_empty(),
            "single-threaded route consumed the parked queue"
        );
        assert_eq!(mgr.active_count(), 1, "parked bounce replayed into active");
        let active = mgr.sm.get(STASH_ACTIVE).expect("active stash exists");
        assert_eq!(active[0].state_id(), id);
        assert_eq!(active[0].pc(), BOUNCE_ADDR, "pc restored to bounce entry");
    });
}

/// angr-05kiw: the explore-end hook `_finalize_parallel_session` calls into
/// `finalize_parallel_session`, which must flush parked bounces too — otherwise
/// an explore() that ends with bounces parked under-reports the resumable
/// frontier in `stash_counts()` versus the serial loop. The flush half is not
/// session-gated (there is no live steady session here).
#[test]
fn finalize_parallel_session_flushes_parked_bounces() {
    Python::initialize();
    Python::attach(|py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        const BOUNCE_ADDR: u64 = 0x40_3000;
        let id = park_reenterable_bounce(&mut mgr, BOUNCE_ADDR);
        assert!(!mgr.parallel_session_active());

        mgr.finalize_parallel_session(py).unwrap();

        assert!(mgr.pending_parallel_bounces.is_empty());
        assert_eq!(mgr.active_count(), 1);
        let active = mgr.sm.get(STASH_ACTIVE).expect("active stash exists");
        assert_eq!(active[0].state_id(), id);
        assert_eq!(active[0].pc(), BOUNCE_ADDR);
    });
}
