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

use crate::exploration::core_outcome::{BounceKind, CoreCounters};
use crate::stash::{STASH_ACTIVE, STASH_AVOID, STASH_FOUND, STASH_PRUNED, STASH_UNCONSTRAINED};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

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

/// Pin `rax` to two different values so the state's path constraints are UNSAT.
fn unsat_state(pc: u64) -> RustSimState {
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_pc(pc);
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "unsat_x", 64)
    };
    state.set_register("rax", x.clone());
    for witness in [1u128, 2u128] {
        let c = {
            let s = state.solver().borrow();
            x.eq(&RustBV::concrete(witness, 64), &s)
        };
        state.add_constraint(c);
    }
    assert!(!state.satisfiable(), "fixture must be UNSAT");
    state
}

#[test]
fn unsat_successor_at_find_pc_routes_to_pruned() {
    Python::initialize();
    Python::attach(|_py| {
        // angr-ph300.13: an UNSAT state reaching a find address via an
        // infeasible path must land in STASH_PRUNED, not vanish. The
        // popped-state path (`check_terminal_conditions`) already did this;
        // the successor path silently dropped it, so `pruned_count` differed
        // by arrival path between two semantically identical explorations.
        let find = 0x40_9000;
        let (mut mgr, _sat, _id) = mgr_and_state(find);
        mgr.set_find_addrs(vec![find]);

        mgr.route_successor(unsat_state(find), true);

        assert_eq!(mgr.stash_count(STASH_FOUND), 0, "UNSAT is not a find");
        assert_eq!(mgr.stash_count(STASH_PRUNED), 1, "tracked, not dropped");
        assert_eq!(mgr.stash_count(STASH_ACTIVE), 0);
    });
}

#[test]
fn sat_successor_at_find_pc_still_routes_to_found() {
    Python::initialize();
    Python::attach(|_py| {
        // Guard the other half of the gate: the prune arm must not swallow a
        // satisfiable successor.
        let find = 0x40_a000;
        let (mut mgr, state, _id) = mgr_and_state(find);
        mgr.set_find_addrs(vec![find]);

        mgr.route_successor(state, true);

        assert_eq!(mgr.stash_count(STASH_FOUND), 1);
        assert_eq!(mgr.stash_count(STASH_PRUNED), 0);
    });
}

#[test]
fn untagged_unsat_residual_at_find_pc_routes_to_pruned() {
    Python::initialize();
    Python::attach(|_py| {
        // The coordinator's untagged residual arm funnels an UNSAT-at-find
        // state through `route_successor(state, true)`, so it inherits the
        // prune routing rather than the old silent drop.
        let find = 0x40_b000;
        let (mut mgr, _sat, _id) = mgr_and_state(find);
        mgr.set_find_addrs(vec![find]);
        let state = unsat_state(find);
        let id = state.state_id();
        let mut bounce_queue = Vec::new();

        mgr.route_materialized_terminal(state, None, 7, &mut bounce_queue);

        assert!(bounce_queue.is_empty());
        assert_eq!(mgr.stash_count(STASH_FOUND), 0);
        assert_eq!(mgr.stash_count(STASH_PRUNED), 1);
        assert_eq!(mgr.sm.get_root(id), Some(7), "lineage root still replayed");
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

#[test]
fn coordinator_routed_find_beyond_num_find_caps_to_active() {
    Python::initialize();
    Python::attach(|_py| {
        // The `worker_found_hint` is bumped ONLY by a worker's pre-step find
        // arm; coordinator-routed finds (bounce->find, untagged residual at a
        // find pc) never touch it (angr-ph300.16). Their `num_find` cap is
        // therefore NOT the hint but `push_found_capped`'s
        // `found_count() >= num_find` gate. Prove that gate holds on the
        // coordinator path independently of the hint: with `num_find == 1`
        // already satisfied, a second coordinator-routed find spills to ACTIVE,
        // keeping the found-set worker-invariant.
        let find = 0x40_8000;
        let (mut mgr, first, _first_id) = mgr_and_state(find);
        mgr.set_find_addrs(vec![find]);
        assert_eq!(mgr.num_find, 1, "default num_find");
        let mut bounce_queue = Vec::new();

        // First coordinator find (untagged residual at find pc) fills FOUND.
        mgr.route_materialized_terminal(first, None, 1, &mut bounce_queue);
        assert_eq!(mgr.stash_count(STASH_FOUND), 1);

        // Second coordinator find must NOT over-collect past num_find — the
        // surplus stays an active, re-findable frontier state.
        let mut second = RustSimState::new("amd64").unwrap();
        second.set_pc(find);
        let second_id = second.state_id();
        mgr.route_materialized_terminal(second, None, 2, &mut bounce_queue);

        assert_eq!(mgr.stash_count(STASH_FOUND), 1, "cap holds at num_find");
        assert_eq!(
            mgr.stash_count(STASH_ACTIVE),
            1,
            "surplus coordinator find spills to active, not found"
        );
        assert_eq!(mgr.get_state_ids(STASH_ACTIVE), vec![second_id]);
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
        // step_one's GAP-6 block consumes it), so it must force serial until
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

/// angr-ph300.24: `add_hook`, `add_hooks`, `register_simprocedure`, and
/// `set_deterministic` each open with `steady_config_guard()` so a mid-steady
/// session is finalized before its snapshotted hook set / solver config goes
/// stale. With no live session the guard is a no-op — this pins that inserting
/// it did not break the mutation itself, and that the four stay guard-first
/// (the guard runs on the no-session path without panicking or clobbering the
/// mutation). The end-to-end finalize-a-live-session proof is the Python
/// `tests/engines/rust/test_parallel_wave.py` steady suite.
#[test]
fn config_mutators_apply_and_are_guard_safe_without_session() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        assert!(
            !mgr.parallel_session_active(),
            "fresh manager has no session"
        );

        mgr.add_hook(0x40_1000);
        assert!(mgr.hooks.contains(&0x40_1000), "add_hook inserted the addr");

        mgr.add_hooks(vec![0x40_2000, 0x40_3000]);
        assert!(
            mgr.hooks.contains(&0x40_2000) && mgr.hooks.contains(&0x40_3000),
            "add_hooks inserted both addrs"
        );

        mgr.register_simprocedure(0x40_4000, "strlen".to_string(), 1, false);
        assert!(
            mgr.hooks.contains(&0x40_4000),
            "register_simprocedure hooked the addr"
        );
        assert!(
            mgr.simprocedures.contains_key(&0x40_4000),
            "register_simprocedure recorded the proc"
        );

        assert!(!mgr.is_deterministic(), "default is non-deterministic");
        mgr.set_deterministic(true);
        assert!(mgr.is_deterministic(), "set_deterministic flipped the flag");

        // The guard ran on the no-session path for every mutator above and
        // left the session absent (nothing to finalize).
        assert!(
            !mgr.parallel_session_active(),
            "guard stays a no-op when no session is live"
        );
    });
}

// ---------------------------------------------------------------------------
// angr-ph300.11: a payload that fails to reattach must not take the rest of
// the wave down with it.
// ---------------------------------------------------------------------------

#[test]
fn corrupt_payload_is_dropped_and_later_payloads_still_route() {
    Python::initialize();
    Python::attach(|_py| {
        let (mut mgr, state, id) = mgr_and_state(0x40_9000);
        let mut kind_map = FxHashMap::default();
        kind_map.insert(id, MatKind::Found);
        let mut root_map = FxHashMap::default();
        root_map.insert(id, 0x7777);

        // Corrupt payload FIRST: under the old `?` this aborted the loop and
        // the trailing found was destroyed with no stash record.
        let payloads = vec![
            StateMigrationPayload::corrupt_for_test(),
            state.detach_for_migration(),
        ];

        let mut bounce_queue = Vec::new();
        let dropped =
            mgr.route_materialized_payloads(payloads, &mut kind_map, &root_map, &mut bounce_queue);

        assert_eq!(dropped, 1, "exactly the corrupt payload was dropped");
        assert_eq!(
            mgr.stash_count(STASH_FOUND),
            1,
            "the found behind the corrupt payload still reached its stash"
        );
        assert_eq!(mgr.sm.get_root(id), Some(0x7777), "lineage root replayed");
        assert!(
            kind_map.is_empty(),
            "routed state's kind entry removed; the corrupt one never had a key"
        );
    });
}

#[test]
fn all_corrupt_payloads_drop_without_routing_anything() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        let mut kind_map = FxHashMap::default();
        let root_map = FxHashMap::default();
        let mut bounce_queue = Vec::new();

        let dropped = mgr.route_materialized_payloads(
            vec![
                StateMigrationPayload::corrupt_for_test(),
                StateMigrationPayload::corrupt_for_test(),
            ],
            &mut kind_map,
            &root_map,
            &mut bounce_queue,
        );

        assert_eq!(dropped, 2);
        assert_eq!(mgr.stash_count(STASH_FOUND), 0);
        assert_eq!(mgr.stash_count(STASH_ACTIVE), 0);
        assert!(bounce_queue.is_empty());
    });
}

/// The steady-finalize drain deadline scales with the configured solver
/// timeout but never drops below the historical 60s floor (angr-e4cys).
/// Cancellation is task-boundary-only, so a worker inside one solve cannot ack
/// until that solve returns — a deadline shorter than the solve would blame a
/// healthy worker for a lost wakeup.
#[test]
fn steady_finalize_deadline_scales_with_solver_timeout() {
    use crate::symbolic::DEFAULT_SOLVER_TIMEOUT_MS;

    // At (and below) the default the floor wins, preserving the old 60s.
    assert_eq!(
        steady_finalize_deadline(DEFAULT_SOLVER_TIMEOUT_MS),
        Duration::from_secs(60)
    );
    assert_eq!(steady_finalize_deadline(0), Duration::from_secs(60));
    assert_eq!(steady_finalize_deadline(1_000), Duration::from_secs(60));

    // Above it the deadline is 2x the solver timeout.
    assert_eq!(steady_finalize_deadline(120_000), Duration::from_secs(240));

    // A pathological timeout must not overflow the millis -> Duration math.
    assert!(steady_finalize_deadline(u32::MAX) > Duration::from_secs(60));
}

/// angr-offd5(b): folding a live session's accounting mid-explore must be safe
/// to repeat — the incremental fold at every `need_callback` return and the
/// final fold in `finalize_steady_session` share one `ParallelShared`, so a
/// non-destructive read would double-count every step and every `CoreCounters`
/// entry. Both halves are M2 (swap `stepped` to zero, `mem::take` the counter
/// queue), which this pins directly: a second fold with no new worker activity
/// contributes exactly nothing.
#[test]
fn folding_shared_counters_twice_does_not_double_count() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        let shared = ParallelShared::seeded(0, 1);
        shared.stepped.store(7, Ordering::SeqCst);
        shared.counters.lock().unwrap().push(CoreCounters {
            native_calls: 3,
            syscall_native_count: 2,
            ..Default::default()
        });

        let stepped = mgr.fold_parallel_shared_counters(&shared);
        assert_eq!(stepped, 7, "first fold reports the workers' step total");
        assert_eq!(mgr.profiling.native_proc_stats.native_calls, 3);
        assert_eq!(mgr.syscall_native_count, 2);

        // Second fold, no worker activity in between: pure no-op.
        let stepped = mgr.fold_parallel_shared_counters(&shared);
        assert_eq!(stepped, 0, "stepped was swapped to zero by the first fold");
        assert_eq!(
            mgr.profiling.native_proc_stats.native_calls, 3,
            "counter queue was drained, not copied"
        );
        assert_eq!(mgr.syscall_native_count, 2);
    });
}

/// angr-n0irt.14 / angr-offd5: `route_steady_terminal` prunes each routed
/// terminal's OWN `root_map`/`kind_map` entry from the live session's shared
/// maps as it routes (`.remove`, not the wave path's non-removing
/// `.get().copied()`), so a long-lived session's maps stay bounded by the live
/// frontier rather than the cumulative terminal count. The pruning was
/// unfalsifiable while inlined in a private fn requiring a live session; the
/// `take_steady_routing` helper isolates it. A revert to a non-removing read
/// leaves the entry behind and trips the "pruned" assertions.
#[test]
fn take_steady_routing_removes_the_routed_terminals_entries() {
    let shared = ParallelShared::seeded(0, 1);
    let id = 0x4141;
    shared.root_map.lock().unwrap().insert(id, 0x7777);
    shared.kind_map.lock().unwrap().insert(id, MatKind::Found);
    // A sibling entry (a still-live Continue successor) that must survive.
    let other = 0x4242;
    shared.root_map.lock().unwrap().insert(other, 0x8888);

    let (root, kind) = take_steady_routing(&shared, id);

    assert_eq!(
        root, 0x7777,
        "returned lineage root matches the seeded entry"
    );
    assert!(
        matches!(kind, Some(MatKind::Found)),
        "returned the seeded kind"
    );
    assert!(
        !shared.root_map.lock().unwrap().contains_key(&id),
        "routed terminal's root_map entry pruned"
    );
    assert!(
        !shared.kind_map.lock().unwrap().contains_key(&id),
        "routed terminal's kind_map entry pruned"
    );
    assert_eq!(
        shared.root_map.lock().unwrap().get(&other).copied(),
        Some(0x8888),
        "sibling Continue successor's entry retained"
    );
}

/// A terminal with no seeded `root_map`/`kind_map` entry (the untagged-residual
/// case) falls back to its own id as the lineage root and a `None` kind — the
/// same `unwrap_or(id)` fallback the wave path uses.
#[test]
fn take_steady_routing_falls_back_to_id_when_absent() {
    let shared = ParallelShared::seeded(0, 1);
    let id = 0x5151;
    let (root, kind) = take_steady_routing(&shared, id);
    assert_eq!(root, id, "absent root_map ⇒ lineage root is the id itself");
    assert!(kind.is_none(), "absent kind_map ⇒ None");
}

/// angr-n0irt.13 / angr-e4cys: on the `finalize_steady_session` drain timeout,
/// the error names exactly the workers that never acked `Paused` — every id in
/// `0..workers` absent from `paused`. The computation was inlined in the live
/// finalize path (needs a running session + pool); `stuck_worker_ids` isolates
/// it. A regression that inverts the predicate (naming the acked workers) or
/// off-by-ones the range trips these assertions.
#[test]
fn stuck_worker_ids_names_the_unacked_workers() {
    // Mixed: workers 1 and 3 acked, 0 / 2 / 4 are stuck. Order-preserving.
    assert_eq!(stuck_worker_ids(5, &[1, 3]), vec![0, 2, 4]);
    // A duplicated ack (Paused dedups, but be robust) still excludes only 2.
    assert_eq!(stuck_worker_ids(4, &[2, 2]), vec![0, 1, 3]);
}

/// The two saturating ends of `stuck_worker_ids`: all workers acked ⇒ empty
/// stuck set (the `Ok(())` finalize path never builds the error), and none
/// acked ⇒ every worker id is stuck.
#[test]
fn stuck_worker_ids_saturates_at_both_ends() {
    assert!(
        stuck_worker_ids(3, &[0, 1, 2]).is_empty(),
        "all acked ⇒ nothing stuck"
    );
    assert_eq!(
        stuck_worker_ids(3, &[]),
        vec![0, 1, 2],
        "none acked ⇒ every worker stuck"
    );
    assert!(
        stuck_worker_ids(0, &[]).is_empty(),
        "zero workers ⇒ empty regardless"
    );
    // Stray acks outside the range (never emitted, but must not panic or leak).
    assert_eq!(stuck_worker_ids(2, &[7, 9]), vec![0, 1]);
}

/// angr-offd5(b): the incremental folder is reachable on the no-session path
/// (a `need_callback` can surface before `ensure_steady_session` ever ran, and
/// the serial/wave loops never build one) and must leave the manager untouched
/// there rather than panicking on the absent session.
#[test]
fn incremental_steady_fold_is_a_noop_without_a_session() {
    Python::initialize();
    Python::attach(|_py| {
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.steps = 41;
        assert!(!mgr.parallel_session_active());

        mgr.fold_steady_counters_incrementally();

        assert_eq!(mgr.steps, 41, "no session ⇒ nothing to fold");
        assert!(!mgr.parallel_session_active());
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
        assert_eq!(mgr.active_count(), 0, "parked bounce is in NO stash");

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

// ---------------------------------------------------------------------------
// seed_steady_session_from_active -> policy.on_state_removed wiring (angr-3xk63)
//
// Same drain-without-notify shape as offload_surplus (angr-ua7fd, pinned in
// scheduler_worker_tests.rs): this drains STASH_ACTIVE straight via
// `s.drain(..).collect()` to seed a live steady session, bypassing
// `policy.select` entirely. A memoizing policy (LoopHeadRoundRobin's
// key_cache) needs `on_state_removed` for every drained state or its memo
// leaks. The session is built by hand from the low-level scheduler
// primitives (mirroring scheduler_tests.rs's "PersistentPool + RunSession
// directly" section) with a trivial terminal-only process closure, so this
// pins the notify wiring without needing real interpreter stepping.
// ---------------------------------------------------------------------------

/// A policy that forwards `select`/`on_fork` to `Lifo` but records every
/// `state_id` passed to `on_state_removed` — same shape as the `SpyPolicy` in
/// `scheduler_worker_tests.rs`, redefined here since that one is private to a
/// sibling test module.
#[derive(Default)]
struct SpyPolicy {
    removed: std::sync::Mutex<Vec<u64>>,
}

impl selection_policy::SelectionPolicy for SpyPolicy {
    fn select(
        &self,
        active: &mut std::collections::VecDeque<RustSimState>,
    ) -> Option<RustSimState> {
        selection_policy::Lifo.select(active)
    }

    fn on_fork(&self, active: &mut std::collections::VecDeque<RustSimState>, state: RustSimState) {
        selection_policy::Lifo.on_fork(active, state);
    }

    fn name(&self) -> &'static str {
        "spy"
    }

    fn on_state_removed(&self, state_id: u64) {
        self.removed.lock().expect("spy poisoned").push(state_id);
    }
}

/// A process closure that materializes every state immediately — no forking,
/// no real VEX stepping. Enough to let a `RunSession` quiesce so the test can
/// stay synchronous.
fn terminal_only_process()
-> impl Fn(RustSimState, &CancelToken, &mut LruCache<u64, Arc<IRSB>>) -> TaskOutcome
+ Send
+ Sync
+ 'static {
    |state, _cancel, _cache| TaskOutcome::terminal(vec![state])
}

#[test]
fn seed_steady_session_from_active_notifies_policy_on_state_removed() {
    Python::initialize();
    Python::attach(|_py| {
        const WORKERS: usize = 2;
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        let spy = Arc::new(SpyPolicy::default());
        mgr.policy = spy.clone();

        // Two active states, parked exactly as any active-stash entry would be
        // (no real memory needed — the process closure never steps them).
        let s1 = RustSimState::new("amd64").unwrap();
        let s2 = RustSimState::new("amd64").unwrap();
        let ids: Vec<u64> = vec![s1.state_id(), s2.state_id()];
        mgr.sm.push(STASH_ACTIVE, s1);
        mgr.sm.push(STASH_ACTIVE, s2);

        // Hand-build a live steady session (mirrors `ensure_steady_session`,
        // but with a trivial process so no real interpreter step is needed).
        let pool = PersistentPool::new(WORKERS);
        let (session, up_rx) =
            RunSession::new_with_policy(Box::new(terminal_only_process()), Arc::clone(&mgr.policy));
        pool.start_session(&session);
        mgr.parallel_session = Some(SteadySession {
            session,
            up_rx: Mutex::new(up_rx),
            shared: Arc::new(ParallelShared::seeded(mgr.found_count(), mgr.num_find)),
            prof: Arc::new(ParallelProfiling::default()),
            workers: WORKERS,
            parked: Vec::new(),
        });
        mgr.parallel_pool = Some(pool);

        mgr.seed_steady_session_from_active();

        let mut removed = spy.removed.lock().expect("spy poisoned").clone();
        removed.sort_unstable();
        let mut expected = ids;
        expected.sort_unstable();
        assert_eq!(
            removed, expected,
            "on_state_removed must fire for every state the steady-session \
             seed drained from STASH_ACTIVE, bypassing policy.select"
        );
        assert!(
            mgr.sm.get(STASH_ACTIVE).is_none_or(|s| s.is_empty()),
            "the drained states left STASH_ACTIVE"
        );
    });
}
