//! Unit tests for the steady-state coordinator: the config guard that
//! finalizes a live session before a stale-snapshot mutation, the drain
//! deadline, the per-terminal routing-map pruning, and the seed path's
//! `on_state_removed` notification. Split out of `run_loop_tests.rs`
//! alongside the module split (angr-9ke6b.49).

use super::*;

use crate::exploration::scheduler::{CancelToken, TaskOutcome};
use crate::stash::STASH_ACTIVE;
use crate::state::RustSimState;
use crate::vex::IRSB;
use lru::LruCache;

/// angr-ph300.24: `add_hook`, `add_hooks`, `register_simprocedure`, and
/// `set_deterministic` each open with `steady_config_guard()` so a mid-steady
/// session is finalized before its snapshotted hook set / solver config goes
/// stale. With no live session the guard is a no-op — this pins that inserting
/// it did not break the mutation itself, and that the four stay guard-first
/// (the guard runs on the no-session path without panicking or clobbering the
/// mutation). The end-to-end finalize-a-live-session proof is the Python
/// `tests/engines/rust/test_parallel_wave.py` steady suite.
///
/// angr-sqfj8.28/.29: the native-technique registrars
/// (`manager_methods_techniques.rs`) and native-procedure mutators
/// (`manager_methods_procedures.rs`) were missing the guard entirely — fixed
/// via the `#[angr_macros::steady_guarded]` attribute macro (which injects
/// `self.steady_config_guard();` as the method's first statement) rather than
/// a hand-written call, so the same class of omission can't recur silently.
/// Covered below alongside the pre-existing sites.
#[test]
fn config_mutators_apply_and_are_guard_safe_without_session() {
    Python::initialize();
    Python::attach(|py| {
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

        // angr-1yge9.3: the two setters that were missing the guard now open
        // with it too, matching the four above.
        mgr.set_solver_timeout(1234);
        assert_eq!(
            mgr.constraint_solver.solver_timeout_ms, 1234,
            "set_solver_timeout applied the new timeout"
        );

        mgr.set_os_name("CGC".to_string());
        assert_eq!(
            mgr.get_os_name(),
            "cgc",
            "set_os_name lowercased and stored the name"
        );

        // angr-9ke6b.53: the frontier cap is snapshotted into the steady
        // session by `ensure_steady_session`, so its setter needs the guard
        // for the same reason the six above do.
        assert!(
            mgr.get_max_active_states().is_none(),
            "unbounded by default"
        );
        mgr.set_max_active_states(Some(7));
        assert_eq!(
            mgr.get_max_active_states(),
            Some(7),
            "set_max_active_states applied the new cap"
        );

        // angr-sqfj8.28: native-technique registrars.
        mgr.register_length_limiter(100, false);
        mgr.register_timeout(5.0);
        mgr.register_loop_bound(3, "spinning");
        mgr.register_merge_point(0x40_5000, 10);
        assert_eq!(
            mgr.native_technique_count(),
            4,
            "all 4 technique registrars applied their mutation"
        );

        // angr-sqfj8.29: native-procedure mutators.
        mgr.disable_native_procedures();
        assert!(!mgr.native_procedures_enabled(), "disable_all applied");
        mgr.enable_native_procedures();
        assert!(mgr.native_procedures_enabled(), "enable_all applied");
        mgr.disable_native_procedure("strlen");
        mgr.enable_native_procedure("strlen");
        mgr.set_python_override("strlen");
        mgr.remove_python_override("strlen");

        let locals = pyo3::types::PyDict::new(py);
        py.run(
            std::ffi::CString::new("f = lambda args: 0")
                .unwrap()
                .as_c_str(),
            None,
            Some(&locals),
        )
        .unwrap();
        let callable: Py<PyAny> = locals.get_item("f").unwrap().unwrap().unbind();
        mgr.register_python_procedure("py_proc".to_string(), 1, false, callable);
        assert!(
            mgr.has_native_procedure("py_proc"),
            "register_python_procedure registered the proc"
        );

        // The guard ran on the no-session path for every mutator above and
        // left the session absent (nothing to finalize).
        assert!(
            !mgr.parallel_session_active(),
            "guard stays a no-op when no session is live"
        );
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
        let (session, up_rx) = RunSession::new_with_policy(
            Box::new(terminal_only_process()),
            Arc::clone(&mgr.policy),
            mgr.max_active_states,
        );
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
