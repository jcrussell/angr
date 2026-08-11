//! Unit tests for the GIL-free parallel worker body (angr-sqfj8.43).
//!
//! [`parallel_process_state`] is the scheduler-thread analogue of `step_one`,
//! and until now it was only exercised *indirectly* through the coordinator
//! tests in `run_loop_wave_tests.rs` / `run_loop_steady_tests.rs`. These pin its
//! own branches directly: the pre-step find/avoid short-circuit, the
//! cancel-preemption bail-out, and the `NeedsPython` bounce arm — plus the
//! side-channel writes ([`ParallelShared::root_map`] / `kind_map` /
//! `worker_found_hint` / `stepped`) each branch owes the coordinator.
//!
//! **Scope note (same honesty as `step_core_tests.rs`).** The arms reachable
//! only *after* a real interpreter step — `Continue`, `Deadended`, `Errored`,
//! `Unconstrained` — need a live lifted block, and block lifting in a default
//! `cargo test` build goes through the Python `lift_block` callback (the native
//! lifter is behind the non-default `libvex-ffi` feature). A test asserting one
//! of those arms would therefore assert a *different* thing depending on the
//! feature set, so they stay out. Their post-step classification is unit-tested
//! directly on [`run_post_step_core`] in `core_outcome_tests.rs`, and the
//! worker's own routing of them is covered end-to-end by the parallel tests in
//! `tests/engines/rust/`. The bounce arm below IS reachable here because a
//! hooked initial pc short-circuits before any lift.

use super::*;

use crate::exploration::RustExplorationManager;
use crate::exploration::core_outcome::ParallelProfiling;
use crate::exploration::step_core::StepContext;
use crate::procedures::NativeProcedureRegistry;
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use crate::syscalls::NativeSyscallRegistry;
use pyo3::Python;

/// Owns everything [`CoreCtx`] borrows, so a test can call the worker without
/// spelling out five locals. `run` re-borrows disjoint fields (`block_cache`
/// mutably, the rest shared), which is exactly the split the coordinator does.
struct Harness {
    ctx: StepContext,
    prof: ParallelProfiling,
    procs: NativeProcedureRegistry,
    syscalls: NativeSyscallRegistry,
    callbacks: PythonCallbacks,
    block_cache: LruCache<u64, std::sync::Arc<IRSB>>,
}

impl Harness {
    fn new(mgr: &RustExplorationManager) -> Self {
        Self {
            ctx: mgr.step_context(),
            prof: ParallelProfiling::default(),
            procs: NativeProcedureRegistry::new(),
            syscalls: NativeSyscallRegistry::new(),
            callbacks: PythonCallbacks::new(),
            block_cache: LruCache::unbounded(),
        }
    }

    fn run(
        &mut self,
        state: RustSimState,
        cancel: &CancelToken,
        shared: &ParallelShared,
    ) -> TaskOutcome {
        let cc = CoreCtx {
            ctx: &self.ctx,
            prof: &self.prof,
            native_procs: &self.procs,
            native_syscalls: &self.syscalls,
            callbacks: Some(&self.callbacks),
        };
        parallel_process_state(
            state,
            cancel,
            &mut self.block_cache,
            &cc,
            &self.callbacks,
            shared,
        )
    }
}

fn state_at(pc: u64) -> RustSimState {
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_pc(pc);
    state
}

/// An UNSAT state parked at `pc` (same fixture shape as `run_loop_wave_tests`).
fn unsat_state_at(pc: u64) -> RustSimState {
    let mut state = state_at(pc);
    let x = {
        let s = state.solver().borrow();
        RustBV::symbolic(&s, "worker_unsat_x", 64)
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

fn kind_of(shared: &ParallelShared, id: u64) -> Option<MatKind> {
    shared.kind_map.lock().unwrap().get(&id).cloned()
}

fn root_of(shared: &ParallelShared, id: u64) -> Option<u64> {
    shared.root_map.lock().unwrap().get(&id).copied()
}

// --- ParallelShared::seeded -------------------------------------------------

#[test]
fn seeded_carries_the_found_count_into_the_early_cancel_hint() {
    let shared = ParallelShared::seeded(3, 5);

    assert_eq!(shared.worker_found_hint.load(Ordering::SeqCst), 3);
    assert_eq!(shared.num_find, 5);
    assert_eq!(shared.stepped.load(Ordering::SeqCst), 0);
    assert!(shared.root_map.lock().unwrap().is_empty());
    assert!(shared.kind_map.lock().unwrap().is_empty());
    assert!(shared.counters.lock().unwrap().is_empty());
}

// --- Pre-step avoid ---------------------------------------------------------

#[test]
fn avoid_addr_short_circuits_to_an_avoided_summary() {
    Python::initialize();
    Python::attach(|_py| {
        let avoid = 0x40_1000;
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_avoid_addrs(vec![avoid]);
        let mut h = Harness::new(&mgr);
        let shared = ParallelShared::seeded(0, 1);
        let state = state_at(avoid);
        let id = state.state_id();

        let out = h.run(state, &CancelToken::new(), &shared);

        assert!(out.continue_states.is_empty());
        assert!(out.terminal_states.is_empty(), "avoided pays no serde");
        assert_eq!(
            out.terminal_summaries,
            vec![TerminalSummary {
                state_id: id,
                pc: avoid,
                disposition: SchedDisposition::Avoided,
            }]
        );
        assert!(!out.request_cancel);
        assert_eq!(
            shared.stepped.load(Ordering::SeqCst),
            0,
            "pre-step route never counts toward self.steps"
        );
        assert!(kind_of(&shared, id).is_none(), "summaries carry no MatKind");
    });
}

/// Find-first, mirroring `step_one` and the Python `Explorer` default
/// (`avoid_priority=False`). The worker must agree with the serial loop here or
/// an overlapping find/avoid address makes the found-set worker-dependent
/// (angr-03vl4.14).
#[test]
fn find_wins_when_a_pc_is_in_both_find_and_avoid() {
    Python::initialize();
    Python::attach(|_py| {
        let addr = 0x40_2000;
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_avoid_addrs(vec![addr]);
        mgr.set_find_addrs(vec![addr]);
        let mut h = Harness::new(&mgr);
        let shared = ParallelShared::seeded(0, 1);
        let state = state_at(addr);
        let id = state.state_id();

        let out = h.run(state, &CancelToken::new(), &shared);

        assert!(
            out.terminal_summaries.is_empty(),
            "a find is materialized, not summarized as avoided"
        );
        assert_eq!(out.terminal_states.len(), 1, "collected as a find");
        assert!(matches!(kind_of(&shared, id), Some(MatKind::Found)));
        assert_eq!(shared.worker_found_hint.load(Ordering::SeqCst), 1);
    });
}

// --- Pre-step find ----------------------------------------------------------

#[test]
fn find_addr_materializes_the_state_and_stamps_root_and_kind() {
    Python::initialize();
    Python::attach(|_py| {
        let find = 0x40_3000;
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_find_addrs(vec![find]);
        let mut h = Harness::new(&mgr);
        // num_find=2 with one already found: this find is the second-to-last,
        // so the hint must NOT trip the pool cancel.
        let shared = ParallelShared::seeded(0, 2);
        let state = state_at(find);
        let id = state.state_id();
        let root = 0xdead_beef;
        shared.root_map.lock().unwrap().insert(id, root);

        let out = h.run(state, &CancelToken::new(), &shared);

        assert_eq!(out.terminal_states.len(), 1, "found crosses the join");
        assert_eq!(out.terminal_states[0].state_id(), id);
        assert!(out.continue_states.is_empty());
        assert!(out.terminal_summaries.is_empty());
        assert!(!out.request_cancel, "1 of 2 finds does not trip cancel");
        assert!(matches!(kind_of(&shared, id), Some(MatKind::Found)));
        assert_eq!(
            root_of(&shared, id),
            Some(root),
            "the seeded lineage root is re-stamped, not overwritten with id"
        );
        assert_eq!(shared.worker_found_hint.load(Ordering::SeqCst), 1);
        assert_eq!(shared.stepped.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn find_reaching_num_find_requests_cancel() {
    Python::initialize();
    Python::attach(|_py| {
        let find = 0x40_4000;
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_find_addrs(vec![find]);
        let mut h = Harness::new(&mgr);
        let shared = ParallelShared::seeded(1, 2);

        let out = h.run(state_at(find), &CancelToken::new(), &shared);

        assert!(
            out.request_cancel,
            "the 2nd of num_find=2 trips the worker-side early cancel"
        );
        assert_eq!(shared.worker_found_hint.load(Ordering::SeqCst), 2);
    });
}

#[test]
fn unsat_state_at_a_find_addr_is_pruned_not_found() {
    Python::initialize();
    Python::attach(|_py| {
        let find = 0x40_5000;
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_find_addrs(vec![find]);
        let mut h = Harness::new(&mgr);
        let shared = ParallelShared::seeded(0, 1);
        let state = unsat_state_at(find);
        let id = state.state_id();

        let out = h.run(state, &CancelToken::new(), &shared);

        assert!(
            out.terminal_states.is_empty(),
            "an infeasible path is no find"
        );
        assert_eq!(
            out.terminal_summaries[0].disposition,
            SchedDisposition::Pruned
        );
        assert!(!out.request_cancel);
        assert!(kind_of(&shared, id).is_none());
        assert_eq!(
            shared.worker_found_hint.load(Ordering::SeqCst),
            0,
            "a pruned find must not consume a num_find slot"
        );
    });
}

#[test]
fn lazy_solves_collects_a_find_without_the_satisfiability_check() {
    Python::initialize();
    Python::attach(|_py| {
        let find = 0x40_6000;
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_find_addrs(vec![find]);
        mgr.set_lazy_solves(true);
        let mut h = Harness::new(&mgr);
        let shared = ParallelShared::seeded(0, 1);
        // Same UNSAT fixture as the test above — under lazy_solves the worker
        // skips `state.satisfiable()` entirely, so it routes to FOUND.
        let state = unsat_state_at(find);
        let id = state.state_id();

        let out = h.run(state, &CancelToken::new(), &shared);

        assert_eq!(out.terminal_states.len(), 1);
        assert!(matches!(kind_of(&shared, id), Some(MatKind::Found)));
        assert!(out.request_cancel);
    });
}

// --- Cancel preemption ------------------------------------------------------

#[test]
fn cancel_preemption_hands_the_frontier_state_back_untagged() {
    Python::initialize();
    Python::attach(|_py| {
        // Bug M1 (angr-op0dn.13.8): a preempted dispatch must NOT drop its
        // state. It comes back as a terminal with no kind_map entry, which the
        // coordinator's untagged arm routes to STASH_ACTIVE.
        let mgr = RustExplorationManager::new("amd64", None).unwrap();
        let mut h = Harness::new(&mgr);
        let shared = ParallelShared::seeded(0, 1);
        let cancel = CancelToken::new();
        cancel.cancel();
        assert!(cancel.preempts_in_flight());
        let state = state_at(0x40_7000);
        let id = state.state_id();

        let out = h.run(state, &cancel, &shared);

        assert_eq!(out.terminal_states.len(), 1, "state is preserved");
        assert_eq!(out.terminal_states[0].state_id(), id);
        assert!(out.continue_states.is_empty());
        assert!(out.terminal_summaries.is_empty());
        assert!(
            !out.request_cancel,
            "a preempted worker does not re-request the cancel"
        );
        assert!(
            kind_of(&shared, id).is_none(),
            "untagged is what makes the coordinator route it to STASH_ACTIVE"
        );
        assert_eq!(
            shared.stepped.load(Ordering::SeqCst),
            0,
            "the bail-out happens before the interpreter step"
        );
    });
}

#[test]
fn find_check_precedes_the_cancel_preemption_bail_out() {
    Python::initialize();
    Python::attach(|_py| {
        let find = 0x40_8000;
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.set_find_addrs(vec![find]);
        let mut h = Harness::new(&mgr);
        let shared = ParallelShared::seeded(0, 1);
        let cancel = CancelToken::new();
        cancel.cancel();
        let state = state_at(find);
        let id = state.state_id();

        let _out = h.run(state, &cancel, &shared);

        assert!(
            matches!(kind_of(&shared, id), Some(MatKind::Found)),
            "a state already sitting on a find addr is collected, not returned \
             to STASH_ACTIVE as an unexplored frontier"
        );
    });
}

// --- NeedsPython (bounce) ---------------------------------------------------

#[test]
fn python_simprocedure_bounce_is_materialized_and_re_pointed_at_its_addr() {
    Python::initialize();
    Python::attach(|_py| {
        // A hooked initial pc short-circuits before any block lift, so this arm
        // is reachable without a lifter (see the module scope note). The name
        // matches no native procedure, so the dispatch declines to Python.
        let addr = 0x40_9000;
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.register_simprocedure(addr, "angr_sqfj8_43_no_such_proc".to_string(), 0, false);
        let mut h = Harness::new(&mgr);
        let shared = ParallelShared::seeded(0, 1);
        let state = state_at(addr);
        let id = state.state_id();
        let root = 0x1234;
        shared.root_map.lock().unwrap().insert(id, root);

        let out = h.run(state, &CancelToken::new(), &shared);

        assert_eq!(
            out.terminal_states.len(),
            1,
            "the bounce state crosses back"
        );
        assert_eq!(
            out.terminal_states[0].pc(),
            addr,
            "angr-op0dn.13.11: the bounce pc is stamped with its re-enterable \
             entry address, not the 0 the step preamble wrote"
        );
        match kind_of(&shared, out.terminal_states[0].state_id()) {
            Some(MatKind::Bounce(BounceKind::SimProcedurePython { addr: a, .. })) => {
                assert_eq!(a, addr);
            }
            _ => panic!("expected a SimProcedurePython bounce"),
        }
        assert_eq!(
            root_of(&shared, out.terminal_states[0].state_id()),
            Some(root)
        );
        assert_eq!(
            shared.stepped.load(Ordering::SeqCst),
            0,
            "M2: a bounce is not a step — the coordinator counts it only if \
             dispatch_bounce later resolves it"
        );
        assert_eq!(
            shared.counters.lock().unwrap().len(),
            1,
            "the step's manager-level counters are still handed over"
        );
    });
}

#[test]
fn bare_hook_bounce_carries_the_hook_kind() {
    Python::initialize();
    Python::attach(|_py| {
        let addr = 0x40_a000;
        let mut mgr = RustExplorationManager::new("amd64", None).unwrap();
        mgr.add_hook(addr);
        let mut h = Harness::new(&mgr);
        let shared = ParallelShared::seeded(0, 1);

        let out = h.run(state_at(addr), &CancelToken::new(), &shared);

        assert_eq!(out.terminal_states.len(), 1);
        let bid = out.terminal_states[0].state_id();
        match kind_of(&shared, bid) {
            Some(MatKind::Bounce(BounceKind::Hook { addr: a })) => assert_eq!(a, addr),
            _ => panic!("expected a Hook bounce"),
        }
        assert_eq!(out.terminal_states[0].pc(), addr);
        assert_eq!(shared.stepped.load(Ordering::SeqCst), 0);
    });
}
