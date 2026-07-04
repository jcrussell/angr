//! Unit tests for the `&mut self`-free post-step core (angr-vh834).
//!
//! These drive a single state through [`run_post_step_core`] and assert the
//! structural outcome (successors / forks / pruned / fork_ids / routing) the
//! legacy `step_state_with_skip` match arms produced inline. The full
//! byte-identical proof is the Python `tests/engines/rust/` suite, which now
//! flows entirely through this core.

use super::*;

use crate::callbacks::{DeferredFork, RunErrorKind, RunResult};
use crate::exploration::RustExplorationManager;
use crate::procedures::NativeProcedureRegistry;
use crate::state::RustSimState;
use crate::syscalls::NativeSyscallRegistry;

fn fresh_ctx() -> super::StepContext {
    let mgr = RustExplorationManager::new("amd64", None).unwrap();
    mgr.step_context()
}

fn block_end(next_addr: u64) -> RunResult {
    RunResult::BlockEnd {
        next_addr,
        jumpkind: "Ijk_Boring".to_string(),
    }
}

#[test]
fn block_end_no_forks_returns_single_main_successor() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let ctx = fresh_ctx();
        let prof = ParallelProfiling::default();
        let procs = NativeProcedureRegistry::new();
        let syscalls = NativeSyscallRegistry::new();

        let mut state = RustSimState::new("amd64").unwrap();
        state.set_pc(0x40_0000);
        let sid = state.state_id();

        let inputs = PostStepInputs {
            result: block_end(0x40_1000),
            deferred_forks: Vec::new(),
            last_condition: None,
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        };
        let outcome = run_post_step_core(
            &CoreCtx {
                ctx: &ctx,
                prof: &prof,
                native_procs: &procs,
                native_syscalls: &syscalls,
            },
            state,
            inputs,
            sid,
        );

        assert!(outcome.pruned.is_empty());
        assert!(outcome.fork_ids.is_empty());
        assert!(outcome.terminal_pushes.is_empty());
        match outcome.ret {
            CoreReturn::Continue(succ) => {
                assert_eq!(succ.len(), 1, "main state only");
                assert_eq!(succ[0].0.state_id(), sid);
                assert!(!succ[0].1.is_fork, "main is not a fork");
                assert!(succ[0].1.root_hint.is_none());
                assert_eq!(succ[0].0.pc(), 0x40_1000);
            }
            _ => panic!("expected Continue"),
        }
    });
}

#[test]
fn block_end_missing_condition_fork_is_materialized_and_dispatched() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let ctx = fresh_ctx();
        let prof = ParallelProfiling::default();
        let procs = NativeProcedureRegistry::new();
        let syscalls = NativeSyscallRegistry::new();

        let mut state = RustSimState::new("amd64").unwrap();
        state.set_pc(0x40_0000);
        let sid = state.state_id();

        // condition_id 999 is absent from stored_conditions -> P15 conservative
        // fork path: base.fork() to the unexplored target, SAT check, dispatch.
        let inputs = PostStepInputs {
            result: block_end(0x40_1000),
            deferred_forks: vec![DeferredFork {
                branch_addr: 0x40_0500,
                path_taken: true,
                unexplored_target: 0x40_2000,
                condition_id: 999,
                push_level: 0,
                condition_ast: None,
            }],
            last_condition: None,
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        };
        let outcome = run_post_step_core(
            &CoreCtx {
                ctx: &ctx,
                prof: &prof,
                native_procs: &procs,
                native_syscalls: &syscalls,
            },
            state,
            inputs,
            sid,
        );

        assert!(outcome.pruned.is_empty(), "fresh state is SAT");
        assert_eq!(outcome.fork_ids.len(), 1, "one fork dispatched");
        match outcome.ret {
            CoreReturn::Continue(succ) => {
                assert_eq!(succ.len(), 2, "main + conservative fork");
                assert_eq!(succ[0].0.state_id(), sid);
                assert!(!succ[0].1.is_fork);
                // The fork carries the stamped root hint and is flagged a fork.
                assert!(succ[1].1.is_fork);
                assert_eq!(succ[1].1.root_hint, Some(sid));
                assert_eq!(succ[1].0.pc(), 0x40_2000);
                assert_eq!(outcome.fork_ids[0], succ[1].0.state_id());
                assert_ne!(succ[1].0.state_id(), sid);
            }
            _ => panic!("expected Continue"),
        }
    });
}

#[test]
fn error_deadend_kind_routes_to_deadended() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let ctx = fresh_ctx();
        let prof = ParallelProfiling::default();
        let procs = NativeProcedureRegistry::new();
        let syscalls = NativeSyscallRegistry::new();

        let state = RustSimState::new("amd64").unwrap();
        let sid = state.state_id();
        let inputs = PostStepInputs {
            result: RunResult::Error {
                message: "unliftable".to_string(),
                addr: 0x40_3000,
                kind: RunErrorKind::Deadend,
            },
            deferred_forks: Vec::new(),
            last_condition: None,
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        };
        let outcome = run_post_step_core(
            &CoreCtx {
                ctx: &ctx,
                prof: &prof,
                native_procs: &procs,
                native_syscalls: &syscalls,
            },
            state,
            inputs,
            sid,
        );
        match outcome.ret {
            CoreReturn::Deadended(s) => {
                assert_eq!(s.state_id(), sid);
                assert_eq!(s.pc(), 0x40_3000);
            }
            _ => panic!("expected Deadended"),
        }
    });
}

#[test]
fn error_fatal_kind_routes_to_errored() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let ctx = fresh_ctx();
        let prof = ParallelProfiling::default();
        let procs = NativeProcedureRegistry::new();
        let syscalls = NativeSyscallRegistry::new();

        let state = RustSimState::new("amd64").unwrap();
        let sid = state.state_id();
        let inputs = PostStepInputs {
            result: RunResult::Error {
                message: "boom".to_string(),
                addr: 0x40_4000,
                kind: RunErrorKind::Fatal,
            },
            deferred_forks: Vec::new(),
            last_condition: None,
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        };
        let outcome = run_post_step_core(
            &CoreCtx {
                ctx: &ctx,
                prof: &prof,
                native_procs: &procs,
                native_syscalls: &syscalls,
            },
            state,
            inputs,
            sid,
        );
        match outcome.ret {
            CoreReturn::Errored(s, msg) => {
                assert_eq!(s.state_id(), sid);
                assert_eq!(msg, "boom");
            }
            _ => panic!("expected Errored"),
        }
    });
}

#[test]
fn symbolic_branch_bounces_to_python() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let ctx = fresh_ctx();
        let prof = ParallelProfiling::default();
        let procs = NativeProcedureRegistry::new();
        let syscalls = NativeSyscallRegistry::new();

        let state = RustSimState::new("amd64").unwrap();
        let inputs = PostStepInputs {
            result: RunResult::SymbolicBranch {
                condition_id: 1,
                true_target: 0x40_5000,
                false_target: 0x40_6000,
            },
            deferred_forks: Vec::new(),
            last_condition: None,
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        };
        let outcome = run_post_step_core(
            &CoreCtx {
                ctx: &ctx,
                prof: &prof,
                native_procs: &procs,
                native_syscalls: &syscalls,
            },
            state,
            inputs,
            0,
        );
        assert!(matches!(
            outcome.ret,
            CoreReturn::NeedsPython(PendingBounce {
                kind: BounceKind::SymbolicBranch { .. },
                ..
            })
        ));
    });
}

/// angr-qhkye: `accumulate_step` must fold EVERY sum-typed interpreter counter
/// (not just cache hits/misses) into the shared accumulator, so the parallel
/// path reports `lift_time_ns` and siblings like the single-threaded
/// `accumulated_stats.merge(&step.step_stats)` does. Guards against the
/// regression where worker-thread `lift_time_ns` silently dropped to 0.
#[test]
fn accumulate_step_folds_full_stats_no_double_count() {
    let prof = ParallelProfiling::default();

    // Two worker dispatches with representative interpreter counters set.
    prof.accumulate_step(&ExecutionStats {
        lift_time_ns: 500,
        cache_hit_count: 3,
        cache_miss_count: 1,
        blocks_executed: 2,
        ..Default::default()
    });
    prof.accumulate_step(&ExecutionStats {
        lift_time_ns: 250,
        cache_hit_count: 4,
        blocks_executed: 1,
        ..Default::default()
    });

    // Independently, the post-step arms fold solver timing into the atomics.
    ParallelProfiling::add(&prof.solver_sat_count, 7);

    let mut stats = ExecutionStats::default();
    prof.fold_into(&mut stats);

    // Sum-typed interpreter counters accumulate across both steps.
    assert_eq!(stats.lift_time_ns, 750);
    assert_eq!(stats.cache_hit_count, 7);
    assert_eq!(stats.cache_miss_count, 1);
    assert_eq!(stats.blocks_executed, 3);
    // Atomic-tracked solver counter still folds (interpreter never sets it, so
    // the full-stats merge cannot double-count it).
    assert_eq!(stats.solver_sat_count, 7);
}

/// `drain_into` must reset the step-stats accumulator so a long-lived
/// (steady-state) accumulator folds deltas, not cumulative totals.
#[test]
fn drain_into_resets_step_stats_accumulator() {
    let prof = ParallelProfiling::default();

    prof.accumulate_step(&ExecutionStats {
        lift_time_ns: 100,
        cache_hit_count: 2,
        ..Default::default()
    });

    let mut first = ExecutionStats::default();
    prof.drain_into(&mut first);
    assert_eq!(first.lift_time_ns, 100);
    assert_eq!(first.cache_hit_count, 2);

    // Second drain with no further accumulation must add zero (the accumulator
    // was reset), not re-add the first step's totals.
    let mut second = ExecutionStats::default();
    prof.drain_into(&mut second);
    assert_eq!(second.lift_time_ns, 0);
    assert_eq!(second.cache_hit_count, 0);
}
