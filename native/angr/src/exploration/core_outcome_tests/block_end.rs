//! `RunResult::BlockEnd` successors, deferred-fork materialization, and the
//! natively-forked `RunResult::SymbolicBranch`: the arms where the core turns
//! one stepped state into the successor set the run loop routes.

use super::*;

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
                callbacks: None,
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

        // condition_id 999 is absent from stored_conditions -> conservative
        // fork path: base.fork() to the unexplored target, SAT check, dispatch.
        let inputs = PostStepInputs {
            result: block_end(0x40_1000),
            deferred_forks: vec![DeferredFork {
                branch_addr: 0x40_0500,
                path_taken: true,
                unexplored_target: 0x40_2000,
                condition_id: 999,
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
                callbacks: None,
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
fn symbolic_branch_forks_both_targets_natively() {
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
                callbacks: None,
            },
            state,
            inputs,
            0,
        );
        // angr-gorvf.14: eager-mode symbolic branches resolve in Rust — both
        // children come back as successors instead of parking for Python.
        let CoreReturn::Continue(succ) = outcome.ret else {
            panic!("symbolic branch did not resolve natively");
        };
        let mut pcs: Vec<u64> = succ.iter().map(|(s, _)| s.pc()).collect();
        pcs.sort_unstable();
        assert_eq!(pcs, vec![0x40_5000, 0x40_6000]);
        assert!(outcome.pruned.is_empty());
    });
}
