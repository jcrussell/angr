//! angr-c7xno.33: `handle_symbolic_jump_target_core`, both arms — the
//! single-target constrain-in-place path and the multi-target fork from an
//! unconstrained base.

use super::*;

const JUMP_COND_ID: u64 = 77;

/// Assert that a symbolic-jump successor pinned `expr` to `target`.
///
/// The pin is only *observable* where a model exists: without Z3 the mock
/// solver's `eval` returns `None` for every symbolic BV, so the no-z3 build
/// asserts exactly that rather than z3-gating the whole test away — the rest
/// of each test (pc advanced, forked vs moved, ids, tags) is engine-agnostic
/// and stays covered in both builds (angr-c7xno.100).
fn assert_jump_pinned(
    state: &RustSimState,
    expr: &crate::symbolic::RustBV,
    target: u64,
    what: &str,
) {
    let expected = cfg!(feature = "vex-engine-z3").then(|| u128::from(target));
    assert_eq!(state.eval(expr), expected, "{what}");
}

/// Drive one `RunResult::SymbolicJumpTarget` through `run_post_step_core` with
/// a fresh symbolic jump expression stored under [`JUMP_COND_ID`] (unless
/// `with_expr` is false, which models the id-missing path).
///
/// Returns `(outcome, state_id, jump_expr)` — the expression comes back so each
/// successor's solver can be asked what it pinned the jump to.
fn dispatch_symbolic_jump(
    targets: Vec<u64>,
    with_expr: bool,
    keep_ip_symbolic: bool,
) -> (CoreOutcome, u64, crate::symbolic::RustBV) {
    let mgr = RustExplorationManager::new("amd64", None).unwrap();
    let ctx = mgr.step_context();
    let prof = ParallelProfiling::default();
    let procs = NativeProcedureRegistry::new();
    let syscalls = NativeSyscallRegistry::new();

    let mut state = RustSimState::new("amd64").unwrap();
    state.set_pc(0x40_0000);
    state.set_keep_ip_symbolic(keep_ip_symbolic);
    let sid = state.state_id();
    let bits = state.arch().bits();
    let expr = crate::symbolic::RustBV::symbolic(&state.solver().borrow(), "jump_target", bits);

    let mut stored_conditions = FxHashMap::default();
    if with_expr {
        stored_conditions.insert(JUMP_COND_ID, expr.clone());
    }

    let outcome = run_post_step_core(
        &CoreCtx {
            ctx: &ctx,
            prof: &prof,
            native_procs: &procs,
            native_syscalls: &syscalls,
            callbacks: None,
        },
        state,
        PostStepInputs {
            result: RunResult::SymbolicJumpTarget {
                targets,
                condition_id: JUMP_COND_ID,
                jumpkind: "Ijk_Ret".to_string(),
            },
            deferred_forks: Vec::new(),
            last_condition: None,
            stored_conditions,
            fork_snapshots: FxHashMap::default(),
        },
        sid,
    );
    (outcome, sid, expr)
}

/// Concretization that produced no targets at all deadends the state rather
/// than continuing it somewhere arbitrary.
#[test]
fn symbolic_jump_no_targets_deadends() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let (outcome, sid, _) = dispatch_symbolic_jump(Vec::new(), true, false);
        match outcome.ret {
            CoreReturn::Deadended(s) => assert_eq!(s.state_id(), sid),
            _ => panic!("expected Deadended for an empty target list"),
        }
        assert!(outcome.fork_ids.is_empty());
        assert!(outcome.terminal_pushes.is_empty());
    });
}

/// Single target: the state moves in place (no fork) and the jump expression is
/// constrained to the concretized address.
#[test]
fn symbolic_jump_single_target_constrains_in_place() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        const TARGET: u64 = 0x40_9000;
        let (outcome, sid, expr) = dispatch_symbolic_jump(vec![TARGET], true, false);
        assert!(outcome.fork_ids.is_empty(), "a lone target must not fork");
        let CoreReturn::Continue(succ) = outcome.ret else {
            panic!("expected Continue");
        };
        assert_eq!(succ.len(), 1);
        assert_eq!(succ[0].0.state_id(), sid, "moved, not forked");
        assert!(!succ[0].1.is_fork);
        assert_eq!(succ[0].0.pc(), TARGET);
        assert_jump_pinned(
            &succ[0].0,
            &expr,
            TARGET,
            "jump expression pinned to the target",
        );
        assert!(
            !matches!(succ[0].0.get_ip(), crate::symbolic::RustBV::Symbolic { .. }),
            "default mode concretizes IP"
        );
    });
}

/// Under `keep_ip_symbolic` the same arm keeps the symbolic IP and adds NO
/// constraint — the whole point of the option is that the jump stays open.
#[test]
fn symbolic_jump_single_target_keep_ip_symbolic_adds_no_constraint() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        const TARGET: u64 = 0x40_9000;
        let (outcome, _, _) = dispatch_symbolic_jump(vec![TARGET], true, true);
        let CoreReturn::Continue(succ) = outcome.ret else {
            panic!("expected Continue");
        };
        assert_eq!(succ[0].0.pc(), TARGET, "pc still advances to the target");
        assert!(
            matches!(succ[0].0.get_ip(), crate::symbolic::RustBV::Symbolic { .. }),
            "IP register must stay symbolic"
        );
        assert_eq!(
            succ[0].0.solver().borrow().num_constraints(),
            0,
            "keep_ip_symbolic must not pin the jump expression"
        );
    });
}

/// A condition id absent from `stored_conditions` still advances the pc — there
/// is simply nothing to constrain.
#[test]
fn symbolic_jump_missing_condition_sets_pc_without_constraining() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        const TARGET: u64 = 0x40_a000;
        let (outcome, _, _) = dispatch_symbolic_jump(vec![TARGET], false, false);
        let CoreReturn::Continue(succ) = outcome.ret else {
            panic!("expected Continue");
        };
        assert_eq!(succ.len(), 1);
        assert_eq!(succ[0].0.pc(), TARGET);
        assert_eq!(succ[0].0.solver().borrow().num_constraints(), 0);
    });
}

/// Multi-target: one successor per concretized address. Each fork is minted
/// from the UNCONSTRAINED original, so every child must pin the jump expression
/// to its own target — a fork taken off the already-constrained first state
/// would come back UNSAT (or evaluate to the first target).
#[test]
fn symbolic_jump_multiple_targets_fork_from_unconstrained_base() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        const TARGETS: [u64; 3] = [0x40_b000, 0x40_c000, 0x40_d000];
        let (outcome, sid, expr) = dispatch_symbolic_jump(TARGETS.to_vec(), true, false);
        let CoreReturn::Continue(succ) = outcome.ret else {
            panic!("expected Continue");
        };
        assert_eq!(succ.len(), TARGETS.len());

        // First successor is the moved original; the rest are forks stamped
        // with the lineage root. Symbolic-jump forks deliberately do NOT go
        // through `dispatch_fork_inspect`, so `fork_ids` stays empty.
        assert_eq!(succ[0].0.state_id(), sid);
        assert!(!succ[0].1.is_fork);
        assert!(outcome.fork_ids.is_empty());
        assert!(outcome.pruned.is_empty());

        let mut seen_ids = vec![succ[0].0.state_id()];
        for (i, (state, tag)) in succ.iter().enumerate() {
            assert_eq!(state.pc(), TARGETS[i], "successor {i} pc");
            assert_jump_pinned(
                state,
                &expr,
                TARGETS[i],
                &format!("successor {i} pinned the jump expression to its own target"),
            );
            assert!(state.satisfiable(), "successor {i} must be SAT");
            if i > 0 {
                assert!(tag.is_fork, "successor {i} is a fork");
                assert_eq!(tag.root_hint, Some(sid));
                assert!(
                    !seen_ids.contains(&state.state_id()),
                    "fork {i} reuses a state id"
                );
                seen_ids.push(state.state_id());
            }
        }
    });
}
