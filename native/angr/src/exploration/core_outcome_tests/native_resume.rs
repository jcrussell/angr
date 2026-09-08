//! angr-c7xno.32: the *production* native-resume path.
//!
//! `handle_native_resume_core` is what both engines actually run when the guest
//! returns to the resume sentinel; `stepping_subcall.rs`'s `handle_native_resume`
//! is a
//! `#[cfg(test)]`-only twin. These drive the core through `run_post_step_core`
//! so the two cannot silently drift.

use super::*;

const RESUME_CALLER_RET: u64 = 0x0040_0123;
const RESUME_SP: u64 = 0x7fff_0000;
const RESUME_GUEST_TARGET: u64 = 0x0040_1000;

/// Continuation-carrying proc: tag 1 returns `saved_arg + 1`, tag 2 issues a
/// *nested* sub-call, tag 3 fails. Covers all three `resume()` outcomes the
/// core has to route.
struct ResumeTestProc;
impl ResumeTestProc {
    const TAG_RETURN: u32 = 1;
    const TAG_NESTED: u32 = 2;
    const TAG_ERR: u32 = 3;
}
impl crate::procedures::NativeSimProcedure for ResumeTestProc {
    fn name(&self) -> &'static str {
        "resume_core_test"
    }
    fn num_args(&self) -> usize {
        1
    }
    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[crate::symbolic::RustBV],
    ) -> Result<Option<crate::symbolic::RustBV>, crate::procedures::ProcedureError> {
        // Never reached: these tests enter at the resume sentinel.
        Ok(None)
    }
    fn resume(
        &self,
        _state: &mut RustSimState,
        resume_tag: u32,
        saved_args: &[crate::symbolic::RustBV],
    ) -> Result<crate::procedures::ProcOutcome, crate::procedures::ProcedureError> {
        match resume_tag {
            Self::TAG_RETURN => {
                let v = saved_args[0].as_u64().expect("saved arg concrete");
                Ok(crate::procedures::ProcOutcome::Return(Some(
                    crate::symbolic::RustBV::concrete((v + 1) as u128, 64),
                )))
            }
            Self::TAG_NESTED => Ok(crate::procedures::ProcOutcome::CallAndResume {
                target: RESUME_GUEST_TARGET,
                args: vec![],
                resume_tag: Self::TAG_RETURN,
            }),
            _ => Err(crate::procedures::ProcedureError::NotImplemented),
        }
    }
}

/// Drive one `RunResult::SimProcedure { name: NATIVE_RESUME_SENTINEL_NAME, .. }`
/// through `run_post_step_core`, i.e. the production dispatch into
/// `handle_native_resume_core`. `seed` prepares the state (resume frame, if
/// any) after a 2-page stack is mapped around [`RESUME_SP`] and the guest's
/// `ret` to the sentinel has been simulated (`sp += 8`).
fn dispatch_native_resume(seed: impl FnOnce(&mut RustSimState)) -> CoreOutcome {
    dispatch_native_resume_seeded(seed, None).0
}

/// [`dispatch_native_resume`] with an optional deferred fork riding into
/// `PostStepInputs` — a symbolic branch the guest sub-call *body* took before
/// returning to the sentinel (angr-6cp06.25). Hands back the main state's id
/// alongside the outcome so the sibling's routing can be checked.
fn dispatch_native_resume_seeded(
    seed: impl FnOnce(&mut RustSimState),
    fork: Option<ForkSeed>,
) -> (CoreOutcome, u64) {
    let mgr = RustExplorationManager::new("amd64", None).unwrap();
    let ctx = mgr.step_context();
    let prof = ParallelProfiling::default();
    let mut procs = NativeProcedureRegistry::new();
    procs.register(std::sync::Arc::new(ResumeTestProc));
    let syscalls = NativeSyscallRegistry::new();

    let mut state = RustSimState::new("amd64").unwrap();
    state.map_memory(RESUME_SP - 0x1000, 0x2000, crate::memory::Permission::RWX);
    // The guest routine has already `ret`-ed to the sentinel: it popped the
    // sentinel slot, so SP sits one word above where the sub-call was set up.
    state.set_sp(crate::symbolic::RustBV::concrete(
        (RESUME_SP + 8) as u128,
        64,
    ));
    state.set_pc(crate::procedures::native_resume_sentinel(8));
    seed(&mut state);
    let sid = state.state_id();

    let mut deferred_forks = Vec::new();
    let mut stored_conditions = FxHashMap::default();
    let mut fork_snapshots = FxHashMap::default();
    if let Some(seed) = fork {
        let guard = crate::symbolic::RustBV::symbolic(&state.solver().borrow(), "resume_guard", 1);
        stored_conditions.insert(FORK_COND_ID, guard);
        if seed.with_snapshot {
            fork_snapshots.insert(
                FORK_COND_ID,
                crate::interpreter::BranchSnapshot {
                    solver: state.solver().borrow().fork(),
                    registers: state.registers().fork(),
                    memory: None,
                },
            );
        }
        deferred_forks.push(DeferredFork {
            branch_addr: RESUME_GUEST_TARGET,
            path_taken: seed.path_taken,
            unexplored_target: seed.unexplored_target,
            condition_id: FORK_COND_ID,
            condition_ast: None,
        });
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
            result: RunResult::SimProcedure {
                addr: crate::procedures::native_resume_sentinel(8),
                name: crate::procedures::NATIVE_RESUME_SENTINEL_NAME.to_string(),
                num_args: 0,
                return_addr: 0,
            },
            deferred_forks,
            last_condition: None,
            stored_conditions,
            fork_snapshots,
        },
        sid,
    );
    (outcome, sid)
}

/// Push the continuation frame the sub-call dispatcher would have left behind.
fn push_resume_frame(state: &mut RustSimState, resume_tag: u32, proc_name: &str) {
    state.push_native_resume_frame(crate::state::NativeResumeFrame {
        proc_name: proc_name.to_string(),
        resume_tag,
        saved_args: vec![crate::symbolic::RustBV::concrete(41, 64)],
        caller_return_addr: RESUME_CALLER_RET,
    });
}

/// Happy path: `ProcOutcome::Return` lands the state at the *frame's*
/// `caller_return_addr` (not [sp]), writes the return register, and drains the
/// resume stack.
#[test]
fn native_resume_core_returns_to_caller_and_drains_frame() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let outcome = dispatch_native_resume(|state| {
            push_resume_frame(state, ResumeTestProc::TAG_RETURN, "resume_core_test");
        });
        match outcome.ret {
            CoreReturn::Continue(succ) => {
                assert_eq!(succ.len(), 1, "no deferred forks -> one main successor");
                let out = &succ[0].0;
                assert_eq!(out.pc(), RESUME_CALLER_RET);
                assert!(out.native_resume_stack().is_empty(), "frame popped");
                // saved_arg + 1 = 42 proves resume() ran with the saved args.
                let ret_reg = RustExplorationManager::new("amd64", None)
                    .unwrap()
                    .environment
                    .calling_convention
                    .return_register();
                assert_eq!(out.get_register_by_offset(ret_reg, 8).as_u64(), Some(42));
            }
            _ => panic!("expected Continue"),
        }
    });
}

/// `ProcOutcome::CallAndResume` from inside `resume()` must set up a *nested*
/// sub-call rather than returning: PC enters the guest routine and a fresh
/// frame replaces the popped one.
#[test]
fn native_resume_core_nested_subcall_pushes_new_frame() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let outcome = dispatch_native_resume(|state| {
            push_resume_frame(state, ResumeTestProc::TAG_NESTED, "resume_core_test");
        });
        match outcome.ret {
            CoreReturn::Continue(succ) => {
                let out = &succ[0].0;
                assert_eq!(out.pc(), RESUME_GUEST_TARGET, "entered the nested callee");
                assert_eq!(out.native_resume_stack().len(), 1, "new frame pushed");
                let f = &out.native_resume_stack()[0];
                assert_eq!(f.resume_tag, ResumeTestProc::TAG_RETURN);
                // The original caller return address rides forward across the
                // nested call — losing it would resume the outermost caller at
                // the sentinel.
                assert_eq!(f.caller_return_addr, RESUME_CALLER_RET);
                assert_eq!(f.saved_args[0].as_u64(), Some(41));
            }
            _ => panic!("expected Continue"),
        }
    });
}

/// The three failure modes all deadend rather than silently continuing at the
/// sentinel address: empty resume stack, proc missing from the registry, and a
/// `resume()` that errors.
#[test]
fn native_resume_core_failures_deadend() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        // (a) sentinel hit with nothing suspended.
        let empty = dispatch_native_resume(|_| {});
        assert!(matches!(empty.ret, CoreReturn::Deadended(_)));

        // (b) frame names a proc the registry doesn't have.
        let unknown = dispatch_native_resume(|state| {
            push_resume_frame(state, ResumeTestProc::TAG_RETURN, "not_registered");
        });
        assert!(matches!(unknown.ret, CoreReturn::Deadended(_)));

        // (c) resume() returns Err.
        let failed = dispatch_native_resume(|state| {
            push_resume_frame(state, ResumeTestProc::TAG_ERR, "resume_core_test");
        });
        assert!(matches!(failed.ret, CoreReturn::Deadended(_)));
    });
}

/// angr-6cp06.25: a branch taken *inside* the sub-call body forks a sibling
/// that has not yet reached the sentinel — so it must still carry the resume
/// frame the main successor just popped.
///
/// `fork_with` copies `native_resume_stack` from the fork base and no
/// `BranchSnapshot` covers that field, so materializing against the
/// already-popped state handed the sibling an empty stack; it would then hit
/// the sentinel itself and be deadended by the "empty resume stack" arm, losing
/// the path. `process_deferred_forks_rewound` rewinds just that field.
#[test]
fn native_resume_fork_keeps_the_frame_the_main_successor_popped() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        for with_snapshot in [true, false] {
            let (outcome, sid) = dispatch_native_resume_seeded(
                |state| push_resume_frame(state, ResumeTestProc::TAG_RETURN, "resume_core_test"),
                Some(ForkSeed {
                    path_taken: true,
                    unexplored_target: RESUME_GUEST_TARGET + 0x20,
                    with_snapshot,
                }),
            );
            assert!(outcome.pruned.is_empty(), "the sibling is SAT");
            let CoreReturn::Continue(succ) = outcome.ret else {
                panic!("expected Continue");
            };
            assert_eq!(succ.len(), 2, "main successor plus the sibling");
            let (main, fork) = (&succ[0].0, &succ[1].0);
            assert_eq!(main.state_id(), sid);
            assert!(
                main.native_resume_stack().is_empty(),
                "main successor consumed the frame (with_snapshot={with_snapshot})"
            );
            assert_eq!(
                fork.native_resume_stack().len(),
                1,
                "sibling never reached the sentinel, so it keeps the frame \
                 (with_snapshot={with_snapshot})"
            );
            assert_eq!(
                fork.native_resume_stack()[0].caller_return_addr,
                RESUME_CALLER_RET
            );
            assert_eq!(fork.pc(), RESUME_GUEST_TARGET + 0x20);
        }
    });
}

/// The nested-sub-call outcome is the mirror: `setup_native_subcall` pushed a
/// *fresh* frame onto the main successor, which the sibling — branched before
/// the nested call was made — must not inherit. It keeps exactly the frame that
/// was live when it branched.
#[test]
fn native_resume_nested_subcall_fork_does_not_inherit_the_new_frame() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let (outcome, _sid) = dispatch_native_resume_seeded(
            |state| push_resume_frame(state, ResumeTestProc::TAG_NESTED, "resume_core_test"),
            Some(ForkSeed {
                path_taken: false,
                unexplored_target: RESUME_GUEST_TARGET + 0x20,
                with_snapshot: true,
            }),
        );
        let CoreReturn::Continue(succ) = outcome.ret else {
            panic!("expected Continue");
        };
        assert_eq!(succ.len(), 2);
        let (main, fork) = (&succ[0].0, &succ[1].0);
        assert_eq!(main.native_resume_stack().len(), 1, "nested frame pushed");
        assert_eq!(
            main.native_resume_stack()[0].resume_tag,
            ResumeTestProc::TAG_RETURN
        );
        assert_eq!(
            fork.native_resume_stack().len(),
            1,
            "sibling keeps its own single pre-nesting frame"
        );
        assert_eq!(
            fork.native_resume_stack()[0].resume_tag,
            ResumeTestProc::TAG_NESTED,
            "and it is the frame that was live when the branch was taken"
        );
    });
}
