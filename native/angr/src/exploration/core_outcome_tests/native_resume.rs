//! angr-c7xno.32: the *production* native-resume path.
//!
//! `handle_native_resume_core` is what both engines actually run when the guest
//! returns to the resume sentinel; `stepping.rs`'s `handle_native_resume` is a
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

    run_post_step_core(
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
            deferred_forks: Vec::new(),
            last_condition: None,
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        },
        sid,
    )
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
