//! angr-c7xno.33: `handle_syscall_core`, all four dispatch outcomes — plus the
//! ABI half of the syscall return path, `CcSnapshot::write_syscall_return`.

use super::*;

/// A syscall number outside every production table in
/// `NativeSyscallRegistry::new()`, so registering [`StubSyscall`] under it
/// cannot shadow (or be shadowed by) a real handler.
const STUB_SYSCALL_NUM: u64 = 0xdead_beef;
const SYSCALL_PC: u64 = 0x40_7000;

/// What [`StubSyscall`] does when `handle_syscall_core` dispatches it.
enum StubSyscallBehavior {
    /// Native success with a concrete return value.
    Continue(u64),
    /// Native success writing a fresh symbolic return value.
    ContinueSymbolic,
    /// Native success that deadends the state (exit / exit_group).
    Exit,
    /// Registered handler that declines — falls through to Python.
    Decline,
}

/// Stands in for any registered native syscall handler, with the outcome and
/// the declared arity both under test control (a `num_args` past the ABI's
/// register window is how the arg-extraction-failure arm is reached).
struct StubSyscall {
    behavior: StubSyscallBehavior,
    num_args: usize,
}

impl crate::syscalls::NativeSyscall for StubSyscall {
    fn name(&self) -> &'static str {
        "stub_syscall_test"
    }
    fn num_args(&self) -> usize {
        self.num_args
    }
    fn call(
        &self,
        state: &mut RustSimState,
        _args: &[crate::symbolic::RustBV],
    ) -> Result<crate::syscalls::SyscallOutcome, crate::syscalls::SyscallError> {
        use crate::syscalls::SyscallOutcome;
        match self.behavior {
            StubSyscallBehavior::Continue(ret) => Ok(SyscallOutcome::Continue { ret }),
            StubSyscallBehavior::ContinueSymbolic => Ok(SyscallOutcome::ContinueSymbolic {
                ret: crate::symbolic::RustBV::symbolic(
                    &state.solver().borrow(),
                    "stub_syscall_ret",
                    state.arch().bits(),
                ),
            }),
            StubSyscallBehavior::Exit => Ok(SyscallOutcome::Exit),
            StubSyscallBehavior::Decline => Err(crate::syscalls::SyscallError::Other(
                "declined by stub".to_string(),
            )),
        }
    }
}

/// Drive one `RunResult::Syscall { num, pc: SYSCALL_PC }` through
/// `run_post_step_core`, optionally with `stub` registered for AMD64 at
/// [`STUB_SYSCALL_NUM`]. Returns `(outcome, state_id)`.
fn dispatch_syscall(num: Option<u64>, stub: Option<StubSyscall>) -> (CoreOutcome, u64) {
    let mgr = RustExplorationManager::new("amd64", None).unwrap();
    let ctx = mgr.step_context();
    let prof = ParallelProfiling::default();
    let procs = NativeProcedureRegistry::new();
    let mut syscalls = NativeSyscallRegistry::new();
    if let Some(stub) = stub {
        syscalls.register("AMD64", STUB_SYSCALL_NUM, std::sync::Arc::new(stub));
    }

    let mut state = RustSimState::new("amd64").unwrap();
    state.set_pc(0x40_0000);
    let sid = state.state_id();

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
            result: RunResult::Syscall {
                num,
                pc: SYSCALL_PC,
            },
            deferred_forks: Vec::new(),
            last_condition: None,
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        },
        sid,
    );
    (outcome, sid)
}

/// The native fast path: `SyscallOutcome::Continue` writes the concrete return
/// value to the ABI return register (rax on amd64), leaves the state at the
/// syscall pc, and counts one native dispatch keyed by the syscall number.
#[test]
fn syscall_native_continue_writes_return_register_and_counts() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let (outcome, sid) = dispatch_syscall(
            Some(STUB_SYSCALL_NUM),
            Some(StubSyscall {
                behavior: StubSyscallBehavior::Continue(0x2a),
                num_args: 0,
            }),
        );

        assert_eq!(outcome.counters.syscall_native_count, 1);
        assert_eq!(
            outcome.counters.syscall_native_by_num[&(STUB_SYSCALL_NUM as i128)],
            1
        );
        assert_eq!(outcome.counters.syscall_python_fallback_count, 0);
        assert!(outcome.terminal_pushes.is_empty());

        let CoreReturn::Continue(succ) = outcome.ret else {
            panic!("expected Continue for a native syscall");
        };
        assert_eq!(succ.len(), 1, "main state only");
        assert_eq!(succ[0].0.state_id(), sid);
        assert!(!succ[0].1.is_fork);
        assert_eq!(succ[0].0.pc(), SYSCALL_PC, "state parked at the syscall pc");
        assert_eq!(
            succ[0].0.get_register("rax").and_then(|bv| bv.as_u64()),
            Some(0x2a),
            "return value landed in the amd64 syscall return register"
        );
    });
}

/// `SyscallOutcome::ContinueSymbolic` takes the same tail but writes the BV
/// straight through, so the return register must come out *symbolic* rather
/// than concretized.
#[test]
fn syscall_native_continue_symbolic_writes_symbolic_return() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let (outcome, _) = dispatch_syscall(
            Some(STUB_SYSCALL_NUM),
            Some(StubSyscall {
                behavior: StubSyscallBehavior::ContinueSymbolic,
                num_args: 0,
            }),
        );
        assert_eq!(outcome.counters.syscall_native_count, 1);
        let CoreReturn::Continue(succ) = outcome.ret else {
            panic!("expected Continue");
        };
        let rax = succ[0].0.get_register("rax").expect("rax exists");
        assert!(
            rax.as_u64().is_none(),
            "symbolic syscall return must not be concretized, got {rax:?}"
        );
    });
}

/// `SyscallOutcome::Exit` deadends: the state leaves via `terminal_pushes`
/// (STASH_DEADENDED) and is NOT also returned as a successor.
#[test]
fn syscall_native_exit_pushes_state_to_deadended() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let (outcome, sid) = dispatch_syscall(
            Some(STUB_SYSCALL_NUM),
            Some(StubSyscall {
                behavior: StubSyscallBehavior::Exit,
                num_args: 0,
            }),
        );
        assert_eq!(outcome.counters.syscall_native_count, 1);
        assert_eq!(outcome.terminal_pushes.len(), 1);
        assert_eq!(outcome.terminal_pushes[0].0.state_id(), sid);
        assert_eq!(outcome.terminal_pushes[0].1, crate::stash::STASH_DEADENDED);
        let CoreReturn::Continue(succ) = outcome.ret else {
            panic!("expected Continue (with no successors)");
        };
        assert!(succ.is_empty(), "exiting state must not also continue");
    });
}

/// A registered handler that *declines* falls through to the Python syscall
/// implementation, counted as a fallback (not as a native dispatch).
#[test]
fn syscall_native_decline_bounces_to_python() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let (outcome, _) = dispatch_syscall(
            Some(STUB_SYSCALL_NUM),
            Some(StubSyscall {
                behavior: StubSyscallBehavior::Decline,
                num_args: 0,
            }),
        );
        assert_eq!(outcome.counters.syscall_native_count, 0);
        assert_eq!(outcome.counters.syscall_python_fallback_count, 1);
        assert_eq!(
            outcome.counters.syscall_python_fallback_by_num[&(STUB_SYSCALL_NUM as i128)],
            1
        );
        match outcome.ret {
            CoreReturn::NeedsPython(bounce) => match bounce.kind {
                BounceKind::SyscallPython { num } => assert_eq!(num, Some(STUB_SYSCALL_NUM)),
                other => panic!("expected SyscallPython bounce, got {other:?}"),
            },
            _ => panic!("expected NeedsPython"),
        }
    });
}

/// Arg-extraction failure takes an *earlier* fallback path than the decline
/// above — the handler never runs. amd64 exposes six syscall arg registers and
/// no stack path, so a handler declaring seven args is a guaranteed
/// `ExtractionError::RegisterOverflow`.
#[test]
fn syscall_arg_extraction_failure_bounces_before_calling_handler() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let (outcome, _) = dispatch_syscall(
            Some(STUB_SYSCALL_NUM),
            Some(StubSyscall {
                // Would deadend the state if it ever ran; it must not.
                behavior: StubSyscallBehavior::Exit,
                num_args: 7,
            }),
        );
        assert_eq!(outcome.counters.syscall_native_count, 0);
        assert_eq!(outcome.counters.syscall_python_fallback_count, 1);
        assert!(
            outcome.terminal_pushes.is_empty(),
            "the handler must not have run"
        );
        assert!(matches!(
            outcome.ret,
            CoreReturn::NeedsPython(PendingBounce {
                kind: BounceKind::SyscallPython {
                    num: Some(STUB_SYSCALL_NUM)
                },
                ..
            })
        ));
    });
}

/// No registered handler at all (and an unknown syscall number) bounces to
/// Python, with the `None` number folded into the `-1` fallback bucket.
#[test]
fn syscall_without_native_handler_bounces_to_python() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let (outcome, _) = dispatch_syscall(None, None);
        assert_eq!(outcome.counters.syscall_native_count, 0);
        assert_eq!(outcome.counters.syscall_python_fallback_count, 1);
        assert_eq!(
            outcome.counters.syscall_python_fallback_by_num[&-1],
            1,
            "an unknown syscall number is bucketed as -1"
        );
        assert!(matches!(
            outcome.ret,
            CoreReturn::NeedsPython(PendingBounce {
                kind: BounceKind::SyscallPython { num: None },
                ..
            })
        ));
    });
}

/// angr-0jh0j.18: a concrete syscall number of `u64::MAX` must keep its own
/// bucket. Under the old `n as i64` key it also produced `-1` and silently
/// merged into the "syscall register was symbolic" bucket asserted above.
#[test]
fn syscall_num_u64_max_does_not_collide_with_unknown_sentinel() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let (outcome, _) = dispatch_syscall(Some(u64::MAX), None);
        assert_eq!(outcome.counters.syscall_python_fallback_count, 1);
        assert_eq!(
            outcome.counters.syscall_python_fallback_by_num[&(u64::MAX as i128)],
            1,
            "u64::MAX must be bucketed under its own key"
        );
        assert!(
            !outcome
                .counters
                .syscall_python_fallback_by_num
                .contains_key(&-1),
            "u64::MAX must not land in the unknown-syscall bucket"
        );
    });
}

/// MIPS O32/N64 report syscall failure in `$a3` rather than by returning a
/// small negative value, so `CcSnapshot::write_syscall_return` must split a raw
/// kernel return into a *positive* errno in `$v0` plus an all-ones flag in
/// `$a3`, and must clear the flag on success. Nothing in the crate exercised
/// the `syscall_error_register` tuple before this (angr-5mnx3.4); a drifted
/// offset or threshold would have mis-classified every MIPS syscall silently.
#[test]
fn write_syscall_return_splits_errno_flag_into_a3_on_mips() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        for (arch, bits) in [("mips32", 32u32), ("mips64", 64u32)] {
            let cc = RustExplorationManager::new(arch, None)
                .unwrap()
                .step_context()
                .cc;
            let (err_reg, errno_start) =
                cc.syscall_error_register.expect("MIPS ABIs carry an $a3 flag");
            let ret_reg = cc.return_register;
            // get_register_by_offset's `size` is in bytes.
            let bytes = bits / 8;
            let all_ones = RustBV::ones(bits).as_u64();

            let write = |ret: RustBV| {
                let mut state = RustSimState::new(arch).unwrap();
                cc.write_syscall_return(&mut state, ret_reg, ret);
                (
                    state.get_register_by_offset(ret_reg, bytes).as_u64(),
                    state.get_register_by_offset(err_reg, bytes).as_u64(),
                )
            };

            // Success: a small return is passed through and $a3 is cleared.
            assert_eq!(
                write(RustBV::concrete(5, bits)),
                (Some(5), Some(0)),
                "{arch}: successful return belongs in $v0 with $a3 clear",
            );

            // Failure: the kernel's -ENOENT surfaces as +2 with $a3 set.
            assert_eq!(
                write(RustBV::concrete(-2i64 as u128, bits)),
                (Some(2), all_ones),
                "{arch}: failure means +errno in $v0 and an all-ones $a3",
            );

            // `errno_start` itself is an error (the compare is `uge`)...
            assert_eq!(
                write(RustBV::concrete(errno_start as u128, bits)),
                (Some(errno_start.unsigned_abs()), all_ones),
                "{arch}: errno_start is inclusive",
            );
            // ...and one below it is not, so it stays a (large) success value.
            let below = RustBV::concrete(errno_start as u128 - 1, bits);
            assert_eq!(
                write(below.clone()),
                (below.as_u64(), Some(0)),
                "{arch}: below the threshold is a success value, passed through",
            );
        }
    });
}

/// The control for the test above: an ABI with no `syscall_error_register`
/// must write the return register verbatim and touch nothing else — a kernel
/// `-ENOENT` stays `0xffff_ffff_ffff_fffe` on amd64.
#[test]
fn write_syscall_return_passes_through_on_abis_without_an_error_register() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let cc = fresh_ctx().cc;
        assert!(cc.syscall_error_register.is_none(), "amd64 has no flag reg");
        let mut state = RustSimState::new("amd64").unwrap();
        cc.write_syscall_return(&mut state, cc.return_register, RustBV::concrete(-2i64 as u128, 64));
        assert_eq!(
            state.get_register_by_offset(cc.return_register, 8).as_u64(),
            Some(-2i64 as u64),
        );
    });
}
