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
use crate::stash::STASH_ACTIVE;
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

        // condition_id 999 is absent from stored_conditions -> P15 conservative
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
                callbacks: None,
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
                callbacks: None,
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

/// angr-91vj9.9: a `Fatal` reported at pc 0 is deadended, not errored —
/// `ErrorRoute::NullAddressDeadend`. Pins the routing behaviour behind the
/// named variant (the classifier itself is unit-tested in `callbacks/events.rs`).
#[test]
fn error_fatal_at_null_addr_routes_to_deadended() {
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
                message: "boom at null".to_string(),
                addr: 0,
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
                callbacks: None,
            },
            state,
            inputs,
            sid,
        );
        match outcome.ret {
            CoreReturn::Deadended(s) => assert_eq!(s.state_id(), sid),
            _ => panic!("expected Deadended for a Fatal error at pc 0"),
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

// --- angr-1i5h7: native dispatch must not swallow a find/avoid target hook ---

const HOOK_ADDR: u64 = 0x50_0000;
const HOOK_RET: u64 = 0x40_2000;

/// Stands in for any zero-arg native libc proc (getenv/time/...) hooked in the
/// extern object, i.e. outside `binary_regions`, where native dispatch fires.
struct FindTargetProc;
impl crate::procedures::NativeSimProcedure for FindTargetProc {
    fn name(&self) -> &'static str {
        "find_target_test"
    }
    fn num_args(&self) -> usize {
        0
    }
    fn call(
        &self,
        state: &mut RustSimState,
        _args: &[crate::symbolic::RustBV],
    ) -> Result<Option<crate::symbolic::RustBV>, crate::procedures::ProcedureError> {
        Ok(Some(crate::symbolic::RustBV::concrete(
            0,
            state.arch().bits(),
        )))
    }
}

/// Drive one `SimProcedure` result for a hook at [`HOOK_ADDR`] with
/// `FindTargetProc` registered, under a manager configured by `cfg` (which
/// seeds find/avoid addresses). Returns the raw outcome.
fn dispatch_hook_with(cfg: impl FnOnce(&mut RustExplorationManager)) -> CoreOutcome {
    dispatch_hook_on_arch("amd64", SpSeed::Default, cfg)
}

/// How [`dispatch_hook_on_arch`] seeds the state's stack pointer.
enum SpSeed {
    /// Leave SP at whatever `RustSimState::new` produced.
    Default,
    /// Seed a concrete SP.
    Concrete(u64),
    /// Seed a fresh symbolic SP — `extract_args_with_abi` never touches SP for
    /// a zero-arg proc, so it stays symbolic through the return (angr-c7xno.29).
    Symbolic,
}

/// [`dispatch_hook_with`] on an arbitrary architecture, optionally seeding SP,
/// so the return path's SP adjustment can be checked per calling convention.
fn dispatch_hook_on_arch(
    arch: &str,
    sp: SpSeed,
    cfg: impl FnOnce(&mut RustExplorationManager),
) -> CoreOutcome {
    let mut mgr = RustExplorationManager::new(arch, None).unwrap();
    cfg(&mut mgr);
    let ctx = mgr.step_context();
    let prof = ParallelProfiling::default();
    let mut procs = NativeProcedureRegistry::new();
    procs.register(std::sync::Arc::new(FindTargetProc));
    let syscalls = NativeSyscallRegistry::new();

    let mut state = RustSimState::new(arch).unwrap();
    state.set_pc(HOOK_ADDR);
    let bits = state.arch().bits();
    match sp {
        SpSeed::Default => {}
        SpSeed::Concrete(sp) => state.set_sp(crate::symbolic::RustBV::concrete(sp as u128, bits)),
        SpSeed::Symbolic => {
            let sym = crate::symbolic::RustBV::symbolic(&state.solver().borrow(), "sym_sp", bits);
            state.set_sp(sym);
        }
    }
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
                addr: HOOK_ADDR,
                name: "find_target_test".to_string(),
                num_args: 0,
                return_addr: HOOK_RET,
            },
            deferred_forks: Vec::new(),
            last_condition: None,
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        },
        sid,
    )
}

/// Control: with no find/avoid targets the hook still dispatches natively.
#[test]
fn native_hook_outside_binary_dispatches_natively() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let outcome = dispatch_hook_with(|_| {});
        assert_eq!(outcome.counters.native_calls, 1, "native proc ran");
        match outcome.ret {
            CoreReturn::Continue(succ) => {
                assert_eq!(succ.len(), 1);
                assert_eq!(
                    succ[0].0.pc(),
                    HOOK_RET,
                    "native dispatch lands at the return address"
                );
            }
            _ => panic!("expected Continue"),
        }
    });
}

/// A hook that IS a find target must bounce to Python instead of running
/// natively — otherwise the state lands at `return_addr` and the run loop's
/// find check never sees the target address (angr-1i5h7).
#[test]
fn native_hook_at_find_addr_bounces_to_python() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let outcome = dispatch_hook_with(|mgr| mgr.set_find_addrs(vec![HOOK_ADDR]));
        assert_eq!(
            outcome.counters.native_calls, 0,
            "native proc must NOT run at a find target"
        );
        match outcome.ret {
            CoreReturn::NeedsPython(bounce) => match bounce.kind {
                BounceKind::SimProcedurePython { addr, .. } => assert_eq!(addr, HOOK_ADDR),
                other => panic!("expected SimProcedurePython bounce, got {other:?}"),
            },
            _ => panic!("expected NeedsPython bounce so the find check can fire"),
        }
    });
}

/// Same guard for avoid targets.
#[test]
fn native_hook_at_avoid_addr_bounces_to_python() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        let outcome = dispatch_hook_with(|mgr| mgr.set_avoid_addrs(vec![HOOK_ADDR]));
        assert_eq!(outcome.counters.native_calls, 0);
        assert!(matches!(
            outcome.ret,
            CoreReturn::NeedsPython(PendingBounce {
                kind: BounceKind::SimProcedurePython {
                    addr: HOOK_ADDR,
                    ..
                },
                ..
            })
        ));
    });
}

// --- angr-sqfj8.37: the native-return SP bump is gated on pops_return_addr ---

const SP_SEED: u64 = 0x7fff_0000;

/// Run the native-return path on `arch` and hand back the successor's SP.
fn native_return_sp(arch: &str) -> u64 {
    let outcome = dispatch_hook_on_arch(arch, SpSeed::Concrete(SP_SEED), |_| {});
    assert_eq!(
        outcome.counters.native_calls, 1,
        "{arch}: native proc must have run"
    );
    match outcome.ret {
        CoreReturn::Continue(succ) => {
            assert_eq!(succ.len(), 1);
            assert_eq!(succ[0].0.pc(), HOOK_RET, "{arch}: landed at return address");
            succ[0]
                .0
                .get_sp()
                .as_u64()
                .unwrap_or_else(|| panic!("{arch}: SP went symbolic"))
        }
        _ => panic!("{arch}: expected Continue"),
    }
}

/// Stack-return ABIs pop the return address, so the parallel path advances SP
/// by one pointer — the control for the link-register cases below.
#[test]
fn native_return_advances_sp_on_stack_return_abi() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        assert_eq!(native_return_sp("amd64"), SP_SEED + 8);
        assert_eq!(native_return_sp("x86"), SP_SEED + 4);
    });
}

/// ARM/ARM64/MIPS return through LR/X30/$ra: nothing was pushed, so nothing may
/// be popped. Before angr-sqfj8.37 `handle_simprocedure_core` bumped SP
/// unconditionally — only `step_one` (run_loop_single.rs) gated it — so every
/// native proc return under the wave/steady/worker engines silently dropped a
/// pointer-sized live stack slot.
#[test]
fn native_return_leaves_sp_untouched_on_link_register_abis() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        for arch in ["armel", "aarch64", "mips32", "mips64"] {
            assert_eq!(
                native_return_sp(arch),
                SP_SEED,
                "{arch} returns via a link register; SP must not move"
            );
        }
    });
}

/// A symbolic SP must survive the native-return bump (angr-c7xno.29).
///
/// `extract_args_with_abi` only reads SP when the argument count overflows the
/// register window, so a zero-arg native proc runs to completion with SP still
/// symbolic. The pre-fix `get_sp().as_u64().unwrap_or(0)` then rewrote that SP
/// to a bogus concrete `ptr_size` (8 on amd64, 4 on x86) — the shared
/// `advance_sp_past_return_addr` helper adds the pointer symbolically instead.
#[test]
fn native_return_keeps_symbolic_sp_symbolic_on_stack_return_abi() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        for arch in ["amd64", "x86"] {
            let outcome = dispatch_hook_on_arch(arch, SpSeed::Symbolic, |_| {});
            let CoreReturn::Continue(succ) = outcome.ret else {
                panic!("{arch}: expected Continue");
            };
            let sp = succ[0].0.get_sp();
            assert!(
                sp.as_u64().is_none(),
                "{arch}: symbolic SP was concretized to {:?} — the unwrap_or(0) bug",
                sp.as_u64()
            );
        }
    });
}

/// Control for the test above on the link-register ABIs: the bump is skipped
/// entirely there, so the SP register must come out as the *same* symbol it
/// went in as. (This one passed pre-fix too — it guards against the shared
/// helper regressing into an unconditional bump.)
#[test]
fn native_return_keeps_symbolic_sp_symbolic_on_link_register_abis() {
    pyo3::Python::initialize();
    pyo3::Python::attach(|_py| {
        for arch in ["armel", "aarch64", "mips32", "mips64"] {
            let outcome = dispatch_hook_on_arch(arch, SpSeed::Symbolic, |_| {});
            let CoreReturn::Continue(succ) = outcome.ret else {
                panic!("{arch}: expected Continue");
            };
            let sp = succ[0].0.get_sp();
            assert!(
                matches!(sp, crate::symbolic::RustBV::Symbolic { .. }),
                "{arch}: SP must be the untouched symbol, got {sp:?}"
            );
        }
    });
}

/// A re-enterable bounce parked in `pending_parallel_bounces` (a state living
/// in NO stash) must be recoverable into STASH_ACTIVE via
/// `flush_parked_bounces_to_active` — restoring the pc to the bounce entry so a
/// later step re-lifts the hook. The parallel found-early-return path calls
/// this so `active_count` reaches parity with the serial loop instead of
/// stranding the queue when `num_find` is hit mid-wave (angr-ph300.8).
#[test]
fn flush_parked_bounce_recovers_reenterable_state_to_active() {
    let mut mgr = RustExplorationManager::new("amd64", None).unwrap();

    // Park a re-enterable SimProcedurePython bounce whose entry addr differs
    // from the state's current pc, so we can prove the flush restored it.
    const BOUNCE_ADDR: u64 = 0x4000;
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_pc(0xdead);
    let id = state.state_id();
    mgr.pending_parallel_bounces.push((
        state,
        BounceKind::SimProcedurePython {
            addr: BOUNCE_ADDR,
            name: "sp".to_string(),
            num_args: 0,
            return_addr: 0x4010,
        },
        id, // lineage root = self
    ));

    // Stash-only view: `active_count` deliberately includes the parked bounce
    // (angr-03vl4.15), so it is `stash_count` that shows the state is in no
    // stash yet.
    assert_eq!(
        mgr.stash_count(STASH_ACTIVE),
        0,
        "parked bounce is in NO stash"
    );

    mgr.flush_parked_bounces_to_active();

    assert!(
        mgr.pending_parallel_bounces.is_empty(),
        "re-enterable bounce drained from the parked queue"
    );
    assert_eq!(mgr.active_count(), 1, "flushed back to STASH_ACTIVE");
    let active = mgr.sm.get(STASH_ACTIVE).expect("active stash exists");
    assert_eq!(active[0].state_id(), id);
    assert_eq!(
        active[0].pc(),
        BOUNCE_ADDR,
        "pc restored to the bounce entry for faithful replay"
    );
}

// --- angr-c7xno.33: handle_syscall_core, all four dispatch outcomes ---

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

// --- angr-c7xno.33: handle_symbolic_jump_target_core, both arms ---

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

// --- angr-c7xno.32: the *production* native-resume path ---
//
// `handle_native_resume_core` is what both engines actually run when the guest
// returns to the resume sentinel; `stepping.rs`'s `handle_native_resume` is a
// `#[cfg(test)]`-only twin. These drive the core through `run_post_step_core`
// so the two cannot silently drift.

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

/// A native proc's unmapped-page error becomes a Python-identical
/// SimSegfaultException message — but only with STRICT_PAGE_ACCESS on, since
/// Python otherwise lazily initializes the page and keeps going (angr-gorvf.13).
#[test]
fn segfault_message_mirrors_python_strict_page_access() {
    use super::handlers::segfault_message;
    use crate::memory::MemoryError;
    use crate::procedures::ProcedureError;

    let mut state = RustSimState::new("amd64").unwrap();
    let unmapped = ProcedureError::Memory(MemoryError::Unmapped {
        addr: 0x1234,
        size: 4096,
    });

    // STRICT_PAGE_ACCESS off: Python services the read, so we must bounce.
    assert_eq!(segfault_message(&state, &unmapped), None);

    state.set_enforce_permissions(true);
    // Page-aligned, matching PrivilegedPagingMixin's `pageno * page_size`.
    assert_eq!(
        segfault_message(&state, &unmapped),
        Some("0x1000 (unmapped)".to_string())
    );

    // Every other decline still falls back to Python.
    for err in [
        ProcedureError::SymbolicArgument("n".into()),
        ProcedureError::NotImplemented,
        ProcedureError::Memory(MemoryError::UnmappedPageInRegion { page_addr: 0x1000 }),
    ] {
        assert_eq!(segfault_message(&state, &err), None);
    }
}
