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

    assert_eq!(mgr.active_count(), 0, "parked bounce is in NO stash");

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
