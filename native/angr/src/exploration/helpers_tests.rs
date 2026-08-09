// Tests for exploration/helpers.rs — extracted from the former inline
// `#[cfg(test)] mod tests` block (see rust-mod-tests-sibling-extraction).
use super::*;
use crate::exploration::{CallbackReason, PendingCallback};
use crate::stash::{STASH_ACTIVE, STASH_ERRORED};
use crate::state::RustSimState;
use crate::symbolic::{RustBV, SymContext};
use rustc_hash::FxHashMap;

// amd64 SystemV arg registers (VEX guest-state offsets): RDI, RSI, RDX, RCX,
// R8, R9. extract_procedure_args reads these before spilling to the stack.
const RDI: u32 = 72;
const RSI: u32 = 64;
const RDX: u32 = 32;
const RCX: u32 = 24;
const R8: u32 = 80;
const R9: u32 = 88;

// MIPS32 VEX guest-state offsets for $a0-$a3 (R4-R7).
const A0: u32 = 24;
const A1: u32 = 28;
const A2: u32 = 32;
const A3: u32 = 36;

/// O32 syscalls with >4 args read args 5+ from the stack at [sp+16].
/// A 6-arg MIPS32 syscall (futex/epoll_pwait/mmap2) must dispatch
/// natively when SP is concrete (angr-tvod).
#[test]
fn mips_o32_extract_syscall_args_reads_stack_window() {
    let mgr = RustExplorationManager::new("mips32", None).expect("mips32 mgr");
    let mut state = RustSimState::new("mips32").expect("mips32 state");

    // Register args $a0-$a3 = 0xa0..0xa3.
    state.set_register_by_offset(A0, RustBV::concrete(0xa0, 32));
    state.set_register_by_offset(A1, RustBV::concrete(0xa1, 32));
    state.set_register_by_offset(A2, RustBV::concrete(0xa2, 32));
    state.set_register_by_offset(A3, RustBV::concrete(0xa3, 32));

    // Stack-resident args 5,6 at [sp+16], [sp+20].
    let sp: u64 = 0x7fff_f000;
    state.map_memory(sp, 0x1000, crate::memory::Permission::RW);
    state.set_sp(RustBV::concrete(sp as u128, 32));
    state
        .memory_store(sp + 16, RustBV::concrete(0xa4, 32))
        .expect("store arg5");
    state
        .memory_store(sp + 20, RustBV::concrete(0xa5, 32))
        .expect("store arg6");

    let args = mgr.extract_syscall_args(&state, 6).expect("6 args");
    assert_eq!(args.len(), 6);
    let vals: Vec<u64> = args.iter().map(|a| a.as_u64().expect("concrete")).collect();
    assert_eq!(vals, vec![0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5]);
}

/// A symbolic SP must NOT fabricate stack args; it returns SpSymbolic so
/// the dispatcher falls through to the Python syscall callback.
#[test]
fn mips_o32_extract_syscall_args_symbolic_sp_falls_back() {
    let mgr = RustExplorationManager::new("mips32", None).expect("mips32 mgr");
    let mut state = RustSimState::new("mips32").expect("mips32 state");
    let ctx = SymContext::new();
    state.set_sp(RustBV::symbolic(&ctx, "sp", 32));

    match mgr.extract_syscall_args(&state, 6) {
        Err(ExtractionError::SpSymbolic) => {}
        other => panic!("expected SpSymbolic, got {other:?}"),
    }
}

/// ABIs that do not spill syscall args to the stack (no
/// syscall_stack_arg_offset) report RegisterOverflow, never a stack read.
#[test]
fn amd64_extract_syscall_args_overflow_is_register_overflow() {
    let mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let state = RustSimState::new("amd64").expect("amd64 state");
    // amd64 exposes 6 syscall arg registers; a 7-arg request overflows.
    match mgr.extract_syscall_args(&state, 7) {
        Err(ExtractionError::RegisterOverflow {
            requested: 7,
            available: 6,
        }) => {}
        other => panic!("expected RegisterOverflow, got {other:?}"),
    }
}

// --- extract_procedure_args (helpers.rs) ---------------------------------
// The 6-register window is covered by every native SimProc integration test;
// these exercise the stack-spill fallback and its two error branches, which
// no suite SimProc reaches (all request <= 6 args). Mirror of the
// extract_syscall_args tests above (szg45.3 gap).

/// Requesting more args than the 6-register window spills the remainder from
/// the stack at [sp + stack_arg_offset()] (= [sp + 8] on amd64, past the
/// return address).
#[test]
fn amd64_extract_procedure_args_reads_stack_window() {
    let mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let mut state = RustSimState::new("amd64").expect("amd64 state");

    // Register args RDI..R9 = 0xa0..0xa5.
    for (off, v) in [
        (RDI, 0xa0u128),
        (RSI, 0xa1),
        (RDX, 0xa2),
        (RCX, 0xa3),
        (R8, 0xa4),
        (R9, 0xa5),
    ] {
        state.set_register_by_offset(off, RustBV::concrete(v, 64));
    }

    // Stack-resident args 7,8 at [sp+8], [sp+16] (offset skips return addr).
    let sp: u64 = 0x7fff_f000;
    state.map_memory(sp, 0x1000, crate::memory::Permission::RW);
    state.set_sp(RustBV::concrete(sp as u128, 64));
    state
        .memory_store(sp + 8, RustBV::concrete(0xa6, 64))
        .expect("store arg7");
    state
        .memory_store(sp + 16, RustBV::concrete(0xa7, 64))
        .expect("store arg8");

    let args = mgr.extract_procedure_args(&state, 8).expect("8 args");
    assert_eq!(args.len(), 8);
    let vals: Vec<u64> = args.iter().map(|a| a.as_u64().expect("concrete")).collect();
    assert_eq!(vals, vec![0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7]);
}

/// A symbolic SP must NOT fabricate stack args; spilling reports SpSymbolic so
/// the dispatcher falls through to the Python SimProcedure callback.
#[test]
fn amd64_extract_procedure_args_symbolic_sp_falls_back() {
    let mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    let ctx = SymContext::new();
    state.set_sp(RustBV::symbolic(&ctx, "sp", 64));

    // 8 args -> 6 from registers, 2 would spill onto the symbolic stack.
    match mgr.extract_procedure_args(&state, 8) {
        Err(ExtractionError::SpSymbolic) => {}
        other => panic!("expected SpSymbolic, got {other:?}"),
    }
}

/// An unmapped stack slot reports StackUnmapped (with the offending arg index
/// and address) rather than fabricating a zero argument.
#[test]
fn amd64_extract_procedure_args_unmapped_stack() {
    let mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let mut state = RustSimState::new("amd64").expect("amd64 state");
    // Concrete SP, but the stack page is never mapped.
    let sp: u64 = 0x7fff_f000;
    state.set_sp(RustBV::concrete(sp as u128, 64));

    match mgr.extract_procedure_args(&state, 8) {
        Err(ExtractionError::StackUnmapped { arg_index, addr }) => {
            // First spilled arg is index 6, read from [sp + 8].
            assert_eq!(arg_index, 6);
            assert_eq!(addr, sp + 8);
        }
        other => panic!("expected StackUnmapped, got {other:?}"),
    }
}

// --- compute_register_tuple_hash (helpers.rs) ----------------------------
// The concrete branch runs during the fauxware uniqueness integration test,
// but the symbolic-sentinel disambiguation and missing-register branches are
// never asserted anywhere (szg45.3 gap).

/// States with equal concrete register values hash equal; the symbolic
/// sentinel, a concrete u64::MAX, and a missing register all hash distinctly.
#[test]
fn amd64_compute_register_tuple_hash_branches() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    mgr.constraint_tracker.uniqueness_registers = vec!["rax".to_string()];

    let mut concrete_a = RustSimState::new("amd64").expect("state a");
    concrete_a.set_register("rax", RustBV::concrete(0x1234, 64));
    let mut concrete_b = RustSimState::new("amd64").expect("state b");
    concrete_b.set_register("rax", RustBV::concrete(0x1234, 64));

    // Equal concrete values -> equal hash.
    let h_a = mgr.compute_register_tuple_hash(&concrete_a);
    let h_b = mgr.compute_register_tuple_hash(&concrete_b);
    assert_eq!(h_a, h_b, "equal concrete registers must hash equal");

    // Symbolic register vs a concrete u64::MAX register: the sentinel adds a
    // disambiguating 1u8 so they must NOT collide.
    let ctx = SymContext::new();
    let mut symbolic = RustSimState::new("amd64").expect("sym state");
    symbolic.set_register("rax", RustBV::symbolic(&ctx, "x", 64));
    let mut max_concrete = RustSimState::new("amd64").expect("max state");
    max_concrete.set_register("rax", RustBV::concrete(u64::MAX as u128, 64));

    let h_sym = mgr.compute_register_tuple_hash(&symbolic);
    let h_max = mgr.compute_register_tuple_hash(&max_concrete);
    assert_ne!(
        h_sym, h_max,
        "symbolic sentinel must not collide with concrete u64::MAX"
    );

    // A missing register hashes distinctly from both the symbolic sentinel and
    // the concrete u64::MAX cases.
    mgr.constraint_tracker.uniqueness_registers = vec!["nonexistent_reg".to_string()];
    let h_missing = mgr.compute_register_tuple_hash(&concrete_a);
    assert_ne!(
        h_missing, h_sym,
        "missing must differ from symbolic sentinel"
    );
    assert_ne!(
        h_missing, h_max,
        "missing must differ from concrete u64::MAX"
    );
}

// --- apply_uniqueness_filter (helpers.rs) --------------------------------
// Driven (but never asserted) by test_named_check_uniqueness_takes_native_path;
// these assert the duplicate actually moves to 'not_unique' (or is dropped
// when drop_terminal_states is set) while distinct states stay active
// (szg45.3 gap).

/// Push two states with identical uniqueness-register values plus one distinct
/// state; after the filter the duplicate moves to 'not_unique' and the two
/// distinct states remain active.
#[test]
fn apply_uniqueness_filter_moves_duplicate_to_not_unique() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    mgr.constraint_tracker.uniqueness_registers = vec!["rax".to_string()];

    for v in [0x10u128, 0x10, 0x20] {
        let mut s = RustSimState::new("amd64").expect("state");
        s.set_register("rax", RustBV::concrete(v, 64));
        mgr.sm.push(STASH_ACTIVE, s);
    }

    mgr.apply_uniqueness_filter();

    let active = mgr.sm.get(STASH_ACTIVE).expect("active stash");
    assert_eq!(active.len(), 2, "two distinct states stay active");
    let not_unique = mgr.sm.get("not_unique").expect("not_unique stash");
    assert_eq!(not_unique.len(), 1, "duplicate moved to not_unique");
}

/// When drop_terminal_states is enabled the duplicate is discarded entirely
/// rather than parked in 'not_unique'.
#[test]
fn apply_uniqueness_filter_drops_duplicate_when_drop_terminal() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    mgr.constraint_tracker.uniqueness_registers = vec!["rax".to_string()];
    mgr.sm.set_drop_terminal_states(true);

    for v in [0x10u128, 0x10] {
        let mut s = RustSimState::new("amd64").expect("state");
        s.set_register("rax", RustBV::concrete(v, 64));
        mgr.sm.push(STASH_ACTIVE, s);
    }

    mgr.apply_uniqueness_filter();

    let active = mgr.sm.get(STASH_ACTIVE).expect("active stash");
    assert_eq!(active.len(), 1, "one unique state remains active");
    assert!(
        mgr.sm
            .get("not_unique")
            .is_none_or(std::collections::VecDeque::is_empty),
        "duplicate dropped, not parked in not_unique"
    );
}

// --- _resume_after_error (resume.rs) -------------------------------------
// test_error_stash.py only fills the errored stash via a native NX violation;
// the callback-error recovery path here is never driven (szg45.3 high gap).

/// With a pending callback state, _resume_after_error records (pc, msg,
/// state_id) into errors and moves the state into the errored stash.
#[test]
fn resume_after_error_records_and_moves_to_errored() {
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let mut state = RustSimState::new("amd64").expect("state");
    state.set_pc(0x401000);
    let sid = state.state_id();

    mgr.pending_callbacks.insert(
        StateId::new(sid),
        PendingCallback {
            state,
            pre_callback_snapshot: None,
            reason: CallbackReason::Error {
                message: "boom".to_string(),
            },
            jumpkind: None,
            solver_ctx: None,
            deferred_forks: Vec::new(),
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        },
    );

    mgr._resume_after_error(sid, "py callback raised")
        .expect("ok");

    let errors = mgr.get_errors();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0], (0x401000, "py callback raised".to_string(), sid));

    let errored = mgr.sm.get(STASH_ERRORED).expect("errored stash");
    assert_eq!(errored.len(), 1);
    assert_eq!(errored[0].state_id(), sid);
    // The pending callback was consumed.
    assert!(mgr.pending_callbacks.is_empty());
}

/// Calling _resume_after_error with no pending state raises RuntimeError.
#[test]
fn resume_after_error_no_pending_state_raises() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let err = mgr
        ._resume_after_error(0, "ignored")
        .expect_err("must raise without a pending state");
    Python::attach(|py| {
        assert!(err.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
    });
}

/// _resume_after_error must not silently drop a deferred fork accumulated
/// earlier in the same step just because the Python callback that parked the
/// state later raised (angr-4xaga.5). The deferred fork diverged BEFORE the
/// callback, so it is unrelated to the error; dropping it makes a find target
/// behind its unexplored side permanently unreachable. Mirrors
/// `deadend_pending_callback_conservative_fork_not_dropped`: the fork here has
/// no condition source at all, so the P15 conservative arm must materialize an
/// unconstrained successor at `unexplored_target` and route it to active while
/// the parking state still lands in the errored stash.
#[test]
fn resume_after_error_deferred_fork_not_dropped() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let mut state = RustSimState::new("amd64").expect("state");
    state.set_pc(0x40_1000);
    let sid = state.state_id();

    mgr.pending_callbacks.insert(
        StateId::new(sid),
        PendingCallback {
            state,
            pre_callback_snapshot: None,
            reason: CallbackReason::Error {
                message: "hook raised".to_string(),
            },
            jumpkind: None,
            solver_ctx: None,
            // condition_id 999 absent from stored_conditions + no condition_ast
            // -> only the P15 conservative arm keeps this fork alive.
            deferred_forks: vec![crate::callbacks::DeferredFork {
                branch_addr: 0x40_0500,
                path_taken: true,
                unexplored_target: 0x40_2000,
                condition_id: 999,
                push_level: 0,
                condition_ast: None,
            }],
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        },
    );

    mgr._resume_after_error(sid, "py callback raised")
        .expect("ok");

    // The parking state itself errored, with the error recorded.
    let errored = mgr.sm.get(STASH_ERRORED).expect("errored stash");
    assert_eq!(errored.len(), 1);
    assert_eq!(errored[0].state_id(), sid);
    assert_eq!(mgr.get_errors().len(), 1);

    // The deferred fork was routed to active at the unexplored target, not
    // dropped when `pending` went out of scope.
    let active = mgr.sm.get(STASH_ACTIVE).expect("active stash");
    assert_eq!(
        active.len(),
        1,
        "conservative fork must be routed, not dropped"
    );
    assert_eq!(active[0].pc(), 0x40_2000);

    assert!(mgr.pending_callbacks.is_empty());
}

/// Forcing-function canary (angr-qwyti.4): a find-predicate pending must never
/// carry deferred forks, because `_resume_find_predicate` — unlike the terminal
/// / branch consumers — deliberately does NOT materialize them (find/avoid
/// pendings are built via `PendingCallback::lightweight`, hardcoded empty). If a
/// future refactor ever routes a fork-carrying pending here it would silently
/// prune the unexplored branch, the exact bug class of angr-4xaga.5. The `assert!`
/// in the consumer trips loudly instead; this test pins that it fires. Real
/// `assert!` (not `debug_assert!`) because the CI/ralph gate runs
/// `cargo test --release`, where debug assertions are compiled out.
#[test]
#[should_panic(expected = "lightweight callbacks must not")]
fn find_predicate_pending_with_deferred_fork_trips_canary() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let mut state = RustSimState::new("amd64").expect("state");
    state.set_pc(0x40_1000);
    let sid = state.state_id();

    // Deliberately bypass `PendingCallback::lightweight` to inject a deferred
    // fork onto a find-predicate pending — the invalid shape the canary guards.
    mgr.pending_callbacks.insert(
        StateId::new(sid),
        PendingCallback {
            state,
            pre_callback_snapshot: None,
            reason: CallbackReason::FindPredicate { addr: 0x40_1000 },
            jumpkind: None,
            solver_ctx: None,
            deferred_forks: vec![crate::callbacks::DeferredFork {
                branch_addr: 0x40_0500,
                path_taken: true,
                unexplored_target: 0x40_2000,
                condition_id: 999,
                push_level: 0,
                condition_ast: None,
            }],
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        },
    );

    // Panics: the consumer refuses to silently drop the fork.
    let _ = mgr._resume_find_predicate(sid, false);
}

/// Sibling canary to `find_predicate_pending_with_deferred_fork_trips_canary`
/// for the avoid-predicate consumer (angr-qwyti.4).
#[test]
#[should_panic(expected = "lightweight callbacks must not")]
fn avoid_predicate_pending_with_deferred_fork_trips_canary() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let mut state = RustSimState::new("amd64").expect("state");
    state.set_pc(0x40_1000);
    let sid = state.state_id();

    mgr.pending_callbacks.insert(
        StateId::new(sid),
        PendingCallback {
            state,
            pre_callback_snapshot: None,
            reason: CallbackReason::AvoidPredicate { addr: 0x40_1000 },
            jumpkind: None,
            solver_ctx: None,
            deferred_forks: vec![crate::callbacks::DeferredFork {
                branch_addr: 0x40_0500,
                path_taken: true,
                unexplored_target: 0x40_2000,
                condition_id: 999,
                push_level: 0,
                condition_ast: None,
            }],
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        },
    );

    let _ = mgr._resume_avoid_predicate(sid, false);
}

/// _deadend_pending_callback must not silently drop a deferred fork whose
/// condition is absent from `stored_conditions` and which carries no
/// `condition_ast` (angr-ph300.7). Before the P11/P15 fallback was ported to
/// the deadend path, such a fork vanished — its unexplored branch was never
/// routed, so a find target behind it was unreachable when the parking
/// callback was an exit/abort SimProcedure. Here the fork has no condition
/// source at all, so the P15 conservative-fork arm must materialize an
/// unconstrained successor at `unexplored_target`.
#[test]
fn deadend_pending_callback_conservative_fork_not_dropped() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let mut state = RustSimState::new("amd64").expect("state");
    state.set_pc(0x40_1000);
    let sid = state.state_id();

    mgr.pending_callbacks.insert(
        StateId::new(sid),
        PendingCallback {
            state,
            pre_callback_snapshot: None,
            reason: CallbackReason::Error {
                message: "exit".to_string(),
            },
            jumpkind: None,
            solver_ctx: None,
            // condition_id 999 is absent from stored_conditions and there is no
            // condition_ast -> neither the direct lookup nor P11 can supply a
            // condition, so the P15 conservative arm is the only path that
            // keeps this fork alive.
            deferred_forks: vec![crate::callbacks::DeferredFork {
                branch_addr: 0x40_0500,
                path_taken: true,
                unexplored_target: 0x40_2000,
                condition_id: 999,
                push_level: 0,
                condition_ast: None,
            }],
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        },
    );

    mgr._deadend_pending_callback(sid).expect("ok");

    // The parking state itself deadended.
    let deadended = mgr
        .sm
        .get(crate::stash::STASH_DEADENDED)
        .expect("deadended stash");
    assert_eq!(deadended.len(), 1);
    assert_eq!(deadended[0].state_id(), sid);

    // The conservative fork was routed to active at the unexplored target,
    // not dropped.
    let active = mgr.sm.get(STASH_ACTIVE).expect("active stash");
    assert_eq!(
        active.len(),
        1,
        "conservative fork must be routed, not dropped"
    );
    assert_eq!(active[0].pc(), 0x40_2000);

    assert!(mgr.pending_callbacks.is_empty());
}

/// A deferred fork resumed through _resume_after_symbolic_branch must be based
/// on the guard-free `pre_callback_snapshot`, not on `true_state` (angr-ph300.9).
/// Before the fix, deferred forks were built from `true_state`, which carries
/// both the deferred fork's own taken constraint (`x == 0`, added by
/// apply_deferred_fork_constraints) and the branch guard (`assume_true`). Its
/// unexplored side `x != 0` therefore contradicted the base solver, came back
/// UNSAT, and a genuinely reachable path was pruned — while
/// pre_callback_snapshot (which predates both) was dropped unused. The fix
/// mirrors _resume_after_simprocedure and forks the deferred branch from that
/// snapshot. With no per-condition BranchSnapshot in `fork_snapshots`,
/// build_unexplored_fork forks the base directly, so the base's cleanliness is
/// what keeps the unexplored side satisfiable.
#[test]
fn resume_symbolic_branch_deferred_fork_uses_pre_callback_snapshot() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let mut state = RustSimState::new("amd64").expect("state");
    state.set_pc(0x40_1000);
    let sid = state.state_id();

    // Build the guard/fork conditions in the state's own solver context so the
    // forked solvers (clones of it) recognize the `x` symbol.
    let mut stored = FxHashMap::default();
    {
        let ctx_ref = state.solver().borrow();
        let ctx: &SymContext = &ctx_ref;
        let x = RustBV::symbolic(ctx, "x", 64);
        let zero = RustBV::zero(64);
        stored.insert(1u64, x.eq(&zero, ctx)); // branch B guard: x == 0
        stored.insert(2u64, x.eq(&zero, ctx)); // deferred fork A condition
    }

    // Snapshot captured here — before apply_deferred_fork_constraints and the
    // branch guard run inside _resume — so it holds no constraint on `x`. The
    // `x` symbol already exists in this context (created above), so the clone
    // recognizes it.
    let snapshot = state.fork();

    mgr.pending_callbacks.insert(
        StateId::new(sid),
        PendingCallback {
            state,
            pre_callback_snapshot: Some(snapshot),
            reason: CallbackReason::SymbolicBranch {
                condition_id: 1,
                true_target: 0x40_1100,
                false_target: 0x40_1200,
            },
            jumpkind: None,
            solver_ctx: None,
            deferred_forks: vec![crate::callbacks::DeferredFork {
                branch_addr: 0x40_0500,
                path_taken: true, // unexplored side is x != 0
                unexplored_target: 0x40_2000,
                condition_id: 2,
                push_level: 0,
                condition_ast: None,
            }],
            stored_conditions: stored,
            fork_snapshots: FxHashMap::default(),
        },
    );

    Python::attach(|py| {
        mgr._resume_after_symbolic_branch(py, sid, 0x40_1100, 0x40_1200, None, None)
            .expect("resume ok");
    });

    // The deferred fork (x != 0) is satisfiable on a guard-free base, so it must
    // be routed to active at its unexplored target rather than pruned as UNSAT.
    let active = mgr.sm.get(STASH_ACTIVE).expect("active stash");
    assert!(
        active.iter().any(|s| s.pc() == 0x40_2000),
        "deferred fork must survive: it inherited the branch guard and was pruned"
    );
}

// --- pending_callbacks keying (angr-1ilq.4) ------------------------------
// The single-slot `pending_callback: Option<_>` became a `state_id`-keyed
// `FxHashMap<StateId, PendingCallback>`. These assert the keying is genuine:
// a resume addressed by the wrong/absent id must NOT consume some other
// pending state, and two pending entries under distinct ids are resolved /
// removed independently with no crosstalk.

/// Build a minimal Error-reason PendingCallback for an amd64 state at `pc`.
/// Cheap to construct (no solver fork / deferred forks), so multi-pending
/// isolation can be exercised purely at the Rust unit-test level.
#[cfg(test)]
fn make_pending_at(pc: u64) -> (u64, PendingCallback) {
    let mut state = RustSimState::new("amd64").expect("state");
    state.set_pc(pc);
    let sid = state.state_id();
    let pending = PendingCallback {
        state,
        pre_callback_snapshot: None,
        reason: CallbackReason::Error {
            message: "pending".to_string(),
        },
        jumpkind: None,
        solver_ctx: None,
        deferred_forks: Vec::new(),
        stored_conditions: FxHashMap::default(),
        fork_snapshots: FxHashMap::default(),
    };
    (sid, pending)
}

/// Resuming with the wrong/absent state_id returns the "no pending callback"
/// error and leaves the genuine pending entry untouched; resuming with the
/// correct id then succeeds and consumes it.
#[test]
fn resume_wrong_state_id_errors_correct_succeeds() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let (sid, pending) = make_pending_at(0x401000);
    mgr.pending_callbacks.insert(StateId::new(sid), pending);

    // Wrong id (sid + 1 is guaranteed absent): must error, must NOT consume sid.
    let wrong = sid.wrapping_add(1);
    let err = mgr
        ._resume_after_error(wrong, "ignored")
        .expect_err("wrong state_id must raise");
    Python::attach(|py| {
        assert!(err.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
    });
    assert!(
        mgr.pending_callbacks.contains_key(&StateId::new(sid)),
        "genuine pending entry must survive a mis-addressed resume"
    );

    // Correct id: succeeds and consumes exactly that entry.
    mgr._resume_after_error(sid, "real error").expect("ok");
    assert!(
        !mgr.pending_callbacks.contains_key(&StateId::new(sid)),
        "correct resume must remove the entry"
    );
    assert!(mgr.pending_callbacks.is_empty());
}

/// Two pending entries under distinct StateIds are resumed/removed
/// independently: resuming one leaves the other in place, and each routes its
/// own state into the errored stash.
#[test]
fn multi_pending_entries_resume_independently() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");

    let (sid_a, pending_a) = make_pending_at(0x401000);
    let (sid_b, pending_b) = make_pending_at(0x402000);
    assert_ne!(sid_a, sid_b, "distinct states must have distinct ids");
    mgr.pending_callbacks.insert(StateId::new(sid_a), pending_a);
    mgr.pending_callbacks.insert(StateId::new(sid_b), pending_b);
    assert_eq!(mgr.pending_callbacks.len(), 2);

    // Resume A: only A is consumed; B remains live.
    mgr._resume_after_error(sid_a, "err a").expect("resume a");
    assert!(!mgr.pending_callbacks.contains_key(&StateId::new(sid_a)));
    assert!(mgr.pending_callbacks.contains_key(&StateId::new(sid_b)));

    // Resume B: now empty.
    mgr._resume_after_error(sid_b, "err b").expect("resume b");
    assert!(mgr.pending_callbacks.is_empty());

    // Both states landed in the errored stash, with their own ids.
    let errored = mgr.sm.get(STASH_ERRORED).expect("errored stash");
    let ids: Vec<u64> = errored
        .iter()
        .map(crate::state::RustSimState::state_id)
        .collect();
    assert!(ids.contains(&sid_a) && ids.contains(&sid_b));
}

/// angr-pwu71: `fold_scheduler_dispatch_stats` must land the worker-summarized
/// `Avoided` split on `sm.avoided_count`, alongside the three dispositions
/// angr-op0dn.13.15 already folded. Without it, a fork successor whose pc hits
/// an avoid address inside a worker is invisible to `stats()["avoided_count"]`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn fold_scheduler_dispatch_stats_folds_avoided_split() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let stats = crate::exploration::scheduler::SchedulerStats {
        summarized_terminals: 10,
        summarized_deadended: 4,
        summarized_errored: 2,
        summarized_pruned: 1,
        summarized_avoided: 3,
        ..Default::default()
    };
    mgr.fold_scheduler_dispatch_stats(&stats);
    assert_eq!(mgr.sm.deadended_count, 4);
    assert_eq!(mgr.sm.errored_count, 2);
    assert_eq!(mgr.sm.pruned_count, 1);
    assert_eq!(
        mgr.sm.avoided_count, 3,
        "worker-avoided terminals must reach avoided_count for worker-count invariance",
    );
}

// ---------------------------------------------------------------------------
// record_migration_sample — the modelled work-stealing scheduler (angr-panhl.1,
// angr-9ke6b.79). Counters only, so the whole model is exercised directly:
// push N-1 states onto the active stash, seed sticky homes in
// `parallel_worker_of`, and hand the Nth id in as the just-dispatched state.
// ---------------------------------------------------------------------------

/// Build a manager plus `n` fresh amd64 states, pushing all but the LAST onto
/// the active stash. Returns the manager and every state id in push order —
/// the final id is the caller's `stepped` argument (already popped from the
/// stash by the time the real call site samples).
fn migration_fixture(workers: usize, n: usize) -> (RustExplorationManager, Vec<u64>) {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    mgr.parallel_num_workers = workers;
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let state = RustSimState::new("amd64").expect("state");
        ids.push(state.state_id());
        if i + 1 < n {
            mgr.sm.push(STASH_ACTIVE, state);
        }
    }
    (mgr, ids)
}

/// A steal needs BOTH halves of the condition: the dispatched state's home
/// worker still holds a backlog (>=2 queued, including this task) AND some
/// other worker is idle. Two states co-homed on worker 0 of two satisfy both.
#[test]
fn migration_sample_counts_steal_when_home_backlogged_and_peer_idle() {
    let (mut mgr, ids) = migration_fixture(2, 2);
    let (other, stepped) = (ids[0], ids[1]);
    mgr.parallel_worker_of.insert(other, 0);
    mgr.parallel_worker_of.insert(stepped, 0);

    mgr.record_migration_sample(stepped);

    assert_eq!(mgr.parallel_migrations, 1, "backlogged home + idle peer");
    assert_eq!(mgr.parallel_tasks, 1, "one dispatch == one task");
    assert_eq!(mgr.parallel_max_active_width, 2, "stepped counts in width");
    assert_eq!(
        mgr.parallel_width_hist,
        [0, 1, 0, 0, 0],
        "width 2 lands in the ==2 bucket",
    );
}

/// Boundary: `load[home] == 1` is NOT a steal even when an idle worker exists —
/// there is no backlog to hand off. Same shape as the test above but with the
/// two states split across workers of a three-worker pool, so worker 2 is idle.
#[test]
fn migration_sample_no_steal_at_load_one_boundary() {
    let (mut mgr, ids) = migration_fixture(3, 2);
    let (other, stepped) = (ids[0], ids[1]);
    mgr.parallel_worker_of.insert(other, 1);
    mgr.parallel_worker_of.insert(stepped, 0);

    mgr.record_migration_sample(stepped);

    assert_eq!(
        mgr.parallel_migrations, 0,
        "load[home]==1 has nothing to steal, idle peer notwithstanding",
    );
}

/// The `>=2` side of the same boundary: add a third state co-homed with the
/// dispatched one so `load[home] == 2`, holding the idle worker fixed.
#[test]
fn migration_sample_counts_steal_at_load_two_boundary() {
    let (mut mgr, ids) = migration_fixture(3, 3);
    let (a, b, stepped) = (ids[0], ids[1], ids[2]);
    mgr.parallel_worker_of.insert(a, 1);
    mgr.parallel_worker_of.insert(b, 0);
    mgr.parallel_worker_of.insert(stepped, 0);

    mgr.record_migration_sample(stepped);

    assert_eq!(mgr.parallel_migrations, 1, "load[home]==2 crosses the sill");
}

/// Homes are sticky across steps for states that are still schedulable, and a
/// state with no home is placed on the least-loaded worker (ties go to the
/// lowest index). Here a/b/stepped pile onto worker 0, leaving load [3, 0, 0],
/// so the homeless `fresh` must land on worker 1.
#[test]
fn migration_sample_keeps_sticky_homes_and_places_new_state_least_loaded() {
    let (mut mgr, ids) = migration_fixture(3, 4);
    let (a, b, fresh, stepped) = (ids[0], ids[1], ids[2], ids[3]);
    mgr.parallel_worker_of.insert(a, 0);
    mgr.parallel_worker_of.insert(b, 0);
    mgr.parallel_worker_of.insert(stepped, 0);

    mgr.record_migration_sample(stepped);

    assert_eq!(mgr.parallel_worker_of.get(&a), Some(&0), "sticky");
    assert_eq!(mgr.parallel_worker_of.get(&b), Some(&0), "sticky");
    assert_eq!(mgr.parallel_worker_of.get(&stepped), Some(&0), "sticky");
    assert_eq!(
        mgr.parallel_worker_of.get(&fresh),
        Some(&1),
        "homeless state goes to the least-loaded worker, not worker 0",
    );
    assert_eq!(mgr.parallel_migrations, 1, "load[0]==3 with worker 2 idle");
}

/// A home recorded against a worker index that no longer exists (the modelled
/// pool shrank) is dropped rather than indexing past `load`, and the state is
/// re-placed by the least-loaded rule.
#[test]
fn migration_sample_drops_home_beyond_worker_count() {
    let (mut mgr, ids) = migration_fixture(2, 2);
    let (stale, stepped) = (ids[0], ids[1]);
    mgr.parallel_worker_of.insert(stale, 7);
    mgr.parallel_worker_of.insert(stepped, 1);

    mgr.record_migration_sample(stepped);

    assert_eq!(
        mgr.parallel_worker_of.get(&stale),
        Some(&0),
        "out-of-range home re-placed on the idle worker",
    );
    assert_eq!(mgr.parallel_worker_of.get(&stepped), Some(&1), "sticky");
    assert_eq!(
        mgr.parallel_migrations, 0,
        "one task per worker, no backlog"
    );
}

/// The width histogram is bucketed BEFORE the `<2 schedulable states` early
/// return, so a narrow single-state step is still visible in the series (that
/// ordering is the whole point of the width audit distinguishing width-1 steps).
#[test]
fn migration_sample_buckets_width_before_early_return() {
    let (mut mgr, ids) = migration_fixture(4, 1);

    mgr.record_migration_sample(ids[0]);

    assert_eq!(
        mgr.parallel_width_hist,
        [1, 0, 0, 0, 0],
        "width-1 step still counted despite the early return",
    );
    assert_eq!(mgr.parallel_tasks, 1);
    assert_eq!(mgr.parallel_migrations, 0);
    assert!(
        mgr.parallel_worker_of.is_empty(),
        "early return leaves the home map untouched",
    );
}

// --- angr-c7xno.29: advance_sp_past_return_addr ---

/// The concrete happy path: one pointer popped off the stack.
#[test]
fn advance_sp_past_return_addr_bumps_concrete_sp() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_sp(RustBV::concrete(0x7fff_0000u128, 64));

    advance_sp_past_return_addr(&mut state, true);

    assert_eq!(state.get_sp().as_u64(), Some(0x7fff_0008));
}

/// Link-register ABIs never pushed a return address, so nothing is popped.
#[test]
fn advance_sp_past_return_addr_is_noop_when_abi_does_not_pop() {
    let mut state = RustSimState::new("amd64").unwrap();
    state.set_sp(RustBV::concrete(0x7fff_0000u128, 64));

    advance_sp_past_return_addr(&mut state, false);

    assert_eq!(state.get_sp().as_u64(), Some(0x7fff_0000));
}

/// The regression this helper exists for: the two native-return sites used to
/// do `get_sp().as_u64().unwrap_or(0)`, rewriting a symbolic SP to the bogus
/// concrete value `ptr_size`. The bump must stay symbolic instead.
#[test]
fn advance_sp_past_return_addr_keeps_symbolic_sp_symbolic() {
    let mut state = RustSimState::new("amd64").unwrap();
    let sym = RustBV::symbolic(&state.solver().borrow(), "sym_sp", 64);
    state.set_sp(sym);

    advance_sp_past_return_addr(&mut state, true);

    let sp = state.get_sp();
    assert_eq!(
        sp.as_u64(),
        None,
        "symbolic SP must not collapse to a concrete value"
    );
    assert_eq!(sp.width(), 64, "the bump preserves SP width");
}
