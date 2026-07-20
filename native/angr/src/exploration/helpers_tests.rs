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

/// `reconstruct_deferred_fork_condition` (the shared P11 helper extracted in
/// angr-ph300.76) short-circuits without ever touching Python in its two
/// non-reconstruction branches: when the condition is already present in
/// `stored_conditions`, and when the fork carries no `condition_ast`. Both
/// return `None` (nothing was reconstructed), leaving `condition.or(...)`
/// intact at the call sites.
#[test]
fn reconstruct_deferred_fork_condition_early_returns_none() {
    let fork_base = RustSimState::new("amd64").expect("base state");

    let ctx = SymContext::new();
    let stored = RustBV::symbolic(&ctx, "c", 1);
    let fork_with_ast = crate::callbacks::DeferredFork {
        branch_addr: 0x40_0500,
        path_taken: true,
        unexplored_target: 0x40_2000,
        condition_id: 7,
        push_level: 0,
        condition_ast: None,
    };

    // Branch 1: stored condition already present -> nothing to reconstruct,
    // even if an AST were also present. No Python attach happens.
    assert!(
        reconstruct_deferred_fork_condition(Some(&stored), &fork_with_ast, &fork_base).is_none(),
        "present stored condition must short-circuit to None"
    );

    // Branch 2: no stored condition and no condition_ast -> P11 cannot supply
    // a condition, so the P15 conservative arm must take over at the call site.
    assert!(
        reconstruct_deferred_fork_condition(None, &fork_with_ast, &fork_base).is_none(),
        "absent condition_ast must yield None"
    );
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

// --- MergePoint native technique (helpers.rs::apply_merge_point) ----------
// ManualMergepoint parity (angr-op0dn.11.5). These drive apply_native_techniques
// directly, constructing states at the merge address (with per-state callstacks
// via push_call_frame) so the group/merge/release logic is asserted without a
// full binary.

/// Push a fresh amd64 state at `pc` with a synthetic single-frame callstack
/// whose return address is `ret` (the merge grouping key), then park it active.
fn push_active_at(mgr: &mut RustExplorationManager, pc: u64, ret: u64) -> u64 {
    let mut s = RustSimState::new("amd64").expect("state");
    s.set_pc(pc);
    // return_addr is the field merge_waiters_by_callstack keys on.
    s.push_call(0xdead, 0xbeef, ret, 0x7fff_0000);
    let sid = s.state_id();
    mgr.sm.push(STASH_ACTIVE, s);
    sid
}

/// Three same-callstack states at the merge address (active otherwise empty)
/// collapse to one merged state; states_merged_native counts all three.
#[test]
fn merge_point_merges_same_callstack_when_active_drains() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    for _ in 0..3 {
        push_active_at(&mut mgr, 0x1000, 0xAAAA);
    }
    mgr.register_merge_point(0x1000, 10);
    assert_eq!(mgr.native_technique_count(), 1);

    mgr.apply_native_techniques();

    let active = mgr.sm.get(STASH_ACTIVE).expect("active");
    assert_eq!(active.len(), 1, "3 same-callstack waiters merge to 1");
    assert_eq!(active[0].pc(), 0x1000, "merged state sits at the merge pc");
    let wait = mgr.sm.get("merge_waiting_0x1000").expect("wait stash");
    assert!(wait.is_empty(), "waiters consumed by the merge");
    assert_eq!(mgr.states_merged_native, 3);
}

/// Waiters with two distinct callstacks form two groups -> two merged states.
#[test]
fn merge_point_groups_by_callstack() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    // Two per callstack so each group actually merges (>=2).
    push_active_at(&mut mgr, 0x2000, 0xA);
    push_active_at(&mut mgr, 0x2000, 0xA);
    push_active_at(&mut mgr, 0x2000, 0xB);
    push_active_at(&mut mgr, 0x2000, 0xB);
    mgr.register_merge_point(0x2000, 10);

    mgr.apply_native_techniques();

    let active = mgr.sm.get(STASH_ACTIVE).expect("active");
    assert_eq!(active.len(), 2, "two callstack groups -> two merged states");
    assert!(mgr.sm.get("merge_waiting_0x2000").unwrap().is_empty());
    assert_eq!(mgr.states_merged_native, 4);
}

/// A lone waiter is released back to active unmerged: count preserved, counter
/// untouched.
#[test]
fn merge_point_single_waiter_released_unmerged() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let sid = push_active_at(&mut mgr, 0x3000, 0xC);
    mgr.register_merge_point(0x3000, 10);

    mgr.apply_native_techniques();

    let active = mgr.sm.get(STASH_ACTIVE).expect("active");
    assert_eq!(active.len(), 1, "lone waiter released back to active");
    assert_eq!(active[0].state_id(), sid, "same state, not a merge product");
    assert!(mgr.sm.get("merge_waiting_0x3000").unwrap().is_empty());
    assert_eq!(mgr.states_merged_native, 0, "nothing merged");
}

/// While the active frontier is non-empty and under the round limit the merge
/// is deferred; once `wait_counter_limit` post-step rounds elapse it fires even
/// with a live non-merge state still active.
#[test]
fn merge_point_defers_then_fires_on_counter() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    push_active_at(&mut mgr, 0x4000, 0xD);
    push_active_at(&mut mgr, 0x4000, 0xD);
    // A live non-merge state keeps the frontier non-empty across rounds.
    let live = push_active_at(&mut mgr, 0x5000, 0xE);
    mgr.register_merge_point(0x4000, 2);

    // Round 1: two waiters parked, but active still holds the live state and
    // counter (1) < limit (2) -> no merge yet.
    mgr.apply_native_techniques();
    assert_eq!(mgr.sm.get("merge_waiting_0x4000").unwrap().len(), 2);
    assert_eq!(mgr.states_merged_native, 0, "merge deferred round 1");
    assert_eq!(
        mgr.sm.get(STASH_ACTIVE).unwrap().len(),
        1,
        "live state stays"
    );

    // Round 2: nothing new arrives, counter (2) reaches the limit -> merge
    // fires even though the live state is still active.
    mgr.apply_native_techniques();
    assert_eq!(mgr.states_merged_native, 2, "counter forced the merge");
    assert!(mgr.sm.get("merge_waiting_0x4000").unwrap().is_empty());
    let active_ids = mgr.sm.state_ids(STASH_ACTIVE);
    assert!(active_ids.contains(&live), "live state untouched");
    assert_eq!(active_ids.len(), 2, "live state + merged product");
}
