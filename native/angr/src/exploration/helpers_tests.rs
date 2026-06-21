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
        mgr.sm.get("not_unique").is_none_or(|s| s.is_empty()),
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

    mgr.pending_callback = Some(PendingCallback {
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
    });

    mgr._resume_after_error("py callback raised").expect("ok");

    let errors = mgr.get_errors();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0], (0x401000, "py callback raised".to_string(), sid));

    let errored = mgr.sm.get(STASH_ERRORED).expect("errored stash");
    assert_eq!(errored.len(), 1);
    assert_eq!(errored[0].state_id(), sid);
    // The pending callback was consumed.
    assert!(mgr.pending_callback.is_none());
}

/// Calling _resume_after_error with no pending state raises RuntimeError.
#[test]
fn resume_after_error_no_pending_state_raises() {
    Python::initialize();
    let mut mgr = RustExplorationManager::new("amd64", None).expect("amd64 mgr");
    let err = mgr
        ._resume_after_error("ignored")
        .expect_err("must raise without a pending state");
    Python::attach(|py| {
        assert!(err.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
    });
}
