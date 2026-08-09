// Unit tests for interpreter/execution.rs (VEXInterpreter block exec / memory / cache).
// Split out per rust-mod-tests-sibling-extraction; included via #[cfg(test)] #[path].
use super::*;
use crate::callbacks::{ErrorRoute, RunErrorKind};

fn new_interp(ctx: &SymContext) -> VEXInterpreter<'_> {
    VEXInterpreter::new(VexArch::AMD64, ctx)
}

fn make_irsb(addr: u64, len_bytes: u32) -> IRSB {
    let mut irsb = IRSB::new(addr, VexArch::AMD64);
    irsb.statements.push(IRStmt::IMark {
        addr,
        len: len_bytes,
        delta: 0,
    });
    irsb
}

#[test]
fn sort_concrete_memory_orders_by_base() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.add_concrete_memory(0x3000, vec![0u8; 0x10]);
    interp.add_concrete_memory(0x1000, vec![0u8; 0x10]);
    interp.add_concrete_memory(0x2000, vec![0u8; 0x10]);
    assert!(!interp.concrete_memory_sorted);
    interp.sort_concrete_memory();
    assert!(interp.concrete_memory_sorted);
    let bases: Vec<u64> = interp.concrete_memory.iter().map(|r| r.base).collect();
    assert_eq!(bases, vec![0x1000, 0x2000, 0x3000]);
}

#[test]
fn sort_concrete_memory_is_idempotent() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.add_concrete_memory(0x1000, vec![0u8; 0x10]);
    interp.add_concrete_memory(0x2000, vec![0u8; 0x10]);
    interp.sort_concrete_memory();
    // Second call is a no-op (already sorted) but should not panic or change order.
    interp.sort_concrete_memory();
    assert_eq!(interp.concrete_memory[0].base, 0x1000);
    assert_eq!(interp.concrete_memory[1].base, 0x2000);
}

#[test]
fn sort_concrete_memory_noop_below_two_regions() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.add_concrete_memory(0x4000, vec![0u8; 0x10]);
    interp.sort_concrete_memory();
    // Single-region case does not flip the sorted flag.
    assert!(!interp.concrete_memory_sorted);
}

#[test]
fn block_cache_round_trip() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb(0x4000, 4);
    assert!(!interp.has_cached_block(0x4000));
    interp.cache_block(0x4000, irsb);
    assert!(interp.has_cached_block(0x4000));
    let got = interp.get_cached_block(0x4000).expect("cached IRSB");
    assert_eq!(got.addr, 0x4000);
}

#[test]
fn pop_block_solver_if_pushed_is_no_op_when_not_pushed() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    assert!(!interp.block_solver_pushed);
    // Should not panic / not double-pop.
    interp.pop_block_solver_if_pushed();
    assert!(!interp.block_solver_pushed);
}

#[test]
fn next_cond_id_is_monotonic() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let a = interp.next_cond_id();
    let b = interp.next_cond_id();
    let c = interp.next_cond_id();
    assert_eq!(a, 0);
    assert_eq!(b, 1);
    assert_eq!(c, 2);
}

#[test]
fn deferred_forks_take_clears_state() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    assert_eq!(interp.num_deferred_forks(), 0);
    // The take_/clear_ APIs should not panic on empty state.
    let taken = interp.take_deferred_forks();
    assert!(taken.is_empty());
    interp.clear_deferred_forks();
    assert_eq!(interp.num_deferred_forks(), 0);
}

#[test]
fn stored_conditions_take_and_lookup() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    // Insert a stored condition via the private map; verify take/get behaviour.
    let cond = RustBV::symbolic(&ctx, "cond", 1);
    interp.stored_conditions.insert(42, cond);
    assert!(interp.get_stored_condition(42).is_some());
    let taken = interp.take_stored_conditions();
    assert_eq!(taken.len(), 1);
    // After take, the map should be empty.
    assert!(interp.get_stored_condition(42).is_none());
}

#[test]
fn take_last_branch_condition_returns_none_initially() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    assert!(interp.take_last_branch_condition().is_none());
}

#[test]
fn take_last_branch_condition_returns_then_clears() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let cond = RustBV::symbolic(&ctx, "b", 1);
    interp.last_branch_condition = Some(cond);
    assert!(interp.take_last_branch_condition().is_some());
    assert!(interp.take_last_branch_condition().is_none());
}

#[test]
fn swap_block_cache_exchanges_caches() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.cache_block(0x4000, make_irsb(0x4000, 4));
    // Swap in an empty cache — should get back the populated one.
    let old = interp.swap_block_cache(LruCache::unbounded());
    assert!(old.contains(&0x4000));
    assert!(!interp.has_cached_block(0x4000));
}

// angr-zzju9: CbExecutionError::run_error_kind classifies the unliftable
// LiftError sentinel as a graceful Deadend, and every other Panic-strategy
// variant — crucially InvalidIR, which a genuinely malformed IRSB maps to —
// as Fatal (errored stash). This locks the typed routing that replaced the
// message-substring matching in exploration/stepping.rs.
#[test]
fn run_error_kind_lift_error_is_deadend() {
    let e = CbExecutionError::LiftError("unliftable block: empty IRSB sentinel".to_string());
    assert_eq!(e.run_error_kind(), RunErrorKind::Deadend);
    // Deadend errors still carry the Panic strategy (no Python recovery).
    assert_eq!(e.strategy(), FallbackStrategy::Panic);
}

#[test]
fn run_error_kind_invalid_ir_is_fatal() {
    let e = CbExecutionError::InvalidIR("IRSB deserialization failed: bad json".to_string());
    assert_eq!(e.run_error_kind(), RunErrorKind::Fatal);
}

#[test]
fn run_error_kind_other_panic_variants_are_fatal() {
    for e in [
        CbExecutionError::Memory(MemoryError::Unmapped { addr: 0, size: 0 }),
        CbExecutionError::UnknownTemp(3),
        CbExecutionError::Callback("py raised".to_string()),
    ] {
        assert_eq!(e.run_error_kind(), RunErrorKind::Fatal);
    }
}

// angr-c7xno.30: `run_error_kind`'s doc used to read as if Fatal meant the
// errored stash unconditionally, but `RunErrorKind::route` downgrades a Fatal
// at pc 0 to a deadend for *every* Fatal-producing variant, not just the
// lift-adjacent ones. Compose the two so the end-to-end classification is
// pinned from a real `CbExecutionError`, non-LiftError included.
#[test]
fn fatal_variants_at_null_pc_route_to_null_address_deadend() {
    for e in [
        CbExecutionError::InvalidIR("IRSB deserialization failed: bad json".to_string()),
        CbExecutionError::Memory(MemoryError::Unmapped { addr: 0, size: 0 }),
        CbExecutionError::UnknownTemp(3),
        CbExecutionError::Callback("py raised".to_string()),
    ] {
        let kind = e.run_error_kind();
        assert_eq!(kind, RunErrorKind::Fatal);
        assert_eq!(kind.route(0), ErrorRoute::NullAddressDeadend);
        // Same variant at a real pc is still an errored state.
        assert_eq!(kind.route(0x40_1000), ErrorRoute::Errored);
    }
    // A LiftError deadends at pc 0 too, but by the *other* route — the two
    // must stay distinguishable.
    assert_eq!(
        CbExecutionError::LiftError("empty IRSB sentinel".to_string())
            .run_error_kind()
            .route(0),
        ErrorRoute::UnliftableDeadend
    );
}

// angr-sqfj8.111: a block whose default exit is a VEX trap (Ijk_Sig*,
// Ijk_Privileged) must stop the run with a Fatal error — Python's
// engines/failure.py raises AngrExitError on such a successor, landing it in
// the errored stash. Before the fix `parse_jumpkind` had no variants for
// these tags, so they arrived as Boring and the interpreter kept executing
// straight through the trap.
#[test]
fn trap_jumpkind_block_end_errors_instead_of_continuing() {
    for (jk, tag) in [
        (JumpKind::SigTRAP, "Ijk_SigTRAP"),
        (JumpKind::SigILL, "Ijk_SigILL"),
        (JumpKind::SigSEGV, "Ijk_SigSEGV"),
        (JumpKind::SigFPE_IntDiv, "Ijk_SigFPE_IntDiv"),
        (JumpKind::Privileged, "Ijk_Privileged"),
    ] {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.add_concrete_memory(0x1000, vec![0u8; 0x1000]);
        let mut irsb = make_irsb(0x1000, 4);
        irsb.next = IRExpr::Const(IRConst::U64(0x1004));
        irsb.jumpkind = jk;
        interp.cache_block(0x1000, irsb);
        interp.set_pc(0x1000);

        let callbacks = PythonCallbacks::new();
        let (result, _blocks, _forks) =
            interp.run_until_event(&callbacks, 4, &std::collections::HashSet::new(), false);
        match result {
            RunResult::Error {
                message,
                addr,
                kind,
            } => {
                assert_eq!(addr, 0x1004, "{tag} should error at the trap target");
                assert_eq!(
                    kind,
                    RunErrorKind::Fatal,
                    "{tag} must reach the errored stash"
                );
                assert!(
                    message.contains(tag),
                    "{tag} missing from message: {message}"
                );
            }
            other => panic!("expected Error for {tag}, got {other:?}"),
        }
        // PC points at the trap target, matching the Python successor.
        assert_eq!(interp.get_pc(), 0x1004);
    }
}

// The converse: Ijk_NoRedir is an ordinary jump with a Valgrind translation
// hint, so it must keep executing rather than tripping the trap check above.
#[test]
fn noredir_jumpkind_is_not_treated_as_a_trap() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.add_concrete_memory(0x1000, vec![0u8; 0x1000]);
    let mut irsb = make_irsb(0x1000, 4);
    irsb.next = IRExpr::Const(IRConst::U64(0x1004));
    irsb.jumpkind = JumpKind::NoRedir;
    interp.cache_block(0x1000, irsb);
    interp.set_pc(0x1000);

    let callbacks = PythonCallbacks::new();
    let (result, _blocks, _forks) =
        interp.run_until_event(&callbacks, 1, &std::collections::HashSet::new(), false);
    assert!(
        !matches!(result, RunResult::Error { .. }),
        "Ijk_NoRedir must not error, got {result:?}"
    );
}
