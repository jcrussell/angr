//! Tests for [`super::super::exits`] — VEX exit/jump dispatch logic.
//!
//! Extracted from the `mod tests` block in `exits.rs` (see
//! `rust-mod-tests-sibling-extraction` bd memory).

use super::*;
use crate::state::CallStackEntry;
use crate::vex::ir::IRType;

fn new_interp(ctx: &SymContext) -> VEXInterpreter<'_> {
    VEXInterpreter::new(VexArch::AMD64, ctx)
}

fn add_internal_region(interp: &mut VEXInterpreter<'_>) {
    // Define a "binary" region so is_in_binary returns true for 0x1000..0x2000.
    interp.add_concrete_memory(0x1000, vec![0u8; 0x1000]);
}

#[test]
fn handle_exit_updates_pc_on_boring_internal_jump() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    add_internal_region(&mut interp);
    let result = interp.handle_exit(0x1500, JumpKind::Boring);
    assert_eq!(interp.get_pc(), 0x1500);
    matches!(
        result,
        BlockResult::BlockEnd {
            next_addr: 0x1500,
            jumpkind: JumpKind::Boring
        }
    );
}

#[test]
fn handle_exit_syscall_returns_syscall_variant() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    add_internal_region(&mut interp);
    // Place syscall number into RAX via register file's put_reg helper.
    interp.registers.put_reg("rax", RustBV::concrete(60, 64));
    let result = interp.handle_exit(0x1234, JumpKind::Sys_syscall);
    match result {
        BlockResult::Syscall { num } => assert_eq!(num, Some(60)),
        other => panic!("expected Syscall, got {:?}", other),
    }
    // PC is updated even for syscalls.
    assert_eq!(interp.get_pc(), 0x1234);
}

#[test]
fn handle_exit_hooked_address_returns_hook() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    add_internal_region(&mut interp);
    interp.add_hook(0x1500);
    match interp.handle_exit(0x1500, JumpKind::Boring) {
        BlockResult::Hook { addr } => assert_eq!(addr, 0x1500),
        other => panic!("expected Hook, got {:?}", other),
    }
}

#[test]
fn handle_exit_call_to_external_returns_unmodeled_call() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    add_internal_region(&mut interp);
    // 0x9000 is not in [0x1000, 0x2000) — external.
    match interp.handle_exit(0x9000, JumpKind::Call) {
        BlockResult::UnmodeledCall {
            addr, symbol_name, ..
        } => {
            assert_eq!(addr, 0x9000);
            // Calls don't get __extern_addr__ tag — that's the post-call path.
            assert_eq!(symbol_name, None);
        }
        other => panic!("expected UnmodeledCall, got {:?}", other),
    }
}

#[test]
fn handle_exit_ret_with_empty_call_stack_returns_unconstrained() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    add_internal_region(&mut interp);
    assert!(interp.call_stack.is_empty());
    // External target + Ret + empty stack -> UnconstrainedJump (see angr-3uye).
    match interp.handle_exit(0x9000, JumpKind::Ret) {
        BlockResult::UnconstrainedJump {
            min_target,
            max_target,
            jumpkind,
            ..
        } => {
            assert_eq!(min_target, 0x9000);
            assert_eq!(max_target, 0x9000);
            assert!(jumpkind.is_ret());
        }
        other => panic!("expected UnconstrainedJump, got {:?}", other),
    }
}

#[test]
fn handle_exit_ret_to_internal_address_is_block_end() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    add_internal_region(&mut interp);
    // Returning to an internal address is a normal block end even with empty stack.
    match interp.handle_exit(0x1800, JumpKind::Ret) {
        BlockResult::BlockEnd {
            next_addr,
            jumpkind,
        } => {
            assert_eq!(next_addr, 0x1800);
            assert!(jumpkind.is_ret());
        }
        other => panic!("expected BlockEnd, got {:?}", other),
    }
}

#[test]
fn handle_exit_external_non_call_non_ret_returns_unmodeled_call() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    add_internal_region(&mut interp);
    // A jump (Boring) to a non-hooked external address is tagged __extern_addr__.
    match interp.handle_exit(0x9000, JumpKind::Boring) {
        BlockResult::UnmodeledCall {
            addr, symbol_name, ..
        } => {
            assert_eq!(addr, 0x9000);
            assert_eq!(symbol_name.as_deref(), Some("__extern_addr__"));
        }
        other => panic!("expected UnmodeledCall, got {:?}", other),
    }
}

#[test]
fn handle_exit_internal_block_end_carries_jumpkind() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    add_internal_region(&mut interp);
    // Internal call (target in binary) — falls through to BlockEnd.
    match interp.handle_exit(0x1234, JumpKind::Call) {
        BlockResult::BlockEnd {
            next_addr,
            jumpkind,
        } => {
            assert_eq!(next_addr, 0x1234);
            assert!(jumpkind.is_call());
        }
        other => panic!("expected BlockEnd, got {:?}", other),
    }
}

#[test]
fn handle_exit_ret_with_nonempty_call_stack_to_external_is_unmodeled() {
    // When a Ret targets external memory but we DO have a call frame, the
    // empty-stack guard should not fire — the fall-through path (external,
    // non-hooked) emits UnmodeledCall with the __extern_addr__ tag.
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    add_internal_region(&mut interp);
    interp.call_stack.push(CallStackEntry {
        call_site_addr: 0x1100,
        callee_addr: 0x1200,
        return_addr: 0x1108,
        stack_ptr: 0x7ffe_0000,
    });
    match interp.handle_exit(0x9000, JumpKind::Ret) {
        BlockResult::UnmodeledCall {
            addr, symbol_name, ..
        } => {
            assert_eq!(addr, 0x9000);
            assert_eq!(symbol_name.as_deref(), Some("__extern_addr__"));
        }
        other => panic!("expected UnmodeledCall, got {:?}", other),
    }
}

/// angr-1c88c gap 4/7: drive `exits.rs::eval_next_addr` (the callback-aware
/// fallthrough evaluator used for symbolic `Ist_Exit` next addresses in
/// non-deferred fork mode).
///
/// Concrete next: the expression is concrete so the symbolic branch is never
/// taken — `eval_next_addr` returns the literal value via the `as_u64()` tail.
#[test]
fn eval_next_addr_concrete_returns_literal() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let mut irsb = IRSB::new(0x1000, VexArch::AMD64);
    irsb.next = IRExpr::Const(IRConst::U64(0x4000));

    Python::initialize();
    Python::attach(|_py| {
        let cb = PythonCallbacks::new();
        let addr = interp
            .eval_next_addr(&cb, &irsb)
            .expect("concrete next address");
        assert_eq!(addr, 0x4000);
    });
}

/// Symbolic next constrained to a single value: `eval_next_addr` concretizes to
/// `Single(addr)` and pins `next == addr` on the solver before returning the
/// concrete target. The pin is a tautology here (the symbol is already
/// constrained), but it exercises the `ConcretizationResult::Single` arm.
#[test]
fn eval_next_addr_single_concretization_returns_target() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let mut irsb = IRSB::new(0x1000, VexArch::AMD64);
    irsb.tyenv.new_temp(IRType::I64);
    irsb.next = IRExpr::RdTmp(0);

    // Symbolic fallthrough target constrained to exactly 0x9000.
    let sym = RustBV::symbolic(&ctx, "next_target", 64);
    let pin = sym.eq(&RustBV::concrete(0x9000, 64), &ctx);
    ctx.assume_true(&pin);
    interp.temps.resize(1, None);
    interp.temps[0] = Some(sym);

    Python::initialize();
    Python::attach(|_py| {
        let cb = PythonCallbacks::new();
        let addr = interp
            .eval_next_addr(&cb, &irsb)
            .expect("single-valued symbolic next address");
        assert_eq!(addr, 0x9000);
    });
}

/// Multi-valued symbolic next: an unconstrained 64-bit symbol concretizes to
/// many targets. `eval_next_addr` cannot fork mid-block, so it surfaces
/// `CbExecutionError::Unsupported` to route the block back to Python.
#[test]
fn eval_next_addr_multivalued_returns_unsupported() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let mut irsb = IRSB::new(0x1000, VexArch::AMD64);
    irsb.tyenv.new_temp(IRType::I64);
    irsb.next = IRExpr::RdTmp(0);

    // Unconstrained symbolic target -> Multiple/TooLarge -> Unsupported.
    let sym = RustBV::symbolic(&ctx, "next_target_open", 64);
    interp.temps.resize(1, None);
    interp.temps[0] = Some(sym);

    Python::initialize();
    Python::attach(|_py| {
        let cb = PythonCallbacks::new();
        let err = interp
            .eval_next_addr(&cb, &irsb)
            .expect_err("multi-valued symbolic next must be unsupported");
        assert!(matches!(err, CbExecutionError::Unsupported(_)));
    });
}

/// Drive the actual call site: a symbolic guard in non-deferred fork mode takes
/// `statements.rs`'s `!use_deferred_forks` branch, which calls `eval_next_addr`
/// for the fallthrough and returns a `SymbolicBranch` carrying both targets.
#[test]
fn exit_nondeferred_symbolic_guard_uses_eval_next_addr_for_fallthrough() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let cfg = crate::callbacks::ExecutionConfig {
        use_deferred_forks: false,
        ..Default::default()
    };
    interp.set_config(cfg);

    let mut irsb = IRSB::new(0x1000, VexArch::AMD64);
    irsb.tyenv.new_temp(IRType::I1);
    // Concrete fallthrough so eval_next_addr returns it directly.
    irsb.next = IRExpr::Const(IRConst::U64(0x4000));

    // Symbolic 1-bit guard -> non-deferred symbolic-branch path.
    let guard = RustBV::symbolic(&ctx, "exit_guard", 1);
    interp.temps.resize(1, None);
    interp.temps[0] = Some(guard);

    let stmt = IRStmt::Exit {
        guard: IRExpr::RdTmp(0),
        dst: 0x9000,
        jk: JumpKind::Boring,
        offsIP: 184,
    };

    Python::initialize();
    Python::attach(|_py| {
        let cb = PythonCallbacks::new();
        let res = interp
            .execute_stmt_with_callbacks(&cb, &stmt, &irsb)
            .expect("symbolic exit");
        match res {
            StmtResult::SymbolicBranch {
                true_target,
                false_target,
                ..
            } => {
                assert_eq!(true_target, 0x9000, "branch-taken target is the dst");
                assert_eq!(
                    false_target, 0x4000,
                    "fallthrough came from eval_next_addr on irsb.next"
                );
            }
            _ => panic!("expected SymbolicBranch from non-deferred symbolic guard"),
        }
    });
}

#[test]
fn get_syscall_num_reads_rax_on_amd64() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.registers.put_reg("rax", RustBV::concrete(231, 64)); // exit_group
    assert_eq!(interp.get_syscall_num(), Some(231));
}

#[test]
fn get_syscall_num_returns_none_for_symbolic_rax() {
    // angr-gffd: symbolic syscall register must produce None so the
    // caller routes to Python instead of dispatching to a native handler.
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let sym = RustBV::symbolic(&ctx, "rax_sym", 64);
    interp.registers.put_reg("rax", sym);
    assert_eq!(interp.get_syscall_num(), None);
}
