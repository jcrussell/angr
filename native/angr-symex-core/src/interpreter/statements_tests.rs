//! Tests for VEX statement execution (extracted from statements.rs).
//!
//! Sibling test modules, each registered from the prod file it covers:
//! `statements_dirty_tests.rs` (`IRStmt::Dirty`) and `statements_cas_tests.rs`
//! (end-to-end `IRStmt::CAS` dispatch). Helpers used by more than one of the
//! three live in `statements_tests_support.rs` (angr-fs8kb.96).
use super::*;
use crate::interpreter::statements_tests_support::{make_irsb_with_temps, new_interp, with_test_host};
use crate::memory::PAGE_SIZE;
use crate::symbolic::Node;
use angr_vex::ir::{Endness, IRType, MBusEvent};

/// An interpreter whose concretizer has `SYMBOLIC_WRITE_ADDRESSES` on, so
/// symbolic-address writes keep the Range strategy and resolve to a candidate
/// set. The default chain is Max-only for unannotated addresses and yields
/// `Single` — see `AddressConcretizer::write_range_applies` (angr-9ke6b.194).
fn new_interp_multiwrite(ctx: &SymContext) -> VEXInterpreter<'_> {
    let mut interp = new_interp(ctx);
    interp.set_concretizer(crate::symbolic::concretize::AddressConcretizer {
        symbolic_write_addresses: true,
        ..crate::symbolic::concretize::AddressConcretizer::new()
    });
    interp
}


#[test]
fn noop_returns_continue() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x1000, &[]);
    with_test_host(|cb| {
        let res = interp
            .execute_stmt_with_callbacks(cb, &IRStmt::NoOp, &irsb)
            .expect("noop");
        assert!(matches!(res, StmtResult::Continue));
    });
}

#[test]
fn imark_updates_current_insn() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x2000, &[]);
    with_test_host(|cb| {
        let stmt = IRStmt::IMark {
            addr: 0x2004,
            len: 4,
            delta: 0,
        };
        let res = interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("imark");
        assert!(matches!(res, StmtResult::Continue));
        assert_eq!(interp.current_insn_addr, 0x2004);
        assert_eq!(interp.current_insn_len, 4);
    });
}

#[test]
fn imark_at_hooked_address_returns_exit() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.add_hook(0x3000);
    let irsb = make_irsb_with_temps(0x3000, &[]);
    with_test_host(|cb| {
        let stmt = IRStmt::IMark {
            addr: 0x3000,
            len: 1,
            delta: 0,
        };
        let res = interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("imark");
        match res {
            StmtResult::Exit { target, jumpkind } => {
                assert_eq!(target, 0x3000);
                assert!(matches!(jumpkind, JumpKind::Boring));
            }
            _ => panic!("expected Exit for hooked IMark"),
        }
    });
}

/// A skipped hook lets its own instruction run once, then is live again, so
/// a loop that comes back to it in the same run stops there (angr-mx00p.132.5).
#[test]
fn skip_hook_once_steps_over_the_first_imark_only() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.add_hook(0x3000);
    interp.set_skip_hook_once(0x3000);
    assert!(!interp.is_hooked(0x3000));
    let irsb = make_irsb_with_temps(0x3000, &[]);
    let stmt = IRStmt::IMark {
        addr: 0x3000,
        len: 1,
        delta: 0,
    };
    with_test_host(|cb| {
        let first = interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("imark");
        assert!(matches!(first, StmtResult::Continue));
        assert!(interp.is_hooked(0x3000));
        let second = interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("imark");
        assert!(matches!(second, StmtResult::Exit { target: 0x3000, .. }));
    });
}

/// A lift cut ends the block at an interior instruction, but not at the
/// block's first, so a state starting there still runs (angr-mx00p.132.60).
#[test]
fn lift_cut_exits_at_an_interior_imark_only() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.add_lift_cut(0x3004);
    let interior = IRStmt::IMark {
        addr: 0x3004,
        len: 1,
        delta: 0,
    };
    with_test_host(|cb| {
        let res = interp
            .execute_stmt_with_callbacks(cb, &interior, &make_irsb_with_temps(0x3000, &[]))
            .expect("imark");
        assert!(matches!(res, StmtResult::Exit { target: 0x3004, .. }));
        let res = interp
            .execute_stmt_with_callbacks(cb, &interior, &make_irsb_with_temps(0x3004, &[]))
            .expect("imark");
        assert!(matches!(res, StmtResult::Continue));
    });
}

/// Run `rax = 0xcafe; [0x4000] = 0xdeadbeef` and then the next instruction's
/// `IMark`, with `host` serving the `instruction` event.
fn fire_after_put_and_store(host: &TestHost) {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.pc = 0x1000;
    let irsb = make_irsb_with_temps(0x1000, &[]);
    let stmts = [
        IRStmt::Put {
            offset: 16,
            data: IRExpr::Const(IRConst::U64(0xcafe)),
        },
        IRStmt::Store {
            addr: IRExpr::Const(IRConst::U64(0x4000)),
            data: IRExpr::Const(IRConst::U32(0xdead_beef)),
            endness: Endness::Little,
        },
        IRStmt::IMark {
            addr: 0x1004,
            len: 4,
            delta: 0,
        },
    ];
    for stmt in &stmts {
        interp
            .execute_stmt_with_callbacks(host, stmt, &irsb)
            .expect("stmt");
    }
}

/// A BP action mid-block reads the interpreter's live values: a register the
/// block wrote and a store still in `pending_stores`, neither yet in the
/// state (angr-mx00p.132.53).
#[test]
fn instruction_fire_publishes_the_live_view() {
    let host = TestHost {
        inspect_instructions: true,
        want_views: true,
        ..TestHost::default()
    };
    fire_after_put_and_store(&host);
    let views = host.instruction_views.borrow();
    let [Some(view)] = views.as_slice() else {
        panic!("one fire with a view expected");
    };
    assert_eq!(view.block_addr(), 0x1000);
    assert_eq!(view.insn_addr(), 0x1004);
    assert_eq!(view.register(16, 8).as_u64(), Some(0xcafe));
    assert_eq!(
        view.load(0x4000, 4).and_then(|v| v.as_u64()),
        Some(0xdead_beef)
    );
    assert!(view.load(0x8000, 4).is_none(), "an unserved load must not read as zero");
    assert!(host.view_live.borrow().is_none(), "the view is dropped after the fire");
}

/// With no BP action registered the fire publishes nothing.
#[test]
fn instruction_fire_without_an_action_takes_no_view() {
    let host = TestHost {
        inspect_instructions: true,
        ..TestHost::default()
    };
    fire_after_put_and_store(&host);
    assert!(matches!(host.instruction_views.borrow().as_slice(), [None]));
}

#[test]
fn abihint_is_no_op() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x1000, &[]);
    let stmt = IRStmt::AbiHint {
        base: Box::new(IRExpr::Const(IRConst::U64(0))),
        len: 0,
        nia: Box::new(IRExpr::Const(IRConst::U64(0x1004))),
    };
    with_test_host(|cb| {
        let res = interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("abihint");
        assert!(matches!(res, StmtResult::Continue));
    });
}

#[test]
fn mbe_fence_is_continue() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x1000, &[]);
    let stmt = IRStmt::MBE(MBusEvent::Fence);
    with_test_host(|cb| {
        // MBE may fall through to default arm but should not error.
        let res = interp.execute_stmt_with_callbacks(cb, &stmt, &irsb);
        assert!(res.is_ok(), "MBE should not error");
    });
}

#[test]
fn put_concrete_writes_register() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x1000, &[]);
    // Write 0xcafe to RAX (offset 16 on AMD64).
    let stmt = IRStmt::Put {
        offset: 16,
        data: IRExpr::Const(IRConst::U64(0xcafe)),
    };
    with_test_host(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("put");
    });
    let val = interp.registers.get(16, 8);
    assert_eq!(val.as_u64(), Some(0xcafe));
}

#[test]
fn wrtmp_concrete_writes_temp() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I32]);
    interp.temps.resize(irsb.tyenv.types.len(), None);
    let stmt = IRStmt::WrTmp {
        tmp: 0,
        data: IRExpr::Const(IRConst::U32(0x1234)),
    };
    with_test_host(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("wrtmp");
    });
    let val = interp.temps[0].as_ref().expect("temp written");
    assert_eq!(val.as_u64(), Some(0x1234));
}

#[test]
fn wrtmp_unknown_temp_errors() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    // No temps allocated, but writing tmp 5.
    let irsb = make_irsb_with_temps(0x1000, &[]);
    let stmt = IRStmt::WrTmp {
        tmp: 5,
        data: IRExpr::Const(IRConst::U32(0)),
    };
    with_test_host(|cb| {
        let res = interp.execute_stmt_with_callbacks(cb, &stmt, &irsb);
        assert!(res.is_err(), "should error on unknown temp");
        if let Err(err) = res {
            assert!(matches!(err, CbExecutionError::UnknownTemp(5)));
        }
    });
}

/// angr-sqfj8.69: LoadG used to drop an out-of-range `dst` write on the floor
/// (bounds check with no `else`), unlike WrTmp/LLSC. Every temp write now goes
/// through `write_tmp`, so a `dst` the tyenv doesn't cover fails loud.
#[test]
fn loadg_unknown_temp_errors() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    // tyenv declares tmp 5 (so the handler gets past its dst-type lookup) but
    // the temps vector was never sized to match. A concrete-false guard takes
    // the alt-value path, so this needs no memory or load callback.
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I32; 6]);
    assert!(interp.temps.is_empty());
    let stmt = IRStmt::LoadG {
        dst: 5,
        addr: Box::new(IRExpr::Const(IRConst::U64(0x4010))),
        alt: Box::new(IRExpr::Const(IRConst::U32(0))),
        guard: Box::new(IRExpr::Const(IRConst::U1(false))),
        cvt: IRLoadGOp::Identity,
        endness: Endness::Little,
    };
    with_test_host(|cb| {
        let res = interp.execute_stmt_with_callbacks(cb, &stmt, &irsb);
        assert!(
            matches!(res, Err(CbExecutionError::UnknownTemp(5))),
            "LoadG must error on an out-of-range dst, not silently drop the write"
        );
    });
}

#[test]
fn exit_with_concrete_false_guard_continues() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x1000, &[]);
    let stmt = IRStmt::Exit {
        guard: IRExpr::Const(IRConst::U1(false)),
        dst: 0x9000,
        jk: JumpKind::Boring,
        offsIP: 184,
    };
    with_test_host(|cb| {
        let res = interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("exit");
        assert!(matches!(res, StmtResult::Continue));
    });
}

#[test]
fn exit_with_concrete_true_guard_takes_branch() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x1000, &[]);
    let stmt = IRStmt::Exit {
        guard: IRExpr::Const(IRConst::U1(true)),
        dst: 0x9000,
        jk: JumpKind::Boring,
        offsIP: 184,
    };
    with_test_host(|cb| {
        let res = interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("exit");
        match res {
            StmtResult::Exit { target, .. } => assert_eq!(target, 0x9000),
            _ => panic!("expected Exit for true guard"),
        }
    });
}

#[test]
fn store_to_concrete_addr_buffers_pending_store() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    // Without rust_memory the fallback path takes a fast buffer route for
    // concrete addresses + concrete data: just append to pending_stores
    // (no callback invoked until flush).
    assert!(!interp.use_rust_memory);
    let irsb = make_irsb_with_temps(0x1000, &[]);
    let stmt = IRStmt::Store {
        addr: IRExpr::Const(IRConst::U64(0x4000)),
        data: IRExpr::Const(IRConst::U32(0xdead_beef)),
        endness: Endness::Little,
    };
    with_test_host(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("store should buffer without erroring");
    });
    // Buffered into pending_stores keyed by address.
    let data = interp
        .pending_stores
        .try_load(0x4000, 4, false)
        .expect("pending store at 0x4000");
    assert_eq!(data, &[0xef, 0xbe, 0xad, 0xde]);
}

/// angr-1c88c: drive `expressions_loads.rs::build_ite_store_from_callbacks`.
///
/// This site is only reachable when `use_rust_memory == false` — during normal
/// exploration the manager always installs `rust_memory` (stepping.rs
/// `set_rust_memory`), and `store_with_concretization` handles a `Multiple`
/// concretization natively (auto-maps candidate pages / silently no-ops for
/// unmapped ones), so it never falls back to the Python store callbacks. A
/// `disable_rust_memory` interpreter with the symbolic-value + load-batch
/// callbacks wired up takes `handle_symbolic_store` → `dispatch_multi_store` →
/// `build_ite_store_from_callbacks`. For each candidate `a_i` the helper builds
/// `ITE(addr == a_i, data, mem[a_i])` in Rust's Z3 context and invokes
/// `memory_store_symbolic_value(a_i, ite)` once per address.
#[test]
fn build_ite_store_invokes_per_addr_ite_callbacks() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    // Site is below the use_rust_memory gate's fallback; the test default is
    // already callback-mode, but assert it to document the precondition.
    assert!(!interp.use_rust_memory);

    let cb = TestHost {
        symbolic_value_store: SymbolicValueStore::Accept,
        serve_zero_loads: true,
        ..TestHost::default()
    };
    assert!(cb.has_memory_store_symbolic_value());

    let addr_expr = RustBV::symbol("store_addr", 64);
    let data_val = RustBV::symbol("store_data", 64);
    let addrs = [0x1000u64, 0x2000u64];

    interp
        .build_ite_store_from_callbacks(&cb, &addrs, &addr_expr, &data_val, None)
        .expect("build_ite_store_from_callbacks");

    let rec = cb.symbolic_value_stores.borrow();
    assert_eq!(rec.len(), 2, "one symbolic-value store per candidate addr");

    let mut seen_addrs = Vec::new();
    for (addr, value) in rec.iter() {
        // Each stored value is the in-Rust ITE chain.
        assert!(
            matches!(value.node(), Some(Node { op: BVOp::Ite, .. })),
            "stored value must be the ITE chain, got {value:?}"
        );
        assert_eq!(value.width(), 64, "ITE store width matches the data width");
        seen_addrs.push(*addr);
    }
    seen_addrs.sort_unstable();
    assert_eq!(seen_addrs, vec![0x1000, 0x2000]);
}

#[test]
fn loadg_unknown_cvt_surfaces_invalid_ir() {
    use angr_vex::ir::IRLoadGOp;
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    // dst temp t0 is I32.
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I32]);
    let stmt = IRStmt::LoadG {
        dst: 0,
        addr: Box::new(IRExpr::Const(IRConst::U64(0x4000))),
        alt: Box::new(IRExpr::Const(IRConst::U32(0))),
        guard: Box::new(IRExpr::Const(IRConst::U8(1))),
        cvt: IRLoadGOp::Unknown,
        endness: Endness::Little,
    };
    with_test_host(|cb| {
        let res = interp.execute_stmt_with_callbacks(cb, &stmt, &irsb);
        // An unrecognized cvt must error rather than silently load+Identity.
        assert!(
            matches!(res, Err(CbExecutionError::InvalidIR(_))),
            "Unknown LoadG cvt should surface InvalidIR"
        );
    });
}

#[test]
fn put_at_high_offset_does_not_overflow_dirty_mask() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x1000, &[]);
    // Offset 600 -> bit index 150, exceeds 128-bit mask. Should NOT panic.
    let stmt = IRStmt::Put {
        offset: 600,
        data: IRExpr::Const(IRConst::U8(0xaa)),
    };
    with_test_host(|cb| {
        let res = interp.execute_stmt_with_callbacks(cb, &stmt, &irsb);
        assert!(res.is_ok(), "high-offset Put should not overflow");
    });
}

// angr-ofyh: a concrete store overwriting a prior symbolic store must evict
// the stale symbolic shadow from both the pending and flushed maps so a
// later load returns the concrete bytes, not the old symbolic value.
#[test]
fn evict_overlapping_symbolic_stores_drops_shadowed_entries() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp
        .pending_symbolic_stores
        .insert(0x2000, RustBV::symbol("p", 64));
    interp
        .all_flushed_symbolic_stores
        .insert(0x2000, RustBV::symbol("f", 64));
    // 8-byte concrete store at 0x2000 fully overwrites both shadows.
    interp.evict_overlapping_symbolic_stores(0x2000, 8);
    assert!(!interp.pending_symbolic_stores.contains_key(&0x2000));
    assert!(!interp.all_flushed_symbolic_stores.contains_key(&0x2000));
}

// angr-fs8kb.64: the eviction overlap test must see a wrapped range, in
// either role. With `saturating_add` both ends clamped to `u64::MAX`, so a
// symbolic store whose tail wrapped kept its stale shadow alive under the
// concrete store that overwrote it -- and a later load of those bytes
// returned the symbolic value instead of the concrete ones.
#[test]
fn evict_overlapping_symbolic_stores_drops_entries_across_address_top() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    // Symbolic store wraps: [MAX-3, MAX] plus [0, 3].
    let wrapped = u64::MAX - 3;
    interp
        .pending_symbolic_stores
        .insert(wrapped, RustBV::symbol("w", 64));
    // Concrete store [0x0,0x2) touches only the wrapped tail.
    interp.evict_overlapping_symbolic_stores(0x0, 2);
    assert!(
        !interp.pending_symbolic_stores.contains_key(&wrapped),
        "a concrete store landing in the wrapped tail must evict the shadow"
    );

    // Mirror case: the *concrete* store is the one that wraps.
    interp
        .all_flushed_symbolic_stores
        .insert(0x2, RustBV::symbol("f", 32));
    interp.evict_overlapping_symbolic_stores(u64::MAX - 1, 8);
    assert!(
        !interp.all_flushed_symbolic_stores.contains_key(&0x2),
        "a wrapped concrete store must evict shadows in its wrapped tail"
    );
}

#[test]
fn evict_overlapping_symbolic_stores_keeps_disjoint_entries() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    // [0x3000,0x3004) and [0x3008,0x300c).
    interp
        .pending_symbolic_stores
        .insert(0x3000, RustBV::symbol("a", 32));
    interp
        .pending_symbolic_stores
        .insert(0x3008, RustBV::symbol("b", 32));
    // Concrete store [0x3002,0x3006) overlaps only the first entry.
    interp.evict_overlapping_symbolic_stores(0x3002, 4);
    assert!(!interp.pending_symbolic_stores.contains_key(&0x3000));
    assert!(interp.pending_symbolic_stores.contains_key(&0x3008));
}

// angr-slbsd: `cas_store_symbolic_data` (the CAS-store path taken when the
// stored value is symbolic -- reached from `cas_dispatch_store` and
// unconditionally from `cas_writeback`'s uncertain-comparison branch) must
// invalidate any stale cached IRSB at the store's (concretized) target
// address, exactly like the ordinary `IRStmt::Store` path. Self-modifying
// `lock cmpxchg` on a packer/protector's own code is a real technique this
// protects against.
//
// This test drives the concrete-address branch, which previously had ZERO
// invalidation and zero Python delegation (raw pending_symbolic_stores /
// pending_stores insert only).
#[test]
fn cas_store_symbolic_data_concrete_addr_invalidates_cached_block() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.add_concrete_memory(0x1000, vec![0u8; 0x1000]);
    // Cached block covers [0x1010, 0x1014) (default IMark len 4 from
    // make_irsb_with_temps).
    interp.cache_block(0x1010, make_irsb_with_temps(0x1010, &[]));
    assert!(interp.has_cached_block(0x1010));

    let addr_expr = IRExpr::Const(IRConst::U64(0x1010));
    let data_bv = RustBV::symbol("cas_data", 32);
    let call_irsb = make_irsb_with_temps(0x1010, &[]);

    with_test_host(|cb| {
        // Bare `TestHost::default()` has no memory_store_symbolic_value, so
        // the store itself must hard-error rather than zero-fill (angr-9ke6b.19,
        // asserted by `cas_symbolic_store_without_callback_errors_not_zero_fills`
        // below). Invalidation happens before that dispatch and still applies.
        let err = interp
            .cas_store_symbolic_data(cb, &addr_expr, &data_bv, &call_irsb)
            .expect_err("symbolic CAS data with no symbolic-store callback must error");
        assert!(matches!(err, CbExecutionError::Unsupported(_)), "{err}");
    });

    assert!(
        !interp.has_cached_block(0x1010),
        "stale cached block must be invalidated by a CAS store to its address"
    );
    assert!(interp.is_code_page_dirtied(0x1010));
}

// angr-9ke6b.19: with `memory_store_symbolic_value` unwired, the symbolic-CAS
// store path used to fall back to `bv_to_bytes` — which yields all zeros for a
// symbolic expression — and push those zeros into `pending_stores`, so a later
// `flush_stores` would overwrite real memory with 0 and report success. That is
// a wrong answer, not a degraded one; the module's
// `avoid-silent-no-op-callback-fallbacks` invariant says hard-error instead.
#[test]
fn cas_symbolic_store_without_callback_errors_not_zero_fills() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.add_concrete_memory(0x1000, vec![0xffu8; 0x1000]);

    let addr_expr = IRExpr::Const(IRConst::U64(0x1010));
    let data_bv = RustBV::symbol("cas_sym", 32);
    let call_irsb = make_irsb_with_temps(0x1010, &[]);

    with_test_host(|cb| {
        assert!(!cb.has_memory_store_symbolic_value());
        let err = interp
            .cas_store_symbolic_data(cb, &addr_expr, &data_bv, &call_irsb)
            .expect_err("symbolic store without its callback must not silently succeed");
        let msg = err.to_string();
        assert!(
            msg.contains("memory_store_symbolic_value") && msg.contains("0x1010"),
            "unexpected error: {msg}"
        );
    });

    // The critical half: no zero bytes were queued for a later flush.
    assert_eq!(
        interp.pending_stores.len(),
        0,
        "the refused store must leave no zero-filled bytes behind"
    );
}

// angr-6cp06.72: the no-callback branch of `cas_store_symbolic_data` used to
// insert into `pending_symbolic_stores` *and* push bytes into
// `pending_stores`. `reject_symbolic_byte_store` short-circuits above it for
// every symbolic value, so the only value that can reach those lines is a
// concrete one -- and a concrete entry in the symbolic map would shadow the
// concrete buffer on load forwarding. This pins the surviving behaviour:
// concrete bytes buffered, symbolic map untouched.
#[test]
fn cas_concrete_store_without_callback_buffers_bytes_only() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.add_concrete_memory(0x1000, vec![0u8; 0x1000]);

    let addr_expr = IRExpr::Const(IRConst::U64(0x1010));
    let data_bv = RustBV::concrete(0xdead_beef, 32);
    let call_irsb = make_irsb_with_temps(0x1010, &[]);

    with_test_host(|cb| {
        assert!(!cb.has_memory_store_symbolic_value());
        interp
            .cas_store_symbolic_data(cb, &addr_expr, &data_bv, &call_irsb)
            .expect("a concrete CAS value needs no symbolic-store callback");
    });

    assert_eq!(
        interp.pending_stores.try_load_exact(0x1010, 4, interp.registers.is_big_endian()),
        Some(&0xdead_beefu32.to_le_bytes()[..]),
        "the concrete value must be buffered as bytes"
    );
    assert!(
        interp.pending_symbolic_stores.is_empty(),
        "a provably-concrete value must not land in the symbolic-store map"
    );
}

// Symbolic-address branch: the CAS-target address itself is unresolved at
// eval time. `cas_store_symbolic_data` concretizes it via
// `concretize_cached_write` (the same helper `handle_symbolic_store` uses for
// the ordinary Store path) and invalidates through `invalidate_code_on_store`
// before dispatching the store. The address is pinned to a single solution
// via `pin_fallback_addr` -- the same `assume_true(addr == chosen)` idiom
// `concretize_write`'s own Max-fallback uses internally -- so the test gets a
// deterministic `Single` result without depending on the concretizer's
// range-enumeration heuristics for an otherwise-unconstrained value.
#[test]
fn cas_store_symbolic_data_symbolic_addr_invalidates_cached_block() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.add_concrete_memory(0x2000, vec![0u8; 0x1000]);
    interp.cache_block(0x2010, make_irsb_with_temps(0x2010, &[]));
    assert!(interp.has_cached_block(0x2010));

    // A symbolic address value, placed in temp 0 so `RdTmp(0)` evaluates to
    // it unchanged -- IRExpr has no way to embed an arbitrary RustBV literal,
    // so this is the standard way tests inject a specific RustBV as an
    // expression's evaluated value. Constrain it to a single solution
    // (0x2010, inside the cached block) so concretization is deterministic.
    let addr_bv = RustBV::symbol("cas_sym_addr", 64);
    crate::symbolic::concretize::pin_fallback_addr(&ctx, &addr_bv, 0x2010);
    interp.temps = vec![Some(addr_bv)];

    let addr_expr = IRExpr::RdTmp(0);
    let data_bv = RustBV::symbol("cas_sym_data", 32);
    let call_irsb = make_irsb_with_temps(0x2010, &[]);

    with_test_host(|cb| {
        let res = interp.cas_store_symbolic_data(cb, &addr_expr, &data_bv, &call_irsb);
        // No memory_store_symbolic_full callback is registered on this bare
        // TestHost::default(), so the store dispatch itself errors out --
        // but invalidation must already have happened before that point.
        assert!(res.is_err());
    });

    assert!(
        !interp.has_cached_block(0x2010),
        "stale cached block must be invalidated even though the CAS-target \
         address is symbolic"
    );
    assert!(interp.is_code_page_dirtied(0x2010));
}

// angr-srk4b: `handle_symbolic_store`'s Single arm invalidates cached code at
// its concretized target, but the Multiple/Strided arms routed straight to
// `dispatch_multi_store` (an `&self` method that structurally cannot touch
// `block_cache`) with zero invalidation -- a symbolic store that concretizes
// to several candidate addresses, one of which lands on a previously-lifted
// page, never evicted the stale IRSB. Fixed by calling
// `invalidate_code_on_store` once (the same dispatcher `try_rust_memory_store`
// and `cas_store_symbolic_data` already use) right after
// `concretize_cached_write`, before the match on the ConcretizationResult
// shape -- so Multiple/Strided (and TooLarge/Failed) get the same per-address
// treatment Single always had.
//
// This test drives the Strided shape via the canonical `a[i]` pattern: an
// index constrained to `{0, 1, 2}` (`idx < 3`) times a stride, added to a
// base -- exactly the "constrained index" example from the bug report,
// verified via a genuine Z3-backed concretization (not a hand-seeded cache
// entry) so the test exercises the real `detect_stride_from_solutions` path.
#[test]
fn handle_symbolic_store_strided_invalidates_cached_candidate_only() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp_multiwrite(&ctx);
    interp.add_concrete_memory(0x5000, vec![0u8; 0x2000]);

    // idx in {0, 1, 2} -> addr in {0x5010, 0x5020, 0x5030} (stride 0x10).
    let idx = RustBV::symbol("idx_strided", 64);
    let bound = idx.ult(&RustBV::concrete(3, 64));
    ctx.assume_true(&bound);
    let stride = RustBV::concrete(0x10, 64);
    let base = RustBV::concrete(0x5010, 64);
    let addr = base.add(&idx.mul(&stride));

    // Sanity-check the construction actually produces Strided, not some other
    // shape (e.g. if range/stride detection heuristics change).
    let pre_check = interp.concretize_cached_write(&addr);
    match &*pre_check {
        ConcretizationResult::Strided {
            base,
            stride,
            count,
        } => {
            assert_eq!((*base, *stride, *count), (0x5010, 0x10, 3));
        }
        other => panic!("expected Strided({{0x5010, 0x10, 3}}), got {other:?}"),
    }

    // Candidate #2 (0x5020) has a cached block -- must be invalidated.
    interp.cache_block(0x5020, make_irsb_with_temps(0x5020, &[]));
    assert!(interp.has_cached_block(0x5020));
    // An unrelated cached block outside every candidate's store window
    // ([0x5010,0x5014), [0x5020,0x5024), [0x5030,0x5034)) must survive --
    // guards against over-eager invalidation of the whole block cache.
    interp.cache_block(0x5100, make_irsb_with_temps(0x5100, &[]));
    assert!(interp.has_cached_block(0x5100));

    let data_val = RustBV::symbol("store_data", 32);
    with_test_host(|cb| {
        // No memory_store_symbolic_value/_full callback registered, so the
        // store dispatch itself errors out -- but invalidation must already
        // have happened before that point (mirrors the CAS symbolic-addr
        // test above).
        let res = interp.handle_symbolic_store(cb, &addr, &data_val, 4);
        assert!(res.is_err());
    });

    assert!(
        !interp.has_cached_block(0x5020),
        "cached block at a Strided candidate address must be invalidated"
    );
    assert!(
        interp.has_cached_block(0x5100),
        "cached block outside every candidate's store window must survive \
         (no over-eager invalidation)"
    );
    assert!(interp.is_code_page_dirtied(0x5020));
}

// Same gap, Multiple shape: index constrained to the non-arithmetic set
// {0, 1, 3} (via explicit disjunction) so stride-detection's GCD collapses to
// 1 and `concretize` genuinely returns `Multiple`, not `Strided`.
#[test]
fn handle_symbolic_store_multiple_invalidates_cached_candidate_only() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp_multiwrite(&ctx);
    interp.add_concrete_memory(0x6000, vec![0u8; 0x2000]);

    let idx = RustBV::symbol("idx_multiple", 64);
    let eq0 = idx.eq(&RustBV::concrete(0, 64));
    let eq1 = idx.eq(&RustBV::concrete(1, 64));
    let eq3 = idx.eq(&RustBV::concrete(3, 64));
    let cond = eq0.or(&eq1).or(&eq3);
    ctx.assume_true(&cond);
    let base = RustBV::concrete(0x6000, 64);
    let addr = base.add(&idx);

    let pre_check = interp.concretize_cached_write(&addr);
    match &*pre_check {
        ConcretizationResult::Multiple(addrs) => {
            assert_eq!(addrs, &vec![0x6000, 0x6001, 0x6003]);
        }
        other => panic!("expected Multiple([0x6000, 0x6001, 0x6003]), got {other:?}"),
    }

    // Candidate 0x6001 has a cached block -- must be invalidated.
    interp.cache_block(0x6001, make_irsb_with_temps(0x6001, &[]));
    assert!(interp.has_cached_block(0x6001));
    // Outside every candidate's store window ([0x6000,0x6004),
    // [0x6001,0x6005), [0x6003,0x6007)) -- must survive.
    interp.cache_block(0x6100, make_irsb_with_temps(0x6100, &[]));
    assert!(interp.has_cached_block(0x6100));

    let data_val = RustBV::symbol("store_data_multi", 32);
    with_test_host(|cb| {
        let res = interp.handle_symbolic_store(cb, &addr, &data_val, 4);
        assert!(res.is_err());
    });

    assert!(
        !interp.has_cached_block(0x6001),
        "cached block at a Multiple candidate address must be invalidated"
    );
    assert!(
        interp.has_cached_block(0x6100),
        "cached block outside every candidate's store window must survive \
         (no over-eager invalidation)"
    );
    assert!(interp.is_code_page_dirtied(0x6001));
}

/// angr-9ke6b.83: a `LoadG` reading an address that an earlier `Store` in the
/// same block wrote must observe the stored value. `resolve_loadg_load` used to
/// call `load_from_callback` directly, skipping the pending-store buffer that
/// ordinary `Load` consults, so the guarded load fell through to (stale) Python
/// memory. Callback-mode variant (`use_rust_memory == false`), where the store
/// only lives in `pending_stores` until flush.
#[test]
fn loadg_sees_pending_store_from_same_block() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    assert!(!interp.use_rust_memory);
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I32]);
    interp.temps.resize(1, None);

    let store = IRStmt::Store {
        addr: IRExpr::Const(IRConst::U64(0x4000)),
        data: IRExpr::Const(IRConst::U32(0xdead_beef)),
        endness: Endness::Little,
    };
    let loadg = IRStmt::LoadG {
        dst: 0,
        addr: Box::new(IRExpr::Const(IRConst::U64(0x4000))),
        alt: Box::new(IRExpr::Const(IRConst::U32(0))),
        guard: Box::new(IRExpr::Const(IRConst::U1(true))),
        cvt: IRLoadGOp::Identity,
        endness: Endness::Little,
    };

    with_test_host(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &store, &irsb)
            .expect("store should buffer");
        interp
            .execute_stmt_with_callbacks(cb, &loadg, &irsb)
            .expect("loadg should read the buffered store");
    });

    assert_eq!(
        interp.temps[0].as_ref().and_then(|bv| bv.as_u64()),
        Some(0xdead_beef),
        "LoadG must observe the pending store, not stale Python memory"
    );
}

/// Rust-memory variant (angr-9ke6b.83): with `use_rust_memory` on (the
/// production configuration — `exploration/step_core.rs` installs the state's
/// memory into the interpreter) stores commit only to `rust_memory` and are
/// never synced to Python, so a `LoadG` that skipped `try_rust_memory_load`
/// read an unrelated value from the Python callback.
#[test]
fn loadg_sees_rust_memory_store_from_same_block() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map_page(0x4000u64, vec![0u8; PAGE_SIZE as usize], Permission::RW);
    interp.set_rust_memory(mem);
    assert!(interp.use_rust_memory);

    let irsb = make_irsb_with_temps(0x1000, &[IRType::I32]);
    interp.temps.resize(1, None);

    let store = IRStmt::Store {
        addr: IRExpr::Const(IRConst::U64(0x4010)),
        data: IRExpr::Const(IRConst::U32(0x1234_5678)),
        endness: Endness::Little,
    };
    let loadg = IRStmt::LoadG {
        dst: 0,
        addr: Box::new(IRExpr::Const(IRConst::U64(0x4010))),
        alt: Box::new(IRExpr::Const(IRConst::U32(0))),
        guard: Box::new(IRExpr::Const(IRConst::U1(true))),
        cvt: IRLoadGOp::Identity,
        endness: Endness::Little,
    };

    with_test_host(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &store, &irsb)
            .expect("store into rust memory");
        interp
            .execute_stmt_with_callbacks(cb, &loadg, &irsb)
            .expect("loadg should read rust memory");
    });

    assert_eq!(
        interp.temps[0].as_ref().and_then(|bv| bv.as_u64()),
        Some(0x1234_5678),
        "LoadG must read the value the Store committed to rust_memory"
    );
}

/// `Single` concretization arm (angr-9ke6b.83): a symbolic address pinned to one
/// solution must go through the same layered read (`load_layered`) rather than
/// straight to the Python callback.
// Needs a real solver: the `Single` branch is only taken when concretization
// finds exactly one solution (angr-9ke6b.236).
#[test]
fn loadg_single_concretization_sees_rust_memory_store() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map_page(0x4000u64, vec![0u8; PAGE_SIZE as usize], Permission::RW);
    interp.set_rust_memory(mem);

    // t0 = symbolic address constrained to exactly 0x4020.
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I64, IRType::I32]);
    interp.temps.resize(2, None);
    let sym_addr = RustBV::symbol("loadg_addr", 64);
    ctx.assume_true(&sym_addr.eq(&RustBV::concrete(0x4020, 64)));
    interp.temps[0] = Some(sym_addr);

    let store = IRStmt::Store {
        addr: IRExpr::Const(IRConst::U64(0x4020)),
        data: IRExpr::Const(IRConst::U32(0xcafe_babe)),
        endness: Endness::Little,
    };
    let loadg = IRStmt::LoadG {
        dst: 1,
        addr: Box::new(IRExpr::RdTmp(0)),
        alt: Box::new(IRExpr::Const(IRConst::U32(0))),
        guard: Box::new(IRExpr::Const(IRConst::U1(true))),
        cvt: IRLoadGOp::Identity,
        endness: Endness::Little,
    };

    with_test_host(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &store, &irsb)
            .expect("store into rust memory");
        interp
            .execute_stmt_with_callbacks(cb, &loadg, &irsb)
            .expect("loadg with symbolic (Single) address");
    });

    assert_eq!(
        interp.temps[1].as_ref().and_then(|bv| bv.as_u64()),
        Some(0xcafe_babe),
        "LoadG on a Single-concretized address must read the just-stored value"
    );
}

/// angr-sqfj8.61: `handle_concrete_store`'s symbolic-store-callback failure
/// path used to `self.ctx.eval(&data_val).unwrap_or(0)` and write the result
/// through `call_memory_store`. When the concretization itself fails, that
/// wrote concrete **zeros** over the target — the exact zero-fill wrong answer
/// `reject_symbolic_byte_store` exists to prevent — while reporting `Ok(())`.
///
/// A value wider than `MAX_CONCRETE_CHUNK` is the deterministic trigger: it
/// cannot round-trip through the `u128` the eval path carries. The store must
/// now fail loudly instead, and must not touch memory at all.
#[test]
fn symbolic_store_fallback_errors_instead_of_zero_filling_when_eval_fails() {
    let ctx = SymContext::new_mock();
    // X86 (32-bit, non-stack address) is the only configuration that arms the
    // `memory_store_symbolic_value` path in `handle_concrete_store`.
    let mut interp = VEXInterpreter::new(VexArch::X86, &ctx);

    let cb = TestHost {
        symbolic_value_store: SymbolicValueStore::Fail,
        accept_stores: true,
        ..TestHost::default()
    };

    // 256 bits = 32 bytes > MAX_CONCRETE_CHUNK, so no u128 witness exists.
    let data_val = RustBV::symbol("wide_store_data", 256);
    let err = interp
        .handle_concrete_store(&cb, 0x10_0000, data_val, 32)
        .expect_err("unconcretizable symbolic store must fail, not zero-fill");
    assert!(
        matches!(err, CbExecutionError::Callback(_)),
        "the original symbolic-store callback error is propagated, got {err:?}"
    );

    assert_eq!(
        cb.stores.borrow().len(),
        0,
        "no byte-level store may reach memory when concretization failed"
    );
}

/// Converse of the test above: when the value *is* concretizable, the fallback
/// still writes the concretized bytes little-endian through `memory_store`.
/// Guards against "fix the zero-fill by disabling the fallback entirely".
///
/// The premise is a value that is symbolic but *concretizable* — `data_val` is
/// pinned to `0xdeadbeef` by a constraint, and the solver reads that back
/// through `eval` (angr-c7xno.100).
#[test]
fn symbolic_store_fallback_writes_concretized_bytes_when_eval_succeeds() {
    let ctx = SymContext::new_mock();
    let mut interp = VEXInterpreter::new(VexArch::X86, &ctx);

    let data_val = RustBV::symbol("store_data", 32);
    ctx.assume_true(&data_val.eq(&RustBV::concrete(0xdead_beef, 32)));

    let cb = TestHost {
        symbolic_value_store: SymbolicValueStore::Fail,
        accept_stores: true,
        ..TestHost::default()
    };

    interp
        .handle_concrete_store(&cb, 0x10_0000, data_val, 4)
        .expect("concretizable symbolic store falls back cleanly");

    let rec = cb.stores.borrow();
    assert_eq!(rec.len(), 1, "exactly one byte-level fallback store");
    let (addr, data) = &rec[0];
    assert_eq!(*addr, 0x10_0000);
    assert_eq!(
        *data,
        vec![0xef, 0xbe, 0xad, 0xde],
        "little-endian concretization"
    );
}

// ---------------------------------------------------------------------------
// angr-5mnx3.23: every `memory_store_symbolic_value` dispatch must buffer.
// ---------------------------------------------------------------------------

/// A host in the production configuration: `memory_store_symbolic_value`
/// exists and absorbs the store — exactly what
/// `rust_manager.py::_cb_memory_store_symbolic_value` does when the callback
/// state's memory is a `RustMemoryProxy` (`state.memory` *is* Rust memory, so
/// the callback deliberately declines to re-enter it). A dispatch that
/// reaches this service without first calling `buffer_store_for_rust_memory`
/// therefore lands nowhere at all.
fn with_proxy_callbacks<R, F: FnOnce(&TestHost) -> R>(f: F) -> R {
    let cb = TestHost {
        symbolic_value_store: SymbolicValueStore::Accept,
        serve_zero_loads: true,
        ..TestHost::default()
    };
    assert!(cb.has_memory_store_symbolic_value());
    f(&cb)
}

/// Build an interpreter that owns its memory (the production configuration —
/// `exploration/step_core.rs` installs the state's memory), so
/// `buffer_store_for_rust_memory`'s `use_rust_memory` gate holds.
fn new_interp_rust_memory(ctx: &SymContext) -> VEXInterpreter<'_> {
    let mut interp = new_interp(ctx);
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map_page(0x4000u64, vec![0u8; PAGE_SIZE as usize], Permission::RW);
    interp.set_rust_memory(mem);
    interp
}

/// The value at `addr` read back both ways the rest of the run sees it: by a
/// later in-block load (`load_layered`), and from `rust_memory` once the
/// block's pending buffers are flushed. angr-vmi2l.168: asserting only that a
/// buffer entry exists stayed green while the entry was invisible to the
/// in-block load and replayed over newer stores.
fn read_back(
    interp: &mut VEXInterpreter<'_>,
    cb: &TestHost,
    addr: u64,
    size: usize,
) -> (RustBV, RustBV) {
    let in_block = interp
        .load_layered(cb, &RustBV::concrete(u128::from(addr), 64), size, None)
        .expect("in-block read-back");
    interp.flush_stores_to_rust_memory();
    let flushed = interp
        .rust_memory
        .as_ref()
        .expect("rust_memory installed")
        .load_concrete(addr, size as u32)
        .expect("flushed read-back");
    (in_block, flushed)
}

/// `bv` evaluated with each `(cond, true/false)` assumption in force.
fn eval_assuming(ctx: &SymContext, bv: &RustBV, assumptions: &[(&RustBV, bool)]) -> Option<u128> {
    ctx.push();
    for &(cond, holds) in assumptions {
        if holds {
            ctx.assume_true(cond);
        } else {
            ctx.assume_false(cond);
        }
    }
    let v = ctx.eval(bv);
    ctx.pop();
    v
}

/// angr-5mnx3.23, angr-vmi2l.168: a `StoreG` with a *symbolic* guard at a
/// concrete address stores `ITE(guard, new, current)`, where `current` is what
/// `rust_memory` holds (here an earlier store in the same block), and both a
/// later in-block load and the flushed memory see it. The ITE used to read
/// the host's stale shadow and park in the pending buffers.
// Needs a real solver: `classify_guard` must return `Symbolic`, which requires
// `check_branch_feasibility` to find both branches satisfiable.
#[test]
fn storeg_symbolic_guard_reads_back_for_rust_memory() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp_rust_memory(&ctx);

    let guard = RustBV::symbol("storeg_guard", 1);
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I1]);
    interp.temps = vec![Some(guard.clone())];

    let prior = IRStmt::Store {
        addr: IRExpr::Const(IRConst::U64(0x4010)),
        data: IRExpr::Const(IRConst::U32(0xcafe_0002)),
        endness: Endness::Little,
    };
    let storeg = IRStmt::StoreG {
        addr: Box::new(IRExpr::Const(IRConst::U64(0x4010))),
        data: Box::new(IRExpr::Const(IRConst::U32(0xdead_beef))),
        guard: Box::new(IRExpr::RdTmp(0)),
        endness: Endness::Little,
    };

    with_proxy_callbacks(|cb| {
        for stmt in [&prior, &storeg] {
            interp
                .execute_stmt_with_callbacks(cb, stmt, &irsb)
                .expect("store, then guarded store with symbolic guard");
        }
        let (in_block, flushed) = read_back(&mut interp, cb, 0x4010, 4);
        for value in [&in_block, &flushed] {
            assert_eq!(eval_assuming(&ctx, value, &[(&guard, true)]), Some(0xdead_beef));
            assert_eq!(
                eval_assuming(&ctx, value, &[(&guard, false)]),
                Some(0xcafe_0002),
                "guard false keeps the earlier store, not the pre-block value"
            );
        }
    });
}

/// angr-vmi2l.168: a plain `Store` after a symbolic-guard `StoreG` to the same
/// address wins on every path. The guarded write used to sit in the pending
/// buffer and be replayed over the newer direct store at flush.
#[test]
fn storeg_then_plain_store_keeps_the_newer_value() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp_rust_memory(&ctx);

    let guard = RustBV::symbol("storeg_later_guard", 1);
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I1]);
    interp.temps = vec![Some(guard.clone())];

    let storeg = IRStmt::StoreG {
        addr: Box::new(IRExpr::Const(IRConst::U64(0x4010))),
        data: Box::new(IRExpr::Const(IRConst::U32(0xcafe_0003))),
        guard: Box::new(IRExpr::RdTmp(0)),
        endness: Endness::Little,
    };
    let later = IRStmt::Store {
        addr: IRExpr::Const(IRConst::U64(0x4010)),
        data: IRExpr::Const(IRConst::U32(0xcafe_0002)),
        endness: Endness::Little,
    };

    with_proxy_callbacks(|cb| {
        for stmt in [&storeg, &later] {
            interp
                .execute_stmt_with_callbacks(cb, stmt, &irsb)
                .expect("guarded store, then plain store");
        }
        let (in_block, flushed) = read_back(&mut interp, cb, 0x4010, 4);
        for value in [&in_block, &flushed] {
            for holds in [true, false] {
                assert_eq!(
                    eval_assuming(&ctx, value, &[(&guard, holds)]),
                    Some(0xcafe_0002),
                    "guard {holds}: the later plain store wins"
                );
            }
        }
    });
}

/// angr-vmi2l.169: a symbolic-guard `StoreG` at a symbolic address pins the
/// address only under the guard. ARM's `strhi r3, [r1, r5, lsl #2]` with the
/// index in `[0, 3]`: the guard-false path stores nothing, so it keeps every
/// index, while the guard-true path is pinned to the Max solution and gets
/// the guard ITE there.
#[test]
fn storeg_symbolic_guard_symbolic_addr_leaves_index_free_when_false() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp_rust_memory(&ctx);

    let guard = RustBV::symbol("storeg_hi", 1);
    let idx = RustBV::symbol("storeg_idx", 64);
    ctx.assume_true(&idx.ule(&RustBV::concrete(3, 64)));
    let addr = RustBV::concrete(0x4000, 64).add(&idx.mul(&RustBV::concrete(4, 64)));
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I1, IRType::I64]);
    interp.temps = vec![Some(guard.clone()), Some(addr)];

    let storeg = IRStmt::StoreG {
        addr: Box::new(IRExpr::RdTmp(1)),
        data: Box::new(IRExpr::Const(IRConst::U32(0xcafe_0003))),
        guard: Box::new(IRExpr::RdTmp(0)),
        endness: Endness::Little,
    };
    with_proxy_callbacks(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &storeg, &irsb)
            .expect("guarded store with symbolic guard and address");
    });

    let idx_values = |want: bool| {
        ctx.push();
        if want {
            ctx.assume_true(&guard);
        } else {
            ctx.assume_false(&guard);
        }
        let mut vals = ctx.eval_upto(&idx, 8);
        ctx.pop();
        vals.sort_unstable();
        vals
    };
    assert_eq!(idx_values(false), vec![0, 1, 2, 3], "guard false: no store, index free");
    assert_eq!(idx_values(true), vec![3], "guard true: pinned to the Max address");
    with_proxy_callbacks(|cb| {
        let (in_block, flushed) = read_back(&mut interp, cb, 0x400c, 4);
        for value in [&in_block, &flushed] {
            assert_eq!(
                eval_assuming(&ctx, value, &[(&guard, true)]),
                Some(0xcafe_0003),
                "the guard ITE lands at the Max address"
            );
            assert_eq!(eval_assuming(&ctx, value, &[(&guard, false)]), Some(0));
        }
    });
}

/// angr-vmi2l.323: under `SYMBOLIC_WRITE_ADDRESSES` the same `StoreG`
/// concretizes to a candidate set. Each candidate gets
/// `ITE(guard && addr == a, data, mem[a])` buffered for Rust memory; it used
/// to go to the host's full store, which a proxy-backed host drops.
#[test]
fn storeg_symbolic_guard_multiwrite_guards_every_candidate() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp_rust_memory(&ctx);
    interp.set_concretizer(crate::symbolic::concretize::AddressConcretizer {
        symbolic_write_addresses: true,
        ..crate::symbolic::concretize::AddressConcretizer::new()
    });

    let guard = RustBV::symbol("storeg_mw_hi", 1);
    let idx = RustBV::symbol("storeg_mw_idx", 64);
    ctx.assume_true(&idx.ule(&RustBV::concrete(3, 64)));
    let addr = RustBV::concrete(0x4000, 64).add(&idx.mul(&RustBV::concrete(4, 64)));
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I1, IRType::I64]);
    interp.temps = vec![Some(guard.clone()), Some(addr)];

    let storeg = IRStmt::StoreG {
        addr: Box::new(IRExpr::RdTmp(1)),
        data: Box::new(IRExpr::Const(IRConst::U32(0xcafe_0003))),
        guard: Box::new(IRExpr::RdTmp(0)),
        endness: Endness::Little,
    };
    with_proxy_callbacks(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &storeg, &irsb)
            .expect("guarded store with symbolic guard and multi-candidate address");
    });

    for k in 0..4u64 {
        let (in_block, flushed) =
            with_proxy_callbacks(|cb| read_back(&mut interp, cb, 0x4000 + 4 * k, 4));
        for cell in [&in_block, &flushed] {
            let eval_under = |hi: bool, at: u64| {
                let at_idx = idx.eq(&RustBV::concrete(u128::from(at), 64));
                eval_assuming(&ctx, cell, &[(&guard, hi), (&at_idx, true)])
            };
            assert_eq!(eval_under(true, k), Some(0xcafe_0003), "guard true, idx={k}: stored");
            assert_eq!(eval_under(false, k), Some(0), "guard false, idx={k}: unchanged");
            assert_eq!(eval_under(true, (k + 1) % 4), Some(0), "other index: unchanged");
        }
    }
    ctx.push();
    ctx.assume_true(&guard);
    let mut vals = ctx.eval_upto(&idx, 8);
    ctx.pop();
    vals.sort_unstable();
    assert_eq!(vals, vec![0, 1, 2, 3], "a complete candidate set pins nothing");
}

/// CAS variant (angr-5mnx3.23, angr-vmi2l.168): `cas_store_symbolic_data`
/// writes back a computed `RustBV` (the success/failure ITE) that cannot be
/// re-expressed as an `IRExpr`. It must land where a later in-block load and
/// the flushed memory both see it.
#[test]
fn cas_symbolic_writeback_reads_back_for_rust_memory() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp_rust_memory(&ctx);
    let irsb = make_irsb_with_temps(0x1000, &[]);

    let swap = RustBV::symbol("cas_swap", 1);
    let data_bv = swap.ite(
        &RustBV::concrete(0x5eed_0001, 32),
        &RustBV::concrete(0x0a0a_0a00, 32),
    );
    let addr_expr = IRExpr::Const(IRConst::U64(0x4020));

    with_proxy_callbacks(|cb| {
        interp
            .cas_store_symbolic_data(cb, &addr_expr, &data_bv, &irsb)
            .expect("CAS symbolic writeback");
        let (in_block, flushed) = read_back(&mut interp, cb, 0x4020, 4);
        for value in [&in_block, &flushed] {
            assert_eq!(eval_assuming(&ctx, value, &[(&swap, true)]), Some(0x5eed_0001));
            assert_eq!(eval_assuming(&ctx, value, &[(&swap, false)]), Some(0x0a0a_0a00));
        }
    });
}

/// Multi-address variant (angr-5mnx3.23, angr-vmi2l.168): a symbolic-address
/// store that concretizes to several candidates within `MAX_ITE_ADDRS` stores
/// `ITE(addr == a_i, data, mem[a_i])` at each candidate, and each reads back
/// in-block and after the flush.
#[test]
fn ite_store_reads_back_every_candidate_for_rust_memory() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp_rust_memory(&ctx);

    let addr_expr = RustBV::symbol("ite_store_addr", 64);
    let data_val = RustBV::concrete(0xcafe_0001, 32);
    let addrs = [0x4010u64, 0x4020u64];

    with_proxy_callbacks(|cb| {
        interp
            .build_ite_store_from_callbacks(cb, &addrs, &addr_expr, &data_val, None)
            .expect("per-candidate ITE store");
        for addr in addrs {
            let at = addr_expr.eq(&RustBV::concrete(u128::from(addr), 64));
            let (in_block, flushed) = read_back(&mut interp, cb, addr, 4);
            for value in [&in_block, &flushed] {
                assert_eq!(eval_assuming(&ctx, value, &[(&at, true)]), Some(0xcafe_0001));
                assert_eq!(eval_assuming(&ctx, value, &[(&at, false)]), Some(0));
            }
        }
    });
}

/// angr-vmi2l.168: a symbolic-guard `LoadG` at a symbolic address reads every
/// candidate, as a plain `Load` does. `idx_constraint` shapes the candidate
/// set (`Multiple` for a sparse set, `Strided` for a range). The `Multiple`
/// shape used to read only the first candidate for every index, and the
/// `Strided` one read the host's shadow, missing the same-block store.
fn loadg_symbolic_addr_reads_every_candidate(name: &str, sparse: bool) {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp_rust_memory(&ctx);

    let guard = RustBV::symbol(format!("{name}_hi"), 1);
    let idx = RustBV::symbol(format!("{name}_idx"), 64);
    let at = |k: u64| idx.eq(&RustBV::concrete(u128::from(k), 64));
    let indices: Vec<u64> = if sparse {
        ctx.assume_true(&at(0).or(&at(1)).or(&at(3)));
        vec![0, 1, 3]
    } else {
        ctx.assume_true(&idx.ule(&RustBV::concrete(3, 64)));
        vec![0, 1, 2, 3]
    };
    let addr = RustBV::concrete(0x4000, 64).add(&idx.mul(&RustBV::concrete(4, 64)));
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I1, IRType::I64, IRType::I32]);
    interp.temps = vec![Some(guard.clone()), Some(addr), None];

    let stores: Vec<IRStmt> = (0..4u64)
        .map(|k| IRStmt::Store {
            addr: IRExpr::Const(IRConst::U64(0x4000 + 4 * k)),
            data: IRExpr::Const(IRConst::U32(0x0d0d_0d00 + k as u32)),
            endness: Endness::Little,
        })
        .collect();
    let loadg = IRStmt::LoadG {
        endness: Endness::Little,
        cvt: IRLoadGOp::Identity,
        dst: 2,
        addr: Box::new(IRExpr::RdTmp(1)),
        alt: Box::new(IRExpr::Const(IRConst::U32(0x99))),
        guard: Box::new(IRExpr::RdTmp(0)),
    };
    with_proxy_callbacks(|cb| {
        for stmt in stores.iter().chain([&loadg]) {
            interp
                .execute_stmt_with_callbacks(cb, stmt, &irsb)
                .expect("stores, then guarded load at a symbolic address");
        }
    });
    let loaded = interp.temps[2].clone().expect("LoadG wrote its temp");
    for k in indices {
        assert_eq!(
            eval_assuming(&ctx, &loaded, &[(&guard, true), (&at(k), true)]),
            Some(0x0d0d_0d00 + u128::from(k)),
            "guard true, idx={k}: reads that candidate"
        );
        assert_eq!(
            eval_assuming(&ctx, &loaded, &[(&guard, false), (&at(k), true)]),
            Some(0x99),
            "guard false, idx={k}: the alt value"
        );
    }
}

#[test]
fn loadg_symbolic_addr_multiple_reads_every_candidate() {
    loadg_symbolic_addr_reads_every_candidate("loadg_multi", true);
}

#[test]
fn loadg_symbolic_addr_strided_reads_every_candidate() {
    loadg_symbolic_addr_reads_every_candidate("loadg_strided", false);
}

/// angr-6cp06.88: a load only *partially* covered by the pending-store buffer
/// must splice the buffered bytes over the lower layers' answer. `try_load`
/// needs one store covering the whole load and `try_load_assembled` needs every
/// byte buffered, so a 1-byte store followed by a 2-byte read-back missed both
/// and the flushed-store layer answered for *both* bytes — silently undoing
/// the buffered write.
#[test]
fn partially_buffered_load_overlays_pending_bytes_on_concrete_lower_layer() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    assert!(!interp.use_rust_memory);
    // Lower layer (a previously flushed store from an earlier block).
    interp.all_flushed_stores.insert(0x4000, vec![0x11, 0x22]);
    // In-flight store covering only the low byte of the coming 2-byte load.
    interp.pending_stores.push(0x4000, vec![0xaa], interp.registers.is_big_endian());

    let irsb = make_irsb_with_temps(0x1000, &[IRType::I16]);
    interp.temps.resize(1, None);
    let load = IRStmt::WrTmp {
        tmp: 0,
        data: IRExpr::Load {
            addr: Box::new(IRExpr::Const(IRConst::U64(0x4000))),
            ty: IRType::I16,
            endness: Endness::Little,
        },
    };

    with_test_host(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &load, &irsb)
            .expect("partially buffered load should resolve");
    });

    assert_eq!(
        interp.temps[0].as_ref().and_then(|bv| bv.as_u64()),
        Some(0x22aa),
        "buffered byte 0xaa must survive; only the uncovered byte comes from \
         the flushed store"
    );
}

/// The symbolic-lower-layer half of the same fix: when the layers below answer
/// with a symbolic BV the overlay is an Extract/Concat splice, not a byte
/// patch, so the uncovered lane must stay symbolic.
#[test]
fn partially_buffered_load_overlays_pending_bytes_on_symbolic_lower_layer() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let sym = RustBV::symbol("flushed", 16);
    interp.all_flushed_symbolic_stores.insert(0x4000, sym);
    interp.pending_stores.push(0x4000, vec![0xaa], interp.registers.is_big_endian());

    let irsb = make_irsb_with_temps(0x1000, &[IRType::I16]);
    interp.temps.resize(1, None);
    let load = IRStmt::WrTmp {
        tmp: 0,
        data: IRExpr::Load {
            addr: Box::new(IRExpr::Const(IRConst::U64(0x4000))),
            ty: IRType::I16,
            endness: Endness::Little,
        },
    };

    with_test_host(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &load, &irsb)
            .expect("partially buffered load should resolve");
    });

    let result = interp.temps[0].as_ref().expect("temp written");
    assert_eq!(result.width(), 16);
    assert!(
        result.is_symbolic(),
        "the uncovered high byte must remain the flushed symbol"
    );
    assert_eq!(
        result.extract(7, 0).as_u64(),
        Some(0xaa),
        "the buffered byte must win over the symbolic lower layer"
    );
}

// ---------------------------------------------------------------------------
// AVOID_MULTIVALUED_WRITES (angr-6cp06.67)
// ---------------------------------------------------------------------------

/// Concretizer with `avoid_multivalued_writes` set to `on` and nothing else
/// changed from the default.
fn mv_write_concretizer(on: bool) -> crate::symbolic::concretize::AddressConcretizer {
    let mut c = crate::symbolic::concretize::AddressConcretizer::new();
    c.configure_strategies(false, None, None, false, false, on);
    c
}

/// Interpreter owning a mapped page at 0x1000 with `backer` planted at the
/// start of it, and `avoid_multivalued_writes` set to `on`.
fn mv_write_interp<'a>(ctx: &'a SymContext, on: bool, backer: u128) -> VEXInterpreter<'a> {
    let mut interp = new_interp(ctx);
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, PAGE_SIZE, crate::memory::Permission::RWX);
    mem.store_concrete(0x1000, RustBV::concrete(backer, 64))
        .expect("plant backer");
    interp.set_rust_memory(mem);
    interp.set_concretizer(mv_write_concretizer(on));
    interp
}

/// Commit any buffered stores and read 0x1000 back out of `rust_memory`.
/// Reads the memory directly rather than through an interpreter load path, so
/// the verification never re-enters the code under test.
fn mv_write_read_back(interp: &mut VEXInterpreter<'_>) -> Option<u64> {
    interp.flush_stores_to_rust_memory();
    let mem = interp.take_rust_memory().expect("interpreter owns memory");
    mem.load_concrete(0x1000, 8)
        .expect("read-back must succeed")
        .as_u64()
}

/// A 64-bit BVS pinned by constraint to 0x1000: symbolic enough for
/// `should_avoid_multivalued_write` to fire, single-valued enough that with the
/// option off the write concretizes to exactly one place.
fn mv_write_addr(ctx: &SymContext, name: &str) -> RustBV {
    let bv = RustBV::symbol(name, 64);
    ctx.assume_true(&bv.eq(&RustBV::concrete(0x1000, 64)));
    assert!(bv.as_u64().is_none(), "address must stay symbolic");
    bv
}

/// `try_rust_memory_store`'s AVOID_MULTIVALUED_WRITES gate returns
/// `Ok(true)` — "handled" — without concretizing or buffering anything, so the
/// backer survives. With the option off the same store lands.
#[test]
fn try_rust_memory_store_drops_the_write_under_avoid_multivalued_writes() {
    let callbacks = TestHost::default();
    let ctx = SymContext::new_mock();
    let backer: u128 = 0xDEAD_BEEF_CAFE_BABE;
    let stored: u128 = 0x0102_0304_0506_0708;

    let mut off = mv_write_interp(&ctx, false, backer);
    let addr = mv_write_addr(&ctx, "mvw_off");
    let handled = off
        .try_rust_memory_store(&callbacks, &addr, &RustBV::concrete(stored, 64), 8, None)
        .expect("store with the option off must succeed");
    assert!(handled, "option off: rust memory must handle the store");
    assert_eq!(
        mv_write_read_back(&mut off),
        Some(stored as u64),
        "option off: the pinned address must concretize and take the write"
    );

    let mut on = mv_write_interp(&ctx, true, backer);
    let addr = mv_write_addr(&ctx, "mvw_on");
    let handled = on
        .try_rust_memory_store(&callbacks, &addr, &RustBV::concrete(stored, 64), 8, None)
        .expect("store with the option on must succeed");
    assert!(
        handled,
        "option on: the drop is still 'handled' — it must not fall through to Python"
    );
    assert_eq!(
        mv_write_read_back(&mut on),
        Some(backer as u64),
        "option on: the write must be dropped, leaving the backer intact"
    );
}

/// `handle_symbolic_store` carries a second copy of the same gate, placed
/// ahead of `flush_stores` and `concretize_cached_write`. Covered separately
/// from `try_rust_memory_store` because deleting either copy leaves the other
/// green.
///
/// This site is only reachable with `use_rust_memory == false` (see
/// `handle_symbolic_store_strided_invalidates_cached_candidate_only`), so the
/// OFF baseline uses that test's idiom: with no `memory_store` callback
/// registered the store dispatch the gate would have skipped errors out. ON
/// returning `Ok(())` on the identical setup is therefore proof the write was
/// dropped before ever reaching that dispatch — the gate's whole contract.
#[test]
fn handle_symbolic_store_drops_the_write_under_avoid_multivalued_writes() {
    let ctx = SymContext::new_mock();
    let stored = RustBV::concrete(0x0102_0304, 32);

    let mut off = new_interp(&ctx);
    off.set_concretizer(mv_write_concretizer(false));
    let off_addr = mv_write_addr(&ctx, "mvhs_off");
    with_test_host(|cb| {
        let res = off.handle_symbolic_store(cb, &off_addr, &stored, 4);
        assert!(
            res.is_err(),
            "option off: the store must reach the unregistered store dispatch, got {res:?}"
        );
    });

    let mut on = new_interp(&ctx);
    on.set_concretizer(mv_write_concretizer(true));
    let on_addr = mv_write_addr(&ctx, "mvhs_on");
    with_test_host(|cb| {
        on.handle_symbolic_store(cb, &on_addr, &stored, 4)
            .expect("option on: the dropped store must report success, not an error");
    });
}


/// angr-vmi2l.324: under `SYMBOLIC_WRITE_ADDRESSES` a plain store at
/// `base + idx` with `idx <= 63` concretizes to more than `MAX_ITE_ADDRS`
/// candidates, and every one of them takes the write, as Python's `Range(128)`
/// conditional store does. `base` is a mapped page or one neither side holds.
fn wide_symbolic_store_reaches_every_candidate(base: u64) {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp_rust_memory(&ctx);
    let mut conc = crate::symbolic::concretize::AddressConcretizer::new();
    conc.configure_strategies(false, None, None, true, false, false);
    interp.set_concretizer(conc);

    let idx = RustBV::symbol("wide_store_idx", 64);
    ctx.assume_true(&idx.ule(&RustBV::concrete(63, 64)));
    let addr = RustBV::concrete(u128::from(base), 64).add(&idx);
    let irsb = make_irsb_with_temps(0x1000, &[IRType::I64]);
    interp.temps = vec![Some(addr)];
    let store = IRStmt::Store {
        addr: IRExpr::RdTmp(0),
        data: IRExpr::Const(IRConst::U8(1)),
        endness: Endness::Little,
    };
    // A page-universe snapshot with no pages in it, as a
    // ZERO_FILL_UNCONSTRAINED_MEMORY seed gives: the host holds none of them.
    let cb = TestHost {
        symbolic_value_store: SymbolicValueStore::Accept,
        serve_zero_loads: true,
        host_pages: Some(Default::default()),
        ..TestHost::default()
    };
    {
        let cb = &cb;
        interp
            .execute_stmt_with_callbacks(cb, &store, &irsb)
            .expect("wide symbolic store");
        for k in [0u64, 5, 63] {
            let at = idx.eq(&RustBV::concrete(u128::from(k), 64));
            let (in_block, flushed) = read_back(&mut interp, cb, base + k, 1);
            for value in [&in_block, &flushed] {
                assert_eq!(eval_assuming(&ctx, value, &[(&at, true)]), Some(1), "idx {k}");
            }
        }
    }
}

#[test]
fn wide_symbolic_store_reaches_every_candidate_mapped() {
    wide_symbolic_store_reaches_every_candidate(0x4000);
}

#[test]
fn wide_symbolic_store_reaches_every_candidate_unmapped() {
    wide_symbolic_store_reaches_every_candidate(0x10000);
}
