//! Tests for VEX statement execution (extracted from statements.rs).
use super::*;
use crate::memory::PAGE_SIZE;
use crate::vex::ir::{Endness, IRType, MBusEvent};

fn new_interp(ctx: &SymContext) -> VEXInterpreter<'_> {
    VEXInterpreter::new(VexArch::AMD64, ctx)
}

/// An interpreter whose concretizer has `SYMBOLIC_WRITE_ADDRESSES` on, so
/// symbolic-address writes keep the Range strategy and resolve to a candidate
/// set. The default chain is Max-only for unannotated addresses and yields
/// `Single` — see `AddressConcretizer::write_range_applies` (angr-9ke6b.194).
fn new_interp_multiwrite(ctx: &SymContext) -> VEXInterpreter<'_> {
    let mut interp = new_interp(ctx);
    interp.set_concretizer(crate::concretize::AddressConcretizer {
        symbolic_write_addresses: true,
        ..crate::concretize::AddressConcretizer::new()
    });
    interp
}

fn make_irsb_with_temps(addr: u64, temp_types: &[IRType]) -> IRSB {
    let mut irsb = IRSB::new(addr, VexArch::AMD64);
    irsb.statements.push(IRStmt::IMark {
        addr,
        len: 4,
        delta: 0,
    });
    for ty in temp_types {
        irsb.tyenv.new_temp(*ty);
    }
    irsb
}

/// Initialize Python once for tests that need to call execute_stmt_with_callbacks.
fn with_python<F, R>(f: F) -> R
where
    F: FnOnce(&PythonCallbacks) -> R,
{
    Python::initialize();
    let callbacks = PythonCallbacks::new();
    Python::attach(|_py| f(&callbacks))
}

#[test]
fn noop_returns_continue() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x1000, &[]);
    with_python(|cb| {
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
    with_python(|cb| {
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
    with_python(|cb| {
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
    with_python(|cb| {
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
    with_python(|cb| {
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
    with_python(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("put");
    });
    let val = interp.registers.get(16, 8, &ctx);
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
    with_python(|cb| {
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
    with_python(|cb| {
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
    with_python(|cb| {
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
    with_python(|cb| {
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
    with_python(|cb| {
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
    with_python(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("store should buffer without erroring");
    });
    // Buffered into pending_stores keyed by address.
    let data = interp
        .pending_stores
        .try_load(0x4000, 4)
        .expect("pending store at 0x4000");
    assert_eq!(data, &[0xef, 0xbe, 0xad, 0xde]);
}

/// angr-1c88c gap 3/7: drive `expressions.rs::build_ite_store_from_callbacks`.
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
    use pyo3::types::{PyDict, PyList};

    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    // Site is below the use_rust_memory gate's fallback; the test default is
    // already callback-mode, but assert it to document the precondition.
    assert!(!interp.use_rust_memory);

    Python::initialize();
    Python::attach(|py| {
        // rustbv_to_claripy imports claripy; skip when it is not importable.
        if py.import("claripy").is_err() {
            return;
        }

        let globals = PyDict::new(py);
        py.run(
            c"_rec = []
def store_cb(addr, ast):
    _rec.append((addr, ast.op, ast.size()))
def load_batch(loads):
    return [(bytes(sz), False, None) for (_a, sz) in loads]
",
            Some(&globals),
            None,
        )
        .expect("define recorder callbacks");

        let store_cb = globals.get_item("store_cb").unwrap().unwrap();
        let load_batch = globals.get_item("load_batch").unwrap().unwrap();

        let mut cb = PythonCallbacks::new();
        cb.set_memory_store_symbolic_value(store_cb.unbind());
        cb.set_memory_load_batch(load_batch.unbind());
        assert!(cb.has_memory_store_symbolic_value());

        let addr_expr = RustBV::symbolic(&ctx, "store_addr", 64);
        let data_val = RustBV::symbolic(&ctx, "store_data", 64);
        let addrs = [0x1000u64, 0x2000u64];

        interp
            .build_ite_store_from_callbacks(&cb, &addrs, &addr_expr, &data_val)
            .expect("build_ite_store_from_callbacks");

        let rec = globals.get_item("_rec").unwrap().unwrap();
        let rec = rec.cast::<PyList>().unwrap();
        assert_eq!(rec.len(), 2, "one symbolic-value store per candidate addr");

        let mut seen_addrs = Vec::new();
        for item in rec.iter() {
            let tup = item.cast::<pyo3::types::PyTuple>().unwrap();
            let addr: u64 = tup.get_item(0).unwrap().extract().unwrap();
            let op: String = tup.get_item(1).unwrap().extract().unwrap();
            let size: u32 = tup.get_item(2).unwrap().extract().unwrap();
            // Each stored value is the in-Rust ITE chain marshalled to claripy.
            assert_eq!(op, "If", "stored value must be the ITE chain");
            assert_eq!(size, 64, "ITE store width matches the data width");
            seen_addrs.push(addr);
        }
        seen_addrs.sort_unstable();
        assert_eq!(seen_addrs, vec![0x1000, 0x2000]);
    });
}

#[test]
fn loadg_unknown_cvt_surfaces_invalid_ir() {
    use crate::vex::ir::IRLoadGOp;
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
    with_python(|cb| {
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
    with_python(|cb| {
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
        .insert(0x2000, RustBV::symbolic(&ctx, "p", 64));
    interp
        .all_flushed_symbolic_stores
        .insert(0x2000, RustBV::symbolic(&ctx, "f", 64));
    // 8-byte concrete store at 0x2000 fully overwrites both shadows.
    interp.evict_overlapping_symbolic_stores(0x2000, 8);
    assert!(!interp.pending_symbolic_stores.contains_key(&0x2000));
    assert!(!interp.all_flushed_symbolic_stores.contains_key(&0x2000));
}

#[test]
fn evict_overlapping_symbolic_stores_keeps_disjoint_entries() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    // [0x3000,0x3004) and [0x3008,0x300c).
    interp
        .pending_symbolic_stores
        .insert(0x3000, RustBV::symbolic(&ctx, "a", 32));
    interp
        .pending_symbolic_stores
        .insert(0x3008, RustBV::symbolic(&ctx, "b", 32));
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
    let data_bv = RustBV::symbolic(&ctx, "cas_data", 32);
    let call_irsb = make_irsb_with_temps(0x1010, &[]);

    with_python(|cb| {
        // Bare `PythonCallbacks::new()` has no memory_store_symbolic_value, so
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
    let data_bv = RustBV::symbolic(&ctx, "cas_sym", 32);
    let call_irsb = make_irsb_with_temps(0x1010, &[]);

    with_python(|cb| {
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

// Symbolic-address branch: the CAS-target address itself is unresolved at
// eval time. `cas_store_symbolic_data` concretizes it via
// `concretize_cached_write` (the same helper `handle_symbolic_store` uses for
// the ordinary Store path) and invalidates through `invalidate_code_on_store`
// before dispatching the store. The address is pinned to a single solution
// via `pin_fallback_addr` -- the same `assume_true(addr == chosen)` idiom
// `concretize_write`'s own Max-fallback uses internally -- so the test gets a
// deterministic `Single` result without depending on the concretizer's
// range-enumeration heuristics for an otherwise-unconstrained value.
#[cfg(feature = "vex-engine-z3")]
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
    let addr_bv = RustBV::symbolic(&ctx, "cas_sym_addr", 64);
    crate::concretize::pin_fallback_addr(&ctx, &addr_bv, 0x2010);
    interp.temps = vec![Some(addr_bv)];

    let addr_expr = IRExpr::RdTmp(0);
    let data_bv = RustBV::symbolic(&ctx, "cas_sym_data", 32);
    let call_irsb = make_irsb_with_temps(0x2010, &[]);

    with_python(|cb| {
        let res = interp.cas_store_symbolic_data(cb, &addr_expr, &data_bv, &call_irsb);
        // No memory_store_symbolic_full callback is registered on this bare
        // PythonCallbacks::new(), so the store dispatch itself errors out --
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
#[cfg(feature = "vex-engine-z3")]
#[test]
fn handle_symbolic_store_strided_invalidates_cached_candidate_only() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp_multiwrite(&ctx);
    interp.add_concrete_memory(0x5000, vec![0u8; 0x2000]);

    // idx in {0, 1, 2} -> addr in {0x5010, 0x5020, 0x5030} (stride 0x10).
    let idx = RustBV::symbolic(&ctx, "idx_strided", 64);
    let bound = idx.ult(&RustBV::concrete(3, 64), &ctx);
    ctx.assume_true(&bound);
    let stride = RustBV::concrete(0x10, 64);
    let base = RustBV::concrete(0x5010, 64);
    let addr = base.add(&idx.mul(&stride, &ctx), &ctx);

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

    let data_val = RustBV::symbolic(&ctx, "store_data", 32);
    with_python(|cb| {
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
#[cfg(feature = "vex-engine-z3")]
#[test]
fn handle_symbolic_store_multiple_invalidates_cached_candidate_only() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp_multiwrite(&ctx);
    interp.add_concrete_memory(0x6000, vec![0u8; 0x2000]);

    let idx = RustBV::symbolic(&ctx, "idx_multiple", 64);
    let eq0 = idx.eq(&RustBV::concrete(0, 64), &ctx);
    let eq1 = idx.eq(&RustBV::concrete(1, 64), &ctx);
    let eq3 = idx.eq(&RustBV::concrete(3, 64), &ctx);
    let cond = eq0.or(&eq1, &ctx).or(&eq3, &ctx);
    ctx.assume_true(&cond);
    let base = RustBV::concrete(0x6000, 64);
    let addr = base.add(&idx, &ctx);

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

    let data_val = RustBV::symbolic(&ctx, "store_data_multi", 32);
    with_python(|cb| {
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

    with_python(|cb| {
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

/// angr-9ke6b.83, Rust-memory variant: with `use_rust_memory` on (the
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

    with_python(|cb| {
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

/// angr-9ke6b.83, `Single` concretization arm: a symbolic address pinned to one
/// solution takes `resolve_loadg_load`'s `ConcretizationResult::Single` branch,
/// which must go through the same layered read (`load_layered_at`) rather than
/// straight to the Python callback.
// Needs a real solver: the `Single` branch is only taken when concretization
// finds exactly one solution, which the no-Z3 mock context never does
// (angr-9ke6b.236, bd memory `vex-engine-z3-test-gate-invariant`).
#[cfg(feature = "vex-engine-z3")]
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
    let sym_addr = RustBV::symbolic(&ctx, "loadg_addr", 64);
    ctx.assume_true(&sym_addr.eq(&RustBV::concrete(0x4020, 64), &ctx));
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

    with_python(|cb| {
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
    use pyo3::types::{PyDict, PyList};

    let ctx = SymContext::new_mock();
    // X86 (32-bit, non-stack address) is the only configuration that arms the
    // `memory_store_symbolic_value` path in `handle_concrete_store`.
    let mut interp = VEXInterpreter::new(VexArch::X86, &ctx);

    Python::initialize();
    Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(
            c"_rec = []
def sym_store_cb(addr, ast):
    raise RuntimeError('symbolic store unavailable')
def store_cb(addr, data):
    _rec.append((addr, bytes(data)))
",
            Some(&globals),
            None,
        )
        .expect("define callbacks");

        let mut cb = PythonCallbacks::new();
        cb.set_memory_store_symbolic_value(
            globals.get_item("sym_store_cb").unwrap().unwrap().unbind(),
        );
        cb.set_memory_store(globals.get_item("store_cb").unwrap().unwrap().unbind());

        // 256 bits = 32 bytes > MAX_CONCRETE_CHUNK, so no u128 witness exists.
        let data_val = RustBV::symbolic(&ctx, "wide_store_data", 256);
        let err = interp
            .handle_concrete_store(&cb, 0x10_0000, data_val, 32)
            .expect_err("unconcretizable symbolic store must fail, not zero-fill");
        assert!(
            matches!(err, CbExecutionError::Callback(_)),
            "the original symbolic-store callback error is propagated, got {err:?}"
        );

        let rec = globals.get_item("_rec").unwrap().unwrap();
        assert_eq!(
            rec.cast::<PyList>().unwrap().len(),
            0,
            "no byte-level store may reach memory when concretization failed"
        );
    });
}

/// Converse of the test above: when the value *is* concretizable, the fallback
/// still writes the concretized bytes little-endian through `memory_store`.
/// Guards against "fix the zero-fill by disabling the fallback entirely".
///
/// Z3-gated (angr-c7xno.100): the premise is a value that is symbolic but
/// *concretizable* — `data_val` is pinned to `0xdeadbeef` by a constraint, and
/// only a real solver can read that back through `eval`. The no-z3 mock has no
/// model at all, so it takes the sibling test's unconcretizable path instead;
/// there is no faithful mock arm to write, short of reimplementing constraint
/// solving.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn symbolic_store_fallback_writes_concretized_bytes_when_eval_succeeds() {
    use pyo3::types::{PyDict, PyList};

    let ctx = SymContext::new_mock();
    let mut interp = VEXInterpreter::new(VexArch::X86, &ctx);

    let data_val = RustBV::symbolic(&ctx, "store_data", 32);
    ctx.assume_true(&data_val.eq(&RustBV::concrete(0xdead_beef, 32), &ctx));

    Python::initialize();
    Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(
            c"_rec = []
def sym_store_cb(addr, ast):
    raise RuntimeError('symbolic store unavailable')
def store_cb(addr, data):
    _rec.append((addr, bytes(data)))
",
            Some(&globals),
            None,
        )
        .expect("define callbacks");

        let mut cb = PythonCallbacks::new();
        cb.set_memory_store_symbolic_value(
            globals.get_item("sym_store_cb").unwrap().unwrap().unbind(),
        );
        cb.set_memory_store(globals.get_item("store_cb").unwrap().unwrap().unbind());

        interp
            .handle_concrete_store(&cb, 0x10_0000, data_val, 4)
            .expect("concretizable symbolic store falls back cleanly");

        let rec = globals.get_item("_rec").unwrap().unwrap();
        let rec = rec.cast::<PyList>().unwrap();
        assert_eq!(rec.len(), 1, "exactly one byte-level fallback store");
        let tup = rec.get_item(0).unwrap();
        let tup = tup.cast::<pyo3::types::PyTuple>().unwrap();
        let addr: u64 = tup.get_item(0).unwrap().extract().unwrap();
        let data: Vec<u8> = tup.get_item(1).unwrap().extract().unwrap();
        assert_eq!(addr, 0x10_0000);
        assert_eq!(
            data,
            vec![0xef, 0xbe, 0xad, 0xde],
            "little-endian concretization"
        );
    });
}

/// angr-c7xno.44: a dirty helper that neither Rust nor Python models must
/// route the block to Python (`NeedPythonFallback`), not fabricate a fresh
/// unconstrained symbolic into the result tmp. Same policy — and same opt-in
/// `ANGR_RUST_FABRICATE_UNSUPPORTED_IROP` escape hatch — as
/// `VEXInterpreter::vex_op_fallback` / `eval_ccall`. Default (gate unset)
/// behavior: fabricating silently diverges, because every downstream
/// condition over the unconstrained tmp explores both branches.
#[test]
fn unmodelled_dirty_call_routes_to_python_not_fabricate() {
    use crate::vex::ir::{DirtyFx, IRCallee, IRDirty};
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x3000, &[IRType::I64]);
    let dirty = IRDirty {
        cee: IRCallee {
            name: "amd64g_dirtyhelper_NOT_A_REAL_HELPER".to_string(),
            addr: 0,
            mcx_mask: 0,
        },
        guard: None,
        tmp: Some(0),
        mFx: DirtyFx::None,
        mAddr: None,
        mSize: 0,
        nFxState: 0,
        args: vec![],
    };
    with_python(|cb| {
        // `with_python` builds a bare PythonCallbacks: no dirty_call callback,
        // which is the branch under test (production always registers one via
        // rust_manager.py::_setup_callbacks).
        assert!(!cb.has_dirty_call());
        let res = interp.execute_stmt_with_callbacks(cb, &IRStmt::Dirty(dirty), &irsb);
        // `StmtResult` is not `Debug`, so describe the outcome by hand.
        assert!(
            matches!(res, Err(CbExecutionError::NeedPythonFallback(_))),
            "unmodelled dirty call must route to Python, got {}",
            match &res {
                Ok(_) => "Ok(..)".to_string(),
                Err(e) => format!("{e:?}"),
            }
        );
        assert_eq!(
            interp.stats.vex_bypass_fabricate_count, 0,
            "unmodelled dirty call fabricated instead of routing to Python"
        );
        assert!(
            interp.temps.first().is_none_or(Option::is_none),
            "no fabricated value may be written into the result tmp"
        );
    });
}

/// angr-03vl4.33: a dirty call naming a result temp that is absent from the
/// block's tyenv is malformed IR. `handle_dirty_call` used to default the
/// width to 64, which silently produced a wrong-width tmp on both the
/// native-handler and Python-callback write paths; it must now fail with
/// `InvalidIR` like the `IRStmt::LLSC` arm does for the same condition.
#[test]
fn dirty_call_result_tmp_missing_from_tyenv_is_invalid_ir() {
    use crate::vex::ir::{DirtyFx, IRCallee, IRDirty};
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    // tyenv declares exactly one temp (t0); the dirty call names t5.
    let irsb = make_irsb_with_temps(0x3000, &[IRType::I64]);
    let dirty = IRDirty {
        cee: IRCallee {
            name: "amd64g_dirtyhelper_NOT_A_REAL_HELPER".to_string(),
            addr: 0,
            mcx_mask: 0,
        },
        guard: None,
        tmp: Some(5),
        mFx: DirtyFx::None,
        mAddr: None,
        mSize: 0,
        nFxState: 0,
        args: vec![],
    };
    with_python(|cb| {
        let res = interp.execute_stmt_with_callbacks(cb, &IRStmt::Dirty(dirty), &irsb);
        match &res {
            Err(CbExecutionError::InvalidIR(msg)) => {
                assert!(
                    msg.contains("not in tyenv") && msg.contains('5'),
                    "InvalidIR message should name the missing temp, got {msg}"
                );
            }
            Ok(_) => panic!("missing tyenv entry silently accepted"),
            Err(e) => panic!("expected InvalidIR, got {e:?}"),
        }
        assert_eq!(
            interp.stats.vex_bypass_fabricate_count, 0,
            "the tyenv lookup must fail before any fabricate/fallback path"
        );
    });
}

// ---------------------------------------------------------------------------
// angr-5mnx3.23: every `memory_store_symbolic_value` dispatch must buffer.
// ---------------------------------------------------------------------------

/// Callbacks in the production callback-memory-proxy configuration: the
/// `memory_store_symbolic_value` slot is wired to a Python function that does
/// nothing — exactly what `rust_manager.py::_cb_memory_store_symbolic_value`
/// is once `memory_is_rust_proxy` is on (`state.memory` *is* Rust memory, so
/// the callback deliberately declines to re-enter it) — and the proxy flag is
/// set. A dispatch that reaches this callback without first calling
/// `buffer_store_for_rust_memory` therefore lands nowhere at all.
///
/// Marshalling a `RustBV` to the callback needs claripy; skip when it is not
/// importable, mirroring `build_ite_store_invokes_per_addr_ite_callbacks`.
fn with_proxy_callbacks<F: FnOnce(&PythonCallbacks)>(f: F) {
    use pyo3::types::PyDict;

    Python::initialize();
    Python::attach(|py| {
        if py.import("claripy").is_err() {
            return;
        }
        let globals = PyDict::new(py);
        py.run(
            c"def proxy_noop_store(addr, ast):
    pass
def load_batch(loads):
    return [(bytes(sz), False, None) for (_a, sz) in loads]
def load_cb(addr, size):
    return (bytes(size), False, None)
",
            Some(&globals),
            None,
        )
        .expect("define proxy no-op callbacks");
        let store_cb = globals.get_item("proxy_noop_store").unwrap().unwrap();
        let load_batch = globals.get_item("load_batch").unwrap().unwrap();

        let mut cb = PythonCallbacks::new();
        cb.set_memory_store_symbolic_value(store_cb.unbind());
        cb.set_memory_load_batch(load_batch.unbind());
        cb.set_memory_load(globals.get_item("load_cb").unwrap().unwrap().unbind());
        cb.py_set_memory_is_rust_proxy(true);
        assert!(cb.has_memory_store_symbolic_value());
        assert!(cb.memory_is_rust_proxy());
        f(&cb);
    });
}

/// Build an interpreter that owns its memory (the production configuration —
/// `exploration/step_core.rs` installs the state's memory), so
/// `buffer_store_for_rust_memory`'s `use_rust_memory` half of the gate holds.
fn new_interp_rust_memory(ctx: &SymContext) -> VEXInterpreter<'_> {
    let mut interp = new_interp(ctx);
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map_page(0x4000u64, vec![0u8; PAGE_SIZE as usize], Permission::RW);
    interp.set_rust_memory(mem);
    interp
}

/// angr-5mnx3.23: a `StoreG` with a *symbolic* guard at a concrete address
/// builds `ITE(guard, new, current)` and hands it to
/// `memory_store_symbolic_value`. Under the proxy gate that callback absorbs
/// nothing, so the ITE must also be buffered in `pending_symbolic_stores` —
/// otherwise the predicated store (ARM conditional stores, masked SIMD) is
/// silently dropped.
// Needs a real solver: `classify_guard` must return `Symbolic`, which requires
// `check_branch_feasibility` to find both branches satisfiable
// (bd memory `vex-engine-z3-test-gate-invariant`).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn storeg_symbolic_guard_buffers_for_rust_memory() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp_rust_memory(&ctx);

    let irsb = make_irsb_with_temps(0x1000, &[IRType::I1]);
    interp.temps.resize(1, None);
    interp.temps[0] = Some(RustBV::symbolic(&ctx, "storeg_guard", 1));

    let storeg = IRStmt::StoreG {
        addr: Box::new(IRExpr::Const(IRConst::U64(0x4010))),
        data: Box::new(IRExpr::Const(IRConst::U32(0xdead_beef))),
        guard: Box::new(IRExpr::RdTmp(0)),
        endness: Endness::Little,
    };

    with_proxy_callbacks(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &storeg, &irsb)
            .expect("guarded store with symbolic guard");

        let buffered = interp
            .pending_symbolic_stores
            .get(&0x4010)
            .expect("symbolic-guard StoreG must be buffered for rust_memory");
        assert_eq!(buffered.width(), 32, "buffered ITE keeps the store width");
        assert!(buffered.is_symbolic(), "the buffered value is the guard ITE");
    });
}

/// angr-5mnx3.23, CAS variant: `cas_store_symbolic_data` writes back a computed
/// `RustBV` (the success/failure ITE) that cannot be re-expressed as an
/// `IRExpr`, so it dispatches the symbolic-value callback directly. Same proxy
/// gate, same requirement to buffer.
#[test]
fn cas_symbolic_writeback_buffers_for_rust_memory() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp_rust_memory(&ctx);
    let irsb = make_irsb_with_temps(0x1000, &[]);

    let data_bv = RustBV::symbolic(&ctx, "cas_writeback", 32);
    let addr_expr = IRExpr::Const(IRConst::U64(0x4020));

    with_proxy_callbacks(|cb| {
        interp
            .cas_store_symbolic_data(cb, &addr_expr, &data_bv, &irsb)
            .expect("CAS symbolic writeback");

        assert!(
            interp.pending_symbolic_stores.contains_key(&0x4020),
            "CAS symbolic writeback must be buffered for rust_memory"
        );
    });
}

/// angr-5mnx3.23, multi-address variant: a symbolic-address store that
/// concretizes to several candidates within `MAX_ITE_ADDRS` emits one
/// `ITE(addr == a_i, data, mem[a_i])` per candidate through the same callback.
/// Each of those must be buffered too.
#[test]
fn ite_store_buffers_every_candidate_for_rust_memory() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp_rust_memory(&ctx);

    let addr_expr = RustBV::symbolic(&ctx, "ite_store_addr", 64);
    let data_val = RustBV::symbolic(&ctx, "ite_store_data", 32);
    let addrs = [0x4010u64, 0x4020u64];

    with_proxy_callbacks(|cb| {
        interp
            .build_ite_store_from_callbacks(cb, &addrs, &addr_expr, &data_val)
            .expect("per-candidate ITE store");

        for addr in addrs {
            assert!(
                interp.pending_symbolic_stores.contains_key(&addr),
                "candidate {addr:#x} must be buffered for rust_memory"
            );
        }
    });
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
    interp.pending_stores.push(0x4000, vec![0xaa]);

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

    with_python(|cb| {
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
    let sym = RustBV::symbolic(&ctx, "flushed", 16);
    interp.all_flushed_symbolic_stores.insert(0x4000, sym);
    interp.pending_stores.push(0x4000, vec![0xaa]);

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

    with_python(|cb| {
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
        result.extract(7, 0, &ctx).as_u64(),
        Some(0xaa),
        "the buffered byte must win over the symbolic lower layer"
    );
}

// ---------------------------------------------------------------------------
// AVOID_MULTIVALUED_WRITES (angr-6cp06.67)
// ---------------------------------------------------------------------------

/// Concretizer with `avoid_multivalued_writes` set to `on` and nothing else
/// changed from the default.
fn mv_write_concretizer(on: bool) -> crate::concretize::AddressConcretizer {
    let mut c = crate::concretize::AddressConcretizer::new();
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
fn mv_write_read_back(interp: &mut VEXInterpreter<'_>, ctx: &SymContext) -> Option<u64> {
    interp.flush_stores_to_rust_memory();
    let mem = interp.take_rust_memory().expect("interpreter owns memory");
    mem.load_concrete(0x1000, 8, ctx)
        .expect("read-back must succeed")
        .as_u64()
}

/// A 64-bit BVS pinned by constraint to 0x1000: symbolic enough for
/// `should_avoid_multivalued_write` to fire, single-valued enough that with the
/// option off the write concretizes to exactly one place.
fn mv_write_addr(ctx: &SymContext, name: &str) -> RustBV {
    let bv = RustBV::symbolic(ctx, name, 64);
    ctx.assume_true(&bv.eq(&RustBV::concrete(0x1000, 64), ctx));
    assert!(bv.as_u64().is_none(), "address must stay symbolic");
    bv
}

/// `try_rust_memory_store`'s AVOID_MULTIVALUED_WRITES gate returns
/// `Ok(true)` — "handled" — without concretizing or buffering anything, so the
/// backer survives. With the option off the same store lands.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn try_rust_memory_store_drops_the_write_under_avoid_multivalued_writes() {
    pyo3::Python::initialize();
    let callbacks = PythonCallbacks::new();
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
        mv_write_read_back(&mut off, &ctx),
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
        mv_write_read_back(&mut on, &ctx),
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
#[cfg(feature = "vex-engine-z3")]
#[test]
fn handle_symbolic_store_drops_the_write_under_avoid_multivalued_writes() {
    let ctx = SymContext::new_mock();
    let stored = RustBV::concrete(0x0102_0304, 32);

    let mut off = new_interp(&ctx);
    off.set_concretizer(mv_write_concretizer(false));
    let off_addr = mv_write_addr(&ctx, "mvhs_off");
    with_python(|cb| {
        let res = off.handle_symbolic_store(cb, &off_addr, &stored, 4);
        assert!(
            res.is_err(),
            "option off: the store must reach the unregistered store dispatch, got {res:?}"
        );
    });

    let mut on = new_interp(&ctx);
    on.set_concretizer(mv_write_concretizer(true));
    let on_addr = mv_write_addr(&ctx, "mvhs_on");
    with_python(|cb| {
        on.handle_symbolic_store(cb, &on_addr, &stored, 4)
            .expect("option on: the dropped store must report success, not an error");
    });
}
