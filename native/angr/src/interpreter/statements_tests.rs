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
fn put_marks_register_dirty() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    let irsb = make_irsb_with_temps(0x1000, &[]);
    assert_eq!(interp.dirty_registers, 0);
    let stmt = IRStmt::Put {
        offset: 16, // RAX -> bit index 4
        data: IRExpr::Const(IRConst::U64(1)),
    };
    with_python(|cb| {
        interp
            .execute_stmt_with_callbacks(cb, &stmt, &irsb)
            .expect("put");
    });
    assert_ne!(interp.dirty_registers, 0);
    assert_eq!(interp.dirty_registers & (1u128 << 4), 1u128 << 4);
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
    let interp = new_interp(&ctx);
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
