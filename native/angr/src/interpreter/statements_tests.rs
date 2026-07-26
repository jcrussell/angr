//! Tests for VEX statement execution (extracted from statements.rs).
use super::*;
use crate::vex::ir::{Endness, IRType, MBusEvent};

fn new_interp(ctx: &SymContext) -> VEXInterpreter<'_> {
    VEXInterpreter::new(VexArch::AMD64, ctx)
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
        interp
            .cas_store_symbolic_data(cb, &addr_expr, &data_bv, &call_irsb)
            .expect("cas_store_symbolic_data (concrete addr)");
    });

    assert!(
        !interp.has_cached_block(0x1010),
        "stale cached block must be invalidated by a CAS store to its address"
    );
    assert!(interp.is_code_page_dirtied(0x1010));
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
