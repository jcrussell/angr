//! Tests for `expressions.rs` — VEX expression evaluation (eval_const, eval_expr, ops).

use super::*;
use crate::vex::ir::{Endness, IRType};

fn new_interp(ctx: &SymContext) -> VEXInterpreter<'_> {
    VEXInterpreter::new(VexArch::AMD64, ctx)
}

#[test]
fn eval_const_u32_matches_width_and_value() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let bv = interp.eval_const(&IRConst::U32(0xdead_beef));
    assert_eq!(bv.width(), 32);
    assert_eq!(bv.as_u64(), Some(0xdead_beef));
}

#[test]
fn eval_const_u1_round_trips() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let t = interp.eval_const(&IRConst::U1(true));
    let f = interp.eval_const(&IRConst::U1(false));
    assert_eq!(t.width(), 1);
    assert_eq!(t.as_u64(), Some(1));
    assert_eq!(f.as_u64(), Some(0));
}

#[test]
fn eval_const_u128_preserves_high_bits() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let value: u128 = (1u128 << 100) | 0xff;
    let bv = interp.eval_const(&IRConst::U128(value));
    assert_eq!(bv.width(), 128);
    assert_eq!(bv.as_u128(), Some(value));
}

#[test]
fn eval_const_v256_keeps_all_four_lanes_at_width_256() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    // Distinct lanes so a dropped/aliased limb is visible; the low half is what
    // the old `RustBV::concrete(.., 128)` arm kept, the high half what it lost.
    let lanes = [0x0011_2233_4455_6677u64, 0x8899_aabb_ccdd_eeff, 0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210];
    let bv = interp.eval_const(&IRConst::V256(lanes));
    assert_eq!(bv.width(), 256);
    // A 256-bit value cannot live in a `Concrete` (u128 payload), so it must be
    // a Concat — `as_u128` returning `None` is the shape assertion.
    assert_eq!(bv.as_u128(), None);
    for (i, lane) in lanes.iter().enumerate() {
        let lo = (i as u32) * 64;
        let slice = bv.extract_no_ctx(lo + 63, lo);
        assert_eq!(slice.width(), 64);
        assert_eq!(slice.as_u64(), Some(*lane), "lane {i} mismatch");
    }
}

#[test]
fn eval_const_v256_zero_is_still_all_zero() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let bv = interp.eval_const(&IRConst::V256([0; 4]));
    assert_eq!(bv.width(), 256);
    for i in 0..4u32 {
        assert_eq!(bv.extract_no_ctx(i * 64 + 63, i * 64).as_u64(), Some(0));
    }
}

#[test]
fn eval_const_f32_packs_to_bits() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let bv = interp.eval_const(&IRConst::F32(1.0));
    assert_eq!(bv.width(), 32);
    assert_eq!(bv.as_u64(), Some(f32::to_bits(1.0) as u64));
}

#[test]
fn eval_expr_simple_const() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let env = TypeEnv::new();
    let bv = interp
        .eval_expr_simple(&IRExpr::Const(IRConst::U64(0x1234)), &env)
        .expect("const eval");
    assert_eq!(bv.as_u64(), Some(0x1234));
}

#[test]
fn eval_expr_simple_rdtmp_returns_stored_value() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp.temps.resize(4, None);
    interp.temps[2] = Some(RustBV::concrete(0xabc, 32));
    let env = TypeEnv::new();
    let bv = interp
        .eval_expr_simple(&IRExpr::RdTmp(2), &env)
        .expect("temp eval");
    assert_eq!(bv.as_u64(), Some(0xabc));
}

#[test]
fn eval_expr_simple_unknown_temp_errors() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let env = TypeEnv::new();
    let err = interp
        .eval_expr_simple(&IRExpr::RdTmp(0), &env)
        .expect_err("missing temp should error");
    assert!(matches!(err, CbExecutionError::UnknownTemp(0)));
}

#[test]
fn eval_expr_simple_get_register_reads_zero_initially() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let env = TypeEnv::new();
    // AMD64 RAX = offset 16, 8 bytes
    let bv = interp
        .eval_expr_simple(
            &IRExpr::Get {
                offset: 16,
                ty: IRType::I64,
            },
            &env,
        )
        .expect("get eval");
    assert_eq!(bv.width(), 64);
    assert_eq!(bv.as_u64(), Some(0));
}

#[test]
fn eval_expr_simple_get_reads_register_after_write() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp
        .registers
        .put_reg("rax", RustBV::concrete(0xfeed, 64));
    let env = TypeEnv::new();
    let bv = interp
        .eval_expr_simple(
            &IRExpr::Get {
                offset: 16, // RAX on AMD64
                ty: IRType::I64,
            },
            &env,
        )
        .expect("get eval");
    assert_eq!(bv.as_u64(), Some(0xfeed));
}

#[test]
fn eval_expr_simple_rejects_complex_load() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let env = TypeEnv::new();
    let load = IRExpr::Load {
        addr: Box::new(IRExpr::Const(IRConst::U64(0x1000))),
        ty: IRType::I64,
        endness: Endness::Little,
    };
    let err = interp
        .eval_expr_simple(&load, &env)
        .expect_err("simple eval should not handle Load");
    assert!(matches!(err, CbExecutionError::Unsupported(_)));
}

#[test]
fn eval_unop_concrete_unsupported_propagates_error_not_zero() {
    // angr-sa3j: a concrete-arg op that VEXOps rejects must surface the
    // typed OpError (so the engine raises RustUnsupportedVexOpError /
    // routes to Python fallback) instead of silently fabricating 0.
    use crate::callbacks::PythonCallbacks;
    Python::initialize();
    let callbacks = PythonCallbacks::new();
    Python::attach(|_py| {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let env = TypeEnv::new();
        // Dispatch a binary op through the unary path: VEXOps::unop
        // returns OpError::NotUnary. Arg is concrete, so the pre-fix
        // code would have returned Ok(concrete 0).
        let arg = IRExpr::Const(IRConst::U64(0x1234));
        let res = interp.eval_unop(&callbacks, IROp::Add(IRType::I64), &arg, &env);
        assert!(
            matches!(res, Err(CbExecutionError::Op(OpError::NotUnary(_)))),
            "concrete unsupported unop must propagate OpError, got {res:?}"
        );
    });
}

#[test]
fn eval_binop_concrete_unsupported_propagates_error_not_zero() {
    // angr-sa3j (binop arm): concrete-arg unsupported binop must propagate.
    use crate::callbacks::PythonCallbacks;
    Python::initialize();
    let callbacks = PythonCallbacks::new();
    Python::attach(|_py| {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let env = TypeEnv::new();
        // Dispatch a unary op through the binary path: VEXOps::binop
        // returns OpError::NotBinary.
        let left = IRExpr::Const(IRConst::U64(0x1));
        let right = IRExpr::Const(IRConst::U64(0x2));
        let res = interp.eval_binop(&callbacks, IROp::Not(IRType::I64), &left, &right, &env);
        assert!(
            matches!(res, Err(CbExecutionError::Op(OpError::NotBinary(_)))),
            "concrete unsupported binop must propagate OpError, got {res:?}"
        );
    });
}

#[test]
fn eval_binop_symbolic_unsupported_routes_to_python_not_fabricate() {
    // angr-oyzvj: an unsupported op with a SYMBOLIC operand must route the
    // block to Python (NeedPythonFallback) rather than fabricate a fresh
    // unconstrained symbolic (which silently diverges — both branches of any
    // downstream condition explored unconstrained). Default behavior with
    // ANGR_RUST_FABRICATE_UNSUPPORTED_IROP unset.
    use crate::callbacks::PythonCallbacks;
    Python::initialize();
    let callbacks = PythonCallbacks::new();
    Python::attach(|_py| {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        // Stash a symbolic value in a temp so the operand is symbolic.
        interp.temps.resize(2, None);
        interp.temps[0] = Some(RustBV::symbolic(&ctx, "sym_in", 64));
        let env = TypeEnv::new();
        // Dispatch a unary op through the binary path: VEXOps::binop returns
        // OpError::NotBinary, and the (symbolic) operand takes the fallback arm.
        let left = IRExpr::RdTmp(0);
        let right = IRExpr::RdTmp(0);
        let res = interp.eval_binop(&callbacks, IROp::Not(IRType::I64), &left, &right, &env);
        assert!(
            matches!(res, Err(CbExecutionError::NeedPythonFallback(_))),
            "symbolic unsupported binop must route to Python, got {res:?}"
        );
        // The fabricate BYPASS must NOT have fired on the default path.
        assert_eq!(
            interp.stats.vex_bypass_fabricate_count, 0,
            "symbolic unsupported op fabricated instead of routing to Python"
        );
    });
}

#[test]
fn eval_binop_dispatch_fabricate_family_counts_as_python_fallback() {
    // angr-9ke6b.85: the three dispatch-fabricate families (VPerm/Pclmul*/
    // Crc32C) route to Python even with all-concrete args, and that IS a
    // Python-VEX-fallback event — it must bump the same per-op + aggregate
    // counters the ordinary unsupported-binop arm does, or capacity analysis
    // built on those counters silently undercounts exactly these three
    // families. The fabricate BYPASS is a different thing and stays 0.
    use crate::callbacks::PythonCallbacks;
    Python::initialize();
    let callbacks = PythonCallbacks::new();
    Python::attach(|_py| {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        let env = TypeEnv::new();
        let left = IRExpr::Const(IRConst::U64(0x1));
        let right = IRExpr::Const(IRConst::U64(0x2));
        let res = interp.eval_binop(&callbacks, IROp::Crc32C, &left, &right, &env);
        assert!(
            matches!(res, Err(CbExecutionError::NeedPythonFallback(_))),
            "dispatch-fabricate family must route to Python, got {res:?}"
        );
        assert_eq!(
            interp.stats.python_vex_binop_fallback_count, 1,
            "dispatch-fabricate fallback did not bump the per-op binop counter"
        );
        assert_eq!(
            interp.stats.python_vex_op_fallback_count, 1,
            "dispatch-fabricate fallback did not bump the aggregate op counter"
        );
        assert_eq!(
            interp.stats.vex_bypass_fabricate_count, 0,
            "dispatch-fabricate family must not take the fabricate BYPASS"
        );
    });
}

#[test]
fn eval_ccall_unsupported_cond_routes_to_python_not_fabricate() {
    // angr-9ke6b.88: a condition-flag CCall the native handler cannot compute
    // must route the block to Python, NOT fabricate a fresh unconstrained
    // symbolic. The result of `*_calculate_condition` IS a branch guard, so an
    // unconstrained stand-in explores both directions regardless of the real
    // flag semantics. Same policy (and same opt-in gate) as `vex_op_fallback`.
    use crate::callbacks::PythonCallbacks;
    Python::initialize();
    let callbacks = PythonCallbacks::new();
    Python::attach(|_py| {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);
        interp.temps.resize(1, None);
        interp.temps[0] = Some(RustBV::concrete(0, 64));
        let env = TypeEnv::new();
        let cee = IRCallee {
            // Matches the is_cond_ccall name test, but with too few args for
            // `handle_ccall_with_ctx` (which wants 5), so it returns None.
            name: "amd64g_calculate_condition".to_string(),
            addr: 0,
            mcx_mask: 0,
        };
        let args = [IRExpr::RdTmp(0)];
        let res = interp.eval_ccall(&callbacks, &cee, IRType::I64, &args, &env);
        assert!(
            matches!(res, Err(CbExecutionError::NeedPythonFallback(_))),
            "unhandled cond-CCall must route to Python, got {res:?}"
        );
        assert_eq!(
            interp.stats.vex_bypass_fabricate_count, 0,
            "unhandled cond-CCall fabricated a symbol instead of routing to Python"
        );
    });
}

#[test]
fn apply_loadg_conversion_widens_zero() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let val = RustBV::concrete(0xff, 8);
    let widened = interp.apply_loadg_conversion(IRLoadGOp::WidenZ { src_bits: 8 }, val, 32);
    assert_eq!(widened.width(), 32);
    assert_eq!(widened.as_u64(), Some(0xff));
}

#[test]
fn apply_loadg_conversion_widens_signed() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    // 0xff as signed i8 is -1; zero-extend says 0xff (255); sign-extend says 0xffffffff.
    let val = RustBV::concrete(0xff, 8);
    let widened = interp.apply_loadg_conversion(IRLoadGOp::WidenS { src_bits: 8 }, val, 32);
    assert_eq!(widened.width(), 32);
    assert_eq!(widened.as_u64(), Some(0xffff_ffff));
}

#[test]
fn apply_loadg_conversion_identity_passes_through() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let val = RustBV::concrete(0xab, 8);
    let same = interp.apply_loadg_conversion(IRLoadGOp::Identity, val, 8);
    assert_eq!(same.width(), 8);
    assert_eq!(same.as_u64(), Some(0xab));
}

#[test]
fn apply_loadg_conversion_same_width_is_no_op() {
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let val = RustBV::concrete(0xdead_beef, 32);
    let same = interp.apply_loadg_conversion(IRLoadGOp::WidenZ { src_bits: 32 }, val, 32);
    assert_eq!(same.width(), 32);
    assert_eq!(same.as_u64(), Some(0xdead_beef));
}

#[test]
fn apply_loadg_conversion_truncates_when_src_wider() {
    // The truncation branch is currently never exercised in production
    // (LoadG always widens), but guard against future refactors:
    // extract(high, low) requires high >= low and yields high - low + 1
    // bits, so the previous extract(0, target_bits) underflowed in
    // release builds. Confirm we now keep the low target_bits.
    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);
    let val = RustBV::concrete(0xdead_beef, 32);
    let truncated = interp.apply_loadg_conversion(IRLoadGOp::Identity, val, 16);
    assert_eq!(truncated.width(), 16);
    assert_eq!(truncated.as_u64(), Some(0xbeef));
}

// angr-ofyh: an offset load fully inside a wider symbolic store must
// extract the covered bytes as a symbolic value, never the concrete-0
// placeholder that bv_to_bytes used to push into the concrete buffer.
#[test]
fn symbolic_overlap_load_returns_symbolic_for_offset_load() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    interp
        .pending_symbolic_stores
        .insert(0x1000, RustBV::symbolic(&ctx, "v", 64));
    // 4-byte load at offset 4 is fully covered by the 8-byte store.
    let bv = interp
        .symbolic_overlap_load(&interp.pending_symbolic_stores, 0x1004, 4)
        .expect("overlap load should cover [0x1004,0x1008)");
    assert_eq!(bv.width(), 32);
    assert!(
        bv.is_symbolic(),
        "offset load into symbolic store must stay symbolic, not concrete 0"
    );
}

#[test]
fn symbolic_overlap_load_misses_when_not_fully_covered() {
    let ctx = SymContext::new_mock();
    let mut interp = new_interp(&ctx);
    // 4-byte store at 0x1000.
    interp
        .pending_symbolic_stores
        .insert(0x1000, RustBV::symbolic(&ctx, "v", 32));
    // Load straddling the end of the store is not fully covered.
    assert!(
        interp
            .symbolic_overlap_load(&interp.pending_symbolic_stores, 0x1002, 4)
            .is_none()
    );
    // Load entirely before the store is not covered.
    assert!(
        interp
            .symbolic_overlap_load(&interp.pending_symbolic_stores, 0x0ffc, 4)
            .is_none()
    );
    // Empty map short-circuits to None.
    let empty = new_interp(&ctx);
    assert!(
        empty
            .symbolic_overlap_load(&empty.pending_symbolic_stores, 0x1000, 4)
            .is_none()
    );
}

// angr-vvzf5: two overlapping-but-different-base-address SYMBOLIC stores
// (the glibc/compiler-generated overlapping-tail-store idiom -- store a wide
// chunk, then store a second wide chunk starting a few bytes in) must not
// coexist in `pending_symbolic_stores`. Before the fix, `handle_concrete_store`
// (statements_store.rs) called `evict_overlapping_symbolic_stores` only on its
// concrete-data branch; the symbolic-data branch (the common 64-bit path,
// since `use_sym_store` is always false on 64-bit) inserted with no eviction,
// so a later overlap load in `symbolic_overlap_load` could return whichever
// entry `FxHashMap` iteration visited first -- hash-order-arbitrary, not
// most-recent.
//
// This test drives the real `handle_concrete_store` entry point (not a
// hand-seeded map) for both stores, so it exercises the eviction call site
// itself, not just `evict_overlapping_symbolic_stores` in isolation.
#[test]
fn handle_concrete_store_symbolic_data_evicts_overlapping_symbolic_shadow() {
    use crate::callbacks::PythonCallbacks;
    Python::initialize();
    let callbacks = PythonCallbacks::new();
    Python::attach(|_py| {
        let ctx = SymContext::new_mock();
        let mut interp = new_interp(&ctx);

        let store_a = RustBV::symbolic(&ctx, "store_a", 64); // [0x6000, 0x6008)
        let store_b = RustBV::symbolic(&ctx, "store_b", 64); // [0x6004, 0x600c) -- overlaps A

        interp
            .handle_concrete_store(&callbacks, 0x6000, store_a, 8)
            .expect("store A");
        interp
            .handle_concrete_store(&callbacks, 0x6004, store_b, 8)
            .expect("store B (overlaps A, more recent)");

        // The stale, fully-shadowed entry for A must be gone -- only B remains.
        assert_eq!(
            interp.pending_symbolic_stores.len(),
            1,
            "store B must evict the overlapping shadow left by store A"
        );
        assert!(!interp.pending_symbolic_stores.contains_key(&0x6000));
        assert!(interp.pending_symbolic_stores.contains_key(&0x6004));

        // A 2-byte load at 0x6006 overlaps both A's and B's original ranges,
        // so pre-fix this hit the FxHashMap-iteration-order bug. With only B
        // left in the map, the overlap fallback in `symbolic_store_load` must
        // resolve deterministically to bits [16:31] of B (not A, and not a
        // value that flips depending on hash-iteration order).
        let loaded = interp
            .symbolic_store_load(&interp.pending_symbolic_stores, 0x6006, 2)
            .expect("load overlapping the surviving store B");
        assert_eq!(loaded.width(), 16);
        let operand_debug = format!("{:?}", loaded.operands().expect("extract operand")[0]);
        assert!(
            operand_debug.contains("store_b"),
            "load must be extracted from the most recent store B, got {operand_debug}"
        );
        assert!(
            !operand_debug.contains("store_a"),
            "load must not read the stale, evicted store A, got {operand_debug}"
        );
    });
}

/// angr-sqfj8.68: `dispatch_multi_load` must apply the same [`MAX_ITE_ADDRS`]
/// cap `dispatch_multi_store` does. A `Multiple`/`Strided` concretization can
/// carry up to `AddressConcretizer::max_solutions` (256) addresses; before the
/// cap the load path always built the ITE chain in Rust, one
/// `memory_load`/`memory_load_batch` round trip and one ITE node per address.
///
/// Drives `dispatch_multi_load` directly (rather than through `load_symbolic_addr`)
/// so the address count is exact instead of whatever the concretizer happens to
/// enumerate.
#[test]
fn dispatch_multi_load_caps_ite_chain_and_defers_to_python() {
    use crate::callbacks::PythonCallbacks;
    use pyo3::types::PyDict;
    Python::initialize();
    Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(
            c"loads = []
fulls = []
def memory_load(addr, size):
    loads.append((addr, size))
    return (b'\\x00' * size, False, None)
def load_full(addr_ast, size):
    fulls.append(size)
    return None
",
            Some(&globals),
            None,
        )
        .expect("define test callbacks");
        let get = |name: &str| {
            globals
                .get_item(name)
                .unwrap()
                .unwrap_or_else(|| panic!("{name} not defined"))
        };
        let recorder_len = |name: &str| get(name).len().expect("recorder is a list");

        let mut cb = PythonCallbacks::new();
        cb.set_memory_load(get("memory_load").unbind());
        cb.set_memory_load_symbolic_full(get("load_full").unbind());

        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let addr_bv = RustBV::symbolic(&ctx, "multi_load_addr", 64);

        // At the cap: resolved in Rust, one Python load per candidate address.
        let addrs: Vec<u64> = (0..MAX_ITE_ADDRS as u64).map(|i| 0x4000 + i * 8).collect();
        let val = interp
            .dispatch_multi_load(&cb, &addrs, &addr_bv, 4)
            .expect("at-cap load resolves in Rust");
        assert_eq!(val.width(), 32);
        assert_eq!(recorder_len("loads"), MAX_ITE_ADDRS);
        assert_eq!(
            recorder_len("fulls"),
            0,
            "an at-cap load must not reach the full symbolic-load callback"
        );

        // One over: handed to Python's memory model instead, with no per-address
        // loads issued from Rust.
        let addrs: Vec<u64> = (0..=MAX_ITE_ADDRS as u64).map(|i| 0x4000 + i * 8).collect();
        let val = interp
            .dispatch_multi_load(&cb, &addrs, &addr_bv, 4)
            .expect("over-cap load defers to Python");
        assert_eq!(val.width(), 32);
        assert_eq!(recorder_len("fulls"), 1);
        assert_eq!(
            recorder_len("loads"),
            MAX_ITE_ADDRS,
            "the over-cap load must not issue per-address loads from Rust"
        );
    });
}

/// The cap yields when the full symbolic-load callback isn't wired up: the
/// in-Rust ITE chain is then the only way to resolve the load, and erroring
/// out would be a regression against the pre-cap behavior.
#[test]
fn dispatch_multi_load_over_cap_without_full_callback_still_builds_ite() {
    use crate::callbacks::PythonCallbacks;
    use pyo3::types::PyDict;
    Python::initialize();
    Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(
            c"def memory_load(addr, size):
    return (b'\\x00' * size, False, None)
",
            Some(&globals),
            None,
        )
        .expect("define test callbacks");
        let mut cb = PythonCallbacks::new();
        cb.set_memory_load(globals.get_item("memory_load").unwrap().unwrap().unbind());
        assert!(!cb.has_memory_load_symbolic_full());

        let ctx = SymContext::new_mock();
        let interp = new_interp(&ctx);
        let addr_bv = RustBV::symbolic(&ctx, "uncapped_addr", 64);
        let addrs: Vec<u64> = (0..64u64).map(|i| 0x5000 + i * 8).collect();
        let val = interp
            .dispatch_multi_load(&cb, &addrs, &addr_bv, 4)
            .expect("over-cap load without the full callback falls back to the ITE chain");
        assert_eq!(val.width(), 32);
    });
}

/// `regarray_offset` computes the rotating window offset for a well-formed
/// descriptor, and rejects `nElems == 0` with `InvalidIR` instead of panicking
/// on the `%` — `nElems` is lifter-derived and never validated on the way in
/// (see the guard in `regarray_offset`).
#[test]
fn regarray_offset_rotates_and_rejects_zero_nelems() {
    use crate::vex::ir::IRRegArray;

    let ctx = SymContext::new_mock();
    let interp = new_interp(&ctx);

    // Well-formed: an 8-element array of I64 based at offset 0x100.
    // idx 6 + bias 3 == 9, which wraps to element 1 => 0x100 + 1 * 8.
    let descr = IRRegArray { base: 0x100, elemTy: IRType::I64, nElems: 8 };
    let ix = RustBV::concrete(6, 64);
    let (offset, elem_size) = interp
        .regarray_offset(&descr, &ix, 3, "GetI")
        .expect("well-formed descriptor resolves");
    assert_eq!(elem_size, 8);
    assert_eq!(offset, 0x100 + 8);

    // Malformed: nElems == 0 would divide by zero. Must be a loud InvalidIR.
    let bad = IRRegArray { base: 0x100, elemTy: IRType::I64, nElems: 0 };
    let err = interp
        .regarray_offset(&bad, &ix, 3, "PutI")
        .expect_err("nElems == 0 must not be executed");
    match err {
        CbExecutionError::InvalidIR(msg) => {
            assert!(msg.contains("PutI"), "site should be named: {msg}");
            assert!(msg.contains("nElems"), "cause should be named: {msg}");
        }
        other => panic!("expected InvalidIR, got {other:?}"),
    }
}
