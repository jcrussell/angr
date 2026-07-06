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
