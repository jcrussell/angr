// angr-3fb7p: operand-width rejection across the packed-vector op family.
//
// Every `vec_*` helper derives its lane offsets from `elem.bits()` and
// `count` — opcode-table constants — and applies them to operands whose
// widths come from guest data. Those preconditions used to be
// `debug_assert_eq!`, which compiles out of `[profile.release]`: a
// width-mismatched operand did not panic in the shipped `.so`, it took the
// `as_u128` concrete fast path and folded a *wrong answer* that reaches
// neither the Python boundary nor Z3. They now route through
// `VEXOps::require_operand_width`, which reports in every profile.
//
// These tests run in every profile, so they pin the release behaviour the
// old `debug_assert!`s could not: an `Err`, not a plausible-looking constant.

use super::*;
use crate::vex::ir::IRType;

/// Assert that `result` is the typed width rejection naming `site`, rather
/// than a folded value.
#[track_caller]
fn assert_width_rejected(result: Result<RustBV, OpError>, site: &str) {
    match result {
        Err(OpError::UnsupportedVectorOp(msg)) => {
            assert!(
                msg.contains(site) && msg.contains("operand width"),
                "expected an operand-width rejection naming {site}, got: {msg}"
            );
        }
        other => panic!("expected an operand-width rejection for {site}, got {other:?}"),
    }
}

/// Widening multiply (`Iop_Mull{N}{S,U}x{M}`) with a too-narrow `left`.
/// `vec_mull` offsets both operands by `elem.bits()`, so a 32-bit `left`
/// under an 8x8 geometry would read lanes past its end.
#[test]
fn test_vec_mull_rejects_narrow_operand() {
    let ctx = SymContext::new_mock();
    let op = IROp::VMull {
        elem: IRType::I8,
        count: 8,
        signed: false,
        even: false,
    };
    assert_width_rejected(
        VEXOps::binop(op, RustBV::concrete(0, 32), RustBV::concrete(0, 64), &ctx),
        "vec_mull left",
    );
    assert_width_rejected(
        VEXOps::binop(op, RustBV::concrete(0, 64), RustBV::concrete(0, 32), &ctx),
        "vec_mull right",
    );
}

/// High-half multiply — the `vec_mulhi` sibling of the case above.
#[test]
fn test_vec_mulhi_rejects_narrow_operand() {
    let ctx = SymContext::new_mock();
    let op = IROp::VMulHi {
        elem: IRType::I16,
        count: 8,
        signed: true,
    };
    assert_width_rejected(
        VEXOps::binop(op, RustBV::concrete(0, 64), RustBV::concrete(0, 128), &ctx),
        "vec_mulhi left",
    );
}

/// Saturating doubling widening multiply (`Iop_QDMull*`).
#[test]
fn test_vec_qdmull_rejects_narrow_operand() {
    let ctx = SymContext::new_mock();
    let op = IROp::VQDMull {
        elem: IRType::I16,
        count: 4,
    };
    assert_width_rejected(
        VEXOps::binop(op, RustBV::concrete(0, 64), RustBV::concrete(0, 32), &ctx),
        "vec_qdmull right",
    );
}

/// Interleave derives its lane geometry from `left`'s width and applies the
/// offsets to `right` as well.
#[test]
fn test_vec_interleave_rejects_mismatched_operand() {
    let ctx = SymContext::new_mock();
    let op = IROp::VInterleaveLO {
        elem: IRType::I8,
        count: 16,
    };
    assert_width_rejected(
        VEXOps::binop(op, RustBV::concrete(0, 128), RustBV::concrete(0, 64), &ctx),
        "vec_interleave right",
    );
}

/// Unary path: per-byte popcount (`Iop_Cnt8x{8,16}`).
#[test]
fn test_vec_cnt_rejects_narrow_operand() {
    let ctx = SymContext::new_mock();
    assert_width_rejected(
        VEXOps::unop(IROp::VCnt { count: 16 }, RustBV::concrete(0, 64), &ctx),
        "vec_cnt arg",
    );
}

/// Unary path with a fixed 128-bit precondition (`Iop_PwBitMtxXpose64x2`).
#[test]
fn test_vec_bit_mtx_xpose_rejects_narrow_operand() {
    let ctx = SymContext::new_mock();
    assert_width_rejected(
        VEXOps::unop(IROp::VPwBitMtxXpose, RustBV::concrete(0, 64), &ctx),
        "vec_bit_mtx_xpose arg",
    );
}

/// Broadcast (`Iop_Dup{N}x{M}`) checks the *scalar* operand against the lane
/// width rather than the total width — the one shape in this family where
/// the expected width is not the vector total.
#[test]
fn test_vec_dup_rejects_wrong_lane_width() {
    let ctx = SymContext::new_mock();
    let op = IROp::VDup {
        elem: IRType::I8,
        count: 8,
    };
    assert_width_rejected(
        VEXOps::unop(op, RustBV::concrete(0, 16), &ctx),
        "vec_dup arg",
    );
}

/// A correctly-shaped operand still folds — the guard rejects mismatches
/// only, it does not break the happy path.
#[test]
fn test_correct_operand_width_still_folds() {
    let ctx = SymContext::new_mock();
    let result = VEXOps::unop(IROp::VCnt { count: 8 }, RustBV::concrete(0xFF, 64), &ctx)
        .expect("well-shaped operand must fold");
    assert_eq!(result.width(), 64);
    // Low byte 0xFF has 8 set bits; every other byte is zero.
    assert_eq!(result.as_u128(), Some(8));
}
