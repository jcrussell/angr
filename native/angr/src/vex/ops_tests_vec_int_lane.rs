// angr-ph300.60: direct coverage for the generic per-lane packed-integer
// dispatcher (ops_vec_int_lane.rs) and the ICmpEq / ICmpGt / ISub IntLaneOp
// impls in ops.rs. Prior to this file VSub and VCmpGT had zero tests (not even
// parse tests), and VCmpEQ had only a concrete i32x4 case (ops_tests_core.rs),
// so the symbolic bool->sign_extend lane-compare path (eq_into/sgt_into then
// sign_extend_into(elem_width)) never executed under Z3. Expected-value arrays
// are independent reference constants, never derived from the code under test.

use super::ops_test_helpers::*;
use super::*;

// =========================================================================
// VSub — concrete wrapping lanes (ISub IntLaneOp)
// =========================================================================

/// PSUBW-style: signed/unsigned wrapping subtract over 8x i16 lanes, chosen to
/// exercise borrow-across-zero (0-1), INT_MIN/INT_MAX boundaries and a plain
/// no-wrap lane. Each expected lane is a hand-computed two's-complement u16.
#[test]
fn test_vec_sub_concrete_i16x8() {
    let ctx = SymContext::new_mock();

    let l: [u128; 8] = [0x0000, 100, 0x8000, 0x7FFF, 5, 0xFFFF, 1, 12345];
    let r: [u128; 8] = [0x0001, 50, 0x0001, 0xFFFF, 5, 0x0001, 2, 345];
    // lane0: 0-1 wraps to 0xFFFF; lane2: INT_MIN-1 -> 0x7FFF; lane3:
    // 0x7FFF-0xFFFF -> 0x8000; lane6: 1-2 -> 0xFFFF.
    let exp: [u128; 8] = [0xFFFF, 50, 0x7FFF, 0x8000, 0, 0xFFFE, 0xFFFF, 12000];

    let lv = pack_lanes_uint(&l, 16);
    let rv = pack_lanes_uint(&r, 16);
    let result = VEXOps::binop(
        IROp::VSub {
            elem: IRType::I16,
            count: 8,
        },
        RustBV::concrete(lv, 128),
        RustBV::concrete(rv, 128),
        &ctx,
    )
    .unwrap();
    assert_int_lanes_eq(result.as_u128().unwrap(), &exp, 16);
}

// =========================================================================
// VCmpGT (signed) — concrete INT_MIN vs INT_MAX (ICmpGtS concrete_lane)
// =========================================================================

/// PCMPGTD-style: signed greater-than over 4x i32 lanes with INT_MIN/INT_MAX/
/// zero/-1 operands. Pins that the concrete path treats the lane bits as signed
/// (via sign_extend_low_to_i128), so INT_MAX > INT_MIN is true but INT_MIN >
/// INT_MAX is false. Result lane is all-ones on true, all-zeros on false.
#[test]
fn test_vec_cmp_gt_signed_concrete_i32x4() {
    let ctx = SymContext::new_mock();

    let l: [u128; 4] = [0x7FFF_FFFF, 0x8000_0000, 0x0000_0000, 0xFFFF_FFFF];
    let r: [u128; 4] = [0x8000_0000, 0x7FFF_FFFF, 0x0000_0000, 0x0000_0000];
    // MAX > MIN -> T; MIN > MAX -> F; 0 > 0 -> F; -1 > 0 -> F.
    let exp: [u128; 4] = [0xFFFF_FFFF, 0, 0, 0];

    let lv = pack_lanes_uint(&l, 32);
    let rv = pack_lanes_uint(&r, 32);
    let result = VEXOps::binop(
        IROp::VCmpGT {
            elem: IRType::I32,
            count: 4,
            signed: true,
        },
        RustBV::concrete(lv, 128),
        RustBV::concrete(rv, 128),
        &ctx,
    )
    .unwrap();
    assert_int_lanes_eq(result.as_u128().unwrap(), &exp, 32);
}

// =========================================================================
// Symbolic lane-compare path — bool -> sign_extend(elem_width)
// =========================================================================

/// Symbolic VCmpEQ over 4x i32: left is a free vector constrained to a concrete
/// packing, forcing the symbolic per-lane path (eq_into then
/// sign_extend_into(32)). Evaluating the model pins that a true lane becomes a
/// full 0xFFFFFFFF (not just bit 0) and a false lane is all-zeros.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_cmp_eq_symbolic_i32x4() {
    let ctx = SymContext::new_mock();

    let l_lanes: [u128; 4] = [5, 5, 5, 5];
    let r_lanes: [u128; 4] = [5, 9, 5, 9];
    let exp: [u128; 4] = [0xFFFF_FFFF, 0, 0xFFFF_FFFF, 0];

    let l = RustBV::symbolic(&ctx, "vcmpeq_l", 128);
    ctx.add_constraint(
        l.to_z3_ast()
            .eq(RustBV::concrete(pack_lanes_uint(&l_lanes, 32), 128).to_z3_ast()),
    );
    let r = RustBV::concrete(pack_lanes_uint(&r_lanes, 32), 128);

    let result = VEXOps::binop(
        IROp::VCmpEQ {
            elem: IRType::I32,
            count: 4,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert!(ctx.is_sat(), "expected SAT after pinning left operand");

    let model = ctx.eval(&result).expect("eval(result) returned None");
    assert_int_lanes_eq(model, &exp, 32);
}

/// Symbolic VCmpGT (signed) over 4x i32 with INT_MIN/INT_MAX operands. Forces
/// the symbolic per-lane path (sgt_into then sign_extend_into(32)) and pins that
/// the Z3 comparison is signed: INT_MAX > INT_MIN true, INT_MIN > INT_MAX false.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_cmp_gt_signed_symbolic_i32x4() {
    let ctx = SymContext::new_mock();

    let l_lanes: [u128; 4] = [0x7FFF_FFFF, 0x8000_0000, 0x0000_0000, 0xFFFF_FFFF];
    let r_lanes: [u128; 4] = [0x8000_0000, 0x7FFF_FFFF, 0x0000_0000, 0x0000_0000];
    let exp: [u128; 4] = [0xFFFF_FFFF, 0, 0, 0];

    let l = RustBV::symbolic(&ctx, "vcmpgt_l", 128);
    ctx.add_constraint(
        l.to_z3_ast()
            .eq(RustBV::concrete(pack_lanes_uint(&l_lanes, 32), 128).to_z3_ast()),
    );
    let r = RustBV::concrete(pack_lanes_uint(&r_lanes, 32), 128);

    let result = VEXOps::binop(
        IROp::VCmpGT {
            elem: IRType::I32,
            count: 4,
            signed: true,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert!(ctx.is_sat(), "expected SAT after pinning left operand");

    let model = ctx.eval(&result).expect("eval(result) returned None");
    assert_int_lanes_eq(model, &exp, 32);
}

// =========================================================================
// VCmpGT (unsigned) — angr-9ke6b.160
// =========================================================================

/// NEON `VCGT.U32` / SSE-style unsigned packed greater-than over 4x u32.
/// Every lane is a value pair where the signed and unsigned answers *differ*
/// (or would, if the lane were misread as signed), so this fails loudly if the
/// unsigned dispatch ever falls back to the signed comparator:
///   lane0: 0xFFFF_FFFF vs 0x0000_0001 -> unsigned true  (signed would be false)
///   lane1: 0x0000_0001 vs 0xFFFF_FFFF -> unsigned false (signed would be true)
///   lane2: 0x8000_0000 vs 0x7FFF_FFFF -> unsigned true  (signed would be false)
///   lane3: 0x7FFF_FFFF vs 0x8000_0000 -> unsigned false (signed would be true)
#[test]
fn test_vec_cmp_gt_unsigned_concrete_u32x4() {
    let ctx = SymContext::new_mock();

    let l: [u128; 4] = [0xFFFF_FFFF, 0x0000_0001, 0x8000_0000, 0x7FFF_FFFF];
    let r: [u128; 4] = [0x0000_0001, 0xFFFF_FFFF, 0x7FFF_FFFF, 0x8000_0000];
    let exp: [u128; 4] = [0xFFFF_FFFF, 0, 0xFFFF_FFFF, 0];

    let result = VEXOps::binop(
        IROp::VCmpGT {
            elem: IRType::I32,
            count: 4,
            signed: false,
        },
        RustBV::concrete(pack_lanes_uint(&l, 32), 128),
        RustBV::concrete(pack_lanes_uint(&r, 32), 128),
        &ctx,
    )
    .unwrap();
    assert_int_lanes_eq(result.as_u128().unwrap(), &exp, 32);
}

/// Unsigned `Iop_CmpGT8Ux16` (byte lanes, the width real unsigned NEON string
/// routines use). Mixes an equal pair (must be false — GT, not GE) with
/// high-bit-set operands that a signed compare would invert.
#[test]
fn test_vec_cmp_gt_unsigned_concrete_u8x16() {
    let ctx = SymContext::new_mock();

    let l: [u128; 16] = [
        0xFF, 0x00, 0x80, 0x7F, 0x41, 0x41, 0x01, 0xFE, 0xFF, 0x7F, 0x80, 0x00, 0xAA, 0x55, 0x10,
        0xEF,
    ];
    let r: [u128; 16] = [
        0xFE, 0x01, 0x7F, 0x80, 0x41, 0x40, 0x02, 0xFF, 0x00, 0x7F, 0x81, 0x00, 0x55, 0xAA, 0x10,
        0xF0,
    ];
    let exp: [u128; 16] = [
        0xFF, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00,
        0x00,
    ];

    let result = VEXOps::binop(
        IROp::VCmpGT {
            elem: IRType::I8,
            count: 16,
            signed: false,
        },
        RustBV::concrete(pack_lanes_uint(&l, 8), 128),
        RustBV::concrete(pack_lanes_uint(&r, 8), 128),
        &ctx,
    )
    .unwrap();
    assert_int_lanes_eq(result.as_u128().unwrap(), &exp, 8);
}

/// Symbolic unsigned VCmpGT over 4x u32 — forces the `ugt_into` +
/// `sign_extend_into(32)` lane path under Z3 (the concrete tests above only
/// cover `concrete_lane`). Same operand pairs as the concrete u32x4 case, so
/// the two paths are pinned to the same reference answer.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_cmp_gt_unsigned_symbolic_u32x4() {
    let ctx = SymContext::new_mock();

    let l_lanes: [u128; 4] = [0xFFFF_FFFF, 0x0000_0001, 0x8000_0000, 0x7FFF_FFFF];
    let r_lanes: [u128; 4] = [0x0000_0001, 0xFFFF_FFFF, 0x7FFF_FFFF, 0x8000_0000];
    let exp: [u128; 4] = [0xFFFF_FFFF, 0, 0xFFFF_FFFF, 0];

    let l = RustBV::symbolic(&ctx, "vcmpgtu_l", 128);
    ctx.add_constraint(
        l.to_z3_ast()
            .eq(RustBV::concrete(pack_lanes_uint(&l_lanes, 32), 128).to_z3_ast()),
    );
    let r = RustBV::concrete(pack_lanes_uint(&r_lanes, 32), 128);

    let result = VEXOps::binop(
        IROp::VCmpGT {
            elem: IRType::I32,
            count: 4,
            signed: false,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert!(ctx.is_sat(), "expected SAT after pinning left operand");

    let model = ctx.eval(&result).expect("eval(result) returned None");
    assert_int_lanes_eq(model, &exp, 32);
}
