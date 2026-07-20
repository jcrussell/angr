// angr-ph300.60: direct coverage for the generic per-lane packed-integer
// dispatcher (ops_vec_int_lane.rs) and the ICmpEq / ICmpGtS / ISub IntLaneOp
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
