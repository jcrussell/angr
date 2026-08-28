// angr-ph300.60: direct coverage for the generic per-lane packed-integer
// dispatcher (ops/vec_int_lane.rs) and the ICmpEq / ICmpGt / ISub IntLaneOp
// impls in `ops/mod.rs`. Prior to this file VSub and VCmpGT had zero tests (not even
// parse tests), and VCmpEQ had only a concrete i32x4 case (ops/tests_core.rs),
// so the symbolic bool->sign_extend lane-compare path (eq_into/sgt_into then
// sign_extend_into(elem_width)) never executed under Z3. Expected-value arrays
// are independent reference constants, never derived from the code under test.

use super::test_helpers::*;
use super::*;
use crate::vex::ir::IRType;

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

// =========================================================================
// AVX2 256-bit compares (angr-sqfj8.112)
// =========================================================================

/// VPCMPEQD/VPCMPGTD-style: 256-bit compares over 8x i32 lanes.
///
/// A 256-bit vector can never be a `Concrete` RustBV — that variant stores its
/// value in a `u128` — so it arrives as a `Concat` and must route through
/// `vec_int_lane_op`'s per-lane symbolic path. That is exactly what the
/// `total_width <= 128` guard there protects: without it the concrete fold
/// would shift a `u128` by up to 224 while assembling the result
/// (`invariant-concrete-bv-u128-16-byte-limit`).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_cmp_avx2_256_bit_i32x8() {
    let ctx = SymContext::new_mock();

    // Low lane first. Covers equal lanes, sign-boundary values and a pair
    // where the signed and unsigned orderings would disagree (0xFFFF_FFFF is
    // -1 signed but the largest value unsigned).
    let l: [u32; 8] = [
        5,
        0x8000_0000,
        0x7FFF_FFFF,
        0xFFFF_FFFF,
        0,
        0x1234_5678,
        0xDEAD_BEEF,
        7,
    ];
    let r: [u32; 8] = [
        5,
        0x7FFF_FFFF,
        0x8000_0000,
        0x0000_0000,
        0,
        0x1234_5679,
        0xDEAD_BEEF,
        6,
    ];

    let pack = |lanes: &[u32; 8]| {
        let mut lo: u128 = 0;
        let mut hi: u128 = 0;
        for (i, lane) in lanes.iter().enumerate() {
            let placed = (*lane as u128) << ((i % 4) * 32);
            if i < 4 {
                lo |= placed;
            } else {
                hi |= placed;
            }
        }
        RustBV::concrete(hi, 128).concat_into(RustBV::concrete(lo, 128), &ctx)
    };
    let lv = pack(&l);
    let rv = pack(&r);
    assert_eq!(lv.width(), 256);

    // Reference semantics, computed independently of the code under test.
    type LaneFn = fn(u32, u32) -> u32;
    let cases: [(IROp, LaneFn); 2] = [
        (
            IROp::VCmpEQ {
                elem: IRType::I32,
                count: 8,
            },
            |a, b| if a == b { u32::MAX } else { 0 },
        ),
        (
            IROp::VCmpGT {
                elem: IRType::I32,
                count: 8,
                signed: true,
            },
            |a, b| {
                if (a as i32) > (b as i32) { u32::MAX } else { 0 }
            },
        ),
    ];

    for (op, expected_lane) in cases {
        let result = VEXOps::binop(op, lv.clone(), rv.clone(), &ctx).unwrap();
        assert_eq!(result.width(), 256, "{op:?} result width");

        // Wider than a u128, so check one lane at a time.
        for i in 0..8u32 {
            let low = i * 32;
            let extracted = result.extract(low + 31, low, &ctx);
            let got = ctx.eval(&extracted).expect("eval(lane) returned None");
            assert_eq!(
                got,
                u128::from(expected_lane(l[i as usize], r[i as usize])),
                "{op:?} lane {i}: {:#010x} vs {:#010x}",
                l[i as usize],
                r[i as usize]
            );
        }
    }
}

// =========================================================================
// VMul — the AVX2 256-bit tier angr-li4ox.1 mapped (Iop_Mul16x16/Mul32x8).
// =========================================================================

/// Parse routing: every mapped `Iop_Mul{N}x{M}` shape lands on `VMul` with the
/// expected `(elem, count)` decomposition, across all three width tiers —
/// D-reg (total=64), Q-reg/SSE (128) and AVX2 (256). The 8-bit lane stops at
/// the Q-reg tier and there is no 64-bit packed multiply at any width, so this
/// table is also the pin on what the parse table must *not* grow (angr-li4ox.1).
#[test]
fn test_parse_vmul_routing() {
    use crate::vex::opcode_map::parse_opcode;

    let cases: &[(&str, IRType, u8)] = &[
        ("Iop_Mul8x8", IRType::I8, 8),
        ("Iop_Mul8x16", IRType::I8, 16),
        ("Iop_Mul16x4", IRType::I16, 4),
        ("Iop_Mul16x8", IRType::I16, 8),
        ("Iop_Mul16x16", IRType::I16, 16),
        ("Iop_Mul32x2", IRType::I32, 2),
        ("Iop_Mul32x4", IRType::I32, 4),
        ("Iop_Mul32x8", IRType::I32, 8),
    ];
    for (op, e, c) in cases {
        match parse_opcode(op) {
            IROp::VMul { elem, count } => {
                assert_eq!(elem, *e, "{op}: elem");
                assert_eq!(count, *c, "{op}: count");
            }
            other => panic!("{op}: expected VMul, got {other:?}"),
        }
    }
}

/// Iop_Mul16x16 — the AVX2 `VPMULLW` shape. 256 bits is past the 16-byte
/// concrete backing store, so the operands are built as a `Concat` of two
/// 128-bit concretes and the result is checked one lane at a time (bd
/// `invariant-concrete-bv-u128-16-byte-limit`, hazard family 3). This is the
/// path `vec_int_lane_op` takes when its `total_width <= 128` concrete fast
/// path declines: per-lane `extract` + symbolic `IMul` + `concat_le_elements`.
#[test]
fn test_vec_mul_16x16_avx2_256_bit() {
    let ctx = SymContext::new_mock();

    // Low lane first. Covers the wrapping cases the width-preserving multiply
    // must truncate (0xFFFF*0xFFFF, 0x8000*2), identity/zero lanes, and a
    // plain no-wrap lane.
    let l: [u16; 16] = [
        0xFFFF, 0x8000, 0, 1, 3, 0x1234, 0x00FF, 0x0100, 7, 0xABCD, 2, 0x7FFF, 0xFFFF, 0x5555,
        0x0F0F, 12345,
    ];
    let r: [u16; 16] = [
        0xFFFF, 2, 0x1234, 0xBEEF, 5, 0x0010, 0x00FF, 0x0100, 0, 1, 0x8000, 2, 0x0002, 0x0003,
        0xF0F0, 3,
    ];

    let pack = |lanes: &[u16; 16]| {
        let mut halves = [0u128; 2];
        for (i, lane) in lanes.iter().enumerate() {
            halves[i / 8] |= (*lane as u128) << ((i % 8) * 16);
        }
        RustBV::concrete(halves[1], 128).concat_into(RustBV::concrete(halves[0], 128), &ctx)
    };

    let result = VEXOps::binop(
        IROp::VMul {
            elem: IRType::I16,
            count: 16,
        },
        pack(&l),
        pack(&r),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 256);

    for i in 0..16u32 {
        let low = i * 16;
        let extracted = result.extract(low + 15, low, &ctx);
        let got = ctx.eval(&extracted).expect("eval(lane) returned None");
        let idx = i as usize;
        // Reference is the wrapping u16 product, computed independently of the
        // lane loop under test.
        let expected = l[idx].wrapping_mul(r[idx]);
        assert_eq!(
            got,
            u128::from(expected),
            "lane {i}: {:#06x} * {:#06x}",
            l[idx],
            r[idx]
        );
    }
}

// =========================================================================
// Over-arity hardening (angr-5mnx3.67)
// =========================================================================

/// An `IntLaneOp` whose `arity()` exceeds `INT_LANE_OP_MAX_ARITY`. No such op
/// exists in the dispatch table today; this stands in for a future one added
/// without widening the const, which is exactly the case the fixed-size `buf`
/// in `VEXOps::vec_int_lane_op` cannot index.
struct OverArityLaneOp;

impl IntLaneOp for OverArityLaneOp {
    fn arity(&self) -> usize {
        INT_LANE_OP_MAX_ARITY + 1
    }
    fn concrete_lane(&self, lanes: &[u128], _elem_width: u32) -> u128 {
        lanes[0]
    }
    fn symbolic_lane(&self, lanes: &[RustBV], _elem_width: u32, _ctx: &SymContext) -> RustBV {
        lanes[0].clone()
    }
}

/// The concrete fast path must return a typed error rather than index `buf`
/// out of bounds. Before angr-5mnx3.67 this was only a `debug_assert!`, so a
/// release build (which is what ships) reached `buf[idx]` and aborted the
/// process. Mirrors the float-side hardening from angr-j60q0.2.
#[test]
fn test_vec_int_lane_over_arity_returns_error() {
    let ctx = SymContext::new_mock();
    let args = [
        RustBV::concrete(0x1111_1111, 32),
        RustBV::concrete(0x2222_2222, 32),
        RustBV::concrete(0x3333_3333, 32),
    ];
    let err = VEXOps::vec_int_lane_op(&args, IRType::I8, 4, &OverArityLaneOp, &ctx)
        .expect_err("over-arity op must not be dispatched");
    match err {
        OpError::UnsupportedVectorOp(msg) => {
            assert!(
                msg.contains("vec_int_lane_op arity 3 exceeds max 2"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("expected UnsupportedVectorOp, got {other:?}"),
    }
}
