// angr-9hleg: vector shift (shl/shr/sar/sal) tests (mirror of ops/vec_shift.rs).

use super::*;
use crate::vex::ir::IRType;

/// Symbolic ShlN16x8: shift count is symbolic; constrain to 4 and verify
/// each lane is `lane << 4`. Exercises the symbolic-shift fallback that
/// builds Z3 `bvshl` expressions per lane.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_shl_n_symbolic_shift() {
    let ctx = SymContext::new_mock();

    // Vector: [0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008]
    let mut v: u128 = 0;
    for i in 0..8u32 {
        v |= ((i + 1) as u128) << (i * 16);
    }
    let vec = RustBV::concrete(v, 128);
    let shift = RustBV::symbolic(&ctx, "shl_amt", 8);

    let result = VEXOps::binop(
        IROp::VShlN {
            elem: IRType::I16,
            count: 8,
        },
        vec,
        shift.clone(),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);

    // Constrain shift == 4.
    let four = RustBV::concrete(4, 8);
    let eq = shift.to_z3_ast().eq(four.to_z3_ast());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT after constraining shift == 4");

    let model = ctx.eval(&result).expect("eval(result) returned None");
    for i in 0..8u32 {
        let lane = (model >> (i * 16)) & 0xFFFF;
        let expected = ((i as u128 + 1) << 4) & 0xFFFF;
        assert_eq!(
            lane, expected,
            "lane {i} expected {expected:#x}, got {lane:#x}"
        );
    }
}

/// Symbolic ShrN32x4: shift count is symbolic; constrain to 8 and verify
/// each lane is `lane >> 8`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_shr_n_symbolic_shift() {
    let ctx = SymContext::new_mock();

    // Vector: [0xAABBCCDD, 0x11223344, 0xDEADBEEF, 0xCAFEBABE]
    let lanes: [u32; 4] = [0xAABBCCDD, 0x11223344, 0xDEADBEEF, 0xCAFEBABE];
    let mut v: u128 = 0;
    for (i, lane) in lanes.iter().enumerate() {
        v |= (*lane as u128) << (i * 32);
    }
    let vec = RustBV::concrete(v, 128);
    let shift = RustBV::symbolic(&ctx, "shr_amt", 8);

    let result = VEXOps::binop(
        IROp::VShrN {
            elem: IRType::I32,
            count: 4,
        },
        vec,
        shift.clone(),
        &ctx,
    )
    .unwrap();

    let eight = RustBV::concrete(8, 8);
    let eq = shift.to_z3_ast().eq(eight.to_z3_ast());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT after constraining shift == 8");

    let model = ctx.eval(&result).expect("eval(result) returned None");
    for (i, lane) in lanes.iter().enumerate() {
        let got = ((model >> (i * 32)) & 0xFFFF_FFFF) as u32;
        let expected = lane >> 8;
        assert_eq!(
            got, expected,
            "lane {i} expected {expected:#x}, got {got:#x}"
        );
    }
}

/// Symbolic SarN16x8 with negative lanes: shift count is symbolic; constrain
/// to 4 and verify sign-extending shift (negative values stay negative).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_sar_n_symbolic_shift() {
    let ctx = SymContext::new_mock();

    // Lanes: mix of positive and negative i16 values.
    let lanes: [i16; 8] = [-1, -32768, -16, 0, 1, 0x4000, -2, 256];
    let mut v: u128 = 0;
    for (i, lane) in lanes.iter().enumerate() {
        v |= ((*lane as u16) as u128) << (i * 16);
    }
    let vec = RustBV::concrete(v, 128);
    let shift = RustBV::symbolic(&ctx, "sar_amt", 8);

    let result = VEXOps::binop(
        IROp::VSarN {
            elem: IRType::I16,
            count: 8,
        },
        vec,
        shift.clone(),
        &ctx,
    )
    .unwrap();

    let four = RustBV::concrete(4, 8);
    let eq = shift.to_z3_ast().eq(four.to_z3_ast());
    ctx.add_constraint(eq);
    assert!(ctx.is_sat(), "expected SAT after constraining shift == 4");

    let model = ctx.eval(&result).expect("eval(result) returned None");
    for (i, lane) in lanes.iter().enumerate() {
        let got = ((model >> (i * 16)) & 0xFFFF) as u16 as i16;
        let expected = lane >> 4; // arithmetic shift in Rust on i16
        assert_eq!(got, expected, "lane {i} expected {expected}, got {got}");
    }
}

/// Symbolic ShlN with unbounded shift: just verify a Z3 expression is
/// produced rather than an UnsupportedVectorOp error. Documents the
/// "fully unconstrained" path stays inside the solver.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_shl_n_unbounded_shift() {
    let ctx = SymContext::new_mock();

    let vec = RustBV::concrete(0x1111_2222_3333_4444u128, 64);
    let shift = RustBV::symbolic(&ctx, "shl_amt_free", 8);

    let result = VEXOps::binop(
        IROp::VShlN {
            elem: IRType::I16,
            count: 4,
        },
        vec,
        shift,
        &ctx,
    )
    .expect("unbounded symbolic shift must not error");
    assert_eq!(result.width(), 64);
}

// =========================================================================
// angr-tukg.7 — NEON vector-shift-by-vector (VShl / VShr / VSar).
// =========================================================================

/// Iop_Shl8x8 — left shift each of 8 lanes by the corresponding count lane.
/// Covers: zero shift (identity), in-range shifts, and out-of-range counts
/// (≥ lane width → zero, matching Z3 bvshl).
#[test]
fn test_vshl_8x8_concrete() {
    let ctx = SymContext::new_mock();
    // Lane layout (LSB→MSB): vec lanes, then shift lanes.
    //   0:  0x01 << 0  = 0x01.
    //   1:  0x01 << 1  = 0x02.
    //   2:  0x01 << 7  = 0x80.
    //   3:  0x01 << 8  = 0 (count == lane width).
    //   4:  0x01 << 255 = 0 (count > lane width).
    //   5:  0xFF << 4  = 0xF0 (high bits shifted out).
    //   6:  0x55 << 1  = 0xAA.
    //   7:  0x80 << 1  = 0 (high bit shifted out).
    let lanes_v: [u8; 8] = [0x01, 0x01, 0x01, 0x01, 0x01, 0xFF, 0x55, 0x80];
    let lanes_s: [u8; 8] = [0, 1, 7, 8, 255, 4, 1, 1];
    let lanes_e: [u8; 8] = [0x01, 0x02, 0x80, 0x00, 0x00, 0xF0, 0xAA, 0x00];
    let mut v: u128 = 0;
    let mut s: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        v |= (lanes_v[i] as u128) << (i * 8);
        s |= (lanes_s[i] as u128) << (i * 8);
        e |= (lanes_e[i] as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VShl {
            elem: IRType::I8,
            count: 8,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Shr16x8 — logical right shift over 8x i16 lanes on a 128-bit vector.
/// Verifies zero-fill (Shr discards the sign bit) and out-of-range counts.
#[test]
fn test_vshr_16x8_concrete() {
    let ctx = SymContext::new_mock();
    let lanes_v: [u16; 8] = [
        0x8000, 0xFFFF, 0xABCD, 0x0001, 0xFFFF, 0x4000, 0x0F0F, 0x1234,
    ];
    let lanes_s: [u16; 8] = [15, 8, 4, 0, 16, 1, 4, 100];
    // 0x8000 >> 15 = 1 (no sign extend).
    // 0xFFFF >> 8  = 0x00FF.
    // 0xABCD >> 4  = 0x0ABC.
    // 0x0001 >> 0  = 0x0001.
    // 0xFFFF >> 16 = 0 (count == width).
    // 0x4000 >> 1  = 0x2000.
    // 0x0F0F >> 4  = 0x00F0.
    // 0x1234 >> 100 = 0 (count > width).
    let lanes_e: [u16; 8] = [1, 0x00FF, 0x0ABC, 0x0001, 0, 0x2000, 0x00F0, 0];
    let mut v: u128 = 0;
    let mut s: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        v |= (lanes_v[i] as u128) << (i * 16);
        s |= (lanes_s[i] as u128) << (i * 16);
        e |= (lanes_e[i] as u128) << (i * 16);
    }
    let result = VEXOps::binop(
        IROp::VShr {
            elem: IRType::I16,
            count: 8,
        },
        RustBV::concrete(v, 128),
        RustBV::concrete(s, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Sar32x2 — arithmetic right shift over 2x i32 lanes on a 64-bit
/// vector. Verifies sign-fill on negative inputs and out-of-range counts.
#[test]
fn test_vsar_32x2_concrete() {
    let ctx = SymContext::new_mock();
    // 0: -1i32 (0xFFFF_FFFF) >> 4   = -1 (sign-fill keeps all bits set).
    // 1: 0x4000_0000   >> 1   = 0x2000_0000 (positive → logical shift).
    // Note: 0xFFFF_FFFF >> 32 would also be all-1 in arithmetic shift,
    // but Z3 bvashr semantics for count >= width are sign-fill which is
    // matched by our concrete fast path.
    let lanes_v: [u32; 2] = [0xFFFF_FFFF, 0x4000_0000];
    let lanes_s: [u32; 2] = [4, 1];
    let lanes_e: [u32; 2] = [0xFFFF_FFFF, 0x2000_0000];
    let mut v: u128 = 0;
    let mut s: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..2 {
        v |= (lanes_v[i] as u128) << (i * 32);
        s |= (lanes_s[i] as u128) << (i * 32);
        e |= (lanes_e[i] as u128) << (i * 32);
    }
    let result = VEXOps::binop(
        IROp::VSar {
            elem: IRType::I32,
            count: 2,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Sar32x2 — out-of-range count produces sign-fill (-1 for negative
/// lanes, 0 for positive). Concrete fast path mirrors Z3 bvashr.
#[test]
fn test_vsar_32x2_oor_sign_fill() {
    let ctx = SymContext::new_mock();
    // 0: -1i32 >> 64 = sign-fill = 0xFFFF_FFFF.
    // 1:  1i32 >> 32 = sign-fill = 0 (positive).
    let lanes_v: [u32; 2] = [0xFFFF_FFFF, 0x0000_0001];
    let lanes_s: [u32; 2] = [64, 32];
    let lanes_e: [u32; 2] = [0xFFFF_FFFF, 0];
    let mut v: u128 = 0;
    let mut s: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..2 {
        v |= (lanes_v[i] as u128) << (i * 32);
        s |= (lanes_s[i] as u128) << (i * 32);
        e |= (lanes_e[i] as u128) << (i * 32);
    }
    let result = VEXOps::binop(
        IROp::VSar {
            elem: IRType::I32,
            count: 2,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Sal8x8 (== Iop_Shl8x8 bit-for-bit): a single concrete check that
/// both opcodes parse to VShl and produce identical results.
#[test]
fn test_vsal_routes_to_vshl_and_matches() {
    use crate::vex::opcode_map::parse_opcode;
    let ctx = SymContext::new_mock();
    let v = 0x1234_5678_9ABC_DEF0u128;
    let s = 0x0102_0304_0506_0708u128; // per-lane counts 8,7,6,5,4,3,2,1
    let shl_res = VEXOps::binop(
        IROp::VShl {
            elem: IRType::I8,
            count: 8,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    // Sal{N}x{M} must parse to the same IROp variant.
    assert!(matches!(parse_opcode("Iop_Sal8x8"), IROp::VShl { .. }));
    let sal_op = parse_opcode("Iop_Sal8x8");
    let sal_res = VEXOps::binop(
        sal_op,
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(sal_res.as_u128().unwrap(), shl_res.as_u128().unwrap());
}

/// Z3 universality parity vs the explicit claripy reference for
/// `Iop_Shl16x4` (operation_map["Shl"] = "__lshift__"; vector dispatch
/// per `_op_vector_mapped` extracts each lane and applies `bvshl`).
/// Symbolic inputs + add_constraint(got ≠ py).not() → assert UNSAT.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vshl_16x4_symbolic_matches_python_ref() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "shl_a_16x4", 64);
    let b = RustBV::symbolic(&ctx, "shl_b_16x4", 64);
    let got = VEXOps::binop(
        IROp::VShl {
            elem: IRType::I16,
            count: 4,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();
    // Build the claripy reference: per-lane bvshl. _op_vector_mapped
    // concats lanes high→low; we use concat_le_elements (low→high), so
    // the resulting BV is structurally equivalent.
    let mut lanes: Vec<RustBV> = Vec::with_capacity(4);
    for i in 0..4 {
        let lo = i * 16;
        let hi = lo + 15;
        let al = a.extract(hi, lo, &ctx);
        let bl = b.extract(hi, lo, &ctx);
        lanes.push(al.shl_into(bl, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);
    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    let universal = !ctx.is_sat();
    ctx.pop();
    assert!(
        universal,
        "VShl 16x4 must match the claripy __lshift__ reference for all 64-bit inputs"
    );
}

/// Z3 universality parity vs the claripy reference for `Iop_Sar8x16`
/// (operation_map["Sar"] = "__rshift__" → bvashr; 16 lanes of 8 bits
/// across a 128-bit vector).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vsar_8x16_symbolic_matches_python_ref() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "sar_a_8x16", 128);
    let b = RustBV::symbolic(&ctx, "sar_b_8x16", 128);
    let got = VEXOps::binop(
        IROp::VSar {
            elem: IRType::I8,
            count: 16,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();
    let mut lanes: Vec<RustBV> = Vec::with_capacity(16);
    for i in 0..16 {
        let lo = i * 8;
        let hi = lo + 7;
        let al = a.extract(hi, lo, &ctx);
        let bl = b.extract(hi, lo, &ctx);
        lanes.push(al.ashr_into(bl, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);
    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    let universal = !ctx.is_sat();
    ctx.pop();
    assert!(
        universal,
        "VSar 8x16 must match the claripy __rshift__ reference for all 128-bit inputs"
    );
}

/// Parse routing: all 8 VShl shapes (8x8 .. 64x2) and the matching Sal
/// aliases land on VShl with the expected (elem, count); same for VShr/VSar.
#[test]
fn test_parse_vshift_routing() {
    use crate::vex::opcode_map::parse_opcode;
    // Only `Sal` has a D-reg 64-bit-lane form (`Iop_Sal64x1`); libVEX declares
    // no `Iop_Shl64x1` / `Shr64x1` / `Sar64x1` (angr-0jh0j.62).
    let shl_cases: &[(&str, IRType, u8)] = &[
        ("Iop_Shl8x8", IRType::I8, 8),
        ("Iop_Shl16x4", IRType::I16, 4),
        ("Iop_Shl32x2", IRType::I32, 2),
        ("Iop_Shl8x16", IRType::I8, 16),
        ("Iop_Shl16x8", IRType::I16, 8),
        ("Iop_Shl32x4", IRType::I32, 4),
        ("Iop_Shl64x2", IRType::I64, 2),
        // Sal aliases route to the same variant.
        ("Iop_Sal8x8", IRType::I8, 8),
        ("Iop_Sal16x4", IRType::I16, 4),
        ("Iop_Sal32x2", IRType::I32, 2),
        ("Iop_Sal64x1", IRType::I64, 1),
        ("Iop_Sal8x16", IRType::I8, 16),
        ("Iop_Sal16x8", IRType::I16, 8),
        ("Iop_Sal32x4", IRType::I32, 4),
        ("Iop_Sal64x2", IRType::I64, 2),
    ];
    for (op, e, c) in shl_cases {
        match parse_opcode(op) {
            IROp::VShl { elem, count } => {
                assert_eq!(elem, *e, "{op}: elem");
                assert_eq!(count, *c, "{op}: count");
            }
            other => panic!("{op}: expected VShl, got {other:?}"),
        }
    }

    let shr_cases: &[(&str, IRType, u8)] = &[
        ("Iop_Shr8x8", IRType::I8, 8),
        ("Iop_Shr16x4", IRType::I16, 4),
        ("Iop_Shr32x2", IRType::I32, 2),
        ("Iop_Shr8x16", IRType::I8, 16),
        ("Iop_Shr16x8", IRType::I16, 8),
        ("Iop_Shr32x4", IRType::I32, 4),
        ("Iop_Shr64x2", IRType::I64, 2),
    ];
    for (op, e, c) in shr_cases {
        match parse_opcode(op) {
            IROp::VShr { elem, count } => {
                assert_eq!(elem, *e, "{op}: elem");
                assert_eq!(count, *c, "{op}: count");
            }
            other => panic!("{op}: expected VShr, got {other:?}"),
        }
    }

    let sar_cases: &[(&str, IRType, u8)] = &[
        ("Iop_Sar8x8", IRType::I8, 8),
        ("Iop_Sar16x4", IRType::I16, 4),
        ("Iop_Sar32x2", IRType::I32, 2),
        ("Iop_Sar8x16", IRType::I8, 16),
        ("Iop_Sar16x8", IRType::I16, 8),
        ("Iop_Sar32x4", IRType::I32, 4),
        ("Iop_Sar64x2", IRType::I64, 2),
    ];
    for (op, e, c) in sar_cases {
        match parse_opcode(op) {
            IROp::VSar { elem, count } => {
                assert_eq!(elem, *e, "{op}: elem");
                assert_eq!(count, *c, "{op}: count");
            }
            other => panic!("{op}: expected VSar, got {other:?}"),
        }
    }
}

// angr-qwyti.17 — sign-fill mask overflow guard for arithmetic vector shifts.
//
// `sar_fill_mask` feeds the negative-lane branch of `shift_lane_u128`, the
// concrete per-lane core both `vec_shift_n` and `vec_shift_vec` shift through
// under `VecShiftKind::Sar`. `elem_width - shift` reaches
// `elem_width`, so a 128-bit lane with `shift == 0` would left-shift a u128 by
// 128 — a panic under panic=abort. These tests pin the guard AND prove the
// reachable `< 128` widths still match the naive formula (behavior-preserving).

/// Naive (unguarded) fill-mask formula, valid only when `elem_width < 128`.
fn naive_fill_mask(elem_mask: u128, elem_width: u32, shift: u32) -> u128 {
    (elem_mask << (elem_width - shift)) & elem_mask
}

#[test]
fn test_sar_fill_mask_128_lane_shift0_does_not_overflow() {
    // elem_width == 128, shift == 0 => fill_bits == 128 => the naive form would
    // shift a u128 by 128 (UB/panic). A shift-by-0 fills no bits, so the guard
    // must return 0.
    let elem_mask = VEXOps::low_bit_mask_u128(128); // u128::MAX
    assert_eq!(VEXOps::sar_fill_mask(elem_mask, 128, 0), 0);
}

#[test]
fn test_sar_fill_mask_matches_naive_for_reachable_widths() {
    // Every realistic NEON lane width, every in-range shift. shift < elem_width
    // is guaranteed by callers (the `else if neg` branch), so exclude
    // shift == elem_width.
    for &elem_width in &[8u32, 16, 32, 64] {
        let elem_mask = VEXOps::low_bit_mask_u128(elem_width);
        for shift in 0..elem_width {
            let got = VEXOps::sar_fill_mask(elem_mask, elem_width, shift);
            let want = naive_fill_mask(elem_mask, elem_width, shift);
            assert_eq!(got, want, "elem_width={elem_width} shift={shift}");
            // Fill mask must live entirely within the lane and cover exactly the
            // top `shift` bits.
            assert_eq!(
                got & !elem_mask,
                0,
                "fill escapes lane w={elem_width} s={shift}"
            );
            assert_eq!(
                got.count_ones(),
                shift,
                "wrong fill width w={elem_width} s={shift}"
            );
        }
    }
}

#[test]
fn test_sar_fill_mask_shift0_is_empty_every_width() {
    // shift == 0 (arithmetic shift by zero) fills no bits at every width,
    // including the 128 boundary.
    for &elem_width in &[8u32, 16, 32, 64, 127, 128] {
        let elem_mask = VEXOps::low_bit_mask_u128(elem_width);
        assert_eq!(
            VEXOps::sar_fill_mask(elem_mask, elem_width, 0),
            0,
            "shift-by-0 must fill nothing at width {elem_width}"
        );
    }
}

/// angr-sqfj8.118: AVX2 256-bit shift-by-immediate (ShlN/ShrN/SarN 32x8).
///
/// A 256-bit vector can never be a `Concrete` RustBV — that variant stores its
/// value in a `u128` — so it arrives as a `Concat` expression and must route
/// through the per-lane path. Before the `total_width <= 128` guard in
/// `vec_shift.rs` the concrete fold would have shifted a `u128` by up to 192.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_shift_n_avx2_256_bit() {
    let ctx = SymContext::new_mock();

    // Low lane first; the mix covers positive, negative and sign-boundary
    // values so the SarN sign fill is actually exercised.
    let lanes: [u32; 8] = [
        0x0000_0001,
        0x8000_0000,
        0x1234_5678,
        0xFFFF_FFF0,
        0x0000_00FF,
        0x7FFF_FFFF,
        0xDEAD_BEEF,
        0x0000_0010,
    ];
    let mut lo: u128 = 0;
    let mut hi: u128 = 0;
    for (i, lane) in lanes.iter().enumerate() {
        let shifted = (*lane as u128) << ((i % 4) * 32);
        if i < 4 {
            lo |= shifted;
        } else {
            hi |= shifted;
        }
    }
    let vec = RustBV::concrete(hi, 128).concat_into(RustBV::concrete(lo, 128), &ctx);
    assert_eq!(vec.width(), 256);

    let shift = RustBV::concrete(4, 8);
    // Reference semantics for one 32-bit lane shifted by 4.
    type LaneFn = fn(u32) -> u32;
    let cases: [(IROp, LaneFn); 3] = [
        (
            IROp::VShlN {
                elem: IRType::I32,
                count: 8,
            },
            |l| l.wrapping_shl(4),
        ),
        (
            IROp::VShrN {
                elem: IRType::I32,
                count: 8,
            },
            |l| l >> 4,
        ),
        (
            IROp::VSarN {
                elem: IRType::I32,
                count: 8,
            },
            |l| ((l as i32) >> 4) as u32,
        ),
    ];

    for (op, expected_lane) in cases {
        let result = VEXOps::binop(op, vec.clone(), shift.clone(), &ctx).unwrap();
        assert_eq!(result.width(), 256, "{op:?} result width");

        // The result is wider than a u128, so check it a lane at a time.
        for (i, lane) in lanes.iter().enumerate() {
            let low = (i as u32) * 32;
            let extracted = result.extract(low + 31, low, &ctx);
            let got = ctx.eval(&extracted).expect("eval(lane) returned None");
            assert_eq!(
                got,
                u128::from(expected_lane(*lane)),
                "{op:?} lane {i} of {lane:#010x}"
            );
        }
    }
}
