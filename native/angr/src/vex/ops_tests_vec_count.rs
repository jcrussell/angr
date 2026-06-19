// angr-9hleg: vector per-lane count (cnt/clz/cls) + polynomial-mul tests (mirror of ops_vec_count.rs).

use super::*;

// =========================================================================
// angr-tukg.6 — NEON per-lane Cnt / Clz / Cls + GF(2) PolynomialMul.
// =========================================================================

/// Iop_Cnt8x8 — per-byte popcount over 8 lanes. Covers all bits-set,
/// no-bits-set, single-bit, and mid-density patterns.
#[test]
fn test_vcnt_8x8_concrete() {
    let ctx = SymContext::new_mock();
    let lanes: [u8; 8] = [0x00, 0xFF, 0x01, 0x80, 0x0F, 0xF0, 0x55, 0xAA];
    // 0,8,1,1,4,4,4,4 — popcount per lane.
    let expected: [u8; 8] = [0, 8, 1, 1, 4, 4, 4, 4];
    let mut a: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (lanes[i] as u128) << (i * 8);
        e |= (expected[i] as u128) << (i * 8);
    }
    let result = VEXOps::unop(IROp::VCnt { count: 8 }, RustBV::concrete(a, 64), &ctx).unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Cnt8x16 — Q-reg popcount: 16 bytes.
#[test]
fn test_vcnt_8x16_concrete() {
    let ctx = SymContext::new_mock();
    let mut a: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..16u32 {
        // Lane i has popcount i % 9.
        let v: u8 = (1u16.wrapping_shl(i % 9).wrapping_sub(1)) as u8;
        a |= (v as u128) << (i * 8);
        e |= ((i % 9) as u128) << (i * 8);
    }
    let result = VEXOps::unop(IROp::VCnt { count: 16 }, RustBV::concrete(a, 128), &ctx).unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_GetMSBs8x16 — PMOVMSKB over 16 bytes: bit i of the I16 result is the
/// MSB of byte i. Covers all-set, none-set, and an alternating pattern.
#[test]
fn test_vgetmsbs_8x16_concrete() {
    let ctx = SymContext::new_mock();
    // Byte i: set its MSB iff bit i of the mask 0xA53C is set.
    let mask: u16 = 0xA53C;
    let mut a: u128 = 0;
    for i in 0..16u32 {
        if (mask >> i) & 1 == 1 {
            a |= 0x80u128 << (i * 8); // MSB set
        } else {
            a |= 0x7Fu128 << (i * 8); // all low bits set, MSB clear
        }
    }
    let result =
        VEXOps::unop(IROp::VGetMSBs { count: 16 }, RustBV::concrete(a, 128), &ctx).unwrap();
    assert_eq!(result.width(), 16);
    assert_eq!(result.as_u128().unwrap(), mask as u128);
}

/// Iop_GetMSBs8x8 — PMOVMSKB over 8 bytes (V64 form) → I8.
#[test]
fn test_vgetmsbs_8x8_concrete() {
    let ctx = SymContext::new_mock();
    let mask: u8 = 0b1011_0010;
    let mut a: u128 = 0;
    for i in 0..8u32 {
        let byte: u128 = if (mask >> i) & 1 == 1 { 0xC0 } else { 0x40 };
        a |= byte << (i * 8);
    }
    let result = VEXOps::unop(IROp::VGetMSBs { count: 8 }, RustBV::concrete(a, 64), &ctx).unwrap();
    assert_eq!(result.width(), 8);
    assert_eq!(result.as_u128().unwrap(), mask as u128);
}

/// Symbolic universality (spec-replay): Iop_GetMSBs8x16 must equal the
/// little-endian concat of each byte's MSB bit.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vgetmsbs_8x16_symbolic_universal() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vgetmsbs_a", 128);
    let got = VEXOps::unop(IROp::VGetMSBs { count: 16 }, a.clone(), &ctx).unwrap();

    let mut bits = Vec::with_capacity(16);
    for i in 0..16u32 {
        let pos = i * 8 + 7;
        bits.push(a.extract(pos, pos, &ctx));
    }
    let py = VEXOps::concat_le_elements(bits, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VGetMSBs 8x16 must match the per-byte-MSB reference"
    );
    ctx.pop();
}

/// Iop_Clz8x8 — per-byte count leading zeros.
#[test]
fn test_vclz_8x8_concrete() {
    let ctx = SymContext::new_mock();
    let lanes: [u8; 8] = [0x00, 0x80, 0x40, 0x01, 0xFF, 0x10, 0x08, 0x7F];
    let expected: [u8; 8] = [8, 0, 1, 7, 0, 3, 4, 1];
    let mut a: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (lanes[i] as u128) << (i * 8);
        e |= (expected[i] as u128) << (i * 8);
    }
    let result = VEXOps::unop(
        IROp::VClz {
            elem: IRType::I8,
            count: 8,
        },
        RustBV::concrete(a, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Clz32x4 — Q-reg 32-bit lane clz. Covers all-zero (→ 32), MSB set
/// (→ 0), and a mid-range value.
#[test]
fn test_vclz_32x4_concrete() {
    let ctx = SymContext::new_mock();
    let lanes: [u32; 4] = [0x00000000, 0x80000000, 0x00010000, 0xFFFFFFFF];
    let expected: [u32; 4] = [32, 0, 15, 0];
    let mut a: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..4 {
        a |= (lanes[i] as u128) << (i * 32);
        e |= (expected[i] as u128) << (i * 32);
    }
    let result = VEXOps::unop(
        IROp::VClz {
            elem: IRType::I32,
            count: 4,
        },
        RustBV::concrete(a, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Cls8x8 — per-byte count leading sign bits (excluding MSB).
/// All-same → N-1; first mismatch determines result.
#[test]
fn test_vcls_8x8_concrete() {
    let ctx = SymContext::new_mock();
    // Inputs covering both sign polarities.
    // 0x00 (0b00000000) → all 7 non-MSB bits match MSB(0) → 7.
    // 0xFF (0b11111111) → all 7 non-MSB bits match MSB(1) → 7.
    // 0x01 (0b00000001) → bit6=0,bit5=0,...,bit1=0 match MSB, bit0=1 differs → 6.
    // 0x02 (0b00000010) → bit1=1 differs at pos 1 → first mismatch at pos 1 → 5.
    // 0x40 (0b01000000) → bit6=1 differs from MSB(0) → 0.
    // 0xC0 (0b11000000) → bit6=1 matches MSB(1); bit5=0 differs at pos 5 → 1.
    // 0x80 (0b10000000) → MSB=1; bits 6..0 all 0, all differ → 0.
    // 0x7F (0b01111111) → MSB=0; bit6=1 differs at pos 6 → 0.
    let lanes: [u8; 8] = [0x00, 0xFF, 0x01, 0x02, 0x40, 0xC0, 0x80, 0x7F];
    let expected: [u8; 8] = [7, 7, 6, 5, 0, 1, 0, 0];
    let mut a: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (lanes[i] as u128) << (i * 8);
        e |= (expected[i] as u128) << (i * 8);
    }
    let result = VEXOps::unop(
        IROp::VCls {
            elem: IRType::I8,
            count: 8,
        },
        RustBV::concrete(a, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_PolynomialMul8x8 — GF(2) multiply per byte, low 8 bits.
/// Reference (closed-form per byte):
///   a=0x01, b=0x57 → 0x01*0x57 = 0x57.
///   a=0x02, b=0x57 → 0x57<<1 = 0xAE; low8 = 0xAE.
///   a=0x03, b=0x57 → 0x57 ^ 0xAE = 0xF9; low8 = 0xF9.
///   a=0x80, b=0x80 → 0x80<<7 = 0x4000; low8 = 0x00.
///   a=0xFF, b=0x01 → XOR of 0x01<<0..7 = 0xFF.
///   a=0xC0, b=0x55 → (0x55<<6)^(0x55<<7) = 0x1540 ^ 0x2A80 = 0x3FC0; low8=0xC0.
///   a=0x00, b=0xFF → 0.
///   a=0xFF, b=0xFF → low8 of GF(2) 0xFF*0xFF; computed below.
#[test]
fn test_vpolynomial_mul_8x8_concrete() {
    let ctx = SymContext::new_mock();
    let a_lanes: [u8; 8] = [0x01, 0x02, 0x03, 0x80, 0xFF, 0xC0, 0x00, 0xFF];
    let b_lanes: [u8; 8] = [0x57, 0x57, 0x57, 0x80, 0x01, 0x55, 0xFF, 0xFF];

    // Compute expected via the same algorithm — keeps the test honest.
    let mut expected: [u8; 8] = [0; 8];
    for i in 0..8 {
        let (a, b) = (a_lanes[i] as u16, b_lanes[i] as u16);
        let mut prod: u16 = 0;
        for bit in 0..8 {
            if (a >> bit) & 1 != 0 {
                prod ^= b << bit;
            }
        }
        expected[i] = (prod & 0xFF) as u8;
    }
    // Sanity-check a couple of the closed-form values to catch a bad
    // expected-table generator.
    assert_eq!(expected[0], 0x57);
    assert_eq!(expected[1], 0xAE);
    assert_eq!(expected[2], 0xF9);
    assert_eq!(expected[3], 0x00);
    assert_eq!(expected[4], 0xFF);
    assert_eq!(expected[6], 0x00);

    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (a_lanes[i] as u128) << (i * 8);
        b |= (b_lanes[i] as u128) << (i * 8);
        e |= (expected[i] as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VPolynomialMul {
            count: 8,
            widen: false,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_PolynomialMull8x8 — widening GF(2) multiply: 8x8 → 8x16. Per lane
/// keeps the full 16-bit product. Output total = 128.
#[test]
fn test_vpolynomial_mull_8x8_concrete() {
    let ctx = SymContext::new_mock();
    let a_lanes: [u8; 8] = [0x01, 0x02, 0x80, 0x40, 0xFF, 0x10, 0x55, 0x00];
    let b_lanes: [u8; 8] = [0x57, 0x57, 0x80, 0x02, 0xFF, 0x10, 0xAA, 0xFF];
    let mut expected: [u16; 8] = [0; 8];
    for i in 0..8 {
        let (a, b) = (a_lanes[i] as u16, b_lanes[i] as u16);
        let mut prod: u16 = 0;
        for bit in 0..8 {
            if (a >> bit) & 1 != 0 {
                prod ^= b << bit;
            }
        }
        expected[i] = prod;
    }
    // Widening keeps the full product. 0x80*0x80 over GF(2) = 0x4000.
    assert_eq!(expected[2], 0x4000);
    // 0x01*0x57 = 0x57; widening must preserve this (low byte only).
    assert_eq!(expected[0], 0x0057);

    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (a_lanes[i] as u128) << (i * 8);
        b |= (b_lanes[i] as u128) << (i * 8);
        e |= (expected[i] as u128) << (i * 16);
    }
    let result = VEXOps::binop(
        IROp::VPolynomialMul {
            count: 8,
            widen: true,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Symbolic universality (spec-replay): Iop_Cnt8x8 must equal the
/// per-byte bit-sum reference for all 64-bit inputs.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vcnt_8x8_symbolic_universal() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vcnt_a", 64);
    let got = VEXOps::unop(IROp::VCnt { count: 8 }, a.clone(), &ctx).unwrap();

    let mut lanes = Vec::with_capacity(8);
    for i in 0..8u32 {
        let lo = i * 8;
        let mut acc = RustBV::concrete(0, 8);
        for b in 0..8 {
            let bit = a.extract(lo + b, lo + b, &ctx);
            acc = acc.add_into(bit.zero_extend_into(8, &ctx), &ctx);
        }
        lanes.push(acc);
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VCnt 8x8 must match the spec-replay reference"
    );
    ctx.pop();
}

/// Symbolic universality (spec-replay): Iop_Clz8x8 must match the
/// claripy-style ITE chain reference per lane.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vclz_8x8_symbolic_universal() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vclz_a", 64);
    let got = VEXOps::unop(
        IROp::VClz {
            elem: IRType::I8,
            count: 8,
        },
        a.clone(),
        &ctx,
    )
    .unwrap();

    let mut lanes = Vec::with_capacity(8);
    let one = RustBV::concrete(1, 1);
    for i in 0..8u32 {
        let lo = i * 8;
        let hi = lo + 7;
        let lane = a.extract(hi, lo, &ctx);
        // Build claripy-style ITE chain inline.
        let mut expr = RustBV::concrete(8, 8);
        for b in 0..8u32 {
            let bit = lane.extract(b, b, &ctx);
            let cond = bit.eq_into(one.clone(), &ctx);
            let then_v = RustBV::concrete((8 - b - 1) as u128, 8);
            expr = cond.ite_into(then_v, expr, &ctx);
        }
        lanes.push(expr);
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VClz 8x8 must match the spec-replay reference"
    );
    ctx.pop();
}

/// Symbolic universality (spec-replay): Iop_PolynomialMul8x8 must match
/// the XOR-of-shifts reference per lane (low 8 bits).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vpolynomial_mul_8x8_symbolic_universal() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vpmul_a", 64);
    let b = RustBV::symbolic(&ctx, "vpmul_b", 64);
    let got = VEXOps::binop(
        IROp::VPolynomialMul {
            count: 8,
            widen: false,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();

    let one = RustBV::concrete(1, 1);
    let mut lanes = Vec::with_capacity(8);
    for i in 0..8u32 {
        let lo = i * 8;
        let hi = lo + 7;
        let a_lane = a.extract(hi, lo, &ctx);
        let b_lane = b.extract(hi, lo, &ctx);
        let b_wide = b_lane.zero_extend_into(16, &ctx);
        let mut acc = RustBV::concrete(0, 16);
        for bit in 0..8u32 {
            let bit_a = a_lane.extract(bit, bit, &ctx);
            let cond = bit_a.eq_into(one.clone(), &ctx);
            let shifted = b_wide
                .clone()
                .shl_into(RustBV::concrete(bit as u128, 16), &ctx);
            let addend = cond.ite_into(shifted, RustBV::concrete(0, 16), &ctx);
            acc = acc.xor_into(addend, &ctx);
        }
        lanes.push(acc.extract(7, 0, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VPolynomialMul 8x8 must match the spec-replay reference"
    );
    ctx.pop();
}

/// Parse routing: all Cnt/Clz/Cls/PolynomialMul opcodes land on their real
/// IROp variants (not NeonUnimplemented).
#[test]
fn test_parse_cnt_clz_cls_pmul_routing() {
    use crate::vex::opcode_map::parse_opcode;

    // VCnt
    for (op, c) in &[("Iop_Cnt8x8", 8u8), ("Iop_Cnt8x16", 16)] {
        match parse_opcode(op) {
            IROp::VCnt { count } => assert_eq!(count, *c, "{}: count", op),
            other => panic!("{}: expected VCnt, got {:?}", op, other),
        }
    }

    // VGetMSBs
    for (op, c) in &[("Iop_GetMSBs8x8", 8u8), ("Iop_GetMSBs8x16", 16)] {
        match parse_opcode(op) {
            IROp::VGetMSBs { count } => assert_eq!(count, *c, "{}: count", op),
            other => panic!("{}: expected VGetMSBs, got {:?}", op, other),
        }
    }

    // VClz
    let clz_cases: &[(&str, IRType, u8)] = &[
        ("Iop_Clz8x8", IRType::I8, 8),
        ("Iop_Clz16x4", IRType::I16, 4),
        ("Iop_Clz32x2", IRType::I32, 2),
        ("Iop_Clz8x16", IRType::I8, 16),
        ("Iop_Clz16x8", IRType::I16, 8),
        ("Iop_Clz32x4", IRType::I32, 4),
    ];
    for (op, e, c) in clz_cases {
        match parse_opcode(op) {
            IROp::VClz { elem, count } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
            }
            other => panic!("{}: expected VClz, got {:?}", op, other),
        }
    }

    // VCls
    let cls_cases: &[(&str, IRType, u8)] = &[
        ("Iop_Cls8x8", IRType::I8, 8),
        ("Iop_Cls16x4", IRType::I16, 4),
        ("Iop_Cls32x2", IRType::I32, 2),
        ("Iop_Cls8x16", IRType::I8, 16),
        ("Iop_Cls16x8", IRType::I16, 8),
        ("Iop_Cls32x4", IRType::I32, 4),
    ];
    for (op, e, c) in cls_cases {
        match parse_opcode(op) {
            IROp::VCls { elem, count } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
            }
            other => panic!("{}: expected VCls, got {:?}", op, other),
        }
    }

    // VPolynomialMul (non-widening and widening)
    let pmul_cases: &[(&str, u8, bool)] = &[
        ("Iop_PolynomialMul8x8", 8, false),
        ("Iop_PolynomialMul8x16", 16, false),
        ("Iop_PolynomialMull8x8", 8, true),
    ];
    for (op, c, w) in pmul_cases {
        match parse_opcode(op) {
            IROp::VPolynomialMul { count, widen } => {
                assert_eq!(count, *c, "{}: count", op);
                assert_eq!(widen, *w, "{}: widen", op);
            }
            other => panic!("{}: expected VPolynomialMul, got {:?}", op, other),
        }
    }
}
