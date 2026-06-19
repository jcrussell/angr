// angr-9hleg: vector pairwise add/min/max + avg tests (mirror of ops_vec_pairwise.rs).

use super::ops_test_helpers::*;
use super::*;

// =========================================================================
// angr-tukg.2 — NEON pairwise add/min/max (VPwAdd / VPwAddL / VPwMin / VPwMax).
// =========================================================================

/// Iop_PwAdd16x4 — pairwise add over 4 lanes of 16 bits, two 64-bit
/// sources. Output[0..2] from `a`, output[2..4] from `b`. Tests both
/// halves and a sample of values.
#[test]
fn test_vpwadd_16x4_concrete() {
    let ctx = SymContext::new_mock();
    // a lanes (LSB→MSB): 1, 2, 3, 4 → pairs (1+2, 3+4) = 3, 7.
    // b lanes:           10, 20, 100, 200 → pairs (30, 300).
    // Output (LSB→MSB): 3, 7, 30, 300.
    let a_lanes: [u16; 4] = [1, 2, 3, 4];
    let b_lanes: [u16; 4] = [10, 20, 100, 200];
    let e_lanes: [u16; 4] = [3, 7, 30, 300];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..4 {
        a |= (a_lanes[i] as u128) << (i * 16);
        b |= (b_lanes[i] as u128) << (i * 16);
        e |= (e_lanes[i] as u128) << (i * 16);
    }
    let result = VEXOps::binop(
        IROp::VPwAdd {
            elem: IRType::I16,
            count: 4,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_PwAdd8x16 — Q-reg variant: 16 lanes of 8 bits. Output[0..8] from
/// `a`, output[8..16] from `b`. Wrap-around per lane (e.g. 0xFF+0x01=0x00).
#[test]
fn test_vpwadd_8x16_concrete() {
    let ctx = SymContext::new_mock();
    // a: pairs (10+20, 30+40, ..., 70+80) → 30, 70, 110, ..., 0xFF+0x01=0x00.
    // Use explicit lanes for clarity.
    let a_lanes: [u8; 16] = [10, 20, 30, 40, 50, 60, 70, 80, 0xFF, 0x01, 0, 0, 0, 0, 0, 0];
    let b_lanes: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0x80, 0x80, 0, 0, 0, 0];
    // a-half output (8 lanes): (10+20, 30+40, 50+60, 70+80, 0xFF+0x01=0x00, 0, 0, 0)
    //                       = (30, 70, 110, 150, 0, 0, 0, 0)
    // b-half output (8 lanes): (1+2, 3+4, 5+6, 7+8, 9+10, 0x80+0x80=0x00, 0, 0)
    //                       = (3, 7, 11, 15, 19, 0, 0, 0)
    let e_lanes: [u8; 16] = [30, 70, 110, 150, 0, 0, 0, 0, 3, 7, 11, 15, 19, 0, 0, 0];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..16 {
        a |= (a_lanes[i] as u128) << (i * 8);
        b |= (b_lanes[i] as u128) << (i * 8);
        e |= (e_lanes[i] as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VPwAdd {
            elem: IRType::I8,
            count: 16,
        },
        RustBV::concrete(a, 128),
        RustBV::concrete(b, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_PwAdd32Fx2 — NEON pairwise FP add (ARM VPADD.F32, D-reg). Two F32
/// lanes per 64-bit operand: a=[a0,a1], b=[b0,b1] → [a0+a1, b0+b1]. Output
/// lane 0 (low) from `a`, lane 1 (high) from `b`.
#[test]
fn test_vfpwadd_32fx2_concrete() {
    let ctx = SymContext::new_mock();
    let a = [1.5f32, 2.5]; // a0+a1 = 4.0
    let b = [-3.0f32, 0.25]; // b0+b1 = -2.75
    let exp = [4.0f32, -2.75];
    let result = VEXOps::binop(
        IROp::VFPwAdd {
            elem: IRType::F32,
            count: 2,
        },
        RustBV::concrete(pack_lanes_f32(&a), 64),
        RustBV::concrete(pack_lanes_f32(&b), 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_f32_lanes_approx(result.as_u128().unwrap(), &exp, 1e-6);
}

/// Symbolic Iop_PwAdd32Fx2: free `left` constrained to [1.0, 2.0], concrete
/// `right` = [5.0, 6.0]. Expect [1.0+2.0, 5.0+6.0] = [3.0, 11.0].
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vfpwadd_32fx2_symbolic() {
    let ctx = SymContext::new_mock();
    let r = RustBV::concrete(pack_lanes_f32(&[5.0f32, 6.0]), 64);
    let l = RustBV::symbolic(&ctx, "vfpwadd_l", 64);
    ctx.add_constraint(
        l.to_z3_ast()
            .eq(RustBV::concrete(pack_lanes_f32(&[1.0f32, 2.0]), 64).to_z3_ast()),
    );
    let result = VEXOps::binop(
        IROp::VFPwAdd {
            elem: IRType::F32,
            count: 2,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert!(ctx.is_sat(), "expected SAT");
    let model = ctx.eval(&result).expect("eval(result) returned None");
    assert_f32_lanes_approx(model, &[3.0f32, 11.0], 1e-6);
}

/// Iop_PwAddL8Sx8 — signed widening pairwise add. Input is 8 lanes × 8 bits;
/// output is 4 lanes × 16 bits. Negative sources must sign-extend before
/// adding so the sum doesn't lose its sign.
#[test]
fn test_vpwaddl_8sx8_concrete() {
    let ctx = SymContext::new_mock();
    // a lanes (signed i8): -100, -100, 100, 100, -1, -1, 1, 1
    // Pairs: -200, 200, -2, 2 (as i16).
    let a_lanes: [i8; 8] = [-100, -100, 100, 100, -1, -1, 1, 1];
    let e_lanes: [i16; 4] = [-200, 200, -2, 2];
    let mut a: u128 = 0;
    let mut e: u128 = 0;
    for (i, &lane) in a_lanes.iter().enumerate() {
        a |= ((lane as u8) as u128) << (i * 8);
    }
    for (i, &lane) in e_lanes.iter().enumerate() {
        e |= ((lane as u16) as u128) << (i * 16);
    }
    let result = VEXOps::unop(
        IROp::VPwAddL {
            elem: IRType::I8,
            count: 8,
            signed: true,
        },
        RustBV::concrete(a, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_PwAddL8Ux16 — unsigned widening pairwise add, Q-reg. Verifies that
/// 0xFF + 0xFF widens to 0x01FE rather than overflowing in 8 bits.
#[test]
fn test_vpwaddl_8ux16_concrete() {
    let ctx = SymContext::new_mock();
    let a_lanes: [u8; 16] = [
        0xFF, 0xFF, 0x80, 0x80, 0x01, 0x02, 0, 0, 100, 50, 200, 100, 0, 0, 0, 0,
    ];
    let e_lanes: [u16; 8] = [0x01FE, 0x0100, 0x0003, 0, 150, 300, 0, 0];
    let mut a: u128 = 0;
    let mut e: u128 = 0;
    for (i, &lane) in a_lanes.iter().enumerate() {
        a |= (lane as u128) << (i * 8);
    }
    for (i, &lane) in e_lanes.iter().enumerate() {
        e |= (lane as u128) << (i * 16);
    }
    let result = VEXOps::unop(
        IROp::VPwAddL {
            elem: IRType::I8,
            count: 16,
            signed: false,
        },
        RustBV::concrete(a, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_PwMin16Sx4 — pairwise signed min. Exercises mixed-sign pairs and
/// confirms output[2..4] comes from `b`.
#[test]
fn test_vpwmin_16sx4_concrete() {
    let ctx = SymContext::new_mock();
    // a: -100, 100, 200, -200 → pairs min(-100,100)=-100, min(200,-200)=-200
    // b: 30000, -1, 0, 0       → pairs min=-1, min=0
    let a_lanes: [i16; 4] = [-100, 100, 200, -200];
    let b_lanes: [i16; 4] = [30000, -1, 0, 0];
    let e_lanes: [i16; 4] = [-100, -200, -1, 0];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..4 {
        a |= ((a_lanes[i] as u16) as u128) << (i * 16);
        b |= ((b_lanes[i] as u16) as u128) << (i * 16);
        e |= ((e_lanes[i] as u16) as u128) << (i * 16);
    }
    let result = VEXOps::binop(
        IROp::VPwMin {
            elem: IRType::I16,
            count: 4,
            signed: true,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_PwMax8Ux8 — pairwise unsigned max. Distinguishes 0x80 (signed -128)
/// from 0x01 to confirm unsigned compare.
#[test]
fn test_vpwmax_8ux8_concrete() {
    let ctx = SymContext::new_mock();
    // a: 0x80, 0x01, 0x10, 0x10, 0xFF, 0xFF, 0, 0
    //   → unsigned max pairs: 0x80, 0x10, 0xFF, 0
    // b: 0, 0, 0, 0, 50, 60, 70, 80
    //   → max: 0, 0, 60, 80
    let a_lanes: [u8; 8] = [0x80, 0x01, 0x10, 0x10, 0xFF, 0xFF, 0, 0];
    let b_lanes: [u8; 8] = [0, 0, 0, 0, 50, 60, 70, 80];
    let e_lanes: [u8; 8] = [0x80, 0x10, 0xFF, 0, 0, 0, 60, 80];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (a_lanes[i] as u128) << (i * 8);
        b |= (b_lanes[i] as u128) << (i * 8);
        e |= (e_lanes[i] as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VPwMax {
            elem: IRType::I8,
            count: 8,
            signed: false,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Symbolic universality (spec-replay): Iop_PwAdd16x4 must produce the same
/// bits as a hand-built reference for any 64-bit input pair. Claripy has no
/// `_op_generic_PwAdd`, so we reference-build inline per `z3-spec-replay-test-template`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vpwadd_16x4_matches_spec_replay() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vpwadd_a", 64);
    let b = RustBV::symbolic(&ctx, "vpwadd_b", 64);
    let got = VEXOps::binop(
        IROp::VPwAdd {
            elem: IRType::I16,
            count: 4,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();

    // Reference: 2 pairs from a, then 2 pairs from b.
    let mut lanes = Vec::with_capacity(4);
    for src in [&a, &b] {
        for i in 0..2u32 {
            let lo0 = 2 * i * 16;
            let lo1 = (2 * i + 1) * 16;
            let l0 = src.extract(lo0 + 15, lo0, &ctx);
            let l1 = src.extract(lo1 + 15, lo1, &ctx);
            lanes.push(l0.add_into(l1, &ctx));
        }
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VPwAdd 16x4 must match the spec-replay reference for all 64-bit inputs"
    );
    ctx.pop();
}

/// Symbolic universality (spec-replay): Iop_PwAddL16Sx4 widens each lane
/// before adding. Reference uses explicit sign-extend.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vpwaddl_16sx4_matches_spec_replay() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::symbolic(&ctx, "vpwaddl_a", 64);
    let got = VEXOps::unop(
        IROp::VPwAddL {
            elem: IRType::I16,
            count: 4,
            signed: true,
        },
        arg.clone(),
        &ctx,
    )
    .unwrap();

    // Reference: 2 pairs, each pair sign-extended to 32 bits then added.
    let mut lanes = Vec::with_capacity(2);
    for i in 0..2u32 {
        let lo0 = 2 * i * 16;
        let lo1 = (2 * i + 1) * 16;
        let l0 = arg.extract(lo0 + 15, lo0, &ctx).sign_extend_into(32, &ctx);
        let l1 = arg.extract(lo1 + 15, lo1, &ctx).sign_extend_into(32, &ctx);
        lanes.push(l0.add_into(l1, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VPwAddL 16Sx4 must match the spec-replay sign-extend reference"
    );
    ctx.pop();
}

/// Symbolic universality (spec-replay): Iop_PwMin16Sx4 — signed pairwise
/// min via SLE/ITE for each pair.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vpwmin_16sx4_matches_spec_replay() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vpwmin_a", 64);
    let b = RustBV::symbolic(&ctx, "vpwmin_b", 64);
    let got = VEXOps::binop(
        IROp::VPwMin {
            elem: IRType::I16,
            count: 4,
            signed: true,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();

    let mut lanes = Vec::with_capacity(4);
    for src in [&a, &b] {
        for i in 0..2u32 {
            let lo0 = 2 * i * 16;
            let lo1 = (2 * i + 1) * 16;
            let l0 = src.extract(lo0 + 15, lo0, &ctx);
            let l1 = src.extract(lo1 + 15, lo1, &ctx);
            let cond = l0.clone().sle_into(l1.clone(), &ctx);
            lanes.push(cond.ite_into(l0, l1, &ctx));
        }
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VPwMin 16Sx4 must match the spec-replay signed-min reference"
    );
    ctx.pop();
}

/// Parse routing: Iop_PwAdd / PwAddL / PwMin / PwMax variants land on
/// VPwAdd / VPwAddL / VPwMin / VPwMax with the expected decomposition.
/// Iop_PwAdd32Fx2 routes to VFPwAdd (angr-cudgw.6).
#[test]
fn test_parse_pairwise_routing() {
    use crate::vex::opcode_map::parse_opcode;

    let pwadd_cases: &[(&str, IRType, u8)] = &[
        ("Iop_PwAdd8x8", IRType::I8, 8),
        ("Iop_PwAdd16x4", IRType::I16, 4),
        ("Iop_PwAdd32x2", IRType::I32, 2),
        ("Iop_PwAdd8x16", IRType::I8, 16),
        ("Iop_PwAdd16x8", IRType::I16, 8),
        ("Iop_PwAdd32x4", IRType::I32, 4),
    ];
    for (op, e, c) in pwadd_cases {
        match parse_opcode(op) {
            IROp::VPwAdd { elem, count } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
            }
            other => panic!("{}: expected VPwAdd, got {:?}", op, other),
        }
    }

    let pwaddl_cases: &[(&str, IRType, u8, bool)] = &[
        ("Iop_PwAddL8Sx8", IRType::I8, 8, true),
        ("Iop_PwAddL8Ux8", IRType::I8, 8, false),
        ("Iop_PwAddL16Sx4", IRType::I16, 4, true),
        ("Iop_PwAddL16Ux4", IRType::I16, 4, false),
        ("Iop_PwAddL32Sx2", IRType::I32, 2, true),
        ("Iop_PwAddL32Ux2", IRType::I32, 2, false),
        ("Iop_PwAddL8Sx16", IRType::I8, 16, true),
        ("Iop_PwAddL8Ux16", IRType::I8, 16, false),
        ("Iop_PwAddL16Sx8", IRType::I16, 8, true),
        ("Iop_PwAddL16Ux8", IRType::I16, 8, false),
        ("Iop_PwAddL32Sx4", IRType::I32, 4, true),
        ("Iop_PwAddL32Ux4", IRType::I32, 4, false),
    ];
    for (op, e, c, s) in pwaddl_cases {
        match parse_opcode(op) {
            IROp::VPwAddL {
                elem,
                count,
                signed,
            } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
                assert_eq!(signed, *s, "{}: signed", op);
            }
            other => panic!("{}: expected VPwAddL, got {:?}", op, other),
        }
    }

    let pwmin_cases: &[(&str, IRType, u8, bool)] = &[
        ("Iop_PwMin8Sx8", IRType::I8, 8, true),
        ("Iop_PwMin8Ux8", IRType::I8, 8, false),
        ("Iop_PwMin16Sx4", IRType::I16, 4, true),
        ("Iop_PwMin16Ux4", IRType::I16, 4, false),
        ("Iop_PwMin32Sx2", IRType::I32, 2, true),
        ("Iop_PwMin32Ux2", IRType::I32, 2, false),
    ];
    for (op, e, c, s) in pwmin_cases {
        match parse_opcode(op) {
            IROp::VPwMin {
                elem,
                count,
                signed,
            } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
                assert_eq!(signed, *s, "{}: signed", op);
            }
            other => panic!("{}: expected VPwMin, got {:?}", op, other),
        }
    }

    let pwmax_cases: &[(&str, IRType, u8, bool)] = &[
        ("Iop_PwMax8Sx8", IRType::I8, 8, true),
        ("Iop_PwMax8Ux8", IRType::I8, 8, false),
        ("Iop_PwMax16Sx4", IRType::I16, 4, true),
        ("Iop_PwMax16Ux4", IRType::I16, 4, false),
        ("Iop_PwMax32Sx2", IRType::I32, 2, true),
        ("Iop_PwMax32Ux2", IRType::I32, 2, false),
    ];
    for (op, e, c, s) in pwmax_cases {
        match parse_opcode(op) {
            IROp::VPwMax {
                elem,
                count,
                signed,
            } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
                assert_eq!(signed, *s, "{}: signed", op);
            }
            other => panic!("{}: expected VPwMax, got {:?}", op, other),
        }
    }

    // Float pairwise add (Iop_PwAdd32Fx2) routes to VFPwAdd via parse_float
    // (angr-cudgw.6), not NeonUnimplemented.
    match parse_opcode("Iop_PwAdd32Fx2") {
        IROp::VFPwAdd { elem, count } => {
            assert_eq!(elem, IRType::F32, "PwAdd32Fx2: elem");
            assert_eq!(count, 2, "PwAdd32Fx2: count");
        }
        other => panic!("Iop_PwAdd32Fx2 expected VFPwAdd, got {:?}", other),
    }
}

// =========================================================================
// angr-tukg.3 — NEON rounding halving add (VAvg).
// =========================================================================

/// Iop_Avg8Ux8 — unsigned rounding-average over 8 lanes of 8 bits.
/// Exercises the round-up at the half-way point (`(a+b+1) >> 1`) and the
/// no-overflow guarantee for 0xFF+0xFF.
#[test]
fn test_vavg_8ux8_concrete() {
    let ctx = SymContext::new_mock();
    // Per-lane: rounded average of u8 values.
    //   lane 0: avg(0, 0) = 0.
    //   lane 1: avg(1, 1) = 1.
    //   lane 2: avg(1, 2) = 2  (round up; truncating would give 1).
    //   lane 3: avg(0xFF, 0xFF) = 0xFF (no overflow — widening absorbs +1).
    //   lane 4: avg(0xFE, 0xFF) = 0xFF (round up; truncating would give 0xFE).
    //   lane 5: avg(0x10, 0x20) = 0x18.
    //   lane 6: avg(0x80, 0x80) = 0x80.
    //   lane 7: avg(0x7F, 0x01) = 0x40.
    let a_lanes: [u8; 8] = [0, 1, 1, 0xFF, 0xFE, 0x10, 0x80, 0x7F];
    let b_lanes: [u8; 8] = [0, 1, 2, 0xFF, 0xFF, 0x20, 0x80, 0x01];
    let e_lanes: [u8; 8] = [0, 1, 2, 0xFF, 0xFF, 0x18, 0x80, 0x40];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (a_lanes[i] as u128) << (i * 8);
        b |= (b_lanes[i] as u128) << (i * 8);
        e |= (e_lanes[i] as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VAvg {
            elem: IRType::I8,
            count: 8,
            signed: false,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Avg16Ux8 — Q-reg variant: 8 lanes of 16 bits. 0xFFFF+0xFFFF
/// rounding-average must stay 0xFFFF (no truncation loss).
#[test]
fn test_vavg_16ux8_concrete() {
    let ctx = SymContext::new_mock();
    let a_lanes: [u16; 8] = [0, 1, 0xFFFF, 0xFFFE, 0x1000, 0x8000, 0x7FFF, 0x0123];
    let b_lanes: [u16; 8] = [0, 2, 0xFFFF, 0xFFFF, 0x2000, 0x8000, 0x0001, 0x0456];
    // Avg = (a+b+1) >> 1
    let e_lanes: [u16; 8] = [0, 2, 0xFFFF, 0xFFFF, 0x1800, 0x8000, 0x4000, 0x02BD];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (a_lanes[i] as u128) << (i * 16);
        b |= (b_lanes[i] as u128) << (i * 16);
        e |= (e_lanes[i] as u128) << (i * 16);
    }
    let result = VEXOps::binop(
        IROp::VAvg {
            elem: IRType::I16,
            count: 8,
            signed: false,
        },
        RustBV::concrete(a, 128),
        RustBV::concrete(b, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_Avg8Sx8 — signed rounding-average. Validates that two -128 lanes
/// give -128 (sign-extension fence) and mixed-sign lanes round correctly.
#[test]
fn test_vavg_8sx8_concrete() {
    let ctx = SymContext::new_mock();
    // Per-lane signed avg with round-half-up.
    //   lane 0: avg(-128, -128) = -128.
    //   lane 1: avg(127, 127)   = 127.
    //   lane 2: avg(-1, 0)      = 0 (round up: (-1+0+1)/2=0).
    //   lane 3: avg(-2, -1)     = -1.
    //   lane 4: avg(-100, 100)  = 0.
    //   lane 5: avg(-100, 101)  = 1.
    //   lane 6: avg(127, -128)  = 0 (the +1 makes the sum -1+1=0; >>1=0).
    //   lane 7: avg(50, 51)     = 51.
    let a_lanes: [i8; 8] = [-128, 127, -1, -2, -100, -100, 127, 50];
    let b_lanes: [i8; 8] = [-128, 127, 0, -1, 100, 101, -128, 51];
    let e_lanes: [i8; 8] = [-128, 127, 0, -1, 0, 1, 0, 51];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= ((a_lanes[i] as u8) as u128) << (i * 8);
        b |= ((b_lanes[i] as u8) as u128) << (i * 8);
        e |= ((e_lanes[i] as u8) as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VAvg {
            elem: IRType::I8,
            count: 8,
            signed: true,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Symbolic universality (spec-replay): Iop_Avg16Ux4 must equal the
/// reference `((zext(a)+zext(b)+1) >> 1)[15:0]` per lane for all 64-bit
/// inputs. Claripy has no `_op_generic_Avg`; the test uses the
/// `z3-spec-replay-test-template` pattern.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vavg_16ux4_symbolic_universal_unsigned() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vavg_a", 64);
    let b = RustBV::symbolic(&ctx, "vavg_b", 64);
    let got = VEXOps::binop(
        IROp::VAvg {
            elem: IRType::I16,
            count: 4,
            signed: false,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();

    let mut lanes = Vec::with_capacity(4);
    for i in 0..4u32 {
        let lo = i * 16;
        let hi = lo + 15;
        let al = a.extract(hi, lo, &ctx).zero_extend_into(17, &ctx);
        let bl = b.extract(hi, lo, &ctx).zero_extend_into(17, &ctx);
        let sum = al
            .add_into(bl, &ctx)
            .add_into(RustBV::concrete(1, 17), &ctx);
        let shifted = sum.lshr_into(RustBV::concrete(1, 17), &ctx);
        lanes.push(shifted.extract(15, 0, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VAvg 16Ux4 must match the unsigned spec-replay reference"
    );
    ctx.pop();
}

/// Symbolic universality (spec-replay): Iop_Avg8Sx8 signed rounding-avg.
/// Reference sign-extends each lane to 9 bits before summing.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vavg_8sx8_symbolic_universal_signed() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vavg_sa", 64);
    let b = RustBV::symbolic(&ctx, "vavg_sb", 64);
    let got = VEXOps::binop(
        IROp::VAvg {
            elem: IRType::I8,
            count: 8,
            signed: true,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();

    let mut lanes = Vec::with_capacity(8);
    for i in 0..8u32 {
        let lo = i * 8;
        let hi = lo + 7;
        let al = a.extract(hi, lo, &ctx).sign_extend_into(9, &ctx);
        let bl = b.extract(hi, lo, &ctx).sign_extend_into(9, &ctx);
        let sum = al.add_into(bl, &ctx).add_into(RustBV::concrete(1, 9), &ctx);
        let shifted = sum.lshr_into(RustBV::concrete(1, 9), &ctx);
        lanes.push(shifted.extract(7, 0, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VAvg 8Sx8 must match the signed spec-replay reference"
    );
    ctx.pop();
}

/// Parse routing: all 12 Iop_Avg variants land on IROp::VAvg with the
/// expected `(elem, count, signed)` decomposition. No Avg op should remain
/// in NeonUnimplemented.
#[test]
fn test_parse_avg_routing() {
    use crate::vex::opcode_map::parse_opcode;

    let cases: &[(&str, IRType, u8, bool)] = &[
        ("Iop_Avg8Ux8", IRType::I8, 8, false),
        ("Iop_Avg16Ux4", IRType::I16, 4, false),
        ("Iop_Avg32Ux2", IRType::I32, 2, false),
        ("Iop_Avg8Sx8", IRType::I8, 8, true),
        ("Iop_Avg16Sx4", IRType::I16, 4, true),
        ("Iop_Avg32Sx2", IRType::I32, 2, true),
        ("Iop_Avg8Ux16", IRType::I8, 16, false),
        ("Iop_Avg16Ux8", IRType::I16, 8, false),
        ("Iop_Avg32Ux4", IRType::I32, 4, false),
        ("Iop_Avg8Sx16", IRType::I8, 16, true),
        ("Iop_Avg16Sx8", IRType::I16, 8, true),
        ("Iop_Avg32Sx4", IRType::I32, 4, true),
    ];
    for (op, e, c, s) in cases {
        match parse_opcode(op) {
            IROp::VAvg {
                elem,
                count,
                signed,
            } => {
                assert_eq!(elem, *e, "{}: elem", op);
                assert_eq!(count, *c, "{}: count", op);
                assert_eq!(signed, *s, "{}: signed", op);
            }
            other => panic!("{}: expected VAvg, got {:?}", op, other),
        }
    }
}
