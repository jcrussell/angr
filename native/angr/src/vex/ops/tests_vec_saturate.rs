// angr-9hleg: vector saturating narrow/add/sub/shift tests (mirror of ops/vec_saturate.rs).

use super::*;

#[test]
fn test_vnarrow_un_16to8x8_concrete() {
    let ctx = SymContext::new_mock();
    // Input V128 = 8 lanes of I16: 0x1122, 0x3344, 0x5566, 0x7788, 0x99AA,
    // 0xBBCC, 0xDDEE, 0xFF00.
    let lanes_in: [u16; 8] = [
        0x1122, 0x3344, 0x5566, 0x7788, 0x99AA, 0xBBCC, 0xDDEE, 0xFF00,
    ];
    let mut v: u128 = 0;
    for (i, &lane) in lanes_in.iter().enumerate() {
        v |= (lane as u128) << (i * 16);
    }
    let arg = RustBV::concrete(v, 128);
    let res = VEXOps::unop(
        IROp::VNarrowUn {
            from: IRType::I16,
            count: 8,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 64);
    let out = res.as_u128().unwrap();
    // Each output lane = low byte of input lane.
    for (i, &lane) in lanes_in.iter().enumerate() {
        let got = (out >> (i * 8)) & 0xFF;
        assert_eq!(got, (lane & 0xFF) as u128, "out lane {i}");
    }
}

#[test]
fn test_vnarrow_bin_16to8x16_concrete() {
    let ctx = SymContext::new_mock();
    // Each input is V128 with 8 lanes of I16. left lanes -> output lanes 0..8;
    // right lanes -> output lanes 8..16.
    let lanes_l: [u16; 8] = [
        0x0011, 0x0022, 0x0033, 0x0044, 0x0055, 0x0066, 0x0077, 0x0088,
    ];
    let lanes_r: [u16; 8] = [
        0x0099, 0x00AA, 0x00BB, 0x00CC, 0x00DD, 0x00EE, 0x00FF, 0x0001,
    ];
    let mut l: u128 = 0;
    let mut r: u128 = 0;
    for (i, &lane) in lanes_l.iter().enumerate() {
        l |= (lane as u128) << (i * 16);
    }
    for (i, &lane) in lanes_r.iter().enumerate() {
        r |= (lane as u128) << (i * 16);
    }
    let res = VEXOps::binop(
        IROp::VNarrowBin {
            from: IRType::I16,
            count: 16,
        },
        RustBV::concrete(l, 128),
        RustBV::concrete(r, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let out = res.as_u128().unwrap();
    for (i, &lane) in lanes_l.iter().enumerate() {
        let got = (out >> (i * 8)) & 0xFF;
        assert_eq!(got, (lane & 0xFF) as u128, "left lane {i}");
    }
    for (i, &lane) in lanes_r.iter().enumerate() {
        let got = (out >> ((8 + i) * 8)) & 0xFF;
        assert_eq!(got, (lane & 0xFF) as u128, "right lane {i}");
    }
}

#[test]
fn test_vqnarrow_un_16sto8sx8_saturates() {
    let ctx = SymContext::new_mock();
    // signed I16 source, signed I8 target. Range [-128, 127].
    // Lane 0 = 1000 (clamps to 127), lane 1 = -200 (clamps to -128 = 0x80),
    // lane 2 = 50 (passes through), lane 3 = -50 (passes through).
    let v: u128 = (1000i16 as u16 as u128)
        | ((-200i16 as u16 as u128) << 16)
        | ((50i16 as u16 as u128) << 32)
        | ((-50i16 as u16 as u128) << 48);
    let arg = RustBV::concrete(v, 128);
    let res = VEXOps::unop(
        IROp::VQNarrowUn {
            from: IRType::I16,
            count: 8,
            src_signed: true,
            dst_signed: true,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 64);
    let out = res.as_u128().unwrap();
    assert_eq!((out & 0xFF) as u8, 127, "lane 0 saturated to 127");
    assert_eq!(((out >> 8) & 0xFF) as u8, 0x80, "lane 1 saturated to -128");
    assert_eq!(((out >> 16) & 0xFF) as u8, 50, "lane 2 unchanged");
    assert_eq!(
        ((out >> 24) & 0xFF) as u8,
        (-50i8) as u8,
        "lane 3 unchanged"
    );
}

#[test]
fn test_vqnarrow_un_16sto8ux8_signed_to_unsigned() {
    let ctx = SymContext::new_mock();
    // Signed source -> unsigned dst. Range [0, 255].
    // -1 (0xFFFF) -> 0; 300 -> 255; 100 -> 100.
    let v: u128 = (0xFFFFu128) | ((300u128) << 16) | ((100u128) << 32);
    let arg = RustBV::concrete(v, 128);
    let res = VEXOps::unop(
        IROp::VQNarrowUn {
            from: IRType::I16,
            count: 8,
            src_signed: true,
            dst_signed: false,
        },
        arg,
        &ctx,
    )
    .unwrap();
    let out = res.as_u128().unwrap();
    assert_eq!((out & 0xFF) as u8, 0, "negative -> 0");
    assert_eq!(((out >> 8) & 0xFF) as u8, 255, "300 -> 255");
    assert_eq!(((out >> 16) & 0xFF) as u8, 100, "passes through");
}

#[test]
#[allow(clippy::identity_op)] // explicit 8-lane layout reads better than the minimized form
fn test_vqnarrow_un_16uto8ux8_unsigned() {
    let ctx = SymContext::new_mock();
    // Unsigned source -> unsigned dst. Range [0, 255]. 256 saturates to 255.
    let v: u128 = (256u128) | ((100u128) << 16) | ((0u128) << 32) | ((0xFFFFu128) << 48);
    let arg = RustBV::concrete(v, 128);
    let res = VEXOps::unop(
        IROp::VQNarrowUn {
            from: IRType::I16,
            count: 8,
            src_signed: false,
            dst_signed: false,
        },
        arg,
        &ctx,
    )
    .unwrap();
    let out = res.as_u128().unwrap();
    assert_eq!((out & 0xFF) as u8, 255, "256 -> 255");
    assert_eq!(((out >> 8) & 0xFF) as u8, 100);
    assert_eq!(((out >> 16) & 0xFF) as u8, 0);
    assert_eq!(((out >> 24) & 0xFF) as u8, 255, "0xFFFF -> 255");
}

#[test]
fn test_vqnarrow_bin_16sto8sx16_two_inputs() {
    let ctx = SymContext::new_mock();
    // 8 lanes per input of signed I16. Left lane 0 = 200 (>127 -> 127),
    // right lane 7 = -300 (< -128 -> -128).
    let mut l: u128 = 0;
    let mut r: u128 = 0;
    l |= 200u128;
    l |= (50i16 as u16 as u128) << 16;
    r |= 5i16 as u16 as u128;
    r |= ((-300i16) as u16 as u128) << (7 * 16);

    let res = VEXOps::binop(
        IROp::VQNarrowBin {
            from: IRType::I16,
            count: 16,
            src_signed: true,
            dst_signed: true,
        },
        RustBV::concrete(l, 128),
        RustBV::concrete(r, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let out = res.as_u128().unwrap();
    // Left lane 0 -> output byte 0: 127.
    assert_eq!((out & 0xFF) as u8, 127, "left lane 0 sat to 127");
    // Left lane 1 -> output byte 1: 50.
    assert_eq!(((out >> 8) & 0xFF) as u8, 50, "left lane 1 unchanged");
    // Right lane 0 -> output byte 8: 5.
    assert_eq!(((out >> 64) & 0xFF) as u8, 5, "right lane 0 unchanged");
    // Right lane 7 -> output byte 15: -128.
    assert_eq!(
        ((out >> (15 * 8)) & 0xFF) as u8,
        0x80,
        "right lane 7 sat to -128"
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vqnarrow_un_symbolic_saturates() {
    // Symbolic I16 saturating to signed I8. Constrain output lane to 127
    // and require source > 127 to confirm saturation kicked in.
    let ctx = SymContext::new_mock();
    let arg = RustBV::symbolic(&ctx, "qn_arg", 128); // 8 lanes I16
    let res = VEXOps::unop(
        IROp::VQNarrowUn {
            from: IRType::I16,
            count: 8,
            src_signed: true,
            dst_signed: true,
        },
        arg.clone(),
        &ctx,
    )
    .unwrap();
    // Constrain output lane 0 = 127 AND source lane 0 = 200.
    let out_lane0 = res.extract(7, 0, &ctx);
    ctx.add_constraint(
        out_lane0
            .to_z3_ast()
            .eq(RustBV::concrete(127, 8).to_z3_ast()),
    );
    let src_lane0 = arg.extract(15, 0, &ctx);
    ctx.add_constraint(
        src_lane0
            .to_z3_ast()
            .eq(RustBV::concrete(200, 16).to_z3_ast()),
    );
    assert!(
        ctx.is_sat(),
        "expected SAT: src lane 0 = 200 saturates to 127"
    );
}

/// angr-36vvn.2 regression: the destination minimum (-128 narrowing I16→I8S)
/// must pass through UNCHANGED, not get mis-clamped to -127. The symbolic
/// `saturate_lane_symbolic` min bound was `-half + 1` (0xFF81) instead of
/// `-half` (0xFF80), so a source lane exactly at the true minimum tripped the
/// `lane < min` test and was clamped to 0x81 (-127).
#[test]
fn test_vqnarrow_un_symbolic_dest_min_passthrough() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::symbolic(&ctx, "qn_min_arg", 128); // 8 lanes I16
    let res = VEXOps::unop(
        IROp::VQNarrowUn {
            from: IRType::I16,
            count: 8,
            src_signed: true,
            dst_signed: true,
        },
        arg.clone(),
        &ctx,
    )
    .unwrap();
    // Source lane 0 = -128 (0xFF80 as sign-extended I16). It is exactly the
    // destination minimum, so it must narrow to 0x80 (-128), unchanged.
    let src_lane0 = arg.extract(15, 0, &ctx);
    ctx.add_constraint(
        src_lane0
            .to_z3_ast()
            .eq(RustBV::concrete(0xFF80, 16).to_z3_ast()),
    );
    let out_lane0 = res.extract(7, 0, &ctx);
    // With the fix out_lane0 == 0x80; require it to be equal (SAT) and the
    // buggy 0x81 to be impossible (UNSAT) under this source constraint.
    ctx.push();
    ctx.add_constraint(
        out_lane0
            .to_z3_ast()
            .eq(RustBV::concrete(0x80, 8).to_z3_ast()),
    );
    assert!(ctx.is_sat(), "dest-min -128 must pass through to 0x80");
    ctx.pop();
    ctx.add_constraint(
        out_lane0
            .to_z3_ast()
            .eq(RustBV::concrete(0x81, 8).to_z3_ast()),
    );
    assert!(
        !ctx.is_sat(),
        "dest-min -128 must NOT be mis-clamped to -127 (0x81)"
    );
}

// =========================================================================
// angr-tukg.1 — NEON saturating add/sub (VQAdd / VQSub).
// =========================================================================

/// Iop_QAdd8Sx8 — signed 8-bit saturating add over 8 lanes. Exercises:
/// non-overflowing add, positive overflow → INT8_MAX (0x7F), negative
/// overflow → INT8_MIN (0x80), exact-boundary cases.
#[test]
fn test_vqadd_8sx8_concrete() {
    let ctx = SymContext::new_mock();
    // Lane layout (LSB→MSB):
    //   0:  100 + 100 = 200, signed overflow → clamp to +127 (0x7F).
    //   1:  -100 + -100 = -200, signed underflow → clamp to -128 (0x80).
    //   2:   50 + 60 = 110, no overflow → 110 (0x6E).
    //   3:  -50 + -60 = -110, no overflow → -110 (0x92).
    //   4:  127 + 1  = INT_MAX+1 → clamp to +127.
    //   5: -128 + -1 = INT_MIN-1 → clamp to -128.
    //   6:  127 + -1 = 126 (no overflow).
    //   7: -128 + 1  = -127 (no overflow).
    let lanes_a: [i8; 8] = [100, -100, 50, -50, 127, -128, 127, -128];
    let lanes_b: [i8; 8] = [100, -100, 60, -60, 1, -1, -1, 1];
    let lanes_e: [i8; 8] = [127, -128, 110, -110, 127, -128, 126, -127];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= ((lanes_a[i] as u8) as u128) << (i * 8);
        b |= ((lanes_b[i] as u8) as u128) << (i * 8);
        e |= ((lanes_e[i] as u8) as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VQAdd {
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

/// Iop_QAdd8Ux8 — unsigned 8-bit saturating add. Overflow clamps to 0xFF.
#[test]
fn test_vqadd_8ux8_concrete() {
    let ctx = SymContext::new_mock();
    // 0: 200+100 = 300 → clamp to 255 (0xFF).
    // 1: 255+1   = 256 → clamp to 255.
    // 2: 0+0 → 0.
    // 3: 200+55 = 255 (boundary, no clamp).
    // 4: 200+56 = 256 → clamp.
    // 5: 50+50  = 100.
    // 6,7: 0 fillers.
    let lanes_a: [u8; 8] = [200, 255, 0, 200, 200, 50, 0, 0];
    let lanes_b: [u8; 8] = [100, 1, 0, 55, 56, 50, 0, 0];
    let lanes_e: [u8; 8] = [255, 255, 0, 255, 255, 100, 0, 0];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..8 {
        a |= (lanes_a[i] as u128) << (i * 8);
        b |= (lanes_b[i] as u128) << (i * 8);
        e |= (lanes_e[i] as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VQAdd {
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

/// Iop_QSub16Sx4 — signed 16-bit saturating sub. Verify both overflow
/// directions and the no-overflow path.
#[test]
fn test_vqsub_16sx4_concrete() {
    let ctx = SymContext::new_mock();
    // 0:  30000 - (-10000) = 40000 → clamp to 32767.
    // 1: -30000 - 10000    = -40000 → clamp to -32768.
    // 2: 100 - 50 = 50 (no overflow).
    // 3: -100 - (-50) = -50 (no overflow).
    let lanes_a: [i16; 4] = [30000, -30000, 100, -100];
    let lanes_b: [i16; 4] = [-10000, 10000, 50, -50];
    let lanes_e: [i16; 4] = [32767, -32768, 50, -50];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..4 {
        a |= ((lanes_a[i] as u16) as u128) << (i * 16);
        b |= ((lanes_b[i] as u16) as u128) << (i * 16);
        e |= ((lanes_e[i] as u16) as u128) << (i * 16);
    }
    let result = VEXOps::binop(
        IROp::VQSub {
            elem: IRType::I16,
            count: 4,
            signed: true,
        },
        RustBV::concrete(a, 64),
        RustBV::concrete(b, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_QSub32Ux4 — unsigned 32-bit saturating sub on a 128-bit Q-reg.
/// Underflow clamps to 0.
#[test]
fn test_vqsub_32ux4_concrete() {
    let ctx = SymContext::new_mock();
    let lanes_a: [u32; 4] = [100, 0xFFFF_FFFF, 1, 0];
    let lanes_b: [u32; 4] = [50, 1, 5, 5]; // underflow on lane 2 and 3
    let lanes_e: [u32; 4] = [50, 0xFFFF_FFFE, 0, 0];
    let mut a: u128 = 0;
    let mut b: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..4 {
        a |= (lanes_a[i] as u128) << (i * 32);
        b |= (lanes_b[i] as u128) << (i * 32);
        e |= (lanes_e[i] as u128) << (i * 32);
    }
    let result = VEXOps::binop(
        IROp::VQSub {
            elem: IRType::I32,
            count: 4,
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

/// Symbolic universality: Iop_QAdd8Sx8 must produce the same bits as the
/// claripy reference at `_op_generic_QAdd` for any 64-bit input. Built per
/// lane using the explicit sign-bit overflow formula (cap_cond + cap).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vqadd_8sx8_symbolic_matches_python_ref() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "vqadd_a", 64);
    let b = RustBV::symbolic(&ctx, "vqadd_b", 64);
    let got = VEXOps::binop(
        IROp::VQAdd {
            elem: IRType::I8,
            count: 8,
            signed: true,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();

    // Reference: per-lane signed saturating add as in irop.py.
    let smax = RustBV::concrete(0x7F, 8);
    let smin = RustBV::concrete(0x80, 8);
    let mut lanes = Vec::with_capacity(8);
    for i in 0..8u32 {
        let lo = i * 8;
        let hi = lo + 7;
        let a_lane = a.extract(hi, lo, &ctx);
        let b_lane = b.extract(hi, lo, &ctx);
        let res = a_lane.clone().add_into(b_lane.clone(), &ctx);
        let top_a = a_lane.extract(7, 7, &ctx);
        let top_b = b_lane.extract(7, 7, &ctx);
        let top_r = res.extract(7, 7, &ctx);
        // ~(top_a ^ top_b) & (top_a ^ top_r) == 1
        let signs_match = top_a.clone().xor_into(top_b, &ctx).not_into(&ctx);
        let r_flipped = top_a.xor_into(top_r.clone(), &ctx);
        let overflow = signs_match
            .and_into(r_flipped, &ctx)
            .eq(&RustBV::concrete(1, 1), &ctx);
        let cap = top_r
            .eq(&RustBV::concrete(1, 1), &ctx)
            .ite(&smax, &smin, &ctx);
        lanes.push(overflow.ite(&cap, &res, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);

    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    assert!(
        !ctx.is_sat(),
        "VQAdd 8Sx8 must match the claripy QAdd reference for all 64-bit inputs"
    );
    ctx.pop();
}

/// Parse routing: Iop_QAdd / Iop_QSub variants land on VQAdd / VQSub with
/// the expected (elem, count, signed) decomposition. Covers a sampling
/// across D-reg (total=64) and Q-reg (total=128) shapes plus both
/// signedness conventions.
#[test]
fn test_parse_vqaddsub_routing() {
    use crate::vex::opcode_map::parse_opcode;

    let qadd_cases: &[(&str, IRType, u8, bool)] = &[
        ("Iop_QAdd8Sx8", IRType::I8, 8, true),
        ("Iop_QAdd16Ux4", IRType::I16, 4, false),
        ("Iop_QAdd32Sx2", IRType::I32, 2, true),
        ("Iop_QAdd64Ux1", IRType::I64, 1, false),
        ("Iop_QAdd8Ux16", IRType::I8, 16, false),
        ("Iop_QAdd16Sx8", IRType::I16, 8, true),
        ("Iop_QAdd64Sx2", IRType::I64, 2, true),
    ];
    for (op, e, c, s) in qadd_cases {
        match parse_opcode(op) {
            IROp::VQAdd {
                elem,
                count,
                signed,
            } => {
                assert_eq!(elem, *e, "{op}: elem");
                assert_eq!(count, *c, "{op}: count");
                assert_eq!(signed, *s, "{op}: signed");
            }
            other => panic!("{op}: expected VQAdd, got {other:?}"),
        }
    }

    let qsub_cases: &[(&str, IRType, u8, bool)] = &[
        ("Iop_QSub8Sx8", IRType::I8, 8, true),
        ("Iop_QSub32Ux4", IRType::I32, 4, false),
        ("Iop_QSub64Sx2", IRType::I64, 2, true),
    ];
    for (op, e, c, s) in qsub_cases {
        match parse_opcode(op) {
            IROp::VQSub {
                elem,
                count,
                signed,
            } => {
                assert_eq!(elem, *e, "{op}: elem");
                assert_eq!(count, *c, "{op}: count");
                assert_eq!(signed, *s, "{op}: signed");
            }
            other => panic!("{op}: expected VQSub, got {other:?}"),
        }
    }
}

// =========================================================================
// angr-tukg.8 — NEON saturating vector shifts (VQShlSat).
// =========================================================================

/// Iop_QShl8x8 — unsigned saturating left shift by vector (D-reg).
/// Covers in-range left shift, OOR left shift (amt ≥ width → UMAX if
/// `a != 0` else 0), overflow saturation to UMAX, and the negative-amt
/// branch (right shift via logical shift, with OOR → 0).
#[test]
fn test_vqshl_8x8_concrete_unsigned() {
    let ctx = SymContext::new_mock();
    // amt is sign-extended as i8: 0xFF = -1, 0xFC = -4, 0xF8 = -8.
    let lanes_v: [u8; 8] = [0x01, 0x01, 0x01, 0x80, 0x40, 0xFF, 0x10, 0x80];
    let lanes_s: [u8; 8] = [0, 7, 8, 1, 1, 0xFF, 0xFC, 0xF8];
    // 0x01<<0  = 0x01.    0x01<<7  = 0x80.   0x01<<8 OOR, a!=0 → 0xFF.
    // 0x80<<1  overflow → 0xFF.              0x40<<1  = 0x80 (no overflow).
    // 0xFF >> 1 (amt=-1) = 0x7F (lshr).      0x10>>4 (amt=-4) = 0x01.
    // 0x80>>8 OOR → 0 (lshr).
    let lanes_e: [u8; 8] = [0x01, 0x80, 0xFF, 0xFF, 0x80, 0x7F, 0x01, 0x00];
    let (mut v, mut s, mut e) = (0u128, 0u128, 0u128);
    for i in 0..8 {
        v |= (lanes_v[i] as u128) << (i * 8);
        s |= (lanes_s[i] as u128) << (i * 8);
        e |= (lanes_e[i] as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VQShlSat {
            elem: IRType::I8,
            count: 8,
            signed: false,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_QSal16x4 — signed saturating left shift by vector (D-reg).
/// Covers positive overflow → SMAX, negative overflow → SMIN, in-range
/// shift, and the negative-amt branch (right shift via ashr, sign-fill).
#[test]
fn test_vqsal_16x4_concrete_signed() {
    let ctx = SymContext::new_mock();
    // i16: SMAX=0x7FFF, SMIN=0x8000, -1=0xFFFF.
    // lane 0: 0x1000 (positive) << 3 = 0x8000 (negative when truncated) →
    //   overflow → SMAX = 0x7FFF.
    // lane 1: 0xF000 (= -4096) << 1 = 0xE000 (= -8192); ashr(0xE000,1)
    //   = 0xF000 == a → no overflow → 0xE000.
    // lane 2: 0xFFFF (= -1) with amt = -1 (0xFFFF sign-extended): ashr
    //   by 1 → 0xFFFF (sign-fill).
    // lane 3: 0x0040 (positive) with amt = 16 (OOR): a > 0 → SMAX.
    let lanes_v: [u16; 4] = [0x1000, 0xF000, 0xFFFF, 0x0040];
    let lanes_s: [u16; 4] = [3, 1, 0xFFFF, 16];
    let lanes_e: [u16; 4] = [0x7FFF, 0xE000, 0xFFFF, 0x7FFF];
    let (mut v, mut s, mut e) = (0u128, 0u128, 0u128);
    for i in 0..4 {
        v |= (lanes_v[i] as u128) << (i * 16);
        s |= (lanes_s[i] as u128) << (i * 16);
        e |= (lanes_e[i] as u128) << (i * 16);
    }
    let result = VEXOps::binop(
        IROp::VQShlSat {
            elem: IRType::I16,
            count: 4,
            signed: true,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 64);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_QSal8x16 — signed saturating left shift over 16x i8 lanes
/// (Q-reg/128-bit). Covers OOR-with-negative-input → SMIN saturation
/// (a < 0 with amt > width).
#[test]
fn test_vqsal_8x16_concrete_smin_saturation() {
    let ctx = SymContext::new_mock();
    // Build a 16-lane vector: alternating positive overflow (a=1, amt=8 OOR)
    // and negative overflow (a=0xFF=-1, amt=8 OOR).
    // amt=8 for all lanes. Positive a=1 (>0) → SMAX=0x7F.
    //                     Negative a=0xFF (<0) → SMIN=0x80.
    let mut v: u128 = 0;
    let mut s: u128 = 0;
    let mut e: u128 = 0;
    for i in 0..16 {
        let (a, expected) = if i % 2 == 0 {
            (1u8, 0x7Fu8)
        } else {
            (0xFFu8, 0x80u8)
        };
        v |= (a as u128) << (i * 8);
        s |= 8u128 << (i * 8);
        e |= (expected as u128) << (i * 8);
    }
    let result = VEXOps::binop(
        IROp::VQShlSat {
            elem: IRType::I8,
            count: 16,
            signed: true,
        },
        RustBV::concrete(v, 128),
        RustBV::concrete(s, 128),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.width(), 128);
    assert_eq!(result.as_u128().unwrap(), e);
}

/// Iop_QShl64x1 — width=64 edge case (single-lane D-reg). Exercises the
/// elem_width = 64 boundary of the concrete fast path (elem_mask uses
/// the full u64 range; sign-extension via the |!elem_mask| branch).
#[test]
fn test_vqshl_64x1_concrete_width_boundary() {
    let ctx = SymContext::new_mock();
    // Unsigned: 0x0000_0000_0000_0001 << 63 = 0x8000_0000_0000_0000.
    // Round-trip: (0x8000... >> 63) = 1 == a. No overflow. Result OK.
    let v = 1u128;
    let s = 63u128;
    let e = 0x8000_0000_0000_0000u128;
    let result = VEXOps::binop(
        IROp::VQShlSat {
            elem: IRType::I64,
            count: 1,
            signed: false,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.as_u128().unwrap(), e);

    // Same input, signed: a=1 (positive), shift left by 63 → 0x8000...
    // which is SMIN as signed. Overflow → SMAX = 0x7FFF_FFFF_FFFF_FFFF.
    let result_s = VEXOps::binop(
        IROp::VQShlSat {
            elem: IRType::I64,
            count: 1,
            signed: true,
        },
        RustBV::concrete(v, 64),
        RustBV::concrete(s, 64),
        &ctx,
    )
    .unwrap();
    assert_eq!(result_s.as_u128().unwrap(), 0x7FFF_FFFF_FFFF_FFFFu128);
}

/// Parse routing for all 16 saturating-shift opcodes: Iop_QShl{N}x{M}
/// (signed=false) and Iop_QSal{N}x{M} (signed=true), 8 shapes each.
#[test]
fn test_parse_vqshlsat_routing() {
    use crate::vex::opcode_map::parse_opcode;
    let shapes: &[(&str, IRType, u8)] = &[
        ("8x8", IRType::I8, 8),
        ("16x4", IRType::I16, 4),
        ("32x2", IRType::I32, 2),
        ("64x1", IRType::I64, 1),
        ("8x16", IRType::I8, 16),
        ("16x8", IRType::I16, 8),
        ("32x4", IRType::I32, 4),
        ("64x2", IRType::I64, 2),
    ];
    for (sfx, elem_e, count_e) in shapes {
        for (prefix, want_signed) in [("Iop_QShl", false), ("Iop_QSal", true)] {
            let name = format!("{prefix}{sfx}");
            match parse_opcode(&name) {
                IROp::VQShlSat {
                    elem,
                    count,
                    signed,
                } => {
                    assert_eq!(elem, *elem_e, "{name}: elem");
                    assert_eq!(count, *count_e, "{name}: count");
                    assert_eq!(signed, want_signed, "{name}: signed");
                }
                other => panic!("{name}: expected VQShlSat, got {other:?}"),
            }
        }
    }
}

/// Symbolic parity: the saturating shift behaves identically to a hand-
/// rolled per-lane ITE chain over Z3 bvshl/bvlshr/bvashr + round-trip
/// overflow detection. There is no `_op_generic_QShl` in claripy, so the
/// reference is the same algorithm encoded straight from the spec. This
/// catches encoding mistakes (wrong cap, swapped then/else, sign-bit
/// extraction errors) without depending on a Python reference.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vqshl_16x4_symbolic_universal_unsigned() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "qshl_a_16x4", 64);
    let b = RustBV::symbolic(&ctx, "qshl_b_16x4", 64);
    let got = VEXOps::binop(
        IROp::VQShlSat {
            elem: IRType::I16,
            count: 4,
            signed: false,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();
    // Reference: per-lane spec replay with the same primitives as the
    // helper. Any encoding drift will surface as a SAT counter-example.
    let umax = RustBV::concrete(0xFFFFu128, 16);
    let zero16 = RustBV::concrete(0, 16);
    let bit_one = RustBV::concrete(1, 1);
    let mut lanes: Vec<RustBV> = Vec::with_capacity(4);
    for i in 0..4 {
        let lo = i * 16;
        let hi = lo + 15;
        let al = a.extract(hi, lo, &ctx);
        let bl = b.extract(hi, lo, &ctx);
        let shl_v = al.clone().shl_into(bl.clone(), &ctx);
        let recovered = shl_v.clone().lshr_into(bl.clone(), &ctx);
        let no_overflow = recovered.eq(&al, &ctx);
        let left_branch = no_overflow.ite(&shl_v, &umax, &ctx);
        let neg_amt = zero16.clone().sub_into(bl.clone(), &ctx);
        let right_branch = al.clone().lshr_into(neg_amt, &ctx);
        let amt_top = bl.extract(15, 15, &ctx);
        let amt_is_neg = amt_top.eq(&bit_one, &ctx);
        lanes.push(amt_is_neg.ite(&right_branch, &left_branch, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);
    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    let universal = !ctx.is_sat();
    ctx.pop();
    assert!(
        universal,
        "VQShlSat 16x4 (unsigned) must match the per-lane spec for all 64-bit inputs"
    );
}

/// Same parity check for the signed (QSal) branch — verifies the SMAX/
/// SMIN cap selection from the data-sign bit.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vqsal_16x4_symbolic_universal_signed() {
    let ctx = SymContext::new_mock();
    let a = RustBV::symbolic(&ctx, "qsal_a_16x4", 64);
    let b = RustBV::symbolic(&ctx, "qsal_b_16x4", 64);
    let got = VEXOps::binop(
        IROp::VQShlSat {
            elem: IRType::I16,
            count: 4,
            signed: true,
        },
        a.clone(),
        b.clone(),
        &ctx,
    )
    .unwrap();
    let smax = RustBV::concrete(0x7FFFu128, 16);
    let smin = RustBV::concrete(0x8000u128, 16);
    let zero16 = RustBV::concrete(0, 16);
    let bit_one = RustBV::concrete(1, 1);
    let mut lanes: Vec<RustBV> = Vec::with_capacity(4);
    for i in 0..4 {
        let lo = i * 16;
        let hi = lo + 15;
        let al = a.extract(hi, lo, &ctx);
        let bl = b.extract(hi, lo, &ctx);
        let shl_v = al.clone().shl_into(bl.clone(), &ctx);
        let recovered = shl_v.clone().ashr_into(bl.clone(), &ctx);
        let no_overflow = recovered.eq(&al, &ctx);
        let a_top = al.extract(15, 15, &ctx);
        let a_is_neg = a_top.eq(&bit_one, &ctx);
        let cap = a_is_neg.ite(&smin, &smax, &ctx);
        let left_branch = no_overflow.ite(&shl_v, &cap, &ctx);
        let neg_amt = zero16.clone().sub_into(bl.clone(), &ctx);
        let right_branch = al.clone().ashr_into(neg_amt, &ctx);
        let amt_top = bl.extract(15, 15, &ctx);
        let amt_is_neg = amt_top.eq(&bit_one, &ctx);
        lanes.push(amt_is_neg.ite(&right_branch, &left_branch, &ctx));
    }
    let py = VEXOps::concat_le_elements(lanes, &ctx);
    ctx.push();
    ctx.add_constraint(got.to_z3_ast().eq(py.to_z3_ast()).not());
    let universal = !ctx.is_sat();
    ctx.pop();
    assert!(
        universal,
        "VQShlSat 16x4 (signed) must match the per-lane spec for all 64-bit inputs"
    );
}
