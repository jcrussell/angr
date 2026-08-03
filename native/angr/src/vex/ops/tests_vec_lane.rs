// angr-9hleg: NEON vector lane get/set/dup/widen/mul tests (mirror of ops/vec_lane.rs).

use super::*;
use crate::vex::ir::IRType;

// -------------------------------------------------------------------------
// NEON SIMD (angr-bkcs.2)
// -------------------------------------------------------------------------

#[test]
fn test_vmul_8x8_concrete() {
    let ctx = SymContext::new_mock();
    // 8 lanes of 8-bit, lane i = i for both → product = i*i mod 256.
    // l = 0x0706050403020100, r = same. result lane i = i*i.
    let l = RustBV::concrete(0x0706_0504_0302_0100u128, 64);
    let r = RustBV::concrete(0x0706_0504_0302_0100u128, 64);
    let res = VEXOps::binop(
        IROp::VMul {
            elem: IRType::I8,
            count: 8,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 64);
    let v = res.as_u128().unwrap();
    // lane i (bits [8i+7:8i]) should equal i*i.
    for i in 0u128..8 {
        let lane = (v >> (i * 8)) & 0xFF;
        assert_eq!(lane, (i * i) & 0xFF, "lane {i} of Mul8x8");
    }
}

#[test]
fn test_vmul_8x16_concrete() {
    let ctx = SymContext::new_mock();
    // All lanes = 3, multiplied by all lanes = 5 → all lanes = 15.
    let lo64 = 0x0303_0303_0303_0303u128;
    let l = RustBV::concrete(lo64 | (lo64 << 64), 128);
    let lo5 = 0x0505_0505_0505_0505u128;
    let r = RustBV::concrete(lo5 | (lo5 << 64), 128);
    let res = VEXOps::binop(
        IROp::VMul {
            elem: IRType::I8,
            count: 16,
        },
        l,
        r,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let v = res.as_u128().unwrap();
    for i in 0..16 {
        let lane = (v >> (i * 8)) & 0xFF;
        assert_eq!(lane, 15, "lane {i} of Mul8x16");
    }
}

#[test]
fn test_vget_elem_8x8_concrete() {
    let ctx = SymContext::new_mock();
    let vec = RustBV::concrete(0x8877_6655_4433_2211u128, 64);
    // Lane 0 = 0x11, lane 7 = 0x88.
    for (lane, expected) in [
        (0u128, 0x11),
        (1, 0x22),
        (2, 0x33),
        (3, 0x44),
        (4, 0x55),
        (5, 0x66),
        (6, 0x77),
        (7, 0x88),
    ] {
        let idx = RustBV::concrete(lane, 8);
        let res = VEXOps::binop(
            IROp::VGetElem {
                elem: IRType::I8,
                count: 8,
            },
            vec.clone(),
            idx,
            &ctx,
        )
        .unwrap();
        assert_eq!(res.width(), 8);
        assert_eq!(res.as_u128().unwrap(), expected, "lane {lane}");
    }
}

#[test]
fn test_vget_elem_16x8_concrete() {
    let ctx = SymContext::new_mock();
    // V128 with 8 lanes of 16 bits. Lane 3 = 0xDEAD.
    let mut payload: u128 = 0;
    payload |= 0xDEAD_u128 << (3 * 16);
    payload |= 0xBEEF_u128 << (7 * 16);
    let vec = RustBV::concrete(payload, 128);
    let res = VEXOps::binop(
        IROp::VGetElem {
            elem: IRType::I16,
            count: 8,
        },
        vec.clone(),
        RustBV::concrete(3, 8),
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 16);
    assert_eq!(res.as_u128().unwrap(), 0xDEAD);

    let res = VEXOps::binop(
        IROp::VGetElem {
            elem: IRType::I16,
            count: 8,
        },
        vec,
        RustBV::concrete(7, 8),
        &ctx,
    )
    .unwrap();
    assert_eq!(res.as_u128().unwrap(), 0xBEEF);
}

#[test]
fn test_vget_elem_64x2_concrete() {
    let ctx = SymContext::new_mock();
    let lo = 0xAAAA_BBBB_CCCC_DDDDu128;
    let hi = 0x1111_2222_3333_4444u128;
    let vec = RustBV::concrete(lo | (hi << 64), 128);
    let r0 = VEXOps::binop(
        IROp::VGetElem {
            elem: IRType::I64,
            count: 2,
        },
        vec.clone(),
        RustBV::concrete(0, 8),
        &ctx,
    )
    .unwrap();
    assert_eq!(r0.as_u128().unwrap(), lo);
    let r1 = VEXOps::binop(
        IROp::VGetElem {
            elem: IRType::I64,
            count: 2,
        },
        vec,
        RustBV::concrete(1, 8),
        &ctx,
    )
    .unwrap();
    assert_eq!(r1.as_u128().unwrap(), hi);
}

#[test]
fn test_vset_elem_8x8_concrete() {
    let ctx = SymContext::new_mock();
    let vec = RustBV::concrete(0x0u128, 64);
    // Set lane 3 to 0xFF.
    let res = VEXOps::binop_with_rm(
        IROp::VSetElem {
            elem: IRType::I8,
            count: 8,
        },
        vec,
        RustBV::concrete(3, 8),
        RustBV::concrete(0xFF, 8),
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 64);
    assert_eq!(res.as_u128().unwrap(), 0xFF00_0000u128);
}

#[test]
fn test_vset_elem_16x8_concrete() {
    let ctx = SymContext::new_mock();
    let vec = RustBV::concrete(0u128, 128);
    // Set lane 5 to 0xCAFE in a 16x8 vector.
    let res = VEXOps::binop_with_rm(
        IROp::VSetElem {
            elem: IRType::I16,
            count: 8,
        },
        vec,
        RustBV::concrete(5, 8),
        RustBV::concrete(0xCAFE, 16),
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    assert_eq!(res.as_u128().unwrap(), 0xCAFE_u128 << (5 * 16));
}

#[test]
fn test_vset_elem_preserves_other_lanes() {
    let ctx = SymContext::new_mock();
    let vec = RustBV::concrete(0xDEAD_BEEF_CAFE_F00Du128, 64);
    // Overwrite lane 2 (byte 2) with 0x77.
    let res = VEXOps::binop_with_rm(
        IROp::VSetElem {
            elem: IRType::I8,
            count: 8,
        },
        vec,
        RustBV::concrete(2, 8),
        RustBV::concrete(0x77, 8),
        &ctx,
    )
    .unwrap();
    let v = res.as_u128().unwrap();
    // Original byte 2 was 0xFE; expect 0x77 in its place, rest unchanged.
    let expected: u128 = (0xDEAD_BEEF_CAFE_F00Du128 & !(0xFFu128 << 16)) | (0x77u128 << 16);
    assert_eq!(v, expected);
}

#[test]
fn test_vset_elem_round_trip_via_get() {
    let ctx = SymContext::new_mock();
    let vec = RustBV::concrete(0u128, 128);
    let inserted = VEXOps::binop_with_rm(
        IROp::VSetElem {
            elem: IRType::I32,
            count: 4,
        },
        vec,
        RustBV::concrete(2, 8),
        RustBV::concrete(0x1234_5678, 32),
        &ctx,
    )
    .unwrap();
    let lane = VEXOps::binop(
        IROp::VGetElem {
            elem: IRType::I32,
            count: 4,
        },
        inserted,
        RustBV::concrete(2, 8),
        &ctx,
    )
    .unwrap();
    assert_eq!(lane.as_u128().unwrap(), 0x1234_5678);
}

// -------------------------------------------------------------------
// NEON Dup / Widen / Narrow / QNarrow tests (angr-hzs0)
// -------------------------------------------------------------------

#[test]
fn test_vdup_8x8_concrete() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(0xABu128, 8);
    let res = VEXOps::unop(
        IROp::VDup {
            elem: IRType::I8,
            count: 8,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 64);
    assert_eq!(res.as_u128().unwrap(), 0xABAB_ABAB_ABAB_ABABu128);
}

#[test]
fn test_vdup_16x8_concrete() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(0xCAFEu128, 16);
    let res = VEXOps::unop(
        IROp::VDup {
            elem: IRType::I16,
            count: 8,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let expected: u128 = (0..8).fold(0u128, |acc, i| acc | (0xCAFE_u128 << (i * 16)));
    assert_eq!(res.as_u128().unwrap(), expected);
}

#[test]
fn test_vdup_32x4_concrete() {
    let ctx = SymContext::new_mock();
    let arg = RustBV::concrete(0xDEAD_BEEFu128, 32);
    let res = VEXOps::unop(
        IROp::VDup {
            elem: IRType::I32,
            count: 4,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let expected: u128 = (0..4).fold(0u128, |acc, i| acc | (0xDEAD_BEEF_u128 << (i * 32)));
    assert_eq!(res.as_u128().unwrap(), expected);
}

#[test]
fn test_vwiden_8sto16x8_signed() {
    let ctx = SymContext::new_mock();
    // 8 lanes of I8: lane 0 = 0xFF (= -1 signed), lane 1 = 0x7F (= 127),
    // lane 2 = 0x80 (= -128), rest = 0.
    let arg_val: u128 = 0xFFu128 | (0x7Fu128 << 8) | (0x80u128 << 16);
    let arg = RustBV::concrete(arg_val, 64);
    let res = VEXOps::unop(
        IROp::VWiden {
            from: IRType::I8,
            count: 8,
            signed: true,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let v = res.as_u128().unwrap();
    // Lane 0: 0xFFFF (sign-ext of 0xFF), Lane 1: 0x007F, Lane 2: 0xFF80.
    assert_eq!(v & 0xFFFF, 0xFFFF, "lane 0");
    assert_eq!((v >> 16) & 0xFFFF, 0x007F, "lane 1");
    assert_eq!((v >> 32) & 0xFFFF, 0xFF80, "lane 2");
    assert_eq!((v >> 48) & 0xFFFF, 0x0000, "lane 3");
}

#[test]
fn test_vwiden_8uto16x8_unsigned() {
    let ctx = SymContext::new_mock();
    let arg_val: u128 = 0xFFu128 | (0x80u128 << 8);
    let arg = RustBV::concrete(arg_val, 64);
    let res = VEXOps::unop(
        IROp::VWiden {
            from: IRType::I8,
            count: 8,
            signed: false,
        },
        arg,
        &ctx,
    )
    .unwrap();
    let v = res.as_u128().unwrap();
    assert_eq!(v & 0xFFFF, 0x00FF, "lane 0 zero-extended");
    assert_eq!((v >> 16) & 0xFFFF, 0x0080, "lane 1 zero-extended");
}

#[test]
fn test_vwiden_32sto64x2_signed() {
    let ctx = SymContext::new_mock();
    // Lane 0 = 0x80000000 (= INT_MIN signed), Lane 1 = 0x12345678.
    let arg_val: u128 = 0x8000_0000u128 | (0x1234_5678u128 << 32);
    let arg = RustBV::concrete(arg_val, 64);
    let res = VEXOps::unop(
        IROp::VWiden {
            from: IRType::I32,
            count: 2,
            signed: true,
        },
        arg,
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 128);
    let v = res.as_u128().unwrap();
    let lo = v as u64;
    let hi = (v >> 64) as u64;
    assert_eq!(lo, 0xFFFF_FFFF_8000_0000u64, "lane 0 sign-extended");
    assert_eq!(hi, 0x0000_0000_1234_5678u64, "lane 1 zero-positive");
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vdup_symbolic_arg() {
    // Symbolic 8-bit value, dup to 8x8. Constrain the output to a known
    // pattern and check the solver picks the right scalar.
    let ctx = SymContext::new_mock();
    let arg = RustBV::symbolic(&ctx, "dup_arg", 8);
    let res = VEXOps::unop(
        IROp::VDup {
            elem: IRType::I8,
            count: 8,
        },
        arg.clone(),
        &ctx,
    )
    .unwrap();
    let target = RustBV::concrete(0x4242_4242_4242_4242u128, 64);
    ctx.add_constraint(res.to_z3_ast().eq(target.to_z3_ast()));
    assert!(ctx.is_sat(), "expected SAT for dup to 0x42 broadcast");
    let model_arg = ctx.eval(&arg).expect("eval(arg) None");
    assert_eq!(model_arg, 0x42);
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vget_elem_symbolic_idx() {
    // Build a concrete vector with distinct lane values, then read
    // through a symbolic idx and constrain it to return a specific lane.
    let ctx = SymContext::new_mock();
    let vec = RustBV::concrete(0x8877_6655_4433_2211u128, 64);
    let sym_idx = RustBV::symbolic(&ctx, "get_idx", 8);
    let res = VEXOps::binop(
        IROp::VGetElem {
            elem: IRType::I8,
            count: 8,
        },
        vec,
        sym_idx.clone(),
        &ctx,
    )
    .unwrap();
    assert_eq!(res.width(), 8);
    // Constrain result to 0x66 → solver must pick idx == 5.
    let target = RustBV::concrete(0x66, 8);
    ctx.add_constraint(res.to_z3_ast().eq(target.to_z3_ast()));
    assert!(ctx.is_sat(), "expected SAT for lane==0x66");
    let model_idx = ctx.eval(&sym_idx).expect("eval(idx) None");
    // idx must be 5 mod 8 (modulo because ITE chain ignores high bits).
    assert_eq!(model_idx & 0x7, 5, "expected idx&7 == 5, got {model_idx}");
}
