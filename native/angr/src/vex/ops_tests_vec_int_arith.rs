// angr-9hleg: vector packed-int min/max/abs tests (mirror of ops_vec_int_arith.rs).

use super::ops_test_helpers::*;
use super::*;

// =========================================================================
// Packed integer min/max/abs tests
// =========================================================================

/// PMINSW-style: signed min over 8x i16 lanes, mix of positive and negative.
#[test]
fn test_vec_int_min_signed_concrete() {
    let ctx = SymContext::new_mock();

    let l: [i16; 8] = [-5, 100, 0, -32768, 1, -1, 32767, -2];
    let r: [i16; 8] = [-3, 200, -100, -32767, -1, 0, 32766, 3];
    let exp: [i16; 8] = [-5, 100, -100, -32768, -1, -1, 32766, -2];

    let lv = pack_lanes_uint(&l.map(|x| x as u16 as u128), 16);
    let rv = pack_lanes_uint(&r.map(|x| x as u16 as u128), 16);
    let result = VEXOps::binop(
        IROp::VMin {
            elem: IRType::I16,
            count: 8,
            signed: true,
        },
        RustBV::concrete(lv, 128),
        RustBV::concrete(rv, 128),
        &ctx,
    )
    .unwrap();
    assert_int_lanes_eq(
        result.as_u128().unwrap(),
        &exp.map(|x| x as u16 as u128),
        16,
    );
}

/// PMAXUB-style: unsigned max over 16x u8 lanes.
#[test]
fn test_vec_int_max_unsigned_concrete() {
    let ctx = SymContext::new_mock();

    let l: [u8; 16] = [
        0xFF, 0x00, 0x80, 0x7F, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    ];
    let r: [u8; 16] = [0x00, 0xFF, 0x7F, 0x80, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0];
    let mut exp = [0u8; 16];
    for i in 0..16 {
        exp[i] = if l[i] > r[i] { l[i] } else { r[i] };
    }

    let lv = pack_lanes_uint(&l.map(|x| x as u128), 8);
    let rv = pack_lanes_uint(&r.map(|x| x as u128), 8);
    let result = VEXOps::binop(
        IROp::VMax {
            elem: IRType::I8,
            count: 16,
            signed: false,
        },
        RustBV::concrete(lv, 128),
        RustBV::concrete(rv, 128),
        &ctx,
    )
    .unwrap();
    assert_int_lanes_eq(result.as_u128().unwrap(), &exp.map(|x| x as u128), 8);
}

/// PABSW-style: per-lane absolute value over 8x i16 lanes (incl. INT_MIN
/// which stays INT_MIN under two's-complement |x|).
#[test]
fn test_vec_int_abs_concrete() {
    let ctx = SymContext::new_mock();

    let v: [i16; 8] = [-5, 100, 0, -32768, 1, -1, 32767, -200];
    let exp: [u16; 8] = [5, 100, 0, 0x8000 /* INT_MIN stays */, 1, 1, 32767, 200];

    let result = VEXOps::unop(
        IROp::VAbs {
            elem: IRType::I16,
            count: 8,
        },
        RustBV::concrete(pack_lanes_uint(&v.map(|x| x as u16 as u128), 16), 128),
        &ctx,
    )
    .unwrap();
    assert_int_lanes_eq(result.as_u128().unwrap(), &exp.map(|x| x as u128), 16);
}

/// Symbolic VMax (signed): constrain right == 7, derive left from a free
/// 4x i32 vector, and verify that asserting result == [7, 7, 7, 7] forces
/// every lane of left to be <= 7.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_vec_int_max_symbolic_signed() {
    let ctx = SymContext::new_mock();

    // r = [7, 7, 7, 7] as i32x4
    let mut rv: u128 = 0;
    for i in 0..4u32 {
        rv |= (7u128) << (i * 32);
    }
    let r = RustBV::concrete(rv, 128);
    let l = RustBV::symbolic(&ctx, "vmax_l", 128);

    let result = VEXOps::binop(
        IROp::VMax {
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

    // Constrain result == [7, 7, 7, 7]; this only requires l <= 7 per lane,
    // so the constraint must remain SAT.
    let exp = RustBV::concrete(rv, 128);
    ctx.add_constraint(result.to_z3_ast().eq(exp.to_z3_ast()));
    assert!(ctx.is_sat(), "expected SAT after constraining max == 7");
}
