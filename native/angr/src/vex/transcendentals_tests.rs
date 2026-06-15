// Tests for vex/transcendentals.rs — extracted from the inline `mod tests`.
// See parent module for the transcendental VEX op implementations under test.

use super::*;

fn bv64(f: f64) -> RustBV {
    RustBV::concrete(f.to_bits() as u128, 64)
}
fn bv32(f: f32) -> RustBV {
    RustBV::concrete(f.to_bits() as u128, 32)
}
fn rm() -> RustBV {
    RustBV::concrete(0, 32)
}
fn extract_f64(bv: RustBV) -> f64 {
    f64::from_bits(bv.as_u128().unwrap() as u64)
}
fn extract_f32(bv: RustBV) -> f32 {
    f32::from_bits(bv.as_u128().unwrap() as u32)
}

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps || ((a - b).abs() / b.abs().max(1e-300)) < eps
}

#[test]
fn sin_cos_tan_concrete() {
    let r = try_concrete_binop_rm(IOP_SIN_F64, &rm(), &bv64(std::f64::consts::PI / 6.0)).unwrap();
    assert!(approx_eq(extract_f64(r), 0.5, 1e-12));

    let r = try_concrete_binop_rm(IOP_COS_F64, &rm(), &bv64(0.0)).unwrap();
    assert_eq!(extract_f64(r), 1.0);

    let r = try_concrete_binop_rm(IOP_TAN_F64, &rm(), &bv64(std::f64::consts::PI / 4.0)).unwrap();
    assert!(approx_eq(extract_f64(r), 1.0, 1e-12));
}

#[test]
fn two_xm1_concrete() {
    // 2^0 - 1 = 0
    let r = try_concrete_binop_rm(IOP_2XM1_F64, &rm(), &bv64(0.0)).unwrap();
    assert_eq!(extract_f64(r), 0.0);
    // 2^1 - 1 = 1
    let r = try_concrete_binop_rm(IOP_2XM1_F64, &rm(), &bv64(1.0)).unwrap();
    assert!(approx_eq(extract_f64(r), 1.0, 1e-12));
    // 2^-1 - 1 = -0.5
    let r = try_concrete_binop_rm(IOP_2XM1_F64, &rm(), &bv64(-1.0)).unwrap();
    assert!(approx_eq(extract_f64(r), -0.5, 1e-12));
}

#[test]
fn yl2x_concrete() {
    // 1 * log2(8) = 3
    let r = try_concrete_triop_rm(IOP_YL2X_F64, &rm(), &bv64(1.0), &bv64(8.0)).unwrap();
    assert!(approx_eq(extract_f64(r), 3.0, 1e-12));
    // 2 * log2(4) = 4
    let r = try_concrete_triop_rm(IOP_YL2X_F64, &rm(), &bv64(2.0), &bv64(4.0)).unwrap();
    assert!(approx_eq(extract_f64(r), 4.0, 1e-12));
}

#[test]
fn yl2xp1_concrete() {
    // 1 * log2(7+1) = 3
    let r = try_concrete_triop_rm(IOP_YL2XP1_F64, &rm(), &bv64(1.0), &bv64(7.0)).unwrap();
    assert!(approx_eq(extract_f64(r), 3.0, 1e-12));
}

#[test]
fn scale_concrete() {
    // 1.5 * 2^trunc(2.7) = 1.5 * 4 = 6
    let r = try_concrete_triop_rm(IOP_SCALE_F64, &rm(), &bv64(1.5), &bv64(2.7)).unwrap();
    assert!(approx_eq(extract_f64(r), 6.0, 1e-12));
    // 1.0 * 2^trunc(-3.5) = 1.0 * 0.125 = 0.125
    let r = try_concrete_triop_rm(IOP_SCALE_F64, &rm(), &bv64(1.0), &bv64(-3.5)).unwrap();
    assert_eq!(extract_f64(r), 0.125);
}

#[test]
fn atan_concrete() {
    // atan2(1, 1) = pi/4
    let r = try_concrete_triop_rm(IOP_ATAN_F64, &rm(), &bv64(1.0), &bv64(1.0)).unwrap();
    assert!(approx_eq(
        extract_f64(r),
        std::f64::consts::FRAC_PI_4,
        1e-12
    ));
    // atan2(0, 1) = 0
    let r = try_concrete_triop_rm(IOP_ATAN_F64, &rm(), &bv64(0.0), &bv64(1.0)).unwrap();
    assert_eq!(extract_f64(r), 0.0);
}

#[test]
fn symbolic_returns_none() {
    let ctx = crate::symbolic::SymContext::new_mock();
    let sym = RustBV::symbolic(&ctx, "x", 64);
    assert!(try_concrete_binop_rm(IOP_SIN_F64, &rm(), &sym).is_none());
    assert!(try_concrete_triop_rm(IOP_YL2X_F64, &rm(), &bv64(1.0), &sym).is_none());
}

#[test]
fn unknown_opcode_returns_none() {
    assert!(try_concrete_binop_rm(0xdead, &rm(), &bv64(1.0)).is_none());
    assert!(try_concrete_triop_rm(0xdead, &rm(), &bv64(1.0), &bv64(1.0)).is_none());
}

#[test]
fn recip_exp_specials() {
    // 2.0 → exponent 1 → 2^-1 = 0.5
    let r = try_concrete_binop_rm(IOP_RECPEXP_F64, &rm(), &bv64(2.0)).unwrap();
    assert_eq!(extract_f64(r), 0.5);
    // 4.0 → exponent 2 → 2^-2 = 0.25
    let r = try_concrete_binop_rm(IOP_RECPEXP_F64, &rm(), &bv64(4.0)).unwrap();
    assert_eq!(extract_f64(r), 0.25);
    // 0 → +inf
    let r = try_concrete_binop_rm(IOP_RECPEXP_F64, &rm(), &bv64(0.0)).unwrap();
    assert_eq!(extract_f64(r), f64::INFINITY);
    // f32: 8.0 → exponent 3 → 2^-3 = 0.125
    let r = try_concrete_binop_rm(IOP_RECPEXP_F32, &rm(), &bv32(8.0)).unwrap();
    assert_eq!(extract_f32(r), 0.125);
}
