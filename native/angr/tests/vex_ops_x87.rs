//! Integration tests for the concrete libm fast path of the x87
//! transcendental ops (bd `angr-i5lj.1`, `angr-i5lj.2`), mirroring the
//! `vex::transcendentals` unit tests through the public API. The symbolic
//! concretize-and-pin tests live with the pins, in
//! `interpreter/transcendental_pins_tests.rs`: the interpreter is not
//! public (angr-mx00p.116.6).

// Integration-test harness code panicking on unwrap/expect is the desired
// behavior, same rationale as #[cfg(test)] code in lib.rs -- not
// production-code debt (angr-9ke6b.212).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rustylib::symbolic::RustBV;
use rustylib::vex::transcendentals::{
    IOP_2XM1_F64, IOP_YL2X_F64, try_concrete_binop_rm, try_concrete_triop_rm,
};

fn bv64(f: f64) -> RustBV {
    RustBV::concrete(f.to_bits() as u128, 64)
}
fn rm() -> RustBV {
    RustBV::concrete(0, 32)
}
fn extract_f64(bv: RustBV) -> f64 {
    f64::from_bits(bv.as_u128().unwrap() as u64)
}
fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps || ((a - b).abs() / b.abs().max(1e-300)) < eps
}

#[test]
fn two_xm1_concrete_via_libm() {
    // 2^0 - 1 = 0
    let r = try_concrete_binop_rm(IOP_2XM1_F64, &rm(), &bv64(0.0)).unwrap();
    assert_eq!(extract_f64(r), 0.0);
    // 2^0.5 - 1 ≈ 0.4142135...
    let r = try_concrete_binop_rm(IOP_2XM1_F64, &rm(), &bv64(0.5)).unwrap();
    assert!(approx_eq(extract_f64(r), 2f64.sqrt() - 1.0, 1e-12));
}

#[test]
fn yl2x_concrete_via_libm() {
    // 3 * log2(2) = 3
    let r = try_concrete_triop_rm(IOP_YL2X_F64, &rm(), &bv64(3.0), &bv64(2.0)).unwrap();
    assert!(approx_eq(extract_f64(r), 3.0, 1e-12));
}
