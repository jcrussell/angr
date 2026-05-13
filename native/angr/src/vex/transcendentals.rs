//! Concrete-only fast paths for x87 FPU transcendentals (Iop_SinF64,
//! Iop_CosF64, Iop_TanF64, Iop_2xm1F64, Iop_AtanF64, Iop_Yl2xF64,
//! Iop_Yl2xp1F64, Iop_ScaleF64) and ARM AArch64 FRECPX (Iop_RecpExpF64,
//! Iop_RecpExpF32). VEX exposes no IROp variants for these, so the FFI
//! lifter passes them through as `IROp::Raw(opcode)`.
//!
//! When all value operands are concrete f32/f64, this module computes the
//! result via Rust's libm bindings. Symbolic operands return `None` so
//! the caller falls back to `unsup_*` fresh-symbolic — Z3 has no native
//! sin/cos/log/exp theory.
//!
//! Opcode values: libvex_ir.h Iop_* enum, base 0x1400. Validated against
//! pyvex.const.enums_to_ints (2026-05-07).

use crate::symbolic::RustBV;

pub const IOP_ATAN_F64: u32 = 0x14de;
pub const IOP_YL2X_F64: u32 = 0x14df;
pub const IOP_YL2XP1_F64: u32 = 0x14e0;
pub const IOP_SCALE_F64: u32 = 0x14e5;
pub const IOP_SIN_F64: u32 = 0x14e6;
pub const IOP_COS_F64: u32 = 0x14e7;
pub const IOP_TAN_F64: u32 = 0x14e8;
pub const IOP_2XM1_F64: u32 = 0x14e9;
pub const IOP_RECPEXP_F64: u32 = 0x14fa;
pub const IOP_RECPEXP_F32: u32 = 0x14fb;

/// Concrete Binop transcendental: (rm, x) → result.
/// Returns None if any value operand is symbolic or the opcode is unknown.
/// `rm` (rounding mode) is consumed but ignored — Rust libm always uses
/// round-to-nearest-even, which matches the typical x87 rm=0 case.
pub fn try_concrete_binop_rm(opcode: u32, _rm: &RustBV, x: &RustBV) -> Option<RustBV> {
    if opcode == IOP_RECPEXP_F32 {
        let xv = x.as_u128()? as u32;
        let r = recip_exp_f32(f32::from_bits(xv));
        return Some(RustBV::concrete(r.to_bits() as u128, 32));
    }

    let xv = x.as_u128()? as u64;
    let xf = f64::from_bits(xv);
    let r = match opcode {
        IOP_SIN_F64 => xf.sin(),
        IOP_COS_F64 => xf.cos(),
        IOP_TAN_F64 => xf.tan(),
        IOP_2XM1_F64 => xf.exp2() - 1.0,
        IOP_RECPEXP_F64 => recip_exp_f64(xf),
        _ => return None,
    };
    Some(RustBV::concrete(r.to_bits() as u128, 64))
}

/// Concrete Triop transcendental: (rm, a, b) → result.
/// Returns None if any value operand is symbolic or the opcode is unknown.
pub fn try_concrete_triop_rm(opcode: u32, _rm: &RustBV, a: &RustBV, b: &RustBV) -> Option<RustBV> {
    let av = a.as_u128()? as u64;
    let bv = b.as_u128()? as u64;
    let af = f64::from_bits(av);
    let bf = f64::from_bits(bv);
    let r = match opcode {
        // Iop_Yl2xF64(rm, y, x) = y * log2(x)
        IOP_YL2X_F64 => af * bf.log2(),
        // Iop_Yl2xp1F64(rm, y, x) = y * log2(x + 1)
        IOP_YL2XP1_F64 => af * (bf + 1.0).log2(),
        // Iop_ScaleF64(rm, x, y) = x * 2^trunc(y) (x87 fscale semantics)
        IOP_SCALE_F64 => af * bf.trunc().exp2(),
        // Iop_AtanF64(rm, y, x) = atan2(y, x) (x87 fpatan semantics)
        IOP_ATAN_F64 => af.atan2(bf),
        _ => return None,
    };
    Some(RustBV::concrete(r.to_bits() as u128, 64))
}

/// IEEE 754 reciprocal exponent (ARM AArch64 FRECPX semantics).
/// Returns 2^(-Exp(x)) preserving sign; special-cases NaN/0/Inf per the
/// architecture reference.
fn recip_exp_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY.copysign(x);
    }
    if x.is_infinite() {
        return 0.0_f64.copysign(x);
    }
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let unbiased = if biased == 0 {
        // Subnormal — treat as smallest normal exponent for the estimate.
        -1022
    } else {
        biased - 1023
    };
    let new_biased = ((1023 - unbiased).clamp(0, 0x7fe)) as u64;
    let sign = bits & 0x8000_0000_0000_0000;
    f64::from_bits(sign | (new_biased << 52))
}

fn recip_exp_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x == 0.0 {
        return f32::INFINITY.copysign(x);
    }
    if x.is_infinite() {
        return 0.0_f32.copysign(x);
    }
    let bits = x.to_bits();
    let biased = ((bits >> 23) & 0xff) as i32;
    let unbiased = if biased == 0 { -126 } else { biased - 127 };
    let new_biased = ((127 - unbiased).clamp(0, 0xfe)) as u32;
    let sign = bits & 0x8000_0000;
    f32::from_bits(sign | (new_biased << 23))
}

#[cfg(test)]
mod tests {
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
        let r =
            try_concrete_binop_rm(IOP_SIN_F64, &rm(), &bv64(std::f64::consts::PI / 6.0)).unwrap();
        assert!(approx_eq(extract_f64(r), 0.5, 1e-12));

        let r = try_concrete_binop_rm(IOP_COS_F64, &rm(), &bv64(0.0)).unwrap();
        assert_eq!(extract_f64(r), 1.0);

        let r =
            try_concrete_binop_rm(IOP_TAN_F64, &rm(), &bv64(std::f64::consts::PI / 4.0)).unwrap();
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
}
