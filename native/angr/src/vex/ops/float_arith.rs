//! Scalar floating-point arithmetic VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via a plain `mod` decl in
//! `ops/mod.rs`), so these `pub(super)` methods stay callable from the unop/binop
//! dispatch in `ops`, and the shared free fns they call (`build_float_expr`,
//! `float_prec_of`, which live in `ops/lane_traits.rs` and are re-exported by
//! `ops`) stay visible by the descendant-module rule.
//!
//! Only the *scalar* float arith ops live here. The `FloatLaneOp` trait, the
//! `FAdd`/`FSub`/… lane structs, the `impl_float_lane_*` macros, and the
//! `build_float_expr`/`float_prec_of` free fns deliberately live in the
//! sibling `ops/lane_traits.rs`: they are shared with the packed/vector float
//! paths (`binop_vec_float`,
//! `vec_float_lane_op`) and the rounding-mode variants.

use super::{OpError, VEXOps, build_float_expr, float_prec_of};
use crate::symbolic::{FloatOpKind, RustBV, SymContext};
use crate::vex::ir::IRType;
use std::ops::Sub;

impl VEXOps {
    pub(super) fn float_neg(arg: RustBV, ty: IRType, ctx: &SymContext) -> Result<RustBV, OpError> {
        // Flip the sign bit
        let sign_bit = match ty {
            IRType::F32 => 31,
            IRType::F64 => 63,
            _ => return Err(OpError::InvalidFloatType(ty)),
        };

        let mask = RustBV::concrete(1u128 << sign_bit, arg.width());
        Ok(arg.xor_into(mask, ctx))
    }

    pub(super) fn float_abs(arg: RustBV, ty: IRType, ctx: &SymContext) -> Result<RustBV, OpError> {
        // Clear the sign bit
        let mask = match ty {
            IRType::F32 => RustBV::concrete(0x7FFFFFFF, 32),
            IRType::F64 => RustBV::concrete(0x7FFFFFFFFFFFFFFF, 64),
            _ => return Err(OpError::InvalidFloatType(ty)),
        };

        Ok(arg.and_into(mask, ctx))
    }

    pub(super) fn float_sqrt(
        arg: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        // For concrete values, compute directly
        if let Some(v) = arg.as_u128() {
            let result = match ty {
                IRType::F32 => {
                    let f = f32::from_bits(v as u32);
                    f.sqrt().to_bits() as u128
                }
                IRType::F64 => {
                    let f = f64::from_bits(v as u64);
                    f.sqrt().to_bits() as u128
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, ty.bits()));
        }

        // Symbolic: build a Z3 FP expression so constraints propagate.
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(FloatOpKind::Sqrt, prec, vec![arg]))
    }

    /// Shared scalar binary float-arith helper. The four public ops
    /// (`float_add`/`sub`/`mul`/`div`) differ only in the concrete operator
    /// (`op32`/`op64`) and the symbolic [`FloatOpKind`]; everything else — the
    /// F32/F64 bit-cast, the `_` -> `InvalidFloatType` guard, and the symbolic
    /// `build_float_expr` fallback — is identical, so it lives here once.
    fn float_arith_scalar(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        kind: FloatOpKind,
        op32: fn(f32, f32) -> f32,
        op64: fn(f64, f64) -> f64,
    ) -> Result<RustBV, OpError> {
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match ty {
                IRType::F32 => {
                    op32(f32::from_bits(l as u32), f32::from_bits(r as u32)).to_bits() as u128
                }
                IRType::F64 => {
                    op64(f64::from_bits(l as u64), f64::from_bits(r as u64)).to_bits() as u128
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, ty.bits()));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(kind, prec, vec![left, right]))
    }

    pub(super) fn float_add(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_arith_scalar(
            left,
            right,
            ty,
            FloatOpKind::Add,
            |a, b| a + b,
            |a, b| a + b,
        )
    }

    pub(super) fn float_sub(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_arith_scalar(
            left,
            right,
            ty,
            FloatOpKind::Sub,
            |a, b| a - b,
            |a, b| a - b,
        )
    }

    pub(super) fn float_mul(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_arith_scalar(
            left,
            right,
            ty,
            FloatOpKind::Mul,
            |a, b| a * b,
            |a, b| a * b,
        )
    }

    pub(super) fn float_div(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_arith_scalar(
            left,
            right,
            ty,
            FloatOpKind::Div,
            |a, b| a / b,
            |a, b| a / b,
        )
    }

    /// `Iop_MaxNumF32`/`Iop_MaxNumF64` — AArch32 VMAXNM.
    pub(super) fn float_max_num(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_arith_scalar(
            left,
            right,
            ty,
            FloatOpKind::MaxNum,
            |a, b| num_min_max(a, b, /*want_max=*/ true),
            |a, b| num_min_max(a, b, /*want_max=*/ true),
        )
    }

    /// `Iop_MinNumF32`/`Iop_MinNumF64` — AArch32 VMINNM.
    pub(super) fn float_min_num(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_arith_scalar(
            left,
            right,
            ty,
            FloatOpKind::MinNum,
            |a, b| num_min_max(a, b, /*want_max=*/ false),
            |a, b| num_min_max(a, b, /*want_max=*/ false),
        )
    }

    /// Fused multiply-add: a*b + c
    pub(super) fn float_madd(
        a: RustBV,
        b: RustBV,
        c: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(av), Some(bv), Some(cv)) = (a.as_u128(), b.as_u128(), c.as_u128()) {
            let result = match ty {
                IRType::F32 => {
                    let af = f32::from_bits(av as u32);
                    let bf = f32::from_bits(bv as u32);
                    let cf = f32::from_bits(cv as u32);
                    af.mul_add(bf, cf).to_bits() as u128
                }
                IRType::F64 => {
                    let af = f64::from_bits(av as u64);
                    let bf = f64::from_bits(bv as u64);
                    let cf = f64::from_bits(cv as u64);
                    af.mul_add(bf, cf).to_bits() as u128
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, ty.bits()));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(FloatOpKind::Fma, prec, vec![a, b, c]))
    }

    /// Fused multiply-sub: a*b - c
    pub(super) fn float_msub(
        a: RustBV,
        b: RustBV,
        c: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(av), Some(bv), Some(cv)) = (a.as_u128(), b.as_u128(), c.as_u128()) {
            let result = match ty {
                IRType::F32 => {
                    let af = f32::from_bits(av as u32);
                    let bf = f32::from_bits(bv as u32);
                    let cf = f32::from_bits(cv as u32);
                    af.mul_add(bf, -cf).to_bits() as u128
                }
                IRType::F64 => {
                    let af = f64::from_bits(av as u64);
                    let bf = f64::from_bits(bv as u64);
                    let cf = f64::from_bits(cv as u64);
                    af.mul_add(bf, -cf).to_bits() as u128
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, ty.bits()));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(FloatOpKind::Fms, prec, vec![a, b, c]))
    }
}

/// Minimal float facade, so each body written against it -- [`num_min_max`]
/// for `Iop_MaxNumF*`/`Iop_MinNumF*`, and [`round_ties_to_even`] /
/// [`apply_rounding`] for the `ops/conversions.rs` float-to-int paths -- is
/// written once instead of once per precision.
pub(super) trait IeeeNum: Copy + PartialOrd + Sub<Output = Self> {
    /// `0.5` in this precision: the tie point [`round_ties_to_even`] tests for.
    const HALF: Self;
    fn is_nan(self) -> bool;
    fn is_sign_negative(self) -> bool;
    fn abs(self) -> Self;
    /// Nearest integer, ties away from zero (Rust's `round`, *not* RNE).
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    /// Whether `self.trunc()` is an even integer, decided in the same-width
    /// signed integer type (`i32` for `f32`, `i64` for `f64`). The width is
    /// observable rather than incidental: Rust's float-to-int `as` cast
    /// saturates, so an out-of-range input lands on `iN::MAX`/`MIN`, whose
    /// parity differs per width.
    fn trunc_is_even(self) -> bool;
}

macro_rules! impl_ieee_num {
    ($t:ty, $int:ty) => {
        impl IeeeNum for $t {
            const HALF: Self = 0.5;
            fn is_nan(self) -> bool {
                <$t>::is_nan(self)
            }
            fn is_sign_negative(self) -> bool {
                <$t>::is_sign_negative(self)
            }
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            fn round(self) -> Self {
                <$t>::round(self)
            }
            fn trunc(self) -> Self {
                <$t>::trunc(self)
            }
            fn floor(self) -> Self {
                <$t>::floor(self)
            }
            fn ceil(self) -> Self {
                <$t>::ceil(self)
            }
            fn trunc_is_even(self) -> bool {
                (<$t>::trunc(self) as $int) % 2 == 0
            }
        }
    };
}
impl_ieee_num!(f32, i32);
impl_ieee_num!(f64, i64);

/// Round to the nearest integer, ties to even (banker's rounding) -- the
/// implicit RNE mode of the rounding-mode-free float-to-int conversions in
/// `ops/conversions.rs`.
pub(super) fn round_ties_to_even<T: IeeeNum>(f: T) -> T {
    let truncated = f.trunc();
    // An exact .5 fractional part is the only input `round` (ties *away from
    // zero*) gets differently from RNE; there, keep the even neighbour.
    if (f - truncated).abs() == T::HALF && f.trunc_is_even() {
        truncated
    } else {
        f.round()
    }
}

/// Apply a VEX rounding mode to a concrete float: 0=nearest (ties to even),
/// 1=down (-inf), 2=up (+inf), 3=zero (truncate).
///
/// Callers pass the raw mode word; only its low two bits select the mode, so
/// the `_` arm below is mode 0 and nothing else.
pub(super) fn apply_rounding<T: IeeeNum>(f: T, rm: u32) -> T {
    match rm & 0x3 {
        1 => f.floor(),
        2 => f.ceil(),
        3 => f.trunc(),
        _ => round_ties_to_even(f),
    }
}

/// Concrete IEEE-754-2008 `maxNum` (`want_max`) / `minNum` on one scalar pair.
///
/// Deliberately *not* `f32::max`/`f64::max`: those map to LLVM `maxnum`, which
/// documents ±0 as "may return either operand". The ARM ARM's FPMax/FPMin --
/// which VMAXNM/VMINNM defer to once NaNs are out of the way -- pin
/// `maxNum(+0,-0) = +0` and `minNum(+0,-0) = -0`, and this op's whole reason
/// to exist is the corner cases the plain compare-and-select `FMax`/`FMin`
/// lane ops in `vex::ops::lane_traits` get differently.
fn num_min_max<T: IeeeNum>(a: T, b: T, want_max: bool) -> T {
    // Exactly-one-NaN: return the other operand (this is what makes it
    // *maxNum*, not max). Both-NaN falls out as `b`, itself a NaN.
    if a.is_nan() {
        return b;
    }
    if b.is_nan() {
        return a;
    }
    if a == b {
        // Reached for equal magnitudes and, the case that matters, for
        // `-0.0 == 0.0` -- which the ordered compare below cannot separate.
        return if a.is_sign_negative() == want_max {
            b
        } else {
            a
        };
    }
    if (a > b) == want_max { a } else { b }
}
