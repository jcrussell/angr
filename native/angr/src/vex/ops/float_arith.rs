//! Scalar floating-point arithmetic VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via a plain `mod` decl in
//! `ops/mod.rs`), so these `pub(super)` methods stay callable from the unop/binop
//! dispatch in `ops`, and the shared free fns they call that remain in `ops`
//! (`build_float_expr`, `float_prec_of`) stay visible by the
//! descendant-module rule.
//!
//! Only the *scalar* float arith ops live here. The `FloatLaneOp` trait, the
//! `FAdd`/`FSub`/… lane structs, the `impl_float_lane_*` macros, and the
//! `build_float_expr`/`float_prec_of` free fns deliberately stay in `ops`:
//! they are shared with the packed/vector float paths (`binop_vec_float`,
//! `vec_float_lane_op`) and the rounding-mode variants.

use super::{OpError, VEXOps, build_float_expr, float_prec_of};
use crate::symbolic::{FloatOpKind, RustBV, SymContext};
use crate::vex::ir::IRType;

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
