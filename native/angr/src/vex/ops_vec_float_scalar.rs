//! SSE scalar-in-vector floating-point VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via `#[path]` in
//! ops.rs), so these `pub(super)` methods stay callable from the binop/unop
//! dispatch in `ops`, and the shared siblings they reference
//! (`Self::concat_le_elements`, the `build_float_expr`/`float_prec_of` free
//! fns, all of which stay in ops.rs) stay visible via the descendant rule.
//!
//! Covers the SSE "scalar" forms that operate on lane 0 only and pass the
//! upper lanes through from the left operand (ADDSS/SUBSS/MULSS/DIVSS and
//! F64 variants, SQRTSS/SQRTSD, MAXSS/MINSS, RCPSS/RSQRTSS estimates) plus
//! the packed fresh-symbolic-per-lane fallbacks used by the
//! reciprocal/rsqrt estimate-and-step ops.

use super::{OpError, VEXOps, build_float_expr, float_prec_of};
use crate::symbolic::{FloatOpKind, RustBV, SymContext};
use crate::vex::ir::IRType;

impl VEXOps {
    /// Scalar float operation in vector (SSE scalar ops like ADDSS, DIVSS).
    /// Operates on element 0 only, passes through other elements from left operand.
    /// Only Add/Sub/Mul/Div are accepted; other kinds are a programmer error.
    #[inline]
    pub(super) fn vec_float_scalar_op(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        kind: FloatOpKind,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(left.width(), 128);
        debug_assert_eq!(right.width(), 128);
        debug_assert!(
            matches!(
                kind,
                FloatOpKind::Add | FloatOpKind::Sub | FloatOpKind::Mul | FloatOpKind::Div
            ),
            "vec_float_scalar_op only supports Add/Sub/Mul/Div, got {:?}",
            kind,
        );

        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match elem {
                IRType::F32 => {
                    let l0 = f32::from_bits(l as u32);
                    let r0 = f32::from_bits(r as u32);
                    let res0 = match kind {
                        FloatOpKind::Add => l0 + r0,
                        FloatOpKind::Sub => l0 - r0,
                        FloatOpKind::Mul => l0 * r0,
                        FloatOpKind::Div => l0 / r0,
                        // Hardened from `unreachable!()` (angr-j60q0.2): `kind`
                        // is contractually Add/Sub/Mul/Div, but return a typed
                        // error in release should a future caller forward an
                        // attacker-derived FloatOpKind.
                        _ => {
                            return Err(OpError::UnsupportedVectorOp(format!(
                                "vec_float_scalar_op expects Add/Sub/Mul/Div, got {kind:?}"
                            )));
                        }
                    };
                    Self::splice_lane0_u128(l, res0.to_bits() as u128, 32)
                }
                IRType::F64 => {
                    let l0 = f64::from_bits(l as u64);
                    let r0 = f64::from_bits(r as u64);
                    let res0 = match kind {
                        FloatOpKind::Add => l0 + r0,
                        FloatOpKind::Sub => l0 - r0,
                        FloatOpKind::Mul => l0 * r0,
                        FloatOpKind::Div => l0 / r0,
                        // Hardened from `unreachable!()` (angr-j60q0.2): `kind`
                        // is contractually Add/Sub/Mul/Div, but return a typed
                        // error in release should a future caller forward an
                        // attacker-derived FloatOpKind.
                        _ => {
                            return Err(OpError::UnsupportedVectorOp(format!(
                                "vec_float_scalar_op expects Add/Sub/Mul/Div, got {kind:?}"
                            )));
                        }
                    };
                    Self::splice_lane0_u128(l, res0.to_bits() as u128, 64)
                }
                _ => return Err(OpError::InvalidFloatType(elem)),
            };
            return Ok(RustBV::concrete(result, 128));
        }
        Self::vec_float_scalar_lane_binop(left, right, elem, kind, ctx)
    }

    /// Scalar sqrt in vector (SQRTSS/SQRTSD).
    pub(super) fn vec_float_scalar_sqrt(
        arg: RustBV,
        elem: IRType,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(arg.width(), 128);

        if let Some(v) = arg.as_u128() {
            let result = match elem {
                IRType::F32 => {
                    let val = f32::from_bits(v as u32);
                    let res = val.sqrt();
                    Self::splice_lane0_u128(v, res.to_bits() as u128, 32)
                }
                IRType::F64 => {
                    let val = f64::from_bits(v as u64);
                    let res = val.sqrt();
                    Self::splice_lane0_u128(v, res.to_bits() as u128, 64)
                }
                _ => return Err(OpError::InvalidFloatType(elem)),
            };
            return Ok(RustBV::concrete(result, 128));
        }
        let prec = float_prec_of(elem).ok_or(OpError::InvalidFloatType(elem))?;
        let lane_bits = prec.bits();
        let lo = arg.extract(lane_bits - 1, 0, ctx);
        let upper = arg.extract(127, lane_bits, ctx);
        let res_lane = build_float_expr(FloatOpKind::Sqrt, prec, vec![lo]);
        Ok(upper.concat_into(res_lane, ctx))
    }

    /// SSE scalar-in-vector reciprocal/rsqrt estimate (RCPSS / RSQRTSS).
    /// VEX leaves the lane-0 result implementation-defined, so we hand back a
    /// fresh symbolic of the lane width; upper 96 bits pass through from arg.
    /// Used for `Iop_RecipEst32F0x4` and `Iop_RSqrtEst32F0x4`. The arg is still
    /// consumed (the upper-lane passthrough preserves it) so dataflow is sane.
    pub(super) fn vec_float_scalar_fresh(
        arg: RustBV,
        elem: IRType,
        name: &str,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(arg.width(), 128);
        let prec = float_prec_of(elem).ok_or(OpError::InvalidFloatType(elem))?;
        let lane_bits = prec.bits();
        let upper = arg.extract(127, lane_bits, ctx);
        let lane = RustBV::symbolic(ctx, name, lane_bits);
        Ok(upper.concat_into(lane, ctx))
    }

    /// Packed FP fresh-symbolic per lane. Used for `VFRecipEst`/`VFRSqrtEst`
    /// (precision implementation-defined) and `VFRecipStep`/`VFRSqrtStep`
    /// (angr Python has no generic handler, so a fresh symbolic per lane is
    /// the conservative match — Newton-Raphson refinement converges anyway).
    pub(super) fn vec_float_fresh_per_lane(
        elem: IRType,
        count: u8,
        name: &str,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let lane_bits = match elem {
            IRType::F32 => 32,
            IRType::F64 => 64,
            _ => return Err(OpError::InvalidFloatType(elem)),
        };
        let mut lanes: Vec<RustBV> = Vec::with_capacity(count as usize);
        for _ in 0..count {
            lanes.push(RustBV::symbolic(ctx, name, lane_bits));
        }
        Ok(Self::concat_le_elements(lanes, ctx))
    }

    /// Packed integer fresh-symbolic per lane. Used for `VIRecipEst` /
    /// `VIRSqrtEst` (ARM URECPE / URSQRTE) where claripy has no generic
    /// handler — emitting a fresh symbolic per lane keeps the Rust engine
    /// from diverging from angr Python (which raises) while still letting
    /// the binary's Newton-Raphson refinement loop converge.
    pub(super) fn vec_int_fresh_per_lane(
        lane_bits: u32,
        count: u8,
        name: &str,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let mut lanes: Vec<RustBV> = Vec::with_capacity(count as usize);
        for _ in 0..count {
            lanes.push(RustBV::symbolic(ctx, name, lane_bits));
        }
        Ok(Self::concat_le_elements(lanes, ctx))
    }

    /// Scalar max/min in vector (MAXSS/MAXSD/MINSS/MINSD). `is_max` picks the
    /// `>`/`<` comparison; both encode SSE's "NaN returns right" semantics
    /// (the concrete branch via Rust's `>`/`<`, the symbolic fallback via
    /// `vec_float_scalar_lane_minmax`'s ITE).
    pub(super) fn vec_float_scalar_minmax(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        is_max: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(left.width(), 128);
        debug_assert_eq!(right.width(), 128);

        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match elem {
                IRType::F32 => {
                    let l0 = f32::from_bits(l as u32);
                    let r0 = f32::from_bits(r as u32);
                    let res0 = if is_max {
                        if l0 > r0 { l0 } else { r0 }
                    } else if l0 < r0 {
                        l0
                    } else {
                        r0
                    };
                    Self::splice_lane0_u128(l, res0.to_bits() as u128, 32)
                }
                IRType::F64 => {
                    let l0 = f64::from_bits(l as u64);
                    let r0 = f64::from_bits(r as u64);
                    let res0 = if is_max {
                        if l0 > r0 { l0 } else { r0 }
                    } else if l0 < r0 {
                        l0
                    } else {
                        r0
                    };
                    Self::splice_lane0_u128(l, res0.to_bits() as u128, 64)
                }
                _ => return Err(OpError::InvalidFloatType(elem)),
            };
            return Ok(RustBV::concrete(result, 128));
        }
        Self::vec_float_scalar_lane_minmax(left, right, elem, is_max, ctx)
    }

    /// Symbolic fallback for SSE scalar binary float ops (Add/Sub/Mul/Div).
    /// Extracts lane 0, runs the op via Z3 FP, concats back with upper bits.
    pub(super) fn vec_float_scalar_lane_binop(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        kind: FloatOpKind,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let prec = float_prec_of(elem).ok_or(OpError::InvalidFloatType(elem))?;
        let lane_bits = prec.bits();
        let l_lo = left.extract(lane_bits - 1, 0, ctx);
        let r_lo = right.extract(lane_bits - 1, 0, ctx);
        let upper = left.extract(127, lane_bits, ctx);
        let res_lane = build_float_expr(kind, prec, vec![l_lo, r_lo]);
        Ok(upper.concat_into(res_lane, ctx))
    }

    /// Symbolic fallback for SSE scalar MAXSS/MINSS/MAXSD/MINSD on lane 0.
    /// Encodes Rust's `>`/`<` semantics (NaN-returns-right) as
    /// `ITE(cond, l, r)` with `cond = FCmpLt(r, l)` for max or
    /// `cond = FCmpLt(l, r)` for min — matches the concrete branch above.
    pub(super) fn vec_float_scalar_lane_minmax(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        is_max: bool,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let prec = float_prec_of(elem).ok_or(OpError::InvalidFloatType(elem))?;
        let lane_bits = prec.bits();
        let l_lo = left.extract(lane_bits - 1, 0, ctx);
        let r_lo = right.extract(lane_bits - 1, 0, ctx);
        let upper = left.extract(127, lane_bits, ctx);
        // max(l, r) = if l > r then l else r ⟺ ITE(r < l, l, r)
        // min(l, r) = if l < r then l else r ⟺ ITE(l < r, l, r)
        let (cmp_left, cmp_right) = if is_max {
            (r_lo.clone(), l_lo.clone())
        } else {
            (l_lo.clone(), r_lo.clone())
        };
        let cond = build_float_expr(FloatOpKind::CmpLt, prec, vec![cmp_left, cmp_right]);
        let res_lane = cond.ite_into(l_lo, r_lo, ctx);
        Ok(upper.concat_into(res_lane, ctx))
    }
}
