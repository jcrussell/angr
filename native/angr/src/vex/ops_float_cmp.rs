//! Floating-point comparison VEX op helpers.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via `#[path]` in
//! ops.rs), so these `pub(super)` methods stay callable from the binop
//! dispatch in `ops`, and the shared free fns / sibling methods they call
//! that remain in `ops` (`build_float_expr`, `float_prec_of`,
//! `Self::concat_le_elements`) stay visible by the descendant-module rule.
//!
//! Covers scalar FP compares (Iop_FCmp{EQ,LT,LE}, Iop_CmpF{32,64}), the
//! SSE scalar-lane compare (Iop_Cmp*32F0x4/64F0x2), and the packed FP
//! compare (Iop_Cmp*{32Fx2,32Fx4,64Fx2}). The `FCmpKind` enum and the
//! `build_float_expr`/`float_prec_of` free fns deliberately stay in `ops`:
//! they are shared with the scalar/packed FP arith paths.

use super::{OpError, VEXOps, build_float_expr, float_prec_of};
use crate::symbolic::{FloatOpKind, FloatPrec, RustBV, SymContext};
use crate::vex::ir::{FCmpKind, IRType};

impl VEXOps {
    /// Concrete IEEE-754 truth value for a single FP-compare lane, shared by the
    /// scalar-lane and packed concrete fast paths. `l`/`r` carry the raw lane
    /// bits (only the low `ty.bits()` are significant). All six `FCmpKind`s are
    /// valid here; Gt/Ge/Un map to Rust's `> >= is_nan`, whose NaN behavior
    /// matches VEX's ordered/unordered semantics.
    fn fcmp_truth(ty: IRType, kind: FCmpKind, l: u128, r: u128) -> Result<bool, OpError> {
        let truth = match ty {
            IRType::F32 => {
                let (lf, rf) = (f32::from_bits(l as u32), f32::from_bits(r as u32));
                match kind {
                    FCmpKind::Eq => lf == rf,
                    FCmpKind::Lt => lf < rf,
                    FCmpKind::Le => lf <= rf,
                    FCmpKind::Gt => lf > rf,
                    FCmpKind::Ge => lf >= rf,
                    FCmpKind::Un => lf.is_nan() || rf.is_nan(),
                }
            }
            IRType::F64 => {
                let (lf, rf) = (f64::from_bits(l as u64), f64::from_bits(r as u64));
                match kind {
                    FCmpKind::Eq => lf == rf,
                    FCmpKind::Lt => lf < rf,
                    FCmpKind::Le => lf <= rf,
                    FCmpKind::Gt => lf > rf,
                    FCmpKind::Ge => lf >= rf,
                    FCmpKind::Un => lf.is_nan() || rf.is_nan(),
                }
            }
            _ => return Err(OpError::InvalidFloatType(ty)),
        };
        Ok(truth)
    }

    /// Build the 1-bit symbolic FP-compare predicate for a single lane, shared
    /// by the scalar-lane and packed symbolic fallbacks. Gt(a,b) ≡ Lt(b,a) and
    /// Ge(a,b) ≡ Le(b,a) (swapped operands — Z3 has no native Gt/Ge); Un is
    /// isNaN(l) OR isNaN(r) (IsNaN is a unary primitive, so no operand clone).
    fn fcmp_predicate_1bit(
        kind: FCmpKind,
        prec: FloatPrec,
        l: RustBV,
        r: RustBV,
        ctx: &SymContext,
    ) -> RustBV {
        match kind {
            FCmpKind::Eq => build_float_expr(FloatOpKind::CmpEq, prec, vec![l, r]),
            FCmpKind::Lt => build_float_expr(FloatOpKind::CmpLt, prec, vec![l, r]),
            FCmpKind::Le => build_float_expr(FloatOpKind::CmpLe, prec, vec![l, r]),
            FCmpKind::Gt => build_float_expr(FloatOpKind::CmpLt, prec, vec![r, l]),
            FCmpKind::Ge => build_float_expr(FloatOpKind::CmpLe, prec, vec![r, l]),
            FCmpKind::Un => {
                let l_nan = build_float_expr(FloatOpKind::IsNaN, prec, vec![l]);
                let r_nan = build_float_expr(FloatOpKind::IsNaN, prec, vec![r]);
                l_nan.or_into(r_nan, ctx)
            }
        }
    }

    /// Scalar FP compare shared by Iop_FCmp{EQ,LT,LE}. The three only differ in
    /// the concrete predicate and the symbolic `FloatOpKind`; everything else
    /// (the F32/F64 `from_bits` split, 1-bit concrete result, and
    /// `build_float_expr` symbolic fallback) is identical.
    fn float_cmp_scalar(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        kind: FCmpKind,
    ) -> Result<RustBV, OpError> {
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let truth = match ty {
                IRType::F32 => {
                    let lf = f32::from_bits(l as u32);
                    let rf = f32::from_bits(r as u32);
                    match kind {
                        FCmpKind::Eq => lf == rf,
                        FCmpKind::Lt => lf < rf,
                        FCmpKind::Le => lf <= rf,
                        _ => return Err(OpError::InvalidFloatType(ty)),
                    }
                }
                IRType::F64 => {
                    let lf = f64::from_bits(l as u64);
                    let rf = f64::from_bits(r as u64);
                    match kind {
                        FCmpKind::Eq => lf == rf,
                        FCmpKind::Lt => lf < rf,
                        FCmpKind::Le => lf <= rf,
                        _ => return Err(OpError::InvalidFloatType(ty)),
                    }
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(truth as u128, 1));
        }
        let op = match kind {
            FCmpKind::Eq => FloatOpKind::CmpEq,
            FCmpKind::Lt => FloatOpKind::CmpLt,
            FCmpKind::Le => FloatOpKind::CmpLe,
            _ => return Err(OpError::InvalidFloatType(ty)),
        };
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(op, prec, vec![left, right]))
    }

    pub(super) fn float_cmp_eq(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_cmp_scalar(left, right, ty, FCmpKind::Eq)
    }

    pub(super) fn float_cmp_lt(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_cmp_scalar(left, right, ty, FCmpKind::Lt)
    }

    pub(super) fn float_cmp_le(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        Self::float_cmp_scalar(left, right, ty, FCmpKind::Le)
    }

    /// SSE scalar-lane FP compare (Iop_Cmp{EQ,LT,LE,UN}{32F0x4,64F0x2}).
    /// Operates on lane 0 only; result is V128 with lane 0 set to all-1s on
    /// true and 0 on false. Upper lanes pass through from `left`.
    pub(super) fn vec_float_scalar_lane_cmp(
        left: RustBV,
        right: RustBV,
        kind: FCmpKind,
        ty: IRType,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        let lane_bits = prec.bits();
        debug_assert_eq!(left.width(), 128);
        debug_assert_eq!(right.width(), 128);
        let l_lo = left.extract(lane_bits - 1, 0, ctx);
        let r_lo = right.extract(lane_bits - 1, 0, ctx);
        let upper = left.extract(127, lane_bits, ctx);

        let lane_mask = Self::low_bit_mask_u128(lane_bits);

        // Concrete fast path
        if let (Some(l), Some(r)) = (l_lo.as_u128(), r_lo.as_u128()) {
            let lane_val = if Self::fcmp_truth(ty, kind, l, r)? {
                lane_mask
            } else {
                0
            };
            let lane = RustBV::concrete(lane_val, lane_bits);
            return Ok(upper.concat_into(lane, ctx));
        }

        // Symbolic path: build 1-bit compare, then sign-extend to lane_bits
        // (sign-extend turns 1 → all-1s, 0 → all-0s).
        let cmp_1bit = Self::fcmp_predicate_1bit(kind, prec, l_lo, r_lo, ctx);
        let lane = cmp_1bit.sign_extend_into(lane_bits, ctx);
        Ok(upper.concat_into(lane, ctx))
    }

    /// Packed FP compare (Iop_Cmp{EQ,LT,LE,GT,GE,UN}{32Fx2,32Fx4,64Fx2}).
    /// Each lane independently produces all-1s (true) or 0 (false) of width
    /// `elem.bits()`. Total result width = elem.bits() * count.
    pub(super) fn vec_float_packed_cmp(
        left: RustBV,
        right: RustBV,
        kind: FCmpKind,
        elem: IRType,
        count: u8,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);

        if elem_width != 32 && elem_width != 64 {
            return Err(OpError::InvalidFloatType(elem));
        }
        let lane_mask = Self::low_bit_mask_u128(elem_width);

        // Concrete fast path: extract each lane, compare, repack. The concrete
        // truth-table for Gt/Ge follows the IEEE 754 ordered semantics — Rust's
        // `>` and `>=` on f32/f64 already return false when either operand is
        // NaN, matching VEX. The Un case independently checks NaN.
        if total_width <= 128
            && let (Some(l), Some(r)) = (left.as_u128(), right.as_u128())
        {
            let mut result: u128 = 0;
            let elem_mask: u128 = Self::low_bit_mask_u128(elem_width);
            for i in 0..count {
                let shift = (i as u32) * elem_width;
                let l_bits = (l >> shift) & elem_mask;
                let r_bits = (r >> shift) & elem_mask;
                let lane_val = if Self::fcmp_truth(elem, kind, l_bits, r_bits)? {
                    lane_mask
                } else {
                    0
                };
                result |= lane_val << shift;
            }
            return Ok(RustBV::concrete(result, total_width));
        }

        // Symbolic per-lane fallback: build a 1-bit predicate per lane,
        // sign-extend to elem width, concatenate (low-order lane first).
        let prec = float_prec_of(elem).ok_or(OpError::InvalidFloatType(elem))?;
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let l_lane = left.extract(hi, lo, ctx);
            let r_lane = right.extract(hi, lo, ctx);
            let cmp_1bit = Self::fcmp_predicate_1bit(kind, prec, l_lane, r_lane, ctx);
            elements.push(cmp_1bit.sign_extend_into(elem_width, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }

    /// x87 FCOM-style compare (Iop_CmpF32, Iop_CmpF64). Returns I32 with the
    /// VEX-defined encoding:
    ///   0x40 = EQ, 0x01 = LT, 0x00 = GT, 0x45 = UN (either operand is NaN).
    pub(super) fn float_com_cc(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        // Concrete fast path
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result: u128 = match ty {
                IRType::F32 => {
                    let lf = f32::from_bits(l as u32);
                    let rf = f32::from_bits(r as u32);
                    if lf.is_nan() || rf.is_nan() {
                        0x45
                    } else if lf < rf {
                        0x01
                    } else if lf == rf {
                        0x40
                    } else {
                        0x00
                    }
                }
                IRType::F64 => {
                    let lf = f64::from_bits(l as u64);
                    let rf = f64::from_bits(r as u64);
                    if lf.is_nan() || rf.is_nan() {
                        0x45
                    } else if lf < rf {
                        0x01
                    } else if lf == rf {
                        0x40
                    } else {
                        0x00
                    }
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, 32));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;

        // Symbolic: compose un/lt/eq predicates, then nest ITEs.
        // un  = isNaN(l) OR isNaN(r)   [unary primitive, no operand clone]
        // lt  = l < r                  [false if either is NaN]
        // eq  = l == r                 [false if either is NaN]
        let l_nan = build_float_expr(FloatOpKind::IsNaN, prec, vec![left.clone()]);
        let r_nan = build_float_expr(FloatOpKind::IsNaN, prec, vec![right.clone()]);
        let un = l_nan.or_into(r_nan, ctx);
        let lt = build_float_expr(FloatOpKind::CmpLt, prec, vec![left.clone(), right.clone()]);
        let eq = build_float_expr(FloatOpKind::CmpEq, prec, vec![left, right]);

        let v_un = RustBV::concrete(0x45, 32);
        let v_lt = RustBV::concrete(0x01, 32);
        let v_eq = RustBV::concrete(0x40, 32);
        let v_gt = RustBV::concrete(0x00, 32);
        // un ? 0x45 : (lt ? 0x01 : (eq ? 0x40 : 0x00))
        let inner = eq.ite_into(v_eq, v_gt, ctx);
        let middle = lt.ite_into(v_lt, inner, ctx);
        Ok(un.ite_into(v_un, middle, ctx))
    }
}
