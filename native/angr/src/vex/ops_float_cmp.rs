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
use crate::symbolic::{FloatOpKind, RustBV, SymContext};
use crate::vex::ir::{FCmpKind, IRType};

impl VEXOps {
    pub(super) fn float_cmp_eq(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match ty {
                IRType::F32 => {
                    let lf = f32::from_bits(l as u32);
                    let rf = f32::from_bits(r as u32);
                    if lf == rf { 1u128 } else { 0u128 }
                }
                IRType::F64 => {
                    let lf = f64::from_bits(l as u64);
                    let rf = f64::from_bits(r as u64);
                    if lf == rf { 1u128 } else { 0u128 }
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, 1));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(
            FloatOpKind::CmpEq,
            prec,
            vec![left, right],
        ))
    }

    pub(super) fn float_cmp_lt(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match ty {
                IRType::F32 => {
                    let lf = f32::from_bits(l as u32);
                    let rf = f32::from_bits(r as u32);
                    if lf < rf { 1u128 } else { 0u128 }
                }
                IRType::F64 => {
                    let lf = f64::from_bits(l as u64);
                    let rf = f64::from_bits(r as u64);
                    if lf < rf { 1u128 } else { 0u128 }
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, 1));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(
            FloatOpKind::CmpLt,
            prec,
            vec![left, right],
        ))
    }

    pub(super) fn float_cmp_le(
        left: RustBV,
        right: RustBV,
        ty: IRType,
        _ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let result = match ty {
                IRType::F32 => {
                    let lf = f32::from_bits(l as u32);
                    let rf = f32::from_bits(r as u32);
                    if lf <= rf { 1u128 } else { 0u128 }
                }
                IRType::F64 => {
                    let lf = f64::from_bits(l as u64);
                    let rf = f64::from_bits(r as u64);
                    if lf <= rf { 1u128 } else { 0u128 }
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            return Ok(RustBV::concrete(result, 1));
        }
        let prec = float_prec_of(ty).ok_or(OpError::InvalidFloatType(ty))?;
        Ok(build_float_expr(
            FloatOpKind::CmpLe,
            prec,
            vec![left, right],
        ))
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

        let lane_mask: u128 = if lane_bits == 32 {
            0xFFFF_FFFF
        } else {
            0xFFFF_FFFF_FFFF_FFFF
        };

        // Concrete fast path
        if let (Some(l), Some(r)) = (l_lo.as_u128(), r_lo.as_u128()) {
            let truth = match (ty, kind) {
                (IRType::F32, FCmpKind::Eq) => f32::from_bits(l as u32) == f32::from_bits(r as u32),
                (IRType::F32, FCmpKind::Lt) => f32::from_bits(l as u32) < f32::from_bits(r as u32),
                (IRType::F32, FCmpKind::Le) => f32::from_bits(l as u32) <= f32::from_bits(r as u32),
                (IRType::F32, FCmpKind::Gt) => f32::from_bits(l as u32) > f32::from_bits(r as u32),
                (IRType::F32, FCmpKind::Ge) => f32::from_bits(l as u32) >= f32::from_bits(r as u32),
                (IRType::F32, FCmpKind::Un) => {
                    f32::from_bits(l as u32).is_nan() || f32::from_bits(r as u32).is_nan()
                }
                (IRType::F64, FCmpKind::Eq) => f64::from_bits(l as u64) == f64::from_bits(r as u64),
                (IRType::F64, FCmpKind::Lt) => f64::from_bits(l as u64) < f64::from_bits(r as u64),
                (IRType::F64, FCmpKind::Le) => f64::from_bits(l as u64) <= f64::from_bits(r as u64),
                (IRType::F64, FCmpKind::Gt) => f64::from_bits(l as u64) > f64::from_bits(r as u64),
                (IRType::F64, FCmpKind::Ge) => f64::from_bits(l as u64) >= f64::from_bits(r as u64),
                (IRType::F64, FCmpKind::Un) => {
                    f64::from_bits(l as u64).is_nan() || f64::from_bits(r as u64).is_nan()
                }
                _ => return Err(OpError::InvalidFloatType(ty)),
            };
            let lane_val = if truth { lane_mask } else { 0 };
            let lane = RustBV::concrete(lane_val, lane_bits);
            return Ok(upper.concat_into(lane, ctx));
        }

        // Symbolic path: build 1-bit compare, then sign-extend to lane_bits
        // (sign-extend turns 1 → all-1s, 0 → all-0s).
        let cmp_1bit = match kind {
            FCmpKind::Eq => build_float_expr(FloatOpKind::CmpEq, prec, vec![l_lo, r_lo]),
            FCmpKind::Lt => build_float_expr(FloatOpKind::CmpLt, prec, vec![l_lo, r_lo]),
            FCmpKind::Le => build_float_expr(FloatOpKind::CmpLe, prec, vec![l_lo, r_lo]),
            // Gt(a,b) = Lt(b,a); Ge(a,b) = Le(b,a). Not emitted by SSE
            // scalar-lane opcodes today, but exhaustive for the shared FCmpKind enum.
            FCmpKind::Gt => build_float_expr(FloatOpKind::CmpLt, prec, vec![r_lo, l_lo]),
            FCmpKind::Ge => build_float_expr(FloatOpKind::CmpLe, prec, vec![r_lo, l_lo]),
            FCmpKind::Un => {
                // un = isNaN(l) OR isNaN(r). IsNaN is a unary primitive so no
                // operand clone is needed (vs. CmpEq(x, x) which doubles x).
                let l_nan = build_float_expr(FloatOpKind::IsNaN, prec, vec![l_lo]);
                let r_nan = build_float_expr(FloatOpKind::IsNaN, prec, vec![r_lo]);
                l_nan.or_into(r_nan, ctx)
            }
        };
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

        let lane_mask: u128 = if elem_width == 32 {
            0xFFFF_FFFF
        } else if elem_width == 64 {
            0xFFFF_FFFF_FFFF_FFFF
        } else {
            return Err(OpError::InvalidFloatType(elem));
        };

        // Concrete fast path: extract each lane, compare, repack. The concrete
        // truth-table for Gt/Ge follows the IEEE 754 ordered semantics — Rust's
        // `>` and `>=` on f32/f64 already return false when either operand is
        // NaN, matching VEX. The Un case independently checks NaN.
        if total_width <= 128
            && let (Some(l), Some(r)) = (left.as_u128(), right.as_u128())
        {
            let mut result: u128 = 0;
            let elem_mask: u128 = (1u128 << elem_width) - 1;
            for i in 0..count {
                let shift = (i as u32) * elem_width;
                let l_bits = (l >> shift) & elem_mask;
                let r_bits = (r >> shift) & elem_mask;
                let truth = match (elem, kind) {
                    (IRType::F32, FCmpKind::Eq) => {
                        f32::from_bits(l_bits as u32) == f32::from_bits(r_bits as u32)
                    }
                    (IRType::F32, FCmpKind::Lt) => {
                        f32::from_bits(l_bits as u32) < f32::from_bits(r_bits as u32)
                    }
                    (IRType::F32, FCmpKind::Le) => {
                        f32::from_bits(l_bits as u32) <= f32::from_bits(r_bits as u32)
                    }
                    (IRType::F32, FCmpKind::Gt) => {
                        f32::from_bits(l_bits as u32) > f32::from_bits(r_bits as u32)
                    }
                    (IRType::F32, FCmpKind::Ge) => {
                        f32::from_bits(l_bits as u32) >= f32::from_bits(r_bits as u32)
                    }
                    (IRType::F32, FCmpKind::Un) => {
                        f32::from_bits(l_bits as u32).is_nan()
                            || f32::from_bits(r_bits as u32).is_nan()
                    }
                    (IRType::F64, FCmpKind::Eq) => {
                        f64::from_bits(l_bits as u64) == f64::from_bits(r_bits as u64)
                    }
                    (IRType::F64, FCmpKind::Lt) => {
                        f64::from_bits(l_bits as u64) < f64::from_bits(r_bits as u64)
                    }
                    (IRType::F64, FCmpKind::Le) => {
                        f64::from_bits(l_bits as u64) <= f64::from_bits(r_bits as u64)
                    }
                    (IRType::F64, FCmpKind::Gt) => {
                        f64::from_bits(l_bits as u64) > f64::from_bits(r_bits as u64)
                    }
                    (IRType::F64, FCmpKind::Ge) => {
                        f64::from_bits(l_bits as u64) >= f64::from_bits(r_bits as u64)
                    }
                    (IRType::F64, FCmpKind::Un) => {
                        f64::from_bits(l_bits as u64).is_nan()
                            || f64::from_bits(r_bits as u64).is_nan()
                    }
                    _ => return Err(OpError::InvalidFloatType(elem)),
                };
                let lane_val = if truth { lane_mask } else { 0 };
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
            let cmp_1bit = match kind {
                FCmpKind::Eq => build_float_expr(FloatOpKind::CmpEq, prec, vec![l_lane, r_lane]),
                FCmpKind::Lt => build_float_expr(FloatOpKind::CmpLt, prec, vec![l_lane, r_lane]),
                FCmpKind::Le => build_float_expr(FloatOpKind::CmpLe, prec, vec![l_lane, r_lane]),
                // Gt(a,b) ≡ Lt(b,a); Ge(a,b) ≡ Le(b,a). Z3 has CmpLt/CmpLe;
                // swapping operands is cheaper than introducing new variants.
                FCmpKind::Gt => build_float_expr(FloatOpKind::CmpLt, prec, vec![r_lane, l_lane]),
                FCmpKind::Ge => build_float_expr(FloatOpKind::CmpLe, prec, vec![r_lane, l_lane]),
                FCmpKind::Un => {
                    // un = isNaN(l) OR isNaN(r); IsNaN is a unary primitive.
                    let l_nan = build_float_expr(FloatOpKind::IsNaN, prec, vec![l_lane]);
                    let r_nan = build_float_expr(FloatOpKind::IsNaN, prec, vec![r_lane]);
                    l_nan.or_into(r_nan, ctx)
                }
            };
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
