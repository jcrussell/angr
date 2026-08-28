//! Generic per-lane packed floating-point VEX op dispatcher.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via a plain `mod` decl in
//! `ops/mod.rs`), so this `pub(super)` method stays callable from the binop/unop
//! dispatch in `ops`, and the shared siblings it references
//! (`Self::concat_le_elements`, which stays in `ops/mod.rs`; the
//! `float_prec_of` free fn, the `FloatLaneOp` trait and the
//! `FLOAT_LANE_OP_MAX_ARITY` const, which live in `ops/lane_traits.rs` and are
//! re-exported by `ops`) stay visible via the descendant rule.

use super::{FLOAT_LANE_OP_MAX_ARITY, FloatLaneOp, OpError, VEXOps, float_prec_of};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ir::IRType;

impl VEXOps {
    /// Generic per-lane FP dispatcher. Handles the lane loop for both the
    /// concrete (extract via shift+mask, run f32/f64 op, repack) and the
    /// symbolic (extract via .extract(hi,lo), build per-lane Z3 expression,
    /// concat) paths. The trait `FloatLaneOp` provides the per-op specifics,
    /// forcing each impl to define both branches in lockstep.
    pub(super) fn vec_float_lane_op(
        args: &[RustBV],
        elem: IRType,
        count: u8,
        op: &dyn FloatLaneOp,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(args.len(), op.arity());
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        for a in args {
            Self::require_operand_width("vec_float_lane_op a", a.width(), total_width)?;
        }

        // Concrete fast path: every operand must fit in u128.
        if total_width <= 128 {
            let concrete: Option<Vec<u128>> =
                args.iter().map(crate::symbolic::RustBV::as_u128).collect();
            if let Some(concrete) = concrete {
                if !matches!(elem, IRType::F32 | IRType::F64) {
                    return Err(OpError::InvalidFloatType(elem));
                }
                let arity = concrete.len();
                // Hardened from a `debug_assert!` (angr-j60q0.2): `buf32`/`buf64`
                // are fixed `FLOAT_LANE_OP_MAX_ARITY`-wide and indexed `[idx]`
                // for `idx in 0..arity` below, so an over-arity caller would be
                // an index-panic process abort in release. Return a typed error
                // instead.
                if arity > FLOAT_LANE_OP_MAX_ARITY {
                    return Err(OpError::UnsupportedVectorOp(format!(
                        "vec_float_lane_op arity {arity} exceeds max {FLOAT_LANE_OP_MAX_ARITY}"
                    )));
                }
                let mut result: u128 = 0;
                let elem_mask: u128 = Self::low_bit_mask_u128(elem_width);
                let mut buf32 = [0f32; FLOAT_LANE_OP_MAX_ARITY];
                let mut buf64 = [0f64; FLOAT_LANE_OP_MAX_ARITY];
                for i in 0..count {
                    let shift = (i as u32) * elem_width;
                    let lane_bits = match elem {
                        IRType::F32 => {
                            for (idx, raw) in concrete.iter().enumerate() {
                                buf32[idx] = f32::from_bits(((raw >> shift) & elem_mask) as u32);
                            }
                            op.concrete_f32(&buf32[..arity]).to_bits() as u128
                        }
                        IRType::F64 => {
                            for (idx, raw) in concrete.iter().enumerate() {
                                buf64[idx] = f64::from_bits(((raw >> shift) & elem_mask) as u64);
                            }
                            op.concrete_f64(&buf64[..arity]).to_bits() as u128
                        }
                        // Hardened from `unreachable!()` (angr-j60q0.2): `elem`
                        // is validated F32|F64 above, but keep the contract in
                        // release should a future caller bypass that guard.
                        _ => return Err(OpError::InvalidFloatType(elem)),
                    };
                    result |= lane_bits << shift;
                }
                return Ok(RustBV::concrete(result, total_width));
            }
        }

        // Symbolic per-lane fallback.
        let prec = float_prec_of(elem).ok_or(OpError::InvalidFloatType(elem))?;
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            let lane_args: Vec<RustBV> = args.iter().map(|a| a.extract(hi, lo, ctx)).collect();
            elements.push(op.symbolic(lane_args, prec, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }
}
