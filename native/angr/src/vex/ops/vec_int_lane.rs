//! Generic per-lane packed-integer VEX op dispatcher.
//!
//! Mirrors `ops/vec_float_lane.rs` for the integer family. Declared as a child
//! module of `ops` (via a plain `mod` decl in `ops/mod.rs`), so this `pub(super)` method stays
//! callable from the unop/binop dispatch in `ops`, and the shared siblings it
//! references (`Self::concat_le_elements` and `Self::low_bit_mask_u128`, which
//! stay in `ops/mod.rs`; the `IntLaneOp` trait and the
//! `INT_LANE_OP_MAX_ARITY` const, which live in `ops/lane_traits.rs` and are
//! re-exported by `ops`) stay visible via the descendant rule.

use super::{INT_LANE_OP_MAX_ARITY, IntLaneOp, OpError, VEXOps};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ir::IRType;

impl VEXOps {
    /// Generic per-lane integer dispatcher. Handles the lane loop for both the
    /// concrete (extract via shift+mask, run the op, re-mask and repack) and the
    /// symbolic (extract via `.extract(hi,lo)`, build per-lane Z3 expression,
    /// concat) paths. The trait `IntLaneOp` provides the per-op specifics,
    /// forcing each impl to define both branches in lockstep.
    pub(super) fn vec_int_lane_op(
        args: &[RustBV],
        elem: IRType,
        count: u8,
        op: &dyn IntLaneOp,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        debug_assert_eq!(args.len(), op.arity());
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;
        for a in args {
            debug_assert_eq!(a.width(), total_width);
        }

        // Concrete fast path: every operand must fit in u128.
        if total_width <= 128 {
            let concrete: Option<Vec<u128>> =
                args.iter().map(crate::symbolic::RustBV::as_u128).collect();
            if let Some(concrete) = concrete {
                let arity = concrete.len();
                // Hardened from a `debug_assert!` (angr-5mnx3.67), mirroring
                // `VEXOps::vec_float_lane_op`'s angr-j60q0.2 fix: `buf` is a
                // fixed `INT_LANE_OP_MAX_ARITY`-wide array indexed `[idx]` for
                // `idx in 0..arity` below, so an over-arity caller would be an
                // index-panic process abort in release. Return a typed error
                // instead.
                if arity > INT_LANE_OP_MAX_ARITY {
                    return Err(OpError::UnsupportedVectorOp(format!(
                        "vec_int_lane_op arity {arity} exceeds max {INT_LANE_OP_MAX_ARITY}"
                    )));
                }
                let elem_mask = Self::low_bit_mask_u128(elem_width);
                let mut buf = [0u128; INT_LANE_OP_MAX_ARITY];
                let mut result: u128 = 0;
                for i in 0..count {
                    let lo = (i as u32) * elem_width;
                    for (idx, raw) in concrete.iter().enumerate() {
                        buf[idx] = (raw >> lo) & elem_mask;
                    }
                    let lane = op.concrete_lane(&buf[..arity], elem_width) & elem_mask;
                    result |= lane << lo;
                }
                return Ok(RustBV::concrete(result, total_width));
            }
        }

        // Symbolic per-lane fallback.
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);
        let mut lane_args: Vec<RustBV> = Vec::with_capacity(op.arity());
        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;
            lane_args.clear();
            for a in args {
                lane_args.push(a.extract(hi, lo, ctx));
            }
            elements.push(op.symbolic_lane(&lane_args, elem_width, ctx));
        }
        Ok(Self::concat_le_elements(elements, ctx))
    }
}
