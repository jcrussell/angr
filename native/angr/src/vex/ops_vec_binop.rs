//! Generic packed-integer element-wise binary op dispatcher.
//!
//! Extracted from the parent `ops` module (angr-cudgw.18) to shrink the
//! VEXOps god-file. Declared as a child module of `ops` (via `#[path]` in
//! ops.rs), so this `pub(super)` method stays callable from the binop
//! dispatch in `ops`, and the shared sibling it references
//! (`Self::concat_le_elements`, which stays in ops.rs) stays visible via the
//! descendant rule.
//!
//! Covers the `add`/`sub`/`mul` per-lane packed-integer family (Iop_Add/Sub/
//! Mul{N}x{M}): a concrete shift+mask fast path plus a symbolic per-lane
//! extract/op/concat path.

use super::{OpError, VEXOps};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ir::IRType;

impl VEXOps {
    /// Vector element-wise binary operation.
    pub(super) fn vec_binop(
        left: RustBV,
        right: RustBV,
        elem: IRType,
        count: u8,
        op: &str,
        ctx: &SymContext,
    ) -> Result<RustBV, OpError> {
        let elem_width = elem.bits();
        let total_width = elem_width * count as u32;

        debug_assert_eq!(left.width(), total_width);
        debug_assert_eq!(right.width(), total_width);

        // For concrete values, we can do this efficiently
        if let (Some(l), Some(r)) = (left.as_u128(), right.as_u128()) {
            let mut result: u128 = 0;

            for i in 0..count {
                let lo = (i as u32) * elem_width;
                let _hi = lo + elem_width - 1;
                let mask = (1u128 << elem_width) - 1;

                let l_elem = (l >> lo) & mask;
                let r_elem = (r >> lo) & mask;

                let res_elem = match op {
                    "add" => l_elem.wrapping_add(r_elem) & mask,
                    "sub" => l_elem.wrapping_sub(r_elem) & mask,
                    "mul" => l_elem.wrapping_mul(r_elem) & mask,
                    _ => return Err(OpError::UnsupportedVectorOp(op.to_string())),
                };

                result |= res_elem << lo;
            }

            return Ok(RustBV::concrete(result, total_width));
        }

        // For symbolic, we need to extract, operate, and concatenate
        let mut elements: Vec<RustBV> = Vec::with_capacity(count as usize);

        for i in 0..count {
            let lo = (i as u32) * elem_width;
            let hi = lo + elem_width - 1;

            let l_elem = left.extract(hi, lo, ctx);
            let r_elem = right.extract(hi, lo, ctx);

            let res_elem = match op {
                "add" => l_elem.add_into(r_elem, ctx),
                "sub" => l_elem.sub_into(r_elem, ctx),
                "mul" => l_elem.mul_into(r_elem, ctx),
                _ => return Err(OpError::UnsupportedVectorOp(op.to_string())),
            };

            elements.push(res_elem);
        }

        Ok(Self::concat_le_elements(elements, ctx))
    }
}
