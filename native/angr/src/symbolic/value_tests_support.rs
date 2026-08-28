//! Shared helpers for the `value_*_tests.rs` family.
//!
//! Both helpers here are used by more than one of the sibling test modules
//! that `value.rs` registers via `test_submod!`; anything used by exactly one
//! of them stays private to that module instead.

use super::*;

/// Recursive structural rendering of a `RustBV` tree.
///
/// `RustBV::PartialEq` on an `Expression` is node identity (same `op`/width
/// plus a pointer-identical operand slice — see the impl's doc), not a
/// structural walk, and its `Debug` stops at the operand *count*, so shape
/// assertions need their own walker.
pub(super) fn bv_shape(bv: &RustBV) -> String {
    match bv {
        RustBV::Expression {
            op,
            width,
            operands,
            ..
        } => {
            let inner: Vec<String> = operands.iter().map(bv_shape).collect();
            format!("Expr({op:?}, {width}, [{}])", inner.join(", "))
        }
        other => format!("{other:?}"),
    }
}

/// Helper: build a raw Extract Expression node without going through
/// `extract_into`. This simulates Extract nodes that bypass the
/// construction-time rewrite (e.g., via `truncate_into`), so we can verify the
/// Z3-emission pass picks them up.
#[cfg(feature = "vex-engine-z3")]
pub(super) fn raw_extract_node(inner: RustBV, high: u32, low: u32) -> RustBV {
    let result_width = high - low + 1;
    RustBV::Expression {
        id: RustBV::EXPRESSION_ID,
        width: result_width,
        op: BVOp::Extract(high, low),
        operands: std::sync::Arc::<[RustBV]>::from([inner]),
        memo: Default::default(),
    }
}
