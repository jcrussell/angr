//! Shared helpers for the `value_*_tests.rs` family.
//!
//! Both helpers here are used by more than one of the sibling test modules
//! that `value.rs` registers via `test_submod!`; anything used by exactly one
//! of them stays private to that module instead.

use super::*;

/// Recursive structural rendering of a `RustBV` tree.
///
/// Reads through [`RustBV::leaf_var`] and [`RustBV::node`], so a shape
/// assertion names leaves and ops rather than clarirs' SMT-LIB rendering.
pub(super) fn bv_shape(bv: &RustBV) -> String {
    if let Some((name, width)) = bv.leaf_var() {
        return format!("Leaf({name}, {width})");
    }
    match bv.node() {
        Some(node) => {
            let inner: Vec<String> = node.operands.iter().map(bv_shape).collect();
            format!("Expr({:?}, {}, [{}])", node.op, bv.width(), inner.join(", "))
        }
        None => format!("{bv:?}"),
    }
}

/// Helper: build an Extract node without going through `extract_into`'s
/// rewrites, so a test can check how such a node behaves (simulates Extract
/// nodes that bypass the construction-time rewrite, e.g. via `truncate_into`).
pub(super) fn raw_extract_node(inner: RustBV, high: u32, low: u32) -> RustBV {
    RustBV::from_op(high - low + 1, BVOp::Extract(high, low), vec![inner])
}
