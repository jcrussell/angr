//! Shared helpers for the `value_*_tests.rs` family.
//!
//! The helper here is used by more than one of the sibling test modules
//! that `value.rs` registers as `#[path]` children; anything used by exactly one
//! of them stays private to that module instead.

use super::*;

/// Recursive structural rendering of a `SymBV` tree.
///
/// Reads through [`SymBV::leaf_var`] and [`SymBV::node`], so a shape
/// assertion names leaves and ops rather than clarirs' SMT-LIB rendering.
pub(super) fn bv_shape(bv: &SymBV) -> String {
    if let Some((name, width)) = bv.leaf_var() {
        return format!("Leaf({name}, {width})");
    }
    match bv.node() {
        Some(node) => {
            let inner: Vec<String> = node.operands.iter().map(bv_shape).collect();
            format!(
                "Expr({:?}, {}, [{}])",
                node.op,
                bv.width(),
                inner.join(", ")
            )
        }
        None => format!("{bv:?}"),
    }
}
