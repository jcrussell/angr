//! ITE (if-then-else) tree builders for symbolic-address memory loads.
//!
//! These helpers turn a set of candidate concrete addresses (either a strided
//! pattern or an arbitrary sorted list) into a balanced binary tree of ITE
//! expressions. Balanced trees give O(log N) depth instead of the O(N) depth
//! of a linear chain, which keeps Z3 ASTs small for large concretization sets.
//!
//! Two flavors live here:
//! - `load_concrete_lazy` based: used by the original symbolic-load path; can
//!   propagate `MemoryError` if a candidate page is unmapped (no lazy region).
//! - `load_concrete_or_unconstrained` based: used by the unified path that has
//!   already prepared addresses; falls back to a fresh symbolic value on miss.
//!
//! All entry points are inherent methods on `SymbolicMemory` re-exported by the
//! parent module, so callers in `memory/mod.rs` keep using `self.method(...)`.
use std::sync::Arc;

use crate::concretize::strided_addr_at;
use crate::symbolic::{RustBV, SymContext};

use super::{MemoryError, SymbolicMemory};

/// Light-weight structural equality used by the ITE builders to detect when
/// two sibling subtrees produce identical loaded values, in which case the
/// surrounding ITE can be collapsed (`ite(c, v, v) -> v`). Returns true only
/// when the two values are provably identical without recursing into
/// expression trees — conservative by design, so it never collapses
/// distinct ASTs.
///
/// Catches the common bead-described case (angr-269l): N-way concretization
/// where every candidate address maps to the same memory content (zero
/// pages, repeated initializers). Those loads return identical
/// `RustBV::Concrete` values, so the entire balanced ITE tree collapses to
/// a single leaf — no `ITE` nodes, no Or-of-equalities to construct.
fn loads_match(a: &RustBV, b: &RustBV) -> bool {
    use RustBV::*;
    if a.width() != b.width() {
        return false;
    }
    match (a, b) {
        (Concrete { value: v1, .. }, Concrete { value: v2, .. }) => v1 == v2,
        (Symbolic { id: i1, .. }, Symbolic { id: i2, .. }) => i1 == i2,
        (
            Constrained {
                id: i1, value: v1, ..
            },
            Constrained {
                id: i2, value: v2, ..
            },
        ) => i1 == i2 && v1 == v2,
        // Two Expression nodes are treated as identical only when they share
        // the same operand `Arc` and op. This catches incidental sharing
        // (e.g. when distinct addresses route to the same cached AST) without
        // paying for a deep structural walk on every ITE construction.
        (
            Expression {
                op: op1,
                operands: ops1,
                ..
            },
            Expression {
                op: op2,
                operands: ops2,
                ..
            },
        ) => op1 == op2 && Arc::ptr_eq(ops1, ops2),
        _ => false,
    }
}

/// The shape of a strided access — fixed for the whole ITE-tree recursion,
/// so it travels as one immutable bundle rather than three repeated params.
struct StridedPattern {
    /// Base address of the strided pattern.
    base: u64,
    /// Stride between consecutive addresses.
    stride: u64,
    /// Number of bytes loaded at each address.
    size: u32,
}

impl SymbolicMemory {
    /// Load from strided addresses using a balanced ITE tree.
    ///
    /// For a strided pattern like base, base+stride, base+2*stride, ...,
    /// this builds a balanced binary tree of ITE expressions with O(log N) depth
    /// instead of the linear O(N) depth of a chain.
    ///
    /// The tree structure:
    /// ```text
    ///                      ITE(addr <= mid_addr)
    ///                     /                    \
    ///        ITE(addr <= lo_mid)        ITE(addr <= hi_mid)
    ///           /      \                    /       \
    ///         ...     ...                ...        ...
    /// ```
    pub(super) fn load_strided_balanced(
        &self,
        addr_expr: &RustBV,
        base: u64,
        stride: u64,
        count: u64,
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        if count == 0 {
            return Err(MemoryError::SymbolicAddress {
                description: "strided access with zero count".to_string(),
            });
        }
        if count == 1 {
            return self.load_concrete_lazy(base, size, ctx);
        }

        // Build the balanced tree recursively
        let pattern = StridedPattern { base, stride, size };
        // overflow-ok: `count == 0` is rejected above, so `count - 1 >= 0`.
        self.build_strided_ite_tree(addr_expr, &pattern, 0, count - 1, ctx)
    }

    /// Recursive helper to build a balanced ITE tree for strided access.
    ///
    /// # Arguments
    /// * `addr_expr` - The symbolic address expression
    /// * `pattern` - Base/stride/size of the strided access (fixed across the recursion)
    /// * `lo` - Lowest index in the current subtree
    /// * `hi` - Highest index in the current subtree
    /// * `ctx` - Solver context
    fn build_strided_ite_tree(
        &self,
        addr_expr: &RustBV,
        pattern: &StridedPattern,
        lo: u64,
        hi: u64,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        let StridedPattern { base, stride, size } = *pattern;

        // Base case: single element
        if lo == hi {
            let addr = strided_addr_at(base, stride, lo);
            return self.load_concrete_lazy(addr, size, ctx);
        }

        // Split at midpoint for balanced tree
        let mid = (lo + hi) / 2;
        let mid_addr = strided_addr_at(base, stride, mid);

        // Build condition: addr <= mid_addr
        let mid_const = RustBV::concrete(mid_addr as u128, addr_expr.width());
        let cond = addr_expr.ule(&mid_const, ctx);

        // Recursively build left subtree (lo..mid) and right subtree (mid+1..hi)
        let left = self.build_strided_ite_tree(addr_expr, pattern, lo, mid, ctx)?;
        let right = self.build_strided_ite_tree(addr_expr, pattern, mid + 1, hi, ctx)?;

        // angr-269l: collapse `ite(c, v, v) -> v` so identical-content
        // strided regions fold to a single leaf instead of an N-deep tree.
        if loads_match(&left, &right) {
            return Ok(left);
        }

        // Build ITE: if (addr <= mid_addr) then left else right
        Ok(cond.ite(&left, &right, ctx))
    }

    /// Build a balanced ITE tree for arbitrary addresses.
    ///
    /// This is similar to the strided version but works with any sorted
    /// list of addresses. Uses binary search style partitioning.
    pub(super) fn build_balanced_ite_load(
        &self,
        addr_expr: &RustBV,
        addrs: &[u64],
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        if addrs.is_empty() {
            return Err(MemoryError::SymbolicAddress {
                description: "empty address list".to_string(),
            });
        }
        if addrs.len() == 1 {
            return self.load_concrete_lazy(addrs[0], size, ctx);
        }

        self.build_balanced_ite_load_inner(addr_expr, addrs, size, ctx)
    }

    /// Recursive helper for balanced ITE tree with arbitrary addresses.
    /// Leaves load via the error-propagating `load_concrete_lazy`.
    fn build_balanced_ite_load_inner(
        &self,
        addr_expr: &RustBV,
        addrs: &[u64],
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        let mut load_leaf =
            |m: &Self, a: u64, s: u32, c: &SymContext| m.load_concrete_lazy(a, s, c);
        self.build_ite_tree_generic(addr_expr, addrs, size, ctx, &mut load_leaf)
    }

    /// Shared recursive ITE-tree scaffold (angr-24pv4.3) parameterized over a
    /// `load_leaf` closure so the only difference between the lazy
    /// (`load_concrete_lazy`, error-propagating) and unified
    /// (`load_concrete_or_unconstrained`, unconstrained fallback) builders is
    /// the leaf loader. Tree shape: 1-addr leaf; 2-addr `eq`-ITE with
    /// `loads_match` dedup; N-addr midpoint split on `ult` then recurse with
    /// identical-subtree collapse (all angr-269l).
    fn build_ite_tree_generic(
        &self,
        addr_expr: &RustBV,
        addrs: &[u64],
        size: u32,
        ctx: &SymContext,
        load_leaf: &mut impl FnMut(&Self, u64, u32, &SymContext) -> Result<RustBV, MemoryError>,
    ) -> Result<RustBV, MemoryError> {
        // Base case: single address
        if addrs.len() == 1 {
            return load_leaf(self, addrs[0], size, ctx);
        }

        // Base case: two addresses - simple ITE
        if addrs.len() == 2 {
            let left_val = load_leaf(self, addrs[0], size, ctx)?;
            let right_val = load_leaf(self, addrs[1], size, ctx)?;

            // angr-269l: skip the ITE when both addresses map to the same content.
            if loads_match(&left_val, &right_val) {
                return Ok(left_val);
            }

            let left_const = RustBV::concrete(addrs[0] as u128, addr_expr.width());
            let cond = addr_expr.eq(&left_const, ctx);

            return Ok(cond.ite(&left_val, &right_val, ctx));
        }

        // Split at midpoint
        let mid = addrs.len() / 2;
        let mid_addr = addrs[mid];

        // Build condition: addr < mid_addr (for binary partition)
        let mid_const = RustBV::concrete(mid_addr as u128, addr_expr.width());
        let cond = addr_expr.ult(&mid_const, ctx);

        // Recursively build left (addrs < mid) and right (addrs >= mid) subtrees
        let left = self.build_ite_tree_generic(addr_expr, &addrs[..mid], size, ctx, load_leaf)?;
        let right = self.build_ite_tree_generic(addr_expr, &addrs[mid..], size, ctx, load_leaf)?;

        // angr-269l: collapse identical subtrees so multi-address dedup
        // propagates up the tree (every leaf identical → root is the leaf).
        if loads_match(&left, &right) {
            return Ok(left);
        }

        // Build ITE: if (addr < mid_addr) then left else right
        Ok(cond.ite(&left, &right, ctx))
    }

    /// Build a balanced ITE tree after addresses have been prepared.
    ///
    /// This is the immutable part of the unified load, called after prepare_addresses_for_ite.
    pub(super) fn build_balanced_ite_load_after_prep(
        &self,
        addr_expr: &RustBV,
        addrs: &[u64],
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        if addrs.is_empty() {
            // angr-03vl4.41: `mem_empty_ite_{size}` alone is not a unique name —
            // two independent empty-address loads of the same width would intern
            // to the same Z3 constant. Same reasoning as the `unc_mem_` fallback
            // in `SymbolicMemory::load_concrete_or_unconstrained`.
            return Ok(ctx.new_bv(&format!("mem_empty_ite_{size}"), size * 8));
        }
        if addrs.len() == 1 {
            return Ok(self.load_concrete_or_unconstrained(addrs[0], size, ctx));
        }

        self.build_ite_tree_inner(addr_expr, addrs, size, ctx)
    }

    /// Recursive helper for building ITE tree (immutable borrow). Leaves load
    /// via the infallible `load_concrete_or_unconstrained`.
    fn build_ite_tree_inner(
        &self,
        addr_expr: &RustBV,
        addrs: &[u64],
        size: u32,
        ctx: &SymContext,
    ) -> Result<RustBV, MemoryError> {
        let mut load_leaf = |m: &Self, a: u64, s: u32, c: &SymContext| {
            Ok(m.load_concrete_or_unconstrained(a, s, c))
        };
        self.build_ite_tree_generic(addr_expr, addrs, size, ctx, &mut load_leaf)
    }
}
