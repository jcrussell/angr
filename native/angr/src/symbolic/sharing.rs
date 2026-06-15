//! Constraint sharing walk (angr-zdho).
//!
//! Walks a population of constraint RustBVs (typically every state's assumed
//! list at end of exploration) and reports:
//!
//! - `total_visits` — recursive descents through the DAG, counting Arc
//!   re-visits as separate. The upper bound on AST work if we had NO cache
//!   (neither per-call nor hash-cons).
//! - `unique_pointers` — distinct `Arc<RustBV>` allocations seen. This is what
//!   today's per-call `to_z3_ast_cached` collapses repeated visits down to.
//! - `unique_shapes` — distinct *structural* shapes seen. Two nodes with the
//!   same op/width and structurally-equal children share a shape. This is the
//!   lower bound a construction-time hash-cons (angr-behq) would reach.
//!
//! `unique_pointers - unique_shapes` is the "structural-duplicate" count from
//! the bead — RustBVs that hash-cons could merge but the current build path
//! keeps distinct.

use std::collections::HashMap;

use super::RustBV;

/// Canonical key for structural equality between two `RustBV` subtrees.
///
/// Two `RustBV` nodes hash to the same `StructuralKey` iff a construction-time
/// hash-cons would treat them as the same node.
#[derive(Clone, PartialEq, Eq, Hash)]
enum StructuralKey {
    /// Concrete leaf: same value at the same width.
    Concrete(u128, u32),
    /// Symbolic leaf: same `id` (ids are minted globally so this implies same
    /// width too, but include it explicitly for clarity).
    Symbolic(u64, u32),
    /// Constrained leaf: symbolic id with a known concrete witness.
    Constrained(u64, u128, u32),
    /// Expression node: same op, same width, structurally-equal operand list.
    Expression(super::value::BVOp, u32, Vec<u64>),
}

/// Accumulator for a constraint-sharing analysis pass.
///
/// Build with `ConstraintSharingWalk::new()`, fold one or more
/// `SymContext`s in via `SymContext::fold_sharing_walk`, then read out
/// the totals with `into_stats()`.
pub struct ConstraintSharingWalk {
    /// Pointer → canonical shape id. A pointer entry is created on first
    /// visit, so `len()` is the count of unique RustBV Arc allocations.
    ptr_to_shape: HashMap<usize, u64>,
    /// Structural shape → canonical id. `len()` is the count of unique
    /// structural shapes — what hash-cons would shrink to.
    shape_to_id: HashMap<StructuralKey, u64>,
    /// Monotonic id allocator for shape interning.
    next_shape_id: u64,
    /// Recursive descents (including Arc re-visits).
    total_visits: u64,
}

impl ConstraintSharingWalk {
    pub fn new() -> Self {
        Self {
            ptr_to_shape: HashMap::new(),
            shape_to_id: HashMap::new(),
            next_shape_id: 0,
            total_visits: 0,
        }
    }

    /// Recursively walk `node`, updating the maps and the visit counter.
    /// Returns the canonical shape id for `node`.
    pub fn visit(&mut self, node: &RustBV) -> u64 {
        self.total_visits = self.total_visits.saturating_add(1);
        let ptr_key = node as *const RustBV as usize;
        if let Some(&id) = self.ptr_to_shape.get(&ptr_key) {
            return id;
        }
        let key = match node {
            RustBV::Concrete { value, width } => StructuralKey::Concrete(*value, *width),
            RustBV::Symbolic { id, width, .. } => StructuralKey::Symbolic(*id, *width),
            RustBV::Constrained { id, value, width } => {
                StructuralKey::Constrained(*id, *value, *width)
            }
            RustBV::Expression {
                op,
                operands,
                width,
                ..
            } => {
                let mut op_ids = Vec::with_capacity(operands.len());
                for operand in operands.iter() {
                    op_ids.push(self.visit(operand));
                }
                StructuralKey::Expression(op.clone(), *width, op_ids)
            }
        };
        let id = if let Some(&existing) = self.shape_to_id.get(&key) {
            existing
        } else {
            let id = self.next_shape_id;
            self.next_shape_id = self.next_shape_id.saturating_add(1);
            self.shape_to_id.insert(key, id);
            id
        };
        self.ptr_to_shape.insert(ptr_key, id);
        id
    }

    /// Consume the walk and return the aggregate stats.
    pub fn into_stats(self) -> ConstraintSharingStats {
        ConstraintSharingStats {
            total_visits: self.total_visits,
            unique_pointers: self.ptr_to_shape.len() as u64,
            unique_shapes: self.shape_to_id.len() as u64,
        }
    }
}

impl Default for ConstraintSharingWalk {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregate output of `ConstraintSharingWalk::into_stats()`.
pub struct ConstraintSharingStats {
    pub total_visits: u64,
    pub unique_pointers: u64,
    pub unique_shapes: u64,
}
