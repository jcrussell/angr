//! Constraint sharing walk (angr-zdho).
//!
//! Walks a population of constraint RustBVs (typically every state's assumed
//! list at end of exploration) and reports:
//!
//! - `total_visits` — recursive descents through the DAG, counting Arc
//!   re-visits as separate. The upper bound on AST work if we had NO cache
//!   (neither per-call nor hash-cons).
//! - `unique_pointers` — distinct in-memory `RustBV` nodes seen. Two operand
//!   positions that alias the same node (because a whole `Arc<[RustBV]>`
//!   operand slice is shared between two expressions) count once. This is what
//!   today's per-call `to_z3_ast_cached` collapses repeated visits down to.
//! - `unique_shapes` — distinct *structural* shapes seen. Two nodes with the
//!   same op/width and structurally-equal children share a shape. This is the
//!   lower bound a construction-time hash-cons (angr-behq) would reach.
//!
//! `unique_pointers - unique_shapes` is the "structural-duplicate" count from
//! the bead — RustBVs that hash-cons could merge but the current build path
//! keeps distinct.
//!
//! # Why the walk takes ownership
//!
//! Node identity is the node's *address*, and addresses are only comparable
//! between values that are simultaneously alive: free a node and the allocator
//! may hand the same address to an unrelated later one. Callers reach this walk
//! through [`SymContext::fold_sharing_walk`](super::SymContext::fold_sharing_walk),
//! which walks the vector `get_assumed_constraints()` *clones* — a temporary
//! that used to be dropped at the end of each state's fold, so the next state's
//! temporary could land on the freed addresses and be scored as already-seen
//! (understating `unique_pointers`, hence `structural_duplicates`, and doing so
//! nondeterministically). [`ConstraintSharingWalk::visit_batch`] therefore takes
//! its batch by value and retains it until [`ConstraintSharingWalk::into_stats`]
//! consumes the walk, so every address the walk ever keyed on stays live and
//! distinct for the whole analysis (angr-gkcxh).

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
    /// visit, so `len()` is the count of unique RustBV nodes walked.
    ptr_to_shape: HashMap<usize, u64>,
    /// Structural shape → canonical id. `len()` is the count of unique
    /// structural shapes — what hash-cons would shrink to.
    shape_to_id: HashMap<StructuralKey, u64>,
    /// Monotonic id allocator for shape interning.
    next_shape_id: u64,
    /// Recursive descents (including Arc re-visits).
    total_visits: u64,
    /// Every batch handed to [`Self::visit_batch`], kept alive so the
    /// addresses in `ptr_to_shape` stay live — and therefore stay distinct
    /// from later batches' — for the whole walk (angr-gkcxh; see the module
    /// doc for why).
    ///
    /// A `Vec` of `Vec`s rather than one flat `Vec`: a flat one would
    /// reallocate as batches accumulate, moving already-keyed nodes out from
    /// under their recorded addresses. Growing the outer `Vec` only moves the
    /// inner `Vec` *headers*; the element buffers they point at never move.
    retained: Vec<Vec<RustBV>>,
}

impl ConstraintSharingWalk {
    pub fn new() -> Self {
        Self {
            ptr_to_shape: HashMap::new(),
            shape_to_id: HashMap::new(),
            next_shape_id: 0,
            total_visits: 0,
            retained: Vec::new(),
        }
    }

    /// Walk a batch of top-level constraint trees, taking ownership of the
    /// batch so its node addresses remain live for the rest of the walk.
    ///
    /// This is the only entry point: `visit` is private precisely so a caller
    /// cannot hand the walk a borrow of a value it is about to drop, which is
    /// the miscount described in the module doc (angr-gkcxh).
    pub fn visit_batch(&mut self, batch: Vec<RustBV>) {
        for node in &batch {
            self.visit(node);
        }
        // Only *after* walking, so `batch`'s element buffer is never reallocated
        // between keying a node's address and the end of the walk. `push` moves
        // the `Vec` header; the buffer the recorded addresses point into stays.
        self.retained.push(batch);
    }

    /// Recursively walk `node`, updating the maps and the visit counter.
    /// Returns the canonical shape id for `node`.
    fn visit(&mut self, node: &RustBV) -> u64 {
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

test_submod!("sharing_tests.rs" => tests);
