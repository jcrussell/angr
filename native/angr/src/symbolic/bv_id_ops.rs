//! BV-construction / id-allocation `&self` helpers for [`SymContext`].
//!
//! Slice 8 of the `symbolic/context.rs` split (angr-a2br.2.6), the design's
//! boundary (2) "BV expression construction". These are the small, low-coupling
//! symbol-allocation helpers: the unique-id counter (`next_id`), the constraint
//! count accessor (`num_constraints`), and the symbolic-bitvector factory pair
//! (`new_bv` / `unique_name`).
//!
//! Coupling is minimal: the only private fields these touch are the two atomics
//! `next_id` and `constraint_count` — promoted to `pub(super)` so this sibling
//! module can reach them. `new_bv` routes construction through
//! `RustBV::symbolic`. None of these touch the constraint-mutation/transaction
//! path (`local_constraints` / `solver` / `push_level` / lineage), which the
//! design defers to a separate, higher-coupling slice 9. See bd memory
//! `a2br2-context-split-impl-block-plan` for the slice plan.
//!
//! Lives as a second `impl SymContext` block in a child module of `symbolic`;
//! `pub(super)` (== `pub(in crate::symbolic)`) keeps the promoted atomics
//! module-private — no public API leak. These methods are not Z3-gated: they
//! exist in both the z3 and non-z3 builds.

use super::RustBV;
use super::SymContext;
use std::sync::atomic::Ordering;

impl SymContext {
    /// Get the next unique ID for a symbolic variable.
    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Get the number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.constraint_count.load(Ordering::SeqCst)
    }

    /// Overwrite the constraint counter. Used only by
    /// [`SymContext::restore_from_snapshot`](crate::symbolic::SymContext::restore_from_snapshot)
    /// to pin the count back to the source's captured value after the
    /// assume-class IR replay (angr-kenpr — the replay can re-assert
    /// live-deduped entries and inflate the counter).
    pub fn set_constraint_count(&self, count: usize) {
        self.constraint_count.store(count, Ordering::SeqCst);
    }

    // =========================================================================
    // Symbol Management
    // =========================================================================

    /// Create a new symbolic bitvector with a unique name.
    pub fn new_bv(&self, name: &str, width: u32) -> RustBV {
        let unique_name = self.unique_name(name);
        RustBV::symbolic(self, &unique_name, width)
    }

    /// Create a unique name for a symbol.
    pub fn unique_name(&self, base: &str) -> String {
        let id = self.next_id();
        format!("{base}_{id}")
    }
}
