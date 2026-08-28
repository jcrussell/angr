//! Unit tests for `SymbolicMemory` and friends.
//!
//! Split into per-feature submodules to keep individual files manageable.
//! Each submodule lives next door to the production code via `super::super`.
//!
//! - `basic`: core ops (concrete load/store, endianness, fork, map_data,
//!   fast-path symbolic, permission/check-executable).
//! - `symbolic_load_store`: core symbolic store/load paths — per-byte concat
//!   fallback, `import_symbolic_value` width contract, load-path resolution
//!   order.
//! - `symbolic_wide`: byte ordering for wide (128-bit) symbolic stores and
//!   the wider-symbolic linear-scan fallback, both endiannesses.
//! - `symbolic_overlap`: a later symbolic store partially covering an earlier
//!   one, and concrete/narrower stores truncating a wider symbolic entry.
//! - `symbolic_cross_page`: symbolic-address concretization across a page
//!   boundary and permission enforcement for page-straddling accesses.
//! - `symbolic_fork`: fork isolation of the symbolic sidecars plus the
//!   `pending_writes` deferred-write lifecycle.
//! - `symbolic_counters`: memory ITE depth, load/store volume,
//!   concretization, lazy-page-fault and symbolic-address counters.
//! - `multi`: `MultiPayload` data structure and the whole Phase 1–4.2
//!   multi-cell install/collapse/cache/coalesce lifecycle, split further into
//!   `payload`, `collapse`, `install`, `cache` and `coalesce` submodules.
//! - `ite_dedup`: ITE deduplication on loads + address-disjunction hoisting.
//! - `merge_cost_shape`: S5a measurement record for the merge cost-shape
//!   question (spike; the optimization it recommended has since shipped).
//! - `merge_divergence`: the shipped divergence-proportional merge — the
//!   `is_shared_identical` CoW page skip, its walk count, and its soundness
//!   against the symbolic-overlay trap.
//! - `merge_multi`: merging a byte that is Multi on both arms into one lazy
//!   Multi cell, plus merge-condition guarding of deferred `pending_writes`.
//! - `merge_prefetch`: soundness-critical paths no integration test reaches —
//!   per-byte merge ITE selection, `load_concrete_or_unconstrained`'s Err
//!   fallback, and `get_region_prefetch_list`.
//! - `merge_sidecars`: every `SymbolicMemory` sidecar field is merged per its
//!   declared `#[merge_policy]` (behavioural half of the `MergePolicy` derive).
//! - `page_boundary_property_tests`: property-based concrete round-trips for
//!   stores/loads straddling a page boundary.

mod basic;
mod ite_dedup;
mod merge_cost_shape;
mod merge_divergence;
mod merge_multi;
mod merge_prefetch;
mod merge_sidecars;
mod multi;
mod page_boundary_property_tests;
mod symbolic_counters;
mod symbolic_cross_page;
mod symbolic_fork;
mod symbolic_load_store;
mod symbolic_overlap;
mod symbolic_wide;

/// Test-only instrumentation for `SymbolicMemory::merge`'s page-walk count
/// (angr-op0dn.11.2.1). The production merge increments this each time it falls
/// through the CoW fast path and materializes a both-present page. Tests reset
/// it, run a merge, and assert the walk count equals the divergent-page count.
pub(crate) mod merge_instrument {
    use std::cell::Cell;

    thread_local! {
        static PAGES_WALKED: Cell<u64> = const { Cell::new(0) };
    }

    /// Called from `SymbolicMemory::merge` on each materialized both-present page.
    #[inline]
    pub(crate) fn note_page_walked() {
        PAGES_WALKED.with(|c| c.set(c.get() + 1));
    }

    /// Reset the counter to zero before a measured merge.
    pub(crate) fn reset() {
        PAGES_WALKED.with(|c| c.set(0));
    }

    /// Read the number of pages walked since the last [`reset`].
    pub(crate) fn walked() -> u64 {
        PAGES_WALKED.with(|c| c.get())
    }
}
