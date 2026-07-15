//! Unit tests for `SymbolicMemory` and friends.
//!
//! Split into per-feature submodules to keep individual files manageable.
//! Each submodule lives next door to the production code via `super::super`.
//!
//! - `basic`: core ops (concrete load/store, endianness, fork, map_data,
//!   fast-path symbolic, permission/check-executable).
//! - `symbolic`: symbolic stores/loads, partial overlap, wide-store paths,
//!   cross-page concretization, fork isolation, counters/metrics.
//! - `multi`: `MultiPayload` data structure and Phase 1–4.2 multi-cell
//!   collapse/coalesce behavior.
//! - `ite_dedup`: ITE deduplication on loads + address-disjunction hoisting.

mod basic;
mod ite_dedup;
mod merge_cost_shape;
mod merge_divergence;
mod merge_prefetch;
mod multi;
mod symbolic;

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
