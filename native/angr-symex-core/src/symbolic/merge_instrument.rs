//! Test-only instrumentation for `SymContext::merge`'s guarded-assertion count
//! (angr-op0dn.11.3). The production merge increments this for every guarded
//! `Or(!cond, c)` it emits plus the final `Or` of merge flags. On the
//! shared-prefix (CoW) path the guarded count collapses to the divergent
//! (local) constraint count + 1; on the fallback path it stays the total
//! constraint count + 1. Tests reset it, run a merge, and assert the count.
//!
//! Split out of `context.rs` (angr-fs8kb.35); still declared from there via
//! `#[path]` so the module path stays `symbolic::context::merge_instrument`.

use std::cell::Cell;

thread_local! {
    static GUARDED_EMITTED: Cell<u64> = const { Cell::new(0) };
}

/// Called from `SymContext::merge` for each guarded `Or` (and the flag `Or`).
#[inline]
pub(crate) fn note_guarded() {
    GUARDED_EMITTED.with(|c| c.set(c.get() + 1));
}

/// Reset the counter to zero before a measured merge.
pub(crate) fn reset() {
    GUARDED_EMITTED.with(|c| c.set(0));
}

/// Read the number of guarded `Or`s emitted since the last [`reset`].
pub(crate) fn emitted() -> u64 {
    GUARDED_EMITTED.with(|c| c.get())
}
