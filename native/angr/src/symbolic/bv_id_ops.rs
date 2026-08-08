//! BV-construction / id-allocation `&self` helpers for [`SymContext`].
//!
//! Slice 8 of the `symbolic/context.rs` split (angr-a2br.2.6), the design's
//! boundary (2) "BV expression construction". These are the small, low-coupling
//! symbol-allocation helpers: the unique-id counter (`next_id`), the constraint
//! count accessor (`num_constraints`), and the symbolic-bitvector factory pair
//! (`new_bv` / `unique_name`).
//!
//! Coupling is minimal: the only `SymContext` field these touch is the
//! `constraint_count` atomic — promoted to `pub(super)` so this sibling module
//! can reach it. Ids do **not** come from a context field at all: `next_id`
//! reads the process-global `NEXT_SYMBOL_ID` static below, for the aliasing
//! reason its own doc gives. `new_bv` routes construction through
//! `RustBV::symbolic`. None of these touch the constraint-mutation/transaction
//! path (`local_constraints` / `solver` / `push_level` / lineage), which the
//! design defers to a separate, higher-coupling slice 9. See bead angr-a2br.2.4
//! for the slice plan.
//!
//! Lives as a second `impl SymContext` block in a child module of `symbolic`;
//! `pub(super)` (== `pub(in crate::symbolic)`) keeps the promoted atomics
//! module-private — no public API leak. These methods are not Z3-gated: they
//! exist in both the z3 and non-z3 builds.

use super::RustBV;
use super::SymContext;
use std::sync::atomic::{AtomicU64, Ordering};

/// Process-global symbol-id allocator.
///
/// Symbol ids MUST be unique per process, not per [`SymContext`]: the maps that
/// resolve an id back to its claripy AST / name / width
/// (`symbolic::registry::GLOBAL_REGISTRY`, reached from the claripy-bridge
/// export path) are process-global and outlive any one manager. A per-context
/// counter starting at 0 therefore made the second exploration in a process
/// mint ids the first one's symbols still own in the registry, so an exported
/// symbol resolved to a *stranger's* claripy AST — the same aliasing failure
/// angr-op0dn.13.14 fixed for snapshot resume, but across two explorations
/// (angr-op0dn.13.16). Aliased symbols collapse a compare-and-branch to a
/// concrete guard, which silently adds or drops whole search subtrees.
///
/// Mirrors the `state-id-never-reused` contract on `state::NEXT_STATE_ID`.
static NEXT_SYMBOL_ID: AtomicU64 = AtomicU64::new(0);

/// Raise the global symbol-id counter so the next minted id is strictly greater
/// than `id`. Used by snapshot restore, whose deserialized leaves carry ids
/// minted by a *different* process (see
/// [`SymContext::restore_from_snapshot`](crate::symbolic::SymContext::restore_from_snapshot)).
pub fn reserve_symbol_id(id: u64) {
    NEXT_SYMBOL_ID.fetch_max(id, Ordering::SeqCst);
}

/// The current high-water mark: every id minted so far is strictly below this.
pub fn symbol_id_watermark() -> u64 {
    NEXT_SYMBOL_ID.load(Ordering::SeqCst)
}

thread_local! {
    /// Offset added to every symbol id deserialized on this thread while a
    /// [`SymbolIdRebase`] guard is live. `0` (the default) is the identity.
    static SYMBOL_ID_REBASE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The active deserialization id offset — see [`SymbolIdRebase`].
pub fn symbol_id_rebase_offset() -> u64 {
    SYMBOL_ID_REBASE.with(std::cell::Cell::get)
}

/// RAII guard that shifts every symbol id deserialized on this thread into a
/// fresh, process-local range (angr-euw28).
///
/// [`reserve_symbol_id`] keeps ids minted *after* a restore clear of the
/// restored leaves, but a snapshot written by a **different process** carries
/// ids from a foreign allocator that started at 0 — those alias ids this
/// process already minted (e.g. the seed state's own symbols), and every
/// id-keyed lookup (the claripy export registry, `stored_conditions`, the
/// symbol table) then resolves a restored leaf to a *stranger's* symbol. The
/// reported symptom was a restored `stdin_0_256` exporting under the seed's
/// `stdin_81_256` name.
///
/// Restoring a foreign envelope therefore rebases the whole snapshot's id space
/// by the current watermark: all leaves shift by the same offset, so ids stay
/// internally consistent, while Z3 identity — which is by NAME, not by id (see
/// `invariant-symbol-identity-is-by-name`) — is untouched, so the replayed
/// constraints still bind to the same variables.
///
/// The offset is thread-local and defaults to 0, so worker-migration payloads
/// (same process, ids already unique) deserialize unchanged.
pub struct SymbolIdRebase {
    prev: u64,
}

impl SymbolIdRebase {
    /// Activate a rebase by `offset` on this thread until the guard drops.
    pub fn activate(offset: u64) -> Self {
        let prev = SYMBOL_ID_REBASE.with(|c| c.replace(offset));
        Self { prev }
    }
}

impl Drop for SymbolIdRebase {
    fn drop(&mut self) {
        SYMBOL_ID_REBASE.with(|c| c.set(self.prev));
    }
}

impl SymContext {
    /// Get the next unique ID for a symbolic variable.
    ///
    /// Allocates from the process-global `NEXT_SYMBOL_ID`, not a per-context
    /// counter — see that static for why.
    pub fn next_id(&self) -> u64 {
        NEXT_SYMBOL_ID.fetch_add(1, Ordering::SeqCst)
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
    ///
    /// Mints exactly **one** id and uses it for both halves of the value: the
    /// `{base}_{id}` display name and the `Symbolic.id` identity key. Going
    /// through [`SymContext::unique_name`] + [`RustBV::symbolic`] instead would
    /// draw two ids per call and leave the visible name one behind the real
    /// `.id` — confusing in debug output and state exports (angr-sqfj8.90).
    pub fn new_bv(&self, name: &str, width: u32) -> RustBV {
        let id = self.next_id();
        RustBV::symbolic_with_id(id, format!("{name}_{id}"), width)
    }

    /// Create a unique name for a symbol.
    pub fn unique_name(&self, base: &str) -> String {
        let id = self.next_id();
        format!("{base}_{id}")
    }
}
