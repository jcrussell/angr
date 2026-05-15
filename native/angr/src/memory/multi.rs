//! Multi-cell byte storage for lazy symbolic memory (Phase 1 of angr-czph).
//!
//! When a symbolic-address store concretizes to multiple candidate addresses,
//! the existing eager path builds an `ITE(addr == cand_i, new, current)` chain
//! per candidate cell and writes it back. Each subsequent overlapping store
//! nests another ITE layer — the bottleneck behind `sym-write` (memory
//! `symwrite-eager-vs-lazy-memory`).
//!
//! `MultiPayload` is the lazy alternative: instead of folding into an ITE at
//! store time, each candidate cell records its alternatives as a flat list of
//! `(cond, value)` pairs. The collapse to an ITE happens at load time, scoped
//! to only the bytes the load actually touches.
//!
//! This module defines the data structures only. Load-time collapse lives in
//! `memory/load.rs` (Phase 1.2, bead angr-n082); store helpers that emit
//! Multi cells live in `memory/store.rs` (Phase 1.3, bead angr-aija).

use std::cell::RefCell;

use super::{MemoryPage, PAGE_MASK, Permission, SymbolicMemory};
use crate::symbolic::{RustBV, SymContext, record_mem_ite_depth};

/// One alternative inside a `MultiPayload`. `cond` is a boolean BV (width 1)
/// that selects this alternative; `value` is the byte stored when `cond`
/// holds.
///
/// Invariants the producers must uphold (enforced by callers, not by this
/// type) — see `MultiPayload` doc comment.
#[derive(Debug, Clone)]
pub struct MultiAlternative {
    /// Boolean BV (width 1) under which `value` is the chosen byte.
    pub cond: RustBV,
    /// Byte value (width 8) for this alternative.
    pub value: RustBV,
}

impl MultiAlternative {
    pub fn new(cond: RustBV, value: RustBV) -> Self {
        debug_assert_eq!(
            value.width(),
            8,
            "MultiAlternative::value must be a single byte (width 8)"
        );
        MultiAlternative { cond, value }
    }
}

/// Memoized result of collapsing a `MultiPayload` to a single BV. The
/// collapse is keyed on the page's concrete byte at the time of computation
/// because that byte is the final `else` leaf of the right-fold ITE chain.
/// If a later concrete store mutates the underlying page byte without
/// clearing the Multi marker, the cache must be rebuilt — that is what the
/// `default_byte` field detects.
#[derive(Debug, Clone)]
struct CachedCollapse {
    default_byte: u8,
    bv: RustBV,
}

/// A set of alternatives for a single byte cell.
///
/// # Invariants (caller-enforced)
///
/// * Conditions are pairwise distinct concretized address equalities
///   (e.g. `addr == 0x1000`, `addr == 0x1004`) so that under any model
///   exactly one alternative's `cond` evaluates to true. Producers in
///   `memory/store.rs` construct these from `ConcretizationResult::Multiple`
///   /`Strided` results.
/// * All `value` BVs have width 8 (one byte). Wider values must be split
///   per byte before populating the payload.
///
/// # Counter contract
///
/// Per memory `invariant-mem-ite-depth-counter`, code that inserts a payload
/// via `SymbolicMemory::set_multi_alternatives` MUST call
/// `crate::symbolic::record_mem_ite_depth(payload.len() as u32)` so the Phase
/// 0 baseline comparison stays direct. The public setter handles this
/// automatically; private mutation paths must not bypass it.
///
/// # Collapse cache (Phase 3, bead angr-j0n4)
///
/// `cached_collapse` memoizes the right-folded ITE BV produced by
/// `collapse`. Reads (e.g. `assemble_load_with_multi`) reuse the cached BV
/// when the page's concrete default byte is unchanged. Any append-mutation
/// (`push`) invalidates the cache; `set_multi_alternatives` always installs
/// a fresh payload (cache starts as `None`) so the merge path in
/// `install_multi_for_candidates` is automatically safe.
#[derive(Debug, Default)]
pub struct MultiPayload {
    alternatives: Vec<MultiAlternative>,
    cached_collapse: RefCell<Option<CachedCollapse>>,
}

impl Clone for MultiPayload {
    fn clone(&self) -> Self {
        MultiPayload {
            alternatives: self.alternatives.clone(),
            // BVs are refcounted on the Z3 side, so cloning the cache is
            // cheap. The fork path benefits from carrying it forward.
            cached_collapse: RefCell::new(self.cached_collapse.borrow().clone()),
        }
    }
}

impl MultiPayload {
    /// Build a payload from a list of alternatives. Caller is responsible for
    /// the invariants documented on the type.
    pub fn from_alternatives(alternatives: Vec<MultiAlternative>) -> Self {
        MultiPayload {
            alternatives,
            cached_collapse: RefCell::new(None),
        }
    }

    /// Number of alternatives in this cell.
    pub fn len(&self) -> usize {
        self.alternatives.len()
    }

    /// True if the payload holds no alternatives. An empty payload should not
    /// be stored in `multi_objects`; callers should drop or replace the cell.
    pub fn is_empty(&self) -> bool {
        self.alternatives.is_empty()
    }

    /// Read-only view of the alternatives, in insertion order. Load-time
    /// collapse iterates this list to build the ITE.
    pub fn alternatives(&self) -> &[MultiAlternative] {
        &self.alternatives
    }

    /// Append one alternative to the payload. Invalidates the collapse cache.
    ///
    /// This is the lazy-store primitive: emitting a `Multi` cell from a
    /// symbolic-address store appends the candidate's `(addr == cand, value)`
    /// pair without rebuilding any ITE. The collapse cost is paid at load
    /// time instead.
    pub fn push(&mut self, alt: MultiAlternative) {
        self.alternatives.push(alt);
        self.cached_collapse.get_mut().take();
    }

    /// Right-fold the alternatives into an ITE BV with `default_byte` as the
    /// final `else` leaf. Reuses the memoized BV when the cached default byte
    /// matches; otherwise recomputes and stores the result.
    ///
    /// Returns an 8-bit `RustBV`. The caller must ensure all alternative
    /// `value` BVs are 8 bits (the type invariant).
    pub fn collapse(&self, default_byte: u8, ctx: &SymContext) -> RustBV {
        if let Some(cached) = self.cached_collapse.borrow().as_ref() {
            if cached.default_byte == default_byte {
                return cached.bv.clone();
            }
        }
        let default = RustBV::concrete(default_byte as u128, 8);
        let mut acc = default;
        for alt in self.alternatives.iter().rev() {
            acc = alt.cond.ite(&alt.value, &acc, ctx);
        }
        *self.cached_collapse.borrow_mut() = Some(CachedCollapse {
            default_byte,
            bv: acc.clone(),
        });
        acc
    }

    /// True if the collapse cache currently holds a value. Test-only helper.
    #[cfg(test)]
    pub(crate) fn has_cached_collapse(&self) -> bool {
        self.cached_collapse.borrow().is_some()
    }
}

impl SymbolicMemory {
    /// Install lazy alternatives at a single byte address.
    ///
    /// Marks the byte's page as Multi (via `MemoryPage::mark_multi`), clears
    /// any conflicting plain-Symbolic state at the same address, and stores
    /// the payload in `multi_objects`. Auto-maps the containing page if it
    /// is not yet present, matching the behavior of `import_symbolic_value`
    /// so callers do not have to pre-map stack regions.
    ///
    /// Per memory `invariant-mem-ite-depth-counter`, this records the
    /// alternative count via `crate::symbolic::record_mem_ite_depth` so the
    /// Phase 0 baseline comparison reflects every Multi insertion.
    ///
    /// A payload with zero alternatives clears the cell instead of
    /// installing an empty entry.
    pub fn set_multi_alternatives(&mut self, addr: u64, payload: MultiPayload) {
        if payload.is_empty() {
            self.clear_multi_at(addr);
            return;
        }

        let depth = payload.len() as u32;

        // Auto-map the page if missing. Matches import_symbolic_value's
        // policy so callers (test rigs, future SimProcedure wiring) do not
        // need to pre-map stack regions.
        let page_num = addr >> 12;
        let offset = (addr & PAGE_MASK) as u16;
        let page_addr = page_num << 12;
        let page = self
            .pages
            .entry(page_num)
            .or_insert_with(|| MemoryPage::new(page_addr, Permission::RW));

        // Clear any prior plain-Symbolic state at this byte: a Multi cell
        // supersedes single-symbolic. The page bitmap and symbolic_objects
        // sidecar must stay in sync.
        page.clear_multi(offset); // no-op if not currently Multi
        page.mark_multi(offset);

        self.symbolic_objects.remove(&addr);
        self.symbolic_spans.remove(&addr);

        self.multi_objects.insert(addr, payload);
        // Phase 4.1: bump per-byte version so the wider-load cache notices
        // this installation. Must run regardless of whether a prior
        // payload existed at this address.
        self.bump_multi_version(addr);
        // Counter contract: callers can't bypass this — this is the only
        // public path that installs a Multi cell.
        record_mem_ite_depth(depth);
    }

    /// Read-only access to the lazy alternatives at a byte address, if any.
    pub fn get_multi_alternatives(&self, addr: u64) -> Option<&MultiPayload> {
        self.multi_objects.get(&addr)
    }

    /// Remove lazy alternatives at a byte address and clear the page bit.
    /// Safe to call on a byte that is not currently Multi (no-op).
    pub fn clear_multi_at(&mut self, addr: u64) {
        let had_payload = self.multi_objects.remove(&addr).is_some();
        let page_num = addr >> 12;
        let offset = (addr & PAGE_MASK) as u16;
        if let Some(page) = self.pages.get_mut(&page_num) {
            page.clear_multi(offset);
        }
        // Phase 4.1: bump version only when a payload was actually present
        // so the no-op case stays free.
        if had_payload {
            self.bump_multi_version(addr);
        }
    }

    /// Count of byte addresses currently carrying lazy Multi alternatives.
    /// Used by tests and by future profiling to track the Phase 1 footprint.
    pub fn multi_cell_count(&self) -> usize {
        self.multi_objects.len()
    }

    /// Materialize every Multi cell into a per-byte symbolic_object so the
    /// state export pipeline (`_sync_rust_symbolic_objects_to_state` in
    /// `rust_state_export.py`) sees the lazy alternatives.
    ///
    /// Phase 2 (angr-qh5u): flips the default symbolic-address store to
    /// install Multi cells. Multi bytes are tracked in a parallel
    /// `multi_bitmap`, not `symbolic_bitmap`, so `symbolic_offsets()`
    /// does not list them. Without this flush the page exporter writes
    /// the underlying concrete `data[]` byte (usually zero), losing the
    /// alternative values entirely.
    ///
    /// Per byte:
    ///   1. Right-fold alternatives into an ITE chain with the page's
    ///      current concrete byte as the default else. Same construction
    ///      as `assemble_load_with_multi` for a 1-byte load.
    ///   2. Insert the ITE BV at `symbolic_objects[byte_addr]` (width 8).
    ///   3. Mark the byte symbolic on the page; drop the Multi marker.
    ///   4. Remove the `multi_objects` entry.
    /// The result is byte-equivalent to an eager store: every later load
    /// of that byte will see the same ITE result through the regular
    /// symbolic_objects path.
    pub fn flush_multi_cells(&mut self, ctx: &crate::symbolic::SymContext) {
        if self.multi_objects.is_empty() {
            return;
        }

        let multi = std::mem::take(&mut self.multi_objects);
        for (byte_addr, payload) in multi {
            // Phase 4.1: this byte is leaving Multi state — bump its
            // version so any cached wider-load entry whose fingerprint
            // snapshotted Multi at this address is invalidated.
            self.bump_multi_version(byte_addr);
            let page_num = byte_addr >> 12;
            let offset = (byte_addr & PAGE_MASK) as u16;

            let concrete_byte: u8 = match self.pages.get(&page_num) {
                Some(page) => page.load_concrete(offset, 1).first().copied().unwrap_or(0),
                None => {
                    // Page was unmapped after the Multi cell was installed.
                    // Drop the cell — there is no byte to merge it with and
                    // no page to mark symbolic. Matches the eager path's
                    // behavior when a candidate page becomes unmapped
                    // mid-execution.
                    continue;
                }
            };

            // Phase 3 collapse cache: identical right-fold shape as the
            // load path, so a load that already populated the cache pays
            // zero extra Z3 work here.
            let acc = payload.collapse(concrete_byte, ctx);

            // Update the page bitmap: clear Multi, set Symbolic.
            // Keep a single mutable borrow for both updates.
            if let Some(page) = self.pages.get_mut(&page_num) {
                page.clear_multi(offset);
                page.mark_symbolic(offset, 1);
            }
            self.symbolic_objects.insert(byte_addr, acc);
            self.dirty_pages.insert(page_num);
        }
    }
}
