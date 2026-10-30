//! Unit tests for `SymbolicMemory::merge`'s generic per-byte and structural
//! branches, which no integration test exercises (angr-szg45.4).
//!
//! Distinct from the other `merge_*` files in this directory, which each cover
//! one specialized aspect: `merge_divergence` the CoW/divergence-proportional
//! walk, `merge_multi` the Multi-on-both-arms lazy cell, `merge_sidecars` the
//! per-field `#[merge_policy]` contract, `merge_cost_shape` the cost-shape
//! spike measurement. This file covers the plain cases those assume:
//! per-byte ITE selection, page-only-in-other adoption, non-page
//! symbolic-object union, and pending-write extension.

use super::super::*;

// ---------------------------------------------------------------------------
// merge: per-byte ITE selection (needs a real solver to evaluate the ITE)
// ---------------------------------------------------------------------------

/// Two memories share a page but differ in one concrete byte. After merge the
/// byte must be `ITE(cond, other, self)`: evaluating under cond=true yields the
/// `other` byte, under cond=false the `self` byte.
#[test]
fn test_merge_selects_other_vs_self_per_byte() {
    let ctx = SymContext::new();

    let mut self_mem = SymbolicMemory::new(Endness::Little);
    self_mem.map(0x1000, 0x1000, Permission::RWX);
    self_mem
        .store_concrete(0x1000, RustBV::concrete(0xAA, 8))
        .unwrap();

    let mut other_mem = SymbolicMemory::new(Endness::Little);
    other_mem.map(0x1000, 0x1000, Permission::RWX);
    other_mem
        .store_concrete(0x1000, RustBV::concrete(0xBB, 8))
        .unwrap();

    let cond = RustBV::symbol("merge_cond", 1);
    assert!(
        self_mem.merge(&other_mem, &cond),
        "merge must report a change when a byte differs"
    );

    let byte = self_mem
        .symbolic_objects
        .get(&Address(0x1000))
        .expect("merged byte must be a symbolic ITE object")
        .clone();

    let t = ctx.fork();
    t.assume_true(&cond.eq(&RustBV::concrete(1, 1)));
    assert_eq!(
        t.eval(&byte),
        Some(0xBB),
        "cond=true must select the `other` byte"
    );

    let f = ctx.fork();
    f.assume_true(&cond.eq(&RustBV::concrete(0, 1)));
    assert_eq!(
        f.eval(&byte),
        Some(0xAA),
        "cond=false must select the `self` byte"
    );
}

// ---------------------------------------------------------------------------
// merge: structural branches (no ITE eval)
// ---------------------------------------------------------------------------

/// A page present only in `other` is adopted wholesale into `self.pages`,
/// preserving its concrete contents.
#[test]
fn test_merge_adopts_page_only_in_other() {

    let mut self_mem = SymbolicMemory::new(Endness::Little);
    self_mem.map(0x1000, 0x1000, Permission::RWX);

    let mut other_mem = SymbolicMemory::new(Endness::Little);
    other_mem.map(0x2000, 0x1000, Permission::RWX);
    other_mem
        .store_concrete(0x2000, RustBV::concrete(0x42, 8))
        .unwrap();

    let cond = RustBV::symbol("c", 1);
    assert!(
        self_mem.merge(&other_mem, &cond),
        "adopting a page must count as a merge"
    );
    assert!(
        self_mem.pages.contains_key(&(0x2000 >> 12)),
        "page only in other must be adopted into self.pages"
    );
    let v = self_mem.load_concrete(0x2000, 1).unwrap();
    assert_eq!(v.as_u64(), Some(0x42), "adopted page bytes must survive");
}

/// A symbolic object living at an address whose page is concrete-identical in
/// both memories (so the per-byte loop short-circuits) is still unioned in via
/// the final non-page symbolic-object pass, populating both
/// `symbolic_objects` and `symbolic_spans`.
#[test]
fn test_merge_unions_non_page_symbolic_object_from_other() {

    let mut self_mem = SymbolicMemory::new(Endness::Little);
    self_mem.map(0x1000, 0x1000, Permission::RWX);

    let mut other_mem = SymbolicMemory::new(Endness::Little);
    other_mem.map(0x1000, 0x1000, Permission::RWX);
    // Insert a symbolic object WITHOUT marking the page symbolic, so
    // op.has_symbolic() stays false and the (s_data==o_data) short-circuit
    // skips the per-byte loop — leaving only the final union pass to copy it.
    let sym = RustBV::symbol("ghost_obj", 8);
    other_mem.symbolic_objects.insert(Address(0x1800), sym);

    let cond = RustBV::symbol("c", 1);
    assert!(
        self_mem.merge(&other_mem, &cond),
        "unioning a symbolic object must count as a merge"
    );
    assert!(
        self_mem.symbolic_objects.contains_key(&Address(0x1800)),
        "non-page symbolic object must be unioned into self.symbolic_objects"
    );
    let span = self_mem
        .symbolic_spans
        .get(&Address(0x1800))
        .expect("symbolic_spans must gain the unioned object");
    assert_eq!(
        *span,
        (Address(0x1800), 8),
        "span must record (addr, width)"
    );
}

/// Pending writes present only in `other` extend `self.pending_writes`.
#[test]
fn test_merge_extends_pending_writes_from_other() {

    let mut self_mem = SymbolicMemory::new(Endness::Little);
    self_mem.map(0x1000, 0x1000, Permission::RWX);

    let mut other_mem = SymbolicMemory::new(Endness::Little);
    other_mem.map(0x1000, 0x1000, Permission::RWX);
    other_mem.add_pending_write(PendingWrite {
        addr: RustBV::symbol("pw_addr", 64),
        value: RustBV::concrete(0x99, 8),
        size: 1,
        condition: None,
    });

    assert_eq!(self_mem.pending_writes_count(), 0);
    let cond = RustBV::symbol("c", 1);
    assert!(
        self_mem.merge(&other_mem, &cond),
        "extending pending writes must count as a merge"
    );
    assert_eq!(
        self_mem.pending_writes_count(),
        1,
        "self.pending_writes must be extended by other's"
    );
}
