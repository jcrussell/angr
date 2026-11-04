//! Unit tests for `SymbolicMemory::merge`'s generic per-byte and structural
//! branches, which no integration test exercises (angr-szg45.4).
//!
//! Distinct from the other `merge_*` files in this directory, which each cover
//! one specialized aspect: `merge_divergence` the CoW/divergence-proportional
//! walk, `merge_multi` the Multi-on-both-arms lazy cell, `merge_sidecars` the
//! per-field `#[merge_policy]` contract, `merge_cost_shape` the cost-shape
//! spike measurement. This file covers the plain cases those assume:
//! per-byte ITE selection, the guarded one-arm page walk, and pending-write
//! extension.

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

/// Evaluate `v` with the merge condition pinned to `arm` (1 = `other`).
fn eval_on_arm(cond: &RustBV, arm: u128, v: &RustBV) -> Option<u64> {
    let ctx = SymContext::new();
    ctx.assume_true(&cond.eq(&RustBV::concrete(arm, 1)));
    ctx.eval(v).map(|x| x as u64)
}

/// A page present only in `other` stays mapped in the merged memory, but its
/// bytes are guarded: `self` never materialized it, so on `self`'s paths it
/// reads as the zero page (angr-vmi2l.187).
#[test]
fn test_merge_other_only_page_is_guarded() {
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
        "an other-only page must count as a merge"
    );
    assert!(
        self_mem.pages.contains_key(&(0x2000 >> 12)),
        "page only in other must stay mapped in self.pages"
    );
    let v = self_mem.load_concrete(0x2000, 1).unwrap();
    assert_eq!(eval_on_arm(&cond, 1, &v), Some(0x42), "other's arm keeps its byte");
    assert_eq!(eval_on_arm(&cond, 0, &v), Some(0), "self's arm reads the zero page");
    let untouched = self_mem.load_concrete(0x2001, 1).unwrap();
    assert_eq!(untouched.as_u64(), Some(0), "a byte neither arm wrote stays concrete");
}

/// A page present only in `self` is guarded the same way: on `other`'s paths
/// it reads as the zero page, not as `self`'s write (angr-vmi2l.187).
#[test]
fn test_merge_self_only_page_is_guarded() {
    let mut self_mem = SymbolicMemory::new(Endness::Little);
    self_mem.map(0x2000, 0x1000, Permission::RWX);
    let sym = RustBV::symbol("w", 16);
    self_mem.store_concrete(0x2010, sym.clone()).unwrap();

    let other_mem = SymbolicMemory::new(Endness::Little);

    let cond = RustBV::symbol("c", 1);
    assert!(self_mem.merge(&other_mem, &cond));
    let v = self_mem.load_concrete(0x2010, 2).unwrap();
    assert_eq!(eval_on_arm(&cond, 1, &v), Some(0), "other's arm reads the zero page");
    let ctx = SymContext::new();
    ctx.assume_true(&cond.eq(&RustBV::concrete(0, 1)));
    ctx.assume_true(&sym.eq(&RustBV::concrete(0xBEEF, 16)));
    assert_eq!(ctx.eval(&v), Some(0xBEEF), "self's arm keeps its write");
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
