//! Unit tests for soundness-critical memory paths that no integration test
//! exercises (angr-szg45.4):
//!
//! - `SymbolicMemory::merge`: per-byte ITE selection, page-only-in-other
//!   adoption, non-page symbolic-object union, and pending-write extension.
//! - `SymbolicMemory::load_concrete_or_unconstrained`: the Err fallback that
//!   only fires for unmapped addresses (the ITE build path always hits Ok).
//! - `SymbolicMemory::get_region_prefetch_list`: the eager-region path that is
//!   off under the production `ExecutionConfig::default()`.

use super::super::*;

/// Extract the variable name from a `Symbolic` BV, panicking otherwise.
fn sym_name(bv: &RustBV) -> String {
    match bv {
        RustBV::Symbolic { name, .. } => name.to_string(),
        other => panic!("expected Symbolic BV, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// merge: per-byte ITE selection (needs a real solver to evaluate the ITE)
// ---------------------------------------------------------------------------

/// Two memories share a page but differ in one concrete byte. After merge the
/// byte must be `ITE(cond, other, self)`: evaluating under cond=true yields the
/// `other` byte, under cond=false the `self` byte.
#[cfg(feature = "vex-engine-z3")]
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

    let cond = RustBV::symbolic(&ctx, "merge_cond", 1);
    assert!(
        self_mem.merge(&other_mem, &cond, &ctx),
        "merge must report a change when a byte differs"
    );

    let byte = self_mem
        .symbolic_objects
        .get(&Address(0x1000))
        .expect("merged byte must be a symbolic ITE object")
        .clone();

    let t = ctx.fork();
    t.assume_true(&cond.eq(&RustBV::concrete(1, 1), &t));
    assert_eq!(
        t.eval(&byte),
        Some(0xBB),
        "cond=true must select the `other` byte"
    );

    let f = ctx.fork();
    f.assume_true(&cond.eq(&RustBV::concrete(0, 1), &f));
    assert_eq!(
        f.eval(&byte),
        Some(0xAA),
        "cond=false must select the `self` byte"
    );
}

// ---------------------------------------------------------------------------
// merge: structural branches (no ITE eval, mock context suffices)
// ---------------------------------------------------------------------------

/// A page present only in `other` is adopted wholesale into `self.pages`,
/// preserving its concrete contents.
#[test]
fn test_merge_adopts_page_only_in_other() {
    let ctx = SymContext::new_mock();

    let mut self_mem = SymbolicMemory::new(Endness::Little);
    self_mem.map(0x1000, 0x1000, Permission::RWX);

    let mut other_mem = SymbolicMemory::new(Endness::Little);
    other_mem.map(0x2000, 0x1000, Permission::RWX);
    other_mem
        .store_concrete(0x2000, RustBV::concrete(0x42, 8))
        .unwrap();

    let cond = RustBV::symbolic(&ctx, "c", 1);
    assert!(
        self_mem.merge(&other_mem, &cond, &ctx),
        "adopting a page must count as a merge"
    );
    assert!(
        self_mem.pages.contains_key(&(0x2000 >> 12)),
        "page only in other must be adopted into self.pages"
    );
    let v = self_mem.load_concrete(0x2000, 1, &ctx).unwrap();
    assert_eq!(v.as_u64(), Some(0x42), "adopted page bytes must survive");
}

/// A symbolic object living at an address whose page is concrete-identical in
/// both memories (so the per-byte loop short-circuits) is still unioned in via
/// the final non-page symbolic-object pass, populating both
/// `symbolic_objects` and `symbolic_spans`.
#[test]
fn test_merge_unions_non_page_symbolic_object_from_other() {
    let ctx = SymContext::new_mock();

    let mut self_mem = SymbolicMemory::new(Endness::Little);
    self_mem.map(0x1000, 0x1000, Permission::RWX);

    let mut other_mem = SymbolicMemory::new(Endness::Little);
    other_mem.map(0x1000, 0x1000, Permission::RWX);
    // Insert a symbolic object WITHOUT marking the page symbolic, so
    // op.has_symbolic() stays false and the (s_data==o_data) short-circuit
    // skips the per-byte loop — leaving only the final union pass to copy it.
    let sym = RustBV::symbolic(&ctx, "ghost_obj", 8);
    other_mem.symbolic_objects.insert(Address(0x1800), sym);

    let cond = RustBV::symbolic(&ctx, "c", 1);
    assert!(
        self_mem.merge(&other_mem, &cond, &ctx),
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
    let ctx = SymContext::new_mock();

    let mut self_mem = SymbolicMemory::new(Endness::Little);
    self_mem.map(0x1000, 0x1000, Permission::RWX);

    let mut other_mem = SymbolicMemory::new(Endness::Little);
    other_mem.map(0x1000, 0x1000, Permission::RWX);
    other_mem.add_pending_write(PendingWrite {
        addr: RustBV::symbolic(&ctx, "pw_addr", 64),
        value: RustBV::concrete(0x99, 8),
        size: 1,
        condition: None,
        page_hint: None,
    });

    assert_eq!(self_mem.pending_writes_count(), 0);
    let cond = RustBV::symbolic(&ctx, "c", 1);
    assert!(
        self_mem.merge(&other_mem, &cond, &ctx),
        "extending pending writes must count as a merge"
    );
    assert_eq!(
        self_mem.pending_writes_count(),
        1,
        "self.pending_writes must be extended by other's"
    );
}

// ---------------------------------------------------------------------------
// load_concrete_or_unconstrained: the unmapped-address fallback
// ---------------------------------------------------------------------------

/// With `zero_fill_unconstrained=true`, an unmapped load returns concrete 0 of
/// width `size*8` and does NOT advance the name counter.
#[test]
fn test_load_concrete_or_unconstrained_zero_fill() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.set_zero_fill_unconstrained(true);

    let mut counter = 0u64;
    let v = mem.load_concrete_or_unconstrained(0x5000u64, 4, &ctx, &mut counter);
    assert_eq!(v.width(), 32, "width must be size*8");
    assert_eq!(v.as_u64(), Some(0), "zero-fill must produce concrete 0");
    assert_eq!(counter, 0, "zero-fill path must not advance the counter");
}

/// With `zero_fill_unconstrained=false` (the default), an unmapped load returns
/// a fresh `unc_mem_`-prefixed symbolic of width `size*8`, and successive calls
/// produce distinct names (counter advances) — guaranteeing distinct constraint
/// identity across unmapped leaves.
#[test]
fn test_load_concrete_or_unconstrained_symbolic_distinct_names() {
    let ctx = SymContext::new_mock();
    let mem = SymbolicMemory::new(Endness::Little);
    assert!(!mem.zero_fill_unconstrained(), "default must be false");

    let mut counter = 0u64;
    let v1 = mem.load_concrete_or_unconstrained(0x5000u64, 4, &ctx, &mut counter);
    let v2 = mem.load_concrete_or_unconstrained(0x5000u64, 4, &ctx, &mut counter);

    assert_eq!(v1.width(), 32, "non-byte size=4 must give a 32-bit value");
    let n1 = sym_name(&v1);
    let n2 = sym_name(&v2);
    assert!(
        n1.starts_with("unc_mem_"),
        "name must be unc_mem_-prefixed: {n1}"
    );
    assert!(
        n2.starts_with("unc_mem_"),
        "name must be unc_mem_-prefixed: {n2}"
    );
    assert_ne!(
        n1, n2,
        "successive unmapped loads must produce distinct names"
    );
    assert_eq!(
        counter, 2,
        "counter must advance once per symbolic fallback"
    );
}

// ---------------------------------------------------------------------------
// get_region_prefetch_list: the eager-region path (off in production config)
// ---------------------------------------------------------------------------

/// Returns exactly the unmapped page addresses in a lazy region, in order,
/// shifted `<<12`. A mapped subset is excluded.
#[test]
fn test_get_region_prefetch_list_unmapped_in_order() {
    let mut mem = SymbolicMemory::new(Endness::Little);
    // Region spans pages 0x10..0x14 (4 pages).
    mem.add_lazy_region(0x10000u64, 0x4000);
    // Map a strict subset: page 0x11.
    mem.map(0x11000, 0x1000, Permission::RWX);

    let list = mem
        .get_region_prefetch_list(0x10000u64, 100)
        .expect("trigger inside region must yield Some");
    assert_eq!(
        list,
        vec![0x10000, 0x12000, 0x13000],
        "must list unmapped pages in order, excluding the mapped 0x11000"
    );
}

/// `max_pages` truncates the returned list.
#[test]
fn test_get_region_prefetch_list_truncates_at_max() {
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.add_lazy_region(0x10000u64, 0x4000);
    mem.map(0x11000, 0x1000, Permission::RWX);

    let list = mem
        .get_region_prefetch_list(0x10000u64, 2)
        .expect("trigger inside region must yield Some");
    assert_eq!(
        list,
        vec![0x10000, 0x12000],
        "max_pages=2 must cap the list at the first two unmapped pages"
    );
}

/// When every page in the region is mapped, the empty list collapses to None.
#[test]
fn test_get_region_prefetch_list_none_when_fully_mapped() {
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.add_lazy_region(0x20000u64, 0x2000); // pages 0x20, 0x21
    mem.map(0x20000, 0x1000, Permission::RWX);
    mem.map(0x21000, 0x1000, Permission::RWX);

    assert!(
        mem.get_region_prefetch_list(0x20000u64, 100).is_none(),
        "fully-mapped region must yield None"
    );
}

/// A trigger page outside any lazy region yields None.
#[test]
fn test_get_region_prefetch_list_none_outside_region() {
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.add_lazy_region(0x30000u64, 0x1000);

    assert!(
        mem.get_region_prefetch_list(0x99000u64, 100).is_none(),
        "trigger outside every lazy region must yield None"
    );
}
