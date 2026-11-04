//! Unit tests for `SymbolicMemory::get_region_prefetch_list` — the eager-region
//! path that is off under the production `ExecutionConfig::default()`, so no
//! integration test reaches it (angr-szg45.4).

use super::super::*;

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
