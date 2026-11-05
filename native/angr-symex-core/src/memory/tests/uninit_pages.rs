//! Never-written bytes on a page mapped with no host bytes (angr-vmi2l.325).
//!
//! `auto_map_zero_page` marks every byte of such a page uninitialized unless
//! ZERO_FILL_UNCONSTRAINED_MEMORY is on; any write clears its bytes, and
//! `uninit_runs` reports the gaps a load must fill.

use super::super::*;

const BASE: u64 = 0x3000_0000;

fn fresh(zero_fill: bool) -> SymbolicMemory {
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.set_zero_fill_unconstrained(zero_fill);
    mem.add_lazy_region(BASE, 0x1000);
    assert!(mem.auto_map_zero_page(BASE));
    mem
}

#[test]
fn store_leaves_gaps_on_either_side_uninit() {
    let mut mem = fresh(false);
    mem.store_concrete(BASE + 2, RustBV::concrete(0x41, 8))
        .expect("store");
    assert_eq!(mem.uninit_runs(BASE, 8), vec![(BASE, 2), (BASE + 3, 5)]);
}

#[test]
fn symbolic_store_clears_uninit() {
    let mut mem = fresh(false);
    mem.store_concrete(BASE, RustBV::symbol("uninit_sym", 32))
        .expect("store");
    assert_eq!(mem.uninit_runs(BASE, 6), vec![(BASE + 4, 2)]);
}

#[test]
fn zero_fill_page_has_no_uninit_bytes() {
    let mem = fresh(true);
    assert!(mem.uninit_runs(BASE, 8).is_empty());
}

#[test]
fn unmapped_bytes_are_not_uninit() {
    let mem = SymbolicMemory::new(Endness::Little);
    assert!(mem.uninit_runs(BASE, 8).is_empty());
}

/// A byte one arm wrote (even with 0) is initialized after the merge; one
/// neither arm wrote stays uninitialized.
#[test]
fn merge_keeps_uninit_only_where_both_arms_have_it() {
    let mut a = fresh(false);
    let mut b = a.clone();
    b.store_concrete(BASE, RustBV::concrete(0, 8)).expect("store");
    let cond = RustBV::symbol("uninit_merge_cond", 1);
    a.merge(&b, &cond);
    assert_eq!(a.uninit_runs(BASE, 4), vec![(BASE + 1, 3)]);
}

/// An arm that never mapped the page reads it as uninitialized, so the
/// other arm's uninit bytes stay so in the merge.
#[test]
fn merge_one_arm_page_keeps_its_uninit_bytes() {
    let mut a = SymbolicMemory::new(Endness::Little);
    a.add_lazy_region(BASE, 0x1000);
    let mut b = fresh(false);
    b.store_concrete(BASE, RustBV::concrete(0x41, 8))
        .expect("store");
    let cond = RustBV::symbol("uninit_merge_cond1", 1);
    a.merge(&b, &cond);
    assert_eq!(a.uninit_runs(BASE, 4), vec![(BASE + 1, 3)]);
}

#[test]
fn uninit_bitmap_survives_serde() {
    let page = MemoryPage::new_uninit(BASE, Permission::RW);
    let back = MemoryPage::from(crate::memory::page::MemoryPageData::from(page));
    assert!(back.is_uninit(7));
}

/// A page mapped from host bytes can carry gaps the host never wrote: the
/// synced SP stack page (angr-mx00p.132.58). `mark_uninit` reopens them and
/// skips bytes on an unmapped page.
#[test]
fn mark_uninit_reopens_gaps_on_a_mapped_page() {
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map_data(BASE, &[0x41; 0x1000], Permission::RW);
    assert!(mem.uninit_runs(BASE, 0x1000).is_empty());
    mem.mark_uninit(BASE + 0xFF8, 0x10);
    assert_eq!(mem.uninit_runs(BASE + 0xFF0, 0x20), vec![(BASE + 0xFF8, 8)]);
}
