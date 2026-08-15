//! Unit tests for the wrapping-distance containment helpers on [`Address`]
//! (angr-xloth.4) — see `invariant-address-containment-wrapping-distance`.

use super::*;

#[test]
fn offset_in_matches_the_naive_form_for_a_normal_region() {
    let base = Address::new(0x1000);
    assert_eq!(Address::new(0x1000).offset_in(base, 8), Some(0));
    assert_eq!(Address::new(0x1007).offset_in(base, 8), Some(7));
    assert_eq!(Address::new(0x1008).offset_in(base, 8), None);
    assert_eq!(Address::new(0xfff).offset_in(base, 8), None);
    // An empty region contains nothing, not even its own base.
    assert_eq!(base.offset_in(base, 0), None);
}

#[test]
fn offset_in_still_works_for_a_region_abutting_the_top_of_the_address_space() {
    // `base + region_size == 2^64`: the naive `addr < base + size` wraps to
    // `addr < 0` and reports every in-region address as outside.
    let base = Address::new(u64::MAX - 7);
    assert_eq!(base.offset_in(base, 8), Some(0));
    assert_eq!(Address::new(u64::MAX).offset_in(base, 8), Some(7));
    // Wrapping past the top is outside, not inside.
    assert_eq!(Address::new(0).offset_in(base, 8), None);
    assert_eq!(Address::new(base.raw() - 1).offset_in(base, 8), None);
}

#[test]
fn range_in_requires_the_whole_access_to_fit() {
    let base = Address::new(0x2000);
    assert!(Address::new(0x2000).range_in(4, base, 8));
    assert!(Address::new(0x2004).range_in(4, base, 8));
    assert!(!Address::new(0x2005).range_in(4, base, 8));
    assert!(!Address::new(0x2008).range_in(4, base, 8));
    // A zero-size access is contained iff its start is.
    assert!(Address::new(0x2007).range_in(0, base, 8));
    assert!(!Address::new(0x2008).range_in(0, base, 8));
}

#[test]
fn range_in_survives_a_region_at_the_top_of_the_address_space() {
    let base = Address::new(u64::MAX - 7);
    assert!(base.range_in(8, base, 8));
    assert!(Address::new(u64::MAX - 3).range_in(4, base, 8));
    // Runs off the end of the region (and of the address space).
    assert!(!Address::new(u64::MAX - 3).range_in(5, base, 8));
    assert!(!Address::new(u64::MAX).range_in(2, base, 8));
}
