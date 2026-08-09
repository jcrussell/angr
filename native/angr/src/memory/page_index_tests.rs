//! Unit tests for [`PageIndex`] (angr-91vj9.8).

use super::*;

#[test]
fn shift_agrees_with_page_size() {
    assert_eq!(PageIndex::SHIFT, 12);
    assert_eq!(1u64 << PageIndex::SHIFT, PAGE_SIZE);
}

#[test]
fn of_drops_the_in_page_offset() {
    assert_eq!(PageIndex::of(0).get(), 0);
    assert_eq!(PageIndex::of(PAGE_MASK).get(), 0);
    assert_eq!(PageIndex::of(PAGE_SIZE).get(), 1);
    assert_eq!(PageIndex::of(0x4001).get(), 4);
}

#[test]
fn base_addr_is_the_inverse_of_of() {
    for addr in [0u64, 0x1, 0xfff, 0x1000, 0xdead_beef] {
        let page = PageIndex::of(addr);
        assert_eq!(page.base_addr(), addr & !PAGE_MASK);
        assert_eq!(PageIndex::of(page.base_addr()), page);
    }
    assert_eq!(PageIndex::from_raw(7).get(), 7);
    assert_eq!(PageIndex::from_raw(7).base_addr(), 7 * PAGE_SIZE);
}

#[test]
fn range_covering_is_inclusive_of_the_last_byte_page() {
    let pages: Vec<u64> = PageIndex::range_covering(0x1000, 1).map(|p| p.get()).collect();
    assert_eq!(pages, vec![1]);

    // Exactly one full page, aligned — must not spill into the next page.
    let pages: Vec<u64> = PageIndex::range_covering(0x1000, PAGE_SIZE)
        .map(|p| p.get())
        .collect();
    assert_eq!(pages, vec![1]);

    // One byte past the page end pulls in the next page.
    let pages: Vec<u64> = PageIndex::range_covering(0x1000, PAGE_SIZE + 1)
        .map(|p| p.get())
        .collect();
    assert_eq!(pages, vec![1, 2]);

    // Unaligned start straddling a boundary.
    let pages: Vec<u64> = PageIndex::range_covering(0xfff, 2).map(|p| p.get()).collect();
    assert_eq!(pages, vec![0, 1]);
}

#[test]
fn range_covering_of_zero_length_is_empty() {
    // The open-coded `first..=last` form the call sites used yielded one page
    // here; a zero-length access touches none.
    assert_eq!(PageIndex::range_covering(0x1000, 0).count(), 0);
}

#[test]
fn range_covering_saturates_instead_of_wrapping() {
    // Without the saturating add, `addr + len - 1` wraps and `first..=last`
    // becomes an empty range — silently skipping the pages it should cover.
    let pages: Vec<u64> = PageIndex::range_covering(u64::MAX - 1, 16)
        .map(|p| p.get())
        .collect();
    assert_eq!(pages, vec![u64::MAX >> PageIndex::SHIFT]);
}
