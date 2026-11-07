//! Unit tests for `PageBitmap`'s range helpers through `MemoryPage`
//! (angr-vmi2l.192).

use super::*;

/// `clamped_end` documents an `offset` past the page end as an empty range
/// for the bitmap helpers. `clear_range` used to hand that inverted range to
/// `FixedBitSet::remove_range`, which panics with "invalid range".
#[test]
fn clear_multi_past_the_page_end_is_a_no_op() {
    let mut page = MemoryPage::new(0x1000, Permission::RW);
    page.mark_multi(0);
    page.clear_multi(PAGE_SIZE as u16 + 1);
    assert!(page.is_multi(0));
    page.clear_multi(0);
    assert!(!page.has_multi());
}
