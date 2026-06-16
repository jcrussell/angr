// Unit tests for segmentlist.rs (SegmentList / SegmentListIter).
// Extracted from the inline `mod tests` block; see rust-mod-tests-sibling-extraction.

use super::{SegmentList, SegmentListIter};

#[test]
fn empty_list() {
    let mut sl = SegmentList::new();
    assert_eq!(sl.occupied_size(), 0);
    sl.release(0, 10);
    assert_eq!(sl.occupied_size(), 0);
}

#[test]
fn single_range() {
    let mut sl = SegmentList::new();
    sl.occupy(10, 5, None);
    assert_eq!(sl.occupied_size(), 5);
    assert!(sl.is_occupied(10));
    assert!(!sl.is_occupied(9));
}

#[test]
fn multi_non_overlapping() {
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, None);
    sl.occupy(20, 5, Some("X".to_string()));
    assert_eq!(sl.occupied_size(), 15);
    sl.release(100, 5);
    assert_eq!(sl.occupied_size(), 15);
}

#[test]
fn overlapping_inserts() {
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, None);
    sl.occupy(5, 10, None);
    assert_eq!(sl.occupied_size(), 15);
}

#[test]
fn iter_snapshot_yields_in_order() {
    let mut sl = SegmentList::new();
    sl.occupy(20, 5, Some("B".to_string()));
    sl.occupy(0, 10, Some("A".to_string()));
    sl.occupy(40, 3, None);

    let mut it = SegmentListIter::snapshot(&sl);
    let collected: Vec<_> = std::iter::from_fn(|| {
        it.segments.get(it.idx).cloned().inspect(|_seg| {
            it.idx += 1;
        })
    })
    .collect();
    assert_eq!(
        collected,
        vec![
            (0, 10, Some("A".to_string())),
            (20, 25, Some("B".to_string())),
            (40, 43, None),
        ]
    );
}

#[test]
fn full_and_partial_release() {
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, None);
    // partial release [3..8)
    sl.release(3, 5);
    assert_eq!(sl.occupied_size(), 5);
    assert!(sl.is_occupied(2));
    assert!(!sl.is_occupied(4));
    assert!(sl.is_occupied(8));
    // full release
    sl.release(0, 10);
    assert_eq!(sl.occupied_size(), 0);
    assert!(sl.is_empty());
}
