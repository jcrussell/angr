// Unit tests for segmentlist.rs (SegmentList / SegmentListIter).
// Extracted from the inline `mod tests` block; see rust-mod-tests-sibling-extraction.

use std::collections::HashSet;

use super::{Segment, SegmentList, SegmentListIter};

/// Drains an iterator without a Python interpreter: `__next__` hands back
/// `Segment` pyobjects, which these tests have no GIL to build.
fn drain(mut it: SegmentListIter) -> Vec<(u64, u64, Option<String>)> {
    std::iter::from_fn(|| {
        it.segments.get(it.idx).cloned().inspect(|_seg| {
            it.idx += 1;
        })
    })
    .collect()
}

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
fn search_half_open_boundary() {
    // Regression for the off-by-one at a segment boundary (angr-zi35f.1).
    // Segments are half-open [start, end); an address exactly at a segment's
    // end boundary belongs to the NEXT segment, not the current one.
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, Some("code".to_string())); // [0, 10)
    sl.occupy(10, 10, Some("nodecode".to_string())); // [10, 20)
    assert_eq!(sl.search(0), Some(0)); // inside first
    assert_eq!(sl.search(9), Some(0)); // last byte of first
    assert_eq!(sl.search(10), Some(1)); // boundary belongs to second
    assert_eq!(sl.search(19), Some(1)); // last byte of second
    assert_eq!(sl.search(20), None); // past everything
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

    assert_eq!(
        drain(SegmentListIter::snapshot(&sl)),
        vec![
            (0, 10, Some("A".to_string())),
            (20, 25, Some("B".to_string())),
            (40, 43, None),
        ]
    );
}

// `iter_backward_from` must yield exactly what `search` + descending
// `__getitem__` did, only without the per-step O(n) re-walk (angr-9ke6b.200).
#[test]
fn iter_backward_from_matches_search_plus_indexing() {
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, Some("A".to_string()));
    sl.occupy(20, 5, Some("B".to_string()));
    sl.occupy(40, 3, None);

    for addr in 0..50u64 {
        let expected: Vec<_> = match sl.search(addr) {
            None => vec![],
            Some(idx) => (0..=idx)
                .rev()
                .map(|i| {
                    let seg = sl.__getitem__(i).expect("index from search is in range");
                    (seg.start, seg.end, seg.sort.clone())
                })
                .collect(),
        };
        assert_eq!(drain(sl.iter_backward_from(addr)), expected, "addr {addr}");
    }
}

// A `search` miss (address past the last segment) must yield nothing rather
// than the whole list — the caller reads an empty walk as "no window here".
#[test]
fn iter_backward_from_past_end_is_empty() {
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, Some("A".to_string()));
    sl.occupy(20, 5, Some("B".to_string()));

    assert_eq!(sl.search(25), None);
    assert!(drain(sl.iter_backward_from(25)).is_empty());
    assert!(drain(sl.iter_backward_from(u64::MAX)).is_empty());
    assert!(drain(SegmentList::new().iter_backward_from(0)).is_empty());
}

// The first item is the segment `addr` lands in even when `addr` sits in a gap
// before it, matching `search`'s "may not actually belong to the block" note.
#[test]
fn iter_backward_from_addr_in_a_gap() {
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, Some("A".to_string()));
    sl.occupy(20, 5, Some("B".to_string()));

    assert_eq!(
        drain(sl.iter_backward_from(15)),
        vec![
            (20, 25, Some("B".to_string())),
            (0, 10, Some("A".to_string()))
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

// `rangemap::RangeMap::{insert,remove}` assert `range.start < range.end`, and this
// crate builds with panic="abort", so an address+size that wraps u64 would take the
// whole process down rather than raise. Both entry points must no-op instead.
#[test]
fn overflowing_release_is_a_noop() {
    let mut sl = SegmentList::new();
    sl.occupy(0x1000, 0x100, Some("code".into()));
    assert_eq!(sl.occupied_size(), 0x100);

    // address + size wraps: u64::MAX - 4 + 16
    sl.release(u64::MAX - 4, 16);
    sl.release(u64::MAX, u64::MAX);
    sl.release(1, u64::MAX);

    // The pre-existing segment is untouched.
    assert_eq!(sl.occupied_size(), 0x100);
    assert!(sl.is_occupied(0x1000));
}

#[test]
fn overflowing_occupy_is_a_noop() {
    let mut sl = SegmentList::new();
    sl.occupy(u64::MAX - 4, 16, Some("code".into()));
    sl.occupy(1, u64::MAX, None);
    assert_eq!(sl.occupied_size(), 0);
    assert!(sl.is_empty());
}

// The exact-fit boundary case must still work: address + size == u64::MAX + 1 is an
// overflow, but address + size == u64::MAX is not.
#[test]
fn release_at_top_of_address_space() {
    let mut sl = SegmentList::new();
    sl.occupy(u64::MAX - 16, 16, Some("code".into()));
    assert_eq!(sl.occupied_size(), 16);
    sl.release(u64::MAX - 16, 16);
    assert_eq!(sl.occupied_size(), 0);
    assert!(sl.is_empty());
}

// `Segment::size()` subtracts without a guard and the release profile has
// `overflow-checks` off, so an inverted `[start, end)` pair must be rejected at
// construction rather than wrapping to a near-u64::MAX size (angr-sqfj8.134).
#[test]
fn segment_new_rejects_inverted_range() {
    assert!(Segment::new(0x2000, 0x1000, None).is_err());
    assert!(Segment::new(u64::MAX, 0, Some("code".into())).is_err());
}

#[test]
fn segment_new_accepts_empty_and_forward_ranges() {
    let empty = Segment::new(0x1000, 0x1000, None).expect("start == end is valid");
    assert_eq!(empty.size(), 0);
    let forward = Segment::new(0x1000, 0x1100, Some("code".into())).expect("start < end is valid");
    assert_eq!(forward.size(), 0x100);
}

// --- __setstate__ ----------------------------------------------------------

#[test]
fn setstate_on_a_populated_list_replaces_rather_than_accumulates() {
    let mut saved = SegmentList::new();
    saved.occupy(0x1000, 0x10, Some("code".to_string()));
    let state = saved.__getstate__();

    // The pickle path always restores onto a fresh instance, but `__setstate__`
    // is plain `#[pymethods]` and any Python caller can aim it at a populated
    // list. Both the map and the byte total must be reset (angr-fs8kb.83).
    let mut sl = SegmentList::new();
    sl.occupy(0x8000, 0x200, Some("data".to_string()));
    assert_eq!(sl.occupied_size(), 0x200);

    sl.__setstate__(state);
    assert_eq!(sl.occupied_size(), 0x10);
    assert_eq!(sl.__len__(), 1);
    assert!(!sl.is_occupied(0x8000));
    assert!(sl.is_occupied(0x1000));
}

#[test]
fn setstate_round_trip_preserves_size_and_sorts() {
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, Some("code".to_string()));
    sl.occupy(20, 10, Some("data".to_string()));
    let state = sl.__getstate__();
    assert_eq!(
        state,
        vec![
            (0, 10, Some("code".to_string())),
            (20, 10, Some("data".to_string())),
        ]
    );

    let mut restored = SegmentList::new();
    restored.__setstate__(state);
    assert_eq!(restored.occupied_size(), sl.occupied_size());
    assert_eq!(restored.__len__(), 2);
    assert_eq!(restored.occupied_by_sort(25), Some("data".to_string()));
}

#[test]
fn setstate_with_empty_state_clears_the_list() {
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, None);
    sl.__setstate__(Vec::new());
    assert_eq!(sl.occupied_size(), 0);
    assert!(sl.is_empty());
    assert!(!sl.has_blocks());
}

// --- has_blocks ------------------------------------------------------------

#[test]
fn has_blocks_tracks_map_emptiness_not_byte_count() {
    let mut sl = SegmentList::new();
    assert!(!sl.has_blocks());
    // A zero-size occupy is a no-op, so it must not flip the flag.
    sl.occupy(0x100, 0, Some("code".to_string()));
    assert!(!sl.has_blocks());
    sl.occupy(0x100, 4, Some("code".to_string()));
    assert!(sl.has_blocks());
    sl.release(0x100, 4);
    assert!(!sl.has_blocks());
}

// --- occupied_by / occupied_by_sort ---------------------------------------

#[test]
fn occupied_by_reports_the_whole_containing_segment() {
    let mut sl = SegmentList::new();
    sl.occupy(0x1000, 0x40, Some("code".to_string()));

    // Any address inside the segment reports the same (start, size, sort) —
    // it is the containing segment, not the queried address, that is described.
    for addr in [0x1000, 0x1020, 0x103f] {
        assert_eq!(
            sl.occupied_by(addr),
            Some((0x1000, 0x40, Some("code".to_string()))),
            "addr {addr:#x}"
        );
    }
    // Half-open: `end` itself is not occupied.
    assert_eq!(sl.occupied_by(0x1040), None);
    assert_eq!(sl.occupied_by(0xfff), None);
}

#[test]
fn occupied_by_sort_conflates_unoccupied_with_a_sortless_segment() {
    let mut sl = SegmentList::new();
    sl.occupy(0x1000, 0x10, None);
    // Both answers are `None`: the sort of a sort-less segment and the sort of
    // an address with no segment at all. Callers that need to tell those apart
    // must use `is_occupied`/`occupied_by`.
    assert_eq!(sl.occupied_by_sort(0x1000), None);
    assert_eq!(sl.occupied_by_sort(0x2000), None);
    assert!(sl.is_occupied(0x1000));
    assert!(!sl.is_occupied(0x2000));

    sl.occupy(0x2000, 0x10, Some("data".to_string()));
    assert_eq!(sl.occupied_by_sort(0x2008), Some("data".to_string()));
}

// --- next_free_pos ---------------------------------------------------------

#[test]
fn next_free_pos_skips_past_adjacent_differently_sorted_segments() {
    let mut sl = SegmentList::new();
    // Adjacent but differently sorted, so they stay two map entries; the gap
    // search must walk both rather than stopping at the first boundary.
    sl.occupy(0, 10, Some("code".to_string()));
    sl.occupy(10, 10, Some("data".to_string()));

    assert_eq!(sl.next_free_pos(0), Some(20));
    assert_eq!(sl.next_free_pos(5), Some(20));
    assert_eq!(sl.next_free_pos(19), Some(20));
    // Already free: the query address itself is the answer.
    assert_eq!(sl.next_free_pos(20), Some(20));
    assert_eq!(sl.next_free_pos(0x9000), Some(0x9000));
}

#[test]
fn next_free_pos_on_an_empty_list_is_the_query_address() {
    let sl = SegmentList::new();
    assert_eq!(sl.next_free_pos(0), Some(0));
    assert_eq!(sl.next_free_pos(0xdead_beef), Some(0xdead_beef));
}

#[test]
fn next_free_pos_returns_none_when_everything_above_is_occupied() {
    let mut sl = SegmentList::new();
    // Covers 0..u64::MAX — the whole searched range, leaving no gap.
    sl.occupy(0, u64::MAX, Some("code".to_string()));
    assert_eq!(sl.next_free_pos(0), None);
    assert_eq!(sl.next_free_pos(0x1000), None);
}

// --- next_pos_with_sort_not_in --------------------------------------------

#[test]
fn next_pos_with_sort_not_in_filters_by_sort() {
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, Some("code".to_string()));
    sl.occupy(20, 10, Some("data".to_string()));

    let skip_code = HashSet::from([Some("code".to_string())]);
    assert_eq!(sl.next_pos_with_sort_not_in(0, skip_code.clone(), None), Some(20));

    let skip_both = HashSet::from([Some("code".to_string()), Some("data".to_string())]);
    assert_eq!(sl.next_pos_with_sort_not_in(0, skip_both, None), None);

    // Nothing filtered out: the first occupied position wins.
    assert_eq!(sl.next_pos_with_sort_not_in(0, HashSet::new(), None), Some(0));
}

#[test]
fn next_pos_with_sort_not_in_clamps_the_hit_to_the_query_address() {
    let mut sl = SegmentList::new();
    sl.occupy(20, 10, Some("data".to_string()));
    let skip_code = HashSet::from([Some("code".to_string())]);

    // Query lands inside the matching segment: the answer is the query address,
    // not the segment start.
    assert_eq!(sl.next_pos_with_sort_not_in(25, skip_code.clone(), None), Some(25));
    // Query below it: the segment start.
    assert_eq!(sl.next_pos_with_sort_not_in(5, skip_code.clone(), None), Some(20));
    // Query above it: nothing left.
    assert_eq!(sl.next_pos_with_sort_not_in(30, skip_code, None), None);
}

#[test]
fn next_pos_with_sort_not_in_honors_max_distance() {
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, Some("code".to_string()));
    sl.occupy(20, 10, Some("data".to_string()));
    let skip_code = HashSet::from([Some("code".to_string())]);

    // Window stops short of the "data" segment.
    assert_eq!(sl.next_pos_with_sort_not_in(0, skip_code.clone(), Some(15)), None);
    // Window reaches one byte into it (half-open: 0..21).
    assert_eq!(sl.next_pos_with_sort_not_in(0, skip_code.clone(), Some(21)), Some(20));
    // An overlong distance saturates instead of wrapping back below `address`.
    assert_eq!(
        sl.next_pos_with_sort_not_in(0x10, skip_code, Some(u64::MAX)),
        Some(20)
    );
}

#[test]
fn next_pos_with_sort_not_in_matches_sortless_segments() {
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, None);
    sl.occupy(20, 10, Some("data".to_string()));

    // `None` is a sort like any other and can itself be filtered out.
    assert_eq!(
        sl.next_pos_with_sort_not_in(0, HashSet::from([None]), None),
        Some(20)
    );
    assert_eq!(
        sl.next_pos_with_sort_not_in(0, HashSet::from([Some("data".to_string())]), None),
        Some(0)
    );
}

// --- update ----------------------------------------------------------------

#[test]
fn update_merges_disjoint_segments_and_sums_bytes() {
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, Some("code".to_string()));
    let mut other = SegmentList::new();
    other.occupy(20, 10, Some("data".to_string()));

    sl.update(&other);
    assert_eq!(sl.occupied_size(), 20);
    assert_eq!(sl.__len__(), 2);
    assert_eq!(sl.occupied_by_sort(25), Some("data".to_string()));
    // `other` is untouched.
    assert_eq!(other.occupied_size(), 10);
    assert_eq!(other.__len__(), 1);
}

#[test]
fn update_does_not_double_count_overlapping_bytes() {
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, Some("code".to_string()));
    let mut other = SegmentList::new();
    other.occupy(5, 10, Some("code".to_string()));

    sl.update(&other);
    // 0..15, not 10 + 10.
    assert_eq!(sl.occupied_size(), 15);
    assert_eq!(sl.__len__(), 1);
    assert_eq!(sl.occupied_by(0), Some((0, 15, Some("code".to_string()))));
}

#[test]
fn update_lets_the_other_lists_sort_win_on_overlap() {
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, Some("code".to_string()));
    let mut other = SegmentList::new();
    other.occupy(0, 10, Some("data".to_string()));

    sl.update(&other);
    assert_eq!(sl.occupied_size(), 10);
    assert_eq!(sl.occupied_by_sort(0), Some("data".to_string()));
}

#[test]
fn update_from_an_empty_list_is_a_noop() {
    let mut sl = SegmentList::new();
    sl.occupy(0, 10, Some("code".to_string()));
    sl.update(&SegmentList::new());
    assert_eq!(sl.occupied_size(), 10);
    assert_eq!(sl.__len__(), 1);
}
