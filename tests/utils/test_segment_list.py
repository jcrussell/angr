from __future__ import annotations

import pickle
import unittest

from angr.rustylib import SegmentList


class TestSegmentList(unittest.TestCase):
    """
    Test the SegmentList class.
    """

    # pylint: disable=no-self-use

    def test_occupy(self):
        seg_list = SegmentList()
        seg_list.occupy(0, 1, "code")
        seg_list.occupy(2, 3, "code")

        assert len(seg_list) == 2
        assert seg_list[0].end == 1
        assert seg_list[1].end == 5
        assert seg_list.is_occupied(4)
        assert seg_list.is_occupied(5) is False

    def test_merging(self):
        seg_list = SegmentList()

        # They should be merged
        seg_list.occupy(0, 1, "code")
        seg_list.occupy(1, 2, "code")

        assert len(seg_list) == 1
        assert seg_list[0].start == 0
        assert seg_list[0].end == 3

    def test_not_merged(self):
        seg_list = SegmentList()

        # They should not be merged
        seg_list.occupy(0, 1, "code")
        seg_list.occupy(1, 2, "data")

        assert len(seg_list) == 2
        assert seg_list[0].start == 0
        assert seg_list[0].end == 1
        assert seg_list[1].start == 1
        assert seg_list[1].end == 3

    def test_multi_merge(self):
        seg_list = SegmentList()

        # They should be merged, and create three different segments
        seg_list.occupy(0, 5, "code")
        seg_list.occupy(5, 5, "code")
        seg_list.occupy(1, 2, "data")

        assert len(seg_list) == 3

        assert seg_list[0].start == 0
        assert seg_list[0].end == 1
        assert seg_list[0].sort == "code"

        assert seg_list[1].start == 1
        assert seg_list[1].end == 3
        assert seg_list[1].sort == "data"

        assert seg_list[2].start == 3
        assert seg_list[2].end == 10
        assert seg_list[2].sort == "code"

    def test_fully_overlapping(self):
        seg_list = SegmentList()

        seg_list.occupy(5, 5, "code")
        seg_list.occupy(4, 1, "code")
        seg_list.occupy(2, 2, "code")

        assert len(seg_list) == 1
        assert seg_list[0].start == 2
        assert seg_list[0].end == 10

    def test_overlapping_not_merged(self):
        seg_list = SegmentList()

        seg_list.occupy(5, 5, "data")
        seg_list.occupy(4, 1, "code")
        seg_list.occupy(2, 2, "data")

        assert len(seg_list) == 3
        assert seg_list[0].start == 2
        assert seg_list[2].end == 10

        seg_list.occupy(3, 2, "data")

        assert len(seg_list) == 1
        assert seg_list[0].start == 2
        assert seg_list[0].end == 10

    def test_partially_overlapping_not_merged(self):
        seg_list = SegmentList()

        seg_list.occupy(10, 20, "code")
        seg_list.occupy(9, 2, "data")

        assert len(seg_list) == 2
        assert seg_list[0].start == 9
        assert seg_list[0].end == 11
        assert seg_list[0].sort == "data"

        assert seg_list[1].start == 11
        assert seg_list[1].end == 30
        assert seg_list[1].sort == "code"

    def _boundary_list(self):
        # [0,10)@code, [10,20)@data, gap [20,30), [30,35)@code
        seg_list = SegmentList()
        seg_list.occupy(0, 10, "code")
        seg_list.occupy(10, 10, "data")
        seg_list.occupy(30, 5, "code")
        return seg_list

    def test_search_boundaries(self):
        seg_list = self._boundary_list()

        # Segments are half-open [start, end): an address at a segment's end
        # boundary belongs to the NEXT segment (regression for zi35f.1).
        assert seg_list.search(0) == 0  # inside first
        assert seg_list.search(9) == 0  # last byte of first
        assert seg_list.search(10) == 1  # boundary -> second
        assert seg_list.search(19) == 1  # last byte of second
        # An address inside a gap resolves to the next segment, not None.
        assert seg_list.search(20) == 2
        assert seg_list.search(29) == 2
        assert seg_list.search(30) == 2  # inside third
        assert seg_list.search(34) == 2  # last byte of third
        assert seg_list.search(35) is None  # past everything

        assert SegmentList().search(0) is None  # empty list

    def test_next_free_pos(self):
        seg_list = self._boundary_list()

        # Contiguous [0,20) occupation -> first free byte is the gap start.
        assert seg_list.next_free_pos(0) == 20
        assert seg_list.next_free_pos(5) == 20
        assert seg_list.next_free_pos(19) == 20
        assert seg_list.next_free_pos(20) == 20  # already free
        assert seg_list.next_free_pos(29) == 29  # free byte before third seg
        assert seg_list.next_free_pos(30) == 35  # inside third -> after its end
        assert seg_list.next_free_pos(35) == 35  # already free

        assert SegmentList().next_free_pos(42) == 42  # empty -> address itself

    def test_next_pos_with_sort_not_in(self):
        seg_list = self._boundary_list()

        # First occupied byte whose sort is not excluded, clamped to >= address.
        assert seg_list.next_pos_with_sort_not_in(0, {"code"}) == 10  # data seg
        assert seg_list.next_pos_with_sort_not_in(0, {"data"}) == 0  # code at 0
        # address past the matching segment's start clamps to address.
        assert seg_list.next_pos_with_sort_not_in(12, {"code"}) == 12
        # Everything excluded -> None.
        assert seg_list.next_pos_with_sort_not_in(0, {"code", "data"}) is None

        # max_distance bounds the (half-open) search window.
        assert seg_list.next_pos_with_sort_not_in(0, {"code"}, 5) is None
        assert seg_list.next_pos_with_sort_not_in(0, {"code"}, 11) == 10

        assert SegmentList().next_pos_with_sort_not_in(0, {"code"}) is None

    def test_occupied_by(self):
        seg_list = self._boundary_list()

        assert seg_list.occupied_by(0) == (0, 10, "code")
        assert seg_list.occupied_by(9) == (0, 10, "code")  # last byte of first
        assert seg_list.occupied_by(10) == (10, 10, "data")  # boundary -> next
        assert seg_list.occupied_by(20) is None  # in the gap
        assert seg_list.occupied_by(34) == (30, 5, "code")  # last byte of third
        assert seg_list.occupied_by(35) is None  # end boundary is exclusive

        assert seg_list.occupied_by_sort(5) == "code"
        assert seg_list.occupied_by_sort(15) == "data"
        assert seg_list.occupied_by_sort(25) is None  # gap

    def test_update(self):
        seg_list = self._boundary_list()
        other = SegmentList()
        other.occupy(20, 10, "data")  # fills the gap + abuts the third seg
        other.occupy(100, 4, "code")

        seg_list.update(other)

        # Gap [20,30) is now occupied; occupied_size gains 10 (gap) + 4 (new).
        assert seg_list.occupied_size == 25 + 10 + 4
        # The new [20,30)@data abuts the existing [10,20)@data and merges into
        # a single [10,30)@data segment.
        assert seg_list.occupied_by(20) == (10, 20, "data")
        assert seg_list.occupied_by(100) == (100, 4, "code")

    def test_copy_independence(self):
        seg_list = self._boundary_list()
        clone = seg_list.copy()

        clone.occupy(100, 5, "code")
        # Mutating the copy must not touch the original.
        assert len(seg_list) == 3
        assert len(clone) == 4
        assert seg_list.occupied_by(100) is None
        assert clone.occupied_by(100) == (100, 5, "code")

    def test_pickle_round_trip(self):
        seg_list = self._boundary_list()
        restored = pickle.loads(pickle.dumps(seg_list))

        assert len(restored) == len(seg_list)
        assert restored.occupied_size == seg_list.occupied_size
        assert restored.search(10) == seg_list.search(10)
        assert [(s.start, s.end, s.sort) for s in restored] == [(s.start, s.end, s.sort) for s in seg_list]

        # An empty list survives the round-trip too.
        empty = pickle.loads(pickle.dumps(SegmentList()))
        assert len(empty) == 0
        assert empty.occupied_size == 0


if __name__ == "__main__":
    unittest.main()
