//! Sorted, non-overlapping address-range bookkeeping.
//!
//! A [`SegmentList`] tracks which byte ranges of an address space are occupied
//! and, optionally, what *sort* of thing occupies each one. It is backed by a
//! `RangeMap<u64, Option<String>>`, so adjacent ranges carrying the same sort
//! coalesce automatically and lookups are logarithmic rather than a linear
//! walk. [`Segment`] is the `[start, end)` + sort value type handed back to
//! Python, and [`SegmentListIter`] iterates a *snapshot* of the segments so
//! Python can mutate the list while iterating.
//!
//! Two consumers drive the design, both on the Python side:
//!
//! * `CFGFast` (`angr/analyses/cfg/cfg_fast.py`) keeps a `SegmentList` of the
//!   binary regions it has already recovered, tagged with sorts like `"code"`
//!   / `"data"`, and asks it for the next unscanned address. This is the hot
//!   path: a CFG recovery performs many thousands of occupy/query calls, which
//!   is why the structure lives in Rust at all.
//! * `HistoryTrackingMixin`
//!   (`angr/storage/memory_mixins/paged_memory/pages/history_tracking_mixin.py`)
//!   uses sort-less lists to record which offsets of a page changed.
//!
//! See the "pure value types" entry for this file in
//! `docs/advanced-topics/rust_engine.rst` for how it fits the Send+Sync
//! export surface.

use std::cmp::{max, min};
use std::collections::HashSet;
use std::ops::Range;

use pyo3::{exceptions::PyStopIteration, prelude::*, types::PyTuple};
use rangemap::RangeMap;

#[pyclass(module = "angr.rustylib.segmentlist", from_py_object)]
#[derive(Clone, Debug)]
#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
pub struct Segment {
    #[pyo3(get)]
    start: u64,
    #[pyo3(get)]
    end: u64,
    #[pyo3(get)]
    sort: Option<String>,
}

impl Segment {
    /// Infallible constructor for a range that already comes from the backing
    /// `RangeMap`, where `start <= end` is a structural guarantee. Python-facing
    /// construction goes through `Segment::new`, which validates.
    fn from_range(start: u64, end: u64, sort: Option<String>) -> Self {
        Segment { start, end, sort }
    }
}

#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pymethods]
impl Segment {
    /// Builds a segment from an explicit `[start, end)` pair.
    ///
    /// This is the `#[new]` constructor, so Python can call it with arbitrary
    /// arguments — including `end < start`, which would make `size()` wrap to a
    /// huge `u64` because the crate builds with `overflow-checks` off in
    /// release. Reject that here so `size()` needs no guard of its own, in the
    /// same spirit as the wrap-around guards in `SegmentList::occupy` /
    /// `SegmentList::release` (angr-sqfj8.134).
    ///
    /// # Errors
    ///
    /// Returns `ValueError` when `start > end`.
    #[new]
    pub fn new(start: u64, end: u64, sort: Option<String>) -> PyResult<Self> {
        if start > end {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Segment start {start:#x} must not exceed end {end:#x}"
            )));
        }
        Ok(Segment::from_range(start, end, sort))
    }

    pub fn __getnewargs__(&self) -> (u64, u64, Option<String>) {
        (self.start, self.end, self.sort.clone())
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Cannot wrap: the fields are read-only from Python and every constructor
    /// (`Segment::new`, `Segment::from_range`) establishes `start <= end`.
    #[getter]
    pub fn size(&self) -> u64 {
        self.end - self.start
    }

    fn __repr__(&self) -> String {
        format!(
            "[{:#x}-{:#x}, {}]",
            self.start,
            self.end,
            self.sort.as_deref().unwrap_or("None")
        )
    }
}

/// A sorted, non-overlapping set of occupied address ranges, each optionally
/// tagged with a *sort* string.
///
/// The name is historical and list-flavored: `SegmentList` is the public name
/// Python imports (`from angr.rustylib import SegmentList`, used by `CFGFast`
/// and declared in `angr/rustylib/__init__.pyi`), and the surface it exposes is
/// list-shaped — `__len__`, `__iter__`, and positional `__getitem__`. Internally
/// it is a *map*: a `RangeMap<u64, Option<String>>` that coalesces adjacent
/// same-sort ranges and answers containment queries in logarithmic time. The
/// two shapes only diverge where the list API forces a positional scan
/// (`__getitem__`, `search`), which is why `CFGFast` avoids indexing in a loop.
/// Renaming to something map-flavored would break every Python importer for no
/// behavioral gain, so the list name stays; see the module doc for the two
/// Python consumers that drive the design.
#[derive(Clone, Default)]
#[pyclass(module = "angr.rustylib.segmentlist", from_py_object)]
#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
pub struct SegmentList {
    map: RangeMap<u64, Option<String>>,
    bytes_occupied: u64,
}

impl SegmentList {
    /// Test-only: the `#[pymethods]` surface below exposes `occupied_size` /
    /// `__len__` to Python; nothing in Rust asks whether the map is empty
    /// outside `segmentlist_tests` (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Total number of bytes of `range` that are already occupied.
    ///
    /// `bytes_occupied` is maintained incrementally, so both `occupy` and
    /// `release` need this same clipped-intersection sum before mutating the
    /// map — `occupy` to avoid double-counting bytes it already owns,
    /// `release` to subtract only what was actually there (angr-03vl4.81).
    fn overlap_bytes(&self, range: &Range<u64>) -> u64 {
        self.map
            .overlapping(range.clone())
            .map(|(r, _)| {
                let s = max(r.start, range.start);
                let e = min(r.end, range.end);
                e.saturating_sub(s)
            })
            .sum()
    }
}

#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pymethods]
impl SegmentList {
    #[new]
    pub fn new() -> Self {
        SegmentList {
            map: RangeMap::new(),
            bytes_occupied: 0,
        }
    }

    pub fn __getnewargs__(&self, py: Python<'_>) -> Py<PyTuple> {
        PyTuple::empty(py).unbind()
    }

    pub fn __getstate__(&self) -> Vec<(u64, u64, Option<String>)> {
        self.map
            .iter()
            .map(|(r, sort)| (r.start, r.end - r.start, sort.clone()))
            .collect()
    }

    pub fn __setstate__(&mut self, state: Vec<(u64, u64, Option<String>)>) {
        self.map.clear();
        for (start, size, sort) in state {
            self.occupy(start, size, sort);
        }
    }

    pub fn __len__(&self) -> usize {
        self.map.len()
    }

    /// Ordinal lookup. This is an O(idx) walk from the start of the map — the
    /// backing `RangeMap` is keyed by address, not by position — so it must not
    /// be called in a loop over consecutive indices. Callers that want to walk
    /// backwards from a `search` hit want `iter_backward_from` instead, which
    /// pays that cost once rather than once per step (angr-9ke6b.200).
    pub fn __getitem__(&self, idx: usize) -> PyResult<Segment> {
        self.map
            .iter()
            .nth(idx)
            .map(|(r, sort)| Segment::from_range(r.start, r.end, sort.clone()))
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!("Index {idx} out of range"))
            })
    }

    pub fn __iter__(slf: PyRef<'_, Self>) -> SegmentListIter {
        SegmentListIter::snapshot(&slf)
    }

    /// Iterates the segment `search(addr)` names and every segment before it,
    /// in descending address order.
    ///
    /// This is the walk `search` + repeated `__getitem__` used to express; done
    /// that way it is quadratic, because `__getitem__` re-walks the map from
    /// index 0 every step. Here the whole prefix is snapshotted in one pass, so
    /// the walk costs what a single `search` already did (angr-9ke6b.200).
    ///
    /// Yields nothing when `addr` is past the last segment, mirroring `search`
    /// returning `None` there.
    pub fn iter_backward_from(&self, addr: u64) -> SegmentListIter {
        let mut segments = Vec::new();
        let mut found = false;
        for (range, sort) in self.map.iter() {
            segments.push((range.start, range.end, sort.clone()));
            // Same predicate `search` uses to pick the segment `addr` lands in.
            if range.end > addr {
                found = true;
                break;
            }
        }
        if !found {
            segments.clear();
        }
        segments.reverse();
        SegmentListIter::new(segments)
    }

    #[getter]
    pub fn occupied_size(&self) -> u64 {
        self.bytes_occupied
    }

    #[getter]
    pub fn has_blocks(&self) -> bool {
        !self.map.is_empty()
    }

    /// Checks which segment the address `addr` should belong to, and returns
    /// that segment's **index** — its ordinal position in the segment list, to
    /// be fed back to `__getitem__`. It is *not* a byte offset.
    ///
    /// Concretely: the index of the first segment whose `end` exceeds `addr`,
    /// or `None` when `addr` is past every segment. Note that `addr` may not
    /// actually fall inside the segment at that index — it can land in the gap
    /// before it.
    ///
    /// Retained as public Python API for API compatibility with the historical
    /// pure-Python `SegmentList` this class replaced: nothing inside angr calls
    /// it any more (`cfg_fast.py::_nodecode_bytes_ratio` switched to
    /// `iter_backward_from` because per-step `__getitem__` indexing is
    /// quadratic), but out-of-tree analyses may. Its behaviour is pinned by
    /// `tests/utils/test_segment_list.py` and `segmentlist_tests.rs`. Do not
    /// treat it as dead code — `dead_code` cannot see `#[pymethods]` callers.
    pub fn search(&self, addr: u64) -> Option<usize> {
        self.map
            .iter()
            .enumerate()
            .find(|(_, (range, _))| range.end > addr)
            .map(|(index, _)| index)
    }

    pub fn next_free_pos(&self, address: u64) -> Option<u64> {
        self.map
            .gaps(&(address..u64::MAX))
            .map(|gap| gap.start)
            .next()
    }

    /// Returns the next occupied position that is not in the given set of sorts.
    #[pyo3(signature = (address, sorts, max_distance = None))]
    pub fn next_pos_with_sort_not_in(
        &self,
        address: u64,
        sorts: HashSet<Option<String>>,
        max_distance: Option<u64>,
    ) -> Option<u64> {
        // Determine the end of the search range
        let end = address.saturating_add(max_distance.unwrap_or(u64::MAX));
        let search_range = address..end;
        // Find the lowest position among the occupied ranges
        self.map
            .overlapping(search_range)
            .filter(|(_, sort)| !sorts.contains(sort))
            .map(|(range, _)| std::cmp::max(range.start, address))
            .next()
    }

    pub fn is_occupied(&self, address: u64) -> bool {
        self.map.contains_key(&address)
    }

    pub fn occupied_by_sort(&self, address: u64) -> Option<String> {
        self.map.get(&address)?.clone()
    }

    pub fn occupied_by(&self, address: u64) -> Option<(u64, u64, Option<String>)> {
        self.map
            .get_key_value(&address)
            .map(|(range, sort)| (range.start, range.end - range.start, sort.clone()))
    }

    pub fn occupy(&mut self, address: u64, size: u64, sort: Option<String>) {
        if size == 0 {
            return;
        }
        // SILENT(cat-a): an address+size that wraps u64 describes no real region, so
        // there is nothing to occupy. Bailing keeps `rangemap` from seeing an inverted
        // range, whose `assert!(range.start < range.end)` would abort the process under
        // this crate's panic="abort" release profile.
        let Some(end) = address.checked_add(size) else {
            return;
        };
        let new_range = address..end;
        let added = size.saturating_sub(self.overlap_bytes(&new_range));
        self.map.insert(new_range, sort);
        self.bytes_occupied = self.bytes_occupied.saturating_add(added);
    }

    pub fn update(&mut self, other: &SegmentList) {
        for (r, sort) in other.map.iter() {
            let size = r.end - r.start;
            self.occupy(r.start, size, sort.clone());
        }
    }

    pub fn release(&mut self, address: u64, size: u64) {
        if size == 0 {
            return;
        }
        // SILENT(cat-a): same wrap-around guard as `occupy` — nothing to release, and
        // `RangeMap::remove` would abort on the inverted range.
        let Some(end) = address.checked_add(size) else {
            return;
        };
        let rem = address..end;
        let removed = self.overlap_bytes(&rem);
        self.map.remove(rem);
        self.bytes_occupied = self.bytes_occupied.saturating_sub(removed);
    }

    pub fn copy(&self) -> SegmentList {
        self.clone()
    }
}

#[pyclass]
#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
pub struct SegmentListIter {
    segments: Vec<(u64, u64, Option<String>)>,
    idx: usize,
}

impl SegmentListIter {
    fn new(segments: Vec<(u64, u64, Option<String>)>) -> Self {
        Self { segments, idx: 0 }
    }

    fn snapshot(list: &SegmentList) -> Self {
        Self::new(
            list.map
                .iter()
                .map(|(range, sort)| (range.start, range.end, sort.clone()))
                .collect(),
        )
    }
}

#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pymethods]
impl SegmentListIter {
    fn __iter__(self_: Bound<'_, Self>) -> Bound<'_, Self> {
        self_
    }

    fn __next__(&mut self) -> PyResult<Segment> {
        match self.segments.get(self.idx) {
            Some((start, end, sort)) => {
                let segment = Segment::from_range(*start, *end, sort.clone());
                self.idx += 1;
                Ok(segment)
            }
            None => Err(PyErr::new::<PyStopIteration, _>("")),
        }
    }
}

#[pymodule]
pub(crate) fn segmentlist(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Segment>()?;
    m.add_class::<SegmentList>()?;
    m.add_class::<SegmentListIter>()?;
    Ok(())
}

#[cfg(test)]
#[path = "segmentlist_tests.rs"]
mod tests;
