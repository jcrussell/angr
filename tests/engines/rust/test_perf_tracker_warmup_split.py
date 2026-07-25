"""Unit tests for the SimProcedure warmup-split gate in PerformanceTracker.

``PerformanceTracker.add_simprocedure_phase`` banks each phase's cost into a
``first_``-prefixed twin during the process's *first* real SimProcedure
crossing, so a bench with a single crossing does not report its one-time
warmup (lazy imports, page-cache fill) as per-crossing steady-state cost.

Regression coverage for angr-zi35f.3: the gate used to key off
``callback_simprocedure_count``, which the fast paths in
rust_callback_dispatch.py (internal passthrough, stale-hook skip, deadend fast
path) bump via ``record_simprocedure_call`` *without* recording any phase. A
fast-path bounce firing before the first real crossing therefore closed the
warmup window early and folded the real warmup into steady-state totals. The
gate now counts phase-recording crossings independently, so fast paths no
longer perturb it. Pure-Python, no Rust manager needed.
"""

from __future__ import annotations

from angr.exploration.rust_perf_tracker import PerformanceTracker


def _crossing(t: PerformanceTracker, sc=1, cp=1, ex=1, sb=1, total=10) -> None:
    """Simulate one real SimProcedure crossing recording all four phases."""
    t.add_simprocedure_phase("state_create", sc)
    t.add_simprocedure_phase("state_copy", cp)
    t.add_simprocedure_phase("execute", ex)
    t.add_simprocedure_phase("sync_back", sb)
    t.record_simprocedure_call(total)


def test_first_crossing_banks_first_twins():
    t = PerformanceTracker()
    _crossing(t, sc=5, cp=6, ex=7, sb=8)
    s = t._stats
    assert s["callback_simprocedure_first_state_create_ns"] == 5
    assert s["callback_simprocedure_first_state_copy_ns"] == 6
    assert s["callback_simprocedure_first_execute_ns"] == 7
    assert s["callback_simprocedure_first_sync_back_ns"] == 8


def test_second_crossing_does_not_bank_first_twins():
    t = PerformanceTracker()
    _crossing(t, sc=5, cp=6, ex=7, sb=8)
    _crossing(t, sc=50, cp=60, ex=70, sb=80)
    s = t._stats
    # first_* frozen at crossing 1's values; steady-state totals include both.
    assert s["callback_simprocedure_first_state_create_ns"] == 5
    assert s["callback_simprocedure_first_sync_back_ns"] == 8
    assert s["callback_simprocedure_state_create_ns"] == 55
    assert s["callback_simprocedure_sync_back_ns"] == 88


def test_fast_path_before_first_crossing_does_not_close_window():
    """The core angr-zi35f.3 regression: a fast-path bounce that only calls
    record_simprocedure_call must not consume the warmup window."""
    t = PerformanceTracker()
    # Three fast-path bounces fire first (deadend / passthrough / stale hook).
    t.record_simprocedure_call(3)
    t.record_simprocedure_call(3)
    t.record_simprocedure_call(3)
    assert t._stats["callback_simprocedure_count"] == 3
    # First *real* crossing must still bank its phases into first_*.
    _crossing(t, sc=5, cp=6, ex=7, sb=8)
    s = t._stats
    assert s["callback_simprocedure_first_state_create_ns"] == 5
    assert s["callback_simprocedure_first_execute_ns"] == 7


def test_increment_only_count_does_not_close_window():
    """The Python VEX-fallback path bumps only the count; it must not close the
    warmup window either."""
    t = PerformanceTracker()
    t.increment_simprocedure_count()
    t.increment_simprocedure_count()
    _crossing(t, sc=9)
    assert t._stats["callback_simprocedure_first_state_create_ns"] == 9


def test_error_path_state_create_only_counts_as_a_crossing():
    """The state==None error path records only ``state_create`` then returns;
    it is still a phase-recording crossing and closes the window for the next
    one."""
    t = PerformanceTracker()
    t.add_simprocedure_phase("state_create", 4)  # error path, no other phases
    t.record_simprocedure_call(4)
    _crossing(t, sc=40)
    s = t._stats
    assert s["callback_simprocedure_first_state_create_ns"] == 4
    assert s["callback_simprocedure_state_create_ns"] == 44
