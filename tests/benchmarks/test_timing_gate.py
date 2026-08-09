"""Unit tests for the rust_time regression gate decision.

Exercises the pure helper ``run_regression.timing_regression_pct`` against
synthetic timings. Stays in-process and angr-free for the same reason
``test_memory_gate.py`` does: importing ``run_regression`` only pulls
``run_single``'s light top-level, so these are millisecond-fast and CI-safe.

Regression guard for angr-z8p3x: a relative-only threshold made the
sub-second benches (sharif7_rev50 at a 0.16s baseline → a ~24ms bar) fail the
gate on host noise, twice, retries included. The helper adds an absolute-delta
floor; these tests pin both bars and, just as importantly, pin that benches
above the floor keep their old behaviour.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import run_regression

FLOOR = run_regression.DEFAULT_REGRESSION_FLOOR_S


def test_regression_flagged_when_both_bars_cleared():
    # 12s vs 10s baseline: +20% > 15% and +2000ms > 50ms → regression.
    pct = run_regression.timing_regression_pct(10.0, 12.0, 0.15, FLOOR)
    assert pct is not None
    assert round(pct) == 20


def test_within_threshold_no_regression():
    # +10% is within a 15% threshold, absolute delta irrelevant.
    assert run_regression.timing_regression_pct(10.0, 11.0, 0.15, FLOOR) is None


def test_sharif_style_noise_excursion_suppressed():
    """The angr-z8p3x case: 0.16s baseline, 0.19s measurement.

    +19% clears the relative bar but the delta is 30ms — under the 50ms
    floor, so it must not fail the gate.
    """
    assert run_regression.timing_regression_pct(0.16, 0.19, 0.15, FLOOR) is None


def test_sub_second_bench_still_fails_on_a_real_regression():
    # Same 0.16s baseline, but now 2.5x slower: 240ms delta clears the floor.
    pct = run_regression.timing_regression_pct(0.16, 0.40, 0.15, FLOOR)
    assert pct is not None
    assert round(pct) == 150


def test_slow_bench_behaviour_unchanged_by_the_floor():
    """Above floor/threshold (~0.33s at the defaults) the floor is inert.

    Any bench whose relative bar alone already implies a >50ms delta must
    decide identically with and without the floor — otherwise the fix would
    have quietly loosened the gate for the benches that matter.
    """
    for baseline in (0.5, 1.0, 3.5, 22.0):
        for mult in (1.0, 1.14, 1.16, 1.5, 3.0):
            current = baseline * mult
            with_floor = run_regression.timing_regression_pct(baseline, current, 0.15, FLOOR)
            relative_only = run_regression.timing_regression_pct(baseline, current, 0.15, 0.0)
            assert (with_floor is None) == (relative_only is None), (baseline, mult)


def test_zero_floor_restores_relative_only_behaviour():
    pct = run_regression.timing_regression_pct(0.16, 0.19, 0.15, 0.0)
    assert pct is not None
    assert round(pct) == 19


def test_no_baseline_no_regression():
    assert run_regression.timing_regression_pct(None, 9.0, 0.15, FLOOR) is None


def test_nonpositive_baseline_no_regression():
    assert run_regression.timing_regression_pct(0.0, 9.0, 0.15, FLOOR) is None
    assert run_regression.timing_regression_pct(-1.0, 9.0, 0.15, FLOOR) is None


def test_missing_current_measurement_no_regression():
    assert run_regression.timing_regression_pct(10.0, None, 0.15, FLOOR) is None


def test_floor_defaults_to_the_module_constant():
    # Called without the floor argument, the helper must apply the same
    # default the CLI advertises — not silently fall back to 0.
    assert run_regression.timing_regression_pct(0.16, 0.19, 0.15) is None
