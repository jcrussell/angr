"""Unit tests for the peak_memory_mb regression gate decision.

Exercises the pure helper ``run_regression.memory_regression_pct`` against
synthetic memory measurements. Stays in-process and angr-free: importing
``run_regression`` only pulls ``run_single``'s light top-level (its
``import angr`` is lazy, inside a function), so these are millisecond-fast
and CI-safe.

Regression guard for angr-cudgw.2: the reviewer worried a zero-solve run
could mask a peak-RSS regression by hiding the memory check behind a
"successful_solves > 0" branch. The helper deliberately has no solve/stats
parameter, so its decision is structurally independent of solve count —
these tests pin that invariant.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import run_regression


def test_regression_flagged_above_threshold():
    # 120MB vs 100MB baseline at 0.15 threshold = +20% > 15% → regression.
    pct = run_regression.memory_regression_pct(100.0, 120.0, 0.15)
    assert pct is not None
    assert round(pct) == 20


def test_within_threshold_no_regression():
    # +10% is within a 15% threshold → no regression.
    assert run_regression.memory_regression_pct(100.0, 110.0, 0.15) is None


def test_just_below_threshold_no_regression():
    # +14% is clearly under a 15% threshold → no regression. (Avoids the
    # exact +15% point, where float rounding of 100*1.15 = 114.9999... can
    # make 115.0 trip — behavior inherited verbatim from the inline check.)
    assert run_regression.memory_regression_pct(100.0, 114.0, 0.15) is None


def test_no_baseline_no_regression():
    # No cached baseline (e.g. --no-memory-check or a new bench) → skip.
    assert run_regression.memory_regression_pct(None, 999.0, 0.15) is None


def test_nonpositive_baseline_no_regression():
    assert run_regression.memory_regression_pct(0.0, 999.0, 0.15) is None
    assert run_regression.memory_regression_pct(-5.0, 999.0, 0.15) is None


def test_missing_current_measurement_no_regression():
    assert run_regression.memory_regression_pct(100.0, None, 0.15) is None


def test_decision_is_independent_of_solve_count():
    """The gate must fire on a memory blowup regardless of solves.

    The helper signature has no solve-count / stats argument, so a
    zero-solve run that still allocated a 3x peak gets flagged exactly
    like a many-solve run with the same peak. This is the angr-cudgw.2
    invariant: peak_memory_mb is a process-wide high-water mark and must
    not be masked by an empty-stats / zero-solve conditional.
    """
    # Same memory numbers, "solve count" is not a parameter at all.
    pct = run_regression.memory_regression_pct(100.0, 300.0, 0.5)
    assert pct is not None
    assert round(pct) == 200
