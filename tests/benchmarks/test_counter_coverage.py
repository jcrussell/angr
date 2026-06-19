"""Unit tests for the baseline_counters.json coverage gate in
``tests/benchmarks/run_regression.py``.

baseline_counters.json is refreshed manually (--update / --update-counters)
and silently rots when a bench is added but the snapshot is never rerun,
which quietly disables that bench's bench_diff regression report. These
tests exercise the pure-Python helpers (``expected_counter_keys``,
``missing_counter_keys``) against synthetic suites plus the real on-disk
snapshot. They stay in-process — no ``angr`` import, no Rust extension —
so they are millisecond-fast and CI-safe.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import run_regression


def test_expected_keys_dfs_suffix():
    suite = [("ex", 30, "bfs"), ("ex", 30, "dfs")]
    assert run_regression.expected_counter_keys(suite) == {"ex", "ex__dfs"}


def test_expected_keys_length_two_entry_defaults_bfs():
    # MEDIUM_SUITE entries are (name, timeout) — strategy defaults to bfs.
    suite = [("medium_ex", 60)]
    assert run_regression.expected_counter_keys(suite) == {"medium_ex"}


def test_expected_keys_excludes_bimodal_by_default():
    bimodal = next(iter(run_regression.BIMODAL_BENCHMARKS))
    suite = [(bimodal, 60), ("plain", 30)]
    assert run_regression.expected_counter_keys(suite) == {"plain"}
    # ...but opt-in includes them.
    assert run_regression.expected_counter_keys(suite, skip_bimodal=False) == {bimodal, "plain"}


def test_missing_counter_keys_flags_gap():
    suite = [("a", 30), ("b", 30, "dfs")]
    counters = {"a": {}}  # missing b__dfs
    assert run_regression.missing_counter_keys(suite, counters) == ["b__dfs"]


def test_missing_counter_keys_none_when_covered():
    suite = [("a", 30), ("b", 30, "dfs")]
    counters = {"a": {}, "b__dfs": {}}
    assert run_regression.missing_counter_keys(suite, counters) == []


def test_real_full_suite_fully_covered():
    """The shipped baseline_counters.json must cover every non-bimodal
    fast+medium SUITE bench. Acts as the regression gate: adding a bench to
    FAST_SUITE/MEDIUM_SUITE without refreshing the counter snapshot reddens
    this test (and the --check-counter-coverage CI step) rather than
    silently dropping its bench_diff report."""
    suite = run_regression.FAST_SUITE + run_regression.MEDIUM_SUITE
    missing = run_regression.missing_counter_keys(suite)
    assert missing == [], f"baseline_counters.json missing keys: {missing}"
