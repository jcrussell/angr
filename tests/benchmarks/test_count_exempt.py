"""Guard the COUNT_EXEMPT / BIMODAL_BENCHMARKS relationship in
``tests/benchmarks/run_regression.py``.

``COUNT_EXEMPT`` (benches skipped by the ``--check-counts`` gate) is
documented as the four *original* bimodal benches and a strict subset of
``BIMODAL_BENCHMARKS``. The source comment hardcodes that count, and it has
drifted before (4 -> 5 when ``CADET_00001_partial`` joined
``BIMODAL_BENCHMARKS`` under angr-027h). These tests pin the invariant so a
future edit that adds a bench to one frozenset but not the comment fails
loudly instead of silently weakening the gate (angr-cudgw.11).

In-process and millisecond-fast: they only read two module-level frozensets.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import run_regression

# The exact bench that is in BIMODAL_BENCHMARKS but intentionally NOT in
# COUNT_EXEMPT: its count baselines are 0/0/0 so the `baseline_val > 0` guard
# already no-ops --check-counts for it (see the COUNT_EXEMPT comment).
_BIMODAL_NOT_EXEMPT = frozenset({"CADET_00001_partial"})


def test_count_exempt_is_strict_subset_of_bimodal():
    # The COUNT_EXEMPT comment claims it is "a strict subset, not equal to"
    # BIMODAL_BENCHMARKS. Pin both halves of that claim.
    assert run_regression.COUNT_EXEMPT < run_regression.BIMODAL_BENCHMARKS


def test_count_exempt_is_the_four_original_bimodal():
    # The comment hardcodes "the four original BIMODAL_BENCHMARKS". If a new
    # bimodal bench is added to COUNT_EXEMPT, update both this count and the
    # source comment in lockstep.
    assert len(run_regression.COUNT_EXEMPT) == 4


def test_bimodal_minus_exempt_is_exactly_cadet():
    # The only bimodal bench excluded from COUNT_EXEMPT is CADET_00001_partial.
    # If this fails, either a new bimodal bench was added (decide whether it
    # belongs in COUNT_EXEMPT and update the comment) or CADET was removed.
    extra = run_regression.BIMODAL_BENCHMARKS - run_regression.COUNT_EXEMPT
    assert extra == _BIMODAL_NOT_EXEMPT
