# pylint: disable=missing-class-docstring,no-self-use
"""Find-all coverage report reusing the ``SimStateEdgeHitmap`` format
(angr-op0dn.13.3.1, M5-B14 sub-goal c).

``mgr.coverage_report()`` OR-folds the ``(prev, cur)`` block edges of every
solution (``found``) state's history into the existing 64KB AFL hitmap
container (:class:`~angr.state_plugins.SimStateEdgeHitmap`) rather than
inventing a new coverage store; ``mgr.coverage_blocks()`` returns the distinct
block set. ``set_exploration_strategy('coverage')`` (the ``CoverageGuided``
policy) guides exploration toward new blocks — the report reads out the result.

The acceptance gate (this file): the coverage report — its distinct-block set
AND the projected hitmap bytes — is IDENTICAL across worker counts (1 vs 4) for
an **exhaustive** find-all run (``num_find=None``, run-to-exhaustion).

Why the report is sourced from the FOUND states and not the coverage policy's
``seen``-set: the found-set is proven worker-count invariant (bd
``findall-parallel-slower-than-serial``), whereas the policy ``seen``-set is a
selection-order artifact — the parallel wave loop hands seed states to workers
WITHOUT ``policy.select``, so the entry block (``main``) is missing under >1
worker (measured: block set + edge set both diverge). The parallel path also
does not retain ``deadended`` / ``pruned`` states, so ONLY ``found`` gives an
invariant report. Hence the gate uses ``num_find=None`` on the bounded
exhaustive synthetic (a ``num_find=k`` early stop lets a wide worker pool
speculatively step extra states before the stop is observed).
"""

from __future__ import annotations

import pytest

from angr.exploration.rust_manager import RustExplorationManager
from tests.engines.conftest import load_pbounce, pbounce_stdin_state


@pytest.fixture(scope="module")
def project():
    return load_pbounce()


def _explore_coverage(project, workers, monkeypatch, strategy="coverage"):
    """Fresh manager under `workers`, optional coverage strategy, exhaustive
    find-all (``num_find=None``) to ``reach_target``. Returns the manager."""
    if workers > 1:
        monkeypatch.setenv("RUST_PARALLEL_WORKERS", str(workers))
    else:
        monkeypatch.delenv("RUST_PARALLEL_WORKERS", raising=False)
    state, _ = pbounce_stdin_state(project)
    mgr = RustExplorationManager(project, [state])
    assert mgr.stats["parallel_real_workers"] == workers
    if strategy is not None:
        mgr.set_exploration_strategy(strategy)
    target = project.loader.find_symbol("reach_target").rebased_addr
    mgr.explore(find=target, num_find=None, n=4096)
    return mgr


class TestCoverageReport:
    def test_coverage_blocks_populated(self, project, monkeypatch):
        """The report exposes a non-empty, sorted, deduped block set."""
        mgr = _explore_coverage(project, 1, monkeypatch)
        blocks = mgr.coverage_blocks()
        assert len(blocks) > 0, "no blocks covered"
        assert blocks == sorted(blocks), "coverage_blocks must be sorted (order-stable)"
        assert len(blocks) == len(set(blocks)), "coverage_blocks must be deduped"

    def test_coverage_report_hitmap_shape(self, project, monkeypatch):
        """The report is a SimStateEdgeHitmap of the standard AFL size."""
        from angr.state_plugins import SimStateEdgeHitmap

        mgr = _explore_coverage(project, 1, monkeypatch)
        report = mgr.coverage_report()
        assert isinstance(report, SimStateEdgeHitmap)
        assert len(report.edge_hitmap) == SimStateEdgeHitmap.HITMAP_SIZE
        # At least one covered block => at least one non-zero bucket.
        assert any(report.edge_hitmap), "hitmap has no covered buckets"

    @pytest.mark.parametrize("workers", [2, 4])
    def test_coverage_report_invariant_across_worker_counts(self, project, workers, monkeypatch):
        """Block set, edge-hitmap bytes, and their counts are identical for
        workers=1 vs workers=N on an exhaustive drain."""
        baseline = _explore_coverage(project, 1, monkeypatch)
        base_blocks = baseline.coverage_blocks()
        base_hitmap = baseline.coverage_report().edge_hitmap
        base_edge_count = sum(1 for b in base_hitmap if b)
        assert base_blocks, "baseline produced no coverage"
        assert base_edge_count > 0, "baseline produced no edges"

        variant = _explore_coverage(project, workers, monkeypatch)
        assert list(variant.errored) == [], f"workers={workers} produced errored states"
        var_blocks = variant.coverage_blocks()
        var_hitmap = variant.coverage_report().edge_hitmap
        var_edge_count = sum(1 for b in var_hitmap if b)

        assert var_blocks == base_blocks, f"workers={workers} block set diverged"
        assert var_hitmap == base_hitmap, f"workers={workers} hitmap bytes diverged"
        assert var_edge_count == base_edge_count, f"workers={workers} edge count diverged"
