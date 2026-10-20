# pylint: disable=missing-class-docstring,no-self-use
"""Find-all / run-to-exhaustion mode: ``explore(find=..., num_find=None)``.

Productizes what the M5-P2 driver proved (angr-op0dn.13.3, sub-goal a): a
run-to-exhaustion mode that collects every reachable solution and terminates on
the frontier draining rather than at a fixed count.

The critical regression this guards is the hang documented in the
``invariant-active-empty-not-partial-found`` memory: a find with
``num_find`` above the reachable-find count must terminate via ``active_empty``,
NOT spin re-invoking ``run()`` on a partial-count ``found`` event. ``num_find=None``
maps to a large sentinel, so it exercises exactly that path.
"""

from __future__ import annotations

import pytest

from angr.exploration.rust_manager import RustExplorationManager
from tests.engines.conftest import load_pbounce, pbounce_stdin_state


@pytest.fixture(scope="module")
def project():
    return load_pbounce()


def _run(project, target, num_find):
    """Fresh manager + state; run explore, return the found-set size."""
    state, _ = pbounce_stdin_state(project)
    mgr = RustExplorationManager(project, [state])
    mgr.explore(find=target, num_find=num_find, n=4096)
    return len(mgr.found)


class TestFindAll:
    def test_find_all_matches_large_explicit_count(self, project):
        """num_find=None collects the SAME exhaustive found-set as a large explicit cap."""
        target = project.loader.find_symbol("reach_target").rebased_addr
        exhaustive = _run(project, target, 10_000)
        find_all = _run(project, target, None)
        assert find_all == exhaustive
        assert find_all >= 1  # the synthetic has at least one reaching path

    def test_find_all_unreachable_terminates(self, project):
        """An unreachable target under num_find=None terminates (active_empty),
        not the unbounded run()-respin hang of invariant-active-empty-not-partial-found."""
        # An address that is never a block entry -> never found; the run must
        # drain the (bounded) synthetic frontier and return with zero finds.
        unreachable = 0x1
        assert _run(project, unreachable, None) == 0
