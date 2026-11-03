# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests: explore()/run() without find/avoid leaves none armed from a
previous explore() (angr-mx00p.132.49).

``SimulationManager.explore`` removes its Explorer when the call ends, so a later
``run()`` has no find/avoid. The Rust manager used to keep the old predicate and
find/avoid addrs armed, so ``run()`` kept moving matches to ``found``.
"""

from __future__ import annotations

import pytest

from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

# fauxware's puts("Welcome to the admin console, trusted user!") call site.
_ACCEPTED = 0x4006ED


def _mgr(project):
    return RustExplorationManager(project, [project.factory.entry_state()])


class TestExploreDisarmsFind:
    def test_run_after_a_never_matching_find_predicate(self, fauxware_project):
        mgr = _mgr(fauxware_project)
        calls = []

        def never(state):
            calls.append(state.addr)
            return False

        mgr.explore(find=never, n=1, max_steps=1)
        before = len(calls)
        mgr.run()
        assert len(calls) == before
        assert not mgr.found
        assert len(mgr.deadended) == 3

    def test_run_after_an_address_find(self, fauxware_project):
        mgr = _mgr(fauxware_project)
        mgr.explore(find=_ACCEPTED)
        assert len(mgr.found) == 1
        mgr.drop(stash="found")
        mgr.run()
        assert not mgr.found
        assert len(mgr.deadended) == 2

    def test_run_after_a_find_predicate(self, fauxware_project):
        mgr = _mgr(fauxware_project)
        mgr.explore(find=lambda s: s.addr == _ACCEPTED)
        assert len(mgr.found) == 1
        mgr.drop(stash="found")
        mgr.run()
        assert not mgr.found
        assert len(mgr.deadended) == 2

    def test_run_after_an_address_avoid(self, fauxware_project):
        mgr = _mgr(fauxware_project)
        mgr.explore(find=lambda s: False, avoid=_ACCEPTED, max_steps=1)
        mgr.run()
        assert not mgr.avoid
        assert len(mgr.deadended) == 3
