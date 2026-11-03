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


class TestStepAfterExplore:
    """A step() after explore() advances the found state as SimulationManager does
    (angr-mx00p.132.19): the explore's find addr and num_find no longer apply."""

    def test_step_advances_a_found_state_moved_to_active(self, fauxware_project):
        mgr = _mgr(fauxware_project)
        mgr.explore(find=_ACCEPTED)
        mgr.drop(stash="active")
        mgr.move(from_stash="found", to_stash="active")
        mgr.step()
        assert not mgr.found
        assert mgr.active
        assert _ACCEPTED not in [s.addr for s in mgr.active]

    def test_step_advances_active_once_num_find_is_reached(self, fauxware_project):
        mgr = _mgr(fauxware_project)
        mgr.explore(find=_ACCEPTED)
        assert len(mgr.found) == 1
        before = sorted(s.addr for s in mgr.active)
        assert before
        mgr.step()
        assert len(mgr.found) == 1
        assert sorted(s.addr for s in mgr.active) != before


def _two_stage(project, engine, first, second):
    state = project.factory.entry_state()
    mgr = RustExplorationManager(project, [state]) if engine == "rust" else project.factory.simgr(state)
    mgr.explore(**first)
    mgr.move(from_stash="found", to_stash="active")
    mgr.explore(**second)
    return {
        "found": sorted(s.addr for s in mgr.found),
        "avoid": len(mgr.avoid),
        "deadended": len(mgr.deadended),
        "active": len(mgr.active),
    }


class TestExploreResetsPerCall:
    """A second explore() arms exactly its own find/avoid, as SimulationManager's per-call Explorer does
    (angr-vmi2l.1)."""

    @pytest.mark.parametrize(
        "first,second",
        [
            pytest.param({"find": 0x40071D, "avoid": _ACCEPTED}, {"find": 0x4006FB}, id="avoid_addrs_reset"),
            pytest.param({"find": lambda s: s.addr == 0x40071D}, {"avoid": 0x4006FD}, id="find_predicate_reset"),
        ],
    )
    def test_matches_python_engine(self, fauxware_project, first, second):
        expected = _two_stage(fauxware_project, "python", first, second)
        assert _two_stage(fauxware_project, "rust", first, second) == expected
