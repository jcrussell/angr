# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests: explore() after step() explores only the frontier step() left (angr-vmi2l.319).

The phase-2 eager retry re-seeds the manager's initial states when an address find exhausts its frontier.
After a step() it re-ran paths the frontier had already left: on pbounce, an explore() on an empty active
stash found six states, where SimulationManager's explore finds nothing. The preinit-prefix replay
(angr-bdeqa) likewise found ``main`` for an explore that started past it.
"""

from __future__ import annotations

import pytest

from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager, TrapPoint, load_pbounce

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")


def _counts(mgr):
    return {stash: len(mgr.stashes.get(stash, [])) for stash in ("found", "deadended", "active")}


def _stepped_then_explored(engine, steps, find_symbol):
    """Step ``steps`` times (stopping early once active empties), then explore toward ``find_symbol``."""
    project = load_pbounce()
    project.hook(project.loader.find_symbol("trap_point").rebased_addr, TrapPoint())
    state = project.factory.entry_state()
    mgr = RustExplorationManager(project, [state]) if engine == "rust" else project.factory.simgr(state)
    for _ in range(steps):
        mgr.step()
        if not mgr.active:
            break
    before = _counts(mgr)
    mgr.explore(find=project.loader.find_symbol(find_symbol).rebased_addr, num_find=8)
    return before, _counts(mgr)


class TestExploreAfterStep:
    @pytest.mark.parametrize("engine", ["python", "rust"])
    def test_explore_on_a_drained_manager_finds_nothing(self, engine):
        before, after = _stepped_then_explored(engine, 40, "reach_target")
        assert before["active"] == 0
        assert after == before

    def test_unfound_explore_after_step_drains_only_the_frontier(self):
        """``main`` is behind the frontier five steps in, so the explore only drains it and ends where stepping
        to the end does; a re-seed from the entry state would deadend every path a second time, and a
        preinit replay would find ``main``."""
        drained, _ = _stepped_then_explored("rust", 40, "reach_target")
        before, after = _stepped_then_explored("rust", 5, "main")
        assert before["active"] and before["deadended"]
        assert after == drained
