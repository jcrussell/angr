# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests: user-code exceptions surface from explore()/run() (angr-vmi2l.298).

A find/avoid predicate or a technique filter()/complete() that raises used to be
logged at debug and read as "no match"/"not complete", so explore() ended with
an empty found stash. ``SimulationManager`` lets the exception propagate.
"""

from __future__ import annotations

import pytest

from angr.exploration_techniques import ExplorationTechnique
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")


class _UserBug(Exception):
    pass


def _raise(_state):
    raise _UserBug("predicate bug")


def _mgr(project):
    return RustExplorationManager(project, [project.factory.entry_state()])


class _FilterRaises(ExplorationTechnique):
    def filter(self, simgr, state, **kwargs):
        raise _UserBug("filter bug")


class _CompleteRaises(ExplorationTechnique):
    def complete(self, simgr):
        raise _UserBug("complete bug")


class TestUserCallbackErrors:
    def test_raising_find_predicate_propagates(self, fauxware_project):
        with pytest.raises(_UserBug, match="predicate bug"):
            _mgr(fauxware_project).explore(find=_raise)

    def test_raising_avoid_predicate_propagates(self, fauxware_project):
        with pytest.raises(_UserBug, match="predicate bug"):
            _mgr(fauxware_project).explore(find=lambda s: False, avoid=_raise)

    def test_manager_still_steps_after_a_predicate_raise(self, fauxware_project):
        """The handler resumes the pending state before re-raising, so the
        manager is not left wedged on an unanswered callback."""
        mgr = _mgr(fauxware_project)
        with pytest.raises(_UserBug):
            mgr.explore(find=_raise)
        mgr.explore(find=lambda s: False)
        assert len(mgr.deadended) == 3

    def test_raising_technique_filter_propagates(self, fauxware_project):
        mgr = _mgr(fauxware_project)
        mgr.use_technique(_FilterRaises())
        with pytest.raises(_UserBug, match="filter bug"):
            mgr.run()

    def test_raising_technique_complete_propagates(self, fauxware_project):
        mgr = _mgr(fauxware_project)
        mgr.use_technique(_CompleteRaises())
        with pytest.raises(_UserBug, match="complete bug"):
            mgr.run()
