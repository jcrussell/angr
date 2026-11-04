"""SimulationManager's ``backend=`` seam (angr-mx00p.136.2).

A pure-Python fake backend claims every other state and steps it through
``project.factory.successors``; the manager must give the same results as
with ``backend=None``, and a technique's ``step_state`` hook must still wrap
the claimed states.
"""

from __future__ import annotations

import os
import pickle

import pytest

import angr
from angr.sim_manager_backend import SimulationBackend
from tests.common import bin_location

FAUXWARE = os.path.join(bin_location, "tests", "x86_64", "fauxware")
FIND = 0x4006ED
AVOID = [0x4006AA, 0x4006FD]


class AlternatingBackend(SimulationBackend):
    """Claims every other state it is asked about and steps it on the Python engine."""

    def __init__(self):
        self.asked = 0
        self.stepped = 0

    def claims(self, simgr, state, successor_func, run_args):
        self.asked += 1
        if self.asked % 2:
            return True, ""
        return False, "odd turn"

    def step_state(self, simgr, state, error_list, **run_args):
        self.stepped += 1
        successors = simgr._project.factory.successors(state, **run_args)
        return {
            None: successors.flat_successors,
            "unsat": successors.unsat_successors,
            "unconstrained": successors.unconstrained_successors,
        }


class CountingTechnique(angr.ExplorationTechnique):
    def __init__(self):
        super().__init__()
        self.seen = 0

    def step_state(self, simgr, state, **kwargs):
        self.seen += 1
        return simgr.step_state(state, **kwargs)


@pytest.fixture(scope="module")
def project():
    return angr.Project(FAUXWARE, auto_load_libs=False)


def _summary(simgr):
    # Block paths, not stdin dumps: the solver's witness for a symbolic byte varies between runs.
    return {
        name: sorted((s.addr, tuple(s.history.bbl_addrs)) for s in states)
        for name, states in simgr.stashes.items()
        if states
    }


def _simgr(project, backend):
    # Built directly so an ANGR_DEFAULT_ENGINE=rust run cannot route it to RustExplorationManager.
    return angr.SimulationManager(project, active_states=[project.factory.entry_state()], backend=backend)


@pytest.mark.parametrize(
    "drive",
    [
        pytest.param(lambda m: [m.step() for _ in range(12)], id="step"),
        pytest.param(lambda m: m.run(), id="run"),
        pytest.param(lambda m: m.explore(find=FIND, avoid=AVOID), id="explore"),
    ],
)
def test_backend_results_match_no_backend(project, drive):
    plain = _simgr(project, None)
    drive(plain)

    backend = AlternatingBackend()
    seamed = _simgr(project, backend)
    drive(seamed)

    assert _summary(seamed) == _summary(plain)
    assert backend.stepped > 0
    assert backend.asked > backend.stepped


def test_technique_hook_wraps_backend(project):
    backend = AlternatingBackend()
    technique = CountingTechnique()
    simgr = _simgr(project, backend)
    simgr.use_technique(technique)
    simgr.run()

    assert backend.stepped > 0
    assert technique.seen == backend.asked


def test_base_backend_claims_nothing(project):
    simgr = _simgr(project, SimulationBackend())
    simgr.run()
    assert _summary(simgr) == _summary(_simgr(project, None).run())


def test_copy_keeps_backend_pickle_drops_it(project):
    backend = AlternatingBackend()
    simgr = _simgr(project, backend)
    assert simgr.copy()._backend is backend
    assert pickle.loads(pickle.dumps(simgr))._backend is None
