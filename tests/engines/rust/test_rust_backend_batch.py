"""SimulationManager+RustBackend: a step's claimed states go to Rust in one call (angr-mx00p.136.13).

With nothing hooking ``step_state``, :meth:`SimulationManager.step` hands
every state the backend claims to one :meth:`RustBackend.step_states` call.
A pass-through ``step_state`` technique keeps the per-state path. Over 50
steps on fauxware and ais3_crackme both paths must leave the same stashes,
and only the batched one may step several states in one call. A state the
selector leaves unstepped keeps its place among the batched successors.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr import claripy
from angr.exploration.rust_backend import RustBackend
from tests.engines.conftest import EXAMPLES_DIR, FAUXWARE_PATH, RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")

STEPS = 50
AIS3_PATH = os.path.join(EXAMPLES_DIR, "ais3_crackme", "ais3_crackme")


class _PerState(angr.ExplorationTechnique):
    """Hooks step_state without changing it, which keeps the manager on the per-state path."""

    def step_state(self, simgr, state, **kwargs):
        return simgr.step_state(state, **kwargs)


def _entry_state(project, name):
    if name == "ais3_crackme":
        return project.factory.entry_state(args=[project.filename, claripy.BVS("argv1", 100 * 8)])
    return project.factory.entry_state()


def _run(name, path, per_state):
    if not os.path.exists(path):
        pytest.skip(f"{name} binary not found: {path}")
    project = angr.Project(path, auto_load_libs=False)
    backend = RustBackend(project)
    batch_sizes = []
    step_states = backend.step_states

    def counting(simgr, states, **run_args):
        batch_sizes.append(len(states))
        return step_states(simgr, states, **run_args)

    backend.step_states = counting
    simgr = angr.SimulationManager(project, [_entry_state(project, name)], backend=backend)
    if per_state:
        simgr.use_technique(_PerState())
    simgr.run(n=STEPS)
    return simgr, backend, batch_sizes


def _summary(simgr):
    return {
        name: [(s.addr, s.history.depth, s.posix.dumps(0), s.posix.dumps(1), len(s.solver.constraints)) for s in states]
        for name, states in simgr.stashes.items()
        if states
    }


@pytest.mark.parametrize(("name", "path"), [("fauxware", FAUXWARE_PATH), ("ais3_crackme", AIS3_PATH)])
def test_batched_and_per_state_steps_leave_the_same_stashes(name, path):
    batched, batched_backend, batched_sizes = _run(name, path, per_state=False)
    per_state, per_state_backend, per_state_sizes = _run(name, path, per_state=True)

    assert batched_backend.stats["claimed"] > 0
    assert batched_backend.stats == per_state_backend.stats
    assert max(batched_sizes) > 1
    assert set(per_state_sizes) == {1}
    assert sum(batched_sizes) == sum(per_state_sizes)
    assert _summary(batched) == _summary(per_state)
    assert [(type(r.error), r.state.addr) for r in batched.errored] == [
        (type(r.error), r.state.addr) for r in per_state.errored
    ]


def test_an_unselected_state_keeps_its_place_among_the_batched_successors():
    project = angr.Project(FAUXWARE_PATH, auto_load_libs=False)
    simgr = angr.SimulationManager(project, [project.factory.entry_state()], backend=RustBackend(project))
    simgr.run(until=lambda sm: len(sm.active) > 1)
    st1, st2 = simgr.active[:2]
    simgr.step(selector_func=lambda s: s is st1)
    assert simgr.active[1] is st2
    assert simgr.active[0] is not st1
