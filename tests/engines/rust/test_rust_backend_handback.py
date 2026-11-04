"""RustBackend mid-run hand-back: a state the Rust engine stops supporting steps on Python (angr-mx00p.136.12).

On fauxware from main, a backend-built manager steps a few blocks in Rust;
then one change makes the next step decline: a breakpoint on an event Rust
does not dispatch, a ``num_inst=`` step, a technique overriding
``successors()``, a refused option added to the state. A custom UberEngine
subclass declines from the first step. Each case asserts the decline reason
counter and that the run ends where the Python engine's does.
"""

from __future__ import annotations

import pytest

import angr
from angr.exploration.rust_backend import RustBackend
from angr.exploration.rust_manager import resolve_main_address, rust_unsupported_options
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, load_fauxware

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")

RUST_STEPS = 5
PYTHON_STEPS = 10


@pytest.fixture(scope="module")
def project():
    return load_fauxware()


def _shape(simgr):
    return {
        name: sorted((s.addr, s.posix.dumps(0), s.posix.dumps(1)) for s in states)
        for name, states in simgr.stashes.items()
        if states and not name.startswith("_")
    }


def _run(project, backend, change, step=lambda simgr: simgr.step()):
    """Step RUST_STEPS times, apply *change* to the manager, then step PYTHON_STEPS times with *step*."""
    simgr = angr.SimulationManager(
        project, [project.factory.blank_state(addr=resolve_main_address(project))], backend=backend
    )
    for _ in range(RUST_STEPS):
        simgr.step()
    if backend is not None:
        assert backend.stats["claimed"] > 0
        assert backend.stats["declined"] == {}
    change(simgr)
    for _ in range(PYTHON_STEPS):
        step(simgr)
    return simgr


def _assert_handed_back(backend, reason):
    assert set(backend.stats["declined"]) == {reason, "lineage was handed back to python"}
    assert backend.stats["handed_back"] > 0


def test_inspect_breakpoint_runs_on_python(project):
    def run(backend):
        hits = []

        def change(simgr):
            for state in simgr.active:
                state.inspect.b("simprocedure", when=angr.BP_BEFORE, action=lambda s: hits.append(s.addr))

        return _shape(_run(project, backend, change)), hits

    backend = RustBackend(project)
    (shape, hits), (python_shape, python_hits) = run(backend), run(None)
    assert shape == python_shape
    assert hits and hits == python_hits
    _assert_handed_back(backend, "state.inspect events not dispatched by the rust engine: simprocedure")


def test_num_inst_step_declines_for_that_call(project):
    def run(backend):
        calls = []

        def step(simgr):
            if not calls:
                simgr.step(num_inst=1)
            else:
                simgr.step()
            calls.append(1)

        return _shape(_run(project, backend, lambda simgr: None, step))

    backend = RustBackend(project)
    assert run(backend) == run(None)
    # The lineage the num_inst step handed back stays on Python.
    _assert_handed_back(backend, "step passes run arguments: num_inst")
    assert backend.stats["declined"]["step passes run arguments: num_inst"] == 1


class _CountingSuccessors(angr.ExplorationTechnique):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def successors(self, simgr, state, **kwargs):
        self.calls += 1
        return simgr.successors(state, **kwargs)


def test_successors_technique_declines(project):
    def run(backend):
        tech = _CountingSuccessors()
        shape = _shape(_run(project, backend, lambda simgr: simgr.use_technique(tech)))
        return shape, tech.calls

    backend = RustBackend(project)
    (shape, calls), (python_shape, python_calls) = run(backend), run(None)
    assert shape == python_shape
    assert calls > 0 and calls == python_calls
    _assert_handed_back(backend, "techniques override successors(): _CountingSuccessors")


class _CountingEngine(angr.engines.UberEngine):
    calls = 0

    def process_successors(self, successors, **kwargs):
        type(self).calls += 1
        return super().process_successors(successors, **kwargs)


def test_custom_engine_declines():
    proj = load_fauxware(engine=_CountingEngine)
    main = resolve_main_address(proj)

    def run(backend):
        simgr = angr.SimulationManager(proj, [proj.factory.blank_state(addr=main)], backend=backend)
        _CountingEngine.calls = 0
        simgr.run(n=RUST_STEPS + PYTHON_STEPS)
        return _shape(simgr), _CountingEngine.calls

    backend = RustBackend(proj)
    (shape, calls), (python_shape, python_calls) = run(backend), run(None)
    assert shape == python_shape
    assert calls > 0 and calls == python_calls
    assert backend.stats["claimed"] == 0
    assert set(backend.stats["declined"]) == {"project uses a custom SimEngine _CountingEngine"}


def test_refused_option_added_mid_run_declines(project):
    option = angr.options.TRACK_JMP_ACTIONS
    assert "TRACK_JMP_ACTIONS" in rust_unsupported_options({option})

    def run(backend):
        def change(simgr):
            for state in simgr.active:
                state.options.add(option)

        return _shape(_run(project, backend, change))

    backend = RustBackend(project)
    assert run(backend) == run(None)
    _assert_handed_back(backend, "state options not honored by the rust engine: TRACK_JMP_ACTIONS")
