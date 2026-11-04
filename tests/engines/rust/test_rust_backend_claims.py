"""RustBackend.claims(): the per-state, per-step decline predicate (angr-mx00p.136.5).

Each decline reason is driven with one state; an eligible state on fauxware
past the init prefix gets no decline reason and is claimed, and an explore
with the backend must equal the Python engine's.
"""

from __future__ import annotations

import logging

import pytest

import angr
from angr.exploration.rust_backend import RustBackend
from angr.exploration.rust_manager import resolve_main_address
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, load_fauxware

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")

FIND = 0x4006ED
AVOID = [0x4006AA, 0x4006FD]


@pytest.fixture(scope="module")
def project():
    return load_fauxware()


@pytest.fixture(scope="module")
def main_addr(project):
    return resolve_main_address(project)


def _reason(project, state, successor_func=None, run_args=None, simgr=None):
    simgr = simgr if simgr is not None else angr.SimulationManager(project, [state])
    return RustBackend(project).decline_reason(simgr, state, successor_func, run_args or {})


def test_eligible_state_at_main(project, main_addr):
    state = project.factory.blank_state(addr=main_addr)
    assert _reason(project, state) is None

    backend = RustBackend(project)
    simgr = angr.SimulationManager(project, [state])
    assert backend.claims(simgr, state, None, {}) == (True, "")
    assert backend.stats["declined"] == {}
    assert backend.stats["claimed"] == 1


def test_unsupported_arch():
    proj = angr.load_shellcode(b"\x60\x00\x00\x00", arch="PPC32")
    assert _reason(proj, proj.factory.blank_state()) == "arch PPC32 not implemented by the rust engine"


def test_refused_option(project, main_addr):
    state = project.factory.blank_state(addr=main_addr, add_options={angr.options.UNICORN})
    assert _reason(project, state) == "state options not honored by the rust engine: UNICORN"


def test_inspect_breakpoint_added_between_steps(project, main_addr):
    state = project.factory.blank_state(addr=main_addr)
    assert _reason(project, state) is None
    state.inspect.b("syscall", when=angr.BP_BEFORE, action=lambda s: None)
    assert _reason(project, state) == "state.inspect events not dispatched by the rust engine: syscall"


def test_successor_func(project, main_addr):
    state = project.factory.blank_state(addr=main_addr)
    assert _reason(project, state, successor_func=project.factory.successors) == "step passes a successor_func"


def test_run_args(project, main_addr):
    state = project.factory.blank_state(addr=main_addr)
    assert _reason(project, state, run_args={"num_inst": 1}) == "step passes run arguments: num_inst"


class _SuccessorsHook(angr.ExplorationTechnique):
    def successors(self, simgr, state, **kwargs):
        return simgr.successors(state, **kwargs)


def test_successors_technique(project, main_addr):
    state = project.factory.blank_state(addr=main_addr)
    simgr = angr.SimulationManager(project, [state])
    simgr.use_technique(angr.exploration_techniques.DFS())
    assert _reason(project, state, simgr=simgr) is None
    simgr.use_technique(_SuccessorsHook())
    assert _reason(project, state, simgr=simgr) == "techniques override successors(): _SuccessorsHook"


class _CustomEngine(angr.engines.UberEngine):
    pass


def test_custom_engine():
    proj = load_fauxware(engine=_CustomEngine)
    state = proj.factory.blank_state(addr=resolve_main_address(proj))
    assert _reason(proj, state) == "project uses a custom SimEngine _CustomEngine"


def test_init_prefix_until_main(project, main_addr):
    entry = project.factory.entry_state()
    backend = RustBackend(project)
    simgr = angr.SimulationManager(project, [entry], backend=backend)
    assert backend.decline_reason(simgr, entry, None, {}) == "state is in the python init prefix"

    simgr.explore(find=main_addr)
    (at_main,) = simgr.found
    assert backend.decline_reason(simgr, at_main, None, {}) is None
    # The lineage stays past the prefix after main returns into libc code.
    succ = project.factory.successors(at_main).flat_successors[0]
    assert backend.decline_reason(simgr, succ, None, {}) is None
    # Explorer steps with extra_stop_points, which the backend honors, so only the prefix declined.
    assert set(backend.stats["declined"]) == {"state is in the python init prefix"}


def test_stop_point_inside_the_block_declines(project, main_addr):
    state = project.factory.blank_state(addr=main_addr)
    end = main_addr + project.factory.block(main_addr).size
    assert _reason(project, state, run_args={"extra_stop_points": {main_addr, end}}) is None
    assert (
        _reason(project, state, run_args={"extra_stop_points": {main_addr + 1}})
        == "an extra_stop_points address lies inside the block"
    )


def test_loader_stub_seed_is_in_init_prefix(project):
    state = project.factory.full_init_state()
    assert _reason(project, state) == "state is in the python init prefix"


def test_handed_back_lineage_is_sticky(project, main_addr):
    state = project.factory.blank_state(addr=main_addr)
    backend = RustBackend(project)
    simgr = angr.SimulationManager(project, [state])
    backend.claims(simgr, state, None, {})
    assert backend.claims(simgr, state, project.factory.successors, {}) == (False, "step passes a successor_func")
    assert backend.stats["handed_back"] == 1

    succ = project.factory.successors(state).flat_successors[0]
    assert backend.claims(simgr, succ, None, {}) == (False, "lineage was handed back to python")
    assert backend.stats["handed_back"] == 1

    # A lineage the backend never owned is not handed back by a decline.
    other = project.factory.blank_state(addr=main_addr)
    backend.claims(simgr, other, project.factory.successors, {})
    assert backend.decline_reason(simgr, other, None, {}) is None


def test_decline_is_debug_logged(project, main_addr, caplog):
    state = project.factory.blank_state(addr=main_addr)
    simgr = angr.SimulationManager(project, [state])
    with caplog.at_level(logging.DEBUG, logger="angr.exploration.rust_backend"):
        RustBackend(project).claims(simgr, state, project.factory.successors, {})
    assert "simgr backend: python (step passes a successor_func)" in caplog.messages


def _explore(project, backend):
    simgr = angr.SimulationManager(project, [project.factory.entry_state()], backend=backend)
    simgr.explore(find=FIND, avoid=AVOID)
    return simgr


def test_explore_with_backend_equals_python(project):
    backend = RustBackend(project)
    with_backend = _explore(project, backend)
    python = _explore(project, None)

    def shape(simgr):
        return {name: sorted(s.addr for s in states) for name, states in simgr.stashes.items() if states}

    assert shape(with_backend) == shape(python)
    assert [s.posix.dumps(0) for s in with_backend.found] == [s.posix.dumps(0) for s in python.found]
    assert backend.stats["claimed"] > 0


def test_step_with_backend_equals_python(project, main_addr):
    def run(backend):
        simgr = angr.SimulationManager(project, [project.factory.blank_state(addr=main_addr)], backend=backend)
        for _ in range(6):
            simgr.step()
        return sorted(s.addr for s in simgr.active)

    backend = RustBackend(project)
    assert run(backend) == run(None)
    assert backend.stats["claimed"] > 0
    assert backend.stats["declined"] == {}
