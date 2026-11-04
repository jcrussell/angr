"""RustBackend.step_state: claimed states step in Rust and come back as SimStates (angr-mx00p.136.6).

On fauxware for x86_64, armel and mips, an explore with the backend must find
the Python engine's stdin, every stash entry must be a SimState, the stash
operations a user script does on the found state must work, and run(n=10)
from main must leave the Python engine's active addresses.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr.exploration.rust_backend import RustBackend
from angr.exploration.rust_manager import resolve_main_address
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, TEST_BINARIES_DIR

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")

ARCHES = ("x86_64", "armel", "mips")


def _project(arch):
    path = os.path.join(os.path.dirname(TEST_BINARIES_DIR), arch, "fauxware")
    if not os.path.exists(path):
        pytest.skip(f"fauxware binary not found: {path}")
    return angr.Project(path, auto_load_libs=False)


@pytest.fixture(scope="module", params=ARCHES)
def project(request):
    return _project(request.param)


def _explore(project, backend):
    simgr = angr.SimulationManager(project, [project.factory.entry_state()], backend=backend)
    simgr.explore(find=lambda s: b"Welcome" in s.posix.dumps(1), avoid=lambda s: b"Go away" in s.posix.dumps(1))
    return simgr


def test_explore_finds_the_python_stdin(project):
    backend = RustBackend(project)
    simgr = _explore(project, backend)
    python = _explore(project, None)
    assert backend.stats["claimed"] > 0
    assert [s.posix.dumps(0) for s in simgr.found] == [s.posix.dumps(0) for s in python.found]
    for name, states in simgr.stashes.items():
        assert all(isinstance(s, angr.SimState) for s in states), name

    simgr.move(from_stash="found", to_stash="win")
    (win,) = simgr.win
    assert b"SOSNEAKY" in win.posix.dumps(0)
    simgr.split(from_stash="active", limit=1)
    assert len(simgr.active) <= 1


def test_run_n_matches_python_active_addrs(project):
    main = resolve_main_address(project)

    def run(backend):
        simgr = angr.SimulationManager(project, [project.factory.blank_state(addr=main)], backend=backend)
        simgr.run(n=10)
        return sorted(s.addr for s in simgr.active)

    backend = RustBackend(project)
    assert run(backend) == run(None)
    assert backend.stats["claimed"] > 0
    assert backend.stats["declined"] == {}


def test_a_copy_of_an_export_is_imported():
    project = _project("x86_64")
    main = resolve_main_address(project)
    backend = RustBackend(project)
    simgr = angr.SimulationManager(project, [project.factory.blank_state(addr=main)], backend=backend)
    simgr.step()
    (child,) = simgr.active
    before = child.addr
    # A copy reads the parked state without being it: stepping the copy must leave the original where it was.
    simgr.move(from_stash="active", to_stash="held")
    simgr.populate("active", [child.copy()])
    simgr.step()
    (stepped,) = simgr.active
    assert stepped.addr != before
    assert simgr.held[0].addr == before
    assert backend.stats["claimed"] == 2
