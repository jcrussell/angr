"""RustBackend.step_state: claimed states step in Rust and come back as SimStates (angr-mx00p.136.6).

On fauxware for x86_64, armel and mips, an explore with the backend must find
the Python engine's stdin, every stash entry must be a SimState, the stash
operations a user script does on the found state must work, and run(n=10)
from main must leave the Python engine's active addresses. A SimState held
across step() keeps its pre-step contents, as on the Python engine
(angr-mx00p.136.7).
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


def _held_after_constraints(project, backend):
    """A simgr whose active[0] carries constraints and is not at a hooked address.

    An exit procedure's step leaves its state at the hook (angr-mx00p.136.30), so such a state would not move.
    """
    simgr = angr.SimulationManager(project, [project.factory.entry_state()], backend=backend)
    for _ in range(200):
        simgr.step()
        if simgr.active and simgr.active[0].solver.constraints and not project.is_hooked(simgr.active[0].addr):
            break
    assert simgr.active and simgr.active[0].solver.constraints
    return simgr


def test_a_held_state_keeps_its_pre_step_contents(project):
    # On the Python engine step() replaces each state with new successor objects (angr-mx00p.136.7).
    backend = RustBackend(project)
    simgr = _held_after_constraints(project, backend)
    held = simgr.active[0]
    addr, sp, constraints = held.addr, held.solver.eval(held.regs.sp), list(held.solver.constraints)
    simgr.step()
    assert backend.stats["claimed"] > 0
    assert held.addr == addr
    assert held.solver.eval(held.regs.sp) == sp
    assert [str(c) for c in held.solver.constraints] == [str(c) for c in constraints]
    assert all(s is not held for s in simgr.active)
    assert any(s.addr != addr for s in simgr.active)


def test_a_write_to_a_held_state_does_not_reach_its_successor(project):
    backend = RustBackend(project)
    simgr = _held_after_constraints(project, backend)
    held = simgr.active[0]
    simgr.step()
    succ_sp = [s.solver.eval(s.regs.sp) for s in simgr.active]
    held.regs.sp = 0x1234
    assert [s.solver.eval(s.regs.sp) for s in simgr.active] == succ_sp
    assert held.solver.eval(held.regs.sp) == 0x1234


def test_a_copy_then_step_leaves_the_copy_unchanged(project):
    backend = RustBackend(project)
    simgr = _held_after_constraints(project, backend)
    copy = simgr.active[0].copy()
    addr, sp = copy.addr, copy.solver.eval(copy.regs.sp)
    simgr.step()
    assert copy.addr == addr
    assert copy.solver.eval(copy.regs.sp) == sp
    assert all(s is not copy for s in simgr.active)
