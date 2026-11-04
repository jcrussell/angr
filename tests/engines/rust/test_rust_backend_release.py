"""RustBackend frees a Rust state once no export of it is referenced (angr-mx00p.136.9).

Python owns stash membership, so a state dropped from every stash, or the
source a step consumed, must not stay in the private Rust manager. A held
export keeps its Rust state and stays readable.
"""

from __future__ import annotations

import gc

import pytest

import angr
from angr import claripy
from angr.exploration.rust_backend import RustBackend
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")

BASE = 0x400000
BUF = 0x600000
# top: mov al, [rsi]; inc rsi; cmp al, 0x41; je top; mov eax, 60; syscall
# Every pass forks: one state loops on a fresh symbolic byte, the other exits.
LOOP = bytes.fromhex("8a0648ffc63c4174f7b83c0000000f05")


def _loop_simgr(steps):
    project = angr.load_shellcode(LOOP, arch="amd64", load_address=BASE, simos="linux")
    state = project.factory.blank_state(addr=BASE)
    state.regs.rsi = BUF
    state.memory.store(BUF, claripy.BVS("buf", 8 * (steps + 16)))
    backend = RustBackend(project)
    return angr.SimulationManager(project, [state], backend=backend), backend


def _rust_population(backend):
    return sum(backend._mgr._rust_mgr.stash_counts().values())


def _live_population(simgr):
    return sum(len(states) for states in simgr.stashes.values())


def test_a_forking_loop_keeps_the_rust_population_bounded():
    steps = 2000
    simgr, backend = _loop_simgr(steps)
    for i in range(1, steps + 1):
        simgr.step()
        simgr.drop(stash="deadended")
        if i % 200 == 0:
            # The consumed sources sit in reference cycles until a collection.
            gc.collect()
            simgr.step()
            simgr.drop(stash="deadended")
            assert _rust_population(backend) <= 2 * _live_population(simgr), i
    assert backend.stats["claimed"] >= steps
    assert len(backend._mgr._state_cache) == 0


def test_a_held_export_dropped_from_its_stash_stays_readable():
    simgr, _ = _loop_simgr(16)
    simgr.step()
    held = simgr.active[0]
    addr, rsi = held.addr, held.solver.eval(held.regs.rsi)
    simgr.drop(stash="active")
    gc.collect()
    simgr.populate("active", [held.copy()])
    simgr.step()
    gc.collect()
    simgr.step()
    assert (held.addr, held.solver.eval(held.regs.rsi)) == (addr, rsi)


def test_a_dropped_export_releases_its_rust_state():
    simgr, backend = _loop_simgr(16)
    seed = simgr.active[0].copy()
    simgr.step()
    simgr.step()
    assert backend._resident
    simgr.drop(stash="active")
    gc.collect()
    # The release is applied before the next step; then only that step's children remain.
    simgr.populate("active", [seed])
    simgr.step()
    simgr.drop(stash="deadended")
    gc.collect()
    assert len(backend._resident) == _live_population(simgr) > 0
