"""The phase-2 eager retry reports each path once (angr-vmi2l.157).

An address-based explore that exhausts phase 1 (deferred forks) below
`num_find` re-seeds the roots in eager mode. Phase 2 re-explores every path
phase 1 did, so phase 1's terminal stashes used to be reported again under
fresh state ids. Each case is compared with the Python engine.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager

FAUXWARE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "binaries", "tests", "x86_64", "fauxware")
FAUXWARE_ACCEPT = 0x4006ED
UNREACHED = 0xDEAD0000
LOAD = 0x400000


def _counts(mgr) -> dict[str, int]:
    names = ("found", "avoid", "deadended", "errored", "unconstrained")
    return {k: len(getattr(mgr, k)) for k in names if len(getattr(mgr, k))}


def _manager(engine, proj, state):
    return RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)


@pytest.fixture(scope="module")
def fauxware():
    return angr.Project(FAUXWARE, auto_load_libs=False)


@pytest.mark.parametrize("find", [FAUXWARE_ACCEPT, UNREACHED], ids=["accept", "unreached"])
def test_fauxware_find_all_matches_python(fauxware, find):
    """Find-all, and a find that is never reached so every path deadends."""
    expected = _counts(_manager("py", fauxware, fauxware.factory.entry_state()).explore(find=find, num_find=10**9))
    mgr = _manager("rust", fauxware, fauxware.factory.entry_state())
    mgr.explore(find=find, num_find=None)
    assert _counts(mgr) == expected
    stdins = [s.posix.dumps(0) for s in mgr.found]
    assert len(stdins) == len(set(stdins)), f"duplicate found paths: {stdins}"


def test_avoid_not_doubled():
    """An unreached find re-seeds phase 2; the avoid state is reported once."""
    # cmp rsi, 10; ja +2; jmp $ (avoid, LOAD+6); jmp $ (find, LOAD+8)
    code = bytes.fromhex("4883fe0a") + b"\x77\x02" + b"\xeb\xfe" + b"\xeb\xfe"
    proj = angr.load_shellcode(code, arch="amd64", load_address=LOAD)

    def explore(engine):
        state = proj.factory.blank_state(addr=LOAD, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS})
        x = claripy.BVS("x", 64)
        state.solver.add(x.ULT(5))
        state.regs.rsi = x
        return _manager(engine, proj, state).explore(find=LOAD + 8, avoid=LOAD + 6)

    assert _counts(explore("rust")) == _counts(explore("py")) == {"avoid": 1}


def test_states_from_before_the_explore_survive(fauxware):
    """The retry drops only what this explore added, not a prior run's stashes."""
    mgr = _manager("rust", fauxware, fauxware.factory.entry_state())
    mgr.explore(find=UNREACHED, num_find=None)
    before = set(mgr._rust_mgr.get_state_ids("deadended"))
    assert before
    mgr._phase2_retried = False
    mgr.explore(find=UNREACHED, num_find=None)
    assert before <= set(mgr._rust_mgr.get_state_ids("deadended"))
