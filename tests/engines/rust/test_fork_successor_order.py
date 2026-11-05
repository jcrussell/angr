"""A symbolic conditional exit's successors are stashed in Python's order (angr-mx00p.132.59).

``HeavyVEXMixin`` adds one successor per ``Ist_Exit`` in statement order and
the default exit last, so the exit target comes first. Rust continues on one
side and defers the other; it used to list the continuing state first, so on a
forward branch ``active[0]`` (and ``one_deadended`` downstream, as in
test_unicorn's test_stops) was the other arm.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager
from tests.engines.conftest import BIN_LOCATION

LOAD = 0x400000
# Forward: cmp rdi,10; je +2; jmp $; jmp $  -> exit target 0x8, fall-through 0x6.
FORWARD = b"\x48\x83\xff\x0a\x74\x02\xeb\xfe\xeb\xfe"
# Backward: nop; cmp rdi,10; je 0; jmp $  -> exit target 0x0, fall-through 0x7.
BACKWARD = b"\x90\x48\x83\xff\x0a\x74\xf9\xeb\xfe"


@pytest.mark.parametrize(
    ("code", "start", "expected"),
    [(FORWARD, 0, [0x8, 0x6]), (BACKWARD, 1, [0x0, 0x7])],
    ids=["forward", "backward"],
)
@pytest.mark.parametrize("engine", ["py", "rust"])
def test_conditional_exit_successors_in_python_order(engine, code, start, expected):
    proj = angr.load_shellcode(code, arch="amd64", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD + start)
    if engine == "rust":
        mgr = RustExplorationManager(proj, [state])
        mgr.set_block_granular(True)
    else:
        mgr = proj.factory.simulation_manager(state)
    mgr.step()
    assert [s.addr - LOAD for s in mgr.active] == expected


@pytest.mark.parametrize("engine", ["py", "rust"])
def test_run_deadends_the_fork_arms_in_python_order(engine, monkeypatch):
    """A default-routed run() deadends the arms of uc_stop's symbolic exit as SimulationManager.run() does.

    A chained step ran the fall-through arm (0x8048591) on to its deadend while the exit-target arm (0x8048587)
    waited in active, so ``one_deadended`` was the other arm (angr-mx00p.132.61).
    """
    monkeypatch.setenv("ANGR_DEFAULT_ENGINE", "rust" if engine == "rust" else "python")
    proj = angr.Project(os.path.join(BIN_LOCATION, "tests", "i386", "uc_stop"), auto_load_libs=False)
    mgr = proj.factory.simulation_manager(proj.factory.entry_state(args=["a"] * 10))
    assert isinstance(mgr, RustExplorationManager) == (engine == "rust")
    mgr.run()
    arms = [a for s in mgr.deadended for a in s.history.bbl_addrs if a in (0x8048587, 0x8048591)]
    assert arms == [0x8048587, 0x8048591]
