"""A symbolic conditional exit's successors are stashed in Python's order (angr-mx00p.132.59).

``HeavyVEXMixin`` adds one successor per ``Ist_Exit`` in statement order and
the default exit last, so the exit target comes first. Rust continues on one
side and defers the other; it used to list the continuing state first, so on a
forward branch ``active[0]`` (and ``one_deadended`` downstream, as in
test_unicorn's test_stops) was the other arm.
"""

from __future__ import annotations

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

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
