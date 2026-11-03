"""``mgr.stashes`` lists every stash, as ``SimulationManager.stashes`` does (angr-9ij0m).

It used to enumerate a fixed list of standard names, so a technique-declared
stash (LengthLimiter's ``cut``, LoopSeer's ``spinning``, a custom
``move()`` target) raised KeyError, and its values were state ids, not states.
"""

from __future__ import annotations

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000

# A 2-block infinite loop: inc rax; jmp +0xb / ...; inc rbx; jmp back.
TWO_BLOCK_LOOP = bytearray(b"\x90" * 0x20)
TWO_BLOCK_LOOP[0:5] = b"\x48\xff\xc0\xeb\x0b"
TWO_BLOCK_LOOP[0x10:0x15] = b"\x48\xff\xc3\xeb\xeb"


def _manager(engine: str):
    proj = angr.load_shellcode(bytes(TWO_BLOCK_LOOP), arch="amd64", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD)
    if engine == "rust":
        return RustExplorationManager(proj, [state])
    return proj.factory.simulation_manager(state)


@pytest.mark.parametrize("engine", ["py", "rust"])
def test_technique_stash_is_listed_with_states(engine):
    mgr = _manager(engine)
    mgr.use_technique(angr.exploration_techniques.LengthLimiter(max_length=6))
    mgr.run(n=20)
    stashes = mgr.stashes
    assert "cut" in stashes
    (state,) = stashes["cut"]
    assert state.addr in (LOAD, LOAD + 0x10)
    assert stashes["active"] == []


@pytest.mark.parametrize("engine", ["py", "rust"])
def test_custom_stash_and_missing_name(engine):
    mgr = _manager(engine)
    mgr.move("active", "parked")
    stashes = mgr.stashes
    assert [s.addr for s in stashes["parked"]] == [LOAD]
    assert stashes["never_made"] == []


def test_internal_round_stashes_are_hidden():
    mgr = _manager("rust")
    mgr.step()
    assert not [name for name in mgr.stashes if name.startswith("_")]
