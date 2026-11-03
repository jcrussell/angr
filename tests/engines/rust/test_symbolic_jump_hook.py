"""A hook that jumps to a symbolic target resumes pinned copies (angr-vmi2l.292).

The hook at LOAD+2 does ``self.jump(rdi)``. ``SimSuccessors`` files that one
successor as one ``flat_successors`` copy per feasible target, each with
``ip == target`` added, or as an unconstrained successor when there are too
many targets. The callback dispatcher used to resume ``all_successors[0]``
at an arbitrary ``eval()`` of the IP without pinning it, so found and avoided
states both still admitted every target, and a free target deadended at 0
instead of landing in ``unconstrained``.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
# LOAD: nop; nop | LOAD+2 (hooked): nop; nop | LOAD+4: jmp $ | LOAD+6: nop; nop | LOAD+8: jmp $
CODE = bytes.fromhex("9090" + "9090" + "ebfe" + "9090" + "ebfe")


class _JumpToRdi(angr.SimProcedure):
    def run(self):  # pylint: disable=arguments-differ
        self.jump(self.state.regs.rdi)


def _setup(free: bool):
    proj = angr.load_shellcode(CODE, arch="amd64", load_address=LOAD)
    proj.hook(LOAD + 2, _JumpToRdi(), length=2)
    state = proj.factory.blank_state(addr=LOAD)
    x = claripy.BVS("x", 64)
    state.regs.rdi = x
    if not free:
        state.solver.add(claripy.Or(x == LOAD + 4, x == LOAD + 8))
    return proj, state, x


def _manager(engine: str, proj, state):
    if engine == "python":
        return proj.factory.simulation_manager(state)
    return RustExplorationManager(proj, [state])


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_found_and_avoided_states_each_pin_their_target(engine):
    proj, state, x = _setup(free=False)
    mgr = _manager(engine, proj, state)
    mgr.explore(find=LOAD + 8, avoid=LOAD + 4)
    assert [s.solver.eval_upto(x, 3) for s in mgr.found] == [[LOAD + 8]]
    assert [s.solver.eval_upto(x, 3) for s in mgr.avoid] == [[LOAD + 4]]


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_stepping_splits_the_jump_into_pinned_actives(engine):
    proj, state, x = _setup(free=False)
    mgr = _manager(engine, proj, state)
    mgr.step()
    mgr.step()
    pinned = sorted((s.addr, tuple(s.solver.eval_upto(x, 3))) for s in mgr.active)
    assert pinned == [(LOAD + 4, (LOAD + 4,)), (LOAD + 8, (LOAD + 8,))]


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_free_target_goes_to_unconstrained(engine):
    proj, state, _ = _setup(free=True)
    mgr = _manager(engine, proj, state)
    for _ in range(3):
        mgr.step()
    counts = {name: len(states) for name, states in mgr.stashes.items() if states}
    assert counts == {"unconstrained": 1}
