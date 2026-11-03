"""A symbolic read over more than 256 in-range addresses stays symbolic (angr-vmi2l.248).

The shellcode loads ``buf[x]`` with ``x < N``, where only ``buf[N-3]`` is 0x7f,
and branches on it. Python's read chain ``Range(1024) -> Any`` keeps every
address whose span is within 1024, so both branches survive. Rust enumerated at
most ``max_solutions`` (256) and reported the overflow as ``TooLarge``, which the
read fallback pinned to one model value, so for N > 256 the hit branch vanished.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager

BUF = 0x60000000
# movzx eax, byte [rdi+rsi]; cmp al, 0x7f; je +2; jmp $; jmp $
CODE = bytes.fromhex("0fb60437" + "3c7f" + "7402" + "ebfe" + "ebfe")
HIT_PC = 0x40000A
MISS_PC = 0x400008


def _state(n):
    proj = angr.load_shellcode(CODE, arch="amd64", load_address=0x400000)
    state = proj.factory.blank_state(addr=0x400000, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS})
    buf = bytearray(n)
    buf[n - 3] = 0x7F
    state.memory.store(BUF, bytes(buf))
    x = claripy.BVS("x", 64)
    state.regs.rdi = BUF
    state.regs.rsi = x
    state.solver.add(x.ULT(n))
    return proj, state, x


def _successors(states, x):
    return {s.solver.eval_one(s.regs.rip): s.solver.eval_upto(x, 3) for s in states}


@pytest.mark.parametrize("n", [256, 300, 1000])
def test_rust_keeps_both_branches_like_python(n):
    proj, state, x = _state(n)
    simgr = proj.factory.simulation_manager(state.copy())
    simgr.step()
    expected = _successors(simgr.active, x)
    assert set(expected) == {HIT_PC, MISS_PC}

    mgr = RustExplorationManager(proj, [state])
    mgr.step()
    got = _successors(mgr.active, x)
    assert set(got) == {HIT_PC, MISS_PC}
    assert got[HIT_PC] == [n - 3]
    assert len(got[MISS_PC]) == 3, "the miss branch must not pin x"
