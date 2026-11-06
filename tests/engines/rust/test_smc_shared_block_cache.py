"""One state's patched code never runs in a sibling that did not patch it (angr-vmi2l.163).

Seed A (rdi=0) rewrites the immediate of ``mov eax,1`` to 2 and then runs it;
seed B (rdi=1) runs the instruction unpatched. The IRSB cache is shared by every
state, and SMC tracking used to last one step, so whichever seed lifted the block
first decided what the other one ran. Python, with ``support_selfmodifying_code``,
gives A=2 and B=1 in either seed order.
"""

from __future__ import annotations

import struct

import pytest

import angr
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
TARGET = LOAD + 0x0F
CODE = (
    b"\x48\x85\xff"  # test rdi,rdi
    + b"\x75\x08"  # jnz TARGET
    + b"\xc6\x04\x25"
    + struct.pack("<I", TARGET + 1)
    + b"\x02"  # mov byte [TARGET+1],2
    + b"\xeb\x00"  # jmp TARGET
    + b"\xb8\x01\x00\x00\x00"  # TARGET: mov eax,1
    + b"\xeb\xfe"  # jmp $
)
FIND = TARGET + 5
OPTS = {angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS, angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY}


def _run(engine, rdis, pad):
    code = CODE.ljust(0x2000, b"\x90") if pad else CODE
    project = angr.load_shellcode(code, arch="amd64", load_address=LOAD, support_selfmodifying_code=True)
    states = []
    for rdi in rdis:
        state = project.factory.blank_state(addr=LOAD, add_options=OPTS)
        state.regs.rdi = rdi
        states.append(state)
    if engine == "rust":
        mgr = RustExplorationManager(project, states)
        mgr.explore(find=FIND, num_find=len(rdis), max_steps=20)
    else:
        mgr = project.factory.simulation_manager(states)
        mgr.explore(find=FIND)
    return sorted((s.solver.eval(s.regs.rdi), s.solver.eval(s.regs.eax)) for s in mgr.found)


@pytest.mark.parametrize("pad", [False, True], ids=["one-page", "padded"])
@pytest.mark.parametrize("rdis", [(0, 1), (1, 0)], ids=["patcher-first", "patcher-second"])
def test_patched_code_stays_with_the_state_that_patched_it(rdis, pad):
    expected = _run("python", rdis, pad)
    assert expected == [(0, 2), (1, 1)]
    assert _run("rust", rdis, pad) == expected
