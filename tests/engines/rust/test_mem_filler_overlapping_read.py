"""A read that starts inside an earlier filler mints a filler for the rest (angr-vmi2l.329).

``mov ebx, [rdi]; mov eax, [rdi+2]`` on never-written memory: the second read's
low half is the first filler's top half, and its high half is bytes nobody has
read, so both engines give it a fresh filler. Python's ``UltraPage.load`` once
measured the first object's end from the read address, which returned a 16-bit
value for the 4-byte read and left ``eax``'s upper half concrete 0.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
X = 0x70000020

# mov ebx, [rdi]; jmp +0; mov eax, [rdi+2]; jmp $
CODE = b"\x8b\x1f" + b"\xeb\x00" + b"\x8b\x47\x02" + b"\xeb\xfe"
FIN = LOAD + len(CODE) - 2


def _found(engine: str):
    proj = angr.load_shellcode(CODE, arch="amd64", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS})
    state.regs.rdi = X
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=FIN)
    assert len(mgr.found) == 1
    return mgr.found[0]


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_read_past_a_filler_gets_a_fresh_filler(engine):
    found = _found(engine)
    ebx, eax = found.regs.ebx, found.regs.eax
    # The overlap is the same bytes.
    assert not found.solver.satisfiable(extra_constraints=[ebx[31:16] != eax[15:0]])
    # The unread bytes are free, not concrete 0.
    assert found.solver.satisfiable(extra_constraints=[eax[31:16] != 0])
    assert found.solver.satisfiable(extra_constraints=[eax[31:16] == 0])


def test_ultra_page_load_at_an_offset_into_an_object():
    proj = angr.load_shellcode(b"\xc3", arch="amd64", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD)
    state.memory.store(X, claripy.BVS("x", 32), endness="Iend_LE")
    value = state.memory.load(X + 2, 4, endness="Iend_LE")
    assert value.size() == 32
    assert not state.solver.satisfiable(
        extra_constraints=[value[15:0] != state.memory.load(X + 2, 2, endness="Iend_LE")]
    )
