"""An annotated concrete store in seed memory reaches Rust as its value (angr-vmi2l.224).

UltraPage keeps an annotated BVV in ``symbolic_data`` and leaves zeros in
``concrete_data``; the seed sync sent ``concrete_data`` and the symbolic
import skipped the object (it has no symbolic variable), so the Rust engine
read zero. Each case loads the value into ``eax`` and compares both engines.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
HEAP = 0x500010
SPIN = b"\xeb\xfe"  # jmp $


class _Taint(claripy.Annotation):
    eliminatable = property(lambda self: False)
    relocatable = property(lambda self: True)


def _run(where: str, engine: str):
    if where == "stack":
        load = b"\x8b\x44\x24\xc0"  # mov eax, [rsp-0x40]
    else:
        addr = HEAP if where == "heap" else LOAD + 0x20
        load = b"\x8b\x04\x25" + struct.pack("<I", addr)  # mov eax, [addr]
    code = load + SPIN
    project = angr.load_shellcode(code + b"\0" * 0x40, arch="AMD64", load_address=LOAD)
    state = project.factory.blank_state(addr=LOAD)
    if where == "stack":
        addr = state.solver.eval(state.regs.rsp) - 0x40
    state.memory.store(addr - 4, claripy.BVV(0x11223344, 32), endness="Iend_LE")
    state.memory.store(addr, claripy.BVV(7, 32).annotate(_Taint()), endness="Iend_LE")
    state.memory.store(addr + 4, claripy.BVS("x", 8))
    find = LOAD + len(load)
    mgr = RustExplorationManager(project, [state]) if engine == "rust" else project.factory.simulation_manager(state)
    mgr.explore(find=find)
    assert len(mgr.found) == 1
    found = mgr.found[0]
    return found.solver.eval(found.regs.eax), found.solver.eval(found.memory.load(addr - 4, 8))


@pytest.mark.parametrize("where", ["heap", "stack", "loader"])
def test_annotated_seed_constant_matches_python(where):
    py = _run(where, "python")
    assert py == (7, 0x4433221107000000)
    assert _run(where, "rust") == py
