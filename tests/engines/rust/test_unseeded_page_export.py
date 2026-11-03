"""A found state's memory agrees with execution on pages the seed never mapped (angr-vmi2l.162).

(a) A store to a page neither side held read back in execution but exported
as a fresh symbol. (c) A deferred-fork sibling that never wrote an
uninitialized byte exported the taken path's store there. Both are compared
with the Python engine. (b), the never-written neighbour of such a store,
still reads 0 under Rust where Python mints a ``mem_`` filler (angr-vmi2l.325).
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
HEAP = 0x30000000
# test sil,1; jz L; mov [rdi],al; jmp FIN; L: mov bl,[rdi]; FIN: jmp $
FORK = b"\x40\xf6\xc6\x01\x74\x04\x88\x07\xeb\x02\x8a\x1f\xeb\xfe"
# mov [rdi],al; mov bl,[rdi]; jmp $
SAME = b"\x88\x07\x8a\x1f\xeb\xfe"
# mov [rdi],al; mov bl,[rdi+1]; jmp $
NEXT = b"\x88\x07\x8a\x5f\x01\xeb\xfe"


def _found(engine: str, code: bytes, where: str):
    proj = angr.load_shellcode(code, arch="amd64", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS})
    tgt = HEAP if where == "heap" else state.solver.eval(state.regs.rsp) - 0x100
    state.regs.rdi = tgt
    state.regs.rsi = claripy.BVS("sel", 64)
    state.regs.al = 0x41
    state.regs.rbx = 0x99
    end = LOAD + len(code) - 2
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=end, num_find=2)
    return list(mgr.found), tgt


def _equal(state, a, b) -> bool:
    return not state.solver.satisfiable(extra_constraints=[a != b])


@pytest.mark.parametrize("where", ["heap", "stack"])
@pytest.mark.parametrize("engine", ["py", "rust"])
def test_store_on_unseeded_page_exports(engine, where):
    (f,), tgt = _found(engine, SAME, where)
    assert f.solver.eval(f.regs.bl) == 0x41
    assert f.solver.eval_upto(f.memory.load(tgt, 1), 2) == [0x41]


@pytest.mark.parametrize("where", ["heap", "stack"])
@pytest.mark.parametrize("engine", ["py", "rust"])
def test_fork_sibling_keeps_its_own_uninitialized_byte(engine, where):
    found, tgt = _found(engine, FORK, where)
    assert len(found) == 2
    by_sel = {f.solver.eval(f.regs.rsi) & 1: f for f in found}
    stored, reader = by_sel[1], by_sel[0]
    assert stored.solver.eval_upto(stored.memory.load(tgt, 1), 2) == [0x41]
    mem = reader.memory.load(tgt, 1)
    assert mem.symbolic
    assert _equal(reader, reader.regs.bl, mem)
    assert reader.solver.satisfiable(extra_constraints=[mem != 0x41])


@pytest.mark.parametrize("where", ["heap", "stack"])
@pytest.mark.parametrize(
    "engine",
    ["py", pytest.param("rust", marks=pytest.mark.xfail(strict=True, reason="angr-vmi2l.325"))],
)
def test_unwritten_neighbour_of_a_store_is_a_filler(engine, where):
    (f,), tgt = _found(engine, NEXT, where)
    mem = f.memory.load(tgt + 1, 1)
    assert mem.symbolic
    assert _equal(f, f.regs.bl, mem)
