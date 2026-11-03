"""A read of memory neither side holds stores its filler, so memory stays a function.

Rust used to hand back a ``mem_<addr>_<size>`` filler without storing it, so
a narrower or shifted re-read of the same bytes was an unrelated symbol and
the found state's memory held a third one (angr-vmi2l.179). Python's
``DefaultFillerMixin`` stores its filler; both engines must agree.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
FIRST = bytes.fromhex("8b1f")  # mov ebx, [rdi]
SECOND = {
    "wide_then_narrow": bytes.fromhex("8a4701"),  # mov al, [rdi+1]
    "shift": bytes.fromhex("8b4702"),  # mov eax, [rdi+2]
}
# ebx bits that the second read's low bits must equal
OVERLAP = {"wide_then_narrow": (15, 8, 7, 0), "shift": (31, 16, 15, 0)}


def _found(engine, case, x):
    code = FIRST + bytes.fromhex("eb00") + SECOND[case] + bytes.fromhex("ebfe")
    fin = LOAD + len(code) - 2
    p = angr.load_shellcode(code, arch="amd64", load_address=LOAD)
    s = p.factory.blank_state(addr=LOAD, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS})
    s.regs.rdi = x
    s.regs.rax = 0
    if engine == "rust":
        mgr = RustExplorationManager(p, [s])
        mgr.explore(find=fin)
    else:
        mgr = p.factory.simulation_manager(s)
        mgr.explore(find=fin)
    assert len(mgr.found) == 1
    return mgr.found[0]


def _valid(state, cond):
    return not state.solver.satisfiable(extra_constraints=[claripy.Not(cond)])


@pytest.mark.parametrize("engine", ["python", "rust"])
@pytest.mark.parametrize("case", sorted(SECOND))
@pytest.mark.parametrize("x", [0x70000020, 0xC0800020], ids=["unmapped", "heap"])
def test_overlapping_reads_of_one_uninitialized_byte_agree(engine, case, x):
    found = _found(engine, case, x)
    ebx, eax = found.regs.ebx, found.regs.eax
    hi, lo, ehi, elo = OVERLAP[case]
    assert _valid(found, ebx[hi:lo] == eax[ehi:elo]), (ebx, eax)
    assert _valid(found, found.memory.load(x, 4, endness="Iend_LE") == ebx), found.memory.load(x, 4)
