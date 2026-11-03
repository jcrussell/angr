"""A symbolic-index load over pages Rust does not hold keeps every candidate.

Rust used to drop each candidate whose page it had not mapped, assert the
address into the survivors, and read a name-interned ``mem_all_unmapped_<n>``
symbol when none survived, so two unrelated reads were provably equal and
ZERO_FILL_UNCONSTRAINED_MEMORY was ignored (angr-vmi2l.177). Both engines
must agree.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
# and rsi,3 ; movzx eax, byte [rdi+rsi] ; movzx ebx, byte [rdx+rsi] ; jmp $
CODE = bytes.fromhex("4883e6030fb604370fb61c32ebfe")


def _found(engine, case):
    p = angr.load_shellcode(CODE, arch="amd64", load_address=LOAD)
    opts = {angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
    if case == "zero_fill":
        opts.add(angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY)
    s = p.factory.blank_state(addr=LOAD, add_options=opts)
    s.regs.rsi = claripy.BVS("idx", 64)
    if case == "straddle":
        # Two candidates below the stack page, two on it.
        s.regs.rdi = (s.solver.eval(s.regs.rsp) & ~0xFFF) - 2
    else:
        s.regs.rdi = 0x50000000
    s.regs.rdx = 0x60000000
    mgr = RustExplorationManager(p, [s]) if engine == "rust" else p.factory.simulation_manager(s)
    mgr.explore(find=LOAD + len(CODE) - 2)
    assert len(mgr.found) == 1
    return mgr.found[0]


@pytest.mark.parametrize("engine", ["python", "rust"])
@pytest.mark.parametrize("case", ["straddle", "unmapped"])
def test_every_index_stays_feasible_and_reads_are_independent(engine, case):
    found = _found(engine, case)
    assert sorted(found.solver.eval_upto(found.regs.rsi & 3, 8)) == [0, 1, 2, 3]
    assert found.solver.satisfiable(extra_constraints=[found.regs.eax != found.regs.ebx])


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_zero_fill_reads_zero(engine):
    found = _found(engine, "zero_fill")
    assert found.solver.eval_upto(found.regs.eax, 2) == [0]
    assert found.solver.eval_upto(found.regs.ebx, 2) == [0]
