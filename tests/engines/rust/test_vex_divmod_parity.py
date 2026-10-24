"""amd64 ``div`` (``Iop_DivModU{64to32,128to64}``) on the Rust engine agrees
with the Python engine when the quotient overflows the divisor's width
(angr-vmi2l.58).

libVEX guards only a zero divisor, so an overflowing quotient (which raises
#DE on hardware) reaches the op. Both engines truncate the quotient to the
low half; the remainder half must not pick up the quotient's high bits.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import (
    RUST_EXPLORATION_AVAILABLE,
    RustExplorationManager,
)

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)

BASE = 0x10000
JMP_SELF = bytes.fromhex("ebfe")

CASES = {
    "div_ecx": "f7f1",
    "div_rcx": "48f7f1",
}

# (rax, rdx, rcx): the bead's overflow repro, a larger overflow with a
# non-zero remainder, and an ordinary in-range division.
OPERANDS = {
    "overflow_by_one": (0, 1, 1),
    "overflow_with_remainder": (0x1234, 0x77, 0x10),
    "in_range": (0x1234, 0, 0x10),
}


def _state(project, operands, symbolic):
    state = project.factory.blank_state(addr=BASE)
    for name, val in zip(("rax", "rdx", "rcx"), operands):
        if symbolic and name == "rax":
            sym = claripy.BVS(name, 64)
            state.solver.add(sym == val)
            setattr(state.regs, name, sym)
        else:
            setattr(state.regs, name, val)
    return state


def _results(found):
    return [sorted(found.solver.eval_upto(getattr(found.regs, r), 2)) for r in ("rax", "rdx")]


@pytest.mark.parametrize("symbolic", [False, True], ids=["concrete", "symbolic"])
@pytest.mark.parametrize("operands", OPERANDS)
@pytest.mark.parametrize("insn", CASES)
def test_div_matches_python_engine(insn, operands, symbolic):
    code = bytes.fromhex(CASES[insn]) + JMP_SELF
    end = BASE + len(code) - len(JMP_SELF)
    vals = OPERANDS[operands]

    project = angr.load_shellcode(code, arch="amd64", load_address=BASE)
    python = project.factory.simulation_manager(_state(project, vals, symbolic))
    python.explore(find=end)
    assert python.found, "Python engine did not reach the end of the block"
    expected = _results(python.found[0])

    project = angr.load_shellcode(code, arch="amd64", load_address=BASE)
    mgr = RustExplorationManager(project, [_state(project, vals, symbolic)])
    mgr.explore(find=end, max_steps=5)
    assert len(mgr.found) == 1 and not mgr.errored, mgr.stash_counts()
    assert _results(mgr.found[0]) == expected


def test_div_ecx_overflow_truncates_quotient():
    """The bead's repro, pinned to the Python engine's answer: rax=0, rdx=0."""
    code = bytes.fromhex(CASES["div_ecx"]) + JMP_SELF
    project = angr.load_shellcode(code, arch="amd64", load_address=BASE)
    mgr = RustExplorationManager(project, [_state(project, (0, 1, 1), False)])
    mgr.explore(find=BASE + 2, max_steps=5)
    assert len(mgr.found) == 1 and not mgr.errored, mgr.stash_counts()
    assert _results(mgr.found[0]) == [[0], [0]]
