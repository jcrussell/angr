"""SSE int<->float conversions on the Rust engine agree with the Python engine
(angr-mx00p.6).

libVEX lifts these to forms the Rust engine used to error on: the binop
``Iop_I64StoF64`` / ``Iop_I64StoF32`` (rounding mode first) for cvtsi2sd/ss
from a 64-bit register, ``Iop_I32StoF32x4`` for cvtdq2ps and
``Iop_F32toI32Sx4`` for cvtps2dq / cvttps2dq. Each test runs one instruction
on both engines and compares xmm0, so a wrong rounding on either side shows up
(claripy's concrete ``fpToSBV`` used to truncate whatever the mode, which made
the Python side of cvtps2dq wrong).
"""

from __future__ import annotations

import pytest

import angr
from tests.engines.conftest import (
    RUST_EXPLORATION_AVAILABLE,
    RustExplorationManager,
)

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)

BASE = 0x400000
JMP_SELF = bytes.fromhex("ebfe")

# 2^53 + 1 is not a double, so cvtsi2sd has to round.
RAX = (1 << 53) + 1
# F32 lanes, lane 0 lowest: -2.5, 1.5, -1.5, 2.5 (ties for the nearest-even
# check, and signs for truncation).
XMM1 = (0x40200000 << 96) | (0xBFC00000 << 64) | (0x3FC00000 << 32) | 0xC0200000

CASES = {
    "cvtsi2sd_rax": "f2480f2ac0",
    "cvtsi2ss_rax": "f3480f2ac0",
    "cvtdq2ps": "0f5bc1",
    "cvtps2dq": "660f5bc1",
    "cvttps2dq": "f30f5bc1",
}


def _state(project):
    state = project.factory.blank_state(addr=BASE)
    state.regs.rax = RAX
    state.regs.xmm1 = XMM1
    return state


@pytest.mark.parametrize("insn", CASES)
def test_conversion_matches_python_engine(insn):
    code = bytes.fromhex(CASES[insn]) + JMP_SELF
    end = BASE + len(code) - len(JMP_SELF)

    project = angr.load_shellcode(code, arch="amd64", load_address=BASE)
    python = project.factory.simulation_manager(_state(project))
    python.explore(find=end)
    assert python.found, "Python engine did not reach the end of the block"

    project = angr.load_shellcode(code, arch="amd64", load_address=BASE)
    mgr = RustExplorationManager(project, [_state(project)])
    mgr.explore(find=end, max_steps=5)
    assert len(mgr.found) == 1 and not mgr.errored, mgr.stash_counts()

    expected = python.found[0].solver.eval(python.found[0].regs.xmm0)
    assert mgr.found[0].solver.eval(mgr.found[0].regs.xmm0) == expected


def test_cvtps2dq_rounds_to_nearest_even():
    """Under the default MXCSR (round to nearest) 1.5 -> 2 and 2.5 -> 2."""
    code = bytes.fromhex(CASES["cvtps2dq"]) + JMP_SELF
    project = angr.load_shellcode(code, arch="amd64", load_address=BASE)
    mgr = RustExplorationManager(project, [_state(project)])
    mgr.explore(find=BASE + len(code) - len(JMP_SELF), max_steps=5)
    xmm0 = mgr.found[0].solver.eval(mgr.found[0].regs.xmm0)
    lanes = [(xmm0 >> (32 * i)) & 0xFFFFFFFF for i in range(4)]
    assert [v - (1 << 32) if v >> 31 else v for v in lanes] == [-2, 2, -2, 2]
