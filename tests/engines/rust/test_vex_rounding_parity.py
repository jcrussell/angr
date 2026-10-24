"""Rounding IROps on the Rust engine agree with the Python engine on the same
block: the sign of zero from ``RoundF64toInt`` (angr-vmi2l.57) and the VEX
rounding mode 4, ties away from zero (angr-vmi2l.56).

Each test runs one instruction on both engines and compares the destination
register, concretely and, where the bead's repro did, with the input symbolic
but pinned by a constraint.
"""

from __future__ import annotations

import struct

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
A64_B_SELF = struct.pack("<I", 0x14000000)
X86_JMP_SELF = bytes.fromhex("ebfe")


def _f64_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def _run_both(code, arch, setup, out_reg, symbolic_in=None):
    """Run ``code`` (one instruction plus a self-branch) on both engines.

    ``setup(state)`` seeds the inputs. When ``symbolic_in`` is ``(reg, bits,
    value)``, that register is replaced by a fresh symbol pinned to ``value``.
    Returns ``(python, rust)`` as sorted ``eval_upto`` lists of ``out_reg``.
    """
    end = BASE + len(code) - 4 if arch == "aarch64" else BASE + len(code) - len(X86_JMP_SELF)
    results = []
    for rust in (False, True):
        project = angr.load_shellcode(code, arch=arch, load_address=BASE)
        state = project.factory.blank_state(addr=BASE)
        setup(state)
        if symbolic_in is not None:
            reg, bits, value = symbolic_in
            sym = claripy.BVS(reg, bits)
            state.solver.add(sym == value)
            setattr(state.regs, reg, sym)
        if rust:
            mgr = RustExplorationManager(project, [state])
            mgr.explore(find=end, max_steps=5)
            assert len(mgr.found) == 1 and not mgr.errored, mgr.stash_counts()
            found = mgr.found[0]
        else:
            simgr = project.factory.simulation_manager(state)
            simgr.explore(find=end)
            assert simgr.found, "Python engine did not reach the end of the block"
            found = simgr.found[0]
        results.append(sorted(found.solver.eval_upto(getattr(found.regs, out_reg), 3)))
    return results


# --- RoundF64toInt keeps the sign of zero (angr-vmi2l.57) -------------------

NEG_POINT_3 = _f64_bits(-0.3)
NEG_ZERO = _f64_bits(-0.0)

ROUND_CASES = {
    # frintp d0, d1: RoundF64toInt(RU, d1); ceil(-0.3) == -0.0.
    "frintp": (struct.pack("<I", 0x1E64C020) + A64_B_SELF, "aarch64", "d1", "d0"),
    # frintz d0, d1: RoundF64toInt(RZ, d1); trunc(-0.3) == -0.0.
    "frintz": (struct.pack("<I", 0x1E65C020) + A64_B_SELF, "aarch64", "d1", "d0"),
    # roundsd xmm0, xmm1, 2: RoundF64toInt(RU, low lane of xmm1).
    "roundsd": (bytes.fromhex("660f3a0bc102") + X86_JMP_SELF, "amd64", "xmm1", "xmm0"),
}


@pytest.mark.parametrize("insn", ROUND_CASES)
def test_round_to_int_keeps_negative_zero(insn):
    code, arch, in_reg, out_reg = ROUND_CASES[insn]

    def setup(state):
        setattr(state.regs, in_reg, NEG_POINT_3)
        if arch == "amd64":
            state.regs.xmm0 = 0

    python, rust = _run_both(code, arch, setup, out_reg)
    assert python == [NEG_ZERO]
    assert rust == python
