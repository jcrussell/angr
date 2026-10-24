"""Rust-native ccalls agree with the Python engine's ``ccall.py`` on the same
block.

The Rust ccalls have concrete and symbolic twins that are diff-fuzzed against
each other, which cannot catch a semantic error the twins share. These tests
run one block on both engines and compare the result register, concretely and
with the inputs symbolic but pinned by a constraint.
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

BASE = 0x400000

# cmp x0, x2 ; sbcs xzr, x1, x3 ; cset x4, hs ; b .
# A 128-bit unsigned compare (x1:x0) >= (x3:x2): the cset reads the SBC64
# carry, whose carry-in (cc_dep3) is the CMP's C (angr-vmi2l.76).
SBC_U128_GE = struct.pack("<4I", 0xEB02001F, 0xFA03003F, 0x9A9F37E4, 0x14000000)
SBC_END = BASE + 12

# (x0, x1, x2, x3) -> expected x4. The first two rows have argL == argR in the
# SBC with (argR != 0) != oldC, where keying the carry off cc_dep2 is wrong.
SBC_CASES = {
    "lo_ge_hi_zero": ((3, 0, 2, 0), 1),
    "lo_lt_hi_equal": ((1, 5, 2, 5), 0),
    "lo_ge_hi_equal": ((2, 5, 1, 5), 1),
    "hi_lt": ((5, 1, 5, 2), 0),
    "hi_gt": ((0, 7, 0, 3), 1),
}


def _sbc_state(project, vals, symbolic):
    state = project.factory.blank_state(addr=BASE)
    for name, val in zip(("x0", "x1", "x2", "x3"), vals):
        if symbolic:
            bv = claripy.BVS(name, 64)
            state.solver.add(bv == val)
            setattr(state.regs, name, bv)
        else:
            setattr(state.regs, name, val)
    state.regs.x4 = 0x77
    return state


@pytest.mark.parametrize("symbolic", [False, True], ids=["concrete", "symbolic"])
@pytest.mark.parametrize("case", SBC_CASES)
def test_arm64_sbc_u128_compare_matches_python(case, symbolic):
    vals, expected = SBC_CASES[case]

    project = angr.load_shellcode(SBC_U128_GE, arch="aarch64", load_address=BASE)
    python = project.factory.simulation_manager(_sbc_state(project, vals, symbolic))
    python.explore(find=SBC_END)
    assert python.found, "Python engine did not reach the end of the block"
    py_found = python.found[0]
    assert py_found.solver.eval_upto(py_found.regs.x4, 2) == [expected]

    project = angr.load_shellcode(SBC_U128_GE, arch="aarch64", load_address=BASE)
    mgr = RustExplorationManager(project, [_sbc_state(project, vals, symbolic)])
    mgr.explore(find=SBC_END, max_steps=5)
    assert len(mgr.found) == 1 and not mgr.errored, mgr.stash_counts()
    found = mgr.found[0]
    assert found.solver.eval_upto(found.regs.x4, 2) == [expected]
