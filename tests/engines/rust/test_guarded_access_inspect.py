"""A guarded store or load fires its ``mem_write`` / ``mem_read`` BP (angr-vmi2l.337).

An ARM conditional ``strhi`` lifts to ``StoreG`` and ``ldrhi`` to ``LoadG``.
Python fires the memory BP for either one unless the guard is provably false,
with the guard as ``mem_*_condition``. Rust performed the access and fired
nothing, so a BP missed it and a value-injecting BP silently did not inject.
Each case runs ARM-mode code from a blob and compares both engines.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD, BUF = 0x10000, 0x80000
STR_R2_R1 = 0xE5812000  # str r2, [r1]
CMP_R0_7 = 0xE3500007  # cmp r0, #7
STRHI_R3 = 0x85813004  # strhi r3, [r1, #4]
LDRHI_R3 = 0x85913004  # ldrhi r3, [r1, #4]
B_SELF = 0xEAFFFFFE  # b .
OPTS = {angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY, angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}


def _condition(cond):
    # Read structurally: a solver query from inside a BP action is not what is under test.
    if cond is None or cond.symbolic:
        return cond if cond is None else "symbolic"
    return cond.is_true()


def _run(engine, guarded, r0, event, when=angr.BP_AFTER, inject=None):
    project = angr.load_shellcode(
        b"".join(struct.pack("<I", w) for w in [STR_R2_R1, CMP_R0_7, guarded, B_SELF]),
        arch="ARMEL",
        load_address=LOAD,
    )
    state = project.factory.blank_state(addr=LOAD, add_options=OPTS)
    state.regs.r0 = r0
    state.regs.r1 = BUF
    state.regs.r2 = 0xCAFE0002
    state.regs.r3 = 0xCAFE0003
    state.memory.store(BUF + 4, claripy.BVV(0x0D0D0D0D, 32), endness="Iend_LE")
    hits = []

    def action(st):
        addr = st.solver.eval(getattr(st.inspect.attrs, f"{event}_address"))
        hits.append((hex(addr), _condition(getattr(st.inspect.attrs, f"{event}_condition"))))
        if inject is not None and addr == BUF + 4:
            st.inspect.attrs.mem_write_expr = inject

    state.inspect.b(event, when=when, action=action)
    mgr = RustExplorationManager(project, [state]) if engine == "rust" else project.factory.simulation_manager(state)
    mgr.explore(find=LOAD + 12, n=10)
    assert len(mgr.found) == 1
    found = mgr.found[0]
    seen = list(hits)  # before the load below, which fires mem_read too
    stored = found.solver.eval_upto(found.memory.load(BUF + 4, 4, endness="Iend_LE"), 3)
    return seen, sorted(stored)


R0_CASES = [9, 5, claripy.BVS("r0", 32)]


@pytest.mark.parametrize("r0", R0_CASES, ids=["guard-true", "guard-false", "guard-symbolic"])
def test_guarded_store_fires_mem_write_like_python(r0):
    expected = _run("python", STRHI_R3, r0, "mem_write")
    assert _run("rust", STRHI_R3, r0, "mem_write") == expected
    # The plain str always fires, with no condition; the strhi unless its guard is false.
    assert expected[0][0] == ("0x80000", None)
    assert len(expected[0]) == (1 if r0 is R0_CASES[1] else 2)


@pytest.mark.parametrize("r0", R0_CASES, ids=["guard-true", "guard-false", "guard-symbolic"])
def test_guarded_load_fires_mem_read_like_python(r0):
    expected = _run("python", LDRHI_R3, r0, "mem_read")
    assert _run("rust", LDRHI_R3, r0, "mem_read") == expected
    assert len(expected[0]) == (0 if r0 is R0_CASES[1] else 1)


@pytest.mark.parametrize("r0", [9, R0_CASES[2]], ids=["guard-true", "guard-symbolic"])
def test_guarded_store_takes_an_injected_value_like_python(r0):
    inject = claripy.BVV(0x11223344, 32)
    expected = _run("python", STRHI_R3, r0, "mem_write", when=angr.BP_BEFORE, inject=inject)
    assert _run("rust", STRHI_R3, r0, "mem_write", when=angr.BP_BEFORE, inject=inject) == expected
    assert 0x11223344 in expected[1]
