"""PSHUFB on the Rust engine agrees with the Python engine (angr-mx00p.6).

libVEX lifts SSSE3 ``pshufb`` to ``Iop_PermOrZero8x16`` (and the MMX form to
``Iop_PermOrZero8x8``). The Rust engine runs it natively in
``VEXOps::vec_perm``; these tests run the same block on both engines and
compare the result, and check the block did not defer to Python.
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

TABLE_ADDR = 0x10000
CTL_ADDR = 0x10100
OUT_ADDR = 0x10200

# movdqu xmm0, [rdi]; movdqu xmm1, [rsi]; pshufb xmm0, xmm1;
# movdqu [rdx], xmm0; jmp $
SSE_CODE = bytes.fromhex("f30f6f07f30f6f0e660f3800c1f30f7f02ebfe")
# movq mm0, [rdi]; movq mm1, [rsi]; pshufb mm0, mm1; movq [rdx], mm0; jmp $
MMX_CODE = bytes.fromhex("0f6f070f6f0e0f3800c10f7f02ebfe")

TABLE = bytes(range(0xA0, 0xB0))
# In-range indices, top-bit lanes (zeroed) and stray bits 6:4.
CTL = bytes([0x0F, 0x00, 0x80, 0x07, 0x1F, 0x8F, 0x33, 0x0A, 0x01, 0xFF, 0x7E, 0x02, 0x40, 0x0C, 0x90, 0x05])


def _state(code, table, ctl):
    proj = angr.load_shellcode(code, arch="amd64", load_address=0x1000)
    state = proj.factory.blank_state(
        addr=0x1000,
        add_options={
            angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
            angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
        },
    )
    state.regs.rdi = TABLE_ADDR
    state.regs.rsi = CTL_ADDR
    state.regs.rdx = OUT_ADDR
    state.memory.store(TABLE_ADDR, table)
    state.memory.store(CTL_ADDR, ctl)
    return proj, state


def _run_python(proj, state):
    succ = proj.factory.successors(state.copy())
    (out,) = succ.flat_successors
    return out


def _run_rust(proj, state):
    mgr = RustExplorationManager(proj, [state.copy()])
    mgr.run(max_steps=1)
    (out,) = mgr.active
    assert mgr.stats["rust_python_vex_binop_fallback_count"] == 0, "pshufb deferred to Python"
    return out


def _expected(table, ctl, n):
    return bytes(0 if c & 0x80 else table[c & (n - 1)] for c in ctl)


@pytest.mark.parametrize(
    ("code", "width"),
    [pytest.param(SSE_CODE, 16, id="xmm"), pytest.param(MMX_CODE, 8, id="mmx")],
)
def test_pshufb_concrete_matches_python(code, width):
    proj, state = _state(code, TABLE[:width], CTL[:width])
    want = _expected(TABLE[:width], CTL[:width], width)
    py = _run_python(proj, state).memory.load(OUT_ADDR, width)
    rs = _run_rust(proj, state).memory.load(OUT_ADDR, width)
    assert py.concrete_value == int.from_bytes(want, "big")
    assert rs.concrete_value == py.concrete_value


def test_pshufb_symbolic_control_matches_python():
    ctl = claripy.BVS("pshufb_ctl", 128)
    proj, state = _state(SSE_CODE, TABLE, ctl)
    py_state = _run_python(proj, state)
    rs_state = _run_rust(proj, state)
    for pinned in (CTL, bytes(range(16)), bytes([0x80] * 16)):
        pin = ctl == int.from_bytes(pinned, "big")
        py = py_state.solver.eval(py_state.memory.load(OUT_ADDR, 16), extra_constraints=[pin], cast_to=bytes)
        rs = rs_state.solver.eval(rs_state.memory.load(OUT_ADDR, 16), extra_constraints=[pin], cast_to=bytes)
        assert py == _expected(TABLE, pinned, 16)
        assert rs == py
