"""PHADDW on the Rust engine agrees with the Python engine (angr-mx00p.125.1).

pyvex lifts SSSE3 ``phaddw`` through ``Iop_CatEvenLanes16x4`` /
``Iop_CatOddLanes16x4``. The Rust engine runs them natively in
``VEXOps::vec_cat_lanes``; these tests run the same block on both engines,
compare the result, and check the block did not defer to Python.
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

DST_ADDR = 0x10000
SRC_ADDR = 0x10100
OUT_ADDR = 0x10200

# movdqu xmm0, [rdi]; movdqu xmm1, [rsi]; phaddw xmm0, xmm1;
# movdqu [rdx], xmm0; jmp $
SSE_CODE = bytes.fromhex("f30f6f07f30f6f0e660f3801c1f30f7f02ebfe")
# movq mm0, [rdi]; movq mm1, [rsi]; phaddw mm0, mm1; movq [rdx], mm0; jmp $
MMX_CODE = bytes.fromhex("0f6f070f6f0e0f3801c10f7f02ebfe")

# Distinct words per lane, with pairs that wrap past 0xFFFF.
DST = struct.pack("<8H", 0x0001, 0x0010, 0x0200, 0x3000, 0xFFFF, 0x0002, 0x8000, 0x8001)
SRC = struct.pack("<8H", 0x1111, 0x2222, 0x0003, 0x0004, 0x7000, 0x9000, 0x00AB, 0xCD00)


def _state(code, dst, src):
    proj = angr.load_shellcode(code, arch="amd64", load_address=0x1000)
    state = proj.factory.blank_state(
        addr=0x1000,
        add_options={
            angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
            angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
        },
    )
    state.regs.rdi = DST_ADDR
    state.regs.rsi = SRC_ADDR
    state.regs.rdx = OUT_ADDR
    state.memory.store(DST_ADDR, dst)
    state.memory.store(SRC_ADDR, src)
    return proj, state


def _run_python(proj, state):
    succ = proj.factory.successors(state.copy())
    (out,) = succ.flat_successors
    return out


def _run_rust(proj, state):
    mgr = RustExplorationManager(proj, [state.copy()])
    mgr.run(max_steps=1)
    (out,) = mgr.active
    assert mgr.stats["rust_python_vex_binop_fallback_count"] == 0, "phaddw deferred to Python"
    return out


def _expected(dst, src):
    """Intel SDM PHADDW: adjacent word pairs of dst, then of src, each sum wrapped to 16 bits."""
    words = struct.unpack(f"<{len(dst) // 2}H", dst) + struct.unpack(f"<{len(src) // 2}H", src)
    return struct.pack(f"<{len(words) // 2}H", *((a + b) & 0xFFFF for a, b in zip(words[::2], words[1::2])))


@pytest.mark.parametrize(
    ("code", "width"),
    [pytest.param(SSE_CODE, 16, id="xmm"), pytest.param(MMX_CODE, 8, id="mmx")],
)
def test_phaddw_concrete_matches_python(code, width):
    proj, state = _state(code, DST[:width], SRC[:width])
    want = _expected(DST[:width], SRC[:width])
    py = _run_python(proj, state).memory.load(OUT_ADDR, width)
    rs = _run_rust(proj, state).memory.load(OUT_ADDR, width)
    assert py.concrete_value == int.from_bytes(want, "big")
    assert rs.concrete_value == py.concrete_value


def test_phaddw_symbolic_source_matches_python():
    src = claripy.BVS("phaddw_src", 128)
    proj, state = _state(SSE_CODE, DST, src)
    py_state = _run_python(proj, state)
    rs_state = _run_rust(proj, state)
    for pinned in (SRC, bytes(range(16)), b"\xff" * 16):
        pin = src == int.from_bytes(pinned, "big")
        py = py_state.solver.eval(py_state.memory.load(OUT_ADDR, 16), extra_constraints=[pin], cast_to=bytes)
        rs = rs_state.solver.eval(rs_state.memory.load(OUT_ADDR, 16), extra_constraints=[pin], cast_to=bytes)
        assert py == _expected(DST, pinned)
        assert rs == py
