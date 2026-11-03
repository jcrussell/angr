"""Guarded and atomic accesses read and write Rust memory like plain ones.

A symbolic-guard StoreG (ARM ``strhi``), a CAS with an undecided comparison
(``lock cmpxchg``) and a symbolic-guard LoadG (ARM ``ldrhi``) used to bypass
``rust_memory``: StoreG read its ``current`` bytes from the host's stale
shadow, both writebacks were parked in the pending buffers (invisible to a
later in-block load, and replayed over a newer plain store at flush), and a
LoadG at a symbolic address read only the first candidate or the stale shadow
(angr-vmi2l.168). Each case compares the Rust engine against the Python one.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

OPTS = {angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY, angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}

ARM_LOAD, ARM_BUF = 0x10000, 0x80000
CMP_R0_7 = 0xE3500007
STRHI_R3_R1 = 0x85813000
STR_R2_R1 = 0xE5812000
LDR_R4_R1 = 0xE5914000
LDRHI_R4_R1_R5_LSL2 = 0x87914105
SPIN = 0xEAFFFFFE

STOREG = {
    "prior": [STR_R2_R1, CMP_R0_7, STRHI_R3_R1],
    "readback": [CMP_R0_7, STRHI_R3_R1, LDR_R4_R1],
    "later": [CMP_R0_7, STRHI_R3_R1, STR_R2_R1, LDR_R4_R1],
}
LOADG = {
    "multi": [0xE581200C, CMP_R0_7, LDRHI_R4_R1_R5_LSL2],  # str r2, [r1, #12] first
    "strided_st": [0xE5812004, CMP_R0_7, LDRHI_R4_R1_R5_LSL2],  # str r2, [r1, #4] first
}
CAS = {
    "readback": b"\xf0\x0f\xb1\x37" + b"\x8b\x1f",  # lock cmpxchg [rdi], esi; mov ebx, [rdi]
    "later": b"\xf0\x0f\xb1\x37" + b"\x89\x17" + b"\x8b\x1f",  # ... mov [rdi], edx; mov ebx, [rdi]
}


def _explore(engine, proj, state, find):
    if engine == "rust":
        from angr.exploration.rust_manager import RustExplorationManager

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=find, max_steps=10)
        errors = [getattr(e, "error", e) for e in mgr.errored]
    else:
        mgr = proj.factory.simulation_manager(state)
        mgr.explore(find=find)
        errors = [e.error for e in mgr.errored]
    assert not errors
    assert len(mgr.found) == 1
    return mgr.found[0]


def _values(state, bv, *extra):
    return sorted(state.solver.eval_upto(bv, 8, extra_constraints=list(extra)))


def _arm(words):
    code = b"".join(struct.pack("<I", w) for w in [*words, SPIN])
    proj = angr.load_shellcode(code, arch="ARMEL", load_address=ARM_LOAD)
    state = proj.factory.blank_state(addr=ARM_LOAD, add_options=OPTS)
    x = claripy.BVS("x", 32)
    state.regs.r0 = x
    state.regs.r1 = ARM_BUF
    state.regs.r2 = 0xCAFE0002
    return proj, state, x, ARM_LOAD + 4 * len(words)


def _storeg(engine, case):
    proj, state, x, end = _arm(STOREG[case])
    state.regs.r3 = 0xCAFE0003
    state.regs.r4 = 0
    state.memory.store(ARM_BUF, claripy.BVV(0x0D0D0D0D, 32), endness="Iend_LE")
    found = _explore(engine, proj, state, end)
    mem = found.memory.load(ARM_BUF, 4, endness="Iend_LE")
    hi, lo = claripy.UGT(x, 7), claripy.ULE(x, 7)
    return [_values(found, v, c) for v in (mem, found.regs.r4) for c in (hi, lo)]


def _loadg(engine, case):
    proj, state, x, end = _arm(LOADG[case])
    idx = claripy.BVS("idx", 32)
    state.regs.r4 = 0x99
    state.regs.r5 = idx
    indices = [0, 1, 3] if case == "multi" else [0, 1, 2, 3]
    state.add_constraints(claripy.Or(*(idx == k for k in indices)))
    for k in range(4):
        state.memory.store(ARM_BUF + 4 * k, claripy.BVV(0x0D0D0D00 + k, 32), endness="Iend_LE")
    found = _explore(engine, proj, state, end)
    hi = claripy.UGT(x, 7)
    return {k: _values(found, found.regs.r4, hi, idx == k) for k in indices}, _values(found, idx, hi)


def _cas(engine, case):
    load, buf = 0x400000, 0x700000
    code = CAS[case]
    proj = angr.load_shellcode(code + b"\xeb\xfe", arch="amd64", load_address=load)
    state = proj.factory.blank_state(addr=load, add_options=OPTS)
    x = claripy.BVS("x", 32)
    state.regs.rax = x.zero_extend(32)
    state.regs.rdi = buf
    state.regs.rsi = 0x5EED0001
    state.regs.rdx = 0x5EED0002
    state.regs.rbx = 0
    state.memory.store(buf, claripy.BVV(0x0A0A0A00, 32), endness="Iend_LE")
    found = _explore(engine, proj, state, load + len(code))
    mem = found.memory.load(buf, 4, endness="Iend_LE")
    swap, noswap = x == 0x0A0A0A00, x != 0x0A0A0A00
    return [_values(found, v, c) for v in (mem, found.regs.ebx) for c in (swap, noswap)]


@pytest.mark.parametrize("case", sorted(STOREG))
def test_storeg_symbolic_guard_matches_python(case):
    assert _storeg("rust", case) == _storeg("py", case)


@pytest.mark.parametrize("case", sorted(LOADG))
def test_loadg_symbolic_address_matches_python(case):
    py = _loadg("py", case)
    assert py[0][1 if case == "strided_st" else 3] == [0xCAFE0002]
    assert _loadg("rust", case) == py


@pytest.mark.parametrize("case", sorted(CAS))
def test_cas_undecided_matches_python(case):
    assert _cas("rust", case) == _cas("py", case)
