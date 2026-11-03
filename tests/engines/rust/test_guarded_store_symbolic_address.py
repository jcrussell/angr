"""A guarded store at a symbolic address pins the address only under its guard.

A symbolic-guard StoreG (ARM ``strhi``) or a CAS whose comparison is undecided
(``lock cmpxchg``) writes only where the guard holds. Python concretizes the
address with the guard as an extra constraint and adds ``guard -> addr == a``.
Rust pinned ``addr == max`` unconditionally, so the guard-false path lost every
other index value (angr-vmi2l.169). Under SYMBOLIC_WRITE_ADDRESSES the address
concretizes to every index; Rust handed that write to the host's full store,
which takes no guard and drops the write under the memory proxy
(angr-vmi2l.323).
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

OPTS = {angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY, angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
SYMWRITE = OPTS | {angr.options.SYMBOLIC_WRITE_ADDRESSES}


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


def _storeg(engine, opts):
    """``cmp r0, #7; strhi r3, [r1, r5, lsl #2]`` with the index in [0, 3]."""
    load, buf = 0x10000, 0x80000
    code = b"".join(struct.pack("<I", w) for w in (0xE3500007, 0x87813105, 0xEAFFFFFE))
    proj = angr.load_shellcode(code, arch="ARMEL", load_address=load)
    state = proj.factory.blank_state(addr=load, add_options=opts)
    x, idx = claripy.BVS("x", 32), claripy.BVS("idx", 32)
    state.regs.r0 = x
    state.regs.r1 = buf
    state.regs.r3 = 0xCAFE0003
    state.regs.r5 = idx
    state.add_constraints(claripy.ULE(idx, 3))
    for k in range(4):
        state.memory.store(buf + 4 * k, claripy.BVV(0x0D0D0D00 + k, 32), endness="Iend_LE")
    found = _explore(engine, proj, state, load + 8)
    hi = claripy.UGT(x, 7)
    mem = [_values(found, found.memory.load(buf + 4 * k, 4, endness="Iend_LE"), hi, idx == k) for k in range(4)]
    return _values(found, idx, claripy.Not(hi)), _values(found, idx, hi), mem


def _cas(engine, opts):
    """``lock cmpxchg [rdi + rcx*4], esi`` with the index in [0, 3]."""
    load, buf = 0x400000, 0x700000
    code = b"\xf0\x0f\xb1\x34\x8f"
    end = load + len(code)
    proj = angr.load_shellcode(code + b"\xeb\xfe", arch="amd64", load_address=load)
    state = proj.factory.blank_state(addr=load, add_options=opts)
    x, i = claripy.BVS("x", 32), claripy.BVS("i", 64)
    state.regs.rax = x.zero_extend(32)
    state.regs.rdi = buf
    state.regs.rsi = 0x5EED0001
    state.regs.rcx = i
    state.add_constraints(claripy.ULE(i, 3))
    for k in range(4):
        state.memory.store(buf + 4 * k, claripy.BVV(0x0A0A0A00 + k, 32), endness="Iend_LE")
    found = _explore(engine, proj, state, end)
    mem = [
        _values(found, found.memory.load(buf + 4 * k, 4, endness="Iend_LE"), x == 0x0A0A0A00 + k, i == k)
        for k in range(4)
    ]
    return _values(found, i, x == 0x12345678), _values(found, i), mem


@pytest.mark.parametrize("opts", [OPTS, SYMWRITE], ids=["default", "symwrite"])
def test_storeg_symbolic_address_matches_python(opts):
    py = _storeg("py", opts)
    assert py[0] == [0, 1, 2, 3]
    if opts is SYMWRITE:
        assert py[2] == [[0xCAFE0003]] * 4
    assert _storeg("rust", opts) == py


@pytest.mark.parametrize("opts", [OPTS, SYMWRITE], ids=["default", "symwrite"])
def test_cas_symbolic_address_matches_python(opts):
    py = _cas("py", opts)
    assert py[0] == [0, 1, 2, 3]
    if opts is SYMWRITE:
        assert py[2] == [[0x5EED0001]] * 4
    assert _cas("rust", opts) == py
