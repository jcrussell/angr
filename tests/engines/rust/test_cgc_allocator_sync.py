"""A Python CGC allocate fallback sees the native allocator, and native sees its result (angr-vmi2l.332).

The native CGC allocate / deallocate move the Rust state's allocation_base
and sinkholes. The callback state's ``state.cgc`` plugin kept its own copy,
so an allocate with a symbolic length (which native declines) carved its
region from the stale base and collided with a page native had mapped. The
program allocates two pages natively, deallocates the second (a native
sinkhole), lets Python allocate into the sinkhole, then allocates natively
again, which must not reuse the sinkhole Python consumed.
"""

from __future__ import annotations

import struct

import pytest

import angr
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x08048000
BUF = 0x0804A000
P1, P2, P3, P4 = (BUF + 0x100 + 4 * i for i in range(4))
RX = BUF + 0x120


def _d(v):
    return struct.pack("<I", v & 0xFFFFFFFF)


def _allocate(length_insn, out):
    # <ebx = length>; mov eax, 5; mov ecx, 0; mov edx, out; int 0x80
    return length_insn + b"\xb8" + _d(5) + b"\xb9" + _d(0) + b"\xba" + _d(out) + b"\xcd\x80"


def _build():
    page = b"\xbb" + _d(0x1000)
    code = (
        _allocate(page, P1)
        + _allocate(page, P2)
        # deallocate([P2], 0x1000): mov eax, 6; mov ebx, [P2]; mov ecx, 0x1000; int 0x80
        + b"\xb8"
        + _d(6)
        + b"\x8b\x1d"
        + _d(P2)
        + b"\xb9"
        + _d(0x1000)
        + b"\xcd\x80"
        # receive(0, BUF, 4, RX): mov eax, 3; mov ebx, 0; mov ecx, BUF; mov edx, 4; mov esi, RX; int 0x80
        + b"\xb8"
        + _d(3)
        + b"\xbb"
        + _d(0)
        + b"\xb9"
        + _d(BUF)
        + b"\xba"
        + _d(4)
        + b"\xbe"
        + _d(RX)
        + b"\xcd\x80"
        # A symbolic length in 1..0xff, which native declines:
        # mov ebx, [BUF]; and ebx, 0xff; or ebx, 1
        + _allocate(b"\x8b\x1d" + _d(BUF) + b"\x81\xe3" + _d(0xFF) + b"\x83\xcb\x01", P3)
        + _allocate(page, P4)
    )
    project = angr.load_shellcode(code + b"\xeb\xfe", arch="X86", simos="cgc", load_address=LOAD)
    state = project.factory.entry_state(
        addr=LOAD,
        add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS, angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY},
    )
    state.memory.map_region(BUF, 0x1000, 3)
    state.memory.store(BUF, b"\0" * 0x200)
    return project, state, LOAD + len(code)


def _outcome(found):
    pointers = tuple(found.solver.eval(found.memory.load(p, 4, endness="Iend_LE")) for p in (P1, P2, P3, P4))
    return pointers, found.solver.eval(found.cgc.allocation_base), sorted(found.cgc.sinkholes)


def test_python_allocate_fallback_shares_the_native_cgc_allocator():
    outcomes = {}
    for engine in ("python", "rust"):
        project, state, end = _build()
        if engine == "rust":
            mgr = RustExplorationManager(project, [state])
        else:
            mgr = project.factory.simulation_manager(state)
        mgr.explore(find=end, n=50)
        assert len(mgr.found) == 1, (engine, mgr.errored)
        outcomes[engine] = _outcome(mgr.found[0])
        if engine == "rust":
            stats = mgr._rust_mgr.stats()
            assert stats["syscall_python_fallback_by_num"] == {5: 1}, stats
    (p1, p2, p3, p4), _, sinkholes = outcomes["python"]
    assert p3 == p2 and p4 not in (p1, p2) and not sinkholes
    assert outcomes["rust"] == outcomes["python"]
