"""A seed's ``heap.mmap_base`` and a Python syscall fallback's ``posix.brk`` reach Rust (angr-vmi2l.150).

The seed push forwarded ``posix.brk`` and ``heap.heap_location`` but not
``heap.mmap_base``, so native addr=0 mmaps ignored a user-set base. The
syscall resume path wrote neither the heap cursors nor ``posix.brk`` back, so
after a Python brk fallback a native ``brk(0)`` reported the stale break.
Each case runs amd64 shellcode on both engines and compares the registers.
"""

from __future__ import annotations

import struct

import pytest

import angr
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
SYSCALL = b"\x0f\x05"
MOV_R12_RAX = b"\x49\x89\xc4"
MOV_R13_RAX = b"\x49\x89\xc5"
MOV_R14_RAX = b"\x49\x89\xc6"
MOV_RDI_R13 = b"\x4c\x89\xef"

_MOVABS = {
    "rax": b"\x48\xb8",
    "rdi": b"\x48\xbf",
    "rsi": b"\x48\xbe",
    "rdx": b"\x48\xba",
    "r10": b"\x49\xba",
    "r8": b"\x49\xb8",
    "r9": b"\x49\xb9",
}


def _mov(reg: str, value: int) -> bytes:
    return _MOVABS[reg] + struct.pack("<q", value)


def _add_rdi(value: int) -> bytes:
    return b"\x48\x81\xc7" + struct.pack("<i", value)


# mmap(0, 0x1000, RW, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
ANON_MMAP = (
    _mov("rax", 9)
    + _mov("rdi", 0)
    + _mov("rsi", 0x1000)
    + _mov("rdx", 3)
    + _mov("r10", 0x22)
    + _mov("r8", -1)
    + _mov("r9", 0)
    + SYSCALL
)
TWO_MMAPS = ANON_MMAP + MOV_R12_RAX + ANON_MMAP + MOV_R13_RAX

# r13 = brk(0); mmap(r13 + 0x2000, 0x1000, RW, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0);
# brk(r13 + 0x4000) collides with the fixed page, so native declines and
# Python's set_brk stops the break at it; then r14 = brk(0), run natively.
BRK_COLLIDE_THEN_QUERY = (
    _mov("rax", 12)
    + _mov("rdi", 0)
    + SYSCALL
    + MOV_R13_RAX
    + MOV_RDI_R13
    + _add_rdi(0x2000)
    + _mov("rax", 9)
    + _mov("rsi", 0x1000)
    + _mov("rdx", 3)
    + _mov("r10", 0x32)
    + _mov("r8", -1)
    + _mov("r9", 0)
    + SYSCALL
    + _mov("rax", 12)
    + MOV_RDI_R13
    + _add_rdi(0x4000)
    + SYSCALL
    + MOV_R12_RAX
    + _mov("rax", 12)
    + _mov("rdi", 0)
    + SYSCALL
    + MOV_R14_RAX
)


def _found(code: bytes, engine: str, mmap_base: int | None = None):
    project = angr.load_shellcode(code + b"\xeb\xfe", arch="AMD64", simos="linux", load_address=LOAD)
    state = project.factory.entry_state(addr=LOAD, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY})
    if mmap_base is not None:
        state.heap.mmap_base = mmap_base
    mgr = RustExplorationManager(project, [state]) if engine == "rust" else project.factory.simulation_manager(state)
    mgr.explore(find=LOAD + len(code), n=100)
    assert len(mgr.found) == 1, f"{engine}: errored={list(mgr.errored)}"
    return mgr.found[0]


def _mmap_base_case(engine: str):
    found = _found(TWO_MMAPS, engine, mmap_base=0x50000000)
    return found.solver.eval(found.regs.r12), found.solver.eval(found.regs.r13), found.heap.mmap_base


def _brk_case(engine: str):
    found = _found(BRK_COLLIDE_THEN_QUERY, engine)
    start = found.solver.eval(found.regs.r13)
    return found.solver.eval(found.regs.r12) - start, found.solver.eval(found.regs.r14) - start


def test_user_set_mmap_base_places_native_mmaps():
    expected = _mmap_base_case("python")
    assert _mmap_base_case("rust") == expected
    assert expected == (0x50000000, 0x50001000, 0x50002000)


def test_native_brk_query_sees_a_python_fallback_break():
    expected = _brk_case("python")
    assert _brk_case("rust") == expected
    assert expected == (0x2000, 0x2000)
