"""Byte-slice memory stores keep memory order on a big-endian Rust state (angr-mx00p.132.74).

The proxy hands Rust a concrete store as bytes in memory order. Rust packed
each 16-byte chunk little-endian and stored that value with the memory's
natural endianness, so on MIPS every chunk landed reversed. A bounced
``fstat`` wrote ``st_blksize`` that way, glibc read garbage, and test_accuracy
mips64 errored inside mmap.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")

LOAD_ADDR = 0x400000
HOOK_ADDR = LOAD_ADDR + 8
END_ADDR = LOAD_ADDR + 0x10
SP = 0x7FFF0000
BUF_OFFSET = 0x100


def _run(arch: str, engine: str | None) -> tuple[list[int], int]:
    """Store a 64-bit value in the arch's endness from a hook; return its bytes as the hook and the end state see them."""
    seen = {}

    def hook(state):
        buf = state.regs.sp - BUF_OFFSET
        state.memory.store(buf, claripy.BVV(0x0102030405060708, 64), endness=state.arch.memory_endness)
        seen["bytes"] = [state.memory.load(buf + i, 1).concrete_value for i in range(8)]

    kwargs = {"engine": engine} if engine else {}
    proj = angr.load_shellcode(b"\x00" * 0x20, arch=arch, start_offset=0, load_address=LOAD_ADDR, **kwargs)
    proj.hook(HOOK_ADDR, hook, length=4)
    state = proj.factory.blank_state(addr=LOAD_ADDR, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY})
    state.regs.sp = SP
    simgr = proj.factory.simulation_manager(state)
    simgr.run(until=lambda m: bool(m.active) and m.active[0].addr >= END_ADDR)
    end = simgr.active[0]
    value = end.memory.load(SP - BUF_OFFSET, 8, endness=end.arch.memory_endness).concrete_value
    return seen["bytes"], value


@pytest.mark.parametrize("arch", ["mips32", "mips64"])
def test_a_hook_store_on_a_big_endian_state_matches_python(arch):
    rust_bytes, rust_value = _run(arch, "rust")
    py_bytes, py_value = _run(arch, None)
    assert py_bytes == [1, 2, 3, 4, 5, 6, 7, 8]
    assert rust_bytes == py_bytes
    assert rust_value == py_value == 0x0102030405060708
