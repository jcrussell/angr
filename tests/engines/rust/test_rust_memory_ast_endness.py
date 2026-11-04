"""A symbolic memory read or write through an export follows the arch's byte order (angr-mx00p.136.32).

``get_state_memory_ast`` / ``set_state_memory_ast`` speak an LSB-first layout,
and Rust memory holds a big-endian state's values in big-endian order, so the
FFI reverses for a big-endian state. Before it did, a mips export read
fauxware's password buffer back to front, and a detached export (pickled,
or handed to the Python engine) found ``YKAENSOS``.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr import claripy
from angr.exploration.rust_backend import RustBackend
from angr.exploration.rust_state_binding import binding_of
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, TEST_BINARIES_DIR

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")

# Past fauxware's first read(0, buf, 8), stepped in Rust.
STEPS = 30


def _export(arch):
    project = angr.Project(os.path.join(TEST_BINARIES_DIR, "..", arch, "fauxware"), auto_load_libs=False)
    simgr = angr.SimulationManager(project, [project.factory.entry_state()], backend=RustBackend(project))
    simgr.run(n=STEPS)
    state = simgr.active[0]
    assert binding_of(state) is not None
    return state


def _same(state, a, b):
    # Distinct stdin bytes are free, so a wrong byte order is not provably equal.
    return state.solver.is_true(a == b)


def _stdin_buffer(state):
    sp = state.regs.sp.concrete_value
    for addr in range(sp, sp + 0x100):
        byte = state.memory.load(addr, 1)
        if byte.symbolic and "stdin" in str(byte):
            return addr
    raise AssertionError("no stdin byte on the stack")


@pytest.mark.parametrize("arch", ["armel", "mips", "x86_64"])
def test_a_rust_written_buffer_loads_in_memory_order(arch):
    state = _export(arch)
    addr = _stdin_buffer(state)
    word = state.memory.load(addr, 4)
    for i in range(4):
        assert _same(state, word[31 - 8 * i : 24 - 8 * i], state.memory.load(addr + i, 1))
    # The arch's own endness gives byte 0 its low byte on a little-endian arch, its high byte otherwise.
    native = state.memory.load(addr, 4, endness=state.arch.memory_endness)
    low_first = state.arch.memory_endness == "Iend_LE"
    assert _same(state, native[7:0] if low_first else native[31:24], state.memory.load(addr, 1))


@pytest.mark.parametrize("arch", ["armel", "mips", "x86_64"])
def test_a_symbolic_store_reads_back_byte_by_byte(arch):
    state = _export(arch)
    addr = state.regs.sp.concrete_value - 0x100
    value = claripy.BVS("probe", 32)
    state.memory.store(addr, value)
    for i in range(4):
        assert _same(state, state.memory.load(addr + i, 1), value[31 - 8 * i : 24 - 8 * i])
