"""A concrete store to a page nobody mapped survives under the Rust engine (angr-sy3xj).

The shellcode stores 1 at 0x600000, which neither the loader nor the seed
state maps, then reads it back in a later block. Python's ``DefaultMemory``
maps the page on the write. Rust used to buffer the store for the current run
only: a block-granular step lost it, and an exported state read the page as a
fresh ``mem_600000`` symbol.
"""

from __future__ import annotations

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

# mov rax, 1; mov [0x600000], rax; jmp +0; mov rbx, [0x600000]; jmp $
CODE = bytes.fromhex("48c7c001000000" + "4889042500006000" + "eb00" + "488b1c2500006000" + "ebfe")
SLOT = 0x600000


def _state():
    proj = angr.load_shellcode(CODE, arch="amd64", load_address=0x400000)
    return proj, proj.factory.blank_state(addr=0x400000)


def test_python_engine_reads_the_store_back():
    proj, state = _state()
    simgr = proj.factory.simulation_manager(state)
    for _ in range(3):
        simgr.step()
    assert simgr.active[0].solver.eval_one(simgr.active[0].regs.rbx) == 1


@pytest.mark.parametrize("granular", [True, False])
def test_rust_reads_the_store_back_across_steps(granular):
    proj, state = _state()
    mgr = RustExplorationManager(proj, [state])
    mgr.set_block_granular(granular)
    for _ in range(3):
        mgr.step()
    rbx = mgr.active[0].regs.rbx
    assert rbx.concrete and mgr.active[0].solver.eval_one(rbx) == 1


@pytest.mark.parametrize("granular", [True, False])
def test_exported_state_loads_the_stored_page(granular):
    proj, state = _state()
    mgr = RustExplorationManager(proj, [state])
    mgr.set_block_granular(granular)
    mgr.step()
    stored = mgr.active[0].memory.load(SLOT, 8, endness="Iend_LE")
    assert stored.concrete and mgr.active[0].solver.eval_one(stored) == 1
