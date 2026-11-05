"""A never-written byte on the seed's SP stack page reads as a fill symbol (angr-mx00p.132.58).

The SP page is mapped into Rust from the UltraPage's ``concrete_load``, which
holds zero for every byte Python never wrote. Python's load mints a ``mem_``
fill for such a byte, so a branch on it forks; Rust read the zero and kept one
path. Both tests compare with the Python engine.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager
from tests.engines.conftest import BIN_LOCATION

LOAD = 0x400000
# mov rbx,[rdi]; jmp $
READ = b"\x48\x8b\x1f\xeb\xfe"


@pytest.mark.parametrize("zero_fill", [False, True])
@pytest.mark.parametrize("engine", ["py", "rust"])
def test_unwritten_byte_on_the_sp_page_is_a_fill(engine, zero_fill):
    proj = angr.load_shellcode(READ, arch="amd64", load_address=LOAD)
    options = {angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
    if zero_fill:
        options.add(angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY)
    state = proj.factory.blank_state(addr=LOAD, add_options=options)
    sp_page = state.solver.eval(state.regs.rsp) & ~0xFFF
    # A write elsewhere on the page makes Python materialize it, so the sync
    # maps it from concrete_load rather than leaving it to the lazy region.
    state.memory.store(sp_page + 0x800, b"\x41" * 8)
    state.regs.rdi = sp_page + 0x10
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=LOAD + len(READ) - 2)
    (found,) = mgr.found
    values = found.solver.eval_upto(found.regs.rbx, 2)
    assert values == [0] if zero_fill else len(values) == 2
    assert found.solver.eval_upto(found.memory.load(sp_page + 0x800, 1), 2) == [0x41]


def test_branch_on_an_unwritten_stack_slot_forks_like_python():
    """uc_stop's stop_symbolic_block_exit compares a never-written local with 10."""
    proj = angr.Project(os.path.join(BIN_LOCATION, "tests", "i386", "uc_stop"), auto_load_libs=False)

    def branch_arms(states):
        return sorted(a for s in states for a in s.history.bbl_addrs if a in (0x8048587, 0x8048591))

    python = angr.SimulationManager(proj, active_states=[proj.factory.entry_state(args=["a"] * 10)]).run()
    rust = RustExplorationManager(proj, [proj.factory.entry_state(args=["a"] * 10)])
    rust.run()
    assert branch_arms(python.deadended) == [0x8048587, 0x8048591]
    assert branch_arms(rust.deadended) == branch_arms(python.deadended)
