"""A multi-valued SimProcedure store maps its missing candidate page and retries.

``RustMemoryProxy`` sends a symbolic-address store through
``state_memory_store_symbolic_multi``. The core install no longer maps a
missing candidate page itself: it used to map it as zeros, shadowing any bytes
the host held for it (angr-vmi2l.350). The shell now fetches the page from the
host or, when the host has none, maps it fresh, then retries the store. These
tests pin that retry: the store still lands through the Multi-cell path, not
the single-witness fallback, on a page Rust held and on one it did not.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
# nop (hooked) ; movzx eax, byte ptr [rbx] ; jmp $
CODE = bytes.fromhex("90") + bytes.fromhex("0fb603") + bytes.fromhex("ebfe")
END = LOAD + 4
OPTS = {angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY, angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
STORED = 0xAA


class _TwoSlotStore(angr.SimProcedure):
    """Store STORED at rbx + idx for idx in {0, 1}: a two-candidate address."""

    def run(self):  # pylint: disable=arguments-differ
        idx = claripy.BVS("idx", 64)
        self.state.add_constraints(claripy.ULE(idx, 1))
        self.state.memory.store(self.state.regs.rbx + idx, claripy.BVV(STORED, 8))
        self.jump(claripy.BVV(LOAD + 1, 64))


def _explore(below_sp, backer):
    """Run the hook with rbx ``below_sp`` bytes under the initial sp.

    ``backer`` (or nothing, when None) is written at rbx + 2 in the initial
    state, which makes the page one Rust holds from the start.
    """
    project = angr.load_shellcode(CODE, arch="AMD64", load_address=LOAD)
    project.hook(LOAD, _TwoSlotStore())
    state = project.factory.blank_state(addr=LOAD, add_options=OPTS)
    target = state.solver.eval(state.regs.rsp) - below_sp
    if backer is not None:
        state.memory.store(target + 2, claripy.BVV(backer, 8))
    state.regs.rbx = target + 2
    # The hook stores at target + 2 + {0, 1}; the code then reads target + 2.
    mgr = RustExplorationManager(project, [state])
    mgr.explore(find=END, n=10)
    assert len(mgr.found) == 1
    return mgr, mgr.found[0], target + 2


@pytest.mark.parametrize(
    ("below_sp", "backer", "other"),
    [
        # A page the initial state wrote: Rust holds it, so no retry runs.
        (0x3000, 0x5A, 0x5A),
        # A stack page nobody wrote: it is missing, the host has no bytes for
        # it, so the shell maps it fresh (zero fill) and retries.
        (0x20000, None, 0),
    ],
    ids=["held-page", "fresh-page"],
)
def test_two_candidate_store_lands_in_the_multi_cell_path(below_sp, backer, other):
    mgr, found, addr = _explore(below_sp, backer)
    assert mgr._stats_proxy_mem_symbolic_addr_fallback == 0, "the store fell back to one witness"
    for off in (0, 1):
        byte = found.memory.load(addr + off, 1)
        prior = other if off == 0 else 0
        assert set(found.solver.eval_upto(byte, 4)) == {STORED, prior}
