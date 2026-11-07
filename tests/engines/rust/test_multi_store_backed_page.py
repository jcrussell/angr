"""A symbolic-address SimProcedure store concretizes as Python's does, and maps its page.

``RustMemoryProxy`` sends a symbolic-address store through
``state_memory_store_symbolic_multi``. The core install no longer maps a
missing candidate page itself: it used to map it as zeros, shadowing any bytes
the host held for it (angr-vmi2l.350). The shell now fetches the page from the
host or, when the host has none, maps it fresh, then retries the store. These
tests pin that retry: the store lands in Rust, not the proxy's single-witness
fallback, on a page Rust held and on one it did not.

They also pin the write strategy against the Python engine. Only an address
carrying a ``MultiwriteAnnotation`` gets the Range strategy with
SYMBOLIC_WRITE_ADDRESSES off, so it writes both candidates; any other address
is concretized to its Max solution, so only rbx + 1 changes (angr-vmi2l.351).
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from angr.storage.memory_mixins.address_concretization_mixin import MultiwriteAnnotation
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

    def run(self, annotate=False):  # pylint: disable=arguments-differ
        idx = claripy.BVS("idx", 64)
        self.state.add_constraints(claripy.ULE(idx, 1))
        addr = self.state.regs.rbx + idx
        if annotate:
            addr = addr.annotate(MultiwriteAnnotation())
        self.state.memory.store(addr, claripy.BVV(STORED, 8))
        self.jump(claripy.BVV(LOAD + 1, 64))


def _explore(below_sp, backer, annotate, rust=True):
    """Run the hook with rbx ``below_sp`` bytes under the initial sp.

    ``backer`` (or nothing, when None) is written at rbx + 2 in the initial
    state, which makes the page one Rust holds from the start.
    """
    project = angr.load_shellcode(CODE, arch="AMD64", load_address=LOAD)
    project.hook(LOAD, _TwoSlotStore(annotate=annotate))
    state = project.factory.blank_state(addr=LOAD, add_options=OPTS)
    target = state.solver.eval(state.regs.rsp) - below_sp
    if backer is not None:
        state.memory.store(target + 2, claripy.BVV(backer, 8))
    state.regs.rbx = target + 2
    # The hook stores at target + 2 + {0, 1}; the code then reads target + 2.
    mgr = RustExplorationManager(project, [state]) if rust else project.factory.simulation_manager(state)
    mgr.explore(find=END, n=10)
    assert len(mgr.found) == 1
    return mgr, mgr.found[0], target + 2


def _byte_values(found, addr):
    return [set(found.solver.eval_upto(found.memory.load(addr + off, 1), 4)) for off in (0, 1)]


@pytest.mark.parametrize("annotate", [False, True], ids=["plain", "multiwrite"])
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
def test_symbolic_store_matches_python(below_sp, backer, other, annotate):
    mgr, found, addr = _explore(below_sp, backer, annotate)
    assert mgr._stats_proxy_mem_symbolic_addr_fallback == 0, "the store fell back to the proxy's witness"
    expected = [{STORED, other}, {STORED, 0}] if annotate else [{other}, {STORED}]
    assert _byte_values(found, addr) == expected
    _, py_found, py_addr = _explore(below_sp, backer, annotate, rust=False)
    assert _byte_values(py_found, py_addr) == expected
