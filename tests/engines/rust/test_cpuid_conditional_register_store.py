"""CPUID results match the Python engine.

A CPUID block runs in Python's VEX engine (its helper takes a GSPTR
argument). The helper writes one leaf's results per
``state.registers.store(..., condition=old_eax == leaf)``; the register
proxy dropped ``condition``, so every leaf's write landed unconditionally and
the last one won (angr-vmi2l.322).
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
OPTS = {angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY, angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
CPUID = b"\x0f\xa2"
REGS = ("eax", "ebx", "ecx", "edx")


def _explore(engine, arch, eax):
    end = LOAD + len(CPUID)
    proj = angr.load_shellcode(CPUID + b"\xeb\xfe", arch=arch, load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD, add_options=OPTS)
    state.regs.eax = eax
    if engine == "rust":
        from angr.exploration.rust_manager import RustExplorationManager

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=end, max_steps=10)
        errors = [getattr(e, "error", e) for e in mgr.errored]
    else:
        mgr = proj.factory.simulation_manager(state)
        mgr.explore(find=end)
        errors = [e.error for e in mgr.errored]
    assert not errors
    assert len(mgr.found) == 1
    return mgr.found[0]


@pytest.mark.parametrize("arch", ["amd64", "x86"])
@pytest.mark.parametrize("leaf", [0, 1, 0x80000000, 0x80000002])
def test_concrete_leaf_matches_python(arch, leaf):
    py = _explore("py", arch, leaf)
    rust = _explore("rust", arch, leaf)
    py_vals = [py.solver.eval(getattr(py.regs, r)) for r in REGS]
    rust_vals = [rust.solver.eval(getattr(rust.regs, r)) for r in REGS]
    assert rust_vals == py_vals


@pytest.mark.parametrize("arch", ["amd64", "x86"])
def test_symbolic_leaf_keeps_every_leaf(arch):
    """A symbolic eax leaves each leaf's results reachable under its guard."""
    leaf = claripy.BVS("leaf", 32)
    py = _explore("py", arch, leaf)
    rust = _explore("rust", arch, leaf)
    for value in (0, 1, 0x80000000):
        py_ebx = py.solver.eval_upto(py.regs.ebx, 4, extra_constraints=[leaf == value])
        rust_ebx = rust.solver.eval_upto(rust.regs.ebx, 4, extra_constraints=[leaf == value])
        assert sorted(rust_ebx) == sorted(py_ebx)
