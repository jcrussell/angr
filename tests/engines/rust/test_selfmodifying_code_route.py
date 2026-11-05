"""Self-modifying code stays on Python, and a proxy reads below SP as Python does (angr-mx00p.132.30).

upstream test_self_modifying_code loads below SP under STRICT_PAGE_ACCESS, then
runs cgc/stuff, which copies its code onto the stack and patches the next
instruction of the running block. Two gaps:

* ``RustSolverProxyPlugin.is_true`` answered False for a concrete BV, so the
  fallback memory's ``PrivilegedPagingMixin`` (``is_true(page.perm_read)``)
  saw a mapped stack page as unreadable and the load segfaulted.
* The Rust engine neither relifts mid-block nor invalidates code cached
  outside the loaded binaries, so the stack code ran stale and looped.
  ``rust_engine_eligible`` and ``RustBackend`` keep such projects on Python.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import angr
from angr import claripy
from angr import options as o
from angr.exploration.rust_backend import RustBackend
from angr.exploration.rust_manager import (
    SELFMODIFYING_CODE_REASON,
    RustExplorationManager,
    resolve_main_address,
    rust_engine_eligible,
)
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, TEST_BINARIES_DIR, load_fauxware

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")

CGC_STUFF = os.path.join(TEST_BINARIES_DIR, "..", "cgc", "stuff")


def test_proxy_reads_a_concrete_bv_as_its_truth_value():
    proj = angr.load_shellcode(b"\x90\xc3", arch="amd64")
    state = RustExplorationManager(proj, [proj.factory.entry_state()]).active[0]
    assert state.solver.is_true(claripy.BVV(1, 3))
    assert not state.solver.is_false(claripy.BVV(2, 3))
    assert state.solver.is_false(claripy.BVV(0, 3))
    assert not state.solver.is_true(claripy.BVV(0, 3))


def test_strict_load_below_sp_reads_the_allocated_stack_page():
    if not os.path.exists(CGC_STUFF):
        pytest.skip("cgc/stuff binary not found")
    proj = angr.Project(CGC_STUFF, auto_load_libs=False)
    state = RustExplorationManager(proj, [proj.factory.entry_state(add_options={o.STRICT_PAGE_ACCESS})]).active[0]
    for offs in range(0, 0x6000, 0x1000):
        assert state.memory.load(state.regs.sp - offs, size=1).size() == 8


def test_selfmodifying_project_is_ineligible():
    project = SimpleNamespace(arch=SimpleNamespace(name="AMD64"), selfmodifying_code=True)
    state = SimpleNamespace(options={"EXTENDED_IROP_SUPPORT"})
    assert rust_engine_eligible(project, [state], {}) == (False, SELFMODIFYING_CODE_REASON)


def test_backend_declines_a_selfmodifying_project():
    proj = load_fauxware(selfmodifying_code=True)
    state = proj.factory.blank_state(addr=resolve_main_address(proj))
    simgr = angr.SimulationManager(proj, [state])
    reason = RustBackend(proj).decline_reason(simgr, state, None, {})
    assert reason == SELFMODIFYING_CODE_REASON
