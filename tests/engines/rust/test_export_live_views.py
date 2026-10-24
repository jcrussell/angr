"""A read through a Rust state view never changes that state (angr-vmi2l.259)."""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager


def _project(code):
    return angr.load_shellcode(code, arch="amd64", load_address=0x400000)


class TestReadOnlySymbolicLoad:
    """angr-vmi2l.259: a symbolic-address load through a view proxy
    never changes the Rust state's constraints."""

    # cmp rdi, 0x601008 ; jb +2 ; jmp $ ; jmp $
    _CODE = bytes.fromhex("4881ff08106000" + "7202" + "ebfe" + "ebfe")

    def _manager(self):
        proj = _project(self._CODE)
        state = proj.factory.blank_state(addr=0x400000)
        x = claripy.BVS("x", 64)
        state.regs.rdi = x
        state.add_constraints(claripy.UGE(x, 0x601000), claripy.ULT(x, 0x601010))
        mgr = RustExplorationManager(proj, [state])
        return mgr, x

    @pytest.mark.parametrize("view", ["proxy"])
    def test_load_leaves_the_state_alone(self, view):
        mgr, x = self._manager()
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        before = len(mgr._rust_mgr.export_state_constraints(sid))
        state = mgr.active_proxies()[0]
        state.memory.load(x, 1)
        assert len(mgr._rust_mgr.export_state_constraints(sid)) == before
        mgr.step()
        assert len(mgr.active) == 2, "the load pinned x and dropped a feasible branch"
