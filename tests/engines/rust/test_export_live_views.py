"""Exported SimStates are live views of their Rust state (angr-mx00p.61).

The contract is ``docs/advanced-topics/rust_engine.rst`` "Exported states are
live views"; the mechanism is ``angr/exploration/rust_state_binding.py``
(``CopyStashForkOwner``). These tests pin each clause of it: an export keeps
its manager alive, a copy is a copy-on-write snapshot in both directions, a
step never moves a copy or a snapshot export, a state Rust dropped raises
``RustStateReleasedError``, and an export's solver carries the Rust path
constraints, also through ``copy()`` (angr-vmi2l.262).
"""

from __future__ import annotations

import gc
import weakref

import pytest

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager
from angr.exploration.rust_state_binding import RustStateReleasedError

ADDR = 0x600000

# mov qword [0x600000], rax ; cmp rax, 5 ; ja +2 ; jmp $ ; jmp $
_CMP_CODE = bytes.fromhex("48890425000060004883f8057702ebfeebfe")
_CMP_FIND = 0x400010

# mov qword [rsp-8], 1 ; jmp +0 ; mov qword [rsp-8], 2 ; jmp $
_TWO_STORES_CODE = bytes.fromhex("48c74424f801000000" + "eb00" + "48c74424f802000000" + "ebfe")
_SECOND_BLOCK = 0x40000B


def _project(code):
    return angr.load_shellcode(code, arch="amd64", load_address=0x400000)


def _explore_cmp():
    proj = _project(_CMP_CODE)
    state = proj.factory.blank_state(addr=0x400000)
    x = claripy.BVS("x", 64)
    state.regs.rax = x
    mgr = RustExplorationManager(proj, [state])
    mgr.explore(find=_CMP_FIND)
    assert mgr.found, "the ja branch must be found"
    return mgr, x


def _load_u64(state, addr=ADDR):
    return state.solver.eval(state.memory.load(addr, 8, endness="Iend_LE"))


def _slot(state):
    """The stack slot ``_TWO_STORES_CODE`` writes."""
    return state.solver.eval(state.regs.rsp) - 8


def _two_stores_manager():
    proj = _project(_TWO_STORES_CODE)
    mgr = RustExplorationManager(proj, [proj.factory.blank_state(addr=0x400000)])
    mgr.set_block_granular(True)
    return mgr


class TestSolverOnExports:
    """angr-vmi2l.262: the path constraints live in Rust; an export's solver
    and every copy of it must see them."""

    def test_copy_keeps_the_rust_path_constraints(self):
        mgr, x = _explore_cmp()
        found = mgr.found[0]
        copied = found.copy()
        copied.add_constraints(claripy.ULT(x, 10))
        assert sorted(copied.solver.eval_upto(x, 20)) == [6, 7, 8, 9]
        assert copied.solver.min(x) == 6
        # The copy's guard stays on the copy.
        assert found.solver.satisfiable(extra_constraints=[x == 100])

    def test_constraints_report_the_rust_path_constraints(self):
        mgr, x = _explore_cmp()
        constraints = mgr.found[0].solver.constraints
        assert constraints, "a path-constrained found state must report its constraints"
        assert any(x.variables & c.variables for c in constraints)
        assert not mgr.found[0].solver.satisfiable(extra_constraints=[x == 3])


class TestLifetime:
    def test_export_keeps_its_manager_alive(self):
        mgr, x = _explore_cmp()
        found = mgr.found[0]
        memory = found.memory
        copied = found.copy()
        mgr_ref = weakref.ref(mgr)
        del mgr, found
        gc.collect()
        assert mgr_ref() is not None, "an export holds its manager"
        assert memory.load(ADDR, 8, endness="Iend_LE") is not None
        # A copy can still fork on its first write.
        copied.memory.store(ADDR, claripy.BVV(7, 64), endness="Iend_LE")
        assert _load_u64(copied) == 7
        assert x.variables <= memory.load(ADDR, 8, endness="Iend_LE").variables
        del memory, copied
        gc.collect()
        assert mgr_ref() is None, "the manager goes once no export is referenced"

    def test_dropped_stash_keeps_its_exports(self):
        mgr, _x = _explore_cmp()
        found = mgr.found[0]
        materialized = found._materialize()
        before = materialized.memory.load(ADDR, 8, endness="Iend_LE")
        mgr.drop(stash="found")
        assert not mgr._rust_mgr.get_state_ids("found")
        assert materialized.memory.load(ADDR, 8, endness="Iend_LE") is before or (
            materialized.memory.load(ADDR, 8, endness="Iend_LE").variables == before.variables
        )
        assert materialized.solver.satisfiable()

    def test_state_rust_dropped_raises_released(self):
        mgr = _two_stores_manager()
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        export = mgr.active[0]._materialize()
        assert mgr._rust_mgr.drop_state_from_stash(sid, "active")
        mgr.step()
        with pytest.raises(RustStateReleasedError, match=f"Rust state {sid}"):
            export.memory.load(0x1000, 8)


class TestStepping:
    """Rust steps stash states without a binding; the views that must not
    follow are detached before it does."""

    def test_copy_taken_before_a_step_keeps_the_old_state(self):
        mgr = _two_stores_manager()
        mgr.step()
        assert mgr.active[0].addr == _SECOND_BLOCK
        slot = _slot(mgr.active[0])
        copied = mgr.active[0].copy()
        assert _load_u64(copied, slot) == 1
        mgr.step()
        assert _load_u64(mgr.active[0], slot) == 2, "the stash entry follows its state"
        assert _load_u64(copied, slot) == 1, "a copy keeps the state as of the copy"
        assert copied.addr == _SECOND_BLOCK

    def test_snapshot_export_keeps_the_old_state(self):
        mgr = _two_stores_manager()
        mgr.step()
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        snap = mgr.get_state_by_id(sid)
        slot = _slot(snap)
        assert _load_u64(snap, slot) == 1
        mgr.step()
        assert _load_u64(snap, slot) == 1
        assert snap.callstack is not None and len(snap.callstack) == 0

    def test_held_solver_follows_the_stepped_state(self):
        proj = _project(_CMP_CODE)
        state = proj.factory.blank_state(addr=0x400000)
        x = claripy.BVS("x", 64)
        state.regs.rax = x
        mgr = RustExplorationManager(proj, [state])
        mgr.set_block_granular(True)
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        solver = mgr.active[0].solver
        assert (solver.min(x), solver.max(x)) == (0, 2**64 - 1), "no path constraint before the branch"
        mgr.step()
        assert sid in mgr._rust_mgr.get_state_ids("active")
        assert solver.constraints, "the branch constrains the state that keeps its id"
        truth = mgr._rust_mgr.fork_state_solver(sid)
        held = (solver.min(x), solver.max(x))
        assert held == (truth.min(x), truth.max(x)), "a held solver kept its pre-step fork"

    def test_step_without_copies_forks_nothing(self):
        mgr = _two_stores_manager()
        mgr.step()
        _ = mgr.active[0].memory.load(_slot(mgr.active[0]), 8)
        assert mgr._copy_stash_fork_owner.detach_views() == 0


class TestCopyIsolation:
    def test_writes_stay_on_their_side(self):
        mgr, _x = _explore_cmp()
        found = mgr.found[0]
        copied = found.copy()
        copied.memory.store(ADDR, claripy.BVV(0x41, 64), endness="Iend_LE")
        assert _load_u64(copied) == 0x41
        assert found.memory.load(ADDR, 8, endness="Iend_LE").symbolic, "the copy's write reached the original"
        other = found.copy()
        found.memory.store(ADDR, claripy.BVV(0x42, 64), endness="Iend_LE")
        assert _load_u64(found) == 0x42
        assert other.memory.load(ADDR, 8, endness="Iend_LE").symbolic, "the original's write reached a copy"
        assert _load_u64(copied) == 0x41

    def test_copy_forks_are_released_with_the_copy(self):
        mgr, _x = _explore_cmp()
        copied = mgr.found[0].copy()
        copied.memory.store(ADDR, claripy.BVV(1, 64), endness="Iend_LE")
        assert len(mgr._rust_mgr.get_state_ids("_copies")) == 1
        del copied
        gc.collect()
        assert mgr._rust_mgr.get_state_ids("_copies") == []


class TestReadOnlySymbolicLoad:
    """angr-vmi2l.259: a symbolic-address load through a view or an export
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

    @pytest.mark.parametrize("view", ["proxy", "export"])
    def test_load_leaves_the_state_alone(self, view):
        mgr, x = self._manager()
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        before = len(mgr._rust_mgr.export_state_constraints(sid))
        state = mgr.active_proxies()[0] if view == "proxy" else mgr.active[0]
        state.memory.load(x, 1)
        assert len(mgr._rust_mgr.export_state_constraints(sid)) == before
        mgr.step()
        assert len(mgr.active) == 2, "the load pinned x and dropped a feasible branch"
