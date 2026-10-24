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


# mov rax, 1 ; mov qword [0x600000], rax ; jmp +0
# 0x400011: cmp rdi, 5 ; ja +2 ; jmp $
# 0x400019: mov rax, 2 ; mov qword [0x600000], rax ; jmp $
_HELD_CODE = bytes.fromhex(
    "48c7c001000000"
    + "4889042500006000"
    + "eb00"
    + "4883ff05"
    + "7702"
    + "ebfe"
    + "48c7c002000000"
    + "4889042500006000"
    + "ebfe"
)
_HELD_BRANCH = 0x400011
_HELD_TAKEN = 0x400019

# cmp rdi, 5 ; ja +9 ; mov rax, 1 ; jmp +7 ; mov rax, 2 ; 0x400016: jmp $
_JOIN_CODE = bytes.fromhex("4883ff05" + "7709" + "48c7c001000000" + "eb07" + "48c7c002000000" + "ebfe")
_JOIN = 0x400016


def _held_managers(code):
    """A Python-engine simgr and a Rust manager over the same start state."""
    proj = _project(code)
    state = proj.factory.blank_state(addr=0x400000)
    rdi = claripy.BVS("rdi", 64)
    state.regs.rdi = rdi
    # Map the slot the code writes: Rust and Python treat a store to a page
    # nobody mapped differently, which is not what these tests are about.
    state.memory.store(ADDR, claripy.BVV(0, 64))
    py = proj.factory.simgr(state.copy())
    mgr = RustExplorationManager(proj, [state.copy()])
    mgr.set_block_granular(True)
    return py, mgr, rdi


def _entry_at(stash, addr):
    matches = [s for s in stash if s.addr == addr]
    assert len(matches) == 1, f"one state at {addr:#x}, got {[hex(s.addr) for s in stash]}"
    return matches[0]


def _view(state, rdi):
    """What a held entry shows: pc, a register, a memory slot, its constraints."""
    return {
        "addr": state.addr,
        "rax": state.solver.eval(state.regs.rax),
        "mem": _load_u64(state),
        "sat_rdi_3": state.solver.satisfiable(extra_constraints=[rdi == 3]),
        "sat_rdi_9": state.solver.satisfiable(extra_constraints=[rdi == 9]),
    }


def _constraint_hashes(state):
    return [c.hash() for c in state.solver.constraints]


def _copies(mgr):
    return list(mgr._rust_mgr.get_state_ids("_copies"))


class TestHeldStashEntries:
    """A stash entry held across a step keeps its pre-step state, as under
    the Python engine; one nobody holds costs no fork."""

    def test_held_entry_keeps_its_pre_step_state(self):
        py, mgr, rdi = _held_managers(_HELD_CODE)
        py.step()
        mgr.step()
        held_py, held = py.active[0], mgr.active[0]
        before = _view(held, rdi)
        assert before == _view(held_py, rdi)
        assert before["addr"] == _HELD_BRANCH and before["rax"] == 1
        constraints = _constraint_hashes(held)
        py.step()
        mgr.step()
        assert _view(held, rdi) == before == _view(held_py, rdi)
        assert _constraint_hashes(held) == constraints
        # The stash hands out fresh entries for the stepped states.
        assert all(s is not held for s in mgr.active)
        taken = _entry_at(mgr.active, _HELD_TAKEN)
        assert not taken.solver.satisfiable(extra_constraints=[rdi == 3])
        assert _view(taken, rdi) == _view(_entry_at(py.active, _HELD_TAKEN), rdi)
        # A second step moves the fresh entry and leaves the held one alone.
        held_taken_py, held_taken = _entry_at(py.active, _HELD_TAKEN), taken
        py.step()
        mgr.step()
        assert _view(held_taken, rdi) == _view(held_taken_py, rdi)
        assert _view(held_taken, rdi)["rax"] == 1
        assert _view(_entry_at(mgr.active, 0x400028), rdi) == _view(_entry_at(py.active, 0x400028), rdi)
        assert _view(held, rdi) == before

    def test_writes_to_a_held_entry_stay_on_it(self):
        py, mgr, rdi = _held_managers(_HELD_CODE)
        py.step()
        mgr.step()
        held_py, held = py.active[0], mgr.active[0]
        py.step()
        mgr.step()
        for state in (held_py, held):
            state.regs.rax = 0x41
            state.memory.store(ADDR, claripy.BVV(0x42, 64), endness="Iend_LE")
            state.add_constraints(rdi == 3)
        assert _view(held, rdi) == _view(held_py, rdi)
        assert _view(held, rdi)["rax"] == 0x41
        views = [sorted((_view(s, rdi) for s in stash), key=lambda v: v["addr"]) for stash in (py.active, mgr.active)]
        assert views[0] == views[1]
        assert [(v["rax"], v["mem"]) for v in views[1]] == [(1, 1), (1, 1)]
        py.step()
        mgr.step()
        assert _view(_entry_at(mgr.active, 0x400028), rdi) == _view(_entry_at(py.active, 0x400028), rdi)

    @pytest.mark.parametrize("how", ["run", "explore"])
    def test_held_entry_across_run_and_explore(self, how):
        py, mgr, rdi = _held_managers(_HELD_CODE)
        py.step()
        mgr.step()
        held_py, held = py.active[0], mgr.active[0]
        before = _view(held, rdi)
        for sm in (py, mgr):
            if how == "run":
                sm.run(n=2)
            else:
                sm.explore(find=0x400028)
        assert _view(held, rdi) == before == _view(held_py, rdi)
        if how == "explore":
            assert _view(mgr.found[0], rdi) == _view(py.found[0], rdi)

    def test_held_entries_across_merge(self):
        py, mgr, rdi = _held_managers(_JOIN_CODE)
        for sm in (py, mgr):
            # A block-granular Rust step may advance one state of the two.
            for _ in range(4):
                if [s.addr for s in sm.active] == [_JOIN, _JOIN]:
                    break
                sm.step()
        held_py, held = list(py.active), list(mgr.active)
        assert sorted(s.addr for s in held) == [_JOIN, _JOIN]
        before = sorted((_view(s, rdi) for s in held), key=lambda v: v["rax"])
        assert before == sorted((_view(s, rdi) for s in held_py), key=lambda v: v["rax"])
        py.merge()
        mgr.merge()
        assert len(mgr.active) == len(py.active) == 1
        assert sorted((_view(s, rdi) for s in held), key=lambda v: v["rax"]) == before
        merged = mgr.active[0]
        assert merged not in held
        assert sorted(merged.solver.eval_upto(merged.regs.rax, 3)) == [1, 2]

    def test_held_entry_whose_state_rust_drops(self):
        py, mgr, rdi = _held_managers(_HELD_CODE)
        py.step()
        mgr.step()
        held_py, held = py.active[0], mgr.active[0]
        sid = held._lazy_state_id
        before = _view(held, rdi)
        # Rust drops the state while it runs, after the step hook detached
        # the held entry (a native merge or a retired id does the same).
        sync = mgr._sync_hooks_before_step

        def drop_while_running():
            sync()
            assert mgr._rust_mgr.drop_state_from_stash(sid, "active")

        mgr._sync_hooks_before_step = drop_while_running
        py.step()
        mgr.step()
        assert mgr._rust_mgr.get_state_pc_by_id(sid) is None, "Rust dropped the state"
        assert _view(held, rdi) == before == _view(held_py, rdi)
        held.memory.store(ADDR, claripy.BVV(5, 64), endness="Iend_LE")
        assert _load_u64(held) == 5

    def test_unheld_entries_fork_nothing(self):
        _py, mgr, rdi = _held_managers(_HELD_CODE)
        mgr.step()
        assert _view(mgr.active[0], rdi)["rax"] == 1
        assert [s.addr for s in mgr.active] == [_HELD_BRANCH]
        mgr.step()
        assert _copies(mgr) == []
        assert len(mgr.active) == 2
        mgr.step()
        mgr.explore(find=0x400028)
        assert _copies(mgr) == []

    def test_held_entry_fork_is_released_with_it(self):
        _py, mgr, rdi = _held_managers(_HELD_CODE)
        mgr.step()
        held = mgr.active[0]
        mgr.step()
        assert len(_copies(mgr)) == 1
        assert _view(held, rdi)["addr"] == _HELD_BRANCH
        mgr.step()
        assert len(_copies(mgr)) == 1, "a pinned entry is not detached again"
        del held
        gc.collect()
        assert _copies(mgr) == []

    def test_dropped_stash_keeps_held_entries(self):
        py, mgr, rdi = _held_managers(_HELD_CODE)
        py.step()
        mgr.step()
        held_py, held = py.active[0], mgr.active[0]
        before = _view(held, rdi)
        py.drop(stash="active")
        mgr.drop(stash="active")
        assert not mgr.active
        assert _view(held, rdi) == before == _view(held_py, rdi)

    def test_symbolic_register_proxy_shares_the_export_binding(self):
        from angr.exploration.rust_state_binding import binding_of
        from angr.exploration.rust_state_proxy import RustRegisterProxy

        _py, mgr, _rdi = _held_managers(_HELD_CODE)
        mgr.step()
        state = mgr.active[0]._materialize()
        assert isinstance(state.registers, RustRegisterProxy), "rdi is symbolic, so registers read Rust live"
        binding = binding_of(state)
        assert binding is not None and not binding.is_virtual
        copied = state.copy()
        copied.regs.rax = 9
        assert binding_of(copied) is not binding
        assert (copied.solver.eval(copied.regs.rax), _load_u64(copied)) == (9, 1)
        assert state.solver.eval(state.regs.rax) == 1


# cmp rax, 4 ; jb +2 ; jmp $
# 0x400008: movzx ebx, byte [rax + 0x600000] ; cmp bl, 3 ; je +2 ; jmp $ ; jmp $
_TABLE_CODE = bytes.fromhex("4883f804" + "7202" + "ebfe" + "0fb69800006000" + "80fb03" + "7402" + "ebfe" + "ebfe")
_TABLE_LOOKUP = 0x400008


class TestHandedToThePythonEngine:
    """An export the Python engine steps gets plugins of its own first
    (``RustMemoryProxy.detach_for_python_engine``), so the run matches a
    Python-only one (angr-vmi2l.260)."""

    @staticmethod
    def _found(code, find, setup):
        proj = _project(code)
        state = proj.factory.blank_state(addr=0x400000)
        x = claripy.BVS("x", 64)
        setup(state, x)
        py = proj.factory.simgr(state.copy())
        py.explore(find=find)
        mgr = RustExplorationManager(proj, [state.copy()])
        mgr.explore(find=find)
        assert len(py.found) == len(mgr.found) == 1
        return proj, py.found[0], mgr, x

    @staticmethod
    def _table(state, x):
        state.regs.rax = x
        state.memory.store(0x600000, b"\x01\x02\x03\x04")

    @staticmethod
    def _summary(sm, x):
        return sorted((s.addr, tuple(sorted(s.solver.eval_upto(x, 5)))) for s in sm.active)

    @pytest.mark.parametrize("which", ["found_states", "copy"])
    def test_step_matches_a_python_run(self, which):
        proj, py_found, mgr, x = self._found(_TABLE_CODE, _TABLE_LOOKUP, self._table)
        export = mgr.found_states[0] if which == "found_states" else mgr.found[0].copy()
        ref = proj.factory.simgr(py_found.copy())
        ref.step()
        sm = proj.factory.simgr(export)
        sm.step()
        # A symbolic-address load through a proxy concretized x to one value
        # and lost the bl == 3 branch.
        assert self._summary(sm, x) == self._summary(ref, x) == [(0x400014, (0, 1, 3)), (0x400016, (2,))]
        from angr.exploration.rust_state_binding import binding_of

        for succ in sm.active:
            assert binding_of(succ) is None, "a Python-stepped successor has plugins of its own"
            assert succ.solver.constraints, "it carries the Rust path constraints"
        assert binding_of(export) is not None, "the export itself stays a view"
        assert export.addr == _TABLE_LOOKUP

    def test_merge_after_a_python_step(self):
        def setup(state, x):
            state.regs.rdi = x

        proj, py_found, mgr, x = self._found(_JOIN_CODE, 0x400000, setup)
        results = []
        for start in (py_found.copy(), mgr.found[0].copy()):
            sm = proj.factory.simgr(start)
            while [s.addr for s in sm.active] != [_JOIN, _JOIN]:
                sm.step()
            sm.merge()
            assert len(sm.active) == 1
            merged = sm.active[0]
            results.append(
                (
                    sorted(merged.solver.eval_upto(merged.regs.rax, 3)),
                    merged.solver.eval(merged.regs.rax, extra_constraints=[x == 9]),
                )
            )
        assert results[0] == results[1] == ([1, 2], 2)

    def test_merging_views_raises(self):
        mgr, _x = _explore_cmp()
        a, b = mgr.found[0].copy(), mgr.found[0].copy()
        with pytest.raises(NotImplementedError, match="cannot merge Rust states"):
            a.merge(b)
