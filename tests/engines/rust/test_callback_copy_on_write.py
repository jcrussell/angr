# pylint: disable=missing-class-docstring,no-self-use
"""Copy-on-write callback successors (angr-mx00p.112, angr-vmi2l.294).

Every successor of a bounced SimProcedure, syscall or Python-VEX-fallback
block is a ``SimState.copy()`` of the callback state. Its Rust proxies hold a
copy-on-write ``RustStateBinding`` (``angr/exploration/rust_state_binding.py``):
the copy reads the callback state as of the copy and its first write forks
its own Rust state. These tests pin that against the Python engine.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from angr.exploration.rust_state_binding import (
    CALLBACK_FORK_STASH,
    CallbackForkScope,
    RustStateReleasedError,
    unscoped_binding,
)
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

ZERO_FILL = {angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY, angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}

HOOK_ADDR = 0x4008C0
TARGET_1 = 0x4008B0
TARGET_2 = 0x4008B8
SCRATCH = 0x600000


def _two_successor_proc(sym):
    """A hook that forks two successors, each writing its own memory and
    register value under its own guard on ``sym``."""

    class TwoSuccessors(angr.SimProcedure):
        ADDS_EXITS = True
        NO_RET = True

        def run(self):  # pylint: disable=arguments-differ
            for target, value in ((TARGET_1, 0x111), (TARGET_2, 0x222)):
                succ = self.state.copy()
                succ.memory.store(SCRATCH, claripy.BVV(value, 64), endness="Iend_LE")
                succ.regs.rax = value
                self.successors.add_successor(succ, target, sym == value, "Ijk_Boring")

    return TwoSuccessors()


def _summary(state, sym, other_values):
    """What a successor looks like from the outside: pc, memory, register,
    the guard's model and whether a sibling's guard is still feasible."""
    return (
        state.addr,
        state.solver.eval(state.memory.load(SCRATCH, 8, endness="Iend_LE")),
        state.solver.eval(state.regs.rax),
        sorted(state.solver.eval_upto(sym, 3)),
        [state.solver.satisfiable(extra_constraints=[sym == v]) for v in other_values],
    )


class TestTwoSuccessorHookParity:
    def test_successors_keep_their_own_writes_and_guards(self, hookable_project):
        proj = hookable_project
        sym = claripy.BVS("cow_guard", 64)
        proj.hook(HOOK_ADDR, _two_successor_proc(sym))

        py_simgr = proj.factory.simgr(proj.factory.blank_state(addr=HOOK_ADDR, add_options=ZERO_FILL))
        py_simgr.step()
        expected = sorted(_summary(s, sym, (0x111, 0x222)) for s in py_simgr.active)
        assert [e[0] for e in expected] == [TARGET_1, TARGET_2]

        mgr = RustExplorationManager(
            proj, [proj.factory.blank_state(addr=HOOK_ADDR, add_options=ZERO_FILL)]
        )
        mgr.run(max_steps=1)
        got = sorted(_summary(s, sym, (0x111, 0x222)) for s in mgr.active)

        assert got == expected
        # Every fork the copies made was claimed or dropped.
        assert mgr._rust_mgr.stash_count(CALLBACK_FORK_STASH) == 0

    def test_rust_constraints_are_per_successor(self, hookable_project):
        """Each successor's own Rust solver holds its own guard and only that
        one (the gated solver proxy used to land both guards on one solver)."""
        proj = hookable_project
        sym = claripy.BVS("cow_guard", 64)
        proj.hook(HOOK_ADDR, _two_successor_proc(sym))
        mgr = RustExplorationManager(
            proj, [proj.factory.blank_state(addr=HOOK_ADDR, add_options=ZERO_FILL)]
        )
        mgr.run(max_steps=1)
        rust = mgr._rust_mgr
        guards = {}
        for sid in rust.get_state_ids("active"):
            guards[rust.get_state_pc_by_id(sid)] = [str(c) for c in rust.export_state_constraints(sid) if c.symbolic]
        assert set(guards) == {TARGET_1, TARGET_2}
        assert len(guards[TARGET_1]) == 1 and "0x111" in guards[TARGET_1][0]
        assert len(guards[TARGET_2]) == 1 and "0x222" in guards[TARGET_2][0]


def test_extra_successor_keeps_pre_bounce_path_constraints():
    """angr-vmi2l.294: an extra successor is forked from the pending Rust
    state, so it keeps the constraint Rust collected before the bounce
    (``rdi < 5``). It used to be rebuilt from the Python callback state,
    which never held that constraint."""
    # 0x400000: cmp rdi, 5 ; jae 0x400010 ; (hook at 0x400006) nops ; ret
    code = bytes.fromhex("4883ff05730a") + b"\x90" * 10 + b"\xc3"
    proj = angr.load_shellcode(code, arch="amd64", start_offset=0, load_address=0x400000)
    sel = claripy.BVS("cow_sel", 64)

    class Fork(angr.SimProcedure):
        ADDS_EXITS = True
        NO_RET = True

        def run(self):  # pylint: disable=arguments-differ
            for target, value in ((0x40000A, 1), (0x40000C, 2)):
                self.successors.add_successor(self.state.copy(), target, sel == value, "Ijk_Boring")

    proj.hook(0x400006, Fork())

    def found_max_rdi(engine_simgr):
        found = engine_simgr.found[0]
        return found.solver.max(found.regs.rdi), found.solver.eval_upto(sel, 3)

    def entry():
        st = proj.factory.blank_state(addr=0x400000, add_options=ZERO_FILL)
        st.regs.rdi = claripy.BVS("cow_x", 64)
        return st

    expected = found_max_rdi(proj.factory.simgr(entry()).explore(find=0x40000C))
    mgr = RustExplorationManager(proj, [entry()])
    mgr.explore(find=0x40000C)
    assert len(mgr.found) == 1
    assert found_max_rdi(mgr) == expected == (4, [2])


# movq xmm0, rdi ; pmaddwd xmm0, xmm0 ; movq rax, xmm0 ; mov [0x600000], rax ;
# test rax, rax ; jz +2 ; nop ; nop ; ret
#
# Rust routes a symbolic ``pmaddwd`` to the Python VEX engine, so the whole
# block, conditional exit included, runs in Python and returns two successors.
_VEX_FALLBACK_CODE = bytes.fromhex("66480f6ec7660ff5c066480f7ec048890425000060004885c074029090c3")
_VEX_TAKEN = 0x40001D
_VEX_FALLTHROUGH = 0x40001B


def test_python_vex_fallback_conditional_exit_parity():
    proj = angr.load_shellcode(_VEX_FALLBACK_CODE, arch="amd64", start_offset=0, load_address=0x400000)
    x = claripy.BVS("cow_vex_x", 64)

    def entry():
        st = proj.factory.blank_state(addr=0x400000, add_options=ZERO_FILL)
        st.regs.rdi = x
        return st

    def summary(state):
        rax = state.regs.rax
        stored = state.memory.load(SCRATCH, 8, endness="Iend_LE")
        return (
            state.addr,
            state.solver.satisfiable(extra_constraints=[rax == 0]),
            state.solver.satisfiable(extra_constraints=[rax != 0]),
            state.solver.satisfiable(extra_constraints=[stored != rax]),
        )

    py_simgr = proj.factory.simgr(entry())
    py_simgr.step()
    expected = sorted(summary(s) for s in py_simgr.active)
    assert [e[0] for e in expected] == [_VEX_FALLTHROUGH, _VEX_TAKEN]

    mgr = RustExplorationManager(proj, [entry()])
    mgr.run(max_steps=1)
    assert mgr._perf_stats._stats["callback_vex_fallback_count"] == 1
    got = sorted(summary(s) for s in mgr.active)
    assert got == expected
    assert mgr._rust_mgr.stash_count(CALLBACK_FORK_STASH) == 0


def test_copy_reads_parent_as_of_copy_and_unclaimed_copy_is_released(hookable_project):
    """A copy sees the callback state as it was when copied, the callback
    state does not see the copy's writes, and a copy that is not a successor
    is released when the callback finishes: reading it raises instead of
    reading whatever state its id would name next."""
    proj = hookable_project
    seen = {}

    class Copier(angr.SimProcedure):
        def run(self):  # pylint: disable=arguments-differ
            snapshot = self.state.copy()
            self.state.memory.store(SCRATCH, claripy.BVV(0xAA, 64), endness="Iend_LE")
            self.state.regs.rbx = 0xAA
            scratch = self.state.copy()
            scratch.memory.store(SCRATCH, claripy.BVV(0xBB, 64), endness="Iend_LE")
            scratch.regs.rbx = 0xBB
            scratch.add_constraints(scratch.regs.rdi == 7)

            def load(s):
                return s.solver.eval(s.memory.load(SCRATCH, 8, endness="Iend_LE"))

            seen["snapshot"] = (load(snapshot), snapshot.solver.eval(snapshot.regs.rbx))
            seen["scratch"] = (load(scratch), scratch.solver.eval(scratch.regs.rbx))
            seen["state"] = (load(self.state), self.state.solver.eval(self.state.regs.rbx))
            seen["state_rdi_free"] = self.state.solver.satisfiable(extra_constraints=[self.state.regs.rdi == 8])
            seen["kept"] = scratch
            return 0

    proj.hook(HOOK_ADDR, Copier())
    st = proj.factory.blank_state(addr=HOOK_ADDR, add_options=ZERO_FILL)
    st.regs.rdi = claripy.BVS("cow_rdi", 64)
    mgr = RustExplorationManager(proj, [st])
    mgr.run(max_steps=1)

    assert seen["snapshot"] == (0, 0)
    assert seen["scratch"] == (0xBB, 0xBB)
    assert seen["state"] == (0xAA, 0xAA)
    assert seen["state_rdi_free"] is True
    with pytest.raises(RustStateReleasedError):
        seen["kept"].memory.load(SCRATCH, 8)
    assert mgr._rust_mgr.stash_count(CALLBACK_FORK_STASH) == 0
    # The resumed state carries the callback state's own writes.
    (resumed,) = [s for s in list(mgr.active) + list(mgr.deadended) + list(mgr.unconstrained)]
    assert resumed.solver.eval(resumed.memory.load(SCRATCH, 8, endness="Iend_LE")) == 0xAA


class _FakeMgr:
    """Just enough of ``RustExplorationManager`` for the scope's bookkeeping:
    ids are minted locally and every fork/drop is recorded."""

    def __init__(self):
        self.next_id = 100
        self.parked = {}
        self.dropped = []
        self._copy_stash_fork_owner = None

    def fork_state_for_copy(self, source_id, stash="_copies"):
        self.next_id += 1
        self.parked[self.next_id] = (source_id, stash)
        return self.next_id

    def drop_copy(self, state_id, stash="_copies"):
        if self.parked.get(state_id, (None, None))[1] != stash:
            return False
        del self.parked[state_id]
        self.dropped.append(state_id)
        return True


class _Holder:
    """A stand-in SimState: ``binding_of`` only reads ``plugins``."""

    def __init__(self, binding):
        self.plugins = {"memory": type("P", (), {})()}
        self.plugins["memory"]._binding = binding


class TestCallbackForkScope:
    def test_unwritten_copies_cost_no_fork_and_first_successor_takes_the_pending_state(self):
        mgr = _FakeMgr()
        scope = CallbackForkScope(mgr, 7)
        copy = scope.root.copy()
        assert copy.state_id == 7 and copy.is_virtual
        resumed, extras = scope.settle([_Holder(copy)])
        assert (resumed, extras) == (7, [])
        assert mgr.parked == {}
        with pytest.raises(RustStateReleasedError):
            _ = scope.root.state_id
        scope.close()
        assert copy.state_id == 7

    def test_writes_fork_and_siblings_and_children_stay_isolated(self):
        mgr = _FakeMgr()
        scope = CallbackForkScope(mgr, 7)
        a = scope.root.copy()
        b = scope.root.copy()
        a_child = a.copy()
        assert a.writable_id() == 101  # a forks from the pending state
        assert a_child.state_id == 102  # sheltered onto a snapshot of a, pre-write
        assert mgr.parked[102] == (101, CALLBACK_FORK_STASH)
        assert b.state_id == 7
        assert scope.root.writable_id() == 7  # b is sheltered before the root is written
        assert b.state_id == 103 and mgr.parked[103] == (7, CALLBACK_FORK_STASH)

        resumed, extras = scope.settle([_Holder(a), _Holder(b)])
        assert (resumed, extras) == (101, [103])
        # The snapshot nobody claimed is dropped at once; the child reading it
        # is released.
        assert mgr.dropped == [102]
        with pytest.raises(RustStateReleasedError):
            _ = a_child.state_id
        scope.consumed(101)
        scope.close()
        # 103 was claimed but never routed: close drops it.
        assert mgr.dropped == [102, 103]
        with pytest.raises(RustStateReleasedError):
            b.writable_id()

    def test_extra_successor_reading_the_pending_state_gets_its_own_fork(self):
        mgr = _FakeMgr()
        scope = CallbackForkScope(mgr, 7)
        extra = scope.root.copy()
        resumed, extras = scope.settle([_Holder(scope.root), _Holder(extra)])
        assert resumed == 7
        assert extras == [101] and mgr.parked[101] == (7, CALLBACK_FORK_STASH)

    def test_close_without_settle_drops_every_fork_and_keeps_the_root(self):
        mgr = _FakeMgr()
        scope = CallbackForkScope(mgr, 7)
        a = scope.root.copy()
        a.writable_id()
        orphan = scope.root.copy()
        scope.close()
        assert mgr.parked == {} and mgr.dropped == [101]
        assert scope.root.state_id == 7
        for b in (a, orphan):
            with pytest.raises(RustStateReleasedError):
                _ = b.state_id

    def test_foreign_successor_is_refused(self):
        scope = CallbackForkScope(_FakeMgr(), 7)
        other = CallbackForkScope(_FakeMgr(), 8)
        with pytest.raises(angr.errors.SimStateError):
            scope.settle([_Holder(other.root)])

    def test_unscoped_binding_without_manager_cannot_fork(self):
        root = unscoped_binding(None, 5)
        assert root.writable_id() == 5
        copy = root.copy()
        assert copy.state_id == 5
        with pytest.raises(NotImplementedError):
            copy.writable_id()
