"""``RustMemoryProxy`` honours ``SimMemory``'s keyword arguments (angr-vmi2l.260).

A SimProcedure's ``state.memory`` is a ``RustMemoryProxy`` during a Rust
callback, and so is an exported state's. Each test runs the same operation
against the Python engine and compares: ``size=`` truncates or (symbolic)
guards each byte, ``condition=`` guards the write, a guarded load yields its
``fallback``, and a keyword the proxy cannot honour raises.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager

LOAD, HOOK, DST, SRC = 0x400000, 0x500000, 0x600000, 0x602000
# mov rax, HOOK ; call rax ; jmp $
_CALL_HOOK = bytes([0x48, 0xB8]) + HOOK.to_bytes(8, "little") + bytes([0xFF, 0xD0, 0xEB, 0xFE])
_AFTER_CALL = LOAD + 12


def _call_state(proc, **regs):
    """A state about to call ``proc`` (hooked at ``HOOK``) with ``regs`` set."""
    proj = angr.load_shellcode(_CALL_HOOK, arch="amd64", load_address=LOAD)
    proj.hook(HOOK, proc, replace=True)
    state = proj.factory.blank_state(
        addr=LOAD,
        add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS, angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY},
    )
    state.regs.rsp = 0x7FFFFFFF0800
    state.memory.store(DST, b"Z" * 64)
    state.memory.store(SRC, b"a" * 64)
    for name, value in regs.items():
        setattr(state.regs, name, value)
    return proj, state


def _found_both(proj, state):
    """The state after the call under the Python and the Rust engine."""
    py = proj.factory.simgr(state.copy())
    py.explore(find=_AFTER_CALL)
    mgr = RustExplorationManager(proj, [state.copy()])
    mgr.explore(find=_AFTER_CALL)
    assert len(py.found) == len(mgr.found) == 1
    return py.found[0], mgr.found[0]


def _libc(name):
    proto = angr.SIM_LIBRARIES["libc.so.6"][0].get_prototype(name)
    return angr.SIM_PROCEDURES["libc"][name](prototype=proto)


def _byte_facts(state, addr, n, *extra_vars):
    """Which values byte ``addr`` can take, alone and against ``n``."""
    b = state.memory.load(addr, 1)
    return {
        "stays": state.solver.satisfiable(extra_constraints=[b == ord("Z")]),
        "written": state.solver.satisfiable(extra_constraints=[b == ord("a")]),
        "written_past_n": state.solver.satisfiable(extra_constraints=[b == ord("a"), claripy.ULE(n, addr - DST)]),
        "kept_below_n": state.solver.satisfiable(extra_constraints=[b == ord("Z"), claripy.UGT(n, addr - DST)]),
    }


class TestSymbolicSizeThroughProcedures:
    """A Python memset / memcpy with a symbolic length writes byte i only where i < n."""

    def test_memset_unbounded_length(self):
        n = claripy.BVS("n", 64)
        proj, state = _call_state(_libc("memset"), rdi=DST, rsi=ord("a"), rdx=n)
        py, rust = _found_both(proj, state)
        for offset in (0, 10, 63):
            assert _byte_facts(rust, DST + offset, n) == _byte_facts(py, DST + offset, n), offset
        # Byte 10 is written exactly where 10 < n (and n fits the data memset
        # built; past that, angr's sized store writes nothing).
        facts = _byte_facts(rust, DST + 10, n)
        assert (facts["stays"], facts["written"], facts["written_past_n"]) == (True, True, False)

    def test_memcpy_bounded_length(self):
        n = claripy.BVS("n", 64)
        proj, state = _call_state(_libc("memcpy"), rdi=DST, rsi=SRC, rdx=n)
        state.add_constraints(claripy.ULE(n, 5000))
        py, rust = _found_both(proj, state)
        for offset in (0, 10, 40):
            assert _byte_facts(rust, DST + offset, n) == _byte_facts(py, DST + offset, n), offset
        assert _byte_facts(rust, DST + 10, n)["stays"]


class TestConditionThroughProcedures:
    def test_sigprocmask_with_null_oldset_writes_nothing(self):
        proc = angr.SIM_PROCEDURES["linux_kernel"]["sigprocmask"]()
        proj, state = _call_state(proc, rdi=0, rsi=0, rdx=0, rcx=8)
        state.memory.store(0, b"Q" * 8)
        py, rust = _found_both(proj, state)
        for found in (py, rust):
            assert found.solver.eval(found.memory.load(0, 8), cast_to=bytes) == b"Q" * 8

    def test_sigprocmask_with_oldset_writes_the_mask(self):
        proc = angr.SIM_PROCEDURES["linux_kernel"]["sigprocmask"]()
        proj, state = _call_state(proc, rdi=0, rsi=0, rdx=DST, rcx=8)
        py, rust = _found_both(proj, state)
        masks = [f.solver.eval(f.memory.load(DST, 8), cast_to=bytes) for f in (py, rust)]
        assert masks[0] == masks[1] != b"Z" * 8


def _export_and_python_state():
    """An exported state's memory proxy, and the Python state it started from."""
    proj = angr.load_shellcode(b"\xeb\xfe", arch="amd64", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD)
    state.memory.store(DST, b"\xee" * 16)
    mgr = RustExplorationManager(proj, [state.copy()])
    export = mgr.active[0].copy()
    return export, state


def _bytes(state, addr=DST, n=8):
    return state.solver.eval(state.memory.load(addr, n), cast_to=bytes)


class TestStoreKeywords:
    """Direct ``state.memory.store`` calls on an export against a Python state."""

    @pytest.mark.parametrize("endness", ["Iend_BE", "Iend_LE"])
    @pytest.mark.parametrize(
        "data,size",
        [
            (claripy.BVV(0x1122334455667788, 64), 3),
            (0x41, 2),
            (b"ABCD", 2),
            (claripy.BVV(0x1122, 16), 0),
            (claripy.BVV(0x11223344, 32), claripy.BVV(1, 64)),
        ],
    )
    def test_concrete_size_truncates(self, data, size, endness):
        export, state = _export_and_python_state()
        for s in (export, state):
            s.memory.store(DST, data, size=size, endness=endness)
        assert _bytes(export) == _bytes(state)

    def test_size_larger_than_the_data_raises(self):
        export, state = _export_and_python_state()
        for s in (export, state):
            with pytest.raises(angr.errors.SimMemoryError):
                s.memory.store(DST, claripy.BVV(1, 16), size=4)

    @pytest.mark.parametrize("cond", ["true", "false"])
    def test_concrete_condition(self, cond):
        export, state = _export_and_python_state()
        c = claripy.true() if cond == "true" else claripy.false()
        for s in (export, state):
            s.memory.store(DST, claripy.BVV(0x4142, 16), condition=c)
            # A false guard writes nothing, also to a page nobody mapped.
            s.memory.store(0x700000, claripy.BVV(0x43, 8), condition=claripy.false())
        assert _bytes(export) == _bytes(state)

    def test_symbolic_condition(self):
        export, state = _export_and_python_state()
        x = claripy.BVS("x", 8)
        for s in (export, state):
            s.memory.store(DST, claripy.BVV(0x4142, 16), condition=x == 1, endness="Iend_LE")
        for s in (export, state):
            word = s.memory.load(DST, 2)
            assert s.solver.eval(word, extra_constraints=[x == 1]) == 0x4241
            assert s.solver.eval(word, extra_constraints=[x == 2]) == 0xEEEE

    def test_symbolic_size(self):
        export, state = _export_and_python_state()
        n = claripy.BVS("n", 64)
        for s in (export, state):
            s.add_constraints(claripy.ULE(n, 4))
            s.memory.store(DST, b"abcd", size=n)
        for s in (export, state):
            for k in range(5):
                assert s.solver.eval(s.memory.load(DST, 4), cast_to=bytes, extra_constraints=[n == k]) == (
                    b"abcd"[:k] + b"\xee" * (4 - k)
                )

    def test_unhonoured_keywords_raise(self):
        export, _state = _export_and_python_state()
        with pytest.raises(NotImplementedError, match="bogus"):
            export.memory.store(DST, b"A", bogus=True)
        with pytest.raises(NotImplementedError, match="action"):
            export.memory.store(DST, b"A", action=object())
        with pytest.raises(NotImplementedError, match="bogus"):
            export.memory.load(DST, 1, bogus=True)
        # Accepted, and meaningful: no actions are recorded, no breakpoint fires.
        export.memory.store(DST, b"A", inspect=False, disable_actions=True, action=None)
        assert export.memory.load(DST, 1, inspect=False, disable_actions=True).concrete_value == ord("A")


class TestLoadKeywords:
    def test_condition_with_fallback(self):
        export, state = _export_and_python_state()
        x = claripy.BVS("x", 8)
        for s in (export, state):
            v = s.memory.load(DST, 2, condition=x == 1, fallback=0x1234)
            assert s.solver.eval(v, extra_constraints=[x == 1]) == 0xEEEE
            assert s.solver.eval(v, extra_constraints=[x == 0]) == 0x1234

    def test_symbolic_size_loads_the_maximum(self):
        export, state = _export_and_python_state()
        n = claripy.BVS("n", 64)
        for s in (export, state):
            s.add_constraints(claripy.ULE(n, 3))
        assert [s.memory.load(DST, n).size() for s in (export, state)] == [24, 24]

    def test_breakpoints_fire(self):
        export, state = _export_and_python_state()
        seen = {id(s): [] for s in (export, state)}
        for s in (export, state):
            log = seen[id(s)]
            s.inspect.b("mem_write", when=angr.BP_AFTER, action=lambda st, log=log: log.append("w"))
            s.inspect.b("mem_read", when=angr.BP_AFTER, action=lambda st, log=log: log.append("r"))
            s.memory.store(DST, b"AB")
            s.memory.load(DST, 2)
            s.memory.load(DST, 2, inspect=False)
        assert seen[id(export)] == seen[id(state)] == ["w", "r"]
