"""A CAS writeback fires its ``mem_write`` BP as Python's guarded store does (angr-vmi2l.175).

Python's VEX engine performs a CAS as a load, a compare and one ``StoreG`` of the
whole value (``dataHi:dataLo`` for a DCAS) guarded by the comparison, so
``mem_write`` fires unless the comparison is provably false, with it as
``mem_write_condition``, and a ``mem_write_expr`` override replaces the value.
Rust stored the swap with no condition (a concrete value), or fired nothing at
all (a symbolic one). Each case runs ``lock cmpxchg`` / ``lock cmpxchg16b`` on
AMD64 and compares both engines.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD, BUF = 0x400000, 0x80000
CMPXCHG = bytes.fromhex("f00fb137")  # lock cmpxchg dword ptr [rdi], esi
CMPXCHG16B = bytes.fromhex("f0480fc70f")  # lock cmpxchg16b xmmword ptr [rdi]
JMP_SELF = bytes.fromhex("ebfe")
OPTS = {angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY, angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
OLD, NEW = 0x11112222, 0x33334444


def _condition(cond):
    # Read structurally: a solver query from inside a BP action is not what is under test.
    if cond is None or cond.symbolic:
        return cond if cond is None else "symbolic"
    return cond.is_true()


def _run(engine, code, setup, size, when=angr.BP_AFTER, inject=None):
    project = angr.load_shellcode(code + JMP_SELF, arch="AMD64", load_address=LOAD)
    state = project.factory.blank_state(addr=LOAD, add_options=OPTS)
    state.regs.rdi = BUF
    setup(state)
    hits = []

    def action(st):
        expr = st.inspect.attrs.mem_write_expr
        hits.append(
            (
                hex(st.solver.eval(st.inspect.attrs.mem_write_address)),
                expr.size(),
                _condition(st.inspect.attrs.mem_write_condition),
            )
        )
        if inject is not None:
            st.inspect.attrs.mem_write_expr = inject

    state.inspect.b("mem_write", when=when, action=action)
    mgr = RustExplorationManager(project, [state]) if engine == "rust" else project.factory.simulation_manager(state)
    mgr.explore(find=LOAD + len(code), n=10)
    assert len(mgr.found) == 1
    found = mgr.found[0]
    stored = found.solver.eval_upto(found.memory.load(BUF, size, endness="Iend_LE"), 3)
    return hits, sorted(stored)


def _single(eax):
    def setup(state):
        state.memory.store(BUF, claripy.BVV(OLD, 32), endness="Iend_LE")
        state.regs.eax = eax
        state.regs.esi = NEW

    return setup


EAX_CASES = [OLD, 0x5, claripy.BVS("eax", 32)]


@pytest.mark.parametrize("eax", EAX_CASES, ids=["match", "mismatch", "symbolic"])
def test_cas_writeback_fires_mem_write_like_python(eax):
    expected = _run("python", CMPXCHG, _single(eax), 4)
    assert _run("rust", CMPXCHG, _single(eax), 4) == expected
    assert len(expected[0]) == (0 if eax is EAX_CASES[1] else 1)


def test_cas_writeback_takes_an_injected_value_like_python():
    inject = claripy.BVV(0x5EED5EED, 32)
    expected = _run("python", CMPXCHG, _single(OLD), 4, when=angr.BP_BEFORE, inject=inject)
    assert _run("rust", CMPXCHG, _single(OLD), 4, when=angr.BP_BEFORE, inject=inject) == expected
    assert expected[1] == [0x5EED5EED]


def _double(rax):
    def setup(state):
        state.memory.store(BUF, claripy.BVV(OLD, 64), endness="Iend_LE")
        state.memory.store(BUF + 8, claripy.BVV(OLD, 64), endness="Iend_LE")
        state.regs.rax = rax
        state.regs.rdx = OLD
        state.regs.rbx = NEW
        state.regs.rcx = NEW + 1

    return setup


@pytest.mark.parametrize("rax", [OLD, claripy.BVS("rax", 64)], ids=["match", "symbolic"])
def test_dcas_writeback_fires_one_mem_write_like_python(rax):
    expected = _run("python", CMPXCHG16B, _double(rax), 16)
    assert _run("rust", CMPXCHG16B, _double(rax), 16) == expected
    assert [(addr, bits) for addr, bits, _ in expected[0]] == [(hex(BUF), 128)]
