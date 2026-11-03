"""A trap exit behind a symbolic guard never continues as an ordinary path.

MIPS ``div a0,a1; teq a1,zero,CODE`` with a symbolic divisor: the ``teq`` is a
conditional exit whose jumpkind is ``Ijk_SigFPE_IntDiv`` (code 7) or
``Ijk_SigTRAP`` (code 0). Python drops the ``SigFPE_IntDiv`` successor and
errors the ``SigTRAP`` one. Rust deferred the trap side as an ``Ijk_Boring``
fork, which ran on to the find address with ``a1 == 0`` satisfiable
(angr-vmi2l.160).
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
FIND = LOAD + 0x10
OPTS = {angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY, angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}


def _project(code):
    words = (0x0085001A, (5 << 21) | (code << 6) | 0x34, 0x00001012, 0, 0x1000FFFF, 0)
    blob = b"".join(struct.pack(">I", w) for w in words)
    return angr.load_shellcode(blob, arch="mips32", load_address=LOAD)


def _explore(engine, code, divisor):
    proj = _project(code)
    state = proj.factory.blank_state(addr=LOAD, add_options=OPTS)
    state.regs.a0 = 100
    state.regs.a1 = divisor
    if engine == "rust":
        from angr.exploration.rust_manager import RustExplorationManager

        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=FIND, num_find=10, n=20)
        errored = [e.state for e in mgr.errored]
        deadended = list(mgr.deadended)
    else:
        mgr = proj.factory.simulation_manager(state)
        mgr.explore(find=FIND, num_find=10, n=20)
        errored = [e.state for e in mgr.errored]
        deadended = list(mgr.deadended)
    return list(mgr.found), errored, deadended


def _summary(found, errored, deadended):
    def zero_sat(st):
        return st.solver.satisfiable(extra_constraints=[st.regs.a1 == 0])

    return (
        sorted((st.addr, zero_sat(st)) for st in found),
        sorted((st.addr, zero_sat(st), st.history.jumpkind) for st in errored),
        len(deadended),
    )


@pytest.mark.parametrize("code", [7, 0])
@pytest.mark.parametrize("engine", ["py", "rust"])
def test_symbolic_divisor_trap_side_is_not_explored(engine, code):
    found, errored, deadended = _explore(engine, code, claripy.BVS("d", 32))
    expected_errored = [] if code == 7 else [(LOAD + 8, True, "Ijk_SigTRAP")]
    assert _summary(found, errored, deadended) == ([(FIND, False)], expected_errored, 0)


@pytest.mark.parametrize("code", [7, 0])
def test_zero_divisor_errors_like_python(code):
    # Python's concrete division by zero errors in the div op, at the block
    # start; Rust's total division reaches the trap and errors there.
    for engine in ("py", "rust"):
        found, errored, deadended = _explore(engine, code, 0)
        assert (len(found), len(errored), len(deadended)) == (0, 1, 0), engine
