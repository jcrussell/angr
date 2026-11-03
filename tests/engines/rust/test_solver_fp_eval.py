"""An FP expression evaluates to a float on a Rust state, as SimSolver answers.

The Rust solver context evaluates an FP AST to its IEEE bit pattern; the
proxy decodes it (angr-mx00p.132.9, upstream test_sqrt_symbolic).
"""

from __future__ import annotations

import math

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")


@pytest.fixture
def sqrtss_state():
    """One step of the lone ``sqrtss xmm1, xmm1`` over a symbolic single-precision input."""
    proj = angr.load_shellcode(b"\xf3\x0f\x51\xc9", "amd64", load_address=0, engine="rust")
    state = proj.factory.blank_state(addr=0)
    xmm1 = claripy.FPS("v", claripy.FSORT_FLOAT)
    state.regs.xmm1 = xmm1.to_bv()
    mgr = proj.factory.simgr(state)
    assert type(mgr).__name__ == "RustExplorationManager"
    mgr.step()
    assert len(mgr.active) == 1
    final = mgr.active[0]
    return final, xmm1, final.regs.xmm1[31:0].raw_to_fp()


def test_eval_of_an_fp_input_is_a_float(sqrtss_state):
    final, xmm1, result = sqrtss_state
    a = final.solver.eval(xmm1, extra_constraints=(result == 2.0,))
    assert isinstance(a, float)
    assert int(a) == 4
    assert math.sqrt(a) == pytest.approx(2.0)


def test_eval_upto_of_an_fp_input_is_floats(sqrtss_state):
    final, xmm1, result = sqrtss_state
    values = final.solver.eval_upto(xmm1, 2, extra_constraints=(result == 4.0,))
    assert values
    assert all(isinstance(v, float) for v in values)
    assert all(math.sqrt(v) == pytest.approx(4.0) for v in values)


def test_eval_of_a_double_cast_to_int(sqrtss_state):
    final, _, _ = sqrtss_state
    d = claripy.FPS("d", claripy.FSORT_DOUBLE)
    assert final.solver.eval(d, cast_to=int, extra_constraints=(d == 12.5,)) == 12
