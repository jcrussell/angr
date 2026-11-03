"""A Bool or FP expression evaluates to a bool or float on a Rust state.

The Rust solver context evaluates a Bool AST as ``ite(b, 1, 0)`` and an FP AST
as its IEEE bits. SimSolver answers ``True``/``False`` and a float when
uncast, ``0``/``1`` and the bits under ``cast_to=int``, and one byte for a
Bool under ``cast_to=bytes`` (angr-vmi2l.218). Each Rust solver surface below
is checked against the Python engine on the same input.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
# nop; nop; nop; nop; jmp $
CODE = bytes([0x90, 0x90, 0x90, 0x90, 0xEB, 0xFE])


def _probe(state):
    """Every eval form on a Bool and an FP expression over rbx."""
    fp = state.regs.rbx.raw_to_fp()
    exprs = {"fp": fp, "true": state.regs.rbx[7:0] == 0, "false": state.regs.rbx[7:0] == 1}
    out = {}
    for name, e in exprs.items():
        out[name, None] = state.solver.eval(e)
        out[name, int] = state.solver.eval(e, cast_to=int)
        out[name, bytes] = state.solver.eval(e, cast_to=bytes)
        out[name, "upto"] = list(state.solver.eval_upto(e, 3))
    return out


def _seed(proj):
    state = proj.factory.blank_state(addr=LOAD)
    y = claripy.BVS("y", 64)
    state.regs.rbx = y
    state.solver.add(y.raw_to_fp() == claripy.FPV(2.5, claripy.FSORT_DOUBLE))
    return state


def _run(engine):
    """Probe a find predicate, a SimProcedure hook and the found state."""
    from angr.exploration.rust_manager import RustExplorationManager

    proj = angr.load_shellcode(CODE, arch="AMD64", load_address=LOAD)
    seen = {}

    class Probe(angr.SimProcedure):
        def run(self):
            seen["hook"] = _probe(self.state)
            self.jump(LOAD + 4)

    proj.hook(LOAD + 2, Probe(), length=2)

    def pred(s):
        if s.addr == LOAD + 4:
            seen["pred"] = _probe(s)
            return True
        return False

    state = _seed(proj)
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=pred)
    assert len(mgr.found) == 1
    seen["found"] = _probe(mgr.found[0])
    return seen


@pytest.fixture(scope="module")
def python_answers():
    return _run("python")


@pytest.fixture(scope="module")
def rust_answers():
    return _run("rust")


@pytest.mark.parametrize("site", ["hook", "pred", "found"])
def test_bool_and_fp_eval_match_simsolver(site, python_answers, rust_answers):
    assert python_answers[site][("true", None)] is True
    assert python_answers[site][("false", None)] is False
    assert python_answers[site][("fp", None)] == 2.5
    rust, python = rust_answers[site], python_answers[site]
    assert rust == python
    for key, value in python.items():
        assert type(rust[key]) is type(value), key


def test_solver_fallback_casts_bool_and_fp():
    """``RustSolverFallback``, which an export attaches over SimSolver, casts too."""
    from angr.exploration.rust_manager import RustExplorationManager
    from angr.exploration.rust_state_export import RustSolverFallback

    proj = angr.load_shellcode(CODE, arch="AMD64", load_address=LOAD)
    state = _seed(proj)
    mgr = RustExplorationManager(proj, [state.copy()])
    (state_id,) = mgr._rust_mgr.get_state_ids("active")
    fallback = RustSolverFallback(state, state_id, mgr._rust_mgr)
    rbx = state.regs.rbx
    true, fp = rbx[7:0] == 0, rbx.raw_to_fp()
    assert fallback.eval(true) is True
    assert fallback.eval(rbx[7:0] == 1) is False
    assert fallback.eval(true, cast_to=int) == 1
    assert fallback.eval(true, cast_to=bytes) == b"\x01"
    assert fallback.eval_upto(true, 3) == [True]
    assert fallback.eval(fp) == 2.5
    assert fallback.eval_upto(fp, 3) == [2.5]
    assert fallback.eval(fp, cast_to=int) == state.solver.eval(fp, cast_to=int)
