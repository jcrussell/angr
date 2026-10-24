# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests: RustSolverFallback must honour solver kwargs (angr-mm5sw).

``RustSolverFallback`` patches ``eval``/``eval_upto``/``min``/``max``/
``satisfiable`` onto every Rust-exported state's ``state.solver``. It accepted
``**kwargs`` but never forwarded ``extra_constraints`` (nor ``signed``) to the
Rust context, so ``satisfiable(extra_constraints=[false])`` answered True and
``eval(x, extra_constraints=[x == 5])`` returned some other value. The
exploration constraints live only in Rust, so the queries cannot be answered
by the Python solver either; the extras must be applied to the Rust context.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from angr.errors import SimUnsatError
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")


def _state_with_rust_only_constraint():
    """A materialized state whose only constraint (x > 100) lives in Rust."""
    proj = angr.load_shellcode(b"\x90\xc3", arch="amd64")
    state = proj.factory.blank_state(addr=proj.entry)
    x = claripy.BVS("x", 32)
    state.regs.eax = x
    mgr = RustExplorationManager(proj, [state])
    sid = mgr._rust_mgr.get_state_ids("active")[0]
    mgr._rust_mgr.add_constraints_to_state(sid, [x.UGT(100)])
    mgr._invalidate_state_export_cache()
    return mgr.active[0], x


class TestSolverFallbackKwargs:
    def test_satisfiable_honours_extra_constraints(self):
        s, x = _state_with_rust_only_constraint()
        assert s.solver.satisfiable(extra_constraints=[claripy.false()]) is False
        assert s.solver.satisfiable(extra_constraints=[x.ULT(10)]) is False
        assert s.solver.satisfiable(extra_constraints=[x == 200]) is True

    def test_eval_honours_extra_constraints(self):
        s, x = _state_with_rust_only_constraint()
        assert s.solver.eval(x, extra_constraints=[x == 200]) == 200
        assert s.solver.eval(x, cast_to=bytes, extra_constraints=[x == 0x41424344]) == b"ABCD"

    def test_eval_unsat_extra_constraints_raises(self):
        """Unsat under the extras must raise, not fall through to the Python
        solver (which lacks the Rust constraints and would answer anything)."""
        s, x = _state_with_rust_only_constraint()
        with pytest.raises(SimUnsatError):
            s.solver.eval(x, extra_constraints=[x.ULT(10)])

    def test_eval_upto_honours_extra_constraints(self):
        s, x = _state_with_rust_only_constraint()
        got = s.solver.eval_upto(x, 5, extra_constraints=[x.UGE(200), x.ULE(201)])
        assert sorted(got) == [200, 201]

    def test_eval_one_goes_through_extras(self):
        s, x = _state_with_rust_only_constraint()
        assert s.solver.eval_one(x, extra_constraints=[x == 7 + 100]) == 107

    def test_min_max_honour_extra_constraints(self):
        s, x = _state_with_rust_only_constraint()
        assert s.solver.min(x, extra_constraints=[x.UGT(500)]) == 501
        assert s.solver.max(x, extra_constraints=[x.ULT(300)]) == 299

    def test_min_max_honour_signed(self):
        s, x = _state_with_rust_only_constraint()
        # Under x > 100 (unsigned) the signed minimum is INT32_MIN (0x80000000).
        assert s.solver.min(x, signed=True) == -(2**31)
        assert s.solver.max(x, signed=True) == 2**31 - 1

    def test_extras_do_not_leak(self):
        s, x = _state_with_rust_only_constraint()
        s.solver.eval(x, extra_constraints=[x == 200])
        s.solver.satisfiable(extra_constraints=[claripy.false()])
        assert s.solver.satisfiable() is True
        assert s.solver.min(x) == 101


class TestCallbackSolverKwargs:
    """The per-callback twin (the callback state's ``RustSolverProxyPlugin``)
    must honour the same kwargs (``invariant-rust-solver-fallback-class``)."""

    def _callback_state(self):
        proj = angr.load_shellcode(b"\x90\xc3", arch="amd64")
        state = proj.factory.blank_state(addr=proj.entry)
        x = claripy.BVS("cbx", 32)
        state.regs.eax = x
        mgr = RustExplorationManager(proj, [state])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        mgr._rust_mgr.add_constraints_to_state(sid, [x.UGT(100)])
        cb_state = proj.factory.blank_state(addr=proj.entry)
        mgr._install_callback_solver_proxy(cb_state, sid)  # what every callback site installs
        return cb_state, x

    def test_min_max_honour_extra_constraints(self):
        s, x = self._callback_state()
        assert s.solver.min(x, extra_constraints=[x.UGT(500)]) == 501
        assert s.solver.max(x, extra_constraints=[x.ULT(300)]) == 299
        assert s.solver.min(x) == 101

    def test_min_max_honour_signed(self):
        s, x = self._callback_state()
        assert s.solver.min(x, signed=True) == -(2**31)
        assert s.solver.max(x, signed=True) == 2**31 - 1

    def test_eval_unsat_extra_constraints_raises(self):
        """Infeasible extras must raise SimUnsatError like SimSolver, not fall
        back to the Python solver without them (angr-vmi2l.293)."""
        s, x = self._callback_state()
        with pytest.raises(SimUnsatError):
            s.solver.eval(x, extra_constraints=[x.ULT(10)])
        with pytest.raises(SimUnsatError):
            s.solver.eval_upto(x, 3, extra_constraints=[x.ULT(10)])
        assert s.solver.satisfiable(extra_constraints=[x.ULT(10)]) is False
        # The extras were scoped: the state itself is still satisfiable.
        assert s.solver.satisfiable() is True

    def test_eval_upto_honours_extra_constraints(self):
        s, x = self._callback_state()
        got = s.solver.eval_upto(x, 5, extra_constraints=[x.UGE(200), x.ULE(201)])
        assert sorted(got) == [200, 201]
        assert isinstance(got, list)

    def test_eval_upto_cast_to_bytes_encodes_the_value(self):
        """``cast_to=bytes`` is the big-endian encoding of the solution, as
        ``SimSolver._cast_to`` produces, not ``bytes(value)`` (a zero buffer
        of that length) (angr-vmi2l.225)."""
        s, x = self._callback_state()
        # A small value: the bug allocates ``bytes(value)``, so a large one
        # would exhaust memory instead of failing the assertion.
        got = s.solver.eval_upto(x, 2, cast_to=bytes, extra_constraints=[x == 0x4142])
        assert [len(b) for b in got] == [4]
        assert got == [(0x4142).to_bytes(4, "big")]
        assert s.solver.eval_upto(x, 2, cast_to=int, extra_constraints=[x == 0x4142]) == [0x4142]


class TestCallbackHookEvalUptoCast:
    """End to end: a user hook reached by the Rust engine sees the same
    ``eval_upto(..., cast_to=...)`` answer as under the Python engine
    (angr-vmi2l.225)."""

    @staticmethod
    def _run(engine):
        # nop; nop; nop; nop; jmp $  -- the hook covers the middle two nops.
        proj = angr.load_shellcode(b"\x90\x90\x90\x90\xeb\xfe", arch="amd64", load_address=0x400000)
        load = 0x400000
        seen = {}

        class P(angr.SimProcedure):
            def run(self):
                rbx = self.state.regs.rbx
                seen["bytes"] = self.state.solver.eval_upto(rbx, 2, cast_to=bytes)
                seen["int"] = self.state.solver.eval_upto(rbx, 2, cast_to=int)
                self.jump(load + 4)

        proj.hook(load + 2, P(), length=2)
        state = proj.factory.blank_state(addr=load)
        x = claripy.BVS("hookx", 64)
        state.regs.rbx = x
        state.solver.add(x == 5)
        if engine == "rust":
            RustExplorationManager(proj, [state]).explore(find=load + 4)
        else:
            proj.factory.simulation_manager(state).explore(find=load + 4)
        return seen

    def test_matches_python_engine(self):
        rust = self._run("rust")
        assert rust, "the hook never ran under the Rust engine"
        assert rust == self._run("python")
        assert rust["bytes"] == [(5).to_bytes(8, "big")]
