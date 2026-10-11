"""claripy 9.2.209 on the z3-solver 5.x bindings (angr/misc/z3_compat.py):
concrete-FP model extraction and solver cloning without Z3_solver_translate."""

from __future__ import annotations

import ctypes
import unittest

import z3
from z3 import z3core

import angr  # noqa: F401  -- installs the shim at import
from angr import claripy
from angr.misc import z3_compat


class TestZ3Compat(unittest.TestCase):
    def test_install_is_idempotent(self):
        first = z3_compat.install()
        second = z3_compat.install()
        assert first == second
        if first:
            assert getattr(z3.Z3_fpa_get_numeral_sign, "_angr_z3_compat_wrapped", False)

    def test_fp_model_value_round_trips(self):
        """The call that raised ctypes.ArgumentError before the shim: a
        concrete FP value read back out of a Z3 model by claripy."""
        f = claripy.FPS("f", claripy.FSORT_DOUBLE)
        for expected in (1.5, -2.25, 1024.0, -0.5):
            s = claripy.Solver()
            s.add(f == claripy.FPV(expected, claripy.FSORT_DOUBLE))
            (got,) = s.eval(f, 1)
            assert got == expected, (expected, got)

    def test_c_int_out_param_is_filled(self):
        """Direct check of the wrapped binding with the argument shape claripy uses."""
        if not getattr(z3core.Z3_fpa_get_numeral_sign, "_angr_z3_compat_wrapped", False):
            self.skipTest("shim not needed for this z3-solver")
        ctx = z3.main_ctx()
        neg = z3.FPVal(-3.0, z3.Float64(), ctx)
        pos = z3.FPVal(3.0, z3.Float64(), ctx)
        for val, want in ((neg, 1), (pos, 0)):
            out = ctypes.c_int()
            z3.Z3_fpa_get_numeral_sign(ctx.ref(), val.as_ast(), ctypes.byref(out))
            assert out.value == want


class TestCloneSolverShim(unittest.TestCase):
    """``BackendZ3.clone_solver`` rebuilt instead of translated (see the module
    docstring for why translate is unusable on Z3 5.x)."""

    def test_shim_installed(self):
        from claripy.backends import backend_z3

        assert getattr(backend_z3.BackendZ3.clone_solver, "_angr_z3_compat_wrapped", False)

    def test_branch_keeps_constraints_and_independence(self):
        x = claripy.BVS("x", 32)
        parent = claripy.Solver()
        parent.add(x > 100)
        parent.add(x < 200)
        assert parent.satisfiable()  # finalizes the Z3 solver so branch() clones it
        child = parent.branch()
        child.add(x == 150)
        (got,) = child.eval(x, 1)
        assert got == 150
        child.add(x == 151)
        assert not child.satisfiable()
        assert parent.satisfiable()  # the clone did not leak into the parent
        assert set(parent.eval(x, 3)) <= set(range(101, 200))

    def test_clone_keeps_tracking_for_unsat_core(self):
        x = claripy.BVS("x", 32)
        parent = claripy.Solver(track=True)
        parent.add(x > 100)
        assert parent.satisfiable()
        child = parent.branch()
        child.add(x < 50)
        assert not child.satisfiable()
        core = child.unsat_core()
        assert len(core) == 2, core

    def test_clone_keeps_timeout(self):
        from claripy.backends import backend_z3

        parent = claripy.Solver(timeout=1234)
        x = claripy.BVS("x", 32)
        parent.add(x > 1)
        assert parent.satisfiable()
        child = parent.branch()
        child.add(x > 2)
        assert child.satisfiable()
        cloned = child._tls.solver
        assert getattr(cloned, backend_z3.BackendZ3.__dict__["clone_solver"].__globals__["_PARAMS_ATTR"]) == (
            1234,
            None,
        )

    def test_clone_does_not_balloon_z3_memory(self):
        """The failure mode: 20 clones of a default solver cost ~8 GB through
        translate on Z3 5.1; rebuilt clones are a few MB."""
        before = z3.Z3_get_estimated_alloc_size()
        x = claripy.BVS("x", 32)
        s = claripy.Solver()
        s.add(x > 1)
        assert s.satisfiable()
        for i in range(20):
            s = s.branch()
            s.add(x > i)
            assert s.satisfiable()
        grown_mb = (z3.Z3_get_estimated_alloc_size() - before) / (1024 * 1024)
        assert grown_mb < 64, grown_mb


if __name__ == "__main__":
    unittest.main()
