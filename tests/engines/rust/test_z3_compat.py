"""claripy 9.2.209 concrete-FP model extraction on the z3-solver 5.x bindings
(angr/misc/z3_compat.py)."""

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


if __name__ == "__main__":
    unittest.main()
