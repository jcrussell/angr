"""Coverage for claripy_bridge import paths that had zero tests (angr-n0irt.20).

Two import.rs arms are exercised here through the public ``RustSolverContext``
constraint/eval surface (the same entry point ``add_constraint_ast`` uses):

* ``reverse_bytes`` symbolic branch — ``claripy.Reverse`` over a symbolic leaf.
  Note claripy *folds* ``Reverse`` of a concrete ``BVV`` to a plain ``BVV`` before
  it reaches the bridge (op becomes ``"BVV"``, not ``"Reverse"``), so the
  concrete branch of ``reverse_bytes`` is unreachable from the import path and
  stays defensive — the live byte-reversal path is the symbolic extract-and-
  concat branch, pinned below via a concrete-valued constraint on the leaf.
* the ``"BoolS"`` arm (angr-q6r1) — a symbolic 1-bit Bool leaf. Exercised as the
  condition of an ``If`` so it survives the FFI boundary and drives a real z3
  ite, mirroring the ``posix.fork`` shape the arm was built for.
"""

from __future__ import annotations

import pytest

# Rust availability guard lives in tests/engines/conftest.py (angr-7gdp).
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


class TestClaripyBridgeImport:
    """import.rs coverage: reverse_bytes (symbolic) and the BoolS arm."""

    @classmethod
    def setup_class(cls):
        from angr.exploration.rust_manager import _setup_shared_z3_context

        _setup_shared_z3_context()

    def test_reverse_symbolic_leaf_byte_reverses(self):
        """Reverse(x) over a symbolic leaf hits reverse_bytes' symbolic branch
        (extract-and-concat) and, pinned to a concrete value, evaluates to the
        byte-reversed word."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("rx", 32)
        ctx.add_constraint_ast(x == 0x01020304)
        # op stays "Reverse" for a symbolic operand -> import.rs "Reverse" arm.
        assert claripy.Reverse(x).op == "Reverse"
        assert ctx.eval(claripy.Reverse(x)) == 0x04030201

    def test_reverse_symbolic_64bit(self):
        """Same symbolic branch at 64-bit width (num_bytes loop > 4)."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("rx64", 64)
        ctx.add_constraint_ast(x == 0x1122334455667788)
        assert ctx.eval(claripy.Reverse(x)) == 0x8877665544332211

    def test_reverse_of_concrete_is_folded_by_claripy(self):
        """Documents why reverse_bytes' concrete branch is unreachable from the
        import path: claripy folds Reverse(BVV) to a BVV before dispatch."""
        import claripy

        folded = claripy.Reverse(claripy.BVV(0x01020304, 32))
        assert folded.op == "BVV"
        assert folded.args[0] == 0x04030201

    def test_bools_leaf_drives_ite(self):
        """If(BoolS(...), a, b) exercises import.rs' "BoolS" arm as the ite
        condition; both branch values must be reachable."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        b = claripy.BoolS("mybool")
        ite = claripy.If(b, claripy.BVV(0xAA, 8), claripy.BVV(0xBB, 8))
        results = set(ctx.eval_upto(ite, 4))
        assert results == {0xAA, 0xBB}

    def test_bools_identity_preserved_across_conversions(self):
        """The same BoolS name used in two constraints stays one symbol: an
        equality between them is satisfiable both ways (hash-lookup / name+width
        identity path in the BoolS arm), never a contradiction."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        b = claripy.BoolS("shared")
        then_a = claripy.If(b, claripy.BVV(1, 8), claripy.BVV(0, 8))
        then_b = claripy.If(b, claripy.BVV(1, 8), claripy.BVV(0, 8))
        # Same underlying leaf -> the two ites must agree bit-for-bit.
        assert set(ctx.eval_upto(then_a - then_b, 4)) == {0}

    def test_explicit_name_bool_and_bv_are_independent_variables(self):
        """``BVS("x", 1, explicit_name=True)`` and ``BoolS("x",
        explicit_name=True)`` are distinct variables in claripy (different
        sorts), so pinning the BV to 1 must not force the Bool true.

        Rust models a Bool as a width-1 ``Symbolic`` and builds its Z3 constant
        from the symbol *name*, so before angr-9ke6b.223 both leaves became the
        same ``BV::new_const("x", 1)`` and this pair was UNSAT.  Only reachable
        with ``explicit_name`` on both: claripy otherwise renames to
        ``x_<counter>_1`` / ``x_<counter>_-1``, which never collide.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        bv = claripy.BVS("x", 1, explicit_name=True)
        flag = claripy.BoolS("x", explicit_name=True)
        assert bv.args[0] == "x" and flag.args[0] == "x"

        ctx.add_constraint_ast(bv == 1)
        ctx.add_constraint_ast(claripy.Not(flag))

        # Independent variables -> both constraints hold simultaneously.
        assert ctx.satisfiable()
        assert ctx.eval(bv) == 1
        assert set(ctx.eval_upto(claripy.If(flag, claripy.BVV(1, 8), claripy.BVV(0, 8)), 4)) == {0}

    def test_explicit_name_bool_keeps_its_own_identity_across_imports(self):
        """The mangled Rust name must be stable: re-importing the same
        ``BoolS`` leaf has to resolve to the same Z3 variable, or a constraint
        added in one import silently stops binding in the next."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        flag = claripy.BoolS("y", explicit_name=True)

        ctx.add_constraint_ast(flag)
        # Second, separate import of the same leaf: it must see the constraint.
        assert set(ctx.eval_upto(claripy.If(flag, claripy.BVV(0xAA, 8), claripy.BVV(0xBB, 8)), 4)) == {0xAA}


class TestClaripyBridgeWidthBounds:
    """import.rs must bound caller-chosen widths (angr-0jh0j.8).

    ``width_guards.rs``'s module doc names *two* trust boundaries — the handle
    API in ``solver/handle_api.rs`` and this one — but only the ``Extract`` arm
    here actually called into it. ``BVV``/``BVS`` took their width straight from
    the claripy AST, and ``ZeroExt``/``SignExt`` derived one with unchecked
    ``u32`` addition, so a Python-chosen width reached Z3 as a multi-gigabit
    sort allocation (or, once the sum wrapped, as a silently *narrowing*
    extension).

    The probes below go through ``RustSolverContext.min``, which propagates the
    bridge error rather than swallowing it: ``eval`` falls back to claripy's own
    Z3 AST when conversion fails, which would hand the huge width to Z3 anyway
    — by claripy's doing, not the engine's.
    """

    # Mirrors symbolic/width_guards.rs::MAX_BV_WIDTH.
    MAX_BV_WIDTH = 1 << 20

    @classmethod
    def setup_class(cls):
        from angr.exploration.rust_manager import _setup_shared_z3_context

        _setup_shared_z3_context()

    def _ctx(self):
        from angr.rustylib.vex_engine import RustSolverContext

        return RustSolverContext()

    def test_bvv_width_above_cap_is_refused(self):
        import claripy

        over = self.MAX_BV_WIDTH + 1
        with pytest.raises(ValueError, match=str(over)):
            self._ctx().min(claripy.BVV(0, over))

    def test_bvs_width_above_cap_is_refused(self):
        import claripy

        over = self.MAX_BV_WIDTH + 1
        with pytest.raises(ValueError, match=str(over)):
            self._ctx().min(claripy.BVS("wide_leaf", over))

    @pytest.mark.parametrize("op_name", ["ZeroExt", "SignExt"])
    def test_extension_past_cap_is_refused(self, op_name):
        import claripy

        op = getattr(claripy, op_name)
        with pytest.raises(ValueError, match=op_name):
            self._ctx().min(op(self.MAX_BV_WIDTH, claripy.BVS(f"ext_{op_name}", 8)))

    @pytest.mark.parametrize("op_name", ["ZeroExt", "SignExt"])
    def test_extension_that_wraps_u32_is_refused(self, op_name):
        """8 + (2**32 - 4) wraps to 4 — a *narrowing* extension the release
        profile (no overflow checks) would otherwise accept silently."""
        import claripy

        op = getattr(claripy, op_name)
        with pytest.raises(ValueError, match="overflows u32"):
            self._ctx().min(op(2**32 - 4, claripy.BVS(f"wrap_{op_name}", 8)))

    def test_ordinary_widths_still_convert(self):
        """The guards are a ceiling, not a new precondition: everyday widths
        keep converting. (That the cap itself is *inclusive* is pinned by
        ``import_tests.rs::test_extended_width_refuses_overflow_and_absurd_widths``
        — asserting it from here would make Z3 build a 1-Mibit sort.)"""
        import claripy

        ctx = self._ctx()
        assert ctx.min(claripy.BVV(0x1234, 32)) == 0x1234
        assert ctx.min(claripy.ZeroExt(24, claripy.BVV(0xAB, 8))) == 0xAB
        assert ctx.min(claripy.SignExt(24, claripy.BVV(0x7F, 8))) == 0x7F
