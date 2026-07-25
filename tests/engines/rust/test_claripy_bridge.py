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
