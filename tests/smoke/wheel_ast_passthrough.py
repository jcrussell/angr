#!/usr/bin/env python
"""Clean-venv AST-passthrough smoke test for the prebuilt wheel (bd angr-9eit).

The whole point of *excluding* (not vendoring) ``libz3.so`` from the wheel
(bd angr-ivwn) is that the Rust extension and ``claripy`` must load the **same**
``libz3.so`` at runtime. AST passthrough hands raw Z3 ``Ast`` pointers across
the FFI boundary, and a pointer is only valid inside the ``Z3_context`` that
minted it. If the wheel had bundled a private ``libz3.so`` (a second
``Z3_context``), the constraint minted by ``claripy`` below would be
meaningless to the Rust solver — it would crash, read garbage, or silently
return a wrong answer.

So this script mints a Z3 AST in ``claripy`` and reads it back through the Rust
``RustSolverContext``. A correct round-trip proves both halves resolved
``site-packages/z3/lib/libz3.so`` via the wheel's ``$ORIGIN/../z3/lib`` RUNPATH.

Runnable two ways:

* Standalone for ``cibuildwheel``'s ``CIBW_TEST_COMMAND``::

      python {project}/tests/smoke/wheel_ast_passthrough.py

  Exits 0 on success, nonzero (with a traceback) on failure.

* Collected by pytest for local verification (``test_*`` function below).

See docs/advanced-topics/rust_wheel_distribution.rst (Open blockers #3).
"""

from __future__ import annotations


def check_ast_passthrough() -> None:
    """Mint a claripy AST, solve it in Rust, assert the shared-context answer.

    Raises AssertionError (or a Z3/FFI exception) if the two halves are not
    sharing one ``libz3.so`` — which is exactly the failure the wheel's
    ``--exclude libz3.so`` repair step is designed to prevent.
    """
    import claripy

    # Initialise the shared Z3 context before touching RustSolverContext —
    # mirrors tests/engines/rust/test_solver_ops.py::setup_class.
    from angr.exploration.rust_manager import _setup_shared_z3_context

    _setup_shared_z3_context()

    from angr.rustylib.vex_engine import RustSolverContext

    # Mint an AST in claripy's Z3 backend...
    x = claripy.BVS("x", 32)

    ctx = RustSolverContext()
    # ...and hand the Z3 Ast pointers across FFI to the Rust solver.
    ctx.add_constraint_ast(x >= 10)
    ctx.add_constraint_ast(x <= 20)

    assert ctx.satisfiable(), "10 <= x <= 20 should be satisfiable"

    # Read the bounds back out of the Rust solver. If claripy and Rust had
    # separate Z3 contexts, these would not reflect the claripy-minted AST.
    lo = ctx.min(x, signed=False)
    hi = ctx.max(x, signed=False)
    assert lo == 10, f"expected min 10, got {lo}"
    assert hi == 20, f"expected max 20, got {hi}"

    # Pin to a single model and evaluate the claripy AST through Rust.
    ctx.add_constraint_ast(x == 17)
    val = ctx.eval(x)
    assert val == 17, f"expected eval 17, got {val}"

    # A contradictory claripy constraint must turn the Rust solver UNSAT —
    # proving the constraint actually reached the same solver state.
    ctx.add_constraint_ast(x == 18)
    assert not ctx.satisfiable(), "x==17 AND x==18 should be UNSAT"


def test_wheel_ast_passthrough() -> None:
    """Pytest entry point (local verification of the wheel smoke test)."""
    check_ast_passthrough()


if __name__ == "__main__":
    check_ast_passthrough()
    print("AST-passthrough smoke test OK: claripy and Rust share one libz3.so")
