"""Tests for RustExplorationManager.

This module tests the Rust-native exploration manager for symbolic execution.
"""

from __future__ import annotations

import warnings

import pytest

import angr

# Rust availability guard, binary-path resolution, and the module-scoped
# fauxware_project fixture all live in tests/engines/conftest.py (angr-7gdp).
from tests.engines.conftest import (  # noqa: F401
    RUST_EXPLORATION_AVAILABLE,
    TEST_BINARIES_DIR,
    ExplorationEvent,
    PythonCallbacks,
    RustExplorationManager,
    RustSimState,
    _RustExplorationManager,
)

# All tests in this module require the Rust extension; skip the whole module
# when it is unavailable (matches tests/engines/test_rust_public_api.py).
pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


class TestSolverOperations:
    """Tests for solver constraint operations."""

    @classmethod
    def setup_class(cls):
        """Ensure the shared Z3 context is initialized for solver tests."""
        from angr.exploration.rust_manager import _setup_shared_z3_context

        _setup_shared_z3_context()

    def test_solver_min_max(self):
        """min() and max() return correct bounds."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x >= 10)
        ctx.add_constraint_ast(x <= 20)

        assert ctx.min(x, signed=False) == 10
        assert ctx.max(x, signed=False) == 20

    def test_unbalanced_pop_raises_valueerror(self):
        """pop() with no matching push() raises ValueError, not a panic.

        Regression for angr-ph300.48: an unbalanced ``pop()`` used to reach
        z3-rs's under-pop panic (surfaced as a PanicException/abort) instead
        of a clean Python exception.
        """
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        with pytest.raises(ValueError):
            ctx.pop()

        # Balanced push/pop still works, and the extra pop is refused again.
        ctx.push()
        ctx.pop()
        with pytest.raises(ValueError):
            ctx.pop()

    def test_pop_to_level_beyond_depth_raises(self):
        """pop_to_level past the actual scope depth raises ValueError.

        Regression for angr-ph300.48: ``pop_to_level(0, 5)`` after only two
        pushes must refuse the surplus pops cleanly.
        """
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        ctx.push()
        ctx.push()
        with pytest.raises(ValueError):
            ctx.pop_to_level(0, 5)

    def test_solver_unsatisfiable(self):
        """Contradictory constraints make solver UNSAT."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 5)
        ctx.add_constraint_ast(x == 10)
        assert not ctx.satisfiable()

    def test_contradictory_constraints_make_unsat(self):
        """Inequality contradictions (x>100 AND x<50) make solver UNSAT and
        leave eval/min/max returning None instead of bogus concrete values.

        Locks down behaviour for angr-eldx — see also
        memory `satisfiable-wrong-answer`: False from satisfiable() must
        mean a definitive UNSAT, never a swallowed exception.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x > 100)
        ctx.add_constraint_ast(x < 50)

        assert ctx.satisfiable() is False
        assert ctx.eval(x) is None
        assert ctx.min(x, signed=False) is None
        assert ctx.max(x, signed=False) is None
        assert ctx.eval_upto(x, 5) == []

    def test_unsat_core_reports_contributing_indices(self):
        """add_constraint_tracked_ast() + unsat_core() reports the indices
        of the constraints participating in the UNSAT core.

        Locks down angr-w2je: untracked constraints (added via
        add_constraint_ast) NEVER appear in unsat_core output — only
        tracked constraints do. The returned indices are 0-based and
        match the order in which add_constraint_tracked_ast() was called.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        # Three tracked constraints, indices 0,1,2. The first two are
        # contradictory; the third (x < 1000) is satisfied by neither
        # because the contradiction has already made the solver UNSAT.
        # Z3 typically returns just the minimal contradicting pair.
        idx0 = ctx.add_constraint_tracked_ast(x == 0)
        idx1 = ctx.add_constraint_tracked_ast(x == 1)
        idx2 = ctx.add_constraint_tracked_ast(x < 1000)
        assert (idx0, idx1, idx2) == (0, 1, 2)

        assert ctx.satisfiable() is False
        core = ctx.unsat_core()
        # Z3 returns at least the two contradicting constraints. It may
        # also include x < 1000 depending on the engine's bookkeeping;
        # the only invariant we check is that 0 and 1 are both present.
        assert 0 in core and 1 in core, f"expected indices 0,1 in core, got {core}"
        assert len(core) >= 2

    def test_add_constraint_tracked_ast_non_bool_input_is_lowered(self):
        """add_constraint_tracked_ast() with a non-Bool (raw bitvector) AST
        must lower it to a `!= 0` Bool constraint like add_constraint_ast
        does, not silently drop it.

        Regression for angr-d01qu: unlike its siblings add_constraint_ast
        and add_constraints, add_constraint_tracked_ast used to skip the
        is_bool() gate on the raw-fast-path Z3 AST it extracts from
        claripy, unsafely wrapping a BV-sorted Z3_ast as z3::ast::Bool and
        handing it to Z3_solver_assert_and_track. Because our context
        installs a no-op Z3 error handler (see
        native/z3-patched/src/context.rs), the resulting sort mismatch was
        swallowed *silently* inside Z3's CHECK_FORMULA guard: no exception,
        no abort, and the constraint was never actually asserted -- while
        add_constraint_tracked_ast still returned a tracked index as if it
        had succeeded (a soundness hole: the solver could produce a model
        violating a constraint the caller believed was added).

        Here `x` is passed directly (not wrapped in a comparison), so its
        semantics as a constraint are the truthiness lowering `x != 0`
        (mirrors add_constraint_ast's fallback for a non-Bool AST). Pre-fix,
        this constraint was dropped, so a subsequent `x == 0` constraint was
        wrongly satisfiable. Post-fix, `x != 0` AND `x == 0` is UNSAT.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 8)

        idx = ctx.add_constraint_tracked_ast(x)
        assert idx == 0

        ctx.add_constraint_ast(x == 0)
        assert ctx.satisfiable() is False, (
            "x != 0 (tracked) should contradict x == 0, but the solver "
            "reports SAT -- the tracked constraint was silently dropped"
        )

    def test_unsat_core_empty_when_untracked(self):
        """unsat_core() returns [] when constraints were added via
        add_constraint_ast() (the untracked fast path), even on UNSAT.

        This is the documented limitation: callers who want core
        extraction must opt in via add_constraint_tracked_ast(). Locks
        down the silent-empty behaviour memo'd in
        `avoid-rust-tracking-actions-silent-ignore`.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0)
        ctx.add_constraint_ast(x == 1)
        assert ctx.satisfiable() is False
        assert ctx.unsat_core() == []

    def test_unsat_core_empty_when_sat(self):
        """unsat_core() returns [] when the solver is satisfiable, even
        if constraints were tracked. Z3 does not produce a core for a
        SAT instance, so the matched indices list is empty.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_tracked_ast(x >= 0)
        ctx.add_constraint_tracked_ast(x <= 100)
        assert ctx.satisfiable() is True
        assert ctx.unsat_core() == []

    def test_solver_fork_independence(self):
        """Forked solver contexts are independent."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx1 = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx1.add_constraint_ast(x >= 0)
        ctx1.add_constraint_ast(x <= 100)

        ctx2 = ctx1.fork()
        ctx2.add_constraint_ast(x == 42)

        # ctx2 is constrained to 42
        assert ctx2.eval(x) == 42

        # ctx1 still has the wider range
        val = ctx1.eval(x)
        assert 0 <= val <= 100

    def test_extend_narrowing_raises_valueerror(self):
        """op_zero_extend/op_sign_extend reject to_width < source width.

        Regression for angr-ph300.38: a narrowing width used to be silently
        swallowed by zero_extend_into (``if to_width <= width { return self }``),
        handing the caller a BV wider than requested. Combined with a real
        narrower value that surfaces much later as a Z3 sort error (process
        abort under panic=abort) or a silent eq-False. The boundary now raises
        a clean ValueError instead.
        """
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        h64 = ctx.create_symbolic("x", 64)

        with pytest.raises(ValueError):
            ctx.op_zero_extend(h64.id, 32)
        with pytest.raises(ValueError):
            ctx.op_sign_extend(h64.id, 32)

        # Equal width is a legitimate no-op (returns a same-width handle).
        assert ctx.op_zero_extend(h64.id, 64).width == 64
        assert ctx.op_sign_extend(h64.id, 64).width == 64

        # Genuine widening still works.
        assert ctx.op_zero_extend(h64.id, 96).width == 96
        assert ctx.op_sign_extend(h64.id, 96).width == 96

    def test_binary_op_width_mismatch_raises_valueerror(self):
        """Binary ops reject mismatched operand widths at the Python boundary.

        Regression for angr-ph300.32: op_eq used to silently fold a width
        mismatch to constant False (where claripy raises), while op_ne and the
        arith/shift/cmp siblings only ``debug_assert`` equal width — so a
        release build handed mismatched sorts to Z3 and aborted the process.
        The op_binary! layer now rejects the mismatch with a ValueError for
        every binary op, uniformly.
        """
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        h32 = ctx.create_symbolic("a", 32)
        h64 = ctx.create_symbolic("b", 64)

        # Comparisons: previously silent-False (eq) or Z3 abort (ne, ordering).
        for op in ("op_eq", "op_ne", "op_ult", "op_ule", "op_sgt", "op_sge"):
            with pytest.raises(ValueError):
                getattr(ctx, op)(h32.id, h64.id)
            with pytest.raises(ValueError):
                getattr(ctx, op)(h64.id, h32.id)

        # Arithmetic / bitwise / shift ops share the same guard.
        for op in ("op_add", "op_sub", "op_and", "op_xor", "op_shl", "op_lshr"):
            with pytest.raises(ValueError):
                getattr(ctx, op)(h32.id, h64.id)

        # Equal-width operands are unaffected.
        h32b = ctx.create_symbolic("c", 32)
        assert ctx.op_eq(h32.id, h32b.id).width == 1
        assert ctx.op_add(h32.id, h32b.id).width == 32

    def test_eval_fp_comparison_bool_no_panic(self):
        """eval()/eval_upto() of a Bool-sorted AST that claripy_to_rustbv
        cannot lower (an fpEQ float comparison) must return a 0/1 value,
        never BV-wrap the Bool node.

        Regression for angr-58ks: extract_z3_ast_ptr returns a raw Z3_ast
        with no sort check; the eval fast paths used to BV::wrap it
        unconditionally (width defaulting to 64). For a Bool-sorted node
        that is a wrong-sort Z3 call, which trips Z3's error handler
        (process abort), not a recoverable PyErr. The fix lowers a Bool to
        a 1-bit BV via ite(1, 0) before evaluating.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        a = claripy.FPS("a", claripy.FSORT_DOUBLE)
        b = claripy.FPS("b", claripy.FSORT_DOUBLE)
        eq = claripy.fpEQ(a, b)  # Bool, not lowered by claripy_to_rustbv

        ctx = RustSolverContext()
        # Unconstrained: the model may make the comparison true or false,
        # but the call must complete (no abort) and yield a 0/1 int.
        val = ctx.eval(eq)
        assert val in (0, 1), f"eval of Bool AST should be 0/1, got {val!r}"

        results = ctx.eval_upto(eq, 5)
        assert set(results) <= {0, 1}, f"eval_upto of Bool AST should be 0/1, got {results!r}"

        # Constrain the comparison true (also exercises the add_constraint
        # Bool fast path) and confirm eval now reports 1.
        ctx2 = RustSolverContext()
        ctx2.add_constraint_ast(eq)
        assert ctx2.satisfiable()
        assert ctx2.eval(eq) == 1
        assert 1 in ctx2.eval_upto(eq, 5)

    def test_fork_constraint_bidirectional_isolation(self):
        """Bidirectional fork isolation: parent and sibling constraints
        added after fork() must not leak across.

        Locks the freeze-in-place fork invariant (commit 589b814e9,
        SymContext::fork at native/angr/src/symbolic/context.rs:1823):
        when push_level==0, fork drains the local assumed/z3 vecs into
        a shared Arc — but each side's post-fork additions go into its
        own fresh local vec and must remain isolated. Regression for
        angr-agvl.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx_a = RustSolverContext()
        x = claripy.BVS("x_iso", 32)
        y = claripy.BVS("y_iso", 32)
        z = claripy.BVS("z_iso", 32)

        # A asserts X=5 — this becomes the frozen prefix at fork time.
        ctx_a.add_constraint_ast(x == 5)

        # Fork: B inherits the X=5 prefix as a shared Arc.
        ctx_b = ctx_a.fork()

        # Each side adds an independent post-fork constraint.
        ctx_a.add_constraint_ast(y == 10)
        ctx_b.add_constraint_ast(z == 20)

        # Both contexts remain satisfiable (no cross-contamination forces
        # a contradiction).
        assert ctx_a.satisfiable()
        assert ctx_b.satisfiable()

        # Frozen prefix (X=5) is visible to both.
        assert ctx_a.eval(x) == 5
        assert ctx_b.eval(x) == 5

        # Each side sees its own post-fork constraint.
        assert ctx_a.eval(y) == 10
        assert ctx_b.eval(z) == 20

        # The proof of isolation: if B's Z=20 had leaked into A, adding
        # Z!=20 to A would make it UNSAT. It must remain SAT.
        ctx_a.add_constraint_ast(z != 20)
        assert ctx_a.satisfiable()

        # Symmetric check: A's Y=10 must not have leaked into B.
        ctx_b.add_constraint_ast(y != 10)
        assert ctx_b.satisfiable()

    def test_fork_inside_push_isolation(self):
        """Fork while inside a push() frame: the in-transaction freeze
        path (freeze_z3_assertions, context.rs:2060) takes a different
        branch — it must allocate a fresh merged Vec rather than draining
        the parent's local in place, because rollback expects local intact.

        Sibling state must still be isolated from the parent's post-fork
        constraints. Regression for angr-agvl.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx_a = RustSolverContext()
        x = claripy.BVS("x_push_iso", 32)
        y = claripy.BVS("y_push_iso", 32)
        z = claripy.BVS("z_push_iso", 32)

        # Frozen prefix at level 0.
        ctx_a.add_constraint_ast(x == 5)

        # Enter a push frame, then add another constraint and fork.
        ctx_a.push()
        ctx_a.add_constraint_ast(y == 10)
        ctx_b = ctx_a.fork()  # B inherits {x==5, y==10}

        # A still inside push — add Z constraint.
        ctx_a.add_constraint_ast(z == 20)

        # B is at push_level==0 (fork resets it) — add a different Z.
        ctx_b.add_constraint_ast(z == 99)

        # Both contexts independently satisfiable.
        assert ctx_a.satisfiable()
        assert ctx_b.satisfiable()

        # A sees its post-fork Z=20.
        assert ctx_a.eval(z) == 20
        # B sees its own Z=99 — A's Z=20 must NOT have leaked.
        assert ctx_b.eval(z) == 99

        # Both still see the shared frozen prefix and the in-frame Y=10.
        assert ctx_a.eval(x) == 5
        assert ctx_b.eval(x) == 5
        assert ctx_a.eval(y) == 10
        assert ctx_b.eval(y) == 10

        # Pop A back: Z=20 should disappear from solver (push/pop is
        # solver-frame-scoped). Locks the existing
        # invariant-symcontext-push-not-cache-aware: pop unwinds the
        # solver but the cache keeps the assertion. Just verify solver
        # behaviour here — adding z!=20 must remain SAT after pop.
        ctx_a.pop()
        ctx_a.add_constraint_ast(z != 20)
        assert ctx_a.satisfiable()

    def test_solver_multiple_variables(self):
        """Solver handles multiple independent symbolic variables."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast(x >= 30)
        ctx.add_constraint_ast(x <= 50)
        ctx.add_constraint_ast(y >= 60)
        ctx.add_constraint_ast(y <= 80)

        assert ctx.satisfiable()
        vx = ctx.eval(x)
        vy = ctx.eval(y)
        assert 30 <= vx <= 50
        assert 60 <= vy <= 80

    def test_solver_eval_upto_wide_bvs(self):
        """eval_upto should handle BVS wider than 128 bits without truncation."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        # 296-bit BVS (like whitehatvn's 37-byte arg)
        x = claripy.BVS("wide_var", 296)
        # Constrain first byte to 'A' (0x41) and last byte to 'Z' (0x5a)
        ctx.add_constraint_ast(claripy.Extract(295, 288, x) == 0x41)
        ctx.add_constraint_ast(claripy.Extract(7, 0, x) == 0x5A)

        results = ctx.eval_upto(x, 2)
        assert len(results) >= 1, "should find at least one solution"
        for r in results:
            nbytes = 37
            val_bytes = r.to_bytes(nbytes, "big")
            assert val_bytes[0] == 0x41, "first byte should be 'A'"
            assert val_bytes[-1] == 0x5A, "last byte should be 'Z'"

    def test_solver_eval_upto_excludes_duplicates(self):
        """eval_upto should return distinct values."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(x >= 1)
        ctx.add_constraint_ast(x <= 5)

        results = ctx.eval_upto(x, 10)
        assert len(results) == 5, "should find exactly 5 solutions for [1..5]"
        assert len(set(results)) == 5, "all solutions should be distinct"

    def test_constraint_weakening_through_ite(self):
        """Constraints on an ITE result must weaken correctly to its branches.

        Build `result = If(x > 5, x, 100)` and constrain `result > 50`. This
        is satisfiable in two disjoint ways:
          - x > 5  AND  x > 50  → x in (50, 2^32)
          - x <= 5 AND  100 > 50 → any x in [0, 5] (else-branch always > 50)

        So x is NOT pinned to any single value; the solver must admit
        solutions like x = 51 (and also x in [0, 5]). A regression that
        loses ITE structure could over-constrain and force x to one branch.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        result = claripy.If(x > 5, x, claripy.BVV(100, 32))
        ctx.add_constraint_ast(result > 50)

        assert ctx.satisfiable()

        # Witness 1: x = 51 must be admissible (then-branch satisfies > 50).
        ctx_then = ctx.fork()
        ctx_then.add_constraint_ast(x == 51)
        assert ctx_then.satisfiable(), "x=51 should be a valid model"

        # Witness 2: x = 3 must also be admissible (else-branch=100 > 50).
        ctx_else = ctx.fork()
        ctx_else.add_constraint_ast(x == 3)
        assert ctx_else.satisfiable(), "x=3 should be a valid model"

        # Counter-witness: x = 30 must NOT be admissible
        # (then-branch=30, else-branch unreachable since x>5).
        ctx_bad = ctx.fork()
        ctx_bad.add_constraint_ast(x == 30)
        assert not ctx_bad.satisfiable(), "x=30 must be ruled out"

    def test_z3_seed_pin_not_attempted(self):
        """``build_solver_params`` does NOT pin ``smt.random_seed`` /
        ``sat.random_seed`` — a deliberate non-pin documented in
        angr-iaol.1 (2026-05-25).

        The audit (parent iaol) hypothesized pinning would deliver model
        stability across fresh ``RustSolverContext`` instances. Empirically
        the three reachable param-name forms via the z3-0.19.7
        ``Params::set_u32`` route are all problematic:

        * ``smt.random_seed`` / ``sat.random_seed``: corrupt the solver.
          ``eval()`` returns models that *violate* the asserted
          constraints (eg ``x=0`` for ``x>=100``).
        * ``random_seed`` (no module prefix): accepted but produces
          *more* variation across instances than no pin, **and** breaks
          ``test_model_stability_constraint_order`` (which passes
          without any pin).

        This test is a sanity check that the solver still respects
        asserted constraints — i.e. no broken seed pin was reintroduced.
        See ``iaol1-seed-pin-empirically-broken`` memory for full
        details and follow-up paths (Z3_global_param_set before
        ``Solver::new``).
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        x = claripy.BVS("x", 32)
        ctx = RustSolverContext()
        ctx.add_constraint_ast(x >= 100)
        ctx.add_constraint_ast(x <= 200)
        v = ctx.eval(x)
        assert v is not None and 100 <= v <= 200, (
            f"Sanity: eval(x) must satisfy x in [100, 200]; got {v}. "
            "If this fails, a seed-pin attempt likely corrupted the solver "
            "— see angr-iaol.1 close-out memory "
            "iaol1-seed-pin-empirically-broken."
        )

    def test_model_stability_constraint_order(self):
        """Adding the same constraints in two orders yields identical
        deterministic bounds (``min``/``max``) and per-solver-valid ``eval``.

        Z3 makes NO guarantee about *which* satisfying model ``eval`` returns,
        so asserting ``eval(x_a) == eval(x_b)`` across two independently
        constructed solvers is fragile: Z3's heuristics legitimately pick
        different valid models depending on process-internal state that other
        tests perturb. That fragility is the whole story behind angr-4st5 —
        the original ``v_a == v_b`` assertion flaked ~1-in-4 under the
        register-proxy gate, but it is NOT gate-specific (it also flakes
        gate-off given the right ``PYTHONHASHSEED``); it is inherent
        eval-model non-determinism, not cross-test solver pollution.

        What our bridge *must* keep order-stable is the constraint SET: a
        reorder or dedup bug would change the satisfiable range. ``min`` and
        ``max`` are unique by construction, so they are the correct
        order-independence witnesses — they catch a dropped/duplicated
        constraint while staying immune to Z3's model-choice latitude.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        x = claripy.BVS("x", 32)
        c1 = x >= 100
        c2 = x <= 200
        c3 = x != 150

        ctx_a = RustSolverContext()
        ctx_a.add_constraint_ast(c1)
        ctx_a.add_constraint_ast(c2)
        ctx_a.add_constraint_ast(c3)

        ctx_b = RustSolverContext()
        ctx_b.add_constraint_ast(c3)
        ctx_b.add_constraint_ast(c2)
        ctx_b.add_constraint_ast(c1)

        # Deterministic order-independence witnesses. A reorder or dedup bug
        # in the bridge would change the satisfiable range, which min/max
        # capture exactly (unlike eval, whose model choice Z3 leaves free).
        assert ctx_a.min(x, signed=False) == ctx_b.min(x, signed=False) == 100
        assert ctx_a.max(x, signed=False) == ctx_b.max(x, signed=False) == 200

        # eval must still return a *valid* model from each solver (in range,
        # != 150) — but we deliberately do NOT assert the two models are
        # equal, because Z3 does not guarantee cross-instance model identity.
        v_a = ctx_a.eval(x)
        v_b = ctx_b.eval(x)
        assert v_a is not None and 100 <= v_a <= 200 and v_a != 150
        assert v_b is not None and 100 <= v_b <= 200 and v_b != 150

    def test_solver_contradictory_find_avoid(self):
        """Same address in find and avoid should avoid (avoid takes priority)."""
        mgr = _RustExplorationManager("amd64")
        # Stub callbacks so run() doesn't raise "callbacks not set". The
        # avoid/find classification happens at the top of the run loop
        # *before* any lift/step, so these are never actually invoked here.
        callbacks = PythonCallbacks()
        callbacks.set_memory_load(lambda a, s: (bytes(s), False, None))
        callbacks.set_memory_store(lambda a, d: None)
        callbacks.set_lift_block(lambda a: "{}")
        mgr.set_callbacks(callbacks)
        mgr.set_find_addrs([0x1000])
        mgr.set_avoid_addrs([0x1000])
        state = RustSimState("amd64")
        state.pc = 0x1000
        mgr.add_state("active", state)
        assert mgr.active_count() == 1

        # Step the manager so the run loop actually classifies the state.
        # run_loop.rs checks avoid_addrs (line ~162) *before* find_addrs
        # (line ~206), so a state already sitting at an address in BOTH
        # stashes must land in "avoid", never "found". Asserting the
        # resulting stash placement (not just no-crash) pins that priority:
        # swapping the two classification blocks, or no-op'ing
        # set_avoid_addrs, would flip the state into "found" and redden this.
        mgr.run(5)
        counts = mgr.stash_counts()
        assert counts.get("avoid", 0) == 1, (
            f"State at 0x1000 (in both find and avoid) should land in 'avoid' "
            f"— avoid takes priority over find; stashes={counts}"
        )
        assert counts.get("found", 0) == 0, (
            f"State at 0x1000 must NOT be found when the address is also an avoid address; stashes={counts}"
        )


class TestDeterministicMode:
    """``RustExplorationManager(deterministic=True)`` — angr-iaol.2.

    The flag pins ``smt.random_seed`` + ``sat.random_seed`` to 0 via
    ``Z3_global_param_set`` before any new solver is constructed. Z3 4.13
    still reserves variable / restart heuristic latitude that is not
    bounded by these seeds, so the flag *narrows* but does not *close*
    run-to-run model variation. See iaol1-seed-pin-empirically-broken
    memory for the prior solver-level attempt that failed.
    """

    def test_set_z3_global_param_smoke(self):
        """``set_z3_global_param`` FFI actually writes the Z3 module param."""
        from angr.rustylib.vex_engine import get_z3_global_param, set_z3_global_param

        # Round-trip via the getter so a no-op or arg-swapped FFI wrapper
        # reddens this. A bare set-with-no-readback passed even when the
        # underlying Z3_global_param_set (a void C fn that silently ignores
        # bad keys) was gutted — masking dead deterministic-seed wiring.
        set_z3_global_param("smt.random_seed", "0")
        assert get_z3_global_param("smt.random_seed") == "0"

        set_z3_global_param("sat.random_seed", "0")
        assert get_z3_global_param("sat.random_seed") == "0"

        # A non-zero value must round-trip too — guards against a wrapper
        # that hard-codes "0" instead of forwarding the caller's value.
        set_z3_global_param("smt.random_seed", "42")
        assert get_z3_global_param("smt.random_seed") == "42"

    def test_deterministic_kwarg_accepted(self, fauxware_project):
        """``deterministic=True`` constructs cleanly + records flag on self."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state], deterministic=True)
        assert mgr._deterministic is True

        # Default keeps existing behavior — flag absent → False.
        state2 = fauxware_project.factory.entry_state()
        mgr2 = RustExplorationManager(fauxware_project, [state2])
        assert mgr2._deterministic is False

    def test_fauxware_explore_stable_under_deterministic(self, fauxware_project):
        """Two fauxware explorations with ``deterministic=True`` produce the
        same found-stash size and the same evaluated stdin for the first
        found state. Constructs fresh managers per run so the global pin
        is the only thing tying the runs together (not in-process solver
        state).

        Z3 4.13 retains heuristic latitude even with the seeds pinned —
        if this test ever flakes on the stdin equality, the right move is
        to weaken to "decoded text equal modulo trailing 0xff padding"
        (the canonical residual from defcamp_r100) rather than disable
        the test. See rust_engine.rst "Deterministic mode" for context.
        """

        def _run() -> tuple[int, bytes]:
            state = fauxware_project.factory.entry_state()
            mgr = RustExplorationManager(fauxware_project, [state], deterministic=True)
            mgr.explore(find=0x4006ED, avoid=0x4006FD, max_steps=50000)
            assert len(mgr.found) > 0, "expected at least one found state"
            stdin = mgr.found[0].posix.dumps(0)
            return len(mgr.found), bytes(stdin)

        n_a, stdin_a = _run()
        n_b, stdin_b = _run()
        assert n_a == n_b, f"found-stash size differs run-to-run: {n_a} vs {n_b}"
        assert stdin_a == stdin_b, (
            f"stdin differs run-to-run despite deterministic=True: "
            f"{stdin_a!r} vs {stdin_b!r}. Z3 heuristic latitude may have "
            "drifted; consider weakening to a padding-tolerant comparison "
            "rather than disabling — see rust_engine.rst 'Deterministic mode'."
        )

    def test_deterministic_reaches_state_solvers(self, fauxware_project):
        """angr-op0dn.10.3: the kwarg reaches each state's Rust SymContext.

        The manager-level flag is only meaningful if the per-state solver
        actually switches to the canonical-witness path — a flag stored on
        ``self`` and consulted nowhere (the pre-10.3 state of the world) would
        pass ``test_deterministic_kwarg_accepted`` and still return arbitrary
        Z3 witnesses. Probe the state solver itself, not the manager mirror.
        """
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state], deterministic=True)
        assert mgr._rust_mgr.is_deterministic() is True
        ids = mgr._rust_mgr.get_state_ids("active")
        assert ids, "expected the seed state in the active stash"
        assert all(mgr._rust_mgr.state_is_deterministic(sid) for sid in ids)

        # Off by default, all the way down.
        state2 = fauxware_project.factory.entry_state()
        mgr2 = RustExplorationManager(fauxware_project, [state2])
        assert mgr2._rust_mgr.is_deterministic() is False
        ids2 = mgr2._rust_mgr.get_state_ids("active")
        assert ids2 and not any(mgr2._rust_mgr.state_is_deterministic(sid) for sid in ids2)

    def test_deterministic_warns_with_multiple_workers(self, fauxware_project, monkeypatch):
        """Multi-worker scheduler + deterministic=True warns, does not raise.

        Steal order in the work-stealing pool is nondeterministic by design
        (scheduler_tests.rs::test_determinism_result_set): the found *set* is
        stable, the order is not. The combination stays legal — witness choice
        is per-state and still canonical — so the user gets a warning.
        """
        monkeypatch.setenv("RUST_PARALLEL_WORKERS", "4")
        state = fauxware_project.factory.entry_state()
        with pytest.warns(RuntimeWarning, match="steal order"):
            mgr = RustExplorationManager(fauxware_project, [state], deterministic=True)
        assert mgr._rust_mgr.is_deterministic() is True

    def test_no_warning_single_worker(self, fauxware_project, monkeypatch):
        """The guard fires on worker count, not on the flag alone."""
        monkeypatch.setenv("RUST_PARALLEL_WORKERS", "1")
        state = fauxware_project.factory.entry_state()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            RustExplorationManager(fauxware_project, [state], deterministic=True)


class TestRustStateRegisters:
    """Tests for register operations on RustSimState."""

    def test_register_set_get(self):
        """Set and get multiple registers."""
        state = RustSimState("amd64")
        state.set_register("rax", 0x1234)
        state.set_register("rbx", 0x5678)
        state.set_register("rcx", 0xABCD)

        assert state.get_register("rax") == 0x1234
        assert state.get_register("rbx") == 0x5678
        assert state.get_register("rcx") == 0xABCD

    def test_register_fork_isolation(self):
        """Forked states have independent registers."""
        state1 = RustSimState("amd64")
        state1.set_register("rax", 100)

        state2 = state1.fork()
        state2.set_register("rax", 200)

        assert state1.get_register("rax") == 100
        assert state2.get_register("rax") == 200

    def test_pc_set_get(self):
        """PC property works correctly."""
        state = RustSimState("amd64")
        state.pc = 0xDEADBEEF
        assert state.pc == 0xDEADBEEF

    def test_state_id_unique(self):
        """Each state has a unique ID."""
        s1 = RustSimState("amd64")
        s2 = RustSimState("amd64")
        s3 = s1.fork()
        ids = {s1.state_id, s2.state_id, s3.state_id}
        assert len(ids) == 3, "State IDs should be unique"


class TestSerializeIRSB:
    """Tests for IRSB serialization (used in lift callbacks)."""

    def test_serialize_basic_block(self, fauxware_project):
        """Serializing a basic block produces valid JSON."""
        import json

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        # Lift a block
        block = fauxware_project.factory.block(fauxware_project.entry)
        irsb = block.vex

        # Serialize
        result = mgr._serialize_irsb(irsb)
        data = json.loads(result)

        assert "addr" in data
        assert "statements" in data
        assert "next" in data
        assert "jumpkind" in data
        assert "tyenv" in data
        assert len(data["statements"]) > 0

    def test_serialize_roundtrip_consistency(self, fauxware_project):
        """Serializing the same block twice produces identical output."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        block = fauxware_project.factory.block(fauxware_project.entry)
        irsb = block.vex

        result1 = mgr._serialize_irsb(irsb)
        result2 = mgr._serialize_irsb(irsb)
        assert result1 == result2


class TestSerializeDescr:
    """Unit tests for _serialize_descr and its GetI/PutI callers."""

    class _Descr:
        base = 40
        elemTy = "Ity_I64"
        nElems = 8

    class RdTmp:
        def __init__(self, tmp):
            self.tmp = tmp

    def test_serialize_descr_all_fields(self):
        from angr.exploration.rust_irsb_serializer import _serialize_descr

        out = _serialize_descr(self._Descr())
        assert out == {"base": 40, "elemTy": "Ity_I64", "nElems": 8}

    def test_serialize_descr_hasattr_guard(self):
        """Missing attributes fall back to defaults instead of raising."""
        from angr.exploration.rust_irsb_serializer import _serialize_descr

        out = _serialize_descr(object())
        assert out == {"base": 0, "elemTy": "Ity_I64", "nElems": 0}

    def test_serialize_geti_expr(self):
        """GetI expression serialization embeds the descriptor."""
        from angr.exploration.rust_irsb_serializer import _serialize_expr

        class GetI:
            descr = TestSerializeDescr._Descr()
            ix = TestSerializeDescr.RdTmp(3)
            bias = 2

        out = _serialize_expr(GetI())
        assert out["tag"] == "Iex_GetI"
        assert out["descr"] == {"base": 40, "elemTy": "Ity_I64", "nElems": 8}
        assert out["bias"] == 2
        assert out["ix"] == {"tag": "Iex_RdTmp", "tmp": 3}

    def test_serialize_puti_stmt(self):
        """PutI statement serialization embeds the descriptor."""
        from angr.exploration.rust_irsb_serializer import _serialize_stmt

        class PutI:
            descr = TestSerializeDescr._Descr()
            ix = TestSerializeDescr.RdTmp(3)
            bias = 2
            data = TestSerializeDescr.RdTmp(9)

        out = _serialize_stmt(PutI())
        assert out["tag"] == "Ist_PutI"
        assert out["descr"] == {"base": 40, "elemTy": "Ity_I64", "nElems": 8}
        assert out["bias"] == 2


class TestSerializeConst:
    """Unit tests for _serialize_const (Ico_ constant serialization)."""

    def _make(self, name, value):
        con = object.__new__(type(name, (), {}))
        con.value = value
        return con

    def test_serialize_u128_low_high(self):
        """U128 must emit {low, high}, matching Rust PyVexConst::U128."""
        from angr.exploration.rust_irsb_serializer import _serialize_const

        val = (0xDEADBEEFCAFEBABE << 64) | 0x0123456789ABCDEF
        out = _serialize_const(self._make("U128", val))
        assert out == {
            "tag": "Ico_U128",
            "low": 0x0123456789ABCDEF,
            "high": 0xDEADBEEFCAFEBABE,
        }

    def test_serialize_v128_low_high(self):
        """V128 keeps its {low, high} shape (regression guard)."""
        from angr.exploration.rust_irsb_serializer import _serialize_const

        val = (0x1111111111111111 << 64) | 0x2222222222222222
        out = _serialize_const(self._make("V128", val))
        assert out == {
            "tag": "Ico_V128",
            "low": 0x2222222222222222,
            "high": 0x1111111111111111,
        }

    def test_serialize_u64_generic_value(self):
        """Non-128-bit constants keep the generic {value} shape."""
        from angr.exploration.rust_irsb_serializer import _serialize_const

        out = _serialize_const(self._make("U64", 0x1234))
        assert out == {"tag": "Ico_U64", "value": 0x1234}


class TestExplorationIntegration:
    """Integration tests with real binaries."""

    def test_explore_with_max_steps(self, fauxware_project):
        """Exploration respects max_steps limit."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED, max_steps=1)

        # 1 step is too few to find target in fauxware
        stats = mgr.stats
        assert stats["ffi_crossings"] > 0, "should have crossed FFI boundary"
        assert len(mgr.found) == 0, "1 step too few to find target in fauxware"

    def test_explore_finds_correct_state(self, fauxware_project):
        """Full exploration finds the expected state."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED, avoid=0x4006FD, max_steps=50000)

        assert len(mgr.found) > 0, "Should find at least one state"

    def test_python_callback_aggregate_counters(self, fauxware_project):
        """angr-b00q: mgr.stats exposes aggregate `python_callback_count` and
        `python_callback_dispatch_us` that sum across all per-kind callback
        buckets in PerformanceTracker. Both must be non-zero after a real
        exploration (driven by lift_block / memory_load / simprocedure
        callbacks) and dispatch_us must be bounded by the exploration's
        wall-clock time.

        Forces `use_native_lift=False` so the pyvex lift_block callback fires:
        with libvex-ffi default-ON (angr-3trr7) cold blocks lift natively and
        fauxware otherwise drives *zero* python callbacks, which would make this
        aggregate-counter assertion build-dependent.
        """
        import time

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state], use_native_lift=False)

        _start = time.perf_counter()
        mgr.explore(find=0x4006ED, avoid=0x4006FD, max_steps=50000)
        elapsed_us = (time.perf_counter() - _start) * 1e6

        stats = mgr.stats
        assert "python_callback_count" in stats
        assert "python_callback_dispatch_us" in stats
        assert stats["python_callback_count"] > 0, (
            f"expected non-zero aggregate callback count; got {stats['python_callback_count']}"
        )
        assert stats["python_callback_dispatch_us"] > 0, (
            f"expected non-zero aggregate dispatch_us; got {stats['python_callback_dispatch_us']}"
        )
        assert stats["python_callback_dispatch_us"] <= elapsed_us, (
            f"dispatch_us must be bounded by exploration wall time; "
            f"got dispatch_us={stats['python_callback_dispatch_us']} > elapsed_us={elapsed_us:.0f}"
        )

        # Aggregate must equal the sum of per-kind buckets we surface.
        # NB `callback_count` (no kind suffix) is a Python-side FFI-crossing
        # bookkeeping counter, not a PerformanceTracker bucket — skip it.
        bucket_count = sum(
            v for k, v in stats.items() if k.startswith("callback_") and k.endswith("_count") and k != "callback_count"
        )
        bucket_ns = sum(v for k, v in stats.items() if k.startswith("callback_") and k.endswith("_total_ns"))
        assert stats["python_callback_count"] == bucket_count
        assert stats["python_callback_dispatch_us"] == bucket_ns // 1000

    def test_callback_interpreter_mem_counter_parity(self, fauxware_project):
        """angr-obrm: callback-interpreter VEX load/store paths bump the
        global `mem_load_count` / `mem_store_count` / `mem_load_bytes` /
        `mem_store_bytes` counters at parity with the native VEX
        interpreter (which bumps them via `SymbolicMemory::load_concrete` /
        `store_concrete`). Before this wiring, callback-heavy binaries
        underreported memory work because the cb-fallback paths bypassed
        SymbolicMemory entirely.

        fauxware exercises a mix of try_rust_memory_{load,store} hits and
        callback-path fallbacks, so both wiring sites contribute. Test
        asserts (a) counters are non-zero post-exploration and (b) bytes
        scale with count (≥ size of a single 1-byte op).
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.reset_solver_stats()

        baseline = mgr.get_solver_stats()
        assert baseline["mem_load_count"] == 0
        assert baseline["mem_store_count"] == 0
        assert baseline["mem_load_bytes"] == 0
        assert baseline["mem_store_bytes"] == 0

        mgr.explore(find=0x4006ED, avoid=0x4006FD, max_steps=50000)

        stats = mgr.get_solver_stats()
        assert stats["mem_load_count"] > 0, (
            f"expected non-zero mem_load_count after fauxware exploration; got {stats['mem_load_count']}"
        )
        assert stats["mem_store_count"] > 0, (
            f"expected non-zero mem_store_count after fauxware exploration; got {stats['mem_store_count']}"
        )
        # Bytes must be at least the count (every op moves ≥1 byte) and
        # bounded by 64 * count (largest VEX load width on amd64 is 64
        # bytes for vector loads, but fauxware is scalar so the typical
        # value is 1-8 bytes per op).
        assert stats["mem_load_bytes"] >= stats["mem_load_count"]
        assert stats["mem_store_bytes"] >= stats["mem_store_count"]
        assert stats["mem_load_bytes"] <= 64 * stats["mem_load_count"]
        assert stats["mem_store_bytes"] <= 64 * stats["mem_store_count"]

    def test_explore_with_timeout_technique(self, fauxware_project):
        """Timeout technique stops exploration before max_steps and finds nothing."""
        from angr.exploration_techniques import Timeout

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.use_technique(Timeout(timeout=0.001))  # 1ms timeout — should trigger quickly
        mgr.explore(find=0x4006ED, max_steps=50000)
        # The 1ms budget expires long before the 50000-step cap, so the target
        # is never found and only a handful of steps execute.
        assert not mgr.found
        assert mgr.stats.get("steps", 0) < 50000

    def test_hook_fp_constraint_uses_z3_ptr_fallback(self, fauxware_project):
        """Constraints with ops that claripy_to_rustbv can't translate (e.g. FP)
        must still reach the Rust solver via the Z3 ptr fallback in
        sync_constraints_from_python (added for angr-1epf).

        Without the fallback, FP constraints raise UnsupportedOp inside
        claripy_to_rustbv and get dropped silently. We compare the Rust-side
        Z3 assertion count against an unhooked baseline run; the hooked run
        must produce strictly more assertions per state.
        """
        import claripy

        proj = fauxware_project

        # Baseline run (no hook).
        baseline_state = proj.factory.entry_state()
        baseline_mgr = RustExplorationManager(proj, [baseline_state])
        baseline_mgr.run(max_steps=300)
        baseline_counts = sorted(
            len(baseline_mgr._rust_mgr.export_z3_constraint_ptrs(sid))
            for stash in ("active", "found", "deadended")
            for sid in baseline_mgr._rust_mgr.get_state_ids(stash)
        )
        assert baseline_counts, "baseline run produced no states"

        # Hooked run: FP constraint goes through the Z3 ptr fallback.
        main_sym = proj.loader.find_symbol("main")
        hook_addr = main_sym.rebased_addr
        hook_fired = []

        def hook(state):
            fp = claripy.FPS("hook_fp_var", claripy.FSORT_DOUBLE)
            state.solver.add(fp == claripy.FPV(1.5, claripy.FSORT_DOUBLE))
            hook_fired.append(True)

        proj.hook(hook_addr, hook=hook, length=0)
        try:
            hooked_state = proj.factory.entry_state()
            hooked_mgr = RustExplorationManager(proj, [hooked_state])
            hooked_mgr.run(max_steps=300)
        finally:
            proj.unhook(hook_addr)

        assert hook_fired, "hook never fired"
        hooked_counts = sorted(
            len(hooked_mgr._rust_mgr.export_z3_constraint_ptrs(sid))
            for stash in ("active", "found", "deadended")
            for sid in hooked_mgr._rust_mgr.get_state_ids(stash)
        )
        assert hooked_counts, "hooked run produced no states"
        assert len(hooked_counts) == len(baseline_counts), (
            f"hooked vs baseline state counts differ: hooked={len(hooked_counts)}, baseline={len(baseline_counts)}"
        )
        # Every paired state should have strictly more Z3 assertions in the
        # hooked run. If claripy_to_rustbv silently drops the FP constraint
        # (the bug this fix targets), the counts would match exactly.
        for h, b in zip(hooked_counts, baseline_counts):
            assert h > b, (
                f"Rust solver gained no extra constraints with FP hook "
                f"(hooked={hooked_counts}, baseline={baseline_counts}); "
                "the Z3 ptr fallback is not engaging."
            )

    def test_hook_constraint_propagates_to_rust_solver(self, fauxware_project):
        """A constraint added inside a Python hook callback must reach the Rust solver.

        Regression for angr-1epf: post-callback the dispatch passes new claripy
        constraints back via resume_after_simprocedure. If that round-trip drops
        the constraint, the Rust state's solver itself will be missing it.
        """
        import claripy

        proj = fauxware_project
        main_sym = proj.loader.find_symbol("main")
        assert main_sym is not None, "fauxware should have a main symbol"
        hook_addr = main_sym.rebased_addr

        captured = {}

        def hook(state):
            x = claripy.BVS("hook_constraint_var", 32)
            state.solver.add(x == 0xCAFEBABE)
            captured.setdefault("var", x)

        proj.hook(hook_addr, hook=hook, length=0)
        try:
            state = proj.factory.entry_state()
            mgr = RustExplorationManager(proj, [state])
            mgr.run(max_steps=300)

            assert "var" in captured, "hook never fired"

            # Inspect every live Rust state's exported constraint ASTs and
            # confirm at least one references our `hook_constraint_var` BVS
            # — the AST must have actually crossed the Python→Rust boundary,
            # not just lived in the Python solver.
            def constraint_mentions_hook_var(ast):
                # claripy assigns BVS a uniqueness suffix (e.g. "_1_32"), so
                # match by prefix rather than exact name.
                try:
                    leaves = list(ast.leaf_asts())
                except Exception:
                    return False
                for leaf in leaves:
                    args = getattr(leaf, "args", ())
                    if args and isinstance(args[0], str) and args[0].startswith("hook_constraint_var"):
                        return True
                return False

            saw = False
            for stash in ("active", "found", "deadended"):
                for sid in mgr._rust_mgr.get_state_ids(stash):
                    try:
                        rust_constraints = mgr._rust_mgr.export_state_constraints(sid)
                    except Exception:
                        continue
                    if any(constraint_mentions_hook_var(c) for c in rust_constraints):
                        saw = True
                        break
                if saw:
                    break

            assert saw, (
                "Rust solver does not contain any constraint that references "
                "`hook_constraint_var`. The hook's constraint appears to have "
                "been dropped on the Python→Rust round-trip."
            )
        finally:
            proj.unhook(hook_addr)

    def test_hook_copies_symbolic_memory_preserves_symbolicity(self):
        """Hook that copies symbolic memory must preserve symbolicity at the dest.

        Regression for angr-ctct (sokohashv2's do_repmovsd pattern). The hook
        runs ``state.memory.load(src, 32) -> state.memory.store(dst, ...)``,
        mirroring the manual ``rep movsd`` in solve.py. After resume, the
        Rust engine must read symbolic bytes from ``dst``, not concrete
        witnesses. The shellcode's ``mov rax, [rdi]`` is interpreted by
        Rust's VEX engine, so an concrete-witness load shows up as a
        non-symbolic ``rax`` that cannot be constrained to alternate values.

        Source bytes are not pre-stored; they're left to angr's
        ``default_filler_mixin`` so the copy operates on a Concat of small
        per-byte/per-chunk symbolic fillers — the exact shape sokohashv2's
        do_repmovsd encounters.
        """

        # 0x1000: nop                (hooked, length=1)
        # 0x1001: mov rax, [rdi]     ; 48 8b 07 — Rust VEX load
        # 0x1004: ret                ; c3
        shellcode = bytes.fromhex("90488b07c3")
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)

        SRC_ADDR = 0x3000
        DST_ADDR = 0x2000

        state = proj.factory.blank_state(addr=0x1000)
        state.regs.rdi = DST_ADDR
        state.regs.rsi = SRC_ADDR
        state.regs.rsp = 0x7FFFFE00
        state.memory.store(0x7FFFFE00, b"\x00" * 8)

        # Dest starts concrete zero; src bytes are not pre-stored so the
        # filler creates per-chunk unconstrained symbols at SRC_ADDR.
        state.memory.store(DST_ADDR, b"\x00" * 32)

        hook_fired = []

        def copy_hook(state):
            buf = state.memory.load(state.regs.rsi, 32)
            state.memory.store(state.regs.rdi, buf)
            hook_fired.append(True)

        proj.hook(0x1000, hook=copy_hook, length=1)
        try:
            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=4)
        finally:
            proj.unhook(0x1000)

        assert hook_fired, "hook never fired"

        all_states = list(mgr.active) + list(mgr.deadended) + list(mgr.unconstrained)
        assert all_states, "expected at least one state after run"
        final = all_states[0]

        # rax should reflect the 8 low bytes of dst (=src after copy),
        # symbolic and constrainable. If the hook's copy preserved symbolic
        # identity through to the Rust VEX load, rax can be constrained to
        # any 64-bit value; if dst was concrete-witnessed during resume,
        # rax is fixed and one of the alternatives below is unsatisfiable.
        rax = final.regs.rax
        assert final.solver.satisfiable(extra_constraints=[rax == 0xDEADBEEFCAFEBABE]), (
            "rax not constrainable to 0xDEADBEEFCAFEBABE — Rust load at "
            "dst returned a concrete witness rather than the hook-copied "
            "symbolic value (angr-ctct)"
        )
        assert final.solver.satisfiable(extra_constraints=[rax == 0x1111222233334444]), (
            "rax not constrainable to 0x1111222233334444 — Rust load at dst was concretized (angr-ctct)"
        )

    def test_filler_materialised_multibyte_symbolic_preserved(self):
        """Init-time ``memory.load(addr, N>1)`` must round-trip every byte.

        Regression for angr-fv81 (sokohashv2 hash routine). When solve.py
        does ``init.memory.load(addr, 8)`` to capture a symbolic input var,
        ``SYMBOL_FILL_UNCONSTRAINED_MEMORY`` materialises an 8-byte symbol
        but ``UltraPage.symbolic_data`` stores ONE entry keyed by the
        region's start offset (the bitmap marks all 8 bytes symbolic, but
        the dict has only the head). The previous angr-ctct fallback
        walked dict keys only, so bytes 1..7 silently became concrete
        zeros on the Rust side — the hash AST collapsed to 4 terms
        (low-byte-only) instead of the expected 15 (all 16-bit halves).

        This test loads an 8-byte symbol at init, then runs a 1-byte ``mov
        al, [rdi+5]`` to verify byte offset 5 is still symbolic after the
        Python↔Rust round trip.
        """

        # 0x1000: mov al, [rdi+5]    ; 8a 47 05
        # 0x1003: ret                ; c3
        shellcode = bytes.fromhex("8a4705c3")
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)

        SYM_ADDR = 0x4000

        # SYMBOL_FILL_UNCONSTRAINED_REGISTERS is raised under the Rust
        # engine (angr-apre) — Rust's RegisterFile always returns zero
        # from vec![0; size] so symbolic-fill cannot be honored. This
        # test exercises memory symbolicity, not register symbolicity,
        # so the memory variant is enough.
        state = proj.factory.blank_state(
            addr=0x1000,
            add_options={angr.options.SYMBOL_FILL_UNCONSTRAINED_MEMORY},
        )
        state.regs.rdi = SYM_ADDR
        state.regs.rsp = 0x7FFFFE00
        state.memory.store(0x7FFFFE00, b"\x00" * 8)

        # User-style symbolic capture: load 8 bytes to materialise a
        # filler-backed symbol covering [SYM_ADDR, SYM_ADDR+8). NO store
        # follows — so changed-history is empty and the fallback path
        # is the only way these bytes make it to Rust.
        _ = state.memory.load(SYM_ADDR, 8)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=4)

        all_states = list(mgr.active) + list(mgr.deadended) + list(mgr.unconstrained)
        assert all_states, "expected at least one state after run"
        final = all_states[0]

        # rax low byte = byte 5 of the symbolic region. If only byte 0
        # made it across, byte 5 would be concrete zero, and the Rust
        # VEX load would store a concrete zero into rax's low byte.
        # solver.satisfiable() ignores extra_constraints on this path,
        # so check symbolicity directly: byte at SYM_ADDR+5 must still
        # carry an AST after the Python↔Rust round trip.
        byte5 = final.memory.load(SYM_ADDR + 5, 1)
        assert byte5.symbolic, (
            f"byte 5 of filler-materialised symbol lost symbolicity "
            f"(got {byte5}) — fallback extracted only the region-head "
            f"byte (angr-fv81)"
        )

    def test_arm_conditional_ldrh_loads_two_bytes(self):
        """ARM ``ldrneh`` (conditional halfword load) must read 2 bytes.

        Regression for angr-21am. ``ldrneh r0, [r1]`` lifts to a VEX
        ``LoadG`` with ``cvt=ILGop_16Uto32``. The Rust interpreter used to
        drop the source width and default a 32-bit-destination widening
        load to a *1-byte* memory read, so only the low byte survived and
        the high byte was silently lost.

        We store 0xABCD at the load address. The Rust engine evaluates the
        guard symbolically (its ARM cc_* thunks read as 0/unconstrained), so
        r0 becomes ``ITE(guard, loaded, alt)``. With the fix ``loaded`` is the
        full 16-bit 0xABCD, so ``r0 == 0xABCD`` is satisfiable; under the old
        1-byte bug ``loaded`` would be 0x00CD and 0xABCD would be unreachable.
        The Python engine (concrete guard) is used as a direct oracle.
        """

        # 0x1000: ldrneh r0, [r1]   ; bytes b0 00 d1 11 (ARMEL, little-endian)
        shellcode = bytes.fromhex("b000d111")
        proj = angr.load_shellcode(shellcode, arch="ARMEL", load_address=0x1000)

        LOAD_ADDR = 0x4000
        VALUE = 0xABCD  # high byte 0xAB != 0 distinguishes 2-byte vs 1-byte load

        def fresh_state():
            st = proj.factory.blank_state(addr=0x1000)
            st.regs.r1 = LOAD_ADDR
            # blank_state zeroes the lazy ARM cc_* flag thunks (cc_op=COPY,
            # cc_dep1=0) -> Z==0 -> the NE condition holds -> the load fires.
            st.memory.store(LOAD_ADDR, VALUE, size=2, endness="Iend_LE")
            return st

        # Python engine oracle.
        py_state = fresh_state()
        py_simgr = proj.factory.simulation_manager(py_state)
        py_simgr.step()
        py_r0 = py_simgr.active[0].solver.eval(py_simgr.active[0].regs.r0)
        assert py_r0 == VALUE, f"Python oracle r0=0x{py_r0:x}, expected 0x{VALUE:x}"

        # Rust engine under test.
        rust_state = fresh_state()
        mgr = RustExplorationManager(proj, [rust_state])
        mgr.run(max_steps=1)
        all_states = list(mgr.active) + list(mgr.deadended) + list(mgr.errored)
        assert all_states, "expected at least one state after run"
        final = all_states[0]
        r0 = final.regs.r0
        # The full 16-bit value must be reachable on the guard-true branch.
        assert final.solver.satisfiable(extra_constraints=[r0 == VALUE]), (
            f"r0 cannot equal 0x{VALUE:x} — the conditional halfword load "
            f"read the wrong number of bytes (1 instead of 2), so the high "
            f"byte 0x{VALUE >> 8:02x} was lost (angr-21am)"
        )

    def test_hook_length_advances_pc_userhook(self, fauxware_project):
        """proj.hook(addr, fn, length=N>0) on a UserHook must skip N bytes.

        Regression for angr-03ej. Confirms the Rust dispatch honors
        the new_pc that UserHook sets via ``state.addr + length`` so
        the hooked N-byte instruction is replaced (not re-executed)
        and the hook does not re-fire in a loop.

        The hook is placed at ``mov rsp, rbp`` (3 bytes) inside main.
        With length=3 control resumes at hook_addr+3; if the dispatch
        somehow stayed at hook_addr the hook would re-fire and the
        test would see an unbounded number of fires.
        """

        proj = fauxware_project
        main_sym = proj.loader.find_symbol("main")
        assert main_sym is not None
        hook_addr = main_sym.rebased_addr + 1  # second instruction in main
        hook_length = 3

        # Sanity-check the disassembly so the test fails clearly if the
        # binary changes shape.
        block = proj.factory.block(hook_addr)
        first_insn_size = block.capstone.insns[0].size
        assert first_insn_size == hook_length, (
            f"Expected a {hook_length}-byte instruction at {hook_addr:#x}, got {first_insn_size}-byte"
        )

        fire_addrs = []

        def my_hook(state):
            fire_addrs.append(state.addr)

        proj.hook(hook_addr, hook=my_hook, length=hook_length)
        try:
            state = proj.factory.entry_state()
            mgr = RustExplorationManager(proj, [state])
            mgr.run(max_steps=300)
        finally:
            proj.unhook(hook_addr)

        assert fire_addrs, "hook never fired"
        # Without the fix, the hook re-executes in an infinite loop
        # because new_pc stays at hook_addr. With the fix, it fires
        # at most once per state path through main (fauxware has a
        # small number of paths).
        assert len(fire_addrs) < 50, (
            f"Hook fired {len(fire_addrs)} times — likely re-executing "
            f"because hook_length was not honored. First addrs: "
            f"{[hex(a) for a in fire_addrs[:5]]}"
        )

    def test_vex_fallback_forks_multi_successors(self, fauxware_project):
        """Multi-successor Python VEX fallback must fork extras instead of dropping them.

        Regression for angr-v8iz: factory.successors(num_inst=99) inside
        _handle_python_vex_fallback can return N>1 successors when the
        fallback block contains a symbolic branch. Previously only
        all_succs[0] was synced back to Rust and the rest were silently
        dropped (warning only) — if the convergent path was the dropped
        one, exploration would spin forever. The handler must call
        _add_forked_state for every extra successor.
        """
        from types import SimpleNamespace

        proj = fauxware_project
        seed_state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [seed_state])

        # Build two distinct successor states off the seed state to play
        # the role of a symbolic-branch fork inside the fallback block.
        succ_a = seed_state.copy()
        succ_a.regs.rax = 0x1111
        succ_b = seed_state.copy()
        succ_b.regs.rax = 0x2222
        all_succs = [succ_a, succ_b]
        succs_obj = SimpleNamespace(all_successors=all_succs)

        event = SimpleNamespace(
            callback_addr=seed_state.addr,
            callback_state_id=None,
            callback_name="test_unsupported_vex_op",
            callback_return_addr=None,
        )

        # Stub helpers so the test exercises only the multi-successor
        # handling logic. _create_state_for_callback returns a fresh
        # SimState; factory.successors returns our pre-built successors.
        mgr._create_state_for_callback = lambda evt: seed_state.copy()  # type: ignore[assignment]
        orig_factory_successors = proj.factory.successors
        proj.factory.successors = lambda *a, **kw: succs_obj  # type: ignore[assignment]

        # Capture resume_after_simprocedure (first successor) and
        # _add_forked_state (every extra successor) calls. The rust_mgr
        # PyO3 object's methods are read-only, so swap the whole handle
        # for a SimpleNamespace recorder. We only need to satisfy the
        # subset of calls _handle_python_vex_fallback issues on the
        # success path.
        resumed_calls = []

        def _record_resume(*args, **kwargs):
            resumed_calls.append((args, kwargs))

        original_rust_mgr = mgr._rust_mgr
        mgr._rust_mgr = SimpleNamespace(
            resume_after_simprocedure=_record_resume,
            resume_after_error=lambda *a, **k: None,
            deadend_pending_callback=lambda *a, **k: None,
        )

        forked_calls = []

        def _record_fork(succ, evt):
            forked_calls.append(succ)

        mgr._add_forked_state = _record_fork  # type: ignore[assignment]

        try:
            mgr._handle_python_vex_fallback(event)
        finally:
            proj.factory.successors = orig_factory_successors  # type: ignore[assignment]
            mgr._rust_mgr = original_rust_mgr

        assert len(resumed_calls) == 1, (
            f"Expected resume_after_simprocedure to fire exactly once for the first successor, got {len(resumed_calls)}"
        )
        assert len(forked_calls) == len(all_succs) - 1, (
            f"Expected {len(all_succs) - 1} forked successors, got {len(forked_calls)} — extras were dropped (the bug)"
        )
        assert forked_calls[0] is succ_b, (
            "The forked successor identity does not match all_succs[1]; wrong state was passed to _add_forked_state."
        )


class TestProxyUnsatCore:
    """``RustSolverProxyPlugin.unsat_core`` parity with ``SimSolver.unsat_core``
    (angr-op0dn.14.2).

    The Rust engine never arms tracking at add time — assumption literals are
    too expensive on the fork path — so the proxy rebuilds a tracked throwaway
    solver from the assumed-constraint IR at query time
    (``SymContext::unsat_core_assumed``). These tests pin that the core is
    *complete* (every contributing constraint is nameable, including ones the
    engine added itself) and that it is reported as the same claripy ASTs
    ``state.solver.constraints`` yields.
    """

    @staticmethod
    def _plugin(mgr, sid):
        from angr.exploration.rust_state_proxy import RustSolverProxyPlugin

        return RustSolverProxyPlugin(mgr, sid)

    @staticmethod
    def _python_core(constraints):
        """Same constraints on a stock Python SimState with tracking on."""
        proj = angr.load_shellcode(b"\x90", arch="amd64")
        state = proj.factory.blank_state(add_options={angr.options.CONSTRAINT_TRACKING_IN_SOLVER})
        state.solver.add(*constraints)
        assert state.solver.satisfiable() is False
        return {str(c) for c in state.solver.unsat_core()}

    def test_multi_constraint_core_matches_python(self):
        """Two mutually contradicting constraints among several: both are
        blamed, the innocent bystander is not — and the Rust core is the same
        set Python's tracking SimSolver reports."""
        import claripy

        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        plugin = self._plugin(mgr, sid)

        x = claripy.BVS("ucore_x", 32)
        y = claripy.BVS("ucore_y", 32)
        cons = [x > 10, y == 3, x < 5]
        plugin.add(*cons)

        core = plugin.unsat_core()
        core_strs = {str(c) for c in core}
        assert core_strs == self._python_core(cons)
        assert str(y == 3) not in core_strs, f"innocent constraint blamed: {core_strs}"

    def test_single_self_contradicting_constraint_is_blamed(self):
        """A lone self-contradicting constraint is blamed on its own.

        No Python cross-check here, because neither shape of a one-constraint
        contradiction survives ``SimSolver.add`` intact: ``And(x == 1, x == 2)``
        constant-folds to a literal ``False`` that never reaches Z3 (empty
        Python core), and ``And(x > 10, x < 5)`` gets split into its two
        conjuncts, so Python blames two constraints where the Rust core — which
        reports entries of ``state.solver.constraints``, and the proxy stores
        the ``And`` whole — blames one. Cross-engine parity is pinned by
        :meth:`test_multi_constraint_core_matches_python`, whose constraints are
        already atomic.
        """
        import claripy

        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        plugin = self._plugin(mgr, sid)

        x = claripy.BVS("ucore_solo", 32)
        cons = [claripy.And(x > 10, x < 5)]
        plugin.add(*cons)

        core = plugin.unsat_core()
        assert [str(c) for c in core] == [str(cons[0])]

    def test_core_entries_are_state_constraints(self):
        """Every core entry is one of ``solver.constraints`` — the core is a
        subset of the exported constraint list, not a re-derived AST."""
        import claripy

        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        plugin = self._plugin(mgr, sid)

        x = claripy.BVS("ucore_sub", 32)
        plugin.add(x == 0, x == 1)

        exported = {str(c) for c in plugin.constraints}
        core_strs = {str(c) for c in plugin.unsat_core()}
        assert core_strs, "expected a non-empty core on an UNSAT state"
        assert core_strs <= exported, f"core {core_strs} not a subset of constraints {exported}"

    def test_core_empty_when_satisfiable(self):
        """A SAT state has no core — matches SimSolver (Z3 produces none)."""
        import claripy

        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        plugin = self._plugin(mgr, sid)

        x = claripy.BVS("ucore_sat", 32)
        plugin.add(x > 10, x < 20)
        assert plugin.unsat_core() == []

    def test_extra_constraints_assumed_but_never_blamed(self):
        """``extra_constraints`` can *cause* the UNSAT but are never reported —
        claripy's tracking solver leaves them untracked too."""
        import claripy

        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        plugin = self._plugin(mgr, sid)

        x = claripy.BVS("ucore_extra", 32)
        plugin.add(x == 7)
        assert plugin.unsat_core() == [], "state alone is SAT"

        core = plugin.unsat_core(extra_constraints=[x == 8])
        core_strs = {str(c) for c in core}
        assert core_strs == {str(x == 7)}, f"expected only the state constraint, got {core_strs}"

    def test_core_empty_without_option_on_bound_state(self):
        """Without CONSTRAINT_TRACKING_IN_SOLVER on the bound state the proxy
        keeps returning [] rather than raising (SimSolver raises) — SimProcedures
        that opportunistically peek at the core must not crash.
        ``invariant-rust-solver-proxy-required-surface``."""
        import types

        import claripy

        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("active")
        plugin = self._plugin(mgr, sid)

        x = claripy.BVS("ucore_noopt", 32)
        plugin.add(x == 0, x == 1)
        plugin.set_state(types.SimpleNamespace(options=set()))
        assert plugin.unsat_core() == []

        plugin.set_state(types.SimpleNamespace(options={"CONSTRAINT_TRACKING_IN_SOLVER"}))
        assert len(plugin.unsat_core()) >= 1
