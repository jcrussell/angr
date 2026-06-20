"""Tests for RustExplorationManager.

This module tests the Rust-native exploration manager for symbolic execution.
"""

from __future__ import annotations

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


class TestExplorationStrategy:
    """Tests for DFS/BFS exploration strategy."""

    def test_set_exploration_strategy_dfs(self, fauxware_project):
        """Test setting DFS strategy finds the same result."""

        find_addr = 0x4006ED
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.set_exploration_strategy("dfs")
        mgr.explore(find=find_addr)
        assert len(mgr.found) > 0, "DFS should find at least one state"

    def test_set_exploration_strategy_bfs(self, fauxware_project):
        """Test BFS strategy (default) works."""

        find_addr = 0x4006ED
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.set_exploration_strategy("bfs")
        mgr.explore(find=find_addr)
        assert len(mgr.found) > 0, "BFS should find at least one state"

    def test_set_exploration_strategy_invalid(self, fauxware_project):
        """Test invalid strategy raises ValueError."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        with pytest.raises(ValueError, match="Unknown exploration strategy"):
            mgr.set_exploration_strategy("random")

    def test_dfs_technique_auto_detection(self, fauxware_project):
        """Test that angr DFS technique is auto-detected."""

        find_addr = 0x4006ED
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.use_technique(angr.exploration_techniques.DFS())
        mgr.explore(find=find_addr)
        assert len(mgr.found) > 0, "DFS technique should find at least one state"

    def test_init_kwarg_strategy_dfs(self, fauxware_project):
        """Constructing with exploration_strategy='dfs' picks LIFO state selection.

        Equivalent to a post-init set_exploration_strategy('dfs') call but
        applied during __init__ so the first step already sees the chosen
        order. Regression for the angr-3ms1 Flavor-1 wiring step.
        """

        find_addr = 0x4006ED
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state], exploration_strategy="dfs")
        mgr.explore(find=find_addr)
        assert len(mgr.found) > 0, "DFS-at-init should find at least one state"

    def test_init_kwarg_strategy_bfs_default(self, fauxware_project):
        """Default exploration_strategy is 'bfs' and explicit 'bfs' both work."""

        find_addr = 0x4006ED
        # Default — no kwarg.
        state1 = fauxware_project.factory.entry_state()
        mgr1 = RustExplorationManager(fauxware_project, [state1])
        mgr1.explore(find=find_addr)
        assert len(mgr1.found) > 0

        # Explicit 'bfs' — same as default.
        state2 = fauxware_project.factory.entry_state()
        mgr2 = RustExplorationManager(fauxware_project, [state2], exploration_strategy="bfs")
        mgr2.explore(find=find_addr)
        assert len(mgr2.found) > 0

    def test_init_kwarg_strategy_invalid(self, fauxware_project):
        """Invalid exploration_strategy at construction time raises ValueError."""

        state = fauxware_project.factory.entry_state()
        with pytest.raises(ValueError, match="Unknown exploration strategy"):
            RustExplorationManager(fauxware_project, [state], exploration_strategy="random")

    def test_init_kwarg_use_shared_lineage_solver_default_off(self, fauxware_project):
        """Default ``use_shared_lineage_solver=False`` keeps the opt-in off and
        exploration matches an explicit ``False`` (angr-3ms1 step 1b).

        The flag is inert in this slice — slice-1c will be the first
        consumer at fork time — so this test asserts the wiring is in
        place (default value stored, explicit False round-trips) without
        depending on any observable engine behavior change.
        """

        state_default = fauxware_project.factory.entry_state()
        mgr_default = RustExplorationManager(fauxware_project, [state_default])
        assert mgr_default._use_shared_lineage_solver is False

        state_explicit = fauxware_project.factory.entry_state()
        mgr_explicit = RustExplorationManager(
            fauxware_project,
            [state_explicit],
            use_shared_lineage_solver=False,
        )
        assert mgr_explicit._use_shared_lineage_solver is False

        # The explore-to-find golden path still works when the kwarg is
        # off — guards against an accidental gate flip in the default
        # path.
        find_addr = 0x4006ED
        mgr_default.explore(find=find_addr)
        assert len(mgr_default.found) > 0

    def test_init_kwarg_use_shared_lineage_solver_on(self, fauxware_project):
        """``use_shared_lineage_solver=True`` is inert today but must not
        break exploration (angr-3ms1 step 1b).

        Slice-1c will be the first consumer of the flag at fork time;
        until then, an opt-in run must explore identically to a
        default-off run. This guards against a stray gate that fires on
        the read alone — easy to introduce by mistake while wiring
        slice-1c later.
        """

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(
            fauxware_project,
            [state],
            use_shared_lineage_solver=True,
        )
        assert mgr._use_shared_lineage_solver is True

        find_addr = 0x4006ED
        mgr.explore(find=find_addr)
        assert len(mgr.found) > 0, "explore() must succeed with the opt-in enabled (flag is inert today)"

    def test_deep_loop_recipe_dfs_plus_length_limiter(self, fauxware_project):
        """Recipe for deep-input-loop binaries (angr-smxp / angr-xel4 spike):
        ``exploration_strategy='dfs'`` combined with the ``LengthLimiter``
        technique. Both knobs route through native Rust setters
        (``set_state_selection_lifo`` + ``register_length_limiter``) and must
        coexist without one overriding the other.

        This smoke-tests the documented recipe in
        ``docs/advanced-topics/rust_engine.rst`` ("Deep-input-loop binaries
        (grub-class)" subsection). fauxware is not a deep-loop binary, so we
        cannot assert the depth-limited regression that motivates the recipe
        — only that the combination constructs, runs, and respects the
        ``max_length`` bound on the active stash at the end of exploration.
        """

        find_addr = 0x4006ED
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(
            fauxware_project,
            [state],
            exploration_strategy="dfs",
            max_active_states=64,
        )
        max_length = 32
        mgr.use_technique(
            angr.exploration_techniques.LengthLimiter(
                max_length=max_length,
                drop=True,
            )
        )

        mgr.explore(find=find_addr, max_steps=200)
        assert len(mgr.found) > 0, "DFS + LengthLimiter recipe must still reach fauxware's find"

        # No state in the active stash should exceed max_length blocks.
        # Use the lightweight proxy iterator so we don't pay the cost of
        # a full SimState export to read history length.
        for proxy in mgr.active_proxies():
            depth = len(proxy.history.bbl_addrs)
            assert depth <= max_length, (
                f"LengthLimiter should drop states past {max_length} blocks, observed depth={depth}"
            )


class TestVexOperationCoverage:
    """Systematic tests for BV operations through the Rust solver.

    Each test constrains a symbolic variable using a specific BV operation,
    solves it, and verifies the result matches the expected concrete value.
    This ensures the Rust Z3 bridge correctly handles each VEX/claripy op.
    """

    @classmethod
    def setup_class(cls):
        from angr.exploration.rust_manager import _setup_shared_z3_context

        _setup_shared_z3_context()

    def _make_ctx(self):
        from angr.rustylib.vex_engine import RustSolverContext

        return RustSolverContext()

    # --- Arithmetic ---

    def test_add_concrete(self):
        """x + 7 == 49 => x == 42."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x + 7 == 49)
        assert ctx.eval(x) == 42

    def test_add_symbolic(self):
        """x + y == 100, y == 30 => x == 70."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast(x + y == 100)
        ctx.add_constraint_ast(y == 30)
        assert ctx.eval(x) == 70

    def test_add_overflow_wraps(self):
        """0xFFFFFFFF + 1 wraps to 0 in 32-bit."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x + 1 == 0)
        assert ctx.eval(x) == 0xFFFFFFFF

    def test_sub_concrete(self):
        """x - 8 == 34 => x == 42."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x - 8 == 34)
        assert ctx.eval(x) == 42

    def test_sub_symbolic(self):
        """x - y == 20, y == 10 => x == 30."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast(x - y == 20)
        ctx.add_constraint_ast(y == 10)
        assert ctx.eval(x) == 30

    def test_sub_underflow_wraps(self):
        """0 - 1 wraps to 0xFFFFFFFF in 32-bit."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x - 1 == 0xFFFFFFFF)
        assert ctx.eval(x) == 0

    def test_mul_concrete(self):
        """x * 6 == 42 => x == 7."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x * 6 == 42)
        # Multiple solutions possible (mod 2^32), but 7 must be one
        val = ctx.eval(x)
        assert (val * 6) & 0xFFFFFFFF == 42

    def test_mul_symbolic(self):
        """x * y == 56, x == 7 => y == 8."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast(x * y == 56)
        ctx.add_constraint_ast(x == 7)
        assert ctx.eval(y) == 8

    # --- Bitwise ---

    def test_and_mask(self):
        """x & 0xFF == 0x42 constrains low byte."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast((x & 0xFF) == 0x42)
        assert (ctx.eval(x) & 0xFF) == 0x42

    def test_and_symbolic(self):
        """x & y == 0x10, x == 0x1F, so y & 0x1F == 0x10."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast((x & y) == 0x10)
        ctx.add_constraint_ast(x == 0x1F)
        val_y = ctx.eval(y)
        assert (0x1F & val_y) == 0x10

    def test_or_bits(self):
        """x | 0xF0 == 0xFF => x & 0x0F must be 0x0F."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast((x | 0xF0) == 0xFF)
        val = ctx.eval(x)
        assert (val | 0xF0) == 0xFF

    def test_or_symbolic(self):
        """x | y == 0xFF, x == 0x0F => y must set high nibble."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        y = claripy.BVS("y", 8)
        ctx.add_constraint_ast((x | y) == 0xFF)
        ctx.add_constraint_ast(x == 0x0F)
        val_y = ctx.eval(y)
        assert (0x0F | val_y) == 0xFF

    def test_xor_concrete(self):
        """x ^ 0xAA == 0x55 => x == 0xFF."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast((x ^ 0xAA) == 0x55)
        assert ctx.eval(x) == 0xFF

    def test_xor_self_is_zero(self):
        """x ^ x == 0 for any x."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast((x ^ x) == 0)
        assert ctx.satisfiable()

    def test_xor_symbolic_inverse(self):
        """x ^ y == 0xFFFFFFFF => y == ~x."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast((x ^ y) == 0xFFFFFFFF)
        ctx.add_constraint_ast(x == 0xDEADBEEF)
        assert ctx.eval(y) == (0xDEADBEEF ^ 0xFFFFFFFF)

    def test_not_bitwise(self):
        """~x == 0x00 => x == 0xFF (8-bit)."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(~x == 0x00)
        assert ctx.eval(x) == 0xFF

    # --- Shifts ---

    def test_shl_concrete(self):
        """x << 4 == 0x120 => low nibble lost, x == 0x12."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast((x << 4) == 0x120)
        val = ctx.eval(x)
        assert (val << 4) & 0xFFFFFFFF == 0x120

    def test_shl_symbolic_amount(self):
        """x << n == 0x80, x == 1 => n == 7."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        n = claripy.BVS("n", 32)
        ctx.add_constraint_ast((x << n) == 0x80)
        ctx.add_constraint_ast(x == 1)
        ctx.add_constraint_ast(n < 32)
        assert ctx.eval(n) == 7

    def test_lshr_concrete(self):
        """LShR(x, 8) == 0x12 => x >> 8 == 0x12."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(claripy.LShR(x, 8) == 0x12)
        val = ctx.eval(x)
        assert (val >> 8) == 0x12

    def test_lshr_vs_arithmetic(self):
        """LShR is logical (zero-fill), not arithmetic."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0x80000000)
        ctx.add_constraint_ast(claripy.LShR(x, 1) == 0x40000000)
        assert ctx.satisfiable()

    def test_arithmetic_shr(self):
        """Arithmetic shift right sign-extends."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0x80000000)
        # x >> 1 (arithmetic) should be 0xC0000000
        ctx.add_constraint_ast((x >> 1) == 0xC0000000)
        assert ctx.satisfiable()

    # --- Extract / Concat ---

    def test_extract_low_byte(self):
        """Extract(7, 0, x) gets the least significant byte."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0xDEADBEEF)
        ctx.add_constraint_ast(claripy.Extract(7, 0, x) == 0xEF)
        assert ctx.satisfiable()

    def test_extract_high_byte(self):
        """Extract(31, 24, x) gets the most significant byte."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0xDEADBEEF)
        ctx.add_constraint_ast(claripy.Extract(31, 24, x) == 0xDE)
        assert ctx.satisfiable()

    def test_extract_middle_word(self):
        """Extract(23, 8, x) gets the middle two bytes."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(claripy.Extract(23, 8, x) == 0xBEEF)
        val = ctx.eval(x)
        assert ((val >> 8) & 0xFFFF) == 0xBEEF

    def test_extract_single_bit(self):
        """Extract(0, 0, x) gets bit 0."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(claripy.Extract(0, 0, x) == 1)
        val = ctx.eval(x)
        assert (val & 1) == 1

    def test_concat_two_bytes(self):
        """Concat(a, b) forms a 16-bit value."""
        import claripy

        ctx = self._make_ctx()
        a = claripy.BVS("a", 8)
        b = claripy.BVS("b", 8)
        ctx.add_constraint_ast(claripy.Concat(a, b) == 0xCAFE)
        assert ctx.eval(a) == 0xCA
        assert ctx.eval(b) == 0xFE

    def test_concat_four_bytes(self):
        """Concat(a, b, c, d) forms a 32-bit value."""
        import claripy

        ctx = self._make_ctx()
        a = claripy.BVS("a", 8)
        b = claripy.BVS("b", 8)
        c = claripy.BVS("c", 8)
        d = claripy.BVS("d", 8)
        ctx.add_constraint_ast(claripy.Concat(a, b, c, d) == 0xDEADBEEF)
        assert ctx.eval(a) == 0xDE
        assert ctx.eval(b) == 0xAD
        assert ctx.eval(c) == 0xBE
        assert ctx.eval(d) == 0xEF

    def test_concat_then_extract_roundtrip(self):
        """Extract undoes Concat for matching bit ranges."""
        import claripy

        ctx = self._make_ctx()
        a = claripy.BVS("a", 8)
        b = claripy.BVS("b", 8)
        ctx.add_constraint_ast(a == 0x12)
        ctx.add_constraint_ast(b == 0x34)
        ab = claripy.Concat(a, b)
        ctx.add_constraint_ast(claripy.Extract(15, 8, ab) == 0x12)
        ctx.add_constraint_ast(claripy.Extract(7, 0, ab) == 0x34)
        assert ctx.satisfiable()

    # --- Extension ---

    def test_zeroext_8_to_32(self):
        """ZeroExt(24, x8) where x8 == 0xFF gives 0x000000FF."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(x == 0xFF)
        extended = claripy.ZeroExt(24, x)
        ctx.add_constraint_ast(extended == 0x000000FF)
        assert ctx.satisfiable()

    def test_zeroext_preserves_value(self):
        """ZeroExt should not change the numeric value of a positive number."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 16)
        ctx.add_constraint_ast(x == 0x1234)
        extended = claripy.ZeroExt(16, x)
        ctx.add_constraint_ast(extended == 0x00001234)
        assert ctx.satisfiable()

    def test_signext_positive(self):
        """SignExt of positive value (MSB=0) is same as ZeroExt."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(x == 0x7F)  # positive in signed 8-bit
        extended = claripy.SignExt(24, x)
        ctx.add_constraint_ast(extended == 0x0000007F)
        assert ctx.satisfiable()

    def test_signext_negative(self):
        """SignExt of negative value (MSB=1) fills with 1s."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(x == 0x80)  # -128 in signed 8-bit
        extended = claripy.SignExt(24, x)
        ctx.add_constraint_ast(extended == 0xFFFFFF80)
        assert ctx.satisfiable()

    def test_signext_ff(self):
        """SignExt(24, 0xFF) == 0xFFFFFFFF."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(x == 0xFF)
        extended = claripy.SignExt(24, x)
        ctx.add_constraint_ast(extended == 0xFFFFFFFF)
        assert ctx.satisfiable()

    # --- Reverse (byte swap) ---

    def test_reverse_16bit(self):
        """Reverse(0x1234) == 0x3412."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 16)
        ctx.add_constraint_ast(x == 0x1234)
        ctx.add_constraint_ast(claripy.Reverse(x) == 0x3412)
        assert ctx.satisfiable()

    def test_reverse_32bit(self):
        """Reverse(0xDEADBEEF) == 0xEFBEADDE."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0xDEADBEEF)
        ctx.add_constraint_ast(claripy.Reverse(x) == 0xEFBEADDE)
        assert ctx.satisfiable()

    def test_reverse_involution(self):
        """Reverse(Reverse(x)) == x."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0xCAFEBABE)
        ctx.add_constraint_ast(claripy.Reverse(claripy.Reverse(x)) == x)
        assert ctx.satisfiable()

    def test_reverse_64bit(self):
        """Reverse of a 64-bit value."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 64)
        ctx.add_constraint_ast(x == 0x0102030405060708)
        ctx.add_constraint_ast(claripy.Reverse(x) == 0x0807060504030201)
        assert ctx.satisfiable()

    # --- Comparisons ---

    def test_uge(self):
        """Unsigned greater-or-equal."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(claripy.UGE(x, 0xFE))
        results = ctx.eval_upto(x, 10)
        assert set(results) == {0xFE, 0xFF}

    def test_ule(self):
        """Unsigned less-or-equal."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(claripy.ULE(x, 2))
        results = ctx.eval_upto(x, 10)
        assert set(results) == {0, 1, 2}

    def test_sgt(self):
        """Signed greater-than constrains to positive range."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(claripy.SGT(x, claripy.BVV(0x7C, 8)))  # > 124 signed
        ctx.add_constraint_ast(claripy.SLT(x, claripy.BVV(0x7F, 8)))  # < 127 signed
        results = ctx.eval_upto(x, 10)
        assert set(results) == {0x7D, 0x7E}  # 125, 126

    def test_sle(self):
        """Signed less-or-equal with negative bound."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        # x <=s -126 (0x82) means x is in {0x80, 0x81, 0x82} = {-128, -127, -126}
        ctx.add_constraint_ast(claripy.SLE(x, claripy.BVV(0x82, 8)))
        ctx.add_constraint_ast(claripy.SGE(x, claripy.BVV(0x80, 8)))
        results = ctx.eval_upto(x, 10)
        assert set(results) == {0x80, 0x81, 0x82}

    # --- If-then-else ---

    def test_ite_true_branch(self):
        """If(True, a, b) == a."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 1)
        result = claripy.If(x == 1, claripy.BVV(0xAA, 8), claripy.BVV(0xBB, 8))
        ctx.add_constraint_ast(result == 0xAA)
        assert ctx.satisfiable()

    def test_ite_false_branch(self):
        """If(False, a, b) == b."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0)
        result = claripy.If(x == 1, claripy.BVV(0xAA, 8), claripy.BVV(0xBB, 8))
        ctx.add_constraint_ast(result == 0xBB)
        assert ctx.satisfiable()

    def test_ite_symbolic_condition(self):
        """ITE with symbolic condition and constrained result."""
        import claripy

        ctx = self._make_ctx()
        cond = claripy.BVS("c", 8)
        a = claripy.BVS("a", 32)
        b = claripy.BVS("b", 32)
        ctx.add_constraint_ast(a == 100)
        ctx.add_constraint_ast(b == 200)
        result = claripy.If(cond == 1, a, b)
        ctx.add_constraint_ast(result == 100)
        assert ctx.eval(cond) == 1

    # --- Combined / complex ---

    def test_add_then_extract(self):
        """(x + y) constrained, then extract a byte of the sum."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast(x == 0x00001000)
        ctx.add_constraint_ast(y == 0x00000234)
        s = x + y
        ctx.add_constraint_ast(claripy.Extract(15, 0, s) == 0x1234)
        assert ctx.satisfiable()

    def test_xor_shl_combo(self):
        """(x ^ key) << 8 == target tests combined ops."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        key = 0x55
        target = 0x0000AA00
        ctx.add_constraint_ast(((x ^ key) << 8) == target)
        val = ctx.eval(x)
        assert (((val ^ key) << 8) & 0xFFFFFFFF) == target

    def test_signext_then_add(self):
        """SignExt then add: common in sign-extended address calculations."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(x == 0xFE)  # -2 in signed 8-bit
        extended = claripy.SignExt(24, x)  # 0xFFFFFFFE
        result = extended + 0x100
        ctx.add_constraint_ast(result == 0x000000FE)
        assert ctx.satisfiable()

    def test_concat_reverse_extract(self):
        """Concat, Reverse, Extract pipeline (common in memory operations)."""
        import claripy

        ctx = self._make_ctx()
        a = claripy.BVS("a", 8)
        b = claripy.BVS("b", 8)
        ctx.add_constraint_ast(a == 0x12)
        ctx.add_constraint_ast(b == 0x34)
        word = claripy.Concat(a, b)  # 0x1234
        swapped = claripy.Reverse(word)  # 0x3412
        lo = claripy.Extract(7, 0, swapped)  # 0x12
        ctx.add_constraint_ast(lo == 0x12)
        assert ctx.satisfiable()

    def test_mul_and_mask(self):
        """x * 3 & 0xFF == result, checking low byte of multiplication."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0x55)
        product = x * 3  # 0xFF
        ctx.add_constraint_ast((product & 0xFF) == 0xFF)
        assert ctx.satisfiable()

    # --- Width variations ---

    def test_add_8bit(self):
        """8-bit addition with overflow."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(x + 1 == 0)
        assert ctx.eval(x) == 0xFF

    def test_add_64bit(self):
        """64-bit addition."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 64)
        ctx.add_constraint_ast(x + 1 == 0x100000000)
        assert ctx.eval(x) == 0xFFFFFFFF

    def test_xor_64bit(self):
        """64-bit XOR."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 64)
        ctx.add_constraint_ast(x ^ 0xDEADBEEFCAFEBABE == 0)
        assert ctx.eval(x) == 0xDEADBEEFCAFEBABE


class TestVexGetMSBsInstruction:
    """Instruction-level coverage for Iop_GetMSBs8x16 (x86 PMOVMSKB).

    The VGetMSBs op (commit af98545ea) shipped with Rust unit tests but no
    Python-boundary regression. glibc's SSE strlen/memchr lift `pmovmskb`
    to Iop_GetMSBs8x16; before the op existed, any real-glibc binary errored
    at startup ("unmapped VEX opcode: Iop_GetMSBs8x16"). This drives the op
    end-to-end through the Rust interpreter on a binary-free shellcode blob:
    a single `pmovmskb eax, xmm0` that reduces the 16 bytes of xmm0 to a
    16-bit mask whose bit i is the MSB (bit 7) of input byte i.
    """

    def _run_pmovmskb(self, xmm0_value):
        import claripy

        import angr
        import angr.sim_options as o

        #   66 0f d7 c0   pmovmskb eax, xmm0
        #   c3            ret
        shellcode = bytes.fromhex("660fd7c0") + b"\xc3"
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x401000)
        state = proj.factory.blank_state(
            addr=0x401000,
            add_options={
                o.ZERO_FILL_UNCONSTRAINED_MEMORY,
                o.ZERO_FILL_UNCONSTRAINED_REGISTERS,
            },
        )
        state.regs.xmm0 = claripy.BVV(xmm0_value, 128)
        # Clean return target so the ret deadends predictably.
        state.regs.rsp = 0x7FFF_0000
        state.memory.store(0x7FFF_0000, b"\x00" * 8)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=10)

        all_states = list(mgr.found) + list(mgr.active) + list(mgr.deadended) + list(mgr.unconstrained)
        assert all_states, "expected at least one state after run"
        s = all_states[0]
        # PMOVMSKB writes the 16-bit mask into eax (zero-extended to rax).
        return s.solver.eval(s.regs.rax) & 0xFFFF

    @staticmethod
    def _expected_mask(value):
        mask = 0
        for i in range(16):
            if (value >> (8 * i + 7)) & 1:
                mask |= 1 << i
        return mask

    def test_pmovmskb_recognizable_pattern(self):
        """Hand-picked mask 0xACE1: byte i has MSB set iff bit i of the mask."""
        mask = 0xACE1
        value = 0
        for i in range(16):
            byte = 0x80 if (mask >> i) & 1 else 0x01
            value |= byte << (8 * i)
        assert self._run_pmovmskb(value) == mask

    def test_pmovmskb_all_high(self):
        """All 16 bytes 0x80 → mask 0xFFFF (all MSBs set)."""
        value = int.from_bytes(b"\x80" * 16, "little")
        assert self._run_pmovmskb(value) == 0xFFFF

    def test_pmovmskb_all_low(self):
        """All 16 bytes 0x7F → mask 0x0000 (no MSBs set)."""
        value = int.from_bytes(b"\x7f" * 16, "little")
        assert self._run_pmovmskb(value) == 0x0000

    def test_pmovmskb_arbitrary_value_matches_oracle(self):
        """Arbitrary 128-bit value matches the per-byte-MSB oracle."""
        value = 0x8001_7FFE_C3D2_E1F0_0F1E_2D3C_4B5A_6978
        assert self._run_pmovmskb(value) == self._expected_mask(value)


class TestVexBitCountInstructions:
    """Instruction-level coverage for scalar Iop_Clz64 / Iop_Ctz64 (lzcnt/tzcnt).

    The sound symbolic-bitcount export (commit e3f46fa9a, acoq) shipped with
    two Rust unit tests but no Python-boundary regression. Before that fix,
    rustbv_to_claripy_memo exported a symbolic Clz/Ctz as a freshly-minted
    *unconstrained* claripy BVS — constraint relationship lost — so a
    Python-side eval could return a Rust-infeasible value. build_sound_bitcount
    now emits a nested-If encoding tied to the operand AST.

    x86 `lzcnt`/`tzcnt` lift to Iop_Clz64/Iop_Ctz64 (verified by lifting:
    lzcnt edx, ecx -> ITE(ecx==0, 32, Clz64(ZeroExt(ecx) << 32)); tzcnt
    ebx, ecx -> ITE(ecx==0, 32, Ctz64(ZeroExt(ecx)))). A binary-free shellcode
    blob drives both ops end-to-end through the Rust interpreter; keeping the
    operand symbolic forces the sound-export path the Cargo tests cover only
    in isolation.
    """

    #   f3 0f bd d1   lzcnt edx, ecx
    #   f3 0f bc d9   tzcnt ebx, ecx
    #   c3            ret
    _SHELLCODE = bytes.fromhex("f30fbdd1" + "f30fbcd9") + b"\xc3"

    def _run(self, ecx_value):
        import angr
        import angr.sim_options as o

        proj = angr.load_shellcode(self._SHELLCODE, arch="AMD64", load_address=0x401000)
        state = proj.factory.blank_state(
            addr=0x401000,
            add_options={
                o.ZERO_FILL_UNCONSTRAINED_MEMORY,
                o.ZERO_FILL_UNCONSTRAINED_REGISTERS,
            },
        )
        state.regs.ecx = ecx_value
        # Clean return target so the ret deadends predictably.
        state.regs.rsp = 0x7FFF_0000
        state.memory.store(0x7FFF_0000, b"\x00" * 8)

        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=10)

        all_states = list(mgr.found) + list(mgr.active) + list(mgr.deadended) + list(mgr.unconstrained)
        assert all_states, "expected at least one state after run"
        return all_states[0]

    @staticmethod
    def _clz32(v):
        v &= 0xFFFFFFFF
        return 32 if v == 0 else 32 - v.bit_length()

    @staticmethod
    def _ctz32(v):
        v &= 0xFFFFFFFF
        return 32 if v == 0 else (v & -v).bit_length() - 1

    def test_lzcnt_tzcnt_concrete(self):
        """Concrete operand: lzcnt (edx) / tzcnt (ebx) match the bit oracle."""
        import claripy

        value = 0x00F0_0F00  # clz=8, ctz=8
        s = self._run(claripy.BVV(value, 32))
        assert s.solver.eval(s.regs.edx) == self._clz32(value)
        assert s.solver.eval(s.regs.ebx) == self._ctz32(value)

    def test_lzcnt_tzcnt_extremes(self):
        """Top-bit-set -> clz 0 / ctz 31; bit-0-set -> clz 31 / ctz 0."""
        import claripy

        s_hi = self._run(claripy.BVV(0x8000_0000, 32))
        assert s_hi.solver.eval(s_hi.regs.edx) == 0
        assert s_hi.solver.eval(s_hi.regs.ebx) == 31

        s_lo = self._run(claripy.BVV(0x0000_0001, 32))
        assert s_lo.solver.eval(s_lo.regs.edx) == 31
        assert s_lo.solver.eval(s_lo.regs.ebx) == 0

    # --- Symbolic soundness (executable spec for the open bug angr-4ju9e) ---
    #
    # These encode the CORRECT Python-engine behaviour: a symbolic operand must
    # leave lzcnt/tzcnt symbolic so constraining the result back-solves a
    # consistent operand. Today the Rust engine concretizes the symbolic
    # Iop_Clz64/Iop_Ctz64 result to 0 *inside the interpreter* (verified:
    # rdx/rbx export as concrete <BV...0x0>, mgr.stats['rust_export_sound_clz']
    # stays 0 — acoq's build_sound_bitcount export, commit e3f46fa9a, never
    # runs for this lifting). So eval(ecx) ignores the constraint and these
    # fail. xfail (non-strict) until angr-4ju9e lands; drop the marker then.
    _XFAIL_4JU9E = pytest.mark.xfail(
        reason="angr-4ju9e: Rust concretizes symbolic lzcnt/tzcnt result to 0",
        strict=False,
    )

    @_XFAIL_4JU9E
    @pytest.mark.parametrize("clz_target", [0, 4, 8, 16, 23, 31])
    def test_lzcnt_symbolic_export_is_sound(self, clz_target):
        """Symbolic operand: constraining the lzcnt result back-solves a value
        whose true clz matches. A fresh unconstrained BVS export (pre-acoq) or
        an interpreter-level concretization (angr-4ju9e) would let eval(ecx)
        return a Rust-infeasible value with the wrong clz.
        """
        import claripy

        x = claripy.BVS("ecx", 32)
        s = self._run(x)
        s.add_constraints(s.regs.edx == clz_target)
        assert s.satisfiable(), f"clz=={clz_target} must be satisfiable"
        value = s.solver.eval(x) & 0xFFFFFFFF
        assert self._clz32(value) == clz_target

    @_XFAIL_4JU9E
    @pytest.mark.parametrize("ctz_target", [0, 4, 8, 16, 31])
    def test_tzcnt_symbolic_export_is_sound(self, ctz_target):
        """Symbolic operand: constraining the tzcnt result back-solves a value
        whose true ctz matches (sound Ctz export, commit e3f46fa9a; gated on
        the interpreter-concretization fix angr-4ju9e).
        """
        import claripy

        x = claripy.BVS("ecx", 32)
        s = self._run(x)
        s.add_constraints(s.regs.ebx == ctz_target)
        assert s.satisfiable(), f"ctz=={ctz_target} must be satisfiable"
        value = s.solver.eval(x) & 0xFFFFFFFF
        assert self._ctz32(value) == ctz_target
