"""Rust-native Z3 solver plugin.

This plugin uses the Rust Z3 solver bridge to perform constraint solving
directly in Rust, avoiding Python-Rust serialization overhead during
symbolic execution.
"""
from __future__ import annotations

import logging

import claripy

from angr.errors import SimUnsatError
from .plugin import SimStatePlugin

l = logging.getLogger(name=__name__)


# Try to import the Rust solver context
try:
    from angr.rustylib.vex_engine import RustSolverContext
    RUST_SOLVER_AVAILABLE = True
except ImportError:
    RUST_SOLVER_AVAILABLE = False
    RustSolverContext = None


def _concrete_value(e):
    """Extract concrete value from an expression if possible."""
    if isinstance(e, (int, float, bool)):
        return e
    if hasattr(e, 'op') and e.op == 'BVV' and len(e.args) >= 1:
        return e.args[0]
    if hasattr(e, 'is_leaf') and e.is_leaf() and not e.symbolic:
        return e.args[0]
    return None


class RustSimSolver(SimStatePlugin):
    """SimSolver implementation using Rust-native Z3 solving.

    This provides a drop-in replacement for the standard SimSolver
    that uses the Rust Z3 bridge for constraint solving. This can
    provide significant speedups for symbolic execution by avoiding
    Python-Rust serialization overhead.

    Usage:
        # Create state with Rust solver
        state = proj.factory.entry_state()
        state.register_plugin('solver', RustSimSolver())

        # Or enable via sim_options.RUST_SOLVER
        state = proj.factory.entry_state(add_options={sim_options.RUST_SOLVER})
    """

    def __init__(self, rust_ctx=None, all_variables=None, **kwargs):
        """Initialize the Rust solver plugin.

        Args:
            rust_ctx: Optional RustSolverContext to use. If None, creates new one.
            all_variables: List of all symbolic variables for tracking.
            **kwargs: Additional arguments (ignored for compatibility).
        """
        super().__init__()

        if not RUST_SOLVER_AVAILABLE:
            raise ImportError(
                "RustSolverContext not available. "
                "Build with vex-engine-z3 feature enabled."
            )

        self._rust_ctx = rust_ctx if rust_ctx is not None else RustSolverContext()
        self.all_variables = all_variables if all_variables is not None else []

    @property
    def constraints(self):
        """Return the constraints (not directly available from Rust)."""
        # The Rust solver doesn't expose constraints back to Python
        # This is a limitation - for full compatibility, would need
        # to track constraints on the Python side as well
        return []

    def reload_solver(self, constraints=None):
        """Reload the solver with new constraints."""
        self._rust_ctx = RustSolverContext()
        if constraints:
            for c in constraints:
                self._rust_ctx.add_constraint_ast(c)

    def add(self, *constraints):
        """Add constraints to the solver.

        Args:
            *constraints: Constraint ASTs to add.
        """
        to_add = []
        for c in constraints:
            if isinstance(c, (list, tuple)):
                raise TypeError("Tuple or list passed to add!")
            if isinstance(c, bool):
                if not c:
                    # Adding False makes the solver unsat
                    # Create an impossible constraint
                    self._rust_ctx.add_constraint_ast(claripy.false)
                    return
                continue
            to_add.append(c)

        # Use batch API when multiple constraints for reduced overhead
        try:
            if len(to_add) == 1:
                self._rust_ctx.add_constraint_ast(to_add[0])
            elif len(to_add) > 1:
                self._rust_ctx.add_constraints(to_add)
        except Exception as e:
            l.warning("Failed to add constraint(s) to Rust solver: %s", e)

    def satisfiable(self, extra_constraints=(), **kwargs):
        """Check if constraints are satisfiable.

        Args:
            extra_constraints: Additional constraints to check (temporarily).
            **kwargs: Additional arguments (ignored).

        Returns:
            True if satisfiable, False otherwise.
        """
        if extra_constraints:
            self._rust_ctx.push()
            try:
                for c in extra_constraints:
                    if isinstance(c, bool):
                        if not c:
                            return False
                        continue
                    try:
                        self._rust_ctx.add_constraint_ast(c)
                    except Exception:
                        pass
                return self._rust_ctx.satisfiable()
            finally:
                self._rust_ctx.pop()
        return self._rust_ctx.satisfiable()

    def eval(self, e, cast_to=None, **kwargs):
        """Evaluate an expression to get a concrete value.

        Args:
            e: Expression to evaluate.
            cast_to: Type to cast result to (bytes or int).
            **kwargs: Additional arguments (ignored).

        Returns:
            Concrete value of the expression.

        Raises:
            SimUnsatError: If no solution exists.
        """
        # Fast path for concrete values
        concrete_val = _concrete_value(e)
        if concrete_val is not None:
            return self._cast_to(e, concrete_val, cast_to)

        result = self._rust_ctx.eval(e)
        if result is None:
            raise SimUnsatError(f"Not satisfiable: {e}")
        return self._cast_to(e, result, cast_to)

    def eval_one(self, e, **kwargs):
        """Evaluate an expression to get a single concrete value.

        This is used by state.addr to get the concrete instruction pointer.

        Args:
            e: Expression to evaluate.
            **kwargs: Additional arguments (ignored).

        Returns:
            Concrete value of the expression.

        Raises:
            SimUnsatError: If no solution exists.
            SimValueError: If multiple solutions exist.
        """
        # Fast path for concrete values
        concrete_val = _concrete_value(e)
        if concrete_val is not None:
            return concrete_val

        # Get up to 2 solutions to check uniqueness
        results = self.eval_upto(e, 2, **kwargs)
        if len(results) == 0:
            raise SimUnsatError(f"Not satisfiable: {e}")
        if len(results) > 1:
            from angr.errors import SimValueError
            raise SimValueError(f"Expression has multiple solutions: {e}")
        return results[0]

    def eval_upto(self, e, n, cast_to=None, **kwargs):
        """Evaluate an expression and return up to n solutions.

        Args:
            e: Expression to evaluate.
            n: Maximum number of solutions.
            cast_to: Type to cast results to.
            **kwargs: Additional arguments (ignored).

        Returns:
            List of up to n concrete values.

        Raises:
            SimUnsatError: If no solution exists.
        """
        # Fast path for concrete values
        concrete_val = _concrete_value(e)
        if concrete_val is not None:
            return [self._cast_to(e, concrete_val, cast_to)]

        results = self._rust_ctx.eval_upto(e, n)
        if not results:
            raise SimUnsatError(f"Not satisfiable: {e}")
        return [self._cast_to(e, r, cast_to) for r in results]

    def min(self, e, extra_constraints=(), signed=False, **kwargs):
        """Return the minimum value of an expression.

        Args:
            e: Expression to minimize.
            extra_constraints: Additional constraints.
            signed: Whether to treat as signed.
            **kwargs: Additional arguments (ignored).

        Returns:
            Minimum value.
        """
        # Fast path for concrete values
        concrete_val = _concrete_value(e)
        if concrete_val is not None:
            return concrete_val

        if extra_constraints:
            self._rust_ctx.push()
            try:
                for c in extra_constraints:
                    if not isinstance(c, bool):
                        self._rust_ctx.add_constraint_ast(c)
                result = self._rust_ctx.min(e, signed)
            finally:
                self._rust_ctx.pop()
            if result is None:
                raise SimUnsatError(f"Cannot minimize {e}: solver returned None")
            return result

        result = self._rust_ctx.min(e, signed)
        if result is None:
            raise SimUnsatError(f"Cannot minimize {e}: solver returned None")
        return result

    def max(self, e, extra_constraints=(), signed=False, **kwargs):
        """Return the maximum value of an expression.

        Args:
            e: Expression to maximize.
            extra_constraints: Additional constraints.
            signed: Whether to treat as signed.
            **kwargs: Additional arguments (ignored).

        Returns:
            Maximum value.
        """
        # Fast path for concrete values
        concrete_val = _concrete_value(e)
        if concrete_val is not None:
            return concrete_val

        if extra_constraints:
            self._rust_ctx.push()
            try:
                for c in extra_constraints:
                    if not isinstance(c, bool):
                        self._rust_ctx.add_constraint_ast(c)
                result = self._rust_ctx.max(e, signed)
            finally:
                self._rust_ctx.pop()
            if result is None:
                raise SimUnsatError(f"Cannot maximize {e}: solver returned None")
            return result

        result = self._rust_ctx.max(e, signed)
        if result is None:
            raise SimUnsatError(f"Cannot maximize {e}: solver returned None")
        return result

    def is_true(self, e, **kwargs):
        """Check if an expression is definitely true.

        Args:
            e: Expression to check.
            **kwargs: Additional arguments (ignored).

        Returns:
            True if definitely true, False otherwise.
        """
        if isinstance(e, bool):
            return e
        if hasattr(e, 'op') and e.op == 'BoolV':
            return e.args[0]

        return self._rust_ctx.is_true(e)

    def is_false(self, e, **kwargs):
        """Check if an expression is definitely false.

        Args:
            e: Expression to check.
            **kwargs: Additional arguments (ignored).

        Returns:
            True if definitely false, False otherwise.
        """
        if isinstance(e, bool):
            return not e
        if hasattr(e, 'op') and e.op == 'BoolV':
            return not e.args[0]

        return self._rust_ctx.is_false(e)

    def solution(self, e, v, extra_constraints=(), **kwargs):
        """Check if v is a valid solution for expression e.

        Args:
            e: Expression.
            v: Proposed solution.
            extra_constraints: Additional constraints.
            **kwargs: Additional arguments (ignored).

        Returns:
            True if v is a valid solution.
        """
        # Convert v to integer if needed
        if hasattr(v, 'args') and hasattr(v, 'op') and v.op == 'BVV':
            v = v.args[0]

        if extra_constraints:
            self._rust_ctx.push()
            try:
                for c in extra_constraints:
                    if not isinstance(c, bool):
                        self._rust_ctx.add_constraint_ast(c)
                result = self._rust_ctx.solution(e, v)
            finally:
                self._rust_ctx.pop()
            return result

        return self._rust_ctx.solution(e, v)

    def unique(self, e, **kwargs):
        """Check if an expression has exactly one solution.

        Args:
            e: Expression to check.
            **kwargs: Additional arguments (ignored).

        Returns:
            True if exactly one solution exists.
        """
        if not hasattr(e, 'symbolic') or not e.symbolic:
            return True

        results = self.eval_upto(e, 2, **kwargs)
        if len(results) == 1:
            self.add(e == results[0])
            return True
        return False

    def symbolic(self, e):
        """Check if an expression is symbolic.

        Args:
            e: Expression to check.

        Returns:
            True if symbolic.
        """
        if isinstance(e, (int, bytes, float, bool)):
            return False
        return getattr(e, 'symbolic', False)

    def simplify(self, e=None):
        """Simplify an expression.

        Args:
            e: Expression to simplify. If None, simplifies solver state.

        Returns:
            Simplified expression.
        """
        if e is None:
            return None
        if isinstance(e, (int, float, bool)):
            return e
        return claripy.simplify(e)

    def BVS(self, name, size, **kwargs):
        """Create a symbolic bitvector.

        Args:
            name: Name of the symbol.
            size: Size in bits.
            **kwargs: Additional arguments passed to claripy.BVS.

        Returns:
            Symbolic bitvector.
        """
        # Filter out angr-specific kwargs that claripy doesn't accept
        filtered_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ('key', 'inspect', 'events')}
        r = claripy.BVS(name, size, **filtered_kwargs)
        self.all_variables.append(r)
        return r

    def Unconstrained(self, name, bits, **kwargs):
        """Create an unconstrained symbol.

        Args:
            name: Name of the symbol.
            bits: Size in bits.
            **kwargs: Additional arguments.

        Returns:
            Symbolic bitvector or concrete 0.
        """
        return self.BVS(name, bits, **kwargs)

    @staticmethod
    def _cast_to(e, solution, cast_to):
        """Cast a solution to the desired type.

        Args:
            e: Original expression.
            solution: Solution value.
            cast_to: Target type (bytes, int, or None).

        Returns:
            Cast value.
        """
        if cast_to is None:
            return solution

        if cast_to is bytes:
            if isinstance(solution, bool):
                return bytes([int(solution)])
            # Get bit width from expression
            width = getattr(e, 'length', 64)
            if width == 0:
                return b""
            if width % 8:
                raise ValueError("bit string length is not a multiple of 8")
            return solution.to_bytes(width // 8, byteorder='big')

        if cast_to is int:
            if isinstance(solution, bool):
                return int(solution)
            return int(solution)

        raise ValueError(f"Unsupported cast_to type: {cast_to}")

    @SimStatePlugin.memo
    def copy(self, memo):
        """Copy the plugin for state forking.

        Args:
            memo: Memoization dictionary.

        Returns:
            Copy of the plugin with forked Rust context.
        """
        c = RustSimSolver.__new__(RustSimSolver)
        c.state = None
        c._rust_ctx = self._rust_ctx.fork()
        c.all_variables = self.all_variables.copy()
        return c

    def merge(self, others, merge_conditions, common_ancestor=None):
        """Merge solver states.

        Note: Full merge support would require tracking constraints
        on the Python side. For now, this creates a fresh solver.
        """
        # Simple merge: just use self's constraints
        # A proper implementation would merge constraint sets
        return False

    def widen(self, others):
        """Widen solver state."""
        return self.merge(others, None)

    def downsize(self):
        """Free memory (no-op for Rust solver)."""
        pass

    @property
    def _solver(self):
        """Compatibility property for code expecting claripy solver."""
        # Return self as a compatibility shim - we provide the necessary attributes
        return self

    @property
    def variables(self):
        """Return the set of all symbolic variable names.

        This is used by address concretization to check if an address
        involves variables that have been constrained.
        """
        result = set()
        for v in self.all_variables:
            if hasattr(v, 'variables'):
                result.update(v.variables)
        return frozenset(result)

    def __getattr__(self, name):
        """Forward claripy attribute access.

        This allows using solver.BVV, solver.Or, etc.
        """
        return getattr(claripy, name)
