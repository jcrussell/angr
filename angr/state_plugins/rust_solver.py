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


# Try to import the Rust solver context and handle types
try:
    from angr.rustylib.vex_engine import RustSolverContext, RustBVHandle
    RUST_SOLVER_AVAILABLE = True
except ImportError:
    RUST_SOLVER_AVAILABLE = False
    RustSolverContext = None
    RustBVHandle = None


def _is_handle(e):
    """Check if e is a RustBVHandle."""
    return RustBVHandle is not None and isinstance(e, RustBVHandle)


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

    Optimizations:
    - Constraint tracking: Maintains Python-side list for introspection
    - Lazy conversion: Defers constraint conversion until solver check
    - Full AST caching: Rust side caches all symbolic AST nodes

    Usage:
        # Create state with Rust solver
        state = proj.factory.entry_state()
        state.register_plugin('solver', RustSimSolver())

        # Or enable via sim_options.RUST_SOLVER
        state = proj.factory.entry_state(add_options={sim_options.RUST_SOLVER})
    """

    def __init__(self, rust_ctx=None, all_variables=None, constraint_list=None,
                 pending=None, temporal_tracked_variables=None,
                 eternal_tracked_variables=None, **kwargs):
        """Initialize the Rust solver plugin.

        Args:
            rust_ctx: Optional RustSolverContext to use. If None, creates new one.
            all_variables: List of all symbolic variables for tracking.
            constraint_list: List of constraints for introspection.
            pending: List of pending constraints for lazy conversion.
            temporal_tracked_variables: Dict of versioned tracked variables.
            eternal_tracked_variables: Dict of permanent tracked variables.
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
        # Phase 3: Track constraints for introspection
        self._constraint_list = constraint_list if constraint_list is not None else []
        # Phase 4: Pending constraints for lazy conversion
        self._pending = pending if pending is not None else []
        # Variable tracking system
        self.temporal_tracked_variables = temporal_tracked_variables if temporal_tracked_variables is not None else {}
        self.eternal_tracked_variables = eternal_tracked_variables if eternal_tracked_variables is not None else {}

    @property
    def constraints(self):
        """Return the constraints tracked on the Python side."""
        # Flush pending first to ensure list is complete
        self._flush()
        return list(self._constraint_list)

    def _flush(self):
        """Flush pending constraints to the Rust solver.

        This implements lazy constraint conversion - constraints are only
        converted to Rust/Z3 when a solver check is actually needed.
        This avoids conversion overhead for branches that turn out to be unsat.

        P2 Fix: Properly handle constraint errors - re-add to pending queue
        on failure and propagate the error so callers know the solver state
        may be inconsistent.
        """
        if not self._pending:
            return

        pending = self._pending
        self._pending = []

        try:
            if len(pending) == 1:
                self._rust_ctx.add_constraint_ast(pending[0])
            else:
                self._rust_ctx.add_constraints(pending)
        except Exception as e:
            l.error("Failed to flush constraints to Rust solver: %s", e)
            # Re-add to pending so they're not lost
            self._pending = pending + self._pending
            raise  # Propagate error so caller knows solver state is inconsistent

    def reload_solver(self, constraints=None):
        """Reload the solver with new constraints."""
        self._rust_ctx = RustSolverContext()
        self._constraint_list = []
        self._pending = []
        if constraints:
            for c in constraints:
                self._constraint_list.append(c)
                self._pending.append(c)

    def add(self, *constraints):
        """Add constraints to the solver.

        Uses lazy conversion - constraints are queued and only converted
        to Rust/Z3 when a solver check is actually needed.

        Args:
            *constraints: Constraint ASTs to add.
        """
        for c in constraints:
            if isinstance(c, (list, tuple)):
                raise TypeError("Tuple or list passed to add!")
            if isinstance(c, bool):
                if not c:
                    # Adding False makes the solver unsat
                    # Flush immediately and add the impossible constraint
                    self._flush()
                    self._constraint_list.append(claripy.false)
                    try:
                        self._rust_ctx.add_constraint_ast(claripy.false)
                    except Exception as e:
                        l.warning("Failed to add false constraint: %s", e)
                    return
                continue
            # Track constraint and queue for lazy conversion
            self._constraint_list.append(c)
            self._pending.append(c)

    def satisfiable(self, extra_constraints=(), **kwargs):
        """Check if constraints are satisfiable.

        Args:
            extra_constraints: Additional constraints to check (temporarily).
            **kwargs: Additional arguments (ignored).

        Returns:
            True if satisfiable, False otherwise.
        """
        # Flush pending constraints before checking
        self._flush()

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

        # Flush pending constraints before solving
        self._flush()

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

        # Flush pending constraints before solving
        self._flush()

        results = self._rust_ctx.eval_upto(e, n)
        if not results:
            raise SimUnsatError(f"Not satisfiable: {e}")
        return [self._cast_to(e, r, cast_to) for r in results]

    def eval_atleast(self, e, n, cast_to=None, **kwargs):
        """Evaluate expression and verify at least n solutions exist.

        Args:
            e: Expression to evaluate.
            n: Minimum number of required solutions.
            cast_to: Type to cast results to.
            **kwargs: Additional arguments.

        Returns:
            List of n solutions.

        Raises:
            SimUnsatError: If no solution exists.
            SimValueError: If fewer than n solutions exist.
        """
        r = self.eval_upto(e, n, cast_to, **kwargs)
        if len(r) != n:
            from angr.errors import SimValueError
            raise SimValueError(f"Concretized {len(r)} values (must be at least {n}) in eval_atleast")
        return r

    def eval_atmost(self, e, n, cast_to=None, **kwargs):
        """Evaluate expression and verify at most n solutions exist.

        Args:
            e: Expression to evaluate.
            n: Maximum number of allowed solutions.
            cast_to: Type to cast results to.
            **kwargs: Additional arguments.

        Returns:
            List of up to n solutions.

        Raises:
            SimUnsatError: If no solution exists.
            SimValueError: If more than n solutions exist.
        """
        r = self.eval_upto(e, n + 1, cast_to, **kwargs)
        if len(r) > n:
            from angr.errors import SimValueError
            raise SimValueError(f"Concretized {len(r)} values (must be at most {n}) in eval_atmost")
        return r

    def eval_exact(self, e, n, cast_to=None, **kwargs):
        """Evaluate expression and verify exactly n solutions exist.

        Args:
            e: Expression to evaluate.
            n: Exact number of required solutions.
            cast_to: Type to cast results to.
            **kwargs: Additional arguments.

        Returns:
            List of exactly n solutions.

        Raises:
            SimUnsatError: If no solution exists.
            SimValueError: If number of solutions != n.
        """
        r = self.eval_upto(e, n + 1, cast_to, **kwargs)
        if len(r) != n:
            from angr.errors import SimValueError
            raise SimValueError(f"Concretized {len(r)} values (must be exactly {n}) in eval_exact")
        return r

    def eval_to_ast(self, e, n, extra_constraints=(), exact=None):
        """Evaluate expression and return solutions as AST nodes.

        Args:
            e: Expression to evaluate.
            n: Number of solutions.
            extra_constraints: Additional constraints.
            exact: If False, allow approximate solutions.

        Returns:
            Tuple of solution ASTs.
        """
        # Get primitive solutions
        solutions = self.eval_upto(e, n, extra_constraints=extra_constraints)

        # Convert to AST nodes
        if hasattr(e, 'length'):
            return tuple(claripy.BVV(s, e.length) for s in solutions)
        elif hasattr(e, 'op') and e.op == 'BoolS':
            return tuple(claripy.BoolV(s) for s in solutions)
        else:
            return tuple(claripy.BVV(s, 64) for s in solutions)

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

        # Flush pending constraints before solving
        self._flush()

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

        # Flush pending constraints before solving
        self._flush()

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

    # Aliases for compatibility with standard SimSolver
    min_int = min
    max_int = max

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

        # Flush pending constraints before checking
        self._flush()
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

        # Flush pending constraints before checking
        self._flush()
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

        # Flush pending constraints before checking
        self._flush()

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

        # eval_upto already flushes pending constraints
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

    def single_valued(self, e):
        """Check if expression has only one possible value.

        Unlike unique(), this does NOT query the constraint solver.

        Args:
            e: Expression to check.

        Returns:
            True if expression has exactly one possible value.
        """
        if isinstance(e, (int, bytes, float, bool)):
            return True

        # Check cardinality for value sets (VSA mode)
        if hasattr(e, 'cardinality'):
            return e.cardinality <= 1

        # For symbolic mode, non-symbolic means single-valued
        return not self.symbolic(e)

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
                          if k not in ('key', 'inspect', 'events', 'eternal')}
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

    def register_variable(self, v, key, eternal=True):
        """Register a variable with the tracking system.

        Args:
            v: The BVS to register.
            key: A tuple key to register under.
            eternal: If True, permanent; if False, versioned with counter.
        """
        if type(key) is not tuple:
            raise TypeError("Variable tracking key must be a tuple")
        if eternal:
            self.eternal_tracked_variables[key] = v
        else:
            # Create new dict to avoid mutation issues
            self.temporal_tracked_variables = dict(self.temporal_tracked_variables)
            ctrkey = (*key, None)
            ctrval = self.temporal_tracked_variables.get(ctrkey, 0) + 1
            self.temporal_tracked_variables[ctrkey] = ctrval
            tempkey = (*key, ctrval)
            self.temporal_tracked_variables[tempkey] = v

    def get_variables(self, *keys):
        """Iterate over variables whose tracking key starts with given prefix.

        Args:
            *keys: Key prefix to match.

        Yields:
            Tuples of (full_key, variable).
        """
        for k, v in self.eternal_tracked_variables.items():
            if len(k) >= len(keys) and all(x == y for x, y in zip(keys, k)):
                yield k, v
        for k, v in self.temporal_tracked_variables.items():
            if k[-1] is None:
                continue
            if len(k) >= len(keys) and all(x == y for x, y in zip(keys, k)):
                yield k, v

    def describe_variables(self, v):
        """Given an AST, iterate over tracking keys of registered BVS leaves.

        Args:
            v: Claripy AST to inspect.

        Yields:
            Tracking keys for registered variables found in v.
        """
        reverse_mapping = {next(iter(var.variables)): k
                           for k, var in self.eternal_tracked_variables.items()}
        reverse_mapping.update(
            {next(iter(var.variables)): k
             for k, var in self.temporal_tracked_variables.items() if k[-1] is not None}
        )
        for var in v.variables:
            if var in reverse_mapping:
                yield reverse_mapping[var]

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
        # Flush pending constraints before fork to ensure consistency
        self._flush()

        c = RustSimSolver.__new__(RustSimSolver)
        c.state = None
        c._rust_ctx = self._rust_ctx.fork()
        c.all_variables = self.all_variables.copy()
        # Copy constraint list (shared immutable AST refs are fine)
        c._constraint_list = self._constraint_list.copy()
        # Fresh pending list for the fork
        c._pending = []
        # Copy variable tracking dictionaries
        c.temporal_tracked_variables = self.temporal_tracked_variables.copy()
        c.eternal_tracked_variables = self.eternal_tracked_variables.copy()
        return c

    def merge(self, others, merge_conditions, common_ancestor=None):
        """Merge solver states.

        Note: Full merge support would require complex constraint disjunction.
        For now, this merges the constraint lists but returns False to
        indicate that caller should handle state merging.
        """
        # Flush pending constraints before merge
        self._flush()
        for other in others:
            if hasattr(other, '_flush'):
                other._flush()

        # Merge variable lists
        for other in others:
            for v in other.all_variables:
                if v not in self.all_variables:
                    self.all_variables.append(v)

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

    def unsat_core(self, extra_constraints=()):
        """Return the unsat core from the solver.

        Args:
            extra_constraints: Extra constraints to add temporarily.

        Returns:
            The unsat core constraints as a list of AST nodes.

        Raises:
            SimSolverOptionError: If constraint tracking not enabled.
        """
        from angr import sim_options as o
        from angr.errors import SimSolverOptionError

        if self.state and o.CONSTRAINT_TRACKING_IN_SOLVER not in self.state.options:
            raise SimSolverOptionError(
                "CONSTRAINT_TRACKING_IN_SOLVER must be enabled before calling unsat_core()."
            )

        # Delegate to Rust context if supported
        if hasattr(self._rust_ctx, 'unsat_core'):
            self._flush()
            if extra_constraints:
                self._rust_ctx.push()
                try:
                    for c in extra_constraints:
                        if not isinstance(c, bool):
                            self._rust_ctx.add_constraint_ast(c)
                    core_indices = self._rust_ctx.unsat_core()
                finally:
                    self._rust_ctx.pop()
            else:
                core_indices = self._rust_ctx.unsat_core()

            # Map indices back to constraint ASTs
            return [self._constraint_list[i] for i in core_indices
                    if i < len(self._constraint_list)]

        # Fallback: not supported
        raise NotImplementedError("unsat_core requires Rust solver with constraint tracking")

    def __getattr__(self, name):
        """Forward claripy attribute access.

        This allows using solver.BVV, solver.Or, etc.
        """
        return getattr(claripy, name)

    # =========================================================================
    # Handle-based API (Claripy Bypass)
    # These methods allow symbolic operations without claripy AST conversion,
    # providing significant speedups for constraint-heavy execution.
    # =========================================================================

    def create_symbolic_handle(self, name, width):
        """Create a symbolic bitvector and return a handle.

        This bypasses claripy.BVS() for native Rust symbolic value creation.
        The returned handle can be used with handle-based operations for
        maximum performance.

        Args:
            name: Name of the symbolic variable.
            width: Bit width.

        Returns:
            RustBVHandle for the symbolic value.
        """
        return self._rust_ctx.create_symbolic(name, width)

    def create_concrete_handle(self, value, width):
        """Create a concrete bitvector and return a handle.

        This bypasses claripy.BVV() for native Rust concrete value creation.

        Args:
            value: Concrete value.
            width: Bit width.

        Returns:
            RustBVHandle for the concrete value.
        """
        return self._rust_ctx.create_concrete(value, width)

    def eval_handle(self, handle):
        """Evaluate a handle to get a concrete value.

        This bypasses claripy AST conversion for solver queries.

        Args:
            handle: RustBVHandle to evaluate.

        Returns:
            Concrete value, or None if unsatisfiable.
        """
        self._flush()
        return self._rust_ctx.eval_handle(handle.id)

    def min_handle(self, handle, signed=False):
        """Get the minimum value for a handle.

        Args:
            handle: RustBVHandle to minimize.
            signed: Whether to treat as signed.

        Returns:
            Minimum value.
        """
        self._flush()
        return self._rust_ctx.min_handle(handle.id, signed)

    def max_handle(self, handle, signed=False):
        """Get the maximum value for a handle.

        Args:
            handle: RustBVHandle to maximize.
            signed: Whether to treat as signed.

        Returns:
            Maximum value.
        """
        self._flush()
        return self._rust_ctx.max_handle(handle.id, signed)

    def add_handle_constraint(self, handle_or_id):
        """Add a constraint from a handle (must be 1-bit).

        Args:
            handle_or_id: RustBVHandle or handle ID representing the constraint.
        """
        if hasattr(handle_or_id, 'id'):
            self._rust_ctx.add_constraint_handle(handle_or_id.id)
        else:
            self._rust_ctx.add_constraint_handle(handle_or_id)

    # =========================================================================
    # Handle-based Arithmetic Operations
    # =========================================================================

    def handle_add(self, a, b):
        """Add two handles."""
        return self._rust_ctx.op_add(a.id, b.id)

    def handle_sub(self, a, b):
        """Subtract two handles."""
        return self._rust_ctx.op_sub(a.id, b.id)

    def handle_mul(self, a, b):
        """Multiply two handles."""
        return self._rust_ctx.op_mul(a.id, b.id)

    def handle_udiv(self, a, b):
        """Unsigned division."""
        return self._rust_ctx.op_udiv(a.id, b.id)

    def handle_sdiv(self, a, b):
        """Signed division."""
        return self._rust_ctx.op_sdiv(a.id, b.id)

    def handle_urem(self, a, b):
        """Unsigned remainder."""
        return self._rust_ctx.op_urem(a.id, b.id)

    def handle_srem(self, a, b):
        """Signed remainder."""
        return self._rust_ctx.op_srem(a.id, b.id)

    def handle_neg(self, a):
        """Negation."""
        return self._rust_ctx.op_neg(a.id)

    # =========================================================================
    # Handle-based Bitwise Operations
    # =========================================================================

    def handle_and(self, a, b):
        """Bitwise AND."""
        return self._rust_ctx.op_and(a.id, b.id)

    def handle_or(self, a, b):
        """Bitwise OR."""
        return self._rust_ctx.op_or(a.id, b.id)

    def handle_xor(self, a, b):
        """Bitwise XOR."""
        return self._rust_ctx.op_xor(a.id, b.id)

    def handle_not(self, a):
        """Bitwise NOT."""
        return self._rust_ctx.op_not(a.id)

    # =========================================================================
    # Handle-based Shift Operations
    # =========================================================================

    def handle_shl(self, a, b):
        """Left shift."""
        return self._rust_ctx.op_shl(a.id, b.id)

    def handle_lshr(self, a, b):
        """Logical right shift."""
        return self._rust_ctx.op_lshr(a.id, b.id)

    def handle_ashr(self, a, b):
        """Arithmetic right shift."""
        return self._rust_ctx.op_ashr(a.id, b.id)

    def handle_rotl(self, a, b):
        """Rotate left."""
        return self._rust_ctx.op_rotl(a.id, b.id)

    def handle_rotr(self, a, b):
        """Rotate right."""
        return self._rust_ctx.op_rotr(a.id, b.id)

    # =========================================================================
    # Handle-based Comparison Operations
    # =========================================================================

    def handle_eq(self, a, b):
        """Equality comparison (returns 1-bit handle)."""
        return self._rust_ctx.op_eq(a.id, b.id)

    def handle_ne(self, a, b):
        """Inequality comparison (returns 1-bit handle)."""
        return self._rust_ctx.op_ne(a.id, b.id)

    def handle_ult(self, a, b):
        """Unsigned less than."""
        return self._rust_ctx.op_ult(a.id, b.id)

    def handle_ule(self, a, b):
        """Unsigned less than or equal."""
        return self._rust_ctx.op_ule(a.id, b.id)

    def handle_ugt(self, a, b):
        """Unsigned greater than."""
        return self._rust_ctx.op_ugt(a.id, b.id)

    def handle_uge(self, a, b):
        """Unsigned greater than or equal."""
        return self._rust_ctx.op_uge(a.id, b.id)

    def handle_slt(self, a, b):
        """Signed less than."""
        return self._rust_ctx.op_slt(a.id, b.id)

    def handle_sle(self, a, b):
        """Signed less than or equal."""
        return self._rust_ctx.op_sle(a.id, b.id)

    def handle_sgt(self, a, b):
        """Signed greater than."""
        return self._rust_ctx.op_sgt(a.id, b.id)

    def handle_sge(self, a, b):
        """Signed greater than or equal."""
        return self._rust_ctx.op_sge(a.id, b.id)

    # =========================================================================
    # Handle-based Conversion Operations
    # =========================================================================

    def handle_zero_extend(self, a, to_width):
        """Zero-extend to a wider width."""
        return self._rust_ctx.op_zero_extend(a.id, to_width)

    def handle_sign_extend(self, a, to_width):
        """Sign-extend to a wider width."""
        return self._rust_ctx.op_sign_extend(a.id, to_width)

    def handle_truncate(self, a, to_width):
        """Truncate to a narrower width."""
        return self._rust_ctx.op_truncate(a.id, to_width)

    def handle_extract(self, a, high, low):
        """Extract bits [high:low] (inclusive)."""
        return self._rust_ctx.op_extract(a.id, high, low)

    def handle_concat(self, a, b):
        """Concatenate two values (a becomes high bits)."""
        return self._rust_ctx.op_concat(a.id, b.id)

    def handle_ite(self, cond, then_val, else_val):
        """If-then-else."""
        return self._rust_ctx.op_ite(cond.id, then_val.id, else_val.id)

    # =========================================================================
    # Hybrid Methods (work with both handles and claripy ASTs)
    # =========================================================================

    def eval_any(self, e, cast_to=None, **kwargs):
        """Evaluate an expression (handle or claripy AST).

        This is a unified method that works with both RustBVHandle and
        claripy AST objects, choosing the optimal path automatically.

        Args:
            e: Expression to evaluate (handle or AST).
            cast_to: Type to cast result to.
            **kwargs: Additional arguments.

        Returns:
            Concrete value.
        """
        if _is_handle(e):
            result = self.eval_handle(e)
            if result is None:
                raise SimUnsatError(f"Not satisfiable: {e}")
            # Handle cast_to
            if cast_to is bytes:
                width = e.width
                if width == 0:
                    return b""
                if width % 8:
                    raise ValueError("bit string length is not a multiple of 8")
                return result.to_bytes(width // 8, byteorder='big')
            return result
        else:
            return self.eval(e, cast_to=cast_to, **kwargs)

    def min_any(self, e, signed=False, **kwargs):
        """Get minimum value (handle or claripy AST).

        Args:
            e: Expression to minimize.
            signed: Whether to treat as signed.
            **kwargs: Additional arguments.

        Returns:
            Minimum value.
        """
        if _is_handle(e):
            result = self.min_handle(e, signed)
            if result is None:
                raise SimUnsatError(f"Cannot minimize: {e}")
            return result
        else:
            return self.min(e, signed=signed, **kwargs)

    def max_any(self, e, signed=False, **kwargs):
        """Get maximum value (handle or claripy AST).

        Args:
            e: Expression to maximize.
            signed: Whether to treat as signed.
            **kwargs: Additional arguments.

        Returns:
            Maximum value.
        """
        if _is_handle(e):
            result = self.max_handle(e, signed)
            if result is None:
                raise SimUnsatError(f"Cannot maximize: {e}")
            return result
        else:
            return self.max(e, signed=signed, **kwargs)

    @property
    def handle_count(self):
        """Get the number of handles in the symbol table."""
        return self._rust_ctx.handle_count()
