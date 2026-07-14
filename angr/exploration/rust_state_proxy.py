"""
Lightweight proxy objects that expose Rust symex state via PyO3 bindings.

Instead of creating full angr SimState objects and syncing bidirectionally,
these proxies delegate reads directly to the Rust engine. This eliminates
caching, sync bugs, and the dual-solver problem for non-SimProcedure paths.

Full SimState creation is only needed for SimProcedure execution (which
requires angr plugins like posix, filesystem, etc.).
"""

from __future__ import annotations

import contextlib
import logging

import claripy

from angr.rustylib.vex_engine import register_size_for_arch

l = logging.getLogger(__name__)


_REGISTER_OVERLAP_CACHE: dict[str, dict[str, frozenset[str]]] = {}


def _get_overlap_map(arch) -> dict[str, frozenset[str]]:
    """Per-arch map of register name -> frozenset of names whose (offset, size) overlaps.

    Cached per architecture. Used by ``RustRegisterProxy.__setattr__`` to
    invalidate stale aliases on partial writes — e.g. writing ``eax`` must
    invalidate cached ``rax`` / ``ax`` / ``al`` / ``ah`` so a subsequent
    ``state.regs.rax`` read fetches the post-write value from Rust rather
    than the pre-write cache entry.
    """
    cached = _REGISTER_OVERLAP_CACHE.get(arch.name)
    if cached is not None:
        return cached
    by_range = []
    for name, info in arch.registers.items():
        if not info or len(info) < 2:
            continue
        off, sz = info[0], info[1]
        by_range.append((name, off, sz))
    overlap: dict[str, frozenset[str]] = {}
    for name, off, sz in by_range:
        end = off + sz
        names = {name}
        for other_name, other_off, other_sz in by_range:
            if other_off < end and other_off + other_sz > off:
                names.add(other_name)
        overlap[name] = frozenset(names)
    _REGISTER_OVERLAP_CACHE[arch.name] = overlap
    return overlap


def _cast_eval_result(expr, result, cast_to):
    """Cast eval result matching angr's SimSolver._cast_to behavior."""
    if cast_to is None:
        return result
    if cast_to is bytes:
        if hasattr(expr, "__len__"):
            nbits = len(expr)
        elif hasattr(expr, "size"):
            nbits = expr.size()
        else:
            nbits = 64
        if nbits == 0:
            return b""
        return result.to_bytes(nbits // 8, byteorder="big")
    return cast_to(result)


def _with_extra_constraints(ctx, fn, *args, extra=(), **kwargs):
    """Run ``fn(*args, **kwargs)`` under a temporary push/pop of ``extra``.

    When ``extra`` is empty the solver call runs directly with no push/pop.
    """
    if extra:
        ctx.push()
        try:
            for c in extra:
                ctx.add_constraint_ast(c)
            return fn(*args, **kwargs)
        finally:
            ctx.pop()
    return fn(*args, **kwargs)


def _resolve_none_extremum(kind, expr, constraints, extra_constraints, signed, satisfiable_fn):
    """Resolve a ``None`` result from ``RustSolverContext.min``/``max``.

    The Rust solver returns ``None`` for BOTH an unsat context and a
    *satisfiable* extremum wider than ``u128`` (``solving_ops.rs``). The two
    proxy ``min``/``max`` sites previously raised ``UnsatError`` on any
    ``None``, so a valid >128-bit bound was mislabeled unsat and the state was
    killed (angr-fjhk9). Disambiguate using the same logic as the
    ``RustSolverFallback`` twin (angr-8iv6j), keeping the four solver-shim
    sites in sync (``invariant-rust-solver-fallback-class``):

    * genuinely unsat -> raise ``claripy.errors.UnsatError``. This preserves
      the proxy-wide exception-type convention (the export.py twin raises
      ``SimUnsatError`` instead — that divergence is intentional; the proxies
      raise ``claripy.errors.UnsatError`` everywhere, incl. ``eval``, and a
      test asserts it). ``satisfiable_fn`` is cheap: the Rust solver caches the
      sat result from the ``min``/``max`` call above.
    * satisfiable but width>128 -> fall back to claripy's big-int solver over
      the exported constraints, which CAN bound it. This path is currently
      unreachable via 64-bit address concretization, so the export cost never
      materializes in practice.
    """
    if not satisfiable_fn():
        raise claripy.errors.UnsatError("unsat")
    solver = claripy.Solver()
    cons = list(constraints)
    if extra_constraints:
        cons.extend(extra_constraints)
    if cons:
        solver.add(cons)
    return solver.min(expr, signed=signed) if kind == "min" else solver.max(expr, signed=signed)


class _SolutionCountMixin:
    """Shared solution-count wrappers for the two solver proxies.

    Subclasses must provide an ``eval_upto(expr, n, **kwargs)`` method.
    """

    def eval_one(self, expr, **kwargs):
        """Evaluate expression expecting exactly one solution."""
        results = self.eval_upto(expr, 2, **kwargs)
        if len(results) != 1:
            raise claripy.errors.ClaripyError(f"expected 1 solution, got {len(results)}")
        return results[0]

    def eval_exact(self, expr, n, **kwargs):
        """Evaluate expression expecting exactly n solutions."""
        results = self.eval_upto(expr, n + 1, **kwargs)
        if len(results) != n:
            raise claripy.errors.ClaripyError(f"expected {n} solutions, got {len(results)}")
        return results

    def eval_atleast(self, expr, n, **kwargs):
        """Evaluate expression expecting at least n solutions."""
        results = self.eval_upto(expr, n, **kwargs)
        if len(results) < n:
            raise claripy.errors.ClaripyError(f"expected at least {n} solutions, got {len(results)}")
        return results


class RustSolverProxyBase(_SolutionCountMixin):
    """Shared delegation core for the two Rust solver proxies.

    ``RustSolverProxy`` (standalone ``state.solver`` read proxy) and
    ``RustSolverProxyPlugin`` (SimProcedure-callback ``state.solver`` plugin)
    duplicate the bulk of their solver-query surface. The only thing that
    differs is *how* each obtains the Rust solver context to query — the
    standalone proxy lazily forks into ``self._solver_ctx`` via
    ``_ensure_solver``, while the plugin caches its fork in
    ``self._rust_ctx_cache`` (invalidated on ``add``). Subclasses express that
    difference through two hooks:

      * ``_query_ctx()`` — return the context to run a query against (forking
        on first use).
      * ``_forked_ctx()`` — return the already-forked context, or ``None`` if
        nothing has been forked yet (used by the timeout setter to propagate
        to a live fork).

    Everything that is identical modulo that access — ``satisfiable`` /
    ``min`` / ``max`` / ``min_int`` / ``max_int`` / ``symbolic`` /
    ``constraints`` / ``timeout`` and the ``_cast_result`` helper — lives
    here. ``eval`` / ``eval_upto`` / ``is_true`` / ``is_false`` / ``solution``
    stay on the subclasses because their bodies genuinely differ (the plugin
    munges the ``eval`` signature, unwraps ``SimActionObject`` constraints, and
    shortcuts concrete ``BoolV`` expressions; the standalone proxy does not).
    """

    _cast_result = staticmethod(_cast_eval_result)

    # ---------------------------------------------------------------
    # Context-access hooks — subclasses supply the fork strategy.
    # ---------------------------------------------------------------

    def _query_ctx(self):
        raise NotImplementedError

    def _forked_ctx(self):
        raise NotImplementedError

    # ---------------------------------------------------------------
    # Shared delegation
    # ---------------------------------------------------------------

    def satisfiable(self, extra_constraints=(), **kwargs):
        """Check if the state's constraints are satisfiable."""
        ctx = self._query_ctx()
        return _with_extra_constraints(ctx, ctx.satisfiable, extra=extra_constraints)

    def min(self, expr, extra_constraints=(), signed=False, exact=None, **kwargs):
        """Get minimum value of expression."""
        ctx = self._query_ctx()
        result = _with_extra_constraints(ctx, ctx.min, expr, extra=extra_constraints, signed=signed)
        if result is not None:
            return result
        return _resolve_none_extremum(
            "min",
            expr,
            self.constraints,
            extra_constraints,
            signed,
            lambda: self.satisfiable(extra_constraints=extra_constraints),
        )

    def max(self, expr, extra_constraints=(), signed=False, exact=None, **kwargs):
        """Get maximum value of expression."""
        ctx = self._query_ctx()
        result = _with_extra_constraints(ctx, ctx.max, expr, extra=extra_constraints, signed=signed)
        if result is not None:
            return result
        return _resolve_none_extremum(
            "max",
            expr,
            self.constraints,
            extra_constraints,
            signed,
            lambda: self.satisfiable(extra_constraints=extra_constraints),
        )

    def symbolic(self, expr):
        """Check if expression contains symbolic variables."""
        if isinstance(expr, claripy.ast.Base):
            return expr.symbolic
        return False

    @property
    def constraints(self):
        """Get all constraints as claripy ASTs."""
        return self._mgr.export_state_constraints(self._state_id)

    @property
    def timeout(self):
        """Z3 solver timeout in milliseconds for this state's solver context."""
        try:
            return self._mgr.get_state_solver_timeout(self._state_id)
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: state may have been GC'd or backend
            # lacks Z3 — return 0 as default. (No log: this property is read
            # frequently from solver hot paths.)
            return 0

    @timeout.setter
    def timeout(self, value):
        """Forward `state.solver.timeout = N` to the Rust solver context.

        Updates the underlying state's SymContext so future forks inherit
        the timeout, and also propagates to the proxy's already-forked
        solver (if any) so the next satisfiable()/eval() honors it.
        """
        if value is None:
            return
        timeout_ms = int(value)
        self._mgr.set_state_solver_timeout(self._state_id, timeout_ms)
        ctx = self._forked_ctx()
        if ctx is not None:
            ctx.set_timeout(timeout_ms)


class RustSolverProxy(RustSolverProxyBase):
    """
    Wraps a RustSolverContext to present a claripy-compatible solver interface.

    Delegates satisfiability checks, evaluation, and constraint operations
    directly to the Rust Z3 solver — no claripy frontend sync needed.

    Shared solver-query methods (satisfiable / min / max / symbolic /
    constraints / timeout) live on ``RustSolverProxyBase``; this subclass adds
    the lazy single-fork strategy (``_ensure_solver`` → ``self._solver_ctx``)
    and the methods whose bodies differ from the plugin's (eval / eval_upto /
    add / is_true / is_false / solution).
    """

    def __init__(self, rust_mgr, state_id, shared_ctx_getter=None):
        self._mgr = rust_mgr
        self._state_id = state_id
        self._solver_ctx = None  # lazy — forked on first access
        # angr-yodz: when set, returns the parent RustStateProxy's single
        # shared forked context so a constraint added here is visible to the
        # same proxy's memory/posix reads. None for standalone construction.
        self._shared_ctx_getter = shared_ctx_getter

    def _ensure_solver(self):
        if self._solver_ctx is None:
            if self._shared_ctx_getter is not None:
                self._solver_ctx = self._shared_ctx_getter()
            else:
                self._solver_ctx = self._mgr.fork_state_solver(self._state_id)

    def _query_ctx(self):
        self._ensure_solver()
        return self._solver_ctx

    def _forked_ctx(self):
        return self._solver_ctx

    def eval(self, expr, n=1, cast_to=None, extra_constraints=(), **kwargs):
        """Evaluate a symbolic expression to a concrete value.

        Matches angr SimSolver.eval: returns a single value (not a tuple).
        """
        self._ensure_solver()
        # Fast path: concrete expression
        if hasattr(expr, "concrete") and expr.concrete:
            val = expr.concrete_value if hasattr(expr, "concrete_value") else expr.args[0]
            return self._cast_result(expr, val, cast_to)
        result = _with_extra_constraints(self._solver_ctx, self._solver_ctx.eval, expr, extra=extra_constraints)
        if result is None:
            raise claripy.errors.UnsatError("unsat")
        return self._cast_result(expr, result, cast_to)

    def _eval_inner(self, expr, n, cast_to):
        if n == 1:
            result = self._solver_ctx.eval(expr)
            if result is None:
                raise claripy.errors.UnsatError("unsat")
            result = self._cast_result(expr, result, cast_to)
            return (result,)
        results = self._solver_ctx.eval_upto(expr, n)
        results = tuple(self._cast_result(expr, r, cast_to) for r in results)
        return results

    def eval_upto(self, expr, n, cast_to=None, extra_constraints=(), **kwargs):
        """Evaluate expression for up to n solutions. Returns a tuple."""
        self._ensure_solver()
        return _with_extra_constraints(self._solver_ctx, self._eval_inner, expr, n, cast_to, extra=extra_constraints)

    def add(self, *constraints):
        """Add constraint(s) to the solver."""
        self._ensure_solver()
        for c in constraints:
            if isinstance(c, (list, tuple)):
                for cc in c:
                    self._solver_ctx.add_constraint_ast(cc)
            else:
                self._solver_ctx.add_constraint_ast(c)

    def is_true(self, expr, **kwargs):
        """Check if expression is definitely true."""
        self._ensure_solver()
        return self._solver_ctx.is_true(expr)

    def is_false(self, expr, **kwargs):
        """Check if expression is definitely false."""
        self._ensure_solver()
        return self._solver_ctx.is_false(expr)

    def solution(self, expr, value, **kwargs):
        """Check if value is a valid solution for expr."""
        self._ensure_solver()
        return self._solver_ctx.solution(expr, value)


class RustSolverProxyPlugin(RustSolverProxyBase):
    """SimSolver-shaped plugin that routes constraint ops through Rust.

    Installed as ``state.solver`` on SimProcedure callback states under the
    ``use_callback_solver_proxy`` gate on ``RustExplorationManager``
    (angr-8oiw, write-through .3). Mirrors the angr-4scu (memory) /
    angr-qj30 (registers) install pattern.

    Write-through model:
      * ``add(constraint)`` routes immediately into the underlying Rust
        state's solver via ``add_constraints_to_state(state_id, [...])`` —
        no parallel Python claripy solver.
      * ``constraints`` reads through ``export_state_constraints(state_id)``.
      * ``eval`` / ``eval_upto`` / ``satisfiable`` / ``min`` / ``max`` /
        ``is_true`` / ``is_false`` / ``solution`` delegate to a lazily-forked
        Rust context (avoids mutating the source state's solver).

    Symbol creation methods (``BVS`` / ``BVV`` / ``Unconstrained``) and the
    variable-tracking helpers (``register_variable`` / ``get_variables`` /
    ``describe_variables``) delegate to ``claripy`` directly — those don't
    touch the solver, so there's no divergence concern. ``all_variables`` /
    ``eternal_tracked_variables`` / ``temporal_tracked_variables`` are
    maintained on the plugin so SimProcedures that read them (e.g. file
    backing for ``open()``) see consistent state.

    Plugin protocol surface (``id`` / ``state`` / ``set_state`` / ``copy`` /
    ``init_state`` / ``merge`` / ``widen``) matches the ``RustRegisterProxy`` /
    ``RustMemoryProxy`` shape: ``STRONGREF_STATE = False``, ``copy()`` returns
    an unbound proxy bound to the same Rust state_id.
    """

    STRONGREF_STATE: bool = False

    def __init__(self, rust_mgr, state_id, python_mgr=None):
        object.__setattr__(self, "_mgr", rust_mgr)
        object.__setattr__(self, "_state_id", state_id)
        # angr-7jv5: optional reference to the Python wrapper so write paths
        # can bump ``_stats_proxy_*`` counters. ``rust_mgr`` is the PyO3
        # builtin and has no Python attributes; only the wrapper does.
        object.__setattr__(self, "_python_mgr", python_mgr)
        # Lazy Rust solver fork — created on first eval/satisfiable/min/max.
        # Invalidated on add() so the next solve picks up the new constraint.
        object.__setattr__(self, "_rust_ctx_cache", None)
        # SimSolver protocol attributes.
        object.__setattr__(self, "id", "solver")
        object.__setattr__(self, "state", None)
        # Variable tracking — SimProcedures call ``register_variable`` and
        # ``get_variables``; keep the bookkeeping on the plugin so they work.
        object.__setattr__(self, "all_variables", [])
        object.__setattr__(self, "temporal_tracked_variables", {})
        object.__setattr__(self, "eternal_tracked_variables", {})
        # ``SimSolver._stored_solver`` is read by ``solver._solver`` —
        # callers occasionally poke at it; set to ``None`` so attribute
        # access doesn't raise. The proxy never uses it.
        object.__setattr__(self, "_stored_solver", None)

    # ---------------------------------------------------------------
    # Plugin protocol
    # ---------------------------------------------------------------

    @property
    def category(self):
        return "solver"

    def set_state(self, state):
        """SimStatePlugin hook — invoked on plugin install / copy."""
        object.__setattr__(self, "state", state)

    def set_strongref_state(self, _state):
        # STRONGREF_STATE=False, so dead code; defined for protocol parity.
        pass

    def init_state(self):
        # The underlying Rust state's solver is already populated by the
        # engine — nothing to initialize on the Python side.
        pass

    def copy(self, _memo=None):
        """Return a new proxy bound to the same Rust state."""
        clone = RustSolverProxyPlugin(self._mgr, self._state_id, python_mgr=self._python_mgr)
        # Carry over variable-tracking dicts — SimSolver.copy() does the
        # same so SimProc state.copy() preserves register_variable refs.
        clone.all_variables = list(self.all_variables)
        clone.temporal_tracked_variables = dict(self.temporal_tracked_variables)
        clone.eternal_tracked_variables = dict(self.eternal_tracked_variables)
        return clone

    def merge(self, _others, _merge_conditions, _common_ancestor=None):
        # Mirrors the angr-8dop.2 gap stubs on the memory / register proxies:
        # cross-state_id solver merge is out of scope for the callback gate.
        return False

    def widen(self, _others):
        return False

    def downsize(self):
        # SimSolver.downsize clears Python claripy caches; the proxy has no
        # Python solver state, so this is a no-op.
        pass

    def simplify(self, e=None):
        """``state.solver.simplify(expr)`` — delegates to claripy.

        SimSolver.simplify is a thin wrapper around ``claripy.simplify``
        when handed an AST; for non-ASTs it returns the input unchanged.
        Used by ``state.memory.store(...)``'s default-page logic and any
        SimProc that simplifies its computed result.
        """
        if e is None:
            return None
        if isinstance(e, claripy.ast.Base):
            return claripy.simplify(e)
        return e

    def reload_solver(self, constraints=None):
        # Proxy has no _stored_solver to reload — constraints live in
        # Rust. Accept the call so SimProcedures that call reload_solver
        # after add() don't crash.
        pass

    def unsat_core(self, extra_constraints=()):
        """Constraints Z3 blames for this state being UNSAT (angr-op0dn.14.2).

        Returns a subset of ``self.constraints`` — the same claripy ASTs
        ``export_state_constraints`` yields — or ``[]`` when the state is SAT,
        matching ``SimSolver.unsat_core``. ``extra_constraints`` are assumed but
        never reported, as in claripy's tracking solver.

        Unlike Python's SimSolver, tracking is not armed at add time: the Rust
        engine asserts path constraints untracked (assumption literals are too
        expensive on the fork path) and rebuilds a tracked solver from the
        assumed-constraint IR here, on demand. So the core is complete even
        though no option was set *before* execution.

        Without ``CONSTRAINT_TRACKING_IN_SOLVER`` on the bound state this keeps
        returning ``[]`` rather than raising (SimSolver raises SimSolverOptionError)
        — ``invariant-rust-solver-proxy-required-surface``: SimProcedures that
        opportunistically peek at the core must not crash.
        """
        opts = getattr(self.state, "options", None) if self.state is not None else None
        if opts is not None and "CONSTRAINT_TRACKING_IN_SOLVER" not in opts:
            return []
        extras = [self._unwrap_constraint(c) for c in extra_constraints]
        return list(self._mgr.state_unsat_core(self._state_id, extras))

    def eval_to_ast(self, e, n, extra_constraints=(), exact=None):
        """Return up to ``n`` concrete solutions as claripy BVVs."""
        if hasattr(e, "concrete") and e.concrete:
            return [e]
        values = self.eval_upto(e, n, extra_constraints=extra_constraints, exact=exact)
        return [claripy.BVV(v, len(e)) for v in values]

    # ---------------------------------------------------------------
    # Write-through: add
    # ---------------------------------------------------------------

    @staticmethod
    def _unwrap_constraint(c):
        """Strip ``SimActionObject`` wrappers and ``True``/``False`` no-ops.

        SimProcedures often pass constraints wrapped in ``SimActionObject``
        for action tracking. Stock SimSolver unwraps them via
        ``_adjust_constraint``. Python ``True`` is a tautology (no-op);
        ``False`` is UNSAT — both are returned as-is so the caller can
        decide what to do.
        """
        # Avoid an import cycle / hot-path import: SimActionObject ships
        # with angr but isn't on the import path for rust_state_proxy.
        from angr.state_plugins.sim_action_object import SimActionObject

        if isinstance(c, SimActionObject):
            return c.ast
        return c

    def add(self, *constraints):
        """Add constraint(s) to the underlying Rust state's solver.

        Write-through: each constraint lands on the Rust state by
        ``state_id`` via ``add_constraints_to_state``. ``find_state_mut`` is
        pending-aware (angr-qj30 fix), so this works during SimProc callbacks
        where the state lives in pending_callback.
        """
        ast_list = []
        for c in constraints:
            if isinstance(c, (list, tuple)):
                for cc in c:
                    ast_list.append(self._unwrap_constraint(cc))
            else:
                ast_list.append(self._unwrap_constraint(c))
        # Filter out Python-True tautologies; bail UNSAT on Python-False
        # (matches SimSolver.add's concrete-bool shortcut).
        filtered = []
        for c in ast_list:
            if c is True:
                continue
            if c is False:
                raise claripy.errors.UnsatError("attempted to add False constraint")
            filtered.append(c)
        if not filtered:
            return ast_list
        # angr-4aach: fire the ``constraints`` inspect event around the
        # write-through, matching Python's SimSolver.add. BP_BEFORE may
        # replace ``added_constraints``; re-read it before installing.
        if self._python_mgr is not None:
            overridden = self._python_mgr._cb_inspect_constraints(self._state_id, "before", added_constraints=filtered)
            if overridden is not None and overridden is not filtered:
                filtered = list(overridden)
                if not filtered:
                    return ast_list
        try:
            if self._python_mgr is not None:
                self._python_mgr._stats_proxy_solver_adds += len(filtered)
            self._mgr.add_constraints_to_state(self._state_id, filtered)
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: write-through failed; the Rust
            # state's solver did not get the constraint. Subsequent eval()
            # may return values inconsistent with the caller's expectation.
            # Re-raise so the failure is visible — silently swallowing
            # would mask the divergence.
            l.warning(
                "RustSolverProxyPlugin.add: write-through failed for state %d: %s",
                self._state_id,
                e,
            )
            raise
        if self._python_mgr is not None:
            self._python_mgr._cb_inspect_constraints(self._state_id, "after")
        # Invalidate the cached fork so the next eval picks up the new
        # constraint (the fork was cloned from the pre-add solver state).
        object.__setattr__(self, "_rust_ctx_cache", None)
        return ast_list

    # ---------------------------------------------------------------
    # Read-through: eval / satisfiable / min / max
    # ---------------------------------------------------------------

    def _get_rust_ctx(self):
        if self._rust_ctx_cache is None:
            object.__setattr__(
                self,
                "_rust_ctx_cache",
                self._mgr.fork_state_solver(self._state_id),
            )
        return self._rust_ctx_cache

    # Context-access hooks for ``RustSolverProxyBase``: the plugin caches its
    # fork in ``_rust_ctx_cache`` (invalidated on ``add``).
    def _query_ctx(self):
        return self._get_rust_ctx()

    def _forked_ctx(self):
        return self._rust_ctx_cache

    def eval(self, expr, n_or_cast=None, cast_to=None, extra_constraints=(), exact=None, **kwargs):
        """``state.solver.eval(expr[, cast_to=bytes])``.

        Matches SimSolver.eval's (expr, cast_to=...) signature — returns a
        single value, not a tuple. The second positional is interpreted as
        ``cast_to`` (SimSolver doesn't support a positional ``n``).
        """
        if cast_to is None and n_or_cast is not None and not isinstance(n_or_cast, int):
            cast_to = n_or_cast
        if hasattr(expr, "concrete") and expr.concrete:
            val = expr.concrete_value if hasattr(expr, "concrete_value") else expr.args[0]
            return self._cast_result(expr, val, cast_to)
        ctx = self._get_rust_ctx()
        result = _with_extra_constraints(ctx, ctx.eval, expr, extra=extra_constraints)
        if result is None:
            raise claripy.errors.UnsatError("unsat")
        return self._cast_result(expr, result, cast_to)

    def eval_upto(self, expr, n, cast_to=None, extra_constraints=(), exact=None, **kwargs):
        ctx = self._get_rust_ctx()
        if hasattr(expr, "concrete") and expr.concrete:
            val = expr.concrete_value if hasattr(expr, "concrete_value") else expr.args[0]
            return [self._cast_result(expr, val, cast_to)]
        results = _with_extra_constraints(ctx, ctx.eval_upto, expr, n, extra=extra_constraints)
        if cast_to is not None:
            results = [self._cast_result(expr, r, cast_to) for r in results]
        return list(results)

    def is_true(self, expr, extra_constraints=(), **kwargs):
        expr = self._unwrap_constraint(expr)
        if isinstance(expr, bool):
            return expr
        if isinstance(expr, claripy.ast.Base) and expr.op == "BoolV":
            return bool(expr.args[0])
        ctx = self._get_rust_ctx()
        return ctx.is_true(expr)

    def is_false(self, expr, extra_constraints=(), **kwargs):
        expr = self._unwrap_constraint(expr)
        if isinstance(expr, bool):
            return not expr
        if isinstance(expr, claripy.ast.Base) and expr.op == "BoolV":
            return not bool(expr.args[0])
        ctx = self._get_rust_ctx()
        return ctx.is_false(expr)

    def solution(self, expr, value, extra_constraints=(), **kwargs):
        expr = self._unwrap_constraint(expr)
        ctx = self._get_rust_ctx()
        return ctx.solution(expr, value)

    def unique(self, expr, **kwargs):
        results = self.eval_upto(expr, 2, **kwargs)
        return len(results) == 1

    def single_valued(self, e):
        """``True`` if ``e`` is concrete or value-set has cardinality 1.

        Mirrors SimSolver.single_valued in non-static mode: any symbolic
        expression is reported as not single-valued (no solver query).
        """
        if isinstance(e, (int, bytes, float, bool)):
            return True
        return not self.symbolic(e)

    # SimSolver exposes ``min_int`` / ``max_int`` as aliases for ``min`` /
    # ``max``; SimProcedures (libc/memcmp.py, mempcpy.py, …) call them as the
    # int-only convenience. Installed only on the plugin (the callback
    # ``state.solver``); the standalone ``RustSolverProxy`` keeps its prior
    # surface and does not expose them.
    min_int = RustSolverProxyBase.min
    max_int = RustSolverProxyBase.max

    # ---------------------------------------------------------------
    # Symbol creation (delegate to claripy — no solver interaction)
    # ---------------------------------------------------------------

    def BVS(self, name, size, explicit_name=False, key=None, eternal=False, **kwargs):
        """``state.solver.BVS(name, size)`` — mints a fresh claripy BVS."""
        kwargs.pop("uninitialized", None)
        kwargs.pop("inspect", None)
        kwargs.pop("events", None)
        kwargs.pop("min", None)
        kwargs.pop("max", None)
        kwargs.pop("stride", None)
        sym = claripy.BVS(name, size, explicit_name=explicit_name, **kwargs)
        if key is not None:
            self.register_variable(sym, key, eternal=eternal)
        self.all_variables.append(sym)
        return sym

    def BVV(self, value, size=None, **kwargs):
        if size is None:
            return claripy.BVV(value, **kwargs) if isinstance(value, int) else claripy.BVV(value)
        return claripy.BVV(value, size)

    def Unconstrained(self, name, bits, **kwargs):
        """Match SimSolver.Unconstrained — return a fresh BVS by default."""
        return self.BVS(name, bits, **kwargs)

    # ---------------------------------------------------------------
    # Variable tracking
    # ---------------------------------------------------------------

    def register_variable(self, v, key, eternal=True):
        if type(key) is not tuple:
            raise TypeError("Variable tracking key must be a tuple")
        if eternal:
            self.eternal_tracked_variables[key] = v
        else:
            self.temporal_tracked_variables = dict(self.temporal_tracked_variables)
            ctrkey = (*key, None)
            ctrval = self.temporal_tracked_variables.get(ctrkey, 0) + 1
            self.temporal_tracked_variables[ctrkey] = ctrval
            tempkey = (*key, ctrval)
            self.temporal_tracked_variables[tempkey] = v

    def get_variables(self, *keys):
        for k, v in self.eternal_tracked_variables.items():
            if len(k) >= len(keys) and all(x == y for x, y in zip(keys, k)):
                yield k, v
        for k, v in self.temporal_tracked_variables.items():
            if k[-1] is None:
                continue
            if len(k) >= len(keys) and all(x == y for x, y in zip(keys, k)):
                yield k, v

    def describe_variables(self, v):
        reverse_mapping = {next(iter(var.variables)): k for k, var in self.eternal_tracked_variables.items()}
        reverse_mapping.update(
            {next(iter(var.variables)): k for k, var in self.temporal_tracked_variables.items() if k[-1] is not None}
        )
        for var in v.variables:
            if var in reverse_mapping:
                yield reverse_mapping[var]


class RustRegisterProxy:
    """
    Provides `state.regs.rax`-style access by delegating to Rust.

    Register values are returned as claripy BVVs for compatibility
    with code that expects symbolic bitvectors.

    Also implements the minimum ``SimMemory`` plugin protocol (``id``,
    ``category``, ``state``, ``set_state``, ``copy``, ``init_state``,
    ``merge``, ``widen``) so that the proxy can be installed as
    ``state.registers`` on a real SimState (angr-qj30 write-through .2,
    under the ``use_callback_register_proxy`` gate on
    ``RustExplorationManager``).
    """

    # SimMemory plugin protocol: registers never carry a strong ref back
    # to the state. The proxy routes every op into Rust by ``state_id``
    # and does not need ``self.state`` to be a live reference.
    STRONGREF_STATE: bool = False
    SUPPORTS_CONCRETE_LOAD: bool = False

    def __init__(self, rust_mgr, state_id, arch, python_mgr=None):
        # Use object.__setattr__ to bypass our own __setattr__ during init
        # (which routes name= writes through to Rust). Without this, the
        # first attribute assignment below would try to look up self._mgr
        # before it exists.
        object.__setattr__(self, "_mgr", rust_mgr)
        object.__setattr__(self, "_state_id", state_id)
        object.__setattr__(self, "_arch", arch)
        object.__setattr__(self, "_cache", {})  # name -> claripy BVV/BVS
        # angr-7jv5: optional Python-wrapper handle for counter bumps.
        object.__setattr__(self, "_python_mgr", python_mgr)
        # SimMemory plugin protocol surface — stored via object.__setattr__
        # so our overridden __setattr__ does not route them through to Rust.
        object.__setattr__(self, "id", "reg")
        object.__setattr__(self, "endness", getattr(arch, "register_endness", "Iend_LE"))
        object.__setattr__(self, "state", None)

    def prefetch(self, names):
        """Batch-fetch multiple registers in one FFI call and cache them."""
        try:
            values = self._mgr.get_state_registers_batch(self._state_id, names)
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: prefetch is a best-effort
            # optimization; missing prefetch falls back to per-register
            # __getattr__ on first access.
            return
        for name, val in zip(names, values):
            width = self._get_register_width(name)
            if val is None:
                self._cache[name] = self._recover_symbolic_register_ast(name, width)
            else:
                self._cache[name] = claripy.BVV(val, width)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._cache:
            return self._cache[name]
        canonical = self._canonical_name(name)
        try:
            val = self._mgr.get_state_register(self._state_id, canonical)
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: re-raise as AttributeError so callers
            # using hasattr()/getattr() see "no such register". Note: this
            # masks transient FFI errors as missing-attribute — log debug so
            # they're visible under --debug.
            l.debug(
                "get_state_register(sid=%d, name=%r) failed; reporting as AttributeError", self._state_id, canonical
            )
            raise AttributeError(f"register '{name}' not found")
        width = self._get_register_width(canonical)
        if val is None:
            result = self._recover_symbolic_register_ast(canonical, width)
        else:
            result = claripy.BVV(val, width)
        self._cache[name] = result
        return result

    def _recover_symbolic_register_ast(self, name, width):
        """Recover Rust's claripy AST for a symbolic register (angr-4pm1).

        ``get_state_register`` returns ``None`` when the register holds a
        symbolic ``RustBV``. Previously we minted a fresh, orphan ``claripy.BVS``
        here — but the orphan symbol has no identity link to Rust's internal
        Z3 AST, so ``state.solver.add(state.regs.<reg> == K)`` constrained a
        ghost symbol and never affected the actual register value.

        Now we ask the manager for the register's claripy AST via
        ``get_state_register_ast`` (which round-trips through
        ``rustbv_to_claripy`` and preserves the underlying Z3 AST), so
        downstream solver operations land on the correct symbol. Only when
        the manager genuinely has no value for the register (unknown name or
        old build without the FFI shim) do we fall back to the orphan BVS —
        same loss-of-identity behavior as before, but at least with a debug
        log so the gap is visible.
        """
        if hasattr(self._mgr, "get_state_register_ast"):
            try:
                ast = self._mgr.get_state_register_ast(self._state_id, name)
            except (RuntimeError, ValueError, KeyError, AttributeError):
                # cat-(b) FALLBACK WITH LOSS: AST recovery failed; we mint
                # an orphan BVS below. Solver writes through that BVS will
                # not affect the actual register value.
                ast = None
            if ast is not None:
                return ast
        l.debug(
            "RustRegisterProxy: no AST recovered for symbolic register %r "
            "(sid=%d); minting orphan BVS — solver writes through the proxy "
            "will not affect this register",
            name,
            self._state_id,
        )
        return claripy.BVS(f"reg_{name}_{self._state_id}", width)

    def _get_register_width(self, name):
        """Get the bit width for a named register.

        Looks up the size in BYTES from the Rust arch table (single source of
        truth) and converts to bits. Falls back to ``arch.bits`` for registers
        Rust doesn't model — the same default the old hand-rolled prefix
        matcher used.
        """
        size_bytes = register_size_for_arch(self._arch.name, name)
        if size_bytes is not None:
            return size_bytes * 8
        return self._arch.bits

    def _canonical_name(self, name):
        """Translate an angr register alias to Rust's canonical name.

        angr exposes aliases like ``ip`` (and other ABI synonyms) that share
        an ``(offset, size)`` with their canonical register
        (e.g. ``ip`` ↔ ``rip`` on AMD64). Rust's register file is keyed by
        the canonical name only; passing ``"ip"`` to
        ``set_state_register_symbolic_ast`` would return "failed to set
        register: ip". Look up the alias's ``(offset, size)`` in
        ``arch.registers`` and translate via
        ``arch.register_size_names[(offset, size)]`` to the canonical name.
        Returns the input unchanged when no translation is required (e.g.
        ``"rip"`` itself, or names not present in ``arch.registers``).
        """
        info = self._arch.registers.get(name)
        if info is None:
            return name
        offset, size = info[0], info[1]
        try:
            return self._arch.register_size_names[(offset, size)]
        except KeyError:
            return name

    # SimMemory plugin protocol surface attribute names that must NOT route
    # through to Rust when assigned. ``state`` is set by ``set_state``,
    # ``id`` / ``endness`` are stamped on the plugin instance, etc.
    _PLUGIN_ATTRS = frozenset({"id", "endness", "state", "category"})

    def __setattr__(self, name, value):
        """Write-through register assignment (angr-j28e).

        ``proxy.regs.<name> = value`` from a state.inspect callback,
        find/avoid predicate, or external user code is forwarded immediately
        to the Rust state via ``set_state_register_symbolic_ast``. Rust is
        the single source of truth — there is no Python-side shadow store.

        ``value`` may be a claripy AST (BVV/BVS/any expression), an int, or
        ``bytes``. Ints/bytes are wrapped in a BVV at the register's native
        width before forwarding. After a successful write, the proxy cache
        is updated so the next ``proxy.regs.<name>`` read returns the new
        value without an FFI round-trip.

        Underscore-prefixed names (``_mgr``, ``_state_id``, ``_cache``,
        ``_arch``) and SimMemory plugin protocol names (``id``, ``endness``,
        ``state``, ``category``) are stored as ordinary Python attributes
        via ``object.__setattr__`` — only public register names route to Rust.
        """
        if name.startswith("_") or name in self._PLUGIN_ATTRS:
            object.__setattr__(self, name, value)
            return
        canonical = self._canonical_name(name)
        width = self._get_register_width(canonical)
        ast = self._coerce_to_ast(value, width)
        if self._python_mgr is not None:
            self._python_mgr._stats_proxy_reg_writes += 1
        self._mgr.set_state_register_symbolic_ast(self._state_id, canonical, ast)
        # angr-yxar: invalidate cached aliases whose (offset, size) overlaps
        # the write. A partial write like ``state.regs.eax = sym32`` must
        # drop any cached ``rax`` / ``ax`` / ``al`` / ``ah`` entry so a later
        # ``state.regs.rax`` read picks up the post-write value from Rust
        # rather than the stale pre-write entry. Without this, the calling-
        # convention return-value path (which clears rax then writes eax)
        # leaves rax cached at BVV(0) even though Rust now holds the
        # symbolic return value.
        overlap = _get_overlap_map(self._arch).get(canonical)
        if overlap:
            for n in overlap:
                self._cache.pop(n, None)
        else:
            self._cache.pop(canonical, None)
        # Keep the cache coherent so a subsequent __getattr__ returns the
        # AST we just wrote (matches the post-write read invariant the
        # caller would otherwise see if no cache existed). Both alias and
        # canonical entries point at the same AST so future reads on either
        # name see the just-written value.
        self._cache[name] = ast
        if canonical != name:
            self._cache[canonical] = ast

    @staticmethod
    def _coerce_to_ast(value, width):
        """Coerce a Python int / bytes / claripy AST to a claripy AST.

        The FFI shim wants a claripy AST so it can route through
        ``claripy_to_rustbv`` and register the symbol in the shared cache.
        Plain ints and bytes are wrapped in a ``BVV`` at the requested
        width — same convention as ``state.regs.<name> = 0x41`` on a
        regular SimState.
        """
        if isinstance(value, claripy.ast.Base):
            return value
        if isinstance(value, (bytes, bytearray)):
            return claripy.BVV(bytes(value), width)
        if isinstance(value, int):
            return claripy.BVV(value, width)
        raise TypeError(f"register write expects claripy AST, int, or bytes; got {type(value).__name__}")

    def load(self, reg_name_or_offset, size=None, **kwargs):
        """Load register by name or by ``(offset, size)`` tuple.

        Mirrors ``SimRegisters.load``: string names route through
        ``__getattr__``; integer offsets are resolved via the architecture's
        ``register_size_names[(offset, size)]`` map (size defaults to
        ``arch.bytes``, matching angr's SimMemory default).

        ``inspect`` / ``disable_actions`` / ``events`` kwargs are accepted
        but ignored — the proxy does not fire BPs (state.inspect raises
        NotImplementedError per docs/advanced-topics/rust_engine.rst).
        """
        if isinstance(reg_name_or_offset, str):
            return getattr(self, reg_name_or_offset)
        if isinstance(reg_name_or_offset, int):
            if size is None:
                size = self._arch.bytes
            try:
                name = self._arch.register_size_names[(reg_name_or_offset, size)]
            except KeyError as e:
                raise NotImplementedError(
                    f"no register for offset {reg_name_or_offset} size {size} on {self._arch.name}"
                ) from e
            return getattr(self, name)
        raise TypeError(f"register load expects str name or int offset, got {type(reg_name_or_offset).__name__}")

    def store(self, addr, data, size=None, **kwargs):
        """Store ``data`` into the register named/offset by ``addr``.

        Mirrors ``SimRegisters.store``: string names route through the
        write-through ``__setattr__``; integer offsets are resolved via
        ``register_size_names[(offset, size)]``. ``inspect`` /
        ``disable_actions`` / ``endness`` kwargs are accepted but ignored
        (same as ``load``).
        """
        if isinstance(addr, str):
            setattr(self, addr, data)
            return
        if isinstance(addr, int):
            if size is None:
                # When size is omitted, infer it from the data width
                # (matches SimRegisters.store's behavior for concrete ints).
                if (hasattr(data, "size") and callable(data.size)) or isinstance(data, claripy.ast.Base):
                    size = data.size() // 8
                elif isinstance(data, (bytes, bytearray)):
                    size = len(data)
                else:
                    size = self._arch.bytes
            try:
                name = self._arch.register_size_names[(addr, size)]
            except KeyError as e:
                raise NotImplementedError(f"no register for offset {addr} size {size} on {self._arch.name}") from e
            setattr(self, name, data)
            return
        raise TypeError(f"register store expects str name or int offset, got {type(addr).__name__}")

    # ---------------------------------------------------------------
    # SimMemory plugin protocol surface (angr-qj30, write-through .2)
    # ---------------------------------------------------------------

    @property
    def category(self):
        return "reg"

    def set_state(self, state):
        """SimStatePlugin hook — invoked on plugin install / copy.

        We don't keep a weakref like the base ``SimStatePlugin`` does
        because the proxy never reads back through ``self.state``; it
        routes every op directly into Rust by ``state_id``. We do stash
        the reference so downstream code that inspects ``plugin.state``
        doesn't see ``None``.
        """
        object.__setattr__(self, "state", state)

    def set_strongref_state(self, _state):
        # SimStatePlugin protocol: invoked when ``STRONGREF_STATE`` is True.
        # We keep it ``False`` (no strong refs from the proxy back to the
        # state) so this is dead code; defined only to match the surface.
        pass

    def init_state(self):
        # SimStatePlugin protocol: called once after ``register_plugin``.
        # Nothing to initialize on the Rust side — the underlying
        # ``RustSimState`` register file is already populated by the
        # engine.
        pass

    def copy(self, _memo=None):
        """Return a new proxy bound to the same Rust state.

        Mirrors ``SimMemoryMixin.copy``: returns an unbound plugin (no
        ``state`` set) of the same type. The underlying Rust state is
        shared by ``state_id`` — the proxy does not own a CoW copy;
        ``copy()`` is only meaningful here as part of ``SimState.copy()``
        plugin walk. True per-state CoW lives in angr-d1dr
        (RustStateProxy.copy()).
        """
        return RustRegisterProxy(self._mgr, self._state_id, self._arch, python_mgr=self._python_mgr)

    def merge(self, _others, _merge_conditions, _common_ancestor=None):
        # angr-8dop.2-style gap: merge / widen / compare on the proxy
        # are stubs that signal "no merge happened" (False), matching
        # SimRegNameView.merge. Real merge would require coordinating
        # the Rust register file across multiple state_ids — out of
        # scope for the callback-install gate.
        return False

    def widen(self, _others):
        return False


class RustMemoryProxy:
    """
    Provides `state.memory.load(addr, size)`-style access via Rust.

    Returns bytes as claripy BVVs for compatibility.

    Implements the minimum ``SimMemory`` plugin protocol (``id``, ``endness``,
    ``category``, ``state``, ``set_state``, ``copy``) so that the proxy can be
    installed as ``state.memory`` on a real SimState (angr-4scu step 3, under
    the ``use_callback_memory_proxy`` gate on ``RustExplorationManager``).
    """

    SUPPORTS_CONCRETE_LOAD: bool = False

    def __init__(
        self,
        rust_mgr,
        state_id,
        arch,
        *,
        endness=None,
        python_mgr=None,
        shared_ctx_getter=None,
        fallback_memory=None,
        addr_witness_cache=None,
    ):
        self._mgr = rust_mgr
        self._state_id = state_id
        self._arch = arch
        self._solver_ctx = None  # lazy — forked on first symbolic-addr load
        # angr-5rjbq: per-address concretization-witness cache, keyed by the
        # symbolic address's structural ``hash()``. The symbolic-addr STORE
        # fallback (p1s02) concretizes ``ebp+off`` to a witness and records it
        # here; a later symbolic-addr LOAD of the *structurally identical* addr
        # reuses that witness instead of asking the Rust solver for a fresh
        # (possibly different) model. Without this, an unconstrained ebp-relative
        # store lands at witness A while the read concretizes to witness B, Rust
        # has no data at B, and the copied bytes read back as zeros — the
        # flareon2015_5 residual (bd flareon-5rjbq-witness-mismatch). Bounded by
        # the number of distinct symbolic-addr stores a callback issues.
        self._addr_witness_cache = addr_witness_cache if addr_witness_cache is not None else {}
        # angr-5rjbq: the Python SimMemory that ``state.memory`` held BEFORE
        # this proxy was swapped in (the callback state's copy of the entry
        # state's DefaultMemory). Loads Rust can't satisfy (unmapped/zero
        # lazy page) fall back to this so setup-time writes — e.g. flareon's
        # ebp-relative pw symbols, stored into Python memory during harness
        # setup and never synced to Rust at the concretized address — survive
        # under the gate, matching gate-off DefaultMemory semantics.
        self._fallback_memory = fallback_memory
        # angr-yodz: shared-context getter from the parent RustStateProxy so
        # symbolic-addr concretization honors constraints added via the same
        # proxy's add_constraints. None for standalone construction.
        self._shared_ctx_getter = shared_ctx_getter
        # angr-7jv5: optional Python-wrapper handle for counter bumps.
        self._python_mgr = python_mgr
        # SimMemory plugin protocol surface.
        self.id = "mem"
        self.endness = endness or getattr(arch, "memory_endness", "Iend_BE")
        self.state = None

    @property
    def category(self):
        return "mem"

    STRONGREF_STATE = False

    def set_state(self, state):
        """SimStatePlugin hook — invoked on plugin install / copy.

        We don't keep a weakref like the base ``SimStatePlugin`` does because
        the proxy never reads back through ``self.state``; it routes every
        op directly into Rust by ``state_id``. We do stash the reference so
        downstream code that inspects ``plugin.state`` doesn't see ``None``.
        """
        self.state = state

    def set_strongref_state(self, _state):
        # SimStatePlugin protocol: invoked when ``STRONGREF_STATE`` is True.
        # We keep it ``False`` (no strong refs from the proxy back to the
        # state) so this is dead code; defined only to match the surface.
        pass

    def init_state(self):
        # SimStatePlugin protocol: called once after ``register_plugin``.
        # Nothing to initialize on the Rust side — the underlying
        # ``RustSimState`` is already populated by the engine.
        pass

    def copy(self, _memo=None):
        """Return a new proxy bound to the same Rust state.

        Mirrors ``SimMemoryMixin.copy``: returns an unbound plugin (no
        ``state`` set) of the same type. The underlying Rust state is shared
        by ``state_id`` — the proxy does not own a CoW copy; ``copy()`` is
        only meaningful here as part of ``SimState.copy()`` plugin walk.
        True per-state CoW lives in angr-d1dr (RustStateProxy.copy()).
        """
        return RustMemoryProxy(
            self._mgr,
            self._state_id,
            self._arch,
            endness=self.endness,
            python_mgr=self._python_mgr,
            fallback_memory=self._fallback_memory,
            addr_witness_cache=dict(self._addr_witness_cache),
        )

    # ---------------------------------------------------------------
    # SimMemoryMixin gap stubs (angr-8dop.2)
    #
    # Step 4 parity validation (2026-06-04) found three benches failing
    # with AttributeError on ``state.memory.permissions(...)`` /
    # ``.merge(...)`` and similar when the callback-install gate is on.
    # These stubs let bench code that calls into mprotect / merge logic
    # continue executing under the proxy. They are intentionally minimal:
    # real permission tracking and cross-state_id merge are Rust-side
    # concerns that the proxy doesn't model.
    # ---------------------------------------------------------------

    def permissions(self, addr, permissions=None, **_kwargs):
        """``state.memory.permissions(addr [, permissions])`` stub.

        Returns a fully-permissive 3-bit BVV (RWX = 7) for reads; silently
        accepts writes. Permission bits are not tracked on the Python side
        of the proxy — Rust's memory model owns enforcement. Returning the
        permissive default keeps callers like ``mprotect`` /
        ``is_bad_ptr`` / ``VirtualProtect`` from crashing; the actual page
        permissions inside Rust are unaffected because no FFI is invoked
        here.
        """
        del addr, permissions
        return claripy.BVV(7, 3)

    def merge(self, _others, _merge_conditions, _common_ancestor=None):
        """SimMemory.merge stub.

        Returns False ("no merge happened") matching the register / solver
        proxy pattern. Cross-state_id memory merge would require Rust-side
        coordination that isn't in scope for the callback-install gate.
        """
        return False

    def widen(self, _others):
        """SimMemory.widen stub — see ``merge``."""
        return False

    def compare(self, _other):
        """SimMemory.compare stub.

        Returns True ("memories considered equal"). The proxy holds no
        Python-side state to diff against; comparing two proxies that
        point at different Rust state_ids would need an FFI that walks
        both states' pages, which is out of scope for the gate.
        """
        return True

    def _ensure_solver(self):
        if self._solver_ctx is None:
            if self._shared_ctx_getter is not None:
                self._solver_ctx = self._shared_ctx_getter()
            else:
                self._solver_ctx = self._mgr.fork_state_solver(self._state_id)

    def _pin_addr_witness(self, addr, conc):
        """Pin a symbolic address to its chosen concretization on the STATE solver.

        angr-5rjbq: distinct unconstrained ebp-relative buffers
        (``ebp-0x80004``, ``ebp-0x70004``, ``ebp-0x40000`` in flareon2015_5)
        are separate claripy ASTs, so each concretizes INDEPENDENTLY via
        ``_solver_ctx.eval`` and Z3 hands back witness ``0`` for every
        unconstrained expression — every buffer ALIASES at address 0 and the
        copies clobber each other (bd flareon-5rjbq-address-witness-collision-root-cause).

        Adding ``addr == conc`` to the state solver the *native* interpreter
        also uses pins the shared base symbol (``ebp``) on the FIRST
        concretization, so subsequent proxy AND native concretizations of the
        other ebp-relative addresses resolve to distinct, non-aliasing
        witnesses — mirroring angr ``DefaultMemory`` write-address
        concretization. This is NOT export-time pre-pinning of a recovered
        model (bd constraint-export-no-pre-pin): we are committing to a write
        address we ourselves chose, on a scratch stack pointer that is not part
        of any goal constraint. Confined to the callback-memory-proxy gate, so
        gate-off runs are unaffected.
        """
        pin = addr == conc
        try:
            self._mgr.add_constraints_to_state(self._state_id, [pin])
        except Exception as e:
            l.debug(
                "RustMemoryProxy._pin_addr_witness: state pin failed for state %d: %s",
                self._state_id,
                e,
            )
        # Also pin the concretization solver the proxy itself evals against —
        # add_constraints_to_state lands on the STATE solver (what native
        # execution uses), but the proxy concretizes symbolic addresses via
        # ``self._solver_ctx`` (a fork or the shared context). Without pinning
        # that too, the NEXT proxy concretization of another ebp-relative
        # address re-picks witness 0 and re-aliases. Guarded by
        # ``_solver_ctx is not None`` — the caller has already ensured it.
        if self._solver_ctx is not None:
            with contextlib.suppress(Exception):
                self._solver_ctx.add_constraint_ast(pin)

    def load(self, addr, size=None, endness=None, **kwargs):
        """Load memory from the Rust state.

        Concrete addresses (int or concrete claripy AST) issue a direct
        FFI load. Symbolic addresses are concretized to a single solution
        under the state's constraints by forking a Rust solver context.
        Unsat addresses raise ``claripy.errors.UnsatError``.

        Routes through ``get_state_memory_ast`` (angr-8dop.1) so symbolic
        memory bytes survive the round-trip — strlen/strchr/memchr/etc.
        SimProcedures running under the callback-memory-proxy gate need
        the actual symbolic AST to build their byte-by-byte ITE chains.
        Returning a solver witness here would silently miss other matches.
        """
        if size is None:
            size = self._arch.bytes
        if isinstance(size, claripy.ast.Base):
            size = size.concrete_value
        # angr-4scu step 4: stock SimMemory.load(addr, 0) returns a 0-width
        # BV without touching memory; mirror that. Reaching the FFI with
        # size=0 panics in Z3 because zero-width BVs are invalid (see
        # ``z3-patched/src/ast/bv.rs``). ``posix/open.py`` triggers this on
        # paths where ``strlen.max_null_index == 0`` (null at offset 0).
        if size == 0:
            return claripy.BVV(0, 0)
        # angr-5rjbq: keep the ORIGINAL address (possibly symbolic) so a
        # fallback to the pre-swap Python memory concretizes it consistently
        # with its own state's constraints — the setup-time ebp pin — rather
        # than the Rust solver's independently-chosen witness.
        orig_addr = addr
        if isinstance(addr, claripy.ast.Base):
            if addr.concrete:
                addr = addr.concrete_value
            else:
                # angr-5rjbq: if a prior symbolic-addr store concretized this
                # exact address, reuse that witness so the read lands where the
                # write did (see ``_addr_witness_cache``). Otherwise ask the
                # solver for any satisfying model.
                cached = self._addr_witness_cache.get(addr.hash())
                if cached is not None:
                    addr = cached
                else:
                    self._ensure_solver()
                    resolved = self._solver_ctx.eval(addr)
                    if resolved is None:
                        raise claripy.errors.UnsatError("symbolic memory load addr is unsat")
                    # angr-5rjbq: pin the chosen witness on the state solver so
                    # the shared base symbol (e.g. ebp) stays consistent across
                    # this and every later ebp-relative concretization — both in
                    # the proxy and in native execution. Also cache it so a
                    # repeat load of the identical AST is stable.
                    self._pin_addr_witness(orig_addr, resolved)
                    self._addr_witness_cache[orig_addr.hash()] = resolved
                    addr = resolved

        ast = self._mgr.get_state_memory_ast(self._state_id, addr, size)
        if ast is None:
            fallback = self._load_from_fallback(addr, orig_addr, size, endness)
            if fallback is not None:
                return fallback
            return claripy.BVV(0, size * 8)

        # The Rust AST is laid out LSB-first (byte i of memory at bits
        # [i*8+7 : i*8]) — matches what ``set_state_memory_concrete`` /
        # ``set_state_memory_ast`` write. angr's ``memory.load()`` default
        # endness is BE (byte 0 at MSB); reverse to match. Iend_LE returns
        # the raw layout verbatim.
        if endness is None:
            endness = "Iend_BE"
        if endness == "Iend_BE":
            return ast.reversed
        return ast

    def _load_from_fallback(self, conc_addr, orig_addr, size, endness):
        """Read from the pre-swap Python memory (angr-5rjbq).

        Invoked only when ``get_state_memory_ast`` returns ``None`` — i.e. the
        Rust state has nothing mapped at the (concretized) address. Delegates
        to the Python ``SimMemory`` that ``state.memory`` held before the proxy
        was installed, recovering setup-time writes such as flareon2015_5's
        ebp-relative pw symbols that were stored into Python memory during
        harness setup and never synced to Rust at the concretized address.

        Reads at ``conc_addr`` — the address ALREADY concretized against the
        proxy's solver, which mirrors the Rust state's constraints (including
        the setup-time ebp pin). The harness setup write physically landed the
        pw bytes at that same pinned witness, and Rust inherited the identical
        pin constraint via ``add_constraints_to_state``, so the concrete
        witness the proxy chose equals where the bytes live in Python memory.
        Passing the *symbolic* ``orig_addr`` here (the prior behaviour) let the
        fallback ``SimMemory`` re-concretize against its OWN solver — but after
        the swap the pin constraint migrated to the Rust solver, so Python's
        solver picks an arbitrary, data-free witness and the read misses the
        setup bytes (bd flareon-5rjbq-ebp-cross-solver-root-cause). Returns the
        loaded AST, or ``None`` if there is no fallback memory or the read
        raised (dead/unmapped path — the caller returns the zero default).
        """
        fallback = self._fallback_memory
        if fallback is None:
            return None
        try:
            result = fallback.load(conc_addr, size, endness=endness)
        except Exception:
            # Fallback memory could not satisfy the read either (unmapped,
            # unsat address, or a plugin that doesn't accept these kwargs):
            # let the caller return the zero default, matching the prior
            # behaviour for addresses neither side has mapped.
            return None
        if result is None:
            return None
        if self._python_mgr is not None:
            self._python_mgr._stats_proxy_mem_fallback_python_load += 1
        return result

    def store(self, addr, data, endness=None, **kwargs):
        """Write-through memory store to the Rust state (angr-j28e, angr-4scu).

        ``proxy.memory.store(addr, value)`` from a state.inspect callback,
        find/avoid predicate, or external user code is forwarded immediately
        to the Rust state. Rust is the single source of truth — there is no
        Python-side shadow store.

        Address handling:

        * Concrete ``int`` or concrete claripy AST → direct FFI store.
        * Symbolic claripy AST → routed through the lazy Multi-cell path
          (``state_memory_store_symbolic_multi``). Single-solution addrs
          short-circuit to the eager concrete store; Multiple/Strided
          concretization installs per-byte Multi alternatives. TooLarge
          or Failed concretization (e.g. unconstrained symbolic addr,
          unsat constraints) raises ``NotImplementedError`` with the
          SimProcedure-hook workaround surfaced (angr-4scu step 2).

        Value handling:

        * ``int`` → wrapped in a ``BVV`` at ``size * 8`` bits (caller must
          pass ``size``).
        * ``bytes`` / ``bytearray`` → wrapped at ``len(value) * 8`` bits.
        * concrete claripy ``BVV`` → fast path through the concrete-bytes
          FFI shim.
        * symbolic claripy AST → routes through ``set_state_memory_ast``
          so the symbol is registered in the shared cache.

        Endianness defaults to ``Iend_BE`` (matches angr's ``memory.store``
        default for raw bytes); pass ``endness='Iend_LE'`` to mirror VEX's
        little-endian convention.
        """
        size = kwargs.pop("size", None)
        if endness is None:
            endness = "Iend_BE"

        if isinstance(addr, claripy.ast.Base) and not addr.concrete:
            data_ast = self._data_to_ast(data, size, endness)
            ok = self._mgr.state_memory_store_symbolic_multi(self._state_id, addr, data_ast)
            if ok:
                return
            # angr-p1s02: the lazy Multi-cell path could not route this
            # symbolic-address write — the address is unbounded/unconstrained
            # (e.g. an unconstrained ``operator new`` / malloc return pointer,
            # or an ``ebp``-relative stack slot in a blank_state whose ``ebp``
            # was concretized-and-pinned at setup) so the Multi path saw too
            # many or zero satisfying solutions. Rather than raise
            # NotImplementedError and let the SimProcedure callback die
            # (dropping a corpus find under the callback-memory-proxy gate —
            # see bd memory callback-memory-proxy-medium-differential),
            # concretize the address to a single satisfiable value and store
            # there, mirroring angr's ``DefaultMemory`` write concretization.
            if self._python_mgr is not None:
                self._python_mgr._stats_proxy_mem_symbolic_addr_fallback += 1
            # Concretize through the proxy's shared solver context (not a fresh
            # per-call fork), so the address resolves consistently with the
            # load path's concretization and any constraints added via this
            # proxy — e.g. an ``ebp`` already pinned at setup resolves to the
            # same value here as on the read side.
            self._ensure_solver()
            conc = self._solver_ctx.eval(addr)
            if conc is None:
                # Genuinely unsatisfiable address (dead path): drop the store,
                # as angr would on an unsat state.
                return
            # angr-5rjbq: remember which concrete witness this symbolic address
            # resolved to so a later load of the same address reads back the
            # bytes we're about to write, rather than concretizing to a
            # different (data-free) witness. See ``_addr_witness_cache``.
            self._addr_witness_cache[addr.hash()] = conc
            # angr-5rjbq: pin ebp (or whatever base the address shares) on the
            # state solver so a later native/proxy concretization of another
            # ebp-relative buffer cannot alias this store's witness.
            self._pin_addr_witness(addr, conc)
            # Re-issue at the concretized address through
            # ``set_state_memory_ast_automap`` (angr-5rjbq), which widens the
            # state's lazy region to cover the target page and AUTO-MAPS it
            # before storing. The plain ``set_state_memory_ast`` only auto-maps
            # pages already inside a lazy region; once ``ebp`` pins to a low
            # value the concretized buffer (e.g. flareon2015_5's ENC at 0x10000)
            # lands OUTSIDE every lazy region and the store was silently dropped,
            # so native execution hashed zeros and the found state went unsat.
            # We chose this address deliberately (concretized a symbolic store)
            # and native reads it back, so mapping it is correct — unlike a
            # garbage unconstrained pointer that nobody reads. We still swallow a
            # residual failure and drop the store rather than kill the
            # SimProcedure state (p1s02 keep-alive semantics).
            with contextlib.suppress(Exception):
                self._mgr.set_state_memory_ast_automap(self._state_id, conc, data_ast)
            return

        if isinstance(addr, claripy.ast.Base):
            addr = addr.concrete_value

        if isinstance(data, claripy.ast.Base):
            width_bits = data.length if hasattr(data, "length") else data.size()
            if data.concrete:
                value = data.concrete_value
                nbytes = width_bits // 8
                byteorder = "little" if endness == "Iend_LE" else "big"
                payload = value.to_bytes(nbytes, byteorder)
                if self._python_mgr is not None:
                    self._python_mgr._stats_proxy_mem_concrete_writes += 1
                self._store_concrete(addr, payload)
                return
            # Symbolic AST — route through the AST FFI so the symbol is
            # registered in the shared cache. Endianness on a symbolic AST
            # is the caller's responsibility (claripy ASTs don't carry an
            # endianness flag); we forward verbatim.
            if self._python_mgr is not None:
                self._python_mgr._stats_proxy_mem_ast_writes += 1
            self._store_ast(addr, data)
            return

        if isinstance(data, (bytes, bytearray)):
            payload = bytes(data)
            if endness == "Iend_LE":
                payload = payload[::-1]
            if self._python_mgr is not None:
                self._python_mgr._stats_proxy_mem_concrete_writes += 1
            self._store_concrete(addr, payload)
            return

        if isinstance(data, int):
            # angr's SimMemory.store infers a bare ``int``'s width from the
            # arch word size (e.g. asprintf writes a malloc'd pointer back via
            # ``memory.store(strp, dst, endness=memory_endness)`` with no
            # explicit size). Mirror that default so the proxy path matches the
            # tracked-writes path under the callback-memory-proxy gate.
            if size is None:
                size = self._arch.bytes
            byteorder = "little" if endness == "Iend_LE" else "big"
            payload = data.to_bytes(size, byteorder)
            if self._python_mgr is not None:
                self._python_mgr._stats_proxy_mem_concrete_writes += 1
            self._store_concrete(addr, payload)
            return

        raise TypeError(f"memory store expects claripy AST, int, or bytes; got {type(data).__name__}")

    def _store_concrete(self, addr, payload):
        """``set_state_memory_concrete`` with angr's unmapped-write semantics."""
        try:
            self._mgr.set_state_memory_concrete(self._state_id, addr, payload)
        except ValueError as exc:
            if not self._is_unmapped_store(addr, exc):
                raise
            self._mgr.set_state_memory_concrete_automap(self._state_id, addr, payload)

    def _store_ast(self, addr, ast):
        """``set_state_memory_ast`` with angr's unmapped-write semantics."""
        try:
            self._mgr.set_state_memory_ast(self._state_id, addr, ast)
        except ValueError as exc:
            if not self._is_unmapped_store(addr, exc):
                raise
            self._mgr.set_state_memory_ast_automap(self._state_id, addr, ast)

    def _is_unmapped_store(self, addr, exc):
        """Classify a failed Rust store; True means "retry with auto-map".

        Rust's ``memory_store`` errors ``Unmapped`` for a page outside every
        lazy region, which under the callback-memory-proxy gate is the *only*
        memory a SimProcedure sees — so the bare ``ValueError`` escaped into
        ``proc.execute()`` and errored the state (angr-ijwp0: it timed out
        ``unmapped_analysis``, whose whole point is a strncpy through a garbage
        pointer). Reproduce what angr's ``DefaultMemory`` would have done:

        * STRICT_PAGE_ACCESS set → raise ``SimSegfaultException`` so the state
          lands in ``errored`` with the segfault angr users match on, not an
          opaque FFI ``ValueError``.
        * otherwise → the page maps on demand, so tell the caller to retry
          through the auto-map setter.
        """
        if "unmapped memory" not in str(exc):
            return False
        from angr import sim_options
        from angr.errors import SimSegfaultException

        if sim_options.STRICT_PAGE_ACCESS in self._state_options():
            raise SimSegfaultException(addr, "write-miss") from exc
        return True

    def _state_options(self):
        """State options for this state_id (empty when there is no manager)."""
        if self._python_mgr is not None:
            return self._python_mgr.get_state_options_py(self._state_id)
        return set()

    def _data_to_ast(self, data, size, endness):
        """Coerce ``data`` to a claripy AST, applying endness for concrete
        ``int`` / ``bytes`` inputs. Used by the symbolic-address store path
        (angr-4scu step 2), which must hand the Rust Multi-cell entry point
        a single AST regardless of the Python-side input form.

        Byte-order convention: Rust's ``memory_store(addr, bv)`` lays out
        the BVV with byte ``i`` of memory equal to ``(value >> (i*8)) & 0xff``.
        The concrete-addr proxy path (``set_state_memory_concrete``) builds
        its RustBV via ``int.from_bytes(payload, 'little')`` so that
        ``payload[0]`` lands at ``addr+0``. We must mirror that for the
        symbolic-addr path — using ``claripy.BVV(bytes)`` would
        big-endian-interpret the bytes and produce the reverse layout.

        Symbolic ASTs are forwarded verbatim (endness is the caller's
        responsibility — claripy ASTs do not carry an endness flag).
        """
        if isinstance(data, claripy.ast.Base):
            return data
        if isinstance(data, (bytes, bytearray)):
            payload = bytes(data)
            if endness == "Iend_LE":
                payload = payload[::-1]
            value = int.from_bytes(payload, "little")
            return claripy.BVV(value, len(payload) * 8)
        if isinstance(data, int):
            # Bare ``int`` width defaults to the arch word size, matching
            # angr's SimMemory.store (see the concrete-store branch above).
            if size is None:
                size = self._arch.bytes
            byteorder = "little" if endness == "Iend_LE" else "big"
            payload = data.to_bytes(size, byteorder)
            value = int.from_bytes(payload, "little")
            return claripy.BVV(value, size * 8)
        raise TypeError(f"memory store expects claripy AST, int, or bytes; got {type(data).__name__}")

    def find(
        self,
        addr,
        data,
        max_search,
        *,
        default=None,
        endness=None,
        chunk_size=None,
        max_symbolic_bytes=None,
        condition=None,
        char_size=1,
        **kwargs,
    ):
        """Search memory at ``addr`` for the byte pattern ``data``.

        Matches angr ``SimMemory.find()`` return shape:
        ``(result_addr, constraints, match_indices)``. For a symbolic-byte
        haystack the result is an ``ite_cases`` chain across every
        SAT-able candidate index (mirrors ``SmartFindMixin.find`` —
        strlen/strchr/memchr rely on multi-index ITE shape so their
        ``max(i)`` / ``add_constraints(Or(...))`` calls do the right thing).

        Supported surface:

        * ``addr`` — concrete int or concrete claripy AST. Symbolic addrs are
          resolved via the forked solver context (one solution).
        * ``data`` — concrete ``bytes``/``bytearray``/``int`` or concrete
          claripy BVV. Symbolic needles raise ``NotImplementedError``.
        * ``char_size`` — must be ``1`` (no wide-char search).
        * ``condition`` — must be ``None`` or syntactically true.

        Symbolic-byte handling (angr-8dop.1): when the haystack contains any
        symbolic byte, each candidate position contributes an ITE case
        ``(haystack[i:i+needle_len] == needle, addr+i)``. Concrete-only
        haystacks short-circuit on the first equality and return a single
        BVV result for parity with the pre-symbolic fast path.
        """
        if max_search is None or (isinstance(max_search, int) and max_search <= 0):
            zero = claripy.BVV(default or 0, self._arch.bits)
            return zero, [], []
        if isinstance(max_search, claripy.ast.Base):
            if not max_search.concrete:
                raise NotImplementedError(
                    "RustMemoryProxy.find(): symbolic max_search is not "
                    "supported. Caller must concretize max_search before "
                    "invoking find() on the proxy."
                )
            max_search = max_search.concrete_value

        if char_size != 1:
            raise NotImplementedError(
                "RustMemoryProxy.find() supports char_size=1 only (wide-char search lives in SimMemory.find())."
            )
        if condition is not None and not (hasattr(condition, "is_true") and condition.is_true()):
            raise NotImplementedError(
                "RustMemoryProxy.find() does not support a symbolic "
                "``condition=`` kwarg. Drop the condition or route through "
                "a SimProcedure hook."
            )

        if isinstance(addr, claripy.ast.Base):
            if addr.concrete:
                addr = addr.concrete_value
            else:
                self._ensure_solver()
                resolved = self._solver_ctx.eval(addr)
                if resolved is None:
                    raise claripy.errors.UnsatError("symbolic find addr is unsat")
                addr = resolved

        if isinstance(data, claripy.ast.Base):
            if not data.concrete:
                raise NotImplementedError(
                    "RustMemoryProxy.find(): symbolic needles are not "
                    "supported. Use a SimProcedure hook so state.memory is "
                    "a full SimMemory plugin."
                )
            width_bits = data.size()
            if width_bits % 8:
                raise ValueError(f"needle width must be a multiple of 8 bits, got {width_bits}")
            needle = data.concrete_value.to_bytes(width_bits // 8, "big")
        elif isinstance(data, (bytes, bytearray)):
            needle = bytes(data)
        elif isinstance(data, int):
            needle = bytes((data & 0xFF,))
        else:
            raise TypeError(f"find() needle must be claripy AST, bytes, or int; got {type(data).__name__}")

        needle_len = len(needle)
        if needle_len == 0:
            # Empty needle matches at offset 0 by convention.
            return claripy.BVV(addr, self._arch.bits), [], [0]

        haystack_size = max_search + needle_len - 1
        haystack_ast = self._mgr.get_state_memory_ast(self._state_id, addr, haystack_size)
        if haystack_ast is None:
            # angr-ric3: the wide load fails on mixed symbolic+concrete or
            # multi-page ranges (Rust ``state.memory_load`` returns Err when
            # any byte in the requested range is in a lazy region or has a
            # symbolic store interleaved with concrete bytes). Fall back to
            # per-byte loads and Concat in LSB-first order so byte i lands at
            # bits [i*8 : i*8+7] — matches the layout the eager wide load
            # would have produced and what the sub-AST extractor below
            # assumes. Without this fallback ``find()`` would otherwise scan
            # an all-zero haystack and match the needle at offset 0 whenever
            # the first byte equals 0, returning the wrong answer for
            # symbolic-buffer strlen/strchr/memchr callers.
            byte_asts = []
            for i in range(haystack_size):
                b = self._mgr.get_state_memory_ast(self._state_id, addr + i, 1)
                if b is None:
                    b = claripy.BVV(0, 8)
                byte_asts.append(b)
            haystack_ast = claripy.Concat(*reversed(byte_asts))

        # Layout: Rust stores byte i at bit positions [i*8 : i*8+7] (LSB-first
        # — matches set_state_memory_concrete). Needle is matched MSB-first in
        # memory address order, so the candidate sub-AST at index i extracts
        # bits [(i+needle_len)*8-1 : i*8] and we compare against the needle
        # interpreted with int.from_bytes(needle, 'little'). For a concrete
        # haystack this reduces to a byte-wise == check (the optimizer folds
        # constants).
        needle_val = int.from_bytes(needle, "little")
        needle_ast = claripy.BVV(needle_val, needle_len * 8)

        cases = []
        match_indices = []
        default_bv = (
            claripy.BVV(default, self._arch.bits)
            if isinstance(default, int)
            else (default if default is not None else claripy.BVV(0, self._arch.bits))
        )
        max_iter = min(max_search, (haystack_ast.length // 8) - needle_len + 1)
        for i in range(max_iter):
            bit_hi = (i + needle_len) * 8 - 1
            bit_lo = i * 8
            sub_ast = haystack_ast[bit_hi:bit_lo]
            eq = sub_ast == needle_ast
            if hasattr(eq, "is_false") and eq.is_false():
                continue
            match_indices.append(i)
            match_addr = claripy.BVV(addr + i, self._arch.bits)
            cases.append((eq, match_addr))
            if hasattr(eq, "is_true") and eq.is_true():
                break

        if not match_indices:
            return default_bv, [], []

        # Stock SmartFindMixin behaviour: if the last case is is_true, treat
        # it as the unconditional default. Otherwise (no concrete match at the
        # end, default=None) emit an Or(...) constraint so the caller adds it.
        constraints = []
        if cases and hasattr(cases[-1][0], "is_true") and cases[-1][0].is_true():
            default_bv = cases.pop(-1)[1]
        elif default is None:
            constraints.append(claripy.Or(*(c for c, _ in cases)))

        result = claripy.ite_cases(cases, default_bv)
        return result, constraints, match_indices


class RustHeapProxy:
    """Proxy for state.heap — exposes mmap_base and allocation metadata
    backed by RustSimState.

    The Rust state owns mmap_base (used by the native mmap syscall handler)
    and a HeapMetadata struct tracking malloc/free regions. Reads/writes go
    through PyO3 accessors so Python user code sees the live Rust value.
    """

    def __init__(self, rust_mgr, state_id):
        self._mgr = rust_mgr
        self._state_id = state_id

    @property
    def mmap_base(self):
        """Current mmap base pointer (mirrors `state.heap.mmap_base`)."""
        try:
            return self._mgr.get_state_mmap_base(self._state_id)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: returning None on FFI error mirrors
            # angr's behavior when state.heap is unavailable.
            l.debug("get_state_mmap_base(sid=%d) failed: %s: %s", self._state_id, type(e).__name__, e)
            return None

    @mmap_base.setter
    def mmap_base(self, value):
        self._mgr.set_state_mmap_base(self._state_id, int(value))

    @property
    def allocations(self):
        """List of (addr, size) for currently-live heap allocations."""
        try:
            allocated, _freed = self._mgr.get_state_heap_metadata(self._state_id)
            return list(allocated)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: empty list when heap metadata is
            # unavailable (e.g., state already cleaned up).
            l.debug("get_state_heap_metadata(sid=%d) failed: %s: %s", self._state_id, type(e).__name__, e)
            return []

    @property
    def freed(self):
        """List of freed addresses in free-call order."""
        try:
            _allocated, freed = self._mgr.get_state_heap_metadata(self._state_id)
            return list(freed)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: empty list when heap metadata is
            # unavailable (e.g., state already cleaned up).
            l.debug("get_state_heap_metadata(sid=%d) failed: %s: %s", self._state_id, type(e).__name__, e)
            return []


class RustScratchProxy:
    """Read-only proxy for `state.scratch` over a Rust state.

    Python's SimStateScratch carries per-block transient values used by the
    Python VEX engine (irsb, bbl_addr, ins_addr, stmt_idx, jumpkind, temps,
    tyenv, ...). The Rust engine doesn't store most of those because they
    belong to the in-flight VEXInterpreter, not the persistent state.

    This proxy exposes the subset that *is* recoverable from the Rust state:
    the most recently entered block address (mirrors `state.pc`) and the
    jumpkind that led to the current state (last detailed-history entry).
    The rest (irsb, temps, tyenv, stmt_idx) are exposed as None — the Rust
    interpreter clears its per-block buffers between blocks, so there are no
    stable post-block values to read.
    """

    # Mirrors angr's Ijk_* string convention. Indices match the u8 values
    # produced by RustSimState::detailed_history (native/angr/src/state/mod.rs).
    _JUMPKIND_NAMES = (
        "Ijk_Boring",
        "Ijk_Call",
        "Ijk_Ret",
        "Ijk_Sys_syscall",
        "Ijk_Other",
    )

    def __init__(self, rust_mgr, state_id):
        self._mgr = rust_mgr
        self._state_id = state_id

    @property
    def bbl_addr(self):
        """Address of the most recently entered block (mirrors state.pc)."""
        try:
            return self._mgr.get_state_pc_by_id(self._state_id)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: best-effort read; None mirrors
            # SimStateScratch's pre-block default.
            l.debug("get_state_pc_by_id(sid=%d) failed: %s: %s", self._state_id, type(e).__name__, e)
            return None

    @property
    def ins_addr(self):
        """Address of the most recently executed instruction.

        The Rust interpreter tracks this in VEXInterpreter.current_insn_addr
        but doesn't persist it on the state — once the block finishes, the
        interpreter is dropped. We surface state.pc as a best-effort proxy:
        for a found state, pc is the find address (the last IMark seen).
        """
        return self.bbl_addr

    @property
    def jumpkind(self):
        """Jumpkind that led to the current state, e.g. "Ijk_Boring".

        Pulled from the last entry of detailed_history. Returns None for a
        freshly-created state with no recorded transitions.
        """
        try:
            history = self._mgr.get_state_detailed_history(self._state_id)
        except (RuntimeError, AttributeError, KeyError) as e:
            # cat-(b) FALLBACK WITH LOSS: history fetch failed; caller sees
            # jumpkind=None as if there were no recorded transitions.
            l.debug("get_state_detailed_history(sid=%d) failed: %s: %s", self._state_id, type(e).__name__, e)
            return None
        if not history:
            return None
        _addr, kind_u8, _target = history[-1]
        if 0 <= kind_u8 < len(self._JUMPKIND_NAMES):
            return self._JUMPKIND_NAMES[kind_u8]
        return "Ijk_Other"

    # SimStateScratch attributes the Rust engine doesn't persist between
    # blocks. Returning None matches how Python's plugin reads them before
    # the first block executes.
    @property
    def irsb(self):
        return None

    @property
    def stmt_idx(self):
        return None

    @property
    def temps(self):
        return []

    @property
    def tyenv(self):
        return None

    @property
    def sim_procedure(self):
        return None


class RustHistoryProxy:
    """Provides state.history.recent_bbl_addrs and similar."""

    # Tail length used when materializing recent_bbl_addrs. Matches the
    # spirit of angr's SimStateHistory.recent_bbl_addrs ("most recent" rather
    # than the full lineage chain). Long-running benches accumulate 100k+ bbl
    # entries; cloning the full Vec across FFI was the regression risk
    # called out in angr-kwpi.1.
    _RECENT_TAIL_DEFAULT = 256

    def __init__(self, rust_mgr, state_id):
        self._mgr = rust_mgr
        self._state_id = state_id
        self._bbl_addrs = None

    @property
    def recent_bbl_addrs(self):
        if self._bbl_addrs is None:
            tail = self._mgr.get_state_bbl_history_tail(
                self._state_id,
                self._RECENT_TAIL_DEFAULT,
            )
            self._bbl_addrs = list(tail) if tail is not None else []
        return self._bbl_addrs

    @property
    def bbl_addrs(self):
        return self.recent_bbl_addrs

    @property
    def block_count(self):
        return len(self.recent_bbl_addrs)


class RustPosixProxy:
    """
    Provides state.posix.dumps(fd) for stdin/stdout extraction.

    For stdout (fd=1): reads from a Rust-side or Python-side buffer
    of accumulated write/puts/printf output.

    For stdin (fd=0): evaluates stdin symbolic variables under the
    state's constraints to extract the concrete input.
    """

    def __init__(self, rust_mgr, state_id, stdin_vars=None, stdout_data=None, shared_ctx_getter=None):
        self._mgr = rust_mgr
        self._state_id = state_id
        self._stdin_vars = stdin_vars or []  # list of (claripy BVS, offset) for stdin
        self._stdout_data = stdout_data or b""  # accumulated stdout bytes
        # angr-yodz: shared-context getter from the parent RustStateProxy so
        # dumps(0) honors constraints added via the same proxy's
        # add_constraints. None for standalone construction.
        self._shared_ctx_getter = shared_ctx_getter

    def dumps(self, fd):
        """Dump file descriptor contents."""
        if fd == 0:
            # stdin — evaluate symbolic variables under constraints
            return self._eval_stdin()
        if fd == 1:
            # stdout — return accumulated output (cached on proxy init)
            return self._stdout_data
        # Other fds (stderr, opened files) — query Rust engine
        try:
            return bytes(self._mgr.get_state_fd_output(self._state_id, fd))
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: empty bytes when the requested
            # fd has no Rust-side buffer. Tools that expect specific
            # output should distinguish "no buffer" from "empty buffer";
            # log so they're visible under --debug.
            l.debug("get_state_fd_output(sid=%d, fd=%d) failed: %s: %s", self._state_id, fd, type(e).__name__, e)
            return b""

    def _eval_stdin(self):
        """Evaluate stdin symbolic variables to concrete bytes."""
        if not self._stdin_vars:
            return b""
        try:
            if self._shared_ctx_getter is not None:
                solver_ctx = self._shared_ctx_getter()
            else:
                solver_ctx = self._mgr.fork_state_solver(self._state_id)
            result = bytearray()
            for var, _offset in sorted(self._stdin_vars, key=lambda x: x[1]):
                val = solver_ctx.eval(var)
                if val is not None:
                    # Convert to bytes
                    byte_len = max(1, var.length // 8)
                    result.extend(val.to_bytes(byte_len, "big"))
            return bytes(result)
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: returning empty bytes when stdin eval
            # fails could be misread as "the input was empty" by callers
            # comparing solution bytes. Logged at warn so the failure is loud.
            l.warning("RustPosixProxy: failed to evaluate stdin: %s", e)
            return b""


class RustCallStackFrameProxy:
    """A single frame view backed by a Rust CallStackEntry tuple.

    The Rust engine returns frames as (call_site_addr, callee_addr, return_addr,
    stack_ptr) tuples. This proxy presents the same attribute names as
    angr's CallStack plugin so user code can read frames uniformly.
    """

    __slots__ = (
        "_index",
        "_owner",
        "call_site_addr",
        "func_addr",
        "ret_addr",
        "stack_ptr",
    )

    def __init__(self, frame_tuple, index, owner):
        call_site_addr, callee_addr, return_addr, stack_ptr = frame_tuple
        self.call_site_addr = call_site_addr
        self.func_addr = callee_addr
        self.ret_addr = return_addr
        self.stack_ptr = stack_ptr
        self._index = index
        self._owner = owner

    @property
    def current_function_address(self):
        return self.func_addr

    @property
    def current_return_target(self):
        return self.ret_addr

    @property
    def current_stack_pointer(self):
        return self.stack_ptr

    @property
    def jumpkind(self):
        return "Ijk_Call"

    @property
    def next(self):
        """Walk one frame down the stack (toward the bottom)."""
        next_index = self._index + 1
        if next_index >= len(self._owner._frames):
            return None
        return RustCallStackFrameProxy(self._owner._frames[next_index], next_index, self._owner)

    def __repr__(self):
        return f"<RustCallStackFrame func=0x{self.func_addr:x} ret=0x{self.ret_addr:x} sp=0x{self.stack_ptr:x}>"


class RustCallStackProxy:
    """Iterable callstack view of a Rust state.

    Frames are stored as a list with the most-recent (top) frame first,
    matching angr's CallStack iteration order. The Rust engine stores
    frames in push order (top last), so we reverse on construction.
    """

    def __init__(self, rust_mgr, state_id):
        self._mgr = rust_mgr
        self._state_id = state_id
        self._frames_cache = None

    @property
    def _frames(self):
        if self._frames_cache is None:
            try:
                raw = self._mgr.get_state_call_stack(self._state_id)
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: empty callstack when FFI fails.
                # Distinguishable from a real empty stack only via the log.
                l.debug("get_state_call_stack(sid=%d) failed: %s: %s", self._state_id, type(e).__name__, e)
                raw = []
            # Rust pushes onto the end → most recent is last → reverse.
            self._frames_cache = list(reversed(raw))
        return self._frames_cache

    def __iter__(self):
        for i, frame in enumerate(self._frames):
            yield RustCallStackFrameProxy(frame, i, self)

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, k):
        if k < 0:
            k += len(self._frames)
        if k < 0 or k >= len(self._frames):
            raise IndexError(k)
        return RustCallStackFrameProxy(self._frames[k], k, self)

    @property
    def top(self):
        if not self._frames:
            return None
        return RustCallStackFrameProxy(self._frames[0], 0, self)

    @property
    def current_function_address(self):
        if not self._frames:
            return 0
        return self._frames[0][1]  # callee_addr

    @property
    def current_return_target(self):
        if not self._frames:
            return 0
        return self._frames[0][2]  # return_addr

    @property
    def current_stack_pointer(self):
        if not self._frames:
            return 0
        return self._frames[0][3]  # stack_ptr

    @property
    def func_addr(self):
        return self.current_function_address

    @property
    def ret_addr(self):
        return self.current_return_target

    @property
    def stack_ptr(self):
        return self.current_stack_pointer

    @property
    def call_site_addr(self):
        if not self._frames:
            return 0
        return self._frames[0][0]

    def __repr__(self):
        return f"<RustCallStackProxy depth={len(self)}>"


class RustCallStackProxyPlugin:
    """SimStatePlugin-shaped proxy that reads frames from Rust on demand.

    Installed as ``state.callstack`` on SimProcedure callback states under
    the ``use_callback_callstack_proxy`` gate on
    ``RustExplorationManager`` (angr-6o9p, write-through .4). Mirrors the
    angr-4scu (memory) / angr-qj30 (registers) / angr-8oiw (solver)
    install pattern.

    Read-through model:
      * Iteration / indexing / ``len()`` route through
        ``get_state_call_stack(state_id)`` — no Python-side mirror of the
        Rust call frames.
      * Top-frame attribute access (``func_addr`` / ``stack_ptr`` /
        ``ret_addr`` / ``call_site_addr`` / ``current_function_address``
        / ``current_stack_pointer`` / ``current_return_target``) reads
        the most-recent Rust frame.
      * ``next`` walks one frame down (returns
        :class:`RustCallStackFrameProxy` or ``None`` at the bottom),
        matching the linked-list shape Python code expects.

    Per-frame Python-only metadata (``locals`` / ``block_counter`` /
    ``procedure_data`` / ``invoke_return_variable``) lives on the
    plugin itself — SimProcedures stash continuation data here.

    ``push`` / ``pop`` / ``call`` / ``ret`` are gap stubs that log at
    debug level and return ``self``. SimProcedures don't manually push
    or pop call frames in practice (the engine drives that), so the
    stub keeps any pathological caller alive without diverging the
    Rust-side stack. Plugin protocol (``id`` / ``state`` /
    ``set_state`` / ``copy`` / ``init_state`` / ``merge`` / ``widen``)
    matches the existing proxy plugins: ``STRONGREF_STATE = False``,
    ``copy()`` returns an unbound proxy bound to the same Rust state_id.
    """

    STRONGREF_STATE: bool = False

    def __init__(self, rust_mgr, state_id):
        import collections as _collections

        self._mgr = rust_mgr
        self._state_id = state_id
        # SimStatePlugin protocol attrs.
        self.id = "callstack"
        self.state = None
        # Per-frame Python-only metadata; SimProcedure continuations and
        # block-counter / ``locals`` users read these even when they don't
        # touch the Rust-side frames.
        self.block_counter = _collections.Counter()
        self.procedure_data = None
        self.locals = {}
        self.invoke_return_variable = None
        self.jumpkind = "Ijk_Call"

    @property
    def category(self):
        return "callstack"

    # ---------------------------------------------------------------
    # Plugin protocol
    # ---------------------------------------------------------------

    def set_state(self, state):
        self.state = state

    def set_strongref_state(self, _state):
        pass

    def init_state(self):
        pass

    def copy(self, _memo=None):
        import collections as _collections

        clone = RustCallStackProxyPlugin(self._mgr, self._state_id)
        clone.block_counter = _collections.Counter(self.block_counter)
        clone.procedure_data = self.procedure_data
        clone.locals = dict(self.locals)
        clone.invoke_return_variable = self.invoke_return_variable
        return clone

    def merge(self, _others, _merge_conditions, _common_ancestor=None):
        # Cross-state_id callstack merge is out of scope for the
        # callback-install gate. Mirrors the solver / memory / register
        # proxy stubs.
        return False

    def widen(self, _others):
        return False

    # ---------------------------------------------------------------
    # Frame reads (no Python-side cache; each access re-reads from
    # Rust so the plugin stays in sync with concurrent Rust-side
    # pushes during the same callback).
    # ---------------------------------------------------------------

    @property
    def _frames(self):
        try:
            raw = self._mgr.get_state_call_stack(self._state_id)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: empty stack when FFI fails;
            # distinguishable from a real empty stack only via the log.
            l.debug(
                "RustCallStackProxyPlugin.get_state_call_stack(sid=%d) failed: %s: %s",
                self._state_id,
                type(e).__name__,
                e,
            )
            return []
        # Rust pushes onto the end (most recent last). Reverse so the
        # top frame is first, matching angr's CallStack iteration order.
        return list(reversed(raw))

    def __iter__(self):
        frames = self._frames
        # Yield the plugin itself as the top "node" so callers that
        # expect each iteration step to behave like a CallStack node
        # (``frame.func_addr`` / ``frame.next``) get back something
        # whose top attributes are populated. Frames below the top are
        # ``RustCallStackFrameProxy`` instances.
        if not frames:
            return
        yield self
        owner = _StaticFrameOwner(frames)
        for i in range(1, len(frames)):
            yield RustCallStackFrameProxy(frames[i], i, owner)

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, k):
        frames = self._frames
        if k < 0:
            k += len(frames)
        if k < 0 or k >= len(frames):
            raise IndexError(k)
        if k == 0:
            return self
        return RustCallStackFrameProxy(
            frames[k],
            k,
            _StaticFrameOwner(frames),
        )

    def __repr__(self):
        return f"<RustCallStackProxyPlugin depth={len(self)}>"

    @property
    def top(self):
        # CallStack.top returns ``self`` (the top frame IS the
        # CallStack object). Mirror that shape.
        return self

    @property
    def next(self):
        """Walk one frame down toward the bottom of the stack.

        Returns a :class:`RustCallStackFrameProxy` for frame[1] or
        ``None`` if the stack has fewer than two frames. Matches the
        linked-list ``CallStack.next`` accessor.
        """
        frames = self._frames
        if len(frames) < 2:
            return None
        return RustCallStackFrameProxy(
            frames[1],
            1,
            _StaticFrameOwner(frames),
        )

    @property
    def current_function_address(self):
        frames = self._frames
        if not frames:
            return 0
        return frames[0][1]  # callee_addr

    @property
    def current_return_target(self):
        frames = self._frames
        if not frames:
            return 0
        return frames[0][2]  # return_addr

    @property
    def current_stack_pointer(self):
        frames = self._frames
        if not frames:
            return 0
        return frames[0][3]  # stack_ptr

    @property
    def func_addr(self):
        return self.current_function_address

    @property
    def ret_addr(self):
        return self.current_return_target

    @property
    def stack_ptr(self):
        return self.current_stack_pointer

    @property
    def call_site_addr(self):
        frames = self._frames
        if not frames:
            return 0
        return frames[0][0]

    # ---------------------------------------------------------------
    # push / pop / call / ret — gap stubs.
    #
    # SimProcedures don't manually push or pop call frames in
    # practice; the Rust interpreter drives stack changes via call /
    # return jumpkinds. Returning ``self`` lets any opportunistic
    # caller continue without crashing while preventing a diverging
    # Python-side mirror of the Rust frames.
    # ---------------------------------------------------------------

    def push(self, _cf):
        l.debug("RustCallStackProxyPlugin.push() is a no-op stub")
        return self

    def pop(self):
        l.debug("RustCallStackProxyPlugin.pop() is a no-op stub")
        return self

    def call(self, _callsite_addr, _addr, _retn_target=None, _stack_pointer=None):
        l.debug("RustCallStackProxyPlugin.call() is a no-op stub")
        return self

    def ret(self, _retn_target=None):
        l.debug("RustCallStackProxyPlugin.ret() is a no-op stub")
        return self

    def _manage(self):
        """No-op stub for ``CallStack._manage``.

        Called by ``angr.engines.successors.add_successor`` to push /
        pop frames based on the state's jumpkind. The Rust interpreter
        already maintains the canonical call stack; mirroring the
        push/pop on the Python proxy would diverge state, so this is
        intentionally inert.
        """
        return None

    def stack_suffix(self, context_sensitivity_level):
        """Generate a tuple-form stack suffix from Rust frames.

        Mirrors ``CallStack.stack_suffix``: returns a tuple of
        ``(call_site_addr, func_addr)`` pairs (outer-first) of length
        ``2 * context_sensitivity_level``, padded with ``None`` when
        the stack is shorter.
        """
        ret: tuple = ()
        for frame in self:
            if len(ret) >= context_sensitivity_level * 2:
                break
            ret = (frame.call_site_addr, frame.func_addr, *ret)
        while len(ret) < context_sensitivity_level * 2:
            ret = (None, None, *ret)
        return ret


class _StaticFrameOwner:
    """Tiny adapter so :class:`RustCallStackFrameProxy` can walk a
    snapshot list captured at iteration / index time."""

    __slots__ = ("_frames",)

    def __init__(self, frames):
        self._frames = frames


_INSPECT_NOT_IMPLEMENTED_MSG = (
    "state.inspect breakpoints are not dispatched by the Rust symex engine. "
    "Registering a breakpoint here would silently never fire. "
    "Switch to the Python engine (use_rust_engine=False) for breakpoint-driven "
    "analyses. See docs/advanced-topics/rust_engine.rst for details."
)

_COPY_NEEDS_MANAGER_MSG = (
    "RustStateProxy.copy() requires the high-level RustExplorationManager "
    "to provide the per-state metadata seeding helper. This proxy was "
    "constructed without a python_mgr — typically only happens in low-level "
    "unit tests that build a proxy directly against `_RustExplorationManager`. "
    "Wire a RustExplorationManager (or pass python_mgr=) before calling copy()."
)


# Single source of truth for the Rust engine's state.inspect dispatch.
#
# Each entry maps an angr `event_types` member to a spec dict with:
#   - bit: position in the Rust callbacks `inspect_enabled` u32 bitmask
#   - attrs: SimInspector attribute names this event populates
#   - when_fired: 'before' or 'after' — when in the Rust pipeline it fires
#   - dispatch_origin: 'rust' (default) when the Rust engine invokes the
#     callback via PythonCallbacks; 'python' when dispatch is fired from
#     within an existing Python-side callback handler in
#     `rust_callback_dispatch.py` / `rust_manager.py`. Python-dispatched
#     events do not need a PythonCallbacks slot because no Rust code path
#     ever calls into the inspect callback for them.
#
# Bits 0..=5 mirror `crate::state::InspectEvent` ordering; bit 4 (fork) is
# reserved for future wiring; bits 6 and 7 are custom (instruction / irsb)
# with no InspectionManager enum slot in Rust; bits 8 and 9 are custom for
# call / return (angr-4ai9 widened the bitmask from u8 to u16 to make
# room — the InspectEvent enum is unchanged). Bits 10..=12 are
# Python-dispatched (simprocedure / syscall / dirty) — angr-xmfj wired
# them from the existing Python callback handlers in
# `rust_callback_dispatch.py` / `rust_manager._cb_dirty_call`.
#
# Wiring a new Rust-origin event requires (mirror the angr-d46u 5-touchpoint
# pattern):
#   1. Add a row here with a unique bit, attrs, when_fired.
#   2. Add a PythonCallbacks slot + setter + dispatch helper in
#      native/angr/src/callbacks.rs and the interpreter dispatch
#      helpers.
#   3. Instrument the corresponding interpreter site in
#      native/angr/src/interpreter/.
#   4. Add a `_cb_inspect_<name>` method on RustExplorationManager
#      that builds the attrs dict and calls `_dispatch_inspect_event`.
#   5. Register the callback in RustExplorationManager.set_callbacks().
#
# For a Python-dispatched event (`dispatch_origin: 'python'`), steps 2/3
# collapse to "call self._cb_inspect_<name>(...) from the existing Python
# handler site". No Rust changes are needed.
#
# The CI test `test_inspect_allowlist_complete_and_consistent` enforces
# that every event in `angr.state_plugins.inspect.event_types` is either
# present here with a `_cb_inspect_<name>` method, or registering a
# breakpoint for it raises NotImplementedError. No silent pass-through.
_INSPECT_EVENT_SPECS: dict = {
    "mem_read": {
        "bit": 0,
        "attrs": (
            "mem_read_address",
            "mem_read_length",
            "mem_read_expr",
            "mem_read_condition",
            "mem_read_endness",
        ),
        "when_fired": "after",
    },
    "mem_write": {
        "bit": 1,
        "attrs": (
            "mem_write_address",
            "mem_write_length",
            "mem_write_expr",
            "mem_write_condition",
            "mem_write_endness",
        ),
        "when_fired": "after",
    },
    "reg_read": {
        "bit": 2,
        "attrs": (
            "reg_read_offset",
            "reg_read_length",
            "reg_read_expr",
            "reg_read_condition",
            "reg_read_endness",
        ),
        "when_fired": "after",
    },
    "reg_write": {
        "bit": 3,
        "attrs": (
            "reg_write_offset",
            "reg_write_length",
            "reg_write_expr",
            "reg_write_condition",
            "reg_write_endness",
        ),
        "when_fired": "after",
    },
    # angr-ysml: fork dispatch fires from `exploration/stepping.rs` for
    # each forked state created by the deferred-fork processing in
    # `handle_block_end` and `process_deferred_forks_into`. Matches
    # Python's `engines/successors.py:203` where `state._inspect("fork",
    # BP_AFTER)` fires on the newly-added successor after constraints +
    # ip are applied. The BP fires BEFORE the Rust-side satisfiability
    # check so UNSAT-pruned forks still surface — same pre-discard
    # intent as Python's add_successor flow. Has NO attrs in
    # `inspect_attributes` (verified against state_plugins/inspect.py);
    # the BP just sees the forked state's id via the proxy.
    "fork": {
        "bit": 4,
        "attrs": (),
        "when_fired": "after",
    },
    "exit": {
        "bit": 5,
        "attrs": (
            "exit_target",
            "exit_guard",
            "exit_jumpkind",
        ),
        "when_fired": "before",
    },
    "instruction": {
        "bit": 6,
        "attrs": ("instruction",),
        "when_fired": "before",
    },
    "irsb": {
        "bit": 7,
        "attrs": ("address",),
        "when_fired": "before",
    },
    # angr-4ai9: call/return dispatched from the BlockEnd path of
    # interpreter/execution.rs (Ijk_Call / Ijk_Ret). Fires `before` and
    # `after` around the Rust call_stack push/pop, matching Python
    # callstack.py:386/419 (call) and :430/432 (return). Only attribute is
    # `function_address` (the resolved call target on call; the popped
    # frame's callee on return).
    "call": {
        "bit": 8,
        "attrs": ("function_address",),
        "when_fired": "before",
    },
    "return": {
        "bit": 9,
        "attrs": ("function_address",),
        "when_fired": "before",
    },
    # angr-xmfj: simprocedure / syscall / dirty dispatch fires from the
    # existing Python callback sites in `rust_callback_dispatch.py`
    # (_handle_simprocedure_callback, _handle_syscall_callback_inner) and
    # `rust_manager._cb_dirty_call`. The Rust engine never invokes these
    # inspect callbacks directly — `dispatch_origin: 'python'` flags that
    # no PythonCallbacks slot is needed. The bit is still flipped in
    # `_update_inspect_bitmask` for consistency, even though Rust doesn't
    # read it. Attrs mirror the Python engine's _inspect call signatures
    # (sim_procedure.py:246/314 for simprocedure, procedure.py:34/50 for
    # syscall, engines/vex/heavy/inspect.py:10/22 for dirty). MVP scope:
    # the BP fires with the engine's chosen handler/args/result; user
    # mutations to those attrs in BP_BEFORE actions do NOT influence the
    # engine (Python's _inspect_getattr override path is not honored).
    "simprocedure": {
        "bit": 10,
        "attrs": (
            "simprocedure_name",
            "simprocedure_addr",
            "simprocedure",
            "simprocedure_result",
        ),
        "when_fired": "before",
        "dispatch_origin": "python",
    },
    "syscall": {
        "bit": 11,
        "attrs": (
            "syscall_name",
            "simprocedure",
        ),
        "when_fired": "before",
        "dispatch_origin": "python",
    },
    "dirty": {
        "bit": 12,
        "attrs": (
            "dirty_name",
            "dirty_handler",
            "dirty_args",
            "dirty_result",
        ),
        "when_fired": "before",
        "dispatch_origin": "python",
    },
    # angr-64pi: tmp_read / tmp_write dispatch fires from the Rust
    # interpreter's `RdTmp` / `WrTmp` arms (interpreter/expressions.rs
    # and interpreter/statements.rs). Each `RdTmp` evaluation pays a
    # single bitmask test in the no-BP case; with a BP the tmp's value
    # is round-tripped to a claripy AST and dispatched as
    # `tmp_read_expr`. `tmp_write` follows the same pattern in the
    # `WrTmp` arm, firing `when='after'` after the value is computed
    # but BEFORE the slot is mutated so the BP sees the value going in.
    # Both events fire many times per IRSB (every binop arg goes
    # through `RdTmp`); the bitmask short-circuit keeps overhead at
    # one `AtomicU16::load` per dispatch site when no BP is set.
    "tmp_read": {
        "bit": 13,
        "attrs": (
            "tmp_read_num",
            "tmp_read_expr",
        ),
        "when_fired": "after",
    },
    "tmp_write": {
        "bit": 14,
        "attrs": (
            "tmp_write_num",
            "tmp_write_expr",
        ),
        "when_fired": "after",
    },
    # angr-t8vf: statement dispatch fires from `execute_block_with_callbacks`
    # in `interpreter/execution.rs`, once per VEX IR statement, before the
    # statement runs. The only attr is `statement` (the integer index into
    # `irsb.statements`) — matches Python's `SimInspectMixin._handle_vex_stmt`
    # BP_BEFORE call signature. BP_AFTER is not wired (same MVP gap as
    # `instruction` BP_AFTER). Bit 15 is the LAST free slot in the
    # `AtomicU16` bitmask; the companion `expr` event widened
    # `inspect_enabled` to `AtomicU32` in angr-lge2.
    "statement": {
        "bit": 15,
        "attrs": ("statement",),
        "when_fired": "before",
    },
    # angr-lge2: expr dispatch fires from `eval_expr_with_callbacks` in
    # `interpreter/expressions.rs`, once per IR expression evaluation
    # (every constant, RdTmp, Get, Load, unop/binop arg, ITE, etc.).
    # `when='after'`, with `expr_result` set to the claripy-reconstructed
    # value of the expression. `expr` itself is passed as `None` — Rust
    # IRExpr doesn't round-trip cleanly into `pyvex.IRExpr`, and the BP
    # would otherwise need an expensive lift on every eval. Bit 16
    # required widening the bitmask from `AtomicU16` to `AtomicU32` (the
    # u16 had filled at bit 15 with `statement`). The bitmask short-circuit
    # is load-bearing for this dispatch site: it's the highest-frequency
    # call in the engine, so the no-BP cost MUST stay at one atomic load.
    # User mutations to `expr_result` are NOT honored (MVP gap pattern).
    "expr": {
        "bit": 16,
        "attrs": (
            "expr",
            "expr_result",
        ),
        "when_fired": "after",
    },
    # angr-vfst: address_concretization dispatch fires from
    # `interpreter/expressions.rs::load_symbolic_addr` (read path) and
    # `interpreter/statements.rs::try_rust_memory_store` (write path) when
    # the load/store address is symbolic. Fires both `before` (with the
    # symbolic addr AST, `result=None`) and `after` (with the concretized
    # `addr_concretization_result`). Strategy / memory / add_constraints
    # attrs are passed as None — the Rust engine has no SimMemory instance
    # or strategy stack to surface to the BP (MVP gap; user mutations to
    # those attrs in BP_BEFORE are not honored, matching the wider Rust
    # inspect MVP scope documented in `rust_engine.rst`).
    "address_concretization": {
        "bit": 17,
        "attrs": (
            "address_concretization_strategy",
            "address_concretization_action",
            "address_concretization_memory",
            "address_concretization_expr",
            "address_concretization_result",
            "address_concretization_add_constraints",
        ),
        "when_fired": "before",
    },
    # angr-vfst: symbolic_variable dispatch fires from
    # `interpreter/mod.rs::load_from_callback` when the engine mints a
    # fresh BVS for an unconstrained memory load (Python returned
    # `is_symbolic=True` with no AST). Fires `when='after'` with
    # `symbolic_name`, `symbolic_size`, and `symbolic_expr` mirroring
    # Python's `solver.py:432-439` BP_AFTER signature. The user-callable
    # `state.solver.BVS()` path still fires the same event from Python
    # natively, independent of this Rust dispatch — the Rust path covers
    # only fresh-BVS minting that originates inside the engine.
    "symbolic_variable": {
        "bit": 18,
        "attrs": (
            "symbolic_name",
            "symbolic_size",
            "symbolic_expr",
        ),
        "when_fired": "after",
    },
    # angr-4aach / angr-op0dn.14.4.1: constraints has TWO dispatch origins,
    # both landing on ``RustExplorationManager._cb_inspect_constraints``:
    #
    #   1. Python — ``RustSolverProxyPlugin.add`` in this module, the proxy
    #      solver installed on SimProcedure callback states. User mutation of
    #      ``added_constraints`` in a BP_BEFORE action IS honored here: the
    #      write-through re-reads the attr and installs the replaced list.
    #   2. Rust — ``exploration::helpers::add_fork_guard_constraint``, the
    #      branch-guard add on the continuing state during deferred-fork
    #      materialization. Mutation is NOT honored there (the guard is
    #      already lowered into a ``RustBV``), and the inverted guard the
    #      *forked* state receives inside ``build_unexplored_fork`` does not
    #      fire its own pair — the ``fork`` BP covers that state's creation.
    #
    # Both mirror Python's ``state_plugins/solver.py``, which fires
    # ``BP_BEFORE`` (with ``added_constraints``) then ``BP_AFTER`` around
    # ``self._solver.add(...)``. No ``dispatch_origin: 'python'`` — the Rust
    # slot IS registered so origin 2 has a target.
    "constraints": {
        "bit": 19,
        "attrs": ("added_constraints",),
        "when_fired": "before",
    },
    # angr-4aach / angr-op0dn.14.4.2: vex_lift has TWO dispatch origins, both
    # landing on ``RustExplorationManager._cb_inspect_vex_lift``:
    #
    #   1. Python — ``_cb_lift_block`` in rust_manager.py, the lift callback the
    #      Rust engine invokes on a block-cache miss.
    #   2. Rust — ``interpreter/execution.rs::try_native_lift``, the in-process
    #      libVEX lift (cargo feature ``libvex-ffi``, and only for blocks whose
    #      bytes are concrete). That path bypasses ``_cb_lift_block`` entirely,
    #      so without its own fire the event would silently vanish on a
    #      feature-on build. Both fires happen after libVEX has run, so exactly
    #      one BEFORE/AFTER pair is emitted per lift even when a native miss
    #      falls through to origin 1.
    #
    # Both mirror Python's ``engines/vex/lifter.py``, which fires ``vex_lift``
    # only when the lifter cache is NOT used — a block-cache miss is exactly
    # that path. ``BP_BEFORE`` carries (``vex_lift_addr``, ``vex_lift_size=None``,
    # ``vex_lift_buff``); ``BP_AFTER`` carries (``vex_lift_addr``,
    # ``vex_lift_size`` = the lifted IRSB's byte size). No
    # ``dispatch_origin: 'python'`` — the Rust slot IS registered so origin 2
    # has a target. User mutation of ``vex_lift_buff`` / ``vex_lift_addr`` /
    # ``vex_lift_size`` in BP_BEFORE is NOT honored on either origin — the lift
    # uses the engine's own bytes.
    "vex_lift": {
        "bit": 20,
        "attrs": (
            "vex_lift_addr",
            "vex_lift_size",
            "vex_lift_buff",
        ),
        "when_fired": "before",
    },
}

# Derived views — DO NOT add entries here; edit _INSPECT_EVENT_SPECS instead.
_RUST_INSPECT_SUPPORTED_EVENTS = frozenset(_INSPECT_EVENT_SPECS)
_RUST_INSPECT_ATTRS_BY_EVENT = {name: spec["attrs"] for name, spec in _INSPECT_EVENT_SPECS.items()}
_RUST_INSPECT_EVENT_BITS = {name: spec["bit"] for name, spec in _INSPECT_EVENT_SPECS.items()}
# Events whose dispatch fires from Python (not from the Rust engine's
# PythonCallbacks invocation path). Used by `_setup_callbacks` to skip
# Rust slot registration and by the allowlist consistency test to relax
# the "must have a `set_inspect_<event>` PyO3 slot" requirement.
_RUST_INSPECT_PYTHON_DISPATCHED_EVENTS = frozenset(
    name for name, spec in _INSPECT_EVENT_SPECS.items() if spec.get("dispatch_origin") == "python"
)


def _format_unsupported_event_msg(event_type: str) -> str:
    """Build the NotImplementedError message for a non-honored inspect event."""
    supported = ", ".join(sorted(_RUST_INSPECT_SUPPORTED_EVENTS))
    return (
        f"Rust engine inspect dispatches {supported} events; "
        f"got event_type={event_type!r}. "
        "Drop to use_rust_engine=False for full state.inspect coverage."
    )


class _NoOpInspectProxy:
    """Stand-in for state.inspect on a RustStateProxy detached from any manager.

    Used only when a RustStateProxy is constructed without a python_mgr
    (mostly low-level unit tests). With a manager attached, the proxy
    routes `.inspect` to the manager's `RustInspectProxy` instead, which
    actually dispatches mem_read / mem_write events from the Rust engine.
    """

    def b(self, *args, **kwargs):
        raise NotImplementedError(_INSPECT_NOT_IMPLEMENTED_MSG)

    def make_breakpoint(self, *args, **kwargs):
        raise NotImplementedError(_INSPECT_NOT_IMPLEMENTED_MSG)

    def add_breakpoint(self, *args, **kwargs):
        raise NotImplementedError(_INSPECT_NOT_IMPLEMENTED_MSG)

    def remove_breakpoint(self, *args, **kwargs):
        raise NotImplementedError(_INSPECT_NOT_IMPLEMENTED_MSG)

    def action(self, *args, **kwargs):
        raise NotImplementedError(_INSPECT_NOT_IMPLEMENTED_MSG)


class RustInspectProxy:
    """state.inspect for Rust-engine states.

    Implements the subset of `SimInspector` needed for the mem_read /
    mem_write MVP. Other event types raise NotImplementedError at
    registration time to make the gap loud.

    Storage is centralized on the owning RustExplorationManager (a single
    set of breakpoints applies across all that manager's states), to keep
    the bitmask the Rust engine reads in sync with what's registered.
    The proxy itself is shared across all per-state RustStateProxy
    instances; `set_state` rebinds the `state` field that BP.check / fire
    receive when dispatch occurs for that state.

    This is an MVP deviation from Python's per-state inspect plugin
    semantics — see `docs/advanced-topics/rust_engine.rst`.
    """

    SUPPORTED_EVENTS = _RUST_INSPECT_SUPPORTED_EVENTS

    def __init__(self, mgr):
        # Manager-owned BP storage; the proxy is a thin facade so all
        # state proxies share the same dispatch set.
        self._mgr = mgr
        self.state = None
        self.action_attrs_set = False
        # Initialize every known inspect attribute to None so BP.check
        # doesn't AttributeError when reading attrs not set by the
        # current event.
        for attrs in _RUST_INSPECT_ATTRS_BY_EVENT.values():
            for a in attrs:
                setattr(self, a, None)

    @property
    def _breakpoints(self):
        """Compat with code that pokes into SimInspector._breakpoints."""
        return self._mgr._inspect_breakpoints

    def set_state(self, state):
        """Bind the state passed to BP.check / BP.fire during the next action().

        Called by the manager-side dispatcher right before invoking
        `action(...)` so the user's BP callable sees the per-state proxy.
        """
        self.state = state

    def b(self, event_type, *args, **kwargs):
        """Alias for make_breakpoint, matching SimInspector.b."""
        return self.make_breakpoint(event_type, *args, **kwargs)

    def make_breakpoint(self, event_type, *args, **kwargs):
        self._check_event(event_type)
        from angr.state_plugins.inspect import BP

        bp = BP(*args, **kwargs)
        self.add_breakpoint(event_type, bp)
        return bp

    def add_breakpoint(self, event_type, bp):
        self._check_event(event_type)
        self._mgr._inspect_breakpoints[event_type].append(bp)
        self._mgr._update_inspect_bitmask()

    def remove_breakpoint(self, event_type, bp=None, filter_func=None):
        self._check_event(event_type)
        if bp is None and filter_func is None:
            raise ValueError('remove_breakpoint(): You must specify either "bp" or "filter".')
        bps = self._mgr._inspect_breakpoints[event_type]
        if bp is not None:
            try:
                bps.remove(bp)
            except ValueError:
                pass
        else:
            self._mgr._inspect_breakpoints[event_type] = [b for b in bps if not filter_func(b)]
        self._mgr._update_inspect_bitmask()

    def action(self, event_type, when, **kwargs):
        """Mirror SimInspector.action: stage attrs, walk BPs, fire matches.

        BP.fire() invokes the user action with `self.state` (set by the
        dispatcher just before this call). On reentrant calls (user action
        triggering another inspect event), staging clobbers attrs — same
        as Python's SimInspector. The Rust dispatch site guards against
        reentrant dispatch (uq4n.4).
        """
        self._check_event(event_type)
        self._set_inspect_attrs(**kwargs)
        self.action_attrs_set = True
        state = self.state
        try:
            for bp in list(self._mgr._inspect_breakpoints[event_type]):
                if not self.action_attrs_set:
                    self._set_inspect_attrs(**kwargs)
                    self.action_attrs_set = True
                if bp.check(state, when):
                    bp.fire(state)
        finally:
            self.action_attrs_set = False

    def _check_event(self, event_type):
        if event_type not in self.SUPPORTED_EVENTS:
            raise NotImplementedError(_format_unsupported_event_msg(event_type))

    def _set_inspect_attrs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class RustStateProxy:
    """
    Lightweight read-through proxy to a Rust symex state.

    Delegates property reads to Rust via PyO3 bindings. No caching,
    no state sync — reads directly from the Rust engine.

    Use this for:
    - ExplorationTechnique filter()/complete() callbacks
    - Callable find/avoid predicates
    - Found-state export and solution extraction

    Full SimState is only needed for SimProcedure execution.
    """

    def __init__(
        self, rust_mgr, state_id, project=None, stdin_vars=None, stdout_data=None, python_mgr=None, owns_copy=False
    ):
        self._mgr = rust_mgr
        self._state_id = state_id
        self._project = project
        self._stdin_vars = stdin_vars
        self._stdout_data = stdout_data or b""
        # High-level RustExplorationManager — used for options/globals lookup.
        # None when constructed standalone (e.g., low-level unit tests); in
        # that case options/globals fall back to empty stand-ins.
        self._python_mgr = python_mgr
        # angr-yhe0: True when this proxy was minted by ``.copy()`` and owns
        # the lifetime of the underlying ``_copies`` stash entry. ``__del__``
        # then asks the manager to drop the Rust-side state so long-running
        # spilling/veritesting workflows don't leak copies until manager
        # teardown. False for proxies that wrap a live exploration state.
        self._owns_copy = owns_copy
        # Lazy-initialized sub-proxies
        self._solver_proxy = None
        self._regs_proxy = None
        self._mem_proxy = None
        self._history_proxy = None
        self._posix_proxy = None
        self._callstack_proxy = None
        self._inspect_proxy = None
        self._heap_proxy = None
        self._scratch_proxy = None
        # angr-yodz: single forked solver context shared by this proxy's
        # solver/memory/posix sub-proxies. Forked lazily on first solver
        # access so add_constraints lands somewhere all three sub-proxies
        # read from. View-local — see _get_shared_solver_ctx.
        self._shared_solver_ctx = None

    @property
    def state_id(self):
        """Rust-side state identifier."""
        return self._state_id

    @property
    def addr(self):
        """Current program counter (O(1) via state index, no full export)."""
        if hasattr(self, "_override_addr") and self._override_addr is not None:
            return self._override_addr
        try:
            pc = self._mgr.get_state_pc_by_id(self._state_id)
            if pc is not None:
                return pc
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: returning 0 when PC lookup fails can
            # spuriously match a find/avoid predicate that includes addr 0.
            # Log at warn so the failure is loud; callers reading proxy.addr
            # in critical paths will see the silent zero behavior in stderr.
            l.warning("RustStateProxy.addr(sid=%d) lookup failed: %s: %s", self._state_id, type(e).__name__, e)
        return 0

    @property
    def ip(self):
        """Alias for addr, as claripy BVV."""
        return claripy.BVV(self.addr, self.arch.bits)

    @property
    def arch(self):
        """Architecture object from the project."""
        if self._project is not None:
            return self._project.arch
        # Fallback: get arch name from Rust and resolve
        arch_name = self._mgr.arch
        import archinfo

        return archinfo.arch_from_id(arch_name)

    @property
    def project(self):
        """The angr Project."""
        return self._project

    def _get_shared_solver_ctx(self):
        """Single forked solver context shared by this proxy's solver/memory/
        posix sub-proxies.

        Forking once (rather than per-sub-proxy) is what makes
        ``proxy.add_constraints(c)`` visible to ``proxy.posix.dumps(0)`` and
        ``proxy.memory.load(symbolic_addr)`` — they all read from this same
        context (angr-yodz). The constraint is **view-local**: it lands on
        this forked context, not the underlying Rust state's solver, so it
        does not leak into the live exploration state or sibling proxies and
        vanishes when the proxy is dropped. To persist a constraint onto the
        state itself, use ``mgr._rust_mgr.add_constraints_to_state`` (the
        write-through path the SimProcedure-callback plugin uses).
        """
        if self._shared_solver_ctx is None:
            self._shared_solver_ctx = self._mgr.fork_state_solver(self._state_id)
        return self._shared_solver_ctx

    @property
    def solver(self):
        """Solver proxy — delegates to Rust Z3 via PyO3."""
        if self._solver_proxy is None:
            self._solver_proxy = RustSolverProxy(
                self._mgr, self._state_id, shared_ctx_getter=self._get_shared_solver_ctx
            )
        return self._solver_proxy

    @property
    def se(self):
        """Legacy alias for solver."""
        return self.solver

    @property
    def regs(self):
        """Register proxy — reads registers from Rust state."""
        if self._regs_proxy is None:
            self._regs_proxy = RustRegisterProxy(self._mgr, self._state_id, self.arch, python_mgr=self._python_mgr)
        return self._regs_proxy

    @property
    def registers(self):
        """Alias for regs."""
        return self.regs

    @property
    def memory(self):
        """Memory proxy — reads memory from Rust state."""
        if self._mem_proxy is None:
            self._mem_proxy = RustMemoryProxy(
                self._mgr,
                self._state_id,
                self.arch,
                python_mgr=self._python_mgr,
                shared_ctx_getter=self._get_shared_solver_ctx,
            )
        return self._mem_proxy

    @property
    def mem(self):
        """Alias for memory."""
        return self.memory

    @property
    def history(self):
        """History proxy."""
        if self._history_proxy is None:
            self._history_proxy = RustHistoryProxy(self._mgr, self._state_id)
        return self._history_proxy

    @property
    def posix(self):
        """Posix proxy for stdin/stdout dumps."""
        if self._posix_proxy is None:
            self._posix_proxy = RustPosixProxy(
                self._mgr,
                self._state_id,
                stdin_vars=self._stdin_vars,
                stdout_data=self._stdout_data,
                shared_ctx_getter=self._get_shared_solver_ctx,
            )
        return self._posix_proxy

    @property
    def callstack(self):
        """Callstack proxy — iterable view of the Rust state's call frames.

        Top frame (most recent) first. Frame attrs match angr's CallStack
        plugin (call_site_addr, func_addr, ret_addr, stack_ptr).
        """
        if self._callstack_proxy is None:
            self._callstack_proxy = RustCallStackProxy(self._mgr, self._state_id)
        return self._callstack_proxy

    @property
    def inspect(self):
        """state.inspect for the Rust engine.

        If this proxy is attached to a RustExplorationManager, returns the
        manager-wide RustInspectProxy that supports mem_read / mem_write
        BP registration. Without a manager (low-level proxy construction),
        registration raises NotImplementedError via _NoOpInspectProxy.
        """
        if self._python_mgr is not None:
            return self._python_mgr._get_inspect_proxy()
        if self._inspect_proxy is None:
            self._inspect_proxy = _NoOpInspectProxy()
        return self._inspect_proxy

    @property
    def heap(self):
        """Heap proxy — exposes mmap_base and allocation tracking."""
        if self._heap_proxy is None:
            self._heap_proxy = RustHeapProxy(self._mgr, self._state_id)
        return self._heap_proxy

    @property
    def scratch(self):
        """Read-only scratch proxy — exposes bbl_addr / ins_addr / jumpkind.

        The Rust engine doesn't store the per-block VEX-temp/tyenv/stmt_idx
        values that the Python SimStateScratch plugin maintains, so those
        attributes return None / empty. The values that survive between
        blocks (block address and last jumpkind) read live from the Rust
        state.
        """
        if self._scratch_proxy is None:
            self._scratch_proxy = RustScratchProxy(self._mgr, self._state_id)
        return self._scratch_proxy

    @property
    def options(self):
        """Per-state options set, backed by the high-level manager.

        The Rust engine doesn't honor SimOptions (LAZY_SOLVES /
        STRICT_PAGE_ACCESS are mirrored separately on the Rust state), but
        user code reads `X in state.options` and writes `state.options.add(X)`.
        Returns a live set; mutations persist for this state_id.
        """
        if self._python_mgr is not None:
            return self._python_mgr.get_state_options_py(self._state_id)
        return set()

    @property
    def globals(self):
        """Per-state globals dict, backed by the high-level manager."""
        if self._python_mgr is not None:
            return self._python_mgr.get_state_globals_py(self._state_id)
        return {}

    def add_constraints(self, *constraints):
        """Add constraints to this proxy's view of the solver.

        angr-yodz: the constraint lands on a single forked context shared by
        this proxy's solver / memory / posix sub-proxies, so a subsequent
        ``proxy.posix.dumps(0)`` or ``proxy.memory.load(symbolic_addr)``
        honors it. The constraint is **view-local** — it does NOT persist
        onto the underlying Rust state's solver, so it neither perturbs live
        exploration nor survives into a freshly built proxy. To persist a
        constraint onto the state itself (SimState semantics), use
        ``mgr._rust_mgr.add_constraints_to_state(state_id, [...])`` — the
        write-through path used by ``RustSolverProxyPlugin`` during
        SimProcedure callbacks.
        """
        self.solver.add(*constraints)

    def satisfiable(self, **kwargs):
        """Check if this state is satisfiable."""
        return self.solver.satisfiable(**kwargs)

    def copy(self):
        """Rust-side CoW deep fork (angr-d1dr).

        Mirrors the ``SimState.copy()`` contract: a fully independent state
        whose mutations do not flow back to the source. The new state is
        minted by ``RustSimState::fork`` and parked in the ``_copies`` stash
        — outside the active/found/avoid loop so it won't be auto-stepped.
        Returns a fresh ``RustStateProxy`` bound to the new state id, with
        its own snapshot of the source's options / globals / stdout buffer.

        Raises ``NotImplementedError`` when the proxy was constructed
        without a high-level manager (``python_mgr`` is None) — there is no
        place to store Python-side options/globals snapshots in that case.
        Low-level unit-test paths that need a plain Rust fork should call
        ``rust_mgr.fork_state_to_stash(state_id, '_copies')`` directly.
        """
        if self._python_mgr is None:
            raise NotImplementedError(_COPY_NEEDS_MANAGER_MSG)
        new_id = self._python_mgr.fork_state_for_copy(self._state_id)
        return RustStateProxy(
            self._mgr,
            new_id,
            project=self._project,
            stdin_vars=self._stdin_vars,
            stdout_data=self._stdout_data,
            python_mgr=self._python_mgr,
            owns_copy=True,
        )

    def __repr__(self):
        try:
            stash = self._mgr.state_stash(self._state_id)
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: __repr__ is best-effort; skip
            # missing stash field rather than fail repr().
            stash = None
        try:
            n_constraints = self._mgr.state_constraint_count(self._state_id)
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: __repr__ is best-effort; skip
            # missing constraint count rather than fail repr().
            n_constraints = None
        parts = [f"id={self._state_id}", f"addr={hex(self.addr)}"]
        if stash is not None:
            parts.append(f"stash={stash}")
        if n_constraints is not None:
            parts.append(f"constraints={n_constraints}")
        return f"<RustStateProxy {' '.join(parts)}>"

    def __del__(self):
        """angr-yhe0: drop the Rust-side ``_copies`` entry when a copy-proxy
        is GC'd by Python.

        Only fires when ``_owns_copy`` is True (set by ``copy()``) — proxies
        wrapping live exploration states never auto-drop. Everything is wrapped
        in try/except because interpreter shutdown can already have torn down
        ``sys.modules`` by the time ``__del__`` runs, and the underlying Rust
        manager may have been dropped first.
        """
        try:
            if not getattr(self, "_owns_copy", False):
                return
            python_mgr = getattr(self, "_python_mgr", None)
            if python_mgr is None:
                return
            python_mgr.drop_copy(self._state_id)
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: __del__ is best-effort; swallow
            # everything to match the SimulationManagerProxy.__del__ pattern.
            pass


# Rust step_state() bucket names -> SimulationManager.step_state() stash keys.
# The Rust engine's live successors arrive under "flat"; the successor-dict
# contract keys them under None (sim_manager.py::SimulationManager.step_state).
_STEP_BUCKET_TO_STASH = {"flat": None}

_SUCCESSOR_FUNC_MSG = (
    "successor_func is not supported by RustSimulationManagerProxy.step_state() "
    "— the Rust engine steps states natively and never builds a SimSuccessors "
    "object for a Python callable to consume. Use use_rust_engine=False for "
    "techniques that pass successor_func."
)


def _rust_state_id(state) -> int:
    """Resolve the Rust state ID behind a RustStateProxy or an exported SimState."""
    state_id = getattr(state, "_state_id", None)
    if state_id is None:
        scratch = getattr(state, "scratch", None)
        state_id = getattr(scratch, "rust_state_id", None)
    if state_id is None:
        raise ValueError(
            f"{state!r} is not backed by a Rust state (no _state_id / "
            "scratch.rust_state_id) — it cannot be stepped by the Rust manager."
        )
    return int(state_id)


class RustSimulationManagerProxy:
    """
    Presents a SimulationManager-like interface backed by Rust stashes.

    Used by ExplorationTechniques that need simgr.stashes, simgr.found, etc.
    States are returned as RustStateProxy objects (O(1) creation, no sync).
    """

    def __init__(self, rust_mgr, project=None, stdin_vars=None, stdout_tracker=None, python_mgr=None):
        self._mgr = rust_mgr
        self._project = project
        self._stdin_vars = stdin_vars
        self._stdout_tracker = stdout_tracker or {}  # state_id -> bytes
        self._python_mgr = python_mgr
        self._errored = []
        # Wired by RustExplorationManager when dispatching ExplorationTechnique
        # step() hooks. The callback advances the Rust engine by one batch and
        # records the resulting ExplorationEvent. When None, step()/successors()
        # raise — they must not silently no-op once a tech overrides them.
        self._step_callback = None

    def _wrap_state(self, state_id):
        """Wrap a Rust state ID in a RustStateProxy."""
        return RustStateProxy(
            self._mgr,
            state_id,
            project=self._project,
            stdin_vars=self._stdin_vars,
            stdout_data=self._stdout_tracker.get(state_id, b""),
            python_mgr=self._python_mgr,
        )

    def _get_stash(self, name):
        """Get all states in a stash as proxy objects."""
        state_ids = self._mgr.get_state_ids(name)
        return [self._wrap_state(sid) for sid in state_ids]

    @property
    def active(self):
        return self._get_stash("active")

    @active.setter
    def active(self, states):
        l.warning("RustSimulationManagerProxy: setting active is not yet supported")

    @property
    def found(self):
        return self._get_stash("found")

    @property
    def deadended(self):
        return self._get_stash("deadended")

    @property
    def avoid(self):
        return self._get_stash("avoid")

    @property
    def errored(self):
        return self._errored

    @property
    def one_found(self):
        """Return first found state or None."""
        found = self.found
        return found[0] if found else None

    @property
    def one_active(self):
        """Return first active state or None."""
        active = self.active
        return active[0] if active else None

    @property
    def stashes(self):
        """Dict-like access to all stashes."""
        return _StashDict(self)

    @property
    def _stashes(self):
        """Direct stash dict access (for techniques that access simgr._stashes)."""
        return self.stashes

    def filter(self, state, filter_func=None):
        """Default filter — delegates to filter_func or returns None.

        Techniques call simgr.filter(state) to delegate to the next technique.
        """
        if filter_func is not None:
            return filter_func(state)
        return None

    def step(self, stash="active", **kwargs):
        """Advance the Rust engine by one batch.

        This is the base step impl that ExplorationTechnique.step() hooks
        wrap. The RustExplorationManager installs `_step_callback` before
        dispatching, so calling this directly without the callback set is a
        programming error — raise rather than silently no-op.
        """
        if self._step_callback is None:
            raise NotImplementedError(
                "RustSimulationManagerProxy.step() can only be called from "
                "inside an ExplorationTechnique step() dispatch (the owning "
                "RustExplorationManager wires the callback)."
            )
        self._step_callback(stash=stash, **kwargs)
        return self

    def step_state(self, state, successor_func=None, error_list=None, **run_args):
        """Step one state and return its successors bucketed by stash.

        Mirrors ``SimulationManager.step_state()``: the return value is a dict
        keyed by stash name, with the live successors under ``None`` (see
        ``sim_manager.py``). Keys always present: ``None`` and ``"unsat"``
        (the Rust engine never produces unsat successors, so that list is
        always empty). ``"unconstrained"``, ``"pruned"``, ``"deadended"`` and
        ``"errored"`` appear only when non-empty.

        Two Rust-path specifics:

        * ``successor_func`` is UNSUPPORTED and raises ``NotImplementedError``
          — the engine steps natively and never materializes a
          ``SimSuccessors`` for a Python callable to post-process. Techniques
          that need it must run on the Python engine
          (``use_rust_engine=False``).
        * Successors are RETURNED, never auto-pushed into a stash. They park
          in the Rust ``_step_out`` quarantine stash (which also keeps them
          alive for the returned SimStates' Rust-solver fallback); the caller
          places them, e.g. ``mgr._rust_mgr.move_state(sid, "_step_out",
          "active")``. The stepped state itself is consumed — Rust takes it
          out of its stash for the step, matching the Python engine's
          ``step()``, which drops the source state after bucketing.

        ``error_list`` is accepted for signature compatibility but unused: a
        step that errors is reported as an ``"errored"`` bucket rather than an
        ``ErrorRecord``, because the Rust engine records the failure itself.
        """
        if successor_func is not None:
            raise NotImplementedError(_SUCCESSOR_FUNC_MSG)
        if self._python_mgr is None:
            raise NotImplementedError(
                "step_state() needs the owning RustExplorationManager to export "
                "successors; this proxy was constructed without one."
            )

        extra_stop_points = run_args.pop("extra_stop_points", None)
        if run_args:
            l.debug("step_state(): ignoring run_args unsupported on the Rust path: %s", sorted(run_args))
        stop_points = sorted({int(addr) for addr in extra_stop_points}) if extra_stop_points else None

        buckets = self._mgr.step_state(_rust_state_id(state), stop_points)

        stashes = {None: [], "unsat": []}
        for bucket, snapshots in buckets.items():
            key = _STEP_BUCKET_TO_STASH.get(bucket, bucket)
            stashes.setdefault(key, []).extend(self._python_mgr._snapshot_to_angr(snap) for snap in snapshots)
        return stashes

    def successors(self, state, **kwargs):
        """Run one state forward, returning SimSuccessors.

        Not supported on the Rust manager: a ``SimSuccessors`` object is a
        Python-engine artifact, and rebuilding one would mean re-running the
        step from Python. Use :meth:`step_state`, which returns the same
        successor buckets the Rust engine already produced.
        """
        raise NotImplementedError(
            "successors() is not supported by RustSimulationManagerProxy "
            "(would require a Python-side re-run). Use step_state() for the "
            "successor-dict contract, or use_rust_engine=False if a registered "
            "ExplorationTechnique needs a SimSuccessors object."
        )

    def move(self, from_stash="active", to_stash="stashed", filter_func=None):
        """Move states between stashes."""
        if filter_func is None:
            count = self._mgr.move_states(from_stash, to_stash)
        else:
            # Move states that match filter
            state_ids = self._mgr.get_state_ids(from_stash)
            moved = 0
            for sid in state_ids:
                proxy = self._wrap_state(sid)
                if filter_func(proxy):
                    self._mgr.move_state(sid, from_stash, to_stash)
                    moved += 1
            count = moved
        return count

    def __repr__(self):
        counts = self._mgr.stash_counts()
        parts = ["<RustSimulationManagerProxy"]
        for name, count in sorted(counts.items()):
            if count > 0:
                parts.append(f" {name}:{count}")
        parts.append(">")
        return "".join(parts)


class _StashDict:
    """Dict-like wrapper for stash access: simgr.stashes['found']."""

    def __init__(self, simgr_proxy):
        self._simgr = simgr_proxy

    def __getitem__(self, key):
        return self._simgr._get_stash(key)

    def __setitem__(self, key, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"_StashDict.__setitem__ expects a list of RustStateProxy objects, got {type(value).__name__}"
            )
        if len(value) == 0:
            # Clear the stash — common pattern: simgr.stashes['found'] = []
            try:
                self._simgr._mgr.clear_stash(key)
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: stash clear failed; user expects
                # the stash to be empty afterward but it may not be. Log so
                # this is visible under --debug.
                l.debug("clear_stash(%r) failed: %s: %s", key, type(e).__name__, e)
            return

        # Non-empty assignment (angr-wxuo): make the stash contain exactly the
        # assigned proxies, in order. Used by Director (goal prioritization,
        # writes simgr.stashes[stash] = [...]) and StochasticSearch (restart /
        # re-weight). Stash members are RustStateProxy objects carrying
        # Rust-side state_ids, so this is done via move_state without any
        # SimState materialization.
        mgr = self._simgr._mgr
        desired_ids = []
        seen = set()
        for proxy in value:
            if not isinstance(proxy, RustStateProxy):
                raise TypeError(
                    f"_StashDict.__setitem__ can only assign RustStateProxy objects "
                    f"to stash '{key}', got {type(proxy).__name__}"
                )
            if getattr(proxy, "_mgr", None) is not mgr:
                raise ValueError(
                    f"_StashDict.__setitem__: state {proxy.state_id} belongs to a "
                    f"different manager and cannot be assigned to stash '{key}'"
                )
            sid = proxy.state_id
            if sid in seen:
                raise ValueError(f"_StashDict.__setitem__: duplicate state {sid} in assignment to stash '{key}'")
            cur = mgr.state_stash(sid)
            if cur is None:
                raise ValueError(
                    f"_StashDict.__setitem__: state {sid} is not tracked by this "
                    f"manager (already dropped?) and cannot be assigned to stash '{key}'"
                )
            seen.add(sid)
            desired_ids.append((sid, cur))

        # Drop current members of `key` that are not in the desired set. Matches
        # Python angr semantics where states omitted from the new list leave the
        # stash; here they are freed since each Rust state lives in exactly one
        # stash.
        for sid in mgr.get_state_ids(key):
            if sid not in seen:
                try:
                    mgr.drop_state_from_stash(sid, key)
                except Exception as e:  # pragma: no cover - defensive
                    l.debug("drop_state_from_stash(%d, %r) failed: %s: %s", sid, key, type(e).__name__, e)

        # Append desired states into `key` in order. move_state with from==to
        # removes and re-appends, so iterating in desired order rebuilds the
        # stash exactly (honoring caller-specified ordering for those already in
        # `key`).
        for sid, cur in desired_ids:
            mgr.move_state(sid, cur, key)

    def __contains__(self, key):
        counts = self._simgr._mgr.stash_counts()
        return key in counts

    def keys(self):
        return self._simgr._mgr.stash_counts().keys()

    def values(self):
        return [self._simgr._get_stash(k) for k in self.keys()]

    def items(self):
        return [(k, self._simgr._get_stash(k)) for k in self.keys()]

    def get(self, key, default=None):
        try:
            return self[key]
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: dict.get() contract — return the
            # caller-supplied default when the key is missing or unfetchable.
            return default
