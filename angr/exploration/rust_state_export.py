"""State export and conversion for the Rust exploration manager.

Handles converting Rust state snapshots to angr SimStates, syncing constraints,
restoring plugins, and managing stash export. These methods are defined as a
mixin class that RustExplorationManager inherits from.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import claripy

from angr.state_plugins.history import SimStateHistory

if TYPE_CHECKING:
    import angr

l = logging.getLogger(name=__name__)


class _RustOwnedSimStateHistory(SimStateHistory):
    """SimStateHistory variant that warns once per process when user code reads
    ``state.history.actions`` or ``state.history.events``.

    The Rust engine does not populate ``recent_actions`` / ``recent_events``,
    so these properties return empty iterators no matter what
    ``TRACK_CONSTRAINT_ACTIONS`` / ``TRACK_MEMORY_ACTIONS`` / ``TRACK_*`` are
    set. The relevant options ship in the default ``symbolic`` mode bundle, so
    we can't warn on add() without spamming every ``entry_state()``; instead
    we install this subclass on every materialized Rust-owned SimState and
    warn the first time the empty stream is actually read.

    Class-level flag (not instance-level) — one warning per process even
    across managers and states.
    """

    _WARNED = False

    @classmethod
    def _warn_once(cls, attr: str) -> None:
        if cls._WARNED:
            return
        cls._WARNED = True
        warnings.warn(
            f"state.history.{attr} is empty: the Rust engine does not "
            "produce SimAction/SimEvent records, so the TRACK_*_ACTIONS / "
            "TRACK_MEMORY_MAPPING SimOptions have no effect under "
            "RustExplorationManager. Use the Python engine (drop "
            "use_rust_engine=True) for action-stream-driven analyses. See "
            "docs/advanced-topics/rust_engine.rst.",
            UserWarning,
            stacklevel=3,
        )

    @property
    def actions(self):
        type(self)._warn_once("actions")
        return super().actions

    @property
    def events(self):
        type(self)._warn_once("events")
        return super().events


def _install_rust_history_warning(state) -> None:
    """Promote ``state.history`` to the warn-on-read variant in place.

    No-op if the plugin is already the warning subclass or is some unrelated
    custom subclass (we only swap a clean ``SimStateHistory``).
    """
    history = getattr(state, "history", None)
    if history is None:
        return
    if type(history) is SimStateHistory:
        history.__class__ = _RustOwnedSimStateHistory


class RustSolverFallback:
    """Wraps state.solver so eval/min/max/satisfiable defer to the Rust solver.

    Holds a forked Rust Z3 context (lazily created, cached across calls) and
    the original Python solver methods. Each wrapper method tries the Rust
    solver first and falls back to Python on failure or when Rust returns
    None. Constraints added to the Python state after attach() (e.g.,
    flareon2015_5 hash equalities) are synced into the cached Rust context
    on the next call.

    INVARIANT (``invariant-rust-solver-fallback-class``; mirrored in
    ``callbacks.rs`` module invariant 5): this class owns the per-state
    Rust solver fallback wiring — cached forked context, original method
    handles, constraint sync counter. Use
    ``RustSolverFallback(state, state_id, rust_mgr).attach()`` instead of
    inline closures; the ``_rust_fallback_attached`` flag on state.solver
    guards against double-patching (which would cause infinite recursion
    since the second wrapper's "originals" would be the first wrapper's
    bound methods). Per-callback solver attachment is the parallel path
    in ``_install_rust_solver_on_callback_state``
    (rust_callback_dispatch.py); both paths must stay in sync if a new
    FFI solver entry point is added.
    """

    _ATTACH_FLAG = "_rust_fallback_attached"

    def __init__(self, state, state_id, rust_mgr):
        self._state = state
        self._state_id = state_id
        self._rust_mgr = rust_mgr
        self._original_eval = state.solver.eval
        self._original_eval_upto = state.solver.eval_upto
        self._original_min = state.solver.min
        self._original_max = state.solver.max
        self._original_satisfiable = state.solver.satisfiable
        # Cache the forked Rust solver: fork_state_solver() clones the Z3
        # solver (~3ms), so caching saves significant time when the solve
        # script calls eval() many times (ais3 ~100 byte evals).
        self._cached_rust_ctx = None
        # Track constraint count at attach time so we can detect when the
        # caller adds constraints post-exploration and replay them into Rust.
        n = len(state.solver.constraints)
        self._initial_constraint_count = n
        self._synced_constraint_count = n

    def attach(self):
        """Bind wrapper methods onto state.solver. No-op if already attached."""
        state = self._state
        state.scratch.rust_mgr = self._rust_mgr
        state.scratch.rust_found_state_id = self._state_id
        # Guard against double-patching (which would cause infinite recursion
        # since the second wrapper's "originals" would be the first wrapper).
        solver = state.solver
        if getattr(solver, self._ATTACH_FLAG, False):
            return
        setattr(solver, self._ATTACH_FLAG, True)
        solver.eval = self.eval
        solver.eval_upto = self.eval_upto
        solver.min = self.min
        solver.max = self.max
        solver.satisfiable = self.satisfiable

    def _get_rust_ctx(self):
        if self._cached_rust_ctx is None:
            ctx = self._rust_mgr.fork_state_solver(self._state_id)
            # Default 30s is too short for complex post-exploration solving (asisctf)
            try:
                ctx.set_timeout(120000)
            except Exception:
                # cat-(a) EXPECTED CONTROL FLOW: set_timeout is a best-effort
                # tuning knob; older Rust contexts may not expose it. Default
                # 30s timeout still applies.
                pass
            self._cached_rust_ctx = ctx
        # Replay constraints that the caller added after attach
        current = len(self._state.solver.constraints)
        if current != self._synced_constraint_count:
            new_constraints = self._state.solver.constraints[self._synced_constraint_count :]
            for c in new_constraints:
                try:
                    self._cached_rust_ctx.add_constraint_ast(c)
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: a post-attach constraint
                    # added by the caller could not be replayed into Rust's
                    # context (e.g., references a Python-only symbol). The
                    # next eval() will use a Rust solver missing this
                    # constraint and may return values inconsistent with
                    # the Python solver.
                    l.debug("Could not replay post-attach constraint into Rust ctx: %s", e)
            self._synced_constraint_count = current
        return self._cached_rust_ctx

    def _rust_eval(self, expr, cast_to):
        rust_ctx = self._get_rust_ctx()
        result = rust_ctx.eval(expr)
        if result is not None:
            if cast_to == bytes:
                nbytes = (expr.length + 7) // 8
                raw = result.to_bytes(nbytes, "little")
                return raw[::-1]
            return result
        # Wide BVS (e.g. 160-bit flag): Rust eval returns None because the
        # full symbol isn't in Rust's table. Decompose into byte-sized evals.
        if hasattr(expr, "length") and expr.length > 64:
            nbytes = expr.length // 8
            byte_vals = []
            for i in range(nbytes):
                hi = expr.length - 1 - i * 8
                lo = hi - 7
                byte_expr = claripy.Extract(hi, lo, expr)
                byte_result = rust_ctx.eval(byte_expr)
                if byte_result is None:
                    return None
                byte_vals.append(byte_result & 0xFF)
            value = 0
            for bv in byte_vals:
                value = (value << 8) | bv
            if cast_to == bytes:
                return value.to_bytes(nbytes, "big")
            return value
        return None

    def eval(self, expr, cast_to=None, **kwargs):
        try:
            result = self._rust_eval(expr, cast_to)
            if result is not None:
                return result
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: Rust eval failed; Python solver
            # is the fallback and has the same constraints (constraint sync
            # ran during exploration), so no wrong-answer risk.
            l.debug("Rust eval failed, falling back to Python: %s", e)
        return self._original_eval(expr, cast_to=cast_to, **kwargs)

    def eval_upto(self, expr, n, cast_to=None, **kwargs):
        try:
            rust_ctx = self._get_rust_ctx()
            results = rust_ctx.eval_upto(expr, n)
            if results:
                if cast_to == bytes:
                    nbytes = (expr.length + 7) // 8
                    return [r.to_bytes(nbytes, "big") for r in results]
                return list(results)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: Rust eval_upto failed; Python
            # solver produces the result. Same constraint set, so no
            # wrong-answer risk.
            l.debug("Rust eval_upto failed, falling back to Python: %s", e)
        return self._original_eval_upto(expr, n, cast_to=cast_to, **kwargs)

    def min(self, expr, **kwargs):
        try:
            rust_ctx = self._get_rust_ctx()
            result = rust_ctx.min(expr, signed=kwargs.get("signed", False))
            if result is not None:
                return result
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: Rust min failed; Python solver
            # is the fallback. Same constraint set.
            l.debug("Rust min failed, falling back to Python: %s", e)
        return self._original_min(expr, **kwargs)

    def max(self, expr, **kwargs):
        try:
            rust_ctx = self._get_rust_ctx()
            result = rust_ctx.max(expr, signed=kwargs.get("signed", False))
            if result is not None:
                return result
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: Rust max failed; Python solver
            # is the fallback. Same constraint set.
            l.debug("Rust max failed, falling back to Python: %s", e)
        return self._original_max(expr, **kwargs)

    def satisfiable(self, **kwargs):
        try:
            rust_ctx = self._get_rust_ctx()
            return rust_ctx.satisfiable()
        except Exception as e_rust:
            # cat-(b) FALLBACK WITH LOSS: Rust solver failed; Python is
            # tried next.
            l.debug("Rust satisfiable() failed, trying Python: %s", e_rust)
        try:
            return self._original_satisfiable(**kwargs)
        except Exception as e_py:
            # cat-(c) WRONG-ANSWER RISK: returning False here would mean
            # "UNSAT" — a wrong answer that masks the underlying solver
            # failure. Re-raise so the caller sees the failure.
            l.warning("Both Rust and Python satisfiable() failed for state — Python error: %s", e_py)
            raise


class _LazySimStateRef:
    """Proxy that defers SimState materialization until first attribute access.

    Returned from `_get_stash_states` in place of eagerly-materialized
    ``SimState`` instances. Iterating a stash list, taking ``len()``, or
    comparing identity does **not** trigger the heavy ``_restore_plugins_to_state``
    / ``_sync_rust_*`` calls. The first read of any non-private attribute
    forwards to ``RustStateExportMixin._materialize_single_state`` and then
    delegates the lookup to the materialized SimState.

    Identity is preserved across repeated stash reads via the per-manager
    ``_lazy_state_refs`` cache (one wrapper per Rust state id).
    """

    __slots__ = ("_lazy_mgr", "_lazy_state_id")

    def __init__(self, mgr, state_id):
        object.__setattr__(self, "_lazy_mgr", mgr)
        object.__setattr__(self, "_lazy_state_id", state_id)

    def _materialize(self):
        return self._lazy_mgr._materialize_single_state(self._lazy_state_id)

    def __getattr__(self, name):
        # __getattr__ only fires when normal lookup misses, so the two slot
        # attributes never recurse. Guard private/dunder names so debugger /
        # pickle / inspect probes don't accidentally materialize the state.
        if name.startswith("_") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._materialize(), name)

    def __setattr__(self, name, value):
        if name in type(self).__slots__:
            object.__setattr__(self, name, value)
            return
        setattr(self._materialize(), name, value)

    def __repr__(self):
        return f"<_LazySimStateRef state_id={self._lazy_state_id}>"

    def __hash__(self):
        return hash(("_LazySimStateRef", self._lazy_state_id))

    def __eq__(self, other):
        if isinstance(other, _LazySimStateRef):
            return self._lazy_state_id == other._lazy_state_id
        return NotImplemented


class RustStateExportMixin:
    """Mixin providing state export/conversion methods for RustExplorationManager.

    This mixin expects the host class to have:
    - self._rust_mgr: The Rust exploration manager PyO3 object
    - self._project: The angr Project
    - self._state_cache: Dict[int, SimState]
    - self._state_roots: Dict[int, int]
    - self._lazy_state_refs: Dict[int, _LazySimStateRef]
    """

    def _invalidate_state_export_cache(self) -> None:
        """Clear the per-state `rust_fully_synced` sentinel on every cached state.

        Called when exploration resumes (`step`, `run`). After Rust takes
        further steps, the cached Python mirror is stale until re-synced by
        the next `_get_stash_states` visit.
        """
        for state in self._state_cache.values():
            scratch = getattr(state, "scratch", None)
            if scratch is not None and getattr(scratch, "rust_fully_synced", False):
                scratch.rust_fully_synced = False

    def _get_stash_states(self, stash: str) -> list:
        """Return lazy SimState references for the given stash.

        Iteration / ``len`` / identity comparisons on the result do NOT
        materialize the underlying SimStates. Materialization happens on
        first attribute access (``state.solver``, ``state.regs``, etc.) and
        is delegated to ``_materialize_single_state``.

        Args:
            stash: The stash name ('found', 'active', 'avoid', etc.)

        Returns:
            List of ``_LazySimStateRef`` objects (one per state id in the
            stash). Identity is preserved across repeated calls via
            ``self._lazy_state_refs``.
        """
        state_ids = self._rust_mgr.get_state_ids(stash)
        refs = self._lazy_state_refs
        result = []
        for sid in state_ids:
            ref = refs.get(sid)
            if ref is None:
                ref = _LazySimStateRef(self, sid)
                refs[sid] = ref
            result.append(ref)
        return result

    def _materialize_single_state(self, state_id: int):
        """Materialize one Rust state as a fully-synced angr SimState.

        Path order matches the legacy ``_get_stash_states`` logic:
          1. Cached SimState in ``_state_cache`` — re-sync if stale
             (``rust_fully_synced == False``).
          2. Parent-root copy if the root is cached.
          3. Currently-stepping-state copy if available in cache.
          4. Full ``export_state(state_id)`` snapshot as last resort.

        Always finishes with ``rust_fully_synced = True``, stdin content
        restore, and a posix weakref fix.

        INVARIANT (``invariant-callstack-sync-export-pipeline``; mirrored
        in ``callbacks.rs`` module invariant 4): every per-state sync
        helper — memory, registers, callstack, mmap_base, posix_brk —
        MUST be wired into ALL FOUR paths below. Path 1 routes through
        ``_sync_cached_state``; paths 2 and 3 call the individual
        ``_sync_rust_*`` helpers explicitly; path 4 calls them after the
        snapshot. Adding a new per-state sync that only updates one path
        produces stash-configuration-dependent divergence that is hard
        to debug.
        """
        state = self._state_cache.get(state_id)
        if state is not None:
            if not getattr(state.scratch, "rust_fully_synced", False):
                self._sync_cached_state(state, state_id)
            self._finalize_materialized_state(state)
            return state

        # Parent-root copy
        root = self._state_roots.get(state_id)
        if root is None:
            try:
                root = self._rust_mgr.get_state_root(state_id)
            except Exception:
                # cat-(a) EXPECTED CONTROL FLOW: probing for a Rust root id;
                # absence is normal for entry / unparented states.
                root = None
        if root is not None and root in self._state_cache:
            state = self._state_cache[root].copy()
            self._sync_cached_state(state, state_id)
            # Cache the copy so it stays alive (prevents weakref death
            # during chained attribute access like sm.active[1].posix.dumps()).
            self._state_cache[state_id] = state
            self._finalize_materialized_state(state)
            return state

        # Currently-stepping-state copy
        stepping_id = getattr(self, "_current_stepping_state_id", None)
        if stepping_id is not None and stepping_id in self._state_cache:
            state = self._state_cache[stepping_id].copy()
            self._sync_cached_state(state, state_id)
            self._state_cache[state_id] = state
            self._finalize_materialized_state(state)
            return state

        # Last resort: full snapshot export
        try:
            snapshot = self._rust_mgr.export_state(state_id)
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: export_state FFI failed; the caller
            # raises so the missing state is loud rather than silent.
            l.warning("export_state failed for state %d: %s", state_id, e)
            raise
        try:
            angr_state = self._snapshot_to_angr(snapshot)
            self._inject_rust_stdout(angr_state, snapshot.state_id)
            self._inject_rust_stdin(angr_state, snapshot.state_id)
            self._sync_rust_memory_to_state(angr_state, snapshot.state_id)
            self._sync_rust_registers_to_state(angr_state, snapshot.state_id)
            self._sync_rust_callstack_to_state(angr_state, snapshot.state_id)
            self._sync_rust_mmap_base_to_state(angr_state, snapshot.state_id)
            self._sync_rust_posix_brk_to_state(angr_state, snapshot.state_id)
            angr_state.scratch.rust_fully_synced = True
            self._state_cache[snapshot.state_id] = angr_state
            self._finalize_materialized_state(angr_state)
            return angr_state
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: caller expected a state and gets an
            # exception; surface the failure rather than returning a wrong
            # placeholder.
            l.warning("Failed to materialize state %d: %s", state_id, e)
            raise

    def _sync_cached_state(self, state, state_id: int) -> None:
        """Run the heavy plugin restore + register/memory/callstack sync.

        Sets ``state.scratch.rust_fully_synced = True`` on success. The
        sentinel is cleared by ``_invalidate_state_export_cache`` whenever
        Rust takes another step.
        """
        self._restore_plugins_to_state(state, state_id)
        self._inject_rust_stdout(state, state_id)
        self._inject_rust_stdin(state, state_id)
        # Attach Rust solver fallback BEFORE constraint sync. The Rust solver
        # has the correct constraints from exploration; constraint sync is
        # expensive and often causes identity mismatches.
        self._attach_rust_solver_fallback(state, state_id)
        self._sync_rust_memory_to_state(state, state_id)
        self._sync_rust_registers_to_state(state, state_id)
        self._sync_rust_callstack_to_state(state, state_id)
        self._sync_rust_mmap_base_to_state(state, state_id)
        self._sync_rust_posix_brk_to_state(state, state_id)
        state.scratch.rust_fully_synced = True

    def _finalize_materialized_state(self, state) -> None:
        """Per-state stdin restore + posix weakref fix.

        Equivalent to the per-state body of the old end-of-``_get_stash_states``
        loop; called once for every successful materialization.
        """
        if self._stdin_content:
            try:
                posix = getattr(state, "posix", None)
                if posix is not None:
                    stdin = getattr(posix, "stdin", None)
                    if stdin is not None and hasattr(stdin, "content") and not stdin.content:
                        stdin.content = list(self._stdin_content)
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: stdin restore best-effort; state's
                # posix plugin shape may differ from the template. Original
                # (empty) content remains.
                l.debug("Could not restore stdin content for state: %s", e)
        self._fix_posix_weakrefs(state)

    @staticmethod
    def _fix_posix_weakrefs(state):
        """Fix weakrefs in posix plugin's nested stream objects.

        After state caching, copying, or export, the weakrefs from
        SimPacketsStream objects (stdin/stdout/stderr) to their parent
        state can become stale. This refreshes them.
        """
        try:
            posix = getattr(state, "posix", None)
            if posix is None:
                return
            for attr in ("stdin", "stdout", "stderr"):
                child = getattr(posix, attr, None)
                if child is not None and hasattr(child, "set_state"):
                    child.set_state(state)
            if hasattr(posix, "fd") and posix.fd:
                for fd_obj in posix.fd.values():
                    if fd_obj is not None and hasattr(fd_obj, "set_state"):
                        fd_obj.set_state(state)
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: states without posix plugin
            # (custom factories, blank states) hit attribute / type errors
            # that we silently absorb. Weakref refresh is fix-up only.
            pass

    def _sync_rust_registers_to_state(self, state: angr.SimState, state_id: int):
        """Sync concrete register values from Rust state to Python state.

        After Rust executes code, register values computed during execution
        are only in Rust's state. This exports named registers and overwrites
        the Python state's values for registers that are concrete in Rust.
        """
        # angr-qj30 (write-through .2): when ``state.registers`` is a
        # ``RustRegisterProxy``, every read already routes live into Rust
        # by state_id — pushing the snapshot's concrete values onto the
        # proxy via setattr would overwrite any SYMBOLIC register the
        # SimProcedure just wrote through the proxy (e.g. a symbolic
        # return value), clobbering it with BVV(0) from
        # ``get_registers_named()`` (which returns the concrete portion
        # of Rust's register file only). Skip the sync entirely.
        from angr.exploration.rust_state_proxy import RustRegisterProxy

        regs_plugin = getattr(state, "registers", None)
        if isinstance(regs_plugin, RustRegisterProxy):
            # angr-4rq7: a materialized state that inherits a callback
            # ``RustRegisterProxy`` (either the live frame cached in
            # ``_state_cache`` or a ``.copy()`` of a parent-root state) carries
            # the proxy's per-name read ``_cache``, which is NEVER invalidated
            # when Rust steps the state. So ``state.addr`` (via
            # ``regs.ip`` -> ``proxy.load('ip')`` -> ``__getattr__('ip')``)
            # returns the value cached at callback time, while the
            # ``RustStateProxy`` accessor reads the LIVE pc via
            # ``get_state_pc_by_id``. That stale cache is the source of the
            # full-export-vs-proxy ``addr`` divergence that blocks promoting
            # the register-proxy write-through gate. Two repairs, both no-ops
            # for a freshly-bound live frame:
            #   1. ``RustRegisterProxy.copy()`` preserves the SOURCE
            #      ``_state_id``; rebind to the id we are materializing so
            #      reads route to the correct Rust state.
            #   2. Drop the stale read cache so every subsequent read goes
            #      live. We deliberately do NOT push the concrete snapshot via
            #      setattr (that would clobber a symbolic register the
            #      SimProcedure wrote through the proxy) — clearing the cache
            #      is sufficient because the proxy already reads through to
            #      Rust on a cache miss.
            if getattr(regs_plugin, "_state_id", None) != state_id:
                object.__setattr__(regs_plugin, "_state_id", state_id)
            cache = getattr(regs_plugin, "_cache", None)
            if cache:
                cache.clear()
            return
        try:
            snapshot = self._rust_mgr.export_state(state_id)
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: registers in Python state are now
            # stale. User code reading state.regs.* after exploration will
            # see pre-exploration values, not the values Rust computed.
            l.warning("export_state(%d) failed during register sync: %s — Python registers may be stale", state_id, e)
            return

        named_regs = snapshot.get_registers_named()
        for reg_name, (value, size_bits) in named_regs.items():
            try:
                setattr(state.regs, reg_name, claripy.BVV(value, size_bits))
            except Exception as e:
                # cat-(a) EXPECTED CONTROL FLOW: VEX internal registers
                # (e.g. ip_at_syscall) that angr's register plugin doesn't
                # expose. Log at debug only.
                l.debug("Skipping register %s during sync: %s", reg_name, e)

    def _sync_rust_mmap_base_to_state(self, state: angr.SimState, state_id: int):
        """Push Rust's per-state mmap_base into Python's state.heap.mmap_base.

        The native mmap syscall handler bumps Rust's mmap_base on addr=0 calls;
        without this sync, a subsequent Python-side allocation (SimProcedure
        fallback or unhandled syscall) reads a stale state.heap.mmap_base and
        hands out an address that overlaps a Rust-allocated region.

        Take max(rust, python) to avoid clobbering a Python-side advance that
        happened between Rust runs (e.g. the user calling state.heap.mmap_base
        = N before re-entering exploration).
        """
        heap = getattr(state, "heap", None)
        if heap is None or not hasattr(heap, "mmap_base"):
            return
        try:
            rust_base = self._rust_mgr.get_state_mmap_base(state_id)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: per-state mmap_base unavailable
            # (state may have been dropped from Rust). Python keeps its
            # current heap.mmap_base; a subsequent Python-side allocation
            # may overlap a Rust-allocated region. Debug only because
            # this state is likely no longer being explored.
            l.debug("get_state_mmap_base(%d) failed: %s", state_id, e)
            return
        if rust_base > heap.mmap_base:
            heap.mmap_base = rust_base

    def _sync_rust_posix_brk_to_state(self, state: angr.SimState, state_id: int):
        """Push Rust's per-state posix_brk into Python's state.posix.brk.

        Mirror of _sync_rust_mmap_base_to_state for the brk(2) heap pointer.
        NativeBrkSyscall bumps Rust's posix_brk on concrete grow calls; without
        this sync, a Python-side fallback (symbolic brk or set_brk collision
        retry) reads a stale state.posix.brk and hands out heap addresses
        overlapping a Rust-allocated region.

        Only applied when state.posix.brk is a plain int — Python's set_brk
        rewrites the field as a claripy BV (concrete BVV after a concrete
        bump, an If(...) tree after a symbolic bump). In those cases Python
        already advanced past the default and we leave the BV alone rather
        than risk an int<->BV type mismatch in downstream Python code.

        Take max(rust, python) to avoid clobbering a Python-side advance.
        """
        posix = getattr(state, "posix", None)
        if posix is None or not hasattr(posix, "brk"):
            return
        py_brk = posix.brk
        if not isinstance(py_brk, int):
            return
        try:
            rust_brk = self._rust_mgr.get_state_posix_brk(state_id)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: per-state posix_brk unavailable.
            # Same risk as mmap_base sync — Python-side fallback may
            # overlap a Rust heap region.
            l.debug("get_state_posix_brk(%d) failed: %s", state_id, e)
            return
        if rust_brk > py_brk:
            posix.brk = rust_brk

    def _sync_rust_callstack_to_state(self, state: angr.SimState, state_id: int):
        """Sync Rust-tracked call frames into state.callstack.

        Rust's call_stack (push order, outermost first) is the source of truth
        for any call/ret that happened during Rust execution.

        Two modes:

        * **Eager reconstruction** (default). The Python state was forked from
          a template before exploration, so its CallStack plugin does not
          reflect Rust-side push/pop. This rebuilds state.callstack as a
          linked list (top = most recent Rust call) and replaces the plugin
          via register_plugin. No-op if Rust has zero frames (preserves the
          template's empty CallStack so we don't clobber pre-Rust call
          history for forks made from non-entry states).
        * **Proxy install** (when ``self._use_export_callstack_proxy`` is on,
          angr-yk2g). Installs ``RustCallStackProxyPlugin`` bound to
          ``state_id`` instead — iteration / top-frame attribute access reads
          frames live from Rust via ``get_state_call_stack``. No eager FFI
          read or chain reconstruction. Matches the write-through model used
          for memory / registers / solver.
        """
        if getattr(self, "_use_export_callstack_proxy", False):
            from angr.exploration.rust_state_proxy import RustCallStackProxyPlugin

            try:
                proxy = RustCallStackProxyPlugin(self._rust_mgr, state_id)
                proxy.set_state(state)
                state.register_plugin("callstack", proxy)
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: proxy install failed; state
                # keeps its pre-Rust CallStack plugin.
                l.debug(
                    "RustCallStackProxyPlugin install failed for %d: %s",
                    state_id,
                    e,
                )
            return

        try:
            frames = self._rust_mgr.get_state_call_stack(state_id)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: state.callstack stays at the
            # template's pre-Rust history. Downstream callers reading
            # callstack will see fewer frames than Rust observed.
            l.debug("get_state_call_stack(%d) failed: %s", state_id, e)
            return

        if not frames:
            return

        from angr.state_plugins.callstack import CallStack

        chain = CallStack()
        for call_site, func, ret_addr, sp in frames:
            chain = CallStack(
                call_site_addr=call_site,
                func_addr=func,
                stack_ptr=sp,
                ret_addr=ret_addr,
                jumpkind="Ijk_Call",
                next_frame=chain,
            )

        try:
            state.register_plugin("callstack", chain)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: register_plugin failed; the
            # state keeps its pre-Rust callstack plugin. Downstream
            # consumers that depend on Rust-tracked frames will see
            # incomplete history.
            l.debug("register_plugin('callstack') failed for %d: %s", state_id, e)

    def _sync_rust_memory_to_state(self, state: angr.SimState, state_id: int):
        """Sync memory from Rust state to Python state.

        Two modes:

        * **Eager writeback** (default). After Rust executes code, the
          materialized SimState's claripy memory has whatever bytes were on
          the template at fork time. This method exports the Rust state's
          memory pages and symbolic expressions, and pushes them into
          ``state.memory`` via ``store(...)`` so ``state.memory.load()``
          returns current values.
        * **Proxy install** (when ``self._use_export_memory_proxy`` is on,
          angr-ul4k). Installs ``RustMemoryProxy`` bound to ``state_id`` as
          ``state.memory`` instead — every ``load`` / ``store`` after that
          routes directly into Rust by ``state_id`` (loads use
          ``get_state_memory_ast`` for symbolic-AST round-trip). No FFI
          page export and no writeback into the SimState. Matches the
          write-through model used for callstack / registers / solver.
        """
        if getattr(self, "_use_export_memory_proxy", False):
            from angr.exploration.rust_state_proxy import RustMemoryProxy

            try:
                proxy = RustMemoryProxy(self._rust_mgr, state_id, state.arch)
                state.register_plugin("memory", proxy)
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: proxy install failed; the
                # state keeps its pre-Rust claripy memory. Downstream
                # consumers that read addresses Rust wrote will see stale
                # template bytes.
                l.debug(
                    "RustMemoryProxy install failed for %d: %s",
                    state_id,
                    e,
                )
            # Symbolic-AST sync (_sync_rust_symbolic_objects_to_state) is
            # for pushing Rust-computed claripy ASTs into Python memory; the
            # proxy reads them live via get_state_memory_ast so it would be
            # a no-op duplicate at best. Skipped under the gate.
            return

        try:
            # Use flushed export to materialize any pending symbolic writes
            # before exporting memory pages to Python.
            snapshot = self._rust_mgr.export_state_flushed(state_id)
        except Exception as e_flushed:
            try:
                snapshot = self._rust_mgr.export_state(state_id)
                # cat-(c) WRONG-ANSWER RISK: flushed failed but unflushed
                # worked — pending symbolic writes may not be visible in
                # Python memory. WARN so the user notices.
                l.warning(
                    "export_state_flushed(%d) failed (%s); falling "
                    "back to unflushed snapshot — pending symbolic "
                    "stores may be missing",
                    state_id,
                    e_flushed,
                )
            except Exception as e_unflushed:
                # cat-(c) WRONG-ANSWER RISK: both export paths failed.
                # Python state.memory will return zeros (or stale data)
                # for any address Rust wrote.
                l.warning(
                    "Both export_state_flushed and export_state failed "
                    "for state %d (flushed: %s; unflushed: %s) — "
                    "Python memory will be stale",
                    state_id,
                    e_flushed,
                    e_unflushed,
                )
                return  # State may not be in Rust stashes anymore

        for i in range(snapshot.page_count()):
            page = snapshot.get_page(i)
            if page is None:
                continue
            page_addr, data, _perms, symbolic_offsets = page
            if symbolic_offsets:
                # Write concrete byte ranges while preserving symbolic regions.
                # symbolic_offsets are byte positions within the page where Python
                # has BVS values — writing zeros there would corrupt them.
                sym_set = set(symbolic_offsets)
                # Find contiguous concrete runs
                start = None
                for j in range(len(data) + 1):
                    if j < len(data) and j not in sym_set:
                        if start is None:
                            start = j
                    else:
                        if start is not None:
                            chunk = data[start:j]
                            try:
                                state.memory.store(
                                    page_addr + start,
                                    claripy.BVV(chunk, len(chunk) * 8),
                                    endness="Iend_BE",
                                    inspect=False,
                                    disable_actions=True,
                                )
                            except Exception as e:
                                # cat-(c) WRONG-ANSWER RISK: a concrete
                                # byte range from Rust failed to write into
                                # Python memory. state.memory.load() of
                                # this region will return whatever the
                                # Python state held pre-exploration (often
                                # zero or a stale BVS).
                                l.warning("Failed to sync concrete chunk to 0x%x: %s", page_addr + start, e)
                            start = None
            else:
                try:
                    # Raw bytes from Rust are in memory order; use Iend_BE so angr
                    # stores them as-is without byte-reversing.
                    state.memory.store(
                        page_addr,
                        claripy.BVV(data, len(data) * 8),
                        endness="Iend_BE",
                        inspect=False,
                        disable_actions=True,
                    )
                except Exception as e:
                    # cat-(c) WRONG-ANSWER RISK: full-page sync failed.
                    # All addresses on this page in Python memory are
                    # stale.
                    l.warning("Failed to sync page at 0x%x: %s", page_addr, e)

        # Export Rust-computed symbolic expressions to Python memory.
        # These are Expression values computed by the Rust VEX interpreter
        # (e.g., flag computations in asisctf). Python-imported BVS values
        # are excluded — they already have proper claripy identity in Python.
        self._sync_rust_symbolic_objects_to_state(state, state_id)

    def _sync_rust_symbolic_objects_to_state(self, state: angr.SimState, state_id: int):
        """Export Rust-computed symbolic expressions as claripy ASTs into Python memory.

        Uses the shared Z3 context: Rust builds Z3 ASTs in Python's Z3 context,
        so we can directly wrap the raw Z3_ast pointers as z3.BitVecRef objects
        and convert them to claripy ASTs via claripy.backends.z3._abstract().
        """
        try:
            sym_asts = self._rust_mgr.get_state_symbolic_z3_asts(state_id)
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: Rust-computed symbolic objects
            # are not exported to Python memory. state.memory.load() of
            # those addresses will see concrete bytes (or template BVS),
            # not the symbolic expression Rust computed.
            l.warning("get_state_symbolic_z3_asts(%d) failed: %s — Rust-side symbolic memory not synced", state_id, e)
            return

        if not sym_asts:
            return

        import ctypes

        import z3 as z3mod

        z3_backend = claripy.backends.z3
        z3_ctx = z3_backend._context  # The shared Z3 context

        for addr, z3_ast_ptr, width_bits in sym_asts:
            try:
                # Wrap raw Z3_ast pointer as z3.BitVecRef
                ast_wrapper = ctypes.c_void_p(z3_ast_ptr)
                ast_wrapper.__class__ = z3mod.z3types.Ast
                ast_wrapper._as_parameter_ = z3_ast_ptr
                z3_bv = z3mod.BitVecRef(ast_wrapper, z3_ctx)

                # Convert to claripy AST
                claripy_ast = z3_backend._abstract(z3_bv)

                # Store using the architecture's endianness to match VEX IR
                # store operations (STle for x86-64, etc.)
                state.memory.store(
                    addr,
                    claripy_ast,
                    endness=state.arch.memory_endness,
                    inspect=False,
                    disable_actions=True,
                )
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: per-AST sync failed (e.g.
                # ctypes binding incompatible with this z3 version). The
                # other ASTs on this state still get synced, but this one
                # is missing — Python sees concrete bytes for the address.
                l.debug("Failed to sync symbolic object at 0x%x: %s", addr, e)

    def _attach_rust_solver_fallback(self, state, state_id):
        """Wrap state.solver so eval/min/max/satisfiable defer to the Rust solver.

        The Rust solver holds the authoritative constraints from exploration;
        syncing them to Python is expensive and can cause identity mismatches.
        Re-attaching the same state is a no-op.
        """
        RustSolverFallback(state, state_id, self._rust_mgr).attach()

    @property
    def found_states(self) -> list:
        """Get found states as angr SimStates.

        This converts Rust exploration states back to angr SimStates that
        can be used to evaluate symbolic values and get solutions.

        Returns:
            List of angr SimStates with constraints and symbolic values.
        """
        states = []
        snapshots = self._rust_mgr.export_found_states()
        for snapshot in snapshots:
            try:
                angr_state = self._snapshot_to_angr(snapshot)
                states.append(angr_state)
            except Exception as e:
                # cat-(c) WRONG-ANSWER RISK: a found state is dropped
                # from the result list. Caller may report fewer
                # solutions than the engine actually found.
                l.warning(f"Failed to convert state {snapshot.state_id}: {e}")
        return states

    def get_state_by_id(self, state_id: int):
        """Get a specific state by ID as an angr SimState.

        Args:
            state_id: The Rust state ID.

        Returns:
            angr SimState, or None if not found.
        """
        try:
            snapshot = self._rust_mgr.export_state(state_id)
            return self._snapshot_to_angr(snapshot)
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: returning None hides the
            # underlying failure. Callers checking "if state is None"
            # may treat this as "state doesn't exist" when in fact
            # the state exists but failed to convert.
            l.warning(f"Failed to get state {state_id}: {e}")
            return None

    def _snapshot_to_angr(self, snapshot) -> angr.SimState:
        """Convert a Rust state snapshot to an angr SimState.

        Creates an angr SimState from the snapshot data, including:
        - Register values
        - Memory contents
        - Basic state metadata

        Note: Symbolic values are evaluated using the Rust solver context
        since the Z3 constraints cannot be directly transferred.

        Args:
            snapshot: The state snapshot from Rust.

        Returns:
            An angr SimState.
        """
        state = self._project.factory.blank_state(addr=snapshot.pc)

        self._load_snapshot_registers(state, snapshot)
        self._load_snapshot_pages(state, snapshot, self._project.arch)

        state.scratch.rust_state_id = snapshot.state_id
        state.scratch.rust_parent_id = snapshot.parent_id

        self._restore_plugins_to_state(
            state,
            snapshot.state_id,
            snapshot_parent_id=snapshot.parent_id,
        )
        # Eval/min/max/satisfiable defer to the Rust solver — required because
        # _apply_symbolic_constraints no longer pre-pins recovered symbols
        # (pre-pinning was UNSAT-prone, see pre-pinning-dangerous memo).
        # Idempotent: re-attach is a no-op via the _ATTACH_FLAG guard.
        self._attach_rust_solver_fallback(state, snapshot.state_id)
        return state

    def _load_snapshot_registers(self, state: angr.SimState, snapshot):
        """Restore register values from snapshot using Rust's named register export."""
        named_regs = snapshot.get_registers_named()
        for reg_name, (value, size_bits) in named_regs.items():
            try:
                setattr(state.regs, reg_name, claripy.BVV(value, size_bits))
            except Exception:
                # cat-(a) EXPECTED CONTROL FLOW: Skip VEX internal
                # registers that angr doesn't expose (ip_at_syscall etc.).
                pass

    def _load_snapshot_pages(self, state: angr.SimState, snapshot, arch):
        """Load memory pages from snapshot, restoring symbolic regions with original ASTs."""
        for i in range(snapshot.page_count()):
            page = snapshot.get_page(i)
            if page is None:
                continue
            page_addr, data, _perms, symbolic_offsets = page
            try:
                state.memory.store(
                    page_addr, claripy.BVV(data, len(data) * 8), endness=arch.memory_endness, inspect=False
                )
                if not symbolic_offsets:
                    continue
                regions = self._find_contiguous_regions(symbolic_offsets)
                self._restore_symbolic_regions(
                    state,
                    snapshot,
                    page_addr,
                    regions,
                    arch,
                )
            except Exception as e:
                # cat-(c) WRONG-ANSWER RISK: page failed to load. State
                # memory at this page is left at its blank-state default,
                # so loads return zeros instead of Rust's values.
                l.warning(f"Failed to load page at 0x{page_addr:x}: {e}")

    @staticmethod
    def _find_contiguous_regions(symbolic_offsets) -> list:
        """Group sorted byte offsets into contiguous (offset, size) regions."""
        sorted_offsets = sorted(symbolic_offsets)
        regions = []
        start = sorted_offsets[0]
        end = start
        for offset in sorted_offsets[1:]:
            if offset == end + 1:
                end = offset
            else:
                regions.append((start, end - start + 1))
                start = offset
                end = offset
        regions.append((start, end - start + 1))
        return regions

    def _restore_symbolic_regions(self, state, snapshot, page_addr, regions, arch) -> None:
        """Overwrite symbolic regions on a page with recovered or fresh symbols."""
        for offset, size in regions:
            sym_addr = page_addr + offset
            ast = self._recover_symbolic_ast(snapshot, sym_addr, size)
            if ast is None:
                # Fallback for symbols created in Rust without a tracked AST.
                # angr-4o7d (2026-05-22): instrumented to measure fire rate.
                # Snapshot-export path, not hot exploration loop — distinct
                # threat model from rust_manager.py's mem_thunk_/sym_load_full_fail_
                # fallbacks (which counter angr-ymoe showed are dead in practice).
                sym_name = f"rust_sym_{sym_addr:x}_{snapshot.state_id}"
                ast = claripy.BVS(sym_name, size * 8)
                self._stats_orphan_bvs_snapshot_restore += 1
            state.memory.store(sym_addr, ast, endness=arch.memory_endness, inspect=False)

    def _recover_symbolic_ast(self, snapshot, sym_addr: int, size: int):
        """Look up the original claripy AST for a symbolic byte at sym_addr.

        Searches address-tracked AST maps for the snapshot's state, parent, and
        root, then the hook-symbolic-memory map. Returns the first AST whose
        tracked size matches; None if no match.
        """
        candidate_ids = [snapshot.state_id]
        if snapshot.parent_id >= 0:
            candidate_ids.append(snapshot.parent_id)
        root_id = self._state_roots.get(snapshot.state_id)
        if root_id is None:
            try:
                root_id = self._rust_mgr.get_state_root(snapshot.state_id)
            except Exception:
                # cat-(a) EXPECTED CONTROL FLOW: probing for Rust root id;
                # absence is normal for entry / unparented states. We
                # then search only state_id and parent_id for the AST.
                root_id = None
        if root_id is not None and root_id != snapshot.state_id:
            candidate_ids.append(root_id)

        for sid in candidate_ids:
            if sid is None:
                continue
            addr_map = self._rust_mgr.get_state_addr_to_ast(sid)
            entry = addr_map.get(sym_addr)
            if entry is not None and entry[1] == size:
                return entry[0]

        hook_map = self._rust_mgr.get_state_hook_symbolic_memory(snapshot.state_id)
        hook_entry = hook_map.get(sym_addr)
        if hook_entry is not None and hook_entry[1] == size:
            return hook_entry[0]
        return None

    def _find_plugin_template_state(
        self,
        state_id: int,
        snapshot_parent_id: int | None = None,
    ) -> angr.SimState | None:
        """Locate the best SimState to source plugins (posix/libc/heap/fs/log) from.

        Walks ancestors in proximity order (closest first):

        1. ``state_id`` itself, if cached. The state's own plugins reflect
           its specific mutations and are always the right choice.
        2. ``snapshot_parent_id`` (immediate parent from a snapshot), when
           the caller has one. Plugin state at fork time is the closest
           thing to "what this state should have started with."
        3. ``_state_roots[state_id]`` then a Rust-side ``get_state_root``
           probe. The root is an entry state — its plugins are the
           unmutated baseline that descendants forked from.
        4. As a last resort, any other cached *root* state. Roots have
           clean baseline plugins so falling back to an unrelated root is
           safe — it may report the wrong ``posix.argv``, but it cannot
           leak another path's mid-exploration mutations (e.g., open fd
           tables, heap allocations) into this state.

        Never falls back to "the first cached state": that was a
        correctness landmine (angr-2k64) — the arbitrary pick could be a
        forked descendant whose plugin state has mutations belonging to
        a different exploration branch.

        Returns ``None`` if nothing usable is cached.
        """
        cached = self._state_cache.get(state_id)
        if cached is not None:
            return cached

        if snapshot_parent_id is not None and snapshot_parent_id >= 0:
            cached = self._state_cache.get(snapshot_parent_id)
            if cached is not None:
                return cached

        root_id = self._state_roots.get(state_id)
        if root_id is None or root_id == state_id:
            try:
                rust_root = self._rust_mgr.get_state_root(state_id)
                if rust_root is not None:
                    root_id = rust_root
            except Exception:
                # cat-(a) EXPECTED CONTROL FLOW: Rust root probe failed
                # (state already evicted / unknown id). Falls through to
                # the any-cached-root fallback below.
                pass
        if root_id is not None:
            cached = self._state_cache.get(root_id)
            if cached is not None:
                return cached

        # Last-resort: any cached *root*. Skip non-root descendants — they
        # carry per-fork mutations that must not leak across paths.
        for candidate_root in self._state_roots.values():
            cached = self._state_cache.get(candidate_root)
            if cached is not None:
                return cached

        return None

    def _restore_plugins_to_state(
        self,
        state: angr.SimState,
        state_id: int,
        snapshot_parent_id: int | None = None,
    ):
        """Restore plugins to an exported state from the closest cached ancestor.

        Exported states are missing critical plugins (posix, libc, heap)
        that scripts expect. This method copies them from the nearest
        cached ancestor SimState — preferring the state itself, then its
        snapshot parent, then its tracked root.

        Args:
            state: The state to restore plugins to.
            state_id: The Rust state ID for lookup.
            snapshot_parent_id: Optional immediate parent ID from a state
                snapshot, used to prefer a same-branch ancestor over the
                generic root template.
        """
        template = self._find_plugin_template_state(state_id, snapshot_parent_id)

        if template is None:
            l.debug("No template state found for plugin restoration (state %d)", state_id)
            _install_rust_history_warning(state)
            return

        # Copy plugins that are commonly needed
        plugins_to_restore = ["posix", "libc", "heap", "fs", "log"]

        for plugin_name in plugins_to_restore:
            try:
                if hasattr(template, plugin_name):
                    plugin = getattr(template, plugin_name)
                    if plugin is not None and hasattr(plugin, "copy"):
                        # Only copy if not already present
                        if not hasattr(state, plugin_name) or getattr(state, plugin_name) is None:
                            state.register_plugin(plugin_name, plugin.copy())
                            l.debug(f"Restored {plugin_name} plugin to state")
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: per-plugin restore failed
                # (incompatible copy()). State is left without that
                # plugin; downstream code that accesses it will fall
                # through to angr's default plugin instantiation.
                l.debug(f"Could not restore {plugin_name} plugin: {e}")

        # Fix nested plugin state references for posix (stdin/stdout/stderr)
        # These nested SimPacketsStream objects hold weakrefs to the state that
        # can become stale after state caching/export cycles
        try:
            posix = getattr(state, "posix", None)
            if posix is not None:
                for attr in ("stdin", "stdout", "stderr"):
                    child = getattr(posix, attr, None)
                    if child is not None and hasattr(child, "set_state"):
                        child.set_state(state)
                # Also fix fd entries
                if hasattr(posix, "fd") and posix.fd:
                    for fd_obj in posix.fd.values():
                        if fd_obj is not None and hasattr(fd_obj, "set_state"):
                            fd_obj.set_state(state)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: posix weakref refresh failed
            # (custom posix plugin shape). Stale weakrefs may surface
            # later as AttributeError when the user inspects stdin/
            # stdout/stderr.
            l.debug(f"Could not fix posix nested state refs: {e}")

        # Promote state.history to the warn-on-read variant. Catches the
        # silent-divergence case where users rely on state.history.actions /
        # state.history.events (populated under Python by TRACK_*_ACTIONS /
        # TRACK_MEMORY_MAPPING; never populated under Rust). The relevant
        # options ship in the default 'symbolic' bundle, so we can't warn on
        # add() without spamming entry_state(); the read-time hook fires
        # only when the empty stream is actually consumed.
        _install_rust_history_warning(state)

    def eval_memory(self, state_id: int, addr: int, size: int) -> bytes | None:
        """Evaluate memory from a Rust state's solver context.

        Args:
            state_id: The Rust state ID.
            addr: Memory address to evaluate.
            size: Number of bytes to read.

        Returns:
            Concrete byte value, or None if evaluation fails.
        """
        result = self._rust_mgr.get_state_memory(state_id, addr, size)
        if result is not None:
            return bytes(result)
        return None
