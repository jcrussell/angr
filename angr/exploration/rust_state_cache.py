"""Mixin for State caching, predicate evaluation, and handle management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import angr

l = logging.getLogger(name=__name__)
_DBG = l.isEnabledFor(logging.DEBUG)


class RustStateCacheMixin:
    """State caching, predicate evaluation, and handle management

    This mixin expects the host class to have the standard
    RustExplorationManager attributes (self._rust_mgr, self._project, etc.).
    """

    def _get_default_state(self) -> angr.SimState | None:
        """Get a default state for callbacks."""
        if self._state_cache:
            return next(iter(self._state_cache.values()))
        return None

    def _get_callback_state(self) -> angr.SimState | None:
        """Get current callback state for memory access during hooks.

        This returns the state that was set when entering a callback,
        allowing memory_load to access the correct symbolic memory context.
        """
        return self._callback_state

    def _set_callback_state(self, state: angr.SimState | None):
        """Set the current callback state for memory access."""
        self._callback_state = state

    def _register_handle(self, handle_id: int, ast: object, addr: int = None, size: int = None, state_id: int = None):
        """Pin a claripy AST to a state address for export recovery.

        When an AST is stored at a concrete address (``addr`` + ``state_id``
        given), it is recorded in the Rust-side per-state ``addr_to_ast`` map
        via ``set_state_addr_to_ast``. Rust holds its own strong ``PyObject``
        ref, so the AST survives until the owning ``RustSimState`` drops — this
        is the path ``_get_stash_states`` uses to recover symbolic memory on
        export (rust_state_export.py).

        Calls without ``addr``/``state_id`` are no-ops: there is no Python-side
        handle cache to populate. The ``handle_id`` argument is retained only
        for call-site compatibility.
        """
        if addr is None or state_id is None:
            return

        # Track address -> AST mapping for state export recovery.
        # Use effective state ID so forked states share parent's data.
        effective_id = self._get_effective_state_id(state_id)
        if effective_id is None:
            return
        actual_size = size if size is not None else (ast.length // 8 if hasattr(ast, "length") else 1)
        try:
            self._rust_mgr.set_state_addr_to_ast(effective_id, addr, ast, actual_size)
        except Exception as e:
            # cat-(a) EXPECTED CONTROL FLOW: state may have been dropped
            # from Rust between fork and registration; the addr->AST
            # link is best-effort and missing it is fine.
            l.debug("set_state_addr_to_ast(sid=%d, addr=%#x) failed: %s: %s", effective_id, addr, type(e).__name__, e)

    def _cleanup_symbolic_pages_cache(self):
        """No-op shim retained for back-compat hooks.

        Per-state metadata storage moved into Rust (``RustSimState``); the
        Python-side ``_state_metadata`` dict no longer exists, and metadata is
        freed automatically when Rust drops the owning state. The eviction
        path is therefore unnecessary.
        """
        return

    def _evaluate_predicates_on_active(self):
        """Evaluate callable find/avoid predicates on all Rust states.

        When find/avoid are callables (not addresses), the Rust engine can't
        evaluate them. Checks both cached Python states and uncached Rust-only
        states (created by Rust forking without SimProcedure callbacks).

        Uses change-detection caching: states are only re-evaluated when their
        address or stdout length has changed since the last evaluation. This
        avoids redundant predicate calls (which involve FFI + state creation).

        States already moved to found/avoid stashes are excluded by the stash
        query (only active + deadended are checked).
        """
        from angr.exploration.rust_state_proxy import RustStateProxy

        # Track states already moved to found/avoid to avoid re-moving them
        if not hasattr(self, "_predicate_matched_ids"):
            self._predicate_matched_ids = set()

        # Cache: state_id -> (addr, stdout_len) at last evaluation time
        # If state's current (addr, stdout_len) matches, skip re-evaluation
        if not hasattr(self, "_predicate_eval_cache"):
            self._predicate_eval_cache = {}

        found_sids = set()
        avoid_sids = set()
        _proxy_states = {}  # state_id -> RustStateProxy for uncached states

        # Collect (state_id, addr, stdout_len) from active + deadended in bulk
        # This is much cheaper than per-state FFI calls
        state_info = {}  # state_id -> (addr, stdout_len)
        for stash in ("active", "deadended"):
            try:
                for sid, addr, stdout_len in self._rust_mgr.get_state_predicate_info(stash):
                    state_info[sid] = (addr, stdout_len)
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: missing predicate info means
                # change-detection cache won't activate for this stash; we
                # still fall through to the cached-state path below.
                l.debug("get_state_predicate_info(stash=%s) failed: %s: %s", stash, type(e).__name__, e)

        # Also include cached states (may have been moved between stashes)
        all_state_ids = set(state_info.keys())
        all_state_ids.update(self._state_cache.keys())

        for state_id in all_state_ids:
            if state_id in self._predicate_matched_ids:
                continue

            # Get or create a state for predicate evaluation
            state = self._state_cache.get(state_id)
            is_cached = state is not None

            # Change-detection cache: only for uncached (Rust-only) states.
            # Cached states may have Python-side stdout from SimProcedure
            # callbacks that isn't reflected in Rust's stdout_len.
            current_info = state_info.get(state_id)
            if not is_cached and current_info is not None:
                cached_info = self._predicate_eval_cache.get(state_id)
                if cached_info == current_info:
                    continue

            if state is None:
                # Uncached state — use RustStateProxy for live register/memory access
                stdout_data = b""
                try:
                    stdout_data = bytes(self._rust_mgr.get_state_stdout(state_id))
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: empty stdout when retrieval
                    # fails; predicate that checks stdout content sees b"".
                    l.debug("get_state_stdout(sid=%d) failed: %s: %s", state_id, type(e).__name__, e)
                state = RustStateProxy(
                    self._rust_mgr,
                    state_id,
                    project=self._project,
                    stdin_vars=getattr(self, "_stdin_vars", None),
                    stdout_data=stdout_data,
                    python_mgr=self,
                )
                _proxy_states[state_id] = state

            try:
                if not isinstance(state, RustStateProxy):
                    self._inject_rust_stdout(state, state_id)
                    self._inject_rust_stdin(state, state_id)

                if self._find_predicate is not None:
                    try:
                        if self._find_predicate(state):
                            found_sids.add(state_id)
                            if not is_cached and current_info is not None:
                                self._predicate_eval_cache[state_id] = current_info
                            continue
                    except Exception as e:
                        # cat-(b) FALLBACK WITH LOSS: user predicate raised; we
                        # skip find for this state and fall through to avoid.
                        # User-code bug — log at debug.
                        l.debug("find_predicate(sid=%d) raised: %s: %s", state_id, type(e).__name__, e)

                if self._avoid_predicate is not None:
                    try:
                        if self._avoid_predicate(state):
                            avoid_sids.add(state_id)
                    except Exception as e:
                        # cat-(b) FALLBACK WITH LOSS: user predicate raised; we
                        # don't avoid this state. User-code bug — log at debug.
                        l.debug("avoid_predicate(sid=%d) raised: %s: %s", state_id, type(e).__name__, e)
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: predicate eval setup (stdout/
                # stdin injection) failed; this state goes unevaluated this
                # sweep but will be retried next sweep.
                l.debug("predicate eval setup for sid=%d failed: %s: %s", state_id, type(e).__name__, e)

            # Mark uncached state as evaluated at its current (addr, stdout_len)
            if not is_cached and current_info is not None:
                self._predicate_eval_cache[state_id] = current_info

        # Move matched states to found/avoid stashes
        for sid in found_sids:
            self._predicate_matched_ids.add(sid)
            for stash in ("active", "deadended"):
                try:
                    if self._rust_mgr.move_state(sid, stash, "found"):
                        break
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: state classified as found but
                    # the move failed; the next sweep marks it matched again
                    # (via _predicate_matched_ids) but won't move it. Log
                    # so this is visible — could mask a real find failure.
                    l.debug("move_state(sid=%d, %s -> found) failed: %s: %s", sid, stash, type(e).__name__, e)
            # Record in Python-side found for predicate-matched states
            if not hasattr(self, "_predicate_found"):
                self._predicate_found = []
            if sid in self._state_cache:
                self._predicate_found.append(self._state_cache[sid])
            elif sid in _proxy_states:
                self._predicate_found.append(_proxy_states[sid])

        for sid in avoid_sids:
            self._predicate_matched_ids.add(sid)
            for stash in ("active", "deadended"):
                try:
                    self._rust_mgr.move_state(sid, stash, "avoid")
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: state classified as avoid but
                    # the move failed; state remains active and may continue
                    # being explored. Log at debug.
                    l.debug("move_state(sid=%d, %s -> avoid) failed: %s: %s", sid, stash, type(e).__name__, e)

    def _create_state_for_predicate(self, state_id):
        """Create a lightweight Python state for predicate evaluation.

        For states created purely in Rust (no SimProcedure callback), we need
        a Python state to evaluate callable predicates. Creates one by copying
        from the closest cached ancestor state.

        Returns None if no suitable parent can be found.
        """
        # Try to find the root state in cache
        try:
            root_id = self._rust_mgr.get_state_root(state_id)
        except Exception as e:
            # cat-(a) EXPECTED CONTROL FLOW: missing Rust root means we fall
            # back to "any cached state" below — the parent search is
            # best-effort.
            l.debug("get_state_root(sid=%d) failed: %s: %s", state_id, type(e).__name__, e)
            root_id = None

        parent_state = None
        if root_id is not None and root_id in self._state_cache:
            parent_state = self._state_cache[root_id]
        else:
            # Fall back to any cached state (typically the init state)
            if self._state_cache:
                parent_state = next(iter(self._state_cache.values()))

        if parent_state is None:
            return None

        try:
            state = parent_state.copy()
            # Update PC from Rust state
            try:
                for stash in ("active", "deadended", "found", "avoid"):
                    ids = self._rust_mgr.get_state_ids(stash)
                    if state_id in ids:
                        idx = ids.index(state_id)
                        pc = self._rust_mgr.get_state_pc(stash, idx)
                        if pc is not None:
                            state.regs._ip = pc
                        break
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: PC sync failed; the predicate
                # state has the parent's PC, not the forked child's. Could
                # cause a predicate to match the wrong addr.
                l.debug("PC sync for predicate state sid=%d failed: %s: %s", state_id, type(e).__name__, e)
            # Attach Rust solver fallback so solver operations work
            self._attach_rust_solver_fallback(state, state_id)
            return state
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: parent.copy() failed; this state
            # gets no Python predicate eval this sweep. Caller (None return)
            # falls back to using a RustStateProxy.
            l.debug("Failed to create predicate state for %d: %s", state_id, e)
            return None
