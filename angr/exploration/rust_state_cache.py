"""Mixin for State caching, predicate evaluation, and handle management."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import angr

l = logging.getLogger(name=__name__)
_DBG = l.isEnabledFor(logging.DEBUG)


class RustStateCacheMixin:
    """State caching, predicate evaluation, and handle management

    This mixin expects the host class to have the standard
    RustExplorationManager attributes (self._rust_mgr, self._project, etc.).
    """

    def _get_default_state(self) -> Optional["angr.SimState"]:
        """Get a default state for callbacks."""
        if self._state_cache:
            return next(iter(self._state_cache.values()))
        return None

    def _get_callback_state(self) -> Optional["angr.SimState"]:
        """Get current callback state for memory access during hooks.

        This returns the state that was set when entering a callback,
        allowing memory_load to access the correct symbolic memory context.
        """
        return self._callback_state

    def _set_callback_state(self, state: Optional["angr.SimState"]):
        """Set the current callback state for memory access."""
        self._callback_state = state

    def _lookup_handle(self, handle_id: int) -> Optional[object]:
        """Look up a claripy AST by its handle ID.

        When claripy ASTs are passed to Rust, they get assigned handle IDs
        that allow us to look them up later for constraint reconstruction.

        Checks both the handle cache and the identity tracker to ensure
        proper symbol identity preservation across FFI boundary.

        Args:
            handle_id: The handle ID assigned by Rust.

        Returns:
            The claripy AST if found, None otherwise.
        """
        # Check handle cache first (fast path)
        result = self._ast_handle_cache.get(handle_id)
        if result is not None:
            return result

        # Fall back to identity tracker (may have been evicted from handle cache)
        return self._identity_tracker.get_original_ast(handle_id)

    def _register_handle(self, handle_id: int, ast: object, addr: int = None, size: int = None, state_id: int = None):
        """Register a claripy AST with its handle ID for later lookup.

        Uses an LRU eviction strategy that preserves actively referenced handles.
        Handles marked as active (via _mark_handle_active) are never evicted.

        This also registers the AST with the identity tracker to ensure
        that the same AST is returned when exported from Rust.

        Args:
            handle_id: The handle ID assigned by Rust.
            ast: The claripy AST to cache.
            addr: Optional memory address where this AST is stored.
            size: Optional size in bytes of the AST.
            state_id: Optional state ID for address tracking.
        """
        self._ast_handle_cache[handle_id] = ast

        # Also register with identity tracker for bidirectional lookup
        self._identity_tracker.register(ast, handle_id)

        # Track address -> AST mapping for state export recovery.
        # Use effective state ID so forked states share parent's data.
        if addr is not None and state_id is not None:
            effective_id = self._get_effective_state_id(state_id)
            if effective_id is not None:
                actual_size = size if size is not None else (ast.length // 8 if hasattr(ast, 'length') else 1)
                try:
                    self._rust_mgr.set_state_addr_to_ast(effective_id, addr, ast, actual_size)
                except Exception as e:
                    # State may have been dropped from Rust between fork and
                    # registration; the addr->AST link is best-effort.
                    l.debug("set_state_addr_to_ast(sid=%d, addr=%#x) failed: %s: %s",
                            effective_id, addr, type(e).__name__, e)

        # Limit cache size to prevent memory issues
        # Use smarter eviction that preserves active handles
        if len(self._ast_handle_cache) > 10000:
            # Get set of active handles (those referenced by current states)
            active_handles = self._get_active_handles()

            # Remove oldest entries that are not active
            eviction_count = 0
            max_evict = 5000
            to_remove = []

            for k in list(self._ast_handle_cache.keys()):
                if eviction_count >= max_evict:
                    break
                if k not in active_handles:
                    to_remove.append(k)
                    eviction_count += 1

            for k in to_remove:
                del self._ast_handle_cache[k]

            l.debug(f"Evicted {len(to_remove)} handles from cache "
                    f"(preserved {len(active_handles)} active)")

    def _get_active_handles(self) -> set:
        """Get the set of handle IDs that are actively in use.

        Returns handle IDs referenced by:
        - Rust pending state (stored conditions, deferred forks)
        - Pending callback state
        - Cached states in state_cache
        - Any constraints in current exploration

        Returns:
            Set of handle IDs that should not be evicted.
        """
        active = set()

        # Query Rust for actively referenced handles
        # These are condition IDs and deferred fork handles that must not be evicted
        if hasattr(self._rust_mgr, 'get_active_handle_ids'):
            try:
                rust_handles = self._rust_mgr.get_active_handle_ids()
                active.update(rust_handles)
            except Exception as e:
                l.debug(f"Could not get active handles from Rust: {e}")

        # Add handles from pending state constraints
        if hasattr(self, '_pending_handles') and self._pending_handles:
            active.update(self._pending_handles)

        # Don't evict recently used handles (LRU protection)
        # Keep the most recent 2000 handles regardless
        recent_handles = list(self._ast_handle_cache.keys())[-2000:]
        active.update(recent_handles)

        return active

    def _mark_handle_active(self, handle_id: int):
        """Mark a handle as actively in use to prevent eviction.

        Called when a handle is being used in an active computation.
        """
        if not hasattr(self, '_pending_handles'):
            self._pending_handles = set()
        self._pending_handles.add(handle_id)

    def _unmark_handle_active(self, handle_id: int):
        """Remove active mark from a handle, allowing eviction."""
        if hasattr(self, '_pending_handles') and handle_id in self._pending_handles:
            self._pending_handles.discard(handle_id)

    @staticmethod
    def _evict_oldest(cache: dict, max_size: int) -> list:
        """Find oldest entries to remove from a dict-based cache.

        Returns list of keys to evict so len(cache) - len(result) <= max_size.
        """
        if len(cache) <= max_size:
            return []
        to_remove = []
        for key in list(cache.keys()):
            to_remove.append(key)
            if len(cache) - len(to_remove) <= max_size:
                break
        return to_remove

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
        if not hasattr(self, '_predicate_matched_ids'):
            self._predicate_matched_ids = set()

        # Cache: state_id -> (addr, stdout_len) at last evaluation time
        # If state's current (addr, stdout_len) matches, skip re-evaluation
        if not hasattr(self, '_predicate_eval_cache'):
            self._predicate_eval_cache = {}

        found_sids = set()
        avoid_sids = set()
        _proxy_states = {}  # state_id -> RustStateProxy for uncached states

        # Collect (state_id, addr, stdout_len) from active + deadended in bulk
        # This is much cheaper than per-state FFI calls
        state_info = {}  # state_id -> (addr, stdout_len)
        for stash in ('active', 'deadended'):
            try:
                for sid, addr, stdout_len in self._rust_mgr.get_state_predicate_info(stash):
                    state_info[sid] = (addr, stdout_len)
            except Exception as e:
                l.debug("get_state_predicate_info(stash=%s) failed: %s: %s",
                        stash, type(e).__name__, e)

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
                    l.debug("get_state_stdout(sid=%d) failed: %s: %s",
                            state_id, type(e).__name__, e)
                state = RustStateProxy(
                    self._rust_mgr, state_id,
                    project=self._project,
                    stdin_vars=getattr(self, '_stdin_vars', None),
                    stdout_data=stdout_data,
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
                        l.debug("find_predicate(sid=%d) raised: %s: %s",
                                state_id, type(e).__name__, e)

                if self._avoid_predicate is not None:
                    try:
                        if self._avoid_predicate(state):
                            avoid_sids.add(state_id)
                    except Exception as e:
                        l.debug("avoid_predicate(sid=%d) raised: %s: %s",
                                state_id, type(e).__name__, e)
            except Exception as e:
                l.debug("predicate eval setup for sid=%d failed: %s: %s",
                        state_id, type(e).__name__, e)

            # Mark uncached state as evaluated at its current (addr, stdout_len)
            if not is_cached and current_info is not None:
                self._predicate_eval_cache[state_id] = current_info

        # Move matched states to found/avoid stashes
        for sid in found_sids:
            self._predicate_matched_ids.add(sid)
            for stash in ('active', 'deadended'):
                try:
                    if self._rust_mgr.move_state(sid, stash, 'found'):
                        break
                except Exception as e:
                    l.debug("move_state(sid=%d, %s -> found) failed: %s: %s",
                            sid, stash, type(e).__name__, e)
            # Record in Python-side found for predicate-matched states
            if not hasattr(self, '_predicate_found'):
                self._predicate_found = []
            if sid in self._state_cache:
                self._predicate_found.append(self._state_cache[sid])
            elif sid in _proxy_states:
                self._predicate_found.append(_proxy_states[sid])

        for sid in avoid_sids:
            self._predicate_matched_ids.add(sid)
            for stash in ('active', 'deadended'):
                try:
                    self._rust_mgr.move_state(sid, stash, 'avoid')
                except Exception as e:
                    l.debug("move_state(sid=%d, %s -> avoid) failed: %s: %s",
                            sid, stash, type(e).__name__, e)

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
            l.debug("get_state_root(sid=%d) failed: %s: %s",
                    state_id, type(e).__name__, e)
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
                for stash in ('active', 'deadended', 'found', 'avoid'):
                    ids = self._rust_mgr.get_state_ids(stash)
                    if state_id in ids:
                        idx = ids.index(state_id)
                        pc = self._rust_mgr.get_state_pc(stash, idx)
                        if pc is not None:
                            state.regs._ip = pc
                        break
            except Exception as e:
                l.debug("PC sync for predicate state sid=%d failed: %s: %s",
                        state_id, type(e).__name__, e)
            # Attach Rust solver fallback so solver operations work
            self._attach_rust_solver_fallback(state, state_id)
            return state
        except Exception as e:
            l.debug("Failed to create predicate state for %d: %s", state_id, e)
            return None

    def _cleanup_state_cache(self):
        """Enforce the state cache size limit."""
        to_remove = self._evict_oldest(self._state_cache, self._max_state_cache_size)
        for state_id in to_remove:
            del self._state_cache[state_id]
            try:
                self._rust_mgr.clear_state_metadata(state_id)
            except Exception as e:
                l.debug("clear_state_metadata(sid=%d) failed in cache cleanup: %s: %s",
                        state_id, type(e).__name__, e)
            self._identity_tracker.mark_inactive(state_id)
        if to_remove:
            l.debug(f"Cleaned up {len(to_remove)} state cache entries")

    def _cleanup_state_refs(self, state_id: int):
        """Clean up references for a state that is no longer needed.

        Call this when a state is moved to deadended/errored stash.
        """
        # Remove from state cache
        self._state_cache.pop(state_id, None)
        # Drop per-state metadata held on the Rust side.
        try:
            self._rust_mgr.clear_state_metadata(state_id)
        except Exception as e:
            l.debug("clear_state_metadata(sid=%d) failed in ref cleanup: %s: %s",
                    state_id, type(e).__name__, e)
        # Remove from predicate evaluation cache
        if hasattr(self, '_predicate_eval_cache'):
            self._predicate_eval_cache.pop(state_id, None)
        # Mark symbols as inactive
        self._identity_tracker.mark_inactive(state_id)

