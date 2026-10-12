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
        evaluate them. Every state in the Rust active and deadended stashes is
        evaluated on a fresh ``RustStateProxy`` that reads registers, memory
        and posix output live from Rust.

        angr-hc1yj: this used to hand predicates the cached Python frame from
        ``_state_cache`` when one existed — a frame frozen at its last
        callback (wrong pc, no constraints) and shared between states, into
        which the live stdout was appended in place. Predicates then saw
        another state's output, and the appended bytes leaked into every
        later copy of that frame. Cached frames are never touched here now.

        Uses change-detection caching: a state is only re-evaluated when its
        address or stdout length has changed since the last evaluation.

        States already moved to found/avoid stashes are excluded by the stash
        query (only active + deadended are checked).
        """
        from angr.exploration.rust_state_proxy import RustStateProxy

        if not hasattr(self, "_predicate_matched_ids"):
            self._predicate_matched_ids = set()

        # Cache: state_id -> (addr, stdout_len) at last evaluation time
        # If state's current (addr, stdout_len) matches, skip re-evaluation
        if not hasattr(self, "_predicate_eval_cache"):
            self._predicate_eval_cache = {}

        found_sids = set()
        avoid_sids = set()

        # Collect (state_id, addr, stdout_len) from active + deadended in bulk
        # This is much cheaper than per-state FFI calls
        state_info = {}  # state_id -> (addr, stdout_len)
        for stash in ("active", "deadended"):
            try:
                for sid, addr, stdout_len in self._rust_mgr.get_state_predicate_info(stash):
                    state_info[sid] = (addr, stdout_len)
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: this stash's states go
                # unevaluated this sweep; the next sweep retries.
                l.debug("get_state_predicate_info(stash=%s) failed: %s: %s", stash, type(e).__name__, e)

        for state_id, current_info in state_info.items():
            if state_id in self._predicate_matched_ids:
                continue
            if self._predicate_eval_cache.get(state_id) == current_info:
                continue

            # stdout_data=None: the proxy reads this state's own Rust stdout.
            state = RustStateProxy(
                self._rust_mgr,
                state_id,
                project=self._project,
                stdin_vars=getattr(self, "_stdin_vars", None),
                python_mgr=self,
            )

            if self._find_predicate is not None:
                try:
                    if self._find_predicate(state):
                        found_sids.add(state_id)
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

            self._predicate_eval_cache[state_id] = current_info

        # Move matched states to found/avoid stashes. Rust's found stash is the
        # only record of a find (``found`` reads it), so nothing is kept here.
        for target, sids in (("found", found_sids), ("avoid", avoid_sids)):
            for sid in sids:
                self._predicate_matched_ids.add(sid)
                for stash in ("active", "deadended"):
                    try:
                        if self._rust_mgr.move_state(sid, stash, target):
                            break
                    except Exception as e:
                        # cat-(b) FALLBACK WITH LOSS: state classified but the
                        # move failed; it stays where it was and is not
                        # re-evaluated (it is in _predicate_matched_ids). Log so
                        # this is visible — could mask a real find failure.
                        l.debug("move_state(sid=%d, %s -> %s) failed: %s: %s", sid, stash, target, type(e).__name__, e)
