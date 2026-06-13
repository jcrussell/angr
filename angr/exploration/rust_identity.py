"""Identity tracking utilities for the Rust-Python symbolic execution boundary.

Tracks memory writes during SimProcedure callbacks.

Note: symbolic-identity preservation across the FFI boundary is handled
entirely on the Rust side. ASTs that must round-trip are pinned via the
manager's ``set_state_addr_to_ast`` (Rust holds its own strong ``PyObject``
ref), so a Python-side identity map is unnecessary. The former
``SymbolicIdentityTracker`` was write-only in production (its only reader,
``_lookup_handle``, had no production callers) and leaked strong refs without
bound — it was removed in angr-iu40.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import angr

l = logging.getLogger(name=__name__)


class CallbackMemoryTracker:
    """Tracks memory writes during SimProcedure callback execution.

    Automatically tracks all state.memory.store() calls during callback
    execution, ensuring memory changes are properly synced back to Rust
    without relying on changed_bytes() comparison.

    Usage:
        with CallbackMemoryTracker(state) as tracker:
            # Execute SimProcedure
            proc.execute(state, successors)
        memory_changes = tracker.get_writes()
        symbolic_writes = tracker.get_symbolic_writes()
    """

    def __init__(self, state: "angr.SimState"):
        """Initialize the tracker for a given state.

        Args:
            state: The angr state to track memory writes on.
        """
        self._state = state
        self._writes: list = []  # List of (addr, data_bytes) tuples
        self._symbolic_writes: list = []  # List of (addr, ast) for symbolic imports
        self._original_store = None
        self._tracking = False

    def __enter__(self):
        """Start tracking memory writes."""
        if hasattr(self._state, 'memory') and hasattr(self._state.memory, 'store'):
            self._original_store = self._state.memory.store
            self._tracking = True

            # Create wrapper that tracks writes
            tracker = self

            def tracking_store(addr, data, size=None, condition=None, **kwargs):
                """Wrapper for memory.store that tracks writes."""
                # Call original store first
                result = tracker._original_store(addr, data, size=size, condition=condition, **kwargs)

                # Track the write
                try:
                    # Get concrete address
                    if hasattr(addr, 'symbolic') and addr.symbolic:
                        # For symbolic addresses, we can't easily track
                        pass
                    else:
                        if isinstance(addr, int):
                            concrete_addr = addr
                        elif hasattr(addr, 'args') and isinstance(addr.args[0], int):
                            concrete_addr = addr.args[0]
                        else:
                            concrete_addr = tracker._state.solver.eval(addr)

                        # Get concrete data
                        if isinstance(data, (bytes, bytearray)):
                            # Raw bytes — track directly
                            data_bytes = bytes(data)
                            if size is None:
                                size = len(data_bytes)
                        elif hasattr(data, 'symbolic') and data.symbolic:
                            # For symbolic data, get a concrete witness
                            concrete_data = tracker._state.solver.eval(data)
                            if size is None:
                                size = data.size() // 8 if hasattr(data, 'size') else 8
                            data_bytes = concrete_data.to_bytes(size, 'little')
                            # Also track the symbolic AST for Rust import
                            tracker._symbolic_writes.append((concrete_addr, data))
                        elif isinstance(data, int):
                            if size is None:
                                size = 8
                            data_bytes = data.to_bytes(size, 'little')
                        else:
                            # claripy concrete value
                            concrete_val = tracker._state.solver.eval(data)
                            if size is None:
                                size = data.size() // 8 if hasattr(data, 'size') else 8
                            data_bytes = concrete_val.to_bytes(size, 'little')

                        tracker._writes.append((concrete_addr, bytes(data_bytes)))
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: tracking failure means this
                    # write won't be replayed back to Rust; the original store
                    # already succeeded so the live state is correct.
                    l.debug(f"Memory tracking error at addr={addr}: {e}")

                return result

            self._state.memory.store = tracking_store

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking memory writes and restore original store."""
        if self._tracking and self._original_store is not None:
            self._state.memory.store = self._original_store
            self._tracking = False
        return False  # Don't suppress exceptions

    def get_writes(self) -> list:
        """Get all tracked memory writes.

        Returns:
            List of (addr, data_bytes) tuples.
        """
        return self._writes

    def get_symbolic_writes(self) -> list:
        """Get all tracked symbolic memory writes.

        Returns:
            List of (addr, ast) tuples for symbolic values that need
            to be imported to Rust.
        """
        return self._symbolic_writes

    def clear(self):
        """Clear tracked writes."""
        self._writes.clear()
        self._symbolic_writes.clear()
