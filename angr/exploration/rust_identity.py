"""Identity tracking utilities for the Rust-Python symbolic execution boundary.

These classes preserve symbolic identity across FFI and track memory writes
during SimProcedure callbacks.
"""
from __future__ import annotations

import logging
import weakref
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    import angr

l = logging.getLogger(name=__name__)


class SymbolicIdentityTracker:
    """Tracks symbolic identity across Python<->Rust boundary.

    This ensures that when a claripy AST (e.g., BVS("x", 32)) is passed
    to Rust and then returned, we get back the same Python object.
    This is critical for constraint consistency - constraints added to
    the original `x` must apply to the exported value.

    The tracker maintains bidirectional mappings:
    - py_to_rust_id: Maps Python AST id() to Rust symbol ID
    - rust_id_to_py: Maps Rust symbol ID to original Python AST

    Uses WeakValueDictionary to allow garbage collection of unreferenced ASTs.
    """

    def __init__(self):
        # Map Python AST id() to Rust symbol ID
        self._py_to_rust_id: Dict[int, int] = {}
        # Map Rust symbol ID to original Python AST (weak refs for GC)
        self._rust_id_to_py: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        # Strong refs for active symbols (prevent premature GC)
        self._active_symbols: Dict[int, object] = {}
        # Map Python hash to AST for hash-based lookup
        self._hash_to_py: Dict[int, object] = {}

    def register(self, py_ast: object, rust_id: int) -> None:
        """Register a Python AST with its Rust symbol ID.

        Args:
            py_ast: The Python claripy AST (BVS, etc.)
            rust_id: The Rust symbol ID assigned to this AST
        """
        py_id = id(py_ast)
        self._py_to_rust_id[py_id] = rust_id
        self._rust_id_to_py[rust_id] = py_ast
        self._active_symbols[rust_id] = py_ast  # Keep strong ref

        # Also store by hash for hash-based lookup
        try:
            py_hash = hash(py_ast)
            self._hash_to_py[py_hash] = py_ast
        except (TypeError, AttributeError):
            # cat-(a) EXPECTED CONTROL FLOW: hash-based lookup is an optional
            # optimization; an unhashable AST falls through to id()-keyed maps.
            pass

    def get_rust_id(self, py_ast: object) -> Optional[int]:
        """Get the Rust symbol ID for a Python AST.

        Returns None if the AST hasn't been registered.
        """
        return self._py_to_rust_id.get(id(py_ast))

    def get_original_ast(self, rust_id: int) -> Optional[object]:
        """Get the original Python AST for a Rust symbol ID.

        This is the critical method for identity preservation on export.
        Returns None if the symbol was created in Rust.
        """
        return self._rust_id_to_py.get(rust_id)

    def get_by_hash(self, py_hash: int) -> Optional[object]:
        """Get a Python AST by its hash value.

        This is used when we have a hash from Rust and need the original AST.
        """
        return self._hash_to_py.get(py_hash)

    def mark_inactive(self, rust_id: int) -> None:
        """Mark a symbol as inactive, allowing it to be GC'd.

        Call this when a state is moved to deadended/errored stash.
        """
        self._active_symbols.pop(rust_id, None)

    def clear(self) -> None:
        """Clear all mappings. Call at start of new exploration."""
        self._py_to_rust_id.clear()
        self._rust_id_to_py.clear()
        self._active_symbols.clear()
        self._hash_to_py.clear()

    def __len__(self) -> int:
        """Return number of registered symbols."""
        return len(self._active_symbols)


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
