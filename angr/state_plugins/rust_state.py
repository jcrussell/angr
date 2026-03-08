"""Rust-native state plugin for Rust-first symbolic execution.

This plugin wraps a RustSimState, providing O(1) forking and minimized
Python-Rust state transfer overhead. The state lives primarily in Rust,
with Python state as an optional overlay for SimProcedures and syscalls.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from .plugin import SimStatePlugin

if TYPE_CHECKING:
    import angr

l = logging.getLogger(name=__name__)

# Try to import the Rust state
try:
    from angr.rustylib.vex_engine import RustSimState as _RustSimState
    RUST_STATE_AVAILABLE = True
except ImportError:
    RUST_STATE_AVAILABLE = False
    _RustSimState = None


class RustStatePlugin(SimStatePlugin):
    """SimStatePlugin wrapping a Rust-native RustSimState.

    This plugin enables Rust-first symbolic execution by:
    - Storing state primarily in Rust (registers, memory, constraints)
    - Providing O(1) forking via copy-on-write
    - Minimizing Python-Rust state transfer

    The plugin acts as a facade, providing Python access to Rust state
    while keeping the hot path (state forking, memory access) in Rust.

    Usage:
        # Create a state with Rust state plugin
        state = proj.factory.entry_state()
        rust_state = RustStatePlugin(arch=proj.arch.name)
        state.register_plugin('rust_state', rust_state)

        # Fork is O(1)
        forked = state.copy()  # Uses Rust CoW forking

        # Access registers/memory through Rust
        rust_state.set_register('rax', 42)
        value = rust_state.get_register('rax')
    """

    def __init__(
        self,
        rust_state: Optional["_RustSimState"] = None,
        arch: str = "amd64",
        **kwargs
    ):
        """Initialize the Rust state plugin.

        Args:
            rust_state: Optional existing RustSimState to wrap. If None,
                       creates a new one for the given architecture.
            arch: Architecture name (e.g., "amd64", "x86", "arm").
            **kwargs: Additional arguments (ignored for compatibility).
        """
        super().__init__()

        if not RUST_STATE_AVAILABLE:
            raise ImportError(
                "RustSimState not available. "
                "Build with vex-engine feature enabled."
            )

        if rust_state is not None:
            self._rust_state = rust_state
        else:
            self._rust_state = _RustSimState(arch)

        self._arch_name = arch

    # =========================================================================
    # Basic Properties
    # =========================================================================

    @property
    def state_id(self) -> int:
        """Get the unique state ID."""
        return self._rust_state.state_id

    @property
    def parent_id(self) -> Optional[int]:
        """Get the parent state ID (None if this is the root state)."""
        return self._rust_state.parent_id

    @property
    def pc(self) -> int:
        """Get the program counter."""
        return self._rust_state.pc

    @pc.setter
    def pc(self, value: int):
        """Set the program counter."""
        self._rust_state.pc = value

    @property
    def arch_name(self) -> str:
        """Get the architecture name."""
        return self._rust_state.arch_name

    @property
    def history(self) -> list:
        """Get the basic block history."""
        return self._rust_state.history()

    # =========================================================================
    # Register Access
    # =========================================================================

    def get_register(self, name: str) -> int:
        """Get a register value by name.

        Args:
            name: Register name (e.g., "rax", "eax", "al").

        Returns:
            Register value as integer.

        Raises:
            ValueError: If the register doesn't exist or has no value.
        """
        return self._rust_state.get_register(name)

    def set_register(self, name: str, value: int):
        """Set a register value by name.

        Args:
            name: Register name.
            value: Value to set.

        Raises:
            ValueError: If the register doesn't exist.
        """
        self._rust_state.set_register(name, value)

    def get_registers_raw(self) -> bytes:
        """Get all register bytes for bulk transfer.

        Returns:
            Raw register file bytes.
        """
        return bytes(self._rust_state.get_registers_raw())

    def set_registers_raw(self, data: bytes):
        """Set all register bytes from bulk transfer.

        Args:
            data: Raw register file bytes.
        """
        self._rust_state.set_registers_raw(data)

    def get_dirty_registers(self) -> list:
        """Get offsets of registers modified since last clear.

        Returns:
            List of 4-byte aligned register offsets.
        """
        return self._rust_state.get_dirty_registers()

    def clear_dirty_registers(self):
        """Clear dirty register tracking."""
        self._rust_state.clear_dirty_registers()

    # =========================================================================
    # Memory Access
    # =========================================================================

    def map_memory(self, addr: int, size: int, permissions: int = 7):
        """Map a memory region.

        Args:
            addr: Base address.
            size: Region size in bytes.
            permissions: Permission bits (R=4, W=2, X=1).
        """
        self._rust_state.map_memory(addr, size, permissions)

    def map_memory_data(self, addr: int, data: bytes, permissions: int = 7):
        """Map memory with initial data.

        Args:
            addr: Base address.
            data: Initial data bytes.
            permissions: Permission bits.
        """
        self._rust_state.map_memory_data(addr, data, permissions)

    def memory_load(self, addr: int, size: int) -> bytes:
        """Load bytes from memory.

        Args:
            addr: Address to load from.
            size: Number of bytes.

        Returns:
            Loaded bytes.

        Raises:
            ValueError: If address is unmapped.
        """
        return bytes(self._rust_state.memory_load(addr, size))

    def memory_store(self, addr: int, data: bytes):
        """Store bytes to memory.

        Args:
            addr: Address to store to.
            data: Bytes to store.

        Raises:
            ValueError: If address is unmapped.
        """
        self._rust_state.memory_store(addr, data)

    def add_lazy_region(self, start_addr: int, size: int):
        """Add a lazy region for on-demand page fetching.

        Args:
            start_addr: Start address of region.
            size: Size in bytes.
        """
        self._rust_state.add_lazy_region(start_addr, size)

    def get_dirty_pages(self) -> list:
        """Get page-aligned addresses of modified pages.

        Returns:
            List of dirty page addresses.
        """
        return self._rust_state.get_dirty_pages()

    def clear_dirty_pages(self):
        """Clear dirty page tracking."""
        self._rust_state.clear_dirty_pages()

    # =========================================================================
    # Hooks
    # =========================================================================

    def add_hook(self, addr: int):
        """Add a hook at the given address."""
        self._rust_state.add_hook(addr)

    def remove_hook(self, addr: int):
        """Remove a hook at the given address."""
        self._rust_state.remove_hook(addr)

    def is_hooked(self, addr: int) -> bool:
        """Check if an address is hooked."""
        return self._rust_state.is_hooked(addr)

    def clear_hooks(self):
        """Clear all hooks."""
        self._rust_state.clear_hooks()

    # =========================================================================
    # Solver
    # =========================================================================

    def satisfiable(self) -> bool:
        """Check if current constraints are satisfiable."""
        return self._rust_state.satisfiable()

    # =========================================================================
    # Configuration
    # =========================================================================

    def configure_concretization(self, use_approximate: bool, range_limit: Optional[int] = None):
        """Configure address concretization strategy.

        Args:
            use_approximate: Whether to use approximate memory indices.
            range_limit: Maximum address range (default: 1024).
        """
        self._rust_state.configure_concretization(use_approximate, range_limit)

    def set_track_history(self, track: bool):
        """Set whether to track basic block history."""
        self._rust_state.set_track_history(track)

    def set_max_history(self, max_len: int):
        """Set maximum history length."""
        self._rust_state.set_max_history(max_len)

    # =========================================================================
    # State Sync (Python <-> Rust)
    # =========================================================================

    def sync_from_angr_state(self, angr_state: "angr.SimState"):
        """Sync register and memory state from an angr SimState.

        This is used to initialize the Rust state from Python state,
        or to update after a SimProcedure runs.

        Args:
            angr_state: The angr SimState to sync from.
        """
        # Sync PC
        if hasattr(angr_state, 'addr'):
            self._rust_state.pc = angr_state.addr

        # Sync registers (if concrete)
        if hasattr(angr_state, 'regs'):
            # Get register plugin
            regs = angr_state.regs
            arch = angr_state.arch

            # Sync common registers
            for reg_name in ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi',
                           'rbp', 'rsp', 'r8', 'r9', 'r10', 'r11',
                           'r12', 'r13', 'r14', 'r15', 'rip']:
                try:
                    reg_val = getattr(regs, reg_name)
                    if not reg_val.symbolic:
                        self.set_register(reg_name, angr_state.solver.eval(reg_val))
                except (AttributeError, KeyError):
                    pass

    def sync_to_angr_state(self, angr_state: "angr.SimState"):
        """Sync register and memory state to an angr SimState.

        This is used after Rust execution to update Python state.

        Args:
            angr_state: The angr SimState to sync to.
        """
        import claripy

        # Sync dirty registers
        dirty_offsets = self.get_dirty_registers()
        for offset in dirty_offsets:
            # Get the value from Rust
            # Note: We need to map offset to register name
            # For now, handle common registers
            offset_to_name = {
                16: 'rax', 24: 'rcx', 32: 'rdx', 40: 'rbx',
                48: 'rsp', 56: 'rbp', 64: 'rsi', 72: 'rdi',
                80: 'r8', 88: 'r9', 96: 'r10', 104: 'r11',
                112: 'r12', 120: 'r13', 128: 'r14', 136: 'r15',
                184: 'rip',
            }
            if offset in offset_to_name:
                reg_name = offset_to_name[offset]
                try:
                    value = self.get_register(reg_name)
                    setattr(angr_state.regs, reg_name, claripy.BVV(value, 64))
                except Exception:
                    pass

        # Sync dirty pages
        for page_addr in self.get_dirty_pages():
            try:
                page_data = self.memory_load(page_addr, 4096)
                # Write to angr's memory
                for i, byte in enumerate(page_data):
                    angr_state.memory.store(page_addr + i, claripy.BVV(byte, 8))
            except Exception:
                pass

    # =========================================================================
    # Plugin Protocol
    # =========================================================================

    @SimStatePlugin.memo
    def copy(self, memo):
        """Copy the plugin for state forking.

        This uses Rust's O(1) CoW fork for the RustSimState.

        Args:
            memo: Memoization dictionary.

        Returns:
            Copy of the plugin with forked Rust state.
        """
        c = RustStatePlugin.__new__(RustStatePlugin)
        c.state = None
        c._rust_state = self._rust_state.fork()
        c._arch_name = self._arch_name
        return c

    def merge(self, others, merge_conditions, common_ancestor=None):
        """Merge solver states.

        Note: Full merge is complex for Rust state. For now, returns
        False to indicate caller should handle merging.

        Returns:
            False (merge not fully supported).
        """
        return False

    def widen(self, others):
        """Widen state."""
        return self.merge(others, None)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def fork(self) -> "RustStatePlugin":
        """Fork this state plugin (O(1) CoW).

        Returns:
            New RustStatePlugin with forked Rust state.
        """
        return self.copy(memo={})

    @property
    def rust_state(self) -> "_RustSimState":
        """Get the underlying RustSimState."""
        return self._rust_state
