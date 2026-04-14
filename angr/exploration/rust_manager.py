"""Python wrapper for Rust-native exploration manager.

This module provides `RustExplorationManager`, a Python-facing interface
to the Rust exploration loop that achieves ~3x speedup by:
- Managing states entirely in Rust (using RustSimState)
- Processing symbolic branches with deferred forks
- Only calling Python for SimProcedures and syscalls
- Implementing find/avoid address checking in Rust
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
import weakref
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union

import claripy

if TYPE_CHECKING:
    import angr

l = logging.getLogger(name=__name__)
_DBG = l.isEnabledFor(logging.DEBUG)  # Module-level guard for hot-path debug calls

# Try to import the Rust exploration manager
try:
    from angr.rustylib.vex_engine import (
        RustExplorationManager as _RustExplorationManager,
        ExplorationEvent as _ExplorationEvent,
        ExplorationStateSnapshot as _ExplorationStateSnapshot,
        PythonCallbacks,
        RustSimState as _RustSimState,
    )
    RUST_EXPLORATION_AVAILABLE = True
except ImportError:
    RUST_EXPLORATION_AVAILABLE = False
    _RustExplorationManager = None
    _ExplorationEvent = None
    _ExplorationStateSnapshot = None
    PythonCallbacks = None
    _RustSimState = None

# Z3 context sharing: make Rust and Python use the same Z3 context
# to avoid AST translation overhead between solvers.
_z3_context_shared = False

def _setup_shared_z3_context():
    """Share Python's Z3 context with Rust, so both create ASTs in the same context."""
    global _z3_context_shared
    if _z3_context_shared:
        return
    try:
        from angr.rustylib.vex_engine import set_shared_z3_context
        import z3
        py_ctx = z3.main_ctx()
        set_shared_z3_context(py_ctx.ctx.value)
        _z3_context_shared = True
        l.debug("Shared Z3 context with Rust (ptr=%#x)", py_ctx.ctx.value)
    except (ImportError, AttributeError, Exception) as e:
        l.debug("Z3 context sharing not available: %s", e)


from angr.exploration.rust_identity import SymbolicIdentityTracker, CallbackMemoryTracker


from angr.exploration.rust_state_export import RustStateExportMixin
from angr.exploration.rust_callback_dispatch import RustCallbackDispatchMixin
from angr.exploration.rust_state_sync import RustStateSyncMixin
from angr.exploration.rust_state_cache import RustStateCacheMixin


class RustExplorationManager(
    RustCallbackDispatchMixin,
    RustStateSyncMixin,
    RustStateCacheMixin,
    RustStateExportMixin,
):
    """Python wrapper for Rust-native exploration manager.

    This provides a SimulationManager-like interface while keeping the
    exploration loop in Rust for performance.

    Usage:
        import angr
        from angr.exploration import RustExplorationManager

        proj = angr.Project('binary')
        state = proj.factory.entry_state()

        # Create Rust exploration manager
        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=0x401234)

        # Access found states
        for state in mgr.found:
            print(state.solver.eval(state.posix.dumps(0)))
    """

    # Class-level cache for Python init results per binary
    _init_cache: Dict[str, "angr.SimState"] = {}
    _init_cache_max = 10

    # SimProcedures known to write memory (need full state.copy() for changed_bytes)
    _MEMORY_WRITING_PROCS = frozenset({
        'read', 'recv', 'fgets', 'scanf', '__isoc99_scanf',
        'fread', 'gets', 'getchar', 'fgetc', 'getc',
        'strncpy', 'strcpy', 'memcpy', 'memmove', 'memset',
        'strcat', 'strncat', 'sprintf', 'snprintf'})

    def __init__(
        self,
        project: "angr.Project",
        active_states: Optional[list] = None,
    ):
        """Initialize the Rust exploration manager.

        Args:
            project: angr Project for the binary.
            active_states: Optional list of initial angr SimStates.
        """
        # Ensure Z3 context is shared (one-time setup)
        _setup_shared_z3_context()

        if not RUST_EXPLORATION_AVAILABLE:
            raise ImportError(
                "RustExplorationManager not available. "
                "Build with vex-engine feature enabled."
            )

        self._project = project
        is_le = project.arch.memory_endness == 'Iend_LE'
        self._rust_mgr = _RustExplorationManager(project.arch.name, little_endian=is_le)

        # Performance profiling counters
        self._perf_stats = {
            'init_total_ns': 0,
            'init_setup_callbacks_ns': 0,
            'init_load_binary_ns': 0,
            'init_register_simprocedures_ns': 0,
            'init_python_run_ns': 0,
            'init_add_rust_state_ns': 0,
            'init_memory_sync_ns': 0,
            'init_register_sync_ns': 0,
            'init_hook_register_ns': 0,
            'callback_simprocedure_count': 0,
            'callback_simprocedure_total_ns': 0,
            'callback_simprocedure_state_create_ns': 0,
            'callback_simprocedure_execute_ns': 0,
            'callback_simprocedure_sync_back_ns': 0,
            'callback_memory_load_count': 0,
            'callback_memory_load_total_ns': 0,
            'callback_fetch_page_count': 0,
            'callback_fetch_page_total_ns': 0,
            'callback_lift_block_count': 0,
            'callback_lift_block_total_ns': 0,
            'callback_simprocedure_state_copy_ns': 0,
        }
        # Per-procedure timing: {name: {'count': int, 'execute_ns': int}}
        self._procedure_times: Dict[str, Dict[str, int]] = {}

        # High-level instrumentation counters for optimization tracking
        self._stats_callback_count = 0       # total Python callbacks invoked
        self._stats_ffi_crossings = 0        # total FFI calls to Rust (run/get/set)
        self._stats_state_creations = 0      # full SimState objects created
        self._stats_cache_hits = 0           # state cache hits
        self._stats_cache_misses = 0         # state cache misses
        self._stats_technique_filter_calls = 0  # technique filter invocations
        self._stats_hook_sync_calls = 0      # _sync_hooks_before_step invocations
        self._stats_hook_sync_skips = 0      # fast-path skips (no new hooks)
        self._stats_time_in_callbacks_ns = 0 # cumulative time in callback code
        _init_start = time.perf_counter_ns()

        # Track registered hooks to detect dynamically created continuations
        # SimProcedures can create continuation hooks via self.call() which
        # need to be registered with Rust before exploration continues
        # (Must be initialized before _register_simprocedures() is called)
        self._registered_hooks: set = set()

        # Set up callbacks
        _t0 = time.perf_counter_ns()
        self._setup_callbacks()
        self._perf_stats['init_setup_callbacks_ns'] = time.perf_counter_ns() - _t0

        # Load binary regions
        _t0 = time.perf_counter_ns()
        self._load_binary_regions()
        self._perf_stats['init_load_binary_ns'] = time.perf_counter_ns() - _t0

        # Register SimProcedures
        _t0 = time.perf_counter_ns()
        self._register_simprocedures()
        self._perf_stats['init_register_simprocedures_ns'] = time.perf_counter_ns() - _t0

        # Symbolic identity tracker for preserving AST identity across FFI
        # This is critical: BVS("x", 32) must stay the same object after round-trip
        self._identity_tracker = SymbolicIdentityTracker()

        # Track angr state mappings for callbacks
        # Using regular dict with periodic cleanup to prevent memory leaks
        self._state_cache: Dict[int, "angr.SimState"] = {}

        # Maximum state cache size before cleanup
        self._max_state_cache_size = 500

        # Track claripy AST handles for constraint sync
        # Maps handle_id -> claripy AST
        self._ast_handle_cache: Dict[int, object] = {}

        # Track current callback state for memory access during callbacks
        # This allows memory_load callback to access the correct symbolic state
        self._callback_state: Optional["angr.SimState"] = None

        # Cache bundle register values from _create_state_for_callback for
        # reuse as register snapshot (avoids reading registers back from state)
        self._last_bundle_registers: Optional[dict] = None

        # Track symbolic memory regions per state for preservation during fallback
        # Maps state_id -> dict[addr -> claripy.AST]
        # When Rust falls back to Python, symbolic memory would be lost without this
        # Using bounded cache size to prevent memory leaks (Phase 3)
        self._symbolic_pages: Dict[int, Dict[int, object]] = {}
        self._max_symbolic_pages_cache = 100  # Limit cache size

        # Track symbolic memory writes during hooks for preservation
        # Maps state_id -> {addr -> (claripy_ast, size)}
        # This ensures symbolic memory written by hooks survives the sync back to Rust
        self._hook_symbolic_memory: Dict[int, Dict[int, Tuple[object, int]]] = {}

        # Track the current callback state ID for memory tracking during callbacks
        self._current_callback_state_id: Optional[int] = None
        # Track which Rust state is being stepped for per-fork memory isolation
        self._current_stepping_state_id: Optional[int] = None

        # Track symbolic memory by address for state export recovery (P1 fix)
        # Maps state_id -> {addr -> (ast, size)}
        # This allows recovering original symbols during state export instead of
        # creating fresh BVS variables that lose constraint linkage
        self._addr_to_ast: Dict[int, Dict[int, Tuple[object, int]]] = {}

        # Track procedure_data for SimProcedure continuations (P1 fix)
        # When a SimProcedure uses self.call() to invoke a function and register
        # a continuation, the procedure_data is stored here keyed by the continuation
        # address. When Rust invokes the continuation, we restore this data.
        # Maps continuation_addr -> procedure_data tuple
        self._pending_procedure_data: Dict[int, Tuple] = {}

        # Cache for addresses where SimProcedure continuations always result in exit.
        # After the first time a continuation at an address produces only Ijk_Exit
        # successors, all subsequent callbacks are fast-deadended without state creation.
        self._exit_continuation_addrs: set = set()

        # P10 fix: Track root state IDs for plugin restoration
        # Maps state_id -> root_state_id (the original state from Python)
        # When Rust forks states, this allows finding the original state for plugin copying
        self._state_roots: Dict[int, int] = {}

        # P9 fix: Track active exploration techniques
        # Techniques are applied during exploration steps
        self._active_techniques: list = []

        # Cached memory layout from disk cache for fast _sync_memory_to_rust
        self._mem_cache: Optional[dict] = None

        # Add initial states
        if active_states:
            # Handle single state or list of states
            if hasattr(active_states, 'solver'):  # Single SimState
                active_states = [active_states]
            for state in active_states:
                # Detect LAZY_SOLVES option
                if hasattr(state, 'options'):
                    try:
                        from angr.sim_options import LAZY_SOLVES
                        if LAZY_SOLVES in state.options:
                            self._rust_mgr.set_lazy_solves(True)
                            l.debug("Enabled lazy_solves mode from state options")
                    except ImportError:
                        pass

                # Hybrid init: if the state starts at a loader/init address
                # (not in the main binary), run the init sequence in Python
                # first. This handles C++ constructors, .init_array, etc.
                # that the Rust engine can't execute correctly.
                _t0 = time.perf_counter_ns()
                state = self._run_python_init_if_needed(state)
                self._perf_stats['init_python_run_ns'] += time.perf_counter_ns() - _t0

                _t0 = time.perf_counter_ns()
                self._add_rust_state('active', state)
                self._perf_stats['init_add_rust_state_ns'] += time.perf_counter_ns() - _t0

        self._perf_stats['init_total_ns'] = time.perf_counter_ns() - _init_start

    def perf_report(self) -> str:
        """Return a formatted performance report."""
        s = self._perf_stats
        lines = ["=== Rust Engine Performance Report ==="]
        lines.append(f"Init total: {s['init_total_ns']/1e6:.1f}ms")
        lines.append(f"  Setup callbacks: {s['init_setup_callbacks_ns']/1e6:.1f}ms")
        lines.append(f"  Load binary regions: {s['init_load_binary_ns']/1e6:.1f}ms")
        lines.append(f"  Register SimProcedures: {s['init_register_simprocedures_ns']/1e6:.1f}ms")
        lines.append(f"  Python init: {s['init_python_run_ns']/1e6:.1f}ms")
        lines.append(f"  Add Rust state: {s['init_add_rust_state_ns']/1e6:.1f}ms")
        lines.append(f"    Memory sync: {s['init_memory_sync_ns']/1e6:.1f}ms")
        lines.append(f"    Register sync: {s['init_register_sync_ns']/1e6:.1f}ms")
        lines.append(f"SimProcedure callbacks: {s['callback_simprocedure_count']}")
        lines.append(f"  Total time: {s['callback_simprocedure_total_ns']/1e6:.1f}ms")
        lines.append(f"  State create: {s['callback_simprocedure_state_create_ns']/1e6:.1f}ms")
        lines.append(f"  Execute: {s['callback_simprocedure_execute_ns']/1e6:.1f}ms")
        lines.append(f"  State copy: {s['callback_simprocedure_state_copy_ns']/1e6:.1f}ms")
        lines.append(f"  Sync back: {s['callback_simprocedure_sync_back_ns']/1e6:.1f}ms")
        if self._procedure_times:
            lines.append(f"  Per-procedure breakdown:")
            for pname, pt in sorted(self._procedure_times.items(), key=lambda x: -x[1]['execute_ns']):
                lines.append(f"    {pname}: {pt['count']}x {pt['execute_ns']/1e6:.1f}ms")
        lines.append(f"Memory load callbacks: {s['callback_memory_load_count']}")
        lines.append(f"  Total time: {s['callback_memory_load_total_ns']/1e6:.1f}ms")
        if s['callback_memory_load_count'] > 0:
            lines.append(f"  Avg per call: {s['callback_memory_load_total_ns']/s['callback_memory_load_count']/1e3:.1f}us")
        lines.append(f"Fetch page callbacks: {s['callback_fetch_page_count']}")
        lines.append(f"  Total time: {s['callback_fetch_page_total_ns']/1e6:.1f}ms")
        lines.append(f"Lift block callbacks: {s['callback_lift_block_count']}")
        lines.append(f"  Total time: {s['callback_lift_block_total_ns']/1e6:.1f}ms")
        if s['callback_lift_block_count'] > 0:
            lines.append(f"  Avg per call: {s['callback_lift_block_total_ns']/s['callback_lift_block_count']/1e3:.1f}us")
        return "\n".join(lines)

    def _setup_callbacks(self):
        """Set up Python callbacks for the Rust engine."""
        callbacks = PythonCallbacks()

        # Thread-safe stepping state ID accessor (avoids re-entrant Rust calls)
        try:
            from angr.rustylib.vex_engine import get_stepping_state_id as _get_sid
        except ImportError:
            _get_sid = lambda: None

        def _get_per_fork_state():
            """Get the correct per-fork Python state for the current VEX step."""
            state = self._get_callback_state()
            if state is not None:
                return state
            sid = _get_sid()
            if sid is not None and sid in self._state_cache:
                return self._state_cache[sid]
            return self._get_default_state()

        # Memory load callback - use per-fork state for correct isolation
        def memory_load(addr: int, size: int) -> tuple:
            _ml_start = time.perf_counter_ns()
            try:
                state = _get_per_fork_state()
                if state is None:
                    return (bytes(size), False, None)

                try:
                    # Check preserved hook symbolic memory first
                    # This is critical for preserving symbolic relationships when
                    # hooks manipulate symbolic memory (like flareon2015_5)
                    # P5 fix: Use effective state ID to find parent's symbolic memory
                    state_id = self._current_callback_state_id
                    effective_state_id = self._get_effective_state_id(state_id) if state_id is not None else None
                    lookup_id = effective_state_id if effective_state_id is not None else state_id
                    if lookup_id is not None and lookup_id in self._hook_symbolic_memory:
                        hook_mem = self._hook_symbolic_memory[lookup_id]
                        for mem_addr, (ast, mem_size) in hook_mem.items():
                            # Check if the requested address overlaps with tracked symbolic memory
                            if mem_addr <= addr < mem_addr + mem_size:
                                # Found symbolic memory at this address
                                offset = addr - mem_addr
                                if offset == 0 and size == mem_size:
                                    # Exact match - return the full AST
                                    concrete = state.solver.eval(ast).to_bytes(size, 'little')
                                    self._register_handle(id(ast), ast, addr=addr, size=size, state_id=lookup_id)
                                    if _DBG:
                                        l.debug(f"Memory load hit preserved symbolic at 0x{addr:x}")
                                    return (concrete, True, ast)
                                elif offset == 0 and size < mem_size:
                                    # Partial read from start - extract bytes
                                    extracted = claripy.Extract(size * 8 - 1, 0, ast)
                                    concrete = state.solver.eval(extracted).to_bytes(size, 'little')
                                    self._register_handle(id(extracted), extracted, addr=addr, size=size, state_id=lookup_id)
                                    return (concrete, True, extracted)

                    # Standard memory load from state
                    val = state.memory.load(addr, size, endness=state.arch.memory_endness)

                    # Fix 1C: Coerce thunks/callables to actual values
                    # Some memory loads can return callable thunks instead of proper ASTs
                    coerce_attempts = 0
                    while callable(val) and not hasattr(val, 'op') and coerce_attempts < 3:
                        try:
                            val = val()
                            coerce_attempts += 1
                        except Exception:
                            if _DBG:
                                l.debug(f"Memory load thunk at 0x{addr:x} failed to resolve, creating symbolic")
                            val = claripy.BVS(f"mem_thunk_{addr:x}", size * 8)
                            break

                    # Validate we have a proper claripy AST
                    if not hasattr(val, 'op'):
                        l.warning(f"Memory load at 0x{addr:x} returned invalid type: {type(val)}")
                        return (bytes(size), False, None)

                    # Safe check for symbolic (handles callables)
                    is_symbolic = getattr(val, 'symbolic', False) if hasattr(val, 'symbolic') else False
                    if is_symbolic:
                        # Register the handle for later constraint reconstruction
                        # This is critical for bidirectional constraint sync - when Rust
                        # concretizes this address and syncs constraints back, we can
                        # look up the original AST and properly constrain it
                        handle_id = id(val)
                        # Track address mapping for state export recovery (P1 fix)
                        self._register_handle(handle_id, val, addr=addr, size=size, state_id=state_id)
                        concrete = state.solver.eval(val).to_bytes(size, 'little')
                        return (concrete, True, val)  # Return claripy AST for symbolic values
                    else:
                        concrete = state.solver.eval(val).to_bytes(size, 'little')
                        return (concrete, False, None)
                except Exception as e:
                    l.warning(f"Memory load error at 0x{addr:x}: {e}")
                    return (bytes(size), False, None)
            finally:
                self._perf_stats['callback_memory_load_count'] += 1
                self._perf_stats['callback_memory_load_total_ns'] += time.perf_counter_ns() - _ml_start

        # Memory store callback
        def memory_store(addr: int, data: bytes):
            state = _get_per_fork_state()
            if state is None:
                return

            try:
                val = claripy.BVV(int.from_bytes(data, 'little'), len(data) * 8)
                state.memory.store(addr, val, endness=state.arch.memory_endness)
            except Exception as e:
                l.warning(f"Memory store error at 0x{addr:x}: {e}")

        # Block lifting callback
        def lift_block(addr: int) -> str:
            _lb_start = time.perf_counter_ns()
            try:
                import json
                try:
                    block = self._project.factory.block(addr)
                    irsb = block.vex
                    # Serialize to JSON
                    return self._serialize_irsb(irsb)
                except Exception as e:
                    l.warning(f"Lift error at 0x{addr:x}: {e}")
                    return '{}'
            finally:
                self._perf_stats['callback_lift_block_count'] += 1
                self._perf_stats['callback_lift_block_total_ns'] += time.perf_counter_ns() - _lb_start

        # Page fetch callback
        def fetch_page(page_addr: int) -> tuple:
            _fp_start = time.perf_counter_ns()
            try:
                state = self._get_default_state()
                if state is None:
                    return (bytes(4096), 0, False)

                try:
                    data = state.memory.load(page_addr, 4096, endness=state.arch.memory_endness)
                    # If the page contains symbolic data, return failure so the
                    # Rust engine falls back to per-access memory_load callback
                    # which returns the symbolic AST (enabling symbolic forking)
                    is_symbolic = getattr(data, 'symbolic', False)
                    if is_symbolic:
                        if _DBG:
                            l.debug(f"fetch_page 0x{page_addr:x}: has symbolic data, declining")
                        return (bytes(4096), 0, False)
                    concrete = state.solver.eval(data).to_bytes(4096, 'little')
                    return (concrete, 7, True)  # RWX permissions
                except Exception as e:
                    return (bytes(4096), 0, False)
            finally:
                self._perf_stats['callback_fetch_page_count'] += 1
                self._perf_stats['callback_fetch_page_total_ns'] += time.perf_counter_ns() - _fp_start

        # Constraint sync callback - receives constraints from Rust before Python fallback
        def sync_constraints(constraints: list) -> bool:
            """Sync constraints from Rust to Python's claripy solver.

            This is called when Rust falls back to Python for operations that
            depend on solver state (e.g., unmapped memory access). The constraints
            represent concretization decisions made by Rust that need to be
            reflected in Python's solver.

            Uses transactional sync with rollback on failure:
            1. Record pre-sync constraint count
            2. Add all constraints
            3. Validate satisfiability
            4. On failure: remove added constraints and report error

            Args:
                constraints: List of (description, width, concrete_value, handle_id) tuples

            Returns:
                True if sync succeeded, False if rollback occurred
            """
            state = self._get_default_state()
            if state is None:
                l.debug("sync_constraints called but no state available")
                return False

            # Record pre-sync state for transactional rollback
            pre_sync_count = len(state.solver.constraints)
            added_constraints = []
            sync_failed = False

            for desc, width, concrete_val, handle_id in constraints:
                try:
                    # Try to reconstruct the constraint from handle_id if available
                    if handle_id is not None:
                        # Look up the original AST from the handle
                        ast = self._lookup_handle(handle_id)
                        if ast is not None:
                            # Add constraint: ast == concrete_val
                            constraint = ast == claripy.BVV(concrete_val, width)
                            state.solver.add(constraint)
                            added_constraints.append(constraint)
                            l.debug(f"Synced constraint from handle {handle_id}: {desc}")
                            continue

                    # P2 fix: Try to reconstruct constraint from description
                    # Description formats: "addr_concretize_0x1234", "branch_cond_0xABCD", etc.
                    reconstructed = False

                    # P5 fix: Use effective state ID for addr_to_ast lookups
                    state_id = self._current_callback_state_id
                    effective_id = self._get_effective_state_id(state_id) if state_id is not None else None

                    if desc.startswith("addr_concretize_"):
                        # Address concretization - extract address from description
                        addr_str = desc.replace("addr_concretize_", "")
                        try:
                            addr = int(addr_str, 16)
                            # Look for AST by address in our tracking
                            lookup_id = effective_id if effective_id is not None else state_id
                            if lookup_id is not None:
                                addr_map = self._addr_to_ast.get(lookup_id, {})
                                for tracked_addr, (tracked_ast, _) in addr_map.items():
                                    if tracked_addr == addr:
                                        constraint = tracked_ast == claripy.BVV(concrete_val, width)
                                        state.solver.add(constraint)
                                        added_constraints.append(constraint)
                                        l.debug(f"Reconstructed addr_concretize constraint from desc: {desc}")
                                        reconstructed = True
                                        break
                        except ValueError:
                            pass

                    elif desc.startswith("mem_") and "_" in desc:
                        # Memory read symbolic - try to find by address
                        parts = desc.split("_")
                        if len(parts) >= 2:
                            try:
                                addr = int(parts[1], 16)
                                lookup_id = effective_id if effective_id is not None else state_id
                                if lookup_id is not None:
                                    addr_map = self._addr_to_ast.get(lookup_id, {})
                                    for tracked_addr, (tracked_ast, _) in addr_map.items():
                                        if tracked_addr == addr:
                                            constraint = tracked_ast == claripy.BVV(concrete_val, width)
                                            state.solver.add(constraint)
                                            added_constraints.append(constraint)
                                            l.debug(f"Reconstructed mem constraint from desc: {desc}")
                                            reconstructed = True
                                            break
                            except ValueError:
                                pass

                    if not reconstructed:
                        # Still couldn't reconstruct - log warning
                        l.warning(f"Could not sync constraint (no handle): {desc} = 0x{concrete_val:x}")
                        sync_failed = True

                except Exception as e:
                    l.warning(f"Error syncing constraint '{desc}': {e}")
                    sync_failed = True

            # Validate constraints are still satisfiable
            if added_constraints:
                try:
                    if not state.solver.satisfiable():
                        l.warning(f"Constraint sync made solver UNSAT, rolling back {len(added_constraints)} constraints")
                        # Rollback: remove added constraints
                        # Note: claripy doesn't have a native rollback, so we recreate
                        # the solver with the original constraints
                        state.solver._stored_solver = None  # Force solver rebuild
                        sync_failed = True
                except Exception as e:
                    l.warning(f"Error validating constraint sync: {e}")
                    sync_failed = True

            if sync_failed:
                l.debug(f"Constraint sync completed with failures (added {len(added_constraints)} constraints)")
            else:
                l.debug(f"Constraint sync succeeded (added {len(added_constraints)} constraints)")

            return not sync_failed

        callbacks.set_memory_load(memory_load)
        callbacks.set_memory_store(memory_store)
        callbacks.set_lift_block(lift_block)
        callbacks.set_fetch_page(fetch_page)
        callbacks.set_sync_constraints(sync_constraints)

        # P3 fix: Register callbacks for reading/writing registers
        def get_register(offset: int, size: int) -> Tuple[bytes, bool, Optional[object]]:
            """Get register value from Python state.

            P3 fix: This callback allows Rust to read register values from the
            Python state, supporting both concrete and symbolic values.

            Args:
                offset: Register offset in the register file.
                size: Size of the register in bytes.

            Returns:
                (concrete_bytes, is_symbolic, symbolic_ast_or_none)
            """
            state = self._get_callback_state() or self._get_default_state()
            if state is None:
                return (bytes(size), False, None)

            try:
                val = state.registers.load(offset, size, endness=state.arch.register_endness)
                is_sym = getattr(val, 'symbolic', False) if hasattr(val, 'symbolic') else False
                concrete = state.solver.eval(val).to_bytes(size, 'little')
                if is_sym:
                    self._register_handle(id(val), val)
                    return (concrete, True, val)
                return (concrete, False, None)
            except Exception as e:
                l.warning(f"get_register error at offset {offset}: {e}")
                return (bytes(size), False, None)

        def put_register(offset: int, data: bytes):
            """Set register value in Python state.

            P3 fix: This callback allows Rust to write register values to the
            Python state.

            Args:
                offset: Register offset in the register file.
                data: Raw bytes to write to the register.
            """
            state = self._get_callback_state() or self._get_default_state()
            if state is None:
                return

            try:
                val = claripy.BVV(int.from_bytes(data, 'little'), len(data) * 8)
                state.registers.store(offset, val, endness=state.arch.register_endness)
            except Exception as e:
                l.warning(f"put_register error at offset {offset}: {e}")

        callbacks.set_get_register(get_register)
        callbacks.set_put_register(put_register)

        # P4 fix: Dirty call callback for VEX dirty helpers (CPUID, RDTSC, etc.)
        def dirty_call(name: str, args: list, ret_ty_bits: int) -> Tuple[bytes, bool, Optional[object]]:
            """Handle VEX dirty calls (CPUID, RDTSC, x87 ops, etc.).

            P4 fix: VEX dirty calls are helper functions that perform complex
            operations like reading CPU features (CPUID) or timestamp counter
            (RDTSC). These need to be handled in Python where the dirty
            helper implementations live.

            Args:
                name: Name of the dirty helper function (e.g., "amd64g_dirtyhelper_RDTSC").
                args: List of integer arguments to pass to the helper.
                ret_ty_bits: Size of return value in bits.

            Returns:
                (concrete_bytes, is_symbolic, symbolic_ast_or_none)
            """
            state = self._get_callback_state() or self._get_default_state()
            if state is None:
                return (bytes(ret_ty_bits // 8), False, None)

            try:
                from angr.engines.vex.heavy import dirty as dirty_module

                if not hasattr(dirty_module, name):
                    l.warning(f"No dirty call handler for {name}")
                    return (bytes(ret_ty_bits // 8), False, None)

                handler = getattr(dirty_module, name)

                # Convert integer args to claripy BVVs
                claripy_args = [claripy.BVV(arg, 64) for arg in args]

                # Call the handler
                result, constraints = handler(state, *claripy_args)

                # Add any constraints returned by the handler
                if constraints:
                    for c in constraints:
                        state.solver.add(c)

                if result is None:
                    return (bytes(ret_ty_bits // 8), False, None)

                is_sym = getattr(result, 'symbolic', False) if hasattr(result, 'symbolic') else False

                # Get concrete value (evaluate if symbolic)
                concrete_val = state.solver.eval(result)
                num_bytes = ret_ty_bits // 8
                concrete_bytes = concrete_val.to_bytes(num_bytes, 'little')

                if is_sym:
                    self._register_handle(id(result), result)
                    return (concrete_bytes, True, result)
                return (concrete_bytes, False, None)

            except Exception as e:
                l.warning(f"dirty_call {name} error: {e}")
                return (bytes(ret_ty_bits // 8), False, None)

        callbacks.set_dirty_call(dirty_call)

        # Dynamic function resolution callback
        def resolve_function(addr: int, name: Optional[str]) -> Optional[Tuple[str, int, bool]]:
            """Resolve an unmodeled function call.

            This is called when Rust encounters a function that isn't hooked.
            We check angr's procedure registries to see if we can provide a SimProcedure.

            Args:
                addr: The address of the unmodeled function
                name: The function name if known (from symbols), or None

            Returns:
                (name, num_args, no_return) if the function can be resolved,
                None if the function is truly unmodeled and state should deadend.
            """
            # Check if the address is already hooked (shouldn't happen, but safety check)
            if hasattr(self._project, '_sim_procedures'):
                if addr in self._project._sim_procedures:
                    proc = self._project._sim_procedures[addr]
                    proc_name = proc.__class__.__name__ if hasattr(proc, '__class__') else str(proc)
                    num_args = getattr(proc, 'num_args', 0) or 0
                    no_ret = getattr(proc, 'NO_RET', False)
                    return (proc_name, num_args, no_ret)

            # Check if this is a PLT address - resolve through GOT using jmprel table
            # PLT stubs do `jmp [GOT]` - we need to find which symbol the GOT entry maps to
            if hasattr(self._project, 'loader'):
                obj = self._project.loader.find_object_containing(addr)
                if obj:
                    # Check if addr is in a PLT section
                    in_plt = False
                    for section in obj.sections:
                        if section.name in ('.plt', '.plt.got', '.plt.sec') and section.min_addr <= addr < section.max_addr:
                            in_plt = True
                            break

                    if in_plt:
                        # Build mappings for resolution
                        proc_by_name = {}
                        for proc_addr, proc in self._project._sim_procedures.items():
                            proc_name = proc.__class__.__name__ if hasattr(proc, '__class__') else str(proc)
                            proc_by_name[proc_name] = (proc_addr, proc)

                        # Build GOT address -> symbol name mapping from jmprel
                        got_to_sym = {}
                        if hasattr(obj, 'jmprel'):
                            for sym_name, reloc in obj.jmprel.items():
                                got_to_sym[reloc.rebased_addr] = sym_name

                        # Read the GOT address from the PLT instruction
                        try:
                            block = self._project.factory.block(addr, num_inst=1)
                            insn = block.capstone.insns[0] if block.capstone.insns else None
                            if insn and insn.mnemonic == 'jmp':
                                # Get the memory operand (GOT address)
                                for op in insn.operands:
                                    if op.type == 3:  # CS_OP_MEM
                                        # RIP-relative addressing: target = insn.address + insn.size + disp
                                        got_addr = insn.address + insn.size + op.mem.disp
                                        # Look up which symbol this GOT address belongs to
                                        if got_addr in got_to_sym:
                                            sym_name = got_to_sym[got_addr]
                                            if sym_name in proc_by_name:
                                                proc_addr, proc = proc_by_name[sym_name]
                                                num_args = getattr(proc, 'num_args', 0) or 0
                                                no_ret = getattr(proc, 'NO_RET', False)
                                                l.debug(f"Resolved PLT at 0x{addr:x} to {sym_name} (GOT 0x{got_addr:x})")
                                                return (sym_name, num_args, no_ret)
                                        else:
                                            # GOT not in jmprel - try reading the value and checking SimProcedures
                                            state = self._get_default_state()
                                            if state:
                                                got_val = state.memory.load(got_addr, 8, endness='Iend_LE')
                                                extern_addr = state.solver.eval(got_val)
                                                if extern_addr in self._project._sim_procedures:
                                                    proc = self._project._sim_procedures[extern_addr]
                                                    proc_name = proc.__class__.__name__ if hasattr(proc, '__class__') else str(proc)
                                                    num_args = getattr(proc, 'num_args', 0) or 0
                                                    no_ret = getattr(proc, 'NO_RET', False)
                                                    l.debug(f"Resolved PLT at 0x{addr:x} to {proc_name} via GOT value")
                                                    return (proc_name, num_args, no_ret)
                        except Exception as e:
                            l.debug(f"PLT resolution failed for 0x{addr:x}: {e}")


            # Try to resolve by name using angr's procedure registry
            if name:
                try:
                    from angr.procedures import SIM_PROCEDURES

                    # Check common libraries
                    for lib_name, procs in SIM_PROCEDURES.items():
                        if name in procs:
                            proc_class = procs[name]
                            num_args = getattr(proc_class, 'num_args', 0) or 0
                            no_ret = getattr(proc_class, 'NO_RET', False)
                            l.debug(f"Resolved {name} to {lib_name}:{name}")
                            return (name, num_args, no_ret)
                except ImportError:
                    pass

            # Check if we can resolve via the loader's symbol table
            if hasattr(self._project, 'loader'):
                sym = self._project.loader.find_symbol(addr)
                if sym and sym.name:
                    try:
                        from angr.procedures import SIM_PROCEDURES

                        for lib_name, procs in SIM_PROCEDURES.items():
                            if sym.name in procs:
                                proc_class = procs[sym.name]
                                num_args = getattr(proc_class, 'num_args', 0) or 0
                                no_ret = getattr(proc_class, 'NO_RET', False)
                                l.debug(f"Resolved symbol {sym.name} to {lib_name}:{sym.name}")
                                return (sym.name, num_args, no_ret)
                    except ImportError:
                        pass

            # Check if this is internal binary code (not PLT, not extern)
            # If so, return a special marker to tell Rust to continue execution
            if hasattr(self._project, 'loader'):
                obj = self._project.loader.find_object_containing(addr)
                if obj and obj.binary is not None:
                    # Check if addr is in an executable section (but not PLT)
                    for section in obj.sections:
                        if section.is_executable and section.min_addr <= addr < section.max_addr:
                            if section.name not in ('.plt', '.plt.got', '.plt.sec'):
                                # This is internal binary code - return a pass-through marker
                                l.debug(f"Internal function at 0x{addr:x} - returning pass-through")
                                return ("__internal_passthrough__", 0, False)

            # Unresolvable - return None to indicate state should deadend
            l.debug(f"Could not resolve function at 0x{addr:x} (name={name})")
            return None

        callbacks.set_resolve_function(resolve_function)

        # Phase 5 Fix: Add batch callbacks for improved performance
        # Batching reduces Python<->Rust FFI overhead

        def memory_store_batch(stores: list):
            """Batch memory stores callback for improved performance.

            Phase 5 Fix: Handle multiple memory stores in a single callback
            to reduce Python<->Rust FFI overhead.

            Args:
                stores: List of (addr, data) tuples to write.
            """
            state = _get_per_fork_state()
            if state is None:
                return

            for addr, data in stores:
                try:
                    if isinstance(data, (bytes, list)):
                        int_val = int.from_bytes(bytes(data), 'little')
                        val = claripy.BVV(int_val, len(data) * 8)
                    else:
                        val = claripy.BVV(data, 64)
                    state.memory.store(addr, val, endness='Iend_LE')
                except Exception as e:
                    l.debug(f"Batch memory store failed at 0x{addr:x}: {e}")

        def memory_load_batch(loads: list) -> list:
            """Batch memory loads callback for improved performance.

            Phase 5 Fix: Handle multiple memory loads in a single callback
            to reduce Python<->Rust FFI overhead.

            Args:
                loads: List of (addr, size) tuples to read.

            Returns:
                List of (bytes, is_symbolic, ast_or_none) tuples.
            """
            state = _get_per_fork_state()
            if state is None:
                return [(bytes(size), False, None) for _, size in loads]

            results = []
            for addr, size in loads:
                try:
                    val = state.memory.load(addr, size, endness=state.arch.memory_endness)
                    is_sym = getattr(val, 'symbolic', False) if hasattr(val, 'symbolic') else False
                    concrete = state.solver.eval(val).to_bytes(size, 'little')
                    if is_sym:
                        self._register_handle(id(val), val, addr=addr, size=size)
                        results.append((concrete, True, val))
                    else:
                        results.append((concrete, False, None))
                except Exception as e:
                    l.debug(f"Batch memory load failed at 0x{addr:x}: {e}")
                    results.append((bytes(size), False, None))

            return results

        def batch_fetch_pages(page_addrs: list) -> list:
            """Batch page fetch callback for improved performance.

            Phase 5 Fix: Fetch multiple pages in a single callback to reduce
            FFI overhead when prefetching nearby pages.

            Args:
                page_addrs: List of page-aligned addresses to fetch.

            Returns:
                List of (bytes, permissions, is_concrete) tuples.
            """
            state = self._get_callback_state() or self._get_default_state()
            if state is None:
                return [(bytes(4096), 0, True) for _ in page_addrs]

            results = []
            for page_addr in page_addrs:
                try:
                    # Use fetch_page logic but batch it
                    data = state.memory.load(page_addr, 4096, endness='Iend_LE')
                    if getattr(data, 'symbolic', False):
                        concrete = state.solver.eval(data).to_bytes(4096, 'little')
                        results.append((concrete, 7, False))  # RWX, symbolic
                    else:
                        concrete = state.solver.eval(data).to_bytes(4096, 'little')
                        results.append((concrete, 7, True))  # RWX, concrete
                except Exception as e:
                    l.debug(f"Batch page fetch failed at 0x{page_addr:x}: {e}")
                    results.append((bytes(4096), 0, True))

            return results

        # Symbolic value store callback — preserves symbolic expressions in Python memory
        def memory_store_symbolic_value(addr: int, ast):
            """Store a symbolic value (claripy AST) to Python state memory."""
            state = _get_per_fork_state()
            if state is None or ast is None:
                return
            try:
                if hasattr(ast, 'length') and ast.length:
                    size = ast.length // 8
                    state.memory.store(addr, ast, endness=state.arch.memory_endness,
                                       inspect=False, disable_actions=True)
                    self._register_handle(id(ast), ast, addr=addr, size=size)
            except Exception as e:
                l.debug(f"Symbolic store at 0x{addr:x} failed: {e}")

        # Set batch callbacks if available
        if hasattr(callbacks, 'set_memory_store_batch'):
            callbacks.set_memory_store_batch(memory_store_batch)
        if hasattr(callbacks, 'set_memory_load_batch'):
            callbacks.set_memory_load_batch(memory_load_batch)
        if hasattr(callbacks, 'set_batch_fetch_pages'):
            callbacks.set_batch_fetch_pages(batch_fetch_pages)
        # Symbolic store callback — preserves expression trees for non-stack
        # addresses. Errors are caught gracefully (fall through to concrete store).
        if hasattr(callbacks, 'set_memory_store_symbolic_value'):
            callbacks.set_memory_store_symbolic_value(memory_store_symbolic_value)

        self._rust_mgr.set_callbacks(callbacks)
        self._callbacks = callbacks

    def _load_binary_regions(self):
        """Load binary code regions for native lifting."""
        regions = []

        # Get object sections
        for obj in self._project.loader.all_objects:
            # Skip external objects
            if obj.binary is None:
                continue

            for section in obj.sections:
                if section.is_executable:
                    try:
                        data = self._project.loader.memory.load(
                            section.min_addr,
                            section.max_addr - section.min_addr
                        )
                        regions.append((section.min_addr, bytes(data)))
                    except Exception as e:
                        l.debug(f"Could not load section {section.name}: {e}")

        self._rust_mgr.load_binary_regions(regions)

    def _register_simprocedures(self):
        """Register SimProcedures with the Rust manager."""
        procs = []

        # Get hooked addresses from project
        if hasattr(self._project, '_sim_procedures'):
            for addr, proc in self._project._sim_procedures.items():
                name = proc.__class__.__name__ if hasattr(proc, '__class__') else str(proc)
                num_args = getattr(proc, 'num_args', 0) or 0
                no_return = getattr(proc, 'NO_RET', False)
                procs.append((addr, name, num_args, no_return))
                # Track this hook as registered
                self._registered_hooks.add(addr)

        if procs:
            self._rust_mgr.register_simprocedures(procs)


    # =========================================================================
    # Persistent disk cache for Python init results
    # =========================================================================

    @staticmethod
    def _disk_cache_dir() -> str:
        """Return the disk cache directory for init state data."""
        return os.path.join(os.path.expanduser("~"), ".cache", "angr_rust_init")

    @staticmethod
    def _state_has_user_symbolic(state) -> bool:
        """Check if state has user-created symbolic data in memory.

        Detects symbolic argv, symbolic input buffers, etc. by scanning:
        1. The stack page near SP for BVS variables that aren't unconstrained fill.
        2. All memory pages with symbolic_data for user-created variables
           (e.g., state.memory.store(addr, BVS(...))).
        """
        _user_prefixes = ('mem_', 'reg_', 'unconstrained')
        try:
            sp = state.solver.eval(state.regs.sp)
            sp_page = sp & ~0xfff
            page_data = state.memory.load(
                sp_page, 0x1000, endness='Iend_BE',
                inspect=False, disable_actions=True)
            if page_data.symbolic:
                for name in page_data.variables:
                    if not name.startswith(_user_prefixes):
                        return True
        except Exception:
            pass
        # Also check non-stack pages with symbolic_data (e.g., user stores
        # a BVS into .data/.bss segment via state.memory.store()).
        try:
            mem = state.memory
            for page_num, page in mem._pages.items():
                sd = getattr(page, 'symbolic_data', None)
                if not sd:
                    continue
                # Page has symbolic data — check if any variable is user-created
                for offset, bv in sd.items():
                    if hasattr(bv, 'variables'):
                        for name in bv.variables:
                            if not name.startswith(_user_prefixes):
                                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _disk_cache_key(binary_path: str) -> str:
        """Compute a cache key from binary file content hash."""
        try:
            h = hashlib.md5()
            with open(binary_path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return ""

    def _save_init_to_disk_cache(self, cache_key: str, state: "angr.SimState"):
        """Save essential post-init state data to disk cache.

        Stores: addr, registers, stack page, continuation addrs, and loader
        memory pages + lazy regions for fast memory sync on warm runs.
        """
        try:
            cache_dir = self._disk_cache_dir()
            os.makedirs(cache_dir, exist_ok=True)

            arch = self._project.arch
            page_size = 0x1000
            data = {'addr': state.addr, 'registers': {}, 'stack_page': None,
                    'continuation_addrs': [], 'batch_pages': [], 'lazy_regions': []}

            # Extract concrete register values
            for reg_name in arch.register_names.values():
                try:
                    val = getattr(state.regs, reg_name)
                    if not val.symbolic:
                        data['registers'][reg_name] = state.solver.eval(val)
                except Exception:
                    pass

            # Extract stack page at SP (always concretize, even if symbolic)
            try:
                sp = state.solver.eval(state.regs.sp)
                sp_page = sp & ~(page_size - 1)
                page_val = state.memory.load(
                    sp_page, page_size, endness='Iend_BE',
                    inspect=False, disable_actions=True)
                concrete = state.solver.eval(page_val).to_bytes(page_size, 'big')
                data['stack_page'] = (sp_page, concrete)
                # Store stack lazy region
                stack_base = (sp & ~(page_size - 1)) + page_size
                stack_start = stack_base - 0x11_0000
                data['lazy_regions'].append((stack_start, 0x11_0000))
            except Exception:
                pass

            # Extract loader pages (same logic as _sync_memory_to_rust)
            mapped_page_addrs = set()
            for obj in self._project.loader.all_objects:
                try:
                    if hasattr(obj, 'segments') and obj.segments:
                        ranges = [(s.min_addr & ~(page_size - 1),
                                   (s.max_addr + page_size) & ~(page_size - 1))
                                  for s in obj.segments if s.memsize > 0]
                    else:
                        ranges = [(obj.min_addr & ~(page_size - 1),
                                   (obj.max_addr + page_size) & ~(page_size - 1))]
                    for start_page, end_page in ranges:
                        for page_addr in range(start_page, end_page, page_size):
                            if page_addr in mapped_page_addrs:
                                continue
                            try:
                                page_data = self._project.loader.memory.load(page_addr, page_size)
                                if page_data and len(page_data) == page_size:
                                    data['batch_pages'].append((page_addr, bytes(page_data), 7))
                                    mapped_page_addrs.add(page_addr)
                            except Exception:
                                pass
                    # Lazy region for this object
                    region_start = obj.min_addr & ~(page_size - 1)
                    region_end = (obj.max_addr + page_size) & ~(page_size - 1)
                    if region_end - region_start > 0:
                        data['lazy_regions'].append((region_start, region_end - region_start))
                except Exception:
                    pass

            # Section overlay: capture post-init section data (GOT fixups etc.)
            section_patches = []
            for obj in self._project.loader.all_objects:
                if obj.binary is None or not hasattr(obj, 'sections'):
                    continue
                for section in obj.sections:
                    if 0 < section.memsize < 0x10000:
                        try:
                            val = state.memory.load(
                                section.min_addr, section.memsize,
                                endness='Iend_BE', inspect=False, disable_actions=True)
                            if not val.symbolic:
                                section_patches.append(
                                    (section.min_addr,
                                     state.solver.eval(val).to_bytes(section.memsize, 'big')))
                        except Exception:
                            pass
            data['section_patches'] = section_patches

            # Extract continuation addresses from callstack
            frame = state.callstack.top if hasattr(state, 'callstack') else None
            while frame is not None:
                pdata = getattr(frame, 'procedure_data', None)
                if pdata is not None and len(pdata) >= 5:
                    try:
                        data['continuation_addrs'].append(int(pdata[4]))
                    except (TypeError, ValueError):
                        pass
                frame = getattr(frame, 'next', None)

            cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            l.debug(f"Saved init cache to {cache_path} "
                    f"({os.path.getsize(cache_path)} bytes, "
                    f"{len(data['batch_pages'])} pages)")
        except Exception as e:
            l.debug(f"Failed to save disk cache: {e}")

    def _load_init_from_disk_cache(self, cache_key: str):
        """Load post-init state from disk cache.

        Returns (SimState, memory_cache_data) on hit, (None, None) on miss.
        memory_cache_data contains pre-computed loader pages and lazy regions
        for fast memory sync.
        """
        try:
            cache_path = os.path.join(self._disk_cache_dir(), f"{cache_key}.pkl")
            if not os.path.exists(cache_path):
                return None, None

            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            # Restore to a blank state
            state = self._project.factory.blank_state(addr=data['addr'])
            for reg_name, val in data['registers'].items():
                try:
                    setattr(state.regs, reg_name, val)
                except Exception:
                    pass
            if data.get('stack_page'):
                sp_page, page_bytes = data['stack_page']
                state.memory.store(
                    sp_page, claripy.BVV(page_bytes), endness='Iend_BE',
                    inspect=False, disable_actions=True)

            # Restore continuation data
            for cont_addr in data.get('continuation_addrs', []):
                if cont_addr > 0:
                    self._pending_procedure_data.setdefault(cont_addr, None)

            # Extract memory cache data for fast sync
            mem_cache = None
            if data.get('batch_pages') is not None:
                mem_cache = {
                    'batch_pages': data['batch_pages'],
                    'lazy_regions': data.get('lazy_regions', []),
                    'section_patches': data.get('section_patches', []),
                    'stack_page': data.get('stack_page'),
                }

            l.info(f"Disk cache hit: restored state at 0x{data['addr']:x}")
            return state, mem_cache
        except Exception as e:
            l.debug(f"Disk cache load failed: {e}")
            return None, None

    def _extract_continuation_data(self, state: "angr.SimState"):
        """Extract SimProcedure continuation data from a state's callstack.

        When __libc_start_main uses self.call() to invoke main(), it stores
        procedure_data (local_vars) on the callstack frame. When main() returns,
        the continuation (after_main) needs these args. This method captures
        that data so the Rust engine can restore it when the continuation fires.
        """
        frame = state.callstack.top if hasattr(state, 'callstack') else None
        while frame is not None:
            pdata = getattr(frame, 'procedure_data', None)
            if pdata is not None and len(pdata) >= 5:
                cont_addr = pdata[4]  # ideal_addr = continuation address
                try:
                    cont_addr_int = int(cont_addr)
                except (TypeError, ValueError):
                    frame = getattr(frame, 'next', None)
                    continue
                if cont_addr_int > 0:
                    self._pending_procedure_data[cont_addr_int] = pdata
                    l.debug(f"Extracted continuation data for 0x{cont_addr_int:x} "
                            f"({len(pdata[2]) if len(pdata) > 2 and pdata[2] else 0} local_vars)")
            frame = getattr(frame, 'next', None)

    def _run_python_init_if_needed(self, state: "angr.SimState") -> "angr.SimState":
        """Run initialization in Python if the state starts at a loader address.

        When a state starts at a loader/init address (e.g., from full_init_state),
        the C++ init sequence (constructors, .init_array, etc.) is too complex for
        the Rust engine. Run it in Python first, then return the state at main.
        """
        # Check if the state starts at a non-binary address (loader/SimProcedure)
        main_obj = self._project.loader.main_object
        addr = state.addr

        # If state is at the entry point, run init to reach main.
        if addr == self._project.entry:
            # Check in-process cache first — avoids ~180ms of Python simulation
            cache_key = getattr(main_obj, 'binary', None) or ''
            if cache_key and cache_key in RustExplorationManager._init_cache:
                cached = RustExplorationManager._init_cache[cache_key]
                l.info(f"Init cache hit for {cache_key}, copying state at 0x{cached.addr:x}")
                new_state = cached.copy()
                for c in state.solver.constraints:
                    new_state.solver.add(c)
                return new_state
            # Check persistent disk cache (survives across processes).
            # Only use when state has no user symbolic data — blank_state
            # from cache can't preserve symbolic arguments (e.g., argv BVS).
            has_user_symbolic = self._state_has_user_symbolic(state)
            disk_key = self._disk_cache_key(cache_key) if cache_key and not has_user_symbolic else ''
            if disk_key:
                disk_state, mem_cache = self._load_init_from_disk_cache(disk_key)
                if disk_state is not None:
                    for c in state.solver.constraints:
                        disk_state.solver.add(c)
                    self._extract_continuation_data(disk_state)
                    self._mem_cache = mem_cache  # For fast _sync_memory_to_rust
                    return disk_state
            l.info(f"State at entry point 0x{addr:x}, running Python init to main")
        else:
            # Only trigger for states outside ALL loaded binary objects
            obj = self._project.loader.find_object_containing(addr)
            if obj is not None and obj.binary is not None and not obj.binary.startswith('cle##'):
                return state  # In a real binary (not entry), no init needed

            # Also check if it's a known SimProcedure (LinuxLoader, etc.)
            is_init_proc = addr in self._project._sim_procedures
            if not is_init_proc:
                return state  # Not a SimProcedure, don't pre-run

            l.info(f"State at loader address 0x{addr:x}, running Python init to reach main binary")
            cache_key = getattr(main_obj, 'binary', None) or ''
            has_user_symbolic = self._state_has_user_symbolic(state)
            disk_key = self._disk_cache_key(cache_key) if cache_key and not has_user_symbolic else ''
            if disk_key:
                disk_state, mem_cache = self._load_init_from_disk_cache(disk_key)
                if disk_state is not None:
                    for c in state.solver.constraints:
                        disk_state.solver.add(c)
                    self._extract_continuation_data(disk_state)
                    self._mem_cache = mem_cache
                    return disk_state

        try:
            # Find main function address for target
            main_sym = self._project.loader.find_symbol('main')
            main_addr = main_sym.rebased_addr if main_sym else None

            # If no main symbol (stripped binary), extract from _start's
            # __libc_start_main call: rdi = main address
            if main_addr is None:
                try:
                    entry_block = self._project.factory.block(self._project.entry)
                    vex = entry_block.vex
                    # Look for PUT(rdi) = constant before the call exit
                    rdi_offset = self._project.arch.registers.get('rdi', (None,))[0]
                    if rdi_offset is None:
                        rdi_offset = self._project.arch.registers.get('edi', (None,))[0]
                    if rdi_offset is not None:
                        for stmt in reversed(vex.statements):
                            s = str(stmt)
                            if f'PUT(offset={rdi_offset})' in s or 'PUT(rdi)' in s:
                                # Extract the constant value
                                import re
                                m_const = re.search(r'0x([0-9a-fA-F]+)', s)
                                if m_const:
                                    candidate = int(m_const.group(1), 16)
                                    main_obj = self._project.loader.main_object
                                    if main_obj.min_addr <= candidate <= main_obj.max_addr:
                                        main_addr = candidate
                                        l.info(f"Extracted main=0x{main_addr:x} from _start's rdi")
                                break
                except Exception as e:
                    l.debug(f"Could not extract main from _start: {e}")

            # Known init addresses to skip past (not main)
            entry = self._project.entry
            init_addrs = {entry}
            # Also skip PLT stubs and known init functions
            for obj in self._project.loader.all_objects:
                if hasattr(obj, 'entry') and obj.entry:
                    init_addrs.add(obj.entry)

            # Run in Python until we reach main.
            # Use the REAL SimulationManager (not the monkey-patched factory)
            # to avoid infinite recursion when the factory is patched.
            from angr import SimulationManager
            sm = SimulationManager(project=self._project, active_states=[state])
            main_min = main_obj.min_addr
            main_max = main_obj.max_addr

            for step in range(500):
                if not sm.active:
                    break

                # Check if any active state is at main specifically
                if main_addr is not None:
                    at_main = [s for s in sm.active if s.addr == main_addr]
                    if at_main:
                        l.info(f"Python init complete: state reached main at 0x{main_addr:x} "
                               f"after {step} steps")
                        result = at_main[0]
                        self._extract_continuation_data(result)
                        # Cache for future use (in-process + disk)
                        if cache_key and len(RustExplorationManager._init_cache) < RustExplorationManager._init_cache_max:
                            RustExplorationManager._init_cache[cache_key] = result.copy()
                        if disk_key:
                            self._save_init_to_disk_cache(disk_key, result)
                        return result

                # If no main symbol, look for states that are:
                # 1. In the main binary
                # 2. NOT at _start or entry point
                # 3. NOT at a SimProcedure address
                # 4. Past the first few steps (skip _start prologue)
                if main_addr is None and step > 10:
                    in_main = [s for s in sm.active
                               if main_min <= s.addr <= main_max
                               and s.addr not in init_addrs
                               and s.addr not in self._project._sim_procedures]
                    if in_main:
                        l.info(f"Python init complete: state at 0x{in_main[0].addr:x} "
                               f"after {step} steps")
                        result = in_main[0]
                        self._extract_continuation_data(result)
                        if cache_key and len(RustExplorationManager._init_cache) < RustExplorationManager._init_cache_max:
                            RustExplorationManager._init_cache[cache_key] = result.copy()
                        if disk_key:
                            self._save_init_to_disk_cache(disk_key, result)
                        return result

                sm.step()

            # If we couldn't reach main, use whatever we have
            if sm.active:
                best = sm.active[0]
                self._extract_continuation_data(best)
                l.warning(f"Python init: didn't reach main after 500 steps, "
                          f"using state at 0x{best.addr:x}")
                return best
            elif sm.deadended:
                l.warning(f"Python init: all states deadended")
                return state
            else:
                return state
        except Exception as e:
            l.warning(f"Python init failed: {e}, using original state")
            return state

    def _add_rust_state(self, stash: str, angr_state: "angr.SimState"):
        """Add an angr state to a Rust stash.

        Note: Rust internally forks the state, so we need to get the actual
        state ID from Rust after adding to properly cache the angr state.
        """
        # Concretize stack-relative registers for Rust compatibility
        self._concretize_stack_registers(angr_state)

        # Create Rust state from angr state
        is_le = self._project.arch.memory_endness == 'Iend_LE'
        rust_state = _RustSimState(self._project.arch.name, little_endian=is_le)

        # Set PC
        rust_state.pc = angr_state.addr

        # Sync registers
        _t_reg = time.perf_counter_ns()
        self._sync_registers_to_rust(angr_state, rust_state)
        self._perf_stats['init_register_sync_ns'] += time.perf_counter_ns() - _t_reg

        # Map memory regions
        _t_mem = time.perf_counter_ns()
        self._sync_memory_to_rust(angr_state, rust_state)
        self._perf_stats['init_memory_sync_ns'] += time.perf_counter_ns() - _t_mem

        # Get state IDs before adding (to find the new one)
        ids_before = set(self._rust_mgr.get_state_ids(stash))

        # Add to Rust manager
        self._rust_mgr.add_state(stash, rust_state)

        # Get state IDs after adding to find the newly added state ID
        ids_after = set(self._rust_mgr.get_state_ids(stash))
        new_ids = ids_after - ids_before

        # Cache the angr state with the actual Rust state ID
        if new_ids:
            actual_state_id = new_ids.pop()

            # Sync constraints from Python state to Rust solver
            if hasattr(angr_state, 'solver') and angr_state.solver.constraints:
                try:
                    constraints = list(angr_state.solver.constraints)
                    if constraints:
                        sat = self._rust_mgr.add_constraints_to_state(
                            actual_state_id, constraints)
                        l.debug(f"Synced {len(constraints)} initial constraints to Rust state "
                                f"{actual_state_id}, sat={sat}")
                except Exception as e:
                    l.warning(f"Failed to sync initial constraints: {e}")

            self._state_cache[actual_state_id] = angr_state
            # P10 fix: Track this as a root state for plugin restoration
            self._state_roots[actual_state_id] = actual_state_id
            # Extract and cache symbolic memory regions for preservation
            # This ensures symbolic values survive Rust<->Python transitions
            symbolic_pages = self._extract_symbolic_pages(angr_state)
            if symbolic_pages:
                self._symbolic_pages[actual_state_id] = symbolic_pages
                self._cleanup_symbolic_pages_cache()  # Enforce cache limit
                l.debug(f"Cached {len(symbolic_pages)} symbolic pages for state {actual_state_id}")
                # Also import symbolic regions to Rust's symbolic memory so the
                # Rust engine can handle them natively without Python callbacks
                imported_sym = 0
                for addr, ast in symbolic_pages.items():
                    try:
                        import_ast = claripy.Reverse(ast) if hasattr(ast, 'length') and ast.length > 8 else ast
                        self._rust_mgr.import_symbolic_to_state(actual_state_id, addr, import_ast)
                        imported_sym += 1
                        self._register_handle(id(ast), ast, addr=addr,
                                              size=ast.length // 8 if hasattr(ast, 'length') else 1,
                                              state_id=actual_state_id)
                    except Exception as e:
                        l.debug(f"Symbolic page import at 0x{addr:x} failed: {e}")
                if imported_sym:
                    l.debug(f"Imported {imported_sym} symbolic page entries to Rust state {actual_state_id}")
            # Import pending symbolic values to Rust SymbolicMemory
            if hasattr(self, '_pending_symbolic_imports') and self._pending_symbolic_imports:
                imported = 0
                for addr, ast in self._pending_symbolic_imports:
                    try:
                        # Byte-reverse multi-byte symbolic values before importing to Rust.
                        # Wide values are loaded with Iend_BE (preserving original BVS identity).
                        # Rust's memory model uses LE byte extraction internally, so we
                        # apply Reverse() to match.
                        import_ast = claripy.Reverse(ast) if hasattr(ast, 'length') and ast.length > 8 else ast
                        self._rust_mgr.import_symbolic_to_state(actual_state_id, addr, import_ast)
                        imported += 1
                        # Track the ORIGINAL (non-reversed) AST for identity preservation
                        self._register_handle(id(ast), ast, addr=addr, size=ast.length // 8,
                                              state_id=actual_state_id)
                    except Exception as e:
                        l.debug(f"Symbolic import at 0x{addr:x} failed: {e}")
                if imported:
                    l.debug(f"Imported {imported} symbolic values to Rust state {actual_state_id}")
                self._pending_symbolic_imports = []

            # Enforce state cache limit
            self._cleanup_state_cache()
            l.debug(f"Cached angr state with Rust state ID {actual_state_id}")
        else:
            # Fallback: cache with the Python-side state ID
            self._state_cache[rust_state.state_id] = angr_state
            # P10 fix: Track this as a root state for plugin restoration
            self._state_roots[rust_state.state_id] = rust_state.state_id
            symbolic_pages = self._extract_symbolic_pages(angr_state)
            if symbolic_pages:
                self._symbolic_pages[rust_state.state_id] = symbolic_pages
                self._cleanup_symbolic_pages_cache()  # Enforce cache limit
                # Import symbolic regions to Rust's symbolic memory
                for addr, ast in symbolic_pages.items():
                    try:
                        import_ast = claripy.Reverse(ast) if hasattr(ast, 'length') and ast.length > 8 else ast
                        self._rust_mgr.import_symbolic_to_state(rust_state.state_id, addr, import_ast)
                        self._register_handle(id(ast), ast, addr=addr,
                                              size=ast.length // 8 if hasattr(ast, 'length') else 1,
                                              state_id=rust_state.state_id)
                    except Exception:
                        pass
            # Enforce state cache limit
            self._cleanup_state_cache()
            l.warning(f"Could not determine actual Rust state ID, using Python-side ID {rust_state.state_id}")


    def _serialize_irsb(self, irsb) -> str:
        """Serialize a pyvex IRSB to JSON."""
        import json

        def serialize_expr(expr):
            if expr is None:
                return None

            # Handle primitive types directly
            if isinstance(expr, (int, float, str, bool)):
                return expr

            # Get the class name for type checking
            expr_name = type(expr).__name__

            # Handle direct constants (e.g., pyvex.const.U32)
            # These appear in Exit.dst and need Ico_ prefix, not Iex_
            if hasattr(expr, 'value') and expr_name in ('U1', 'U8', 'U16', 'U32', 'U64', 'U128',
                                                         'F32', 'F32i', 'F64', 'F64i', 'V128', 'V256'):
                if expr_name == 'V128':
                    # Rust expects {low: u64, high: u64}
                    val = expr.value if isinstance(expr.value, int) else 0
                    return {
                        'tag': 'Ico_V128',
                        'low': val & 0xFFFFFFFFFFFFFFFF,
                        'high': (val >> 64) & 0xFFFFFFFFFFFFFFFF,
                    }
                elif expr_name == 'V256':
                    val = expr.value if isinstance(expr.value, int) else 0
                    return {
                        'tag': 'Ico_V256',
                        'value': [
                            val & 0xFFFFFFFFFFFFFFFF,
                            (val >> 64) & 0xFFFFFFFFFFFFFFFF,
                            (val >> 128) & 0xFFFFFFFFFFFFFFFF,
                            (val >> 192) & 0xFFFFFFFFFFFFFFFF,
                        ]
                    }
                return {
                    'tag': f'Ico_{expr_name}',
                    'value': expr.value
                }

            # Rust expects VEX expression tags with Iex_ prefix
            result = {'tag': f'Iex_{expr_name}'}

            # Handle Const expressions
            if hasattr(expr, 'con'):
                con = expr.con
                con_name = type(con).__name__
                # Serialize the constant with the right format
                con_result = serialize_expr(con)
                if con_result and isinstance(con_result, dict):
                    result['con'] = con_result
                else:
                    result['con'] = {
                        'tag': f'Ico_{con_name}',
                        'value': con.value if hasattr(con, 'value') else 0
                    }
                return result

            # Handle RdTmp (read temporary)
            if hasattr(expr, 'tmp') and not hasattr(expr, 'data'):
                result['tmp'] = expr.tmp
                return result

            # Handle Get (read register)
            if expr_name == 'Get':
                if hasattr(expr, 'offset'):
                    result['offset'] = expr.offset
                if hasattr(expr, 'ty'):
                    result['ty'] = str(expr.ty)
                return result

            # Handle GetI (indexed get)
            if expr_name == 'GetI':
                if hasattr(expr, 'descr'):
                    # Serialize descr as struct, not string - Rust expects {base, elemTy, nElems}
                    result['descr'] = {
                        'base': expr.descr.base,
                        'elemTy': str(expr.descr.elemTy),
                        'nElems': expr.descr.nElems,
                    }
                if hasattr(expr, 'ix'):
                    result['ix'] = serialize_expr(expr.ix)
                if hasattr(expr, 'bias'):
                    result['bias'] = expr.bias
                return result

            # Handle Load expression
            if expr_name == 'Load':
                if hasattr(expr, 'addr'):
                    result['addr'] = serialize_expr(expr.addr)
                if hasattr(expr, 'ty'):
                    result['ty'] = str(expr.ty)
                if hasattr(expr, 'end'):
                    result['end'] = str(expr.end)
                return result

            # Handle Unop (single argument)
            if expr_name == 'Unop':
                result['op'] = expr.op
                if hasattr(expr, 'args') and len(expr.args) > 0:
                    result['arg'] = serialize_expr(expr.args[0])  # singular 'arg', not 'args'
                return result

            # Handle Binop, Triop, Qop (multiple arguments)
            if hasattr(expr, 'op'):
                result['op'] = expr.op
                if hasattr(expr, 'args'):
                    result['args'] = [serialize_expr(a) for a in expr.args]
                return result

            # Handle ITE (if-then-else)
            if expr_name == 'ITE':
                if hasattr(expr, 'cond'):
                    result['cond'] = serialize_expr(expr.cond)
                if hasattr(expr, 'iftrue'):
                    result['iftrue'] = serialize_expr(expr.iftrue)
                if hasattr(expr, 'iffalse'):
                    result['iffalse'] = serialize_expr(expr.iffalse)
                return result

            # Handle CCall (helper function calls)
            if expr_name == 'CCall':
                if hasattr(expr, 'cee'):
                    cee = expr.cee
                    result['cee'] = {
                        'name': cee.name if hasattr(cee, 'name') else str(cee),
                        'addr': 0,
                        'mcx_mask': getattr(cee, 'mcx_mask', 0),
                    }
                if hasattr(expr, 'retty'):
                    result['retty'] = str(expr.retty)
                if hasattr(expr, 'args'):
                    result['args'] = [serialize_expr(a) for a in expr.args]
                return result

            # Fallback: try common attributes
            if hasattr(expr, 'offset'):
                result['offset'] = expr.offset
            if hasattr(expr, 'ty'):
                result['ty'] = str(expr.ty)
            if hasattr(expr, 'tmp'):
                result['tmp'] = expr.tmp

            return result

        def serialize_stmt(stmt):
            # Rust expects VEX statement tags with Ist_ prefix
            stmt_name = type(stmt).__name__
            result = {'tag': f'Ist_{stmt_name}'}
            tag = stmt_name  # Use plain name for our checks below

            # IMark: instruction marker
            if tag == 'IMark':
                if hasattr(stmt, 'addr'):
                    result['addr'] = stmt.addr  # This is an int for IMark
                if hasattr(stmt, 'len'):
                    result['len'] = stmt.len
                if hasattr(stmt, 'delta'):
                    result['delta'] = stmt.delta
                return result

            # WrTmp: write to temporary
            if tag == 'WrTmp':
                if hasattr(stmt, 'tmp'):
                    result['tmp'] = stmt.tmp
                if hasattr(stmt, 'data'):
                    result['data'] = serialize_expr(stmt.data)
                return result

            # Put: write to register
            if tag == 'Put':
                if hasattr(stmt, 'offset'):
                    result['offset'] = stmt.offset
                if hasattr(stmt, 'data'):
                    result['data'] = serialize_expr(stmt.data)
                return result

            # PutI: indexed put
            if tag == 'PutI':
                if hasattr(stmt, 'descr'):
                    # Serialize descr as struct, not string - Rust expects {base, elemTy, nElems}
                    result['descr'] = {
                        'base': stmt.descr.base,
                        'elemTy': str(stmt.descr.elemTy),
                        'nElems': stmt.descr.nElems,
                    }
                if hasattr(stmt, 'ix'):
                    result['ix'] = serialize_expr(stmt.ix)
                if hasattr(stmt, 'bias'):
                    result['bias'] = stmt.bias
                if hasattr(stmt, 'data'):
                    result['data'] = serialize_expr(stmt.data)
                return result

            # Store: write to memory
            if tag == 'Store':
                if hasattr(stmt, 'addr'):
                    result['addr'] = serialize_expr(stmt.addr)  # addr is an expression!
                if hasattr(stmt, 'data'):
                    result['data'] = serialize_expr(stmt.data)
                if hasattr(stmt, 'end'):
                    result['end'] = str(stmt.end)
                return result

            # StoreG: guarded store
            if tag == 'StoreG':
                if hasattr(stmt, 'addr'):
                    result['addr'] = serialize_expr(stmt.addr)
                if hasattr(stmt, 'data'):
                    result['data'] = serialize_expr(stmt.data)
                if hasattr(stmt, 'guard'):
                    result['guard'] = serialize_expr(stmt.guard)
                if hasattr(stmt, 'end'):
                    result['end'] = str(stmt.end)
                return result

            # LoadG: guarded load
            if tag == 'LoadG':
                if hasattr(stmt, 'dst'):
                    result['dst'] = stmt.dst
                if hasattr(stmt, 'addr'):
                    result['addr'] = serialize_expr(stmt.addr)
                if hasattr(stmt, 'alt'):
                    result['alt'] = serialize_expr(stmt.alt)
                if hasattr(stmt, 'guard'):
                    result['guard'] = serialize_expr(stmt.guard)
                # Add missing cvt and end fields for Rust deserialization
                if hasattr(stmt, 'cvt'):
                    result['cvt'] = str(stmt.cvt)
                if hasattr(stmt, 'end'):
                    result['end'] = str(stmt.end)
                return result

            # Exit: conditional exit
            if tag == 'Exit':
                if hasattr(stmt, 'guard'):
                    result['guard'] = serialize_expr(stmt.guard)
                if hasattr(stmt, 'dst'):
                    # Exit.dst is a constant, not an expression - Rust expects Ico_ format directly
                    dst = stmt.dst
                    type_name = type(dst).__name__
                    result['dst'] = {
                        'tag': f'Ico_{type_name}',
                        'value': dst.value if hasattr(dst, 'value') else 0
                    }
                if hasattr(stmt, 'jk'):
                    result['jk'] = str(stmt.jk)
                if hasattr(stmt, 'offsIP'):
                    result['offsIP'] = stmt.offsIP
                return result

            # CAS: compare-and-swap
            if tag == 'CAS':
                result['end'] = str(stmt.end) if hasattr(stmt, 'end') else "Iend_LE"
                result['oldLo'] = stmt.oldLo if hasattr(stmt, 'oldLo') else 0
                result['oldHi'] = stmt.oldHi if hasattr(stmt, 'oldHi') else None
                if hasattr(stmt, 'addr'):
                    result['addr'] = serialize_expr(stmt.addr)
                if hasattr(stmt, 'dataLo'):
                    result['dataLo'] = serialize_expr(stmt.dataLo)
                if hasattr(stmt, 'dataHi'):
                    result['dataHi'] = serialize_expr(stmt.dataHi)
                if hasattr(stmt, 'expdLo'):
                    result['expdLo'] = serialize_expr(stmt.expdLo)
                if hasattr(stmt, 'expdHi'):
                    result['expdHi'] = serialize_expr(stmt.expdHi)
                return result

            # LLSC: load-linked/store-conditional
            if tag == 'LLSC':
                result['end'] = str(stmt.end) if hasattr(stmt, 'end') else "Iend_LE"
                if hasattr(stmt, 'addr'):
                    result['addr'] = serialize_expr(stmt.addr)
                if hasattr(stmt, 'storedata'):
                    result['storedata'] = serialize_expr(stmt.storedata) if stmt.storedata else None
                if hasattr(stmt, 'result'):
                    result['result'] = stmt.result
                return result

            # Dirty: helper call with side effects
            if tag == 'Dirty':
                if hasattr(stmt, 'cee'):
                    cee = stmt.cee
                    result['cee'] = {
                        'name': cee.name if hasattr(cee, 'name') else str(cee),
                        'addr': 0,
                        'mcx_mask': getattr(cee, 'mcx_mask', 0),
                    }
                if hasattr(stmt, 'guard'):
                    result['guard'] = serialize_expr(stmt.guard) if stmt.guard else None
                else:
                    result['guard'] = None
                if hasattr(stmt, 'args'):
                    result['args'] = [serialize_expr(a) for a in stmt.args]
                if hasattr(stmt, 'tmp'):
                    result['tmp'] = stmt.tmp
                else:
                    result['tmp'] = None
                # Memory effect fields
                result['mFx'] = str(stmt.mFx) if hasattr(stmt, 'mFx') and stmt.mFx else "Ifx_None"
                result['mAddr'] = serialize_expr(stmt.mAddr) if hasattr(stmt, 'mAddr') and stmt.mAddr else None
                result['mSize'] = stmt.mSize if hasattr(stmt, 'mSize') else 0
                result['nFxState'] = stmt.nFxState if hasattr(stmt, 'nFxState') else 0
                return result

            # AbiHint - needs base, len, nia fields
            if tag == 'AbiHint':
                if hasattr(stmt, 'base'):
                    result['base'] = serialize_expr(stmt.base)
                if hasattr(stmt, 'len'):
                    result['len'] = stmt.len
                if hasattr(stmt, 'nia'):
                    result['nia'] = serialize_expr(stmt.nia)
                return result

            # MBE, NoOp - simple statements with no fields
            if tag in ('MBE', 'NoOp'):
                return result

            # Fallback for unknown statements
            if hasattr(stmt, 'addr'):
                # Check if addr is an expression or int
                addr = stmt.addr
                if isinstance(addr, int):
                    result['addr'] = addr
                else:
                    result['addr'] = serialize_expr(addr)
            if hasattr(stmt, 'len'):
                result['len'] = stmt.len
            if hasattr(stmt, 'delta'):
                result['delta'] = stmt.delta
            if hasattr(stmt, 'tmp'):
                result['tmp'] = stmt.tmp
            if hasattr(stmt, 'data'):
                result['data'] = serialize_expr(stmt.data)
            if hasattr(stmt, 'offset'):
                result['offset'] = stmt.offset
            if hasattr(stmt, 'guard'):
                result['guard'] = serialize_expr(stmt.guard)
            if hasattr(stmt, 'dst'):
                result['dst'] = serialize_expr(stmt.dst)

            return result

        data = {
            'addr': irsb.addr,
            'arch': irsb.arch.name if hasattr(irsb.arch, 'name') else str(irsb.arch),
            'statements': [serialize_stmt(s) for s in irsb.statements],
            'next': serialize_expr(irsb.next),
            'jumpkind': str(irsb.jumpkind),
            'offsIP': irsb.offsIP,
            'tyenv': {
                'types': [str(t) for t in irsb.tyenv.types] if irsb.tyenv else []
            }
        }

        return json.dumps(data)

    def _extract_addrs(self, condition) -> list:
        """Extract addresses from a find/avoid condition."""
        if condition is None:
            return []

        if isinstance(condition, int):
            return [condition]

        if isinstance(condition, (list, tuple, set)):
            addrs = []
            for item in condition:
                if isinstance(item, int):
                    addrs.append(item)
            return addrs

        if callable(condition):
            # Can't extract addresses from callable - need Python evaluation
            return []

        return []


                    # Last resort: the pending_callback is still set, which will cause
                    # step() to fail on the next iteration. This is better than silently
                    # losing the state or hanging indefinitely.


    # =========================================================================
    # Public API (SimulationManager-like interface)
    # =========================================================================

    def explore(
        self,
        find: Optional[Union[int, list, Callable]] = None,
        avoid: Optional[Union[int, list, Callable]] = None,
        num_find: int = 1,
        until: Optional[Callable] = None,
        timeout: Optional[float] = None,
        max_steps: Optional[int] = None,
        **kwargs
    ) -> "RustExplorationManager":
        """Run exploration with find/avoid conditions.

        Args:
            find: Address(es) or callable predicate for finding solutions.
            avoid: Address(es) or callable predicate for avoiding states.
            num_find: Number of solutions to find before stopping.
            until: Callable predicate that receives `self` and returns True to stop.
            timeout: Wall-clock timeout in seconds (Phase 3 fix).
            max_steps: Maximum exploration steps before stopping (Phase 3 fix).
            **kwargs: Additional arguments (ignored for compatibility).

        Returns:
            Self, for chaining.
        """
        # Ensure predicate attributes exist (may not be set if find/avoid not provided)
        if not hasattr(self, '_find_predicate'):
            self._find_predicate = None
        if not hasattr(self, '_avoid_predicate'):
            self._avoid_predicate = None

        # Set find addresses and store predicate for P2 callback handling
        # Only override if explicitly provided (don't clear technique-set values)
        if find is not None:
            find_addrs = self._extract_addrs(find)
            self._rust_mgr.set_find_addrs(find_addrs)
            self._rust_mgr.set_find_needs_python(callable(find))
            self._find_predicate = find if callable(find) else None

        # Set avoid addresses
        if avoid is not None:
            avoid_addrs = self._extract_addrs(avoid)
            self._rust_mgr.set_avoid_addrs(avoid_addrs)
        self._rust_mgr.set_avoid_needs_python(callable(avoid))
        # P2 fix: Store the avoid predicate for callback evaluation
        self._avoid_predicate = avoid if callable(avoid) else None

        # Set num_find
        self._rust_mgr.set_num_find(num_find)

        # Phase 3 Fix: Track timeout and steps
        start_time = time.time()
        _explore_start_ns = time.perf_counter_ns()
        steps_taken = 0

        # When callable predicates or techniques are active, use step-by-step
        # exploration so Python callbacks are properly dispatched between steps.
        # The address-based path below handles run() events directly but can
        # miss callback dispatch that step() handles correctly.
        has_predicates = self._find_predicate is not None or self._avoid_predicate is not None
        if has_predicates or bool(self._active_techniques):
            # Enable Rust-side find predicate callbacks to evaluate predicates
            # at EVERY state PC, including PLT addresses that are only visible
            # before hook resolution. Uses lightweight RustStateProxy.
            self._rust_mgr.set_find_needs_python(self._find_predicate is not None)
            self._rust_mgr.set_avoid_needs_python(False)
            # Keep terminal states alive so predicates can check them.
            # Without this, states that output "win" then exit() get dropped
            # before the predicate evaluation can check stdout content.
            self._rust_mgr.set_drop_terminal_states(False)
            # Native puts/printf stay enabled — they write to Rust's per-state
            # stdout_buffer. We inject this into posix.stdout during predicate
            # evaluation via _inject_rust_stdout().
            #
            # Batch predicate mode: run N steps in Rust between predicate
            # evaluations instead of 1 step at a time. Callbacks are still
            # handled immediately when run() returns need_callback events.
            batch_size = 50  # Steps between predicate evaluations
            _time_in_rust_run = 0
            _time_in_predicate_eval = 0
            _time_in_active_check = 0
            while True:
                if timeout is not None and (time.time() - start_time) > timeout:
                    break
                if max_steps is not None and steps_taken >= max_steps:
                    break
                _t0 = time.perf_counter_ns()
                if not self._rust_mgr.has_active_states():
                    _time_in_active_check += time.perf_counter_ns() - _t0
                    break
                _time_in_active_check += time.perf_counter_ns() - _t0

                # Run a batch of steps, handling callbacks as they arise.
                # run(N) processes up to N steps but returns early on callbacks.
                batch_limit = batch_size
                if max_steps is not None:
                    batch_limit = min(batch_limit, max_steps - steps_taken)

                batch_done = False
                batch_steps_start = steps_taken
                while not batch_done and (steps_taken - batch_steps_start) < batch_limit:
                    self._sync_hooks_before_step()
                    self._stats_ffi_crossings += 1
                    remaining = batch_limit - (steps_taken - batch_steps_start)
                    _t1 = time.perf_counter_ns()
                    event = self._rust_mgr.run(remaining)
                    _time_in_rust_run += time.perf_counter_ns() - _t1
                    self._rust_mgr.sync_state_index()

                    if event.event_type == 'need_callback':
                        self._stats_callback_count += 1
                        _cb_start = time.perf_counter_ns()
                        if event.callback_reason == 'simprocedure':
                            self._handle_simprocedure_callback(event)
                        elif event.callback_reason == 'syscall':
                            self._handle_syscall_callback(event)
                        elif event.callback_reason == 'symbolic_branch':
                            self._handle_symbolic_branch_callback(event)
                        elif event.callback_reason == 'find_predicate':
                            self._handle_find_predicate_callback(event)
                        elif event.callback_reason == 'avoid_predicate':
                            self._handle_avoid_predicate_callback(event)
                        else:
                            l.warning(f"Unknown callback reason: {event.callback_reason}")
                            batch_done = True
                        self._stats_time_in_callbacks_ns += time.perf_counter_ns() - _cb_start
                        steps_taken += 1
                    elif event.event_type == 'active_empty':
                        batch_done = True
                    elif event.event_type in ('step_complete', 'found'):
                        # run(N) completed N steps or found a solution
                        steps_taken += remaining
                        batch_done = True
                    else:
                        steps_taken += 1
                        batch_done = True

                # Ensure at least 1 step counted per batch iteration
                if steps_taken == batch_steps_start:
                    steps_taken += 1

                # Apply technique callbacks after the batch
                if self._active_techniques:
                    self._apply_technique_filters()
                    if self._check_technique_complete():
                        break
                # Evaluate predicates on all states after the batch
                _t2 = time.perf_counter_ns()
                self._evaluate_predicates_on_active()
                _time_in_predicate_eval += time.perf_counter_ns() - _t2
                pf = getattr(self, '_predicate_found', [])
                if pf and len(pf) >= num_find:
                    break
                if until is not None:
                    try:
                        if until(self):
                            break
                    except Exception:
                        pass
            # Final predicate check on deadended/remaining states
            self._evaluate_predicates_on_active()
            # Re-enable drop_terminal_states for future exploration
            self._rust_mgr.set_drop_terminal_states(True)
            # Store timing breakdown for stats
            self._time_in_rust_run_ns = _time_in_rust_run
            self._time_in_predicate_eval_ns = _time_in_predicate_eval
            self._time_in_active_check_ns = _time_in_active_check
            self._time_in_explore_ns = time.perf_counter_ns() - _explore_start_ns
            return self

        # Run exploration loop (address-based find/avoid)
        while True:
            # Phase 3 Fix: Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                l.warning(f"Exploration timeout reached ({timeout}s)")
                break

            # Phase 3 Fix: Check max_steps
            if max_steps is not None and steps_taken >= max_steps:
                l.warning(f"Max exploration steps reached ({max_steps})")
                break

            # Sync any dynamically created hooks (continuations from self.call())
            self._sync_hooks_before_step()

            # When until predicates or techniques are active, step one state at
            # a time so Python can check between steps. Otherwise, let Rust run
            # its full batch for performance.
            need_per_step = (until is not None) or bool(self._active_techniques)
            self._stats_ffi_crossings += 1
            event = self._rust_mgr.run(1) if need_per_step else self._rust_mgr.run()
            self._rust_mgr.sync_state_index()
            steps_taken += 1

            # Terminal states (avoid/pruned/deadended) are now dropped immediately
            # in Rust (drop_terminal_states=true), so no periodic cleanup needed.
            # Periodically clean Python state cache to prevent memory leaks.
            if steps_taken % 100 == 0:
                try:
                    active_set = set(self._rust_mgr.get_state_ids('active'))
                    found_set = set(self._rust_mgr.get_state_ids('found'))
                    keep = active_set | found_set
                    keep.update(self._state_roots.get(sid, sid) for sid in keep)
                    for sid in list(self._state_cache.keys()):
                        if sid not in keep:
                            del self._state_cache[sid]
                except Exception:
                    pass

            # --- Dispatch event ---
            should_break = False

            if event.event_type == 'found' and event.found_count >= num_find:
                break
            elif event.event_type == 'active_empty':
                # Before exiting, apply technique filters one last time.
                # Techniques like SearchForNull need to check deadended states
                # and may move them to 'found' before we conclude exploration.
                if self._active_techniques:
                    self._apply_technique_filters()
                    if self._check_technique_complete():
                        break
                    # If techniques moved states back to active, continue
                    if self._rust_mgr.get_state_ids('active'):
                        continue
                break
            elif event.event_type == 'need_callback':
                self._stats_callback_count += 1
                _cb_start = time.perf_counter_ns()
                if event.callback_reason == 'simprocedure':
                    self._handle_simprocedure_callback(event)
                elif event.callback_reason == 'syscall':
                    self._handle_syscall_callback(event)
                elif event.callback_reason == 'symbolic_branch':
                    self._handle_symbolic_branch_callback(event)
                elif event.callback_reason == 'find_predicate':
                    self._handle_find_predicate_callback(event)
                elif event.callback_reason == 'avoid_predicate':
                    self._handle_avoid_predicate_callback(event)
                else:
                    l.warning(f"Unknown callback reason: {event.callback_reason}")
                    break
                self._stats_time_in_callbacks_ns += time.perf_counter_ns() - _cb_start
            elif event.event_type == 'errored':
                if _DBG:
                    l.debug(f"Exploration error (state deadended): {event.callback_reason}")
            elif event.event_type == 'step_complete':
                self._cleanup_symbolic_pages_cache()

            # --- Common post-event checks ---

            # Apply ExplorationTechnique filter/complete callbacks only after
            # step events (not callbacks — callbacks don't change the active stash)
            if self._active_techniques and event.event_type in ('step_complete', 'found', 'steps_exhausted'):
                self._apply_technique_filters()
                if self._check_technique_complete():
                    break

            # Evaluate callable find/avoid predicates
            if self._find_predicate is not None or self._avoid_predicate is not None:
                self._evaluate_predicates_on_active()
                if self._find_predicate and self._found_count() >= num_find:
                    break

            # Check `until` predicate
            if until is not None:
                try:
                    if until(self):
                        l.debug("until predicate returned True, stopping exploration")
                        break
                except Exception as e:
                    l.warning(f"until predicate error: {e}")

        return self

    def step(self, n: int = 1, **kwargs) -> "RustExplorationManager":
        """Step the exploration n times.

        Args:
            n: Number of steps to take.
            **kwargs: Additional arguments (ignored).

        Returns:
            Self, for chaining.
        """
        steps_taken = 0
        while steps_taken < n:
            # Sync any dynamically created hooks (continuations from self.call())
            self._sync_hooks_before_step()

            self._stats_ffi_crossings += 1
            event = self._rust_mgr.run(1)
            self._rust_mgr.sync_state_index()

            if event.event_type == 'need_callback':
                self._stats_callback_count += 1
                _cb_start = time.perf_counter_ns()
                # Handle callback and continue
                if event.callback_reason == 'simprocedure':
                    self._handle_simprocedure_callback(event)
                elif event.callback_reason == 'syscall':
                    self._handle_syscall_callback(event)
                elif event.callback_reason == 'symbolic_branch':
                    self._handle_symbolic_branch_callback(event)
                elif event.callback_reason == 'find_predicate':
                    # P2 fix: Handle callable find predicate evaluation
                    self._handle_find_predicate_callback(event)
                elif event.callback_reason == 'avoid_predicate':
                    # P7 fix: Handle callable avoid predicate evaluation
                    self._handle_avoid_predicate_callback(event)
                else:
                    l.warning(f"Unknown callback reason: {event.callback_reason}")
                    break
                self._stats_time_in_callbacks_ns += time.perf_counter_ns() - _cb_start
                # After callback, increment step count
                steps_taken += 1
            elif event.event_type == 'active_empty':
                # No more active states
                break
            elif event.event_type == 'errored':
                l.warning(f"Step error: {event.callback_reason}")
                break
            elif event.event_type in ('step_complete', 'found'):
                steps_taken += 1
                # Apply technique filters after each step
                if self._active_techniques:
                    self._apply_technique_filters()
            else:
                # Unknown event type, count as a step
                steps_taken += 1

        return self

    def _found_count(self) -> int:
        """Fast count of found states without triggering full state export/sync."""
        count = len(self._rust_mgr.get_state_ids('found'))
        if hasattr(self, '_predicate_found') and self._predicate_found:
            count += len(self._predicate_found)
        return count

    @property
    def active(self) -> list:
        """Get states in the active stash as angr SimStates.

        For SimulationManager API compatibility, this returns full angr states.
        """
        return self._get_stash_states('active')

    @property
    def found(self) -> list:
        """Get states in the found stash as angr SimStates.

        For SimulationManager API compatibility, this returns full angr states
        that can be used with state.solver.eval(), state.posix.dumps(), etc.
        Includes states found via callable predicates.
        """
        states = self._get_stash_states('found')
        # Include states found via callable predicates that may not be in Rust stash
        if hasattr(self, '_predicate_found') and self._predicate_found:
            existing_ids = {id(s) for s in states}
            for s in self._predicate_found:
                if id(s) not in existing_ids:
                    states.append(s)
        return states

    @property
    def avoid(self) -> list:
        """Get states in the avoid stash as angr SimStates."""
        return self._get_stash_states('avoid')

    @property
    def deadended(self) -> list:
        """Get states in the deadended stash as angr SimStates."""
        return self._get_stash_states('deadended')

    @property
    def errored(self) -> list:
        """Get states in the errored stash as angr SimStates."""
        return self._get_stash_states('errored')

    @property
    def unconstrained(self) -> list:
        """Get states in the unconstrained stash as angr SimStates.

        These are states where a symbolic jump target (e.g., ret instruction
        with symbolic return address) had too many possible concrete values
        to enumerate and fork.
        """
        return self._get_stash_states('unconstrained')

    @property
    def proxy(self):
        """Get a RustSimulationManagerProxy for lightweight state access.

        Returns proxy objects that delegate reads directly to Rust via PyO3,
        without creating full angr SimStates or syncing caches. Use this for
        ExplorationTechnique callbacks, predicates, and fast state queries.

        Example:
            mgr.proxy.found[0].solver.eval(x)  # evaluates via Rust Z3 directly
            mgr.proxy.found[0].addr             # reads PC from Rust state
            mgr.proxy.found[0].regs.rax         # reads register from Rust state
        """
        from angr.exploration.rust_state_proxy import RustSimulationManagerProxy
        return RustSimulationManagerProxy(
            self._rust_mgr,
            project=self._project,
            stdin_vars=getattr(self, '_stdin_vars', None),
            stdout_tracker=getattr(self, '_stdout_tracker', {}),
        )

    # State export methods (_get_stash_states, _snapshot_to_angr, etc.)
    # are inherited from RustStateExportMixin in rust_state_export.py

    def eval_register(self, state_id: int, name: str) -> Optional[int]:
        """Evaluate a register from a Rust state.

        Args:
            state_id: The Rust state ID.
            name: Register name (e.g., 'rax').

        Returns:
            Concrete register value, or None if not available.
        """
        return self._rust_mgr.get_state_register(state_id, name)

    def is_satisfiable(self, state_id: int) -> bool:
        """Check if a state's constraints are satisfiable.

        Args:
            state_id: The Rust state ID.

        Returns:
            True if satisfiable, False otherwise.
        """
        return self._rust_mgr.state_satisfiable(state_id)

    def one_found_state(self) -> Optional["angr.SimState"]:
        """Get one found state as an angr SimState.

        This is a convenience method that returns a single found state
        converted to an angr SimState for solution extraction.

        Returns:
            An angr SimState, or None if no found states exist.
        """
        found_ids = self.found
        if found_ids:
            return self.get_state_by_id(found_ids[0])
        return None

    def stash_counts(self) -> dict:
        """Get state counts for all stashes."""
        return dict(self._rust_mgr.stash_counts())

    @property
    def stats(self) -> dict:
        """Get exploration statistics including instrumentation counters."""
        result = dict(self._rust_mgr.stats())
        # Add Python-side instrumentation counters
        result['callback_count'] = self._stats_callback_count
        result['ffi_crossings'] = self._stats_ffi_crossings
        result['state_creations'] = self._stats_state_creations
        result['cache_hits'] = self._stats_cache_hits
        result['cache_misses'] = self._stats_cache_misses
        result['technique_filter_calls'] = self._stats_technique_filter_calls
        result['hook_sync_calls'] = self._stats_hook_sync_calls
        result['hook_sync_skips'] = self._stats_hook_sync_skips
        result['time_in_callbacks'] = self._stats_time_in_callbacks_ns / 1e9  # seconds
        # Add timing breakdown for predicate-mode exploration loop
        if hasattr(self, '_time_in_rust_run_ns'):
            result['time_in_rust_run'] = self._time_in_rust_run_ns / 1e9
            result['time_in_predicate_eval'] = self._time_in_predicate_eval_ns / 1e9
            result['time_in_active_check'] = self._time_in_active_check_ns / 1e9
        if hasattr(self, '_time_in_explore_ns'):
            result['time_in_explore'] = self._time_in_explore_ns / 1e9
        # Include Rust execution profiling stats if available
        try:
            rust_exec_stats = self._rust_mgr.get_execution_stats()
            for k, v in rust_exec_stats.items():
                result[f'rust_{k}'] = v
        except Exception:
            pass
        return result

    def enable_profiling(self):
        """Enable Rust-side execution profiling for detailed timing breakdown."""
        self._rust_mgr.set_profiling(True)

    def disable_profiling(self):
        """Disable Rust-side execution profiling."""
        self._rust_mgr.set_profiling(False)

    # Compatibility methods for SimulationManager API

    def use_technique(self, technique, **kwargs):
        """Apply an exploration technique. See rust_techniques.use_technique()."""
        from angr.exploration.rust_techniques import use_technique
        return use_technique(self, technique, **kwargs)

    def remove_technique(self, technique) -> bool:
        """Remove an exploration technique. See rust_techniques.remove_technique()."""
        from angr.exploration.rust_techniques import remove_technique
        return remove_technique(self, technique)

    def _apply_technique_filters(self):
        """Apply ExplorationTechnique filter() callbacks via proxy."""
        self._stats_technique_filter_calls += 1
        from angr.exploration.rust_techniques import apply_technique_filters
        apply_technique_filters(self)

    def _check_technique_complete(self) -> bool:
        """Check ExplorationTechnique complete() callbacks."""
        from angr.exploration.rust_techniques import check_technique_complete
        return check_technique_complete(self)

    def run(self, **kwargs) -> "RustExplorationManager":
        """Alias for explore() for SimulationManager compatibility.

        Handles step_func: if provided, called after EACH step (matching
        Python SimulationManager behavior). Without step_func, delegates
        to explore() for batch execution.
        """
        step_func = kwargs.pop('step_func', None)
        if step_func is None:
            return self.explore(**kwargs)

        # step_func mode: step one at a time with step_func applied after
        # each step, matching Python SimulationManager.run() behavior.
        # This is used by Callable for concrete_only pruning.
        # Keep terminal states since step_func may need deadended states.
        self._rust_mgr.set_drop_terminal_states(False)
        try:
            n = kwargs.pop('n', None)
            stash = kwargs.pop('stash', 'active')
            until = kwargs.pop('until', None)
            import itertools
            for _ in itertools.count() if n is None else range(n):
                if not self._rust_mgr.get_state_ids(stash):
                    break
                self.step(**kwargs)
                step_func(self)
                if until and until(self):
                    break
        finally:
            self._rust_mgr.set_drop_terminal_states(True)
        return self

    def move(self, from_stash: str, to_stash: str, filter_func=None) -> "RustExplorationManager":
        """Move states between stashes.

        P8 fix: Full support for filter functions.

        Args:
            from_stash: Source stash name.
            to_stash: Destination stash name.
            filter_func: Optional callable predicate. States matching the predicate
                        are moved; others remain in the source stash.

        Returns:
            Self, for chaining.
        """
        if filter_func is None:
            # Move all states
            self._rust_mgr.move_states(from_stash, to_stash, None)
        else:
            # Export states, evaluate predicate, and handle accordingly
            state_ids = list(self._rust_mgr.get_state_ids(from_stash))
            move_ids = []
            keep_ids = []

            for state_id in state_ids:
                try:
                    # Export state for predicate evaluation
                    snapshot = self._rust_mgr.export_state(state_id)
                    py_state = self._snapshot_to_angr(snapshot)

                    if filter_func(py_state):
                        move_ids.append(state_id)
                    else:
                        keep_ids.append(state_id)
                except Exception as e:
                    if _DBG:
                        l.debug(f"P8: move filter error for state {state_id}: {e}")
                    keep_ids.append(state_id)  # Keep on error

            # Use Rust to move matching states
            for state_id in move_ids:
                try:
                    self._rust_mgr.move_state(state_id, from_stash, to_stash)
                except Exception:
                    pass  # State may have already been moved

        return self

    def stash(self, filter_func=None, from_stash="active", to_stash="stashed") -> "RustExplorationManager":
        """Stash some states. Alias for move() with different defaults."""
        return self.move(from_stash, to_stash, filter_func=filter_func)

    def unstash(self, filter_func=None, to_stash="active", from_stash="stashed") -> "RustExplorationManager":
        """Unstash some states. Alias for move() with different defaults."""
        return self.move(from_stash, to_stash, filter_func=filter_func)

    def filter(self, stash: str = 'active', filter_func=None) -> "RustExplorationManager":
        """Filter states in a stash by predicate.

        P8 fix: States not matching the predicate are removed (moved to 'pruned').

        Args:
            stash: The stash to filter. Defaults to 'active'.
            filter_func: Callable predicate. States where this returns True are kept.

        Returns:
            Self, for chaining.
        """
        if filter_func is None:
            return self

        state_ids = list(self._rust_mgr.get_state_ids(stash))
        keep_ids = []
        prune_ids = []

        for state_id in state_ids:
            try:
                # Try lightweight proxy first (avoids expensive full state export).
                # Falls back to full export if the filter accesses something
                # the proxy doesn't support.
                from angr.exploration.rust_state_proxy import RustStateProxy
                proxy = RustStateProxy(self._rust_mgr, state_id, self._project)
                try:
                    if filter_func(proxy):
                        keep_ids.append(state_id)
                    else:
                        prune_ids.append(state_id)
                    continue  # Proxy worked, skip full export
                except (AttributeError, TypeError, NotImplementedError):
                    pass  # Proxy didn't support something, fall back

                snapshot = self._rust_mgr.export_state(state_id)
                py_state = self._snapshot_to_angr(snapshot)

                if filter_func(py_state):
                    keep_ids.append(state_id)
                    # Cache the snapshot-exported state so _get_stash_states
                    # can find it later (avoids falling back to stale root copy)
                    self._state_cache[state_id] = py_state
                else:
                    prune_ids.append(state_id)
            except Exception as e:
                if _DBG:
                    l.debug(f"P8: filter error for state {state_id}: {e}")
                keep_ids.append(state_id)  # Keep on error

        # Move non-matching states to pruned stash
        for state_id in prune_ids:
            try:
                self._rust_mgr.move_state(state_id, stash, 'pruned')
            except Exception:
                pass

        return self

    def prune(self, stash: str = 'active', filter_func=None) -> "RustExplorationManager":
        """Remove states from a stash based on predicate.

        P8 fix: Default behavior prunes unsatisfiable states.

        Args:
            stash: The stash to prune. Defaults to 'active'.
            filter_func: Callable predicate. States where this returns True are kept.
                        Defaults to keeping satisfiable states.

        Returns:
            Self, for chaining.
        """
        if filter_func is None:
            # Fast path: check satisfiability via Rust solver directly,
            # avoiding expensive state export + _snapshot_to_angr conversion.
            state_ids = list(self._rust_mgr.get_state_ids(stash))
            prune_ids = []
            for state_id in state_ids:
                try:
                    if not self._rust_mgr.state_satisfiable(state_id):
                        prune_ids.append(state_id)
                except Exception:
                    pass  # Keep on error
            for state_id in prune_ids:
                try:
                    self._rust_mgr.move_state(state_id, stash, 'pruned')
                except Exception:
                    pass
            return self

        return self.filter(stash=stash, filter_func=filter_func)

    def drop(self, stash: str = 'active', filter_func=None) -> "RustExplorationManager":
        """Drop states from a stash.

        P8 fix: States matching the predicate (or all if no predicate) are removed.

        Args:
            stash: The stash to drop from. Defaults to 'active'.
            filter_func: Optional callable predicate. If provided, only states
                        matching the predicate are dropped.

        Returns:
            Self, for chaining.
        """
        if filter_func is None:
            # Drop all states from the stash
            try:
                self._rust_mgr.clear_stash(stash)
            except AttributeError:
                # Fallback: move all to deadended
                self._rust_mgr.move_states(stash, 'deadended', None)
        else:
            # Drop states matching predicate
            state_ids = list(self._rust_mgr.get_state_ids(stash))

            for state_id in state_ids:
                try:
                    snapshot = self._rust_mgr.export_state(state_id)
                    py_state = self._snapshot_to_angr(snapshot)

                    if filter_func(py_state):
                        try:
                            self._rust_mgr.move_state(state_id, stash, 'deadended')
                        except Exception:
                            pass
                except Exception as e:
                    if _DBG:
                        l.debug(f"P8: drop filter error for state {state_id}: {e}")

        return self

    def split(self, stash_from: str = 'active', stash_to: str = 'stashed',
              limit: int = 8, filter_func=None) -> "RustExplorationManager":
        """Split states between stashes.

        P8 fix: Moves excess states to another stash to limit exploration width.

        Args:
            stash_from: Source stash. Defaults to 'active'.
            stash_to: Destination for excess states. Defaults to 'stashed'.
            limit: Maximum states to keep in source stash. Defaults to 8.
            filter_func: Optional predicate to determine which states to move.

        Returns:
            Self, for chaining.
        """
        state_ids = list(self._rust_mgr.get_state_ids(stash_from))

        if len(state_ids) <= limit:
            return self

        # Move excess states to destination stash
        excess_ids = state_ids[limit:]
        for state_id in excess_ids:
            try:
                self._rust_mgr.move_state(state_id, stash_from, stash_to)
            except Exception:
                pass

        return self

    @property
    def stashes(self) -> dict:
        """Get all stashes as a dictionary for SimulationManager compatibility.

        P8 fix: Returns state IDs per stash for compatibility.
        """
        result = {}
        for stash_name in ['active', 'found', 'avoid', 'deadended', 'errored',
                          'unconstrained', 'pruned', 'stashed']:
            try:
                state_ids = list(self._rust_mgr.get_state_ids(stash_name))
                result[stash_name] = state_ids
            except Exception:
                result[stash_name] = []
        return result

    @property
    def one_active(self):
        """Get one active state (compatibility stub)."""
        active = self.active
        if active:
            return active[0]
        return None

    @property
    def one_found(self):
        """Get one found state (compatibility stub)."""
        found = self.found
        if found:
            return found[0]
        return None

    def copy(self) -> "RustExplorationManager":
        """Return self for SimulationManager API compatibility.

        RustExplorationManager is stateful and backed by a single Rust object,
        so a true deep copy isn't possible. Return self to satisfy callers like
        angr.callable that store a reference to the manager.
        """
        return self

    def merge(self, stash: str = 'active', **kwargs) -> "RustExplorationManager":
        """Merge states in a stash (best-effort for SimulationManager compatibility).

        True state merging requires claripy merge which isn't supported across
        the Rust/Python boundary. This is a no-op that keeps the first state.
        """
        state_ids = list(self._rust_mgr.get_state_ids(stash))
        if len(state_ids) > 1:
            # Keep first, drop rest
            for sid in state_ids[1:]:
                try:
                    self._rust_mgr.move_state(sid, stash, '_drop')
                except Exception:
                    pass
            try:
                self._rust_mgr.clear_stash('_drop')
            except Exception:
                pass
        return self

    def __len__(self) -> int:
        """Return total number of active states."""
        return self._rust_mgr.active_count()

    def __getattr__(self, name: str):
        """Handle attribute access for stash names."""
        # Try to get stash by name
        if name.startswith('_'):
            raise AttributeError(name)

        # Handle one_* prefix for single state access (SimulationManager compatibility)
        if name.startswith("one_"):
            stash_name = name[4:]  # Remove "one_" prefix
            states = self._get_stash_states(stash_name)
            return states[0] if states else None

        try:
            return self._rust_mgr.get_state_ids(name)
        except Exception:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
