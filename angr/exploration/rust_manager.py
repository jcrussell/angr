"""Python wrapper for Rust-native exploration manager.

This module provides `RustExplorationManager`, a Python-facing interface
to the Rust exploration loop that achieves ~3x speedup by:
- Managing states entirely in Rust (using RustSimState)
- Processing symbolic branches with deferred forks
- Only calling Python for SimProcedures and syscalls
- Implementing find/avoid address checking in Rust
"""
from __future__ import annotations

import logging
import time
import weakref
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union

import claripy

if TYPE_CHECKING:
    import angr

l = logging.getLogger(name=__name__)

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


from angr.exploration.rust_identity import SymbolicIdentityTracker, CallbackMemoryTracker


from angr.exploration.rust_state_export import RustStateExportMixin


class RustExplorationManager(RustStateExportMixin):
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
        if not RUST_EXPLORATION_AVAILABLE:
            raise ImportError(
                "RustExplorationManager not available. "
                "Build with vex-engine feature enabled."
            )

        self._project = project
        self._rust_mgr = _RustExplorationManager(project.arch.name)

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
        }

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

        # P10 fix: Track root state IDs for plugin restoration
        # Maps state_id -> root_state_id (the original state from Python)
        # When Rust forks states, this allows finding the original state for plugin copying
        self._state_roots: Dict[int, int] = {}

        # P9 fix: Track active exploration techniques
        # Techniques are applied during exploration steps
        self._active_techniques: list = []

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
        lines.append(f"  Sync back: {s['callback_simprocedure_sync_back_ns']/1e6:.1f}ms")
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

    def _sync_hooks_before_step(self):
        """Sync dynamically created hooks (like continuations) before stepping.

        SimProcedures can create continuation hooks via self.call() which are
        added to the project dynamically. This method ensures those hooks are
        registered with Rust before exploration continues.

        This is critical for __libc_start_main and other procedures that use
        continuations to chain function calls (init -> main -> fini).
        """
        self._stats_hook_sync_calls += 1

        if not hasattr(self._project, '_sim_procedures'):
            self._stats_hook_sync_skips += 1
            return

        # Fast path: if dict length hasn't changed, no new hooks were added.
        # This avoids O(n) set construction on every step.
        proc_len = len(self._project._sim_procedures)
        if proc_len == len(self._registered_hooks):
            self._stats_hook_sync_skips += 1
            return

        current_hooks = set(self._project._sim_procedures.keys())
        new_hooks = current_hooks - self._registered_hooks

        if not new_hooks:
            self._stats_hook_sync_skips += 1
            return

        # Register newly created hooks with Rust
        procs = []
        for addr in new_hooks:
            proc = self._project._sim_procedures[addr]
            name = proc.__class__.__name__ if hasattr(proc, '__class__') else str(proc)
            num_args = getattr(proc, 'num_args', 0) or 0
            no_return = getattr(proc, 'NO_RET', False)
            procs.append((addr, name, num_args, no_return))
            self._registered_hooks.add(addr)
            l.debug(f"Syncing dynamically created hook at 0x{addr:x}: {name}")

        if procs:
            self._rust_mgr.register_simprocedures(procs)
            l.debug(f"Synced {len(procs)} dynamically created hooks")

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
            # Check init cache first — avoids ~180ms of Python simulation
            cache_key = getattr(main_obj, 'binary', None) or ''
            if cache_key and cache_key in RustExplorationManager._init_cache:
                cached = RustExplorationManager._init_cache[cache_key]
                l.info(f"Init cache hit for {cache_key}, copying state at 0x{cached.addr:x}")
                new_state = cached.copy()
                # Transfer solver constraints from original state
                for c in state.solver.constraints:
                    new_state.solver.add(c)
                return new_state
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
                        # Cache for future use
                        if cache_key and len(RustExplorationManager._init_cache) < RustExplorationManager._init_cache_max:
                            RustExplorationManager._init_cache[cache_key] = result.copy()
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
        rust_state = _RustSimState(self._project.arch.name)

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

    def _sync_registers_to_rust(self, angr_state: "angr.SimState", rust_state: "_RustSimState"):
        """Sync registers from angr state to Rust state."""
        regs = angr_state.regs
        arch = angr_state.arch

        # Sync common registers based on architecture
        if arch.name in ('AMD64', 'X86_64'):
            reg_names = ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi',
                        'rbp', 'rsp', 'r8', 'r9', 'r10', 'r11',
                        'r12', 'r13', 'r14', 'r15', 'rip']
        elif arch.name == 'X86':
            reg_names = ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi',
                        'ebp', 'esp', 'eip']
        elif arch.name.startswith('ARM'):
            reg_names = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7',
                        'r8', 'r9', 'r10', 'r11', 'r12', 'sp', 'lr', 'pc']
        else:
            reg_names = []

        # Build dict of all register values, then send in single FFI call
        bulk_regs = {}
        for reg_name in reg_names:
            try:
                reg_val = getattr(regs, reg_name)
                # Fast path: if concrete, extract directly without solver
                if not reg_val.symbolic:
                    concrete_val = reg_val.args[0] if reg_val.op == 'BVV' else angr_state.solver.eval(reg_val)
                else:
                    concrete_val = angr_state.solver.eval(reg_val)
                bulk_regs[reg_name] = concrete_val
            except (AttributeError, KeyError, Exception):
                pass
        if bulk_regs:
            rust_state.set_registers_bulk(bulk_regs)

    def _sync_memory_to_rust(self, angr_state: "angr.SimState", rust_state: "_RustSimState"):
        """Sync memory from angr state to Rust state.

        Strategy: map pages for each loaded segment (not the entire address
        space). Uses the Python state's memory for relocations/initialized data.
        """
        page_size = 0x1000
        pages_mapped = 0

        # Identify pages containing user-written symbolic data. These pages
        # should NOT be pre-populated with concrete loader data, so that Rust
        # falls back to the Python memory_load callback which returns the
        # symbolic AST (enabling symbolic forking on comparisons).
        symbolic_pages = set()
        if hasattr(angr_state.memory, 'get_symbolic_addrs'):
            try:
                for addr in angr_state.memory.get_symbolic_addrs():
                    symbolic_pages.add(addr & ~(page_size - 1))
            except Exception:
                pass
        if not symbolic_pages and hasattr(angr_state.memory, '_pages'):
            # Fallback: scan pages for symbolic content
            mem_page_size = getattr(angr_state.memory, 'page_size', page_size)
            for page_num in list(angr_state.memory._pages.keys()):
                page = angr_state.memory._pages.get(page_num)
                if page is not None and hasattr(page, 'symbolic_bitmap'):
                    sb = page.symbolic_bitmap
                    if sb is not None and any(sb):
                        symbolic_pages.add(page_num * mem_page_size)
        if symbolic_pages:
            l.debug(f"Skipping {len(symbolic_pages)} pages with symbolic data during memory sync")

        # Map pages for each loaded object's segments
        for obj in self._project.loader.all_objects:
            try:
                start_page = obj.min_addr & ~(page_size - 1)
                end_page = (obj.max_addr + page_size) & ~(page_size - 1)
                for page_addr in range(start_page, end_page, page_size):
                    if page_addr in symbolic_pages:
                        continue  # Don't overwrite symbolic data with concrete
                    try:
                        # Use loader memory directly (fast, no Z3 eval)
                        data = self._project.loader.memory.load(page_addr, page_size)
                        if data and len(data) == page_size:
                            rust_state.map_memory_data(page_addr, bytes(data), 7)
                            pages_mapped += 1
                    except Exception:
                        pass
            except Exception:
                pass

        # Overlay relocated data from Python state (GOT entries, etc.)
        # Only do small concrete sections to avoid expensive Z3 eval
        for obj in self._project.loader.all_objects:
            if obj.binary is None or not hasattr(obj, 'sections'):
                continue
            for section in obj.sections:
                if section.memsize > 0 and section.memsize < 0x10000:
                    try:
                        val = angr_state.memory.load(
                            section.min_addr, section.memsize,
                            endness='Iend_BE', inspect=False,
                            disable_actions=True)
                        if not val.symbolic:
                            data = angr_state.solver.eval(val).to_bytes(section.memsize, 'big')
                            rust_state.map_memory_data(section.min_addr, data, 7)
                    except Exception:
                        pass

        l.debug(f"Pre-populated {pages_mapped} pages from loaded objects")

        # Add lazy regions for ALL loader objects + stack so the fetch_page
        # callback can populate any unmapped page on demand.
        for obj in self._project.loader.all_objects:
            try:
                region_start = obj.min_addr & ~(page_size - 1)
                region_end = (obj.max_addr + page_size) & ~(page_size - 1)
                region_size = region_end - region_start
                if region_size > 0:
                    rust_state.add_lazy_region(region_start, region_size)
            except Exception:
                pass

        arch = self._project.arch
        try:
            sp = angr_state.solver.eval(angr_state.regs.sp)
        except Exception:
            sp = 0x7fff_fff0_0000 if arch.bits == 64 else 0x7fff_0000
        stack_base = (sp & ~(page_size - 1)) + page_size
        stack_start = stack_base - 0x11_0000
        rust_state.add_lazy_region(stack_start, 0x11_0000)

        # Pre-populate stack page at SP from the Python state.
        # Only load the page containing SP - this has the active stack frame
        # with return addresses and saved registers. All other stack pages
        # use lazy fetching via fetch_page callback when actually accessed.
        # This avoids expensive solver.eval() on unconstrained fill pages.
        sp_page = sp & ~(page_size - 1)
        pages_synced = 0
        symbolic_regions = []  # (addr, claripy_ast) pairs to import
        for page_addr in [sp_page]:
            try:
                page_data = angr_state.memory.load(
                    page_addr, page_size, endness='Iend_BE',
                    inspect=False, disable_actions=True
                )
                # Map concrete representation of the page
                concrete = angr_state.solver.eval(page_data).to_bytes(page_size, 'big')
                rust_state.map_memory_data(page_addr, concrete, 6)
                pages_synced += 1

                # Only extract symbolic regions for meaningful symbols.
                # Skip the expensive byte-by-byte scan for pages that only
                # contain unconstrained fill (common for stack pages at init).
                if page_data.symbolic:
                    leaf_names = list(page_data.variables)
                    has_user_sym = any(
                        not n.startswith('mem_') and not n.startswith('reg_')
                        and not n.startswith('unconstrained')
                        for n in leaf_names
                    )
                    if has_user_sym:
                        self._extract_symbolic_regions(
                            angr_state, page_addr, page_size,
                            arch.bytes, symbolic_regions)
            except Exception:
                pass
        if pages_synced:
            l.debug(f"Pre-populated {pages_synced} stack pages in Rust memory")

        # Scan non-stack memory pages for user-written symbolic data.
        # Import WIDE symbolic objects (not byte-by-byte) to preserve identity.
        # This ensures the Rust engine creates a single symbol that can be
        # unified with the original user-created BVS during state export.
        #
        # OPTIMIZATION: Only scan pages with actual user symbolic data, NOT
        # unconstrained fill pages. After Python init, thousands of pages get
        # filled with mem_*/unconstrained symbols. Calling memory.load() on
        # each is ~0.3ms/page = 600ms+ for 2000 pages. Instead:
        # 1. get_symbolic_addrs() returns only user-written addresses (fast)
        # 2. Fall back to page.symbolic_data dict (O(1) per page, non-empty
        #    only for pages with explicit stores, not default fill)
        try:
            pages = getattr(angr_state.memory, '_pages', {})
            # Build set of page numbers worth scanning
            user_sym_pages = set()
            if hasattr(angr_state.memory, 'get_symbolic_addrs'):
                try:
                    for addr in angr_state.memory.get_symbolic_addrs():
                        user_sym_pages.add(addr // page_size)
                except Exception:
                    pass
            if not user_sym_pages:
                # Cheap filter: check symbolic_data dict on each page object.
                # symbolic_data is non-empty only for pages with explicit stores,
                # not for unconstrained fill from the default filler mixin.
                for page_no in pages:
                    page = pages[page_no]
                    if hasattr(page, 'symbolic_data'):
                        sd = page.symbolic_data
                        if sd and len(sd) > 0:
                            user_sym_pages.add(page_no)

            for page_no in sorted(user_sym_pages):
                page_addr = page_no * page_size
                if stack_start <= page_addr < stack_base:
                    continue  # Skip stack pages
                try:
                    page_data = angr_state.memory.load(
                        page_addr, page_size, endness='Iend_BE',
                        inspect=False, disable_actions=True)
                    if page_data.symbolic:
                        leaf_names = list(page_data.variables)
                        has_user_sym = any(
                            not n.startswith('mem_') and not n.startswith('reg_')
                            and not n.startswith('unconstrained')
                            for n in leaf_names
                        )
                        if has_user_sym:
                            self._extract_wide_symbolic_regions(
                                angr_state, page_addr, page_size, symbolic_regions)
                except Exception:
                    pass
        except Exception:
            pass

        if symbolic_regions:
            self._pending_symbolic_imports = symbolic_regions
            l.debug(f"Found {len(symbolic_regions)} symbolic regions for import")

    def _extract_wide_symbolic_regions(self, angr_state, page_addr, page_size, out):
        """Extract WIDE symbolic regions from a page.

        Groups contiguous symbolic bytes that share the same variable and
        imports them as a single wide object. This preserves symbolic identity
        across the Rust/Python boundary (avoids creating rust_sym_XXX aliases).
        """
        offset = 0
        while offset < page_size:
            addr = page_addr + offset
            try:
                val = angr_state.memory.load(addr, 1, endness='Iend_BE',
                                              inspect=False, disable_actions=True)
                if not val.symbolic:
                    offset += 1
                    continue
                leaf_names = list(val.variables)
                is_user = any(
                    not n.startswith('mem_') and not n.startswith('reg_')
                    and not n.startswith('unconstrained')
                    for n in leaf_names
                )
                if not is_user:
                    offset += 1
                    continue

                # Found a symbolic byte — scan forward to find the full region
                # that shares the same variable set
                region_start = addr
                region_vars = frozenset(leaf_names)
                region_len = 1
                while offset + region_len < page_size:
                    next_addr = page_addr + offset + region_len
                    try:
                        next_val = angr_state.memory.load(
                            next_addr, 1, endness='Iend_BE',
                            inspect=False, disable_actions=True)
                        if next_val.symbolic and frozenset(next_val.variables) == region_vars:
                            region_len += 1
                        else:
                            break
                    except Exception:
                        break

                # Load the full region as a single wide object
                try:
                    wide_val = angr_state.memory.load(
                        region_start, region_len, endness='Iend_BE',
                        inspect=False, disable_actions=True)
                    out.append((region_start, wide_val))
                except Exception:
                    # Fall back to byte-by-byte
                    out.append((addr, val))

                offset += region_len
            except Exception:
                offset += 1

    def _extract_symbolic_regions(self, angr_state, page_addr, page_size, ptr_size, out):
        """Extract symbolic memory regions from a page for import to Rust.

        Only imports BYTE-LEVEL symbolic values that contain user-defined
        symbols (BVS with names not starting with 'mem_' or 'reg_').
        Skips unconstrained fill variables.
        """
        for offset in range(0, page_size, 1):
            addr = page_addr + offset
            try:
                val = angr_state.memory.load(addr, 1,
                                             endness='Iend_BE',
                                             inspect=False, disable_actions=True)
                if val.symbolic:
                    # Check if this contains a user-defined variable
                    # (not just unconstrained fill from entry_state)
                    leaf_names = list(val.variables)
                    is_user_sym = any(
                        not n.startswith('mem_') and not n.startswith('reg_')
                        and not n.startswith('unconstrained')
                        for n in leaf_names
                    )
                    if is_user_sym:
                        out.append((addr, val))
            except Exception:
                pass

    def _concretize_stack_registers(self, state: "angr.SimState"):
        """Concretize stack registers for Rust memory mapping compatibility.

        This prevents symbolic address issues during Rust exploration by
        ensuring stack-relative registers have concrete values.

        Uses solver.eval() to get the same concrete values that Python
        already evaluated, ensuring Rust has consistent state with what
        Python set up (e.g., address calculations like ebp - 0x80004).
        """
        arch = state.arch

        # Determine which registers to concretize based on architecture
        if arch.name in ('AMD64', 'X86_64'):
            stack_regs = ['rsp', 'rbp']
            bp_reg = 'rbp'
            sp_reg = 'rsp'
        elif arch.name == 'X86':
            stack_regs = ['esp', 'ebp']
            bp_reg = 'ebp'
            sp_reg = 'esp'
        elif arch.name.startswith('ARM'):
            stack_regs = ['sp']
            bp_reg = None
            sp_reg = 'sp'
        else:
            stack_regs = []
            bp_reg = None
            sp_reg = None

        # First, concretize SP if needed (we need a concrete SP for BP default)
        sp_val = None
        if sp_reg:
            try:
                reg_val = getattr(state.regs, sp_reg)
                if reg_val.symbolic:
                    sp_val = state.solver.eval(reg_val)
                    state.solver.add(reg_val == sp_val)
                    setattr(state.regs, sp_reg, sp_val)
                    l.debug(f"Concretized {sp_reg} to 0x{sp_val:x} (constraint added)")
                else:
                    sp_val = state.solver.eval(reg_val)
            except Exception as e:
                l.debug(f"Could not concretize {sp_reg}: {e}")

        # Then concretize BP - always use solver.eval() to get the same value
        # that Python already used to calculate addresses. This ensures
        # Rust gets consistent register values with what Python set up.
        if bp_reg:
            try:
                reg_val = getattr(state.regs, bp_reg)
                if reg_val.symbolic:
                    # Use existing solver evaluation (respects any prior concretization)
                    concrete_val = state.solver.eval(reg_val)
                    setattr(state.regs, bp_reg, concrete_val)
                    l.debug(f"Concretized {bp_reg} to 0x{concrete_val:x}")
            except Exception as e:
                l.debug(f"Could not concretize {bp_reg}: {e}")

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

    def _init_callback_history(self, state: "angr.SimState", event: "_ExplorationEvent"):
        """Initialize history for callback state to prevent IndexError.

        Python hooks often access `state.history.recent_bbl_addrs[-1]` which
        causes IndexError if history is empty. This method initializes history
        from the Rust pending state.

        Args:
            state: The angr callback state to initialize.
            event: The exploration event that triggered the callback.
        """
        # Use cached bundle data if available (avoids 2 extra FFI calls)
        rust_history = getattr(state.scratch, '_rust_bundle_history', None)
        rust_jumpkind = getattr(state.scratch, '_rust_bundle_jumpkind', None)
        if rust_history is None:
            try:
                rust_history = self._rust_mgr.get_pending_history()
                rust_jumpkind = self._rust_mgr.get_pending_jumpkind()
            except Exception as e:
                l.debug(f"Could not get Rust history: {e}")
                rust_history = []
                rust_jumpkind = "Ijk_Boring"
        if rust_jumpkind is None:
            rust_jumpkind = "Ijk_Boring"

        # Ensure history has at least one entry (the callback address)
        callback_addr = event.callback_addr or 0

        # Directly set recent_bbl_addrs to prevent IndexError
        # This is the critical fix - hooks access state.history.recent_bbl_addrs[-1]
        if hasattr(state.history, 'recent_bbl_addrs'):
            # Use Rust history if available, otherwise use callback address
            # P1 Fix: Always ensure history has at least one entry, even if callback_addr is 0
            if rust_history:
                state.history.recent_bbl_addrs = list(rust_history)
            else:
                # Use callback_addr, or state.addr if callback_addr is None/0
                addr_to_use = callback_addr if callback_addr else (state.addr if state.addr else 0)
                state.history.recent_bbl_addrs = [addr_to_use]

        # Set jumpkind to avoid callstack._manage() pushing a new frame
        # Ijk_Boring prevents the callstack from being modified
        try:
            state.history.jumpkind = rust_jumpkind
        except Exception:
            pass

        l.debug(f"Initialized callback history with {len(state.history.recent_bbl_addrs)} entries, "
                f"jumpkind={rust_jumpkind}, addr=0x{callback_addr:x}")

    def _init_callback_callstack(self, state: "angr.SimState", event: "_ExplorationEvent"):
        """Initialize callstack for SimProcedure continuations.

        SimProcedures may need procedure_data for continuations (e.g., when
        using self.call()). This method initializes the required data.

        P1 fix: Check for stored procedure_data from a previous self.call() and
        restore it. This is critical for continuations like __libc_start_main
        which call init/fini functions and expect to resume with saved args.

        Args:
            state: The angr callback state to initialize.
            event: The exploration event that triggered the callback.
        """
        if event.callback_reason != 'simprocedure':
            return

        addr = event.callback_addr
        # Ensure consistent int type for lookup
        addr_int = int(addr) if addr is not None else None

        # P1 fix: Check for stored procedure_data from a previous self.call()
        # This restores the full context (arguments, local vars) for continuations
        if addr_int in self._pending_procedure_data:
            stored_data = self._pending_procedure_data.get(addr_int)
            try:
                if hasattr(state.callstack, 'top') and state.callstack.top is not None:
                    state.callstack.top.procedure_data = stored_data
                    l.debug(f"Restored procedure_data for continuation at 0x{addr:x}: "
                            f"args={len(stored_data[1]) if len(stored_data) > 1 else 0}")
                    return
            except Exception as e:
                l.debug(f"Could not restore procedure_data: {e}")

        # Fallback: Get SP from Rust for saved state
        try:
            sp_val = self._rust_mgr.get_pending_register('rsp')
            if sp_val is None:
                sp_val = self._rust_mgr.get_pending_register('esp')
            saved_sp = sp_val or 0
        except Exception:
            saved_sp = 0

        # Initialize procedure_data for continuations
        # This prevents crashes when SimProcedures use self.call()
        try:
            if hasattr(state.callstack, 'top') and state.callstack.top is not None:
                # Set procedure_data: (saved_sp, sim_args, saved_local_vars, saved_lr, ideal_addr)
                state.callstack.top.procedure_data = (
                    saved_sp,              # saved_sp
                    [],                    # sim_args (populated by SimProcedure)
                    [],                    # saved_local_vars
                    None,                  # saved_lr
                    addr,                  # ideal_addr
                )
                l.debug(f"Initialized callstack procedure_data at 0x{addr:x}")
        except Exception as e:
            l.debug(f"Could not initialize callstack procedure_data: {e}")

    def _install_rust_solver_on_callback_state(self, state: "angr.SimState"):
        """Make the Rust solver the single source of truth for callback states.

        Instead of syncing constraints from Rust to Python (which can create
        UNSAT due to variable identity mismatches across the FFI boundary),
        this method monkey-patches the Python state's solver to delegate all
        solving operations to the forked Rust solver context.

        This covers:
        - state.solver.eval() — used by SimProcedures and concretization strategies
        - state.solver.satisfiable() — used by concretization and feasibility checks
        - state.solver.min()/max() — used by concretization strategies
        - state.solver.eval_upto() — used by concretization strategies
        - state.solver.add() — forwards constraints to both Python and Rust
        """
        rust_ctx = getattr(state.scratch, 'rust_solver_ctx', None)
        if rust_ctx is None:
            # No Rust solver available, fall back to constraint sync
            self._sync_rust_constraints_to_python(state)
            return

        original_eval = state.solver.eval
        original_satisfiable = state.solver.satisfiable
        original_min = state.solver.min
        original_max = state.solver.max
        original_eval_upto = state.solver.eval_upto
        original_add = state.solver.add

        def _rust_eval(expr, cast_to=None, **kwargs):
            kwargs.pop('exact', None)
            extra = kwargs.pop('extra_constraints', ())
            try:
                if extra:
                    rust_ctx.push()
                    try:
                        for c in extra:
                            rust_ctx.add_constraint_ast(c)
                        result = rust_ctx.eval(expr)
                    finally:
                        rust_ctx.pop()
                else:
                    result = rust_ctx.eval(expr)
                if result is None:
                    raise claripy.errors.UnsatError("UNSAT in Rust solver")
                if cast_to == bytes:
                    nbytes = (expr.length + 7) // 8
                    return result.to_bytes(nbytes, 'big')
                return result
            except claripy.errors.UnsatError:
                raise
            except Exception:
                return original_eval(expr, cast_to=cast_to, **kwargs)

        def _rust_satisfiable(**kwargs):
            kwargs.pop('exact', None)
            extra = kwargs.pop('extra_constraints', ())
            try:
                if extra:
                    rust_ctx.push()
                    try:
                        for c in extra:
                            rust_ctx.add_constraint_ast(c)
                        return rust_ctx.satisfiable()
                    finally:
                        rust_ctx.pop()
                return rust_ctx.satisfiable()
            except Exception:
                return original_satisfiable(**kwargs)

        def _rust_min(expr, **kwargs):
            kwargs.pop('exact', None)
            kwargs.pop('extra_constraints', None)
            kwargs.pop('signed', None)
            try:
                return rust_ctx.min(expr, signed=False)
            except Exception:
                return original_min(expr, **kwargs)

        def _rust_max(expr, **kwargs):
            kwargs.pop('exact', None)
            kwargs.pop('extra_constraints', None)
            kwargs.pop('signed', None)
            try:
                return rust_ctx.max(expr, signed=False)
            except Exception:
                return original_max(expr, **kwargs)

        def _rust_eval_upto(expr, n, cast_to=None, **kwargs):
            kwargs.pop('exact', None)
            extra = kwargs.pop('extra_constraints', ())
            try:
                if extra:
                    rust_ctx.push()
                    try:
                        for c in extra:
                            rust_ctx.add_constraint_ast(c)
                        results = rust_ctx.eval_upto(expr, n)
                    finally:
                        rust_ctx.pop()
                else:
                    results = rust_ctx.eval_upto(expr, n)
                if cast_to is not None:
                    results = tuple(cast_to(r) for r in results)
                return results
            except Exception:
                return original_eval_upto(expr, n, cast_to=cast_to, **kwargs)

        def _rust_add(*constraints):
            # Forward to both Rust and Python solvers
            for c in constraints:
                try:
                    rust_ctx.add_constraint_ast(c)
                except Exception:
                    pass
            original_add(*constraints)

        state.solver.eval = _rust_eval
        state.solver.satisfiable = _rust_satisfiable
        state.solver.min = _rust_min
        state.solver.max = _rust_max
        state.solver.eval_upto = _rust_eval_upto
        state.solver.add = _rust_add

        l.debug(f"Installed Rust solver delegation on callback state "
                f"({rust_ctx.num_constraints()} Rust constraints)")

    def _sync_rust_constraints_to_python(self, state: "angr.SimState"):
        """Sync constraints from Rust solver to Python state.

        This exports constraints from the Rust pending state and adds them
        to the Python state's solver. This ensures Python hooks see the
        same constraint context as Rust.

        Phase 2 Fix: Enhanced to export assumed path constraints (condition == true/false)
        not just stored branch conditions. This ensures Python's claripy solver has
        all the path constraints that Rust accumulated during exploration.

        Note: This relies on export_pending_constraints() which converts
        Rust constraints back to claripy ASTs.
        """
        try:
            # Check if export_pending_constraints is available
            if not hasattr(self._rust_mgr, 'export_pending_constraints'):
                l.debug("export_pending_constraints not available, skipping constraint sync")
                return

            constraints = self._rust_mgr.export_pending_constraints()
            synced = 0
            skipped = 0

            # Get existing constraint hashes to avoid duplicates
            existing_hashes = set()
            try:
                for c in state.solver.constraints:
                    existing_hashes.add(hash(c))
            except Exception:
                pass  # If we can't get existing constraints, add all

            for ast in constraints:
                if ast is not None:
                    try:
                        # Phase 2 Fix: Skip duplicate constraints
                        ast_hash = hash(ast)
                        if ast_hash in existing_hashes:
                            skipped += 1
                            continue

                        # Convert BV constraints to Bool for Python's Z3 backend.
                        # Rust's assume_true produces 1-bit BV constraints that
                        # Z3 can't cast to Bool directly.
                        if getattr(ast, 'length', None) is not None:
                            state.solver.add(ast != 0)
                        else:
                            state.solver.add(ast)
                        existing_hashes.add(ast_hash)
                        synced += 1
                    except Exception as e:
                        l.debug(f"Could not add constraint: {e}")

            if synced > 0 or skipped > 0:
                l.debug(f"Synced {synced} constraints from Rust to Python state ({skipped} duplicates skipped)")

        except Exception as e:
            l.debug(f"Could not sync Rust constraints: {e}")

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

        # Track address -> AST mapping for state export recovery (P1 fix)
        # P5 fix: Use effective state ID to ensure forked states share parent's data
        if addr is not None and state_id is not None:
            effective_id = self._get_effective_state_id(state_id)
            if effective_id is not None:
                if effective_id not in self._addr_to_ast:
                    self._addr_to_ast[effective_id] = {}
                actual_size = size if size is not None else (ast.length // 8 if hasattr(ast, 'length') else 1)
                self._addr_to_ast[effective_id][addr] = (ast, actual_size)

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

    def _cleanup_symbolic_pages_cache(self):
        """Enforce the symbolic pages cache size limit.

        Removes oldest entries when cache exceeds _max_symbolic_pages_cache.
        Also cleans up _addr_to_ast to prevent memory leaks.
        """
        if len(self._symbolic_pages) <= self._max_symbolic_pages_cache:
            return

        # Find entries to remove (oldest first)
        to_remove = []
        for state_id in list(self._symbolic_pages.keys()):
            to_remove.append(state_id)
            if len(self._symbolic_pages) - len(to_remove) <= self._max_symbolic_pages_cache:
                break

        # Remove old entries
        for state_id in to_remove:
            del self._symbolic_pages[state_id]
            # Also clean up address tracking for this state (P1 fix)
            if state_id in self._addr_to_ast:
                del self._addr_to_ast[state_id]
            # Mark symbols as inactive for GC
            self._identity_tracker.mark_inactive(state_id)

        if to_remove:
            l.debug(f"Cleaned up {len(to_remove)} symbolic page cache entries")

    def _inject_rust_stdout(self, state, state_id):
        """Inject Rust-side stdout buffer into the state's posix stdout plugin.

        Native puts/printf write to a per-state stdout_buffer in Rust. This
        method fetches that buffer and writes it into the Python state's posix
        stdout so that predicates calling state.posix.dumps(1) see the output.
        """
        try:
            rust_stdout = self._rust_mgr.get_state_stdout(state_id)
        except Exception:
            return
        if not rust_stdout:
            return
        posix = getattr(state, 'posix', None)
        if posix is None:
            return
        stdout = getattr(posix, 'stdout', None)
        if stdout is None:
            return
        try:
            stdout.write(None, claripy.BVV(bytes(rust_stdout)), events=False)
        except Exception as e:
            l.debug("Failed to inject Rust stdout into posix: %s", e)

    def _evaluate_predicates_on_active(self):
        """Evaluate callable find/avoid predicates on cached Python states.

        When find/avoid are callables (not addresses), the Rust engine can't
        evaluate them. Only evaluates states not yet checked (tracked via
        _evaluated_state_ids). New forked states get evaluated; already-checked
        states are skipped.
        """
        if not self._state_cache:
            return

        if not hasattr(self, '_evaluated_state_ids'):
            self._evaluated_state_ids = set()

        found_sids = set()
        avoid_sids = set()

        for state_id, state in list(self._state_cache.items()):
            if state_id in self._evaluated_state_ids:
                continue
            self._evaluated_state_ids.add(state_id)

            try:
                self._restore_plugins_to_state(state, state_id)
                self._inject_rust_stdout(state, state_id)

                if self._find_predicate is not None:
                    try:
                        if self._find_predicate(state):
                            found_sids.add(state_id)
                            continue
                    except Exception:
                        pass

                if self._avoid_predicate is not None:
                    try:
                        if self._avoid_predicate(state):
                            avoid_sids.add(state_id)
                    except Exception:
                        pass
            except Exception:
                pass

        # Move matched states to found/avoid stashes
        for sid in found_sids:
            for stash in ('active', 'deadended'):
                try:
                    if self._rust_mgr.move_state(sid, stash, 'found'):
                        break
                except Exception:
                    pass
            # Even if not in any Rust stash, record in Python-side found
            if not hasattr(self, '_predicate_found'):
                self._predicate_found = []
            if sid in self._state_cache:
                self._predicate_found.append(self._state_cache[sid])

        for sid in avoid_sids:
            for stash in ('active', 'deadended'):
                try:
                    self._rust_mgr.move_state(sid, stash, 'avoid')
                except Exception:
                    pass

    def _cleanup_state_cache(self):
        """Enforce the state cache size limit.

        Removes oldest entries when cache exceeds _max_state_cache_size.
        """
        if len(self._state_cache) <= self._max_state_cache_size:
            return

        # Find entries to remove (oldest first)
        to_remove = []
        for state_id in list(self._state_cache.keys()):
            to_remove.append(state_id)
            if len(self._state_cache) - len(to_remove) <= self._max_state_cache_size:
                break

        # Remove old entries
        for state_id in to_remove:
            del self._state_cache[state_id]
            # Also clean up symbolic pages
            self._symbolic_pages.pop(state_id, None)
            self._identity_tracker.mark_inactive(state_id)

        if to_remove:
            l.debug(f"Cleaned up {len(to_remove)} state cache entries")

    def _cleanup_state_refs(self, state_id: int):
        """Clean up references for a state that is no longer needed.

        Call this when a state is moved to deadended/errored stash.
        """
        # Remove from state cache
        self._state_cache.pop(state_id, None)
        # Remove from symbolic pages cache
        self._symbolic_pages.pop(state_id, None)
        # Mark symbols as inactive
        self._identity_tracker.mark_inactive(state_id)

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

    def _handle_simprocedure_callback(self, event: "_ExplorationEvent"):
        """Handle SimProcedure callback from Rust.

        This handles the full SimProcedure lifecycle:
        1. Creates angr state from cached state (preserving symbolic memory)
        2. Executes the SimProcedure
        3. Syncs state changes back to Rust
        4. Handles multiple successors (forked states)
        5. Updates state cache for future callbacks

        Special handling for hooks with length=0:
        - These hooks don't replace code, they just run before the instruction
        - After the hook, we need to execute the instruction at the hook address
        - But we must NOT re-trigger the hook (which would cause infinite loop)
        - Solution: Tell Rust to skip the hook for this address on next step
        """
        _sp_total_start = time.perf_counter_ns()
        addr = event.callback_addr
        name = event.callback_name
        state_id = event.callback_state_id

        # Track current callback state ID for symbolic memory preservation
        self._current_callback_state_id = state_id

        # Handle internal passthrough - this is internal binary code, just continue execution
        if name == "__internal_passthrough__":
            l.debug(f"Internal passthrough at 0x{addr:x} - continuing execution")
            # Resume execution at this address, no SimProcedure to run
            self._rust_mgr.resume_after_simprocedure(addr, None, None)
            self._perf_stats['callback_simprocedure_count'] += 1
            self._perf_stats['callback_simprocedure_total_ns'] += time.perf_counter_ns() - _sp_total_start
            return

        # Find the SimProcedure - first by address, then by name
        proc = self._project._sim_procedures.get(addr)
        if proc is None and name:
            # Address lookup failed (e.g., PLT address) - try to find by name
            for proc_addr, candidate in self._project._sim_procedures.items():
                proc_name = candidate.__class__.__name__ if hasattr(candidate, '__class__') else str(candidate)
                if proc_name == name:
                    proc = candidate
                    l.debug(f"Found SimProcedure {name} at 0x{proc_addr:x} (callback was at 0x{addr:x})")
                    break
        if proc is None:
            l.warning(f"SimProcedure not found at 0x{addr:x} (name={name})")
            self._rust_mgr.resume_after_simprocedure(addr + 1, None, None)
            self._perf_stats['callback_simprocedure_count'] += 1
            self._perf_stats['callback_simprocedure_total_ns'] += time.perf_counter_ns() - _sp_total_start
            return

        # Defensive type check
        if isinstance(proc, (list, tuple)):
            l.error(f"Invalid SimProcedure at 0x{addr:x}: got {type(proc).__name__}, expected callable")
            self._rust_mgr.resume_after_simprocedure(addr + 1, None, None)
            self._perf_stats['callback_simprocedure_count'] += 1
            self._perf_stats['callback_simprocedure_total_ns'] += time.perf_counter_ns() - _sp_total_start
            return

        # Get hook length - this determines if the hook replaces code
        hook_length = getattr(proc, 'kwargs', {}).get('length', 0)
        if hook_length == 0:
            hook_length = getattr(proc, 'length', 0)
        is_zero_length_hook = (hook_length == 0)

        # Create angr state for the SimProcedure (uses cached state if available)
        _sp_state_create_start = time.perf_counter_ns()
        state = self._create_state_for_callback(event)
        if state is None:
            l.warning(f"Could not create state for SimProcedure at 0x{addr:x}")
            self._rust_mgr.resume_after_simprocedure(addr + 1, None, None)
            self._set_callback_state(None)
            self._current_callback_state_id = None
            self._perf_stats['callback_simprocedure_state_create_ns'] += time.perf_counter_ns() - _sp_state_create_start
            self._perf_stats['callback_simprocedure_count'] += 1
            self._perf_stats['callback_simprocedure_total_ns'] += time.perf_counter_ns() - _sp_total_start
            return
        self._perf_stats['callback_simprocedure_state_create_ns'] += time.perf_counter_ns() - _sp_state_create_start

        # Save original state for extracting changes after SimProcedure execution.
        # Copy the state for any SimProcedure that writes to memory, so
        # changed_bytes() can detect modifications. The CallbackMemoryTracker
        # also captures writes, but changed_bytes() serves as a backup.
        # Zero-length hooks (UserHook) may also write memory (e.g., storing
        # known bytes before a function call), so always copy for those.
        writes_memory = is_zero_length_hook or name in (
            'read', 'recv', 'fgets', 'scanf', '__isoc99_scanf',
            'fread', 'gets', 'getchar', 'fgetc', 'getc',
            'strncpy', 'strcpy', 'memcpy', 'memmove', 'memset',
            'strcat', 'strncat', 'sprintf', 'snprintf')
        orig_state = state.copy() if writes_memory else state

        # GAP 2 fix: Save original constraint COUNT before hook execution.
        # Building set(constraints) is expensive (~6ms per call with many constraints).
        # Save just the count; only build the full set if count changes after callback.
        orig_constraint_count = len(state.solver.constraints) if hasattr(state, 'solver') else 0
        orig_constraints = None  # Deferred — only built if needed

        # GAP 5: Track memory writes during callback execution
        memory_tracker = CallbackMemoryTracker(state)

        # Run the SimProcedure
        _sp_execute_start = time.perf_counter_ns()
        try:
            from angr.engines.successors import SimSuccessors

            # Create SimSuccessors object for the procedure
            successors = SimSuccessors(addr=addr, initial_state=state)

            # Execute the procedure with memory tracking (GAP 5)
            with memory_tracker:
                proc_instance = proc
                if hasattr(proc, 'run'):
                    proc.execute(state, successors)
                else:
                    # It's a class, instantiate it
                    proc_instance = proc()
                    proc_instance.execute(state, successors)
            self._perf_stats['callback_simprocedure_execute_ns'] += time.perf_counter_ns() - _sp_execute_start

            # Get tracked memory writes from callback execution
            tracked_writes = memory_tracker.get_writes()
            tracked_symbolic_writes = memory_tracker.get_symbolic_writes()
            if tracked_writes:
                l.debug(f"Tracked {len(tracked_writes)} memory writes during callback")
            if tracked_symbolic_writes:
                l.debug(f"Tracked {len(tracked_symbolic_writes)} symbolic memory writes during callback")

            # Handle successors - this includes sync back to Rust
            _sp_sync_start = time.perf_counter_ns()
            all_succs = successors.all_successors

            # P1 fix: Capture procedure_data from successors that use self.call()
            # When a SimProcedure uses self.call() to invoke another function,
            # it stores arguments and continuation info in procedure_data.
            # We capture this here so we can restore it when the continuation runs.
            for succ in all_succs:
                try:
                    cs = succ.callstack
                    top = cs.top if cs else None
                    if top is None:
                        continue

                    # P1 fix: When jumpkind is Ijk_Call, add_successor pushes a NEW callstack frame.
                    # The procedure_data is on the PREVIOUS frame (the caller's frame).
                    # Check both top and top.next for procedure_data.
                    frames_to_check = [top]
                    if hasattr(top, 'next') and top.next is not None:
                        frames_to_check.append(top.next)

                    for frame in frames_to_check:
                        pdata = getattr(frame, 'procedure_data', None)
                        if pdata is not None and len(pdata) >= 5:
                            cont_addr = pdata[4]  # ideal_addr is continuation address
                            # Convert to int for consistent key type
                            if hasattr(cont_addr, 'concrete'):
                                cont_addr_int = int(cont_addr)
                            elif isinstance(cont_addr, int):
                                cont_addr_int = cont_addr
                            else:
                                continue
                            if cont_addr_int != addr:
                                self._pending_procedure_data[cont_addr_int] = pdata
                                l.debug(f"Stored procedure_data for continuation at 0x{cont_addr_int:x}")

                                # C1 Fix: Register continuation hook with Rust IMMEDIATELY
                                # This prevents "Cannot execute external address" errors
                                # when Rust tries to execute the continuation before the
                                # normal hook sync happens via _sync_hooks_before_step()
                                if cont_addr_int not in self._registered_hooks:
                                    cont_proc = self._project._sim_procedures.get(cont_addr_int)
                                    if cont_proc:
                                        cont_name = cont_proc.__class__.__name__ if hasattr(cont_proc, '__class__') else str(cont_proc)
                                        cont_num_args = getattr(cont_proc, 'num_args', 0) or 0
                                        cont_no_return = getattr(cont_proc, 'NO_RET', False)
                                        self._rust_mgr.register_simprocedures([(cont_addr_int, cont_name, cont_num_args, cont_no_return)])
                                        self._registered_hooks.add(cont_addr_int)
                                        l.debug(f"Immediately registered continuation hook at 0x{cont_addr_int:x}: {cont_name}")
                except Exception as e:
                    l.debug(f"Could not capture procedure_data: {e}")

            if all_succs:
                # Check for no-return procedures (exit, abort, etc.)
                # Only deadend if the procedure has NO continuations
                # (__libc_start_main has NO_RET but uses self.call() for continuations)
                proc_no_ret = getattr(proc, 'NO_RET', False) if proc else False
                # Only deadend for explicit termination procedures.
                # Exclude internal angr procedures that have NO_RET for other reasons.
                no_ret_names = {'exit', '_exit', 'abort', '__stack_chk_fail'}
                if proc_no_ret and name in no_ret_names:
                    # Deadend the state by resuming at address 0
                    l.debug(f"No-return procedure {name} with successors — deadending")
                    self._rust_mgr.resume_after_simprocedure(0, None, None)
                    self._set_callback_state(None)
                    self._current_callback_state_id = None
                    return

                # First successor continues in Rust
                first_succ = all_succs[0]

                # When a SimProcedure uses self.call() (Ijk_Call jumpkind),
                # the continuation address is in the callstack but NOT on the
                # stack memory. Push it so the Rust engine's `ret` instruction
                # can find it when the called function returns.
                if first_succ.history.jumpkind == 'Ijk_Call':
                    try:
                        # Find continuation address from pending_procedure_data
                        cont_addr = None
                        for cont_a, pdata in self._pending_procedure_data.items():
                            if cont_a != addr:  # Not the current address
                                cont_addr = cont_a
                                break
                        if cont_addr is not None:
                            sp = first_succ.solver.eval(first_succ.regs._sp)
                            ptr_size = first_succ.arch.bytes
                            # Push continuation address: decrement SP and store
                            new_sp = sp - ptr_size
                            first_succ.regs._sp = new_sp
                            first_succ.memory.store(new_sp,
                                claripy.BVV(cont_addr, ptr_size * 8),
                                endness='Iend_LE')
                            l.debug(f"Pushed continuation addr 0x{cont_addr:x} to stack at 0x{new_sp:x}")
                    except Exception as e:
                        l.debug(f"Could not push continuation addr: {e}")

                # For zero-length hooks, if successor has same address as hook,
                # the hook just modifies state and we should continue execution
                # at the same address WITHOUT re-triggering the hook
                #
                # Note: Check symbolic IP before accessing .addr to prevent
                # SimValueError when IP has multiple possible values
                succ_ip_symbolic = first_succ.regs._ip.symbolic
                succ_addr_matches = (not succ_ip_symbolic and first_succ.addr == addr)
                if is_zero_length_hook and succ_addr_matches:
                    self._resume_with_state(first_succ, orig_state, event, skip_hook_addr=addr,
                                           tracked_writes=tracked_writes,
                                           tracked_symbolic_writes=tracked_symbolic_writes,
                                           orig_constraints=orig_constraints,
                                           orig_constraint_count=orig_constraint_count)
                elif succ_ip_symbolic:
                    self._resume_with_state(first_succ, orig_state, event,
                                           tracked_writes=tracked_writes,
                                           tracked_symbolic_writes=tracked_symbolic_writes,
                                           orig_constraints=orig_constraints,
                                           orig_constraint_count=orig_constraint_count)
                else:
                    self._resume_with_state(first_succ, orig_state, event,
                                           tracked_writes=tracked_writes,
                                           tracked_symbolic_writes=tracked_symbolic_writes,
                                           orig_constraints=orig_constraints,
                                           orig_constraint_count=orig_constraint_count)

                # Additional successors are added as new active states
                for succ in all_succs[1:]:
                    self._add_forked_state(succ, event)

            else:
                # No successors — check if this is a known terminal procedure.
                # Only deadend for specific terminal names (not all NO_RET),
                # because UserHook also has NO_RET=True but should continue
                # execution at the same address for zero-length hooks.
                # CallReturn is the terminal hook used by factory.callable().
                no_ret_terminal = name in ('exit', '_exit', 'abort', '__stack_chk_fail', 'CallReturn')
                if no_ret_terminal:
                    l.debug(f"Terminal procedure {name} — deadending state at 0x{addr:x}")
                    if is_zero_length_hook:
                        # For zero-length terminal hooks (e.g., CallReturn at callable's
                        # return address), skip the hook and resume at addr. The Rust
                        # engine can't lift code there, so it deadends with PC=addr.
                        # This preserves the PC for code that checks state.addr.
                        try:
                            self._rust_mgr.set_skip_hook_addr(addr)
                        except Exception:
                            pass
                        self._rust_mgr.resume_after_simprocedure(addr, None, None)
                    else:
                        self._rust_mgr.resume_after_simprocedure(0, None, None)
                elif is_zero_length_hook:
                    # Tell Rust to skip the hook and execute from addr
                    # GAP 2: Pass original constraints for constraint sync
                    # GAP 5: Pass tracked writes for memory sync
                    self._resume_with_skip_hook(addr, state, orig_state, event, orig_constraints,
                                               tracked_writes=tracked_writes,
                                               tracked_symbolic_writes=tracked_symbolic_writes,
                                               orig_constraint_count=orig_constraint_count)
                else:
                    # Non-zero-length, non-terminal: check NO_RET as fallback
                    proc_no_ret = getattr(proc, 'NO_RET', False)
                    if proc_no_ret:
                        l.debug(f"No-return procedure {name} — deadending state")
                        self._rust_mgr.resume_after_simprocedure(0, None, None)
                    else:
                        ret_addr = event.callback_return_addr or (addr + 1)
                        for sym_addr, ast in (tracked_symbolic_writes or []):
                            try:
                                self._rust_mgr.import_symbolic_memory(sym_addr, ast)
                            except Exception:
                                pass
                        self._rust_mgr.resume_after_simprocedure(ret_addr, None, tracked_writes or None)

        except TypeError as e:
            # Continuation procedure missing local_vars (e.g., after_main without args).
            # This happens when the init phase's procedure_data wasn't captured.
            # Treat as a graceful exit (deadend) rather than a hard error.
            if 'missing' in str(e) and 'positional argument' in str(e):
                l.warning(f"Continuation at 0x{addr:x} missing args (likely after_main) — deadending")
                try:
                    self._rust_mgr.resume_after_simprocedure(0, None, None, None)
                except Exception:
                    try:
                        self._rust_mgr.resume_after_error(str(e))
                    except Exception:
                        pass
                self._set_callback_state(None)
                self._current_callback_state_id = None
                self._perf_stats['callback_simprocedure_count'] += 1
                self._perf_stats['callback_simprocedure_total_ns'] += time.perf_counter_ns() - _sp_total_start
                return
            # Other TypeErrors fall through to generic handler
            raise
        except Exception as e:
            # P17: On exception, move state to errored stash instead of resuming with corrupted state
            l.warning(f"P17: SimProcedure execution error at 0x{addr:x}: {e}")
            import traceback
            traceback.print_exc()
            # Signal error to Rust - this will move the state to errored stash
            try:
                self._rust_mgr.resume_after_error(str(e))
            except Exception as resume_err:
                l.warning(f"P17: Could not signal error to Rust: {resume_err}")
            # P19: Clear callback state since we've handled the error
            self._set_callback_state(None)
            self._current_callback_state_id = None
            self._perf_stats['callback_simprocedure_count'] += 1
            self._perf_stats['callback_simprocedure_total_ns'] += time.perf_counter_ns() - _sp_total_start
            return
        # P19: Only clear callback state on success, not in finally
        # This ensures state isn't lost if resume fails
        self._set_callback_state(None)
        self._current_callback_state_id = None
        self._perf_stats['callback_simprocedure_sync_back_ns'] += time.perf_counter_ns() - _sp_sync_start
        self._perf_stats['callback_simprocedure_count'] += 1
        self._perf_stats['callback_simprocedure_total_ns'] += time.perf_counter_ns() - _sp_total_start

    def _resume_with_state(
        self,
        succ_state: "angr.SimState",
        orig_state: "angr.SimState",
        event: "_ExplorationEvent",
        skip_hook_addr: Optional[int] = None,
        tracked_writes: Optional[list] = None,
        tracked_symbolic_writes: Optional[list] = None,
        orig_constraints: Optional[set] = None,
        orig_constraint_count: Optional[int] = None
    ):
        """Resume Rust execution with a successor state.

        Extracts register, memory, and constraint changes, syncs them to Rust,
        and updates the state cache for future callbacks.

        Args:
            succ_state: The successor state after callback execution.
            orig_state: The original state before callback.
            event: The exploration event that triggered the callback.
            skip_hook_addr: If set, tells Rust to skip the hook at this address
                           for the next step (prevents infinite loops with
                           zero-length hooks).
            tracked_writes: GAP 5 - Memory writes tracked during callback execution.
            tracked_symbolic_writes: List of (addr, ast) for symbolic memory imports.
        """
        # Handle symbolic IP: pick first concrete solution if symbolic
        if succ_state.regs._ip.symbolic:
            try:
                new_pc = succ_state.solver.eval_one(succ_state.regs._ip)
            except Exception:
                # Multiple solutions or other error - pick any valid one
                new_pc = succ_state.solver.eval(succ_state.regs._ip)
        else:
            new_pc = succ_state.addr

        # Extract changes
        reg_changes = self._extract_register_changes(orig_state, succ_state)
        mem_changes, symbolic_imports = self._extract_memory_changes(orig_state, succ_state)

        # GAP 5: Merge tracked writes with extracted memory changes
        if tracked_writes:
            existing_addrs = {addr for addr, _ in mem_changes}
            for addr, data in tracked_writes:
                if addr not in existing_addrs:
                    mem_changes.append((addr, data))
                    existing_addrs.add(addr)
            l.debug(f"Merged {len(tracked_writes)} tracked writes with memory changes")

        # Collect all symbolic addresses to exclude from concrete memory changes
        # This prevents apply_changes from overwriting symbolic imports with concrete values
        all_symbolic_addrs = {addr for addr, _ in symbolic_imports}
        if tracked_symbolic_writes:
            all_symbolic_addrs.update(addr for addr, _ in tracked_symbolic_writes)

        # Filter out symbolic addresses from mem_changes
        if all_symbolic_addrs:
            mem_changes = [(addr, data) for addr, data in mem_changes if addr not in all_symbolic_addrs]

        # Extract any new constraints added during callback
        new_constraints = self._extract_new_constraints(orig_state, succ_state,
                                                        orig_constraints=orig_constraints,
                                                        orig_constraint_count=orig_constraint_count)
        if new_constraints:
            l.debug(f"Extracted {len(new_constraints)} new constraints from callback")

        # For zero-length hooks, tell Rust to skip the hook on next step
        if skip_hook_addr is not None:
            try:
                self._rust_mgr.set_skip_hook_addr(skip_hook_addr)
                l.debug(f"Set skip_hook_addr to 0x{skip_hook_addr:x}")
            except Exception as e:
                l.debug(f"Could not set skip_hook_addr: {e}")

        # Resume Rust with the changes and any new constraints.
        # IMPORTANT: This must happen BEFORE symbolic imports, because
        # apply_changes writes concrete data which clears symbolic page markers.
        # Importing symbolic values after resume re-sets the markers correctly.
        self._rust_mgr.resume_after_simprocedure(
            new_pc, reg_changes, mem_changes or None, new_constraints or None
        )

        # Import symbolic memory to Rust AFTER resume.
        # The resume's apply_changes writes concrete witnesses which clear
        # symbolic page markers. Re-importing symbolic values here restores
        # the markers and stores the symbolic objects for VEX loads.
        # Use import_symbolic_to_state (by state_id) since pending_callback
        # was consumed by resume_after_simprocedure.
        state_id = event.callback_state_id
        all_sym_imports = list(symbolic_imports)
        if tracked_symbolic_writes:
            existing_sym_addrs = {addr for addr, _ in symbolic_imports}
            for addr, ast in tracked_symbolic_writes:
                if addr not in existing_sym_addrs:
                    all_sym_imports.append((addr, ast))

        for addr, ast in all_sym_imports:
            try:
                self._rust_mgr.import_symbolic_to_state(state_id, addr, ast)
                l.debug(f"Imported symbolic memory at 0x{addr:x} to state {state_id}")
            except Exception as e:
                l.debug(f"Could not import symbolic memory at 0x{addr:x}: {e}")

        # Update cache for future callbacks
        state_id = event.callback_state_id
        if state_id is not None:
            self._state_cache[state_id] = succ_state

    def _resume_with_skip_hook(
        self,
        addr: int,
        state: "angr.SimState",
        orig_state: "angr.SimState",
        event: "_ExplorationEvent",
        orig_constraints: Optional[set] = None,
        tracked_writes: Optional[list] = None,
        tracked_symbolic_writes: Optional[list] = None,
        orig_constraint_count: Optional[int] = None
    ):
        """Resume Rust execution after a zero-length hook with no successors.

        This handles hooks that modify state in-place without creating successor
        states. We need to extract all state changes and sync them to Rust.

        Args:
            addr: Original hook address (used for skip_hook).
            state: The modified state after hook execution.
            orig_state: The original state before hook execution.
            event: The exploration event.
            orig_constraints: Original constraints before callback (for constraint sync).
            tracked_writes: Memory writes tracked during callback execution.
            tracked_symbolic_writes: List of (addr, ast) for symbolic memory imports.
        """
        # Tell Rust to skip the hook at this address for the next step
        # This prevents infinite loops when resuming at a zero-length hook address
        try:
            self._rust_mgr.set_skip_hook_addr(addr)
            l.debug(f"Set skip_hook_addr to 0x{addr:x}")
        except Exception as e:
            l.debug(f"Could not set skip_hook_addr: {e}")

        # Extract actual next PC from the modified state
        # This handles hooks that manually set the return address (e.g., pop ret simulation)
        if state.regs._ip.symbolic:
            try:
                new_pc = state.solver.eval_one(state.regs._ip)
            except Exception:
                try:
                    new_pc = state.solver.eval(state.regs._ip)
                except Exception:
                    new_pc = addr  # Fallback to original address
        else:
            new_pc = state.addr

        if new_pc != addr:
            l.debug(f"Hook modified IP from 0x{addr:x} to 0x{new_pc:x}")

        # Extract register changes between original and modified state
        # This syncs all register modifications made by the hook
        reg_changes = self._extract_register_changes(orig_state, state)

        # Extract memory changes between original and modified state
        mem_changes, symbolic_imports = self._extract_memory_changes(orig_state, state)

        # Merge tracked writes with extracted memory changes
        # Tracked writes capture symbolic stores that _extract_memory_changes might miss
        if tracked_writes:
            existing_addrs = {write_addr for write_addr, _ in mem_changes} if mem_changes else set()
            for write_addr, data in tracked_writes:
                if write_addr not in existing_addrs:
                    mem_changes.append((write_addr, data))
                    existing_addrs.add(write_addr)
            l.debug(f"Merged {len(tracked_writes)} tracked writes with memory changes")

        # Collect all symbolic addresses to exclude from concrete memory changes
        # This prevents apply_changes from overwriting symbolic imports with concrete values
        all_symbolic_addrs = {sym_addr for sym_addr, _ in symbolic_imports}
        if tracked_symbolic_writes:
            all_symbolic_addrs.update(sym_addr for sym_addr, _ in tracked_symbolic_writes)

        # Filter out symbolic addresses from mem_changes
        if all_symbolic_addrs:
            mem_changes = [(write_addr, data) for write_addr, data in mem_changes if write_addr not in all_symbolic_addrs]

        # Extract new constraints added during hook execution
        new_constraints = None
        if hasattr(state, 'solver'):
            try:
                current_count = len(state.solver.constraints)
                # Fast path: if count unchanged, skip expensive set construction
                if orig_constraint_count is not None and current_count == orig_constraint_count:
                    pass  # No new constraints
                elif orig_constraints is not None:
                    current_constraints = set(state.solver.constraints)
                    new_constraints = list(current_constraints - orig_constraints)
                    if new_constraints:
                        l.debug(f"Extracted {len(new_constraints)} constraints from hook")
                elif orig_constraint_count is not None:
                    # Count changed — need full diff
                    prior = set(orig_state.solver.constraints)
                    current_constraints = set(state.solver.constraints)
                    new_constraints = list(current_constraints - prior)
                    if new_constraints:
                        l.debug(f"Extracted {len(new_constraints)} constraints from hook")
            except Exception as e:
                l.debug(f"Could not extract constraints: {e}")

        # Resume Rust with all extracted changes.
        # IMPORTANT: This must happen BEFORE symbolic imports (same as _resume_with_state).
        self._rust_mgr.resume_after_simprocedure(
            new_pc,
            reg_changes or None,
            mem_changes or None,
            new_constraints or None
        )

        # Import symbolic memory AFTER resume (see _resume_with_state for rationale)
        state_id = event.callback_state_id
        all_sym_imports = list(symbolic_imports)
        if tracked_symbolic_writes:
            existing_sym_addrs = {sym_addr for sym_addr, _ in symbolic_imports}
            for sym_addr, ast in tracked_symbolic_writes:
                if sym_addr not in existing_sym_addrs:
                    all_sym_imports.append((sym_addr, ast))

        for sym_addr, ast in all_sym_imports:
            try:
                self._rust_mgr.import_symbolic_to_state(state_id, sym_addr, ast)
                l.debug(f"Imported symbolic memory at 0x{sym_addr:x} to state {state_id}")
            except Exception as e:
                l.debug(f"Could not import symbolic memory at 0x{sym_addr:x}: {e}")

        # Update cache with modified state for future callbacks
        if state_id is not None:
            self._state_cache[state_id] = state

    def _extract_new_constraints(
        self,
        orig_state: "angr.SimState",
        new_state: "angr.SimState",
        orig_constraints: Optional[set] = None,
        orig_constraint_count: Optional[int] = None
    ) -> list:
        """Extract constraints added during callback execution.

        Compares constraint sets between original and successor states,
        returning any new constraints that were added during the callback.

        Args:
            orig_constraints: Pre-captured set of constraints from before the
                callback. Use this instead of orig_state.solver.constraints
                when orig_state may alias the successor (no copy was made).
            orig_constraint_count: Fast-path: just the count before callback.
                If count hasn't changed, skip expensive set construction.
        """
        new_constraints = []

        # Extract constraints from Python's claripy solver
        try:
            new_constraint_list = new_state.solver.constraints

            # Fast path: if constraint count hasn't changed, no new constraints
            # were added. Skip expensive set construction.
            if orig_constraint_count is not None and orig_constraints is None:
                if len(new_constraint_list) == orig_constraint_count:
                    # No new constraints — skip the expensive set diff
                    pass
                else:
                    # Count changed — build sets and diff
                    prior = set(orig_state.solver.constraints)
                    state_constraints = set(new_constraint_list)
                    python_added = state_constraints - prior
                    if python_added:
                        l.debug(f"Callback added {len(python_added)} new Python constraints")
                        new_constraints.extend(python_added)
            elif orig_constraints is not None:
                prior = orig_constraints
                state_constraints = set(new_constraint_list)
                python_added = state_constraints - prior
                if python_added:
                    l.debug(f"Callback added {len(python_added)} new Python constraints")
                    new_constraints.extend(python_added)
            else:
                prior = set(orig_state.solver.constraints)
                state_constraints = set(new_constraint_list)
                python_added = state_constraints - prior
                if python_added:
                    l.debug(f"Callback added {len(python_added)} new Python constraints")
                    new_constraints.extend(python_added)
        except Exception as e:
            l.debug(f"Could not extract Python constraints: {e}")

        if new_constraints:
            l.debug(f"Total new constraints to sync: {len(new_constraints)}")

        return new_constraints

    def _add_forked_state(self, succ_state: "angr.SimState", event: "_ExplorationEvent"):
        """Add a forked state from a SimProcedure to Rust active stash.

        When a SimProcedure creates multiple successors (e.g., fork on
        symbolic condition), additional states are added to Rust for
        continued exploration.

        This method properly inherits the solver context from the pending
        state to ensure constraints are preserved across forks.
        """
        # Try to fork the pending solver context for this state
        # This ensures the forked state inherits all constraints
        try:
            forked_solver = self._rust_mgr.fork_pending_solver()
            # Attach forked solver to the state
            succ_state.scratch.rust_solver_ctx = forked_solver
            l.debug(f"Forked solver context for additional successor "
                    f"({forked_solver.num_constraints()} constraints)")
        except Exception as e:
            l.debug(f"Could not fork solver for additional successor: {e}")

        # Extract any constraints specific to this fork path
        # These may differ from the main successor due to branching conditions
        fork_constraints = []
        if hasattr(succ_state, 'scratch') and hasattr(succ_state.scratch, 'rust_solver_ctx'):
            try:
                # If the state has path-specific constraints, extract them
                if hasattr(succ_state.solver, 'constraints'):
                    fork_constraints = list(succ_state.solver.constraints)
            except Exception:
                pass

        # Create a new Rust state for this successor
        self._add_rust_state('active', succ_state)

        # If we have fork-specific constraints, sync them to the new Rust state
        if fork_constraints:
            try:
                # The state was just added, so sync constraints to the pending/active state
                self._rust_mgr.add_constraints_to_pending(fork_constraints)
                l.debug(f"Synced {len(fork_constraints)} fork constraints to Rust state")
            except Exception as e:
                l.debug(f"Could not sync fork constraints: {e}")

        l.debug(f"Added forked state at PC 0x{succ_state.addr:x}")

    def _handle_syscall_callback(self, event: "_ExplorationEvent"):
        """Handle syscall callback from Rust.

        Uses cached state to preserve symbolic memory and constraints,
        then syncs changes back to Rust after syscall execution.
        """
        syscall_num = event.callback_syscall_num
        state_id = event.callback_state_id

        # Track current callback state ID for symbolic memory preservation
        self._current_callback_state_id = state_id

        # Create a state for syscall handling (uses cached state if available)
        state = self._create_state_for_callback(event)
        if state is None:
            l.warning(f"Could not create state for syscall {syscall_num}")
            # Continue after syscall
            pc = self._rust_mgr.get_pending_register('rip') or 0
            self._rust_mgr.resume_after_syscall(pc + 1, None, None)
            return

        # Run the syscall
        try:
            # Get syscall handler from project
            engine = self._project.factory.default_engine

            # Execute syscall
            successors = engine.process(state, procedure=None)

            all_succs = successors.all_successors
            if all_succs:
                # First successor continues in Rust
                succ_state = all_succs[0]

                # Handle symbolic IP: pick first concrete solution if symbolic
                if succ_state.regs._ip.symbolic:
                    try:
                        new_pc = succ_state.solver.eval_one(succ_state.regs._ip)
                    except claripy.errors.ClaripyError:
                        new_pc = succ_state.solver.eval(succ_state.regs._ip)
                else:
                    new_pc = succ_state.addr

                reg_changes = self._extract_register_changes(state, succ_state)
                mem_changes = self._extract_memory_changes(state, succ_state)
                new_constraints = self._extract_new_constraints(state, succ_state)

                self._rust_mgr.resume_after_syscall(
                    new_pc, reg_changes, mem_changes, new_constraints or None
                )

                # Update cache
                state_id = event.callback_state_id
                if state_id is not None:
                    self._state_cache[state_id] = succ_state

                # Handle additional successors
                for succ in all_succs[1:]:
                    self._add_forked_state(succ, event)
            else:
                pc = state.addr + 1
                self._rust_mgr.resume_after_syscall(pc, None, None)

        except Exception as e:
            l.warning(f"Syscall execution error: {e}")
            pc = state.addr + 1
            self._rust_mgr.resume_after_syscall(pc, None, None)
        finally:
            # Clear callback state to avoid stale references
            self._set_callback_state(None)
            self._current_callback_state_id = None

    def _handle_find_predicate_callback(self, event: "_ExplorationEvent"):
        """Handle callable find predicate evaluation callback from Rust.

        P2 fix: When find is a callable (lambda/function), Rust cannot evaluate
        it directly. This handler creates an angr state and evaluates the
        predicate, then tells Rust whether the state matched.

        Args:
            event: The exploration event from Rust.
        """
        state_id = event.callback_state_id
        addr = event.callback_addr

        # Track current callback state ID
        self._current_callback_state_id = state_id

        try:
            # Create angr state for predicate evaluation
            state = self._create_state_for_callback(event)
            if state is None:
                l.warning(f"Could not create state for find predicate at 0x{addr:x}")
                self._rust_mgr.resume_find_predicate(False)
                return

            # Evaluate the find predicate
            if self._find_predicate is None:
                l.warning("Find predicate callback but no predicate stored")
                self._rust_mgr.resume_find_predicate(False)
                return

            try:
                result = self._find_predicate(state)
                matched = bool(result) if result is not None else False
                l.debug(f"Find predicate at 0x{addr:x} returned: {matched}")
            except Exception as e:
                l.warning(f"Find predicate evaluation error at 0x{addr:x}: {e}")
                matched = False

            # Tell Rust the result
            self._rust_mgr.resume_find_predicate(matched)

            # If matched, update state cache for later retrieval
            if matched and state_id is not None:
                self._state_cache[state_id] = state

        except Exception as e:
            l.warning(f"Find predicate callback error: {e}")
            self._rust_mgr.resume_find_predicate(False)
        finally:
            # Clear callback state to avoid stale references
            self._set_callback_state(None)
            self._current_callback_state_id = None

    def _handle_avoid_predicate_callback(self, event: "_ExplorationEvent"):
        """Handle callable avoid predicate evaluation callback from Rust.

        P7 fix: When avoid is a callable (lambda/function), Rust cannot evaluate
        it directly. This handler creates an angr state and evaluates the
        predicate, then tells Rust whether the state should be avoided.

        Args:
            event: The exploration event from Rust.
        """
        state_id = event.callback_state_id
        addr = event.callback_addr

        # Track current callback state ID
        self._current_callback_state_id = state_id

        try:
            # Create angr state for predicate evaluation
            state = self._create_state_for_callback(event)
            if state is None:
                l.warning(f"Could not create state for avoid predicate at 0x{addr:x}")
                self._rust_mgr.resume_avoid_predicate(False)
                return

            # Evaluate the avoid predicate
            if self._avoid_predicate is None:
                l.warning("Avoid predicate callback but no predicate stored")
                self._rust_mgr.resume_avoid_predicate(False)
                return

            try:
                result = self._avoid_predicate(state)
                matched = bool(result) if result is not None else False
                l.debug(f"Avoid predicate at 0x{addr:x} returned: {matched}")
            except Exception as e:
                l.warning(f"Avoid predicate evaluation error at 0x{addr:x}: {e}")
                matched = False

            # Tell Rust the result
            self._rust_mgr.resume_avoid_predicate(matched)

        except Exception as e:
            l.warning(f"Avoid predicate callback error: {e}")
            self._rust_mgr.resume_avoid_predicate(False)
        finally:
            # Clear callback state to avoid stale references
            self._set_callback_state(None)
            self._current_callback_state_id = None

    def _handle_symbolic_branch_callback(self, event: "_ExplorationEvent"):
        """Handle symbolic branch callback from Rust.

        When a symbolic branch with both paths feasible is encountered,
        Rust returns to Python for proper state forking with constraints.
        This ensures symbolic branches are handled correctly even when
        hooks/callbacks occur, preventing lost forks.

        The handler:
        1. Gets the branch condition from Rust
        2. Creates two forked states with appropriate constraints
        3. Adds both states back to Rust's active stash
        """
        true_target = event.branch_true_target
        false_target = event.branch_false_target
        condition_id = event.branch_condition_id
        state_id = event.callback_state_id

        l.debug(f"Handling symbolic branch: true=0x{true_target:x}, false=0x{false_target:x}, "
                f"cond_id={condition_id}")

        try:
            # Get the branch condition from Rust as a claripy AST
            condition = self._rust_mgr.get_pending_branch_condition()

            l.debug(f"Got branch condition from Rust: {condition}")

            # Defensive: ensure condition is a claripy AST, not Python bool/int
            # This can happen if rustbv_to_claripy() returns the wrong type
            if isinstance(condition, (bool, int)):
                l.warning(f"Branch condition is {type(condition).__name__}, wrapping to claripy")
                import claripy
                condition = claripy.BoolV(bool(condition))

            # Create true branch constraint: condition != 0 (condition is true)
            # For a VEX guard, "true" means the guard evaluates to non-zero
            true_constraint = condition != 0

            # Create false branch constraint: condition == 0 (condition is false)
            false_constraint = condition == 0

            l.debug(f"True constraint: {true_constraint}")
            l.debug(f"False constraint: {false_constraint}")

            # Resume Rust with the forked states
            # Pass constraints as lists for each branch
            # Get active state IDs before fork
            ids_before = set(self._rust_mgr.get_state_ids('active'))

            self._rust_mgr.resume_after_symbolic_branch(
                true_target,
                false_target,
                [true_constraint],
                [false_constraint],
            )

            # Cache Python states for new forked Rust states
            ids_after = set(self._rust_mgr.get_state_ids('active'))
            new_ids = ids_after - ids_before
            if new_ids and state_id in self._state_cache:
                parent_state = self._state_cache[state_id]
                for new_id in new_ids:
                    forked_state = parent_state.copy()
                    self._state_cache[new_id] = forked_state
                    # Track lineage for plugin restoration
                    root = self._state_roots.get(state_id, state_id)
                    self._state_roots[new_id] = root

            l.debug(f"Resumed after symbolic branch with {len(new_ids)} forked states")

        except Exception as e:
            l.warning(f"Symbolic branch handling error: {e}")
            import traceback
            l.warning(traceback.format_exc())

            # Fallback: just fork without proper constraints
            # This is less accurate but at least continues exploration
            try:
                self._rust_mgr.resume_after_symbolic_branch(
                    true_target,
                    false_target,
                    None,
                    None,
                )
                l.warning("Resumed after symbolic branch with fallback (no constraints)")
            except Exception as e2:
                l.error(f"Failed to resume after symbolic branch: {e2}")
                # Recovery: Move the pending state to errored stash to avoid hanging
                # This uses the same error handling as other callback failures (P17 fix)
                try:
                    self._rust_mgr.resume_after_error(f"symbolic_branch_error: {e2}")
                    l.warning("Moved state to errored stash after symbolic branch failure")
                except Exception as e3:
                    l.error(f"Failed to move state to errored stash: {e3}")
                    # Last resort: the pending_callback is still set, which will cause
                    # step() to fail on the next iteration. This is better than silently
                    # losing the state or hanging indefinitely.

    def _get_pending_parent_id(self) -> Optional[int]:
        """Get parent state ID of current pending callback state.

        When Rust forks a state during exploration, the forked state gets a
        new state_id but the Python caches only have the parent's state_id.
        This method retrieves the parent_id from the pending state snapshot
        so we can look up the parent's caches.

        Returns:
            The parent state ID, or None if not available.
        """
        try:
            snapshot = self._rust_mgr.export_pending_state()
            return snapshot.parent_id
        except Exception:
            return None

    def _get_pending_root_state_id(self) -> Optional[int]:
        """Get root state ID for the pending callback state.

        When Rust forks states internally (multi-level forks), Python only has
        cached data for the original state that was added via Python. This method
        returns the root state ID (the original ancestor) for any forked descendant.

        Returns:
            The root state ID if available, or None if not tracked.
        """
        try:
            return self._rust_mgr.get_pending_root_state_id()
        except Exception:
            return None

    def _get_effective_state_id(self, state_id: Optional[int]) -> Optional[int]:
        """Get effective state ID following lineage for lookups.

        P5 fix: When a state is forked in Rust, its ID changes but Python's caches
        are keyed by the original state ID. This method follows the lineage chain
        to find a state ID that exists in our caches.

        Args:
            state_id: The current state ID to look up.

        Returns:
            The effective state ID (either the original or an ancestor that's cached).
        """
        if state_id is None:
            return None

        # Fast path: direct hit
        if state_id in self._state_cache:
            return state_id

        # Try root state from pending callback (most common case for forked states)
        try:
            root_id = self._get_pending_root_state_id()
            if root_id is not None and root_id in self._state_cache:
                return root_id
        except Exception:
            pass

        # Walk full ancestry chain from pending callback
        try:
            ancestry = self._get_pending_ancestry()
            for ancestor_id in ancestry:
                if ancestor_id in self._state_cache:
                    return ancestor_id
        except Exception:
            pass

        # No cached ancestor found, return original
        return state_id

    def _get_pending_ancestry(self) -> list:
        """Get full ancestry chain for the pending callback state.

        Returns a list of state IDs: [state_id, parent_id, grandparent_id, ...].
        This allows Python to find cached state data even for multi-level forks.

        Returns:
            List of state IDs in the ancestry chain.
        """
        try:
            return self._rust_mgr.get_pending_ancestry()
        except Exception:
            # Fallback to single parent lookup
            parent_id = self._get_pending_parent_id()
            return [parent_id] if parent_id is not None else []

    def _create_state_for_callback(self, event: "_ExplorationEvent") -> Optional["angr.SimState"]:
        """Create an angr state from the pending Rust state.

        This method uses the cached angr state (if available) to preserve
        symbolic memory and constraints, then syncs concrete register values
        from the Rust pending state.

        CRITICAL: We fork the Rust solver context to ensure callbacks inherit
        all constraints accumulated during Rust exploration. Without this,
        callbacks would create fresh solver contexts, leading to incorrect
        symbolic evaluation.

        Also initializes:
        - History (to prevent IndexError on state.history.recent_bbl_addrs[-1])
        - Callstack procedure_data (for SimProcedure continuations)
        - Callback state tracking (for memory access during hooks)
        """
        state_id = event.callback_state_id
        state = None

        # Use cached state if available - this preserves symbolic memory
        # For forked states, follow the ancestry chain to find cached state
        cached_state = None
        lookup_state_id = state_id

        if state_id is not None and state_id in self._state_cache:
            cached_state = self._state_cache[state_id]
            lookup_state_id = state_id
            # Fix 1B: Validate cached state has required attributes
            if cached_state is not None:
                if not hasattr(cached_state, 'solver') or not hasattr(cached_state, 'memory'):
                    l.warning(f"Cached state {state_id} invalid type: {type(cached_state)}, clearing")
                    del self._state_cache[state_id]
                    cached_state = None
        elif state_id is not None:
            # Forked state - try root state first (most likely to be cached)
            root_id = self._get_pending_root_state_id()
            if root_id is not None and root_id in self._state_cache:
                cached_state = self._state_cache[root_id]
                lookup_state_id = root_id
                l.debug(f"Using root state {root_id} cache for forked state {state_id}")
            else:
                # Walk full ancestry chain to find any cached ancestor
                for ancestor_id in self._get_pending_ancestry():
                    if ancestor_id in self._state_cache:
                        cached_state = self._state_cache[ancestor_id]
                        lookup_state_id = ancestor_id
                        l.debug(f"Using ancestor {ancestor_id} cache for forked state {state_id}")
                        break

        if cached_state is not None:
            self._stats_cache_hits += 1
            # Copy when callable predicates need stdout history, skip otherwise
            has_predicates = (getattr(self, '_find_predicate', None) is not None or
                             getattr(self, '_avoid_predicate', None) is not None)
            if has_predicates:
                self._stats_state_creations += 1
                state = cached_state.copy()
            else:
                state = cached_state

            # Use the callback bundle API to get registers + solver + history
            # in a single FFI call instead of ~20 individual calls.
            try:
                arch = self._project.arch
                reg_names = self._get_arch_register_names(arch)
                bundle = self._rust_mgr.export_callback_bundle(reg_names)

                # Apply solver from bundle
                forked_solver = bundle['solver']
                state.scratch.rust_solver_ctx = forked_solver
                constraint_count = bundle['constraint_count']
                l.debug(f"Bundle: solver with {constraint_count} constraints for state {state_id}")

                # Apply registers from bundle (batch)
                registers = bundle['registers']
                for reg_name, val in registers.items():
                    try:
                        if val is not None:
                            setattr(state.regs, reg_name, claripy.BVV(val, arch.bits))
                        else:
                            # Symbolic register — fetch AST individually
                            try:
                                ast = self._rust_mgr.get_pending_register_ast(reg_name)
                                if ast is not None:
                                    setattr(state.regs, reg_name, ast)
                            except Exception:
                                pass
                    except Exception:
                        pass

                # Cache history and jumpkind from bundle for later use
                state.scratch._rust_bundle_history = bundle.get('history', [])
                state.scratch._rust_bundle_jumpkind = bundle.get('jumpkind', 'Ijk_Boring')
            except Exception as e:
                l.debug(f"Bundle API failed, falling back to individual calls: {e}")
                # Fallback to individual calls
                try:
                    forked_solver = self._rust_mgr.fork_pending_solver()
                    state.scratch.rust_solver_ctx = forked_solver
                except Exception as e2:
                    l.warning(f"Could not fork solver context: {e2}")
                self._sync_registers_from_rust_pending(state)

            # Install memory proxy: wrap state.memory.load to check Rust
            # memory first for addresses that the Python state doesn't have
            # (stack frames created during VEX execution).
            self._install_rust_memory_proxy(state)

            # Restore symbolic memory regions that may have been lost during
            # Rust execution fallback. This is critical for callbacks that need
            # to read symbolic values from memory (e.g., symbolic buffer access)
            # Use lookup_state_id to find cached pages (handles forked states)
            self._restore_symbolic_pages(state, lookup_state_id)

            # Also restore any hook-tracked symbolic memory
            # This handles symbolic memory written by previous hooks that needs
            # to be preserved across multiple callbacks
            self._restore_hook_symbolic_memory(state, lookup_state_id)

            l.debug(f"Using cached state {state_id} for callback (lookup_id={lookup_state_id})")
        else:
            # Fallback to blank state (original behavior)
            self._stats_cache_misses += 1
            self._stats_state_creations += 1
            l.warning(f"No cached state for ID {state_id}, using blank state fallback")
            state = self._create_blank_state_fallback(event)

        if state is not None:
            # Initialize history to prevent IndexError in hooks
            self._init_callback_history(state, event)

            # Initialize callstack for SimProcedure continuations
            self._init_callback_callstack(state, event)

            # Set callback state for memory access during this callback
            self._set_callback_state(state)

            # Make the Rust solver the single source of truth for this callback.
            # Instead of syncing constraints (which can create UNSAT due to
            # variable identity mismatches), delegate solver operations to the
            # forked Rust solver context.
            self._install_rust_solver_on_callback_state(state)

            # Phase 3 Fix: Ensure critical plugins are present
            # Some scripts assume posix/libc plugins exist - restore if missing
            self._ensure_critical_plugins(state, event.callback_state_id)

        return state

    def _ensure_critical_plugins(self, state: "angr.SimState", state_id: Optional[int]):
        """Ensure critical plugins are present on the state.

        Phase 3 Fix: Some scripts and SimProcedures expect plugins like
        posix and libc to be present. If they're missing after state copy/creation,
        restore them from a template state.

        Args:
            state: The state to check/fix.
            state_id: The state ID for root state lookup.
        """
        # Check if critical plugins are missing
        missing_plugins = []
        for plugin_name in ['posix', 'libc', 'heap']:
            if not hasattr(state, plugin_name) or getattr(state, plugin_name) is None:
                missing_plugins.append(plugin_name)

        if not missing_plugins:
            return  # All plugins present

        # Find template state for plugin restoration
        template = None
        if state_id is not None:
            root_id = self._state_roots.get(state_id, state_id)
            if root_id in self._state_cache:
                template = self._state_cache[root_id]

        # Fall back to any cached state
        if template is None and self._state_cache:
            template = next(iter(self._state_cache.values()))

        if template is None:
            l.debug(f"Phase 3: No template for plugin restoration, missing: {missing_plugins}")
            return

        # Restore missing plugins
        for plugin_name in missing_plugins:
            try:
                if hasattr(template, plugin_name):
                    plugin = getattr(template, plugin_name)
                    if plugin is not None and hasattr(plugin, 'copy'):
                        state.register_plugin(plugin_name, plugin.copy())
                        l.debug(f"Phase 3: Restored {plugin_name} plugin")
            except Exception as e:
                l.debug(f"Phase 3: Could not restore {plugin_name}: {e}")

    def _create_blank_state_fallback(self, event: "_ExplorationEvent") -> Optional["angr.SimState"]:
        """Create a blank state as fallback when no cached state is available."""
        try:
            # Create a blank state
            state = self._project.factory.blank_state()

            # Fork the Rust solver context even for blank states
            try:
                forked_solver = self._rust_mgr.fork_pending_solver()
                state.scratch.rust_solver_ctx = forked_solver
                l.debug(f"Forked Rust solver for blank fallback state "
                        f"({forked_solver.num_constraints()} constraints)")
            except Exception as e:
                l.debug(f"Could not fork solver for blank state: {e}")

            # Copy registers from pending Rust state
            self._sync_registers_from_rust_pending(state)

            return state

        except Exception as e:
            l.warning(f"Error creating blank state for callback: {e}")
            return None

    def _sync_registers_from_rust_pending(self, state: "angr.SimState"):
        """Sync register values from Rust pending state to angr state.

        Handles both concrete and symbolic registers. Concrete values are
        set directly. Symbolic values are converted from Rust Z3 BVs to
        claripy ASTs via rustbv_to_claripy, preserving symbolic identity.
        """
        arch = self._project.arch
        reg_names = self._get_arch_register_names(arch)

        for reg_name in reg_names:
            try:
                # Try concrete first (fast path)
                val = self._rust_mgr.get_pending_register(reg_name)
                if val is not None:
                    setattr(state.regs, reg_name, claripy.BVV(val, arch.bits))
                else:
                    # Register is symbolic — convert to claripy AST
                    try:
                        ast = self._rust_mgr.get_pending_register_ast(reg_name)
                        if ast is not None:
                            setattr(state.regs, reg_name, ast)
                    except Exception:
                        pass  # Skip if conversion fails
            except Exception:
                pass

    def _get_arch_register_names(self, arch) -> list:
        """Get register names for an architecture."""
        if arch.name in ('AMD64', 'X86_64'):
            return ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi',
                    'rbp', 'rsp', 'r8', 'r9', 'r10', 'r11',
                    'r12', 'r13', 'r14', 'r15', 'rip']
        elif arch.name == 'X86':
            return ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi',
                    'ebp', 'esp', 'eip']
        elif arch.name.startswith('ARM'):
            return ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7',
                    'r8', 'r9', 'r10', 'r11', 'r12', 'sp', 'lr', 'pc']
        else:
            return []

    def _extract_register_changes(
        self,
        old_state: "angr.SimState",
        new_state: "angr.SimState"
    ) -> list:
        """Extract register changes between states.

        Handles both concrete and symbolic register values. For symbolic
        values (especially return registers like RAX), the value is converted
        to a claripy AST and stored in Rust's pending symbolic state.

        Returns:
            List of (offset, size, data_bytes) tuples for concrete changes.
        """
        changes = []
        arch = old_state.arch

        # Architecture-specific register offset maps
        # Offsets verified against native/angr/src/arch/*.rs
        if arch.name in ('AMD64', 'X86_64'):
            reg_map = {
                'rax': (16, 8), 'rcx': (24, 8), 'rdx': (32, 8), 'rbx': (40, 8),
                'rsp': (48, 8), 'rbp': (56, 8), 'rsi': (64, 8), 'rdi': (72, 8),
                'r8': (80, 8), 'r9': (88, 8), 'r10': (96, 8), 'r11': (104, 8),
                'r12': (112, 8), 'r13': (120, 8), 'r14': (128, 8), 'r15': (136, 8),
                'rip': (184, 8),
            }
            return_regs = {'rax'}
        elif arch.name == 'X86':
            # X86 32-bit (verified: native/angr/src/arch/x86.rs)
            reg_map = {
                'eax': (8, 4), 'ecx': (12, 4), 'edx': (16, 4), 'ebx': (20, 4),
                'esp': (24, 4), 'ebp': (28, 4), 'esi': (32, 4), 'edi': (36, 4),
                'eip': (68, 4),
            }
            return_regs = {'eax'}
        elif arch.name in ('ARMEL', 'ARMHF', 'ARM'):
            # ARM 32-bit (verified: native/angr/src/arch/arm.rs)
            reg_map = {
                'r0': (8, 4), 'r1': (12, 4), 'r2': (16, 4), 'r3': (20, 4),
                'r4': (24, 4), 'r5': (28, 4), 'r6': (32, 4), 'r7': (36, 4),
                'r8': (40, 4), 'r9': (44, 4), 'r10': (48, 4), 'r11': (52, 4),
                'r12': (56, 4), 'sp': (60, 4), 'lr': (64, 4), 'pc': (68, 4),
            }
            return_regs = {'r0'}
        elif arch.name == 'AARCH64':
            # ARM 64-bit - not yet implemented
            l.warning(f"AARCH64 register extraction not yet implemented")
            return []
        else:
            l.warning(f"Unknown architecture {arch.name} for register extraction")
            return []

        # Return register is critical - always sync it

        for reg_name, (offset, size) in reg_map.items():
            try:
                old_val = getattr(old_state.regs, reg_name)
                new_val = getattr(new_state.regs, reg_name)

                if new_val.symbolic:
                    # Symbolic register value - sync to Rust
                    if reg_name in return_regs:
                        try:
                            self._sync_symbolic_register_to_rust(reg_name, new_val)
                            l.debug(f"Synced symbolic return register {reg_name} to Rust")
                        except Exception as e:
                            l.debug(f"Could not sync symbolic {reg_name}: {e}")
                            try:
                                new_concrete = new_state.solver.eval(new_val)
                                data = new_concrete.to_bytes(size, 'little')
                                changes.append((offset, size, bytes(data)))
                            except Exception:
                                pass
                else:
                    new_concrete = new_state.solver.eval(new_val)
                    if old_val.symbolic or old_state.solver.eval(old_val) != new_concrete:
                        data = new_concrete.to_bytes(size, 'little')
                        changes.append((offset, size, bytes(data)))
            except Exception:
                pass

        return changes

    def _sync_symbolic_register_to_rust(self, reg_name: str, value):
        """Sync a symbolic register value to Rust pending state.

        Tries direct AST sync first (best approach), falls back to handle-based
        sync if the direct method is not available.
        """
        try:
            # Best approach: directly sync claripy AST to Rust
            if hasattr(self._rust_mgr, 'set_pending_register_symbolic_ast'):
                self._rust_mgr.set_pending_register_symbolic_ast(reg_name, value)
                l.debug(f"Synced symbolic register {reg_name} to Rust via AST")
                return

            # Fallback: use handle-based sync
            if hasattr(self._rust_mgr, 'claripy_ast_to_handle'):
                handle = self._rust_mgr.claripy_ast_to_handle(value)
                self._rust_mgr.set_pending_register_symbolic(reg_name, handle.id())
                l.debug(f"Synced symbolic register {reg_name} to Rust via handle")
                return

            # Final fallback: just register the handle for later retrieval
            handle_id = id(value)
            self._register_handle(handle_id, value)
            l.debug(f"Registered symbolic register {reg_name} handle for later retrieval")

        except Exception as e:
            l.debug(f"Could not sync symbolic {reg_name}: {e}")

    def _extract_memory_changes(
        self,
        old_state: "angr.SimState",
        new_state: "angr.SimState"
    ) -> tuple:
        """Extract memory changes between states for Rust sync.

        Uses angr's changed_bytes() to detect memory modifications,
        groups them into contiguous regions, and returns concrete values.
        Also tracks symbolic values for constraint propagation.

        Returns:
            Tuple of:
            - List of (addr, data_bytes) for concrete memory changes
            - List of (addr, ast) for symbolic memory that needs import
        """
        concrete_changes = []
        symbolic_imports = []  # Collect symbolic ASTs for import to Rust
        try:
            # Use angr's changed_bytes to find modifications
            changed = new_state.memory.changed_bytes(old_state.memory)
            if not changed:
                return [], []

            # Limit to prevent timeouts on large diffs (e.g., unconstrained fill)
            if len(changed) > 10000:
                l.debug(f"Too many changed bytes ({len(changed)}), truncating to 10000")
                changed = set(sorted(changed)[:10000])

            # Group consecutive changed bytes into regions
            for item in self._group_changed_bytes(new_state, changed):
                if item[0] == 'concrete':
                    _, start, size, data = item
                    concrete_changes.append((start, bytes(data)))
                elif item[0] == 'symbolic':
                    _, start, size, data, handle_id, ast = item
                    # Provide concrete witness for Rust memory sync
                    concrete_changes.append((start, bytes(data)))
                    # Cache symbolic value for later constraint sync
                    # The handle is already registered in _emit_memory_region
                    l.debug(f"Tracked symbolic memory change at 0x{start:x} (handle={handle_id})")

                    # Collect symbolic AST for import to Rust
                    symbolic_imports.append((start, ast))

                    # Track symbolic AST for state restoration during callbacks
                    # This is critical: hooks that copy symbolic memory would lose
                    # the symbolic relationship without this tracking
                    state_id = self._current_callback_state_id
                    if state_id is not None:
                        if state_id not in self._hook_symbolic_memory:
                            self._hook_symbolic_memory[state_id] = {}
                        self._hook_symbolic_memory[state_id][start] = (ast, size)
                        l.debug(f"Preserved symbolic memory at 0x{start:x} for state {state_id}")

        except Exception as e:
            l.debug(f"Error extracting memory changes: {e}")

        return concrete_changes, symbolic_imports

    def _group_changed_bytes(self, state: "angr.SimState", changed_addrs):
        """Group consecutive changed bytes into contiguous regions.

        Yields tuples from _emit_memory_region:
            ('concrete', start_addr, size, data_bytes) for concrete regions
            ('symbolic', start_addr, size, data_bytes, handle_id, ast) for symbolic regions
        """
        if not changed_addrs:
            return

        sorted_addrs = sorted(changed_addrs)
        start = sorted_addrs[0]
        end = start + 1

        for addr in sorted_addrs[1:]:
            if addr == end:
                # Contiguous with current region
                end += 1
            else:
                # Gap found - emit current region and start new one
                yield from self._emit_memory_region(state, start, end - start)
                start = addr
                end = addr + 1

        # Emit final region
        yield from self._emit_memory_region(state, start, end - start)

    def _emit_memory_region(self, state: "angr.SimState", start: int, size: int):
        """Emit a memory region with concrete bytes and optional symbolic info.

        Yields tuples with symbolic value info for constraint reconstruction:
        - ('concrete', start_addr, size, data_bytes) for concrete values
        - ('symbolic', start_addr, size, data_bytes, handle_id, ast) for symbolic values

        For symbolic regions, emits byte-by-byte to produce simple ASTs
        (individual BVS or Extract) that convert cleanly to RustBV. This
        avoids complex Concat trees from multi-byte loads that may fail
        in claripy_to_rustbv conversion.
        """
        # Limit region size to avoid memory issues
        MAX_REGION_SIZE = 4096
        if size > MAX_REGION_SIZE:
            for offset in range(0, size, MAX_REGION_SIZE):
                chunk_size = min(MAX_REGION_SIZE, size - offset)
                yield from self._emit_memory_region(state, start + offset, chunk_size)
            return

        try:
            val = state.memory.load(start, size, endness=state.arch.memory_endness)
            if not val.symbolic:
                concrete = state.solver.eval(val)
                data = concrete.to_bytes(size, 'little')
                yield ('concrete', start, size, data)
            else:
                # For small symbolic regions (typical SimProcedure writes),
                # emit byte-by-byte for simple ASTs. For large regions,
                # emit as one chunk to avoid 1000s of solver.eval() calls.
                if size > 128:
                    # Large region: emit whole (may produce complex AST)
                    handle_id = id(val)
                    self._register_handle(handle_id, val)
                    try:
                        concrete = state.solver.eval(val)
                        data = concrete.to_bytes(size, 'little')
                    except Exception:
                        data = bytes(size)
                    yield ('symbolic', start, size, data, handle_id, val)
                    return

                # Small region: byte-by-byte for simple ASTs
                for byte_offset in range(size):
                    byte_addr = start + byte_offset
                    try:
                        byte_val = state.memory.load(
                            byte_addr, 1, endness='Iend_BE',
                            inspect=False, disable_actions=True)
                        if byte_val.symbolic:
                            handle_id = id(byte_val)
                            self._register_handle(handle_id, byte_val)
                            try:
                                concrete_byte = state.solver.eval(byte_val)
                                data = bytes([concrete_byte & 0xff])
                            except Exception:
                                data = bytes(1)
                            yield ('symbolic', byte_addr, 1, data, handle_id, byte_val)
                        else:
                            try:
                                concrete_byte = state.solver.eval(byte_val)
                                data = bytes([concrete_byte & 0xff])
                            except Exception:
                                data = bytes(1)
                            yield ('concrete', byte_addr, 1, data)
                    except Exception:
                        yield ('concrete', byte_addr, 1, bytes(1))
        except Exception as e:
            l.debug(f"Error emitting memory region at 0x{start:x}: {e}")
            pass

    def _extract_symbolic_pages(self, state: "angr.SimState") -> dict:
        """Extract symbolic memory regions from an angr state.

        This identifies memory regions containing symbolic values and caches
        them for later restoration. This is critical for preserving symbolic
        memory when Rust falls back to Python callbacks.

        Args:
            state: The angr state to extract symbolic pages from.

        Returns:
            Dict mapping address -> claripy AST for symbolic memory locations.
        """
        symbolic_regions = {}
        try:
            # Strategy 1: Use angr's internal symbolic tracking if available
            # This is the most accurate method as angr tracks symbolic bytes precisely
            if hasattr(state.memory, 'get_symbolic_addrs'):
                try:
                    # get_symbolic_addrs returns addresses of symbolic bytes
                    symbolic_addrs = state.memory.get_symbolic_addrs()
                    if symbolic_addrs:
                        # Group contiguous symbolic regions
                        for addr in symbolic_addrs:
                            try:
                                # Load individual symbolic bytes
                                val = state.memory.load(addr, 1, endness=state.arch.memory_endness)
                                if hasattr(val, 'symbolic') and val.symbolic:
                                    symbolic_regions[addr] = val
                                    self._register_handle(id(val), val)
                            except Exception:
                                pass
                        if symbolic_regions:
                            l.debug(f"Extracted {len(symbolic_regions)} symbolic bytes via get_symbolic_addrs")
                            return symbolic_regions
                except Exception as e:
                    l.debug(f"get_symbolic_addrs failed: {e}")

            # Strategy 2: Scan pages for symbolic content
            # Note: _pages keys are page NUMBERS, not addresses
            if hasattr(state.memory, '_pages'):
                page_size = getattr(state.memory, 'page_size', 4096)

                for page_num in list(state.memory._pages.keys()):
                    page = state.memory._pages.get(page_num)
                    if page is None:
                        continue

                    page_addr = page_num * page_size  # Convert page number to address

                    # UltraPage: use all_bytes_changed_in_history() for written bytes,
                    # then check symbolic_bitmap to filter to symbolic ones
                    if hasattr(page, 'all_bytes_changed_in_history') and hasattr(page, 'symbolic_bitmap'):
                        try:
                            changed = page.all_bytes_changed_in_history()
                            sb = page.symbolic_bitmap
                            # changed is a SegmentList, iterate over Segment objects
                            for segment in changed:
                                # Segment has start/end attributes
                                start = getattr(segment, 'start', None)
                                end = getattr(segment, 'end', None)
                                if start is not None and end is not None:
                                    for offset in range(start, end):
                                        if offset < len(sb) and sb[offset]:
                                            addr = page_addr + offset
                                            try:
                                                val = state.memory.load(addr, 1, endness=state.arch.memory_endness)
                                                if hasattr(val, 'symbolic') and val.symbolic:
                                                    symbolic_regions[addr] = val
                                                    self._register_handle(id(val), val)
                                            except Exception:
                                                pass
                        except Exception:
                            pass
                    # ListPage: stored_offset tracks all written bytes
                    elif hasattr(page, 'stored_offset') and page.stored_offset:
                        for offset in page.stored_offset:
                            addr = page_addr + offset
                            try:
                                val = state.memory.load(addr, 1, endness=state.arch.memory_endness)
                                if hasattr(val, 'symbolic') and val.symbolic:
                                    symbolic_regions[addr] = val
                                    self._register_handle(id(val), val)
                            except Exception:
                                pass
                    # Fallback: Check alternative tracking attributes
                    elif hasattr(page, '_symbolic_bitmap') and page._symbolic_bitmap:
                        for offset in range(page_size):
                            if page._symbolic_bitmap.get(offset, False):
                                addr = page_addr + offset
                                try:
                                    val = state.memory.load(addr, 1, endness=state.arch.memory_endness)
                                    if hasattr(val, 'symbolic') and val.symbolic:
                                        symbolic_regions[addr] = val
                                        self._register_handle(id(val), val)
                                except Exception:
                                    pass
                    elif hasattr(page, 'symbolic_byte_map') and page.symbolic_byte_map:
                        for offset, sym_val in page.symbolic_byte_map.items():
                            addr = page_addr + offset
                            symbolic_regions[addr] = sym_val
                            self._register_handle(id(sym_val), sym_val)

            if symbolic_regions:
                l.debug(f"Extracted {len(symbolic_regions)} symbolic memory regions")

        except Exception as e:
            l.debug(f"Error extracting symbolic pages: {e}")
        return symbolic_regions

    def _install_rust_memory_proxy(self, state: "angr.SimState"):
        """Sync stack data from Rust to Python callback state.

        Loads the SP page in a single bulk FFI call (pending_memory_load_page)
        then writes non-zero pointer-sized values to the Python state.
        This is ~30x faster than 64 individual pending_memory_load calls.
        """
        try:
            sp = state.solver.eval(state.regs._sp) if not state.regs._sp.symbolic else None
            if not sp:
                return
            ptr_size = state.arch.bytes

            # Load the full page containing SP in ONE FFI call
            sp_page = sp & ~0xFFF
            sp_offset = sp - sp_page
            try:
                page_data = self._rust_mgr.pending_memory_load_page(sp_page)
                if page_data and len(page_data) == 0x1000:
                    # Write non-zero pointer-sized values from SP upward
                    # Covers 64 slots (~512 bytes on x64) for args + locals
                    end_offset = min(sp_offset + 64 * ptr_size, 0x1000)
                    for off in range(sp_offset, end_offset, ptr_size):
                        chunk = page_data[off:off + ptr_size]
                        if len(chunk) == ptr_size:
                            int_val = int.from_bytes(chunk, 'little')
                            if int_val != 0:
                                addr = sp_page + off
                                val = claripy.BVV(int_val, ptr_size * 8)
                                state.memory.store(addr, val, endness='Iend_LE',
                                                   inspect=False, disable_actions=True)
                    return
            except Exception:
                pass

            # Fallback: individual loads
            for i in range(16):
                addr = sp + i * ptr_size
                try:
                    data = self._rust_mgr.pending_memory_load(addr, ptr_size)
                    if data and len(data) == ptr_size:
                        int_val = int.from_bytes(data, 'little')
                        if int_val != 0:
                            val = claripy.BVV(int_val, ptr_size * 8)
                            state.memory.store(addr, val, endness='Iend_LE',
                                               inspect=False, disable_actions=True)
                except Exception:
                    pass
        except Exception:
            pass

    def _restore_symbolic_pages(self, state: "angr.SimState", state_id: int):
        """Restore symbolic memory regions to an angr state.

        This restores symbolic values that were previously extracted and
        cached, ensuring that Python callbacks see the correct symbolic
        memory context after fallback from Rust.

        For forked states, tries parent chain if direct lookup fails.

        Args:
            state: The angr state to restore symbolic pages to.
            state_id: The state ID to look up cached symbolic pages.
        """
        # Try direct lookup, then ancestry chain for forked states
        symbolic_pages = None
        if state_id in self._symbolic_pages:
            symbolic_pages = self._symbolic_pages[state_id]
        else:
            # Try root state first (most likely to have cached pages)
            root_id = self._get_pending_root_state_id()
            if root_id is not None and root_id in self._symbolic_pages:
                symbolic_pages = self._symbolic_pages[root_id]
                l.debug(f"Using root {root_id} symbolic pages for state {state_id}")
            else:
                # Walk full ancestry chain
                for ancestor_id in self._get_pending_ancestry():
                    if ancestor_id in self._symbolic_pages:
                        symbolic_pages = self._symbolic_pages[ancestor_id]
                        l.debug(f"Using ancestor {ancestor_id} symbolic pages for state {state_id}")
                        break

        if symbolic_pages is None:
            l.debug(f"No symbolic pages found for state {state_id} or ancestors")
            return
        restored_count = 0
        failed_count = 0

        # Group contiguous regions for more efficient restoration
        # This reduces the number of store operations
        sorted_addrs = sorted(symbolic_pages.keys())
        i = 0
        while i < len(sorted_addrs):
            start_addr = sorted_addrs[i]
            ast = symbolic_pages[start_addr]

            # Check for single-byte symbolic values (most common case after byte-granular extraction)
            if ast.length == 8:  # 8 bits = 1 byte
                try:
                    state.memory.store(start_addr, ast, endness=state.arch.memory_endness)
                    restored_count += 1
                    self._register_handle(id(ast), ast)
                except Exception as e:
                    l.debug(f"Error restoring symbolic byte at 0x{start_addr:x}: {e}")
                    failed_count += 1
                i += 1
            else:
                # Multi-byte symbolic value - store directly
                try:
                    state.memory.store(start_addr, ast, endness=state.arch.memory_endness)
                    restored_count += 1
                    self._register_handle(id(ast), ast)
                except Exception as e:
                    l.debug(f"Error restoring symbolic memory at 0x{start_addr:x}: {e}")
                    failed_count += 1
                i += 1

        if restored_count > 0:
            l.debug(f"Restored {restored_count} symbolic memory regions for state {state_id}")
        if failed_count > 0:
            l.warning(f"Failed to restore {failed_count} symbolic memory regions")

    def _restore_hook_symbolic_memory(self, state: "angr.SimState", state_id: int):
        """Restore symbolic memory that was tracked during hook execution.

        When hooks copy or manipulate symbolic memory, the symbolic ASTs are
        tracked in `_hook_symbolic_memory`. This method restores those ASTs
        to the callback state so subsequent operations preserve symbolic
        relationships.

        This is critical for examples like flareon2015_5 where hooks copy
        symbolic password bytes to new memory locations.

        For forked states, tries parent chain if direct lookup fails.

        Args:
            state: The angr state to restore symbolic memory to.
            state_id: The state ID to look up tracked symbolic memory.
        """
        # Try direct lookup, then ancestry chain for forked states
        hook_memory = None
        if state_id in self._hook_symbolic_memory:
            hook_memory = self._hook_symbolic_memory[state_id]
        else:
            # Try root state first (most likely to have hook memory)
            root_id = self._get_pending_root_state_id()
            if root_id is not None and root_id in self._hook_symbolic_memory:
                hook_memory = self._hook_symbolic_memory[root_id]
                l.debug(f"Using root {root_id} hook symbolic memory for state {state_id}")
            else:
                # Walk full ancestry chain
                for ancestor_id in self._get_pending_ancestry():
                    if ancestor_id in self._hook_symbolic_memory:
                        hook_memory = self._hook_symbolic_memory[ancestor_id]
                        l.debug(f"Using ancestor {ancestor_id} hook symbolic memory for state {state_id}")
                        break

        if hook_memory is None:
            return
        restored_count = 0

        for addr, (ast, size) in hook_memory.items():
            try:
                state.memory.store(addr, ast, endness=state.arch.memory_endness)
                self._register_handle(id(ast), ast)
                restored_count += 1
                l.debug(f"Restored hook symbolic memory at 0x{addr:x} (size={size})")
            except Exception as e:
                l.debug(f"Could not restore hook symbolic at 0x{addr:x}: {e}")

        if restored_count > 0:
            l.debug(f"Restored {restored_count} hook symbolic memory regions for state {state_id}")

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
            # Disable Rust-side predicate callbacks — we handle predicates
            # on the Python cached states (which have stdout from printf).
            self._rust_mgr.set_find_needs_python(False)
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
                if not self._rust_mgr.get_state_ids('active'):
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
            # Default: prune unsatisfiable states
            filter_func = lambda s: s.solver.satisfiable()

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
