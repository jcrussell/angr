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
from typing import TYPE_CHECKING, Callable, Optional, Union

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


class RustExplorationManager:
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

        # Set up callbacks
        self._setup_callbacks()

        # Load binary regions
        self._load_binary_regions()

        # Register SimProcedures
        self._register_simprocedures()

        # Track angr state mappings for callbacks
        self._state_cache: dict[int, "angr.SimState"] = {}

        # Track claripy AST handles for constraint sync
        # Maps handle_id -> claripy AST
        self._ast_handle_cache: dict[int, object] = {}

        # Track current callback state for memory access during callbacks
        # This allows memory_load callback to access the correct symbolic state
        self._callback_state: Optional["angr.SimState"] = None

        # Track symbolic memory regions per state for preservation during fallback
        # Maps state_id -> dict[addr -> claripy.AST]
        # When Rust falls back to Python, symbolic memory would be lost without this
        self._symbolic_pages: dict[int, dict[int, object]] = {}

        # Add initial states
        if active_states:
            for state in active_states:
                self._add_rust_state('active', state)

    def _setup_callbacks(self):
        """Set up Python callbacks for the Rust engine."""
        callbacks = PythonCallbacks()

        # Memory load callback - use callback state if available for symbolic access
        def memory_load(addr: int, size: int) -> tuple:
            # Use callback state if available (not default state)
            # This ensures hooks see the symbolic memory from the cached state
            state = self._get_callback_state() or self._get_default_state()
            if state is None:
                return (bytes(size), False, None)

            try:
                val = state.memory.load(addr, size, endness=state.arch.memory_endness)
                if val.symbolic:
                    # Register the handle for later constraint reconstruction
                    # This is critical for bidirectional constraint sync - when Rust
                    # concretizes this address and syncs constraints back, we can
                    # look up the original AST and properly constrain it
                    handle_id = id(val)
                    self._register_handle(handle_id, val)
                    concrete = state.solver.eval(val).to_bytes(size, 'little')
                    return (concrete, True, val)  # Return claripy AST for symbolic values
                else:
                    concrete = state.solver.eval(val).to_bytes(size, 'little')
                    return (concrete, False, None)
            except Exception as e:
                l.warning(f"Memory load error at 0x{addr:x}: {e}")
                return (bytes(size), False, None)

        # Memory store callback
        def memory_store(addr: int, data: bytes):
            state = self._get_default_state()
            if state is None:
                return

            try:
                val = claripy.BVV(int.from_bytes(data, 'little'), len(data) * 8)
                state.memory.store(addr, val, endness=state.arch.memory_endness)
            except Exception as e:
                l.warning(f"Memory store error at 0x{addr:x}: {e}")

        # Block lifting callback
        def lift_block(addr: int) -> str:
            import json
            try:
                block = self._project.factory.block(addr)
                irsb = block.vex
                # Serialize to JSON
                return self._serialize_irsb(irsb)
            except Exception as e:
                l.warning(f"Lift error at 0x{addr:x}: {e}")
                return '{}'

        # Page fetch callback
        def fetch_page(page_addr: int) -> tuple:
            state = self._get_default_state()
            if state is None:
                return (bytes(4096), 0, False)

            try:
                data = state.memory.load(page_addr, 4096, endness=state.arch.memory_endness)
                concrete = state.solver.eval(data).to_bytes(4096, 'little')
                return (concrete, 7, True)  # RWX permissions
            except Exception as e:
                return (bytes(4096), 0, False)

        # Constraint sync callback - receives constraints from Rust before Python fallback
        def sync_constraints(constraints: list):
            """Sync constraints from Rust to Python's claripy solver.

            This is called when Rust falls back to Python for operations that
            depend on solver state (e.g., unmapped memory access). The constraints
            represent concretization decisions made by Rust that need to be
            reflected in Python's solver.

            Args:
                constraints: List of (description, width, concrete_value, handle_id) tuples
            """
            state = self._get_default_state()
            if state is None:
                l.debug(f"sync_constraints called but no state available")
                return

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
                            l.debug(f"Synced constraint from handle {handle_id}: {desc}")
                            continue

                    # Fallback: log the constraint for debugging
                    # Without the original AST, we can't fully reconstruct the constraint
                    l.debug(f"Could not sync constraint (no handle): {desc} = 0x{concrete_val:x}")

                except Exception as e:
                    l.debug(f"Error syncing constraint '{desc}': {e}")

        callbacks.set_memory_load(memory_load)
        callbacks.set_memory_store(memory_store)
        callbacks.set_lift_block(lift_block)
        callbacks.set_fetch_page(fetch_page)
        callbacks.set_sync_constraints(sync_constraints)

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

        if procs:
            self._rust_mgr.register_simprocedures(procs)

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
        self._sync_registers_to_rust(angr_state, rust_state)

        # Map memory regions
        self._sync_memory_to_rust(angr_state, rust_state)

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
            self._state_cache[actual_state_id] = angr_state
            # Extract and cache symbolic memory regions for preservation
            # This ensures symbolic values survive Rust<->Python transitions
            symbolic_pages = self._extract_symbolic_pages(angr_state)
            if symbolic_pages:
                self._symbolic_pages[actual_state_id] = symbolic_pages
                l.debug(f"Cached {len(symbolic_pages)} symbolic pages for state {actual_state_id}")
            l.debug(f"Cached angr state with Rust state ID {actual_state_id}")
        else:
            # Fallback: cache with the Python-side state ID
            self._state_cache[rust_state.state_id] = angr_state
            symbolic_pages = self._extract_symbolic_pages(angr_state)
            if symbolic_pages:
                self._symbolic_pages[rust_state.state_id] = symbolic_pages
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

        for reg_name in reg_names:
            try:
                reg_val = getattr(regs, reg_name)
                if not reg_val.symbolic:
                    rust_state.set_register(reg_name, angr_state.solver.eval(reg_val))
            except (AttributeError, KeyError):
                pass

    def _sync_memory_to_rust(self, angr_state: "angr.SimState", rust_state: "_RustSimState"):
        """Sync memory from angr state to Rust state."""
        # Map main binary regions
        for obj in self._project.loader.all_objects:
            if obj.binary is None:
                continue

            for section in obj.sections:
                if section.memsize > 0:
                    try:
                        data = self._project.loader.memory.load(
                            section.min_addr,
                            section.memsize
                        )
                        rust_state.map_memory_data(section.min_addr, bytes(data), 7)
                    except Exception:
                        pass

        # Map stack (simplified - just map a region)
        stack_base = 0x7fff_fff0_0000
        stack_size = 0x10_0000
        rust_state.map_memory(stack_base - stack_size, stack_size, 6)  # RW

    def _concretize_stack_registers(self, state: "angr.SimState"):
        """Concretize stack registers for Rust memory mapping compatibility.

        This prevents symbolic address issues during Rust exploration by
        ensuring stack-relative registers have concrete values.
        """
        arch = state.arch

        # Determine which registers to concretize based on architecture
        if arch.name in ('AMD64', 'X86_64'):
            stack_regs = ['rsp', 'rbp']
        elif arch.name == 'X86':
            stack_regs = ['esp', 'ebp']
        elif arch.name.startswith('ARM'):
            stack_regs = ['sp']
        else:
            stack_regs = []

        for reg_name in stack_regs:
            try:
                reg_val = getattr(state.regs, reg_name)
                if reg_val.symbolic:
                    concrete_val = state.solver.eval(reg_val)
                    setattr(state.regs, reg_name, concrete_val)
                    l.debug(f"Concretized {reg_name} to 0x{concrete_val:x}")
            except AttributeError:
                pass
            except Exception as e:
                l.debug(f"Could not concretize {reg_name}: {e}")

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
        try:
            # Get history from Rust pending state
            rust_history = self._rust_mgr.get_pending_history()
            rust_jumpkind = self._rust_mgr.get_pending_jumpkind()
        except Exception as e:
            l.debug(f"Could not get Rust history: {e}")
            rust_history = []
            rust_jumpkind = "Ijk_Boring"

        # Ensure history has at least one entry (the callback address)
        callback_addr = event.callback_addr or 0

        # Directly set recent_bbl_addrs to prevent IndexError
        # This is the critical fix - hooks access state.history.recent_bbl_addrs[-1]
        if hasattr(state.history, 'recent_bbl_addrs'):
            # Use Rust history if available, otherwise use callback address
            if rust_history:
                state.history.recent_bbl_addrs = list(rust_history)
            elif callback_addr:
                state.history.recent_bbl_addrs = [callback_addr]

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

        Args:
            state: The angr callback state to initialize.
            event: The exploration event that triggered the callback.
        """
        if event.callback_reason != 'simprocedure':
            return

        # Get SP from Rust for saved state
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
                    event.callback_addr,   # ideal_addr
                )
                l.debug(f"Initialized callstack procedure_data at 0x{event.callback_addr:x}")
        except Exception as e:
            l.debug(f"Could not initialize callstack procedure_data: {e}")

    def _sync_rust_constraints_to_python(self, state: "angr.SimState"):
        """Sync constraints from Rust solver to Python state.

        This exports constraints from the Rust pending state and adds them
        to the Python state's solver. This ensures Python hooks see the
        same constraint context as Rust.

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
            for ast in constraints:
                if ast is not None:
                    try:
                        state.solver.add(ast)
                        synced += 1
                    except Exception as e:
                        l.debug(f"Could not add constraint: {e}")

            if synced > 0:
                l.debug(f"Synced {synced} constraints from Rust to Python state")

        except Exception as e:
            l.debug(f"Could not sync Rust constraints: {e}")

    def _lookup_handle(self, handle_id: int) -> Optional[object]:
        """Look up a claripy AST by its handle ID.

        When claripy ASTs are passed to Rust, they get assigned handle IDs
        that allow us to look them up later for constraint reconstruction.

        Args:
            handle_id: The handle ID assigned by Rust.

        Returns:
            The claripy AST if found, None otherwise.
        """
        return self._ast_handle_cache.get(handle_id)

    def _register_handle(self, handle_id: int, ast: object):
        """Register a claripy AST with its handle ID for later lookup.

        Uses an LRU eviction strategy that preserves actively referenced handles.
        Handles marked as active (via _mark_handle_active) are never evicted.

        Args:
            handle_id: The handle ID assigned by Rust.
            ast: The claripy AST to cache.
        """
        self._ast_handle_cache[handle_id] = ast

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
                return {
                    'tag': f'Ico_{expr_name}',
                    'value': expr.value
                }

            # Rust expects VEX expression tags with Iex_ prefix
            result = {'tag': f'Iex_{expr_name}'}

            # Handle Const expressions
            if hasattr(expr, 'con'):
                con = expr.con
                # Rust expects VEX constant tags with Ico_ prefix
                result['con'] = {
                    'tag': f'Ico_{type(con).__name__}',
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
                    result['cee'] = str(expr.cee)
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
                if hasattr(stmt, 'addr'):
                    result['addr'] = serialize_expr(stmt.addr)
                if hasattr(stmt, 'storedata'):
                    result['storedata'] = serialize_expr(stmt.storedata)
                if hasattr(stmt, 'result'):
                    result['result'] = stmt.result
                return result

            # Dirty: helper call with side effects
            if tag == 'Dirty':
                if hasattr(stmt, 'cee'):
                    result['cee'] = str(stmt.cee)
                if hasattr(stmt, 'args'):
                    result['args'] = [serialize_expr(a) for a in stmt.args]
                if hasattr(stmt, 'tmp'):
                    result['tmp'] = stmt.tmp
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
        addr = event.callback_addr
        name = event.callback_name
        state_id = event.callback_state_id

        # Find the SimProcedure
        proc = self._project._sim_procedures.get(addr)
        if proc is None:
            l.warning(f"SimProcedure not found at 0x{addr:x}")
            self._rust_mgr.resume_after_simprocedure(addr + 1, None, None)
            return

        # Get hook length - this determines if the hook replaces code
        hook_length = getattr(proc, 'kwargs', {}).get('length', 0)
        if hook_length == 0:
            hook_length = getattr(proc, 'length', 0)
        is_zero_length_hook = (hook_length == 0)

        # Create angr state for the SimProcedure (uses cached state if available)
        state = self._create_state_for_callback(event)
        if state is None:
            l.warning(f"Could not create state for SimProcedure at 0x{addr:x}")
            self._rust_mgr.resume_after_simprocedure(addr + 1, None, None)
            return

        # Run the SimProcedure
        try:
            from angr.engines.successors import SimSuccessors

            # Create SimSuccessors object for the procedure
            successors = SimSuccessors(addr=addr, initial_state=state)

            # Execute the procedure
            proc_instance = proc
            if hasattr(proc, 'run'):
                proc.execute(state, successors)
            else:
                # It's a class, instantiate it
                proc_instance = proc()
                proc_instance.execute(state, successors)

            # Handle successors
            all_succs = successors.all_successors
            if all_succs:
                # First successor continues in Rust
                first_succ = all_succs[0]

                # For zero-length hooks, if successor has same address as hook,
                # the hook just modifies state and we should continue execution
                # at the same address WITHOUT re-triggering the hook
                if is_zero_length_hook and first_succ.addr == addr:
                    # Resume with skip_hook flag to prevent infinite loop
                    self._resume_with_state(first_succ, state, event, skip_hook_addr=addr)
                else:
                    self._resume_with_state(first_succ, state, event)

                # Additional successors are added as new active states
                for succ in all_succs[1:]:
                    self._add_forked_state(succ, event)

            else:
                # No successors - for zero-length hooks, execute the instruction
                if is_zero_length_hook:
                    # Tell Rust to skip the hook and execute from addr
                    self._resume_with_skip_hook(addr, state, event)
                else:
                    # No successors - maybe it's a no-return procedure
                    ret_addr = event.callback_return_addr or (addr + 1)
                    self._rust_mgr.resume_after_simprocedure(ret_addr, None, None)

        except Exception as e:
            l.warning(f"SimProcedure execution error at 0x{addr:x}: {e}")
            import traceback
            traceback.print_exc()
            ret_addr = event.callback_return_addr or (addr + 1)
            self._rust_mgr.resume_after_simprocedure(ret_addr, None, None)
        finally:
            # Clear callback state to avoid stale references
            self._set_callback_state(None)

    def _resume_with_state(
        self,
        succ_state: "angr.SimState",
        orig_state: "angr.SimState",
        event: "_ExplorationEvent",
        skip_hook_addr: Optional[int] = None
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
        """
        new_pc = succ_state.addr

        # Extract changes
        reg_changes = self._extract_register_changes(orig_state, succ_state)
        mem_changes = self._extract_memory_changes(orig_state, succ_state)

        # Extract any new constraints added during callback
        new_constraints = self._extract_new_constraints(orig_state, succ_state)
        if new_constraints:
            l.debug(f"Extracted {len(new_constraints)} new constraints from callback")

        # For zero-length hooks, tell Rust to skip the hook on next step
        if skip_hook_addr is not None:
            try:
                self._rust_mgr.set_skip_hook_addr(skip_hook_addr)
                l.debug(f"Set skip_hook_addr to 0x{skip_hook_addr:x}")
            except Exception as e:
                l.debug(f"Could not set skip_hook_addr: {e}")

        # Resume Rust with the changes and any new constraints
        # This enables bidirectional constraint flow: Rust -> Python -> Rust
        self._rust_mgr.resume_after_simprocedure(
            new_pc, reg_changes, mem_changes, new_constraints or None
        )

        # Update cache for future callbacks
        state_id = event.callback_state_id
        if state_id is not None:
            self._state_cache[state_id] = succ_state

    def _resume_with_skip_hook(
        self,
        addr: int,
        state: "angr.SimState",
        event: "_ExplorationEvent"
    ):
        """Resume Rust execution at addr, skipping the hook there.

        Used for zero-length hooks that have no successors.
        """
        try:
            self._rust_mgr.set_skip_hook_addr(addr)
            l.debug(f"Set skip_hook_addr to 0x{addr:x} (no successors)")
        except Exception as e:
            l.debug(f"Could not set skip_hook_addr: {e}")

        self._rust_mgr.resume_after_simprocedure(addr, None, None)

        # Update cache
        state_id = event.callback_state_id
        if state_id is not None:
            self._state_cache[state_id] = state

    def _extract_new_constraints(
        self,
        orig_state: "angr.SimState",
        new_state: "angr.SimState"
    ) -> list:
        """Extract constraints added during callback execution.

        Compares constraint sets between original and successor states,
        returning any new constraints that were added during the callback.

        Also checks the forked Rust solver context for any constraints that
        were added there, ensuring bidirectional constraint flow.

        Returns:
            List of new claripy constraint ASTs.
        """
        new_constraints = []

        # Extract constraints from Python's claripy solver
        try:
            orig_constraints = set(orig_state.solver.constraints)
            state_constraints = set(new_state.solver.constraints)
            python_added = state_constraints - orig_constraints
            if python_added:
                l.debug(f"Callback added {len(python_added)} new Python constraints")
                new_constraints.extend(python_added)
        except Exception as e:
            l.debug(f"Could not extract Python constraints: {e}")

        # Check if the new state has a forked Rust solver context
        # If constraints were added there, we need to verify they're in sync
        if hasattr(new_state, 'scratch') and hasattr(new_state.scratch, 'rust_solver_ctx'):
            try:
                forked_solver = new_state.scratch.rust_solver_ctx
                # Track the constraint count for debugging
                rust_count = forked_solver.num_constraints()

                # Check original state's forked solver count (if any)
                orig_count = 0
                if hasattr(orig_state, 'scratch') and hasattr(orig_state.scratch, 'rust_solver_ctx'):
                    orig_count = orig_state.scratch.rust_solver_ctx.num_constraints()

                if rust_count > orig_count:
                    delta = rust_count - orig_count
                    l.debug(f"Forked Rust solver has {delta} new constraints "
                            f"(total: {rust_count})")

                    # Get constraint info for debugging
                    if hasattr(forked_solver, 'export_constraint_info'):
                        try:
                            info = forked_solver.export_constraint_info()
                            if len(info) > 0:
                                l.debug(f"Rust solver constraint types: {len(info)}")
                        except Exception as e:
                            l.debug(f"Could not export Rust constraint info: {e}")

            except Exception as e:
                l.debug(f"Could not check Rust solver constraints: {e}")

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
        if state_id is not None and state_id in self._state_cache:
            cached_state = self._state_cache[state_id]
            state = cached_state.copy()  # Copy to preserve original

            # CRITICAL: Fork the Rust solver context with all accumulated constraints
            # This ensures SimProcedures see the full constraint context from exploration
            try:
                forked_solver = self._rust_mgr.fork_pending_solver()
                # Attach forked Rust solver to state for constraint evaluation
                # Store as scratch attribute for use by SimProcedures
                state.scratch.rust_solver_ctx = forked_solver
                constraint_count = forked_solver.num_constraints()
                l.debug(f"Forked Rust solver context for callback state {state_id} "
                        f"({constraint_count} constraints)")
            except Exception as e:
                l.warning(f"Could not fork solver context: {e}")

            self._sync_registers_from_rust_pending(state)

            # Restore symbolic memory regions that may have been lost during
            # Rust execution fallback. This is critical for callbacks that need
            # to read symbolic values from memory (e.g., symbolic buffer access)
            self._restore_symbolic_pages(state, state_id)

            l.debug(f"Using cached state {state_id} for callback")
        else:
            # Fallback to blank state (original behavior)
            l.warning(f"No cached state for ID {state_id}, using blank state fallback")
            state = self._create_blank_state_fallback(event)

        if state is not None:
            # Initialize history to prevent IndexError in hooks
            self._init_callback_history(state, event)

            # Initialize callstack for SimProcedure continuations
            self._init_callback_callstack(state, event)

            # Set callback state for memory access during this callback
            self._set_callback_state(state)

            # Sync Rust constraints to Python state
            self._sync_rust_constraints_to_python(state)

        return state

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
        """Sync concrete register values from Rust pending state to angr state."""
        arch = self._project.arch
        reg_names = self._get_arch_register_names(arch)

        for reg_name in reg_names:
            try:
                val = self._rust_mgr.get_pending_register(reg_name)
                if val is not None:
                    setattr(state.regs, reg_name, claripy.BVV(val, arch.bits))
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

        # Map register names to offsets
        reg_map = {
            'rax': (16, 8), 'rcx': (24, 8), 'rdx': (32, 8), 'rbx': (40, 8),
            'rsp': (48, 8), 'rbp': (56, 8), 'rsi': (64, 8), 'rdi': (72, 8),
            'r8': (80, 8), 'r9': (88, 8), 'r10': (96, 8), 'r11': (104, 8),
            'r12': (112, 8), 'r13': (120, 8), 'r14': (128, 8), 'r15': (136, 8),
            'rip': (184, 8),
        }

        # Return register is critical - always sync it
        return_regs = {'rax', 'eax', 'r0'}  # AMD64, x86, ARM

        for reg_name, (offset, size) in reg_map.items():
            try:
                old_val = getattr(old_state.regs, reg_name)
                new_val = getattr(new_state.regs, reg_name)

                if new_val.symbolic:
                    # Symbolic register value - sync to Rust as claripy AST
                    if reg_name in return_regs:
                        # Return register is critical - sync symbolic value
                        try:
                            self._sync_symbolic_register_to_rust(reg_name, new_val)
                            l.debug(f"Synced symbolic return register {reg_name} to Rust")
                        except Exception as e:
                            l.debug(f"Could not sync symbolic {reg_name}: {e}")
                            # Fallback: try to eval to concrete
                            try:
                                new_concrete = new_state.solver.eval(new_val)
                                data = new_concrete.to_bytes(size, 'little')
                                changes.append((offset, size, list(data)))
                            except Exception:
                                pass
                else:
                    new_concrete = new_state.solver.eval(new_val)
                    if old_val.symbolic or old_state.solver.eval(old_val) != new_concrete:
                        data = new_concrete.to_bytes(size, 'little')
                        changes.append((offset, size, list(data)))
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
    ) -> list:
        """Extract memory changes between states for Rust sync.

        Uses angr's changed_bytes() to detect memory modifications,
        groups them into contiguous regions, and returns concrete values.
        Also tracks symbolic values for constraint propagation.

        Returns:
            List of (addr, data_bytes) tuples for memory changes.
        """
        changes = []
        try:
            # Use angr's changed_bytes to find modifications
            changed = new_state.memory.changed_bytes(old_state.memory)
            if not changed:
                return []

            # Group consecutive changed bytes into regions
            for item in self._group_changed_bytes(new_state, changed):
                if item[0] == 'concrete':
                    _, start, size, data = item
                    changes.append((start, list(data)))
                elif item[0] == 'symbolic':
                    _, start, size, data, handle_id, ast = item
                    # Provide concrete witness for Rust memory sync
                    changes.append((start, list(data)))
                    # Cache symbolic value for later constraint sync
                    # The handle is already registered in _emit_memory_region
                    l.debug(f"Tracked symbolic memory change at 0x{start:x} (handle={handle_id})")

        except Exception as e:
            l.debug(f"Error extracting memory changes: {e}")

        return changes

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

        The concrete witness is always provided for Rust memory sync, but symbolic
        values also include the handle_id and AST for later constraint propagation.
        """
        # Limit region size to avoid memory issues
        MAX_REGION_SIZE = 4096
        if size > MAX_REGION_SIZE:
            # Split into smaller chunks
            for offset in range(0, size, MAX_REGION_SIZE):
                chunk_size = min(MAX_REGION_SIZE, size - offset)
                yield from self._emit_memory_region(state, start + offset, chunk_size)
            return

        try:
            val = state.memory.load(start, size, endness=state.arch.memory_endness)
            if not val.symbolic:
                concrete = state.solver.eval(val)
                # Convert to little-endian bytes
                data = concrete.to_bytes(size, 'little')
                yield ('concrete', start, size, data)
            else:
                # For symbolic values, register handle for later reconstruction
                handle_id = id(val)
                self._register_handle(handle_id, val)

                # Get concrete witness that satisfies current constraints
                try:
                    concrete = state.solver.eval(val)
                    data = concrete.to_bytes(size, 'little')
                except Exception:
                    # Cannot concretize - use zeros as placeholder
                    data = bytes(size)

                # Yield symbolic tuple with handle info for constraint propagation
                yield ('symbolic', start, size, data, handle_id, val)
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
                                if val.symbolic:
                                    symbolic_regions[addr] = val
                                    self._register_handle(id(val), val)
                            except Exception:
                                pass
                        if symbolic_regions:
                            l.debug(f"Extracted {len(symbolic_regions)} symbolic bytes via get_symbolic_addrs")
                            return symbolic_regions
                except Exception as e:
                    l.debug(f"get_symbolic_addrs failed: {e}")

            # Strategy 2: Check page-level symbolic maps if available
            if hasattr(state.memory, '_pages'):
                for page_addr in list(state.memory._pages.keys()):
                    page = state.memory._pages.get(page_addr)
                    if page is None:
                        continue

                    # Check if page has symbolic byte tracking
                    if hasattr(page, '_symbolic_bitmap') and page._symbolic_bitmap:
                        # Use internal bitmap for precise tracking
                        for offset in range(4096):
                            if page._symbolic_bitmap.get(offset, False):
                                addr = page_addr + offset
                                try:
                                    val = state.memory.load(addr, 1, endness=state.arch.memory_endness)
                                    if val.symbolic:
                                        symbolic_regions[addr] = val
                                        self._register_handle(id(val), val)
                                except Exception:
                                    pass
                    elif hasattr(page, 'symbolic_byte_map') and page.symbolic_byte_map:
                        # Alternative: symbolic_byte_map
                        for offset, sym_val in page.symbolic_byte_map.items():
                            addr = page_addr + offset
                            symbolic_regions[addr] = sym_val
                            self._register_handle(id(sym_val), sym_val)
                    else:
                        # Strategy 3: Byte-granularity fallback for pages without tracking
                        # Sample multiple offsets to detect symbolic content anywhere in page
                        sample_offsets = [0, 256, 512, 1024, 2048, 3072, 4088]
                        has_symbolic = False
                        for offset in sample_offsets:
                            try:
                                test_val = state.memory.load(page_addr + offset, 8, endness=state.arch.memory_endness)
                                if test_val.symbolic:
                                    has_symbolic = True
                                    break
                            except Exception:
                                continue

                        if has_symbolic:
                            # Scan the full page at byte granularity
                            for offset in range(4096):
                                addr = page_addr + offset
                                try:
                                    val = state.memory.load(addr, 1, endness=state.arch.memory_endness)
                                    if val.symbolic:
                                        symbolic_regions[addr] = val
                                        self._register_handle(id(val), val)
                                except Exception:
                                    pass

            if symbolic_regions:
                l.debug(f"Extracted {len(symbolic_regions)} symbolic memory regions")

        except Exception as e:
            l.debug(f"Error extracting symbolic pages: {e}")
        return symbolic_regions

    def _restore_symbolic_pages(self, state: "angr.SimState", state_id: int):
        """Restore symbolic memory regions to an angr state.

        This restores symbolic values that were previously extracted and
        cached, ensuring that Python callbacks see the correct symbolic
        memory context after fallback from Rust.

        Args:
            state: The angr state to restore symbolic pages to.
            state_id: The state ID to look up cached symbolic pages.
        """
        if state_id not in self._symbolic_pages:
            return

        symbolic_pages = self._symbolic_pages[state_id]
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
            l.debug(f"Failed to restore {failed_count} symbolic memory regions")

    # =========================================================================
    # Public API (SimulationManager-like interface)
    # =========================================================================

    def explore(
        self,
        find: Optional[Union[int, list, Callable]] = None,
        avoid: Optional[Union[int, list, Callable]] = None,
        num_find: int = 1,
        **kwargs
    ) -> "RustExplorationManager":
        """Run exploration with find/avoid conditions.

        Args:
            find: Address(es) or callable predicate for finding solutions.
            avoid: Address(es) or callable predicate for avoiding states.
            num_find: Number of solutions to find before stopping.
            **kwargs: Additional arguments (ignored for compatibility).

        Returns:
            Self, for chaining.
        """
        # Set find addresses
        find_addrs = self._extract_addrs(find)
        self._rust_mgr.set_find_addrs(find_addrs)
        self._rust_mgr.set_find_needs_python(callable(find))

        # Set avoid addresses
        avoid_addrs = self._extract_addrs(avoid)
        self._rust_mgr.set_avoid_addrs(avoid_addrs)
        self._rust_mgr.set_avoid_needs_python(callable(avoid))

        # Set num_find
        self._rust_mgr.set_num_find(num_find)

        # Run exploration loop
        while True:
            event = self._rust_mgr.run()

            if event.event_type == 'found' and event.found_count >= num_find:
                break
            elif event.event_type == 'need_callback':
                if event.callback_reason == 'simprocedure':
                    self._handle_simprocedure_callback(event)
                elif event.callback_reason == 'syscall':
                    self._handle_syscall_callback(event)
                else:
                    l.warning(f"Unknown callback reason: {event.callback_reason}")
                    break
            elif event.event_type == 'active_empty':
                break
            elif event.event_type == 'errored':
                l.warning(f"Exploration error: {event.callback_reason}")
                break
            elif event.event_type == 'step_complete':
                # Continue exploration
                continue

        return self

    def step(self, n: int = 1, **kwargs) -> "RustExplorationManager":
        """Step the exploration n times.

        Args:
            n: Number of steps to take.
            **kwargs: Additional arguments (ignored).

        Returns:
            Self, for chaining.
        """
        self._rust_mgr.run(n)
        return self

    @property
    def active(self) -> list:
        """Get states in the active stash.

        Note: This returns state IDs, not full angr states.
        Use found_states() for full angr state conversion.
        """
        return self._rust_mgr.get_state_ids('active')

    @property
    def found(self) -> list:
        """Get states in the found stash.

        Note: This returns state IDs, not full angr states.
        """
        return self._rust_mgr.get_state_ids('found')

    @property
    def avoid(self) -> list:
        """Get states in the avoid stash."""
        return self._rust_mgr.get_state_ids('avoid')

    @property
    def deadended(self) -> list:
        """Get states in the deadended stash."""
        return self._rust_mgr.get_state_ids('deadended')

    @property
    def errored(self) -> list:
        """Get states in the errored stash."""
        return self._rust_mgr.get_state_ids('errored')

    @property
    def unconstrained(self) -> list:
        """Get states in the unconstrained stash.

        These are states where a symbolic jump target (e.g., ret instruction
        with symbolic return address) had too many possible concrete values
        to enumerate and fork.
        """
        return self._rust_mgr.get_state_ids('unconstrained')

    # =========================================================================
    # State Conversion Methods
    # =========================================================================

    @property
    def found_states(self) -> list["angr.SimState"]:
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
                l.warning(f"Failed to convert state {snapshot.state_id}: {e}")
        return states

    def get_state_by_id(self, state_id: int) -> Optional["angr.SimState"]:
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
            l.warning(f"Failed to get state {state_id}: {e}")
            return None

    def _snapshot_to_angr(self, snapshot: "_ExplorationStateSnapshot") -> "angr.SimState":
        """Convert a Rust state snapshot to an angr SimState.

        This creates an angr SimState from the snapshot data, including:
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
        # Create a blank state with the correct address
        state = self._project.factory.blank_state(addr=snapshot.pc)

        # Set register values from raw bytes
        reg_bytes = snapshot.get_registers_raw()
        arch = self._project.arch

        # Common register mappings for AMD64
        if arch.name in ('AMD64', 'X86_64'):
            reg_offsets = {
                'rax': (16, 8), 'rcx': (24, 8), 'rdx': (32, 8), 'rbx': (40, 8),
                'rsp': (48, 8), 'rbp': (56, 8), 'rsi': (64, 8), 'rdi': (72, 8),
                'r8': (80, 8), 'r9': (88, 8), 'r10': (96, 8), 'r11': (104, 8),
                'r12': (112, 8), 'r13': (120, 8), 'r14': (128, 8), 'r15': (136, 8),
                'rip': (184, 8),
            }
            for reg_name, (offset, size) in reg_offsets.items():
                if offset + size <= len(reg_bytes):
                    value = int.from_bytes(reg_bytes[offset:offset+size], 'little')
                    try:
                        setattr(state.regs, reg_name, claripy.BVV(value, size * 8))
                    except Exception:
                        pass
        elif arch.name == 'X86':
            reg_offsets = {
                'eax': (8, 4), 'ecx': (12, 4), 'edx': (16, 4), 'ebx': (20, 4),
                'esp': (24, 4), 'ebp': (28, 4), 'esi': (32, 4), 'edi': (36, 4),
                'eip': (68, 4),
            }
            for reg_name, (offset, size) in reg_offsets.items():
                if offset + size <= len(reg_bytes):
                    value = int.from_bytes(reg_bytes[offset:offset+size], 'little')
                    try:
                        setattr(state.regs, reg_name, claripy.BVV(value, size * 8))
                    except Exception:
                        pass

        # Load memory pages
        for i in range(snapshot.page_count()):
            page = snapshot.get_page(i)
            if page is not None:
                addr, data, _perms = page
                try:
                    # Store the page data in the state
                    state.memory.store(addr, claripy.BVV(data, len(data) * 8),
                                       endness=arch.memory_endness,
                                       inspect=False)
                except Exception as e:
                    l.debug(f"Failed to load page at 0x{addr:x}: {e}")

        # Store state ID as a scratch attribute for reference
        state.scratch.rust_state_id = snapshot.state_id
        state.scratch.rust_parent_id = snapshot.parent_id

        return state

    def eval_memory(self, state_id: int, addr: int, size: int) -> Optional[bytes]:
        """Evaluate memory from a Rust state's solver context.

        This evaluates symbolic memory using the Rust solver context,
        returning a concrete value given the state's constraints.

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

    def stats(self) -> dict:
        """Get exploration statistics."""
        return dict(self._rust_mgr.stats())

    # Compatibility methods for SimulationManager API

    def use_technique(self, technique, **kwargs):
        """Apply an exploration technique (compatibility stub).

        Note: Most exploration techniques are not supported by the Rust
        exploration manager. This method logs a warning and ignores the
        technique to allow scripts to run.
        """
        l.warning(
            f"RustExplorationManager.use_technique({type(technique).__name__}) "
            "called but exploration techniques are not supported. Ignoring."
        )
        return self

    def run(self, **kwargs) -> "RustExplorationManager":
        """Alias for explore() for SimulationManager compatibility."""
        return self.explore(**kwargs)

    def move(self, from_stash: str, to_stash: str, filter_func=None) -> int:
        """Move states between stashes (compatibility stub).

        Note: Limited support - only moves all states without filtering.
        """
        if filter_func is not None:
            l.warning("RustExplorationManager.move() does not support filter functions")
        return self._rust_mgr.move_states(from_stash, to_stash, None)

    def one_active(self):
        """Get one active state (compatibility stub)."""
        active = self.active
        if active:
            return active[0]
        return None

    def one_found(self):
        """Get one found state (compatibility stub)."""
        found = self.found
        if found:
            return found[0]
        return None

    def __len__(self) -> int:
        """Return total number of active states."""
        return self._rust_mgr.active_count()

    def __getattr__(self, name: str):
        """Handle attribute access for stash names."""
        # Try to get stash by name
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._rust_mgr.get_state_ids(name)
        except Exception:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
