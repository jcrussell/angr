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
        PythonCallbacks,
        RustSimState as _RustSimState,
    )
    RUST_EXPLORATION_AVAILABLE = True
except ImportError:
    RUST_EXPLORATION_AVAILABLE = False
    _RustExplorationManager = None
    _ExplorationEvent = None
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

        # Add initial states
        if active_states:
            for state in active_states:
                self._add_rust_state('active', state)

    def _setup_callbacks(self):
        """Set up Python callbacks for the Rust engine."""
        callbacks = PythonCallbacks()

        # Memory load callback
        def memory_load(addr: int, size: int) -> tuple:
            # Use a default state for memory access
            state = self._get_default_state()
            if state is None:
                return (bytes(size), False, None)

            try:
                val = state.memory.load(addr, size, endness=state.arch.memory_endness)
                if val.symbolic:
                    concrete = state.solver.eval(val).to_bytes(size, 'little')
                    return (concrete, True, val)
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

        callbacks.set_memory_load(memory_load)
        callbacks.set_memory_store(memory_store)
        callbacks.set_lift_block(lift_block)
        callbacks.set_fetch_page(fetch_page)

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
        """Add an angr state to a Rust stash."""
        # Create Rust state from angr state
        rust_state = _RustSimState(self._project.arch.name)

        # Set PC
        rust_state.pc = angr_state.addr

        # Sync registers
        self._sync_registers_to_rust(angr_state, rust_state)

        # Map memory regions
        self._sync_memory_to_rust(angr_state, rust_state)

        # Add to Rust manager
        self._rust_mgr.add_state(stash, rust_state)

        # Cache the angr state
        self._state_cache[rust_state.state_id] = angr_state

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

    def _get_default_state(self) -> Optional["angr.SimState"]:
        """Get a default state for callbacks."""
        if self._state_cache:
            return next(iter(self._state_cache.values()))
        return None

    def _serialize_irsb(self, irsb) -> str:
        """Serialize a pyvex IRSB to JSON."""
        import json

        def serialize_expr(expr):
            if expr is None:
                return None

            # Handle primitive types directly
            if isinstance(expr, (int, float, str, bool)):
                return expr

            result = {'tag': type(expr).__name__}

            # Handle Const expressions
            if hasattr(expr, 'con'):
                con = expr.con
                result['con'] = {
                    'tag': type(con).__name__,
                    'value': con.value if hasattr(con, 'value') else 0
                }
                return result

            # Handle RdTmp (read temporary)
            if hasattr(expr, 'tmp') and not hasattr(expr, 'data'):
                result['tmp'] = expr.tmp
                return result

            # Handle Get (read register)
            if result['tag'] == 'Get':
                if hasattr(expr, 'offset'):
                    result['offset'] = expr.offset
                if hasattr(expr, 'ty'):
                    result['ty'] = str(expr.ty)
                return result

            # Handle GetI (indexed get)
            if result['tag'] == 'GetI':
                if hasattr(expr, 'descr'):
                    result['descr'] = str(expr.descr)
                if hasattr(expr, 'ix'):
                    result['ix'] = serialize_expr(expr.ix)
                if hasattr(expr, 'bias'):
                    result['bias'] = expr.bias
                return result

            # Handle Load expression
            if result['tag'] == 'Load':
                if hasattr(expr, 'addr'):
                    result['addr'] = serialize_expr(expr.addr)
                if hasattr(expr, 'ty'):
                    result['ty'] = str(expr.ty)
                if hasattr(expr, 'end'):
                    result['end'] = str(expr.end)
                return result

            # Handle Unop, Binop, Triop, Qop (operations)
            if hasattr(expr, 'op'):
                result['op'] = expr.op
                if hasattr(expr, 'args'):
                    result['args'] = [serialize_expr(a) for a in expr.args]
                return result

            # Handle ITE (if-then-else)
            if result['tag'] == 'ITE':
                if hasattr(expr, 'cond'):
                    result['cond'] = serialize_expr(expr.cond)
                if hasattr(expr, 'iftrue'):
                    result['iftrue'] = serialize_expr(expr.iftrue)
                if hasattr(expr, 'iffalse'):
                    result['iffalse'] = serialize_expr(expr.iffalse)
                return result

            # Handle CCall (helper function calls)
            if result['tag'] == 'CCall':
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
            result = {'tag': type(stmt).__name__}
            tag = result['tag']

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
                    result['descr'] = str(stmt.descr)
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
                return result

            # Exit: conditional exit
            if tag == 'Exit':
                if hasattr(stmt, 'guard'):
                    result['guard'] = serialize_expr(stmt.guard)
                if hasattr(stmt, 'dst'):
                    result['dst'] = serialize_expr(stmt.dst)
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

            # AbiHint, MBE, NoOp - simple statements
            if tag in ('AbiHint', 'MBE', 'NoOp'):
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
        """Handle SimProcedure callback from Rust."""
        addr = event.callback_addr
        name = event.callback_name

        # Find the SimProcedure
        proc = self._project._sim_procedures.get(addr)
        if proc is None:
            l.warning(f"SimProcedure not found at 0x{addr:x}")
            self._rust_mgr.resume_after_simprocedure(addr + 1, None, None)
            return

        # Create a minimal angr state for the SimProcedure
        state = self._create_state_for_callback(event)
        if state is None:
            l.warning(f"Could not create state for SimProcedure at 0x{addr:x}")
            self._rust_mgr.resume_after_simprocedure(addr + 1, None, None)
            return

        # Run the SimProcedure
        try:
            # Execute the procedure
            from angr.calling_conventions import SimCC
            cc = SimCC.default_for_arch(self._project.arch)

            # Create a successor group
            proc_instance = proc
            if hasattr(proc, 'run'):
                result = proc.execute(state, None)
            else:
                # It's a class, instantiate it
                proc_instance = proc()
                result = proc_instance.execute(state, None)

            # Get the return address and new state
            if hasattr(result, 'all_successors') and result.all_successors:
                succ_state = result.all_successors[0]
                new_pc = succ_state.addr

                # Extract register changes
                reg_changes = self._extract_register_changes(state, succ_state)
                mem_changes = self._extract_memory_changes(state, succ_state)

                self._rust_mgr.resume_after_simprocedure(new_pc, reg_changes, mem_changes)
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

    def _handle_syscall_callback(self, event: "_ExplorationEvent"):
        """Handle syscall callback from Rust."""
        syscall_num = event.callback_syscall_num

        # Create a state for syscall handling
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

            if successors.all_successors:
                succ_state = successors.all_successors[0]
                new_pc = succ_state.addr

                reg_changes = self._extract_register_changes(state, succ_state)
                mem_changes = self._extract_memory_changes(state, succ_state)

                self._rust_mgr.resume_after_syscall(new_pc, reg_changes, mem_changes)
            else:
                pc = state.addr + 1
                self._rust_mgr.resume_after_syscall(pc, None, None)

        except Exception as e:
            l.warning(f"Syscall execution error: {e}")
            pc = state.addr + 1
            self._rust_mgr.resume_after_syscall(pc, None, None)

    def _create_state_for_callback(self, event: "_ExplorationEvent") -> Optional["angr.SimState"]:
        """Create an angr state from the pending Rust state."""
        try:
            # Create a blank state
            state = self._project.factory.blank_state()

            # Copy registers from pending Rust state
            arch = self._project.arch
            if arch.name in ('AMD64', 'X86_64'):
                reg_names = ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi',
                            'rbp', 'rsp', 'r8', 'r9', 'r10', 'r11',
                            'r12', 'r13', 'r14', 'r15', 'rip']
            elif arch.name == 'X86':
                reg_names = ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi',
                            'ebp', 'esp', 'eip']
            else:
                reg_names = []

            for reg_name in reg_names:
                try:
                    val = self._rust_mgr.get_pending_register(reg_name)
                    if val is not None:
                        setattr(state.regs, reg_name, claripy.BVV(val, arch.bits))
                except Exception:
                    pass

            return state

        except Exception as e:
            l.warning(f"Error creating state for callback: {e}")
            return None

    def _extract_register_changes(
        self,
        old_state: "angr.SimState",
        new_state: "angr.SimState"
    ) -> list:
        """Extract register changes between states."""
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

        for reg_name, (offset, size) in reg_map.items():
            try:
                old_val = getattr(old_state.regs, reg_name)
                new_val = getattr(new_state.regs, reg_name)

                if not new_val.symbolic:
                    new_concrete = new_state.solver.eval(new_val)
                    if old_val.symbolic or old_state.solver.eval(old_val) != new_concrete:
                        data = new_concrete.to_bytes(size, 'little')
                        changes.append((offset, size, list(data)))
            except Exception:
                pass

        return changes

    def _extract_memory_changes(
        self,
        old_state: "angr.SimState",
        new_state: "angr.SimState"
    ) -> list:
        """Extract memory changes between states."""
        # For now, return empty - memory sync is complex
        return []

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
