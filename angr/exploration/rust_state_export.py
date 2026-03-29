"""State export and conversion for the Rust exploration manager.

Handles converting Rust state snapshots to angr SimStates, syncing constraints,
restoring plugins, and managing stash export. These methods are defined as a
mixin class that RustExplorationManager inherits from.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import claripy

if TYPE_CHECKING:
    import angr

l = logging.getLogger(name=__name__)


class RustStateExportMixin:
    """Mixin providing state export/conversion methods for RustExplorationManager.

    This mixin expects the host class to have:
    - self._rust_mgr: The Rust exploration manager PyO3 object
    - self._project: The angr Project
    - self._state_cache: Dict[int, SimState]
    - self._state_roots: Dict[int, int]
    - self._addr_to_ast: Dict[int, Dict[int, Tuple]]
    - self._hook_symbolic_memory: Dict[int, Dict[int, Tuple]]
    - self._identity_tracker: SymbolicIdentityTracker
    """

    def _get_stash_states(self, stash: str) -> list:
        """Get states from a stash as angr SimStates.

        Args:
            stash: The stash name ('found', 'active', 'avoid', etc.)

        Returns:
            List of angr SimStates converted from Rust states.
        """
        states = []

        # First try cached Python states (have proper memory from forking)
        state_ids = self._rust_mgr.get_state_ids(stash)
        for state_id in state_ids:
            if state_id in self._state_cache:
                state = self._state_cache[state_id]
                self._restore_plugins_to_state(state, state_id)
                self._sync_exported_constraints(state, state_id)
                # Sync Rust PC to the Python state so multi-stage exploration
                # (re-init from found state) gets the correct program counter.
                try:
                    # Find state's index in the stash
                    all_ids = self._rust_mgr.get_state_ids(stash)
                    idx = all_ids.index(state_id) if state_id in all_ids else -1
                    if idx >= 0:
                        pc = self._rust_mgr.get_state_pc(stash, idx)
                        if pc is not None:
                            state.regs._ip = pc
                except Exception:
                    pass
                states.append(state)

        # For states not in cache, try parent state or snapshot export
        cached_ids = {sid for sid in state_ids if sid in self._state_cache}
        uncached_ids = [sid for sid in state_ids if sid not in self._state_cache]

        if uncached_ids:
            # First try: look up parent state in cache (for intercepted find/avoid states)
            for sid in uncached_ids:
                root = self._state_roots.get(sid)
                if root is not None and root in self._state_cache:
                    state = self._state_cache[root].copy()
                    self._restore_plugins_to_state(state, sid)
                    self._sync_exported_constraints(state, sid)
                    states.append(state)
                    cached_ids.add(sid)

            # Remaining: try stepping state (the state that was being stepped
            # when the find/avoid was detected)
            stepping_id = getattr(self, '_current_stepping_state_id', None)
            for sid in uncached_ids:
                if sid in cached_ids:
                    continue
                if stepping_id is not None and stepping_id in self._state_cache:
                    state = self._state_cache[stepping_id].copy()
                    self._restore_plugins_to_state(state, sid)
                    self._sync_exported_constraints(state, sid)
                    states.append(state)
                    cached_ids.add(sid)

            # Last resort: snapshot export
            remaining = [sid for sid in uncached_ids if sid not in cached_ids]
            if remaining:
                try:
                    snapshots = self._rust_mgr.export_stash(stash)
                    for snapshot in snapshots:
                        if snapshot.state_id not in cached_ids:
                            try:
                                angr_state = self._snapshot_to_angr(snapshot)
                                self._sync_exported_constraints(angr_state, snapshot.state_id)
                                states.append(angr_state)
                            except Exception as e:
                                l.warning(f"Failed to convert state from {stash}: {e}")
                except Exception as e:
                    l.warning(f"export_stash failed for {stash}: {e}")

        return states

    def _sync_exported_constraints(self, state, state_id):
        """Sync constraints from Rust solver to Python state.

        Adds Rust-exported constraints to the Python state. Skips register
        constraints and constraints that use symbols with different identity
        than existing Python symbols (which would cause UNSAT).
        """
        try:
            rust_constraints = self._rust_mgr.export_state_constraints(state_id)
            synced = 0
            skipped = 0

            # Build a map of existing Python leaf ASTs by variable name.
            # Used to detect identity mismatches: same name, different object.
            existing_leaves = {}
            for c in state.solver.constraints:
                for leaf in c.leaf_asts():
                    if hasattr(leaf, 'args') and len(leaf.args) > 0 and isinstance(leaf.args[0], str):
                        existing_leaves[leaf.args[0]] = leaf

            for c in rust_constraints:
                if c is None:
                    continue
                try:
                    c_str = str(c)
                    # Skip register-related constraints
                    if 'reg_' in c_str:
                        skipped += 1
                        continue

                    # Check for identity mismatch: if a Rust constraint uses
                    # a symbol with the same NAME as a Python symbol but a
                    # DIFFERENT object identity, skip it to prevent UNSAT.
                    identity_conflict = False
                    if existing_leaves:
                        for leaf in c.leaf_asts():
                            if hasattr(leaf, 'args') and len(leaf.args) > 0 and isinstance(leaf.args[0], str):
                                name = leaf.args[0]
                                if name in existing_leaves and existing_leaves[name] is not leaf:
                                    identity_conflict = True
                                    break
                    if identity_conflict:
                        skipped += 1
                        continue

                    if hasattr(c, 'op'):
                        if getattr(c, 'length', None) is None:
                            state.solver.add(c)
                        else:
                            state.solver.add(c != 0)
                        synced += 1
                except Exception:
                    pass
            if synced or skipped:
                l.debug(f"Synced {synced} constraints to state {state_id} "
                        f"({skipped} skipped for identity/register)")
        except Exception as e:
            l.debug(f"Could not sync constraints for state {state_id}: {e}")

    @property
    def found_states(self) -> list:
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

    def get_state_by_id(self, state_id: int):
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

    def _snapshot_to_angr(self, snapshot) -> "angr.SimState":
        """Convert a Rust state snapshot to an angr SimState.

        Creates an angr SimState from the snapshot data, including:
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

        # Set register values from Rust's named register export.
        # This uses Rust's architecture register tables directly,
        # eliminating Python-side offset mapping.
        named_regs = snapshot.get_registers_named()
        for reg_name, (value, size_bits) in named_regs.items():
            try:
                setattr(state.regs, reg_name, claripy.BVV(value, size_bits))
            except Exception:
                pass  # Skip VEX internal registers that angr doesn't expose

        arch = self._project.arch

        # Load memory pages
        for i in range(snapshot.page_count()):
            page = snapshot.get_page(i)
            if page is not None:
                page_addr, data, _perms, symbolic_offsets = page
                try:
                    # First store the entire page as concrete data
                    state.memory.store(page_addr, claripy.BVV(data, len(data) * 8),
                                       endness=arch.memory_endness,
                                       inspect=False)

                    # Then overwrite symbolic regions with fresh symbolic variables
                    if symbolic_offsets:
                        # Find contiguous symbolic regions to create multi-byte symbols
                        sorted_offsets = sorted(symbolic_offsets)
                        regions = []
                        start = sorted_offsets[0]
                        end = start
                        for offset in sorted_offsets[1:]:
                            if offset == end + 1:
                                end = offset
                            else:
                                regions.append((start, end - start + 1))
                                start = offset
                                end = offset
                        regions.append((start, end - start + 1))

                        # P6 fix: Track symbolic values for constraint sync
                        symbolic_values_to_constrain = []

                        # Restore symbolic values - try to recover original ASTs first (P1 fix)
                        for offset, size in regions:
                            sym_addr = page_addr + offset
                            original_ast = None

                            # Try to find original AST from address tracking
                            state_addr_map = self._addr_to_ast.get(snapshot.state_id, {})
                            if sym_addr in state_addr_map:
                                tracked_ast, tracked_size = state_addr_map[sym_addr]
                                if tracked_size == size:
                                    original_ast = tracked_ast
                                    l.debug(f"Recovered original AST at 0x{sym_addr:x} for state {snapshot.state_id}")

                            # Also check parent state for inherited symbolic values
                            if original_ast is None and snapshot.parent_id >= 0:
                                parent_addr_map = self._addr_to_ast.get(snapshot.parent_id, {})
                                if sym_addr in parent_addr_map:
                                    tracked_ast, tracked_size = parent_addr_map[sym_addr]
                                    if tracked_size == size:
                                        original_ast = tracked_ast
                                        l.debug(f"Recovered original AST from parent at 0x{sym_addr:x}")

                            # Also check hook symbolic memory
                            hook_mem = self._hook_symbolic_memory.get(snapshot.state_id, {})
                            if original_ast is None and sym_addr in hook_mem:
                                tracked_ast, tracked_size = hook_mem[sym_addr]
                                if tracked_size == size:
                                    original_ast = tracked_ast
                                    l.debug(f"Recovered original AST from hook memory at 0x{sym_addr:x}")

                            # Use original AST if found, otherwise create fresh symbol
                            if original_ast is not None:
                                state.memory.store(sym_addr, original_ast,
                                                   endness=arch.memory_endness,
                                                   inspect=False)
                                # P6 fix: Track for constraint sync
                                symbolic_values_to_constrain.append((sym_addr, size, original_ast))
                            else:
                                # Fallback: create fresh symbolic (for Rust-created symbols)
                                sym_name = f"rust_sym_{sym_addr:x}_{snapshot.state_id}"
                                sym_val = claripy.BVS(sym_name, size * 8)
                                state.memory.store(sym_addr, sym_val,
                                                   endness=arch.memory_endness,
                                                   inspect=False)
                                # P6 fix: Also track fresh symbols for constraint sync
                                symbolic_values_to_constrain.append((sym_addr, size, sym_val))

                        # P6 fix: Add constraints for symbolic values based on Rust solver evaluation
                        for sym_addr, size, ast in symbolic_values_to_constrain:
                            try:
                                # Evaluate the symbolic value using Rust's solver context
                                concrete_bytes = self._rust_mgr.get_state_memory(
                                    snapshot.state_id, sym_addr, size
                                )
                                if concrete_bytes is not None:
                                    # Convert bytes to int (little endian)
                                    concrete_val = int.from_bytes(concrete_bytes, 'little')
                                    # Add constraint: original_ast == concrete_value
                                    constraint = ast == claripy.BVV(concrete_val, size * 8)
                                    state.solver.add(constraint)
                                    l.debug(f"P6: Added constraint at 0x{sym_addr:x}: "
                                            f"{ast} == {concrete_val:#x}")
                            except Exception as e:
                                l.debug(f"P6: Could not add constraint at 0x{sym_addr:x}: {e}")

                except Exception as e:
                    l.warning(f"Failed to load page at 0x{page_addr:x}: {e}")

        # Store state ID as a scratch attribute for reference
        state.scratch.rust_state_id = snapshot.state_id
        state.scratch.rust_parent_id = snapshot.parent_id

        # P10 fix: Restore state plugins from initial state template
        self._restore_plugins_to_state(state, snapshot.state_id)

        return state

    def _restore_plugins_to_state(self, state: "angr.SimState", state_id: int):
        """Restore plugins to an exported state from the initial state template.

        P10 fix: Exported states are missing critical plugins (posix, libc, heap)
        that scripts expect. This method restores them from the template state.

        Args:
            state: The state to restore plugins to.
            state_id: The Rust state ID for lookup.
        """
        # Find the initial state from cache or template
        template = None

        # Try to find root state ID
        root_id = self._state_roots.get(state_id, state_id)
        if root_id in self._state_cache:
            template = self._state_cache[root_id]

        # Fall back to any cached state for plugin extraction
        if template is None and self._state_cache:
            # Use the first cached state as template
            template = next(iter(self._state_cache.values()))

        if template is None:
            l.debug("P10: No template state found for plugin restoration")
            return

        # Copy plugins that are commonly needed
        plugins_to_restore = ['posix', 'libc', 'heap', 'fs', 'log']

        for plugin_name in plugins_to_restore:
            try:
                if hasattr(template, plugin_name):
                    plugin = getattr(template, plugin_name)
                    if plugin is not None and hasattr(plugin, 'copy'):
                        # Only copy if not already present
                        if not hasattr(state, plugin_name) or getattr(state, plugin_name) is None:
                            state.register_plugin(plugin_name, plugin.copy())
                            l.debug(f"P10: Restored {plugin_name} plugin to state")
            except Exception as e:
                l.debug(f"P10: Could not restore {plugin_name} plugin: {e}")

    def eval_memory(self, state_id: int, addr: int, size: int) -> Optional[bytes]:
        """Evaluate memory from a Rust state's solver context.

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
