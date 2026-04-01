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
                self._attach_rust_solver_fallback(state, state_id)
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
                if root is None:
                    try:
                        root = self._rust_mgr.get_state_root(sid)
                    except Exception:
                        pass
                if root is not None and root in self._state_cache:
                    state = self._state_cache[root].copy()
                    self._restore_plugins_to_state(state, sid)
                    self._sync_exported_constraints(state, sid)
                    self._attach_rust_solver_fallback(state, sid)
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
                    self._attach_rust_solver_fallback(state, sid)
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
                                self._attach_rust_solver_fallback(angr_state, snapshot.state_id)
                                states.append(angr_state)
                            except Exception as e:
                                l.warning(f"Failed to convert state from {stash}: {e}")
                except Exception as e:
                    l.warning(f"export_stash failed for {stash}: {e}")

        return states

    def _attach_rust_solver_fallback(self, state, state_id):
        """Monkey-patch state.solver.eval to fallback to Rust solver on UNSAT.

        When Python constraint sync creates UNSAT due to variable identity
        mismatches, the Rust solver (which has the correct answer) is used
        as a fallback for eval() calls.
        """
        import types

        rust_mgr = self._rust_mgr
        state.scratch.rust_mgr = rust_mgr
        state.scratch.rust_found_state_id = state_id

        original_eval = state.solver.eval

        def eval_with_fallback(expr, cast_to=None, **kwargs):
            try:
                return original_eval(expr, cast_to=cast_to, **kwargs)
            except Exception as orig_err:
                # Python solver failed (likely UNSAT), try Rust solver
                try:
                    rust_ctx = rust_mgr.fork_state_solver(state_id)
                    result = rust_ctx.eval(expr)
                    if result is not None:
                        if cast_to == bytes:
                            nbytes = (expr.length + 7) // 8
                            # Rust solver returns LE bytes; reverse for BE user variables
                            raw = result.to_bytes(nbytes, 'little')
                            return raw[::-1]
                        return (result,) if isinstance(result, int) else result
                except Exception:
                    pass
                raise orig_err  # Re-raise original if Rust also fails

        state.solver.eval = eval_with_fallback

    def _sync_exported_constraints(self, state, state_id):
        """Sync constraints from Rust solver to Python state.

        Adds Rust-exported constraints to the Python state. Skips register
        constraints and constraints that use symbols with different identity
        than existing Python symbols (which would cause UNSAT).

        Note: if constraint sync causes UNSAT due to identity mismatches,
        the Rust solver fallback in _attach_rust_solver_fallback handles
        eval() calls by delegating to Rust's Z3 solver directly.
        """
        try:
            root_id = self._state_roots.get(state_id, state_id)
            if root_id == state_id:
                try:
                    rust_root = self._rust_mgr.get_state_root(state_id)
                    if rust_root is not None:
                        root_id = rust_root
                except Exception:
                    pass

            rust_constraints = self._rust_mgr.export_state_constraints(state_id)
            synced = 0
            skipped = 0

            existing_leaves = {}
            for c in state.solver.constraints:
                for leaf in c.leaf_asts():
                    if hasattr(leaf, 'args') and len(leaf.args) > 0 and isinstance(leaf.args[0], str):
                        existing_leaves[leaf.args[0]] = leaf

            # Build substitution map: rust_sym_ADDR → original AST
            rust_sym_to_original = {}
            for lookup_id in [state_id, root_id]:
                addr_map = self._addr_to_ast.get(lookup_id, {})
                for addr, (ast, size) in addr_map.items():
                    rust_name = f"rust_sym_{addr:x}"
                    rust_sym_to_original[rust_name] = ast

            for c in rust_constraints:
                if c is None:
                    continue
                try:
                    c_str = str(c)
                    # Skip register-related constraints
                    if 'reg_' in c_str:
                        skipped += 1
                        continue

                    # Replace rust_sym_XXXX references with original ASTs
                    # This is critical for constraint identity unification
                    if rust_sym_to_original and 'rust_sym_' in c_str:
                        try:
                            for rust_name, orig_ast in rust_sym_to_original.items():
                                for leaf in c.leaf_asts():
                                    if hasattr(leaf, 'args') and len(leaf.args) > 0:
                                        leaf_name = leaf.args[0] if isinstance(leaf.args[0], str) else ''
                                        if leaf_name.startswith(rust_name):
                                            # Substitute: replace rust symbol with original
                                            c = c.replace(leaf, orig_ast)
                                            break
                        except Exception:
                            pass  # Substitution failed, use constraint as-is

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

            # Post-sync UNSAT check: warn if constraints are contradictory
            if synced > 0:
                try:
                    if not state.solver.satisfiable():
                        l.warning(f"State {state_id} is UNSAT after constraint sync "
                                  f"({synced} synced, {skipped} skipped)")
                except Exception:
                    pass
        except Exception as e:
            l.debug(f"Could not sync constraints for state {state_id}: {e}")

    def _replace_with_rust_snapshot(self, state, state_id):
        """Fix a UNSAT state by clearing constraints and pinning symbolic values.

        When constraint sync creates UNSAT due to identity mismatches,
        this method creates a brand new blank state with the correct PC,
        copies plugins from the template, and adds pinning constraints
        that map original symbolic variables to their Rust-solved values.
        """
        try:
            # Build a fresh state from snapshot (has correct concrete memory)
            snapshot = self._rust_mgr.export_state(state_id)
            fresh = self._snapshot_to_angr(snapshot)
            self._restore_plugins_to_state(fresh, state_id)

            # Now pin original symbolic variables to their Rust concrete values
            root_id = self._state_roots.get(state_id, state_id)
            if root_id == state_id:
                try:
                    rust_root = self._rust_mgr.get_state_root(state_id)
                    if rust_root is not None:
                        root_id = rust_root
                except Exception:
                    pass

            for lookup_id in [state_id, root_id]:
                addr_map = self._addr_to_ast.get(lookup_id, {})
                for addr, (ast, size) in addr_map.items():
                    try:
                        concrete_bytes = self._rust_mgr.get_state_memory(
                            state_id, addr, size)
                        if concrete_bytes is not None:
                            concrete_val = int.from_bytes(concrete_bytes, 'little')
                            fresh.solver.add(ast == claripy.BVV(concrete_val, size * 8))
                    except Exception:
                        pass

            # Replace the original state's internals
            state.memory = fresh.memory
            state.solver = fresh.solver
            if hasattr(fresh, '_ip'):
                state.regs._ip = fresh.addr
            l.debug(f"Replaced UNSAT state {state_id} with pinned Rust values")
        except Exception as e:
            l.warning(f"Could not replace UNSAT state {state_id}: {e}")

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

                            # Also check parent and root states for inherited symbolic values
                            if original_ast is None and snapshot.parent_id >= 0:
                                parent_addr_map = self._addr_to_ast.get(snapshot.parent_id, {})
                                if sym_addr in parent_addr_map:
                                    tracked_ast, tracked_size = parent_addr_map[sym_addr]
                                    if tracked_size == size:
                                        original_ast = tracked_ast
                                        l.debug(f"Recovered original AST from parent at 0x{sym_addr:x}")

                            # Check root state (for deeply forked states)
                            if original_ast is None:
                                root_id = self._state_roots.get(snapshot.state_id)
                                if root_id is None:
                                    try:
                                        root_id = self._rust_mgr.get_state_root(snapshot.state_id)
                                    except Exception:
                                        pass
                                if root_id is not None and root_id != snapshot.state_id:
                                    root_addr_map = self._addr_to_ast.get(root_id, {})
                                    if sym_addr in root_addr_map:
                                        tracked_ast, tracked_size = root_addr_map[sym_addr]
                                        if tracked_size == size:
                                            original_ast = tracked_ast

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
        if root_id == state_id:
            try:
                rust_root = self._rust_mgr.get_state_root(state_id)
                if rust_root is not None:
                    root_id = rust_root
            except Exception:
                pass
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
