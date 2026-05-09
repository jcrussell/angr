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


class RustSolverFallback:
    """Wraps state.solver so eval/min/max/satisfiable defer to the Rust solver.

    Holds a forked Rust Z3 context (lazily created, cached across calls) and
    the original Python solver methods. Each wrapper method tries the Rust
    solver first and falls back to Python on failure or when Rust returns
    None. Constraints added to the Python state after attach() (e.g.,
    flareon2015_5 hash equalities) are synced into the cached Rust context
    on the next call.
    """

    _ATTACH_FLAG = '_rust_fallback_attached'

    def __init__(self, state, state_id, rust_mgr):
        self._state = state
        self._state_id = state_id
        self._rust_mgr = rust_mgr
        self._original_eval = state.solver.eval
        self._original_eval_upto = state.solver.eval_upto
        self._original_min = state.solver.min
        self._original_max = state.solver.max
        self._original_satisfiable = state.solver.satisfiable
        # Cache the forked Rust solver: fork_state_solver() clones the Z3
        # solver (~3ms), so caching saves significant time when the solve
        # script calls eval() many times (ais3 ~100 byte evals).
        self._cached_rust_ctx = None
        # Track constraint count at attach time so we can detect when the
        # caller adds constraints post-exploration and replay them into Rust.
        n = len(state.solver.constraints)
        self._initial_constraint_count = n
        self._synced_constraint_count = n

    def attach(self):
        """Bind wrapper methods onto state.solver. No-op if already attached."""
        state = self._state
        state.scratch.rust_mgr = self._rust_mgr
        state.scratch.rust_found_state_id = self._state_id
        # Guard against double-patching (which would cause infinite recursion
        # since the second wrapper's "originals" would be the first wrapper).
        solver = state.solver
        if getattr(solver, self._ATTACH_FLAG, False):
            return
        setattr(solver, self._ATTACH_FLAG, True)
        solver.eval = self.eval
        solver.eval_upto = self.eval_upto
        solver.min = self.min
        solver.max = self.max
        solver.satisfiable = self.satisfiable

    def _get_rust_ctx(self):
        if self._cached_rust_ctx is None:
            ctx = self._rust_mgr.fork_state_solver(self._state_id)
            # Default 30s is too short for complex post-exploration solving (asisctf)
            try:
                ctx.set_timeout(120000)
            except Exception:
                # cat-(a) EXPECTED CONTROL FLOW: set_timeout is a best-effort
                # tuning knob; older Rust contexts may not expose it. Default
                # 30s timeout still applies.
                pass
            self._cached_rust_ctx = ctx
        # Replay constraints that the caller added after attach
        current = len(self._state.solver.constraints)
        if current != self._synced_constraint_count:
            new_constraints = self._state.solver.constraints[self._synced_constraint_count:]
            for c in new_constraints:
                try:
                    self._cached_rust_ctx.add_constraint_ast(c)
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: a post-attach constraint
                    # added by the caller could not be replayed into Rust's
                    # context (e.g., references a Python-only symbol). The
                    # next eval() will use a Rust solver missing this
                    # constraint and may return values inconsistent with
                    # the Python solver.
                    l.debug("Could not replay post-attach constraint into Rust ctx: %s", e)
            self._synced_constraint_count = current
        return self._cached_rust_ctx

    def _rust_eval(self, expr, cast_to):
        rust_ctx = self._get_rust_ctx()
        result = rust_ctx.eval(expr)
        if result is not None:
            if cast_to == bytes:
                nbytes = (expr.length + 7) // 8
                raw = result.to_bytes(nbytes, 'little')
                return raw[::-1]
            return result
        # Wide BVS (e.g. 160-bit flag): Rust eval returns None because the
        # full symbol isn't in Rust's table. Decompose into byte-sized evals.
        if hasattr(expr, 'length') and expr.length > 64:
            nbytes = expr.length // 8
            byte_vals = []
            for i in range(nbytes):
                hi = expr.length - 1 - i * 8
                lo = hi - 7
                byte_expr = claripy.Extract(hi, lo, expr)
                byte_result = rust_ctx.eval(byte_expr)
                if byte_result is None:
                    return None
                byte_vals.append(byte_result & 0xFF)
            value = 0
            for bv in byte_vals:
                value = (value << 8) | bv
            if cast_to == bytes:
                return value.to_bytes(nbytes, 'big')
            return value
        return None

    def eval(self, expr, cast_to=None, **kwargs):
        try:
            result = self._rust_eval(expr, cast_to)
            if result is not None:
                return result
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: Rust eval failed; Python solver
            # is the fallback and has the same constraints (constraint sync
            # ran during exploration), so no wrong-answer risk.
            l.debug("Rust eval failed, falling back to Python: %s", e)
        return self._original_eval(expr, cast_to=cast_to, **kwargs)

    def eval_upto(self, expr, n, cast_to=None, **kwargs):
        try:
            rust_ctx = self._get_rust_ctx()
            results = rust_ctx.eval_upto(expr, n)
            if results:
                if cast_to == bytes:
                    nbytes = (expr.length + 7) // 8
                    return [r.to_bytes(nbytes, 'big') for r in results]
                return list(results)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: Rust eval_upto failed; Python
            # solver produces the result. Same constraint set, so no
            # wrong-answer risk.
            l.debug("Rust eval_upto failed, falling back to Python: %s", e)
        return self._original_eval_upto(expr, n, cast_to=cast_to, **kwargs)

    def min(self, expr, **kwargs):
        try:
            rust_ctx = self._get_rust_ctx()
            result = rust_ctx.min(expr, signed=kwargs.get('signed', False))
            if result is not None:
                return result
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: Rust min failed; Python solver
            # is the fallback. Same constraint set.
            l.debug("Rust min failed, falling back to Python: %s", e)
        return self._original_min(expr, **kwargs)

    def max(self, expr, **kwargs):
        try:
            rust_ctx = self._get_rust_ctx()
            result = rust_ctx.max(expr, signed=kwargs.get('signed', False))
            if result is not None:
                return result
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: Rust max failed; Python solver
            # is the fallback. Same constraint set.
            l.debug("Rust max failed, falling back to Python: %s", e)
        return self._original_max(expr, **kwargs)

    def satisfiable(self, **kwargs):
        try:
            rust_ctx = self._get_rust_ctx()
            return rust_ctx.satisfiable()
        except Exception as e_rust:
            # cat-(b) FALLBACK WITH LOSS: Rust solver failed; Python is
            # tried next.
            l.debug("Rust satisfiable() failed, trying Python: %s", e_rust)
        try:
            return self._original_satisfiable(**kwargs)
        except Exception as e_py:
            # cat-(c) WRONG-ANSWER RISK: returning False here would mean
            # "UNSAT" — a wrong answer that masks the underlying solver
            # failure. Re-raise so the caller sees the failure.
            l.warning("Both Rust and Python satisfiable() failed for "
                      "state — Python error: %s", e_py)
            raise


class RustStateExportMixin:
    """Mixin providing state export/conversion methods for RustExplorationManager.

    This mixin expects the host class to have:
    - self._rust_mgr: The Rust exploration manager PyO3 object
    - self._project: The angr Project
    - self._state_cache: Dict[int, SimState]
    - self._state_roots: Dict[int, int]
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
                self._inject_rust_stdout(state, state_id)
                self._inject_rust_stdin(state, state_id)
                # Attach Rust solver fallback BEFORE constraint sync.
                # The Rust solver has the correct constraints from exploration.
                # Constraint sync is expensive (5.9s for sym-write) and often
                # causes identity mismatches. The fallback handles eval/eval_upto/
                # min/max/satisfiable directly via Rust solver.
                self._attach_rust_solver_fallback(state, state_id)
                # Sync concrete memory from Rust to Python state so that
                # memory modified during Rust execution is visible to the user
                self._sync_rust_memory_to_state(state, state_id)
                # Sync all registers from Rust state to Python state so that
                # register values computed during Rust execution are visible
                # (e.g., rdi holding a computed flag address in asisctf).
                self._sync_rust_registers_to_state(state, state_id)
                # Sync Rust-tracked call frames so state.callstack reflects
                # the call/ret events that happened during Rust execution.
                self._sync_rust_callstack_to_state(state, state_id)
                self._sync_rust_mmap_base_to_state(state, state_id)
                self._sync_rust_posix_brk_to_state(state, state_id)
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
                        # cat-(a) EXPECTED CONTROL FLOW: probing for a Rust
                        # root id; absence is normal for entry / unparented
                        # states. Caller falls through to other lookups.
                        pass
                if root is not None and root in self._state_cache:
                    state = self._state_cache[root].copy()
                    self._restore_plugins_to_state(state, sid)
                    self._inject_rust_stdout(state, sid)
                    self._inject_rust_stdin(state, sid)
                    # Skip _sync_exported_constraints — Rust solver fallback
                    # handles all solver operations directly.
                    self._attach_rust_solver_fallback(state, sid)
                    # Sync memory and registers from Rust (state was copied from
                    # root, so it doesn't have Rust-computed values yet)
                    self._sync_rust_memory_to_state(state, sid)
                    self._sync_rust_registers_to_state(state, sid)
                    self._sync_rust_callstack_to_state(state, sid)
                    self._sync_rust_mmap_base_to_state(state, sid)
                    self._sync_rust_posix_brk_to_state(state, sid)
                    # Cache the copy so it stays alive (prevents weakref death
                    # during chained attribute access like sm.active[1].posix.dumps())
                    self._state_cache[sid] = state
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
                    self._inject_rust_stdout(state, sid)
                    self._inject_rust_stdin(state, sid)
                    # Skip _sync_exported_constraints — Rust solver fallback
                    # handles all solver operations directly.
                    self._attach_rust_solver_fallback(state, sid)
                    self._sync_rust_memory_to_state(state, sid)
                    self._sync_rust_registers_to_state(state, sid)
                    self._sync_rust_callstack_to_state(state, sid)
                    self._sync_rust_mmap_base_to_state(state, sid)
                    self._sync_rust_posix_brk_to_state(state, sid)
                    self._state_cache[sid] = state
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
                                self._inject_rust_stdout(angr_state, snapshot.state_id)
                                self._inject_rust_stdin(angr_state, snapshot.state_id)
                                # Skip _sync_exported_constraints — it's O(n^2) on
                                # constraint ASTs (5.9s for sym-write) and causes
                                # identity mismatches. The Rust solver fallback
                                # handles eval/eval_upto/min/max/satisfiable via
                                # Rust's Z3 solver which already has the correct
                                # constraints from exploration.
                                self._attach_rust_solver_fallback(angr_state, snapshot.state_id)
                                self._sync_rust_memory_to_state(angr_state, snapshot.state_id)
                                self._sync_rust_registers_to_state(angr_state, snapshot.state_id)
                                self._sync_rust_callstack_to_state(angr_state, snapshot.state_id)
                                self._sync_rust_mmap_base_to_state(angr_state, snapshot.state_id)
                                self._sync_rust_posix_brk_to_state(angr_state, snapshot.state_id)
                                self._state_cache[snapshot.state_id] = angr_state
                                states.append(angr_state)
                            except Exception as e:
                                # cat-(c) WRONG-ANSWER RISK: a state in
                                # the Rust stash is dropped from the
                                # Python-visible result. The user expects
                                # N states and sees fewer. WARN ensures
                                # the missing state is observable.
                                l.warning(f"Failed to convert state from {stash}: {e}")
                except Exception as e:
                    # cat-(c) WRONG-ANSWER RISK: entire export_stash() FFI
                    # failed; *all* uncached states are dropped from the
                    # result. Caller will see fewer states than the Rust
                    # mgr reports.
                    l.warning(f"export_stash failed for {stash}: {e}")

        # Restore stdin content for states that were forked purely in Rust.
        # These states have empty posix.stdin.content because they never went
        # through a SimProcedure callback in Python. The _stdin_content tracks
        # BVS packets captured during callbacks (fgets, read, etc.).
        if self._stdin_content:
            for s in states:
                try:
                    posix = getattr(s, 'posix', None)
                    if posix is None:
                        continue
                    stdin = getattr(posix, 'stdin', None)
                    if stdin is None or not hasattr(stdin, 'content'):
                        continue
                    if not stdin.content:
                        stdin.content = list(self._stdin_content)
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: stdin restore best-effort;
                    # state's posix plugin shape may differ from the template
                    # (e.g., custom posix subclass). Original (empty) content
                    # remains; user solving for stdin won't see the captured
                    # packets but the exploration result is still valid.
                    l.debug("Could not restore stdin content for state: %s", e)

        # Fix posix nested weakrefs on all returned states
        # This ensures stdin/stdout/stderr have valid state references
        # regardless of which code path created the state
        for s in states:
            self._fix_posix_weakrefs(s)

        return states

    @staticmethod
    def _fix_posix_weakrefs(state):
        """Fix weakrefs in posix plugin's nested stream objects.

        After state caching, copying, or export, the weakrefs from
        SimPacketsStream objects (stdin/stdout/stderr) to their parent
        state can become stale. This refreshes them.
        """
        try:
            posix = getattr(state, 'posix', None)
            if posix is None:
                return
            for attr in ('stdin', 'stdout', 'stderr'):
                child = getattr(posix, attr, None)
                if child is not None and hasattr(child, 'set_state'):
                    child.set_state(state)
            if hasattr(posix, 'fd') and posix.fd:
                for fd_obj in posix.fd.values():
                    if fd_obj is not None and hasattr(fd_obj, 'set_state'):
                        fd_obj.set_state(state)
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: states without posix plugin
            # (custom factories, blank states) hit attribute / type errors
            # that we silently absorb. Weakref refresh is fix-up only.
            pass

    def _sync_rust_registers_to_state(self, state: "angr.SimState", state_id: int):
        """Sync concrete register values from Rust state to Python state.

        After Rust executes code, register values computed during execution
        are only in Rust's state. This exports named registers and overwrites
        the Python state's values for registers that are concrete in Rust.
        """
        try:
            snapshot = self._rust_mgr.export_state(state_id)
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: registers in Python state are now
            # stale. User code reading state.regs.* after exploration will
            # see pre-exploration values, not the values Rust computed.
            l.warning("export_state(%d) failed during register sync: %s — "
                      "Python registers may be stale", state_id, e)
            return

        named_regs = snapshot.get_registers_named()
        for reg_name, (value, size_bits) in named_regs.items():
            try:
                setattr(state.regs, reg_name, claripy.BVV(value, size_bits))
            except Exception as e:
                # cat-(a) EXPECTED CONTROL FLOW: VEX internal registers
                # (e.g. ip_at_syscall) that angr's register plugin doesn't
                # expose. Log at debug only.
                l.debug("Skipping register %s during sync: %s", reg_name, e)

    def _sync_rust_mmap_base_to_state(self, state: "angr.SimState", state_id: int):
        """Push Rust's per-state mmap_base into Python's state.heap.mmap_base.

        The native mmap syscall handler bumps Rust's mmap_base on addr=0 calls;
        without this sync, a subsequent Python-side allocation (SimProcedure
        fallback or unhandled syscall) reads a stale state.heap.mmap_base and
        hands out an address that overlaps a Rust-allocated region.

        Take max(rust, python) to avoid clobbering a Python-side advance that
        happened between Rust runs (e.g. the user calling state.heap.mmap_base
        = N before re-entering exploration).
        """
        heap = getattr(state, "heap", None)
        if heap is None or not hasattr(heap, "mmap_base"):
            return
        try:
            rust_base = self._rust_mgr.get_state_mmap_base(state_id)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: per-state mmap_base unavailable
            # (state may have been dropped from Rust). Python keeps its
            # current heap.mmap_base; a subsequent Python-side allocation
            # may overlap a Rust-allocated region. Debug only because
            # this state is likely no longer being explored.
            l.debug("get_state_mmap_base(%d) failed: %s", state_id, e)
            return
        if rust_base > heap.mmap_base:
            heap.mmap_base = rust_base

    def _sync_rust_posix_brk_to_state(self, state: "angr.SimState", state_id: int):
        """Push Rust's per-state posix_brk into Python's state.posix.brk.

        Mirror of _sync_rust_mmap_base_to_state for the brk(2) heap pointer.
        NativeBrkSyscall bumps Rust's posix_brk on concrete grow calls; without
        this sync, a Python-side fallback (symbolic brk or set_brk collision
        retry) reads a stale state.posix.brk and hands out heap addresses
        overlapping a Rust-allocated region.

        Only applied when state.posix.brk is a plain int — Python's set_brk
        rewrites the field as a claripy BV (concrete BVV after a concrete
        bump, an If(...) tree after a symbolic bump). In those cases Python
        already advanced past the default and we leave the BV alone rather
        than risk an int<->BV type mismatch in downstream Python code.

        Take max(rust, python) to avoid clobbering a Python-side advance.
        """
        posix = getattr(state, "posix", None)
        if posix is None or not hasattr(posix, "brk"):
            return
        py_brk = posix.brk
        if not isinstance(py_brk, int):
            return
        try:
            rust_brk = self._rust_mgr.get_state_posix_brk(state_id)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: per-state posix_brk unavailable.
            # Same risk as mmap_base sync — Python-side fallback may
            # overlap a Rust heap region.
            l.debug("get_state_posix_brk(%d) failed: %s", state_id, e)
            return
        if rust_brk > py_brk:
            posix.brk = rust_brk

    def _sync_rust_callstack_to_state(self, state: "angr.SimState", state_id: int):
        """Sync Rust-tracked call frames into state.callstack.

        Rust's call_stack (push order, outermost first) is the source of truth
        for any call/ret that happened during Rust execution. The Python state
        was forked from a template before exploration, so its CallStack plugin
        does not reflect Rust-side push/pop. This rebuilds state.callstack as a
        linked list (top = most recent Rust call) and replaces the plugin via
        register_plugin. No-op if Rust has zero frames (preserves the
        template's empty CallStack so we don't clobber pre-Rust call history
        for forks made from non-entry states).
        """
        try:
            frames = self._rust_mgr.get_state_call_stack(state_id)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: state.callstack stays at the
            # template's pre-Rust history. Downstream callers reading
            # callstack will see fewer frames than Rust observed.
            l.debug("get_state_call_stack(%d) failed: %s", state_id, e)
            return

        if not frames:
            return

        from angr.state_plugins.callstack import CallStack

        chain = CallStack()
        for call_site, func, ret_addr, sp in frames:
            chain = CallStack(
                call_site_addr=call_site,
                func_addr=func,
                stack_ptr=sp,
                ret_addr=ret_addr,
                jumpkind="Ijk_Call",
                next_frame=chain,
            )

        try:
            state.register_plugin("callstack", chain)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: register_plugin failed; the
            # state keeps its pre-Rust callstack plugin. Downstream
            # consumers that depend on Rust-tracked frames will see
            # incomplete history.
            l.debug("register_plugin('callstack') failed for %d: %s", state_id, e)

    def _sync_rust_memory_to_state(self, state: "angr.SimState", state_id: int):
        """Sync memory from Rust state to Python state.

        After Rust executes code, memory modified during execution is only in
        Rust's memory. This method exports the Rust state's memory pages and
        symbolic expressions, and applies them to the Python state so that
        state.memory.load() returns current values.
        """
        try:
            # Use flushed export to materialize any pending symbolic writes
            # before exporting memory pages to Python.
            snapshot = self._rust_mgr.export_state_flushed(state_id)
        except Exception as e_flushed:
            try:
                snapshot = self._rust_mgr.export_state(state_id)
                # cat-(c) WRONG-ANSWER RISK: flushed failed but unflushed
                # worked — pending symbolic writes may not be visible in
                # Python memory. WARN so the user notices.
                l.warning("export_state_flushed(%d) failed (%s); falling "
                          "back to unflushed snapshot — pending symbolic "
                          "stores may be missing", state_id, e_flushed)
            except Exception as e_unflushed:
                # cat-(c) WRONG-ANSWER RISK: both export paths failed.
                # Python state.memory will return zeros (or stale data)
                # for any address Rust wrote.
                l.warning("Both export_state_flushed and export_state failed "
                          "for state %d (flushed: %s; unflushed: %s) — "
                          "Python memory will be stale",
                          state_id, e_flushed, e_unflushed)
                return  # State may not be in Rust stashes anymore

        for i in range(snapshot.page_count()):
            page = snapshot.get_page(i)
            if page is None:
                continue
            page_addr, data, _perms, symbolic_offsets = page
            if symbolic_offsets:
                # Write concrete byte ranges while preserving symbolic regions.
                # symbolic_offsets are byte positions within the page where Python
                # has BVS values — writing zeros there would corrupt them.
                sym_set = set(symbolic_offsets)
                # Find contiguous concrete runs
                start = None
                for j in range(len(data) + 1):
                    if j < len(data) and j not in sym_set:
                        if start is None:
                            start = j
                    else:
                        if start is not None:
                            chunk = data[start:j]
                            try:
                                state.memory.store(
                                    page_addr + start,
                                    claripy.BVV(chunk, len(chunk) * 8),
                                    endness="Iend_BE",
                                    inspect=False,
                                    disable_actions=True,
                                )
                            except Exception as e:
                                # cat-(c) WRONG-ANSWER RISK: a concrete
                                # byte range from Rust failed to write into
                                # Python memory. state.memory.load() of
                                # this region will return whatever the
                                # Python state held pre-exploration (often
                                # zero or a stale BVS).
                                l.warning("Failed to sync concrete chunk to 0x%x: %s",
                                          page_addr + start, e)
                            start = None
            else:
                try:
                    # Raw bytes from Rust are in memory order; use Iend_BE so angr
                    # stores them as-is without byte-reversing.
                    state.memory.store(
                        page_addr,
                        claripy.BVV(data, len(data) * 8),
                        endness="Iend_BE",
                        inspect=False,
                        disable_actions=True,
                    )
                except Exception as e:
                    # cat-(c) WRONG-ANSWER RISK: full-page sync failed.
                    # All addresses on this page in Python memory are
                    # stale.
                    l.warning("Failed to sync page at 0x%x: %s", page_addr, e)

        # Export Rust-computed symbolic expressions to Python memory.
        # These are Expression values computed by the Rust VEX interpreter
        # (e.g., flag computations in asisctf). Python-imported BVS values
        # are excluded — they already have proper claripy identity in Python.
        self._sync_rust_symbolic_objects_to_state(state, state_id)

    def _sync_rust_symbolic_objects_to_state(self, state: "angr.SimState", state_id: int):
        """Export Rust-computed symbolic expressions as claripy ASTs into Python memory.

        Uses the shared Z3 context: Rust builds Z3 ASTs in Python's Z3 context,
        so we can directly wrap the raw Z3_ast pointers as z3.BitVecRef objects
        and convert them to claripy ASTs via claripy.backends.z3._abstract().
        """
        try:
            sym_asts = self._rust_mgr.get_state_symbolic_z3_asts(state_id)
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: Rust-computed symbolic objects
            # are not exported to Python memory. state.memory.load() of
            # those addresses will see concrete bytes (or template BVS),
            # not the symbolic expression Rust computed.
            l.warning("get_state_symbolic_z3_asts(%d) failed: %s — "
                      "Rust-side symbolic memory not synced", state_id, e)
            return

        if not sym_asts:
            return

        import z3 as z3mod
        import ctypes

        z3_backend = claripy.backends.z3
        z3_ctx = z3_backend._context  # The shared Z3 context

        for addr, z3_ast_ptr, width_bits in sym_asts:
            try:
                # Wrap raw Z3_ast pointer as z3.BitVecRef
                ast_wrapper = ctypes.c_void_p(z3_ast_ptr)
                ast_wrapper.__class__ = z3mod.z3types.Ast
                ast_wrapper._as_parameter_ = z3_ast_ptr
                z3_bv = z3mod.BitVecRef(ast_wrapper, z3_ctx)

                # Convert to claripy AST
                claripy_ast = z3_backend._abstract(z3_bv)

                # Store using the architecture's endianness to match VEX IR
                # store operations (STle for x86-64, etc.)
                state.memory.store(
                    addr,
                    claripy_ast,
                    endness=state.arch.memory_endness,
                    inspect=False,
                    disable_actions=True,
                )
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: per-AST sync failed (e.g.
                # ctypes binding incompatible with this z3 version). The
                # other ASTs on this state still get synced, but this one
                # is missing — Python sees concrete bytes for the address.
                l.debug("Failed to sync symbolic object at 0x%x: %s", addr, e)

    def _attach_rust_solver_fallback(self, state, state_id):
        """Wrap state.solver so eval/min/max/satisfiable defer to the Rust solver.

        The Rust solver holds the authoritative constraints from exploration;
        syncing them to Python is expensive and can cause identity mismatches.
        Re-attaching the same state is a no-op.
        """
        RustSolverFallback(state, state_id, self._rust_mgr).attach()

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
                    # cat-(a) EXPECTED CONTROL FLOW: probing for a Rust
                    # root id; absence is normal for entry / unparented
                    # states.
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
                if lookup_id is None:
                    continue
                addr_map = self._rust_mgr.get_state_addr_to_ast(lookup_id)
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
                            # cat-(b) FALLBACK WITH LOSS: rust_sym_*
                            # substitution failed (claripy AST shape
                            # mismatch). Constraint is added as-is and
                            # its rust_sym_ leaf may not match the
                            # Python AST identity, leading to a downstream
                            # identity-mismatch skip.
                            pass

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
                    # cat-(b) FALLBACK WITH LOSS: this individual
                    # constraint failed to add (rare malformed AST).
                    # Other constraints still sync. The Rust solver
                    # fallback supplies eval/satisfiable so the missing
                    # constraint is not a wrong-answer source — Python
                    # solver path may diverge but is not used.
                    pass
            if synced or skipped:
                l.debug(f"Synced {synced} constraints to state {state_id} "
                        f"({skipped} skipped for identity/register)")

            # Post-sync UNSAT check removed — it was a diagnostic that
            # called satisfiable() on the full constraint set, which for
            # LAZY_SOLVES examples (hackcon) takes 60s+ with zero benefit.
            # The Rust solver fallback handles eval() correctly regardless.
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: outer constraint sync failed.
            # The Rust solver fallback (attached separately by the caller)
            # serves eval/min/max/satisfiable from Rust's full constraint
            # set, so missing Python-side constraints are not a wrong-
            # answer source.
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
                    # cat-(a) EXPECTED CONTROL FLOW: probing for Rust root.
                    pass

            for lookup_id in [state_id, root_id]:
                if lookup_id is None:
                    continue
                addr_map = self._rust_mgr.get_state_addr_to_ast(lookup_id)
                for addr, (ast, size) in addr_map.items():
                    try:
                        concrete_bytes = self._rust_mgr.get_state_memory(
                            state_id, addr, size)
                        if concrete_bytes is not None:
                            concrete_val = int.from_bytes(concrete_bytes, 'little')
                            fresh.solver.add(ast == claripy.BVV(concrete_val, size * 8))
                    except Exception:
                        # cat-(b) FALLBACK WITH LOSS: per-symbol pin
                        # failed (memory read or constraint add). Other
                        # pins still apply but this symbol is left
                        # unconstrained — eval() may return arbitrary
                        # solutions where Rust had a concrete answer.
                        pass

            # Replace the original state's internals
            state.memory = fresh.memory
            state.solver = fresh.solver
            if hasattr(fresh, '_ip'):
                state.regs._ip = fresh.addr
            l.debug(f"Replaced UNSAT state {state_id} with pinned Rust values")
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: replacement failed entirely; the
            # caller-supplied state is unchanged and may still be UNSAT,
            # producing wrong eval() results downstream.
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
                # cat-(c) WRONG-ANSWER RISK: a found state is dropped
                # from the result list. Caller may report fewer
                # solutions than the engine actually found.
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
            # cat-(c) WRONG-ANSWER RISK: returning None hides the
            # underlying failure. Callers checking "if state is None"
            # may treat this as "state doesn't exist" when in fact
            # the state exists but failed to convert.
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
        state = self._project.factory.blank_state(addr=snapshot.pc)

        self._load_snapshot_registers(state, snapshot)
        self._load_snapshot_pages(state, snapshot, self._project.arch)

        state.scratch.rust_state_id = snapshot.state_id
        state.scratch.rust_parent_id = snapshot.parent_id

        self._restore_plugins_to_state(state, snapshot.state_id)
        return state

    def _load_snapshot_registers(self, state: "angr.SimState", snapshot):
        """Restore register values from snapshot using Rust's named register export."""
        named_regs = snapshot.get_registers_named()
        for reg_name, (value, size_bits) in named_regs.items():
            try:
                setattr(state.regs, reg_name, claripy.BVV(value, size_bits))
            except Exception:
                # cat-(a) EXPECTED CONTROL FLOW: Skip VEX internal
                # registers that angr doesn't expose (ip_at_syscall etc.).
                pass

    def _load_snapshot_pages(self, state: "angr.SimState", snapshot, arch):
        """Load memory pages from snapshot, restoring symbolic regions with original ASTs."""
        for i in range(snapshot.page_count()):
            page = snapshot.get_page(i)
            if page is None:
                continue
            page_addr, data, _perms, symbolic_offsets = page
            try:
                state.memory.store(page_addr, claripy.BVV(data, len(data) * 8),
                                   endness=arch.memory_endness,
                                   inspect=False)
                if not symbolic_offsets:
                    continue
                regions = self._find_contiguous_regions(symbolic_offsets)
                sym_to_constrain = self._restore_symbolic_regions(
                    state, snapshot, page_addr, regions, arch,
                )
                self._apply_symbolic_constraints(state, snapshot.state_id, sym_to_constrain)
            except Exception as e:
                # cat-(c) WRONG-ANSWER RISK: page failed to load. State
                # memory at this page is left at its blank-state default,
                # so loads return zeros instead of Rust's values.
                l.warning(f"Failed to load page at 0x{page_addr:x}: {e}")

    @staticmethod
    def _find_contiguous_regions(symbolic_offsets) -> list:
        """Group sorted byte offsets into contiguous (offset, size) regions."""
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
        return regions

    def _restore_symbolic_regions(self, state, snapshot, page_addr, regions, arch) -> list:
        """Overwrite symbolic regions on a page with recovered or fresh symbols.

        Returns a list of (sym_addr, size, ast) for downstream constraint sync.
        """
        sym_to_constrain = []
        for offset, size in regions:
            sym_addr = page_addr + offset
            ast = self._recover_symbolic_ast(snapshot, sym_addr, size)
            if ast is None:
                # Fallback for symbols created in Rust without a tracked AST
                sym_name = f"rust_sym_{sym_addr:x}_{snapshot.state_id}"
                ast = claripy.BVS(sym_name, size * 8)
            state.memory.store(sym_addr, ast,
                               endness=arch.memory_endness,
                               inspect=False)
            sym_to_constrain.append((sym_addr, size, ast))
        return sym_to_constrain

    def _recover_symbolic_ast(self, snapshot, sym_addr: int, size: int):
        """Look up the original claripy AST for a symbolic byte at sym_addr.

        Searches address-tracked AST maps for the snapshot's state, parent, and
        root, then the hook-symbolic-memory map. Returns the first AST whose
        tracked size matches; None if no match.
        """
        candidate_ids = [snapshot.state_id]
        if snapshot.parent_id >= 0:
            candidate_ids.append(snapshot.parent_id)
        root_id = self._state_roots.get(snapshot.state_id)
        if root_id is None:
            try:
                root_id = self._rust_mgr.get_state_root(snapshot.state_id)
            except Exception:
                # cat-(a) EXPECTED CONTROL FLOW: probing for Rust root id;
                # absence is normal for entry / unparented states. We
                # then search only state_id and parent_id for the AST.
                root_id = None
        if root_id is not None and root_id != snapshot.state_id:
            candidate_ids.append(root_id)

        for sid in candidate_ids:
            if sid is None:
                continue
            addr_map = self._rust_mgr.get_state_addr_to_ast(sid)
            entry = addr_map.get(sym_addr)
            if entry is not None and entry[1] == size:
                return entry[0]

        hook_map = self._rust_mgr.get_state_hook_symbolic_memory(snapshot.state_id)
        hook_entry = hook_map.get(sym_addr)
        if hook_entry is not None and hook_entry[1] == size:
            return hook_entry[0]
        return None

    def _apply_symbolic_constraints(self, state, state_id: int, sym_to_constrain: list):
        """Pin recovered symbolic values to their concrete Rust evaluations."""
        for sym_addr, size, ast in sym_to_constrain:
            try:
                concrete_bytes = self._rust_mgr.get_state_memory(state_id, sym_addr, size)
                if concrete_bytes is None:
                    continue
                concrete_val = int.from_bytes(concrete_bytes, 'little')
                state.solver.add(ast == claripy.BVV(concrete_val, size * 8))
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: per-symbol pin failed; the
                # symbol is left unconstrained while concrete bytes for
                # the address are still loaded by the page sync. The
                # solver may pick a different value than what Rust held.
                l.debug(f"Could not add constraint at 0x{sym_addr:x}: {e}")

    def _restore_plugins_to_state(self, state: "angr.SimState", state_id: int):
        """Restore plugins to an exported state from the initial state template.

        Exported states are missing critical plugins (posix, libc, heap)
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
                # cat-(a) EXPECTED CONTROL FLOW: probing for Rust root;
                # falls through to the next-state-cache template.
                pass
        if root_id in self._state_cache:
            template = self._state_cache[root_id]

        # Fall back to any cached state for plugin extraction
        if template is None and self._state_cache:
            # Use the first cached state as template
            template = next(iter(self._state_cache.values()))

        if template is None:
            l.debug("No template state found for plugin restoration")
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
                            l.debug(f"Restored {plugin_name} plugin to state")
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: per-plugin restore failed
                # (incompatible copy()). State is left without that
                # plugin; downstream code that accesses it will fall
                # through to angr's default plugin instantiation.
                l.debug(f"Could not restore {plugin_name} plugin: {e}")

        # Fix nested plugin state references for posix (stdin/stdout/stderr)
        # These nested SimPacketsStream objects hold weakrefs to the state that
        # can become stale after state caching/export cycles
        try:
            posix = getattr(state, 'posix', None)
            if posix is not None:
                for attr in ('stdin', 'stdout', 'stderr'):
                    child = getattr(posix, attr, None)
                    if child is not None and hasattr(child, 'set_state'):
                        child.set_state(state)
                # Also fix fd entries
                if hasattr(posix, 'fd') and posix.fd:
                    for fd_obj in posix.fd.values():
                        if fd_obj is not None and hasattr(fd_obj, 'set_state'):
                            fd_obj.set_state(state)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: posix weakref refresh failed
            # (custom posix plugin shape). Stale weakrefs may surface
            # later as AttributeError when the user inspects stdin/
            # stdout/stderr.
            l.debug(f"Could not fix posix nested state refs: {e}")

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
