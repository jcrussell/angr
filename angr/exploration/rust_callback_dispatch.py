"""Mixin for Callback dispatch and SimProcedure handling for Rust exploration."""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import TYPE_CHECKING

import claripy

from angr.errors import AngrUnsupportedSyscallError, SimUnsatError
from angr.exploration.rust_identity import CallbackMemoryTracker
from angr.exploration.rust_state_export import POSIX_SYNC_FDS, _concrete_stream_len, _posix_stream
from angr.storage.file import SimFile, SimFileDescriptor

if TYPE_CHECKING:
    import angr

l = logging.getLogger(name=__name__)
_DBG = l.isEnabledFor(logging.DEBUG)


def _simproc_dispatch_name(proc) -> str:
    """Canonical name used to dispatch a SimProcedure to the Rust native registry.

    Prefer ``display_name`` over ``__class__.__name__``: SimLibrary instantiates
    unimplemented-libc stubs as ``ReturnUnconstrained(display_name="setenv")``,
    so keying off the class name would lump every stub under
    ``"ReturnUnconstrained"`` and the native registry (which keys off the
    symbol, e.g. ``"setenv"``) would never match. For first-class SimProcs the
    base class defaults ``display_name`` to ``type(self).__name__``, so this
    coincides with the previous behaviour.
    """
    name = getattr(proc, "display_name", None)
    if name:
        return str(name)
    return proc.__class__.__name__ if hasattr(proc, "__class__") else str(proc)


#: Set ``ANGR_BOUNCE_TRACE=1`` to have every SimProcedure park-and-bounce into
#: Python emit one ``[bounce] ...`` line on stderr. Off by default; the check is
#: a module-level bool so the hot path pays a single load. Consumed by
#: ``tests/benchmarks/zeropy_bounce_census.py`` (angr-gorvf.12) to attribute each
#: bounce to a proc name + PC + defining module without guessing from the binary.
_BOUNCE_TRACE = bool(os.environ.get("ANGR_BOUNCE_TRACE"))


def _emit_bounce_trace(name, addr, proc) -> None:
    """Emit one machine-parseable line per SimProcedure bounce into Python."""
    cls = type(proc).__name__ if proc is not None else "<unresolved>"
    module = type(proc).__module__ if proc is not None else "-"
    addr_s = f"0x{int(addr):x}" if addr is not None else "-"
    print(
        f"[bounce] name={name} addr={addr_s} cls={cls} module={module}",
        file=sys.stderr,
        flush=True,
    )


class RustCallbackDispatchMixin:
    """Callback dispatch and SimProcedure handling for Rust exploration

    This mixin expects the host class to have the standard
    RustExplorationManager attributes (self._rust_mgr, self._project, etc.).
    """

    def _init_callback_history(self, state: angr.SimState, event: _ExplorationEvent):
        """Initialize history for callback state to prevent IndexError.

        Python hooks often access `state.history.recent_bbl_addrs[-1]` which
        causes IndexError if history is empty. This method initializes history
        from the Rust pending state.

        Args:
            state: The angr callback state to initialize.
            event: The exploration event that triggered the callback.
        """
        # Use cached bundle data if available (avoids 2 extra FFI calls)
        rust_history = getattr(state.scratch, "_rust_bundle_history", None)
        rust_jumpkind = getattr(state.scratch, "_rust_bundle_jumpkind", None)
        if rust_history is None:
            try:
                rust_history, rust_jumpkind = self._rust_mgr.get_pending_history_and_jumpkind(event.callback_state_id)
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: pending Rust history unavailable;
                # fall back to a single-element history. State.history may be
                # shorter than truth — affects detailed_history-based predicates.
                # Debug-logs.
                if _DBG:
                    l.debug(f"Could not get Rust history: {e}")
                rust_history = []
                rust_jumpkind = "Ijk_Boring"
        if rust_jumpkind is None:
            rust_jumpkind = "Ijk_Boring"

        # Ensure history has at least one entry (the callback address)
        callback_addr = event.callback_addr or 0

        # Directly set recent_bbl_addrs to prevent IndexError
        # This is the critical fix - hooks access state.history.recent_bbl_addrs[-1]
        if hasattr(state.history, "recent_bbl_addrs"):
            # Use Rust history if available, otherwise use callback address.
            # Always ensure history has at least one entry, even if callback_addr is 0.
            if rust_history:
                state.history.recent_bbl_addrs = list(rust_history)
            else:
                # Use callback_addr, or state.addr if callback_addr is None/0
                addr_to_use = callback_addr or (state.addr or 0)
                state.history.recent_bbl_addrs = [addr_to_use]

        # Set jumpkind to avoid callstack._manage() pushing a new frame
        # Ijk_Boring prevents the callstack from being modified
        try:
            state.history.jumpkind = rust_jumpkind
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: jumpkind assignment failed (e.g.,
            # state.history is read-only); callstack._manage may push an
            # extra frame on the next step.
            pass

        if _DBG:
            l.debug(
                f"Initialized callback history with {len(state.history.recent_bbl_addrs)} entries, "
                f"jumpkind={rust_jumpkind}, addr=0x{callback_addr:x}"
            )

    def _init_callback_callstack(self, state: angr.SimState, event: _ExplorationEvent):
        """Initialize callstack for SimProcedure continuations.

        SimProcedures may need procedure_data for continuations (e.g., when
        using self.call()). This method initializes the required data.

        Check for stored procedure_data from a previous self.call() and
        restore it. This is critical for continuations like __libc_start_main
        which call init/fini functions and expect to resume with saved args.

        Args:
            state: The angr callback state to initialize.
            event: The exploration event that triggered the callback.
        """
        if event.callback_reason != "simprocedure":
            return

        addr = event.callback_addr
        # Ensure consistent int type for lookup
        addr_int = int(addr) if addr is not None else None

        # Check for stored procedure_data from a previous self.call().
        # This restores the full context (arguments, local vars) for continuations.
        if addr_int in self._pending_procedure_data:
            stored_data = self._pending_procedure_data.pop(addr_int)
            try:
                if hasattr(state.callstack, "top") and state.callstack.top is not None:
                    state.callstack.top.procedure_data = stored_data
                    l.debug(
                        f"Restored procedure_data for continuation at 0x{addr:x}: "
                        f"args={len(stored_data[1]) if len(stored_data) > 1 else 0}"
                    )
                    return
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: procedure_data restore failed;
                # continuation runs without the saved args/locals — likely
                # triggers the TypeError handler upstream. Debug-logs.
                l.debug(f"Could not restore procedure_data: {e}")

        # Fallback: Get SP from Rust for saved state
        try:
            sp_val = self._rust_mgr.get_pending_register(event.callback_state_id, "rsp")
            if sp_val is None:
                sp_val = self._rust_mgr.get_pending_register(event.callback_state_id, "esp")
            saved_sp = sp_val or 0
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: pending SP read failed;
            # fall back to saved_sp=0. Procedures that index off saved_sp
            # will see a wrong base.
            saved_sp = 0

        # Initialize procedure_data for continuations
        # This prevents crashes when SimProcedures use self.call()
        try:
            if hasattr(state.callstack, "top") and state.callstack.top is not None:
                # Set procedure_data: (saved_sp, sim_args, saved_local_vars, saved_lr, ideal_addr)
                state.callstack.top.procedure_data = (
                    saved_sp,  # saved_sp
                    [],  # sim_args (populated by SimProcedure)
                    [],  # saved_local_vars
                    None,  # saved_lr
                    addr,  # ideal_addr
                )
                l.debug(f"Initialized callstack procedure_data at 0x{addr:x}")
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: callstack procedure_data init
            # failed; future self.call() inside this procedure may crash on
            # missing fields. Debug-logs.
            l.debug(f"Could not initialize callstack procedure_data: {e}")

    def _install_rust_solver_on_callback_state(self, state: angr.SimState, rust_ctx):
        """Make the Rust solver the single source of truth for callback states.

        Instead of syncing constraints from Rust to Python (which can create
        UNSAT due to variable identity mismatches across the FFI boundary),
        this method monkey-patches the Python state's solver to delegate all
        solving operations to the forked Rust solver context.

        Contract (angr-ye2n): the caller MUST pass a non-None ``rust_ctx``
        AND attach the same context to ``state.scratch.rust_solver_ctx``
        before calling. The installed closures re-read
        ``state.scratch.rust_solver_ctx`` on every invocation so the manager
        only needs to update that one attribute between callbacks; no
        separate solver-handle scratchpad (formerly ``_rust_solver_ref``) is
        required. If a fork path failed and no context is available, the
        caller MUST skip the install rather than passing ``None`` —
        previously this method silently no-op'd when scratch had no
        ``rust_solver_ctx``, which let Python's solver diverge from Rust's
        without raising.

        This covers:
        - state.solver.eval() — used by SimProcedures and concretization strategies
        - state.solver.satisfiable() — used by concretization and feasibility checks
        - state.solver.min()/max() — used by concretization strategies
        - state.solver.eval_upto() — used by concretization strategies
        - state.solver.add() — forwards constraints to both Python and Rust

        Sibling helper to ``RustSolverFallback`` (rust_state_export.py): both
        attach Rust-backed eval/satisfiable/min/max/eval_upto onto a Python
        ``state.solver``. The fallback class handles post-exploration stash
        states (with the ``_rust_fallback_attached`` double-patch guard);
        this method handles per-callback states. Any new FFI solver entry
        point should be considered for both wiring sites — see Rust
        ``callbacks.rs`` module invariant 5
        (``invariant-rust-solver-fallback-class``).
        """
        assert rust_ctx is not None, (
            "_install_rust_solver_on_callback_state requires a non-None "
            "rust_ctx; see angr-ye2n for the explicit-contract rationale"
        )

        # Idempotency: closures read state.scratch.rust_solver_ctx on every
        # call, so a re-install would just rewrap our own closures as the
        # "originals" — an infinite-recursion footgun. Guard with a per-
        # state flag.
        if getattr(state.scratch, "_rust_solver_installed", False):
            return

        # First time: save real originals and create closures once.
        # _with_extra_constraints lives in rust_state_proxy (extracted there by
        # the 24pv4.4 dedup refactor); import here to keep the closures below
        # from raising NameError on every eval/satisfiable/eval_upto call.
        from angr.exploration.rust_state_proxy import _with_extra_constraints

        original_eval = state.solver.eval
        original_satisfiable = state.solver.satisfiable
        original_min = state.solver.min
        original_max = state.solver.max
        original_eval_upto = state.solver.eval_upto
        original_add = state.solver.add
        state.scratch._rust_solver_installed = True

        def _rust_eval(expr, cast_to=None, **kwargs):
            _ctx = state.scratch.rust_solver_ctx
            kwargs.pop("exact", None)
            extra = kwargs.pop("extra_constraints", ())
            try:
                result = _with_extra_constraints(_ctx, _ctx.eval, expr, extra=extra)
                if result is None:
                    return original_eval(expr, cast_to=cast_to, **kwargs)
                if cast_to == bytes:
                    nbytes = (expr.length + 7) // 8
                    return result.to_bytes(nbytes, "big")
                return result
            except claripy.errors.UnsatError:
                # cat-(a) EXPECTED CONTROL FLOW: claripy UnsatError must propagate
                # so the caller sees the constraint conflict — re-raise.
                raise
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: Rust solver eval failed; fall back
                # to Python's claripy solver. The two solvers may diverge on
                # satisfying value selection.
                return original_eval(expr, cast_to=cast_to, **kwargs)

        def _rust_satisfiable(**kwargs):
            _ctx = state.scratch.rust_solver_ctx
            kwargs.pop("exact", None)
            extra = kwargs.pop("extra_constraints", ())
            try:
                return _with_extra_constraints(_ctx, _ctx.satisfiable, extra=extra)
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: Rust solver satisfiable() raised;
                # fall back to Python claripy. May report sat/unsat differently
                # from Rust.
                return original_satisfiable(**kwargs)

        def _rust_min(expr, **kwargs):
            kwargs.pop("exact", None)
            kwargs.pop("extra_constraints", None)
            kwargs.pop("signed", None)
            _ctx = state.scratch.rust_solver_ctx
            try:
                result = _ctx.min(expr, signed=False)
                if result is not None:
                    return result
                # Rust returned no bound (None). Two disjoint causes
                # (solving_ops.rs min()): the ctx is unsat, or expr width > 128.
                #  - unsat: the state is dead; claripy would only re-derive the
                #    same unsat and raise. Raise SimUnsatError now rather than
                #    waste a full Python solve to reach the identical failure.
                #    satisfiable() is cached by the min() call above, so it's free.
                #  - width > 128: the u128 result can't hold the extremum, but
                #    claripy's big-int solver can — fall through to it (below).
                unsat = not _ctx.satisfiable()
            except claripy.errors.UnsatError:
                # cat-(a) EXPECTED CONTROL FLOW: propagate a genuine unsat.
                raise
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: Rust solver min() raised; fall back
                # to Python claripy. Concretization strategies may pick a different
                # minimum than Rust would.
                unsat = False
            if unsat:
                raise SimUnsatError("Rust solver context is unsat (callback min)")
            return original_min(expr, **kwargs)

        def _rust_max(expr, **kwargs):
            kwargs.pop("exact", None)
            kwargs.pop("extra_constraints", None)
            kwargs.pop("signed", None)
            _ctx = state.scratch.rust_solver_ctx
            try:
                result = _ctx.max(expr, signed=False)
                if result is not None:
                    return result
                # Same None disambiguation as _rust_min: unsat -> raise now
                # (avoid a wasted Python solve to re-derive the dead state);
                # width > 128 -> fall through to claripy's big-int solver.
                unsat = not _ctx.satisfiable()
            except claripy.errors.UnsatError:
                # cat-(a) EXPECTED CONTROL FLOW: propagate a genuine unsat.
                raise
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: Rust solver max() raised; fall back
                # to Python claripy. Same divergence risk as min().
                unsat = False
            if unsat:
                raise SimUnsatError("Rust solver context is unsat (callback max)")
            return original_max(expr, **kwargs)

        def _rust_eval_upto(expr, n, cast_to=None, **kwargs):
            _ctx = state.scratch.rust_solver_ctx
            kwargs.pop("exact", None)
            extra = kwargs.pop("extra_constraints", ())
            try:
                results = _with_extra_constraints(_ctx, _ctx.eval_upto, expr, n, extra=extra)
                if cast_to is not None:
                    results = tuple(cast_to(r) for r in results)
                return results
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: Rust solver eval_upto() raised; fall
                # back to Python claripy. Returned solution sets may differ.
                return original_eval_upto(expr, n, cast_to=cast_to, **kwargs)

        def _rust_add(*constraints):
            _ctx = state.scratch.rust_solver_ctx
            # Forward to both Rust and Python solvers
            for c in constraints:
                try:
                    _ctx.add_constraint_ast(c)
                except Exception:
                    # cat-(b) FALLBACK WITH LOSS: forwarding constraint to Rust solver
                    # failed; the Python solver still has it (added below), but Rust
                    # may produce different sat/values until the next callback
                    # re-attaches rust_solver_ctx.
                    pass
            original_add(*constraints)

        state.solver.eval = _rust_eval
        state.solver.satisfiable = _rust_satisfiable
        state.solver.min = _rust_min
        state.solver.max = _rust_max
        state.solver.eval_upto = _rust_eval_upto
        state.solver.add = _rust_add

    def _inject_rust_stdout(self, state, state_id):
        """Inject Rust-side fd output buffers into the state's posix plugin.

        Native puts/printf/fwrite write to per-fd buffers in Rust. This method
        fetches them and appends whatever the Python state is missing, so that
        predicates calling ``state.posix.dumps(1)`` — and a bounced
        SimProcedure that reads its own stdout — see the native output.

        Idempotent: only the suffix past the Python stream's current length is
        written, so calling it twice on the same state (the cached callback
        state is reused across bounces) does not duplicate the output. The
        write-back for the other direction is
        ``_sync_state_posix_to_rust``.
        """
        _px_start = time.perf_counter_ns()
        try:
            for fd in POSIX_SYNC_FDS:
                self._inject_rust_fd_output(state, state_id, fd)
        finally:
            self._perf_stats.record_posix_call(time.perf_counter_ns() - _px_start)

    def _inject_rust_fd_output(self, state, state_id, fd):
        # Fast check: skip FFI if this state never wrote to the fd.
        if not self._rust_mgr.has_state_fd_output(state_id, fd):
            return
        try:
            rust_out = bytes(self._rust_mgr.get_state_fd_output(state_id, fd))
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: Rust fd fetch failed; posix.dumps(fd)
            # will not include any native-side output for this state.
            return
        stream = _posix_stream(state, fd)
        if stream is None:
            return
        py_len = _concrete_stream_len(stream)
        if py_len is None or py_len >= len(rust_out):
            # Symbolic-size stream (cannot tell what is missing), or the
            # Python side is already caught up.
            return
        try:
            stream.write(None, claripy.BVV(rust_out[py_len:]), events=False)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: Rust fd write to posix failed;
            # posix.dumps(fd) misses the native output. Debug-logs.
            l.debug("Failed to inject Rust fd %d output into posix: %s", fd, e)

    def _inject_rust_fds(self, state, state_id):
        """Seed fds that native code opened into the callback state's posix table.

        Inbound half of the fd-table channel whose outbound half is
        ``rust_state_export::_sync_state_posix_fds_to_rust`` (angr-op0dn.14.1.6).
        A native ``open`` allocates the fd on Rust's ``FileSystem`` only; the
        cached callback state's ``state.posix.fd`` never learns of it. Two
        consequences this fixes: a bounced proc that reads/writes such an fd
        sees a closed fd, and ``posix._pick_fd`` — which picks the lowest fd
        *it* does not know about — can hand a bounced ``open()`` a number Rust
        is already using, so the two sides end up pointing at different files.

        Idempotent: an fd already in ``state.posix.fd`` is left alone, which
        matters because the cached callback state is reused across bounces.
        """
        # Fast check: the standard three fds always exist on both sides, so a
        # state that opened nothing natively never crosses the FFI further.
        if not self._rust_mgr.has_state_extra_fds(state_id):
            return
        posix = state.plugins.get("posix")
        if posix is None:
            return
        try:
            rust_fds = self._rust_mgr.get_state_open_fds(state_id)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: without the fd list the bounced proc
            # keeps the pre-fix behavior (native fds invisible to it).
            l.debug("get_state_open_fds(%d) failed: %s", state_id, e)
            return
        for fd, name, position, flags, _content_len, is_open in rust_fds:
            if fd <= 2 or not is_open or fd in posix.fd or not name:
                continue
            try:
                content = bytes(self._rust_mgr.get_state_fd_content(state_id, fd))
            except Exception as e:
                l.debug("get_state_fd_content(%d, %d) failed: %s", state_id, fd, e)
                continue
            path = name.encode() if isinstance(name, str) else name
            simfile = state.fs.get(path)
            if simfile is None:
                # Rust descriptors hold concrete bytes, so the mirror SimFile is
                # a concrete one. has_end is left to the state's FILES_HAVE_EOF
                # option, matching what posix.open would have built.
                simfile = SimFile(path, content=content, size=len(content))
                if not state.fs.insert(path, simfile):
                    l.debug("could not insert mirror SimFile for native fd %d (%s)", fd, name)
                    continue
            # else: a SimFile for this path already exists (e.g. pre-seeded by
            # the harness). cat-(b) FALLBACK WITH LOSS: bytes the native side
            # appended to it are not merged in — clobbering a possibly symbolic
            # pre-seeded file with Rust's concrete snapshot would be worse.
            simfd = SimFileDescriptor(simfile, flags)
            simfd.set_state(state)
            simfd._pos = position
            posix.fd[fd] = simfd

    def _inject_rust_stdin(self, state, state_id):
        """Inject Rust-side stdin data into posix.dumps(0).

        Native fgets/fgetc/getchar create symbolic stdin bytes in Rust.
        Since these BVS variables live in the Rust Z3 context (not Python's),
        we evaluate them via the Rust solver and inject the concrete result
        into the stdin stream so posix.dumps(0) returns the correct value.

        Must catch ALL exceptions (including AttributeError) to prevent
        propagation through @property descriptors which triggers __getattr__.
        """
        _px_start = time.perf_counter_ns()
        try:
            self._inject_rust_stdin_inner(state, state_id)
        finally:
            self._perf_stats.record_posix_call(time.perf_counter_ns() - _px_start)

    def _inject_rust_stdin_inner(self, state, state_id):
        try:
            if not self._rust_mgr.has_state_stdin_symbols(state_id):
                return
            stdin_symbols = self._rust_mgr.get_state_stdin_symbols(state_id)
            if not stdin_symbols:
                return
            posix = getattr(state, "posix", None)
            if posix is None:
                return
            stdin_stream = getattr(posix, "stdin", None)
            if stdin_stream is None:
                return
            # Evaluate each stdin symbol via Rust solver to get concrete bytes
            concrete_bytes = bytearray()
            for name, bits in stdin_symbols:
                val = self._rust_mgr.eval_stdin_symbol(state_id, name)
                if val is not None:
                    concrete_bytes.append(val & 0xFF)
                else:
                    concrete_bytes.append(0)
            if concrete_bytes:
                data = claripy.BVV(bytes(concrete_bytes))
                size = claripy.BVV(len(concrete_bytes), state.arch.bits)
                stdin_stream.content.append((data, size))
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: Rust stdin injection raised; posix.
            # dumps(0) will not show the symbolic-stdin bytes that Rust read.
            # Debug-logs. Note: must catch ALL exceptions including AttributeError
            # to avoid @property-descriptor propagation triggering __getattr__.
            l.debug("Failed to inject Rust stdin data into posix: %s", e)

    _SIMPROC_NO_RET_TERMINAL = {"exit", "_exit", "abort", "__stack_chk_fail"}

    def _handle_simprocedure_callback(self, event: _ExplorationEvent):
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

        Enters Python from ``RunResult::SimProcedure`` / ``RunResult::Hook``
        in the Rust callbacks (``call_on_hook`` in ``callbacks.rs`` —
        ``avoid-silent-no-op-callback-fallbacks`` ensures the Rust side
        hard-errors when the hook is unset rather than silently producing
        a wrong PC).
        """
        _sp_total_start = time.perf_counter_ns()
        addr = event.callback_addr
        name = event.callback_name
        addr_int = int(addr) if addr is not None else None

        # Track current callback state ID for symbolic memory preservation
        self._current_callback_state_id = event.callback_state_id

        # Handle internal passthrough - this is internal binary code, just continue execution
        if name == "__internal_passthrough__":
            if _BOUNCE_TRACE:
                _emit_bounce_trace(name, addr, None)
            if _DBG:
                l.debug(f"Internal passthrough at 0x{addr:x} - continuing execution")
            # Resume execution at this address, no SimProcedure to run
            self._rust_mgr.resume_after_simprocedure(event.callback_state_id, addr, None, None)
            self._perf_stats.record_simprocedure_call(time.perf_counter_ns() - _sp_total_start)
            return

        # Find the SimProcedure
        proc = self._find_simprocedure(addr, name)
        if _BOUNCE_TRACE:
            _emit_bounce_trace(_simproc_dispatch_name(proc) if proc is not None else name, addr, proc)
        if proc is None:
            # Stale/unknown hook fired (e.g. proj.unhook raced the Rust hook
            # table, or a hook addr with no backing proc). Resuming at addr+1
            # would land on a misaligned PC. Instead skip the hook for this
            # address and re-execute the real instruction at addr (angr-969g).
            try:
                self._rust_mgr.set_skip_hook_addr(addr)
            except Exception as e:
                if _DBG:
                    l.debug(f"Could not set skip_hook_addr for stale hook 0x{addr:x}: {e}")
            self._rust_mgr.resume_after_simprocedure(event.callback_state_id, addr, None, None)
            self._perf_stats.record_simprocedure_call(time.perf_counter_ns() - _sp_total_start)
            return

        # Fast paths for procedures that always deadend (NO_RET terminals,
        # cached exit-only continuations) — skip state creation entirely.
        if self._try_simproc_deadend_fast_path(proc, name, addr, addr_int):
            self._perf_stats.record_simprocedure_call(time.perf_counter_ns() - _sp_total_start)
            return

        # Get hook length - this determines if the hook replaces code
        hook_length = getattr(proc, "kwargs", {}).get("length", 0)
        if hook_length == 0:
            hook_length = getattr(proc, "length", 0)
        is_zero_length_hook = hook_length == 0

        # Create angr state for the SimProcedure (uses cached state if available)
        _sp_state_create_start = time.perf_counter_ns()
        state = self._create_state_for_callback(event)
        if state is None:
            l.warning(f"Could not create state for SimProcedure at 0x{addr:x}")
            self._rust_mgr.resume_after_simprocedure(event.callback_state_id, addr + 1, None, None)
            self._set_callback_state(None)
            self._current_callback_state_id = None
            self._perf_stats.add_simprocedure_phase("state_create", time.perf_counter_ns() - _sp_state_create_start)
            self._perf_stats.record_simprocedure_call(time.perf_counter_ns() - _sp_total_start)
            return
        self._perf_stats.add_simprocedure_phase("state_create", time.perf_counter_ns() - _sp_state_create_start)

        # Snapshot original state (full copy / register snapshot / no copy)
        # for change extraction after the SimProcedure runs.
        _sp_copy_start = time.perf_counter_ns()
        orig_state = self._snapshot_orig_state(state, proc, name, is_zero_length_hook)
        self._perf_stats.add_simprocedure_phase("state_copy", time.perf_counter_ns() - _sp_copy_start)

        # Save original constraint COUNT before hook execution.
        # Building set(constraints) is expensive (~6ms per call with many constraints).
        # Save just the count; only build the full set if count changes after callback.
        # When using a register snapshot (no full state copy), eagerly capture
        # constraints since we can't access orig_state.solver.constraints later.
        orig_constraint_count = len(state.solver.constraints) if hasattr(state, "solver") else 0
        if isinstance(orig_state, dict):
            # Snapshot mode: capture constraints eagerly for the rare case
            # where a non-memory-writing proc adds constraints
            orig_constraints = set(state.solver.constraints) if hasattr(state, "solver") else set()
        else:
            orig_constraints = None  # Deferred — only built if needed

        # Track memory writes during callback execution
        memory_tracker = CallbackMemoryTracker(state)

        # state.inspect simprocedure BEFORE (angr-xmfj). The proc may be a
        # class or instance; pass the raw object so user BPs can introspect.
        # `simprocedure_result` is `None` on BEFORE because the proc has
        # not produced a return value yet.
        sp_display_name = name or proc.__class__.__name__
        self._cb_inspect_simprocedure(
            event.callback_state_id,
            "before",
            sp_display_name,
            addr,
            proc,
            None,
        )

        # Run the SimProcedure
        _sp_execute_start = time.perf_counter_ns()
        try:
            from angr.engines.successors import SimSuccessors

            # Create SimSuccessors object for the procedure
            successors = SimSuccessors(addr=addr, initial_state=state)

            # Execute the procedure with memory tracking
            with memory_tracker:
                if hasattr(proc, "run"):
                    proc.execute(state, successors)
                else:
                    # It's a class, instantiate it
                    proc().execute(state, successors)
            # state.inspect simprocedure AFTER (angr-xmfj). MVP scope:
            # `simprocedure_result` is `None` — capturing the proc's raw
            # return value would require wrapping `proc.execute()` to
            # observe the inner `inst.run_func` return; for now BPs see
            # the result via `successors.artifacts` or the proc instance.
            self._cb_inspect_simprocedure(
                event.callback_state_id,
                "after",
                sp_display_name,
                addr,
                proc,
                None,
            )
            _sp_exec_elapsed = time.perf_counter_ns() - _sp_execute_start
            self._perf_stats.add_simprocedure_phase("execute", _sp_exec_elapsed)
            # Per-procedure timing
            _proc_name = name or proc.__class__.__name__
            if _proc_name not in self._procedure_times:
                self._procedure_times[_proc_name] = {"count": 0, "execute_ns": 0}
            self._procedure_times[_proc_name]["count"] += 1
            self._procedure_times[_proc_name]["execute_ns"] += _sp_exec_elapsed

            # Get tracked memory writes from callback execution. When the
            # angr-4scu step-3 gate is on, ``state.memory`` is a
            # ``RustMemoryProxy`` and every store landed in Rust during the
            # callback; the tracker still wrapped ``state.memory.store`` and
            # accumulated entries (so any user code introspecting it sees a
            # stable surface), but we discard the lists to avoid a
            # double-write through ``resume_after_simprocedure``.
            if getattr(self, "_use_callback_memory_proxy", False):
                tracked_writes = None
                tracked_symbolic_writes = None
            else:
                tracked_writes = memory_tracker.get_writes()
                tracked_symbolic_writes = memory_tracker.get_symbolic_writes()
            if _DBG:
                if tracked_writes:
                    l.debug(f"Tracked {len(tracked_writes)} memory writes during callback")
                if tracked_symbolic_writes:
                    l.debug(f"Tracked {len(tracked_symbolic_writes)} symbolic memory writes during callback")

            # Handle successors - this includes sync back to Rust
            _sp_sync_start = time.perf_counter_ns()
            all_succs = successors.all_successors

            self._capture_stdin_from_successors(all_succs, name)
            self._capture_continuation_data(all_succs, addr)

            if all_succs:
                handled_early = self._handle_callback_with_successors(
                    all_succs,
                    proc,
                    name,
                    addr,
                    addr_int,
                    event,
                    state,
                    orig_state,
                    is_zero_length_hook,
                    tracked_writes,
                    tracked_symbolic_writes,
                    orig_constraints,
                    orig_constraint_count,
                    _sp_total_start,
                )
                if handled_early:
                    return
            else:
                self._handle_callback_no_successors(
                    proc,
                    name,
                    addr,
                    event,
                    state,
                    orig_state,
                    is_zero_length_hook,
                    tracked_writes,
                    tracked_symbolic_writes,
                    orig_constraints,
                    orig_constraint_count,
                )

        except TypeError as e:
            # cat-(c) WRONG-ANSWER RISK: continuation procedure missing args
            # (e.g., after_main run without procedure_data). Treated as a
            # graceful deadend. Already warns. Other TypeErrors fall through
            # to the generic handler below.
            # Continuation procedure missing local_vars (e.g., after_main without args).
            # This happens when the init phase's procedure_data wasn't captured.
            # Treat as a graceful exit (deadend) rather than a hard error.
            if "missing" in str(e) and "positional argument" in str(e):
                l.warning(f"Continuation at 0x{addr:x} missing args (likely after_main) — deadending")
                try:
                    self._rust_mgr.resume_after_simprocedure(event.callback_state_id, 0, None, None, None)
                except Exception:
                    # cat-(b) FALLBACK WITH LOSS: resume_after_simprocedure failed in
                    # the missing-args path; try resume_after_error next.
                    try:
                        self._rust_mgr.resume_after_error(event.callback_state_id, str(e))
                    except Exception:
                        # cat-(b) FALLBACK WITH LOSS: resume_after_error also failed; the
                        # pending state stays pending and the next run() iteration will
                        # notice / hang. Tolerated — better than throwing through PyO3.
                        pass
                self._set_callback_state(None)
                self._current_callback_state_id = None
                self._perf_stats.record_simprocedure_call(time.perf_counter_ns() - _sp_total_start)
                return
            # Other TypeErrors fall through to generic handler
            raise
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: SimProcedure execution raised; we
            # move the state to errored stash. Already warns. Without this
            # path, the state would resume in Rust with corrupted post-
            # callback state.
            # On exception, move state to errored stash instead of resuming with corrupted state
            l.warning(f"SimProcedure execution error at 0x{addr:x}: {e}")
            import traceback

            traceback.print_exc()
            # Signal error to Rust - this will move the state to errored stash
            try:
                self._rust_mgr.resume_after_error(event.callback_state_id, str(e))
            except Exception as resume_err:
                # cat-(b) FALLBACK WITH LOSS: resume_after_error itself raised;
                # the pending state stays pending. Already warns.
                l.warning(f"Could not signal error to Rust: {resume_err}")
            # Clear callback state since we've handled the error
            self._set_callback_state(None)
            self._current_callback_state_id = None
            self._perf_stats.record_simprocedure_call(time.perf_counter_ns() - _sp_total_start)
            return
        # Only clear callback state on success, not in finally.
        self._set_callback_state(None)
        self._current_callback_state_id = None
        self._perf_stats.add_simprocedure_phase("sync_back", time.perf_counter_ns() - _sp_sync_start)
        self._perf_stats.record_simprocedure_call(time.perf_counter_ns() - _sp_total_start)

    def _find_simprocedure(self, addr: int, name: str | None):
        """Locate a SimProcedure by address, then by class name as fallback.

        PLT addresses can drift between Python and Rust; the name fallback
        recovers from that case. Returns None on not-found or invalid type
        (caller is expected to log a warning and resume past addr).
        """
        proc = self._project._sim_procedures.get(addr)
        if proc is None and name:
            for proc_addr, candidate in self._project._sim_procedures.items():
                proc_name = _simproc_dispatch_name(candidate)
                if proc_name == name:
                    if _DBG:
                        l.debug(f"Found SimProcedure {name} at 0x{proc_addr:x} (callback was at 0x{addr:x})")
                    proc = candidate
                    break
        if proc is None:
            l.warning(f"SimProcedure not found at 0x{addr:x} (name={name})")
            return None
        if isinstance(proc, (list, tuple)):
            l.error(f"Invalid SimProcedure at 0x{addr:x}: got {type(proc).__name__}, expected callable")
            return None
        return proc

    def _try_simproc_deadend_fast_path(self, proc, name: str | None, addr: int, addr_int: int | None) -> bool:
        """Skip state creation for procedures that always deadend.

        Two cases:
        - NO_RET terminal procedures (exit, abort, etc.) — known statically.
        - Exit-continuation cache hits — observed dynamically when ALL
          successors had Ijk_Exit on a NO_RET procedure (see
          invariant-exit-continuation-cache memory).

        Returns True if the callback was handled (caller should return).
        """
        proc_no_ret = getattr(proc, "NO_RET", False)
        is_terminal = proc_no_ret and name in self._SIMPROC_NO_RET_TERMINAL
        is_cached_exit = addr_int in self._exit_continuation_addrs
        if not (is_terminal or is_cached_exit):
            return False
        if _DBG:
            reason = "no-return procedure" if is_terminal else "exit continuation"
            l.debug(f"Fast path: {reason} {name} at 0x{addr:x} — deadending")
        self._rust_mgr.deadend_pending_callback(self._current_callback_state_id)
        self._current_callback_state_id = None
        _proc_name = name or proc.__class__.__name__
        if _proc_name not in self._procedure_times:
            self._procedure_times[_proc_name] = {"count": 0, "execute_ns": 0}
        self._procedure_times[_proc_name]["count"] += 1
        return True

    def _snapshot_orig_state(self, state, proc, name: str | None, is_zero_length_hook: bool):
        """Snapshot the pre-callback state for later change extraction.

        Returns one of three things:
        - state.copy() for memory-writing procs and UserHooks (need full memory diff)
        - register snapshot dict for non-memory-writing zero-length hooks
          (built from bundle registers if available, else read from state)
        - the state itself for the remaining case (no copy needed; the caller
          will diff registers/memory directly against the post-execution state)
        """
        is_user_hook = proc.__class__.__name__ == "UserHook"
        needs_full_copy = is_user_hook or name in self._MEMORY_WRITING_PROCS
        if needs_full_copy:
            return state.copy()
        if is_zero_length_hook:
            bundle_regs = getattr(self, "_last_bundle_registers", None)
            if bundle_regs is not None:
                snapshot = self._snapshot_registers_from_bundle(bundle_regs, self._project.arch)
                self._last_bundle_registers = None
                return snapshot
            return self._snapshot_registers(state)
        return state

    def _capture_stdin_from_successors(self, all_succs, name: str | None) -> None:
        """Track stdin packets added by SimProcedures (fgets/read/etc.).

        Found states forked purely in Rust later use this to restore stdin
        when extracting concrete inputs.
        """
        for succ in all_succs:
            try:
                posix = getattr(succ, "posix", None)
                if posix is None:
                    continue
                stdin = getattr(posix, "stdin", None)
                if stdin is None or not hasattr(stdin, "content"):
                    continue
                if stdin.content and len(stdin.content) > len(self._stdin_content):
                    self._stdin_content = list(stdin.content)
                    if _DBG:
                        l.debug(f"Captured {len(stdin.content)} stdin packets from {name}")
                # All successors share stdin — only need one
                break
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: per-successor stdin probe failed;
                # we still try the remaining successors. If all fail, _stdin_content
                # stays at its previous value — found-state stdin restoration may
                # miss the latest packet.
                pass

    def _capture_continuation_data(self, all_succs, addr: int) -> None:
        """Stash procedure_data and pre-register continuation hooks.

        When a SimProcedure uses self.call() to invoke another function, the
        continuation address and saved locals are stored in callstack frame
        procedure_data. We stash that here so the continuation callback can
        restore it, and immediately register the hook with Rust to avoid
        "Cannot execute external address" errors.
        """
        for succ in all_succs:
            try:
                cs = succ.callstack
                top = cs.top if cs else None
                if top is None:
                    continue

                # When jumpkind is Ijk_Call, add_successor pushes a NEW callstack frame.
                # The procedure_data is on the PREVIOUS frame (the caller's frame).
                # Check both top and top.next for procedure_data.
                frames_to_check = [top]
                if hasattr(top, "next") and top.next is not None:
                    frames_to_check.append(top.next)

                for frame in frames_to_check:
                    pdata = getattr(frame, "procedure_data", None)
                    if pdata is None or len(pdata) < 5:
                        continue
                    cont_addr = pdata[4]  # ideal_addr is continuation address
                    if hasattr(cont_addr, "concrete"):
                        cont_addr_int = int(cont_addr)
                    elif isinstance(cont_addr, int):
                        cont_addr_int = cont_addr
                    else:
                        continue
                    if cont_addr_int == addr:
                        continue
                    self._pending_procedure_data[cont_addr_int] = pdata
                    if _DBG:
                        l.debug(f"Stored procedure_data for continuation at 0x{cont_addr_int:x}")

                    # C1 Fix: Register continuation hook with Rust IMMEDIATELY
                    # so Rust doesn't error trying to execute the continuation
                    # before the next _sync_hooks_before_step() pass.
                    if cont_addr_int in self._registered_hooks:
                        continue
                    cont_proc = self._project._sim_procedures.get(cont_addr_int)
                    if not cont_proc:
                        continue
                    cont_name = _simproc_dispatch_name(cont_proc)
                    cont_num_args = getattr(cont_proc, "num_args", 0) or 0
                    cont_no_return = getattr(cont_proc, "NO_RET", False)
                    self._rust_mgr.register_simprocedures([(cont_addr_int, cont_name, cont_num_args, cont_no_return)])
                    self._registered_hooks.add(cont_addr_int)
                    if _DBG:
                        l.debug(f"Immediately registered continuation hook at 0x{cont_addr_int:x}: {cont_name}")
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: continuation-data capture / hook
                # pre-registration raised on this successor; the continuation
                # may fail to register before Rust hits it (errors as 'Cannot
                # execute external address'). Debug-logs.
                if _DBG:
                    l.debug(f"Could not capture procedure_data: {e}")

    def _handle_callback_with_successors(
        self,
        all_succs,
        proc,
        name: str | None,
        addr: int,
        addr_int: int | None,
        event: _ExplorationEvent,
        state,
        orig_state,
        is_zero_length_hook: bool,
        tracked_writes,
        tracked_symbolic_writes,
        orig_constraints,
        orig_constraint_count,
        _sp_total_start: int,
    ) -> bool:
        """Handle the success path when the SimProcedure produced successors.

        Returns True if the caller should return immediately (deadend was
        signalled and counters were finalized); False if the caller should
        continue with the normal sync_back timing/cleanup.
        """
        proc_no_ret = getattr(proc, "NO_RET", False) if proc else False

        # NO_RET termination procedure with successors — deadend.
        # __libc_start_main has NO_RET but uses self.call() for continuations,
        # so we restrict this to explicit termination names.
        if proc_no_ret and name in self._SIMPROC_NO_RET_TERMINAL:
            if _DBG:
                l.debug(f"No-return procedure {name} with successors — deadending")
            self._rust_mgr.deadend_pending_callback(event.callback_state_id)
            self._set_callback_state(None)
            self._current_callback_state_id = None
            return True

        # Detect exit-only continuations: if ALL successors have Ijk_Exit
        # AND the procedure declares NO_RET, this address always deadends
        # (e.g., __libc_start_main's after_main continuation). Cache so
        # future callbacks at this address skip state creation.
        # NO_RET requirement prevents wrong-cache: a NO_RET=False procedure
        # might happen to all-exit for one state but normal-return for others.
        if proc_no_ret and addr_int is not None:
            _all_exit = all(getattr(s.history, "jumpkind", None) == "Ijk_Exit" for s in all_succs)
            if _all_exit:
                if _DBG:
                    l.debug(f"Detected exit-only continuation at 0x{addr:x} — caching for fast deadend")
                self._exit_continuation_addrs.add(addr_int)
                self._rust_mgr.deadend_pending_callback(event.callback_state_id)
                self._set_callback_state(None)
                self._current_callback_state_id = None
                self._perf_stats.record_simprocedure_call(time.perf_counter_ns() - _sp_total_start)
                return True

        # First successor continues in Rust
        first_succ = all_succs[0]

        # When a SimProcedure uses self.call() (Ijk_Call), angr's
        # ``cc.setup_callsite`` pushes the continuation return address onto the
        # stack and decrements sp. The sp decrement rides back to Rust via
        # ``reg_changes``, but the PUSHED BYTES are recovered separately in
        # ``_resume_with_state`` (the Ijk_Call stack-region capture) because
        # ``call()`` runs ``setup_callsite`` on a *copy* of the executing
        # state, bypassing the CallbackMemoryTracker's wrapped ``store``. Do
        # NOT manually re-push here: the former ``_push_continuation_address``
        # (removed iter35) also re-decremented sp, leaving the callee's stack
        # 8 bytes low so its ``ret`` popped garbage and deadended (angr-aca6y,
        # xmllint's hooked pthread_once continuation).

        # For zero-length hooks where the successor stays at the same
        # address, the hook just modified state — continue at the same
        # address WITHOUT re-triggering the hook.
        # Check symbolic IP before .addr to prevent SimValueError.
        succ_ip_symbolic = first_succ.regs._ip.symbolic
        succ_addr_matches = not succ_ip_symbolic and first_succ.addr == addr
        skip_hook_addr = addr if (is_zero_length_hook and succ_addr_matches) else None
        self._resume_with_state(
            first_succ,
            orig_state,
            event,
            skip_hook_addr=skip_hook_addr,
            tracked_writes=tracked_writes,
            tracked_symbolic_writes=tracked_symbolic_writes,
            orig_constraints=orig_constraints,
            orig_constraint_count=orig_constraint_count,
        )

        # Additional successors are added as new active states
        for succ in all_succs[1:]:
            self._add_forked_state(succ, event)
        return False

    def _handle_callback_no_successors(
        self,
        proc,
        name: str | None,
        addr: int,
        event: _ExplorationEvent,
        state,
        orig_state,
        is_zero_length_hook: bool,
        tracked_writes,
        tracked_symbolic_writes,
        orig_constraints,
        orig_constraint_count,
    ) -> None:
        """Handle the path where a SimProcedure produced no successors.

        Three sub-cases:
        - Terminal procedures (exit/abort/CallReturn) → deadend or skip-hook resume.
        - Zero-length non-terminal hooks → resume past the hook with synced changes.
        - Other procedures → deadend (NO_RET) or resume at return address.
        """
        # CallReturn is the terminal hook used by factory.callable().
        no_ret_terminal = name in ("exit", "_exit", "abort", "__stack_chk_fail", "CallReturn")
        if no_ret_terminal:
            if _DBG:
                l.debug(f"Terminal procedure {name} — deadending state at 0x{addr:x}")
            if is_zero_length_hook:
                # For zero-length terminal hooks (e.g., CallReturn at callable's
                # return address), skip the hook and resume at addr. The Rust
                # engine can't lift code there, so it deadends with PC=addr,
                # preserving the PC for code that checks state.addr.
                try:
                    self._rust_mgr.set_skip_hook_addr(addr)
                except Exception:
                    # cat-(b) FALLBACK WITH LOSS: set_skip_hook_addr failed for terminal
                    # zero-length hook; resume_after_simprocedure may re-trigger the
                    # hook on next step.
                    pass
                self._rust_mgr.resume_after_simprocedure(event.callback_state_id, addr, None, None)
            else:
                self._rust_mgr.deadend_pending_callback(event.callback_state_id)
            return

        if is_zero_length_hook:
            self._resume_with_skip_hook(
                addr,
                state,
                orig_state,
                event,
                orig_constraints,
                tracked_writes=tracked_writes,
                tracked_symbolic_writes=tracked_symbolic_writes,
                orig_constraint_count=orig_constraint_count,
            )
            return

        # Non-zero-length, non-terminal: check NO_RET as fallback
        if getattr(proc, "NO_RET", False):
            if _DBG:
                l.debug(f"No-return procedure {name} — deadending state")
            self._rust_mgr.deadend_pending_callback(event.callback_state_id)
            return

        ret_addr = event.callback_return_addr or (addr + 1)
        for sym_addr, ast in tracked_symbolic_writes or []:
            try:
                self._rust_mgr.import_symbolic_memory(event.callback_state_id, sym_addr, ast)
            except Exception:
                # cat-(c) WRONG-ANSWER RISK: tracked symbolic write not imported
                # to Rust; subsequent loads at sym_addr will see concrete bytes
                # instead of the symbolic value.
                pass
        self._sync_state_heap_to_rust(state, event.callback_state_id)
        self._sync_state_posix_to_rust(state, event.callback_state_id)
        self._sync_state_posix_fds_to_rust(state, event.callback_state_id)
        self._rust_mgr.resume_after_simprocedure(event.callback_state_id, ret_addr, None, tracked_writes or None)

    @staticmethod
    def _resolve_next_pc(state, fallback_addr=None):
        """Resolve a concrete next-PC from a (possibly symbolic) state IP.

        Concrete IP returns ``state.addr``. A symbolic IP is concretized via
        ``eval_one``, falling back to ``eval`` (picks any solution) when more
        than one solution exists. If ``eval`` also raises and ``fallback_addr``
        is given, that address is returned; otherwise the error propagates.
        """
        if not state.regs._ip.symbolic:
            return state.addr
        try:
            return state.solver.eval_one(state.regs._ip)
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: eval_one raised (>1 solution or
            # solver error); pick any concrete value via eval.
            try:
                return state.solver.eval(state.regs._ip)
            except Exception:
                if fallback_addr is not None:
                    return fallback_addr
                raise

    @staticmethod
    def _merge_tracked_writes(mem_changes, tracked_writes):
        """Append ``tracked_writes`` whose addr isn't already in ``mem_changes``.

        Mutates and returns ``mem_changes``. Tracked writes capture symbolic
        stores that ``_extract_memory_changes`` might miss.
        """
        if tracked_writes:
            existing_addrs = {addr for addr, _ in mem_changes}
            for addr, data in tracked_writes:
                if addr not in existing_addrs:
                    mem_changes.append((addr, data))
                    existing_addrs.add(addr)
        return mem_changes

    @staticmethod
    def _filter_symbolic_addrs(mem_changes, symbolic_imports, tracked_symbolic_writes):
        """Drop concrete ``mem_changes`` at any addr covered by a symbolic import.

        Prevents ``apply_changes`` from overwriting symbolic imports (from
        ``symbolic_imports`` or ``tracked_symbolic_writes``) with concrete values.
        """
        all_symbolic_addrs = {addr for addr, _ in symbolic_imports}
        if tracked_symbolic_writes:
            all_symbolic_addrs.update(addr for addr, _ in tracked_symbolic_writes)
        if all_symbolic_addrs:
            return [(addr, data) for addr, data in mem_changes if addr not in all_symbolic_addrs]
        return mem_changes

    @staticmethod
    def _merge_symbolic_imports(symbolic_imports, tracked_symbolic_writes):
        """Combine ``symbolic_imports`` with ``tracked_symbolic_writes``, deduped by addr."""
        all_sym_imports = list(symbolic_imports)
        if tracked_symbolic_writes:
            existing_sym_addrs = {addr for addr, _ in symbolic_imports}
            for addr, ast in tracked_symbolic_writes:
                if addr not in existing_sym_addrs:
                    all_sym_imports.append((addr, ast))
        return all_sym_imports

    def _resume_with_state(
        self,
        succ_state: angr.SimState,
        orig_state: angr.SimState,
        event: _ExplorationEvent,
        skip_hook_addr: int | None = None,
        tracked_writes: list | None = None,
        tracked_symbolic_writes: list | None = None,
        orig_constraints: set | None = None,
        orig_constraint_count: int | None = None,
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
            tracked_writes: Memory writes tracked during callback execution.
            tracked_symbolic_writes: List of (addr, ast) for symbolic memory imports.
        """
        # Handle symbolic IP: pick first concrete solution if symbolic
        new_pc = self._resolve_next_pc(succ_state)

        # Extract changes
        is_snapshot = isinstance(orig_state, dict)
        # angr-qj30 (write-through .2): when the register-proxy gate is on
        # every ``state.regs.<name> = ...`` issued by the SimProc already
        # landed in Rust via ``RustRegisterProxy.__setattr__`` →
        # ``set_state_register_symbolic_ast``. Skip the diff so we don't
        # double-write through ``resume_after_simprocedure``.
        if getattr(self, "_use_callback_register_proxy", False):
            reg_changes = []
        else:
            reg_changes = self._extract_register_changes(orig_state, succ_state)

        # Skip memory extraction when orig_state is a register snapshot
        # (non-memory-writing extern SimProcedures don't modify memory)
        if is_snapshot:
            mem_changes = []
            symbolic_imports = []
        else:
            mem_changes, symbolic_imports = self._extract_memory_changes(orig_state, succ_state)

        # Merge tracked writes with extracted memory changes
        if tracked_writes:
            mem_changes = self._merge_tracked_writes(mem_changes, tracked_writes)
            if _DBG:
                l.debug(f"Merged {len(tracked_writes)} tracked writes with memory changes")

        # Capture the stack region pushed by a self.call() continuation
        # (Ijk_Call). angr's ``SimProcedure.call()`` runs ``cc.setup_callsite``
        # on a *copy* of the executing state, so the continuation return
        # address it pushes bypasses the CallbackMemoryTracker's wrapped
        # ``store`` (which only sees the tracked state). In register-snapshot
        # mode ``_extract_memory_changes`` is skipped too, so without this the
        # pushed continuation never reaches Rust: the callee's frame keeps a
        # stale word and its ``ret`` jumps to garbage (angr-aca6y, xmllint's
        # hooked pthread_once). The sp decrement itself rides in via
        # ``reg_changes``; only the pushed bytes are missing here. We read the
        # grown region [succ_sp, orig_sp) straight from the successor. Note:
        # this is the correct, non-double-counting replacement for the former
        # ``_push_continuation_address`` (removed iter35), which re-pushed AND
        # re-decremented sp.
        if getattr(succ_state.history, "jumpkind", None) == "Ijk_Call":
            try:
                orig_sp = self._snapshot_sp(orig_state)
                sp_ast = succ_state.regs.sp
                if orig_sp is not None and not sp_ast.symbolic:
                    succ_sp = succ_state.solver.eval(sp_ast)
                    grew = orig_sp - succ_sp
                    if 0 < grew <= 256:
                        region = succ_state.solver.eval(
                            succ_state.memory.load(succ_sp, grew, endness="Iend_BE")
                        ).to_bytes(grew, "big")
                        covered = {a for a, _ in mem_changes}
                        if not any((succ_sp + off) in covered for off in range(grew)):
                            mem_changes.append((succ_sp, region))
            except Exception as _e:
                # cat-(b) FALLBACK WITH LOSS: capturing the continuation push
                # failed; the callee may resume with a stale return slot.
                if _DBG:
                    l.debug(f"continuation stack capture failed: {_e!r}")

        # Exclude symbolic-import addresses from concrete memory changes so
        # apply_changes can't overwrite them with concrete values.
        mem_changes = self._filter_symbolic_addrs(mem_changes, symbolic_imports, tracked_symbolic_writes)

        # Extract any new constraints added during callback.
        # Skip when using shared solver — constraints go directly to the pending
        # state's solver, so syncing them again would double-add.
        rust_ctx = getattr(succ_state.scratch, "rust_solver_ctx", None)
        using_shared_solver = rust_ctx is not None and hasattr(rust_ctx, "is_shared") and rust_ctx.is_shared()
        if using_shared_solver:
            new_constraints = []
        else:
            # When orig_state is a snapshot, rely on orig_constraint_count fast path
            new_constraints = self._extract_new_constraints(
                succ_state if is_snapshot else orig_state,
                succ_state,
                orig_constraints=orig_constraints,
                orig_constraint_count=orig_constraint_count,
            )
            if new_constraints:
                if _DBG:
                    l.debug(f"Extracted {len(new_constraints)} new constraints from callback")

        # For zero-length hooks, tell Rust to skip the hook on next step
        if skip_hook_addr is not None:
            try:
                self._rust_mgr.set_skip_hook_addr(skip_hook_addr)
                if _DBG:
                    l.debug(f"Set skip_hook_addr to 0x{skip_hook_addr:x}")
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: set_skip_hook_addr failed; the next
                # step may re-trigger the zero-length hook. Debug-logs.
                if _DBG:
                    l.debug(f"Could not set skip_hook_addr: {e}")

        # A bounced allocator bumped the Python break pointers; push them back
        # so a later native malloc does not overlap what it handed out.
        self._sync_state_heap_to_rust(succ_state, event.callback_state_id)
        self._sync_state_posix_to_rust(succ_state, event.callback_state_id)
        self._sync_state_posix_fds_to_rust(succ_state, event.callback_state_id)

        # Resume Rust with the changes and any new constraints.
        # IMPORTANT: This must happen BEFORE symbolic imports, because
        # apply_changes writes concrete data which clears symbolic page markers.
        # Importing symbolic values after resume re-sets the markers correctly.
        self._rust_mgr.resume_after_simprocedure(
            event.callback_state_id, new_pc, reg_changes, mem_changes or None, new_constraints or None
        )

        # Import symbolic memory to Rust AFTER resume.
        # The resume's apply_changes writes concrete witnesses which clear
        # symbolic page markers. Re-importing symbolic values here restores
        # the markers and stores the symbolic objects for VEX loads.
        # Use import_symbolic_to_state (by state_id) since pending_callback
        # was consumed by resume_after_simprocedure.
        state_id = event.callback_state_id
        all_sym_imports = self._merge_symbolic_imports(symbolic_imports, tracked_symbolic_writes)

        for addr, ast in all_sym_imports:
            try:
                self._rust_mgr.import_symbolic_to_state(state_id, addr, ast)
                if _DBG:
                    l.debug(f"Imported symbolic memory at 0x{addr:x} to state {state_id}")
            except Exception as e:
                # cat-(c) WRONG-ANSWER RISK: import_symbolic_to_state failed; Rust
                # loses the symbolic relationship at this address — downstream
                # loads see concrete witnesses only. Debug-logs.
                if _DBG:
                    l.debug(f"Could not import symbolic memory at 0x{addr:x}: {e}")

        # Update cache for future callbacks
        state_id = event.callback_state_id
        if state_id is not None:
            self._cache_state_mirror(state_id, succ_state)

    def _resume_with_skip_hook(
        self,
        addr: int,
        state: angr.SimState,
        orig_state: angr.SimState,
        event: _ExplorationEvent,
        orig_constraints: set | None = None,
        tracked_writes: list | None = None,
        tracked_symbolic_writes: list | None = None,
        orig_constraint_count: int | None = None,
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
            if _DBG:
                l.debug(f"Set skip_hook_addr to 0x{addr:x}")
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: set_skip_hook_addr failed; same as
            # 942 — next step may re-trigger the hook. Debug-logs.
            if _DBG:
                l.debug(f"Could not set skip_hook_addr: {e}")

        # Extract actual next PC from the modified state
        # This handles hooks that manually set the return address (e.g., pop ret simulation)
        new_pc = self._resolve_next_pc(state, fallback_addr=addr)

        if new_pc != addr:
            if _DBG:
                l.debug(f"Hook modified IP from 0x{addr:x} to 0x{new_pc:x}")

        # Extract register changes between original and modified state
        # This syncs all register modifications made by the hook.
        # angr-qj30 (write-through .2): with the register-proxy gate on,
        # hook writes to ``state.regs.<name>`` already landed in Rust.
        is_snapshot = isinstance(orig_state, dict)
        if getattr(self, "_use_callback_register_proxy", False):
            reg_changes = []
        else:
            reg_changes = self._extract_register_changes(orig_state, state)

        # Skip memory extraction when orig_state is a register snapshot
        # (non-memory-writing extern SimProcedures don't modify memory)
        if is_snapshot:
            mem_changes = []
            symbolic_imports = []
        else:
            mem_changes, symbolic_imports = self._extract_memory_changes(orig_state, state)

        # Merge tracked writes with extracted memory changes
        # Tracked writes capture symbolic stores that _extract_memory_changes might miss
        if tracked_writes:
            mem_changes = self._merge_tracked_writes(mem_changes, tracked_writes)
            if _DBG:
                l.debug(f"Merged {len(tracked_writes)} tracked writes with memory changes")

        # Exclude symbolic-import addresses from concrete memory changes so
        # apply_changes can't overwrite them with concrete values.
        mem_changes = self._filter_symbolic_addrs(mem_changes, symbolic_imports, tracked_symbolic_writes)

        # Extract new constraints added during hook execution
        new_constraints = None
        if hasattr(state, "solver"):
            try:
                current_count = len(state.solver.constraints)
                # Fast path: if count unchanged, skip expensive set construction
                if orig_constraint_count is not None and current_count == orig_constraint_count:
                    pass  # No new constraints
                elif orig_constraints is not None:
                    current_constraints = set(state.solver.constraints)
                    new_constraints = list(current_constraints - orig_constraints)
                    if new_constraints:
                        if _DBG:
                            l.debug(f"Extracted {len(new_constraints)} constraints from hook")
                elif orig_constraint_count is not None and not is_snapshot:
                    # Count changed — need full diff (only when orig_state is a real state)
                    prior = set(orig_state.solver.constraints)
                    current_constraints = set(state.solver.constraints)
                    new_constraints = list(current_constraints - prior)
                    if new_constraints:
                        if _DBG:
                            l.debug(f"Extracted {len(new_constraints)} constraints from hook")
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: constraint extraction raised; new
                # constraints from the hook are not synced to Rust. Debug-logs.
                if _DBG:
                    l.debug(f"Could not extract constraints: {e}")

        # Push back any heap bump the hook made (see _resume_with_state).
        self._sync_state_heap_to_rust(state, event.callback_state_id)
        self._sync_state_posix_to_rust(state, event.callback_state_id)
        self._sync_state_posix_fds_to_rust(state, event.callback_state_id)

        # Resume Rust with all extracted changes.
        # IMPORTANT: This must happen BEFORE symbolic imports (same as _resume_with_state).
        self._rust_mgr.resume_after_simprocedure(
            event.callback_state_id, new_pc, reg_changes or None, mem_changes or None, new_constraints or None
        )

        # Import symbolic memory AFTER resume (see _resume_with_state for rationale)
        state_id = event.callback_state_id
        all_sym_imports = self._merge_symbolic_imports(symbolic_imports, tracked_symbolic_writes)

        for sym_addr, ast in all_sym_imports:
            try:
                self._rust_mgr.import_symbolic_to_state(state_id, sym_addr, ast)
                if _DBG:
                    l.debug(f"Imported symbolic memory at 0x{sym_addr:x} to state {state_id}")
            except Exception as e:
                # cat-(c) WRONG-ANSWER RISK: import_symbolic_to_state failed in the
                # zero-length-hook path; same divergence as 973. Debug-logs.
                if _DBG:
                    l.debug(f"Could not import symbolic memory at 0x{sym_addr:x}: {e}")

        # Update cache with modified state for future callbacks
        if state_id is not None:
            self._cache_state_mirror(state_id, state)

    def _extract_new_constraints(
        self,
        orig_state: angr.SimState,
        new_state: angr.SimState,
        orig_constraints: set | None = None,
        orig_constraint_count: int | None = None,
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
                        if _DBG:
                            l.debug(f"Callback added {len(python_added)} new Python constraints")
                        new_constraints.extend(python_added)
            elif orig_constraints is not None:
                prior = orig_constraints
                state_constraints = set(new_constraint_list)
                python_added = state_constraints - prior
                if python_added:
                    if _DBG:
                        l.debug(f"Callback added {len(python_added)} new Python constraints")
                    new_constraints.extend(python_added)
            else:
                prior = set(orig_state.solver.constraints)
                state_constraints = set(new_constraint_list)
                python_added = state_constraints - prior
                if python_added:
                    if _DBG:
                        l.debug(f"Callback added {len(python_added)} new Python constraints")
                    new_constraints.extend(python_added)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: constraint set diff raised; treat as
            # 'no new constraints' rather than crashing. Debug-logs.
            if _DBG:
                l.debug(f"Could not extract Python constraints: {e}")

        if new_constraints:
            if _DBG:
                l.debug(f"Total new constraints to sync: {len(new_constraints)}")

        return new_constraints

    def _add_forked_state(self, succ_state: angr.SimState, event: _ExplorationEvent):
        """Add a forked state from a SimProcedure to Rust active stash.

        When a SimProcedure creates multiple successors (e.g., fork on
        symbolic condition), additional states are added to Rust for
        continued exploration.

        Two paths controlled by the ``use_simproc_fork_via_rust`` gate
        (angr-t3mr, write-through boundary):

        - **Gate on** (write-through): fork the parent Rust state directly
          via ``fork_state_to_stash(parent_id, 'active')`` and apply any
          path-specific constraints via ``add_constraints_to_state(new_id,
          ...)``. No Python-side ``_add_rust_state`` push and no
          ``add_constraints_to_pending`` write to the parent.

        - **Gate off** (legacy diff-and-push, default): re-push every
          register / memory page / constraint from ``succ_state`` via
          ``_add_rust_state('active', succ_state)``, then add the path
          constraints back to the pending state.
        """
        if getattr(self, "_use_simproc_fork_via_rust", False):
            self._add_forked_state_via_rust(succ_state, event)
            return

        # Try to fork the pending solver context for this state
        # This ensures the forked state inherits all constraints
        try:
            forked_solver = self._rust_mgr.fork_pending_solver(event.callback_state_id)
            # Attach forked solver to the state
            succ_state.scratch.rust_solver_ctx = forked_solver
            if _DBG:
                l.debug(
                    f"Forked solver context for additional successor ({forked_solver.num_constraints()} constraints)"
                )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: solver-context fork for additional
            # successor failed; the forked state inherits no Rust constraints.
            # Debug-logs.
            if _DBG:
                l.debug(f"Could not fork solver for additional successor: {e}")

        # Extract any constraints specific to this fork path
        # These may differ from the main successor due to branching conditions
        fork_constraints = []
        if hasattr(succ_state, "scratch") and hasattr(succ_state.scratch, "rust_solver_ctx"):
            try:
                # If the state has path-specific constraints, extract them
                if hasattr(succ_state.solver, "constraints"):
                    fork_constraints = list(succ_state.solver.constraints)
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: per-fork constraint extraction failed;
                # the forked state syncs no path-specific constraints to Rust.
                pass

        # Create a new Rust state for this successor
        new_sid = self._add_rust_state("active", succ_state)

        # Lineage-aware demotion (angr-qluof pt2): _add_rust_state's
        # _export_fs_files_to_rust re-registers eligible SimFiles, re-arming
        # any symbolic-file path the parent lineage's native write had demoted.
        # Re-apply from the parent so the forked guest keeps seeing the
        # content-identical Python fallback.
        self._reapply_demoted_paths(self._rust_mgr, event.callback_state_id, new_sid)

        # If we have fork-specific constraints, sync them to the new Rust state
        if fork_constraints:
            try:
                # The state was just added, so sync constraints to the pending/active state
                self._rust_mgr.add_constraints_to_pending(event.callback_state_id, fork_constraints)
                if _DBG:
                    l.debug(f"Synced {len(fork_constraints)} fork constraints to Rust state")
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: add_constraints_to_pending failed
                # for the forked state; constraints stay only on the Python side.
                # Debug-logs.
                if _DBG:
                    l.debug(f"Could not sync fork constraints: {e}")

        if _DBG:
            l.debug(f"Added forked state at PC 0x{succ_state.addr:x}")

    def _add_forked_state_via_rust(self, succ_state: angr.SimState, event: _ExplorationEvent) -> None:
        """Write-through fork: ask Rust to fork the parent state directly.

        Used when ``use_simproc_fork_via_rust`` is on (angr-t3mr). The
        parent state is the pending callback state (``event.callback_state_id``
        — ``find_state`` looks through the pending callback when the ID is
        not yet in a stash, so the fork sees the constraints/registers/
        memory that the SimProcedure has just written via the proxy
        plugins). Any path-specific constraints the SimProcedure added to
        ``succ_state.solver`` are pushed onto the new fork via
        ``add_constraints_to_state`` so per-fork branch conditions land on
        the fork's own solver instead of the pending one.
        """
        parent_id = event.callback_state_id
        if parent_id is None:
            # cat-(b) FALLBACK WITH LOSS: no parent ID on the event — the
            # callback was already torn down. Skip the fork.
            if _DBG:
                l.debug("_add_forked_state_via_rust: no callback_state_id, skipping fork")
            return

        try:
            new_id = self._rust_mgr.fork_state_to_stash(parent_id, "active")
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: Rust-owned fork failed; this
            # successor is silently dropped. Debug-logs.
            if _DBG:
                l.debug(f"fork_state_to_stash({parent_id}) failed: {e}")
            return

        # Path-specific constraints from succ_state. The fork already
        # carries every constraint the parent had (Rust forked the parent
        # state, including its solver), so add_constraints_to_state will
        # mostly re-assert duplicates that Z3 dedups — but it also lands
        # any branch conditions the SimProc added to this successor's
        # claripy solver (when the callback solver proxy is off and the
        # parallel claripy solver carries them).
        try:
            if hasattr(succ_state.solver, "constraints"):
                fork_constraints = list(succ_state.solver.constraints)
                if fork_constraints:
                    self._rust_mgr.add_constraints_to_state(new_id, fork_constraints)
                    if _DBG:
                        l.debug(
                            f"fork_state_to_stash: added {len(fork_constraints)} fork constraints to state {new_id}"
                        )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: constraint sync onto the new fork
            # failed; the fork inherits only the parent's constraints.
            if _DBG:
                l.debug(f"add_constraints_to_state({new_id}) failed: {e}")

        if _DBG:
            l.debug(
                f"_add_forked_state_via_rust: forked parent {parent_id} -> state {new_id} at PC 0x{succ_state.addr:x}"
            )

    def _handle_syscall_callback(self, event: _ExplorationEvent):
        """Handle syscall callback from Rust.

        Uses cached state to preserve symbolic memory and constraints,
        then syncs changes back to Rust after syscall execution.

        Entered from ``RunResult::Syscall`` via ``call_on_syscall``
        (``callbacks.rs``). The Rust side hard-errors when the
        ``on_syscall`` hook is ``None`` per
        ``avoid-silent-no-op-callback-fallbacks`` (callbacks.rs module
        invariant 1); never assume a missing hook turns syscalls into
        no-ops.
        """
        _sc_total_start = time.perf_counter_ns()
        try:
            self._handle_syscall_callback_inner(event)
        finally:
            self._perf_stats.record_syscall_call(time.perf_counter_ns() - _sc_total_start)

    def _handle_syscall_callback_inner(self, event: _ExplorationEvent):
        syscall_num = event.callback_syscall_num
        state_id = event.callback_state_id

        # Track current callback state ID for symbolic memory preservation
        self._current_callback_state_id = state_id

        # Create a state for syscall handling (uses cached state if available)
        state = self._create_state_for_callback(event)
        if state is None:
            l.warning(f"Could not create state for syscall {syscall_num}")
            # Continue after syscall
            pc = self._rust_mgr.get_pending_register(state_id, "rip") or 0
            self._rust_mgr.resume_after_syscall(state_id, pc + 1, None, None)
            return

        # Run the syscall
        try:
            # Resolve the syscall handler so the BP attrs match Python's
            # engine (procedure.py:34 passes `procedure.display_name`).
            # If resolution fails, fall back to the numeric ID so the BP
            # still fires.
            try:
                sc_proc = self._project.simos.syscall(state)
                sc_name = getattr(sc_proc, "display_name", None) or str(syscall_num)
            except Exception:
                # cat-(a) EXPECTED CONTROL FLOW: SIMOS may not expose
                # syscall() (non-Unix targets). Fall back to numeric ID.
                sc_proc = None
                sc_name = str(syscall_num)

            if sc_proc is None:
                raise AngrUnsupportedSyscallError(f"no syscall handler for {syscall_num}")

            # Linux's syscall CC returns to the ``ip_at_syscall`` register,
            # which VEX sets on the syscall exit. Rust builds the callback
            # state directly (pc already points past the syscall) and never
            # populates it, so ``cc.teardown_callsite`` would hand back a
            # return address of 0 (angr-89w70).
            if "ip_at_syscall" in state.arch.registers:
                state.regs.ip_at_syscall = state.addr

            # state.inspect syscall BEFORE (angr-xmfj).
            self._cb_inspect_syscall(state_id, "before", sc_name, None)

            # Execute the resolved handler on the procedure engine. Driving
            # ``default_engine.process(state, procedure=None)`` instead raised
            # unconditionally: ``SimEngineSyscall.process_successors`` forwards
            # **kwargs into ``process_procedure``, which already receives the
            # procedure positionally, so ``procedure=None`` was a duplicate
            # argument. The TypeError was then swallowed below (angr-89w70).
            # The syscall mixin cannot resolve the handler for us anyway — the
            # callback state has no ``Ijk_Sys*`` parent jumpkind.
            successors = self._project.factory.procedure_engine.process(state, procedure=sc_proc)

            # state.inspect syscall AFTER (angr-xmfj). `simprocedure` is
            # the resolved syscall handler (None if resolution failed),
            # matching the Python engine's `simprocedure=inst` kwarg.
            self._cb_inspect_syscall(state_id, "after", sc_name, sc_proc)

            all_succs = successors.all_successors
            if all_succs:
                # First successor continues in Rust
                succ_state = all_succs[0]

                # Handle symbolic IP: pick first concrete solution if symbolic
                new_pc = self._resolve_next_pc(succ_state)

                # angr-qj30 (write-through .2): with the register-proxy
                # gate on, syscall writes to ``state.regs.<name>`` already
                # landed in Rust during the callback.
                if getattr(self, "_use_callback_register_proxy", False):
                    reg_changes = []
                else:
                    reg_changes = self._extract_register_changes(state, succ_state)
                # ``_extract_memory_changes`` returns (concrete, symbolic);
                # passing the 2-tuple straight to ``resume_after_syscall``
                # raised inside PyO3 and the except-swallow below then resumed
                # at PC+1 with every change discarded (angr-89w70).
                mem_changes, symbolic_imports = self._extract_memory_changes(state, succ_state)
                mem_changes = self._filter_symbolic_addrs(mem_changes, symbolic_imports, None)
                new_constraints = self._extract_new_constraints(state, succ_state)

                # Resume BEFORE the symbolic imports: apply_changes writes
                # concrete witnesses that clear symbolic page markers, so the
                # imports must re-set them afterwards (mirrors
                # ``_resume_with_state``).
                self._rust_mgr.resume_after_syscall(
                    state_id, new_pc, reg_changes, mem_changes or None, new_constraints or None
                )

                for sym_addr, ast in symbolic_imports:
                    try:
                        self._rust_mgr.import_symbolic_to_state(state_id, sym_addr, ast)
                    except Exception as e:
                        # cat-(c) WRONG-ANSWER RISK: Rust loses the symbolic
                        # relationship at this address — downstream loads see
                        # the concrete witness only.
                        if _DBG:
                            l.debug(f"Could not import symbolic syscall memory at 0x{sym_addr:x}: {e}")

                # Update cache
                state_id = event.callback_state_id
                if state_id is not None:
                    self._cache_state_mirror(state_id, succ_state)

                # Handle additional successors
                for succ in all_succs[1:]:
                    self._add_forked_state(succ, event)
            else:
                pc = state.addr + 1
                self._rust_mgr.resume_after_syscall(state_id, pc, None, None)

        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: syscall execution raised; we resume
            # at PC+1 rather than re-running the syscall, discarding every
            # register/memory change the syscall made. Counted so a run that
            # hit this path is identifiable from stats() (angr-89w70).
            self._perf_stats.record_syscall_error()
            l.warning(f"Syscall execution error: {e}")
            pc = state.addr + 1
            self._rust_mgr.resume_after_syscall(state_id, pc, None, None)
        finally:
            # Clear callback state to avoid stale references
            self._set_callback_state(None)
            self._current_callback_state_id = None

    def _handle_find_predicate_callback(self, event: _ExplorationEvent):
        """Handle callable find predicate evaluation callback from Rust.

        Uses a lightweight RustStateProxy to evaluate the predicate without
        creating a full SimState. This avoids the ~30ms per-callback overhead
        of _create_state_for_callback while providing live register/memory
        access from the Rust pending state.

        Args:
            event: The exploration event from Rust.

        Predicate evaluation depends on ``drop_terminal_states == False``
        (``drop-terminal-vs-predicates``; mirrored in
        ``callbacks.rs`` module invariant 2). When predicates are active,
        states that print then ``exit()`` must survive until predicate
        evaluation runs — without that, callable-find examples like
        ``sym-write`` produce zero found states.
        """
        _fp_total_start = time.perf_counter_ns()
        try:
            self._handle_find_predicate_callback_inner(event)
        finally:
            self._perf_stats.record_find_predicate_call(time.perf_counter_ns() - _fp_total_start)

    def _handle_find_predicate_callback_inner(self, event: _ExplorationEvent):
        from angr.exploration.rust_state_proxy import RustStateProxy

        state_id = event.callback_state_id
        addr = event.callback_addr

        if self._find_predicate is None:
            self._rust_mgr.resume_find_predicate(state_id, False)
            return

        try:
            # Use lightweight proxy — reads registers/memory directly from
            # Rust pending state. No SimState creation needed.
            # Build a proxy with the pending state's PC for ip access.
            proxy = RustStateProxy(
                self._rust_mgr,
                state_id,
                project=self._project,
                stdin_vars=getattr(self, "_stdin_vars", None),
                python_mgr=self,
            )
            # Override IP to use the callback address (the PLT/hook address),
            # since the pending state may not be in the state index yet.
            proxy._override_addr = addr

            try:
                result = self._find_predicate(proxy)
                matched = bool(result) if result is not None else False
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: find predicate raised on this state;
                # treated as not-matched. Debug-logs.
                if _DBG:
                    l.debug(f"Find predicate at 0x{addr:x}: {e}")
                matched = False

            self._rust_mgr.resume_find_predicate(state_id, matched)

            # If matched, store the proxy as the found state
            if matched:
                if not hasattr(self, "_predicate_found"):
                    self._predicate_found = []
                self._predicate_found.append(proxy)

        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: outer find-predicate handler raised;
            # state is reported as not-matched even if it would have been.
            # Already warns.
            l.warning(f"Find predicate callback error: {e}")
            self._rust_mgr.resume_find_predicate(state_id, False)

    def _handle_avoid_predicate_callback(self, event: _ExplorationEvent):
        """Handle callable avoid predicate evaluation callback from Rust.

        Uses a lightweight RustStateProxy to evaluate the predicate without
        creating a full SimState.

        Args:
            event: The exploration event from Rust.

        Same ``drop-terminal-vs-predicates`` invariant as
        ``_handle_find_predicate_callback`` — callable avoid is paired
        with callable find at the manager level; both rely on terminal
        states remaining alive for evaluation. See ``callbacks.rs``
        module invariant 2.
        """
        _ap_total_start = time.perf_counter_ns()
        try:
            self._handle_avoid_predicate_callback_inner(event)
        finally:
            self._perf_stats.record_avoid_predicate_call(time.perf_counter_ns() - _ap_total_start)

    def _handle_avoid_predicate_callback_inner(self, event: _ExplorationEvent):
        from angr.exploration.rust_state_proxy import RustStateProxy

        state_id = event.callback_state_id
        addr = event.callback_addr

        if self._avoid_predicate is None:
            self._rust_mgr.resume_avoid_predicate(state_id, False)
            return

        try:
            proxy = RustStateProxy(
                self._rust_mgr,
                state_id,
                project=self._project,
                stdin_vars=getattr(self, "_stdin_vars", None),
                python_mgr=self,
            )
            proxy._override_addr = addr

            try:
                result = self._avoid_predicate(proxy)
                matched = bool(result) if result is not None else False
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: avoid predicate raised on this state;
                # treated as not-matched (state stays in active). Debug-logs.
                if _DBG:
                    l.debug(f"Avoid predicate at 0x{addr:x}: {e}")
                matched = False

            self._rust_mgr.resume_avoid_predicate(state_id, matched)

        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: outer avoid-predicate handler raised;
            # state is reported as not-matched even if it would have been
            # avoided. Already warns.
            l.warning(f"Avoid predicate callback error: {e}")
            self._rust_mgr.resume_avoid_predicate(state_id, False)

    def _handle_symbolic_branch_callback(self, event: _ExplorationEvent):
        """Handle symbolic branch callback from Rust.

        When a symbolic branch with both paths feasible is encountered,
        Rust returns to Python for proper state forking with constraints.
        This ensures symbolic branches are handled correctly even when
        hooks/callbacks occur, preventing lost forks.

        The handler:
        1. Gets the branch condition from Rust
        2. Creates two forked states with appropriate constraints
        3. Adds both states back to Rust's active stash

        Entered from ``RunResult::SymbolicBranch`` (callbacks.rs). The
        per-state sync helpers invoked here (memory, registers,
        callstack) MUST be mirrored across the four export paths in
        ``rust_state_export.py::_materialize_single_state`` per
        ``invariant-callstack-sync-export-pipeline`` — callbacks.rs
        module invariant 4. Any new per-state sync added during forking
        needs the matching export-path wiring.
        """
        _sb_total_start = time.perf_counter_ns()
        try:
            self._handle_symbolic_branch_callback_inner(event)
        finally:
            self._perf_stats.record_symbolic_branch_call(time.perf_counter_ns() - _sb_total_start)

    def _handle_symbolic_branch_callback_inner(self, event: _ExplorationEvent):
        true_target = event.branch_true_target
        false_target = event.branch_false_target
        condition_id = event.branch_condition_id
        state_id = event.callback_state_id

        if _DBG:
            l.debug(
                f"Handling symbolic branch: true=0x{true_target:x}, false=0x{false_target:x}, cond_id={condition_id}"
            )

        try:
            # Get the branch condition from Rust as a claripy AST
            condition = self._rust_mgr.get_pending_branch_condition(state_id)

            if _DBG:
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

            if _DBG:
                l.debug(f"True constraint: {true_constraint}")
            if _DBG:
                l.debug(f"False constraint: {false_constraint}")

            # Resume Rust with the forked states
            # Pass constraints as lists for each branch
            # Get active state IDs before fork
            ids_before = set(self._rust_mgr.get_state_ids("active"))

            self._rust_mgr.resume_after_symbolic_branch(
                state_id,
                true_target,
                false_target,
                [true_constraint],
                [false_constraint],
            )

            # Cache Python states for new forked Rust states
            ids_after = set(self._rust_mgr.get_state_ids("active"))
            new_ids = ids_after - ids_before
            if new_ids and state_id in self._state_cache:
                parent_state = self._state_cache[state_id]
                for new_id in new_ids:
                    forked_state = parent_state.copy()
                    self._cache_state_mirror(new_id, forked_state)
                    # Track lineage for plugin restoration
                    root = self._state_roots.get(state_id, state_id)
                    self._state_roots[new_id] = root

            if _DBG:
                l.debug(f"Resumed after symbolic branch with {len(new_ids)} forked states")

        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: symbolic branch handler raised; we
            # still try the no-constraint fallback below, but until that runs
            # constraints are dropped. Already warns.
            l.warning(f"Symbolic branch handling error: {e}")
            import traceback

            l.warning(traceback.format_exc())

            # Fallback: just fork without proper constraints
            # This is less accurate but at least continues exploration
            try:
                self._rust_mgr.resume_after_symbolic_branch(
                    state_id,
                    true_target,
                    false_target,
                    None,
                    None,
                )
                l.warning("Resumed after symbolic branch with fallback (no constraints)")
            except Exception as e2:
                # cat-(c) WRONG-ANSWER RISK: fallback resume_after_symbolic_branch
                # (without constraints) failed; try moving to errored next.
                # Already warns.
                l.error(f"Failed to resume after symbolic branch: {e2}")
                # Recovery: Move the pending state to errored stash to avoid hanging.
                # Uses the same error handling as other callback failures.
                try:
                    self._rust_mgr.resume_after_error(state_id, f"symbolic_branch_error: {e2}")
                    l.warning("Moved state to errored stash after symbolic branch failure")
                except Exception as e3:
                    # cat-(b) FALLBACK WITH LOSS: even errored-stash recovery failed;
                    # the pending state hangs the next step. Already warns.
                    l.error(f"Failed to move state to errored stash: {e3}")

    def _get_pending_parent_id(self) -> int | None:
        """Get parent state ID of current pending callback state.

        When Rust forks a state during exploration, the forked state gets a
        new state_id but the Python caches only have the parent's state_id.
        This method retrieves the parent_id from the pending state snapshot
        so we can look up the parent's caches.

        Returns:
            The parent state ID, or None if not available.
        """
        try:
            snapshot = self._rust_mgr.export_pending_state(self._current_callback_state_id)
            return snapshot.parent_id
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: parent_id lookup failed; the cache
            # fallback to root/ancestry is the next line.
            return None

    def _get_pending_root_state_id(self) -> int | None:
        """Get root state ID for the pending callback state.

        When Rust forks states internally (multi-level forks), Python only has
        cached data for the original state that was added via Python. This method
        returns the root state ID (the original ancestor) for any forked descendant.

        Returns:
            The root state ID if available, or None if not tracked.
        """
        try:
            return self._rust_mgr.get_pending_root_state_id(self._current_callback_state_id)
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: root_id lookup failed; ancestry walk
            # below picks up.
            return None

    def _get_effective_state_id(self, state_id: int | None) -> int | None:
        """Get effective state ID following lineage for lookups.

        When a state is forked in Rust, its ID changes but Python's caches
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
            # cat-(b) FALLBACK WITH LOSS: root_id lookup raised; fall through
            # to ancestry walk.
            pass

        # Walk full ancestry chain from pending callback
        try:
            ancestry = self._get_pending_ancestry()
            for ancestor_id in ancestry:
                if ancestor_id in self._state_cache:
                    return ancestor_id
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: ancestry walk raised; fall back to
            # original state_id.
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
            return self._rust_mgr.get_pending_ancestry(self._current_callback_state_id)
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: get_pending_ancestry failed; fall
            # back to single parent_id lookup.
            # Fallback to single parent lookup
            parent_id = self._get_pending_parent_id()
            return [parent_id] if parent_id is not None else []

    def _create_state_for_callback(self, event: _ExplorationEvent) -> angr.SimState | None:
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
                if not hasattr(cached_state, "solver") or not hasattr(cached_state, "memory"):
                    l.warning(f"Cached state {state_id} invalid type: {type(cached_state)}, clearing")
                    del self._state_cache[state_id]
                    cached_state = None
        elif state_id is not None:
            # Forked state - try root state first (most likely to be cached)
            root_id = self._get_pending_root_state_id()
            if root_id is not None and root_id in self._state_cache:
                cached_state = self._state_cache[root_id]
                lookup_state_id = root_id
                if _DBG:
                    l.debug(f"Using root state {root_id} cache for forked state {state_id}")
            else:
                # Walk full ancestry chain to find any cached ancestor
                for ancestor_id in self._get_pending_ancestry():
                    if ancestor_id in self._state_cache:
                        cached_state = self._state_cache[ancestor_id]
                        lookup_state_id = ancestor_id
                        if _DBG:
                            l.debug(f"Using ancestor {ancestor_id} cache for forked state {state_id}")
                        break

        if cached_state is not None:
            self._stats_cache_hits += 1
            # Copy when callable predicates need stdout history, skip otherwise
            has_predicates = (
                getattr(self, "_find_predicate", None) is not None
                or getattr(self, "_avoid_predicate", None) is not None
            )
            # angr-4scu step 3: when the callback-memory-proxy gate is on we
            # always copy the cached state. The gate swaps ``state.memory``
            # to a ``RustMemoryProxy`` later in this function; mutating the
            # cached state's memory plugin in place would leak the proxy
            # into the cached copy and break ordinary (non-callback) reads
            # against that cache entry. The copy ensures the swap only
            # affects this one callback frame.
            if (
                has_predicates
                or getattr(self, "_use_callback_memory_proxy", False)
                or getattr(self, "_use_callback_register_proxy", False)
                or getattr(self, "_use_callback_callstack_proxy", False)
            ):
                self._stats_state_creations += 1
                _sc_copy_start = time.perf_counter_ns()
                state = cached_state.copy()
                self._perf_stats.add_state_create_subphase("copy", time.perf_counter_ns() - _sc_copy_start)
            else:
                state = cached_state

            # Use the callback bundle API to get registers + solver + history
            # in a single FFI call instead of ~20 individual calls.
            _sc_bundle_start = time.perf_counter_ns()
            try:
                arch = self._project.arch
                reg_names = self._get_arch_register_names(arch)
                bundle = self._rust_mgr.export_callback_bundle(state_id, reg_names)

                # Apply solver from bundle
                forked_solver = bundle["solver"]
                state.scratch.rust_solver_ctx = forked_solver
                constraint_count = bundle["constraint_count"]
                if _DBG:
                    l.debug(f"Bundle: solver with {constraint_count} constraints for state {state_id}")

                # Apply registers from bundle using direct store (bypasses claripy BVV creation)
                # Save bundle registers for snapshot reuse in _handle_simprocedure_callback
                # angr-qj30 (write-through .2): when the register-proxy gate
                # is on, skip the bundle register apply entirely. The proxy
                # we install below at ``_install_callback_register_proxy``
                # reads every register live from Rust by ``state_id`` — the
                # cached state's register file will be replaced wholesale,
                # so writing to it here would be wasted work.
                registers = bundle["registers"]
                if not getattr(self, "_use_callback_register_proxy", False):
                    self._last_bundle_registers = registers
                    reg_map = self._get_register_offset_map(arch)
                    for reg_name, val in registers.items():
                        try:
                            if val is not None:
                                offset_size = reg_map.get(reg_name)
                                if offset_size is not None:
                                    offset, size = offset_size
                                    state.registers.store(offset, val, size=size)
                                else:
                                    setattr(state.regs, reg_name, claripy.BVV(val, arch.bits))
                            else:
                                # Symbolic register — fetch AST individually
                                try:
                                    ast = self._rust_mgr.get_pending_register_ast(state_id, reg_name)
                                    if ast is not None:
                                        setattr(state.regs, reg_name, ast)
                                except Exception:
                                    # cat-(b) FALLBACK WITH LOSS: symbolic register AST fetch failed;
                                    # the register stays at the blank-state default rather than the
                                    # pending Rust value.
                                    pass
                        except Exception:
                            # cat-(b) FALLBACK WITH LOSS: per-register apply failed; that
                            # register may diverge from the pending Rust state.
                            pass
                else:
                    # Gate-on path: still null the bundle-snapshot field so
                    # ``_snapshot_orig_state`` falls through to the
                    # state-based snapshot (which then routes through the
                    # proxy → live Rust read). No diff is run end-of-callback
                    # so this snapshot is mostly inert, but keeping the
                    # field tidy avoids stale data from a prior callback.
                    self._last_bundle_registers = None

                # Cache history and jumpkind from bundle for later use
                state.scratch._rust_bundle_history = bundle.get("history", [])
                state.scratch._rust_bundle_jumpkind = bundle.get("jumpkind", "Ijk_Boring")
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: callback bundle API failed; falls
                # back to the slower individual-call path below. Debug-logs.
                if _DBG:
                    l.debug(f"Bundle API failed, falling back to individual calls: {e}")
                self._last_bundle_registers = None  # Clear on fallback
                # Fallback to individual calls — try shared (borrow) first, fork as last resort
                try:
                    shared_solver = self._rust_mgr.borrow_pending_solver(state_id)
                    state.scratch.rust_solver_ctx = shared_solver
                except Exception:
                    # cat-(b) FALLBACK WITH LOSS: borrow_pending_solver failed; try
                    # fork_pending_solver next.
                    try:
                        forked_solver = self._rust_mgr.fork_pending_solver(state_id)
                        state.scratch.rust_solver_ctx = forked_solver
                    except Exception as e2:
                        # cat-(c) WRONG-ANSWER RISK: even fork_pending_solver failed;
                        # the callback state has no Rust solver context — eval/satisfiable
                        # fall back to the (potentially out-of-date) Python solver.
                        # Already warns.
                        l.warning(f"Could not fork solver context: {e2}")
                # angr-qj30 (write-through .2): with the register-proxy gate
                # on, the proxy will be installed below and reads route
                # live to Rust — skip the pre-population sync path entirely.
                if not getattr(self, "_use_callback_register_proxy", False):
                    self._sync_registers_from_rust_pending(state)

            self._perf_stats.add_state_create_subphase("bundle", time.perf_counter_ns() - _sc_bundle_start)

            # Install memory proxy: wrap state.memory.load to check Rust
            # memory first for addresses that the Python state doesn't have
            # (stack frames created during VEX execution).
            _sc_memory_start = time.perf_counter_ns()
            self._install_rust_memory_proxy(state)
            self._perf_stats.add_state_create_subphase("meminstall", time.perf_counter_ns() - _sc_memory_start)
            _sc_replay_start = time.perf_counter_ns()

            # angr-3tek.2: replay Rust-side memory mutations (concrete +
            # symbolic) recorded in dirty_pages since the last callback.
            # Must run AFTER _install_rust_memory_proxy (which writes
            # concrete pointer slots from the SP page) and BEFORE
            # _restore_symbolic_pages (which restores Python-pushed
            # snapshots). Inside the helper, symbolic stores happen LAST
            # per page so they overwrite concrete defaults at the same
            # addresses — see invariant-3tek2-replay-ordering.
            self._replay_rust_dirty_pages(state)
            self._perf_stats.add_state_create_subphase("memreplay", time.perf_counter_ns() - _sc_replay_start)
            self._perf_stats.add_state_create_subphase("memory", time.perf_counter_ns() - _sc_memory_start)

            # Restore symbolic memory regions — only needed for copied states
            # (predicates case) or on first callback for a new state.
            # When reusing the same state object, symbolic pages persist.
            _sc_sympage_start = time.perf_counter_ns()
            if has_predicates or not getattr(state, "_rust_sympage_restored", False):
                self._restore_symbolic_pages(state, lookup_state_id)
                self._restore_hook_symbolic_memory(state, lookup_state_id)
                state._rust_sympage_restored = True
            self._perf_stats.add_state_create_subphase("sympage", time.perf_counter_ns() - _sc_sympage_start)

            if _DBG:
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
            # angr-ye2n: the implicit-contract check formerly hidden inside
            # _install_rust_solver_on_callback_state is now explicit here.
            # If every prior fork/borrow attempt failed (see warnings above),
            # we have no Rust context to hand to the install — increment the
            # defensive counter and skip the install. This leaves Python's
            # solver untouched (no closures installed; the cached_state's
            # prior closures, if any, still route to the latest
            # state.scratch.rust_solver_ctx).
            # angr-8oiw (write-through .3): when the solver-proxy gate is
            # on, install ``RustSolverProxyPlugin`` instead of the
            # monkey-patched closure path. The proxy writes through to the
            # Rust state's solver by state_id (no parallel Python claripy
            # solver). Skip the closure install entirely in that mode.
            if getattr(self, "_use_callback_solver_proxy", False) and event.callback_state_id is not None:
                self._install_callback_solver_proxy(state, event.callback_state_id)
            else:
                _cb_rust_ctx = getattr(state.scratch, "rust_solver_ctx", None)
                if _cb_rust_ctx is None:
                    # angr-bs71/h0dv: counter should stay at 0 across the
                    # benchmark suite. Non-zero readings here mean every
                    # solver attach path (bundle, borrow, fork) failed for
                    # this callback state and Python's solver may diverge
                    # from Rust's.
                    self._stats_rust_ctx_missing += 1
                    l.warning(
                        "rust_solver_ctx not attached to callback state at 0x%x; "
                        "skipping solver install — Python solver may diverge from Rust",
                        event.callback_addr or 0,
                    )
                else:
                    self._install_rust_solver_on_callback_state(state, _cb_rust_ctx)

            # Phase 3 Fix: Ensure critical plugins are present
            # Some scripts assume posix/libc plugins exist - restore if missing
            self._ensure_critical_plugins(state, event.callback_state_id)

            # angr-4scu step 3: gated install of RustMemoryProxy as
            # ``state.memory``. When the flag is on, every load/store the
            # SimProc issues routes directly into Rust by state_id —
            # eliminating the cached-state shadow memory + the
            # ``CallbackMemoryTracker`` diff-and-push at end-of-callback.
            # The matching site in ``_handle_simprocedure_callback`` forces
            # ``tracked_writes=None`` so the replay path is a no-op.
            if getattr(self, "_use_callback_memory_proxy", False) and event.callback_state_id is not None:
                self._install_callback_memory_proxy(state, event.callback_state_id)

            # angr-qj30 (write-through .2): gated install of
            # ``RustRegisterProxy`` as ``state.registers``. When on, every
            # ``state.regs.<name>`` read/write (and ``state.registers.{load,
            # store}``) routes directly into Rust by state_id — eliminating
            # the bundle register apply above and the post-callback
            # ``_extract_register_changes`` diff-and-push. The matching site
            # in ``_handle_simprocedure_callback`` forces ``reg_changes=[]``
            # so the resume path skips the apply.
            if getattr(self, "_use_callback_register_proxy", False) and event.callback_state_id is not None:
                self._install_callback_register_proxy(state, event.callback_state_id)

            # angr-6o9p (write-through .4): gated install of
            # ``RustCallStackProxyPlugin`` as ``state.callstack``. When on,
            # ``state.callstack`` iteration / top-frame attribute access
            # reads frames live from Rust by ``state_id`` — replacing the
            # cached-state's empty entry-state callstack with a live view
            # of the Rust call frames. Off keeps the cached state's plugin
            # untouched (callbacks see whatever was on the entry state).
            if getattr(self, "_use_callback_callstack_proxy", False) and event.callback_state_id is not None:
                self._install_callback_callstack_proxy(state, event.callback_state_id)

            # angr-op0dn.14.1.3: seed Rust's break pointers into the heap plugin
            # the first time the bounced proc touches it.
            if event.callback_state_id is not None:
                self._install_lazy_heap_sync(state, event.callback_state_id)
                # angr-op0dn.14.1.4: a bounced proc that reads its own stdout
                # (or appends to it) must see the native output that preceded
                # it. Idempotent, so the reused cached state is safe.
                self._inject_rust_stdout(state, event.callback_state_id)
                # angr-op0dn.14.1.6: fds a native open() created must be usable
                # by the bounced proc — and must be visible to posix._pick_fd so
                # a bounced open() does not collide with one of them.
                self._inject_rust_fds(state, event.callback_state_id)

        return state

    def _install_lazy_heap_sync(self, state, state_id):
        """Seed Rust's heap_brk / mmap_base into ``state.heap`` on first access.

        A bounced malloc/calloc/operator-new reads ``state.heap.heap_location``,
        which on a freshly copied callback state is still the default
        0xC0000000 — the same base Rust's bump allocator starts from. Any native
        allocation that already ran has moved Rust's break pointer past it, so
        the Python proc would hand out an address Rust already gave away.

        The sync cannot be eager: the heap plugin is lazy and materializing it
        costs ~65ms (``SimHeapBase.init_state`` maps the 8MB heap region), which
        every callback would pay whether or not it allocates. So hook the
        materialization instead — ``SimState.__getattr__`` routes an unregistered
        plugin through ``self.get_plugin``, so an instance-level override sees
        exactly the callbacks that do touch the heap and nothing else.

        The write-back (Python bump -> Rust) is
        ``_sync_state_heap_to_rust``, called from the resume paths.
        """
        if "heap" in state.plugins:
            # Already materialized (the init-time push in ``_add_rust_state``
            # touches ``state.heap``, so a state copied from the cache normally
            # lands here). Sync now — no hook needed.
            self._sync_rust_heap_brk_to_state(state, state_id)
            self._sync_rust_mmap_base_to_state(state, state_id)
            return

        orig_get_plugin = state.get_plugin

        def get_plugin(name):
            fresh = name not in state.plugins
            plugin = orig_get_plugin(name)
            if name == "heap" and fresh:
                # Both helpers re-enter through state.heap, but the plugin is
                # registered by now so this hook sees fresh=False and stops.
                self._sync_rust_heap_brk_to_state(state, state_id)
                self._sync_rust_mmap_base_to_state(state, state_id)
            return plugin

        state.get_plugin = get_plugin

    def _install_callback_memory_proxy(self, state, state_id):
        """Swap ``state.memory`` for a ``RustMemoryProxy`` bound to ``state_id``.

        angr-4scu step 3. Caller has already copied the cached state, so
        replacing the plugin here only affects this callback frame.
        """
        from angr.exploration.rust_state_proxy import RustMemoryProxy

        # angr-5rjbq: capture the Python SimMemory the callback state holds
        # BEFORE the swap. Setup-time writes (e.g. flareon2015_5's ebp-relative
        # pw symbols) live in this memory but were never synced to Rust at the
        # concretized address, so a proxy load that Rust can't satisfy falls
        # back here — matching gate-off DefaultMemory semantics.
        prev_memory = getattr(state, "memory", None)
        proxy = RustMemoryProxy(
            self._rust_mgr, state_id, self._project.arch, python_mgr=self, fallback_memory=prev_memory
        )
        proxy.set_state(state)
        # ``SimState.register_plugin`` would re-run ``set_state`` and update
        # ``state.plugins`` bookkeeping; use it so removal / merge code that
        # walks ``state.plugins`` finds the proxy under the ``"memory"`` key.
        state.register_plugin("memory", proxy)

    def _install_callback_register_proxy(self, state, state_id):
        """Swap ``state.registers`` for a ``RustRegisterProxy`` bound to ``state_id``.

        angr-qj30 (write-through .2). Caller has already copied the cached
        state, so replacing the plugin here only affects this callback frame.
        ``state.regs`` (the SimRegNameView) delegates to ``state.registers``
        via ``load(name)`` / ``store(name, val)``, so installing the proxy
        under the ``"registers"`` key is sufficient — no separate swap of
        ``state.regs`` is needed.
        """
        from angr.exploration.rust_state_proxy import RustRegisterProxy

        proxy = RustRegisterProxy(self._rust_mgr, state_id, self._project.arch, python_mgr=self)
        proxy.set_state(state)
        state.register_plugin("registers", proxy)

    def _install_callback_solver_proxy(self, state, state_id):
        """Swap ``state.solver`` for a ``RustSolverProxyPlugin`` bound to ``state_id``.

        angr-8oiw (write-through .3). Caller has already copied the cached
        state, so replacing the plugin here only affects this callback frame.
        The proxy routes ``state.solver.add`` through to the underlying Rust
        state's solver (no parallel Python claripy solver), and
        ``constraints`` / ``eval`` / ``satisfiable`` read through Rust.
        """
        from angr.exploration.rust_state_proxy import RustSolverProxyPlugin

        proxy = RustSolverProxyPlugin(self._rust_mgr, state_id, python_mgr=self)
        proxy.set_state(state)
        state.register_plugin("solver", proxy)

    def _install_callback_callstack_proxy(self, state, state_id):
        """Swap ``state.callstack`` for a ``RustCallStackProxyPlugin`` bound to ``state_id``.

        angr-6o9p (write-through .4). Caller has already copied the cached
        state, so replacing the plugin here only affects this callback
        frame. The proxy reads frames live from Rust by ``state_id`` via
        ``get_state_call_stack`` — no Python-side mirror.
        """
        from angr.exploration.rust_state_proxy import RustCallStackProxyPlugin

        proxy = RustCallStackProxyPlugin(self._rust_mgr, state_id)
        proxy.set_state(state)
        state.register_plugin("callstack", proxy)

    def _ensure_critical_plugins(self, state: angr.SimState, state_id: int | None):
        """Ensure critical plugins are present on the state.

        Phase 3 Fix: Some scripts and SimProcedures expect plugins like
        posix and libc to be present. If they're missing after state copy/creation,
        restore them from a template state.

        Args:
            state: The state to check/fix.
            state_id: The state ID for root state lookup.
        """
        # Check if critical plugins are missing.
        # IMPORTANT: Use state.plugins dict directly instead of hasattr/getattr.
        # hasattr() triggers __getattr__ which may lazy-initialize plugins
        # (e.g., 'heap' takes ~80ms to initialize on first access).
        missing_plugins = []
        for plugin_name in ["posix", "libc"]:
            if plugin_name not in state.plugins:
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
            if _DBG:
                l.debug(f"Phase 3: No template for plugin restoration, missing: {missing_plugins}")
            return

        # Restore missing plugins
        for plugin_name in missing_plugins:
            try:
                if hasattr(template, plugin_name):
                    plugin = getattr(template, plugin_name)
                    if plugin is not None and hasattr(plugin, "copy"):
                        state.register_plugin(plugin_name, plugin.copy())
                        if _DBG:
                            l.debug(f"Phase 3: Restored {plugin_name} plugin")
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: critical plugin restoration failed
                # for this plugin (e.g., template plugin lacks copy()). The state
                # proceeds without it; SimProcedures that touch it may crash.
                # Debug-logs.
                if _DBG:
                    l.debug(f"Phase 3: Could not restore {plugin_name}: {e}")

    def _create_blank_state_fallback(self, event: _ExplorationEvent) -> angr.SimState | None:
        """Create a blank state as fallback when no cached state is available."""
        try:
            # Create a blank state
            state = self._project.factory.blank_state()

            # Fork the Rust solver context even for blank states
            try:
                forked_solver = self._rust_mgr.fork_pending_solver(self._current_callback_state_id)
                state.scratch.rust_solver_ctx = forked_solver
                if _DBG:
                    l.debug(
                        f"Forked Rust solver for blank fallback state ({forked_solver.num_constraints()} constraints)"
                    )
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: solver fork on blank state failed;
                # the blank fallback proceeds without Rust solver context.
                # Debug-logs.
                if _DBG:
                    l.debug(f"Could not fork solver for blank state: {e}")

            # Copy registers from pending Rust state
            self._sync_registers_from_rust_pending(state)

            return state

        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: blank state creation raised; caller
            # treats None as 'no callback state' and may resume with corrupted
            # Rust state. Already warns.
            l.warning(f"Error creating blank state for callback: {e}")
            return None

    def _handle_python_vex_fallback(self, event: _ExplorationEvent):
        """Handle Python VEX engine fallback for unsupported Rust operations.

        When the Rust VEX interpreter encounters an unsupported operation
        (CAS, dirty calls, SIMD, etc.), it returns to Python to step the
        block using SimEngineVEX, then syncs results back to Rust.
        """
        _vf_total_start = time.perf_counter_ns()
        try:
            self._handle_python_vex_fallback_inner(event)
        finally:
            self._perf_stats.record_vex_fallback_call(time.perf_counter_ns() - _vf_total_start)

    def _handle_python_vex_fallback_inner(self, event: _ExplorationEvent):
        addr = event.callback_addr
        state_id = event.callback_state_id
        reason = event.callback_name or "unknown"

        if _DBG:
            l.debug(f"Python VEX fallback at 0x{addr:x}: {reason} (state {state_id})")

        try:
            # Create a full SimState from the Rust state
            state = self._create_state_for_callback(event)
            if state is None:
                l.warning(f"Could not create state for VEX fallback at 0x{addr:x}")
                self._rust_mgr.resume_after_simprocedure(state_id, addr, None, None)
                return

            # Step the state through Python's VEX engine for one block
            try:
                succs_obj = self._project.factory.successors(state, num_inst=99)
            except Exception as e:
                # cat-(c) WRONG-ANSWER RISK: Python VEX engine failed; we try
                # resume_after_error to errored-stash the state. Already warns.
                l.warning(f"Python VEX engine failed at 0x{addr:x}: {e}")
                try:
                    self._rust_mgr.resume_after_error(state_id, f"python_vex_fallback_error: {e}")
                except Exception:
                    # cat-(b) FALLBACK WITH LOSS: resume_after_error itself failed;
                    # the pending state hangs the next step. Tolerated.
                    pass
                return

            all_succs = succs_obj.all_successors
            if not all_succs:
                if _DBG:
                    l.debug(f"VEX fallback produced no successors at 0x{addr:x} — deadending")
                self._rust_mgr.deadend_pending_callback(state_id)
                return

            # Use the first successor as the primary result
            succ = all_succs[0]
            new_pc = succ.addr

            # Extract register changes
            # angr-qj30 (write-through .2): with the register-proxy gate
            # on, VEX fallback writes to ``state.regs.<name>`` already
            # landed in Rust during execution.
            if getattr(self, "_use_callback_register_proxy", False):
                reg_changes = []
            else:
                reg_changes = self._extract_register_changes(state, succ)

            # Extract memory changes (returns concrete_changes, symbolic_imports)
            mem_changes, symbolic_imports = self._extract_memory_changes(state, succ)

            # Extract new constraints
            orig_constraints = set(state.solver.constraints)
            new_constraints = [c for c in succ.solver.constraints if c not in orig_constraints]

            # Resume Rust with the first successor
            self._rust_mgr.resume_after_simprocedure(
                state_id, new_pc, reg_changes, mem_changes or None, new_constraints or None
            )

            # Additional successors (symbolic branches in the fallback block)
            # are forked into new Rust active states so the convergent path
            # isn't silently dropped.
            for extra_succ in all_succs[1:]:
                try:
                    self._add_forked_state(extra_succ, event)
                except Exception as fork_err:
                    # cat-(b) FALLBACK WITH LOSS: forking an additional VEX-fallback
                    # successor failed; that branch is silently dropped. Already warns.
                    l.warning(
                        f"VEX fallback at 0x{addr:x}: failed to fork successor at 0x{extra_succ.addr:x}: {fork_err}"
                    )

            self._perf_stats.increment_simprocedure_count()

        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: outer VEX fallback handler raised;
            # we try resume_after_error. Already warns.
            l.warning(f"Python VEX fallback error at 0x{addr:x}: {e}")
            try:
                self._rust_mgr.resume_after_error(state_id, f"python_vex_fallback_error: {e}")
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: resume_after_error in the outer
                # handler also failed; pending state hangs. Tolerated.
                pass
