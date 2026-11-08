"""Python wrapper for Rust-native exploration manager.

This module provides `RustExplorationManager`, a Python-facing interface
to the Rust exploration loop. It keeps states entirely in Rust
(`RustSimState`), processes symbolic branches with deferred forks,
checks find/avoid addresses in Rust, and calls back into Python only for
SimProcedures, syscalls and other Python-owned hooks. User-facing
behaviour (SimOption coverage, inspect dispatch, techniques, errors) is
documented in ``docs/advanced-topics/rust_engine.rst``.

Cross-mixin invariants
======================

The manager class composes several mixins (RustStateCacheMixin,
RustStateExportMixin, RustStateSyncMixin, RustCallbackDispatchMixin)
plus helpers in this module. The invariants below cut across those
boundaries — breaking any one of them tends to produce silent correctness
bugs (lost state, infinite loops) rather than loud failures, so they are
documented here and pinned by the named regression tests. The numbers are stable
identifiers cited from other files (``native/angr/src/stash.rs``,
``native/angr-symex-native/src/syscalls/mmap.rs``), so a retired invariant leaves a gap
rather than renumbering the rest.

I5. Register filter at the FFI boundary
    archinfo defines registers (cr0..8, fs_seg, ds_seg, cmstart,
    cmlen, fpreg, ...) the Rust engine does not model; it models a subset
    (rax-r15+rip on amd64). Any register dict passed to Rust via
    `set_registers_bulk` MUST be filtered to `_supported_register_names`
    in rust_state_sync.py, otherwise PyValueError 'unknown register: cr0'.
    Do not "fix" by adding cr0 etc. to amd64.rs unless the interpreter
    actually consumes them — it doesn't, and `_sync_registers_to_rust`
    skips them too.

I6. State-cache pinning and the metadata-clear contract
    Owned by `_cleanup_state_cache`'s docstring (what is pinned, the LRU
    eviction order, and why eviction never calls `clear_state_metadata`).
    Without the pins, a freshly-mutated state can race-evict between
    callbacks on the same state. Regression tests:
    `test_cleanup_state_cache_evicts_oldest_first`,
    `test_cleanup_state_cache_drops_dead_states` and
    `test_cleanup_state_cache_skips_pinned` in
    ``tests/engines/rust/test_state_sync.py``.

I7. Rust ↔ Python field sync uses max(), not overwrite
    Fields that both sides can mutate (`mmap_base`, `posix_brk`, ...) sync
    on stash export by taking `max(rust_value, python_value)`. A Python-
    side advance (user-set `state.heap.mmap_base` before re-entering
    exploration, or a fallback SimProcedure mutation) must not be reverted
    to a smaller Rust value. Tests:
    `TestMmapBaseSync.test_export_path_does_not_clobber_higher_python_mmap_base`
    and `TestPosixBrkSync.test_export_path_syncs_rust_posix_brk_into_state_posix`
    in ``tests/engines/rust/test_state_sync.py``.

I8. Exploration-loop termination conditions
    `_explore_with_predicates` and `_explore_with_addresses` must terminate
    on EITHER (a) Python predicate match, OR (b) Rust-native find_addr hit
    (`event_type == 'found'`), OR (c) all paths exhausted
    (`active_empty` / `has_active_states == false`). The find check uses
    `_found_count()`, which covers both Rust-native and Python-predicate
    finds: checking only Python predicate flags infinite-loops when
    `find=int` is combined with a non-predicate technique like DFS (the
    technique makes `_active_techniques` non-empty, routing through the
    predicate path).

I9. `max_active_states` is enforced via a helper, not raw `push_back`
    A Rust-side invariant, stated on `push_to_active_or_drop`'s rustdoc in
    ``native/angr/src/exploration/helpers.rs``: every site that pushes
    into the active stash goes through that helper, because a raw
    `push_back` bypasses the cap.

I10. RustExplorationManager exposes `stats` as a @property
     `mgr.stats()` raises TypeError — it is not callable. The inner
     `mgr._rust_mgr` (PyO3 object) does expose `stats()` as a method, and
     `get_fallback_stats()` exists ONLY on `_rust_mgr` (the Python wrapper
     does not re-export it). Tests that need full fallback details must
     reach in via `mgr._rust_mgr.get_fallback_stats()`.
"""

from __future__ import annotations

import contextlib
import functools
import gc
import logging
import os
import pickle
import re
import struct
import sys
import time
import types
import warnings
import weakref
from collections import defaultdict
from collections.abc import Callable
from itertools import chain
from typing import TYPE_CHECKING

import cle
import mulpyplexer
import pyvex
from capstone import x86 as cs_x86
from pyvex.errors import PyVEXError

from angr import claripy
from angr.claripy.errors import ClaripyError
from angr.engines.vex.lifter import VEX_IRSB_MAX_SIZE
from angr.errors import (
    AngrError,
    AngrSyscallError,
    SimEngineError,
    SimError,
    SimIRSBError,
    SimIRSBNoDecodeError,
    SimMemoryAddressError,
    SimMemoryError,
    SimMergeError,
    SimOperationError,
    SimSegfaultException,
    SimSolverError,
    SimStateError,
    SimUnsupportedError,
    SimZeroDivisionException,
    UnsupportedDirtyError,
)
from angr.exploration._constants import PAGE_SIZE
from angr.exploration._fallback import suppress_recoverable
from angr.exploration.rust_callback_dispatch import (
    RustCallbackDispatchMixin,
    _pending_syscall_resume,
    _proc_no_return,
)
from angr.exploration.rust_perf_tracker import PerformanceTracker
from angr.exploration.rust_state_binding import binding_of
from angr.exploration.rust_state_cache import RustStateCacheMixin
from angr.exploration.rust_state_export import RustStateExportMixin
from angr.exploration.rust_state_sync import RustStateSyncMixin, _ultrapage_bytes
from angr.exploration.rust_techniques import _FIND_AVOID_TECH_NAMES, _VT_HOLD_STASH

# The globals key of the time() watermark (``procedures/linux_kernel/time.py``'s
# ``time.KEY``); Rust holds it as ``RustSimState::last_time`` (angr-vmi2l.130).
_SYS_LAST_TIME = "sys_last_time"

# Envelope head written by RustExplorationManager.dump_snapshot. A file without
# it is a pre-bucket-D snapshot (bare Rust bytes) and still loads.
_SNAPSHOT_MAGIC = b"ANGRSNAP\x01"

# Reserved key in the snapshot envelope's pickled overlay dict, whose real keys
# are Rust state ids (u64, so never negative). Holds the seed state's
# posix.stdin.content ASTs — see `_capture_seed_stdin_content`.
_SEED_STDIN_KEY = -1

# Holding stashes for one `step()` round (angr-vmi2l.246): the round's
# not-yet-stepped states wait in the queue, and each stepped state's successors
# wait in the done stash, so the policy's `run(1)` only ever sees the one state
# the round is stepping.
_STEP_ROUND_QUEUE = "_step_round_queue"
_STEP_ROUND_DONE = "_step_round_done"

# `HistoryEntry`'s jumpkind byte for each `Ijk_*` it names (native
# `state/types.rs`); a syscall is 3 and anything else 4.
_HISTORY_JUMPKIND_CODES = {"Ijk_Boring": 0, "Ijk_Call": 1, "Ijk_Ret": 2}

if TYPE_CHECKING:
    import angr
    from angr.exploration.rust_state_proxy import RustInspectProxy
    from angr.state_plugins import SimStateEdgeHitmap

l = logging.getLogger(name=__name__)
_DBG = l.isEnabledFor(logging.DEBUG)  # Module-level guard for hot-path debug calls

# SimOptions mirrored onto the Rust state for native SimProcedures to read via
# ``RustSimState.has_option`` (angr-kzjv6). Add one only when a native proc
# consults it. SYMBOLIC_INITIAL_VALUES is seeded on in Rust (every stock bundle
# has it), so the mirror mostly exists to clear it after remove_options=
# (angr-c7xno.61). The CGC options gate the DECREE handlers in syscalls/cgc.rs
# (angr-vmi2l.139).
_NATIVE_SIMOPTIONS = frozenset(
    {
        "CGC_ENFORCE_FD",
        "CGC_NON_BLOCKING_FDS",
        "CGC_NO_SYMBOLIC_RECEIVE_LENGTH",
        "SHORT_READS",
        "SYMBOLIC_INITIAL_VALUES",
    }
)


def _options_on(state) -> set:
    """The names of the boolean SimOptions switched on in ``state.options``."""
    return {name for name, value in state.options._options.items() if value is True}


def _pristine_seed_copy(state):
    """A copy of seed ``state`` with plugins of its own, for ``_initial_seed_states``.

    A seed exported by another manager is a live view of that manager's Rust
    state, so a plain copy kept its proxies, and through them the source manager
    and its core, alive for as long as this manager lives. A multi-stage loop
    (``simgr = factory.simgr(simgr.found[0])``) then kept every earlier stage's
    manager alive and ran out of memory (angr-f4ldp). The copy gets a Python
    solver holding the Rust path constraints (``rust_solver=False``), since a
    ``RustSolverFallback`` would hold the source core too.
    """
    copy = state.copy()
    binding = binding_of(copy)
    if binding is not None:
        exporter = getattr(binding.owner, "manager", None)
        if exporter is None:
            raise SimStateError(f"cannot add {state!r}: it is a view of a Rust state whose manager is gone")
        exporter._detach_export_proxies(copy, binding.state_id, rust_solver=False)
    return copy


def _is_rust_memory_proxy(plugin) -> bool:
    """True when ``plugin`` is a ``RustMemoryProxy`` (the callback state's memory).

    Used by synchronous Rust→Python callbacks (``_cb_memory_load`` /
    ``_cb_memory_store`` / ``_cb_fetch_page``) to detect re-entry into the
    manager. Calling proxy methods would invoke FFI on the same
    ``_RustExplorationManager`` PyCell that ``run()`` currently
    ``&mut self`` borrows, raising "Already mutably borrowed" (angr-hcok).
    Cheap class-name check avoids importing ``rust_state_proxy`` at module
    top-level (would create a circular import).
    """
    return plugin is not None and type(plugin).__name__ == "RustMemoryProxy"


# Try to import the Rust exploration manager
try:
    from angr.rustylib.vex_engine import (
        ExplorationEvent as _ExplorationEvent,
    )
    from angr.rustylib.vex_engine import (
        ExplorationStateSnapshot as _ExplorationStateSnapshot,
    )
    from angr.rustylib.vex_engine import (
        PythonCallbacks,
    )
    from angr.rustylib.vex_engine import (
        RustExplorationManager as _RustExplorationManager,
    )
    from angr.rustylib.vex_engine import (
        RustSimState as _RustSimState,
    )

    RUST_EXPLORATION_AVAILABLE = True
except ImportError:
    # cat-(b) FALLBACK WITH LOSS: Rust extension not built; the manager
    # is import-safe but constructing one will raise from __init__.
    RUST_EXPLORATION_AVAILABLE = False
    _RustExplorationManager = None
    _ExplorationEvent = None
    _ExplorationStateSnapshot = None
    PythonCallbacks = None
    _RustSimState = None

# SimOption ledgers. The per-option rationale is the "SimOption coverage
# matrix" in docs/advanced-topics/rust_engine.rst; keep the two in step. An
# option the engine honors natively, honors transparently through the Python
# fallback, or that nothing consults appears in none of these sets.

# Warn once per option per manager: the engine ignores them, but they may be
# inherited from a parent state rather than chosen. TRACK_CONSTRAINT_ACTIONS and
# TRACK_MEMORY_MAPPING are deliberately absent: they ship in the default
# `symbolic` bundle, so warning on them would fire on every entry_state(); the
# first warns where it is consumed (reading state.history.actions), the second
# is read by nothing (angr-op0dn.14.7).
_REJECTED_OPTION_NAMES = frozenset(
    {
        "CONSERVATIVE_READ_STRATEGY",  # the write variant raises instead
        "UNINITIALIZED_ACCESS_AWARENESS",
        "BEST_EFFORT_MEMORY_STORING",
        "TRUE_RET_EMULATION_GUARD",  # a guard tweak; DO_RET_EMULATION itself raises
        "SUPER_FASTPATH",
        "FAST_MEMORY",
        "FAST_REGISTERS",
        "UNDER_CONSTRAINED_SYMEXEC",
        "BYPASS_VERITESTING_EXCEPTIONS",  # only Veritesting reads it, on its own Python manager (angr-6rz8)
        # Native time syscalls always return a fresh symbol (angr-0y0v); RDTSC
        # runs Python's dirty helper, which does honor it (angr-vmi2l.69).
        "USE_SYSTEM_TIMES",
    }
)

# Raise at manager construction: silently ignoring these would change what the
# analysis explores or returns, and none ships in the default `symbolic` bundle.
_RAISE_OPTION_NAMES = frozenset(
    {
        # The engine records no SimActions (TRACK_ACTION_HISTORY is only a flag
        # the preconstrainer toggles, so it is honored vacuously; angr-fkvt).
        "TRACK_MEMORY_ACTIONS",
        "TRACK_REGISTER_ACTIONS",
        "TRACK_TMP_ACTIONS",
        "TRACK_JMP_ACTIONS",
        "TRACK_OP_ACTIONS",
        "CONCRETIZE",  # eager concretization of every new symbol (angr-gmrc)
        "CONSERVATIVE_WRITE_STRATEGY",  # the engine always concretizes within limits (angr-csmm)
        "DO_RET_EMULATION",  # no emulated ret successor (angr-cf9h)
        "CALLLESS",  # calls execute instead of being skipped (angr-cf9h)
        "SYMBOL_FILL_UNCONSTRAINED_REGISTERS",  # unset registers read as concrete 0 (angr-apre)
        # Op/TypeMismatch/InvalidIR errors move the state to errored without a
        # Python substitution (angr-6rz8); the BYPASS_UNSUPPORTED_* siblings are
        # honored through the Python fallback.
        "BYPASS_ERRORED_IROP",
        "BYPASS_ERRORED_IRCCALL",
        "BYPASS_ERRORED_IRSTMT",
        "PRODUCE_ZERODIV_SUCCESSORS",  # Z3 division is total, so no SigFPE successor exists (angr-op0dn.14.9)
    }
)

# Route a default-routed manager to Python, but are not refused by an explicit
# engine="rust" manager, which ignores them. UNICORN makes one Python step run
# every concrete block up to the next symbolic one, so step counts (and where a
# state errors) change; the Rust engine has no unicorn mode (angr-mx00p.132.10).
# FAST_MEMORY / FAST_REGISTERS give the state a FastMemory plugin, whose
# contents the state sync cannot read, so they would reach Rust lost
# (angr-mx00p.132.39).
_PYTHON_ROUTED_OPTION_NAMES = frozenset({"UNICORN", "FAST_MEMORY", "FAST_REGISTERS"})

# Options whose ABSENCE the engine cannot honor. EXTENDED_IROP_SUPPORT ships in
# every bundle and only widens Python's op table; unset, a user asks for the
# narrow table, which the interpreter has no mode for (angr-op0dn.14.7).
_REQUIRED_OPTION_NAMES = frozenset({"EXTENDED_IROP_SUPPORT"})


def rust_missing_required_options(options) -> list[str]:
    """Return the sorted ``_REQUIRED_OPTION_NAMES`` *absent* from *options*.

    ``options is None`` means "no state to judge" (a bare container, not a
    SimState) and yields no offenders; an explicitly empty option set does
    offend, since that state really would run with the narrow op table under
    Python.
    """
    if options is None:
        return []
    return sorted(name for name in _REQUIRED_OPTION_NAMES if name not in options)


def rust_unsupported_options(options) -> list[str]:
    """Return the option settings in *options* that the Rust engine refuses.

    Single source of truth for "the Rust engine refuses this state". The
    manager constructor (:meth:`RustExplorationManager._check_raise_options`)
    and the engine dispatcher in :meth:`angr.factory.AngrObjectFactory.simulation_manager`
    both consume this rather than re-deriving the option list.

    Covers both polarities: a ``_RAISE_OPTION_NAMES`` member that is set, and a
    ``_REQUIRED_OPTION_NAMES`` member that is not (reported as ``unset NAME``).

    Args:
        options: A ``SimState.options`` set (or any container of option
            names / SimOption objects), possibly ``None``.

    Returns:
        Sorted list of offending settings; empty when the state is
        Rust-eligible.
    """
    if options is None:
        return []
    offending = [name for name in _RAISE_OPTION_NAMES if name in options]
    offending += [f"unset {name}" for name in rust_missing_required_options(options)]
    return sorted(offending)


def state_requires_python_engine(state) -> bool:
    """True when *state* sets a SimOption the Rust engine cannot honor."""
    return bool(rust_unsupported_options(getattr(state, "options", None)))


def unsupported_rust_manager_kwargs(kwargs) -> list[str]:
    """Return sorted *kwargs* names that :class:`RustExplorationManager` cannot honor.

    ``RustExplorationManager.__init__`` takes no ``**kwargs``, so a
    SimulationManager-only constructor argument — ``hierarchy``,
    ``resilience``, ``techniques``, … — is a TypeError there (angr-vmi2l.243).
    Callers that forward user kwargs (notably the factory) use this to name
    them all in one error, or to route the manager to the Python engine.
    """
    import inspect

    accepted = {
        name
        for name, param in inspect.signature(RustExplorationManager.__init__).parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return sorted(name for name in kwargs if name not in accepted)


# --- Engine dispatch (angr-op0dn.14.5.2, angr-mx00p.62) ----------------------
#
# A project built with `engine="rust"` (or under ANGR_DEFAULT_ENGINE=rust) asks
# `factory.simulation_manager()` to "run this on Rust if Rust can run it, else
# Python". The routing decision is computed here so it is testable without a
# factory, and so the raise-option list stays a single source of truth
# (`rust_unsupported_options`).

# Environment variables an earlier selection API read. Each one set is a hard
# error at project construction rather than a silently ignored setting.
_REMOVED_ENGINE_ENV_VARS = {
    "ANGR_RUST_AUTO": "build the project with engine='rust' (or set ANGR_DEFAULT_ENGINE=rust for every project) "
    "to run its simulation managers on the Rust engine where the workload is eligible",
}


def rust_default_engine_env() -> bool:
    """True when ``ANGR_DEFAULT_ENGINE=rust`` asks new projects to default to the Rust engine.

    The process-wide default of ``Project(..., engine="rust")``: a project
    built while this is set routes ``simulation_manager()`` calls that name no
    engine through the eligibility predicate (:func:`rust_engine_eligible`).
    Read at *project construction* time (not at import), so a test can set the
    variable and build a project without reloading the module. Any other value
    (including ``python``) leaves the default alone.

    Raises:
        AngrError: a removed engine-selection variable (``_REMOVED_ENGINE_ENV_VARS``)
            is set. Ignoring it would run the workload on the Python engine
            while its owner believes it runs on Rust.
    """
    for name, replacement in _REMOVED_ENGINE_ENV_VARS.items():
        if os.environ.get(name, ""):
            raise AngrError(f"The {name} environment variable was removed: unset it and {replacement}.")
    return os.environ.get("ANGR_DEFAULT_ENGINE", "").strip().lower() == "rust"


def rust_state_option_refusals(options) -> list[str]:
    """The option settings in *options* that route a state to the Python engine.

    :func:`rust_unsupported_options` (what the Rust engine refuses), then each
    ``_PYTHON_ROUTED_OPTION_NAMES`` member that is set (what it would accept but
    the Python engine serves). Shared by :func:`rust_engine_eligible` and
    :meth:`angr.exploration.rust_backend.RustBackend.decline_reason`.
    """
    offending = rust_unsupported_options(options)
    if options is not None:
        offending += sorted(name for name in _PYTHON_ROUTED_OPTION_NAMES if name in options)
    return offending


def rust_supports_arch(arch_name) -> bool:
    """True when the Rust engine implements *arch_name*.

    Delegates to the native ``arch_supported`` (backed by ``arch_from_name``,
    ``native/angr-symex-core/src/arch/registry.rs``) so the Python answer cannot drift from
    the
    arches the interpreter actually has. Six today: X86 / AMD64 / ARM / ARM64 /
    MIPS32 / MIPS64. False (→ Python engine) for PPC32/PPC64/S390X and when the
    extension is not built.
    """
    if not RUST_EXPLORATION_AVAILABLE or not arch_name:
        return False
    from angr.rustylib.vex_engine import arch_supported

    return arch_supported(str(arch_name))


#: Why a ``selfmodifying_code`` project stays on Python (angr-mx00p.132.30): a
#: store invalidates cached IRSBs only inside loaded binary regions, and a store
#: into a later instruction of the running block does not relift it as
#: ``SimReliftException`` does, so code patched on the stack runs stale.
SELFMODIFYING_CODE_REASON = "project enables selfmodifying_code, which the rust engine does not relift"


def custom_engine_reason(project) -> str | None:
    """Why *project*'s SimEngine keeps it on Python, or None for the stock ``UberEngine``.

    The Rust engine replaces the whole SimEngine, so a subclass's ``process_successors``
    and mixin overrides would never run (angr-vmi2l.281). Shared by
    :func:`rust_engine_eligible` and
    :meth:`angr.exploration.rust_backend.RustBackend.decline_reason`.
    """
    from angr.engines import UberEngine

    engine = project.factory.default_engine_factory
    if engine is UberEngine:
        return None
    return f"project uses a custom SimEngine {engine.__name__}"


def rust_engine_eligible(project, states, kwargs) -> tuple[bool, str]:
    """Decide whether *project* / *states* / *kwargs* can run on the Rust engine.

    The eligibility predicate behind ``simulation_manager()`` on a ``Project(engine="rust")`` project.
    Every ineligible answer names a signal the Rust engine would otherwise turn
    into a raise, a warning, or a silent divergence — routing to Python instead
    makes the whole loud surface transparent.

    Args:
        project: The :class:`angr.Project` the manager would drive.
        states: The seeding states (list of :class:`SimState`).
        kwargs: The SimulationManager constructor kwargs the caller passed.

    Returns:
        ``(eligible, reason)``. *reason* is a short human-readable string the
        factory logs at debug level, so the route is always observable.
    """
    if not RUST_EXPLORATION_AVAILABLE:
        return False, "rust extension not built"

    arch_name = getattr(getattr(project, "arch", None), "name", None)
    if not rust_supports_arch(arch_name):
        return False, f"arch {arch_name} not implemented by the rust engine"

    engine_reason = custom_engine_reason(project)
    if engine_reason is not None:
        return False, engine_reason

    if getattr(project, "selfmodifying_code", False):
        return False, SELFMODIFYING_CODE_REASON

    unsupported_kwargs = unsupported_rust_manager_kwargs(kwargs)
    if unsupported_kwargs:
        return False, "simulation_manager kwargs not honored by the rust engine: " + ", ".join(unsupported_kwargs)

    for state in states or ():
        offending = rust_state_option_refusals(getattr(state, "options", None))
        if offending:
            return False, "state options not honored by the rust engine: " + ", ".join(offending)

        events = _unsupported_inspect_events(state)
        if events:
            return False, "state.inspect events not dispatched by the rust engine: " + ", ".join(events)

    # The Rust engine applies these manager-wide (angr-vmi2l.241).
    mixed = [
        name
        for name in ("LAZY_SOLVES", "ZERO_FILL_UNCONSTRAINED_MEMORY")
        if len({name in getattr(state, "options", ()) for state in states or ()}) > 1
    ]
    if mixed:
        return False, "state options set on only some seeds: " + ", ".join(mixed)

    return True, "rust engine supports this project's engine, state options, kwargs, and inspect breakpoints"


def _unsupported_inspect_events(state) -> list[str]:
    """Inspect events *state* already has breakpoints for that Rust cannot dispatch.

    Registering one of these on a Rust-owned state raises (see
    ``_format_unsupported_event_msg``, ``rust_state_proxy.py``), so a seeding
    state that carries one is a Python workload — the raise would otherwise be
    the regression the moment the default engine flips. The events in
    ``_RUST_INSPECT_BOUNCE_ONLY_EVENTS`` register, but would silently miss
    every natively served call, so they count too, as does an event with a BP
    whose ``when`` matches a phase Rust never fires (``_rust_unfired_phases``).
    """
    inspector = getattr(state, "inspect", None)
    breakpoints = getattr(inspector, "_breakpoints", None)
    if not breakpoints:
        return []
    from angr.exploration.rust_state_proxy import (
        _RUST_INSPECT_BOUNCE_ONLY_EVENTS,
        _RUST_INSPECT_SUPPORTED_EVENTS,
        _rust_unfired_phases,
    )

    return sorted(
        event
        for event, bps in breakpoints.items()
        if bps
        and (
            event not in _RUST_INSPECT_SUPPORTED_EVENTS
            or event in _RUST_INSPECT_BOUNCE_ONLY_EVENTS
            or any(_rust_unfired_phases(event, bp.when) for bp in bps)
        )
    )


def resolve_main_address(project) -> int | None:
    """Find main function address; for stripped binaries, parse _start's PUT(rdi)."""
    main_sym = project.loader.find_symbol("main")
    if main_sym:
        return main_sym.rebased_addr
    try:
        entry_block = project.factory.block(project.entry)
        vex = entry_block.vex
        rdi_offset = project.arch.registers.get("rdi", (None,))[0]
        if rdi_offset is None:
            rdi_offset = project.arch.registers.get("edi", (None,))[0]
        if rdi_offset is not None:
            for stmt in reversed(vex.statements):
                s = str(stmt)
                if f"PUT(offset={rdi_offset})" in s or "PUT(rdi)" in s:
                    import re

                    m_const = re.search(r"0x([0-9a-fA-F]+)", s)
                    if m_const:
                        candidate = int(m_const.group(1), 16)
                        main_obj = project.loader.main_object
                        if main_obj.min_addr <= candidate <= main_obj.max_addr:
                            l.info(f"Extracted main=0x{candidate:x} from _start's rdi")
                            return candidate
                    break
        # Fallback: a "thin" entry that is a direct `call main` (no
        # __libc_start_main trampoline, e.g. DECREE/CGC binaries) leaves no
        # rdi/edi PUT to parse. The entry block's call target IS main when it
        # lands on real code in the main object (a SimProcedure target means
        # this is the __libc_start_main case, which the rdi parse handles).
        # Without this, _step_python_to_main's main_addr=None heuristic waits
        # `step > 10` and grabs an arbitrary mid-init address (e.g. inside
        # __libc_csu_init), starting Rust exploration off the real CFG.
        if vex.jumpkind == "Ijk_Call":
            import pyvex

            if isinstance(vex.next, pyvex.expr.Const):
                tgt = vex.next.con.value
                main_obj = project.loader.main_object
                if (
                    main_obj.min_addr <= tgt <= main_obj.max_addr
                    and tgt != project.entry
                    and tgt not in project._sim_procedures
                ):
                    l.info(f"Resolved main=0x{tgt:x} from entry's direct call target")
                    return tgt
    except MemoryError:
        raise
    except Exception as e:
        # cat-(b) FALLBACK WITH LOSS: main address extraction from _start
        # disassembly failed; caller resolves None and uses the post-init
        # state at whatever PC step_python_to_main lands on. Debug-logs.
        l.debug(f"Could not extract main from _start: {e}")
    return None


def is_python_init_seed(project, addr) -> bool:
    """True when a state at *addr* starts the loader's init sequence, which runs on the Python engine.

    Only a real init seed qualifies: the entry of a binary-format main
    object, or the simos loader stub (``full_init_state``). A shellcode
    blob has no init, and a user hook or extern SimProcedure seed is
    ordinary code (angr-vmi2l.291).
    """
    if addr is None:
        return False
    if addr == project.entry:
        return not isinstance(project.loader.main_object, cle.backends.Blob)
    return addr == getattr(project.simos, "_loader_addr", None)


# Every manager constructed in this process that is still alive (see _collect_dead_managers).
_live_managers: weakref.WeakSet = weakref.WeakSet()

# A collection may take at most 1 / (1 + _GC_DUTY) of the time spent building
# managers (see _collect_dead_managers).
_GC_DUTY = 2.0
_last_collect_end = 0.0
_last_collect_cost = 0.0


def _collect_dead_managers() -> None:
    """Free finished managers before another one is built.

    A manager sits in several reference cycles (its PythonCallbacks hold bound methods, its proxies point back at
    it), so only the cyclic garbage collector frees it, and with it its Rust states and Z3 terms. claripy ASTs are
    native objects the collector does not count, so a collection can be far off, and a process that builds manager
    after manager would keep every finished one resident until then.

    A full collection costs tens of milliseconds in a large angr process, which dominated a ``Callable`` call (one
    manager per call; angr-mx00p.96). So collect when an earlier manager is alive, unless the last collection ended
    less than ``_GC_DUTY`` times its own duration ago: managers built in quick succession share a collection, and
    ones built further apart (a test suite, a long exploration) are each collected first, as before. A count-based
    batch kept too many large finished managers resident between tests.
    """
    global _last_collect_end, _last_collect_cost
    if not _live_managers:
        return
    now = time.perf_counter()
    if now - _last_collect_end < _GC_DUTY * _last_collect_cost:
        return
    gc.collect()
    _last_collect_end = time.perf_counter()
    _last_collect_cost = _last_collect_end - now


_z3_deterministic_applied = False


def _apply_deterministic_z3_globals() -> None:
    """Pin Z3 module-level random seeds for run-to-run model stability.

    Sets ``smt.random_seed`` and ``sat.random_seed`` to 0 via
    ``Z3_global_param_set``. Z3 reads these on every subsequent
    ``Solver::new`` call (existing solvers are unaffected). Idempotent:
    safe to call from every manager constructed with
    ``deterministic=True``.

    Caveat: Z3 4.13 still reserves variable / restart heuristic latitude
    that is not bound by these seeds (bd memory `iaol1-seed-pin-empirically-broken`).
    See the rust_engine.rst "Deterministic mode" section for user-facing docs.
    """
    global _z3_deterministic_applied
    if _z3_deterministic_applied:
        return
    from angr.rustylib.vex_engine import set_z3_global_param

    set_z3_global_param("smt.random_seed", "0")
    set_z3_global_param("sat.random_seed", "0")
    _z3_deterministic_applied = True
    l.debug("Pinned Z3 smt.random_seed=0 + sat.random_seed=0 (deterministic mode)")


def _warn_if_parallel_nondeterministic() -> bool:
    """Warn when ``deterministic=True`` meets a multi-worker scheduler.

    ``RUST_PARALLEL_WORKERS`` > 1 runs the work-stealing pool, whose steal
    order is nondeterministic *by design*: the set of found states still
    matches a serial run, but which state is found first — and hence the
    order results are reported in — does not. Witness choice stays canonical
    (that part is per-state), so this is a warning, not an error: the
    combination is legitimate when only the found *set* matters.

    Returns True when the warning fired (multi-worker + deterministic).
    Deliberately does NOT touch Z3's parallel mode (bd memory `avoid-z3-parallel-enable`).

    Enforcement-level asymmetry (intentional, not a bug): the *explicit*
    ``parallel_workers>1`` kwarg combined with ``deterministic=True`` is a
    hard ``ValueError`` in ``__init__`` (see the guard near
    ``self._requested_parallel_workers``), because the caller programmatically
    asked for two mutually exclusive things and deserves a clear, early
    failure. This function handles the *other* source — an ambient
    ``RUST_PARALLEL_WORKERS`` env var, which benches set process-wide — where a
    hard error would be hostile: the env is not the caller's explicit intent,
    the found *set* still matches a serial run, and only report order differs.
    So the ambient case warns and proceeds; the explicit case rejects.
    """
    try:
        workers = int(os.environ.get("RUST_PARALLEL_WORKERS", "1"))
    except ValueError:
        return False
    if workers <= 1:
        return False
    warnings.warn(
        f"deterministic=True with RUST_PARALLEL_WORKERS={workers}: witness choice is canonical, "
        "but the work-stealing scheduler's steal order is nondeterministic, so the order in which "
        "states are found is not reproducible. Set RUST_PARALLEL_WORKERS=1 for a fully reproducible run.",
        RuntimeWarning,
        stacklevel=3,
    )
    return True


def set_rust_log_level(level: str = "info") -> None:
    """Set the Rust-side log level.

    Args:
        level: One of "error", "warn", "info", "debug", "trace", "off".
    """
    from angr.rustylib.vex_engine import set_rust_log_level as _set_level

    _set_level(level)


def cfg_distance_map(cfg, target_addr: int) -> dict[int, int]:
    """Build an ``addr -> distance-to-target`` snapshot from an angr CFG.

    Computed **once** at setup for the 'directed' exploration strategy
    (angr-a32jl.4). Distance is the shortest block-count path *to* the target,
    measured by a reverse-BFS over the CFG transition graph so every node's
    value is the number of blocks between it and ``target_addr``. When several
    context-sensitive CFG nodes share a block address, the smallest distance
    wins. Blocks with no path to the target are simply absent from the map (the
    Rust ``DirectedCfgDistance`` policy treats absent blocks as unreachable).

    Args:
        cfg: an angr CFG with a ``.graph`` networkx ``DiGraph`` of nodes that
            expose ``.addr`` (``CFGFast`` / ``CFGEmulated`` both qualify).
        target_addr: the block address to steer toward.

    Returns:
        Mapping of block address to distance-in-blocks from that block to the
        target. Suitable to pass as ``distances=`` to
        :meth:`RustExplorationManager.set_exploration_strategy`.
    """
    import collections

    graph = cfg.graph
    targets = [n for n in graph if getattr(n, "addr", None) == target_addr]
    if not targets:
        raise ValueError(f"target address {target_addr:#x} not found in CFG")

    # Reverse-BFS from the target node(s): a hop against a real edge means one
    # block closer, so BFS depth on the reversed graph is distance-to-target.
    dist: dict[int, int] = {}
    seen: set = set()
    queue: collections.deque = collections.deque()
    for t in targets:
        seen.add(t)
        queue.append((t, 0))
    while queue:
        node, d = queue.popleft()
        addr = getattr(node, "addr", None)
        if addr is not None and d < dist.get(addr, 1 << 62):
            dist[addr] = d
        for pred in graph.predecessors(node):
            if pred not in seen:
                seen.add(pred)
                queue.append((pred, d + 1))
    return dist


_rust_log_env_applied = False


def _apply_rust_log_env() -> None:
    """Honor RUST_LOG (or legacy ANGR_RUST_LOG) on first manager construction.

    Accepts either a single level word (``debug`` / ``info`` / ``warn`` / ...)
    or a full RUST_LOG-style filter spec
    (``rustylib::exploration=debug,rustylib::interpreter=info``; targets are
    module paths under the crate's lib name ``rustylib``). Per-module filters
    are honored — the Rust side delegates filter parsing to env_logger.
    Applies once per process. With neither variable set, the ``warn`` filter
    the extension installs at import stays in effect.

    Precedence: ``RUST_LOG`` wins if set, else ``ANGR_RUST_LOG`` (kept for
    backcompat with the pre-env_logger logger). ``ANGR_RUST_LOG`` retains
    documentation value because the Rust extension does not auto-honor
    ``RUST_LOG`` — env-driven activation only fires when a
    ``RustExplorationManager`` is constructed, which is when this hook runs.
    """
    global _rust_log_env_applied
    if _rust_log_env_applied:
        return
    _rust_log_env_applied = True
    level = os.environ.get("RUST_LOG") or os.environ.get("ANGR_RUST_LOG")
    if not level:
        return
    try:
        set_rust_log_level(level)
    except MemoryError:
        raise
    except Exception as e:
        # cat-(b) FALLBACK WITH LOSS: env-driven level could not be applied;
        # manager construction proceeds, but Rust-side log output stays at
        # whatever level was set previously (``warn`` by default).
        l.debug("Failed to set Rust log level from env (%r): %s", level, e)


def _resolve_env_flag(kwarg: bool | None, env_var: str, default: bool = False) -> bool:
    """Resolve a boolean gate from an explicit kwarg or an env-var fallback.

    Returns ``bool(kwarg)`` when the kwarg is not ``None``. Otherwise, an unset
    (or empty) ``env_var`` yields ``default``, and a set one is truthy only for
    ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive) — so a default-on gate
    is forced back off with ``<VAR>=0``. Centralizes the truthy-token set so a
    new gate can't drift.
    """
    if kwarg is not None:
        return bool(kwarg)
    raw = os.environ.get(env_var, "")
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


# Per-arch GPR snapshot lists for RustErrorRecord.registers. Conservative: PC,
# SP, BP/FP, and standard GPRs. Vector/floating-point registers are excluded.
_ARCH_REG_SNAPSHOT: dict[str, tuple[str, ...]] = {
    "AMD64": (
        "rip",
        "rsp",
        "rbp",
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
    ),
    "X86": ("eip", "esp", "ebp", "eax", "ebx", "ecx", "edx", "esi", "edi"),
    "ARM": ("pc", "sp", "lr", "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12"),
    "ARMEL": ("pc", "sp", "lr", "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12"),
    "ARMHF": ("pc", "sp", "lr", "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12"),
    "AARCH64": ("pc", "sp", "lr", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x29", "x30"),
    "MIPS32": ("pc", "sp", "ra", "v0", "v1", "a0", "a1", "a2", "a3"),
    "MIPS64": ("pc", "sp", "ra", "v0", "v1", "a0", "a1", "a2", "a3"),
}


# The SimError subclass the Python engine raises for each Rust error class.
_SIM_ERROR_BY_CLASS = {
    "nodecode": SimIRSBNoDecodeError,
    "lift": SimIRSBError,
    "memory": SimMemoryError,
    "operation": SimOperationError,
    "unsupported": SimUnsupportedError,
}

# MemoryError's Display for the accesses Python reports as SimSegfaultException
# (native/angr-symex-core/src/memory/mod.rs, enum MemoryError).
_SEGFAULT_RE = re.compile(r"memory error: (?:unmapped memory|unmapped page|permission violation) at (0x[0-9a-f]+)")

# The required permission of a violation, named as Python's SimSegfaultException.reason names it
# (privileged_mixin.py for loads and stores, the VEX engine's NX check for a fetch).
_SEGFAULT_REASONS = (
    ("execute: true", "non-executable"),
    ("write: true", "non-writable"),
    ("read: true", "non-readable"),
)


def sim_error_for_rust_message(message: str, error_class: str) -> SimError:
    """The SimError the Python engine raises where the Rust engine errored with *message*.

    *error_class* is :meth:`RustErrorRecord._classify` of *message*. An
    unmapped or permission-violating access is a SimSegfaultException with
    Python's reason; a class with no Python counterpart is a plain SimError
    (angr-mx00p.136.8).
    """
    if message.startswith(RustErrorRecord._ZERODIV_TRAP):
        return SimZeroDivisionException(message)
    m = _SEGFAULT_RE.match(message)
    if m is not None:
        # Only the required half names the violated permission: the "have" half
        # of a write into an R+X page also says "execute: true" (angr-vmi2l.273).
        required = message.partition("required ")[2].partition(", have ")[0]
        reason = next((r for perm, r in _SEGFAULT_REASONS if perm in required), "unmapped")
        return SimSegfaultException(int(m.group(1), 16), reason)
    return _SIM_ERROR_BY_CLASS.get(error_class, SimError)(message)


class RustErrorRecord:
    """Container for an errored state, matching angr's ErrorRecord interface.

    Attributes:
        state:             The SimState at the point of error.
        error:             The SimError the Python engine raises for this
                           failure (:func:`sim_error_for_rust_message`).
        message:           The Rust engine's error message.
        addr:              The instruction address where the error occurred.
        error_class:       Stable taxonomy string for the error
                           (e.g. 'unsupported', 'memory', 'lift', 'callback').
                           Derived from the message prefix; matches the
                           CbExecutionError variants in
                           native/angr-symex-core/src/interpreter/execution_error.rs.
        constraint_count:  Number of solver constraints on the state at error
                           time. 0 if state is None or the count cannot be read.
        registers:         dict mapping register name -> int (concrete) or str
                           (symbolic AST repr). Empty dict if state is None or
                           arch isn't in the snapshot table.
        last_statements:   Last few basic-block addresses leading up to the
                           error. Per-VEX-statement granularity isn't tracked
                           today; this is the closest available history.
    """

    # Stable error-class taxonomy. Prefixes match the Display impls of
    # CbExecutionError variants in native/angr-symex-core/src/interpreter/execution_error.rs
    # (pinned by display_prefixes_match_the_python_error_classifier in its
    # execution_tests.rs) and the formatted error strings of the stepping loop.
    _ERROR_CLASS_PREFIXES = (
        ("ir decoding error", "nodecode"),
        ("memory error", "memory"),
        ("operation error", "operation"),
        ("invalid vex ir", "invalid_ir"),
        ("unsupported", "unsupported"),
        ("type mismatch", "type_mismatch"),
        ("unknown temporary", "unknown_temp"),
        ("callback error", "callback"),
        ("lift error", "lift"),
        ("need lift at", "need_lift"),
        ("need python fallback", "need_python_fallback"),
        ("resolve_function error", "resolve_function"),
        ("unexecutable vex trap exit", "trap"),
    )

    # The interpreter takes this trap exit only on a concretely zero divisor,
    # where Python's irop.py raises SimZeroDivisionException (angr-mx00p.132.10).
    _ZERODIV_TRAP = "unexecutable VEX trap exit Ijk_SigFPE_IntDiv"

    def __init__(self, state, message: str, addr: int = 0):
        self.state = state
        self.message = message
        self.addr = addr
        self.error_class = self._classify(message)
        self.error = sim_error_for_rust_message(message, self.error_class)
        self.constraint_count = self._count_constraints(state)
        self.registers = self._snapshot_registers(state)
        self.last_statements = self._tail_history(state)

    @classmethod
    def _classify(cls, message: str) -> str:
        msg = message.lower()
        for prefix, klass in cls._ERROR_CLASS_PREFIXES:
            if msg.startswith(prefix):
                return klass
        # Substring fallbacks for nested/wrapped error messages.
        if "timeout" in msg:
            return "timeout"
        if "unmapped" in msg:
            return "unmapped"
        if "panic" in msg:
            return "rust_panic"
        return "unknown"

    @staticmethod
    def _count_constraints(state) -> int:
        if state is None:
            return 0
        try:
            return len(state.solver.constraints)
        except MemoryError:
            raise
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: best-effort error report; constraint
            # count of 0 on read failure is fine — RustErrorRecord is diagnostic.
            return 0

    @staticmethod
    def _snapshot_registers(state) -> dict:
        if state is None:
            return {}
        try:
            arch_name = state.arch.name
        except MemoryError:
            raise
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: best-effort error report; if arch
            # is unreadable, return empty registers dict.
            return {}
        names = _ARCH_REG_SNAPSHOT.get(arch_name, ())
        snap: dict = {}
        for name in names:
            try:
                val = getattr(state.regs, name)
            except MemoryError:
                raise
            except Exception:
                # cat-(a) EXPECTED CONTROL FLOW: best-effort error report; skip a
                # register that fails to read.
                continue
            try:
                if val.concrete:
                    snap[name] = val.concrete_value
                else:
                    snap[name] = str(val)
            except MemoryError:
                raise
            except Exception:
                # cat-(a) EXPECTED CONTROL FLOW: best-effort error report; .concrete
                # probe failed — fall through to str() repr.
                snap[name] = str(val)
        return snap

    @staticmethod
    def _tail_history(state, n: int = 5) -> list:
        if state is None:
            return []
        try:
            bbls = list(state.history.recent_bbl_addrs)
        except MemoryError:
            raise
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: best-effort error report; missing
            # history attribute is fine — return empty list.
            return []
        return bbls[-n:]

    def reraise(self):
        raise self.error

    def __repr__(self):
        return f'<State errored at {hex(self.addr)} class={self.error_class} with "{self.message}">'


# Default active-stash safety cap (angr-o4q3). A divergent / path-exploding
# exploration grows the active stash without bound until it exhausts RAM and is
# OOM-killed (observed: ralph run 20260613 iters 28-30, a CADET_00001 explore
# leaking ~1.6 states/step). This is a *count* backstop that converts unbounded
# growth into bounded growth — it is NOT a precise memory limit (per-state size
# varies), so it is set far above any realistic workload (CTF explores peak in
# the hundreds-low thousands of active states). When the cap is hit, excess
# forks are pruned to the ``pruned`` stash and a one-time warning is logged.
# Pass ``max_active_states=None`` to disable, or a smaller int to bound tighter.
# Memory-precise bounding is tracked separately (angr-wcxi).
DEFAULT_MAX_ACTIVE_STATES = 100_000


# Single source of truth for the boilerplate stash accessors on
# RustExplorationManager. Each entry is
# ``stash -> (doc paragraph, emit `mgr.<stash>` property, emit
# `mgr.<stash>_proxies()` method)``; ``_install_stash_accessors`` turns it into
# the actual attributes at class-creation time. A ``False`` means the class
# already defines that accessor by hand (``found``'s property also merges
# predicate hits) or deliberately has none (``pruned`` has no proxy accessor —
# ``docs/advanced-topics/rust_engine.rst`` documents the proxy set). Adding a
# stash here wires up both halves at once, which is the point: the accessors
# used to be hand-written one-liners, so a new stash could reach one list and
# be forgotten in the other.
_STASH_ACCESSORS = {
    "active": ("These are the states that the next :meth:`step` call will advance, each once.", True, True),
    "found": ("", False, True),
    "avoid": ("These are the states that matched the ``avoid`` predicate.", True, True),
    "deadended": ("These are the states that produced no successors.", True, True),
    "unconstrained": (
        (
            "These are states where a symbolic jump target (e.g., ret instruction "
            "with symbolic return address) had too many possible concrete values "
            "to enumerate and fork."
        ),
        True,
        True,
    ),
    "pruned": (
        (
            "These are states that were determined to be unsatisfiable during "
            "exploration (e.g., both branches of a conditional were infeasible "
            "given the current constraints)."
        ),
        True,
        False,
    ),
}


def _install_stash_accessors(cls):
    """Class decorator generating the per-stash accessors in ``_STASH_ACCESSORS``.

    Both halves route through the hand-written ``_get_stash_states`` /
    ``_stash_proxies`` chokepoints; only the wrapping property/method is
    generated. Accessors the class defines itself are never overwritten, so a
    hand-written body (``found``, ``errored``) always wins over the table.
    """
    for stash, (doc, emit_property, emit_proxies) in _STASH_ACCESSORS.items():
        if emit_property and stash not in cls.__dict__:

            def _states(self, _stash=stash):
                return self._get_stash_states(_stash)

            _states.__doc__ = (
                f"Get states in the ``{stash}`` stash as angr SimStates.\n\n{doc}\n\n"
                f"For SimulationManager API compatibility this returns full "
                f"angr states."
            )
            setattr(cls, stash, property(_states))

        name = f"{stash}_proxies"
        if emit_proxies and name not in cls.__dict__:

            def _proxies(self, _stash=stash):
                return self._stash_proxies(_stash)

            _proxies.__name__ = name
            _proxies.__qualname__ = f"{cls.__qualname__}.{name}"
            _proxies.__doc__ = (
                f"Return the ``{stash}`` stash as ``list[RustStateProxy]``.\n\n"
                f"Counterpart to :attr:`{stash}`, but returns lightweight "
                f"proxies that delegate reads to Rust via PyO3 instead of "
                f"materializing full angr ``SimState`` objects. Use this when "
                f"querying many states cheaply (e.g. counting states matching "
                f"``proxy.addr == X``); use :attr:`{stash}` when you need full "
                f"SimState plugins (``posix.dumps``, ``solver.eval`` of complex "
                f"claripy ASTs, etc.)."
            )
            setattr(cls, name, _proxies)
    return cls


def _native_lift_settings(project) -> dict | None:
    """The lifter settings for ``RustExplorationManager.set_native_lift``, or
    None when native lifting could not match the ``lift_block`` callback.

    The callback lifts through ``project.factory.block(addr)``, so a native
    lift only stands in for it when that would be a plain libVEX lift with the
    engine's settings: libVEX is the arch's first pyvex lifter and no
    postprocessor rewrites its output, the engine keeps the stock
    ``VEXLifter.lift_vex``. ``kb.patches`` keep it on: a stepped state lifts
    its own memory, which native lift reads (angr-vmi2l.75).
    """
    from pyvex.lifting import LibVEXLifter
    from pyvex.lifting.lift_function import lifters, postprocessors

    from angr.engines.vex.lifter import VEXLifter

    arch_name = project.arch.name
    arch_lifters = lifters.get(arch_name) or []
    if not arch_lifters or arch_lifters[0] is not LibVEXLifter or postprocessors.get(arch_name):
        return None
    engine = project.factory.default_engine
    if not isinstance(engine, VEXLifter) or type(engine).lift_vex is not VEXLifter.lift_vex:
        return None
    return {
        "default_opt_level": engine._default_opt_level,
        "strict_block_end": bool(engine.default_strict_block_end),
        # A stateless `factory.block` lift always has cross_insn_opt on.
        "cross_insn_opt": True,
        "selfmodifying_code": bool(engine.selfmodifying_code),
    }


def _syscall_tables(simos) -> list[tuple[str, dict[int, str]]]:
    """``[(abi, {number: name})]`` for each syscall ABI ``simos`` dispatches.

    The input to ``set_syscall_tables``: angr's
    ``syscall_library.syscall_number_mapping`` for every ABI in
    ``simos.syscall_abis``, or ``[]`` when the SimOS has no syscall library.
    """
    library = getattr(simos, "syscall_library", None)
    if library is None:
        return []
    return [(abi, dict(library.syscall_number_mapping[abi])) for abi in getattr(simos, "syscall_abis", {})]


# Every ``Ijk_Sys*`` jumpkind the Rust engine lifts (``JumpKind::is_syscall``).
_SYSCALL_JUMPKINDS = (
    "Ijk_Sys_syscall",
    "Ijk_Sys_int128",
    "Ijk_Sys_int129",
    "Ijk_Sys_int130",
    "Ijk_Sys_int145",
    "Ijk_Sys_int210",
    "Ijk_Sys_sysenter",
    "Ijk_Sys_int",
    "Ijk_Sys_int32",
    "Ijk_Sys",
)


def _unresolvable_syscall_jumpkinds(simos, arch) -> list[str]:
    """The syscall jumpkinds on which ``simos.syscall_abi`` raises ``AngrSyscallError`` for ``arch``.

    The second input to ``set_syscall_tables``. Python's
    ``SimSuccessors._categorize_successor`` resolves a syscall in the step that
    reaches it and files the successor as unsat when that raises (SimLinux on
    AMD64 knows only ``syscall`` and ``int 0x80``). ``syscall_abi`` reads only
    the arch and the jumpkind there, so a stand-in state probes it once per
    kind; a probe that fails any other way leaves the kind to the per-syscall
    Python path, which still errors the state.
    """
    syscall_abi = getattr(simos, "syscall_abi", None)
    if syscall_abi is None:
        return []
    out = []
    for jk in _SYSCALL_JUMPKINDS:
        probe = types.SimpleNamespace(arch=arch, history=types.SimpleNamespace(jumpkind=jk, parent=None))
        try:
            syscall_abi(probe)
        except AngrSyscallError:
            out.append(jk)
        except MemoryError:
            raise
        except Exception:  # pylint:disable=broad-exception-caught
            # cat-(a) EXPECTED CONTROL FLOW: a jumpkind the probe cannot
            # classify stays on the Python syscall path.
            l.debug("syscall_abi probe for %s failed; leaving it to the Python syscall path", jk, exc_info=True)
    return out


def _concretization_key(state) -> tuple[bool, int, int, bool, bool, bool]:
    """The arguments ``configure_concretization_strategies`` takes for *state*.

    The Rust engine applies them manager-wide, so every state one manager
    runs must share this key (angr-vmi2l.241, angr-vmi2l.346).
    """
    from angr import sim_options as o

    read_limit = 1024  # Python default
    write_limit = 128  # Python default
    mem = getattr(state, "memory", None)
    for strat in getattr(mem, "read_strategies", None) or ():
        if hasattr(strat, "_limit"):
            read_limit = strat._limit
            break
    for strat in getattr(mem, "write_strategies", None) or ():
        if hasattr(strat, "_limit"):
            write_limit = strat._limit
            break
    return (
        o.APPROXIMATE_MEMORY_INDICES in state.options,
        read_limit,
        write_limit,
        o.SYMBOLIC_WRITE_ADDRESSES in state.options,
        o.AVOID_MULTIVALUED_READS in state.options,
        o.AVOID_MULTIVALUED_WRITES in state.options,
    )


def _plt_got_slot(obj, insn, op, bits: int) -> int | None:
    """The GOT slot an x86 PLT entry's ``jmp *mem`` reads, or None for an operand shape no PLT uses.

    The slot depends on the operand's base register: RIP/EIP-relative (x86-64,
    ``jmp *disp(%rip)``), absolute (non-PIC i386, ``jmp *abs32``), or relative to
    ``%ebx``, which a PIC i386 PLT holds at ``_GLOBAL_OFFSET_TABLE_``, the start of
    ``.got.plt`` (angr-vmi2l.245).
    """
    mem = op.mem
    if mem.index != cs_x86.X86_REG_INVALID:
        return None
    if mem.base in (cs_x86.X86_REG_RIP, cs_x86.X86_REG_EIP):
        slot = insn.address + insn.size + mem.disp
    elif mem.base == cs_x86.X86_REG_INVALID:
        slot = mem.disp
    elif mem.base == cs_x86.X86_REG_EBX:
        got_plt = next((sec for sec in obj.sections if sec.name == ".got.plt"), None)
        if got_plt is None:
            return None
        slot = got_plt.min_addr + mem.disp
    else:
        return None
    return slot & ((1 << bits) - 1)


@_install_stash_accessors
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

    # SimulationManager.DROP: a to_stash that discards states (split(to_stash=DROP)).
    DROP = "_DROP"

    # Class-level cache for loader-pages output keyed (weakly) by the
    # cle.Loader instance. The output is a pure function of loader state,
    # so reusing it across Callable-spawned RustExplorationManagers on the
    # same project skips the per-init `loader.memory.load` + `map_memory_batch`
    # cost (significant for mma_howtouse, see angr-i9f2). Value is a dict with
    # 'batch_pages' (list of (page_addr, bytes, perms)) and 'lazy_regions'
    # (list of (start, len)). WeakKeyDictionary auto-evicts entries when
    # the Project/Loader is garbage-collected, so we avoid stale hits if
    # Python recycles ids across short-lived projects.
    _loader_pages_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def __init__(
        self,
        project: angr.Project,
        active_states: list | None = None,
        save_unconstrained: bool = True,
        solver_timeout_ms: int = 30000,
        max_active_states: int | None = DEFAULT_MAX_ACTIVE_STATES,
        max_history: int = 1000,
        exploration_strategy: str = "bfs",
        use_shared_lineage_solver: bool = False,
        deterministic: bool = False,
        prefer_native_library_hooks: bool | None = None,
        use_native_lift: bool = True,
        parallel_workers: int = 1,
        symlinks: dict | None = None,
        resilience=None,
    ):
        """Initialize the Rust exploration manager.

        Args:
            project: angr Project for the binary.
            active_states: Optional list of initial angr SimStates.
            save_unconstrained: If True (the default, as in angr's
                ``SimulationManager``), save states with unconstrained IP to
                the 'unconstrained' stash instead of silently dropping them
                (angr-x39ae).
            solver_timeout_ms: Z3 solver timeout in milliseconds (default: 30000).
            max_active_states: Maximum number of states in the active stash.
                When reached, new forked states are pruned to the ``pruned``
                stash and a one-time warning is logged. Defaults to
                :data:`DEFAULT_MAX_ACTIVE_STATES` (a runaway-explosion backstop
                so a divergent explore fails bounded instead of OOM-killing the
                process). Pass ``None`` to disable the cap entirely, or a
                smaller int to bound tighter. This is a count cap, not a memory
                limit.
            max_history: Maximum length of each state's history /
                detailed_history ring buffer (default: 1000). 0 means
                unlimited — only safe for short runs since long explorations
                can OOM. Applied to every state created or added via this
                manager.
            exploration_strategy: ``"bfs"`` (default, FIFO state selection)
                or ``"dfs"`` (LIFO). Equivalent to constructing with the
                default and then calling
                :meth:`set_exploration_strategy`, but settable at
                construction time so techniques and other one-shot setup
                code observe the chosen order from the first step. Raises
                ``ValueError`` for any other value.
            use_shared_lineage_solver: Opt this manager's seed states in
                to fork-time ``SharedLineageSolver`` materialization
                (angr-3ms1 step 1b). Default off: the slice-1c fork-time
                gate stays inert so plain BFS runs keep their current
                solver shape and CI baselines hold. When on, every seed
                state's solver context gets the flag set and forks
                inherit it (so descendants opt in transparently). Inert
                in this slice — slice-1c will be the first consumer of
                the flag at fork time.
            prefer_native_library_hooks: If True (default), let native
                procedures serve ``use_sim_procedures`` hooks whose address
                lands inside a *non-main* loaded object (libc &c. under
                ``auto_load_libs``). Main-object hooks always stay on Python so
                ``proj.hook()`` overrides win. When ``None`` (the default), env
                var ``ANGR_RUST_PREFER_NATIVE_LIBRARY_HOOKS=0`` forces the old
                behavior of bouncing every in-object hook to Python.
            use_native_lift: If True (default), lift cold blocks in-process
                through the native libVEX seam instead of the pyvex Python
                callback. It stays off when the callback would not be a
                plain libVEX lift (a custom pyvex lifter or ``lift_vex``).
                Blocks whose bytes are not fully concrete in the Rust memory
                sidecar, Thumb blocks, and blocks libVEX
                cannot fully decode fall back to the callback with an
                identical IRSB, so setting this False only costs speed,
                never fidelity. See docs/advanced-topics/rust_libvex_ffi.rst.
            deterministic: If True, pin ``smt.random_seed`` and
                ``sat.random_seed`` to 0 via ``Z3_global_param_set``
                before any new solver is constructed (angr-iaol.2).
                The pin is process-wide and applies to every solver
                built afterwards (including in other managers in the
                same process). Z3 4.13 still reserves variable /
                restart heuristic latitude that is not bounded by
                these seeds, so this flag *narrows* but does not
                *close* run-to-run model variation. Default False
                preserves the current non-deterministic behavior.
            parallel_workers: Number of real work-stealing scheduler
                workers to engage (angr-op0dn.13.5). Default 1
                (single-threaded, zero-regression). Values >1 request
                the parallel coordinator; it engages only on an
                *eligible* explore — address-based ``find``/``avoid``
                (no callable predicates), where worker-local frontiers
                are safe. Ineligible explores (callable ``find``/``avoid``)
                fall back to single-threaded with a logged reason so the
                found-set stays correct. Engages the wave coordinator (the
                worker-invariant parallel path); the steady-state loop stays an
                explicit ``RUST_PARALLEL_STEADY`` opt-in.
                The ``RUST_PARALLEL_WORKERS`` env var, when set, overrides
                this kwarg (benches set it explicitly). Passing >1 together
                with ``deterministic=True`` raises ``ValueError`` — the
                work-stealing steal order is nondeterministic by design.
            resilience: As ``SimulationManager``'s (``None``, ``True``,
                ``False`` or a tuple of exception types). A state the Rust
                engine errors still lands in ``errored``; when its error
                (:func:`sim_error_for_rust_message`, a ``SimError``) is not
                one of these types, the ``step()`` / ``explore()`` that
                errored it raises that error, as the Python engine does.
        """
        # angr-op0dn.13.5 (M5-B16): the work-stealing steal order is
        # nondeterministic by design, so a reproducible (deterministic) run and
        # multiple workers are mutually exclusive. Reject early, before any Rust
        # construction, so the caller gets a clear error rather than a silently
        # non-reproducible run. M2 owns the same guard for its own paths; this
        # only rejects the constructor combination. The ambient env-var source
        # (RUST_PARALLEL_WORKERS) is handled with a *warning* instead of an
        # error by _warn_if_parallel_nondeterministic() — see its docstring for
        # why the explicit-kwarg and ambient-env enforcement levels differ.
        if deterministic and int(parallel_workers) > 1:
            raise ValueError(
                "deterministic=True is incompatible with parallel_workers>1: the "
                "work-stealing scheduler's steal order is nondeterministic by design, "
                "so a reproducible run must be single-threaded. Set parallel_workers=1 "
                "(or RUST_PARALLEL_WORKERS=1) for a deterministic run."
            )
        # Requested worker count for the programmatic (non-env) path; the env var
        # RUST_PARALLEL_WORKERS (read once in Rust new()) wins when set, so record
        # whether it is present to skip the programmatic setter in that case.
        self._requested_parallel_workers = max(1, int(parallel_workers))
        self._parallel_workers_env_set = "RUST_PARALLEL_WORKERS" in os.environ
        # Parsed env worker count, recorded so an ineligible explore (callable
        # predicates or `until`) can downgrade to serial for that explore and a
        # later eligible explore can restore the env-configured count
        # (angr-ph300.77). Malformed values degrade to 1.
        self._parallel_workers_env_value = 1
        if self._parallel_workers_env_set:
            try:
                self._parallel_workers_env_value = max(1, int(os.environ["RUST_PARALLEL_WORKERS"]))
            except (ValueError, TypeError):
                self._parallel_workers_env_value = 1
        # Init pipeline sequenced into named phases (angr-wqao.4).
        # boot:       construct the Rust manager + apply basic config.
        # config:     resolve gate flags, init counters/caches, register
        #             callbacks + simprocedures.
        # activate:   add initial states to the Rust 'active' stash. A state
        #             found by another manager is copied in, never adopted
        #             with that manager's core (angr-vmi2l.2):
        #             ``_add_rust_state`` moves its constraints by Z3 pointer.
        self._phase_boot(
            project,
            save_unconstrained,
            solver_timeout_ms,
            max_active_states,
            max_history,
            exploration_strategy,
            deterministic,
            use_native_lift,
        )
        self._phase_config(
            use_shared_lineage_solver,
            prefer_native_library_hooks,
        )
        # Pre-existing symlinks to seed into the Rust FileSystem at
        # state-creation time (angr-m7s7y). Mirrors readlink(2): each value is
        # the raw target bytes (NOT NUL-terminated). Python ``state.fs``
        # symlinks are NOT auto-mirrored — the harness opts in explicitly,
        # same trade-off as ``register_known_path`` (angr-11djq.6.2).
        # ``_add_rust_state`` replays these onto every seed state; forks
        # inherit via the Rust ``FileSystem`` Arc clone.
        self._pending_symlinks: dict[str, bytes] = {}
        from angr.sim_manager import resilience_errors

        self._resilience = resilience_errors(resilience)
        self._errors_raised: set[int] = set()
        if symlinks:
            for link, target in symlinks.items():
                self._pending_symlinks[str(link)] = (
                    target if isinstance(target, (bytes, bytearray)) else str(target).encode()
                )
        self._phase_activate(active_states)

    def _phase_boot(
        self,
        project,
        save_unconstrained,
        solver_timeout_ms,
        max_active_states,
        max_history,
        exploration_strategy,
        deterministic,
        use_native_lift,
    ):
        """Phase 1 (boot): construct the Rust manager and apply basic config."""
        _apply_rust_log_env()
        _collect_dead_managers()
        _live_managers.add(self)

        # angr-iaol.2: pin Z3 module-level random seeds BEFORE the first
        # Solver::new. The per-solver Z3_solver_set_params route is
        # broken for these keys (angr-iaol.1) — only the global path
        # takes effect. Apply early in __init__ before _RustExplorationManager
        # constructs its first solver. The pin is process-global; once
        # set it persists for every subsequent solver.
        if deterministic:
            _apply_deterministic_z3_globals()
            _warn_if_parallel_nondeterministic()
        self._deterministic = bool(deterministic)

        if not RUST_EXPLORATION_AVAILABLE:
            raise ImportError("RustExplorationManager not available: angr.rustylib.vex_engine failed to import.")

        self._project = project
        self._save_unconstrained = save_unconstrained
        is_le = project.arch.memory_endness == "Iend_LE"
        self._rust_mgr = _RustExplorationManager(project.arch.name, little_endian=is_le)

        # Native syscall handlers are registered by syscall name; the numbers
        # come from the SimOS's own syscall library, for each ABI it lists
        # (angr-mx00p.35). That also routes DECREE: SimCGC lists only the
        # cgcabi numbering, whose 1-7 collide with Linux i386. A SimOS without
        # a syscall library (plain SimOS, Windows) has no syscall procedures on
        # the Python engine either (``SimOS.syscall`` returns None), so it gets
        # no native handlers.
        simos = getattr(project, "simos", None)
        self._rust_mgr.set_syscall_tables(_syscall_tables(simos), _unresolvable_syscall_jumpkinds(simos, project.arch))

        # angr-op0dn.10.3: strict-deterministic witness selection. The Z3 seed
        # pin above only stabilizes *which model Z3 builds*; this makes the
        # witness a function of the constraints alone (unsigned-minimum for
        # `eval`, ascending prefix for `eval_upto`), which is what makes a
        # truncated result reproducible. Every state entering a stash inherits
        # it, and forks inherit from their parent.
        if self._deterministic:
            self._rust_mgr.set_deterministic(True)

        # Configure solver timeout
        if solver_timeout_ms != 30000:
            self._rust_mgr.set_solver_timeout(solver_timeout_ms)

        # Configure max active states limit
        if max_active_states is not None:
            self._rust_mgr.set_max_active_states(max_active_states)

        # Configure per-state history cap (1000 is the Rust default; only push
        # a non-default value to keep the FFI surface quiet in the common case)
        if max_history != 1000:
            self._rust_mgr.set_max_history(max_history)

        # Native (in-process) libVEX cold-block lifting (angr-z087y), on by
        # default (angr-op0dn.2.2; rationale in
        # docs/advanced-topics/rust_libvex_ffi.rst): any block the native
        # lifter cannot serve falls back to the pyvex callback with an
        # identical IRSB.
        self._use_native_lift = bool(use_native_lift)
        if self._use_native_lift:
            settings = _native_lift_settings(project)
            if settings is not None:
                self._rust_mgr.set_native_lift(project.arch, **settings)

        # The native x86g_use_seg_selector serves only protected mode; a
        # state whose globals clear CR0.PE turns it off (_note_x86_cr0).
        self._rust_mgr.set_x86_protected_mode(bool(project.arch.vex_archinfo.get("x86_cr0", 0) & 1))

        # Configure exploration strategy. Reuses the post-init setter so the
        # validation and FFI-call shape live in one place. Always invoke so a
        # bad value raises ValueError eagerly during construction; the
        # default-bfs path is cheap (one FFI hop into the Rust setter).
        self.set_exploration_strategy(exploration_strategy)

    def _phase_config(
        self,
        use_shared_lineage_solver,
        prefer_native_library_hooks,
    ):
        """Phase 2 (config): gate flags, counters/caches, callbacks + simprocedures."""
        # angr-3ms1 step 1b: opt-in flag for fork-time
        # SharedLineageSolver materialization. Stashed here so
        # _add_rust_state can push the value onto every seed state's
        # solver context. Default off keeps slice-1c's gate inert (and
        # the v5a5-spike baby-re BFS-thrash regression out of CI).
        self._use_shared_lineage_solver = bool(use_shared_lineage_solver)

        # angr-a8epx / angr-gorvf.3.2: native-dispatch gate for library hooks.
        # Off, native procedures fire only for hooks OUTSIDE every loaded object
        # (the extern-object stubs). On a dynamically-linked binary loaded with
        # ``auto_load_libs=True, use_sim_procedures=True`` every libc hook lands
        # inside the loaded libc's .text, so that would send all of them to
        # Python. On (the default; angr-gorvf.6) the native registry serves
        # those hooks too; main-object hooks still go to Python, preserving
        # ``proj.hook()`` overrides. It was off while the native string procs
        # scanned a wider symbolic window than angr's Python ones and could
        # fork-storm on symbolic data; ``MAX_SYMBOLIC_SCAN_BYTES`` (angr-gorvf.5)
        # closed that gap. ``ANGR_RUST_PREFER_NATIVE_LIBRARY_HOOKS=0`` forces the
        # old Python-dispatch behavior back.
        self._prefer_native_library_hooks = _resolve_env_flag(
            prefer_native_library_hooks,
            "ANGR_RUST_PREFER_NATIVE_LIBRARY_HOOKS",
            default=True,
        )

        # Performance profiling counters
        self._perf_stats = PerformanceTracker()
        # Per-procedure timing: {name: {'count': int, 'execute_ns': int}}
        self._procedure_times: dict[str, dict[str, int]] = {}

        # High-level instrumentation counters for optimization tracking
        self._stats_callback_count = 0  # total Python callbacks invoked
        self._stats_ffi_crossings = 0  # total FFI calls to Rust (run/get/set)
        self._stats_state_creations = 0  # full SimState objects created
        self._stats_cache_hits = 0  # state cache hits
        self._stats_cache_misses = 0  # state cache misses
        self._stats_technique_filter_calls = 0  # technique filter invocations
        self._stats_hook_sync_calls = 0  # _sync_hooks_before_step invocations
        self._stats_hook_sync_skips = 0  # fast-path skips (no new hooks)
        self._stats_time_in_callbacks_ns = 0  # cumulative time in callback code
        # angr-ymoe: orphan-BVS fallback counters. Both paths mint a Python
        # claripy.BVS that has no Rust counterpart — measure how often they
        # fire to decide between hard-error / Rust-side fresh symbol / delete.
        self._stats_orphan_bvs_mem_thunk = 0
        self._stats_orphan_bvs_sym_load_full_fail = 0
        # angr-4o7d: snapshot-restore orphan-BVS at rust_state_export.py
        # _restore_symbolic_regions. Different threat model from the two
        # rust_manager.py fallbacks above — runs at snapshot-export time,
        # not in the hot exploration loop.
        self._stats_orphan_bvs_snapshot_restore = 0
        # angr-7jv5: per-FFI counters for the RustStateProxy write-through
        # surfaces. These instrument SimProc-callback proxy traffic so we can
        # decide whether a within-callback write buffer would pay off. Each
        # counter increments on the Python side immediately before the PyO3
        # call in `rust_state_proxy.py`. Aggregates across the whole run.
        self._stats_proxy_mem_concrete_writes = 0
        self._stats_proxy_mem_ast_writes = 0
        # angr-p1s02: symbolic-address stores the lazy Multi-cell path could
        # not route (unbounded/unconstrained addr), fallen back to a
        # single-address concretization (angr's Max write strategy).
        self._stats_proxy_mem_symbolic_addr_fallback = 0
        # angr-5rjbq: proxy loads that Rust could not satisfy (unmapped/zero
        # lazy page) and that fell back to the pre-swap Python SimMemory —
        # recovers setup-time writes (e.g. flareon2015_5 ebp-relative pw
        # symbols) the callback memory proxy would otherwise miss.
        self._stats_proxy_mem_fallback_python_load = 0
        # angr-vmi2l.270: cat-(c) losses that used to be silent. A
        # SimProcedure store the automap path could not land, and a dirty page
        # whose concrete bytes could not be fetched for the Python mirror.
        self._stats_proxy_mem_automap_store_dropped = 0
        self._stats_dirty_page_fetch_failed = 0
        self._stats_proxy_reg_writes = 0
        self._stats_proxy_solver_adds = 0
        # angr-4ref8: symbolic-file export observability. Every v1 scope-gate
        # rejection in _export_fs_files_to_rust was debug-log-only, so a file
        # silently losing native serving (e.g. a 65537-byte file over the size
        # cap) was not counter-attributable against the symfile_* surface.
        # Count successful exports plus per-reason skips; pre-seed every reason
        # so the keys always appear (stable for bench_diff / --dump-counters).
        # Reasons mirror the docstring scope gate; `error`/`preamble` are the
        # two cat-(b) fallback-with-loss paths.
        self._stats_symfile_exports = 0
        self._stats_symfile_export_skips = dict.fromkeys(
            (
                "subclass",
                "has_end",
                "not_seekable",
                "file_exists",
                "endness",
                "size",
                "path_utf8",
                "error",
                "preamble",
            ),
            0,
        )
        # angr-qluof: count re-applied symbolic-file demotions on the merged
        # re-add path (lineage-aware demotion), a measurement surface for the
        # write-demotion re-arm.
        self._stats_symfile_redemotions = 0
        # angr-op0dn.11.7: Veritesting (step_state hook) dispatch observability.
        # `dispatches` counts step_state batches that routed through
        # dispatch_step_state_with_hooks(); `applied` counts states a step_state
        # hook actually rewrote (Veritesting's nested-analysis merge fired) as
        # opposed to declining to the native fallback.
        self._stats_veritesting_dispatches = 0
        self._stats_veritesting_applied = 0
        self._init_start = time.perf_counter_ns()

        # Track registered hooks to detect dynamically created continuations
        # SimProcedures can create continuation hooks via self.call() which
        # need to be registered with Rust before exploration continues
        # (Must be initialized before _register_simprocedures() is called)
        self._registered_hooks: dict = {}  # addr -> proc (see _register_hooks)

        # Set up callbacks
        _t0 = time.perf_counter_ns()
        self._setup_callbacks()
        self._perf_stats.set_init_phase("setup_callbacks", time.perf_counter_ns() - _t0)

        # Load binary regions
        _t0 = time.perf_counter_ns()
        self._load_binary_regions()
        self._perf_stats.set_init_phase("load_binary", time.perf_counter_ns() - _t0)

        # Register SimProcedures
        _t0 = time.perf_counter_ns()
        self._register_simprocedures()
        self._perf_stats.set_init_phase("register_simprocedures", time.perf_counter_ns() - _t0)

        # Track angr state mappings for callbacks
        # Using regular dict with periodic cleanup to prevent memory leaks
        self._state_cache: dict[int, angr.SimState] = {}

        # angr-lyvf2: seeds the Python init skip fast-forwarded past their own
        # PC, as (pre-init addr, un-advanced SimState, (stash, rust id) of each
        # state the pre-step produced). `explore()` consumes this once, then
        # drops the refs.
        self._preinit_seeds: list[tuple[int, angr.SimState, list[tuple[str, int | None]]]] = []
        # Every seed's ZERO_FILL_UNCONSTRAINED_MEMORY, which they must agree on (_phase_activate); None before
        # the first seed.
        self._seed_zero_fill: bool | None = None

        # Lazy SimState references handed out by _get_stash_states. Keyed by
        # Rust state id so repeated stash reads return the same wrapper
        # (preserves the `mgr.active[0] is mgr.active[0]` invariant). Wrappers
        # materialize their SimState on first attribute access — see
        # `_LazySimStateRef` in rust_state_export.py. Pruned in
        # `_cleanup_state_cache` alongside _state_cache.
        from angr.exploration.rust_state_export import _LazySimStateRef

        self._lazy_state_refs: dict[int, _LazySimStateRef] = {}
        # Bumped by every handoff to Rust (_invalidate_state_export_cache). A
        # held RustStateProxy keys its register memo and solver fork on it,
        # since a non-forking step keeps the state id (angr-vmi2l.266).
        self._rust_step_epoch = 0

        # Maximum state cache size. Cache is bounded to the in-flight callback
        # state plus a small LRU window of recently-mutated states; root states
        # are pinned and never count against the cap. A cache that tracked
        # O(active_states) was the dominant driver of Python-side memory
        # growth (angr-qm7w).
        self._max_state_cache_size = 8

        # Track current callback state for memory access during callbacks
        # This allows memory_load callback to access the correct symbolic state
        self._callback_state: angr.SimState | None = None

        # Per-state metadata (symbolic_pages / hook_symbolic_memory /
        # addr_to_ast) is now stored on the Rust side in `RustSimState`. Access
        # goes through `self._rust_mgr.{get,set}_state_*` PyO3 methods so the
        # storage and the state lifetime are unified — when Rust drops a state,
        # its metadata is freed automatically.

        # Track the current callback state ID for memory tracking during callbacks
        self._current_callback_state_id: int | None = None
        # Track which Rust state is being stepped for per-fork memory isolation
        self._current_stepping_state_id: int | None = None

        # Track procedure_data for SimProcedure continuations.
        # When a SimProcedure uses self.call() to invoke a function and register
        # a continuation, the procedure_data is stored here keyed by the continuation
        # address. When Rust invokes the continuation, we restore this data.
        # Maps continuation_addr -> procedure_data tuple.
        self._pending_procedure_data: dict[int, tuple] = {}
        # The procedure_data tuple the current callback was seeded with
        # (see _init_callback_callstack).
        self._injected_procedure_data: tuple | None = None

        # Cache for addresses where SimProcedure continuations always result in exit.
        # After the first time a continuation at an address produces only Ijk_Exit
        # successors, all subsequent callbacks are fast-deadended without state creation.
        # The value is the concrete exit status the fast path records (None: no status).
        self._exit_continuation_addrs: dict = {}

        # Track root state IDs for plugin restoration.
        # Maps state_id -> root_state_id (the original state from Python).
        # When Rust forks states, this allows finding the original state for plugin copying.
        # Pruned in `_cleanup_state_cache`: an entry whose key state is no
        # longer in any Rust stash is dropped (state IDs are monotonically
        # allocated and never reused, so this is safe).
        self._state_roots: dict[int, int] = {}
        # Plugin templates kept outside `_state_cache`, by root state id: a
        # SimState whose posix/libc/heap/fs/log/globals a root's exports copy, for an
        # owner (`RustBackend`) whose roots' own ids are exported and so may
        # not be cached. The owner drops each entry with its lineage.
        self._lineage_templates: dict[int, angr.SimState] = {}
        # The SimState an owner (`RustBackend`) last stepped each state id from, which it does
        # not cache either: what a callback on the id, or its successors' plugins, copy from.
        # Set as a step begins, so a successor exported mid-step never stands in for its parent.
        self._step_sources: weakref.WeakValueDictionary[int, angr.SimState] = weakref.WeakValueDictionary()
        # Called after every explore(n=) round when set, by an owner (`RustBackend`) standing in
        # for a technique whose step() reorders the stashes, as DFS's does. See
        # `_explore_reordered_rounds`.
        self._round_reorder: Callable[[], None] | None = None
        # Called after every explore(n=) round when set, by an owner (`RustBackend`) that keeps
        # some of the run's states stepping on Python: it steps them once and returns whether
        # any is still active and how many of them it has found so far. See
        # `_explore_reordered_rounds`.
        self._round_side_step: Callable[[], tuple[bool, int]] | None = None

        # Per-state-id Python-side stand-ins for state.options and state.globals.
        # The Rust engine doesn't honor SimOptions (LAZY_SOLVES / STRICT_PAGE_ACCESS
        # are mirrored separately on the Rust state itself), but user-facing code
        # — predicates, exploration techniques, callbacks — frequently reads
        # state.options.add(X) and state.globals[k] = v. Storing the set/dict
        # Python-side keyed by state_id lets RustStateProxy expose live mutable
        # views without round-tripping through Rust. Children inherit a deep
        # copy from their root on first access (see get_state_options_py /
        # get_state_globals_py).
        self._py_state_options: dict[int, set] = {}
        self._py_state_globals: dict[int, dict] = {}

        # Track which silently-divergent SimOptions we've already warned about
        # for this manager so the warn-once helper does not spam during runs
        # that add many states. Cleared at __init__ time, so a fresh manager
        # re-warns for the same option.
        self._warned_rejected_options: set = set()

        # Track active exploration techniques (applied during exploration steps).
        self._active_techniques: list = []

        # state.inspect MVP storage (angr-uq4n, angr-d46u).
        # Manager-wide breakpoint registry — shared across all states this
        # manager owns. RustInspectProxy is the user-facing facade.
        # Initialized from the single source of truth in rust_state_proxy
        # (`_INSPECT_EVENT_SPECS`); to wire a new event, edit that table
        # and follow the recipe documented there.
        # See docs/advanced-topics/rust_engine.rst for the MVP scope.
        from angr.exploration.rust_state_proxy import (
            _RUST_INSPECT_EVENT_BITS,
            _RUST_INSPECT_SUPPORTED_EVENTS,
        )

        self._inspect_breakpoints: dict[str, list] = {evt: [] for evt in _RUST_INSPECT_SUPPORTED_EVENTS}
        # Caller-owned states handed to ``_add_rust_state``: a BP set on one
        # after it was added is adopted at the next step (angr-vmi2l.176).
        self._inspect_seed_states: list[weakref.ref] = []
        self._INSPECT_EVENT_BITS = dict(_RUST_INSPECT_EVENT_BITS)
        # Reentrancy guard: when a user action callback runs, suppress
        # nested inspect dispatch on the same manager. uq4n.4 covers
        # the full guard test.
        self._inspect_dispatch_depth = 0
        # (event, attr) pairs whose ignored BP write already warned, so
        # `_warn_ignored_inspect_writes` logs each once (angr-vmi2l.206).
        self._inspect_ignored_write_warned: set[tuple[str, str]] = set()
        # Lazy RustInspectProxy instance (one per manager, shared across proxies).
        self._inspect_proxy: RustInspectProxy | None = None

        # Track stdin BVS variables for state export.
        # List of (claripy_bvs, size_ast) tuples from SimPacketsStream.content.
        # Populated during SimProcedure callbacks that read from stdin (fgets, read, etc.).
        # Used to restore stdin content on found states that were forked purely in Rust.
        self._stdin_content: list = []

    def _phase_activate(self, active_states):
        """Phase 3 (activate): add initial states to the Rust 'active' stash."""
        # Add initial states
        if active_states:
            # Handle single state or list of states
            if hasattr(active_states, "solver"):  # Single SimState
                active_states = [active_states]
            # angr-027h two-phase explore: keep the pristine seed states. If a
            # find-based explore exhausts to active_empty without finding,
            # `_explore_with_addresses` re-seeds them in eager mode. These MUST
            # be copies, not references: the callback-dispatch path re-uses the
            # seed SimState object as the Python-side mirror of its root Rust
            # state, so a Python bounce mutates it in place (pc lands on the
            # bounced SimProcedure, constraints/memory move with the path). A
            # reference here made the phase-2 retry replay a mid-path,
            # constraint-less state that could reach the find address and land
            # in `found` carrying only the find gate (angr-je2xt). Skip when
            # re-seeding (the retry passes copies of these and must not
            # overwrite the pristine originals).
            if not getattr(self, "_phase2_reseeding", False):
                self._initial_seed_states = [_pristine_seed_copy(s) for s in active_states]
                self._frontier_is_seeds = True
                self._seed_of_state_id = {}
            seed_concretization = None
            for seed_index, state in enumerate(active_states):
                # Detect state options
                if hasattr(state, "options"):
                    # Hard-fail on options the Rust engine cannot honor
                    # before doing any further work; warn-once on the rest.
                    # This runs on the user-supplied state, before
                    # _run_python_init_if_needed steps it to main.
                    self._check_raise_options(state.options)
                    self._warn_rejected_options(state.options)
                    try:
                        from angr import sim_options as o

                        # LAZY_SOLVES is latched from the stepped states at each run
                        # (_latch_lazy_solves) and ZERO_FILL_UNCONSTRAINED_MEMORY per state
                        # by _add_rust_state (angr-vmi2l.241). The page callbacks answer
                        # from one seed's memory, though, so the seeds must agree on it.
                        zero_fill = o.ZERO_FILL_UNCONSTRAINED_MEMORY in state.options
                        if self._seed_zero_fill not in (None, zero_fill):
                            raise NotImplementedError(
                                "RustExplorationManager: ZERO_FILL_UNCONSTRAINED_MEMORY is set on some seeds "
                                "and not others; the Rust engine's page callbacks serve every state from one "
                                "seed's memory, so build one manager per setting."
                            )
                        self._seed_zero_fill = zero_fill
                        # Configure concretization strategies to match Python's
                        concretization = _concretization_key(state)
                        if seed_index and concretization != seed_concretization:
                            # One manager-wide concretizer: a later seed must not overwrite an
                            # earlier one's (angr-vmi2l.241).
                            raise NotImplementedError(
                                "RustExplorationManager: the seeds differ in their memory concretization "
                                "options or strategy limits, which the Rust engine applies manager-wide; "
                                "build one manager per configuration."
                            )
                        seed_concretization = concretization
                        self._rust_mgr.configure_concretization_strategies(*concretization)
                        l.debug(
                            "Configured concretization: approx=%s, read_limit=%d, "
                            "write_limit=%d, sym_write=%s, avoid_reads=%s, "
                            "avoid_writes=%s",
                            *concretization,
                        )
                    except ImportError:
                        # cat-(a) EXPECTED CONTROL FLOW: optional sim_options import.
                        # If absent, skip option-driven Rust configuration.
                        pass

                # Hybrid init: if the state starts at a loader/init address
                # (not in the main binary), run the init sequence in Python
                # first. This handles C++ constructors, .init_array, etc.
                # that the Rust engine can't execute correctly.
                _t0 = time.perf_counter_ns()
                seed_state, seed_addr = state, state.addr
                carried = self._run_python_init_if_needed(state)
                self._perf_stats.add_init_phase("python_run", time.perf_counter_ns() - _t0)

                _t0 = time.perf_counter_ns()
                placed = [(stash, self._add_rust_state(stash, s)) for stash, s in carried]
                self._perf_stats.add_init_phase("add_rust_state", time.perf_counter_ns() - _t0)

                # angr-lyvf2: the init skip fast-forwards the seed past its own
                # PC, so Rust's pre-step find check can never see `seed_addr`.
                # Park the un-advanced seed so `explore()` can route it to FOUND
                # if the caller's find targets that address (vanilla angr's
                # Explorer matches it at filter() time, before any step).
                # The phase-2 retry (angr-027h) re-activates copies of the same
                # seeds mid-explore; parking them again would double-report the
                # find (angr-bdeqa), so only the original activation parks.
                pre_stepped = carried != [("active", seed_state)]
                if not getattr(self, "_phase2_reseeding", False):
                    self._seed_of_state_id.update((state_id, seed_index) for _, state_id in placed)
                    if pre_stepped:
                        self._preinit_seeds.append((seed_addr, seed_state, placed))

        self._install_python_servable_pages()

        self._perf_stats.set_init_phase("total", time.perf_counter_ns() - self._init_start)

    def _install_python_servable_pages(self) -> None:
        """Tell Rust which pages ``_cb_fetch_page`` could actually serve.

        angr-gorvf.4.6: every page the run loop still fetches on the ZeroPy
        FAIL benches is a lazy-stack page Python DECLINES (measured: every one,
        across the benches whose whole residual GIL was this one site).
        Rust paid a GIL attach per fetch just to be told no. Python's page
        universe is frozen after ``_sync_extra_python_pages`` hands Rust every
        page it can serve, so we can snapshot the servable set once here and
        let Rust decline everything outside it natively, with no crossing.

        The snapshot is skipped entirely (filter off, every fetch crosses) when
        a seed has no page dict, or has ``ZERO_FILL_UNCONSTRAINED_MEMORY`` —
        under which Python synthesizes a zero page for *any* address, so no
        finite set describes what it can serve. Individual pages the fast
        classifier cannot rule on are kept servable, so they too keep crossing.

        Note the servable set is NOT empty in general: ``_sync_extra_python_pages``
        deliberately leaves all-zero pages lazy once a state has more than
        ``zero_eager_cap`` of them (mma_howtouse), and Python does serve those.
        """
        callbacks = self._callbacks
        if callbacks is None or not hasattr(callbacks, "set_python_servable_pages"):
            return
        servable: set[int] = set()
        universe: set[int] = set()
        for state in self._state_cache.values():
            mem_pages = getattr(state.memory, "_pages", None)
            if mem_pages is None or "ZERO_FILL_UNCONSTRAINED_MEMORY" in state.options:
                return
            for page_no in list(mem_pages.keys()):
                page_addr = page_no * PAGE_SIZE
                universe.add(page_addr)
                fast = self._fetch_page_from_ultrapage(state, page_addr)
                # ``None`` means the fast classifier can't rule on this page
                # (not a 4096-byte UltraPage). Call it servable: Rust keeps
                # crossing for it and Python answers on the slow load+eval
                # path, exactly as before. Only pages we can positively prove
                # Python would decline are filtered out.
                if fast is None or fast[2]:
                    servable.add(page_addr)
        callbacks.set_python_servable_pages(sorted(servable))
        # angr-gorvf.4.7: the load-side oracle. Deliberately the *unfiltered*
        # page set — a page Python declines to serve as a concrete fetch (it
        # holds symbolic bytes) still answers a ``memory_load`` from that
        # symbolic data, so it must stay crossable. Only a page Python has no
        # object for at all can be served natively with a filler.
        #
        # The loader regions must be folded in too, and that is not optional:
        # angr's memory serves a loader-backed address straight from the CLE
        # backer, materializing the page on first touch. Such a page is absent
        # from ``_pages`` until something reads it, so a universe built from
        # ``_pages`` alone would let Rust synthesize a filler over real binary
        # bytes. (Measured when it did: csgames2018 regressed — the fabricated
        # symbols turned concrete loads symbolic and dragged in solver work and
        # SimProcedure bounces.)
        if hasattr(callbacks, "set_python_page_universe"):
            for region_start, region_size in self._get_loader_pages_cache(PAGE_SIZE)["lazy_regions"]:
                base = region_start - (region_start % PAGE_SIZE)
                for page_addr in range(base, region_start + region_size, PAGE_SIZE):
                    universe.add(page_addr)
            callbacks.set_python_page_universe(sorted(universe))

    def perf_report(self) -> str:
        """Return a formatted performance report."""
        s = self._perf_stats
        lines = ["=== Rust Engine Performance Report ==="]
        lines.append(f"Init total: {s['init_total_ns'] / 1e6:.1f}ms")
        lines.append(f"  Setup callbacks: {s['init_setup_callbacks_ns'] / 1e6:.1f}ms")
        lines.append(f"  Load binary regions: {s['init_load_binary_ns'] / 1e6:.1f}ms")
        lines.append(f"  Register SimProcedures: {s['init_register_simprocedures_ns'] / 1e6:.1f}ms")
        lines.append(f"  Python init: {s['init_python_run_ns'] / 1e6:.1f}ms")
        lines.append(f"  Add Rust state: {s['init_add_rust_state_ns'] / 1e6:.1f}ms")
        lines.append(f"    Memory sync: {s['init_memory_sync_ns'] / 1e6:.1f}ms")
        lines.append(f"    Register sync: {s['init_register_sync_ns'] / 1e6:.1f}ms")
        lines.append(f"SimProcedure callbacks: {s['callback_simprocedure_count']}")
        lines.append(f"  Total time: {s['callback_simprocedure_total_ns'] / 1e6:.1f}ms")
        lines.append(f"  State create: {s['callback_simprocedure_state_create_ns'] / 1e6:.1f}ms")
        lines.append(f"  Execute: {s['callback_simprocedure_execute_ns'] / 1e6:.1f}ms")
        lines.append(f"  State copy: {s['callback_simprocedure_state_copy_ns'] / 1e6:.1f}ms")
        lines.append(f"  Sync back: {s['callback_simprocedure_sync_back_ns'] / 1e6:.1f}ms")
        if self._procedure_times:
            lines.append("  Per-procedure breakdown:")
            for pname, pt in sorted(self._procedure_times.items(), key=lambda x: -x[1]["execute_ns"]):
                lines.append(f"    {pname}: {pt['count']}x {pt['execute_ns'] / 1e6:.1f}ms")
        lines.append(f"Memory load callbacks: {s['callback_memory_load_count']}")
        lines.append(f"  Total time: {s['callback_memory_load_total_ns'] / 1e6:.1f}ms")
        if s["callback_memory_load_count"] > 0:
            lines.append(
                f"  Avg per call: {s['callback_memory_load_total_ns'] / s['callback_memory_load_count'] / 1e3:.1f}us"
            )
        lines.append(f"Fetch page callbacks: {s['callback_fetch_page_count']}")
        lines.append(f"  Total time: {s['callback_fetch_page_total_ns'] / 1e6:.1f}ms")
        lines.append(f"Lift block callbacks: {s['callback_lift_block_count']}")
        lines.append(f"  Total time: {s['callback_lift_block_total_ns'] / 1e6:.1f}ms")
        if s["callback_lift_block_count"] > 0:
            lines.append(
                f"  Avg per call: {s['callback_lift_block_total_ns'] / s['callback_lift_block_count'] / 1e3:.1f}us"
            )
        # angr-xtse.1: per-category Python callback timing for upper-bound
        # speedup analysis. Every path below is instrumented from the public
        # _handle_* entry points in rust_callback_dispatch.py.
        for label, key in (
            ("Syscall", "syscall"),
            ("Find predicate", "find_predicate"),
            ("Avoid predicate", "avoid_predicate"),
            ("Symbolic branch", "symbolic_branch"),
            ("Python VEX fallback", "vex_fallback"),
            ("Posix inject", "posix"),  # (angr-afbx)
        ):
            count = s.get(f"callback_{key}_count", 0)
            ns = s.get(f"callback_{key}_total_ns", 0)
            lines.append(f"{label} callbacks: {count}")
            lines.append(f"  Total time: {ns / 1e6:.1f}ms")
            if count > 0:
                lines.append(f"  Avg per call: {ns / count / 1e3:.1f}us")
        # Per-category fallback counters (angr-md0m). Pulled live from
        # self.stats — values may be 0 if the run never tripped the path.
        try:
            fb = self.stats
        except MemoryError:
            raise
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: stats unavailable; the perf report
            # omits the fallback-counter section but still prints other phases.
            fb = {}
        if fb:
            lines.append("Fallback counters:")
            lines.append(f"  SimProcedure -> Python: {fb.get('simprocedure_python_fallback_count', 0)}")
            by_name = fb.get("simprocedure_fallback_by_name", {}) or {}
            if by_name:
                top = sorted(by_name.items(), key=lambda kv: -kv[1])[:10]
                for pname, pcount in top:
                    lines.append(f"    {pname}: {pcount}")
            lines.append(f"  Syscall -> Python: {fb.get('syscall_python_fallback_count', 0)}")
            lines.append(f"  Dirty call -> Python: {fb.get('rust_python_dirty_call_count', 0)}")
            lines.append(f"  VEX op fallback (silent): {fb.get('rust_python_vex_op_fallback_count', 0)}")
            lines.append(
                f"    unop {fb.get('rust_python_vex_unop_fallback_count', 0)}, "
                f"binop {fb.get('rust_python_vex_binop_fallback_count', 0)}, "
                f"triop {fb.get('rust_python_vex_triop_fallback_count', 0)}, "
                f"qop {fb.get('rust_python_vex_qop_fallback_count', 0)}"
            )
            lines.append(f"  DCAS unsupported: {fb.get('dcas_unsupported_count', 0)}")
            lines.append(f"  VEX block fallback (PythonVEXFallback): {fb.get('vex_fallback_count', 0)}")
        return "\n".join(lines)

    def _setup_callbacks(self):
        """Set up Python callbacks for the Rust engine.

        Registers bound methods as callbacks with the Rust PythonCallbacks object.
        Each _cb_* method implements one callback type.

        PythonCallbacks (PyO3 pyclass) implements __traverse__/__clear__ so the
        cycle (mgr -> _callbacks -> bound method -> mgr) is GC-collectible.
        Without that GC support, the manager and its _state_cache (4030 angr
        pages per call in mma_howtouse) would leak permanently.
        """
        # Cache the stepping state ID accessor for _get_per_fork_state
        try:
            from angr.rustylib.vex_engine import get_stepping_state_id

            self._get_stepping_state_id = get_stepping_state_id
        except ImportError:
            # cat-(b) FALLBACK WITH LOSS: older Rust build without
            # get_stepping_state_id; dispatcher uses a None-returning lambda,
            # so per-fork state lookup falls back to the default state.
            self._get_stepping_state_id = lambda: None

        callbacks = PythonCallbacks()
        callbacks.set_memory_load(self._cb_memory_load)
        callbacks.set_memory_store(self._cb_memory_store)
        callbacks.set_lift_block(self._cb_lift_block)
        callbacks.set_fetch_page(self._cb_fetch_page)
        callbacks.set_dirty_call(self._cb_dirty_call)
        callbacks.set_resolve_function(self._cb_resolve_function)
        if hasattr(callbacks, "set_memory_store_batch"):
            callbacks.set_memory_store_batch(self._cb_memory_store_batch)
        if hasattr(callbacks, "set_memory_load_batch"):
            callbacks.set_memory_load_batch(self._cb_memory_load_batch)
        if hasattr(callbacks, "set_batch_fetch_pages"):
            callbacks.set_batch_fetch_pages(self._cb_batch_fetch_pages)
        if hasattr(callbacks, "set_memory_store_symbolic_value"):
            callbacks.set_memory_store_symbolic_value(self._cb_memory_store_symbolic_value)
        if hasattr(callbacks, "set_memory_store_symbolic_full"):
            callbacks.set_memory_store_symbolic_full(self._cb_memory_store_symbolic_full)
        if hasattr(callbacks, "set_memory_load_symbolic_full"):
            callbacks.set_memory_load_symbolic_full(self._cb_memory_load_symbolic_full)
        # state.inspect MVP (angr-uq4n.2, angr-d46u) — register dispatchers
        # even when no BPs are set so the Rust side has a target if
        # instrumentation fires unexpectedly. The bitmask gates dispatch.
        # Iterate the single source of truth so adding a new event
        # auto-wires registration. Read directly from the module rather
        # than self._INSPECT_EVENT_BITS so this method works even before
        # __init__ has finished. The hasattr() check tolerates older .so
        # builds. Skip events whose dispatch fires from Python
        # (simprocedure/syscall/dirty; angr-xmfj); they have no PythonCallbacks slot
        # because the Rust engine never invokes them.
        from angr.exploration.rust_state_proxy import (
            _RUST_INSPECT_EVENT_BITS,
            _RUST_INSPECT_PYTHON_DISPATCHED_EVENTS,
        )

        for evt in _RUST_INSPECT_EVENT_BITS:
            if evt in _RUST_INSPECT_PYTHON_DISPATCHED_EVENTS:
                continue
            setter_name = f"set_inspect_{evt}"
            cb_name = f"_cb_inspect_{evt}"
            if hasattr(callbacks, setter_name):
                getattr(callbacks, setter_name)(getattr(self, cb_name))
        self._rust_mgr.set_callbacks(callbacks)
        self._callbacks = callbacks

    # ---- Per-fork state resolution ----

    def _get_per_fork_state(self):
        """Get the correct per-fork Python state for the current VEX step."""
        state = self._get_callback_state()
        if state is not None:
            return state
        sid = self._get_stepping_state_id()
        if sid is not None and sid in self._state_cache:
            return self._state_cache[sid]
        return self._get_default_state()

    # ---- Individual callback methods ----

    def _cb_memory_load(self, addr: int, size: int) -> tuple:
        _ml_start = time.perf_counter_ns()
        try:
            state = self._get_per_fork_state()
            if state is None:
                return (bytes(size), False, None)

            try:
                # Check preserved hook symbolic memory first
                state_id = self._current_callback_state_id
                effective_state_id = self._get_effective_state_id(state_id) if state_id is not None else None
                lookup_id = effective_state_id if effective_state_id is not None else state_id
                hook_mem = self._rust_mgr.get_state_hook_symbolic_memory(lookup_id) if lookup_id is not None else {}
                if hook_mem:
                    for mem_addr, (ast, mem_size) in hook_mem.items():
                        if mem_addr <= addr < mem_addr + mem_size:
                            offset = addr - mem_addr
                            if offset == 0 and size == mem_size:
                                concrete = state.solver.eval(ast).to_bytes(size, "little")
                                self._register_handle(id(ast), ast, addr=addr, size=size, state_id=lookup_id)
                                if _DBG:
                                    l.debug(f"Memory load hit preserved symbolic at 0x{addr:x}")
                                return (concrete, True, ast)
                            if offset == 0 and size < mem_size:
                                extracted = claripy.Extract(size * 8 - 1, 0, ast)
                                concrete = state.solver.eval(extracted).to_bytes(size, "little")
                                self._register_handle(
                                    id(extracted), extracted, addr=addr, size=size, state_id=lookup_id
                                )
                                return (concrete, True, extracted)

                # angr-hcok: when the state's ``memory`` plugin is a
                # ``RustMemoryProxy``, it re-enters ``_rust_mgr.get_state_memory_ast(...)`` —
                # but ``_cb_memory_load`` fires from inside ``_rust_mgr.run()``
                # which holds ``&mut self`` on the manager PyCell, so the
                # re-entry raises "Already mutably borrowed". Synthesize a
                # filler BVS instead (mirrors angr's filler_mixin default for
                # unmapped memory). Rust stores the AST in its symbolic memory
                # so subsequent loads at the same address don't re-enter this
                # path. Skip the proxy round-trip — Rust already owns memory.
                if _is_rust_memory_proxy(state.memory):
                    ast = claripy.BVS(f"mem_filler_{addr:x}_{size}", size * 8)
                    self._register_handle(id(ast), ast, addr=addr, size=size, state_id=state_id)
                    if _DBG:
                        l.debug(
                            "Memory load 0x%x size=%d: proxy-gate filler BVS",
                            addr,
                            size,
                        )
                    return (bytes(size), True, ast)

                val = state.memory.load(addr, size, endness=state.arch.memory_endness)

                # Coerce thunks/callables to actual values
                coerce_attempts = 0
                while callable(val) and not hasattr(val, "op") and coerce_attempts < 3:
                    try:
                        val = val()
                        coerce_attempts += 1
                    except MemoryError:
                        raise
                    except Exception:
                        # cat-(b) FALLBACK WITH LOSS: thunk in memory failed to resolve;
                        # replace with a fresh BVS so the load proceeds. Caller pays a
                        # downstream symbolic constraint instead of a hard error.
                        # angr-ymoe: no fires across the fast-tier
                        # benches. Path is dead in practice; counter acts as a watchdog
                        # — if it ever goes non-zero, escalate to a Rust-side fresh
                        # symbol (the _set_state_register_symbolic_ast FFI shim
                        # shows the pattern).
                        if _DBG:
                            l.debug(f"Memory load thunk at 0x{addr:x} failed to resolve, creating symbolic")
                        val = claripy.BVS(f"mem_thunk_{addr:x}", size * 8)
                        self._stats_orphan_bvs_mem_thunk += 1
                        break

                if not hasattr(val, "op"):
                    l.warning(f"Memory load at 0x{addr:x} returned invalid type: {type(val)}")
                    return (bytes(size), False, None)

                is_symbolic = getattr(val, "symbolic", False)
                if is_symbolic:
                    handle_id = id(val)
                    self._register_handle(handle_id, val, addr=addr, size=size, state_id=state_id)
                    concrete = state.solver.eval(val).to_bytes(size, "little")
                    return (concrete, True, val)
                concrete = state.solver.eval(val).to_bytes(size, "little")
                return (concrete, False, None)
            except (SimError, ClaripyError) as e:
                # cat-(c) WRONG-ANSWER RISK: memory load returned zero bytes after
                # Sim/Claripy error — Rust sees concrete zero where the program
                # might have stored real data. Already warns.
                l.warning(f"Memory load error at 0x{addr:x}: {e}")
                return (bytes(size), False, None)
        finally:
            self._perf_stats.record_memory_load(time.perf_counter_ns() - _ml_start)

    def _cb_memory_store(self, addr: int, data: bytes):
        state = self._get_per_fork_state()
        if state is None:
            return
        # angr-hcok: when ``state.memory`` is a ``RustMemoryProxy``, it
        # routes ``store(...)`` back into
        # ``_rust_mgr.set_state_memory_concrete(...)`` — but we're inside
        # ``_rust_mgr.run()`` which holds ``&mut self``, so the re-entry
        # raises "Already mutably borrowed". Rust already performed the
        # store internally before invoking this callback, so the Python
        # shadow is redundant then; just acknowledge and return.
        if _is_rust_memory_proxy(state.memory):
            return
        try:
            val = claripy.BVV(int.from_bytes(data, "little"), len(data) * 8)
            state.memory.store(addr, val, endness=state.arch.memory_endness)
        except (SimError, ClaripyError) as e:
            # cat-(c) WRONG-ANSWER RISK: memory store silently dropped on Sim/
            # Claripy error — subsequent loads see stale data. Already warns.
            l.warning(f"Memory store error at 0x{addr:x}: {e}")

    def _cb_lift_block(self, addr: int, opt_level: int | None = None, dirty_bytes: bytes | None = None):
        """Lift the block at ``addr`` for the Rust engine, which reads the
        returned ``pyvex.IRSB`` directly (``vex/pyvex_reader.rs``). Returns None
        when no block can be lifted there."""
        _lb_start = time.perf_counter_ns()
        try:
            try:
                # angr-4aach: fire the vex_lift inspect event around this lift.
                # _cb_lift_block is the Rust block-cache miss path, mirroring
                # Python's lifter which only fires vex_lift when its cache is
                # not used. Attributed to the callback/default state (lifts are
                # state-independent in the Rust engine's shared block cache).
                if self._inspect_breakpoints.get("vex_lift"):
                    self._cb_inspect_vex_lift(-1, "before", addr, None, buff=dirty_bytes)
                kwargs = {}
                if opt_level is not None:
                    kwargs["opt_level"] = opt_level
                if dirty_bytes is not None:
                    # SMC: Rust signaled that this lift range is on a page that
                    # has been overwritten via state.memory. The cle static
                    # binary buffer is stale; lift the fresh bytes Rust sent
                    # instead.
                    kwargs["byte_string"] = dirty_bytes
                elif self._project.kb.patches.items():
                    # A stateless lift with kb.patches set reads the patched
                    # entry state, but a stepped state lifts its own memory
                    # (VEXLifter._load_bytes). An unstored page of state
                    # memory is the loaded image; a patched state's pages
                    # arrive as dirty_bytes (angr-vmi2l.75).
                    with contextlib.suppress(KeyError):
                        kwargs["byte_string"] = self._project.loader.memory.load(addr, VEX_IRSB_MAX_SIZE)
                block = self._project.factory.block(addr, **kwargs)
                irsb = block.vex
                if self._inspect_breakpoints.get("vex_lift"):
                    self._cb_inspect_vex_lift(-1, "after", addr, irsb.size)
                return irsb
            except (SimEngineError, ClaripyError, PyVEXError) as e:
                # Rust routes None to the deadended stash (an unliftable block).
                # A hooked address lands here on purpose: a zero-length terminal
                # hook (CallReturn, the Callable's return sentinel) resumes at
                # its own address so the state deadends with that PC; one per
                # Callable call is not worth a warning (angr-mx00p.96).
                log = l.debug if self._project.is_hooked(addr) else l.warning
                log(f"Lift error at 0x{addr:x}: {e}")
                return None
        finally:
            self._perf_stats.record_lift_block(time.perf_counter_ns() - _lb_start)

    def _cb_fetch_page(self, page_addr: int) -> tuple:
        _fp_start = time.perf_counter_ns()
        try:
            state = self._get_default_state()
            if state is None:
                return (bytes(4096), 0, False)
            # angr-hcok: when ``state.memory`` is a ``RustMemoryProxy``, it
            # calls back into ``_rust_mgr`` —
            # invalid while ``run()`` holds ``&mut self``. Decline the page;
            # Rust will fall back to its own zero-fill / static-binary page
            # source rather than asking Python for a copy.
            if _is_rust_memory_proxy(state.memory):
                return (bytes(4096), 0, False)
            try:
                return self._fetch_page_bytes(state, page_addr)
            except (SimError, ClaripyError):
                # cat-(b) FALLBACK WITH LOSS: page fetch failed; return empty page
                # with perms=0 so Rust marks it inaccessible (a load there will
                # error rather than silently succeed). Already debug-logs.
                l.debug("fetch_page 0x%x: failed to load/eval, returning empty", page_addr, exc_info=True)
                return (bytes(4096), 0, False)
        finally:
            self._perf_stats.record_fetch_page(time.perf_counter_ns() - _fp_start)

    def _cb_dirty_call(self, name: str, args: list, ret_ty_bits: int) -> tuple[bytes, bool, object | None]:
        """Run the Python dirty helper ``name`` for the Rust interpreter.

        The handler runs against the callback/default SimState, not the Rust
        state, so its own memory and register effects do not reach the
        engine. That is why Rust routes a helper with a memory effect to
        Python's VEX engine instead (`VEXInterpreter::handle_dirty_call`),
        and why a handler's constraints are refused here rather than added to
        a solver the Rust state never sees. A missing handler or a handler
        exception raises, as the Python engine does (angr-vmi2l.170); Rust
        reports the state errored. A symbolic result travels as its AST.
        """
        state = self._get_callback_state() or self._get_default_state()
        if state is None:
            raise UnsupportedDirtyError(f"dirty call {name}: no state to run the handler on")
        # state.inspect dispatch (angr-xmfj). state_id falls back to -1
        # because dirty calls run from a Rust interpreter dispatch where the
        # state-id is not threaded through this callback yet; the inspect
        # proxy falls back to the cached default state, which is the same
        # SimState the handler sees here. BPs that read `state.inspect.*`
        # see the live attrs.
        sid = getattr(self, "_current_callback_state_id", None)
        if sid is None:
            sid = -1
        from angr.engines.vex.heavy import dirty as dirty_module

        handler = getattr(dirty_module, name, None)
        if handler is None:
            raise UnsupportedDirtyError(f"Unsupported dirty helper {name}")
        claripy_args = [claripy.BVV(arg, 64) for arg in args]

        self._cb_inspect_dirty(sid, "before", name, handler, claripy_args, None)

        result, constraints = handler(state, *claripy_args)
        if constraints:
            raise UnsupportedDirtyError(f"dirty call {name}: its constraints cannot be applied to the Rust state")

        override = self._cb_inspect_dirty(sid, "after", name, handler, claripy_args, result)
        if override is not None:
            # User BP set state.inspect.dirty_result — honor the override
            # in place of the handler's return value (angr-uy32).
            result = override

        num_bytes = ret_ty_bits // 8
        if result is None:
            return (bytes(num_bytes), False, None)
        if getattr(result, "symbolic", False):
            return (bytes(num_bytes), True, result)
        return (state.solver.eval(result).to_bytes(num_bytes, "little"), False, None)

    def _cb_resolve_function(self, addr: int, name: str | None) -> tuple[str, int, bool] | None:
        """Resolve an unmodeled function call."""
        if hasattr(self._project, "_sim_procedures") and addr in self._project._sim_procedures:
            proc = self._project._sim_procedures[addr]
            proc_name = proc.__class__.__name__ if hasattr(proc, "__class__") else str(proc)
            num_args = getattr(proc, "num_args", 0) or 0
            no_ret = _proc_no_return(proc)
            return (proc_name, num_args, no_ret)

        if hasattr(self._project, "loader"):
            obj = self._project.loader.find_object_containing(addr)
            if obj:
                in_plt = False
                for section in obj.sections:
                    if section.name in (".plt", ".plt.got", ".plt.sec") and section.min_addr <= addr < section.max_addr:
                        in_plt = True
                        break

                if in_plt:
                    proc_by_name = {}
                    for proc_addr, proc in self._project._sim_procedures.items():
                        proc_name = proc.__class__.__name__ if hasattr(proc, "__class__") else str(proc)
                        proc_by_name[proc_name] = (proc_addr, proc)

                    got_to_sym = {}
                    if hasattr(obj, "jmprel"):
                        for sym_name, reloc in obj.jmprel.items():
                            got_to_sym[reloc.rebased_addr] = sym_name

                    try:
                        block = self._project.factory.block(addr, num_inst=1)
                        insn = block.capstone.insns[0] if block.capstone.insns else None
                        if insn and insn.mnemonic == "jmp":
                            for op in insn.operands:
                                if op.type == 3:  # CS_OP_MEM
                                    got_addr = _plt_got_slot(obj, insn, op, self._project.arch.bits)
                                    if got_addr is None:
                                        continue
                                    if got_addr in got_to_sym:
                                        sym_name = got_to_sym[got_addr]
                                        if sym_name in proc_by_name:
                                            proc_addr, proc = proc_by_name[sym_name]
                                            num_args = getattr(proc, "num_args", 0) or 0
                                            no_ret = _proc_no_return(proc)
                                            l.debug(f"Resolved PLT at 0x{addr:x} to {sym_name} (GOT 0x{got_addr:x})")
                                            return (sym_name, num_args, no_ret)
                                    else:
                                        state = self._get_default_state()
                                        if state:
                                            ptr_size = self._project.arch.bytes
                                            got_val = state.memory.load(got_addr, ptr_size, endness="Iend_LE")
                                            extern_addr = state.solver.eval(got_val)
                                            if extern_addr in self._project._sim_procedures:
                                                proc = self._project._sim_procedures[extern_addr]
                                                proc_name = (
                                                    proc.__class__.__name__ if hasattr(proc, "__class__") else str(proc)
                                                )
                                                num_args = getattr(proc, "num_args", 0) or 0
                                                no_ret = _proc_no_return(proc)
                                                l.debug(f"Resolved PLT at 0x{addr:x} to {proc_name} via GOT value")
                                                return (proc_name, num_args, no_ret)
                    except MemoryError:
                        raise
                    except Exception as e:
                        # cat-(b) FALLBACK WITH LOSS: PLT block lift / capstone parse
                        # failed; falls through to the next resolution heuristic (symbol
                        # name match). Already debug-logs.
                        l.debug(f"PLT resolution failed for 0x{addr:x}: {e}")

        if name:
            try:
                from angr.procedures import SIM_PROCEDURES

                for lib_name, procs in SIM_PROCEDURES.items():
                    if name in procs:
                        proc_class = procs[name]
                        num_args = getattr(proc_class, "num_args", 0) or 0
                        no_ret = _proc_no_return(proc_class)
                        l.debug(f"Resolved {name} to {lib_name}:{name}")
                        return (name, num_args, no_ret)
            except ImportError:
                # cat-(a) EXPECTED CONTROL FLOW: optional SIM_PROCEDURES import;
                # if absent, skip name-based resolution.
                pass

        if hasattr(self._project, "loader"):
            sym = self._project.loader.find_symbol(addr)
            if sym and sym.name:
                try:
                    from angr.procedures import SIM_PROCEDURES

                    for lib_name, procs in SIM_PROCEDURES.items():
                        if sym.name in procs:
                            proc_class = procs[sym.name]
                            num_args = getattr(proc_class, "num_args", 0) or 0
                            no_ret = _proc_no_return(proc_class)
                            l.debug(f"Resolved symbol {sym.name} to {lib_name}:{sym.name}")
                            return (sym.name, num_args, no_ret)
                except ImportError:
                    # cat-(a) EXPECTED CONTROL FLOW: optional SIM_PROCEDURES import
                    # (symbol name path); same as above.
                    pass

        if hasattr(self._project, "loader"):
            obj = self._project.loader.find_object_containing(addr)
            if obj and obj.binary is not None:
                for section in obj.sections:
                    if (
                        section.is_executable
                        and section.min_addr <= addr < section.max_addr
                        and section.name not in (".plt", ".plt.got", ".plt.sec")
                    ):
                        l.debug(f"Internal function at 0x{addr:x} - returning pass-through")
                        return ("__internal_passthrough__", 0, False)

        l.debug(f"Could not resolve function at 0x{addr:x} (name={name})")
        return None

    def _cb_memory_store_batch(self, stores: list):
        state = self._get_per_fork_state()
        if state is None:
            return
        # angr-hcok: proxy gate — Rust already performed every store before
        # invoking the batch callback. Skip the redundant Python shadow.
        if _is_rust_memory_proxy(state.memory):
            return
        for addr, data in stores:
            try:
                if isinstance(data, (bytes, list)):
                    int_val = int.from_bytes(bytes(data), "little")
                    val = claripy.BVV(int_val, len(data) * 8)
                else:
                    val = claripy.BVV(data, 64)
                # Rust buffers a store as its value's little-endian bytes
                # (`bv_to_bytes`) and its own memory applies the arch endness,
                # so this store must too (angr-vmi2l.234).
                state.memory.store(addr, val, endness=state.arch.memory_endness)
            except (SimError, ClaripyError) as e:
                # cat-(c) WRONG-ANSWER RISK: batch memory store partially failed;
                # some addresses keep stale data. Logs at debug; promote upstream
                # if a divergence is observed.
                l.debug(f"Batch memory store failed at 0x{addr:x}: {e}")

    def _cb_memory_load_batch(self, loads: list) -> list:
        state = self._get_per_fork_state()
        if state is None:
            return [(bytes(size), False, None) for _, size in loads]

        # angr-hcok: proxy gate — generate filler BVSs per entry rather
        # than re-entering Rust through the proxy.
        if _is_rust_memory_proxy(state.memory):
            state_id = self._current_callback_state_id
            results = []
            for addr, size in loads:
                ast = claripy.BVS(f"mem_filler_{addr:x}_{size}", size * 8)
                self._register_handle(id(ast), ast, addr=addr, size=size, state_id=state_id)
                results.append((bytes(size), True, ast))
            return results

        results = []
        for addr, size in loads:
            try:
                val = state.memory.load(addr, size, endness=state.arch.memory_endness)
                is_sym = getattr(val, "symbolic", False)
                concrete = state.solver.eval(val).to_bytes(size, "little")
                if is_sym:
                    results.append((concrete, True, val))
                else:
                    results.append((concrete, False, None))
            except (SimError, ClaripyError) as e:
                # cat-(c) WRONG-ANSWER RISK: batch memory load partial failure;
                # returns zero bytes for that entry — Rust sees concrete zero.
                # Logs at debug; promote upstream if a divergence is observed.
                l.debug(f"Batch memory load failed at 0x{addr:x}: {e}")
                results.append((bytes(size), False, None))
        return results

    def _fetch_page_from_ultrapage(self, state, page_addr: int) -> tuple | None:
        """Serve one page fetch straight off the UltraPage side tables.

        Returns the ``(data, perms, is_mapped)`` triple Rust expects, or None
        when this state's memory backend isn't an UltraPage and the caller must
        fall back to ``memory.load`` + ``solver.eval``.

        angr-gorvf.4.5: the load-based path builds a 32768-bit AST for the whole
        page just to answer "is it symbolic?". Every page the ZeroPy FAIL benches
        still fetch is a lazy-stack page that IS symbolic (measured: every one on
        google2016_unbreakable_1) and therefore gets DECLINED right after — so
        the AST was pure GIL cost. Classify off ``symbolic_data`` /
        ``symbolic_bitmap`` / ``concrete_data`` instead, the same fast path
        ``RustStateSyncMixin._sync_extra_python_pages`` uses at setup:

        * page not materialized, or any byte still default-fill (bitmap bit set)
          → whatever Python would synthesize on read. Under
          ZERO_FILL_UNCONSTRAINED_MEMORY that is zeros, so serve a zero page;
          otherwise it is an unconstrained symbol, so DECLINE and let the
          per-load ``memory_load`` callback preserve the AST — exactly what the
          load-based path did, minus the AST.
        * explicit symbolic store on the page (``symbolic_data``) → DECLINE.
        * fully concrete → hand Rust ``concrete_data`` with no solver call.
        """
        mem_pages = getattr(state.memory, "_pages", None)
        if mem_pages is None:
            return None
        # Option constants are plain strings (see the module header note).
        zero_fill = "ZERO_FILL_UNCONSTRAINED_MEMORY" in state.options
        page_obj = mem_pages.get(page_addr // 4096)
        if page_obj is None:
            return (bytes(4096), 7, True) if zero_fill else (bytes(4096), 0, False)
        concrete_data = _ultrapage_bytes(page_obj)
        symbolic_bitmap = getattr(page_obj, "symbolic_bitmap", None)
        if concrete_data is None:
            return None  # non-UltraPage backend — caller uses the slow path
        if getattr(page_obj, "symbolic_data", None):
            return (bytes(4096), 0, False)
        if not zero_fill and (symbolic_bitmap is None or symbolic_bitmap.any_set(0, symbolic_bitmap.size)):
            # Uninitialized bytes on the page would default-fill symbolic.
            return (bytes(4096), 0, False)
        return (bytes(concrete_data), 7, True)

    def _fetch_page_bytes(self, state, page_addr: int) -> tuple:
        """The ``(data, perms, is_mapped)`` triple for one page of ``state``.

        Shared by ``_cb_fetch_page`` and ``_cb_batch_fetch_pages``; their
        callers own the error handling. ``data`` is the page in memory order,
        so the load is ``Iend_LE`` whatever the arch: only then do the
        evaluated int's little-endian bytes come out in address order. An
        arch-endness load hands a big-endian page to Rust reversed end to end
        (angr-vmi2l.234).
        """
        # angr-gorvf.4.5: UltraPage side tables answer this without
        # materializing a page-wide AST. See _fetch_page_from_ultrapage.
        fast = self._fetch_page_from_ultrapage(state, page_addr)
        if fast is not None:
            return fast
        data = state.memory.load(page_addr, 4096, endness="Iend_LE")
        if getattr(data, "symbolic", False):
            # angr-gorvf.4.3: a symbolic page is DECLINED (is_mapped=False)
            # — Rust keeps serving it through the per-load memory_load
            # callback, which preserves the AST. Concretizing it here would be
            # pure waste: a `solver.eval` of a 32768-bit symbolic AST (a full
            # Z3 solve; nearly all of the callback GIL on
            # google2016_unbreakable_1) whose result `fetch_pages_batch` then
            # throws away, since it skips every !is_mapped entry.
            if _DBG:
                l.debug(f"fetch_page 0x{page_addr:x}: has symbolic data, declining")
            return (bytes(4096), 0, False)
        return (state.solver.eval(data).to_bytes(4096, "little"), 7, True)

    def _cb_batch_fetch_pages(self, page_addrs: list) -> list:
        state = self._get_callback_state() or self._get_default_state()
        if state is None:
            # Match _cb_fetch_page: decline (is_mapped=False) so Rust falls
            # back to its own page source and re-triggers the callback later,
            # rather than permanently baking an all-zero page into Rust memory.
            return [(bytes(4096), 0, False) for _ in page_addrs]

        # angr-hcok: proxy gate — decline; Rust falls back to its own
        # zero-fill / static-binary page source.
        if _is_rust_memory_proxy(state.memory):
            return [(bytes(4096), 0, False) for _ in page_addrs]

        results = []
        for page_addr in page_addrs:
            try:
                results.append(self._fetch_page_bytes(state, page_addr))
            except (SimError, ClaripyError) as e:
                # cat-(b) FALLBACK WITH LOSS: batch page fetch partial failure.
                # Decline (is_mapped=False) so Rust marks the page inaccessible
                # and can retry via the per-load callback, rather than baking a
                # transient failure in as a permanent all-zero page (matches the
                # single-page _cb_fetch_page error arm).
                l.debug(f"Batch page fetch failed at 0x{page_addr:x}: {e}")
                results.append((bytes(4096), 0, False))
        return results

    def _cb_memory_store_symbolic_value(self, addr: int, ast):
        """Store a symbolic value (claripy AST) to Python state memory."""
        state = self._get_per_fork_state()
        if state is None or ast is None:
            return
        # angr-hcok: proxy gate — Rust already holds the symbolic AST in
        # its own memory; skip the Python shadow write.
        if _is_rust_memory_proxy(state.memory):
            return
        try:
            if hasattr(ast, "length") and ast.length:
                state.memory.store(addr, ast, endness=state.arch.memory_endness, inspect=False, disable_actions=True)
        except (SimError, ClaripyError) as e:
            # cat-(c) WRONG-ANSWER RISK: symbolic store at concrete addr failed;
            # Rust may serve stale concrete bytes from a prior store. Debug-
            # logs.
            l.debug(f"Symbolic store at 0x{addr:x} failed: {e}")

    def _cb_memory_store_symbolic_full(self, addr_ast, data_ast):
        """Store a symbolic value at a *symbolic* address.

        Called from Rust when address concretization yields TooLarge (or for the
        Multiple/StoreG fallback paths in interpreter/statements.rs). Python's
        memory model natively supports symbolic addresses via angr's
        SimConcretizationStrategy chain, which is the whole reason the *_full
        callback exists.

        Phase 1.4 (angr-5zw8): if ``addr_ast`` carries a
        ``MultiwriteAnnotation`` (attached by ``libc/strchr.py``,
        ``libc/memchr.py``, ``libc/gets.py``, ``libc/fgets.py``), route the
        store through the Rust Multi-cell lazy path on the current state
        before falling back to Python's memory model. On success the write
        lands directly in Rust memory and the Python state is skipped; on
        failure we fall through to the existing Python path so the write is
        not lost.
        """
        state = self._get_per_fork_state()
        if state is None or addr_ast is None or data_ast is None:
            return
        if self._try_multi_cell_store(addr_ast, data_ast):
            return
        # angr-hcok: proxy gate — a store through the proxy would re-enter
        # Rust under &mut self, and returning would drop the write. Rust
        # stores every enumerable candidate set itself, so only an address
        # it cannot resolve (TooLarge with no Max fallback, a Failed
        # concretization) gets here; fail loud, as Python's empty strategy
        # chain does (angr-vmi2l.324).
        if _is_rust_memory_proxy(state.memory):
            raise SimMemoryAddressError(f"Rust could not concretize the symbolic store address {addr_ast!r}")
        try:
            state.memory.store(
                addr_ast, data_ast, endness=state.arch.memory_endness, inspect=False, disable_actions=True
            )
        except (SimError, ClaripyError) as e:
            # cat-(c) WRONG-ANSWER RISK: symbolic-address store failed; Python
            # memory model could not apply the symbolic write. Debug-logs.
            l.debug(f"Symbolic-address store failed: {e}")

    def _try_multi_cell_store(self, addr_ast, data_ast) -> bool:
        """Phase 1.4 (angr-5zw8): route MultiwriteAnnotation-tagged stores
        to Rust's ``state_memory_store_symbolic_multi`` instead of Python.

        Returns True iff the store landed in Rust memory. Caller must fall
        back to the Python store path on False so writes are not lost.
        """
        try:
            has_anno = getattr(addr_ast, "has_annotation_type", None)
            if has_anno is None:
                return False
            from angr.storage.memory_mixins.address_concretization_mixin import (
                MultiwriteAnnotation,
            )

            if not has_anno(MultiwriteAnnotation):
                return False
        except MemoryError:
            raise
        except (ImportError, AttributeError, Exception) as e:
            # cat-(a) EXPECTED CONTROL FLOW: annotation introspection failed
            # (claripy API drift, exotic AST). Fall through to Python path.
            l.debug(f"MultiwriteAnnotation probe failed: {e}")
            return False

        sid = self._get_stepping_state_id() if self._get_stepping_state_id else None
        if sid is None:
            return False
        try:
            ok = self._rust_state_memory_store_symbolic_multi(sid, addr_ast, data_ast)
        except AttributeError:
            # cat-(a) EXPECTED CONTROL FLOW: older Rust build without the
            # Phase 1.4 entry point — keep the Python path.
            return False
        except MemoryError:
            raise
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: Rust side raised; debug-log and
            # let the Python fallback try.
            l.debug(f"state_memory_store_symbolic_multi raised: {e}")
            return False
        return bool(ok)

    def _rust_state_memory_store_symbolic_multi(self, state_id: int, addr_ast, data_ast) -> bool:
        """Thin Python wrapper around the read-only PyO3 method so tests can
        monkey-patch the routing without touching the Rust manager."""
        return self._rust_mgr.state_memory_store_symbolic_multi(state_id, addr_ast, data_ast)

    def _cb_memory_load_symbolic_full(self, addr_ast, size: int):
        """Load `size` bytes at a *symbolic* address, returning a claripy AST.

        Counterpart to `_cb_memory_store_symbolic_full`. Rust delegates here when
        a load address has a TooLarge solution range (expressions.rs:149); Python
        builds the appropriate ITE chain or memory access via angr's
        concretization strategies and returns the AST. Rust converts the AST
        back via claripy_to_rustbv (or a fresh sym_pyref_* placeholder if
        conversion fails).
        """
        state = self._get_per_fork_state()
        if state is None or addr_ast is None:
            return claripy.BVV(0, size * 8)
        # angr-hcok: proxy gate — synthesize a fresh BVS rather than
        # re-entering Rust via the proxy mid-run().
        if _is_rust_memory_proxy(state.memory):
            return claripy.BVS(f"sym_addr_load_{size}", size * 8, explicit_name=False)
        try:
            return state.memory.load(
                addr_ast, size, endness=state.arch.memory_endness, inspect=False, disable_actions=True
            )
        except (SimError, ClaripyError) as e:
            # cat-(c) WRONG-ANSWER RISK: symbolic-address load failed; returns
            # a fresh BVS that has no relationship to the actual symbolic value
            # in memory — downstream constraint sync will diverge. Debug-logs.
            # angr-ymoe: no fires across the fast-tier
            # benches. Path is dead in practice; counter acts as a watchdog.
            # `test_cb_memory_load_symbolic_full_swallows_sim_memory_error`
            # locks in the swallowing behavior (Rust upstream wraps the AST
            # in a sym_pyref_* placeholder via expressions.rs).
            l.debug(f"Symbolic-address load failed: {e}")
            self._stats_orphan_bvs_sym_load_full_fail += 1
            return claripy.BVS(f"sym_load_full_fail_{size}", size * 8, explicit_name=False)

    # ---- state.inspect MVP dispatcher (angr-uq4n.2) ----

    def _get_inspect_proxy(self):
        """Return the manager-wide RustInspectProxy (lazy)."""
        if self._inspect_proxy is None:
            from angr.exploration.rust_state_proxy import RustInspectProxy

            self._inspect_proxy = RustInspectProxy(self)
        return self._inspect_proxy

    def _update_inspect_bitmask(self):
        """Recompute the Rust-side `inspect_enabled` bitmask.

        Called after every BP add/remove. The bitmask is the OR of bits
        for every event with at least one registered breakpoint.
        Rust VEX dispatch sites read this with a single `& != 0` test
        before any payload work, so when no BPs exist the cost is one
        branch per Load/Store.
        """
        if self._callbacks is None or not hasattr(self._callbacks, "set_inspect_enabled"):
            return
        mask = 0
        for event, bit in self._INSPECT_EVENT_BITS.items():
            if self._inspect_breakpoints.get(event):
                mask |= 1 << bit
        self._callbacks.set_inspect_enabled(mask)
        # A BP action may read the stepping state, which the step holds, so
        # ask Rust for the mid-step view only while one is registered
        # (angr-mx00p.132.53).
        self._callbacks.set_inspect_wants_view(
            any(bp.action is not None for bps in self._inspect_breakpoints.values() for bp in bps)
        )

    def _seed_inspect_breakpoints(self, angr_state) -> None:
        """Adopt the breakpoints *angr_state*'s own ``SimInspector`` carries.

        ``SimulationManager`` steps the seed state itself, so BPs set with
        ``state.inspect.b(...)`` before ``factory.simgr(state)`` fire; the Rust
        registry is manager-wide, so without this they are silently dropped
        (angr-mx00p.132.52). ``rust_engine_eligible`` has already routed a
        state with an event Rust cannot dispatch to Python. Identity dedup
        keeps a re-added state (a found hit, a merge result, a copy sharing
        the same ``BP`` objects) from registering a BP twice.
        """
        plugins = getattr(angr_state, "plugins", None)
        inspector = plugins.get("inspect") if isinstance(plugins, dict) else None
        breakpoints = getattr(inspector, "_breakpoints", None)
        if not breakpoints or breakpoints is self._inspect_breakpoints:
            return
        from angr.exploration.rust_state_proxy import _format_unfired_phase_msg, _rust_unfired_phases

        added = False
        for event, bps in breakpoints.items():
            if not bps:
                continue
            registry = self._inspect_breakpoints.get(event)
            if registry is None:
                raise NotImplementedError(f"state.inspect event {event!r} is not dispatched by the rust engine")
            for bp in bps:
                unfired = _rust_unfired_phases(event, bp.when)
                if unfired:
                    raise NotImplementedError(_format_unfired_phase_msg(event, bp.when, unfired))
                if not any(bp is known for known in registry):
                    registry.append(bp)
                    added = True
        if added:
            self._update_inspect_bitmask()

    def _adopt_late_inspect_breakpoints(self, *, release_seeds: bool) -> None:
        """Adopt BPs set after construction on a seed state or a stash mirror.

        Under ``SimulationManager`` the stepped state is the one the caller
        holds, so ``state.inspect.b(...)`` on the seed after ``simgr(state)``,
        or on ``simgr.active[i]``, fires. Here those are a caller-owned seed
        and the Python mirror of a Rust state, neither of which Rust steps,
        so every entry that hands control to Rust re-reads their inspectors
        (``_invalidate_state_export_cache``) (angr-vmi2l.176). A seed stops
        being read once Rust steps (*release_seeds*): ``SimulationManager``
        steps its seed once, and a BP set on it later never fires. A removal
        is not mirrored: only ``RustInspectProxy.remove_breakpoint`` drops a BP.
        """
        live = [ref for ref in self._inspect_seed_states if ref() is not None]
        self._inspect_seed_states = [] if release_seeds else live
        for ref in live:
            self._seed_inspect_breakpoints(ref())
        for state in list(self._state_cache.values()):
            self._seed_inspect_breakpoints(state)

    def _make_inspect_state_for(self, state_id: int):
        """Build the state object the user action sees during inspect dispatch.

        Returns a RustStateProxy bound to `state_id` (preferred). Falls
        back to whatever the cached default state is if the state_id is
        unknown — keeps the BP firing rather than silently dropping.
        """
        if state_id is not None and state_id >= 0:
            from angr.exploration.rust_state_proxy import RustInspectViewStateProxy, RustStateProxy
            from angr.rustylib.vex_engine import inspect_view_addrs

            # A fire mid-step publishes the stepping state's view; its reads
            # must not touch the borrowed manager (angr-mx00p.132.53).
            proxy_cls = RustStateProxy if inspect_view_addrs() is None else RustInspectViewStateProxy
            # cat-(a) EXPECTED CONTROL FLOW: state may have been
            # dropped from Rust between dispatch and proxy build.
            with suppress_recoverable():
                return proxy_cls(self._rust_mgr, state_id, self._project, python_mgr=self)
        return self._get_callback_state() or self._get_default_state()

    def _dispatch_inspect_event(self, event_type: str, state_id: int, when: str, **attrs):
        """Generic dispatch path for state.inspect events.

        Builds a RustStateProxy for `state_id`, then forwards `attrs` as
        keyword arguments to `SimInspector.action` (via the manager-wide
        RustInspectProxy). Reentrancy is suppressed by
        `_inspect_dispatch_depth` to prevent a BP action that recursively
        triggers another event from clobbering in-flight attributes.

        Per-event-type kwargs are constructed by the `_cb_inspect_*`
        callbacks — this layer is event-agnostic.

        Returns the RustInspectProxy after dispatch so callers can read
        back attributes the user's BP action may have mutated (value
        injection, angr-uy32). Returns ``None`` when no BP fired (reentrancy
        guard, bitmask race, or no live state), in which case nothing was
        mutated and the caller should keep its original value.
        """
        if self._inspect_dispatch_depth > 0:
            return None  # reentrancy guard (uq4n.4)
        bps = self._inspect_breakpoints.get(event_type)
        if not bps:
            return None  # bitmask race — Rust fired but Python already cleared
        state = self._make_inspect_state_for(state_id)
        if state is None:
            return None
        proxy = self._get_inspect_proxy()
        proxy.set_state(state)
        self._inspect_dispatch_depth += 1
        try:
            proxy.action(event_type, when, **attrs)
        finally:
            self._inspect_dispatch_depth -= 1
        return proxy

    def _addr_attr_for(self, addr: int):
        """Wrap an integer address in a claripy BVV at the project's word size."""
        if not isinstance(addr, int):
            return addr
        bits = self._project.arch.bits if self._project is not None else 64
        return claripy.BVV(addr, bits)

    def _fire_inspect_event(
        self,
        event_type: str,
        state_id: int,
        when: str,
        *,
        _inject: str | None = None,
        _original=None,
        **attrs,
    ):
        """Shared body of every ``_cb_inspect_*`` callback (angr-12jjk.28).

        Forwards to :meth:`_dispatch_inspect_event` with `attrs` and
        contains the error handling every event callback used to
        hand-roll. Each `_cb_inspect_*` is then just its docstring plus the
        per-event attr names, which is the only thing that actually differs
        between them.

        When `_inject` names an attribute, the user's BP action is allowed
        to override that value: the attr is read back off the post-dispatch
        proxy and returned when the user swapped the object (identity
        check — a needless round-trip back to Rust costs an AST->RustBV
        conversion and its width checks on every untouched event). Returns
        ``None`` otherwise, meaning "caller keeps its original value".
        Fire-and-forget events pass no `_inject` and ignore the return.

        cat-(b) FALLBACK WITH LOSS: a raising user handler drops this one
        event (no breakpoint fired) and the engine keeps stepping; the
        exception type is open since handlers are user code. For an
        `_inject` event the drop additionally means the *original* value
        stands, so a half-applied override is never observable.
        """
        try:
            proxy = self._dispatch_inspect_event(event_type, state_id, when, **attrs)
            if proxy is not None:
                self._warn_ignored_inspect_writes(proxy, event_type, when, _inject, attrs)
            if _inject is not None and proxy is not None:
                mutated = getattr(proxy, _inject)
                if mutated is not _original:
                    return mutated
        except MemoryError:
            raise
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: the BP's effect (and any
            # injected value) is lost and execution goes on without it.
            l.warning("inspect %s dispatch failed: %s: %s", event_type, type(e).__name__, e)
        return None

    def _warn_ignored_inspect_writes(self, proxy, event_type: str, when: str, inject, attrs: dict) -> None:
        """Warn once per (event, attr) when a BP replaced an attr nobody reads back.

        Python angr reads back attrs such as ``reg_write_expr`` or
        ``expr_result`` after a BP fires; the Rust engine reads back only
        the one `inject` attr (rust_engine.rst "Read-only (mutations
        ignored)"), so any other replaced attr would otherwise drop with no
        signal (angr-vmi2l.206). Same identity check as the write-back: an
        in-place mutation of a staged object is not seen.
        """
        for name, original in attrs.items():
            if name == inject or getattr(proxy, name, original) is original:
                continue
            key = (event_type, name)
            if key in self._inspect_ignored_write_warned:
                continue
            self._inspect_ignored_write_warned.add(key)
            l.warning(
                "inspect %s (when=%s): the BP's write to %s is ignored by the Rust engine, "
                "which keeps the original value (this warning is shown once per event and attribute)",
                event_type,
                when,
                name,
            )

    def _cb_inspect_mem_read(
        self,
        state_id: int,
        when: str,
        addr: int,
        size: int,
        value_ast,
        endness: str,
        condition=None,
    ):
        """PyO3 callback target for mem_read events from Rust.

        Returns the possibly-mutated ``mem_read_expr`` AST when the user's
        BP action overrode it (value injection, angr-uy32); the Rust caller
        substitutes it for the loaded value. Returns ``None`` when unchanged,
        so the original load result stands. ``condition`` is a guarded
        load's 1-bit guard (``None`` for a plain load); it becomes
        ``mem_read_condition`` as ``guard == 1``, the Bool Python's VEX
        engine passes.
        """
        return self._fire_inspect_event(
            "mem_read",
            state_id,
            when,
            _inject="mem_read_expr",
            _original=value_ast,
            mem_read_address=self._addr_attr_for(addr),
            mem_read_length=size,
            mem_read_expr=value_ast,
            mem_read_endness=endness,
            mem_read_condition=None if condition is None else condition == 1,
        )

    def _cb_inspect_mem_write(
        self,
        state_id: int,
        when: str,
        addr: int,
        size: int,
        value_ast,
        endness: str,
        condition=None,
    ):
        """PyO3 callback target for mem_write events from Rust.

        Returns the possibly-mutated ``mem_write_expr`` AST when a BP_BEFORE
        action overrode it (value injection, angr-inh0); the Rust caller
        substitutes it for the stored value before commit. Returns ``None``
        when unchanged (and always for ``when='after'``, post-commit), so the
        original store value stands. ``condition`` is as for
        ``_cb_inspect_mem_read``.
        """
        return self._fire_inspect_event(
            "mem_write",
            state_id,
            when,
            _inject="mem_write_expr",
            _original=value_ast,
            mem_write_address=self._addr_attr_for(addr),
            mem_write_length=size,
            mem_write_expr=value_ast,
            mem_write_endness=endness,
            mem_write_condition=None if condition is None else condition == 1,
        )

    def _cb_inspect_reg_read(
        self,
        state_id: int,
        when: str,
        offset: int,
        size: int,
        value_ast,
    ):
        """PyO3 callback target for reg_read events from Rust (IRExpr::Get)."""
        self._fire_inspect_event(
            "reg_read",
            state_id,
            when,
            reg_read_offset=offset,
            reg_read_length=size,
            reg_read_expr=value_ast,
            reg_read_condition=None,
            reg_read_endness=None,
        )

    def _cb_inspect_reg_write(
        self,
        state_id: int,
        when: str,
        offset: int,
        size: int,
        value_ast,
    ):
        """PyO3 callback target for reg_write events from Rust (IRStmt::Put)."""
        self._fire_inspect_event(
            "reg_write",
            state_id,
            when,
            reg_write_offset=offset,
            reg_write_length=size,
            reg_write_expr=value_ast,
            reg_write_condition=None,
            reg_write_endness=None,
        )

    def _cb_inspect_instruction(self, state_id: int, when: str, addr: int):
        """PyO3 callback target for instruction events (one per IMark)."""
        self._fire_inspect_event("instruction", state_id, when, instruction=addr)

    def _cb_inspect_irsb(self, state_id: int, when: str, addr: int):
        """PyO3 callback target for irsb events (one per basic block)."""
        self._fire_inspect_event("irsb", state_id, when, address=addr)

    def _cb_inspect_exit(
        self,
        state_id: int,
        when: str,
        target: int,
        jumpkind: str,
        guard_ast,
    ):
        """PyO3 callback target for VEX conditional exit events."""
        self._fire_inspect_event(
            "exit",
            state_id,
            when,
            exit_target=self._addr_attr_for(target),
            exit_guard=guard_ast,
            exit_jumpkind=jumpkind,
        )

    def _cb_inspect_call(self, state_id: int, when: str, function_address: int):
        """PyO3 callback target for Ijk_Call events (function entry).

        Mirrors the `call` BP pair in Python's `CallStack._manage`
        (`state_plugins/callstack.py`): fires `when='before'` with
        the resolved call target, then `when='after'` once the Rust
        call_stack push has happened. `function_address` is wrapped in a
        word-sized BVV so user code may compare it with `state.regs._ip`
        which is also symbolic.
        """
        self._fire_inspect_event(
            "call",
            state_id,
            when,
            function_address=self._addr_attr_for(function_address),
        )

    def _cb_inspect_return(self, state_id: int, when: str, function_address: int):
        """PyO3 callback target for Ijk_Ret events (function exit).

        Mirrors the `return` BP pairs in Python's `CallStack._manage`
        (`state_plugins/callstack.py`): fires `when='before'` with
        the func_addr of the frame being popped, then `when='after'`.
        function_address is 0 when the Rust call stack is empty (popping
        from an unentered frame), matching the int convention.
        """
        self._fire_inspect_event(
            "return",
            state_id,
            when,
            function_address=self._addr_attr_for(function_address),
        )

    def _cb_inspect_tmp_read(self, state_id: int, when: str, tmp_num: int, value_ast):
        """PyO3 callback target for tmp_read events (VEX `RdTmp`).

        Fired `when='after'` once the tmp slot has been read. `value_ast`
        is the claripy reconstruction of the stored RustBV.
        """
        self._fire_inspect_event(
            "tmp_read",
            state_id,
            when,
            tmp_read_num=tmp_num,
            tmp_read_expr=value_ast,
        )

    def _cb_inspect_tmp_write(self, state_id: int, when: str, tmp_num: int, value_ast):
        """PyO3 callback target for tmp_write events (VEX `WrTmp`).

        Fired `when='after'` with the value about to be stored into the
        tmp. Note: the slot mutation happens after this callback returns,
        so user BP_AFTER overrides are not honored — same MVP gap as the
        Python-dispatched events documented in `rust_engine.rst`.
        """
        self._fire_inspect_event(
            "tmp_write",
            state_id,
            when,
            tmp_write_num=tmp_num,
            tmp_write_expr=value_ast,
        )

    def _cb_inspect_statement(self, state_id: int, when: str, stmt_idx: int):
        """PyO3 callback target for statement events (per VEX IR statement).

        Fired `when='before'` from `execute_block_with_callbacks` for each
        statement in the IRSB, with `stmt_idx` (the position in
        `irsb.statements`) as the only attr. Matches Python's
        `SimInspectMixin._handle_vex_stmt` BP_BEFORE attr signature.
        The BP_AFTER mirror is not wired — same MVP scope as `instruction`.
        """
        self._fire_inspect_event("statement", state_id, when, statement=stmt_idx)

    def _cb_inspect_expr(self, state_id: int, when: str, expr_result):
        """PyO3 callback target for expr events (per VEX IR expression eval).

        Fired `when='after'` from `eval_expr_with_callbacks` once the
        expression has been reduced to a value. `expr_result` is the
        claripy reconstruction of the computed RustBV; `expr` itself is
        passed as `None` because Rust IRExpr doesn't round-trip cleanly
        into a `pyvex.IRExpr`. User mutations to `expr_result` in BP
        actions are NOT honored — same MVP gap as the other inspect
        events.
        """
        self._fire_inspect_event("expr", state_id, when, expr=None, expr_result=expr_result)

    def _cb_inspect_address_concretization(
        self,
        state_id: int,
        when: str,
        action: str,
        addr_ast,
        result,
    ):
        """PyO3 callback target for address_concretization events.

        Fired BEFORE/AFTER around the Rust concretizer when a symbolic
        address gets concretized for a load or store. `action` is 'load'
        or 'store'; `addr_ast` is the claripy reconstruction of the
        symbolic address; `result` is the list of concrete addresses
        produced by the concretizer (None on BEFORE). The strategy,
        memory and add_constraints attrs are passed as None — the Rust
        engine doesn't surface those objects (MVP gap).
        """
        self._fire_inspect_event(
            "address_concretization",
            state_id,
            when,
            address_concretization_strategy=None,
            address_concretization_action=action,
            address_concretization_memory=None,
            address_concretization_expr=addr_ast,
            address_concretization_result=result,
            address_concretization_add_constraints=None,
        )

    def _cb_inspect_fork(self, state_id: int, when: str):
        """PyO3 callback target for fork events.

        Fired `when='after'` once per forked state created by the
        deferred-fork processing (`apply_core_outcome` in
        `exploration/stepping.rs`, `process_deferred_forks_into` in
        `exploration/stepping_forks.rs`). `state_id` is the FORKED state's
        id (matching Python's `SimSuccessors._preprocess_successor` in
        `engines/successors.py`, where the BP fires on the newly-added
        successor, not the parent). The fork event has no attrs in
        angr's `inspect_attributes` table; the BP just gets the per-state
        proxy via `state.inspect`. UNSAT-pruned forks still fire the BP
        (the Rust dispatch is pre-satisfiability check, matching
        Python's pre-discard fire).
        """
        self._fire_inspect_event("fork", state_id, when)

    def _cb_inspect_symbolic_variable(
        self,
        state_id: int,
        when: str,
        name: str,
        size: int,
        expr_ast,
    ):
        """PyO3 callback target for symbolic_variable events.

        Fired `when='after'` when the Rust engine mints a fresh BVS for
        an unconstrained memory load (load_from_callback fresh-symbol
        fallback). Attrs mirror Python `solver.py:432-439`:
        `symbolic_name`, `symbolic_size`, `symbolic_expr`. The
        user-callable `state.solver.BVS()` path still fires the same
        event from Python natively, independent of this dispatch.
        """
        self._fire_inspect_event(
            "symbolic_variable",
            state_id,
            when,
            symbolic_name=name,
            symbolic_size=size,
            symbolic_expr=expr_ast,
        )

    def _cb_inspect_simprocedure(
        self,
        state_id: int,
        when: str,
        sp_name,
        sp_addr,
        sp_inst,
        sp_result,
    ):
        """Python-side dispatch target for SimProcedure inspect events.

        Invoked from `_handle_simprocedure_callback` (in the dispatch
        mixin) wrapping the proc execution. Attrs mirror Python's
        `sim_procedure.py:246/314` — `simprocedure_name`,
        `simprocedure_addr`, `simprocedure` (the instance), and
        `simprocedure_result` (the procedure return value on AFTER;
        `NO_OVERRIDE` on BEFORE).
        """
        self._fire_inspect_event(
            "simprocedure",
            state_id,
            when,
            simprocedure_name=sp_name,
            simprocedure_addr=sp_addr,
            simprocedure=sp_inst,
            simprocedure_result=sp_result,
        )

    def _cb_inspect_syscall(
        self,
        state_id: int,
        when: str,
        syscall_name,
        sp_inst=None,
    ):
        """Python-side dispatch target for syscall inspect events.

        Invoked from `_handle_syscall_callback_inner` wrapping the syscall
        handler. Attrs mirror Python's `procedure.py:34/50` —
        `syscall_name`, and `simprocedure` (the executed instance) on
        AFTER. `simprocedure` is `None` on BEFORE (matches the Python
        engine, which does not pass it before execution).
        """
        self._fire_inspect_event(
            "syscall",
            state_id,
            when,
            syscall_name=syscall_name,
            simprocedure=sp_inst,
        )

    def _cb_inspect_dirty(
        self,
        state_id: int,
        when: str,
        dirty_name,
        dirty_handler,
        dirty_args,
        dirty_result,
    ):
        """Python-side dispatch target for VEX Dirty-call inspect events.

        Invoked from `_cb_dirty_call` wrapping the resolved dirty handler
        invocation. Attrs mirror Python's `engines/vex/heavy/inspect.py:10/22`
        — `dirty_name`, `dirty_handler` (the resolved handler callable),
        `dirty_args` (the actual call arguments — claripy BVVs), and
        `dirty_result` (the handler's return value on AFTER; `None` on
        BEFORE).

        Returns the possibly-mutated ``dirty_result`` when the user's BP
        action overrode it (short-circuit / value injection, angr-uy32);
        `_cb_dirty_call` substitutes it for the handler's result. Returns
        ``None`` when unchanged (or no BP fired), so the original result
        stands.
        """
        return self._fire_inspect_event(
            "dirty",
            state_id,
            when,
            _inject="dirty_result",
            _original=dirty_result,
            dirty_name=dirty_name,
            dirty_handler=dirty_handler,
            dirty_args=dirty_args,
            dirty_result=dirty_result,
        )

    def _cb_inspect_constraints(self, state_id: int, when: str, added_constraints=None):
        """Python-side dispatch target for constraints inspect events (angr-4aach).

        Invoked from ``RustSolverProxyPlugin.add`` around the write-through
        into the Rust state's solver. Mirrors Python's
        ``state_plugins/solver.py`` which fires ``constraints`` BP_BEFORE
        (with ``added_constraints``) then BP_AFTER around the solver add.

        On BP_BEFORE, returns the possibly-mutated ``added_constraints``
        list when the user's action replaced it (value injection); the
        proxy installs the returned list instead of the original. Returns
        ``None`` when unchanged or no BP fired, so the caller keeps its
        original list. BP_AFTER always returns ``None`` (post-install).
        """
        return self._fire_inspect_event(
            "constraints",
            state_id,
            when,
            # BP_AFTER runs post-install, so there is nothing left to inject.
            _inject="added_constraints" if when == "before" else None,
            _original=added_constraints,
            added_constraints=added_constraints,
        )

    def _cb_inspect_vex_lift(self, state_id: int, when: str, addr, size, buff=None):
        """Python-side dispatch target for vex_lift inspect events (angr-4aach).

        Invoked from ``_cb_lift_block`` — the Rust block-cache miss path —
        mirroring Python's ``engines/vex/lifter.py`` which fires ``vex_lift``
        only when its lifter cache is not used. Fires BP_BEFORE
        (``vex_lift_addr``, ``vex_lift_size=None``, ``vex_lift_buff``) before
        the lift and BP_AFTER (``vex_lift_addr``, ``vex_lift_size`` = the
        lifted IRSB byte size) after. Attributed to the callback/default
        state (``state_id`` is ``-1``): Rust block lifts are
        state-independent (shared block cache). User mutation of the attrs
        is not honored (MVP gap; the engine uses its own lift bytes).
        """
        # Unlike its siblings this one resolves the owning state before
        # dispatching, and that resolution can itself raise — so it keeps a
        # try/except of its own; the dispatch below still goes through the
        # shared `_fire_inspect_event` body.
        try:
            # Lifts are state-independent; when the caller has no owning
            # state_id (-1) attribute the event to a representative active
            # state so `state.inspect` routes through the RustInspectProxy
            # (a real-SimState fallback would not see the staged attrs).
            if state_id is None or state_id < 0:
                # The native-lift fire (angr-op0dn.14.4.2) arrives mid-step,
                # while the Rust manager is mutably borrowed — `get_state_ids`
                # would raise "Already mutably borrowed" there. The thread-local
                # stepping id names the state whose block is being lifted and
                # costs no borrow, so prefer it and keep the stash query as the
                # out-of-step fallback (`_cb_lift_block` called directly).
                stepping_id = self._get_stepping_state_id()
                if stepping_id is not None:
                    state_id = stepping_id
                else:
                    active_ids = self._rust_mgr.get_state_ids("active")
                    if active_ids:
                        state_id = active_ids[0]
        except MemoryError:
            raise
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: no state to attribute the lift to;
            # this vex_lift event is dropped (no breakpoint fired).
            l.warning("inspect vex_lift dispatch failed: %s: %s", type(e).__name__, e)
            return
        self._fire_inspect_event(
            "vex_lift",
            state_id,
            when,
            vex_lift_addr=self._addr_attr_for(addr),
            vex_lift_size=size,
            vex_lift_buff=buff,
        )

    def _load_binary_regions(self):
        """Load binary code regions for native lifting."""
        regions = []

        main_object = self._project.loader.main_object
        for obj in self._project.loader.all_objects:
            # Skip cle pseudo-objects (externs/tls/kernel) — their `binary`
            # is a synthetic string like 'cle##externs', not None, so the
            # plain None check below is not enough.
            #
            # `binary is None` also matches stream-backed real objects such as
            # the Blob produced by `load_shellcode` (and `Project(BytesIO(...),
            # backend='blob')`). Those carry the actual code, so we must NOT
            # skip the main object on a None binary — doing so left the Rust
            # engine with zero concrete regions, so every block resolved its
            # successor to 0x0 and deadended on the first step (angr-1lzq).
            # Non-main None-binary objects (kernel/tls pseudo-objects) stay
            # skipped to preserve the original behavior.
            if obj.binary is None and obj is not main_object:
                continue
            if isinstance(obj.binary, str) and obj.binary.startswith("cle##"):
                continue

            # Prefer sections; fall back to segments for loaders (e.g. Blob)
            # that don't expose sections. Without the segment fallback,
            # `is_in_binary` returns false for in-bounds branches and the
            # interpreter mistakes them for unmodeled calls.
            executable_ranges = [s for s in obj.sections if s.is_executable]
            if not executable_ranges:
                executable_ranges = [s for s in obj.segments if s.is_executable]

            for region in executable_ranges:
                try:
                    # cle's Region.max_addr is INCLUSIVE (last valid byte), so the
                    # byte count is max_addr - min_addr + 1. Omitting the +1 drops
                    # the final byte of every executable region; harmless on real
                    # binaries (trailing padding) but fatal for tiny blobs, where
                    # it truncated a 5-byte load_shellcode to 4 bytes and left the
                    # native lifter with a half-decoded instruction (angr-1lzq).
                    size = region.max_addr - region.min_addr + 1
                    data = self._project.loader.memory.load(region.min_addr, size)
                    regions.append((region.min_addr, bytes(data)))
                except MemoryError:
                    raise
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: executable region not loaded into Rust;
                    # attempts to lift a block in this region will fall through to
                    # Python via lift_block. Debug-logs.
                    name = getattr(region, "name", repr(region))
                    l.debug(f"Could not load region {name}: {e}")

        self._rust_mgr.load_binary_regions(regions)
        # Every loaded object, pseudo-objects included: a transfer outside all
        # of them into code the state holds is executed (angr-vmi2l.158).
        # cle's max_addr is inclusive; the Rust side wants half-open ranges.
        self._rust_mgr.set_loader_object_ranges(
            [(obj.min_addr, obj.max_addr + 1) for obj in self._project.loader.all_objects]
        )

        # Main-object span for the native-dispatch gate: hooks landing inside the
        # main object are user `proj.hook()` territory and always defer to Python,
        # while `use_sim_procedures` hooks inside a *loaded library* may prefer the
        # native registry when `prefer_native_library_hooks` is on (angr-a8epx).
        # cle's max_addr is inclusive; the Rust side wants a half-open range.
        try:
            self._rust_mgr.set_main_object_range(main_object.min_addr, main_object.max_addr + 1)
        except AttributeError:
            l.debug("Main object exposes no min_addr/max_addr; native-dispatch gate stays default")
        self._rust_mgr.set_prefer_native_library_hooks(self._prefer_native_library_hooks)

    def _register_simprocedures(self):
        """Register the project's SimProcedures with the Rust manager."""
        if hasattr(self._project, "_sim_procedures"):
            self._register_hooks(self._project._sim_procedures.items())

    def _extract_continuation_data(self, state: angr.SimState):
        """Extract SimProcedure continuation data from a state's callstack.

        When __libc_start_main uses self.call() to invoke main(), it stores
        procedure_data (local_vars) on the callstack frame. When main() returns,
        the continuation (after_main) needs these args. This method captures
        that data so the Rust engine can restore it when the continuation fires.
        """
        frame = state.callstack.top if hasattr(state, "callstack") else None
        while frame is not None:
            pdata = getattr(frame, "procedure_data", None)
            if pdata is not None and len(pdata) >= 5:
                cont_addr = pdata[4]  # ideal_addr = continuation address
                try:
                    cont_addr_int = int(cont_addr)
                except (TypeError, ValueError):
                    # cat-(a) EXPECTED CONTROL FLOW: continuation addr is symbolic /
                    # non-castable; walk to the next frame.
                    frame = getattr(frame, "next", None)
                    continue
                if cont_addr_int > 0:
                    self._pending_procedure_data[cont_addr_int] = pdata
                    l.debug(
                        f"Extracted continuation data for 0x{cont_addr_int:x} "
                        f"({len(pdata[2]) if len(pdata) > 2 and pdata[2] else 0} local_vars)"
                    )
            frame = getattr(frame, "next", None)

    def _run_python_init_if_needed(self, state: angr.SimState) -> list[tuple[str, angr.SimState]]:
        """Run initialization in Python if the state starts at a loader address.

        When a state starts at a loader/init address (e.g., from full_init_state),
        the C++ init sequence (constructors, .init_array, etc.) is too complex for
        the Rust engine. Run it in Python first, then return every state it
        produced as ``(stash, state)`` pairs (see :meth:`_step_python_to_main`).
        A seed that needs no init comes back alone, as ``[("active", state)]``.

        Only a real init seed qualifies: the entry of a binary-format main
        object, or the simos loader stub (``full_init_state``). A shellcode
        blob has no init, and a user hook or extern SimProcedure seed is
        ordinary code: pre-stepping either runs the program in Python, and
        when that pre-run ends without reaching main it is discarded and Rust
        runs it again, calling every user hook twice (angr-vmi2l.291).
        """
        main_obj = self._project.loader.main_object
        addr = state.addr
        if not is_python_init_seed(self._project, addr):
            return [("active", state)]
        l.info(f"State at init seed 0x{addr:x}, running Python init to main")

        try:
            main_addr = resolve_main_address(self._project)
            return self._step_python_to_main(state, main_addr, main_obj)
        except MemoryError:
            raise
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: Python init step failed; original
            # state is used unmodified, so init-side-effects (constructors,
            # .init_array) may not have run. Already warns.
            l.warning(f"Python init failed: {e}, using original state")
            return [("active", state)]

    def _step_python_to_main(
        self, state: angr.SimState, main_addr: int | None, main_obj
    ) -> list[tuple[str, angr.SimState]]:
        """Run Python SimulationManager until every path reaches main or ends.

        Returns ``(stash, state)`` pairs for every state the prefix produced:
        each fork that reached main (it stops there), each fork still running
        when the step budget ran out, and every deadended, unconstrained and
        errored state. Keeping only the first arrival silently dropped the
        other forks of a branching init (angr-vmi2l.291).
        """
        # Init-only addresses we never want to land on as "main"
        init_addrs = {self._project.entry}
        for obj in self._project.loader.all_objects:
            if hasattr(obj, "entry") and obj.entry:
                init_addrs.add(obj.entry)

        # Use the REAL SimulationManager (not the monkey-patched factory)
        # to avoid infinite recursion when the factory is patched.
        from angr import SimulationManager

        sm = SimulationManager(project=self._project, active_states=[state])
        main_min = main_obj.min_addr
        main_max = main_obj.max_addr

        def at_main(s: angr.SimState, step: int) -> bool:
            if main_addr is not None:
                return s.addr == main_addr
            # No main symbol: a state inside the main binary that isn't at
            # _start, isn't at a SimProcedure, and is past the prologue.
            return (
                step > 10
                and main_min <= s.addr <= main_max
                and s.addr not in init_addrs
                and s.addr not in self._project._sim_procedures
            )

        def spins(s: angr.SimState) -> bool:
            # A block whose only effect is jumping to itself (`jmp $`) never
            # changes the state again, so stepping it to the budget is waste.
            if self._project.is_hooked(s.addr):
                return False
            try:
                irsb = self._project.factory.block(s.addr).vex
            except (SimEngineError, PyVEXError):
                return False
            return (
                irsb.jumpkind == "Ijk_Boring"
                and isinstance(irsb.next, pyvex.expr.Const)
                and irsb.next.con.value == s.addr
                and all(isinstance(stmt, pyvex.stmt.IMark) for stmt in irsb.statements)
            )

        for step in range(500):
            sm.move(from_stash="active", to_stash="at_main", filter_func=lambda s, step=step: at_main(s, step))
            sm.move(from_stash="active", to_stash="spinning", filter_func=spins)
            if not sm.active:
                break
            sm.step()

        arrived = sm.stashes.get("at_main", [])
        if arrived:
            l.info(f"Python init complete: {len(arrived)} state(s) reached main at 0x{arrived[0].addr:x}")
        if sm.active:
            l.warning(f"Python init: {len(sm.active)} state(s) didn't reach main after 500 steps, using them as-is")
        if sm.stashes.get("spinning"):
            l.info(f"Python init: {len(sm.spinning)} state(s) spin on a self-jump, carrying them as-is")
            sm.move(from_stash="spinning", to_stash="active")
        # Step-capped forks go first: a LIFO policy (`Lifo::select` pops the
        # back) then descends the arrivals before a fork that may spin
        # forever in init, e.g. r100's ptrace anti-debug `jmp $`.
        carried = [("active", s) for s in sm.active + arrived]
        for _, s in carried:
            self._extract_continuation_data(s)
        carried += [("deadended", s) for s in sm.deadended]
        carried += [("unconstrained", s) for s in sm.unconstrained]
        carried += [("errored", rec.state) for rec in sm.errored]
        if not carried:
            # Every path was pruned (e.g. an unsat seed). An unstepped
            # SimulationManager would still hold the seed, so keep it.
            l.warning("Python init: no state survived the prefix, using original state")
            return [("active", state)]
        return carried

    def _warn_rejected_options(self, options) -> None:
        """Emit a one-time UserWarning per silently-divergent SimOption.

        See ``_REJECTED_OPTION_NAMES`` for the rationale. Some entries
        (notably ``TRACK_CONSTRAINT_ACTIONS`` and ``TRACK_MEMORY_MAPPING``)
        ship in the default ``symbolic`` mode bundle, so we warn once per
        option per manager rather than raise.
        """
        if not options:
            return
        # state.options is a SimStateOptions container (not a plain set), so
        # `intersection(options)` would fail on Python's set protocol — it
        # tries to look up each frozenset member as a state option, which
        # raises on names like "0". Iterate the small constant set instead.
        unseen = {
            name for name in _REJECTED_OPTION_NAMES if name in options and name not in self._warned_rejected_options
        }
        if not unseen:
            return
        for name in sorted(unseen):
            warnings.warn(
                f"SimOption {name!r} is set on a state owned by "
                "RustExplorationManager, but the Rust engine does not honor "
                "it. Behavior may diverge from the Python engine. See "
                "docs/advanced-topics/rust_engine.rst for the full matrix.",
                UserWarning,
                stacklevel=3,
            )
        self._warned_rejected_options.update(unseen)

    def _check_raise_options(self, options) -> None:
        """Raise NotImplementedError on a SimOption setting Rust cannot provide.

        Both polarities (see :func:`rust_unsupported_options`): a
        ``_RAISE_OPTION_NAMES`` member that is set — action streams, eager
        concretization, conservative-write refusal, ret-emulation, calless
        short-circuits, symbolic register fill, … — or a
        ``_REQUIRED_OPTION_NAMES`` member that is not. Silent divergence has
        burned users in the past, so we hard-fail at the manager boundary to
        force a drop to the Python engine.
        """
        offending = rust_unsupported_options(options)
        if not offending:
            return
        names = ", ".join(offending)
        raise NotImplementedError(
            f"SimOption setting(s) {{{names}}} request behavior the Rust "
            "engine cannot provide. Drop engine='rust' (or fix these "
            "options on state.options) and rerun with the Python "
            "engine. See docs/advanced-topics/rust_engine.rst for the "
            "full matrix."
        )

    # Maximum concretized file size (bytes) eligible for the Rust-side
    # symbolic-content export (the v1 scope gate; angr-0xyq2). One RustBV
    # per byte, so this bounds both export time and per-state memory.
    _FS_EXPORT_MAX_FILE_SIZE = 65536

    def _seed_stdin_to_rust(self, angr_state: angr.SimState, state_id: int) -> None:
        """Push a harness-filled ``posix.stdin.content`` into Rust's fd 0 (angr-mb09c).

        A harness that seeds stdin itself --- ``state.posix.stdin.content.append((BVS, n))``
        on a blank_state, or ``entry_state(stdin=SimFileStream(content=BVS))`` --- expects
        the found state to solve for THAT symbol. Without this push the native
        ``read(0, ...)`` mints its own ``stdin_*`` bytes and the harness's BVS stays
        unconstrained: ``solver.eval(bvs)`` returns zeros and ``posix.dumps(0)`` shows a
        bogus zero prefix ahead of the injected bytes. Attaching the same claripy byte
        ASTs as fd 0's symbolic content makes the Rust path condition reference the
        harness's own symbol (the bridge import preserves BVS identity), so both queries
        answer correctly off the Rust solver fallback.

        Best-effort: any stream shape we cannot flatten to whole bytes is skipped and
        Rust keeps the fresh-symbol behavior.
        """
        try:
            stdin = getattr(getattr(angr_state, "posix", None), "stdin", None)
            content = getattr(stdin, "content", None)
            asts = []
            consumed = 0
            # Only a SimFile can end: a SimPackets stream mints a fresh packet
            # past its content, as Rust does past a frontier seed.
            has_end = False
            if isinstance(content, list) and content:
                # Packets before ``pos`` were already read (a state Python
                # stepped past a stdin read; angr-mx00p.89).
                read_packets = stdin.pos if isinstance(getattr(stdin, "pos", None), int) else 0
                for i, (data, _size) in enumerate(content):
                    if not isinstance(data, claripy.ast.BV) or data.length % 8:
                        return
                    asts.extend(data.chop(8))  # chop(8)[0] is the byte at stream offset 0
                    if i < read_packets:
                        consumed += data.length // 8
            elif content is None and hasattr(stdin, "load") and hasattr(stdin, "size"):
                # angr-23n3r: ``blank_state(stdin=BVS)`` makes a file-backed
                # SimFileStream with no packet list; read it whole while its
                # size is concrete (the same bytes Python's read would serve).
                size = getattr(stdin, "size", None)
                if size is None or getattr(size, "symbolic", True):
                    return
                n = angr_state.solver.eval(size)
                has_end = getattr(stdin, "has_end", None) is True
                # An empty stream that ends is still pushed, so a native
                # reader sees the EOF (angr-vmi2l.110).
                if n <= 0 and not has_end:
                    return
                if n > 0:
                    data = stdin.load(0, n)
                    if not isinstance(data, claripy.ast.BV) or data.length % 8:
                        return
                    asts.extend(data.chop(8))
            else:
                return
            if (not asts and not has_end) or len(asts) > self._FS_EXPORT_MAX_FILE_SIZE:
                return
            self._rust_mgr.seed_stdin_content(state_id, asts, consumed, has_end)
        except MemoryError:
            raise
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: stdin seeding failed; native read(0)
            # mints fresh symbols, so the harness's BVS stays unconstrained on
            # exported states. Debug-logs.
            l.debug("stdin seed push failed for state %d: %s: %s", state_id, type(e).__name__, e)

    # Caps for the initial-environment bridge (`_seed_environ_to_rust`): at most
    # this many envp slots, and this many bytes per "KEY=VALUE" entry. Both are
    # far above any realistic `entry_state(env=...)`; they exist so a corrupt or
    # symbolic envp pointer cannot turn state-add into an unbounded memory walk.
    _MAX_ENVIRON_ENTRIES = 512
    _MAX_ENVIRON_ENTRY_LEN = 4096
    # Bytes per chunked cstring load. Typical "KEY=VALUE" fits in one.
    _ENVIRON_SCAN_CHUNK = 64

    def _read_environ_cstring(self, angr_state: angr.SimState, addr: int) -> bytes | None:
        """Read one NUL-terminated concrete ``KEY=VALUE`` entry from ``addr``.

        Returns ``None`` when a symbolic byte is hit at or before the NUL, or
        when no NUL appears within :attr:`_MAX_ENVIRON_ENTRY_LEN` --- neither
        shape can be represented in Rust's byte-keyed environment map, so the
        entry is skipped rather than guessed at. Bytes *after* the NUL are
        never inspected, so a chunk that overruns the string into unmapped or
        symbolic memory is harmless.
        """
        out = bytearray()
        while len(out) < self._MAX_ENVIRON_ENTRY_LEN:
            n = min(self._ENVIRON_SCAN_CHUNK, self._MAX_ENVIRON_ENTRY_LEN - len(out))
            chunk = angr_state.memory.load(addr + len(out), n, endness="Iend_BE")
            for b in chunk.chop(8):  # chop(8)[0] is the byte at the lowest address
                if b.symbolic:
                    return None
                v = angr_state.solver.eval(b)
                if v == 0:
                    return bytes(out)
                out.append(v)
        return None

    # The native procedures that read or write ``RustSimState``'s environment
    # map (``native/angr-symex-native/src/procedures/getenv.rs``). They go off together: a
    # native setenv whose value only the map holds would be invisible to a
    # Python getenv.
    _NATIVE_ENV_PROCS = ("getenv", "setenv", "unsetenv", "clearenv", "putenv")

    def disable_native_procedure(self, name: str) -> None:
        """Serve every hook dispatched as ``name`` from its Python SimProcedure.

        The supported opt-out from the native procedure of that name; a hook
        whose SimProcedure is not stock angr already runs on Python
        (``_native_dispatch_eligible``).
        """
        self._rust_mgr.disable_native_procedure(name)

    def _disable_native_env_procs(self, state_id: int, reason: str) -> None:
        """Serve the env family from Python for this manager: the map cannot hold this state's envp."""
        l.debug("state %d: %s; the env procedures run on Python", state_id, reason)
        for name in self._NATIVE_ENV_PROCS:
            self._rust_mgr.disable_native_procedure(name)

    def _seed_environ_to_rust(self, angr_state: angr.SimState, state_id: int) -> None:
        """Bridge the guest's initial ``entry_state(env=...)`` into Rust (angr-6cp06.12).

        angr models the initial environment purely in memory: ``simos/linux.py``
        dumps a ``KEY=VALUE`` string table onto the stack and stores the envp
        array pointer as ``state.posix.environ``; the Python ``getenv``
        SimProcedure walks that array. Rust's native ``getenv``/``setenv``/
        ``putenv`` instead read ``RustSimState``'s byte-keyed environment map,
        which starts empty and is only ever written by those same native procs.
        Without this push a guest ``getenv("FLAG")`` on an
        ``entry_state(env={"FLAG": "1"})`` state returns NULL under the Rust
        engine where Python finds the value --- silently taking the wrong branch
        --- and native ``setenv(..., overwrite=0)`` clobbers such a var because
        its existence check misses too.

        A symbolic envp pointer or slot, or an entry with a symbolic byte, has
        no place in the byte-keyed map, and a native ``getenv`` would return
        NULL where Python returns the symbolic value (angr-mx00p.132.2). Such a
        state turns the whole native env family off for this manager
        (:meth:`_disable_native_env_procs`), so Python's procedures, which read
        envp, serve every call. An entry with no ``=`` is not a variable at all
        and is skipped. A state with no modeled environment at all
        (``posix.environ is None``) turns the family off the same way.
        """
        try:
            posix = getattr(angr_state, "posix", None)
            if posix is None:
                return
            envp = posix.environ
            if envp is None:
                # No modeled environment (``blank_state``/``call_state``):
                # Python's getenv mints a ``getenv__<name>`` symbol keeping
                # both the NULL and non-NULL branches, which the byte-keyed
                # map cannot express (angr-vmi2l.124).
                self._disable_native_env_procs(state_id, "no modeled environment (posix.environ is None)")
                return
            if isinstance(envp, claripy.ast.Base):
                if envp.symbolic:
                    self._disable_native_env_procs(state_id, "symbolic envp pointer")
                    return
                envp = angr_state.solver.eval(envp)
            envp = int(envp)
            if envp == 0:
                return

            ptr_size = angr_state.arch.bytes
            entries: list[tuple[bytes, bytes, int]] = []
            for i in range(self._MAX_ENVIRON_ENTRIES):
                slot = angr_state.memory.load(envp + i * ptr_size, ptr_size, endness=angr_state.arch.memory_endness)
                if slot.symbolic:
                    self._disable_native_env_procs(state_id, f"symbolic envp[{i}]")
                    return
                ptr = angr_state.solver.eval(slot)
                if ptr == 0:
                    break
                line = self._read_environ_cstring(angr_state, ptr)
                if line is None:
                    self._disable_native_env_procs(state_id, f"envp[{i}] has a symbolic or unterminated entry")
                    return
                key, sep, value = line.partition(b"=")
                if not sep or not key:
                    continue
                # The value's own address, so native getenv returns a
                # pointer into this string as Python's getenv does.
                entries.append((key, value, ptr + len(key) + 1))
            else:
                # No NULL terminator inside the cap — treat the array as
                # unrecognizable rather than seeding a truncated environment.
                self._disable_native_env_procs(state_id, "envp has no NULL terminator")
                return

            if entries:
                self._rust_mgr.seed_environment(state_id, entries)
        except MemoryError:
            raise
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: initial-environment push failed; the
            # Rust environment map stays empty, so native getenv returns NULL
            # for vars the harness passed via entry_state(env=...). Debug-logs.
            l.debug("environ seed push failed for state %d: %s: %s", state_id, type(e).__name__, e)

    def _export_fs_files_to_rust(self, angr_state: angr.SimState, state_id: int) -> None:
        """Export eligible ``state.fs._files`` entries into the Rust
        FileSystem's path-keyed symbolic-content registry, so a native
        guest ``open()`` attaches the content and reads are served in Rust
        instead of bouncing to Python (angr-0xyq2 Phase 3).

        Scope gate (v1) — a file is exported iff ALL of:

        * ``type(simfile) is SimFile`` exactly (subclasses like
          ``SimFileStream`` / ``SimPackets`` have different position/EOF
          models),
        * ``simfile.has_end is True`` (bounded-EOF model only),
        * ``simfile.seekable`` is truthy,
        * ``simfile.file_exists is True`` (the native ``open()`` model
          cannot express Python's ``If(file_exists, fd, -1)`` return),
        * ``simfile.endness == "Iend_BE"`` (an LE file's Python read
          window is address-reversed per read, so no fixed byte order
          matches it — review-verified),
        * the size concretizes uniquely (``solver.eval_one``) to
          ``0 < size <= _FS_EXPORT_MAX_FILE_SIZE``,
        * the path decodes as UTF-8 (the Rust path model is UTF-8-lossy).

        Anything else is skipped silently — those files keep today's
        Python-fallback behavior. Already-open ``state.posix.fd`` entries
        are deliberately NOT synced (v1): the Rust and Python fd tables are
        unsynced by design (angr-8j16), so only path-keyed content that a
        future native ``open()`` attaches is pushed.

        Known v1 limitation: this also runs on mid-run re-adds (legacy
        Python-fork push, ``merge()``, cross-manager transfer), where it
        re-registers content on a path an ancestor's native write had
        demoted to Python ownership. That serves the same bytes Python
        would (the refused write never landed in either model — the guest
        saw ``-1``), so it re-arms the documented write-demotion
        limitation rather than corrupting content. ``merge()`` now closes
        this gap (angr-qluof): it queries each source lineage's demoted
        paths via ``get_demoted_paths`` and re-applies them with
        ``demote_file_path`` after the re-add. The legacy-fork-push and
        cross-manager-transfer re-add sites are not yet covered.
        """
        try:
            from angr.storage.file import SimFile

            fs = angr_state.fs
            # Python ``_files`` keys are cwd-normalized (default
            # ``/home/user``) while the Rust FileSystem cwd defaults to
            # ``/`` — push the cwd unconditionally (even with no files) so
            # native getcwd/relative-path normalization matches Python
            # regardless of whether any SimFile was inserted. cwd is
            # always bytes on the Python side.
            self._rust_mgr.set_fs_cwd(state_id, fs.cwd)
            files = fs._files
        except MemoryError:
            raise
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: fs export preamble failed (non-UTF-8
            # cwd, missing fs plugin, FFI error); every file in this state
            # stays Python-served (the pre-Phase-3 behavior). Debug-logs.
            self._stats_symfile_export_skips["preamble"] += 1
            l.debug("fs file export skipped for state %d: %s: %s", state_id, type(e).__name__, e)
            return

        exported = 0
        for path, simfile in files.items():
            try:
                # v1 scope gate — see docstring. Exact-type check on purpose.
                # Each rejection bumps a per-reason counter (angr-4ref8) so a
                # file dropping out of native serving is attributable.
                if type(simfile) is not SimFile:
                    self._stats_symfile_export_skips["subclass"] += 1
                    continue
                if simfile.has_end is not True:
                    self._stats_symfile_export_skips["has_end"] += 1
                    continue
                if not simfile.seekable:
                    self._stats_symfile_export_skips["not_seekable"] += 1
                    continue
                if simfile.file_exists is not True:
                    self._stats_symfile_export_skips["file_exists"] += 1
                    continue
                if simfile.endness != "Iend_BE":
                    self._stats_symfile_export_skips["endness"] += 1
                    continue
                try:
                    size = angr_state.solver.eval_one(simfile.size)
                except SimSolverError:
                    # symbolic size without a unique value (or unsat)
                    self._stats_symfile_export_skips["size"] += 1
                    continue
                if not 0 < size <= self._FS_EXPORT_MAX_FILE_SIZE:
                    self._stats_symfile_export_skips["size"] += 1
                    continue
                try:
                    path_str = path.decode("utf-8")
                except UnicodeDecodeError:
                    # the Rust path model is UTF-8-lossy — a non-UTF-8 path
                    # can't be keyed natively.
                    self._stats_symfile_export_skips["path_utf8"] += 1
                    continue
                # Big-endian load: chop(8)[0] is the byte at file offset 0.
                data = simfile.load(0, size, disable_actions=True, inspect=False)
                self._rust_mgr.register_file_content(state_id, path_str, data.chop(8))
                exported += 1
            except MemoryError:
                raise
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: this file stays Python-served
                # (the pre-Phase-3 behavior). Debug-logs.
                self._stats_symfile_export_skips["error"] += 1
                l.debug("fs file export failed for %r: %s: %s", path, type(e).__name__, e)
        self._stats_symfile_exports += exported
        if exported:
            l.debug("Exported %d symbolic file(s) to Rust state %d", exported, state_id)

    @staticmethod
    def _push_python_call_frames(angr_state: angr.SimState, rust_state) -> None:
        """Push ``angr_state.callstack``'s concrete frames onto ``rust_state``, outermost first."""
        if not angr_state.has_plugin("callstack"):
            return
        frames = []
        for frame in angr_state.callstack:
            if not frame.func_addr and not frame.ret_addr:
                continue  # the bottom sentinel (None or 0 fields)
            sp = frame.stack_ptr
            sp = sp.concrete_value if isinstance(sp, claripy.ast.BV) and not sp.symbolic else sp
            if not all(isinstance(v, int) for v in (frame.call_site_addr, frame.func_addr, frame.ret_addr, sp)):
                return  # a symbolic or missing field: keep Rust's stack empty rather than partial
            frames.append((frame.call_site_addr, frame.func_addr, frame.ret_addr, sp))
        for call_site, callee, ret_addr, sp in reversed(frames):
            rust_state.push_call_frame(call_site, callee, ret_addr, sp)

    @staticmethod
    def _seed_python_jumpkind(angr_state: angr.SimState, rust_state) -> None:
        """Record the jumpkind that led to ``angr_state`` as ``rust_state``'s last history entry.

        An export reads ``history.jumpkind`` back from that entry, and analyses
        key on it (Veritesting starts its CFG from ``(addr, jumpkind)``), so a
        state Python stepped and handed over must not come back as
        ``Ijk_Boring`` (angr-mx00p.89). A state that has not stepped has no
        jumpkind to carry.
        """
        history = angr_state.history
        jumpkind = history.jumpkind
        if not jumpkind or history.depth == 0:
            return
        code = _HISTORY_JUMPKIND_CODES.get(jumpkind)
        if code is None:
            code = 3 if jumpkind.startswith("Ijk_Sys") else 4
        src = history.addr
        rust_state.add_detailed_history(src if isinstance(src, int) else 0, code, angr_state.addr)
        if jumpkind.startswith("Ijk_Sys"):
            # Python stepped up to the syscall exit; the syscall itself is the
            # next step's work, so Rust must run it before the block at pc
            # (angr-mx00p.132.12). Its pc is the syscall's return address,
            # not the kernel stub ``SimSuccessors`` pointed ip at.
            syscall = _pending_syscall_resume(angr_state)
            if syscall is not None:
                rust_state.pc = syscall[0]
            rust_state.set_pending_syscall(jumpkind)

    def _add_rust_state(self, stash: str, angr_state: angr.SimState):
        """Add an angr state to a Rust stash.

        Note: Rust internally forks the state, so we need to get the actual
        state ID from Rust after adding to properly cache the angr state.

        Returns the resolved Rust state id (the id-diff result, or the
        Python-side fallback id when the diff was inconclusive) so callers
        like ``merge()`` can post-process the freshly added state
        (angr-qluof lineage-aware demotion).
        """
        # Operate on a private copy so this never mutates the caller-owned
        # SimState (angr-hv4lt.7). _concretize_stack_registers pins a symbolic
        # sp/bp to one concrete value and adds a permanent sp-equality
        # constraint; doing that in place silently corrupts the symbolic
        # identity + constraint set of any object the caller still holds a
        # reference to (e.g. a hybrid Python-vs-Rust comparison, or reusing the
        # seed across two manager constructions). The copy becomes the cached
        # Python mirror below, so Rust still sees the concretized+constrained
        # view. SimStateScratch.copy() drops the cross-manager transfer attrs
        # (rust_mgr / rust_found_state_id set by RustSolverFallback.attach),
        # so carry them over explicitly — the Z3-pointer transfer path below
        # reads them.
        caller_state = angr_state
        angr_state = caller_state.copy()
        for _scratch_attr in ("rust_mgr", "rust_found_state_id"):
            if hasattr(caller_state.scratch, _scratch_attr):
                setattr(angr_state.scratch, _scratch_attr, getattr(caller_state.scratch, _scratch_attr))
        # An exported state (this manager's or another's) is a live view of a
        # Rust state: the sync below reads Python plugins, so give the copy
        # its own first. Its solver then carries the cross-manager transfer
        # attributes for the state it read (``RustSolverFallback.attach``).
        binding = binding_of(angr_state)
        if binding is not None:
            exporter = getattr(binding.owner, "manager", None)
            if exporter is None:
                raise SimStateError(f"cannot add {caller_state!r}: it is a view of a Rust state whose manager is gone")
            exporter._detach_export_proxies(angr_state, binding.state_id)

        self._seed_inspect_breakpoints(caller_state)
        if not any(ref() is caller_state for ref in self._inspect_seed_states):
            self._inspect_seed_states.append(weakref.ref(caller_state))

        # Concretize stack-relative registers for Rust compatibility
        self._concretize_stack_registers(angr_state)

        # Create Rust state from angr state
        is_le = self._project.arch.memory_endness == "Iend_LE"
        rust_state = _RustSimState(self._project.arch.name, little_endian=is_le)

        # angr-3ms1 step 1b: push the manager-wide opt-in for fork-time
        # SharedLineageSolver materialization onto this seed state's
        # solver context. Descendants inherit via SymContext::fork, so
        # this single call propagates to every state derived from this
        # seed. Default off keeps slice-1c's gate inert.
        if self._use_shared_lineage_solver:
            rust_state.set_use_shared_lineage_solver(True)

        # Set PC
        rust_state.pc = angr_state.addr

        # Carry the Python call stack (the frames of the init prefix Python ran
        # before handing over, e.g. __libc_start_main -> main) so Rust's frames
        # extend it rather than start empty (angr-mx00p.91). CallStack iterates
        # top-first; Rust's call stack is push-ordered. The bottom sentinel
        # frame is not a call.
        self._push_python_call_frames(angr_state, rust_state)
        # Rust frames have no procedure_data: keep the continuations the
        # prefix armed (angr-mx00p.136.32).
        self._extract_continuation_data(angr_state)
        self._seed_python_jumpkind(angr_state, rust_state)

        # Push posix.brk so Rust's NativeBrkSyscall starts from the correct
        # base. The angr loader sets state.posix.brk to (binary last_addr +
        # page) — distinct from Rust's hardcoded default 0x1B00000 — so
        # without this push a brk syscall in Rust would compare against the
        # wrong base and either overlap mapped memory or hand out an address
        # the program can't use. Only push if it's still a plain int; once
        # Python's set_brk has wrapped it as a BV, we don't try to flatten.
        try:
            py_brk = getattr(getattr(angr_state, "posix", None), "brk", None)
            if isinstance(py_brk, int):
                rust_state.posix_brk = py_brk
        except MemoryError:
            raise
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: posix.brk push to Rust failed; Rust's
            # brk syscall will base-from its hardcoded default and may overlap
            # mapped memory. Debug-logs.
            l.debug("posix.brk init push failed: %s", e)

        # Push posix.pid / ppid so the native getpid/gettid/getppid return a
        # user-set value, not SimSystemPosix's defaults (angr-vmi2l.149).
        posix = getattr(angr_state, "posix", None)
        pid = getattr(posix, "pid", None)
        ppid = getattr(posix, "ppid", None)
        if isinstance(pid, int) and isinstance(ppid, int):
            rust_state.posix_pids = (pid, ppid)

        # Push state.cgc's allocator so the native CGC allocate starts where
        # a Python-side allocate left it (angr-vmi2l.332). A plugin that was
        # never materialized holds the defaults Rust already has.
        cgc = angr_state.plugins.get("cgc")
        if cgc is not None:
            base = cgc.allocation_base
            if not isinstance(base, int) and not angr_state.solver.symbolic(base):
                base = angr_state.solver.eval_one(base)
            if isinstance(base, int) and all(isinstance(v, int) for hole in cgc.sinkholes for v in hole):
                rust_state.cgc_allocator = (base, list(cgc.sinkholes))

        # Push state.heap.heap_location so Rust's malloc bump allocator
        # (heap_alloc) starts past any Python-side allocation. Symmetric with
        # the posix.brk push above and the export-side
        # _sync_rust_heap_brk_to_state: a Python fallback SimProcedure that
        # mallocs bumps heap_location, and without this push a subsequent
        # native alloc in Rust would hand out an overlapping address
        # (angr-um39j). Only push a plain int that's past Rust's default base.
        # heap.mmap_base rides along so a user-set base places native addr=0
        # mmaps where Python's would (angr-vmi2l.150); the plugin default
        # equals Rust's.
        try:
            heap = getattr(angr_state, "heap", None)
            py_loc = getattr(heap, "heap_location", None)
            if isinstance(py_loc, int):
                rust_state.heap_brk = py_loc
            py_mmap = getattr(heap, "mmap_base", None)
            if isinstance(py_mmap, int):
                rust_state.mmap_base = py_mmap
        except MemoryError:
            raise
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: heap_location push to Rust failed;
            # Rust's native allocator bases from its default and may overlap a
            # Python-allocated region. Debug-logs.
            l.debug("heap.heap_location / mmap_base init push failed: %s", e)

        # Push the locale ctype table pointers so the native __ctype_b_loc /
        # __ctype_tolower_loc / __ctype_toupper_loc procs can return them
        # without a Python round-trip. Python's __libc_start_main init pass
        # (which runs before Rust takes over) mallocs + fills the tables in
        # shared memory and records the pointers on state.libc; we forward
        # just the pointer values. Only set when the init pass actually ran
        # (a plain int) — a blank_state entry leaves them None and the native
        # proc falls back to Python.
        try:
            libc = getattr(angr_state, "libc", None)
            if libc is not None:
                for attr, setter in (
                    ("ctype_b_loc_table_ptr", "ctype_b_loc_table_ptr"),
                    ("ctype_tolower_loc_table_ptr", "ctype_tolower_loc_table_ptr"),
                    ("ctype_toupper_loc_table_ptr", "ctype_toupper_loc_table_ptr"),
                ):
                    val = getattr(libc, attr, None)
                    if isinstance(val, int):
                        setattr(rust_state, setter, val)
        except MemoryError:
            raise
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: ctype table-ptr push failed; the
            # native __ctype_*_loc procs will defer to Python. Debug-logs.
            l.debug("libc ctype table-ptr init push failed: %s", e)

        # Push the loader-resolved guest addresses of the getopt(3) extern
        # globals (optind/optarg/optopt) so the native getopt proc
        # can write the cursor/optarg/optopt back to guest
        # memory the way real getopt does. Python is the only side with
        # loader.find_symbol, so this is a Python->native init-push (mirrors
        # the ctype table-ptr channel above, not the Rust->Python posix_brk
        # sync). Each symbol may be absent (statically linked away, or a
        # blank_state with no real loader pass) — then we skip it and the
        # native proc defers to Python for that global.
        try:
            loader = getattr(self._project, "loader", None)
            if loader is not None:
                for name, setter in (
                    ("optind", "getopt_optind_addr"),
                    ("optarg", "getopt_optarg_addr"),
                    ("optopt", "getopt_optopt_addr"),
                ):
                    sym = loader.find_symbol(name)
                    if sym is not None:
                        setattr(rust_state, setter, sym.rebased_addr)
        except MemoryError:
            raise
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: getopt extern-addr push failed; the
            # native getopt proc will defer to Python. Debug-logs.
            l.debug("getopt extern-addr init push failed: %s", e)

        # Push glibc getopt's scan state kept on state.libc, so a native getopt
        # resumes a scan Python started (angr-vmi2l.123). The outbound half is
        # ``_inject_rust_getopt``.
        try:
            libc = angr_state.plugins.get("libc")
            if libc is not None:
                rust_state.getopt_cursor = (
                    libc.getopt_optind,
                    libc.getopt_nextchar,
                    libc.getopt_first_nonopt,
                    libc.getopt_last_nonopt,
                )
        except MemoryError:
            raise
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: getopt cursor push failed (a cursor
            # field outside u32); native getopt starts a fresh scan. Debug-logs.
            l.debug("getopt cursor init push failed: %s", e)

        # Sync registers
        _t_reg = time.perf_counter_ns()
        self._sync_registers_to_rust(angr_state, rust_state)
        self._perf_stats.add_init_phase("register_sync", time.perf_counter_ns() - _t_reg)

        # Map memory regions
        _t_mem = time.perf_counter_ns()
        self._sync_memory_to_rust(angr_state, rust_state)
        self._perf_stats.add_init_phase("memory_sync", time.perf_counter_ns() - _t_mem)

        # Mirror angr's STRICT_PAGE_ACCESS and ENABLE_NX: when set on the
        # SimState, the Rust memory model rejects loads/stores that violate
        # per-page R/W bits (STRICT_PAGE_ACCESS) and instruction fetches from
        # non-X pages (ENABLE_NX, which Python additionally gates on
        # STRICT_PAGE_ACCESS — see angr/engines/vex/heavy/heavy.py:115-124).
        # add_state forks the state internally; both flags are preserved
        # through forks (see SymbolicMemory::fork in native/angr-symex-core/src/memory/mod.rs).
        if hasattr(angr_state, "options"):
            try:
                from angr import sim_options as o

                if o.STRICT_PAGE_ACCESS in angr_state.options:
                    rust_state.set_enforce_permissions(True)
                if o.ENABLE_NX in angr_state.options:
                    rust_state.set_enforce_nx(True)
                if o.NO_IP_CONCRETIZATION in angr_state.options:
                    rust_state.set_no_ip_concretization(True)
                if o.NO_SYMBOLIC_JUMP_RESOLUTION in angr_state.options:
                    rust_state.set_no_symbolic_jump_resolution(True)
                if o.KEEP_IP_SYMBOLIC in angr_state.options:
                    rust_state.set_keep_ip_symbolic(True)
                # angr-kzjv6: thread the symex-relevant SimOption subset onto
                # the Rust state so native SimProcedures can branch on them
                # (e.g. SHORT_READS gates faithful fgets short-read/EOF). Only
                # options a native proc actually consults are mirrored; the full
                # option set stays Python-side (rust_state_proxy.options).
                # Mirrored in BOTH directions: a Rust state seeds
                # SYMBOLIC_INITIAL_VALUES ON (stock-angr default), so absence
                # on the SimState has to be pushed across as an explicit
                # clear, not merely "not set".
                for _opt in _NATIVE_SIMOPTIONS:
                    rust_state.set_option(_opt, _opt in angr_state.options)
            except MemoryError:
                raise
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: option detection failed; Rust
                # permission/NX enforcement and IP-handling gating stay off —
                # accesses/jumps that Python would handle differently are
                # silently allowed/concretized. Debug-logs.
                l.debug(
                    "STRICT_PAGE_ACCESS / ENABLE_NX / NO_IP_CONCRETIZATION / "
                    "NO_SYMBOLIC_JUMP_RESOLUTION / "
                    f"KEEP_IP_SYMBOLIC detection failed: {e}"
                )

            self._check_raise_options(angr_state.options)
            self._warn_rejected_options(angr_state.options)

        # Seed harness-registered symlinks (angr-m7s7y) so native
        # readlink/readlinkat can resolve them. Forks inherit via the Rust
        # FileSystem Arc clone, so only the seed state needs the push.
        for link, target in getattr(self, "_pending_symlinks", {}).items():
            try:
                rust_state.register_symlink(link, target)
            except MemoryError:
                raise
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: symlink push failed; native
                # readlink for this link returns -1 (the pre-6.2 behavior).
                l.debug("symlink seed push failed for %r: %s", link, e)

        # Python lifts a patched project's code from the stepping state's
        # memory (VEXLifter._load_bytes); a dirtied page makes the Rust lift
        # read it too (angr-vmi2l.75).
        for patch in self._project.kb.patches.values():
            rust_state.mark_code_pages_dirtied(patch.addr, len(patch))

        # Get state IDs before adding (to find the new one)
        ids_before = set(self._rust_mgr.get_state_ids(stash))

        # Add to Rust manager. add_state copies the manager's zero-fill switch onto the
        # state's own memory, so set it from this state alone (angr-vmi2l.241).
        self._rust_mgr.set_zero_fill_unconstrained("ZERO_FILL_UNCONSTRAINED_MEMORY" in angr_state.options)
        self._rust_mgr.add_state(stash, rust_state)

        # Get state IDs after adding to find the newly added state ID
        ids_after = set(self._rust_mgr.get_state_ids(stash))
        new_ids = ids_after - ids_before

        # Cache the angr state with the actual Rust state ID
        if new_ids:
            actual_state_id = new_ids.pop()

            # Install constraints into the new Rust state with a single call
            # (angr-tvpk). Two paths,
            # mutually exclusive:
            #   - Cross-manager transfer (state came from a previous
            #     RustExplorationManager): copy Z3 ASTs by pointer from the
            #     old solver.
            #     Lossless and avoids a claripy round-trip.
            #   - Fresh Python state: install angr_state.solver.constraints.
            # Removed: the claripy round-trip via export_state_constraints
            # (bidirectional cycle) and the post-Z3 Python re-sync (second
            # installer call). Under the write-through model, Rust owns the
            # constraint set after this point; subsequent Python-side adds
            # route through RustSolverFallback's replay path or through
            # SimSolver write-through (angr-8oiw), not through re-installing
            # constraints at init.
            old_rust_mgr = getattr(angr_state.scratch, "rust_mgr", None)
            old_state_id = getattr(angr_state.scratch, "rust_found_state_id", None)
            installed = False
            if old_rust_mgr is not None and old_state_id is not None:
                try:
                    z3_ptrs = old_rust_mgr.export_z3_constraint_ptrs(old_state_id)
                    if z3_ptrs:
                        sat = self._rust_mgr.import_z3_constraint_ptrs(actual_state_id, z3_ptrs)
                        l.debug(
                            f"Transferred {len(z3_ptrs)} Z3 constraints from previous "
                            f"Rust manager (state {old_state_id}), sat={sat}"
                        )
                        installed = True
                    # The mirror cached below now stands for actual_state_id:
                    # point its solver fallback and the scratch transfer
                    # attributes at this core, or they keep the source core
                    # alive and answer for the source state (angr-f4ldp).
                    self._attach_rust_solver_fallback(angr_state, actual_state_id)
                except MemoryError:
                    raise
                except (AttributeError, Exception) as e:
                    # cat-(b) FALLBACK WITH LOSS: Z3 pointer transfer from old manager
                    # failed; falls through to Python-side install below, which on a
                    # state from a previous Rust manager typically has 0 constraints
                    # (RustSolverFallback keeps them on the old Rust solver). Debug-logs.
                    l.debug(f"Z3 pointer transfer failed: {e}")

            if not installed:
                py_constraints = None
                if hasattr(angr_state, "solver") and angr_state.solver.constraints:
                    py_constraints = list(angr_state.solver.constraints)
                if py_constraints:
                    try:
                        sat = self._rust_mgr.add_constraints_to_state(actual_state_id, py_constraints)
                        l.debug(
                            f"Installed {len(py_constraints)} Python constraints to Rust state "
                            f"{actual_state_id}, sat={sat}"
                        )
                    except MemoryError:
                        raise
                    except Exception as e:
                        # cat-(c) WRONG-ANSWER RISK: constraint install to Rust failed; the
                        # new state has fewer constraints than the source state — eval /
                        # satisfiable on it may produce wrong values. Already warns.
                        l.warning(f"Failed to install initial constraints: {e}")

            # Export eligible symbolic files from state.fs into the Rust
            # path-keyed content registry so native open()/read() serve them
            # without a Python bounce (angr-0xyq2 Phase 3). After the
            # constraint install: the bridge import preserves BVS identity,
            # so content bytes referenced by installed constraints resolve
            # to the same Rust symbols.
            self._export_fs_files_to_rust(angr_state, actual_state_id)

            # Same channel for a harness-seeded posix.stdin (angr-mb09c): after
            # the constraint install, so bytes referenced by installed
            # constraints resolve to the same Rust symbols.
            self._seed_stdin_to_rust(angr_state, actual_state_id)

            # Bridge entry_state(env=...) into Rust's environment map so the
            # native getenv/setenv/putenv family sees the guest's initial
            # environment, not just vars set at runtime (angr-6cp06.12).
            self._seed_environ_to_rust(angr_state, actual_state_id)

            # The time() watermark a Python time() left, so a native time()
            # stays monotonic against it (angr-vmi2l.130).
            last_time = angr_state.globals.get(_SYS_LAST_TIME) if "globals" in angr_state.plugins else None
            if last_time is not None:
                self._rust_mgr.set_state_last_time(actual_state_id, last_time)

            # Lineage-aware demotion (angr-qluof pt2): on a cross-manager
            # transfer the export above re-arms any symbolic-file path the
            # source Rust manager's lineage had demoted. Re-apply from the old
            # manager's state so the guest keeps seeing the Python fallback.
            self._reapply_demoted_paths(old_rust_mgr, old_state_id, actual_state_id)

            self._state_cache[actual_state_id] = angr_state
            # Track this as a root state for plugin restoration
            self._state_roots[actual_state_id] = actual_state_id
            self._record_history_anchor(actual_state_id, angr_state)
            # Seed Python-side options/globals from the source SimState so the
            # proxy returns the user-supplied values rather than empty stand-ins.
            # SimStateOptions is a dict-backed custom mapping that doesn't iterate
            # like a set (set(state.options) raises SimStateOptionsError on
            # numeric keys); pull names whose boolean switch is True directly
            # from the underlying _options dict.
            try:
                src_opts = getattr(angr_state, "options", None)
                inner = getattr(src_opts, "_options", None)
                if isinstance(inner, dict):
                    self._py_state_options[actual_state_id] = {name for name, value in inner.items() if value is True}
            except MemoryError:
                raise
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: seeding _py_state_options from source
                # state failed; child uses an empty options set on first access.
                # Debug-logs.
                l.debug("seed py_state_options(sid=%d) failed: %s: %s", actual_state_id, type(e).__name__, e)
            try:
                if "globals" in getattr(angr_state, "plugins", {}):
                    self._py_state_globals[actual_state_id] = dict(angr_state.globals)
                    self._note_x86_cr0(angr_state.globals)
            except MemoryError:
                raise
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: seeding _py_state_globals from source
                # state failed; child sees an empty globals dict on first access.
                # Debug-logs.
                l.debug("seed py_state_globals(sid=%d) failed: %s: %s", actual_state_id, type(e).__name__, e)
            # Extract and cache symbolic memory regions for preservation
            # This ensures symbolic values survive Rust<->Python transitions
            symbolic_pages = self._extract_symbolic_pages(angr_state)
            if symbolic_pages:
                self._rust_mgr.set_state_symbolic_pages(actual_state_id, symbolic_pages)
                l.debug(f"Cached {len(symbolic_pages)} symbolic pages for state {actual_state_id}")
                # Also import symbolic regions to Rust's symbolic memory so the
                # Rust engine can handle them natively without Python callbacks
                imported_sym = 0
                for addr, ast in symbolic_pages.items():
                    try:
                        import_ast = claripy.Reverse(ast) if hasattr(ast, "length") and ast.length > 8 else ast
                        self._rust_mgr.import_symbolic_to_state(actual_state_id, addr, import_ast)
                        imported_sym += 1
                        self._register_memory_order_ast(ast, addr, actual_state_id, angr_state.arch)
                    except MemoryError:
                        raise
                    except Exception as e:
                        # cat-(c) WRONG-ANSWER RISK: symbolic page import to Rust failed;
                        # Rust sees only the concrete-witness bytes for this page, losing
                        # the symbolic relationship. Debug-logs.
                        l.debug(f"Symbolic page import at 0x{addr:x} failed: {e}")
                if imported_sym:
                    l.debug(f"Imported {imported_sym} symbolic page entries to Rust state {actual_state_id}")
            # Import pending symbolic values to Rust SymbolicMemory
            if hasattr(self, "_pending_symbolic_imports") and self._pending_symbolic_imports:
                imported = 0
                for addr, ast in self._pending_symbolic_imports:
                    try:
                        # Byte-reverse multi-byte symbolic values before importing to Rust.
                        # Wide values are loaded with Iend_BE (preserving original BVS identity).
                        # Rust's memory model uses LE byte extraction internally, so we
                        # apply Reverse() to match.
                        import_ast = claripy.Reverse(ast) if hasattr(ast, "length") and ast.length > 8 else ast
                        self._rust_mgr.import_symbolic_to_state(actual_state_id, addr, import_ast)
                        imported += 1
                        self._register_memory_order_ast(ast, addr, actual_state_id, angr_state.arch)
                    except MemoryError:
                        raise
                    except Exception as e:
                        # cat-(c) WRONG-ANSWER RISK: pending symbolic import to Rust failed;
                        # the symbolic value is not visible to Rust — downstream loads see
                        # concrete witnesses only. Debug-logs.
                        l.debug(f"Symbolic import at 0x{addr:x} failed: {e}")
                if imported:
                    l.debug(f"Imported {imported} symbolic values to Rust state {actual_state_id}")
                self._pending_symbolic_imports = []

            # Enforce state cache limit
            self._cleanup_state_cache()
            l.debug(f"Cached angr state with Rust state ID {actual_state_id}")
            return actual_state_id
        # Fallback: cache with the Python-side state ID
        self._state_cache[rust_state.state_id] = angr_state
        # Track this as a root state for plugin restoration
        self._state_roots[rust_state.state_id] = rust_state.state_id
        self._record_history_anchor(rust_state.state_id, angr_state)
        symbolic_pages = self._extract_symbolic_pages(angr_state)
        if symbolic_pages:
            self._rust_mgr.set_state_symbolic_pages(rust_state.state_id, symbolic_pages)
            # Import symbolic regions to Rust's symbolic memory
            for addr, ast in symbolic_pages.items():
                try:
                    import_ast = claripy.Reverse(ast) if hasattr(ast, "length") and ast.length > 8 else ast
                    self._rust_mgr.import_symbolic_to_state(rust_state.state_id, addr, import_ast)
                    self._register_memory_order_ast(ast, addr, rust_state.state_id, angr_state.arch)
                except (TypeError, ValueError, RuntimeError):
                    # cat-(c) WRONG-ANSWER RISK: same as 2199 but on the fallback path
                    # where actual_state_id was not determined; Python-side state ID is
                    # used. Debug-logs (with exc_info).
                    l.debug("Failed to import symbolic region at 0x%x", addr, exc_info=True)
        # Enforce state cache limit
        self._cleanup_state_cache()
        l.warning(f"Could not determine actual Rust state ID, using Python-side ID {rust_state.state_id}")
        return rust_state.state_id

    def _reapply_demoted_paths(self, source_mgr, source_sid, target_sid) -> None:
        """Re-apply a single source lineage's symbolic-file demotions on a
        freshly re-added state (angr-qluof lineage-aware demotion, pt2).

        ``_add_rust_state``'s ``_export_fs_files_to_rust`` re-registers
        eligible SimFiles, re-arming any path an ancestor's native write had
        demoted (content-harmless — the refused write never landed — but it
        reverts the guest's write→-1). Querying the source state's demoted
        paths and re-applying them via ``demote_file_path`` keeps the guest
        seeing the content-identical Python fallback.

        ``source_mgr`` is a *native* manager (``self._rust_mgr`` for the
        legacy-fork-push site, or ``scratch.rust_mgr`` for the cross-manager
        transfer site — a different RustExplorationManager's native handle).
        Demoted paths are plain cwd-normalized strings, so they port across
        managers. This is the single-source counterpart to ``merge()``'s
        union-across-lineages re-demotion (which must query before its
        _merge_drop, so it can't share this helper).
        """
        if source_mgr is None or source_sid is None or target_sid is None:
            return
        try:
            demoted = source_mgr.get_demoted_paths(source_sid)
        except MemoryError:
            raise
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: a failed query just means the
            # re-added state may
            # re-arm a native write-demotion (the documented, content-harmless
            # v1 limitation).
            l.debug("get_demoted_paths(%s) failed: %s: %s", source_sid, type(e).__name__, e)
            return
        for path in demoted:
            try:
                if self._rust_mgr.demote_file_path(target_sid, path):
                    self._stats_symfile_redemotions += 1
            except MemoryError:
                raise
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: the state may re-arm a native
                # write-demotion, the content-harmless v1 limitation.
                l.debug("re-demote %r on state %s failed: %s: %s", path, target_sid, type(e).__name__, e)

    # Field descriptor types for table-driven serialization:
    #   'val'  — copy attribute value directly (int, str)
    #   'str'  — convert attribute to string via str()
    #   'expr' — serialize as VEX expression (recursive)
    #   'exprs'— serialize list of VEX expressions
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

    def _dispatch_callback(self, event) -> bool:
        """Dispatch a need_callback event to the appropriate handler.

        Returns True if exploration should stop (unknown callback reason).
        """
        self._stats_callback_count += 1
        _cb_start = time.perf_counter_ns()
        reason = event.callback_reason
        if reason == "simprocedure":
            self._handle_simprocedure_callback(event)
        elif reason == "syscall":
            self._handle_syscall_callback(event)
        elif reason == "symbolic_branch":
            self._handle_symbolic_branch_callback(event)
        elif reason == "find_predicate":
            self._handle_find_predicate_callback(event)
        elif reason == "avoid_predicate":
            self._handle_avoid_predicate_callback(event)
        elif reason == "python_vex_fallback":
            self._handle_python_vex_fallback(event)
        else:
            l.warning(f"Unknown callback reason: {reason}")
            self._stats_time_in_callbacks_ns += time.perf_counter_ns() - _cb_start
            return True
        # Bound the cache between callbacks (angr-qm7w). Pre-fix the cache
        # tracked O(active_states); now it is tied to the LRU window of
        # recently-mutated states + pinned roots / current dispatcher.
        self._cleanup_state_cache()
        self._stats_time_in_callbacks_ns += time.perf_counter_ns() - _cb_start
        return False

    def _check_limits(self, start_time, steps_taken, timeout, max_steps) -> bool:
        """Check if timeout or max_steps limits have been reached."""
        self._fire_progress_if_due(start_time, steps_taken)

        if timeout is not None and (time.time() - start_time) > timeout:
            l.warning(f"Exploration timeout reached ({timeout}s)")
            return True
        if max_steps is not None and steps_taken >= max_steps:
            l.warning(f"Max exploration steps reached ({max_steps})")
            return True
        return False

    def _fire_progress_if_due(self, start_time, steps_taken) -> None:
        """Fire the progress callback when `interval_steps` steps have elapsed.

        Called both at the top of an exploration iteration (via
        ``_check_limits``) and right after a batch's steps are counted: a run
        that finds its target inside a single native batch breaks out of the
        loop without a second limit check, so without the post-batch call the
        callback would never fire at all (angr-gorvf.15).
        """
        cb = getattr(self, "_progress_callback", None)
        if cb is not None:
            interval = getattr(self, "_progress_interval", 100)
            last = getattr(self, "_progress_last_fired", 0)
            if steps_taken - last >= interval:
                self._progress_last_fired = steps_taken
                counts = self._rust_mgr.stash_counts()
                # cat-(b) FALLBACK WITH LOSS: progress callback raised; suppress so
                # user code can't break exploration. The callback's view skips this
                # tick.
                with suppress_recoverable():
                    cb(
                        {
                            "step_count": steps_taken,
                            "active_count": counts.get("active", 0),
                            "found_count": counts.get("found", 0),
                            "deadended_count": counts.get("deadended", 0),
                            "elapsed_seconds": time.time() - start_time,
                        }
                    )

    def set_progress_callback(self, callback: Callable, interval_steps: int = 100) -> None:
        """Set a progress callback that fires every `interval_steps` steps.

        The callback receives a dict with:
            step_count, active_count, found_count, deadended_count, elapsed_seconds.

        Args:
            callback: Callable that receives the progress dict.
            interval_steps: How often to fire (default: every 100 steps).
        """
        self._progress_callback = callback
        self._progress_interval = interval_steps
        self._progress_last_fired = 0

    def set_exploration_strategy(
        self,
        strategy: str,
        seed: int = 0,
        distances: dict[int, int] | None = None,
        beam_width: int = 2,
    ):
        """Set exploration strategy: 'bfs' (default), 'dfs', 'random', 'coverage', 'loop_head', 'directed', or 'find_directed'.

        Args:
            strategy: 'bfs' (FIFO), 'dfs' (LIFO), 'random' (uniformly-random
                active-state selection), 'coverage' (new-block-first),
                'loop_head' (round-robin over (loop-head, callstack-class)
                buckets), 'directed' (CFG-distance beam search), or
                'find_directed' (novelty/CFG-distance find-first dispatch under
                num_find=1). All non-default strategies
                are opt-in only.
            seed: SplitMix64 seed for 'random' — fixes the selection stream so a
                run is reproducible. Ignored for the other strategies.
            distances: required for 'directed' and 'find_directed' — an
                ``addr -> distance-to-target`` snapshot computed once from the
                angr CFG (e.g. via :func:`cfg_distance_map`; pass the find address
                as the target for 'find_directed'). Shipped into Rust as immutable
                metadata; unmapped blocks are treated as unreachable.
            beam_width: for 'directed', the number of closest states stepped as a
                beam (default 2). ``beam_width == 1`` is greedy best-first and
                traps on data-dependent targets (see the Rust
                ``DirectedCfgDistance`` docs); ``>= 2`` recovers.
        """
        strategy = strategy.lower()
        if strategy == "dfs":
            self._rust_mgr.set_state_selection_lifo()
        elif strategy == "bfs":
            self._rust_mgr.set_state_selection_fifo()
        elif strategy == "random":
            self._rust_mgr.set_state_selection_random(int(seed) & 0xFFFFFFFFFFFFFFFF)
        elif strategy == "coverage":
            self._rust_mgr.set_state_selection_coverage()
        elif strategy == "loop_head":
            self._rust_mgr.set_state_selection_loop_head()
        elif strategy == "directed":
            if not distances:
                raise ValueError("strategy 'directed' requires a non-empty distances map")
            dmap = {int(a) & 0xFFFFFFFFFFFFFFFF: int(d) & 0xFFFFFFFFFFFFFFFF for a, d in distances.items()}
            self._rust_mgr.set_state_selection_directed(dmap, int(beam_width))
        elif strategy == "find_directed":
            if not distances:
                raise ValueError("strategy 'find_directed' requires a non-empty distances map")
            dmap = {int(a) & 0xFFFFFFFFFFFFFFFF: int(d) & 0xFFFFFFFFFFFFFFFF for a, d in distances.items()}
            self._rust_mgr.set_state_selection_find_directed(dmap)
        else:
            raise ValueError(
                f"Unknown exploration strategy: {strategy!r}. "
                "Use 'bfs', 'dfs', 'random', 'coverage', 'loop_head', 'directed', or 'find_directed'."
            )

    def coverage_blocks(self, stashes: tuple[str, ...] = ("found",)) -> list[int]:
        """Sorted set of distinct block addresses covered by the solution states
        of a find-all / exhaustive run (angr-op0dn.13.3.1, M5-B14 sub-goal c).

        The set is the union of the full basic-block histories of every state in
        ``stashes`` (default: ``found``). It is derived from the *solution*
        states rather than the scheduling policy's ``seen``-set on purpose: the
        found-set is proven worker-count invariant (bd
        ``findall-parallel-slower-than-serial``), whereas the coverage-guided
        policy's ``seen``-set is a selection-order artifact — the parallel wave
        loop hands seed states to workers WITHOUT going through
        ``policy.select``, so a block like the entry ``main`` would be missing
        under >1 worker. Use ``set_exploration_strategy('coverage')`` to guide
        exploration toward new blocks; this report reads out the result.

        A history is the blocks a state executed, as Python's
        ``history.bbl_addrs`` is: the entry block is in, the state's own
        address (not yet executed) is not (angr-v625t).

        Note: histories are capped by the state's ``max_history``; on a very
        long-running real binary the report reflects the retained tail.
        """
        return self._coverage_blocks_of(sid for stash in stashes for sid in self._rust_mgr.get_state_ids(stash))

    def _coverage_blocks_of(self, state_ids) -> list[int]:
        """:meth:`coverage_blocks` over *state_ids* instead of whole stashes."""
        blocks: set[int] = set()
        for sid in state_ids:
            hist = self._rust_mgr.get_state_bbl_history_tail(sid, 0)
            if hist:
                blocks.update(hist)
        return sorted(blocks)

    def coverage_report(self, stashes: tuple[str, ...] = ("found",)) -> SimStateEdgeHitmap:
        """Edge-coverage report over a find-all / exhaustive run, in the existing
        :class:`~angr.state_plugins.SimStateEdgeHitmap` (64KB AFL) format.

        Reuses that plugin's container rather than adding a new coverage store
        (angr-op0dn.13.3.1): the report OR-folds the ``(prev_block, cur_block)``
        edges of every solution state's basic-block history into the hitmap. The
        fold is order-independent and computed over the worker-invariant found
        set, so the report bytes — and the edge/block counts derived from them —
        are identical across worker counts for an exhaustive run.

        ``stashes`` selects which terminal stashes contribute (default:
        ``found``). Only ``found`` is guaranteed worker-invariant; the parallel
        wave loop does not retain ``deadended`` / ``pruned`` states, so widening
        ``stashes`` trades that guarantee for a fuller picture on serial runs.
        """
        return self._coverage_report_of(sid for stash in stashes for sid in self._rust_mgr.get_state_ids(stash))

    def _coverage_report_of(self, state_ids) -> SimStateEdgeHitmap:
        """:meth:`coverage_report` over *state_ids* instead of whole stashes."""
        from angr.state_plugins import SimStateEdgeHitmap

        size = SimStateEdgeHitmap.HITMAP_SIZE
        hitmap = bytearray(size)
        for sid in state_ids:
            hist = self._rust_mgr.get_state_bbl_history_tail(sid, 0)
            if not hist:
                continue
            for prev, cur in zip(hist, hist[1:]):
                # AFL indexes an edge by the two block ids; we have no
                # instrumentation ids, so hash the block addresses
                # deterministically. Shifting off the low 4 bits collapses
                # within-block offsets; the prev-shift keeps A->B and B->A
                # distinct. OR the bit in — a set folded into a bitmap.
                idx = ((prev >> 4) ^ ((cur >> 4) << 1)) % size
                hitmap[idx] = 0xFF
        return SimStateEdgeHitmap(edge_hitmap=bytes(hitmap))

    def register_uniqueness_filter(self, register_names: list[str]):
        """Enable the native uniqueness filter keyed on the given registers.

        States whose listed-register values collide with an already-seen
        combination are dropped, collapsing reconverging paths. Pass the
        register names (e.g. ``["rip"]`` or ``["rax", "rbx"]``) the filter
        should hash. Replaces any previously-registered filter and resets its
        seen-set. See the Rust core
        (``exploration/mod.rs::register_uniqueness_filter``).
        """
        self._rust_mgr.register_uniqueness_filter(list(register_names))

    def disable_uniqueness_filter(self):
        """Disable the native uniqueness filter and clear its seen-set."""
        self._rust_mgr.disable_uniqueness_filter()

    def uniqueness_filter_enabled(self) -> bool:
        """Return True if the native uniqueness filter is currently active."""
        return self._rust_mgr.uniqueness_filter_enabled()

    def uniqueness_set_size(self) -> int:
        """Return the number of distinct register-combinations seen so far."""
        return self._rust_mgr.uniqueness_set_size()

    def _route_preinit_seed_finds(self, find_addrs: list[int]) -> None:
        """Route seeds the Python init skip fast-forwarded past a find address.

        A state constructed at the entry point is stepped to ``main`` in Python
        before it ever reaches Rust (``_run_python_init_if_needed``), so Rust's
        pre-step ``find_addrs`` check never sees the seed's own PC and
        ``explore(find=proj.entry)`` would explore forever instead of finding
        (angr-lyvf2). Vanilla angr's ``Explorer`` matches such a state in
        ``filter()``, before the first step, so it belongs in FOUND unstepped:
        push the un-advanced seed into the found stash and drop its
        fast-forwarded descendants from their stashes, exactly as ``filter()`` moves
        the state out of active.

        Seeds whose own PC is not a target stay parked: the find address may
        still lie strictly *inside* the init prefix, which
        :meth:`_replay_preinit_prefix_for_find` checks once the exploration
        proper has come up empty (angr-bdeqa). ``explore()`` releases the refs
        when it returns.
        """
        seeds, self._preinit_seeds = self._preinit_seeds, []
        targets = set(find_addrs)
        routed = False
        for seed_addr, seed_state, placed in seeds:
            if seed_addr not in targets:
                self._preinit_seeds.append((seed_addr, seed_state, placed))
                continue
            for stash, sid in placed:
                if sid is None:
                    continue
                try:
                    if self._rust_mgr.drop_state_from_stash(sid, stash):
                        self._state_cache.pop(sid, None)
                        self._state_roots.pop(sid, None)
                except MemoryError:
                    raise
                except Exception as e:
                    # cat-(b) BENIGN: the advanced state stays in its stash and
                    # is explored redundantly; the seed is still reported as found.
                    l.debug("preinit seed drop(sid=%d) failed: %s: %s", sid, type(e).__name__, e)
            self._add_rust_state("found", seed_state)
            routed = True
        if routed:
            self._invalidate_state_export_cache(detach_held=False)

    def _replay_preinit_prefix_for_find(self, find_addrs: list[int], avoid_addrs: list[int]) -> None:
        """Re-run the skipped Python init prefix, matching find addresses inside it.

        ``_run_python_init_if_needed`` fast-forwards an entry-point seed to
        ``main`` in Python without matching any find address on the way, so
        an address the prefix traverses — inside ``__libc_csu_init``,
        ``frame_dummy``, a ``.init_array`` ctor — is never a PC Rust sees, and
        ``explore(find=<that addr>)`` would run to exhaustion (angr-bdeqa).
        Vanilla angr's ``Explorer`` matches it during those first steps.

        Rather than pay the prefix replay on every address-based ``explore()``,
        it runs lazily: only once the Rust exploration has finished with an
        empty found stash, which is exactly the buggy case. The state pushed to
        FOUND is the real mid-init state at the target address, produced by the
        same Python stepping the init skip does — not a stand-in.

        ``avoid_addrs`` is honored *within the replay* (a prefix path that hits
        an avoid address before the find address does not match), but an avoid
        address inside the prefix does not retroactively kill the states Rust
        already explored from ``main``. See the state-serialization/init notes
        in ``docs/advanced-topics/rust_engine.rst``.
        """
        if not self._preinit_seeds:
            return
        from angr import SimulationManager

        targets = set(find_addrs)
        avoid = set(avoid_addrs)
        main_addr = resolve_main_address(self._project)
        routed = False

        for _seed_addr, seed_state, _placed in self._preinit_seeds:
            sm = SimulationManager(project=self._project, active_states=[seed_state.copy()])
            hit = None
            for _ in range(500):
                if not sm.active:
                    break
                hit = next((s for s in sm.active if s.addr in targets), None)
                if hit is not None:
                    break
                # Past the prefix (or avoided): Rust already owns the rest.
                # NB: assigning ``sm.active`` would shadow the dynamic stash
                # attribute with a frozen list — move states out instead.
                sm.move(
                    from_stash="active",
                    to_stash="stashed",
                    filter_func=lambda s: s.addr == main_addr or s.addr in avoid,
                )
                if not sm.active:
                    break
                sm.step()
            if hit is not None:
                l.info("preinit replay: find address 0x%x lies inside the Python init prefix", hit.addr)
                self._add_rust_state("found", hit)
                routed = True
        if routed:
            self._invalidate_state_export_cache(detach_held=False)

    def explore(
        self,
        find: int | list | Callable | None = None,
        avoid: int | list | Callable | None = None,
        num_find: int | None = 1,
        until: Callable | None = None,
        timeout: float | None = None,
        max_steps: int | None = None,
        stash: str = "active",
        n: int | None = None,
        find_stash: str = "found",
        avoid_stash: str = "avoid",
        cfg=None,
        avoid_priority: bool = False,
        **kwargs,
    ) -> RustExplorationManager:
        """Run exploration with find/avoid conditions.

        Args:
            find: Address(es) or callable predicate for finding solutions.
            avoid: Address(es) or callable predicate for avoiding states.
            num_find: Number of solutions to find before stopping. Pass ``None``
                to request find-all / run-to-exhaustion: every reachable
                solution is collected and the run terminates when the active
                frontier drains (``active_empty``) rather than at a fixed count.
                Exhaustive find-all does not raise ``max_active_states``; a
                genuine fork explosion still prunes to that backstop, so a
                non-empty ``pruned`` stash means the run was not exhaustive.
            until: Callable predicate that receives `self` and returns True to stop.
            timeout: Wall-clock timeout in seconds.
            max_steps: Maximum exploration steps before stopping.
            stash: ``"active"`` only, as in SimulationManager.explore.
            n: Step at most this many times, each step a SimulationManager step (every active state advances one
                block), as SimulationManager.explore(n=) does.
            find_stash: Stash this call's finds move to (SimulationManager.explore's ``find_stash``).
            avoid_stash: Stash this call's avoided states move to.
            cfg: ``None`` only. Explorer's CFG pruning has no Rust counterpart.
            avoid_priority: ``False`` only.
            **kwargs: Nothing. SimulationManager.explore forwards these to step(), whose engine arguments the Rust
                run loop has no counterpart for. Like every argument above outside its supported value, any
                kwarg hands a not-yet-stepped default-routed manager off to the Python engine, and raises
                NotImplementedError on any other manager (angr-vmi2l.243).

        Returns:
            Self, for chaining.
        """
        # num_find=None requests find-all / run-to-exhaustion: collect every
        # reachable solution and stop when the frontier drains. The Rust run
        # loop returns active_empty (never a partial-count 'found') once active
        # empties with found_count < num_find — see the
        # invariant-active-empty-not-partial-found memory — so a large sentinel
        # makes the ">= num_find" early-stop never fire and lets the loop run to
        # exhaustion. sys.maxsize fits usize on 64-bit and never realistically
        # collides with an actual found_count. Kept out of the num_find==1
        # find-directed opt-in path, so find-all stays breadth-first.
        unhonored = sorted(kwargs)
        unhonored += [f"stash={stash!r}"] if stash != "active" else []
        unhonored += ["cfg"] if cfg is not None else []
        unhonored += ["avoid_priority"] if avoid_priority else []
        unhonored += self._mixed_lazy_solves()
        if self._refuse_or_hand_off("explore", unhonored):
            if until is not None:
                kwargs["until"] = until
            return self.explore(
                stash=stash,
                n=n,
                find=find,
                avoid=avoid,
                find_stash=find_stash,
                avoid_stash=avoid_stash,
                cfg=cfg,
                num_find=num_find,
                avoid_priority=avoid_priority,
                **kwargs,
            )
        settled_before = {
            source: set(self._rust_mgr.get_state_ids(source))
            for source, target in (("found", find_stash), ("avoid", avoid_stash))
            if target != source
        }

        if num_find is None:
            num_find = sys.maxsize
        else:
            # As SimulationManager.explore does: num_find counts new finds, so a
            # repeat explore() does not stop at once on the finds already
            # stashed by the last one (angr-mx00p.132.15).
            num_find += self._found_count()

        # Re-entry into Rust execution invalidates the state-export cache:
        # any previously-cached Python mirrors are about to go stale.
        self._invalidate_state_export_cache()

        # Ensure predicate attributes exist (may not be set if find/avoid not provided)
        if not hasattr(self, "_find_predicate"):
            self._find_predicate = None
        if not hasattr(self, "_avoid_predicate"):
            self._avoid_predicate = None

        # As SimulationManager.explore removes its Explorer when the call ends:
        # a find/avoid left out of this call must not stay armed from the last
        # one (angr-mx00p.132.49). A native Explorer technique owns the Rust
        # find/avoid addrs instead, so they are left alone while one is active.
        explorer_technique_armed = any(type(t).__name__ in _FIND_AVOID_TECH_NAMES for t in self._active_techniques)
        if find is None and not explorer_technique_armed:
            self._rust_mgr.set_find_addrs([])
            self._find_predicate = None
            self._explore_find_addrs = None

        # Set find addresses and store predicate for callback handling
        if find is not None:
            find_addrs = self._extract_addrs(find)
            self._rust_mgr.set_find_addrs(find_addrs)
            self._rust_mgr.set_find_needs_python(callable(find))
            self._find_predicate = find if callable(find) else None
            # angr-027h: remember whether this is an address-based find so the
            # two-phase eager retry only fires when there is a concrete target.
            self._explore_find_addrs = None if callable(find) else find_addrs
            if not callable(find):
                self._route_preinit_seed_finds(find_addrs)

        # Set avoid addresses
        avoid_addrs: list[int] = []
        if avoid is not None:
            avoid_addrs = self._extract_addrs(avoid)
        if avoid is not None or not explorer_technique_armed:
            self._rust_mgr.set_avoid_addrs(avoid_addrs)
        self._rust_mgr.set_avoid_needs_python(callable(avoid))
        self._avoid_predicate = avoid if callable(avoid) else None
        self._explore_avoid_addrs = avoid_addrs

        # Set num_find
        self._rust_mgr.set_num_find(num_find)

        # Options added or removed since construction count (angr-vmi2l.241).
        self._latch_lazy_solves(self._rust_mgr.get_state_ids("active"))

        # Reset solver profiling stats for this exploration run
        try:
            from angr.rustylib.vex_engine import RustExplorationManager as _REM

            _REM.reset_solver_stats()
        except (ImportError, RuntimeError, AttributeError):
            # cat-(b) FALLBACK WITH LOSS: solver-stats reset failed (e.g., the
            # Rust extension was built without Z3); the run accumulates over
            # whatever counters survived the previous explore().
            pass

        # The phase-2 retry and the preinit-prefix replay both re-run paths
        # from the seeds, so they only stand in for this explore when it
        # starts from them: after a step() or an earlier explore they revive
        # paths the frontier already left (angr-vmi2l.319).
        self._explore_from_seeds = getattr(self, "_frontier_is_seeds", False)
        self._frontier_is_seeds = False

        # Route to appropriate exploration strategy
        has_predicates = self._find_predicate is not None or self._avoid_predicate is not None
        # Parallel engages only on the pure address-based full-batch path — the
        # same conditions the Rust steady loop requires (no callable predicates,
        # no `until`, no techniques). Everything else runs single-threaded so the
        # found-set stays correct (angr-op0dn.13.5).
        parallel_eligible = not has_predicates and until is None and not self._active_techniques
        self._engage_parallel_workers(parallel_eligible)
        restore_veritesting = self._install_veritesting_target_filter(find, avoid)
        try:
            try:
                # SimulationManager.run() asks complete() before its first step, so a technique that is already
                # complete steps nothing (angr-mx00p.132.40); the loops below only ask after a step.
                if self._active_techniques and self._check_technique_complete():
                    result = self
                elif n is not None:
                    result = self._explore_rounds(n, num_find, until)
                elif has_predicates or bool(self._active_techniques):
                    result = self._explore_with_predicates(num_find, until, timeout, max_steps)
                else:
                    result = self._explore_with_addresses(num_find, until, timeout, max_steps)
            finally:
                restore_veritesting()
                self._release_gone_exports()

            # angr-bdeqa: an address-based find that came up empty may be targeting
            # an address strictly inside the Python init prefix, which Rust never
            # steps through. Replay the prefix once, then release the parked seeds.
            if find is not None and not callable(find):
                if self._preinit_seeds and self._explore_from_seeds and self._found_count() == 0:
                    self._replay_preinit_prefix_for_find(self._extract_addrs(find), avoid_addrs)
                self._preinit_seeds = []
        finally:
            if not explorer_technique_armed:
                self._disarm_explore_targets()
            for source, before in settled_before.items():
                target = find_stash if source == "found" else avoid_stash
                for sid in self._rust_mgr.get_state_ids(source):
                    if sid not in before:
                        self._rust_mgr.move_state(sid, source, target)
        self._raise_uncovered_error()
        return result

    def _explore_rounds(self, n: int, num_find: int, until: Callable | None) -> RustExplorationManager:
        """Run explore(n=)'s budget: at most *n* SimulationManager steps, as SimulationManager.explore does.

        Each round is a :meth:`step` (every active state advances one block), so *n* bounds the run as it does on
        the Python engine, and a find that is never reached still returns (angr-vmi2l.243).
        """
        rust = self._rust_mgr
        # Python evaluates callable predicates between rounds, where Explorer's filter runs before each step. A
        # Rust-side predicate bounce would spend a round's run(1) on the callback and leave the state unstepped
        # (angr-mx00p.136.35).
        rust.set_find_needs_python(False)
        rust.set_avoid_needs_python(False)
        drop_terminal = rust.stats()["drop_terminal_states"]
        rust.set_drop_terminal_states(False)
        was_granular = self.set_block_granular(True)
        # Python checks the found count once a round is done. Left to Rust, the round's run(1)
        # stops at the find that reaches num_find, so the states after it are not stepped in the
        # last round, as SimulationManager's last step steps them (angr-mx00p.136.14).
        rust.set_num_find(sys.maxsize)
        try:
            if self._round_reorder is not None or self._round_side_step is not None:
                self._explore_reordered_rounds(n, num_find)
                return self
            self._evaluate_predicates_on_active()
            for round_no in range(n):
                # SimulationManager's Explorer filters before it steps, so nothing reached in the budget's
                # last step is stashed; the run ends with those states active (angr-mx00p.136.35).
                if round_no + 1 == n and not self._active_techniques:
                    self._explore_last_round()
                    return self
                self._invalidate_state_export_cache()
                # An empty visible stash still steps: the round's run(1) flushes parked bounces.
                if not self._step_loop(1):
                    break
                self._evaluate_predicates_on_active()
                if self._found_count() >= num_find:
                    if not self._active_techniques and until is None:
                        self._explore_last_round()
                        return self
                    break
                if self._active_techniques and self._check_technique_complete():
                    break
                if until is not None and until(self):
                    break
            self._evaluate_predicates_on_active()
        finally:
            rust.set_num_find(num_find)
            self.set_block_granular(was_granular)
            rust.set_drop_terminal_states(drop_terminal)
        return self

    def _explore_reordered_rounds(self, n: int, num_find: int) -> None:
        """Run :meth:`_explore_rounds`'s budget with :attr:`_round_reorder` and :attr:`_round_side_step` called after every round.

        The reorder hook may move states between stashes, as DFS's step() does, so a
        state that arrives at a find or avoid address must still be active when
        it runs: each round filters first and steps with the targets disarmed,
        as SimulationManager's Explorer does, and the run stops once the found
        count is reached or ``active`` is empty after the hook (angr-mx00p.136.15).
        The side-step hook's states are part of that round, its finds of the count
        and its active states of ``active``, so the run stops on the step the
        Python loop over all of them stops on (angr-mx00p.136.40.2).
        """
        rust = self._rust_mgr
        side_active, side_found = self._round_side_step is not None, 0
        for _ in range(n):
            if not rust.get_state_ids("active") and not side_active:
                return
            # A round that empties ``active`` stops the step loop, but the hook can refill it from
            # the states it moved aside, as DFS does.
            if not self._explore_last_round() and rust.get_state_ids("active"):
                return
            if self._round_reorder is not None:
                self._round_reorder()
            if self._round_side_step is not None:
                side_active, side_found = self._round_side_step()
            if self._found_count() + side_found >= num_find:
                return

    def _explore_last_round(self) -> bool:
        """Filter the active states as Explorer does, then step them once with no find or avoid armed.

        SimulationManager's Explorer filters a state before stepping it, so a
        state reaching a find address is stashed one step after it arrives,
        and in that step every other active state advances once more. Rust
        stashes on arrival, so the round that reaches ``num_find`` is one
        round short of the Python run; this is that round (angr-mx00p.136.14).
        The last round of an ``n`` budget is one too (angr-mx00p.136.35). A
        state arriving at a find or avoid address in it stays active, as on
        Python. The filter stands in for Rust's pre-step check, which the
        disarm turns off: a state that starts the round at a target (a seed
        when ``n=1``) is stashed unstepped, find first as Explorer checks.
        The targets stay lift cuts for the step, so a block ends at one it
        holds, as Explorer's ``extra_stop_points`` split it (angr-mx00p.132.60).
        """
        rust = self._rust_mgr
        self._evaluate_predicates_on_active()
        find_addrs = set(self._explore_find_addrs or ())
        avoid_addrs = set(self._explore_avoid_addrs or ())
        if find_addrs or avoid_addrs:
            for sid, addr, _stdout_len in rust.get_state_predicate_info("active"):
                if addr in find_addrs:
                    rust.move_state(sid, "active", "found")
                elif addr in avoid_addrs:
                    rust.move_state(sid, "active", "avoid")
        rust.set_find_addrs([])
        rust.set_avoid_addrs([])
        rust.set_find_needs_python(False)
        rust.set_avoid_needs_python(False)
        self._invalidate_state_export_cache()
        rust.set_lift_cut_addrs(sorted(find_addrs | avoid_addrs))
        try:
            return self._step_loop(1)
        finally:
            rust.set_lift_cut_addrs([])

    def _disarm_explore_targets(self) -> None:
        """Drop this explore()'s find/avoid/num_find from the Rust manager.

        SimulationManager.explore removes its Explorer when the call returns, so
        a later step()/run() neither stashes a state at a find/avoid address nor
        stops on the found count. Left armed, step() on a found state moved back
        to active re-stashed it as found without advancing (angr-mx00p.132.19).
        The Python-side predicates and ``_explore_*_addrs`` stay for the
        callback dispatch and the next explore(), which re-arms everything.
        """
        self._rust_mgr.set_find_addrs([])
        self._rust_mgr.set_avoid_addrs([])
        self._rust_mgr.set_num_find(sys.maxsize)

    def _install_veritesting_target_filter(self, find, avoid) -> Callable[[], None]:
        """Make the first Veritesting stop its nested run at a find/avoid state.

        ``SimulationManager.explore`` does the same: Veritesting steps a merge
        region in its own SimulationManager, which knows nothing of the
        explore targets, so without this deviation filter a merge steps
        straight through the find address and the outer manager never sees a
        state there (angr-mx00p.89). Returns the callable that restores the
        technique's own filter.
        """
        from angr.exploration_techniques import Veritesting

        vt = next((t for t in self._active_techniques if isinstance(t, Veritesting)), None)
        if vt is None or (find is None and avoid is None):
            return lambda: None

        def target_test(cond):
            if cond is None:
                return lambda _s: False
            if callable(cond):
                return lambda s: bool(cond(s))
            addrs = frozenset(self._extract_addrs(cond))
            return lambda s: s.addr in addrs

        is_find, is_avoid = target_test(find), target_test(avoid)
        saved = vt.options.get("deviation_filter")
        if saved is None:
            vt.options["deviation_filter"] = lambda s: is_find(s) or is_avoid(s)
        else:
            vt.options["deviation_filter"] = lambda s: is_find(s) or is_avoid(s) or saved(s)

        def restore() -> None:
            if saved is None:
                vt.options.pop("deviation_filter", None)
            else:
                vt.options["deviation_filter"] = saved

        return restore

    def _engage_parallel_workers(self, eligible: bool) -> None:
        """Apply the programmatic ``parallel_workers=`` request for this explore.

        Precedence and eligibility, in order (angr-op0dn.13.5):

        * ``RUST_PARALLEL_WORKERS`` env set — the env value was already applied
          once in Rust ``new()`` and wins for eligible explores (benches set it
          explicitly). An *ineligible* explore (callable predicates or ``until``)
          still downgrades to serial for that explore and restores the env count
          when the next explore is eligible (angr-ph300.77); older Rust builds
          without the setter cannot downgrade and leave the env value active.
        * ``parallel_workers <= 1`` requested — nothing to engage; no-op.
        * eligible explore (address-based find/avoid, no ``until``/techniques) —
          engage the requested worker count on the wave coordinator (the
          13.3-validated, worker-invariant parallel path).
        * ineligible explore (callable find/avoid) — fall back to single-threaded
          and log the reason; there is no worker-local skip-state analogue for
          Python predicates, so a correct found-set requires the serial loop.

        Steady-state auto-engagement (``RUST_PARALLEL_STEADY``) is intentionally
        NOT armed here. The found-accounting over-collection that once blocked it
        (steady=12 vs serial=8 on the M5 synthetic) is FIXED — ``push_found_capped``
        in the Rust routing helper now caps the parallel found set at ``num_find``,
        so steady is worker-invariant and count-equal to serial (angr-op0dn.13.17).
        What keeps steady opt-in is *performance*, not correctness: it is
        net-negative on the CTF corpus and actively hurts ``num_find=1`` first-find
        explores (barrier-free speculative waste), reaching only parity on the
        exhaustive partial-bounce workloads it targets
        (bd memory `steady-state-loop-opt-in-net-negative-corpus`). Auto-arming it for every
        eligible address-based explore would regress those benches, so the kwarg
        engages the wave loop only; steady stays an explicit ``RUST_PARALLEL_STEADY``
        opt-in for the exhaustive workloads that benefit.

        Older Rust builds without the ``set_parallel_workers`` pymethod degrade
        to a no-op (the env gate remains the only opt-in there).
        """
        setter = getattr(self._rust_mgr, "set_parallel_workers", None)
        if self._parallel_workers_env_set:
            # The env value was applied once in Rust new() and wins for eligible
            # explores. But an ineligible explore (callable predicates or
            # `until`) must run serial: a parallel wave runs its frontier to
            # quiescence with the GIL released, ignoring the per-batch
            # run(n)/until budget and hanging a non-terminating frontier
            # (angr-ph300.77). Downgrade to 1 for this explore, and restore the
            # env-configured count when a later explore is eligible. Older Rust
            # builds without the setter cannot downgrade — leave the env value
            # active (the pre-fix behaviour).
            if setter is None or self._parallel_workers_env_value <= 1:
                return
            if eligible:
                setter(self._parallel_workers_env_value)
            else:
                setter(1)
                l.info(
                    "RUST_PARALLEL_WORKERS=%d set but this explore has callable find/avoid "
                    "predicates or an `until` (no worker-local skip-state analogue, and the "
                    "wave ignores the per-batch budget); running single-threaded",
                    self._parallel_workers_env_value,
                )
            return
        if self._requested_parallel_workers <= 1:
            return
        if setter is None:
            l.debug("set_parallel_workers unavailable on this Rust build; parallel_workers= inert")
            return
        if eligible:
            setter(self._requested_parallel_workers)
        else:
            setter(1)
            l.info(
                "parallel_workers=%d requested but this explore has callable find/avoid "
                "predicates (no worker-local skip-state analogue); running single-threaded",
                self._requested_parallel_workers,
            )

    def _explore_with_predicates(self, num_find, until, timeout, max_steps):
        """Exploration loop for callable predicates or active techniques.

        Runs in batches of 50 steps, evaluating predicates between batches.
        Callbacks are handled immediately when Rust returns need_callback events.
        """
        self._rust_mgr.set_find_needs_python(self._find_predicate is not None)
        # Avoid is asked at every block, as find is: the between-batch sweep
        # alone misses a state that leaves active/deadended mid-batch (e.g.
        # returns into an unconstrained address) (angr-mx00p.132.47).
        self._rust_mgr.set_avoid_needs_python(self._avoid_predicate is not None)
        # Keep terminal states alive so predicates can check them; the caller's
        # setting comes back afterwards (angr-mx00p.132.49).
        drop_terminal = self._rust_mgr.stats()["drop_terminal_states"]
        self._rust_mgr.set_drop_terminal_states(False)

        # An `until` predicate is evaluated after EVERY step by Python's
        # SimulationManager.run(); batching 50 native steps between checks
        # would run straight past the step the caller wanted to stop on. See
        # `_explore_with_addresses` for the full rationale (angr-gorvf.15).
        batch_size = 1 if until is not None else 50
        start_time = time.time()
        _explore_start_ns = time.perf_counter_ns()
        steps_taken = 0
        _time_in_rust_run = 0
        _time_in_predicate_eval = 0
        _time_in_active_check = 0

        while True:
            if self._check_limits(start_time, steps_taken, timeout, max_steps):
                break
            _t0 = time.perf_counter_ns()
            if not self._rust_mgr.has_active_states():
                _time_in_active_check += time.perf_counter_ns() - _t0
                break
            _time_in_active_check += time.perf_counter_ns() - _t0

            # Run a batch of steps, handling callbacks as they arise
            batch_limit = batch_size
            if max_steps is not None:
                batch_limit = min(batch_limit, max_steps - steps_taken)

            if until is not None:
                # As in _explore_with_addresses: `until` must not read a
                # pre-step mirror (angr-mx00p.132.29).
                self._invalidate_state_export_cache()
            steps_taken, _time_in_rust_run = self._run_predicate_batch(steps_taken, batch_limit, _time_in_rust_run)
            self._fire_progress_if_due(start_time, steps_taken)

            # Apply technique callbacks after the batch
            if self._active_techniques:
                self._apply_technique_filters()
                if self._check_technique_complete():
                    break

            # Evaluate predicates on all states after the batch
            _t2 = time.perf_counter_ns()
            self._evaluate_predicates_on_active()
            _time_in_predicate_eval += time.perf_counter_ns() - _t2
            # Stop when num_find is reached — counts both Python-predicate matches
            # and Rust-native find_addr matches. Without this check, exploring with
            # an int find addr but an active technique (use_technique path) would
            # never terminate even after Rust populated the found stash.
            if self._found_count() >= num_find:
                break
            if until is not None:
                try:
                    if until(self):
                        break
                except MemoryError:
                    raise
                except Exception:
                    # cat-(b) FALLBACK WITH LOSS: until predicate raised; we keep
                    # exploring and let the user fix their predicate. Debug-logs with
                    # exc_info.
                    l.debug("until predicate raised exception", exc_info=True)

        # Final predicate check on deadended/remaining states
        self._evaluate_predicates_on_active()
        self._rust_mgr.set_drop_terminal_states(drop_terminal)

        # Store timing breakdown for stats
        self._time_in_rust_run_ns = _time_in_rust_run
        self._time_in_predicate_eval_ns = _time_in_predicate_eval
        self._time_in_active_check_ns = _time_in_active_check
        self._time_in_explore_ns = time.perf_counter_ns() - _explore_start_ns
        return self

    def _run_predicate_batch(self, steps_taken, batch_limit, _time_in_rust_run):
        """Run up to batch_limit steps, dispatching callbacks immediately.

        Returns updated (steps_taken, _time_in_rust_run).
        """
        batch_steps_start = steps_taken
        batch_done = False
        while not batch_done and (steps_taken - batch_steps_start) < batch_limit:
            self._sync_hooks_before_step()
            self._stats_ffi_crossings += 1
            remaining = batch_limit - (steps_taken - batch_steps_start)
            _t1 = time.perf_counter_ns()
            if self._has_technique_step_state_hooks():
                # One step per dispatch: a step_state technique (Veritesting)
                # must see every state before it advances, as under
                # SimulationManager.step (angr-mx00p.89). `remaining` is also
                # what a step_complete event is credited with below.
                remaining = 1
                event = self._run_with_step_state_hooks(remaining)
            elif self._has_technique_step_hooks():
                event = self._run_with_step_hooks(remaining)
            else:
                event = self._rust_mgr.run(remaining)
            _time_in_rust_run += time.perf_counter_ns() - _t1
            self._rust_mgr.sync_state_index()

            if event.event_type == "need_callback":
                if self._dispatch_callback(event):
                    batch_done = True
                steps_taken += 1
            elif event.event_type == "active_empty":
                batch_done = True
            elif event.event_type in ("step_complete", "found"):
                steps_taken += remaining
                batch_done = True
            else:
                steps_taken += 1
                batch_done = True

        # Ensure at least 1 step counted per batch iteration
        if steps_taken == batch_steps_start:
            steps_taken += 1

        return steps_taken, _time_in_rust_run

    def _explore_with_addresses(self, num_find, until, timeout, max_steps):
        """Exploration loop for address-based find/avoid (no callable predicates).

        Lets Rust run full batches for performance. Callbacks still return
        immediately from Rust regardless of batch size.
        """
        start_time = time.time()
        steps_taken = 0
        need_per_step = (until is not None) or bool(self._active_techniques)

        # angr-nkoct steady-state: this address-based loop (no predicates, no
        # techniques — those route to _explore_with_predicates) reads nothing
        # from the ACTIVE stash between run() calls except guarded cache
        # cleanup, so worker frontiers may stay resident across the
        # Python-callback boundary. Eligible only for the full-batch path
        # (until is None); an `until` predicate inspects state each step and is
        # not eligible. This is one of the Rust-side steady engagement
        # conditions (also gated on RUST_PARALLEL_STEADY and >=2 workers), so
        # it is a safe no-op when steady mode is off.
        residency = not need_per_step
        self._set_frontier_residency(residency)
        # angr-vmi2l.157: what each stash held before this explore, so a phase-2
        # retry can discard exactly phase 1's terminal states.
        self._pre_explore_stash_ids = self._stash_id_snapshot()

        while True:
            if self._check_limits(start_time, steps_taken, timeout, max_steps):
                break

            self._sync_hooks_before_step()
            self._stats_ffi_crossings += 1

            if need_per_step:
                # `until` is a per-STEP predicate in Python's SimulationManager
                # (checked after each step). Batching native steps between
                # checks runs past the intended stop point: fauxware's
                # `run(until=lambda sm: len(sm.active) > 1)` wants the two
                # states at the auth branch, but a 50-step batch executes the
                # whole program and leaves the active stash empty (angr-gorvf.15).
                # Before the native read/open widening, the SimProcedure bounce
                # returned control to Python after ~1 step and masked this.
                # Techniques keep the 50-step batch (they filter stashes between
                # batches rather than pinpointing a step).
                batch_size = 1 if until is not None else 50
                if max_steps is not None:
                    batch_size = min(batch_size, max_steps - steps_taken)
                if until is not None:
                    # The step-start invalidation step() does: a state that
                    # keeps its Rust id across the step would otherwise show
                    # `until` its cached pre-step mirror (angr-mx00p.132.29).
                    self._invalidate_state_export_cache()
                if self._has_technique_step_state_hooks():
                    # See _run_predicate_batch: one step per step_state dispatch.
                    event = self._run_with_step_state_hooks(1)
                elif self._has_technique_step_hooks():
                    event = self._run_with_step_hooks(batch_size)
                else:
                    event = self._rust_mgr.run(batch_size)
            elif max_steps is not None:
                # An unbounded native run() overshoots a step budget: it only
                # returns on a callback / found / active_empty, so `max_steps`
                # was enforced only at whatever step the next Python bounce
                # happened to land on. With the SimProcedure bounces retired
                # (angr-gorvf.15) a `run(max_steps=1)` on fauxware executed the
                # whole program and emptied the active stash. Bound the batch.
                event = self._rust_mgr.run(max_steps - steps_taken)
            else:
                event = self._rust_mgr.run()
            self._rust_mgr.sync_state_index()

            # Count steps: callbacks = 1, batched runs = event total
            if event.event_type == "need_callback":
                steps_taken += 1
            else:
                steps_taken += max(1, event.steps_taken)
            self._fire_progress_if_due(start_time, steps_taken)

            # Drop unconstrained states if save_unconstrained=False
            if not self._save_unconstrained:
                # cat-(b) FALLBACK WITH LOSS: clear_stash on the unconstrained
                # stash failed (no such stash, race with Rust); states may persist
                # in 'unconstrained' even though save_unconstrained=False.
                with contextlib.suppress(RuntimeError, KeyError):
                    self._rust_mgr.clear_stash("unconstrained")

            # Periodically clean Python state cache to prevent memory leaks
            if steps_taken % 100 == 0:
                self._cleanup_state_cache()

            # Dispatch event
            if event.event_type == "found" and event.found_count >= num_find:
                break
            if event.event_type == "active_empty":
                if self._active_techniques:
                    self._apply_technique_filters()
                    if self._check_technique_complete():
                        break
                    if self._rust_mgr.get_state_ids("active"):
                        continue
                # angr-027h two-phase explore: deferred-fork mode (phase 1)
                # exhausted without reaching the find target. Loop-exit forks
                # behind a symbolic loop (CADET easter-egg) were dropped at the
                # unconstrained jump, so the only egg-reaching paths never
                # materialized. Re-seed the pristine initial states in EAGER
                # mode (phase 2) and run again. Benches that find in phase 1
                # (e.g. whitehatvn2015_re400) break on the `found` event before
                # ever reaching active_empty, so they never pay the eager cost.
                if self._maybe_phase2_eager_retry(num_find):
                    steps_taken = 0
                    start_time = time.time()
                    continue
                break
            if event.event_type == "need_callback":
                if self._dispatch_callback(event):
                    break
            elif event.event_type == "errored" and _DBG:
                l.debug(f"Exploration error (state deadended): {event.callback_reason}")

            # Apply technique filters after step events
            if self._active_techniques and event.event_type in ("step_complete", "found", "steps_exhausted"):
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
                except MemoryError:
                    raise
                except Exception as e:
                    # cat-(c) WRONG-ANSWER RISK: until predicate raised; exploration
                    # continues past the user's intended stop. Already warns.
                    l.warning(f"until predicate error: {e}")

        # Every loop exit above is a `break`, so this runs on all normal exits:
        # drain any resident worker frontier back into STASH_ACTIVE so the
        # post-explore stashes are truthful, then clear the residency flag. An
        # exception propagating out of the loop is handled by the manager's Drop
        # (cancel + pool teardown), so a try/finally is not required here.
        self._finalize_parallel_session()
        if residency:
            self._set_frontier_residency(False)

        # angr-87e56: after the parallel session joins its workers, free any
        # owned solver forks their off-thread GC stranded in the graveyard.
        self._drain_solver_graveyard()
        return self

    def _maybe_phase2_eager_retry(self, num_find):
        """angr-027h: re-seed the initial states in eager-fork mode (phase 2).

        Fires at most once per explore, only for address-based finds that
        exhausted phase 1 (deferred) without reaching the target, and only
        when the explore started from the seed frontier (``_explore_from_seeds``). Returns True
        when a retry was seeded (the caller should reset its step/time budget and
        continue), False otherwise.
        """
        if getattr(self, "_phase2_retried", False):
            return False
        if not getattr(self, "_explore_find_addrs", None):
            return False
        if not getattr(self, "_explore_from_seeds", False):
            return False
        if self._found_count() >= num_find:
            return False
        # A find-all (num_find=None) or a resumed explore exhausts with
        # found < num_find even when phase 1 DID reach the target. Phase 2
        # re-explores from the root, so it would find every such state again
        # (angr-mx00p.132.48): retry only while no found state sits at a target.
        targets = set(self._explore_find_addrs)
        if any(self._rust_mgr.get_state_pc_by_id(sid) in targets for sid in self._rust_mgr.get_state_ids("found")):
            return False
        seeds = getattr(self, "_initial_seed_states", None)
        if not seeds:
            return False
        self._phase2_retried = True
        # (angr-027h)
        l.debug("phase 1 (deferred) exhausted without find; re-seeding %d state(s) in eager mode", len(seeds))
        try:
            self._rust_mgr.set_use_deferred_forks(False)
        except AttributeError:
            # Rust extension predates set_use_deferred_forks; cannot retry.
            return False
        # Phase 2 re-explores every path phase 1 did from the same roots, so
        # phase 1's found/avoid/deadended/errored/pruned states would each be
        # reported twice (angr-vmi2l.157). Keep only what predates this explore.
        self._drop_states_added_since(getattr(self, "_pre_explore_stash_ids", None) or {})
        # Re-seed from copies so the pristine originals stay reusable.
        self._phase2_reseeding = True
        try:
            self._phase_activate([s.copy() for s in seeds])
        finally:
            self._phase2_reseeding = False
        self._rust_mgr.sync_state_index()
        # Guard against a no-op re-seed (would otherwise spin active_empty).
        return bool(self._rust_mgr.get_state_ids("active"))

    def _stash_id_snapshot(self) -> dict[str, set[int]]:
        """Map each public stash to the state ids it holds now."""
        return {
            stash: set(self._rust_mgr.get_state_ids(stash))
            for stash in self._rust_mgr.stash_counts()
            if not stash.startswith("_")
        }

    def _drop_states_added_since(self, snapshot: dict[str, set[int]]) -> None:
        """Drop every public-stash state that is not in ``snapshot``.

        Internal ``_``-prefixed stashes (``_copies``) back live proxies and
        are left alone.
        """
        for stash, ids in self._stash_id_snapshot().items():
            for sid in ids - snapshot.get(stash, set()):
                self._rust_mgr.drop_state_from_stash(sid, stash)

    def get_state_options_py(self, state_id: int) -> set:
        """Return the Python-side options set for a Rust state.

        The Rust engine doesn't honor SimOptions (only LAZY_SOLVES /
        STRICT_PAGE_ACCESS and ``_NATIVE_SIMOPTIONS`` are mirrored onto the
        Rust state, at seed time), so this set lives Python-side and is
        advisory to native code. It is the state's one options store: every
        export of the state carries it (:meth:`_apply_state_metadata`). A
        fork's first access copies the nearest ancestor's set.

        Returns a live set — mutations propagate to subsequent accesses.
        """
        return self._state_metadata_entry(self._py_state_options, state_id, set)

    def get_state_globals_py(self, state_id: int) -> dict:
        """Return the Python-side globals dict for a Rust state.

        The state's one globals store, as :meth:`get_state_options_py` is its
        options store. Returns a live dict — mutations propagate.
        """
        return self._state_metadata_entry(self._py_state_globals, state_id, dict)

    def _state_metadata_entry(self, store: dict, state_id: int, copy, *, create: bool = True):
        """``store[state_id]``, first seeded with a ``copy`` of the nearest ancestor's entry.

        Nearest, not the lineage root: a value a technique wrote on an
        intermediate state is what its forks start from, as a
        ``SimState.copy`` would carry it (angr-vmi2l.265). With no entry on
        the state or any ancestor, ``None`` without ``create``; with it, the
        lineage's plugin template SimState seeds the entry, and only with no
        template is it empty. An empty options entry strips every option off
        the next export (:meth:`_apply_state_metadata`), so "no record" must
        not read as "the user turned everything off" (angr-kfchf).
        """
        entry = store.get(state_id)
        if entry is not None:
            return entry
        for ancestor in self._state_ancestry(state_id)[1:]:
            src = store.get(ancestor)
            if src is not None:
                entry = copy(src)
                break
        else:
            if not create:
                return None
            entry = self._unrecorded_metadata_entry(store, state_id, copy)
        store[state_id] = entry
        return entry

    def _unrecorded_metadata_entry(self, store: dict, state_id: int, copy):
        """A fresh options/globals entry for a lineage with no record, from its plugin template.

        The template (:meth:`_find_plugin_template_state`) is the SimState an
        export of ``state_id`` takes its plugins from, so seeding from it
        leaves that export's options and globals as they were.
        """
        template = self._find_plugin_template_state(state_id)
        if template is None:
            return copy()
        if store is self._py_state_options:
            if isinstance(getattr(getattr(template, "options", None), "_options", None), dict):
                return _options_on(template)
            return copy()
        if "globals" in template.plugins:
            return dict(template.globals)
        return copy()

    def _state_ancestry(self, state_id: int) -> list[int]:
        """``[state_id, parent, ..., root]`` as Rust still holds them (``get_state_ancestry``)."""
        try:
            return list(self._rust_mgr.get_state_ancestry(state_id))
        except MemoryError:
            raise
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: the ancestry lookup failed; treat the
            # state as having no ancestors, so its metadata starts empty.
            return [state_id]

    def _inherit_state_metadata(self, state_ids=None) -> None:
        """Give each of ``state_ids`` (default: the active stash) its own options/globals entry now.

        A fork copies its ancestor's entry on first access, and the walk to
        that ancestor ends where Rust no longer holds a state, so the copy is
        taken here too: when Python hands control back to Rust and before a
        cache cleanup prunes the entries of states gone from every stash.
        Skipped while every entry is a lineage root's: the lazy lookup then
        reads the same entry.
        """
        keys = self._py_state_options.keys() | self._py_state_globals.keys()
        if not keys or keys <= set(self._state_roots.values()):
            return
        if state_ids is None:
            state_ids = self._rust_mgr.get_state_ids("active")
        for sid in state_ids:
            if sid not in self._py_state_options:
                self.get_state_options_py(sid)
            if sid not in self._py_state_globals:
                self.get_state_globals_py(sid)

    def _apply_state_metadata(self, state, state_id: int, *, mirror: bool = False) -> None:
        """Give the SimState ``state`` (an export or callback frame of ``state_id``) its options and globals store.

        A ``mirror`` (the stash mirror ``mgr.active[i]``) is a view of the
        store: its globals plugin backs onto the store's dict, and the options
        it gains or loses are taken back when Python hands control to Rust
        (:meth:`_reconcile_mirror_options`). Anything else gets copies, and a
        callback frame's edits come back through :meth:`_record_state_metadata`.
        A state of no recorded lineage keeps what it has.
        """
        from angr.state_plugins.globals import SimStateGlobals

        options = self._state_metadata_entry(self._py_state_options, state_id, set, create=False)
        if options is not None:
            on = _options_on(state)
            for name in on - options:
                state.options.discard(name)
            for name in options - on:
                try:
                    state.options.add(name)
                except MemoryError:
                    raise
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: a name the proxy accepted that
                    # SimStateOptions does not know; the export goes without it.
                    l.debug("export of state %d: cannot add option %r: %s", state_id, name, e)
            if mirror:
                state.scratch.rust_applied_options = frozenset(_options_on(state))
        glb = self._state_metadata_entry(self._py_state_globals, state_id, dict, create=False)
        if glb is not None:
            state.register_plugin("globals", SimStateGlobals(glb if mirror else dict(glb)))
        # Rust's time() watermark is authoritative; _record_state_metadata
        # takes a Python time()'s update back (angr-vmi2l.130).
        last_time = self._rust_mgr.get_state_last_time(state_id)
        if last_time is not None:
            state.globals[_SYS_LAST_TIME] = last_time

    def _record_state_metadata(self, state_id: int, state) -> None:
        """Take a callback frame's options and globals back as ``state_id``'s store.

        A SimProcedure's ``state.globals`` / ``state.options`` edits land on
        the frame; the proxies and later exports read the store. In place, so
        a mirror backed onto the store keeps seeing it.
        """
        if isinstance(getattr(getattr(state, "options", None), "_options", None), dict):
            options = self.get_state_options_py(state_id)
            options.clear()
            options.update(_options_on(state))
        if "globals" in state.plugins:
            glb = self.get_state_globals_py(state_id)
            frame = dict(state.globals)
            glb.clear()
            glb.update(frame)
            self._note_x86_cr0(frame)
            last_time = frame.get(_SYS_LAST_TIME)
            if last_time is not None:
                self._rust_mgr.set_state_last_time(state_id, last_time)

    def _note_x86_cr0(self, glb) -> None:
        """Turn the native ``x86g_use_seg_selector`` off for good once a state's
        ``globals["x86_cr0"]`` may clear CR0.PE, which ccall.py reads ahead of
        ``arch.vex_archinfo`` (angr-vmi2l.78)."""
        cr0 = glb.get("x86_cr0")
        if cr0 is not None and not (isinstance(cr0, int) and cr0 & 1):
            self._rust_mgr.set_x86_protected_mode(False)

    def _reconcile_mirror_options(self) -> None:
        """Take the options a stash mirror gained or lost since its export into its store."""
        for sid, state in self._state_cache.items():
            scratch = getattr(state, "scratch", None)
            applied = getattr(scratch, "rust_applied_options", None)
            if applied is None or getattr(scratch, "rust_mirror_id", None) != sid:
                continue
            on = _options_on(state)
            if on != applied:
                options = self.get_state_options_py(sid)
                options.difference_update(applied - on)
                options.update(on - applied)
                scratch.rust_applied_options = frozenset(on)

    @functools.cached_property
    def _copy_stash_fork_owner(self):
        """Fork owner for proxy bindings outside a callback scope.

        See ``rust_state_binding.CopyStashForkOwner`` (angr-mx00p.112).
        """
        from angr.exploration.rust_state_binding import CopyStashForkOwner

        return CopyStashForkOwner(self)

    def fork_state_for_copy(self, source_state_id: int, stash: str = "_copies") -> int:
        """Rust-side CoW fork backing ``RustStateProxy.copy()``.

        Mints a new state by forking ``source_state_id`` into the dedicated
        ``_copies`` stash (or ``stash``: the copy-on-write proxy bindings of a
        callback state fork into ``_cb_forks``, angr-mx00p.112). The new state's Python-side metadata (options set,
        globals dict) is seeded from the source's current metadata — NOT the
        lineage root — so a caller that has mutated the source between fork
        time and copy time sees those mutations on the copy. The forked state
        inherits the parent's lineage root so downstream lookups
        (``get_state_options_py`` / ``get_state_globals_py``) still resolve.

        The ``_copies`` stash is treated as a holding area: ``step()`` does
        not advance it (StashManager only iterates ``active``), and
        ``find_state`` resolves it like any other stash so proxy reads on
        the returned id keep working.

        Memory note: copies linger in ``_copies`` until the caller drops the
        proxy. There is no automatic GC — that would require a back-reference
        from the proxy to the manager, which today is intentionally weak.
        """
        new_id = self._rust_mgr.fork_state_to_stash(source_state_id, stash)
        src_opts = self._py_state_options.get(source_state_id)
        if src_opts is not None:
            self._py_state_options[new_id] = set(src_opts)
        src_glb = self._py_state_globals.get(source_state_id)
        if src_glb is not None:
            self._py_state_globals[new_id] = dict(src_glb)
        return new_id

    def drop_copy(self, copy_state_id: int, stash: str = "_copies") -> bool:
        """Drop a clone produced by ``fork_state_for_copy`` from ``_copies``
        (angr-yhe0), or from the ``stash`` it was forked into.

        Removes the Rust-side state from the ``_copies`` stash, clears the
        per-state Python metadata (options set, globals dict),
        and frees Rust-side per-state metadata via
        ``clear_state_metadata``. Returns ``True`` when a state was actually
        dropped, ``False`` when the id is not in ``_copies`` (e.g. it was
        already dropped, moved to a different stash, or never a copy).

        This is the manual cleanup primitive backing ``RustStateProxy.__del__``
        for proxies returned by ``proxy.copy()``. The proxy invokes this in
        its finalizer so long-running Spiller / ManualMergepoint workflows
        don't leak copies until manager teardown. Callers can also invoke it
        directly when they know a copy is no longer needed but still hold a
        reference.
        """
        try:
            dropped = self._rust_mgr.drop_state_from_stash(copy_state_id, stash)
        except MemoryError:
            raise
        except Exception as e:
            # cat-(a) EXPECTED CONTROL FLOW: state may already be gone (e.g.
            # double-drop) or the Rust manager torn down. drop_copy is
            # best-effort.
            l.debug("drop_state_from_stash(sid=%d) failed: %s: %s", copy_state_id, type(e).__name__, e)
            return False
        if not dropped:
            return False
        self._py_state_options.pop(copy_state_id, None)
        self._py_state_globals.pop(copy_state_id, None)
        try:
            self._rust_mgr.clear_state_metadata(copy_state_id)
        except MemoryError:
            raise
        except Exception as e:
            # cat-(a) EXPECTED CONTROL FLOW: metadata clear is best-effort.
            l.debug("clear_state_metadata(sid=%d) after drop_copy failed: %s: %s", copy_state_id, type(e).__name__, e)
        return True

    def _set_frontier_residency(self, enabled):
        """Toggle the Rust steady-state frontier-residency engagement flag.

        No-op on older Rust builds without the pymethod (steady mode absent).
        """
        with contextlib.suppress(AttributeError):
            self._rust_mgr.set_parallel_frontier_residency(bool(enabled))

    def _finalize_parallel_session(self):
        """Drain any live steady-state session's resident frontier back into the
        stashes (angr-nkoct). No-op without a live session or on older Rust
        builds."""
        with contextlib.suppress(AttributeError):
            self._rust_mgr.finalize_parallel_session()

    def _prune_state_metadata(self, live: set[int]) -> None:
        """Drop ``_py_state_options`` / ``_py_state_globals`` entries outside ``live``.

        A lineage root's entry survives while any of its forks lives:
        ``get_state_options_py`` / ``get_state_globals_py`` seed a fork's first
        access from the root's entry, and the root's own id can leave every
        stash long before its forks do (angr-mx00p.136.42).
        """
        candidates = (self._py_state_options.keys() | self._py_state_globals.keys()) - live
        if not candidates:
            return
        candidates -= set(self._rust_mgr.live_lineage_roots())
        for sid in candidates:
            self._py_state_options.pop(sid, None)
            self._py_state_globals.pop(sid, None)

    def _cleanup_state_cache(self):
        """Bound ``_state_cache`` size while preserving correctness invariants.

        Step 1 — drop entries whose state no longer exists in any live stash
        (deadended/errored/avoid GC).  Step 2 — pin root states + the
        currently-dispatching callback state + the state being stepped.
        Step 3 — if the cache is still over ``_max_state_cache_size``, LRU-evict
        non-pinned entries (Python ``dict`` preserves insertion order; entries
        that were re-written most recently sit at the back, so evicting from
        the front removes the least-recently-touched first).

        Also prunes shadow structures keyed by state id (``_state_roots``,
        ``_predicate_matched_ids``) of entries whose state no longer exists
        in *any* Rust stash. Rust state IDs are monotonically allocated and
        never reused, so a dropped entry can never become relevant again.

        Metadata-clear contract: this path does NOT call
        ``clear_state_metadata`` on evicted ids. A state can be live in a
        Rust stash and still LRU-evicted from the Python mirror; freeing
        Rust-side metadata in that case would corrupt subsequent stepping.
        Per-state Rust metadata is dropped exclusively by ``RustSimState``'s
        own ``Drop`` when the state leaves every stash.
        """
        # angr-nkoct steady-state: while a session is live, some frontier states
        # are RESIDENT inside worker Z3 contexts and appear in NO stash. Pruning
        # now would treat them as dead and evict their _state_roots mirror,
        # causing a blank-state fallback when one later bounces. Skip the tick;
        # the session finalizes (draining every state back into the stashes) at
        # the step budget / find / exhaustion, where cleanup resumes normally.
        try:
            if self._rust_mgr.parallel_session_active():
                return
        except AttributeError:
            pass

        try:
            active_set = set(self._rust_mgr.get_state_ids("active"))
            # A step_state hook parks its live successors in the hold stash
            # while the native run steps the rest; they rejoin active next
            # batch, and their cached SimState is the only record of the stdin
            # Python read for them (angr-mx00p.132.26).
            active_set.update(self._rust_mgr.get_state_ids(_VT_HOLD_STASH))
            found_set = set(self._rust_mgr.get_state_ids("found"))
        except (RuntimeError, KeyError):
            # cat-(b) FALLBACK WITH LOSS: cannot read live state IDs; skip this
            # cleanup tick. Cache may temporarily exceed cap until next call.
            return

        # Broader "known-to-Rust" set for shadow-structure pruning. State IDs
        # outside this set are unreachable (no stash holds them) and Rust will
        # never resurrect them, so it's safe to drop any Python-side mapping
        # keyed on them. Failure here is non-fatal — fall back to the narrower
        # active/found set so we never over-prune.
        # Every stash counts, the private ones too: an export RustBackend handed
        # out waits in its ``_simgr_backend_parked`` stash, and a Spiller vault
        # can hand that live export back after missing steps, so its options
        # and globals entries must survive (angr-l0k4o).
        try:
            any_stash = active_set | found_set
            for name in self._rust_mgr.stash_counts():
                any_stash.update(self._rust_mgr.get_state_ids(name))
        except (RuntimeError, KeyError):
            # cat-(b) FALLBACK WITH LOSS: a missing stash view means the
            # shadow-prune below sees a smaller live set and is more aggressive
            # than ideal: mappings for states in the other stashes are dropped
            # this tick, and a held export of one of them loses its options.
            any_stash = active_set | found_set

        live = active_set | found_set
        # Materialize the root lookups into a separate set before updating
        # `live`: feeding a generator that reads `live` straight into
        # `live.update()` mutates the set mid-iteration ("Set changed size
        # during iteration") whenever a root ID is not already present — which
        # happens once forking produces enough distinct roots (reliably under
        # eager-fork mode, latent otherwise). See angr-027h.
        live.update({self._state_roots.get(sid, sid) for sid in live})
        # Always keep root state cache entries: a forked state can fire its
        # first Python callback without itself or its (Rust-only) ancestors
        # being in cache; the only way to avoid the blank-state fallback is
        # to fall back to the root, so the root must survive eviction even
        # when it is no longer in active/found. ``_state_roots`` holds only the
        # roots Python registered; a root whose own id left every stash while a
        # Rust-side fork lives is in ``live_lineage_roots`` alone (angr-mx00p.114).
        # Only a plain SimState mirror is worth pinning there: one whose proxy
        # plugins read the root's own Rust state fails once that state is gone.
        rust_roots = {
            sid
            for sid in self._rust_mgr.live_lineage_roots()
            if (cached := self._state_cache.get(sid)) is not None and binding_of(cached) is None
        }
        live |= set(self._state_roots.values()) | rust_roots
        for sid in list(self._state_cache.keys()):
            if sid not in live:
                del self._state_cache[sid]
        self._inherit_state_metadata(active_set | found_set)
        self._prune_state_metadata(live | any_stash)

        # Prune _state_roots: drop entries whose key state no longer exists in
        # any Rust stash. The root state's own self-entry survives only while
        # the root is still tracked by Rust. Without this, _state_roots grows
        # monotonically across explore() calls (every forked state's id stays
        # forever) — the root pinning above would then pin a growing set of
        # dead roots in _state_cache, defeating the cap.
        for sid in list(self._state_roots.keys()):
            if sid not in any_stash:
                del self._state_roots[sid]

        # Prune _predicate_matched_ids similarly: once the underlying state is
        # gone, the "already moved to found/avoid" mark is irrelevant. Lazy-
        # initialized in rust_state_cache.py, so guard with hasattr.
        matched = getattr(self, "_predicate_matched_ids", None)
        if matched is not None:
            matched.intersection_update(any_stash)

        # Prune _predicate_eval_cache: drop change-detection entries for states
        # no longer in any Rust stash. Formerly pruned per-state by the removed
        # `_cleanup_state_refs`; folded here so the (addr, stdout_len) cache
        # stays bounded by live stash size across long explorations.
        eval_cache = getattr(self, "_predicate_eval_cache", None)
        if eval_cache is not None:
            for sid in list(eval_cache.keys()):
                if sid not in any_stash:
                    del eval_cache[sid]

        # Drop _LazySimStateRef wrappers for state ids that no longer exist
        # in any Rust stash. The wrappers are cheap (two slots) but pruning
        # them keeps the dict bounded by live stash size across long runs.
        for sid in list(self._lazy_state_refs.keys()):
            if sid not in any_stash:
                del self._lazy_state_refs[sid]

        pinned = set(self._state_roots.values()) | rust_roots
        if self._current_callback_state_id is not None:
            pinned.add(self._current_callback_state_id)
            # cat-(b) FALLBACK WITH LOSS: effective-state lookup failed; only
            # the direct callback id is pinned, so an in-flight forked state
            # could be evicted. Tolerated — eviction is recoverable via re-
            # fetch on next access.
            with suppress_recoverable():
                eff_id = self._get_effective_state_id(self._current_callback_state_id)
                if eff_id is not None:
                    pinned.add(eff_id)
        if self._current_stepping_state_id is not None:
            pinned.add(self._current_stepping_state_id)

        cap = self._max_state_cache_size
        overflow = len(self._state_cache) - cap
        if overflow <= 0:
            return
        for sid in list(self._state_cache.keys()):
            if overflow <= 0:
                break
            if sid in pinned:
                continue
            del self._state_cache[sid]
            overflow -= 1

    def set_block_granular(self, enabled: bool = True) -> bool:
        """Toggle block-granular stepping (angr-bmyx).

        When enabled, the Rust VEX interpreter stops chaining basic blocks and
        returns to the step boundary after every block, so each ``step()``
        advances exactly one block and every interior address is observable —
        matching Python angr's block-granular ``step()`` semantics. This lets a
        bare step-loop (``while True: sm.step(); break if any active.addr ==
        TARGET``, e.g. CADET solve.py phase 3) detect a mid-path target that the
        chained interpreter would otherwise run straight through.

        ``explore(find=...)`` does not need this — address-based finds already
        break the chain at the specific target addresses — and leaving it off
        preserves chaining throughput on the ``explore``/benchmark paths.

        Args:
            enabled: True to step one block at a time, False to restore chaining.

        Returns:
            The previous setting, so a scoped step-loop can restore it. Returns
            False on a Rust extension that predates this method (no-op).
        """
        try:
            return bool(self._rust_mgr.set_block_granular(bool(enabled)))
        except AttributeError:
            # Rust extension predates set_block_granular; chaining stays on.
            return False

    def set_materialize_unconstrained_forks(self, enabled: bool = True) -> bool:
        """Toggle materialization of loop-exit forks at unconstrained jumps (angr-ckdy).

        In deferred-fork mode the loop-exit forks accumulated when a state goes
        unconstrained (too many symbolic jump targets) are materialized eagerly
        and routed to the active stash by default (angr-x39ae), so no sibling
        path is lost. Disabling drops them instead: faster, but those paths are
        gone unless an address-based ``explore()``'s two-phase eager retry
        (angr-027h) re-derives them, which it does not always do. A bare
        step-loop (CADET solve.py phase 3) relies on them being kept; pair with
        ``set_block_granular(True)`` so the target block is observable at a step
        boundary before the materialized subtree explodes.

        Args:
            enabled: True to materialize (keep) the forks, False to drop them.

        Returns:
            The previous setting, so a scoped step-loop can restore it. Returns
            False on a Rust extension that predates this method (no-op).
        """
        try:
            return bool(self._rust_mgr.set_materialize_unconstrained_forks(bool(enabled)))
        except AttributeError:
            # Rust extension predates set_materialize_unconstrained_forks.
            return False

    def step(self, n: int = 1, **kwargs) -> RustExplorationManager:
        """Step the exploration n times.

        Args:
            n: Number of steps to take.
            **kwargs: ``stash="active"`` only. Anything else (an engine run arg such as ``num_inst``, a
                selector, another stash) changes what a SimulationManager step executes, so a not-yet-stepped
                default-routed manager hands off to the Python engine and any other manager raises
                (angr-mx00p.132.23).

        Returns:
            Self, for chaining.
        """
        unhonored = sorted(k for k, v in kwargs.items() if not (k == "stash" and v == "active"))
        unhonored += self._mixed_lazy_solves()
        if self._refuse_or_hand_off("step", unhonored):
            for _ in range(n):
                self.step(**kwargs)
            return self
        self._latch_lazy_solves(self._rust_mgr.get_state_ids("active"))

        # Re-entry into Rust execution invalidates the state-export cache:
        # any previously-cached Python mirrors are about to go stale.
        self._invalidate_state_export_cache()
        self._frontier_is_seeds = False

        # A default-routed manager promises Python's step semantics: one basic
        # block per step, so the successor's ``scratch.irsb`` is the block it
        # just left (BackwardSlice reads it; angr-mx00p.132.24). An explicit
        # ``engine="rust"`` manager keeps block chaining for throughput and
        # opts in with ``set_block_granular`` (``block-granular-step-mode``).
        was_granular = self.set_block_granular(True) if self._python_handoff_kwargs is not None else None
        try:
            self._step_loop(n)
        finally:
            if was_granular is not None:
                self.set_block_granular(was_granular)

        self._release_gone_exports()
        self._drain_solver_graveyard()
        self._raise_uncovered_error()
        return self

    def _raise_uncovered_error(self) -> None:
        """Raise the error of a newly errored state that ``resilience`` does not cover.

        The state stays in ``errored``; each error id is raised at most once, so a later
        ``step()`` goes on with the other states. Every Rust error maps to a ``SimError``,
        so a resilience that covers ``SimError`` (the default and ``True``) never raises.
        """
        if any(issubclass(SimError, kind) for kind in self._resilience):
            return
        for _, message, state_id in self._rust_mgr.get_errors():
            if state_id in self._errors_raised:
                continue
            self._errors_raised.add(state_id)
            error = sim_error_for_rust_message(message, RustErrorRecord._classify(message))
            if not isinstance(error, self._resilience):
                raise error

    def _step_loop(self, n: int) -> bool:
        """Run ``n`` steps for :meth:`step`, stopping early on a callback or an empty stash; False when it stopped.

        A step is a SimulationManager step: each state in ``active`` when it begins is advanced once, and its
        successors are not re-stepped in the same step (angr-vmi2l.246). A technique step()/step_state() hook
        wraps that round, as it wraps SimulationManager.step's (angr-vmi2l.317).
        """
        for _ in range(n):
            self._sync_hooks_before_step()
            hooked = self._has_technique_step_state_hooks() or self._has_technique_step_hooks()
            if not (self._step_hooked_round() if hooked else self._step_round()):
                return False
        return True

    def _step_round(self) -> bool:
        """Advance every active state once, then apply technique filters; return False when stepping should stop."""
        keep_going, stepped = self._advance_round()
        if stepped and self._active_techniques:
            self._apply_technique_filters()
        return keep_going

    def _step_hooked_round(self) -> bool:
        """Run :meth:`_advance_round` as the native step under the technique hooks; False when stepping should stop.

        A step_state hook that rewrites a state takes it out of ``active`` before the round, so the round advances
        only the states every hook declined; a step() hook that never delegates leaves the round unrun.
        """
        from angr.exploration.rust_techniques import _SyntheticStepEvent

        outcome = {"keep_going": True}

        def run_round(_batch_size):
            outcome["keep_going"], _ = self._advance_round()
            counts = self._rust_mgr.stash_counts()
            return _SyntheticStepEvent(counts.get("found", 0), counts.get("active", 0))

        if self._has_technique_step_state_hooks():
            self._run_with_step_state_hooks(1, native_run=run_round)
        else:
            self._run_with_step_hooks(1, native_run=run_round)
        self._rust_mgr.sync_state_index()
        if outcome["keep_going"]:
            self._apply_technique_filters()
        return outcome["keep_going"]

    def _advance_round(self) -> tuple[bool, bool]:
        """Advance every active state once; return ``(keep_going, stepped)``.

        The round's states wait in ``_STEP_ROUND_QUEUE`` and are released into ``active`` one at a time, so the
        ``run(1)`` that steps each sees only that state whatever the selection policy. Successors collect in
        ``_STEP_ROUND_DONE`` and rejoin ``active`` in stepping order when the round ends, as do any states an
        early stop left unstepped.
        """
        rust = self._rust_mgr
        if rust.move_states("active", _STEP_ROUND_QUEUE, None) == 0:
            # Nothing visible to queue; a plain run(1) still flushes parked bounces or reports active_empty.
            self._stats_ffi_crossings += 1
            event = rust.run(1)
            rust.sync_state_index()
            kind = event.event_type
            return self._step_event_keeps_going(event), kind not in ("found", "active_empty")
        keep_going = True
        stepped = False
        try:
            for sid in rust.get_state_ids(_STEP_ROUND_QUEUE):
                rust.move_state(sid, _STEP_ROUND_QUEUE, "active")
                self._stats_ffi_crossings += 1
                event = rust.run(1)
                rust.sync_state_index()
                # num_find is reached on "found": run(1) returns before popping, so no state in the round can advance.
                keep_going = self._step_event_keeps_going(event) and event.event_type != "found"
                stepped = stepped or event.event_type != "found"
                rust.move_states("active", _STEP_ROUND_DONE, None)
                if not keep_going:
                    break
        finally:
            rust.move_states(_STEP_ROUND_DONE, "active", None)
            rust.move_states(_STEP_ROUND_QUEUE, "active", None)
        return keep_going, stepped

    def _step_ids_once(self, ids) -> dict[int, dict[str, list[int]]]:
        """Step each active state of ``ids`` once, as one SimulationManager step does; map it to its children.

        The primitive the simgr backend steps through (angr-mx00p.136.3). Each id is advanced by its own
        block-granular ``run(1)`` with every other state parked, so whatever lands in a stash during that run
        is its child. A state that does not fork keeps its id, so a deadended state maps to itself under
        ``"deadended"``. The returned dict has one entry per id, keyed by the Rust stash each child landed in
        (``"active"`` for the flat successors), and the active children rejoin ``active``. Terminal states are
        kept, the find/avoid targets stay disarmed (no explore() is running), and the manager's settings and
        the other active states are restored before it returns.
        """
        rust = self._rust_mgr
        self._invalidate_state_export_cache()
        self._frontier_is_seeds = False
        self._disarm_explore_targets()
        wanted = list(ids)
        rust.move_states("active", _STEP_ROUND_QUEUE, None)
        queued = set(rust.get_state_ids(_STEP_ROUND_QUEUE))
        missing = [sid for sid in wanted if sid not in queued]
        if missing:
            rust.move_states(_STEP_ROUND_QUEUE, "active", None)
            raise ValueError(f"_step_ids_once: states {missing} are not active")

        private = {"active", _STEP_ROUND_QUEUE, _STEP_ROUND_DONE}
        counts = {k: v for k, v in rust.stash_counts().items() if k not in private}
        seen = {k: set(rust.get_state_ids(k)) for k, v in counts.items() if v}
        drop_terminal = rust.stats()["drop_terminal_states"]
        rust.set_drop_terminal_states(False)
        was_granular = self.set_block_granular(True)
        children: dict[int, dict[str, list[int]]] = {}
        try:
            for sid in wanted:
                rust.move_state(sid, _STEP_ROUND_QUEUE, "active")
                # Each state steps alone, so it steps under its own LAZY_SOLVES.
                self._latch_lazy_solves([sid])
                self._stats_ffi_crossings += 1
                event = rust.run(1)
                rust.sync_state_index()
                if event.event_type == "need_callback":
                    self._dispatch_callback(event)
                elif event.event_type == "errored":
                    l.debug("_step_ids_once: state %d errored: %s", sid, event.callback_reason)
                buckets = {"active": rust.get_state_ids("active")}
                for name, count in rust.stash_counts().items():
                    if name in private or count == counts.get(name, 0):
                        continue
                    counts[name] = count
                    now = rust.get_state_ids(name)
                    before = seen.setdefault(name, set())
                    buckets[name] = [cid for cid in now if cid not in before]
                    before.update(now)
                children[sid] = {name: cids for name, cids in buckets.items() if cids}
                rust.move_states("active", _STEP_ROUND_DONE, None)
        finally:
            self.set_block_granular(was_granular)
            rust.set_drop_terminal_states(drop_terminal)
            rust.move_states(_STEP_ROUND_DONE, "active", None)
            rust.move_states(_STEP_ROUND_QUEUE, "active", None)
        self._release_gone_exports()
        return children

    def _step_event_keeps_going(self, event) -> bool:
        """Dispatch a pending callback from a step's ``run(1)`` event; return False when stepping should stop."""
        kind = event.event_type
        if kind == "need_callback":
            return not self._dispatch_callback(event)
        if kind == "errored":
            l.warning(f"Step error: {event.callback_reason}")
            return False
        return kind != "active_empty"

    @staticmethod
    def _drain_solver_graveyard() -> None:
        """Free solver forks stranded by off-thread GC (angr-87e56).

        Called on the coordinator (main) thread after each step()/explore().
        Under RUST_PARALLEL_WORKERS>1 an owned RustSolverContext whose proxy is
        collected on a scheduler worker cannot be dropped there (pyo3 refuses
        the unsendable cross-thread dealloc); its finalizer buries it and this
        drain closes it on the owning coordinator thread. No-op (cheap locked
        empty check) in the single-worker case, where forks close inline.
        """
        # cat-(a) EXPECTED CONTROL FLOW: best-effort cleanup; never let a
        # drain hiccup escape into the step loop.
        with suppress_recoverable():
            from angr.exploration.rust_state_proxy import drain_solver_graveyard

            drain_solver_graveyard()

    def _found_count(self) -> int:
        """Fast count of found states without triggering full state export/sync."""
        # Rust's found stash holds every find, predicate matches included
        # (adding _predicate_found would double-count them; angr-hc1yj).
        return len(self._rust_mgr.get_state_ids("found"))

    # ``active`` / ``avoid`` / ``deadended`` / ``unconstrained`` / ``pruned``
    # are generated from ``_STASH_ACCESSORS`` by ``_install_stash_accessors``;
    # only the stashes needing a non-trivial body are written out here.

    @property
    def found(self) -> list:
        """Get states in the found stash as angr SimStates.

        For SimulationManager API compatibility, this returns full angr states
        that can be used with state.solver.eval(), state.posix.dumps(), etc.
        Includes states found via callable predicates.
        """
        # Predicate matches are moved into Rust's found stash by both the
        # callback and the sweep path, so the stash is complete; merging a
        # second Python-side list listed every match twice (angr-hc1yj).
        return self._get_stash_states("found")

    @property
    def errored(self) -> list:
        """Get states in the errored stash as RustErrorRecord objects.

        Each RustErrorRecord has .state, .error, and .addr attributes,
        matching the interface of angr's ErrorRecord class.
        """
        states = self._get_stash_states("errored")
        if not states:
            return states

        # Build error lookup: state_id -> (addr, message)
        error_lookup = {}
        try:
            for addr, message, state_id in self._rust_mgr.get_errors():
                error_lookup[state_id] = (addr, message)
        except (RuntimeError, IndexError):
            # cat-(b) FALLBACK WITH LOSS: get_errors() failed; error records
            # are built with default (addr=0, message='unknown error') rather
            # than skipped, so callers still see the right number of states.
            pass

        # Map states to error records using state_ids from the stash
        state_ids = self._rust_mgr.get_state_ids("errored")
        records = []
        for i, state in enumerate(states):
            state_id = state_ids[i] if i < len(state_ids) else None
            addr, message = error_lookup.get(state_id, (0, "unknown error"))
            records.append(RustErrorRecord(state, message, addr))
        return records

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
            python_mgr=self,
        )

    # State export methods (_get_stash_states, _snapshot_to_angr, etc.)
    # are inherited from RustStateExportMixin in rust_state_export.py

    def _stash_proxies(self, stash_name: str) -> list:
        """Wrap every state_id in ``stash_name`` as a ``RustStateProxy``.

        O(1) per state — no SimState materialization, no cache sync. Used by
        the ``X_proxies()`` accessors below as a single chokepoint so the
        proxy-construction args stay in sync with ``mgr.proxy``.
        """
        return self._proxies_of(self._rust_mgr.get_state_ids(stash_name))

    def _proxies_of(self, state_ids) -> list:
        """Wrap each of *state_ids* as a ``RustStateProxy`` (see :meth:`_stash_proxies`)."""
        from angr.exploration.rust_state_proxy import RustStateProxy

        return [
            RustStateProxy(
                self._rust_mgr,
                sid,
                project=self._project,
                python_mgr=self,
            )
            for sid in state_ids
        ]

    # ``found_proxies`` / ``active_proxies`` / ``avoid_proxies`` /
    # ``deadended_proxies`` / ``unconstrained_proxies`` are generated from
    # ``_STASH_ACCESSORS`` by ``_install_stash_accessors``.

    def stash_counts(self) -> dict:
        """Get state counts for all stashes."""
        return dict(self._rust_mgr.stash_counts())

    @property
    def backend(self) -> RustExplorationManager:
        """This manager, so code reading the Rust-only API as ``mgr.backend.<member>`` also
        runs on a :class:`~angr.sim_manager.SimulationManager` whose backend is a
        :class:`~angr.exploration.rust_backend.RustBackend` (angr-mx00p.136.16)."""
        return self

    @property
    def stats(self) -> dict:
        """Get exploration statistics including instrumentation counters."""
        result = dict(self._rust_mgr.stats())
        # Add Python-side instrumentation counters
        result["callback_count"] = self._stats_callback_count
        result["ffi_crossings"] = self._stats_ffi_crossings
        result["state_creations"] = self._stats_state_creations
        result["cache_hits"] = self._stats_cache_hits
        result["cache_misses"] = self._stats_cache_misses
        result["technique_filter_calls"] = self._stats_technique_filter_calls
        result["hook_sync_calls"] = self._stats_hook_sync_calls
        result["hook_sync_skips"] = self._stats_hook_sync_skips
        result["time_in_callbacks"] = self._stats_time_in_callbacks_ns / 1e9  # seconds
        # angr-xtse.1: surface PerformanceTracker callback counts/times so
        # run_single.py --counters-json picks them up alongside the Rust-side
        # counters. Keys are kept verbatim ("callback_<kind>_count",
        # "callback_<kind>_total_ns") so downstream analysis can extract the
        # bucket by name without translation.
        #
        # angr-b00q: aggregate `python_callback_count` and
        # `python_callback_dispatch_us` give a single top-level signal for
        # "are Python round-trips dominating this run?" without forcing
        # callers to sum the per-kind buckets themselves.
        callback_total_count = 0
        callback_total_ns = 0
        for key, val in self._perf_stats.as_dict().items():
            if key.startswith("callback_"):
                result[key] = val
                if key.endswith("_count"):
                    callback_total_count += val
                elif key.endswith("_total_ns"):
                    callback_total_ns += val
        result["python_callback_count"] = callback_total_count
        result["python_callback_dispatch_us"] = callback_total_ns // 1000
        # angr-ymoe: orphan-BVS fallback counters
        result["orphan_bvs_mem_thunk"] = self._stats_orphan_bvs_mem_thunk
        result["orphan_bvs_sym_load_full_fail"] = self._stats_orphan_bvs_sym_load_full_fail
        # angr-4o7d: snapshot-restore orphan-BVS counter
        result["orphan_bvs_snapshot_restore"] = self._stats_orphan_bvs_snapshot_restore
        # angr-7jv5: proxy write-through FFI counters
        result["proxy_mem_concrete_writes"] = self._stats_proxy_mem_concrete_writes
        result["proxy_mem_ast_writes"] = self._stats_proxy_mem_ast_writes
        result["proxy_mem_symbolic_addr_fallback"] = self._stats_proxy_mem_symbolic_addr_fallback
        result["proxy_mem_fallback_python_load"] = self._stats_proxy_mem_fallback_python_load
        result["proxy_mem_automap_store_dropped"] = self._stats_proxy_mem_automap_store_dropped
        result["dirty_page_fetch_failed"] = self._stats_dirty_page_fetch_failed
        result["proxy_reg_writes"] = self._stats_proxy_reg_writes
        result["proxy_solver_adds"] = self._stats_proxy_solver_adds
        # angr-4ref8: symbolic-file export observability. `symfile_exports` is
        # the count of files handed to the native registry; each
        # `symfile_export_skip_<reason>` attributes a v1 scope-gate rejection
        # (subclass/has_end/not_seekable/file_exists/endness/size/path_utf8) or
        # a cat-(b) fallback (error/preamble). Complements the Rust-side
        # symfile_reads_native / symfile_write_demotions counters.
        result["symfile_exports"] = self._stats_symfile_exports
        for reason, count in self._stats_symfile_export_skips.items():
            result[f"symfile_export_skip_{reason}"] = count
        result["symfile_redemotions"] = self._stats_symfile_redemotions
        # angr-op0dn.11.7: Veritesting step_state() dispatch counters.
        result["veritesting_dispatches"] = self._stats_veritesting_dispatches
        result["veritesting_applied"] = self._stats_veritesting_applied
        # Add timing breakdown for predicate-mode exploration loop
        if hasattr(self, "_time_in_rust_run_ns"):
            result["time_in_rust_run"] = self._time_in_rust_run_ns / 1e9
            result["time_in_predicate_eval"] = self._time_in_predicate_eval_ns / 1e9
            result["time_in_active_check"] = self._time_in_active_check_ns / 1e9
        if hasattr(self, "_time_in_explore_ns"):
            result["time_in_explore"] = self._time_in_explore_ns / 1e9
        # Include Rust execution profiling stats if available
        try:
            rust_exec_stats = self._rust_mgr.get_execution_stats()
            for k, v in rust_exec_stats.items():
                result[f"rust_{k}"] = v
        except (RuntimeError, AttributeError):
            # cat-(b) FALLBACK WITH LOSS: Rust execution stats unavailable;
            # the stats dict still has Python-side counters.
            pass
        # Include Z3 solver profiling stats
        try:
            from angr.rustylib.vex_engine import RustExplorationManager as _REM

            solver_stats = _REM.get_solver_stats()
            for k, v in solver_stats.items():
                result[k] = v
        except (ImportError, RuntimeError, AttributeError):
            # cat-(b) FALLBACK WITH LOSS: Z3 solver stats unavailable (built
            # without Z3); the stats dict omits z3_* keys.
            pass
        return result

    def enable_profiling(self):
        """Enable Rust-side execution profiling for detailed timing breakdown."""
        self._rust_mgr.set_profiling(True)

    def disable_profiling(self):
        """Disable Rust-side execution profiling."""
        self._rust_mgr.set_profiling(False)

    def get_solver_stats(self) -> dict:
        """Return Z3 solver profiling counters as a dict.

        Counters are global (process-wide atomics) and accumulate across all
        SymContexts. Includes overall query/sat/unsat/timeout counts, total
        time spent in solver.check(), assume/branch fast-path counters, and
        per-call-site breakdowns.

        Returns an empty dict if the Rust extension was built without Z3.
        """
        try:
            from angr.rustylib.vex_engine import RustExplorationManager as _REM

            return dict(_REM.get_solver_stats())
        except (ImportError, RuntimeError, AttributeError):
            # cat-(b) FALLBACK WITH LOSS: Z3 solver stats unavailable on the
            # public getter; return an empty dict.
            return {}

    def reset_solver_stats(self):
        """Reset all global Z3 solver profiling counters to zero."""
        try:
            from angr.rustylib.vex_engine import RustExplorationManager as _REM

            _REM.reset_solver_stats()
        except (ImportError, RuntimeError, AttributeError):
            # cat-(b) FALLBACK WITH LOSS: solver-stats reset failed on public
            # resetter; counters keep accumulating from whatever state they
            # were in.
            pass

    # Compatibility methods for SimulationManager API

    # The SimulationManager kwargs of a manager the factory routed here because the project defaults to the Rust
    # engine (angr-mx00p.132.25); None for one the caller asked for by name. See _hand_off_to_python.
    _python_handoff_kwargs: dict | None = None

    def use_technique(self, technique, **kwargs):
        """Apply an exploration technique. See rust_techniques.use_technique()."""
        from angr.exploration.rust_techniques import technique_needs_python_engine, use_technique

        if self._python_handoff_kwargs is not None and self._rust_mgr.step_count == 0:
            python_reason = technique_needs_python_engine(technique)
            if python_reason is not None:
                self._hand_off_to_python(python_reason, stashes=self._handoff_stashes())
                return self.use_technique(technique)
        return use_technique(self, technique, **kwargs)

    def _mixed_lazy_solves(self) -> list[str]:
        """``["LAZY_SOLVES on only some active states"]`` when the active states disagree on it, else ``[]``.

        Rust's lazy-solves switch is manager-wide (``set_lazy_solves``), so a run over states that disagree would
        prune one side's forks as the other side asks (angr-vmi2l.241). Reads each state's options store, after
        taking in what a stash mirror gained or lost.
        """
        self._reconcile_mirror_options()
        if len(self._lazy_solves_of(self._rust_mgr.get_state_ids("active"))) > 1:
            return ["LAZY_SOLVES on only some active states"]
        return []

    def _lazy_solves_of(self, state_ids) -> set[bool]:
        """Whether each of ``state_ids`` has LAZY_SOLVES in its options store, as a set."""
        return {"LAZY_SOLVES" in self.get_state_options_py(sid) for sid in state_ids}

    def _latch_lazy_solves(self, state_ids) -> None:
        """Set Rust's manager-wide lazy-solves switch to what ``state_ids`` agree on, on or off.

        Called before every run, so an option added or removed since construction counts. A caller that steps
        states which disagree refuses first (:meth:`_mixed_lazy_solves`); with no states it is left as it is.
        """
        values = self._lazy_solves_of(state_ids)
        if len(values) == 1:
            self._rust_mgr.set_lazy_solves(values.pop())

    def _refuse_or_hand_off(self, method: str, unhonored: list[str]) -> bool:
        """Fail loudly on SimulationManager arguments *method* cannot honor; True when the manager handed off.

        A not-yet-stepped, default-routed manager hands off to the Python engine (:meth:`_hand_off_to_python`), so
        the caller re-dispatches *method* on the SimulationManager it became, with the states of
        :meth:`_handoff_stashes`; any other manager raises NotImplementedError naming each argument
        (angr-mx00p.132.23, angr-vmi2l.243).
        """
        if not unhonored:
            return False
        reason = f"{method}() cannot honor {', '.join(unhonored)}"
        if self._python_handoff_kwargs is None or self._rust_mgr.step_count != 0:
            raise NotImplementedError(f"RustExplorationManager.{reason}; build the manager without engine='rust'.")
        self._hand_off_to_python(reason, stashes=self._handoff_stashes())
        return True

    # State plugins the Rust engine cannot host: SimSymbolizer's init_state needs angr's paged memory, and its
    # mem_write BP action rewrites the store mid-step (angr-mx00p.132.37). See _hand_off_for_python_plugin.
    _PYTHON_ENGINE_PLUGINS = frozenset({"symbolizer"})

    def _hand_off_for_python_plugin(self, name: str) -> bool:
        """Hand off to the Python engine before plugin *name* first registers on one of this manager's states.

        The factory routes on the seeds, so a plugin the caller adds through ``simgr.one_active`` arrives too late
        for ``rust_engine_eligible``; the states carry over as :meth:`_handoff_stashes` gives them. Returns False,
        leaving the plugin to raise, for a manager that cannot hand off (stepped, or ``engine="rust"`` by name).
        """
        if name not in self._PYTHON_ENGINE_PLUGINS:
            return False
        if self._python_handoff_kwargs is None or self._rust_mgr.step_count != 0:
            return False
        self._hand_off_to_python(f"state plugin {name!r} needs the python engine", stashes=self._handoff_stashes())
        return True

    def _handoff_stashes(self) -> dict[str, list]:
        """The states a not-yet-stepped manager hands the Python engine: those it would hold had it never been Rust.

        Without an init seed they are this manager's stashes as they are now (:meth:`_carried_over_stashes`), so a
        write the caller made through ``simgr.one_active`` survives (angr-mx00p.114). The constructor already stepped
        an init seed to main on Python, where the Python engine would start at the seed itself, so then they are the
        pristine seeds with the caller's writes replayed onto them (:meth:`_record_handoff_write`,
        angr-mx00p.132.64), and every lazy ref handed out is pinned to the seed it descends from.
        """
        seeds = getattr(self, "_initial_seed_states", None) or []
        if not any(is_python_init_seed(self._project, seed.addr) for seed in seeds):
            return self._carried_over_stashes()
        writes = getattr(self, "_handoff_writes", None) or []
        seed_of = self._seed_of_state_id
        written = defaultdict(set)
        for state_id, _ in writes:
            written[seed_of[state_id]].add(state_id)
        if any(len(ids) > 1 for ids in written.values()):
            raise NotImplementedError(
                "RustExplorationManager cannot hand off to the python engine: writes were made to more than one "
                "state descended from one seed, and the python engine holds only the seed; build the manager "
                "without engine='rust'."
            )
        for state_id, replay in writes:
            replay(seeds[seed_of[state_id]])
        from angr.exploration.rust_state_export import _LazySimStateRef

        for refs in self.stashes.values():
            for ref in refs:
                if isinstance(ref, _LazySimStateRef) and ref._lazy_state_id in seed_of:
                    object.__setattr__(ref, "_lazy_pinned", seeds[seed_of[ref._lazy_state_id]])
        return {"active": seeds}

    def _record_handoff_write(self, binding, replay: Callable[[angr.SimState], object]) -> None:
        """Log a write a caller made through an exported stash state, for :meth:`_handoff_stashes` to replay.

        The Rust proxy plugins call this after every outermost write (``_ProxyStatePluginMixin._handoff_write``).
        Only a default-routed manager that has not stepped can still hand off, and only a write to a state its
        seeds placed (an export's own binding, not a copy's fork or a callback scope) lands in a stash the
        handoff rebuilds from the seed.
        """
        if self._python_handoff_kwargs is None or self._rust_mgr.step_count != 0:
            return
        from angr.exploration.rust_state_binding import CopyStashForkOwner

        state_id = binding.state_id
        if not isinstance(binding.owner, CopyStashForkOwner) or state_id not in self.__dict__.get(
            "_seed_of_state_id", ()
        ):
            return
        self.__dict__.setdefault("_handoff_writes", []).append((state_id, replay))

    def _carried_over_stashes(self) -> dict[str, list]:
        """This not-yet-stepped manager's stashes as Python states, for :meth:`_hand_off_to_python`.

        Each state is a copy given plugins of its own (``detach_for_python_engine``), so a write the caller made
        through an export survives the handoff, and every lazy ref handed out is pinned to its copy.
        """
        from angr.exploration.rust_state_export import _LazySimStateRef

        stashes = {}
        for stash, refs in self.stashes.items():
            states = []
            for ref in refs:
                if not isinstance(ref, _LazySimStateRef):
                    states.append(ref)
                    continue
                state = ref._materialize().copy()
                detach = getattr(state.plugins.get("memory"), "detach_for_python_engine", None)
                if detach is not None:
                    detach()
                object.__setattr__(ref, "_lazy_pinned", state)
                states.append(state)
            if states:
                stashes[stash] = states
        return stashes

    def _hand_off_to_python(self, reason: str, stashes: dict[str, list]) -> None:
        """Turn this not-yet-stepped, default-routed manager into a Python ``SimulationManager`` in place.

        The factory decides the engine before any technique exists, so a technique the Rust manager cannot honor
        arrives too late for ``rust_engine_eligible``. A project-default route never diverges (see
        docs/advanced-topics/rust_engine.rst), so the caller's reference becomes the manager the Python engine would
        have built: the states of :meth:`_handoff_stashes` (*stashes*: its ``active`` list seeds the manager and
        every other stash is populated as is), the caller's kwargs, and the techniques already registered. Swapping
        the class before ``SimulationManager.__init__`` runs keeps the hooks it installs bound to this object.
        """
        from angr.sim_manager import SimulationManager

        l.info("simulation_manager: handing off to the python engine (%s)", reason)
        project = self._project
        seeds = stashes.get("active", [])
        kwargs = self._python_handoff_kwargs
        techniques = list(self._active_techniques)
        _live_managers.discard(self)
        self.__dict__.clear()
        self.__class__ = SimulationManager
        SimulationManager.__init__(self, project, active_states=seeds, **kwargs)
        for stash, states in stashes.items():
            if stash != "active":
                self.populate(stash, states)
        for tech in techniques:
            self.use_technique(tech)

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

    def _has_technique_step_hooks(self) -> bool:
        """True iff an active technique has a non-native step() hook to dispatch."""
        if not self._active_techniques:
            return False
        from angr.exploration.rust_techniques import manager_has_step_hooks

        return manager_has_step_hooks(self)

    def _run_with_step_hooks(self, batch_size, native_run=None):
        """Run one batch under ExplorationTechnique step() hook composition.

        Returns the captured ExplorationEvent. When the step-hook stack never
        delegates to simgr.step() (e.g. a stash-only step hook), falls back to
        a direct run (``native_run`` when given, as for
        ``dispatch_step_with_hooks``) so the loop still makes progress.
        """
        from angr.exploration.rust_techniques import dispatch_step_with_hooks

        event = dispatch_step_with_hooks(self, batch_size, native_run=native_run)
        if event is None:
            event = native_run(batch_size) if native_run is not None else self._rust_mgr.run(batch_size)
        return event

    def _has_technique_step_state_hooks(self) -> bool:
        """True iff an active technique has a step_state() hook to dispatch.

        Veritesting (angr-op0dn.11.7) is the canonical case: it overrides
        step_state() to drive the CMU merging analysis. Checked ahead of the
        step() hook path in the run loops because a step_state hook needs the
        per-state export/re-import dispatch, not the batch step() proxy.
        """
        if not self._active_techniques:
            return False
        from angr.exploration.rust_techniques import manager_has_step_state_hooks

        return manager_has_step_state_hooks(self)

    def _run_with_step_state_hooks(self, batch_size, native_run=None):
        """Run one batch under ExplorationTechnique step_state() hook composition.

        Returns the ExplorationEvent (real, over the natively-advanced declined
        states, or a synthetic step_complete when every state was merged).
        Bumps the Veritesting dispatch/applied counters (angr-op0dn.11.7).
        """
        from angr.exploration.rust_techniques import dispatch_step_state_with_hooks

        self._stats_veritesting_dispatches += 1
        event, applied = dispatch_step_state_with_hooks(self, batch_size, native_run=native_run)
        self._stats_veritesting_applied += applied
        return event

    def run(self, **kwargs) -> RustExplorationManager:
        """Alias for explore() for SimulationManager compatibility.

        Handles step_func: if provided, called after EACH step (matching
        Python SimulationManager behavior). Without step_func, delegates
        to explore() for batch execution.
        """
        step_func = kwargs.pop("step_func", None)
        n = kwargs.pop("n", None)

        # A bounded `n` means SimulationManager.run(n=N) semantics: step at most
        # N times (or until the watched stash empties), exactly like calling
        # step() N times. explore() IGNORES `n` (it lands in **kwargs) and runs
        # to completion, which silently over-runs the requested budget. For a
        # script that does `sm.run(n=4); sm.step(...); sm.active[0]` (e.g.
        # ekopartyctf2015_rev100, asisctffinals2015_license) the over-run drives
        # the lone state into a deadend that drop_terminal_states then discards,
        # leaving every stash empty -> IndexError on active[0]/found[0]
        # (angr-58v9a). Only delegate to the run-to-completion explore() path
        # when neither `n` nor step_func is given.
        if n is None and step_func is None:
            # A default-routed manager runs one block per pop, as its step() does: a chained step runs the
            # fall-through arm of a fork on to its deadend while the exit-target arm waits in active, so the
            # deadended stash lists the arms in the reverse of SimulationManager.run()'s order
            # (angr-mx00p.132.61; ``block-granular-step-mode``).
            was_granular = self.set_block_granular(True) if self._python_handoff_kwargs is not None else None
            try:
                return self.explore(**kwargs)
            finally:
                if was_granular is not None:
                    self.set_block_granular(was_granular)

        # Bounded / step_func mode: step one at a time. step_func (used by
        # Callable for concrete_only pruning) is applied after each step,
        # matching Python SimulationManager.run() behavior. Keep terminal
        # states so they land in their stash (deadended/errored) instead of
        # being dropped, matching Python. Each step advances one basic block,
        # as a Python step does: with the interpreter's block chaining on, N
        # steps run far past the budget the caller calibrated against Python
        # (test_serialization's run(n=10) reached a divide-by-zero the Python
        # engine is nowhere near after ten steps; angr-mx00p.132.22).
        drop_terminal = self._rust_mgr.stats()["drop_terminal_states"]
        self._rust_mgr.set_drop_terminal_states(False)
        was_granular = self.set_block_granular(True)
        try:
            stash = kwargs.pop("stash", "active")
            until = kwargs.pop("until", None)
            import itertools

            for _ in itertools.count() if n is None else range(n):
                if not self._rust_mgr.get_state_ids(stash) or self._check_technique_complete():
                    break
                self.step(**kwargs)
                if step_func is not None:
                    step_func(self)
                if until and until(self):
                    break
        finally:
            self.set_block_granular(was_granular)
            self._rust_mgr.set_drop_terminal_states(drop_terminal)
        return self

    def _evaluate_predicate_on_stash(self, stash: str, filter_func) -> tuple[list[int], list[int], list[int]]:
        """Evaluate ``filter_func`` against every state in ``stash``.

        Shared backbone for move()/prune()/drop(),
        which previously hand-rolled this same proxy-then-full-export block
        near-identically (angr-hv4lt.3). Tries the lightweight
        ``RustStateProxy`` first; on ``(AttributeError, TypeError,
        NotImplementedError)`` — the predicate touched something the proxy
        doesn't expose — falls back to a full ``export_state`` +
        ``_snapshot_to_angr`` and re-runs the predicate against the py_state.

        Returns ``(matched_ids, unmatched_ids, errored_ids)``:
          - matched_ids: predicate returned truthy,
          - unmatched_ids: predicate returned falsy,
          - errored_ids: predicate raised an unexpected exception (cat-(b)
            fallback-with-loss; the state is neither matched nor unmatched so
            callers leave it in place). Debug-logged.
        """
        from angr.exploration.rust_state_proxy import RustStateProxy

        matched_ids: list[int] = []
        unmatched_ids: list[int] = []
        errored_ids: list[int] = []

        for state_id in list(self._rust_mgr.get_state_ids(stash)):
            try:
                proxy = RustStateProxy(self._rust_mgr, state_id, self._project, python_mgr=self)
                try:
                    matched = bool(filter_func(proxy))
                except (AttributeError, TypeError, NotImplementedError):
                    # cat-(a) EXPECTED CONTROL FLOW: proxy didn't support an
                    # attribute the predicate accessed; fall back to full export.
                    snapshot = self._rust_mgr.export_state(state_id)
                    py_state = self._snapshot_to_angr(snapshot)
                    matched = bool(filter_func(py_state))
                (matched_ids if matched else unmatched_ids).append(state_id)
            except MemoryError:
                raise
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: predicate raised on a state; leave
                # it where it is rather than moving/dropping it. Debug-logs.
                if _DBG:
                    l.debug(f"predicate error for state {state_id} in {stash}: {e}")
                errored_ids.append(state_id)

        return matched_ids, unmatched_ids, errored_ids

    def move(self, from_stash: str, to_stash: str, filter_func=None) -> RustExplorationManager:
        """Move states between stashes.

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
            # Export states, evaluate predicate, and handle accordingly.
            # Matched states move; unmatched/errored stay in the source stash.
            move_ids, _keep_ids, _errored_ids = self._evaluate_predicate_on_stash(from_stash, filter_func)

            # Use Rust to move matching states
            for state_id in move_ids:
                # cat-(a) EXPECTED CONTROL FLOW: state may already have moved
                # (e.g., another technique deadended it). Suppress.
                # State may have already been moved
                with contextlib.suppress(RuntimeError, KeyError):
                    self._rust_mgr.move_state(state_id, from_stash, to_stash)

        return self

    def stash(self, filter_func=None, from_stash="active", to_stash="stashed") -> RustExplorationManager:
        """Stash some states. Alias for move() with different defaults."""
        return self.move(from_stash, to_stash, filter_func=filter_func)

    def unstash(self, filter_func=None, to_stash="active", from_stash="stashed") -> RustExplorationManager:
        """Unstash some states. Alias for move() with different defaults."""
        return self.move(from_stash, to_stash, filter_func=filter_func)

    def prune(self, filter_func=None, from_stash="active", to_stash="pruned") -> RustExplorationManager:
        """Move the unsatisfiable states of ``from_stash`` to ``to_stash``.

        Same contract as ``SimulationManager.prune``: a state is pruned iff
        ``filter_func`` is None or matches it, *and* it is unsat. A
        satisfiable state stays put whatever the filter says (angr-vmi2l.235).
        Satisfiability goes through the Rust solver directly, so no state is
        exported unless the filter needs more than the proxy exposes.

        Args:
            filter_func: Only prune states that match this predicate.
            from_stash: Prune states from this stash. Defaults to 'active'.
            to_stash: Put pruned states in this stash. Defaults to 'pruned'.

        Returns:
            Self, for chaining.
        """
        if filter_func is None:
            candidate_ids = list(self._rust_mgr.get_state_ids(from_stash))
        else:
            candidate_ids, _unmatched_ids, _errored_ids = self._evaluate_predicate_on_stash(from_stash, filter_func)
        prune_ids = []
        for state_id in candidate_ids:
            try:
                if not self._rust_mgr.state_satisfiable(state_id):
                    prune_ids.append(state_id)
            except (RuntimeError, KeyError):
                # cat-(b) FALLBACK WITH LOSS: state_satisfiable() raised; keep the
                # state — if it's actually unsat, downstream solver use will catch.
                pass  # Keep on error
        for state_id in prune_ids:
            # cat-(a) EXPECTED CONTROL FLOW: state may already have moved.
            with contextlib.suppress(RuntimeError, KeyError):
                self._rust_mgr.move_state(state_id, from_stash, to_stash)
        return self

    def drop(self, filter_func=None, stash: str = "active") -> RustExplorationManager:
        """Drop states from a stash, with ``SimulationManager.drop``'s signature and contract.

        States matching the predicate (or all if no predicate) are discarded, as upstream's
        ``move(stash, DROP, filter_func)`` does; they do not land in another stash (angr-vmi2l.236).

        Args:
            filter_func: Optional callable predicate. If provided, only states
                        matching the predicate are dropped.
            stash: The stash to drop from. Defaults to 'active'.

        Returns:
            Self, for chaining.
        """
        if filter_func is None:
            # Drop all states from the stash. An export of a dropped state
            # keeps reading it as it was (``CopyStashForkOwner.preserve``).
            self._preserve_exports(self._rust_mgr.get_state_ids(stash))
            try:
                self._rust_mgr.clear_stash(stash)
            except AttributeError:
                # cat-(a) EXPECTED CONTROL FLOW: probing for the optional
                # clear_stash API on older Rust builds; fall back to move_states.
                # Fallback: move all to deadended
                self._rust_mgr.move_states(stash, "deadended", None)
        else:
            # Drop states matching predicate; unmatched/errored stay in place.
            drop_ids, _keep_ids, _errored_ids = self._evaluate_predicate_on_stash(stash, filter_func)
            self._preserve_exports(drop_ids)
            for state_id in drop_ids:
                self._rust_mgr.drop_state_from_stash(state_id, stash)

        return self

    def split(
        self,
        stash_splitter=None,
        stash_ranker=None,
        state_ranker=None,
        limit: int = 8,
        from_stash: str = "active",
        to_stash: str = "stashed",
    ) -> RustExplorationManager:
        """Split ``from_stash`` in two, with ``SimulationManager.split``'s signature and contract.

        ``stash_splitter`` overrides ``stash_ranker``, which overrides ``state_ranker``; with none of them the first
        ``limit`` states stay. ``from_stash`` keeps the kept states in the order the callable returned them, and
        ``to_stash == DROP`` drops the rest (angr-mx00p.132.35).

        Returns:
            Self, for chaining.
        """
        state_ids = list(self._rust_mgr.get_state_ids(from_stash))
        if stash_splitter is None and stash_ranker is None and state_ranker is None:
            keep_ids, split_ids = state_ids[:limit], state_ids[limit:]
            reorder = False
        else:
            refs = self._get_stash_states(from_stash)
            sid_of = {id(ref): ref._lazy_state_id for ref in refs}

            def to_ids(states) -> list[int]:
                try:
                    return [sid_of[id(st)] for st in states]
                except KeyError:
                    raise ValueError(
                        f"split(): a ranker or splitter returned a state that is not in stash {from_stash!r}"
                    ) from None

            if stash_splitter is not None:
                keep, split = stash_splitter(refs)
            elif stash_ranker is not None:
                ranked = list(stash_ranker(refs))
                keep, split = ranked[:limit], ranked[limit:]
            else:
                ranked = sorted(refs, key=state_ranker)
                keep, split = ranked[:limit], ranked[limit:]
            keep_ids, split_ids = to_ids(keep), to_ids(split)
            reorder = True

        if to_stash == self.DROP:
            self._preserve_exports(split_ids)
            for state_id in split_ids:
                self._rust_mgr.drop_state_from_stash(state_id, from_stash)
        else:
            for state_id in split_ids:
                self._rust_mgr.move_state(state_id, from_stash, to_stash)
        if reorder:
            # Rust appends on move, so a round trip through a staging stash leaves from_stash in keep_ids order.
            for state_id in keep_ids:
                self._rust_mgr.move_state(state_id, from_stash, "_split_keep")
            self._rust_mgr.move_states("_split_keep", from_stash, None)

        return self

    @property
    def stashes(self) -> defaultdict[str, list]:
        """Every stash's states, as ``SimulationManager.stashes`` returns them.

        Covers each stash Rust holds (technique-declared ones such as ``cut`` and ``spinning`` included, the
        ``_step_round_*`` staging stashes excluded) plus the standard names, and reads a missing name as ``[]``
        (angr-9ij0m).
        """
        result = defaultdict(list)
        for name in self._stash_names():
            result[name] = self._get_stash_states(name)
        return result

    def _stash_names(self) -> dict[str, None]:
        """The stash names this manager answers for, in order: the standard ones (a name Rust has not created
        yet reads as empty, as ``SimulationManager``'s integral stashes do) plus every non-staging stash Rust
        holds, empty ones included. ``get_state_ids`` reads any other name as empty, so this set, not Rust,
        decides whether a name is a stash (angr-vmi2l.238)."""
        names = dict.fromkeys(
            ["active", "found", "avoid", "deadended", "errored", "unconstrained", "pruned", "stashed", "unsat"]
        )
        names.update(dict.fromkeys(n for n in self._rust_mgr.stash_counts() if not n.startswith("_")))
        return names

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

    def copy(self, deep: bool = False) -> RustExplorationManager:
        """Refuse: this manager cannot be copied (angr-vmi2l.239).

        ``SimulationManager.copy`` returns an independent manager to use as a
        checkpoint. Every stash here lives in one Rust manager, so returning
        ``self`` would alias the original: stepping the "copy" would step it.
        A ``SimulationManager`` built by ``project.factory.simgr`` runs on the
        Rust engine through its backend and supports ``copy``.
        """
        raise NotImplementedError(
            "RustExplorationManager.copy() is not supported: its stashes live in "
            "one Rust manager, so a copy would alias the original. Use "
            "project.factory.simgr(...), whose copy() works on the Rust backend."
        )

    def _native_merge_key(self, state_id: int) -> tuple | None:
        """Upstream's ``SimulationManager._merge_key`` read from Rust, or None.

        The pc, the call stack's function addresses and the set of open fds,
        so two states at one pc under different callers or fd tables stay
        apart (angr-vmi2l.240). Rust's root frame and closed-fd tombstones
        are dropped on every state alike, so equal keys mean equal upstream
        keys. None (a symbolic pc, or a lookup that failed) sends every state
        to the export path, which applies upstream's key itself.
        """
        pc = self._rust_mgr.get_state_pc_by_id(state_id)
        if pc is None:
            return None
        try:
            frames = self._rust_mgr.get_state_call_stack(state_id)
            fds = self._rust_mgr.get_state_open_fds(state_id)
        except (RuntimeError, KeyError) as e:
            # cat-(a) EXPECTED CONTROL FLOW: the export path groups this state
            # on the same key.
            l.debug("native merge key for state %d unavailable: %s", state_id, e)
            return None
        return pc, tuple(frame[1] for frame in frames), frozenset(info[0] for info in fds if info[5])

    def _merge_native(self, stash: str, state_ids: list[int]) -> list[int] | None:
        """Merge native states in-Rust, no export round trip (M3-4).

        Groups ``state_ids`` on ``_native_merge_key``, Rust's reading of
        upstream's default ``SimulationManager._merge_key`` (angr-vmi2l.240),
        then calls ``merge_states`` per multi-member group. The Rust ``merge_states``
        forks + merges the source states and pushes the result to ``stash`` but
        does NOT remove the sources, so we drop them afterwards (matching the
        Python path's ``_merge_drop`` dance).

        The native path never exports state, so the symbolic-file re-demotion
        dance is unnecessary (angr-qluof): the merged state inherits its base's demoted
        symbolic-file paths directly (no ``_export_fs_files_to_rust`` re-arms
        them). ``states_merged_native`` is surfaced Rust-side by ``merge_states``.

        A group whose stdin, stdout or stderr differ across members is left
        for the Python merge (``merge_refusal``, angr-vmi2l.199): the native
        merge keeps one member's streams, while ``SimPackets.merge``
        ITE-merges them or raises ``SimMergeError``.

        Returns:
            The ids of the groups left for the Python export path (empty when
            the native path handled every group), or None if a key lookup
            failed and the caller should merge every state in Python.
        """
        groups: dict[tuple, list[int]] = {}
        for sid in state_ids:
            key = self._native_merge_key(sid)
            if key is None:
                return None
            groups.setdefault(key, []).append(sid)

        merged_any = False
        left_for_python: list[int] = []
        for group in groups.values():
            if len(group) <= 1:
                # Singleton stays in the stash untouched.
                continue
            refusal = self._rust_mgr.merge_refusal(group)
            if refusal is not None:
                l.debug("merging group %s in Python: %s", group, refusal)
                left_for_python.extend(group)
                continue
            try:
                self._rust_mgr.merge_states(group, stash)
            except (RuntimeError, ValueError) as e:
                # cat-(b) FALLBACK WITH LOSS: native merge failed for this group;
                # leave its states unmerged and continue with other groups.
                l.warning("native merge_states failed for group %s: %s", group, e)
                continue
            merged_any = True
            # Drop the now-merged source states from the stash.
            for sid in group:
                # cat-(a) EXPECTED CONTROL FLOW: source already moved.
                with contextlib.suppress(RuntimeError, KeyError):
                    self._rust_mgr.move_state(sid, stash, "_merge_drop")
        if merged_any:
            # An export of a merged-away state keeps reading it as it was.
            self._preserve_exports(self._rust_mgr.get_state_ids("_merge_drop"))
            # cat-(a) EXPECTED CONTROL FLOW: intermediate stash never created.
            with contextlib.suppress(RuntimeError, KeyError):
                self._rust_mgr.clear_stash("_merge_drop")
        return left_for_python

    def merge(self, merge_func=None, merge_key=None, stash: str = "active", prune=True) -> RustExplorationManager:
        """Merge states in a stash, with ``SimulationManager.merge``'s signature.

        ``prune`` prunes ``stash`` first, and the default ``merge_key`` is
        upstream's ``_merge_key`` (pc, call-stack function addresses, open
        fds), read natively by ``_native_merge_key`` (angr-vmi2l.240). Groups
        the native merge cannot take are exported, merged in Python, and
        re-imported. A group whose merge fails stays unmerged.
        """
        if prune:
            self.prune(from_stash=stash)
        state_ids = list(self._rust_mgr.get_state_ids(stash))
        if len(state_ids) <= 1:
            return self

        # M3-4 (angr-op0dn.11.4): native fast path. When there is no custom
        # merge_func and grouping uses the default key (s.addr == pc), every
        # group member is a Rust-native state, so we can merge in-Rust via
        # merge_states() and skip the export -> Python state.merge() ->
        # re-import round trip entirely. A custom merge_func or merge_key needs
        # exported SimStates, so those fall through to the Python path below.
        if merge_func is None and merge_key is None:
            left_for_python = self._merge_native(stash, state_ids)
            if left_for_python is not None:
                if not left_for_python:
                    return self
                state_ids = left_for_python

        # Export all states to Python SimStates for merging
        try:
            py_states = []
            # Map each exported py_state back to its source Rust state id so
            # the merged re-add can inherit the group's demoted symbolic-file
            # paths (angr-qluof lineage-aware demotion). Queried BEFORE the
            # _merge_drop below removes the source states.
            sid_by_state_key = {}
            demoted_by_sid = {}
            for sid in state_ids:
                py_state = self.get_state_by_id(sid)
                if py_state is not None:
                    # Python merges it and the result is re-imported after the
                    # _merge_drop below removes ``sid``, so it needs plugins of
                    # its own rather than proxies bound to ``sid``
                    # (angr-vmi2l.314, angr-mx00p.61). Its Python solver holds
                    # the Rust path constraints, as SimStateSolver.merge ORs
                    # each state's constraints under its merge condition, and
                    # stdin is the symbols Rust read, so the merged packets are
                    # solvable under each condition (angr-mx00p.132.54).
                    self._prepare_for_python_step(py_state, sid, rust_solver=False)
                    py_states.append(py_state)
                    sid_by_state_key[id(py_state)] = sid
                    try:
                        demoted_by_sid[sid] = self._rust_mgr.get_demoted_paths(sid)
                    except MemoryError:
                        raise
                    except Exception as e:
                        # cat-(b) FALLBACK WITH LOSS: a missing demoted-path query just means the
                        # merged state may re-arm a native write-demotion (the
                        # documented, content-harmless v1 limitation).
                        l.debug("get_demoted_paths(%d) failed: %s: %s", sid, type(e).__name__, e)

            if len(py_states) <= 1:
                return self

            if merge_key is None:
                from angr.sim_manager import SimulationManager

                merge_key = SimulationManager._merge_key

            # Grouped by equality, as upstream does: its default key holds a
            # list and a set, so it is not hashable.
            groups: list[tuple[object, list]] = []
            for s in py_states:
                key = merge_key(s)
                for group_key, members in groups:
                    if group_key == key:
                        members.append(s)
                        break
                else:
                    groups.append((key, [s]))

            # Each entry pairs a state to re-add with the set of symbolic-file
            # paths its source lineage(s) demoted, so the re-add can re-apply
            # the demotion the export step re-arms (angr-qluof).
            merged = []

            def _group_demoted(grp):
                paths = set()
                for s in grp:
                    sid = sid_by_state_key.get(id(s))
                    if sid is not None:
                        paths.update(demoted_by_sid.get(sid, ()))
                return paths

            for key, group in groups:
                group_demoted = _group_demoted(group)
                if len(group) <= 1:
                    # Unmerged single state keeps its own demoted paths.
                    merged.extend((s, _group_demoted([s])) for s in group)
                elif merge_func is not None:
                    try:
                        merged.append((merge_func(*group), group_demoted))
                    except (TypeError, ValueError, RuntimeError):
                        # cat-(b) FALLBACK WITH LOSS: user merge_func failed for this
                        # group; keep the group's states unmerged. Already warns.
                        l.warning("merge_func failed for group at %s, keeping unmerged", key)
                        merged.extend((s, _group_demoted([s])) for s in group)
                else:
                    try:
                        base = group[0]
                        others = group[1:]
                        m, _, _ = base.merge(*others)
                        merged.append((m, group_demoted))
                    except (AttributeError, TypeError, ValueError, SimMergeError, ClaripyError):
                        # cat-(b) FALLBACK WITH LOSS: built-in state.merge() failed;
                        # keep the group unmerged. Already warns.
                        l.warning("State merge failed for group at %s, keeping unmerged", key)
                        merged.extend((s, _group_demoted([s])) for s in group)

            # Clear the Rust stash and re-add merged states
            for sid in state_ids:
                # cat-(a) EXPECTED CONTROL FLOW: source stash entry already moved.
                with contextlib.suppress(RuntimeError, KeyError):
                    self._rust_mgr.move_state(sid, stash, "_merge_drop")
            # An export of a merged-away state keeps reading it as it was.
            self._preserve_exports(self._rust_mgr.get_state_ids("_merge_drop"))
            # cat-(a) EXPECTED CONTROL FLOW: clear of intermediate _merge_drop
            # stash failed (already empty / never created).
            with contextlib.suppress(RuntimeError, KeyError):
                self._rust_mgr.clear_stash("_merge_drop")

            # Re-add merged states
            for ms, ms_demoted in merged:
                try:
                    new_sid = self._add_rust_state(stash, ms)
                    l.debug("Added merged state to %s at 0x%x", stash, ms.addr)
                    # Lineage-aware demotion (angr-qluof): _add_rust_state's
                    # _export_fs_files_to_rust re-registers eligible SimFiles,
                    # re-arming any path an ancestor's native write had demoted.
                    # Re-apply those demotions on the new state so the guest
                    # keeps seeing the (content-identical) Python fallback.
                    if new_sid is not None and ms_demoted:
                        for path in ms_demoted:
                            try:
                                if self._rust_mgr.demote_file_path(new_sid, path):
                                    self._stats_symfile_redemotions += 1
                            except MemoryError:
                                raise
                            except Exception as e:
                                # cat-(b) FALLBACK WITH LOSS: as in _reapply_demoted_paths.
                                l.debug(
                                    "re-demote %r on state %s failed: %s: %s",
                                    path,
                                    new_sid,
                                    type(e).__name__,
                                    e,
                                )
                except MemoryError:
                    raise
                except Exception as e:
                    # cat-(c) WRONG-ANSWER RISK: failed to re-add merged state; the
                    # merge result is lost — caller sees fewer states than expected.
                    # Already warns.
                    l.warning("Failed to re-add merged state: %s", e)

        except MemoryError:
            raise
        except Exception:
            # cat-(c) WRONG-ANSWER RISK: outer merge raised; states are left
            # unmerged (some already moved to _merge_drop and cleared above).
            # Already warns with exc_info.
            l.warning("State merge failed entirely, keeping states unmerged", exc_info=True)

        return self

    def dump_snapshot(self, path: str) -> None:
        """Write a stash-manager snapshot to ``path`` (opt-in, angr-x04s.1.4).

        Captures every state in every stash via the Rust-side
        :class:`StashManager` codec (bucket A/B/C — pc, registers, memory,
        history, fs, posix, call_stack, …) plus the
        ``SymContext::assumed_constraints`` log per state. Bucket-D
        ``Py<PyAny>`` overlays (``symbolic_pages`` /
        ``hook_symbolic_memory`` / ``addr_to_ast``) are NOT captured by
        Rust and are restored empty; for fauxware-level workflows that's
        a non-issue because the Rust SimProcedures + native memory plugin
        keep those overlays empty.

        Args:
            path: Filesystem path to write the snapshot to. Overwrites if
                the file exists.

        The format-version byte at envelope head lets :meth:`load_snapshot`
        reject a stale snapshot fast (see :exc:`ValueError`).

        Known limitation (prototype scope, angr-x04s.1):
            Constraints added via :meth:`add_constraint_raw` (the path the
            Python claripy-sync layer uses for initial-state constraints)
            are NOT serialized — only the ``assume_true``/``assume_false``-
            tracked subset is. After restore, the resumed solver may have
            fewer constraints than the original and consequently solve to
            a different model. End-to-end equality of ``posix.dumps(0)``
            across a snapshot round-trip is therefore not guaranteed.

        Live parallel sessions (angr-op0dn.13.6):
            Under the steady-state parallel loop the frontier is resident in
            the worker Z3 contexts and belongs to no stash, so a mid-session
            dump would silently omit it. The session is finalized (frontier
            drained back into the active stash) before the capture, both here
            and again Rust-side inside ``dump_snapshot_bytes``.
        """
        self._finalize_parallel_session()
        bytes_blob = bytes(self._rust_mgr.dump_snapshot_bytes())
        payload = self._capture_bucket_d()
        stdin_content = self._capture_seed_stdin_content()
        if stdin_content:
            payload[_SEED_STDIN_KEY] = stdin_content
        overlays = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path, "wb") as f:
            f.write(_SNAPSHOT_MAGIC)
            f.write(struct.pack("<Q", len(bytes_blob)))
            f.write(bytes_blob)
            f.write(overlays)

    def _capture_bucket_d(self) -> dict[int, dict[str, dict]]:
        """Collect the per-state Python-AST overlays for every stashed state.

        The Rust ``StashManager`` codec cannot serialize the ``Py<PyAny>``
        overlays (``symbolic_pages`` / ``hook_symbolic_memory`` /
        ``addr_to_ast``), so it restores them empty. A state that bounced
        through a *Python* SimProcedure keeps its post-bounce symbolic values
        only in those overlays: drop them and the resumed state reads those
        addresses as unconstrained, never forks on them, and the whole
        deferred-fork subtree below it is lost (angr-op0dn.13.14).
        """
        out: dict[int, dict[str, dict]] = {}
        for stash in self.stash_counts():
            for sid in self._rust_mgr.get_state_ids(stash):
                pages = dict(self._rust_mgr.get_state_symbolic_pages(sid))
                hook_mem = dict(self._rust_mgr.get_state_hook_symbolic_memory(sid))
                addr_map = dict(self._rust_mgr.get_state_addr_to_ast(sid))
                if pages or hook_mem or addr_map:
                    out[sid] = {
                        "symbolic_pages": pages,
                        "hook_symbolic_memory": hook_mem,
                        "addr_to_ast": addr_map,
                    }
        return out

    def _capture_seed_stdin_content(self) -> list | None:
        """The harness-seeded ``posix.stdin.content`` ASTs, for the envelope.

        A snapshot restores a state's constraints — which, for a harness that
        seeded stdin itself (see :meth:`_seed_stdin_to_rust`), are phrased over
        the *original* manager's stdin BVS. The resumed manager's own seed state
        carries a freshly-minted claripy symbol with a different name, so
        ``found[i].posix.dumps(0)`` solves an unconstrained variable and comes
        back all-zero — identical across every found state (angr-op0dn.13.14).

        Carrying the original byte ASTs in the envelope keeps the names aligned:
        claripy pickling preserves them, so once :meth:`load_snapshot` installs
        them as ``_stdin_content`` — the list the materialization tail
        (``_finalize_materialized_state``) grafts onto any state whose stdin came
        back empty — the restored content and the restored constraints talk about
        the same symbol again.
        """
        roots = (self._state_cache.get(rid) for rid in self._state_roots.values())
        # Roots first (the pristine seed), but any cached state will do: a
        # harness-seeded stdin stream is never rewritten mid-exploration, so a
        # descendant carries the very same byte ASTs. `_cleanup_state_cache` can
        # evict the root outright, and a manager that never materialized a state
        # has an empty cache — hence the `_initial_seed_states` backstop.
        seeds = getattr(self, "_initial_seed_states", None) or ()
        for state in chain(roots, self._state_cache.values(), seeds):
            if state is None:
                continue
            content = getattr(getattr(getattr(state, "posix", None), "stdin", None), "content", None)
            if content:
                return list(content)
        return None

    def _restore_bucket_d(self, overlays: dict[int, dict[str, dict]]) -> None:
        """Inverse of :meth:`_capture_bucket_d`. Unknown state ids are skipped
        by the Rust setters, so a partial stash restore stays safe."""
        for sid, buckets in overlays.items():
            if sid == _SEED_STDIN_KEY:
                continue
            pages = buckets.get("symbolic_pages")
            if pages:
                self._rust_mgr.set_state_symbolic_pages(sid, pages)
            for addr, (ast, size) in buckets.get("hook_symbolic_memory", {}).items():
                self._rust_mgr.set_state_hook_symbolic_memory(sid, addr, ast, size)
            for addr, (ast, size) in buckets.get("addr_to_ast", {}).items():
                self._rust_mgr.set_state_addr_to_ast(sid, addr, ast, size)

    def load_snapshot(self, path: str) -> None:
        """Restore a stash-manager snapshot from ``path`` (opt-in; angr-x04s.1.4).

        Inverse of :meth:`dump_snapshot`. Replaces every stash in this
        manager wholesale; manager-level configuration (find/avoid addrs,
        hooks, simprocedures, solver/memory config) is preserved.

        Args:
            path: Filesystem path to read the snapshot from.

        Raises:
            ValueError: When the envelope is empty or carries a stale
                format-version byte.
        """
        with open(path, "rb") as f:
            blob = f.read()
        overlays: dict[int, dict[str, dict]] = {}
        if blob.startswith(_SNAPSHOT_MAGIC):
            head = len(_SNAPSHOT_MAGIC)
            (rust_len,) = struct.unpack_from("<Q", blob, head)
            head += struct.calcsize("<Q")
            bytes_blob = blob[head : head + rust_len]
            overlays = pickle.loads(blob[head + rust_len :])
        else:
            # Pre-bucket-D envelope (angr-x04s.1.4): bare Rust bytes.
            bytes_blob = blob
        self._rust_mgr.load_snapshot_bytes(bytes_blob)
        # Re-point the materialization-time stdin restore
        # (`_finalize_materialized_state`) at the ASTs the snapshot was taken
        # over, so the exported posix and the restored constraints name the same
        # claripy symbols. See `_capture_seed_stdin_content`.
        seed_stdin = overlays.get(_SEED_STDIN_KEY)
        if seed_stdin:
            self._stdin_content = list(seed_stdin)
        self._restore_bucket_d(overlays)
        # The post-restore Rust state_ids are the original ones (snapshot
        # preserves them), so external state-export caches indexed by id
        # must be flushed.
        self._invalidate_state_export_cache(detach_held=False)
        self._reset_python_shadow_maps_for_restore()
        # angr-op0dn.13.12: the restored frontier — not the state this manager
        # was constructed with — is now the exploration's root. The phase-2
        # eager retry (angr-027h) re-seeds `_initial_seed_states`, i.e. the caller's
        # pre-load placeholder state (`load_from_disk` passes none at all), which
        # would restart the exploration from a path the user never asked to
        # resume and additionally flip `use_deferred_forks` off globally (on the
        # parallel path that drops the stored branch conditions and forks
        # unconstrained). The mid-path frontier cannot be replayed eagerly from
        # its own states, so phase 2 is simply disabled for a resumed manager.
        self._initial_seed_states = None
        self._phase2_retried = True

    def _reset_python_shadow_maps_for_restore(self) -> None:
        """Drop the id-keyed Python shadow maps across a snapshot restore.

        The Python counterpart of ``load_snapshot_bytes`` clearing the Rust
        ``ConstraintTracker``'s per-run sets (angr-fs8kb.89).

        ``StashManager::from_snapshot`` rebuilds ``state_index`` from each
        state's own serialized ``state_id``, so a restore into a *live* manager
        resurrects the original ids — the one case the
        ``invariant-state-id-never-reused`` guarantee does not cover. Every
        structure below is keyed by that id and holds a *verdict* or a *lineage*
        computed against the pre-load world, neither of which
        :meth:`_invalidate_state_export_cache` can repair (it only unsets
        ``scratch.rust_fully_synced``, which re-reads a mirror's contents from
        Rust — the right treatment for ``_state_cache``, which holds mirrorable
        state *contents*, and the wrong one for these).

        A restore is a whole-world replacement, not a rewind within one
        exploration, so the recorded decision for each of these is *clear*:

        * ``_predicate_matched_ids`` — :meth:`_evaluate_predicates_on_active`
          ``continue``\\ s outright on a member, so a restored state whose id
          matched pre-dump would never have its find/avoid predicate evaluated
          again: a silently lost ``found``. Direct mirror of .89's
          ``skip_find_predicate_states`` / ``skip_avoid_predicate_states``.
        * ``_predicate_eval_cache`` — ``state_id -> (addr, stdout_len)``
          change-detection memo. A restored state sitting at the same pair as
          its pre-dump self would reuse a verdict computed under different
          constraints.
        * ``_state_roots`` — pre-load lineage. A stale entry maps a restored id
          to a root from a different lineage, which both misdirects plugin
          restoration and pins the wrong entries in
          :meth:`_cleanup_state_cache`. Rust carries its own roots, so the map
          rebuilds itself from ``get_state_root``.
        * ``_py_state_options`` / ``_py_state_globals`` — user-visible
          stand-ins the snapshot does not capture at all, so a surviving entry
          hands a *different* state's options/globals to the restored id.
          Clearing is lossless: :meth:`get_state_options_py` /
          :meth:`get_state_globals_py` re-seed lazily from the lineage root.
        """
        self._predicate_matched_ids = set()
        self._predicate_eval_cache = {}
        self._state_roots = {}
        self._py_state_options = {}
        self._py_state_globals = {}

    @classmethod
    def load_from_disk(
        cls,
        path: str,
        project: angr.Project,
        **kwargs,
    ) -> RustExplorationManager:
        """Construct a fresh manager and restore its stash from ``path``.

        Convenience wrapper around the ``RustExplorationManager(project) +
        load_snapshot(path)`` two-step. Use this when you want to resume an
        exploration without re-deriving an entry state just to give the
        constructor a placeholder.

        Args:
            path: Filesystem path to read the snapshot from. Same format as
                :meth:`dump_snapshot` writes.
            project: angr Project matching the binary the snapshot was
                captured against. The caller is responsible for the match
                — there is no binary-hash cross-check.
            **kwargs: Passed through to ``__init__`` so manager-level
                configuration (``solver_timeout_ms``, ``max_active_states``,
                ``max_history``, ``exploration_strategy``, …) can be
                overridden on resume. The snapshot does NOT capture these.

        Returns:
            A new :class:`RustExplorationManager` whose stashes contain the
            restored states. Find/avoid addresses, hooks, simprocedures,
            inspection breakpoints, and other Python-side configuration are
            NOT restored — the caller must re-register them, same as on a
            fresh manager.

        Raises:
            ValueError: When the envelope is empty or carries a stale
                format-version byte (propagated from :meth:`load_snapshot`).
        """
        mgr = cls(project, active_states=None, **kwargs)
        mgr.load_snapshot(path)
        return mgr

    def __len__(self) -> int:
        """Return total number of active states."""
        return self._rust_mgr.active_count()

    def __getattr__(self, name: str):
        """Handle attribute access for stash names."""
        # Try to get stash by name
        if name.startswith("_"):
            raise AttributeError(name)

        # Handle one_* prefix for single state access (SimulationManager compatibility)
        if name.startswith("one_"):
            stash_name = name[4:]  # Remove "one_" prefix
            states = self._get_stash_states(stash_name)
            return states[0] if states else None

        # mp_* reads a stash as a mulpyplexer, as SimulationManager._fetch_states does (angr-mx00p.132.42).
        if name.startswith("mp_"):
            stash_name = name[3:]
            if stash_name == "_ALL":
                return mulpyplexer.MP(list(chain.from_iterable(self.stashes.values())))
            try:
                states = getattr(self, stash_name)
            except AttributeError:
                # cat-(a) EXPECTED CONTROL FLOW: SimulationManager reads a missing stash's mp_ form as empty.
                states = []
            return mulpyplexer.MP(list(states))

        # A typo or feature probe raises, as SimulationManager._fetch_states does, so hasattr / getattr
        # defaults work; a stash the caller named (explore(find_stash=)) reads as states (angr-vmi2l.238).
        if name not in self._stash_names():
            raise AttributeError(f"No such stash: {name}")
        return self._get_stash_states(name)
