"""Python wrapper for Rust-native exploration manager.

This module provides `RustExplorationManager`, a Python-facing interface
to the Rust exploration loop that achieves ~3x speedup by:
- Managing states entirely in Rust (using RustSimState)
- Processing symbolic branches with deferred forks
- Only calling Python for SimProcedures and syscalls
- Implementing find/avoid address checking in Rust

Cross-mixin invariants
======================

The manager class composes several mixins (RustStateCacheMixin,
RustStateExportMixin, RustStateSyncMixin, RustCallbackDispatchMixin) plus
helpers in this module. The invariants below cut across those boundaries —
breaking any one of them tends to produce silent correctness bugs (cache
poisoning, lost state, infinite loops) rather than loud failures, so they
are documented here and pinned by the named regression tests.

I1. Disk-cache key axes
    `_disk_cache_key` mixes (binary path, `_RUST_CACHE_VERSION`,
    `_PYTHON_METADATA_VERSION`, arch name) into the filename hash and
    memoizes on `(binary_path, arch_name)` in `_disk_key_cache`. When you
    add a new dimension that affects the serialized init state, you must
    bump the matching version constant AND extend the memo tuple — all
    three (constant, hash, memo key) move together. Old-format pkls are
    silently rejected by the outer `try/except Exception` in
    `_load_init_pickle`.

I2. Init pipeline phases (post angr-khth split)
    `_run_python_init_if_needed` is the single orchestrator:
    in-memory cache → disk cache → full Python init. The disk-load path
    is split so each phase owns a single concern:
      * `_load_init_pickle`        — pure I/O.
      * `_deserialize_init_state`  — pure SimState construction; no
                                     manager-owned mutation.
      * `_apply_init_side_effects` — populates `_pending_procedure_data`
                                     and `_precomputed_regs` on self.
      * `_load_init_from_disk_cache` — thin wrapper over the three.
    When you save a NEW field to the disk cache, decide whether the pure
    deserialization or the side-effect phase owns it. Don't mix the two —
    that is the whole point of the split.

I3. Init-cache user-symbolic gate
    Disk and in-memory init caches are BOTH gated on
    `_state_has_user_symbolic(state)`. `blank_state` round-trips lose
    user-created BVS identity (e.g. `argv` symbols), so `_compute_disk_init_key`
    and `_compute_mem_init_key` return `''` to suppress load AND save when
    the input state holds user-symbolic data. Tests rely on this so user
    stores from one test do not bleed into the next via the class-level
    `_init_cache`. See the `_isolate_class_caches` autouse fixture in
    `TestEdgeCases` (tests/engines/test_rust_exploration.py:6601) which
    clears the cache between tests as defense-in-depth.

I4. `_apply_state_metadata` is an allow-list
    On a cached/disk-loaded init state, `_apply_state_metadata` copies
    constraints, globals, and only the `LAZY_SOLVES` + `STRICT_PAGE_ACCESS`
    SimOptions from the source. Every other option (`TRACK_*`,
    `CONCRETIZE`, `DO_RET_EMULATION`, ...) is silently dropped on the
    cache-hit path. To check user-set options reliably, do it on the user-
    supplied state in `__init__` BEFORE `_run_python_init_if_needed` runs,
    not on the post-init state in `_add_rust_state`.

I5. Register filter at the FFI boundary
    The disk init cache pickles ALL archinfo registers (cr0..8, ymm0..15,
    fs_seg, ds_seg, cmstart, cmlen, fpreg, ...) via
    `arch.register_names.values()`. The Rust engine only models a subset
    (rax-r15+rip on amd64). When passing the cached dict to Rust via
    `set_registers_bulk`, you MUST filter to `_supported_register_names`
    in rust_state_sync.py, otherwise PyValueError 'unknown register: cr0'.
    Do not "fix" by adding cr0 etc. to amd64.rs unless the interpreter
    actually consumes them — it doesn't, and the slow path skips them too.

I6. State-cache pinning + manager-vs-mixin override
    `_cleanup_state_cache` (manager override; takes precedence over the
    mixin version) runs three steps: (1) drop entries whose state is no
    longer in active/found, (2) pin every root in `_state_roots`, plus
    `_current_callback_state_id` (and its effective id via
    `_get_effective_state_id`) and `_current_stepping_state_id`, (3)
    LRU-evict non-pinned entries past `_max_state_cache_size`. Without
    those pins, a freshly-mutated state can race-evict between callbacks
    on the same state. Regression tests:
    `TestStateCacheSizeBound.test_cleanup_state_cache_evicts_oldest_first`,
    `..._drops_dead_states`, `..._skips_pinned` (lines 2017, 2064, 2096).
    Note: the manager path does NOT call `clear_state_metadata` on
    eviction — it relies on `RustSimState`'s own drop. The mixin version
    of `_cleanup_state_cache` (rust_state_cache.py) DOES clear metadata.

I7. Rust ↔ Python field sync uses max(), not overwrite
    Fields that both sides can mutate (`mmap_base`, `posix_brk`, ...) sync
    on stash export by taking `max(rust_value, python_value)`. A Python-
    side advance (user-set `state.heap.mmap_base` before re-entering
    exploration, or a fallback SimProcedure mutation) must not be reverted
    to a smaller Rust value. Tests:
    `TestMmapBaseSync.test_export_path_does_not_clobber_higher_python_mmap_base`
    (line 1629), `TestPosixBrkSync.test_export_path_syncs_rust_posix_brk_into_state_posix`.

I8. Exploration-loop termination conditions
    `_explore_with_predicates` and `_explore_with_addresses` must terminate
    on EITHER (a) Python predicate match, OR (b) Rust-native find_addr hit
    (`event_type == 'found'`), OR (c) all paths exhausted
    (`active_empty` / `has_active_states == false`). Earlier code only
    checked Python predicate flags, which infinite-looped when `find=int`
    was combined with a non-predicate technique like DFS (technique made
    `_active_techniques` non-empty, routing through the predicate path).
    Fixed by switching the check to `_found_count()` which covers both
    Rust-native and Python-predicate finds.

I9. `max_active_states` is enforced via a helper, not raw `push_back`
    Every site that pushes into the active stash must go through
    `push_to_active_or_drop` (in native/angr/src/exploration/helpers.rs).
    The helper uses `sm.push()` which respects the cap; raw
    `stashes_mut().entry(STASH_ACTIVE).push_back()` bypasses it and the
    limit silently breaks again.

I10. RustExplorationManager exposes `stats` as a @property
     `mgr.stats()` raises TypeError — it is not callable. The inner
     `mgr._rust_mgr` (PyO3 object) does expose `stats()` as a method, and
     `get_fallback_stats()` exists ONLY on `_rust_mgr` (the Python wrapper
     does not re-export it). Tests that need full fallback details must
     reach in via `mgr._rust_mgr.get_fallback_stats()`.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
import warnings
import weakref
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union

import claripy
from claripy.errors import ClaripyError
from pyvex.errors import PyVEXError

from angr.errors import SimEngineError, SimError
from angr.exploration.rust_irsb_serializer import serialize_irsb
from angr.exploration.rust_perf_tracker import PerformanceTracker
from angr.exploration._constants import PAGE_SIZE, PAGE_MASK, STACK_SIZE, MAX_OVERLAY_SECTION_SIZE

if TYPE_CHECKING:
    import angr

l = logging.getLogger(name=__name__)
_DBG = l.isEnabledFor(logging.DEBUG)  # Module-level guard for hot-path debug calls

# Disk cache versioning is split across two axes so each side can invalidate
# without forcing a full cache rebuild on the other:
#   _RUST_CACHE_VERSION:      bump when Rust engine changes affect serialized
#                             init state (memory page format, register values,
#                             callstack frame layout produced by Rust).
#   _PYTHON_METADATA_VERSION: bump when Python-side SimState attributes that
#                             we read or restore change shape (e.g., new
#                             callstack frame field, new register imports).
# Both are mixed into the cache key together with the arch name, so a key
# from a different (rust, python, arch) tuple lands at a different file and
# is treated as a miss — never deserialized into a current-format slot.
_RUST_CACHE_VERSION = 2
_PYTHON_METADATA_VERSION = 1

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
    # cat-(b) FALLBACK WITH LOSS: Rust extension not built; the manager
    # is import-safe but constructing one will raise from __init__.
    RUST_EXPLORATION_AVAILABLE = False
    _RustExplorationManager = None
    _ExplorationEvent = None
    _ExplorationStateSnapshot = None
    PythonCallbacks = None
    _RustSimState = None

# SimOptions tagged "(b) explicitly reject" in docs/advanced-topics/rust_engine.rst.
# Setting any of these on a state owned by RustExplorationManager would change
# Python-engine semantics, but the Rust engine silently ignores them — without
# a warning users can spend hours chasing a divergence between engines. We
# warn-once per option per manager rather than raise, since the user may have
# inherited the option from a parent state without explicit consent. Two
# entries from the matrix's (b) category — `TRACK_CONSTRAINT_ACTIONS` and
# `TRACK_MEMORY_MAPPING` — are intentionally **excluded** here because they
# ship in the default `symbolic` mode bundle (sim_options.py:391, 374). Every
# `factory.entry_state()` would otherwise trigger a warning the user did not
# choose. They remain divergence-risk in the doc; this set covers the options
# a user must opt into.
_REJECTED_OPTION_NAMES = frozenset({
    # Conservative read strategy: refuses to concretize on range-check
    # failure. (The write strategy variant raises, see _RAISE_OPTION_NAMES.)
    "CONSERVATIVE_READ_STRATEGY",
    # SimMemory error-handling tweaks.
    "UNINITIALIZED_ACCESS_AWARENESS", "BEST_EFFORT_MEMORY_STORING",
    # Ret-emulation guard sibling. The DO_RET_EMULATION half raises (see
    # _RAISE_OPTION_NAMES); the guard alone is harmless without it.
    "TRUE_RET_EMULATION_GUARD",
    # Alternate Python engines / memory plugins.
    "SUPER_FASTPATH", "FAST_MEMORY", "FAST_REGISTERS", "UNDER_CONSTRAINED_SYMEXEC",
})


# SimOptions that we hard-fail rather than warn on. The Rust engine never
# produces SimAction or SimEvent records, so anything driven by
# state.history.actions or unsat_core() will silently get empty data under
# Rust. Loud failure beats hours of chasing a phantom divergence. None of
# these ship in the default `symbolic` bundle (sim_options.py:391), so this
# only fires when a user explicitly added the option. TRACK_OP_ACTIONS does
# ship in the `fastpath` mode bundle — fastpath users will hit this and must
# drop to the Python engine for action-stream-driven analyses.
#
# CONCRETIZE eagerly concretizes every symbol introduced (Python:
# sim_options.py + SimSolver.BatchedConcretizationBacker). Silent ignore
# under Rust would totally change semantics — symbol-driven solver tests
# would behave like concrete tests but without the speedup. Promoted to
# raise (angr-gmrc, 2026-05-16).
#
# CONSERVATIVE_WRITE_STRATEGY tells SimMemory to refuse symbolic-write
# address concretization on range-check failure (Python: state_plugins/
# symbolic_memory.py SimSymbolicMemory.concretize_write_addr). Rust's
# SymbolicMemory always concretizes within strategy limits, so silently
# accepting this would mask the user's intent to keep an analysis
# conservative. Promoted to raise (angr-csmm, 2026-05-16).
#
# DO_RET_EMULATION asks the engine to add an emulated successor at every
# ret site (Python: SimEngineVEX returns the ret successor with a guard
# that's true unless TRUE_RET_EMULATION_GUARD is also set). Rust does
# not emulate rets at all, so the emulated successor is silently missing
# under Rust; that's a divergence in the successor set that Callable
# workflows (the typical caller) depend on. Promoted to raise (angr-cf9h,
# 2026-05-16). TRUE_RET_EMULATION_GUARD stays in _REJECTED_OPTION_NAMES
# because alone it's just a guard tweak with no effect.
#
# CALLLESS replaces every call with an unconstraining of the return
# register, used by Callable to short-circuit function bodies. Rust has
# no equivalent path, so calls execute normally; silently accepting the
# option means Callable workflows would step into the callee instead of
# skipping it — a structural divergence, not a precision one. Promoted
# to raise (angr-cf9h, 2026-05-16).
#
# EFFICIENT_STATE_MERGING asks SimStateHistory to retain a strong
# reference to each ancestor state so state.merge() can find a common
# ancestor for plugin merging (Python: state_plugins/history.py
# set_strongref_state). The Rust engine does not drive SimStateHistory's
# strongref path, so the option is silently ignored. Auto-added by
# Veritesting (exploration_techniques/veritesting.py), which requires
# real per-plugin merging to work — Veritesting under Rust would
# silently lose ancestor refs and then fall back to weak-ref merging
# inside the export path. Promoted to raise (angr-n129, 2026-05-16).
# The paired SIMPLIFY_MERGED_CONSTRAINTS is NOT promoted because it
# ships in the default `symbolic` mode bundle (simplification set);
# it is honored implicitly through the Python state.merge() fallback
# inside RustExplorationManager.merge().
#
# SYMBOL_FILL_UNCONSTRAINED_REGISTERS asks the Python filler to create
# a fresh symbolic BVS on every read of an uninitialized register, and
# suppresses the otherwise-emitted warning (Python: state_plugins/
# light_registers.py _fill, storage/memory_mixins/default_filler_mixin.py
# _default_value). The Rust RegisterFile always returns concrete zero
# from its vec![0; size] storage — there is no "uninitialized" marker,
# so register reads cannot generate fresh symbols regardless of options.
# Silently accepting the option means a user who opted into symbolic-fill
# would get concrete-zero registers and never know — paths that depend
# on unconstrained initial register values would simply not be explored.
# Promoted to raise (angr-apre, 2026-05-17). The MEMORY variant
# SYMBOL_FILL_UNCONSTRAINED_MEMORY is NOT promoted because Rust's
# load_concrete_lazy (native/angr/src/memory/load.rs:333-339) falls back
# to a fresh `unc_mem_*` symbolic BVS when zero_fill_unconstrained is
# unset — i.e., symbolic-fill is already Rust's default for memory.
_RAISE_OPTION_NAMES = frozenset({
    "TRACK_MEMORY_ACTIONS", "TRACK_REGISTER_ACTIONS", "TRACK_TMP_ACTIONS",
    "TRACK_JMP_ACTIONS", "TRACK_OP_ACTIONS", "TRACK_ACTION_HISTORY",
    "CONCRETIZE",
    "CONSERVATIVE_WRITE_STRATEGY",
    "DO_RET_EMULATION",
    "CALLLESS",
    "EFFICIENT_STATE_MERGING",
    "SYMBOL_FILL_UNCONSTRAINED_REGISTERS",
})


# Z3 context sharing: make Rust and Python use the same Z3 context
# to avoid AST translation overhead between solvers.
_z3_context_shared = False

def _setup_shared_z3_context():
    """Share Python's Z3 context with Rust, so both create ASTs in the same context."""
    global _z3_context_shared
    if _z3_context_shared:
        return
    try:
        from angr.rustylib.vex_engine import set_shared_z3_context, reset_shared_z3_context
        import atexit
        import z3
        py_ctx = z3.main_ctx()
        set_shared_z3_context(py_ctx.ctx.value)
        _z3_context_shared = True
        l.debug("Shared Z3 context with Rust (ptr=%#x)", py_ctx.ctx.value)

        # Register cleanup to run BEFORE Python's Z3 context is freed.
        # Without this, Rust's Solver objects may reference a freed Z3 context
        # at process exit, causing a segfault.
        atexit.register(reset_shared_z3_context)
    except (ImportError, AttributeError, Exception) as e:
        # cat-(b) FALLBACK WITH LOSS: shared Z3 context unavailable; AST
        # round-tripping pays a translation hop on each Rust<->Python move.
        # Already debug-logs the cause.
        l.debug("Z3 context sharing not available: %s", e)


_z3_deterministic_applied = False


def _apply_deterministic_z3_globals() -> None:
    """Pin Z3 module-level random seeds for run-to-run model stability.

    Sets ``smt.random_seed`` and ``sat.random_seed`` to 0 via
    ``Z3_global_param_set``. Z3 reads these on every subsequent
    ``Solver::new`` call (existing solvers are unaffected). Idempotent:
    safe to call from every manager constructed with
    ``deterministic=True``.

    Caveat: Z3 4.13 still reserves variable / restart heuristic latitude
    that is not bound by these seeds. See angr-iaol.1 close-out memory
    ``iaol1-seed-pin-empirically-broken`` for the full audit and the
    rust_engine.rst "Deterministic mode" section for user-facing docs.
    """
    global _z3_deterministic_applied
    if _z3_deterministic_applied:
        return
    try:
        from angr.rustylib.vex_engine import set_z3_global_param
    except ImportError:
        # cat-(b) FALLBACK WITH LOSS: the Rust extension was built without
        # the vex-engine-z3 feature; determinism pin cannot be applied.
        # Manager construction proceeds with current (nondeterministic)
        # Z3 globals.
        l.debug("set_z3_global_param not available; deterministic=True ignored")
        return
    set_z3_global_param("smt.random_seed", "0")
    set_z3_global_param("sat.random_seed", "0")
    _z3_deterministic_applied = True
    l.debug("Pinned Z3 smt.random_seed=0 + sat.random_seed=0 (deterministic mode)")


def set_rust_log_level(level: str = "info") -> None:
    """Set the Rust-side log level.

    Args:
        level: One of "error", "warn", "info", "debug", "trace", "off".
    """
    from angr.rustylib.vex_engine import set_rust_log_level as _set_level
    _set_level(level)


_rust_log_env_applied = False

def _apply_rust_log_env() -> None:
    """Honor ANGR_RUST_LOG on first manager construction.

    Set ANGR_RUST_LOG=debug (or error/warn/info/trace/off) to surface
    Rust-side log::debug!/info! output. Applies once per process.
    The Rust engine uses a custom StderrLogger plus log::set_max_level,
    not env_logger, so the standard RUST_LOG env var is *not* honored
    and per-module filters are not supported.
    """
    global _rust_log_env_applied
    if _rust_log_env_applied:
        return
    _rust_log_env_applied = True
    level = os.environ.get("ANGR_RUST_LOG")
    if not level:
        return
    try:
        set_rust_log_level(level)
    except Exception as e:  # noqa: BLE001 — never fail manager construction
        # cat-(b) FALLBACK WITH LOSS: ANGR_RUST_LOG could not be applied;
        # manager construction proceeds, but Rust-side log output stays at
        # whatever level was set previously (typically off).
        l.debug("Failed to set Rust log level from ANGR_RUST_LOG=%r: %s", level, e)


from angr.exploration.rust_identity import SymbolicIdentityTracker, CallbackMemoryTracker


from angr.exploration.rust_state_export import RustStateExportMixin
from angr.exploration.rust_callback_dispatch import RustCallbackDispatchMixin
from angr.exploration.rust_state_sync import RustStateSyncMixin
from angr.exploration.rust_state_cache import RustStateCacheMixin


def _extract_register_snapshot(state, arch) -> Dict[str, int]:
    """Extract concrete register values from a SimState.

    Skips symbolic registers and any access errors. Returns a dict
    mapping register name → concrete int value.
    """
    registers: Dict[str, int] = {}
    for reg_name in arch.register_names.values():
        try:
            val = getattr(state.regs, reg_name)
            if not val.symbolic:
                registers[reg_name] = state.solver.eval(val)
        except (AttributeError, KeyError, TypeError, ValueError):
            # cat-(a) EXPECTED CONTROL FLOW: arch lists a register the SimState
            # doesn't expose, or the read errors on a symbolic value — skip.
            pass
    return registers


def _extract_stack_page(state, page_size: int):
    """Extract the stack page at SP and the lazy stack region descriptor.

    Returns (stack_page, lazy_region) where stack_page is
    ``(page_addr, bytes)`` and lazy_region is ``(start_addr, length)``.
    Both are None if extraction fails.
    """
    try:
        sp = state.solver.eval(state.regs.sp)
        sp_page = sp & ~(page_size - 1)
        page_val = state.memory.load(
            sp_page, page_size, endness='Iend_BE',
            inspect=False, disable_actions=True)
        concrete = state.solver.eval(page_val).to_bytes(page_size, 'big')
        stack_base = (sp & ~(page_size - 1)) + page_size
        stack_start = stack_base - STACK_SIZE
        return (sp_page, concrete), (stack_start, STACK_SIZE)
    except (AttributeError, TypeError, ValueError):
        # cat-(a) EXPECTED CONTROL FLOW: SP is symbolic or stack page can't
        # be loaded; caller treats (None, None) as 'no eager extraction' and
        # falls back to the lazy stack region.
        return None, None


def _extract_loader_pages(loader, page_size: int):
    """Extract concrete loader memory pages and per-object lazy regions.

    Returns (batch_pages, lazy_regions, mapped_page_addrs):
    - batch_pages: list of (page_addr, bytes, perms) tuples
    - lazy_regions: list of (start_addr, length) tuples (one per loader object)
    - mapped_page_addrs: set of page_addr ints already captured
    """
    batch_pages = []
    lazy_regions = []
    mapped_page_addrs = set()
    for obj in loader.all_objects:
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
                        page_data = loader.memory.load(page_addr, page_size)
                        if page_data and len(page_data) == page_size:
                            batch_pages.append((page_addr, bytes(page_data), 7))
                            mapped_page_addrs.add(page_addr)
                    except (KeyError, TypeError, ValueError):
                        # cat-(a) EXPECTED CONTROL FLOW: per-page loader read failed (e.g.
                        # unmapped gap inside the segment range); skip and continue.
                        pass
            region_start = obj.min_addr & ~(page_size - 1)
            region_end = (obj.max_addr + page_size) & ~(page_size - 1)
            if region_end - region_start > 0:
                lazy_regions.append((region_start, region_end - region_start))
        except (AttributeError, KeyError, TypeError):
            # cat-(b) FALLBACK WITH LOSS: per-object iteration failed (loader
            # object lacks expected attributes); that object's pages won't be
            # eagerly mapped — Rust will fetch them on demand via fetch_page.
            pass
    return batch_pages, lazy_regions, mapped_page_addrs


def _extract_section_patches(state, loader) -> list:
    """Extract concrete post-init section bytes (e.g., GOT fixups).

    Returns a list of (min_addr, bytes) tuples for sections smaller
    than MAX_OVERLAY_SECTION_SIZE whose memory loads as concrete.
    """
    section_patches = []
    for obj in loader.all_objects:
        if obj.binary is None or not hasattr(obj, 'sections'):
            continue
        for section in obj.sections:
            if 0 < section.memsize < MAX_OVERLAY_SECTION_SIZE:
                try:
                    val = state.memory.load(
                        section.min_addr, section.memsize,
                        endness='Iend_BE', inspect=False, disable_actions=True)
                    if not val.symbolic:
                        section_patches.append(
                            (section.min_addr,
                             state.solver.eval(val).to_bytes(section.memsize, 'big')))
                except (AttributeError, TypeError, ValueError):
                    # cat-(b) FALLBACK WITH LOSS: section overlay extraction failed;
                    # Rust sees raw loader bytes for this section without the post-init
                    # concrete patches (e.g. GOT relocations).
                    pass
    return section_patches


def _extract_extra_pages(state, page_size: int, mapped_page_addrs: set,
                         stack_page_addr: Optional[int]) -> list:
    """Extract non-loader pages created during init (e.g., ctype tables).

    Skips pages already captured as loader pages or as the stack page.
    Returns a list of (page_addr, bytes) tuples.
    """
    extra_pages = []
    if hasattr(state.memory, '_pages'):
        mem_page_size = getattr(state.memory, 'page_size', page_size)
        for page_num in state.memory._pages:
            page_addr = page_num * mem_page_size
            if page_addr in mapped_page_addrs:
                continue
            if stack_page_addr is not None and page_addr == stack_page_addr:
                continue
            page = state.memory._pages[page_num]
            if page is None:
                continue
            try:
                page_data = page.concrete_load(0, mem_page_size)
                if any(page_data):
                    extra_pages.append((page_addr, bytes(page_data)))
            except (AttributeError, TypeError, ValueError):
                # cat-(b) FALLBACK WITH LOSS: extra (non-loader) page extraction
                # failed (e.g., concrete_load on a symbolic page). Rust will fetch
                # the page lazily via the fetch_page callback if it is accessed.
                pass
    return extra_pages


def _extract_callstack_snapshot(state):
    """Walk the SimState callstack and collect frames + continuation addrs.

    Returns (callstack_frames, continuation_addrs).
    """
    callstack_frames = []
    continuation_addrs = []
    frame = state.callstack.top if hasattr(state, 'callstack') else None
    while frame is not None:
        pdata = getattr(frame, 'procedure_data', None)
        frame_data = {
            'call_site_addr': frame.call_site_addr,
            'func_addr': frame.func_addr,
            'ret_addr': frame.ret_addr,
            'stack_ptr': frame.stack_ptr,
        }
        if pdata is not None and len(pdata) >= 5:
            try:
                continuation_addrs.append(int(pdata[4]))
            except (TypeError, ValueError):
                # cat-(a) EXPECTED CONTROL FLOW: continuation addr in procedure_data
                # is not int-castable (symbolic). Skip — saves no continuation, the
                # normal procedure-data restore path handles it later.
                pass
        callstack_frames.append(frame_data)
        frame = getattr(frame, 'next', None)
    return callstack_frames, continuation_addrs


# Per-arch GPR snapshot lists for RustErrorRecord.registers. Conservative: PC,
# SP, BP/FP, and standard GPRs. Vector/floating-point registers are excluded.
_ARCH_REG_SNAPSHOT: Dict[str, Tuple[str, ...]] = {
    'AMD64': ('rip', 'rsp', 'rbp', 'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi',
              'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15'),
    'X86':   ('eip', 'esp', 'ebp', 'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi'),
    'ARM':     ('pc', 'sp', 'lr', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6',
                'r7', 'r8', 'r9', 'r10', 'r11', 'r12'),
    'ARMEL':   ('pc', 'sp', 'lr', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6',
                'r7', 'r8', 'r9', 'r10', 'r11', 'r12'),
    'ARMHF':   ('pc', 'sp', 'lr', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6',
                'r7', 'r8', 'r9', 'r10', 'r11', 'r12'),
    'AARCH64': ('pc', 'sp', 'lr', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                'x7', 'x8', 'x29', 'x30'),
    'MIPS32':  ('pc', 'sp', 'ra', 'v0', 'v1', 'a0', 'a1', 'a2', 'a3'),
    'MIPS64':  ('pc', 'sp', 'ra', 'v0', 'v1', 'a0', 'a1', 'a2', 'a3'),
}


class RustErrorRecord:
    """Container for an errored state, matching angr's ErrorRecord interface.

    Attributes:
        state:             The SimState at the point of error.
        error:             An Exception describing what went wrong.
        addr:              The instruction address where the error occurred.
        error_class:       Stable taxonomy string for the error
                           (e.g. 'unsupported', 'memory', 'lift', 'callback').
                           Derived from the message prefix; matches the
                           CbExecutionError variants in
                           native/angr/src/interpreter/mod.rs.
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
    # CbExecutionError variants in native/angr/src/interpreter/mod.rs and
    # the formatted error strings in native/angr/src/exploration/stepping.rs.
    _ERROR_CLASS_PREFIXES = (
        ('memory error',           'memory'),
        ('operation error',        'operation'),
        ('invalid vex ir',         'invalid_ir'),
        ('unsupported',            'unsupported'),
        ('type mismatch',          'type_mismatch'),
        ('unknown temporary',      'unknown_temp'),
        ('callback error',         'callback'),
        ('lift error',             'lift'),
        ('need lift at',           'need_lift'),
        ('need python fallback',   'need_python_fallback'),
        ('resolve_function error', 'resolve_function'),
    )

    def __init__(self, state, message: str, addr: int = 0):
        self.state = state
        self.error = RuntimeError(message)
        self.addr = addr
        self.error_class = self._classify(message)
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
        if 'timeout' in msg:
            return 'timeout'
        if 'unmapped' in msg:
            return 'unmapped'
        if 'panic' in msg:
            return 'rust_panic'
        return 'unknown'

    @staticmethod
    def _count_constraints(state) -> int:
        if state is None:
            return 0
        try:
            return len(state.solver.constraints)
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
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: best-effort error report; if arch
            # is unreadable, return empty registers dict.
            return {}
        names = _ARCH_REG_SNAPSHOT.get(arch_name, ())
        snap: dict = {}
        for name in names:
            try:
                val = getattr(state.regs, name)
            except Exception:
                # cat-(a) EXPECTED CONTROL FLOW: best-effort error report; skip a
                # register that fails to read.
                continue
            try:
                if val.concrete:
                    snap[name] = val.concrete_value
                else:
                    snap[name] = str(val)
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
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: best-effort error report; missing
            # history attribute is fine — return empty list.
            return []
        return bbls[-n:]

    def reraise(self):
        raise self.error

    def __repr__(self):
        return (
            f'<State errored at {hex(self.addr)} '
            f'class={self.error_class} with "{self.error}">'
        )


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

    # Class-level cache for disk cache keys (MD5 of binary content).
    # Keyed by (binary_path, arch_name) so cross-arch lookups don't collide.
    # Avoids re-hashing the same file on every RustExplorationManager construction.
    _disk_key_cache: Dict[Tuple[str, str], str] = {}

    # Class-level cache for blank_state objects keyed by (binary_path, addr).
    # blank_state() is expensive (~1ms); caching + copy() is <0.1ms.
    _blank_state_cache: Dict[tuple, "angr.SimState"] = {}
    _blank_state_cache_max = 10

    # Class-level cache for loader-pages output keyed (weakly) by the
    # cle.Loader instance. The output is a pure function of loader state,
    # so reusing it across Callable-spawned RustExplorationManagers on the
    # same project skips the per-init `loader.memory.load` + `map_memory_batch`
    # cost (~36ms for mma_howtouse, see angr-i9f2). Value is a dict with
    # 'batch_pages' (list of (page_addr, bytes, perms)) and 'lazy_regions'
    # (list of (start, len)). WeakKeyDictionary auto-evicts entries when
    # the Project/Loader is garbage-collected, so we avoid stale hits if
    # Python recycles ids across short-lived projects.
    _loader_pages_cache: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()

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
        save_unconstrained: bool = False,
        solver_timeout_ms: int = 30000,
        max_active_states: Optional[int] = None,
        max_history: int = 1000,
        clear_caches_on_cleanup: bool = False,
        exploration_strategy: str = "bfs",
        use_shared_lineage_solver: bool = False,
        deterministic: bool = False,
        **kwargs,
    ):
        """Initialize the Rust exploration manager.

        Args:
            project: angr Project for the binary.
            active_states: Optional list of initial angr SimStates.
            save_unconstrained: If True, save states with unconstrained IP
                to the 'unconstrained' stash instead of dropping them.
            solver_timeout_ms: Z3 solver timeout in milliseconds (default: 30000).
            max_active_states: Maximum number of states in the active stash.
                When reached, new forked states are pruned. None = no limit.
            max_history: Maximum length of each state's history /
                detailed_history ring buffer (default: 1000). 0 means
                unlimited — only safe for short runs since long explorations
                can OOM. Applied to every state created or added via this
                manager.
            clear_caches_on_cleanup: If True, the manager flushes the
                Rust-side thread-local claripy AST translation caches
                from ``cleanup()`` / ``__del__``. Off by default because
                the caches are typically helpful for a single long
                exploration; enable for Callable-heavy workloads that
                spawn many short-lived managers on the same thread
                (e.g. mma_howtouse runs 45 ``callable()`` invocations
                and the cache accumulates O(n) entries across them).
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
        """
        # Ensure Z3 context is shared (one-time setup)
        _setup_shared_z3_context()
        _apply_rust_log_env()

        # angr-iaol.2: pin Z3 module-level random seeds BEFORE the first
        # Solver::new. The per-solver Z3_solver_set_params route is
        # broken for these keys (angr-iaol.1) — only the global path
        # takes effect. Apply early in __init__ before _RustExplorationManager
        # constructs its first solver. The pin is process-global; once
        # set it persists for every subsequent solver.
        if deterministic:
            _apply_deterministic_z3_globals()
        self._deterministic = bool(deterministic)

        if not RUST_EXPLORATION_AVAILABLE:
            raise ImportError(
                "RustExplorationManager not available. "
                "Build with vex-engine feature enabled."
            )

        self._project = project
        self._save_unconstrained = save_unconstrained
        # See ``cleanup()`` — only honored when the manager has a real Rust
        # backend (i.e. not the multi-stage-reuse early return below).
        self._clear_caches_on_cleanup = clear_caches_on_cleanup
        is_le = project.arch.memory_endness == 'Iend_LE'
        self._rust_mgr = _RustExplorationManager(project.arch.name, little_endian=is_le)

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

        # Configure exploration strategy. Reuses the post-init setter so the
        # validation and FFI-call shape live in one place. Always invoke so a
        # bad value raises ValueError eagerly during construction; the
        # default-bfs path is cheap (one FFI hop into the Rust setter).
        self.set_exploration_strategy(exploration_strategy)

        # angr-3ms1 step 1b: opt-in flag for fork-time
        # SharedLineageSolver materialization. Stashed here so
        # _add_rust_state can push the value onto every seed state's
        # solver context. Default off keeps slice-1c's gate inert (and
        # the v5a5-slice-4c.3-retry-failed-bfs-thrash-fundamental
        # baby-re regression out of CI).
        self._use_shared_lineage_solver = bool(use_shared_lineage_solver)

        # Performance profiling counters
        self._perf_stats = PerformanceTracker()
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
        # angr-bs71/h0dv: defensive counter for Path A (rust_solver_ctx attach)
        # regressions. Stays at 0 in production; non-zero means a callback site
        # forgot to attach rust_solver_ctx and Python's solver may diverge from
        # Rust's. The legacy Path B (Rust→Python constraint AST push) was
        # removed in angr-h0dv after a 20-bench soak proved it was dead.
        self._stats_rust_ctx_missing = 0
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
        _init_start = time.perf_counter_ns()

        # Track registered hooks to detect dynamically created continuations
        # SimProcedures can create continuation hooks via self.call() which
        # need to be registered with Rust before exploration continues
        # (Must be initialized before _register_simprocedures() is called)
        self._registered_hooks: set = set()

        # Set up callbacks
        _t0 = time.perf_counter_ns()
        self._setup_callbacks()
        self._perf_stats.set_init_phase('setup_callbacks', time.perf_counter_ns() - _t0)

        # Load binary regions
        _t0 = time.perf_counter_ns()
        self._load_binary_regions()
        self._perf_stats.set_init_phase('load_binary', time.perf_counter_ns() - _t0)

        # Register SimProcedures
        _t0 = time.perf_counter_ns()
        self._register_simprocedures()
        self._perf_stats.set_init_phase('register_simprocedures', time.perf_counter_ns() - _t0)

        # Symbolic identity tracker for preserving AST identity across FFI
        # This is critical: BVS("x", 32) must stay the same object after round-trip
        self._identity_tracker = SymbolicIdentityTracker()

        # Track angr state mappings for callbacks
        # Using regular dict with periodic cleanup to prevent memory leaks
        self._state_cache: Dict[int, "angr.SimState"] = {}

        # Lazy SimState references handed out by _get_stash_states. Keyed by
        # Rust state id so repeated stash reads return the same wrapper
        # (preserves the `mgr.active[0] is mgr.active[0]` invariant). Wrappers
        # materialize their SimState on first attribute access — see
        # `_LazySimStateRef` in rust_state_export.py. Pruned in
        # `_cleanup_state_cache` alongside _state_cache.
        from angr.exploration.rust_state_export import _LazySimStateRef
        self._lazy_state_refs: Dict[int, _LazySimStateRef] = {}

        # Maximum state cache size. Cache is bounded to the in-flight callback
        # state plus a small LRU window of recently-mutated states; root states
        # are pinned and never count against the cap. Pre-angr-qm7w this was
        # 500 and the cache tracked O(active_states), making it the dominant
        # driver of Python-side memory growth.
        self._max_state_cache_size = 8

        # Track claripy AST handles for constraint sync
        # Maps handle_id -> claripy AST
        self._ast_handle_cache: Dict[int, object] = {}

        # Cache claripy AST -> Z3 AST pointer for register sync.
        # Skips redundant z3_backend.convert(reg_val) + .as_ast().value lookups
        # when the same symbolic register is re-imported across SimProcedure
        # callbacks. Holds a strong ref to the z3 object so the AST pointer
        # stays valid (Z3 ASTs are refcounted; the shared context outlives the
        # manager). Bounded size with simple drop-and-rebuild eviction.
        self._z3_ptr_cache: Dict[tuple, tuple] = {}
        self._z3_ptr_cache_max = 1024
        self._z3_ptr_cache_hits = 0
        self._z3_ptr_cache_misses = 0

        # Track current callback state for memory access during callbacks
        # This allows memory_load callback to access the correct symbolic state
        self._callback_state: Optional["angr.SimState"] = None

        # Cache bundle register values from _create_state_for_callback for
        # reuse as register snapshot (avoids reading registers back from state)
        self._last_bundle_registers: Optional[dict] = None

        # Per-state metadata (symbolic_pages / hook_symbolic_memory /
        # addr_to_ast) is now stored on the Rust side in `RustSimState`. Access
        # goes through `self._rust_mgr.{get,set}_state_*` PyO3 methods so the
        # storage and the state lifetime are unified — when Rust drops a state,
        # its metadata is freed automatically.
        self._max_symbolic_pages_cache = 100  # Retained for back-compat hooks.

        # Track the current callback state ID for memory tracking during callbacks
        self._current_callback_state_id: Optional[int] = None
        # Track which Rust state is being stepped for per-fork memory isolation
        self._current_stepping_state_id: Optional[int] = None

        # Track procedure_data for SimProcedure continuations.
        # When a SimProcedure uses self.call() to invoke a function and register
        # a continuation, the procedure_data is stored here keyed by the continuation
        # address. When Rust invokes the continuation, we restore this data.
        # Maps continuation_addr -> procedure_data tuple.
        self._pending_procedure_data: Dict[int, Tuple] = {}

        # Cache for addresses where SimProcedure continuations always result in exit.
        # After the first time a continuation at an address produces only Ijk_Exit
        # successors, all subsequent callbacks are fast-deadended without state creation.
        self._exit_continuation_addrs: set = set()

        # Track root state IDs for plugin restoration.
        # Maps state_id -> root_state_id (the original state from Python).
        # When Rust forks states, this allows finding the original state for plugin copying.
        # Pruned in `_cleanup_state_cache`: an entry whose key state is no
        # longer in any Rust stash is dropped (state IDs are monotonically
        # allocated and never reused, so this is safe).
        self._state_roots: Dict[int, int] = {}

        # Per-state-id Python-side stand-ins for state.options and state.globals.
        # The Rust engine doesn't honor SimOptions (LAZY_SOLVES / STRICT_PAGE_ACCESS
        # are mirrored separately on the Rust state itself), but user-facing code
        # — predicates, exploration techniques, callbacks — frequently reads
        # state.options.add(X) and state.globals[k] = v. Storing the set/dict
        # Python-side keyed by state_id lets RustStateProxy expose live mutable
        # views without round-tripping through Rust. Children inherit a deep
        # copy from their root on first access (see get_state_options_py /
        # get_state_globals_py).
        self._py_state_options: Dict[int, set] = {}
        self._py_state_globals: Dict[int, dict] = {}

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
        # and follow the 5-touchpoint pattern documented there.
        # See docs/advanced-topics/rust_engine.rst for the MVP scope.
        from angr.exploration.rust_state_proxy import (
            _RUST_INSPECT_EVENT_BITS,
            _RUST_INSPECT_SUPPORTED_EVENTS,
        )
        self._inspect_breakpoints: Dict[str, list] = {
            evt: [] for evt in _RUST_INSPECT_SUPPORTED_EVENTS
        }
        self._INSPECT_EVENT_BITS = dict(_RUST_INSPECT_EVENT_BITS)
        # Reentrancy guard: when a user action callback runs, suppress
        # nested inspect dispatch on the same manager. uq4n.4 covers
        # the full guard test.
        self._inspect_dispatch_depth = 0
        # Lazy RustInspectProxy instance (one per manager, shared across proxies).
        self._inspect_proxy: Optional["RustInspectProxy"] = None

        # Cached memory layout from disk cache for fast _sync_memory_to_rust
        self._mem_cache: Optional[dict] = None

        # Pre-computed register dict from disk cache for fast register sync
        self._precomputed_regs: Optional[dict] = None

        # Track stdin BVS variables for state export.
        # List of (claripy_bvs, size_ast) tuples from SimPacketsStream.content.
        # Populated during SimProcedure callbacks that read from stdin (fgets, read, etc.).
        # Used to restore stdin content on found states that were forked purely in Rust.
        self._stdin_content: list = []

        # Multi-stage explore reuse: if the initial state came from a previous
        # RustExplorationManager for the same project, reuse the old Rust manager
        # instead of creating a new one. This avoids lossy constraint transfer
        # that fails after ~50 stages (Z3 dedup causes UNSAT, claripy loses constraints).
        self._reused_from = None
        if active_states:
            _single = active_states[0] if isinstance(active_states, (list, tuple)) else active_states
            old_mgr = getattr(getattr(_single, 'scratch', None), 'rust_mgr', None)
            old_state_id = getattr(getattr(_single, 'scratch', None), 'rust_found_state_id', None)
            if old_mgr is not None and old_state_id is not None:
                try:
                    old_mgr.reset_for_stage(old_state_id)
                    # Reuse the old Rust manager — it has the correct solver state
                    self._rust_mgr = old_mgr
                    self._reused_from = old_state_id
                    # Cache the angr state with the reused state ID
                    self._state_cache[old_state_id] = _single
                    self._state_roots[old_state_id] = old_state_id
                    l.debug(f"Multi-stage reuse: reset manager for state {old_state_id}")
                    # Skip ALL remaining init — callbacks, binary, simprocedures
                    # are already set up on the old manager
                    self._perf_stats.set_init_phase('total', time.perf_counter_ns() - _init_start)
                    return
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: multi-stage manager reuse failed;
                    # falls back to building a fresh Rust manager from this state.
                    # Already debug-logs the cause.
                    l.debug(f"Multi-stage reuse failed, falling back to normal init: {e}")

        # Add initial states
        if active_states:
            # Handle single state or list of states
            if hasattr(active_states, 'solver'):  # Single SimState
                active_states = [active_states]
            for state in active_states:
                # Detect state options
                if hasattr(state, 'options'):
                    # Hard-fail on options the Rust engine cannot honor
                    # before doing any further work; warn-once on the rest.
                    # The post-Python-init state passed to _add_rust_state
                    # may come from a cached path that strips options down
                    # to LAZY_SOLVES + STRICT_PAGE_ACCESS via
                    # _apply_state_metadata.
                    self._check_raise_options(state.options)
                    self._warn_rejected_options(state.options)
                    try:
                        from angr import sim_options as o
                        if o.LAZY_SOLVES in state.options:
                            self._rust_mgr.set_lazy_solves(True)
                            l.debug("Enabled lazy_solves mode from state options")
                        if o.ZERO_FILL_UNCONSTRAINED_MEMORY in state.options:
                            self._rust_mgr.set_zero_fill_unconstrained(True)
                            l.debug("Enabled zero_fill_unconstrained from state options")
                        # Configure concretization strategies to match Python's
                        use_approx = o.APPROXIMATE_MEMORY_INDICES in state.options
                        sym_write = o.SYMBOLIC_WRITE_ADDRESSES in state.options
                        # Read Python's strategy limits from memory plugin
                        read_limit = 1024  # Python default
                        write_limit = 128  # Python default
                        if hasattr(state, 'memory'):
                            mem = state.memory
                            if hasattr(mem, 'read_strategies') and mem.read_strategies:
                                for strat in mem.read_strategies:
                                    if hasattr(strat, '_limit'):
                                        read_limit = strat._limit
                                        break
                            if hasattr(mem, 'write_strategies') and mem.write_strategies:
                                for strat in mem.write_strategies:
                                    if hasattr(strat, '_limit'):
                                        write_limit = strat._limit
                                        break
                        self._rust_mgr.configure_concretization_strategies(
                            use_approx,
                            read_limit,
                            write_limit,
                            sym_write,
                        )
                        l.debug(
                            "Configured concretization: approx=%s, read_limit=%d, "
                            "write_limit=%d, sym_write=%s",
                            use_approx, read_limit, write_limit, sym_write,
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
                state = self._run_python_init_if_needed(state)
                self._perf_stats.add_init_phase('python_run', time.perf_counter_ns() - _t0)

                _t0 = time.perf_counter_ns()
                self._add_rust_state('active', state)
                self._perf_stats.add_init_phase('add_rust_state', time.perf_counter_ns() - _t0)

        self._perf_stats.set_init_phase('total', time.perf_counter_ns() - _init_start)

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
        # angr-xtse.1: per-category Python callback timing for upper-bound
        # speedup analysis. All five paths are instrumented from the public
        # _handle_* entry points in rust_callback_dispatch.py.
        for label, key in (
            ("Syscall", "syscall"),
            ("Find predicate", "find_predicate"),
            ("Avoid predicate", "avoid_predicate"),
            ("Symbolic branch", "symbolic_branch"),
            ("Python VEX fallback", "vex_fallback"),
        ):
            count = s.get(f"callback_{key}_count", 0)
            ns = s.get(f"callback_{key}_total_ns", 0)
            lines.append(f"{label} callbacks: {count}")
            lines.append(f"  Total time: {ns/1e6:.1f}ms")
            if count > 0:
                lines.append(f"  Avg per call: {ns/count/1e3:.1f}us")
        # Per-category fallback counters (angr-md0m). Pulled live from
        # self.stats — values may be 0 if the run never tripped the path.
        try:
            fb = self.stats
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: stats unavailable; the perf report
            # omits the fallback-counter section but still prints other phases.
            fb = {}
        if fb:
            lines.append("Fallback counters:")
            lines.append(f"  SimProcedure -> Python: {fb.get('simprocedure_python_fallback_count', 0)}")
            by_name = fb.get('simprocedure_fallback_by_name', {}) or {}
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

    def get_exploration_summary(self) -> str:
        """Return a high-level summary of the exploration run."""
        s = self._perf_stats
        lines = ["=== Exploration Summary ==="]

        # Duration
        explore_ns = getattr(self, '_time_in_explore_ns', 0)
        init_ns = s.get('init_total_ns', 0)
        total_ns = init_ns + explore_ns
        lines.append(f"Total time: {total_ns/1e6:.1f}ms (init: {init_ns/1e6:.1f}ms, explore: {explore_ns/1e6:.1f}ms)")

        # Steps and throughput
        explore_s = explore_ns / 1e9 if explore_ns > 0 else 0
        steps_per_sec = f" ({self._stats_ffi_crossings / explore_s:.0f} steps/sec)" if explore_s > 0.001 else ""
        lines.append(f"Steps: {self._stats_ffi_crossings}{steps_per_sec}")

        # State counts
        try:
            n_found = len(self._rust_mgr.get_state_ids('found'))
            n_active = len(self._rust_mgr.get_state_ids('active'))
            n_deadended = len(self._rust_mgr.get_state_ids('deadended'))
            n_avoided = len(self._rust_mgr.get_state_ids('avoided'))
            lines.append(f"States: {n_found} found, {n_active} active, {n_avoided} avoided, {n_deadended} deadended")
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: state-count snapshot failed;
            # the summary skips the per-stash count line.
            pass

        # Callback breakdown
        cb_total = self._stats_callback_count
        sp_count = s.get('callback_simprocedure_count', 0)
        native_count = cb_total - sp_count  # memory/lift/fetch callbacks
        lines.append(f"Callbacks: {cb_total} total ({sp_count} SimProcedure, {native_count} other)")
        if self._stats_time_in_callbacks_ns > 0:
            lines.append(f"  Time in callbacks: {self._stats_time_in_callbacks_ns/1e6:.1f}ms")

        # Per-procedure breakdown (top 5)
        if self._procedure_times:
            sorted_procs = sorted(self._procedure_times.items(), key=lambda x: -x[1]['execute_ns'])
            lines.append(f"SimProcedure breakdown ({len(sorted_procs)} unique):")
            for pname, pt in sorted_procs[:5]:
                lines.append(f"  {pname}: {pt['count']}x, {pt['execute_ns']/1e6:.1f}ms")

        # FFI stats
        lines.append(f"State creations: {self._stats_state_creations}")
        lines.append(f"Cache: {self._stats_cache_hits} hits, {self._stats_cache_misses} misses")

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
        callbacks.set_get_register(self._cb_get_register)
        callbacks.set_put_register(self._cb_put_register)
        callbacks.set_dirty_call(self._cb_dirty_call)
        callbacks.set_resolve_function(self._cb_resolve_function)
        if hasattr(callbacks, 'set_memory_store_batch'):
            callbacks.set_memory_store_batch(self._cb_memory_store_batch)
        if hasattr(callbacks, 'set_memory_load_batch'):
            callbacks.set_memory_load_batch(self._cb_memory_load_batch)
        if hasattr(callbacks, 'set_batch_fetch_pages'):
            callbacks.set_batch_fetch_pages(self._cb_batch_fetch_pages)
        if hasattr(callbacks, 'set_memory_store_symbolic_value'):
            callbacks.set_memory_store_symbolic_value(self._cb_memory_store_symbolic_value)
        if hasattr(callbacks, 'set_memory_store_symbolic_full'):
            callbacks.set_memory_store_symbolic_full(self._cb_memory_store_symbolic_full)
        if hasattr(callbacks, 'set_memory_load_symbolic_full'):
            callbacks.set_memory_load_symbolic_full(self._cb_memory_load_symbolic_full)
        # state.inspect MVP (angr-uq4n.2, angr-d46u) — register dispatchers
        # even when no BPs are set so the Rust side has a target if
        # instrumentation fires unexpectedly. The bitmask gates dispatch.
        # Iterate the single source of truth so adding a new event
        # auto-wires registration. Read directly from the module rather
        # than self._INSPECT_EVENT_BITS so this method works even before
        # __init__ has finished. The hasattr() check tolerates older .so
        # builds.
        from angr.exploration.rust_state_proxy import _RUST_INSPECT_EVENT_BITS
        for evt in _RUST_INSPECT_EVENT_BITS:
            setter_name = f'set_inspect_{evt}'
            cb_name = f'_cb_inspect_{evt}'
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
                hook_mem = (
                    self._rust_mgr.get_state_hook_symbolic_memory(lookup_id)
                    if lookup_id is not None else {}
                )
                if hook_mem:
                    for mem_addr, (ast, mem_size) in hook_mem.items():
                        if mem_addr <= addr < mem_addr + mem_size:
                            offset = addr - mem_addr
                            if offset == 0 and size == mem_size:
                                concrete = state.solver.eval(ast).to_bytes(size, 'little')
                                self._register_handle(id(ast), ast, addr=addr, size=size, state_id=lookup_id)
                                if _DBG:
                                    l.debug(f"Memory load hit preserved symbolic at 0x{addr:x}")
                                return (concrete, True, ast)
                            elif offset == 0 and size < mem_size:
                                extracted = claripy.Extract(size * 8 - 1, 0, ast)
                                concrete = state.solver.eval(extracted).to_bytes(size, 'little')
                                self._register_handle(id(extracted), extracted, addr=addr, size=size, state_id=lookup_id)
                                return (concrete, True, extracted)

                val = state.memory.load(addr, size, endness=state.arch.memory_endness)

                # Coerce thunks/callables to actual values
                coerce_attempts = 0
                while callable(val) and not hasattr(val, 'op') and coerce_attempts < 3:
                    try:
                        val = val()
                        coerce_attempts += 1
                    except Exception:
                        # cat-(b) FALLBACK WITH LOSS: thunk in memory failed to resolve;
                        # replace with a fresh BVS so the load proceeds. Caller pays a
                        # downstream symbolic constraint instead of a hard error.
                        # angr-ymoe (2026-05-22): measured 0 fires across 21 fast-tier
                        # benches. Path is dead in practice; counter acts as a watchdog
                        # — if it ever goes non-zero, escalate to a Rust-side fresh
                        # symbol (see angr-4pm1's _set_state_register_symbolic_ast
                        # FFI shim for the pattern).
                        if _DBG:
                            l.debug(f"Memory load thunk at 0x{addr:x} failed to resolve, creating symbolic")
                        val = claripy.BVS(f"mem_thunk_{addr:x}", size * 8)
                        self._stats_orphan_bvs_mem_thunk += 1
                        break

                if not hasattr(val, 'op'):
                    l.warning(f"Memory load at 0x{addr:x} returned invalid type: {type(val)}")
                    return (bytes(size), False, None)

                is_symbolic = getattr(val, 'symbolic', False)
                if is_symbolic:
                    handle_id = id(val)
                    self._register_handle(handle_id, val, addr=addr, size=size, state_id=state_id)
                    concrete = state.solver.eval(val).to_bytes(size, 'little')
                    return (concrete, True, val)
                else:
                    concrete = state.solver.eval(val).to_bytes(size, 'little')
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
        try:
            val = claripy.BVV(int.from_bytes(data, 'little'), len(data) * 8)
            state.memory.store(addr, val, endness=state.arch.memory_endness)
        except (SimError, ClaripyError) as e:
            # cat-(c) WRONG-ANSWER RISK: memory store silently dropped on Sim/
            # Claripy error — subsequent loads see stale data. Already warns.
            l.warning(f"Memory store error at 0x{addr:x}: {e}")

    def _cb_lift_block(self, addr: int, opt_level: int = None, dirty_bytes: bytes = None) -> str:
        _lb_start = time.perf_counter_ns()
        try:
            try:
                kwargs = {}
                if opt_level is not None:
                    kwargs['opt_level'] = opt_level
                if dirty_bytes is not None:
                    # SMC: Rust signaled that this lift range is on a page that
                    # has been overwritten via state.memory. The cle static
                    # binary buffer is stale; lift the fresh bytes Rust sent
                    # instead.
                    kwargs['byte_string'] = dirty_bytes
                block = self._project.factory.block(addr, **kwargs)
                irsb = block.vex
                return self._serialize_irsb(irsb)
            except (SimEngineError, ClaripyError, PyVEXError) as e:
                # cat-(c) WRONG-ANSWER RISK: lift returned empty IRSB; Rust will
                # treat the block as a no-op step. Already warns.
                l.warning(f"Lift error at 0x{addr:x}: {e}")
                return '{}'
        finally:
            self._perf_stats.record_lift_block(time.perf_counter_ns() - _lb_start)

    def _cb_fetch_page(self, page_addr: int) -> tuple:
        _fp_start = time.perf_counter_ns()
        try:
            state = self._get_default_state()
            if state is None:
                return (bytes(4096), 0, False)
            try:
                data = state.memory.load(page_addr, 4096, endness=state.arch.memory_endness)
                is_symbolic = getattr(data, 'symbolic', False)
                if is_symbolic:
                    if _DBG:
                        l.debug(f"fetch_page 0x{page_addr:x}: has symbolic data, declining")
                    return (bytes(4096), 0, False)
                concrete = state.solver.eval(data).to_bytes(4096, 'little')
                return (concrete, 7, True)
            except (SimError, ClaripyError):
                # cat-(b) FALLBACK WITH LOSS: page fetch failed; return empty page
                # with perms=0 so Rust marks it inaccessible (a load there will
                # error rather than silently succeed). Already debug-logs.
                l.debug("fetch_page 0x%x: failed to load/eval, returning empty", page_addr, exc_info=True)
                return (bytes(4096), 0, False)
        finally:
            self._perf_stats.record_fetch_page(time.perf_counter_ns() - _fp_start)

    def _cb_get_register(self, offset: int, size: int) -> Tuple[bytes, bool, Optional[object]]:
        state = self._get_callback_state() or self._get_default_state()
        if state is None:
            return (bytes(size), False, None)
        try:
            val = state.registers.load(offset, size, endness=state.arch.register_endness)
            is_sym = getattr(val, 'symbolic', False)
            concrete = state.solver.eval(val).to_bytes(size, 'little')
            if is_sym:
                self._register_handle(id(val), val)
                return (concrete, True, val)
            return (concrete, False, None)
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: register read failed — Rust receives
            # zero bytes where the SimState may have a real value. Already
            # warns.
            l.warning(f"get_register error at offset {offset}: {e}")
            return (bytes(size), False, None)

    def _cb_put_register(self, offset: int, data: bytes):
        state = self._get_callback_state() or self._get_default_state()
        if state is None:
            return
        try:
            val = claripy.BVV(int.from_bytes(data, 'little'), len(data) * 8)
            state.registers.store(offset, val, endness=state.arch.register_endness)
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: register write failed — Python state
            # diverges from Rust on this register. Already warns.
            l.warning(f"put_register error at offset {offset}: {e}")

    def _cb_dirty_call(self, name: str, args: list, ret_ty_bits: int) -> Tuple[bytes, bool, Optional[object]]:
        state = self._get_callback_state() or self._get_default_state()
        if state is None:
            return (bytes(ret_ty_bits // 8), False, None)
        try:
            from angr.engines.vex.heavy import dirty as dirty_module

            if not hasattr(dirty_module, name):
                l.warning(f"No dirty call handler for {name}")
                return (bytes(ret_ty_bits // 8), False, None)

            handler = getattr(dirty_module, name)
            claripy_args = [claripy.BVV(arg, 64) for arg in args]
            result, constraints = handler(state, *claripy_args)

            if constraints:
                for c in constraints:
                    state.solver.add(c)

            if result is None:
                return (bytes(ret_ty_bits // 8), False, None)

            is_sym = getattr(result, 'symbolic', False)
            concrete_val = state.solver.eval(result)
            num_bytes = ret_ty_bits // 8
            concrete_bytes = concrete_val.to_bytes(num_bytes, 'little')

            if is_sym:
                self._register_handle(id(result), result)
                return (concrete_bytes, True, result)
            return (concrete_bytes, False, None)

        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: dirty call handler raised; Rust sees
            # zero bytes for the result. Already warns.
            l.warning(f"dirty_call {name} error: {e}")
            return (bytes(ret_ty_bits // 8), False, None)

    def _cb_resolve_function(self, addr: int, name: Optional[str]) -> Optional[Tuple[str, int, bool]]:
        """Resolve an unmodeled function call."""
        if hasattr(self._project, '_sim_procedures'):
            if addr in self._project._sim_procedures:
                proc = self._project._sim_procedures[addr]
                proc_name = proc.__class__.__name__ if hasattr(proc, '__class__') else str(proc)
                num_args = getattr(proc, 'num_args', 0) or 0
                no_ret = getattr(proc, 'NO_RET', False)
                return (proc_name, num_args, no_ret)

        if hasattr(self._project, 'loader'):
            obj = self._project.loader.find_object_containing(addr)
            if obj:
                in_plt = False
                for section in obj.sections:
                    if section.name in ('.plt', '.plt.got', '.plt.sec') and section.min_addr <= addr < section.max_addr:
                        in_plt = True
                        break

                if in_plt:
                    proc_by_name = {}
                    for proc_addr, proc in self._project._sim_procedures.items():
                        proc_name = proc.__class__.__name__ if hasattr(proc, '__class__') else str(proc)
                        proc_by_name[proc_name] = (proc_addr, proc)

                    got_to_sym = {}
                    if hasattr(obj, 'jmprel'):
                        for sym_name, reloc in obj.jmprel.items():
                            got_to_sym[reloc.rebased_addr] = sym_name

                    try:
                        block = self._project.factory.block(addr, num_inst=1)
                        insn = block.capstone.insns[0] if block.capstone.insns else None
                        if insn and insn.mnemonic == 'jmp':
                            for op in insn.operands:
                                if op.type == 3:  # CS_OP_MEM
                                    got_addr = insn.address + insn.size + op.mem.disp
                                    if got_addr in got_to_sym:
                                        sym_name = got_to_sym[got_addr]
                                        if sym_name in proc_by_name:
                                            proc_addr, proc = proc_by_name[sym_name]
                                            num_args = getattr(proc, 'num_args', 0) or 0
                                            no_ret = getattr(proc, 'NO_RET', False)
                                            l.debug(f"Resolved PLT at 0x{addr:x} to {sym_name} (GOT 0x{got_addr:x})")
                                            return (sym_name, num_args, no_ret)
                                    else:
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
                        num_args = getattr(proc_class, 'num_args', 0) or 0
                        no_ret = getattr(proc_class, 'NO_RET', False)
                        l.debug(f"Resolved {name} to {lib_name}:{name}")
                        return (name, num_args, no_ret)
            except ImportError:
                # cat-(a) EXPECTED CONTROL FLOW: optional SIM_PROCEDURES import;
                # if absent, skip name-based resolution.
                pass

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
                    # cat-(a) EXPECTED CONTROL FLOW: optional SIM_PROCEDURES import
                    # (symbol name path); same as above.
                    pass

        if hasattr(self._project, 'loader'):
            obj = self._project.loader.find_object_containing(addr)
            if obj and obj.binary is not None:
                for section in obj.sections:
                    if section.is_executable and section.min_addr <= addr < section.max_addr:
                        if section.name not in ('.plt', '.plt.got', '.plt.sec'):
                            l.debug(f"Internal function at 0x{addr:x} - returning pass-through")
                            return ("__internal_passthrough__", 0, False)

        l.debug(f"Could not resolve function at 0x{addr:x} (name={name})")
        return None

    def _cb_memory_store_batch(self, stores: list):
        state = self._get_per_fork_state()
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
            except (SimError, ClaripyError) as e:
                # cat-(c) WRONG-ANSWER RISK: batch memory store partially failed;
                # some addresses keep stale data. Logs at debug; promote upstream
                # if a divergence is observed.
                l.debug(f"Batch memory store failed at 0x{addr:x}: {e}")

    def _cb_memory_load_batch(self, loads: list) -> list:
        state = self._get_per_fork_state()
        if state is None:
            return [(bytes(size), False, None) for _, size in loads]

        results = []
        for addr, size in loads:
            try:
                val = state.memory.load(addr, size, endness=state.arch.memory_endness)
                is_sym = getattr(val, 'symbolic', False)
                concrete = state.solver.eval(val).to_bytes(size, 'little')
                if is_sym:
                    self._register_handle(id(val), val, addr=addr, size=size)
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

    def _cb_batch_fetch_pages(self, page_addrs: list) -> list:
        state = self._get_callback_state() or self._get_default_state()
        if state is None:
            return [(bytes(4096), 0, True) for _ in page_addrs]

        results = []
        for page_addr in page_addrs:
            try:
                data = state.memory.load(page_addr, 4096, endness='Iend_LE')
                if getattr(data, 'symbolic', False):
                    concrete = state.solver.eval(data).to_bytes(4096, 'little')
                    results.append((concrete, 7, False))
                else:
                    concrete = state.solver.eval(data).to_bytes(4096, 'little')
                    results.append((concrete, 7, True))
            except (SimError, ClaripyError) as e:
                # cat-(c) WRONG-ANSWER RISK: batch page fetch partial failure;
                # returns zero page — Rust sees concrete zero. Debug-logs.
                l.debug(f"Batch page fetch failed at 0x{page_addr:x}: {e}")
                results.append((bytes(4096), 0, True))
        return results

    def _cb_memory_store_symbolic_value(self, addr: int, ast):
        """Store a symbolic value (claripy AST) to Python state memory."""
        state = self._get_per_fork_state()
        if state is None or ast is None:
            return
        try:
            if hasattr(ast, 'length') and ast.length:
                size = ast.length // 8
                state.memory.store(addr, ast, endness=state.arch.memory_endness,
                                   inspect=False, disable_actions=True)
                self._register_handle(id(ast), ast, addr=addr, size=size)
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
        ``libc/gets.py``, ``libc/fgets.py``), route the store through the
        Rust Multi-cell lazy path on the current state before falling back
        to Python's memory model. On success the write lands directly in
        Rust memory and the Python state is skipped; on failure we fall
        through to the existing Python path so the write is not lost.
        """
        state = self._get_per_fork_state()
        if state is None or addr_ast is None or data_ast is None:
            return
        if self._try_multi_cell_store(addr_ast, data_ast):
            return
        try:
            state.memory.store(addr_ast, data_ast, endness=state.arch.memory_endness,
                               inspect=False, disable_actions=True)
            self._register_handle(id(data_ast), data_ast)
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
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: Rust side raised; debug-log and
            # let the Python fallback try.
            l.debug(f"state_memory_store_symbolic_multi raised: {e}")
            return False
        if ok:
            self._register_handle(id(data_ast), data_ast)
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
        try:
            ast = state.memory.load(addr_ast, size, endness=state.arch.memory_endness,
                                    inspect=False, disable_actions=True)
            if ast is not None:
                self._register_handle(id(ast), ast, size=size)
            return ast
        except (SimError, ClaripyError) as e:
            # cat-(c) WRONG-ANSWER RISK: symbolic-address load failed; returns
            # a fresh BVS that has no relationship to the actual symbolic value
            # in memory — downstream constraint sync will diverge. Debug-logs.
            # angr-ymoe (2026-05-22): measured 0 fires across 21 fast-tier
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
        if self._callbacks is None or not hasattr(self._callbacks, 'set_inspect_enabled'):
            return
        mask = 0
        for event, bit in self._INSPECT_EVENT_BITS.items():
            if self._inspect_breakpoints.get(event):
                mask |= 1 << bit
        self._callbacks.set_inspect_enabled(mask)

    def _make_inspect_state_for(self, state_id: int):
        """Build the state object the user action sees during inspect dispatch.

        Returns a RustStateProxy bound to `state_id` (preferred). Falls
        back to whatever the cached default state is if the state_id is
        unknown — keeps the BP firing rather than silently dropping.
        """
        if state_id is not None and state_id >= 0:
            from angr.exploration.rust_state_proxy import RustStateProxy
            try:
                return RustStateProxy(self._rust_mgr, state_id, self._project,
                                      python_mgr=self)
            except Exception:
                # cat-(a) EXPECTED CONTROL FLOW: state may have been
                # dropped from Rust between dispatch and proxy build.
                pass
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
        """
        if self._inspect_dispatch_depth > 0:
            return  # reentrancy guard (uq4n.4)
        bps = self._inspect_breakpoints.get(event_type)
        if not bps:
            return  # bitmask race — Rust fired but Python already cleared
        state = self._make_inspect_state_for(state_id)
        if state is None:
            return
        proxy = self._get_inspect_proxy()
        proxy.set_state(state)
        self._inspect_dispatch_depth += 1
        try:
            proxy.action(event_type, when, **attrs)
        finally:
            self._inspect_dispatch_depth -= 1

    def _addr_attr_for(self, addr: int):
        """Wrap an integer address in a claripy BVV at the project's word size."""
        if not isinstance(addr, int):
            return addr
        bits = self._project.arch.bits if self._project is not None else 64
        return claripy.BVV(addr, bits)

    def _cb_inspect_mem_read(
        self,
        state_id: int,
        when: str,
        addr: int,
        size: int,
        value_ast,
        endness: str,
    ):
        """PyO3 callback target for mem_read events from Rust."""
        try:
            self._dispatch_inspect_event(
                "mem_read", state_id, when,
                mem_read_address=self._addr_attr_for(addr),
                mem_read_length=size,
                mem_read_expr=value_ast,
                mem_read_endness=endness,
            )
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: user BP action errored. Log and
            # swallow so the engine keeps stepping; the user can see the
            # warning in stderr.
            l.warning("inspect mem_read dispatch failed: %s: %s", type(e).__name__, e)

    def _cb_inspect_mem_write(
        self,
        state_id: int,
        when: str,
        addr: int,
        size: int,
        value_ast,
        endness: str,
    ):
        """PyO3 callback target for mem_write events from Rust."""
        try:
            self._dispatch_inspect_event(
                "mem_write", state_id, when,
                mem_write_address=self._addr_attr_for(addr),
                mem_write_length=size,
                mem_write_expr=value_ast,
                mem_write_endness=endness,
            )
        except Exception as e:
            l.warning("inspect mem_write dispatch failed: %s: %s", type(e).__name__, e)

    def _cb_inspect_reg_read(
        self,
        state_id: int,
        when: str,
        offset: int,
        size: int,
        value_ast,
    ):
        """PyO3 callback target for reg_read events from Rust (IRExpr::Get)."""
        try:
            self._dispatch_inspect_event(
                "reg_read", state_id, when,
                reg_read_offset=offset,
                reg_read_length=size,
                reg_read_expr=value_ast,
                reg_read_condition=None,
                reg_read_endness=None,
            )
        except Exception as e:
            l.warning("inspect reg_read dispatch failed: %s: %s", type(e).__name__, e)

    def _cb_inspect_reg_write(
        self,
        state_id: int,
        when: str,
        offset: int,
        size: int,
        value_ast,
    ):
        """PyO3 callback target for reg_write events from Rust (IRStmt::Put)."""
        try:
            self._dispatch_inspect_event(
                "reg_write", state_id, when,
                reg_write_offset=offset,
                reg_write_length=size,
                reg_write_expr=value_ast,
                reg_write_condition=None,
                reg_write_endness=None,
            )
        except Exception as e:
            l.warning("inspect reg_write dispatch failed: %s: %s", type(e).__name__, e)

    def _cb_inspect_instruction(self, state_id: int, when: str, addr: int):
        """PyO3 callback target for instruction events (one per IMark)."""
        try:
            self._dispatch_inspect_event(
                "instruction", state_id, when,
                instruction=addr,
            )
        except Exception as e:
            l.warning("inspect instruction dispatch failed: %s: %s", type(e).__name__, e)

    def _cb_inspect_irsb(self, state_id: int, when: str, addr: int):
        """PyO3 callback target for irsb events (one per basic block)."""
        try:
            self._dispatch_inspect_event(
                "irsb", state_id, when,
                address=addr,
            )
        except Exception as e:
            l.warning("inspect irsb dispatch failed: %s: %s", type(e).__name__, e)

    def _cb_inspect_exit(
        self,
        state_id: int,
        when: str,
        target: int,
        jumpkind: str,
        guard_ast,
    ):
        """PyO3 callback target for VEX conditional exit events."""
        try:
            self._dispatch_inspect_event(
                "exit", state_id, when,
                exit_target=self._addr_attr_for(target),
                exit_guard=guard_ast,
                exit_jumpkind=jumpkind,
            )
        except Exception as e:
            l.warning("inspect exit dispatch failed: %s: %s", type(e).__name__, e)

    def _load_binary_regions(self):
        """Load binary code regions for native lifting."""
        regions = []

        for obj in self._project.loader.all_objects:
            # Skip cle pseudo-objects (externs/tls/kernel) — their `binary`
            # is a synthetic string like 'cle##externs', not None, so the
            # plain None check below is not enough.
            if obj.binary is None:
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
                    data = self._project.loader.memory.load(
                        region.min_addr,
                        region.max_addr - region.min_addr
                    )
                    regions.append((region.min_addr, bytes(data)))
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: executable region not loaded into Rust;
                    # attempts to lift a block in this region will fall through to
                    # Python via lift_block. Debug-logs.
                    name = getattr(region, 'name', repr(region))
                    l.debug(f"Could not load region {name}: {e}")

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
        """Check if state has user-created symbolic data in memory or registers.

        Detects symbolic argv, symbolic input buffers, ``state.regs.a0 =
        BVS(...)``-style register mutations, etc. by scanning:
        1. The stack page near SP for BVS variables that aren't unconstrained fill.
        2. All memory pages with symbolic_data for user-created variables
           (e.g., state.memory.store(addr, BVS(...))).
        3. Architectural registers for user-set symbolic values
           (e.g., state.regs.a0 = BVS(...)). Without this, the disk init
           cache silently replaces the user's state with a cached blank_state,
           losing the user's symbolic register mutations — see angr-g9hy.
        """
        _user_prefixes = ('mem_', 'reg_', 'unconstrained')
        try:
            sp = state.solver.eval(state.regs.sp)
            sp_page = sp & ~PAGE_MASK
            page_data = state.memory.load(
                sp_page, PAGE_SIZE, endness='Iend_BE',
                inspect=False, disable_actions=True)
            if page_data.symbolic:
                for name in page_data.variables:
                    if not name.startswith(_user_prefixes):
                        return True
        except (AttributeError, KeyError, TypeError):
            # cat-(a) EXPECTED CONTROL FLOW: stack-page probe for user symbolic
            # data failed; fall through to the per-page bitmap scan below.
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
        except (AttributeError, KeyError, TypeError):
            # cat-(a) EXPECTED CONTROL FLOW: per-page symbolic-data scan hit
            # missing attribute; conclude no user symbolic data, return False.
            pass
        # Check the register file for user-set symbolic values. blank_state's
        # default symbol-fill is lazy (BVS allocated only on first read), so
        # uninitialized registers do NOT appear in `symbolic_data`. The
        # values that DO appear are either initialization writes or user
        # mutations like `state.regs.a0 = BVS(...)`. A non-default variable
        # prefix on any of these is treated as user-supplied — the cache
        # can't round-trip it.
        try:
            regs_mem = state.registers
            for _page_num, page in regs_mem._pages.items():
                sd = getattr(page, 'symbolic_data', None)
                if not sd:
                    continue
                for _offset, bv in sd.items():
                    if hasattr(bv, 'variables'):
                        for name in bv.variables:
                            if not name.startswith(_user_prefixes):
                                return True
        except (AttributeError, KeyError, TypeError):
            # cat-(a) EXPECTED CONTROL FLOW: registers storage lacks _pages
            # (non-DefaultMemory plugin?); skip — caller treats False as "no
            # user symbolic registers detected".
            pass
        return False

    @classmethod
    def _disk_cache_key(cls, binary_path: str, arch_name: str = "") -> str:
        """Compute a cache key from binary content hash, version axes, and arch.

        Combines (binary_hash, _RUST_CACHE_VERSION, _PYTHON_METADATA_VERSION,
        arch_name) so a change on any axis lands at a different filename and
        treats stale entries as misses. Results are memoized per
        (binary_path, arch_name) to avoid re-hashing the same file on every
        RustExplorationManager construction (~0.5ms for 100KB binary).
        """
        memo_key = (binary_path, arch_name)
        cached = cls._disk_key_cache.get(memo_key)
        if cached is not None:
            return cached
        try:
            h = hashlib.md5()
            # Mix all version dimensions into the hash so any one bumping
            # produces a fresh key without colliding with old cache files.
            h.update(
                f"r{_RUST_CACHE_VERSION}:p{_PYTHON_METADATA_VERSION}:"
                f"a{arch_name}:".encode()
            )
            with open(binary_path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    h.update(chunk)
            result = h.hexdigest()
            cls._disk_key_cache[memo_key] = result
            return result
        except OSError:
            # cat-(b) FALLBACK WITH LOSS: cannot read binary for hash; disk
            # cache disabled for this binary (empty key returns ''-keyed nothing).
            return ""

    def _save_init_to_disk_cache(self, cache_key: str, state: "angr.SimState"):
        """Save essential post-init state data to disk cache.

        Stores: addr, registers, stack page, continuation addrs, and loader
        memory pages + lazy regions for fast memory sync on warm runs.
        """
        try:
            cache_dir = self._disk_cache_dir()
            os.makedirs(cache_dir, exist_ok=True)

            page_size = PAGE_SIZE
            loader = self._project.loader

            stack_page, stack_lazy_region = _extract_stack_page(state, page_size)
            batch_pages, loader_lazy_regions, mapped_page_addrs = _extract_loader_pages(
                loader, page_size)

            lazy_regions = []
            if stack_lazy_region is not None:
                lazy_regions.append(stack_lazy_region)
            lazy_regions.extend(loader_lazy_regions)

            callstack_frames, continuation_addrs = _extract_callstack_snapshot(state)

            data = {
                'addr': state.addr,
                'registers': _extract_register_snapshot(state, self._project.arch),
                'stack_page': stack_page,
                'continuation_addrs': continuation_addrs,
                'batch_pages': batch_pages,
                'lazy_regions': lazy_regions,
                'section_patches': _extract_section_patches(state, loader),
                'extra_pages': _extract_extra_pages(
                    state, page_size, mapped_page_addrs,
                    stack_page[0] if stack_page is not None else None),
                'callstack_frames': callstack_frames,
            }

            cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            l.debug(f"Saved init cache to {cache_path} "
                    f"({os.path.getsize(cache_path)} bytes, "
                    f"{len(batch_pages)} pages)")
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: disk cache write failed (e.g., OOM,
            # permission, ENOSPC). Run continues without persistent caching.
            l.debug(f"Failed to save disk cache: {e}")

    def _get_cached_blank_state(self, addr: int) -> "angr.SimState":
        """Get a blank state, using class-level cache when possible.

        blank_state() is expensive (~1ms) due to plugin initialization.
        Caching + copy() is <0.1ms.
        """
        binary_path = getattr(self._project.loader.main_object, 'binary', None) or ''
        cache_key = (binary_path, addr)
        cached = RustExplorationManager._blank_state_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        state = self._project.factory.blank_state(addr=addr)
        if binary_path and len(RustExplorationManager._blank_state_cache) < RustExplorationManager._blank_state_cache_max:
            RustExplorationManager._blank_state_cache[cache_key] = state.copy()
        return state

    def _load_init_pickle(self, cache_key: str):
        """Read and unpickle the disk cache file. Returns the raw data dict
        on hit, None on miss or read failure. Pure I/O — no state mutation."""
        try:
            cache_path = os.path.join(self._disk_cache_dir(), f"{cache_key}.pkl")
            if not os.path.exists(cache_path):
                return None
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError) as e:
            # cat-(b) FALLBACK WITH LOSS: disk cache read failed / corrupt;
            # treated as a cache miss. Caller pays full Python init.
            l.debug(f"Disk cache read failed: {e}")
            return None

    def _deserialize_init_state(self, data: dict):
        """Build a SimState + memory_cache from a cache data dict. Pure
        function over `self._project` and the cached blank-state pool — no
        mutation of manager-owned metadata dicts (those happen in
        `_apply_init_side_effects`)."""
        state = self._get_cached_blank_state(data['addr'])
        for reg_name, val in data['registers'].items():
            try:
                setattr(state.regs, reg_name, val)
            except (AttributeError, TypeError, ValueError):
                # cat-(b) FALLBACK WITH LOSS: per-register restore failed; that
                # register stays at the blank-state default — may diverge from the
                # pre-cached value.
                pass
        if data.get('stack_page'):
            sp_page, page_bytes = data['stack_page']
            state.memory.store(
                sp_page, claripy.BVV(page_bytes), endness='Iend_BE',
                inspect=False, disable_actions=True)

        # Restore extra memory pages created during init
        # (e.g., ctype tables at 0xc0000000 written by SimProcedures)
        for page_addr, page_bytes in data.get('extra_pages', []):
            try:
                state.memory.store(
                    page_addr, claripy.BVV(page_bytes), endness='Iend_BE',
                    inspect=False, disable_actions=True)
            except (TypeError, ValueError):
                # cat-(b) FALLBACK WITH LOSS: extra page restore failed; that page
                # stays blank and Rust sees concrete zeros there.
                pass

        # Restore callstack frames
        callstack_frames = data.get('callstack_frames', [])
        if callstack_frames and len(callstack_frames) > 1:
            # The first frame is the top of the callstack (main's frame).
            # We need to push frames from bottom to top.
            try:
                from angr.state_plugins.callstack import CallStack
                cs = state.callstack
                for frame_data in reversed(callstack_frames[:-1]):
                    # Skip the bottom sentinel frame (all zeros)
                    if frame_data['call_site_addr'] == 0 and frame_data['func_addr'] == 0:
                        continue
                    cs.call(
                        frame_data['call_site_addr'],
                        frame_data['func_addr'],
                        return_address=frame_data['ret_addr'],
                        stack_pointer=frame_data['stack_ptr'],
                    )
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: callstack restoration failed; the
                # state has a top-of-stack frame but ret-chain may be incomplete.
                # Debug-logs.
                l.debug(f"Failed to restore callstack: {e}")

        mem_cache = None
        if data.get('batch_pages') is not None:
            mem_cache = {
                'batch_pages': data['batch_pages'],
                'lazy_regions': data.get('lazy_regions', []),
                'section_patches': data.get('section_patches', []),
                'stack_page': data.get('stack_page'),
            }
        return state, mem_cache

    def _apply_init_side_effects(self, data: dict) -> None:
        """Populate manager-owned metadata from a cache data dict:
        `_pending_procedure_data` (continuation slots) and `_precomputed_regs`
        (fast Rust register sync). Separated from deserialization so the
        SimState construction can be tested in isolation."""
        for cont_addr in data.get('continuation_addrs', []):
            if cont_addr > 0:
                self._pending_procedure_data.setdefault(cont_addr, None)
        self._precomputed_regs = data.get('registers', {})

    def _load_init_from_disk_cache(self, cache_key: str):
        """Load post-init state from disk cache.

        Returns (SimState, memory_cache_data) on hit, (None, None) on miss.
        memory_cache_data contains pre-computed loader pages and lazy regions
        for fast memory sync.

        Phases (each independently testable):
        1. `_load_init_pickle` — pure I/O.
        2. `_deserialize_init_state` — pure SimState construction.
        3. `_apply_init_side_effects` — manager-owned metadata mutation.
        """
        data = self._load_init_pickle(cache_key)
        if data is None:
            return None, None
        try:
            state, mem_cache = self._deserialize_init_state(data)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: pickle deserialization succeeded but
            # state construction failed; treat as cache miss. Debug-logs.
            l.debug(f"Disk cache deserialization failed: {e}")
            return None, None
        self._apply_init_side_effects(data)
        l.info(f"Disk cache hit: restored state at 0x{data['addr']:x}")
        return state, mem_cache

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
                    # cat-(a) EXPECTED CONTROL FLOW: continuation addr is symbolic /
                    # non-castable; walk to the next frame.
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
        main_obj = self._project.loader.main_object
        addr = state.addr
        cache_key = getattr(main_obj, 'binary', None) or ''
        mem_key = self._compute_mem_init_key(state, cache_key)
        disk_key = self._compute_disk_init_key(state, cache_key)

        if addr == self._project.entry:
            cached = self._try_in_memory_init_cache(state, mem_key)
            if cached is not None:
                return cached
            cached = self._try_disk_init_cache(state, disk_key)
            if cached is not None:
                return cached
            l.info(f"State at entry point 0x{addr:x}, running Python init to main")
        else:
            obj = self._project.loader.find_object_containing(addr)
            if obj is not None and obj.binary is not None and not obj.binary.startswith('cle##'):
                return state  # In a real binary (not entry), no init needed
            if addr not in self._project._sim_procedures:
                return state  # Not a SimProcedure (e.g., LinuxLoader), don't pre-run
            l.info(f"State at loader address 0x{addr:x}, running Python init to reach main binary")
            cached = self._try_disk_init_cache(state, disk_key)
            if cached is not None:
                return cached

        try:
            main_addr = self._resolve_main_address()
            return self._step_python_to_main(state, main_addr, mem_key, disk_key, main_obj)
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: Python init step failed; original
            # state is used unmodified, so init-side-effects (constructors,
            # .init_array) may not have run. Already warns.
            l.warning(f"Python init failed: {e}, using original state")
            return state

    def _apply_state_metadata(self, src_state: "angr.SimState",
                              dst_state: "angr.SimState") -> None:
        """Copy constraints, globals, and LAZY_SOLVES / STRICT_PAGE_ACCESS /
        ENABLE_NX / NO_IP_CONCRETIZATION / NO_SYMBOLIC_JUMP_RESOLUTION /
        KEEP_IP_SYMBOLIC options from src to dst.

        Options are mirrored — added when src has them, removed when src
        doesn't. The remove half matters for the in-memory init cache: a
        cached state populated from a prior STRICT_PAGE_ACCESS run on the
        same binary would otherwise leak that option to a subsequent caller
        that didn't request it (and downstream `set_enforce_permissions(True)`
        would then surface spurious permission errors). Same reasoning for
        ENABLE_NX → `set_enforce_nx(True)`, NO_IP_CONCRETIZATION →
        `set_no_ip_concretization(True)`, NO_SYMBOLIC_JUMP_RESOLUTION →
        `set_no_symbolic_jump_resolution(True)`, and KEEP_IP_SYMBOLIC →
        `set_keep_ip_symbolic(True)`.
        """
        for c in src_state.solver.constraints:
            dst_state.solver.add(c)
        if 'globals' in src_state.plugins:
            for k, v in src_state.globals.items():
                dst_state.globals[k] = v
        try:
            from angr import sim_options as o
            for opt in (
                o.LAZY_SOLVES,
                o.STRICT_PAGE_ACCESS,
                o.ENABLE_NX,
                o.NO_IP_CONCRETIZATION,
                o.NO_SYMBOLIC_JUMP_RESOLUTION,
                o.KEEP_IP_SYMBOLIC,
            ):
                if opt in src_state.options:
                    dst_state.options.add(opt)
                else:
                    dst_state.options.discard(opt)
        except (ImportError, Exception):
            # cat-(a) EXPECTED CONTROL FLOW: sim_options optional import;
            # without it the option-mirror step is skipped.
            pass

    def _compute_disk_init_key(self, state: "angr.SimState", cache_key: str) -> str:
        """Compute disk init cache key. Empty string means caching is disabled
        (no binary path, or state has user symbolic data that blank_state can't
        round-trip)."""
        if not cache_key:
            return ''
        if self._state_has_user_symbolic(state):
            return ''
        arch_name = getattr(self._project.arch, 'name', '') or ''
        return self._disk_cache_key(cache_key, arch_name)

    def _compute_mem_init_key(self, state: "angr.SimState", cache_key: str) -> str:
        """In-memory init cache key. Returns '' (caching disabled) when the
        state has user-created symbolic data, mirroring _compute_disk_init_key.
        Without this gate, a user-symbolic store on the input state survives
        through Python init and ends up in the cached post-init state. Later
        callers that hit the cache via .copy() inherit those stores while
        their own stores are silently lost — _apply_state_metadata copies
        constraints/options but not memory pages.
        """
        if not cache_key:
            return ''
        if self._state_has_user_symbolic(state):
            return ''
        return cache_key

    def _try_in_memory_init_cache(self, state: "angr.SimState",
                                  cache_key: str) -> Optional["angr.SimState"]:
        """Try the per-process init cache (~180ms savings). Returns ready state or None."""
        if not cache_key or cache_key not in RustExplorationManager._init_cache:
            return None
        cached = RustExplorationManager._init_cache[cache_key]
        l.info(f"Init cache hit for {cache_key}, copying state at 0x{cached.addr:x}")
        new_state = cached.copy()
        self._apply_state_metadata(state, new_state)
        return new_state

    def _try_disk_init_cache(self, state: "angr.SimState",
                             disk_key: str) -> Optional["angr.SimState"]:
        """Try the persistent disk init cache. Returns ready state or None.

        Note: only safe when the source state has no user symbolic data —
        blank_state from cache can't preserve symbolic arguments (e.g., argv BVS).
        Caller must enforce that via _compute_disk_init_key.
        """
        if not disk_key:
            return None
        disk_state, mem_cache = self._load_init_from_disk_cache(disk_key)
        if disk_state is None:
            return None
        self._apply_state_metadata(state, disk_state)
        self._extract_continuation_data(disk_state)
        self._mem_cache = mem_cache  # For fast _sync_memory_to_rust
        return disk_state

    def _resolve_main_address(self) -> Optional[int]:
        """Find main function address; for stripped binaries, parse _start's PUT(rdi)."""
        main_sym = self._project.loader.find_symbol('main')
        if main_sym:
            return main_sym.rebased_addr
        try:
            entry_block = self._project.factory.block(self._project.entry)
            vex = entry_block.vex
            rdi_offset = self._project.arch.registers.get('rdi', (None,))[0]
            if rdi_offset is None:
                rdi_offset = self._project.arch.registers.get('edi', (None,))[0]
            if rdi_offset is None:
                return None
            for stmt in reversed(vex.statements):
                s = str(stmt)
                if f'PUT(offset={rdi_offset})' in s or 'PUT(rdi)' in s:
                    import re
                    m_const = re.search(r'0x([0-9a-fA-F]+)', s)
                    if m_const:
                        candidate = int(m_const.group(1), 16)
                        main_obj = self._project.loader.main_object
                        if main_obj.min_addr <= candidate <= main_obj.max_addr:
                            l.info(f"Extracted main=0x{candidate:x} from _start's rdi")
                            return candidate
                    break
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: main address extraction from _start
            # disassembly failed; caller resolves None and uses the post-init
            # state at whatever PC step_python_to_main lands on. Debug-logs.
            l.debug(f"Could not extract main from _start: {e}")
        return None

    def _save_init_state_to_caches(self, result: "angr.SimState",
                                   cache_key: str, disk_key: str) -> None:
        """Persist a freshly-built init state to in-memory and disk caches."""
        if (cache_key and len(RustExplorationManager._init_cache)
                < RustExplorationManager._init_cache_max):
            RustExplorationManager._init_cache[cache_key] = result.copy()
        if disk_key:
            self._save_init_to_disk_cache(disk_key, result)

    def _step_python_to_main(self, state: "angr.SimState",
                             main_addr: Optional[int],
                             cache_key: str, disk_key: str,
                             main_obj) -> "angr.SimState":
        """Run Python SimulationManager until reaching main, then cache+return."""
        # Init-only addresses we never want to land on as "main"
        init_addrs = {self._project.entry}
        for obj in self._project.loader.all_objects:
            if hasattr(obj, 'entry') and obj.entry:
                init_addrs.add(obj.entry)

        # Use the REAL SimulationManager (not the monkey-patched factory)
        # to avoid infinite recursion when the factory is patched.
        from angr import SimulationManager
        sm = SimulationManager(project=self._project, active_states=[state])
        main_min = main_obj.min_addr
        main_max = main_obj.max_addr

        for step in range(500):
            if not sm.active:
                break

            if main_addr is not None:
                at_main = [s for s in sm.active if s.addr == main_addr]
                if at_main:
                    l.info(f"Python init complete: state reached main at 0x{main_addr:x} "
                           f"after {step} steps")
                    result = at_main[0]
                    self._extract_continuation_data(result)
                    self._save_init_state_to_caches(result, cache_key, disk_key)
                    return result

            # No main symbol: pick the first state inside the main binary that
            # isn't at _start, isn't at a SimProcedure, and is past the prologue.
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
                    self._save_init_state_to_caches(result, cache_key, disk_key)
                    return result

            sm.step()

        # Couldn't reach main within budget — fall back to whatever we have.
        if sm.active:
            best = sm.active[0]
            self._extract_continuation_data(best)
            l.warning(f"Python init: didn't reach main after 500 steps, "
                      f"using state at 0x{best.addr:x}")
            return best
        if sm.deadended:
            l.warning(f"Python init: all states deadended")
        return state

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
        unseen = {name for name in _REJECTED_OPTION_NAMES
                  if name in options and name not in self._warned_rejected_options}
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
        """Raise NotImplementedError if any ``_RAISE_OPTION_NAMES`` are set.

        These SimOptions request behavior the Rust engine cannot provide
        (action streams, eager concretization, conservative-write refusal,
        ret-emulation, calless short-circuits, ancestor strongrefs,
        symbolic register fill, etc.). Silent divergence has burned users
        in the past, so we hard-fail at the manager boundary to force a
        drop to the Python engine.
        """
        if not options:
            return
        offending = sorted(name for name in _RAISE_OPTION_NAMES if name in options)
        if not offending:
            return
        names = ", ".join(offending)
        raise NotImplementedError(
            f"SimOption(s) {{{names}}} request behavior the Rust engine "
            "cannot provide. Drop use_rust_engine=True (or remove these "
            "options from state.options) and rerun with the Python "
            "engine. See docs/advanced-topics/rust_engine.rst for the "
            "full matrix."
        )

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

        # angr-3ms1 step 1b: push the manager-wide opt-in for fork-time
        # SharedLineageSolver materialization onto this seed state's
        # solver context. Descendants inherit via SymContext::fork, so
        # this single call propagates to every state derived from this
        # seed. Default off keeps slice-1c's gate inert.
        if self._use_shared_lineage_solver:
            rust_state.set_use_shared_lineage_solver(True)

        # Set PC
        rust_state.pc = angr_state.addr

        # Push posix.brk so Rust's NativeBrkSyscall starts from the correct
        # base. The angr loader sets state.posix.brk to (binary last_addr +
        # page) — distinct from Rust's hardcoded default 0x1B00000 — so
        # without this push a brk syscall in Rust would compare against the
        # wrong base and either overlap mapped memory or hand out an address
        # the program can't use. Only push if it's still a plain int; once
        # Python's set_brk has wrapped it as a BV, we don't try to flatten.
        try:
            py_brk = getattr(getattr(angr_state, 'posix', None), 'brk', None)
            if isinstance(py_brk, int):
                rust_state.posix_brk = py_brk
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: posix.brk push to Rust failed; Rust's
            # brk syscall will base-from its hardcoded default and may overlap
            # mapped memory. Debug-logs.
            l.debug("posix.brk init push failed: %s", e)

        # Sync registers (use precomputed dict from disk cache when available)
        _t_reg = time.perf_counter_ns()
        precomputed = self._precomputed_regs
        self._precomputed_regs = None  # Consume once
        self._sync_registers_to_rust(angr_state, rust_state,
                                     precomputed_regs=precomputed)
        self._perf_stats.add_init_phase('register_sync', time.perf_counter_ns() - _t_reg)

        # Map memory regions
        _t_mem = time.perf_counter_ns()
        self._sync_memory_to_rust(angr_state, rust_state)
        self._perf_stats.add_init_phase('memory_sync', time.perf_counter_ns() - _t_mem)

        # Mirror angr's STRICT_PAGE_ACCESS and ENABLE_NX: when set on the
        # SimState, the Rust memory model rejects loads/stores that violate
        # per-page R/W bits (STRICT_PAGE_ACCESS) and instruction fetches from
        # non-X pages (ENABLE_NX, which Python additionally gates on
        # STRICT_PAGE_ACCESS — see angr/engines/vex/heavy/heavy.py:115-124).
        # add_state forks the state internally; both flags are preserved
        # through forks (see SymbolicMemory::fork in native/angr/src/memory/mod.rs).
        if hasattr(angr_state, 'options'):
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

            # Sync constraints from Python state to Rust solver.
            # When re-using a found state from a previous RustExplorationManager
            # (multi-stage explore pattern), the Python solver may have 0 constraints
            # because they all live in the old Rust solver. Detect this and transfer
            # constraints from the old Rust manager.
            #
            # Prefer Z3 pointer transfer (lossless) over claripy AST round-trip
            # (which silently drops constraints where claripy_to_rustbv fails).
            z3_transferred = False
            old_rust_mgr = getattr(angr_state.scratch, 'rust_mgr', None)
            old_state_id = getattr(angr_state.scratch, 'rust_found_state_id', None)
            if old_rust_mgr is not None and old_state_id is not None:
                try:
                    z3_ptrs = old_rust_mgr.export_z3_constraint_ptrs(old_state_id)
                    if z3_ptrs:
                        # Debug: compare solver states
                        try:
                            old_info = old_rust_mgr.debug_solver_info(old_state_id)
                            l.debug(f"OLD solver ({old_state_id}): {old_info[:300]}")
                        except Exception as e:
                            # cat-(a) EXPECTED CONTROL FLOW: debug-only solver-info dump;
                            # failure is informational only.
                            l.debug(f"OLD solver debug failed: {e}")

                        sat = self._rust_mgr.import_z3_constraint_ptrs(
                            actual_state_id, z3_ptrs)

                        try:
                            new_info = self._rust_mgr.debug_solver_info(actual_state_id)
                            l.debug(f"NEW solver ({actual_state_id}): {new_info[:300]}")
                        except Exception as e:
                            # cat-(a) EXPECTED CONTROL FLOW: debug-only solver-info dump;
                            # same as above.
                            l.debug(f"NEW solver debug failed: {e}")

                        l.debug(f"Transferred {len(z3_ptrs)} Z3 constraints from previous "
                                f"Rust manager (state {old_state_id}), sat={sat}")
                        z3_transferred = True
                except (AttributeError, Exception) as e:
                    # cat-(b) FALLBACK WITH LOSS: Z3 pointer transfer from old manager
                    # failed; falls back to lossy claripy round-trip below. Debug-logs.
                    l.debug(f"Z3 pointer transfer failed, falling back to claripy: {e}")

            if not z3_transferred:
                # Fallback: claripy AST round-trip (lossy but works without Z3)
                constraints = []
                if hasattr(angr_state, 'solver') and angr_state.solver.constraints:
                    constraints = list(angr_state.solver.constraints)
                if old_rust_mgr is not None and old_state_id is not None:
                    try:
                        exported = old_rust_mgr.export_state_constraints(old_state_id)
                        if exported:
                            l.debug(f"Transferring {len(exported)} constraints from previous "
                                    f"Rust manager (state {old_state_id})")
                            constraints = exported + constraints
                    except Exception as e:
                        # cat-(b) FALLBACK WITH LOSS: legacy export from old Rust manager
                        # failed; constraints from the old run are not transferred.
                        # Debug-logs.
                        l.debug(f"Could not export constraints from old Rust manager: {e}")
                if constraints:
                    try:
                        sat = self._rust_mgr.add_constraints_to_state(
                            actual_state_id, constraints)
                        l.debug(f"Synced {len(constraints)} initial constraints to Rust state "
                                f"{actual_state_id}, sat={sat}")
                    except Exception as e:
                        # cat-(c) WRONG-ANSWER RISK: claripy-fallback constraint sync to
                        # Rust failed; the new state has fewer constraints than the source
                        # state — eval/satisfiable on it may produce wrong values.
                        # Already warns.
                        l.warning(f"Failed to sync initial constraints: {e}")

            # Also sync any Python-side constraints (user-added post-exploration)
            if z3_transferred and hasattr(angr_state, 'solver') and angr_state.solver.constraints:
                py_constraints = list(angr_state.solver.constraints)
                if py_constraints:
                    try:
                        self._rust_mgr.add_constraints_to_state(
                            actual_state_id, py_constraints)
                        l.debug(f"Synced {len(py_constraints)} additional Python constraints")
                    except Exception as e:
                        # cat-(b) FALLBACK WITH LOSS: post-Z3-transfer sync of additional
                        # Python constraints failed; primary Z3 ptr transfer already
                        # carries the bulk. Debug-logs.
                        l.debug(f"Could not sync Python constraints: {e}")

            self._state_cache[actual_state_id] = angr_state
            # Track this as a root state for plugin restoration
            self._state_roots[actual_state_id] = actual_state_id
            # Seed Python-side options/globals from the source SimState so the
            # proxy returns the user-supplied values rather than empty stand-ins.
            # SimStateOptions is a dict-backed custom mapping that doesn't iterate
            # like a set (set(state.options) raises SimStateOptionsError on
            # numeric keys); pull names whose boolean switch is True directly
            # from the underlying _options dict.
            try:
                src_opts = getattr(angr_state, 'options', None)
                inner = getattr(src_opts, '_options', None)
                if isinstance(inner, dict):
                    self._py_state_options[actual_state_id] = {
                        name for name, value in inner.items() if value is True
                    }
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: seeding _py_state_options from source
                # state failed; child uses an empty options set on first access.
                # Debug-logs.
                l.debug("seed py_state_options(sid=%d) failed: %s: %s",
                        actual_state_id, type(e).__name__, e)
            try:
                if 'globals' in getattr(angr_state, 'plugins', {}):
                    self._py_state_globals[actual_state_id] = dict(angr_state.globals)
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: seeding _py_state_globals from source
                # state failed; child sees an empty globals dict on first access.
                # Debug-logs.
                l.debug("seed py_state_globals(sid=%d) failed: %s: %s",
                        actual_state_id, type(e).__name__, e)
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
                        import_ast = claripy.Reverse(ast) if hasattr(ast, 'length') and ast.length > 8 else ast
                        self._rust_mgr.import_symbolic_to_state(actual_state_id, addr, import_ast)
                        imported_sym += 1
                        self._register_handle(id(ast), ast, addr=addr,
                                              size=ast.length // 8 if hasattr(ast, 'length') else 1,
                                              state_id=actual_state_id)
                    except Exception as e:
                        # cat-(c) WRONG-ANSWER RISK: symbolic page import to Rust failed;
                        # Rust sees only the concrete-witness bytes for this page, losing
                        # the symbolic relationship. Debug-logs.
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
        else:
            # Fallback: cache with the Python-side state ID
            self._state_cache[rust_state.state_id] = angr_state
            # Track this as a root state for plugin restoration
            self._state_roots[rust_state.state_id] = rust_state.state_id
            symbolic_pages = self._extract_symbolic_pages(angr_state)
            if symbolic_pages:
                self._rust_mgr.set_state_symbolic_pages(rust_state.state_id, symbolic_pages)
                # Import symbolic regions to Rust's symbolic memory
                for addr, ast in symbolic_pages.items():
                    try:
                        import_ast = claripy.Reverse(ast) if hasattr(ast, 'length') and ast.length > 8 else ast
                        self._rust_mgr.import_symbolic_to_state(rust_state.state_id, addr, import_ast)
                        self._register_handle(id(ast), ast, addr=addr,
                                              size=ast.length // 8 if hasattr(ast, 'length') else 1,
                                              state_id=rust_state.state_id)
                    except (TypeError, ValueError, RuntimeError):
                        # cat-(c) WRONG-ANSWER RISK: same as 2199 but on the fallback path
                        # where actual_state_id was not determined; Python-side state ID is
                        # used. Debug-logs (with exc_info).
                        l.debug("Failed to import symbolic region at 0x%x", addr, exc_info=True)
            # Enforce state cache limit
            self._cleanup_state_cache()
            l.warning(f"Could not determine actual Rust state ID, using Python-side ID {rust_state.state_id}")


    # Field descriptor types for table-driven serialization:
    #   'val'  — copy attribute value directly (int, str)
    #   'str'  — convert attribute to string via str()
    #   'expr' — serialize as VEX expression (recursive)
    #   'exprs'— serialize list of VEX expressions
    def _serialize_irsb(self, irsb) -> str:
        """Serialize a pyvex IRSB to JSON for the Rust VEX interpreter."""
        return serialize_irsb(irsb)

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
        if reason == 'simprocedure':
            self._handle_simprocedure_callback(event)
        elif reason == 'syscall':
            self._handle_syscall_callback(event)
        elif reason == 'symbolic_branch':
            self._handle_symbolic_branch_callback(event)
        elif reason == 'find_predicate':
            self._handle_find_predicate_callback(event)
        elif reason == 'avoid_predicate':
            self._handle_avoid_predicate_callback(event)
        elif reason == 'python_vex_fallback':
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
        # Fire progress callback if due
        cb = getattr(self, '_progress_callback', None)
        if cb is not None:
            interval = getattr(self, '_progress_interval', 100)
            last = getattr(self, '_progress_last_fired', 0)
            if steps_taken - last >= interval:
                self._progress_last_fired = steps_taken
                counts = self._rust_mgr.stash_counts()
                try:
                    cb({
                        'step_count': steps_taken,
                        'active_count': counts.get('active', 0),
                        'found_count': counts.get('found', 0),
                        'deadended_count': counts.get('deadened', 0),
                        'elapsed_seconds': time.time() - start_time,
                    })
                except Exception:
                    # cat-(b) FALLBACK WITH LOSS: progress callback raised; suppress so
                    # user code can't break exploration. The callback's view skips this
                    # tick.
                    pass

        if timeout is not None and (time.time() - start_time) > timeout:
            l.warning(f"Exploration timeout reached ({timeout}s)")
            return True
        if max_steps is not None and steps_taken >= max_steps:
            l.warning(f"Max exploration steps reached ({max_steps})")
            return True
        return False

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

    def set_exploration_strategy(self, strategy: str):
        """Set exploration strategy: 'bfs' (default) or 'dfs'."""
        strategy = strategy.lower()
        if strategy == 'dfs':
            self._rust_mgr.set_state_selection_lifo()
        elif strategy == 'bfs':
            self._rust_mgr.set_state_selection_fifo()
        else:
            raise ValueError(f"Unknown exploration strategy: {strategy!r}. Use 'bfs' or 'dfs'.")

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
            timeout: Wall-clock timeout in seconds.
            max_steps: Maximum exploration steps before stopping.
            **kwargs: Additional arguments (ignored for compatibility).

        Returns:
            Self, for chaining.
        """
        # Re-entry into Rust execution invalidates the state-export cache:
        # any previously-cached Python mirrors are about to go stale.
        self._invalidate_state_export_cache()

        # Ensure predicate attributes exist (may not be set if find/avoid not provided)
        if not hasattr(self, '_find_predicate'):
            self._find_predicate = None
        if not hasattr(self, '_avoid_predicate'):
            self._avoid_predicate = None

        # Set find addresses and store predicate for callback handling
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
        self._avoid_predicate = avoid if callable(avoid) else None

        # Set num_find
        self._rust_mgr.set_num_find(num_find)

        # Re-check state options that may have been set after construction
        # (e.g., sm.one_active.options.add(LAZY_SOLVES) after simgr creation)
        try:
            from angr import sim_options as o
            for state in self._state_cache.values():
                if hasattr(state, 'options') and o.LAZY_SOLVES in state.options:
                    self._rust_mgr.set_lazy_solves(True)
                    l.debug("Enabled lazy_solves from cached state options at explore() time")
                    break
        except (ImportError, Exception):
            # cat-(a) EXPECTED CONTROL FLOW: optional sim_options import; if
            # absent, lazy_solves stays at the value set during construction.
            pass

        # Reset solver profiling stats for this exploration run
        try:
            from angr.rustylib.vex_engine import RustExplorationManager as _REM
            _REM.reset_solver_stats()
        except (ImportError, RuntimeError, AttributeError):
            # cat-(b) FALLBACK WITH LOSS: solver-stats reset failed (e.g., the
            # Rust extension was built without Z3); the run accumulates over
            # whatever counters survived the previous explore().
            pass

        # Route to appropriate exploration strategy
        has_predicates = self._find_predicate is not None or self._avoid_predicate is not None
        if has_predicates or bool(self._active_techniques):
            return self._explore_with_predicates(num_find, until, timeout, max_steps)
        return self._explore_with_addresses(num_find, until, timeout, max_steps)

    def _explore_with_predicates(self, num_find, until, timeout, max_steps):
        """Exploration loop for callable predicates or active techniques.

        Runs in batches of 50 steps, evaluating predicates between batches.
        Callbacks are handled immediately when Rust returns need_callback events.
        """
        self._rust_mgr.set_find_needs_python(self._find_predicate is not None)
        self._rust_mgr.set_avoid_needs_python(False)
        # Keep terminal states alive so predicates can check them
        self._rust_mgr.set_drop_terminal_states(False)

        batch_size = 50
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

            steps_taken, _time_in_rust_run = self._run_predicate_batch(
                steps_taken, batch_limit, _time_in_rust_run
            )

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
                except Exception:
                    # cat-(b) FALLBACK WITH LOSS: until predicate raised; we keep
                    # exploring and let the user fix their predicate. Debug-logs with
                    # exc_info.
                    l.debug("until predicate raised exception", exc_info=True)

        # Final predicate check on deadended/remaining states
        self._evaluate_predicates_on_active()
        self._rust_mgr.set_drop_terminal_states(True)

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
            event = self._rust_mgr.run(remaining)
            _time_in_rust_run += time.perf_counter_ns() - _t1
            self._rust_mgr.sync_state_index()

            if event.event_type == 'need_callback':
                if self._dispatch_callback(event):
                    batch_done = True
                steps_taken += 1
            elif event.event_type == 'active_empty':
                batch_done = True
            elif event.event_type in ('step_complete', 'found'):
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

        while True:
            if self._check_limits(start_time, steps_taken, timeout, max_steps):
                break

            self._sync_hooks_before_step()
            self._stats_ffi_crossings += 1

            if need_per_step:
                batch_size = 50
                if max_steps is not None:
                    batch_size = min(batch_size, max_steps - steps_taken)
                event = self._rust_mgr.run(batch_size)
            else:
                event = self._rust_mgr.run()
            self._rust_mgr.sync_state_index()

            # Count steps: callbacks = 1, batched runs = event total
            if event.event_type == 'need_callback':
                steps_taken += 1
            else:
                steps_taken += max(1, event.steps_taken)

            # Drop unconstrained states if save_unconstrained=False
            if not self._save_unconstrained:
                try:
                    self._rust_mgr.clear_stash('unconstrained')
                except (RuntimeError, KeyError):
                    # cat-(b) FALLBACK WITH LOSS: clear_stash on the unconstrained
                    # stash failed (no such stash, race with Rust); states may persist
                    # in 'unconstrained' even though save_unconstrained=False.
                    pass

            # Periodically clean Python state cache to prevent memory leaks
            if steps_taken % 100 == 0:
                self._cleanup_state_cache()

            # Dispatch event
            if event.event_type == 'found' and event.found_count >= num_find:
                break
            elif event.event_type == 'active_empty':
                if self._active_techniques:
                    self._apply_technique_filters()
                    if self._check_technique_complete():
                        break
                    if self._rust_mgr.get_state_ids('active'):
                        continue
                break
            elif event.event_type == 'need_callback':
                if self._dispatch_callback(event):
                    break
            elif event.event_type == 'errored':
                if _DBG:
                    l.debug(f"Exploration error (state deadended): {event.callback_reason}")
            elif event.event_type == 'step_complete':
                self._cleanup_symbolic_pages_cache()

            # Apply technique filters after step events
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
                    # cat-(c) WRONG-ANSWER RISK: until predicate raised; exploration
                    # continues past the user's intended stop. Already warns.
                    l.warning(f"until predicate error: {e}")

        return self

    def get_state_options_py(self, state_id: int) -> set:
        """Return the Python-side options set for a Rust state.

        The Rust engine doesn't honor SimOptions (only LAZY_SOLVES /
        STRICT_PAGE_ACCESS are mirrored onto the Rust state separately), so
        this set lives Python-side. On first access for a forked state, copy
        from the root state's options so children inherit a snapshot.

        Returns a live set — mutations propagate to subsequent accesses.
        """
        opts = self._py_state_options.get(state_id)
        if opts is not None:
            return opts
        try:
            root_id = self._rust_mgr.get_state_root(state_id)
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: get_state_root lookup failed;
            # treat as no parent and start from a fresh empty options set.
            root_id = None
        if root_id is not None and root_id != state_id:
            parent_opts = self._py_state_options.get(root_id)
            if parent_opts is not None:
                opts = set(parent_opts)
                self._py_state_options[state_id] = opts
                return opts
        opts = set()
        self._py_state_options[state_id] = opts
        return opts

    def get_state_globals_py(self, state_id: int) -> dict:
        """Return the Python-side globals dict for a Rust state.

        Children inherit a shallow copy of the root state's globals on first
        access. Returns a live dict — mutations propagate.
        """
        glb = self._py_state_globals.get(state_id)
        if glb is not None:
            return glb
        try:
            root_id = self._rust_mgr.get_state_root(state_id)
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: get_state_root lookup failed;
            # treat as no parent and start from a fresh empty globals dict.
            root_id = None
        if root_id is not None and root_id != state_id:
            parent_glb = self._py_state_globals.get(root_id)
            if parent_glb is not None:
                glb = dict(parent_glb)
                self._py_state_globals[state_id] = glb
                return glb
        glb = {}
        self._py_state_globals[state_id] = glb
        return glb

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
        """
        try:
            active_set = set(self._rust_mgr.get_state_ids('active'))
            found_set = set(self._rust_mgr.get_state_ids('found'))
        except (RuntimeError, KeyError):
            # cat-(b) FALLBACK WITH LOSS: cannot read live state IDs; skip this
            # cleanup tick. Cache may temporarily exceed cap until next call.
            return

        # Broader "known-to-Rust" set for shadow-structure pruning. State IDs
        # outside this set are unreachable (no stash holds them) and Rust will
        # never resurrect them, so it's safe to drop any Python-side mapping
        # keyed on them. Failure here is non-fatal — fall back to the narrower
        # active/found set so we never over-prune.
        try:
            avoid_set = set(self._rust_mgr.get_state_ids('avoid'))
            deadended_set = set(self._rust_mgr.get_state_ids('deadended'))
            any_stash = active_set | found_set | avoid_set | deadended_set
        except (RuntimeError, KeyError):
            # cat-(b) FALLBACK WITH LOSS: missing avoid/deadended view means
            # the shadow-prune below sees a smaller live set and is more
            # aggressive than ideal. Mappings for states currently sitting in
            # those stashes will be dropped this tick (harmless: the state
            # itself is no longer being stepped, and a new explore() call
            # rebuilds mappings as states get re-registered).
            any_stash = active_set | found_set

        live = active_set | found_set
        live.update(self._state_roots.get(sid, sid) for sid in live)
        # Always keep root state cache entries: a forked state can fire its
        # first Python callback without itself or its (Rust-only) ancestors
        # being in cache; the only way to avoid the blank-state fallback is
        # to fall back to the root, so the root must survive eviction even
        # when it is no longer in active/found.
        live |= set(self._state_roots.values())
        for sid in list(self._state_cache.keys()):
            if sid not in live:
                del self._state_cache[sid]
        # Drop options/globals for state IDs that no longer exist in any stash.
        live_with_roots = live
        for sid in list(self._py_state_options.keys()):
            if sid not in live_with_roots:
                del self._py_state_options[sid]
        for sid in list(self._py_state_globals.keys()):
            if sid not in live_with_roots:
                del self._py_state_globals[sid]

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
        matched = getattr(self, '_predicate_matched_ids', None)
        if matched is not None:
            matched.intersection_update(any_stash)

        # Drop _LazySimStateRef wrappers for state ids that no longer exist
        # in any Rust stash. The wrappers are cheap (two slots) but pruning
        # them keeps the dict bounded by live stash size across long runs.
        for sid in list(self._lazy_state_refs.keys()):
            if sid not in any_stash:
                del self._lazy_state_refs[sid]

        pinned = set(self._state_roots.values())
        if self._current_callback_state_id is not None:
            pinned.add(self._current_callback_state_id)
            try:
                eff_id = self._get_effective_state_id(self._current_callback_state_id)
                if eff_id is not None:
                    pinned.add(eff_id)
            except Exception:
                # cat-(b) FALLBACK WITH LOSS: effective-state lookup failed; only
                # the direct callback id is pinned, so an in-flight forked state
                # could be evicted. Tolerated — eviction is recoverable via re-
                # fetch on next access.
                pass
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

    def step(self, n: int = 1, **kwargs) -> "RustExplorationManager":
        """Step the exploration n times.

        Args:
            n: Number of steps to take.
            **kwargs: Additional arguments (ignored).

        Returns:
            Self, for chaining.
        """
        # Re-entry into Rust execution invalidates the state-export cache:
        # any previously-cached Python mirrors are about to go stale.
        self._invalidate_state_export_cache()

        steps_taken = 0
        while steps_taken < n:
            self._sync_hooks_before_step()
            self._stats_ffi_crossings += 1
            event = self._rust_mgr.run(1)
            self._rust_mgr.sync_state_index()

            if event.event_type == 'need_callback':
                if self._dispatch_callback(event):
                    break
                steps_taken += 1
            elif event.event_type == 'active_empty':
                break
            elif event.event_type == 'errored':
                l.warning(f"Step error: {event.callback_reason}")
                break
            elif event.event_type in ('step_complete', 'found'):
                steps_taken += 1
                if self._active_techniques:
                    self._apply_technique_filters()
            else:
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
        """Get states in the errored stash as RustErrorRecord objects.

        Each RustErrorRecord has .state, .error, and .addr attributes,
        matching the interface of angr's ErrorRecord class.
        """
        states = self._get_stash_states('errored')
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
        state_ids = self._rust_mgr.get_state_ids('errored')
        records = []
        for i, state in enumerate(states):
            state_id = state_ids[i] if i < len(state_ids) else None
            addr, message = error_lookup.get(state_id, (0, "unknown error"))
            records.append(RustErrorRecord(state, message, addr))
        return records

    @property
    def unconstrained(self) -> list:
        """Get states in the unconstrained stash as angr SimStates.

        These are states where a symbolic jump target (e.g., ret instruction
        with symbolic return address) had too many possible concrete values
        to enumerate and fork.
        """
        return self._get_stash_states('unconstrained')

    @property
    def pruned(self) -> list:
        """Get states in the pruned stash as angr SimStates.

        These are states that were determined to be unsatisfiable during
        exploration (e.g., both branches of a conditional were infeasible
        given the current constraints).
        """
        return self._get_stash_states('pruned')

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
        from angr.exploration.rust_state_proxy import RustStateProxy
        stdin_vars = getattr(self, '_stdin_vars', None)
        stdout_tracker = getattr(self, '_stdout_tracker', {}) or {}
        state_ids = self._rust_mgr.get_state_ids(stash_name)
        return [
            RustStateProxy(
                self._rust_mgr,
                sid,
                project=self._project,
                stdin_vars=stdin_vars,
                stdout_data=stdout_tracker.get(sid, b""),
                python_mgr=self,
            )
            for sid in state_ids
        ]

    def found_proxies(self) -> list:
        """Return the ``found`` stash as ``list[RustStateProxy]``.

        Counterpart to :attr:`found`, but returns lightweight proxies that
        delegate reads to Rust via PyO3 instead of materializing full angr
        ``SimState`` objects. Use this when querying many states cheaply
        (e.g. counting states matching ``proxy.addr == X``); use
        :attr:`found` when you need full SimState plugins (``posix.dumps``,
        ``solver.eval`` of complex claripy ASTs, etc.).
        """
        return self._stash_proxies('found')

    def active_proxies(self) -> list:
        """Return the ``active`` stash as ``list[RustStateProxy]``. See
        :meth:`found_proxies` for when to prefer proxies over full states."""
        return self._stash_proxies('active')

    def avoid_proxies(self) -> list:
        """Return the ``avoid`` stash as ``list[RustStateProxy]``. See
        :meth:`found_proxies` for when to prefer proxies over full states."""
        return self._stash_proxies('avoid')

    def deadended_proxies(self) -> list:
        """Return the ``deadended`` stash as ``list[RustStateProxy]``. See
        :meth:`found_proxies` for when to prefer proxies over full states."""
        return self._stash_proxies('deadended')

    def unconstrained_proxies(self) -> list:
        """Return the ``unconstrained`` stash as ``list[RustStateProxy]``.
        See :meth:`found_proxies` for when to prefer proxies over full
        states."""
        return self._stash_proxies('unconstrained')

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
        result['z3_ptr_cache_hits'] = self._z3_ptr_cache_hits
        result['z3_ptr_cache_misses'] = self._z3_ptr_cache_misses
        # angr-xtse.1: surface PerformanceTracker callback counts/times so
        # run_single.py --counters-json picks them up alongside the Rust-side
        # counters. Keys are kept verbatim ("callback_<kind>_count",
        # "callback_<kind>_total_ns") so downstream analysis can extract the
        # bucket by name without translation.
        for key, val in self._perf_stats.as_dict().items():
            if key.startswith("callback_"):
                result[key] = val
        # angr-h0dv: defensive counter for Path A (rust_solver_ctx attach)
        # regressions. The other legacy constraint-sync counters were retired
        # after a 20-bench soak proved Path B was dead code.
        result['rust_ctx_missing'] = self._stats_rust_ctx_missing
        # angr-ymoe: orphan-BVS fallback counters
        result['orphan_bvs_mem_thunk'] = self._stats_orphan_bvs_mem_thunk
        result['orphan_bvs_sym_load_full_fail'] = self._stats_orphan_bvs_sym_load_full_fail
        # angr-4o7d: snapshot-restore orphan-BVS counter
        result['orphan_bvs_snapshot_restore'] = self._stats_orphan_bvs_snapshot_restore
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
                    # Try lightweight proxy first (avoids expensive full state
                    # export). Falls back to full export if the filter accesses
                    # something the proxy doesn't support. Mirrors filter().
                    from angr.exploration.rust_state_proxy import RustStateProxy
                    proxy = RustStateProxy(self._rust_mgr, state_id, self._project,
                                           python_mgr=self)
                    try:
                        if filter_func(proxy):
                            move_ids.append(state_id)
                        else:
                            keep_ids.append(state_id)
                        continue  # Proxy worked, skip full export
                    except (AttributeError, TypeError, NotImplementedError):
                        # cat-(a) EXPECTED CONTROL FLOW: proxy didn't support an
                        # attribute the predicate accessed; fall back to full export.
                        pass

                    snapshot = self._rust_mgr.export_state(state_id)
                    py_state = self._snapshot_to_angr(snapshot)

                    if filter_func(py_state):
                        move_ids.append(state_id)
                    else:
                        keep_ids.append(state_id)
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: move filter raised on a state; keep
                    # the state in the source stash rather than dropping it. Debug-logs.
                    if _DBG:
                        l.debug(f"move filter error for state {state_id}: {e}")
                    keep_ids.append(state_id)  # Keep on error

            # Use Rust to move matching states
            for state_id in move_ids:
                try:
                    self._rust_mgr.move_state(state_id, from_stash, to_stash)
                except (RuntimeError, KeyError):
                    # cat-(a) EXPECTED CONTROL FLOW: state may already have moved
                    # (e.g., another technique deadended it). Suppress.
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

        States not matching the predicate are removed (moved to 'pruned').

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
                proxy = RustStateProxy(self._rust_mgr, state_id, self._project,
                                        python_mgr=self)
                try:
                    if filter_func(proxy):
                        keep_ids.append(state_id)
                    else:
                        prune_ids.append(state_id)
                    continue  # Proxy worked, skip full export
                except (AttributeError, TypeError, NotImplementedError):
                    # cat-(a) EXPECTED CONTROL FLOW: proxy didn't support an attribute
                    # the predicate accessed; fall back to full export below.
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
                # cat-(b) FALLBACK WITH LOSS: filter raised; keep the state in the
                # source stash. Debug-logs.
                if _DBG:
                    l.debug(f"filter error for state {state_id}: {e}")
                keep_ids.append(state_id)  # Keep on error

        # Move non-matching states to pruned stash
        for state_id in prune_ids:
            try:
                self._rust_mgr.move_state(state_id, stash, 'pruned')
            except (RuntimeError, KeyError):
                # cat-(a) EXPECTED CONTROL FLOW: state may already have moved.
                pass

        return self

    def prune(self, stash: str = 'active', filter_func=None) -> "RustExplorationManager":
        """Remove states from a stash based on predicate.

        Default behavior prunes unsatisfiable states.

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
                except (RuntimeError, KeyError):
                    # cat-(b) FALLBACK WITH LOSS: state_satisfiable() raised; keep the
                    # state — if it's actually unsat, downstream solver use will catch.
                    pass  # Keep on error
            for state_id in prune_ids:
                try:
                    self._rust_mgr.move_state(state_id, stash, 'pruned')
                except (RuntimeError, KeyError):
                    # cat-(a) EXPECTED CONTROL FLOW: state may already have moved.
                    pass
            return self

        return self.filter(stash=stash, filter_func=filter_func)

    def drop(self, stash: str = 'active', filter_func=None) -> "RustExplorationManager":
        """Drop states from a stash.

        States matching the predicate (or all if no predicate) are removed.

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
                # cat-(a) EXPECTED CONTROL FLOW: probing for the optional
                # clear_stash API on older Rust builds; fall back to move_states.
                # Fallback: move all to deadended
                self._rust_mgr.move_states(stash, 'deadended', None)
        else:
            # Drop states matching predicate
            state_ids = list(self._rust_mgr.get_state_ids(stash))

            for state_id in state_ids:
                try:
                    # Try lightweight proxy first (avoids expensive full state
                    # export). Falls back to full export if the filter accesses
                    # something the proxy doesn't support. Mirrors filter().
                    from angr.exploration.rust_state_proxy import RustStateProxy
                    proxy = RustStateProxy(self._rust_mgr, state_id, self._project,
                                           python_mgr=self)
                    matched = None
                    try:
                        matched = bool(filter_func(proxy))
                    except (AttributeError, TypeError, NotImplementedError):
                        # cat-(a) EXPECTED CONTROL FLOW: proxy didn't support an
                        # attribute the predicate accessed; fall back to full export.
                        pass

                    if matched is None:
                        snapshot = self._rust_mgr.export_state(state_id)
                        py_state = self._snapshot_to_angr(snapshot)
                        matched = filter_func(py_state)

                    if matched:
                        try:
                            self._rust_mgr.move_state(state_id, stash, 'deadended')
                        except (RuntimeError, KeyError):
                            # cat-(a) EXPECTED CONTROL FLOW: state may already have moved.
                            pass
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: drop filter raised; that state is
                    # left in the source stash. Debug-logs.
                    if _DBG:
                        l.debug(f"drop filter error for state {state_id}: {e}")

        return self

    def split(self, stash_from: str = 'active', stash_to: str = 'stashed',
              limit: int = 8, filter_func=None) -> "RustExplorationManager":
        """Split states between stashes.

        Moves excess states to another stash to limit exploration width.

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
            except (RuntimeError, KeyError):
                # cat-(a) EXPECTED CONTROL FLOW: state may already have moved.
                pass

        return self

    @property
    def stashes(self) -> dict:
        """Get all stashes as a dictionary for SimulationManager compatibility.

        Returns state IDs per stash.
        """
        result = {}
        for stash_name in ['active', 'found', 'avoid', 'deadended', 'errored',
                          'unconstrained', 'pruned', 'stashed']:
            try:
                state_ids = list(self._rust_mgr.get_state_ids(stash_name))
                result[stash_name] = state_ids
            except (RuntimeError, KeyError):
                # cat-(a) EXPECTED CONTROL FLOW: stash name unknown to Rust;
                # return an empty list for that stash key.
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

    def merge(self, stash: str = 'active', merge_func=None, merge_key=None,
              prune=True, **kwargs) -> "RustExplorationManager":
        """Merge states in a stash.

        Exports states to Python, performs merge via claripy, then replaces
        the stash with merged states. Falls back to keeping all states
        unmerged if merge fails.
        """
        state_ids = list(self._rust_mgr.get_state_ids(stash))
        if len(state_ids) <= 1:
            return self

        # Export all states to Python SimStates for merging
        try:
            py_states = []
            for sid in state_ids:
                py_state = self.get_state_by_id(sid)
                if py_state is not None:
                    py_states.append(py_state)

            if len(py_states) <= 1:
                return self

            # Group by merge key (default: PC)
            if merge_key is None:
                merge_key = lambda s: s.addr

            groups = {}
            for s in py_states:
                key = merge_key(s)
                groups.setdefault(key, []).append(s)

            merged = []
            for key, group in groups.items():
                if len(group) <= 1:
                    merged.extend(group)
                elif merge_func is not None:
                    try:
                        merged.append(merge_func(*group))
                    except (TypeError, ValueError, RuntimeError):
                        # cat-(b) FALLBACK WITH LOSS: user merge_func failed for this
                        # group; keep the group's states unmerged. Already warns.
                        l.warning("merge_func failed for group at %s, keeping unmerged", key)
                        merged.extend(group)
                else:
                    try:
                        base = group[0]
                        others = group[1:]
                        m, _, _ = base.merge(*others)
                        merged.append(m)
                    except (AttributeError, TypeError, ValueError):
                        # cat-(b) FALLBACK WITH LOSS: built-in state.merge() failed;
                        # keep the group unmerged. Already warns.
                        l.warning("State merge failed for group at %s, keeping unmerged", key)
                        merged.extend(group)

            # Clear the Rust stash and re-add merged states
            for sid in state_ids:
                try:
                    self._rust_mgr.move_state(sid, stash, '_merge_drop')
                except (RuntimeError, KeyError):
                    # cat-(a) EXPECTED CONTROL FLOW: source stash entry already moved.
                    pass
            try:
                self._rust_mgr.clear_stash('_merge_drop')
            except (RuntimeError, KeyError):
                # cat-(a) EXPECTED CONTROL FLOW: clear of intermediate _merge_drop
                # stash failed (already empty / never created).
                pass

            # Re-add merged states
            for ms in merged:
                try:
                    self._add_rust_state(stash, ms)
                    l.debug("Added merged state to %s at 0x%x", stash, ms.addr)
                except Exception as e:
                    # cat-(c) WRONG-ANSWER RISK: failed to re-add merged state; the
                    # merge result is lost — caller sees fewer states than expected.
                    # Already warns.
                    l.warning("Failed to re-add merged state: %s", e)

        except Exception:
            # cat-(c) WRONG-ANSWER RISK: outer merge raised; states are left
            # unmerged (some already moved to _merge_drop and cleared above).
            # Already warns with exc_info.
            l.warning("State merge failed entirely, keeping states unmerged", exc_info=True)

        return self

    def cleanup(self) -> None:
        """Release this manager's hold on per-process AST caches.

        Flushes the Rust-side thread-local claripy AST translation caches
        (``AST_CACHE`` / ``CLARIPY_AST_CACHE`` / ``EXPRESSION_CACHE`` /
        ``EXPRESSION_BY_OPERANDS_PTR`` in ``claripy_bridge``). Those caches
        outlive a single manager because they are thread-local, so without
        a flush they accumulate O(n) across managers in Callable-heavy
        workloads — see ``docs/advanced-topics/rust_engine.rst`` for the
        mma_howtouse case study.

        Safe to call multiple times. Does NOT touch the global symbolic
        identity registry (shared across managers, so a clear from one
        manager would invalidate symbol IDs held live by another).
        """
        try:
            from angr.rustylib.vex_engine import clear_ast_cache
            clear_ast_cache()
        except Exception as e:
            # cat-(a) EXPECTED CONTROL FLOW: vex_engine module may be
            # absent in degraded builds; cleanup is best-effort.
            l.debug("clear_ast_cache unavailable, skipping: %s", e)

    def __del__(self):
        # Best-effort cleanup. Skip silently when the flag was never
        # opted into (single-long-exploration users see no behavior
        # change). Wrap everything because interpreter shutdown can
        # already have torn down sys.modules by the time __del__ runs.
        try:
            if getattr(self, '_clear_caches_on_cleanup', False):
                self.cleanup()
        except Exception:
            pass

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
        except (RuntimeError, KeyError):
            # cat-(a) EXPECTED CONTROL FLOW: stash name unknown to Rust;
            # raise AttributeError so getattr-style probes return the default.
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
