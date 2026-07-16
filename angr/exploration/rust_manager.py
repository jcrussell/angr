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
RustStateExportMixin, RustStateSyncMixin, RustCallbackDispatchMixin,
RustDiskCacheManager) plus helpers in this module. The invariants below cut
across those boundaries —
breaking any one of them tends to produce silent correctness bugs (cache
poisoning, lost state, infinite loops) rather than loud failures, so they
are documented here and pinned by the named regression tests.

I1. Disk-cache key axes
    `_disk_cache_key` (now in the RustDiskCacheManager mixin,
    `rust_disk_cache.py`, alongside `_save_init_to_disk_cache` and the
    `_extract_*` snapshot helpers) mixes (binary path, `_RUST_CACHE_VERSION`,
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
    (now in the RustDiskCacheManager mixin, `rust_disk_cache.py`, alongside
    the save path) is split so each phase owns a single concern:
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
    `TestEdgeCases` (tests/engines/rust/test_annotations_edge.py) which
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
    `_cleanup_state_cache` runs three steps: (1) drop entries whose state
    is no longer in active/found, (2) pin every root in `_state_roots`,
    plus `_current_callback_state_id` (and its effective id via
    `_get_effective_state_id`) and `_current_stepping_state_id`, (3)
    LRU-evict non-pinned entries past `_max_state_cache_size`. Without
    those pins, a freshly-mutated state can race-evict between callbacks
    on the same state. Regression tests:
    `TestStateCacheSizeBound.test_cleanup_state_cache_evicts_oldest_first`,
    `..._drops_dead_states`, `..._skips_pinned` (lines 2017, 2064, 2096).
    Metadata-clear contract: cache eviction does NOT call
    `clear_state_metadata`. A state can be in `active`/`found` on Rust and
    still LRU-evicted from the Python mirror; in that case the Rust-side
    metadata must remain. Per-state Rust metadata is freed when the
    underlying `RustSimState` drops naturally (i.e. once the state leaves
    every Rust stash).

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

import logging
import os
import pickle
import struct
import sys
import time
import warnings
import weakref
from collections.abc import Callable
from itertools import chain
from typing import TYPE_CHECKING

import claripy
from claripy.errors import ClaripyError
from pyvex.errors import PyVEXError

from angr.errors import SimEngineError, SimError, SimSolverError
from angr.exploration.rust_irsb_serializer import serialize_irsb
from angr.exploration.rust_perf_tracker import PerformanceTracker

# Envelope head written by RustExplorationManager.dump_snapshot. A file without
# it is a pre-bucket-D snapshot (bare Rust bytes) and still loads.
_SNAPSHOT_MAGIC = b"ANGRSNAP\x01"

# Reserved key in the snapshot envelope's pickled overlay dict, whose real keys
# are Rust state ids (u64, so never negative). Holds the seed state's
# posix.stdin.content ASTs — see `_capture_seed_stdin_content`.
_SEED_STDIN_KEY = -1

if TYPE_CHECKING:
    import angr

l = logging.getLogger(name=__name__)
_DBG = l.isEnabledFor(logging.DEBUG)  # Module-level guard for hot-path debug calls

# angr-kzjv6: symex-relevant SimOptions mirrored onto the Rust state so native
# SimProcedures can branch on them via ``RustSimState.has_option``. Kept as the
# raw option strings (the ``angr.sim_options`` constants are plain ``str``) to
# avoid an import-order dependency. Add an entry here only when a native proc
# actually consults the option; the full per-state option set otherwise stays
# Python-side (``rust_state_proxy.options``).
_NATIVE_SIMOPTIONS = frozenset({"SHORT_READS"})


def _is_rust_memory_proxy(plugin) -> bool:
    """True when ``plugin`` is a ``RustMemoryProxy`` (callback-memory-proxy gate).

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
# choose. This set covers the options a user must opt into; the two
# default-bundle ones are resolved at their *consumption* site instead
# (angr-op0dn.14.7): TRACK_CONSTRAINT_ACTIONS' only observable effect is the
# SimActionConstraint stream, and reading `state.history.actions` warns once via
# `_RustOwnedSimStateHistory` (rust_state_export.py); TRACK_MEMORY_MAPPING is
# vestigial — no code in angr or its dependencies reads it, only
# analyses/identifier/runner.py adds it — so ignoring it cannot diverge.
_REJECTED_OPTION_NAMES = frozenset(
    {
        # Conservative read strategy: refuses to concretize on range-check
        # failure. (The write strategy variant raises, see _RAISE_OPTION_NAMES.)
        "CONSERVATIVE_READ_STRATEGY",
        # SimMemory error-handling tweaks.
        "UNINITIALIZED_ACCESS_AWARENESS",
        "BEST_EFFORT_MEMORY_STORING",
        # Ret-emulation guard sibling. The DO_RET_EMULATION half raises (see
        # _RAISE_OPTION_NAMES); the guard alone is harmless without it.
        "TRUE_RET_EMULATION_GUARD",
        # Alternate Python engines / memory plugins.
        "SUPER_FASTPATH",
        "FAST_MEMORY",
        "FAST_REGISTERS",
        "UNDER_CONSTRAINED_SYMEXEC",
        # BYPASS_VERITESTING_EXCEPTIONS is consulted only from
        # angr/analyses/veritesting.py (resilience= kwarg passed to nested
        # SimulationManager.run). Veritesting under Rust already raises via
        # EFFICIENT_STATE_MERGING (Veritesting auto-adds that option), so a
        # user driving Veritesting hits the raise on EFFICIENT_STATE_MERGING
        # first. Outside Veritesting, BYPASS_VERITESTING_EXCEPTIONS is a
        # no-op — `resilience` bundle users carry it implicitly. Reject with
        # a warn-once rather than raise so adding `angr.options.resilience`
        # to a non-Veritesting state does not crash. (angr-6rz8 2026-06-03)
        "BYPASS_VERITESTING_EXCEPTIONS",
        # USE_SYSTEM_TIMES tells Python's posix sim_time procedures
        # (procedures/posix/sim_time.py) to return the host's real
        # `int(time.time())` instead of a fresh symbolic timeval/timespec.
        # The native handlers (native/angr/src/syscalls/sim_time.rs:
        # gettimeofday/time/clock_gettime) always write a fresh symbolic
        # value and never consult the option — wiring host-time into the
        # native path would be a behavior change with no benchmark demand.
        # Warn-once so a user who opted into concrete host times learns the
        # native syscalls are ignoring it rather than silently exploring a
        # symbolic-time path. Not in any default mode bundle, so this only
        # fires on explicit opt-in. (angr-0y0v 2026-06-15)
        "USE_SYSTEM_TIMES",
    }
)


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
# address concretization on range-check failure (Python: storage/
# memory_mixins/address_concretization_mixin.py concretize_write_addr). Rust's
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
# set_strongref_state). It was raise-listed (angr-n129, 2026-05-16)
# while the Rust engine merged only through the Python export path,
# which needs that common-ancestor walk. DEMOTED (angr-op0dn.11.6):
# merge is now native. RustExplorationManager.merge() runs the M3-4
# fast path (_merge_native -> _rust_mgr.merge_states, no export, no
# SimStateHistory walk), and the M3-5 native MergePoint technique
# (ManualMergepoint -> register_merge_point) forks-and-merges entirely
# in Rust. Neither path consults SimStateHistory's strongref, so the
# option's Python rationale is moot: it is honored by NOT raising, which
# lets a state carrying it (including Veritesting's auto-added copy)
# explore under the native merge machinery instead of hard-failing at
# the manager boundary. The paired SIMPLIFY_MERGED_CONSTRAINTS was never
# promoted because it ships in the default `symbolic` mode bundle
# (simplification set); it is honored implicitly through the Python
# state.merge() fallback that still backs custom merge_func/merge_key.
# NOTE for M6 coordination: this option is M3-owned — the M6 dispatcher
# routes it to Python meanwhile and must not double-count it.
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
# load_concrete_lazy (native/angr/src/memory/load.rs fn load_concrete_lazy) falls back
# to a fresh `unc_mem_*` symbolic BVS when zero_fill_unconstrained is
# unset — i.e., symbolic-fill is already Rust's default for memory.
#
# BYPASS_ERRORED_IROP / BYPASS_ERRORED_IRCCALL / BYPASS_ERRORED_IRSTMT
# tell Python's HeavyResilienceMixin (engines/vex/heavy/resilience.py) to
# catch SimError / SimOperationError raised during op / ccall / stmt
# evaluation and substitute a default symbol or zero. Rust's interpreter
# routes its own errors via FallbackStrategy (interpreter/mod.rs:268-278):
# UnsupportedFeature falls back to Python so the BYPASS_UNSUPPORTED_*
# bypass fires, but Op / TypeMismatch / InvalidIR are Panic-strategy and
# move the state to the errored stash without ever giving Python a chance
# to substitute. The "ERRORED" bypasses therefore silently do nothing
# under Rust when Rust hits a Panic-strategy variant — a divergence the
# user paid into resilience for. Promoted to raise (angr-6rz8,
# 2026-06-03). The "UNSUPPORTED" siblings (BYPASS_UNSUPPORTED_IROP /
# IRDIRTY / IRCCALL / SYSCALL) are honored transparently via the
# UnsupportedFeature -> Python fallback path and therefore stay out of
# both _RAISE and _REJECTED. BYPASS_UNSUPPORTED_IREXPR and
# BYPASS_UNSUPPORTED_IRSTMT are vestigial — defined but not consulted
# anywhere in angr today; also honored vacuously. The modifier options
# UNSUPPORTED_BYPASS_ZERO_DEFAULT and UNSUPPORTED_FORCE_CONCRETIZE only
# affect what value Python substitutes when its bypass fires, so they
# are also honored transparently through the same path.
# CONSTRAINT_TRACKING_IN_SOLVER tells SimSolver to build a tracking
# claripy Solver (state_plugins/solver.py _init_add_constraints) so that
# unsat_core() can name the constraints that made a state infeasible;
# without it, unsat_core() refuses to run at all (raises
# SimSolverOptionError). It was raise-listed (angr-op0dn.14.8) while the
# Rust engine had no core to give: constraints go onto the shared Z3
# solver with no assumption literals, so an opting-in user got a
# *silently empty* core — the one failure mode the option exists to
# prevent. DEMOTED (angr-op0dn.14.2): RustSolverProxyPlugin.unsat_core now
# computes the core on demand (SymContext::unsat_core_assumed) by rebuilding
# a throwaway assumption-guarded solver from the assumed-constraint IR, so
# the core is complete — it names the engine's own fork guards too, which a
# core read off the live solver could not — and nothing is paid on the hot
# add path. The proxy still honors the option: without it on the bound
# state, unsat_core() raises SimSolverOptionError like Python's.
#
# CONCRETIZE_SYMBOLIC_WRITE_SIZES is NOT promoted. Its only in-tree
# consumer is SimFileBase._prep_generic (storage/file.py), and every
# native write path (syscalls/write.rs, the CGC `transmit` in
# syscalls/cgc.rs) falls back to Python on a symbolic count — so the
# option is honored transparently wherever it can fire. (The memory-side
# knob of the same name is a SimMemory *constructor kwarg*, not this
# option.)
#
# CGC_NON_BLOCKING_FDS is NOT promoted either: the native `fdwait`
# (syscalls/cgc.rs NativeFdwaitSyscall) implements exactly the
# option-is-set behavior, and since angr-op0dn.14.8 it falls back to
# Python when the option is unset — where the Python proc's
# unconstrained ready bits are produced. Honored in both directions.
#
# TRACK_ACTION_HISTORY is NOT in _RAISE_OPTION_NAMES (demoted angr-fkvt,
# 2026-06-06). Unlike its TRACK_*_ACTIONS siblings, it does not gate
# action recording — angr's heavy/actions.py only consults
# TRACK_REGISTER_ACTIONS / TRACK_MEMORY_ACTIONS to populate
# state.history.recent_events. TRACK_ACTION_HISTORY's only consumer in
# current angr is preconstrainer.py (state_plugins/preconstrainer.py:98)
# which uses it as a metadata flag — temporarily clears it during
# preconstraint to suppress action recording, then restores. Under Rust
# that clear/restore is a vacuous no-op (Rust never records actions
# regardless) so honoring the option silently is safe. Unblocks AEG
# workloads (insomnihack_aeg, angr-86c4) which set the option but never
# inspect state.history.actions directly. The TRACK_*_ACTIONS family
# remains raise-listed because those DO gate action recording.
_RAISE_OPTION_NAMES = frozenset(
    {
        "TRACK_MEMORY_ACTIONS",
        "TRACK_REGISTER_ACTIONS",
        "TRACK_TMP_ACTIONS",
        "TRACK_JMP_ACTIONS",
        "TRACK_OP_ACTIONS",
        "CONCRETIZE",
        "CONSERVATIVE_WRITE_STRATEGY",
        "DO_RET_EMULATION",
        "CALLLESS",
        "SYMBOL_FILL_UNCONSTRAINED_REGISTERS",
        "BYPASS_ERRORED_IROP",
        "BYPASS_ERRORED_IRCCALL",
        "BYPASS_ERRORED_IRSTMT",
        # PRODUCE_ZERODIV_SUCCESSORS (promoted angr-op0dn.14.9). Python's
        # zero-division path is a *lifting* one: irop.py raises
        # SimZeroDivisionException on a concrete zero divisor, the resilience
        # mixin turns that into an `Ijk_SigFPE_IntDiv` exit, and
        # engines/successors.py::add_successor keeps that successor only when
        # this option is set (it drops it otherwise). The Rust interpreter has
        # no such path at all: DivS/DivU lower to Z3's bvsdiv/bvudiv, which are
        # *total* (Z3 defines x/0), so no SigFPE exit is ever produced and the
        # option can never fire. It ships only in the `tracing` bundle, which a
        # user reaches by passing mode= — an opt-in, so the raise is routable:
        # the auto-dispatcher sends those workloads to Python rather than
        # silently dropping the divide-by-zero successor.
        "PRODUCE_ZERODIV_SUCCESSORS",
    }
)

# Inverse-polarity gate (angr-op0dn.14.7): options Rust cannot honor by their
# *absence*. Same shape as the CGC_NON_BLOCKING_FDS polarity trap (angr-op0dn.14.8):
# the divergent configuration is the one where the option is UNSET.
#
# EXTENDED_IROP_SUPPORT ships in every mode bundle, so `_RAISE_OPTION_NAMES`
# could never carry it — routing on its presence routes every user to Python.
# But the option only ever *widens* Python's op table: `vexop_to_simop`
# (engines/vex/claripy/irop.py) auto-generates a SimIROp from the op name when
# `extended=True` and raises UnsupportedIROpError when it is False. Set (the
# default) it is therefore honored transparently — Rust runs the op natively, or
# defers the block to Python where extended=True applies. Unset, the user is
# asking for the *narrow* table, and the Rust interpreter has no narrow mode:
# it would execute an auto-generated-only op where Python would have refused.
# That is the only direction it can diverge, and it is an explicit opt-out of a
# default option, so we make it loud (and auto-dispatch routes it to Python).
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

    ``RustExplorationManager.__init__`` accepts ``**kwargs`` (for forward
    compat with subclasses) and would otherwise swallow SimulationManager-only
    constructor arguments — ``hierarchy``, ``resilience``, ``techniques``, …
    — silently. Callers that forward user kwargs (notably the factory) use
    this to fail loudly instead of dropping them.
    """
    import inspect

    accepted = {
        name
        for name, param in inspect.signature(RustExplorationManager.__init__).parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return sorted(name for name in kwargs if name not in accepted)


# --- Auto engine dispatch (angr-op0dn.14.5.2) ---------------------------------
#
# `factory.simulation_manager(use_rust_engine=None)` asks "run this on Rust if
# Rust can run it, else Python". The routing decision is computed here so it is
# testable without a factory, and so the raise-option list stays a single source
# of truth (`rust_unsupported_options`).
#
# Auto mode is OFF by default: with the flag off, `use_rust_engine=None` behaves
# exactly like `use_rust_engine=False` (plain SimulationManager), so the default
# flip (angr-op0dn.14.6) is a one-line policy change, not a behavior rewrite.
_AUTO_DISPATCH_ENABLED = os.environ.get("ANGR_RUST_AUTO", "").lower() not in ("", "0", "false", "no")


def set_rust_auto_dispatch(enabled: bool) -> None:
    """Enable/disable auto engine selection for ``simulation_manager()``.

    Only consulted when the caller leaves ``use_rust_engine`` unset (``None``).
    Explicit ``True``/``False`` always wins. Also settable process-wide via the
    ``ANGR_RUST_AUTO`` environment variable (read once at import).
    """
    global _AUTO_DISPATCH_ENABLED
    _AUTO_DISPATCH_ENABLED = bool(enabled)


def rust_auto_dispatch_enabled() -> bool:
    """True when ``use_rust_engine=None`` may select the Rust engine."""
    return _AUTO_DISPATCH_ENABLED


def rust_default_engine_env() -> bool:
    """True when ``ANGR_DEFAULT_ENGINE=rust`` asks new projects to default to the Rust engine.

    The process-wide equivalent of ``Project(..., engine="rust")``: a project
    built while this is set routes ``simulation_manager()`` calls that leave
    ``use_rust_engine`` unset through the auto-dispatch predicate
    (:func:`rust_engine_eligible`), the same way the explicit sentinel does.
    Read at *project construction* time (not at import), so a test can set the
    variable and build a project without reloading the module. Any other value
    (including ``python``) leaves the default alone.
    """
    return os.environ.get("ANGR_DEFAULT_ENGINE", "").strip().lower() == "rust"


def rust_supports_arch(arch_name) -> bool:
    """True when the Rust engine implements *arch_name*.

    Delegates to the native ``arch_supported`` (backed by ``arch_from_name``,
    ``native/angr/src/arch/mod.rs``) so the Python answer cannot drift from the
    arches the interpreter actually has. Six today: X86 / AMD64 / ARM / ARM64 /
    MIPS32 / MIPS64. False (→ Python engine) for PPC32/PPC64/S390X and when the
    extension is not built.
    """
    if not RUST_EXPLORATION_AVAILABLE or not arch_name:
        return False
    from angr.rustylib.vex_engine import arch_supported

    return arch_supported(str(arch_name))


def rust_engine_eligible(project, states, kwargs) -> tuple[bool, str]:
    """Decide whether *project* / *states* / *kwargs* can run on the Rust engine.

    The eligibility predicate behind ``simulation_manager(use_rust_engine=None)``.
    Every ineligible answer names a signal the Rust engine would otherwise turn
    into a raise, a warning, or a silent divergence — routing to Python instead
    makes the whole loud surface transparent.

    Args:
        project: The :class:`angr.Project` the manager would drive.
        states: The seeding states (list of :class:`SimState`).
        kwargs: The SimulationManager constructor kwargs the caller passed.

    Returns:
        ``(eligible, reason)``. *reason* is a short human-readable string
        recorded on the returned manager as ``dispatch_reason`` and logged at
        debug level, so the route is always observable.
    """
    if not RUST_EXPLORATION_AVAILABLE:
        return False, "rust extension not built"

    arch_name = getattr(getattr(project, "arch", None), "name", None)
    if not rust_supports_arch(arch_name):
        return False, f"arch {arch_name} not implemented by the rust engine"

    unsupported_kwargs = unsupported_rust_manager_kwargs(kwargs)
    if unsupported_kwargs:
        return False, "simulation_manager kwargs not honored by the rust engine: " + ", ".join(unsupported_kwargs)

    for state in states or ():
        offending = rust_unsupported_options(getattr(state, "options", None))
        if offending:
            return False, "state options not honored by the rust engine: " + ", ".join(offending)

        events = _unsupported_inspect_events(state)
        if events:
            return False, "state.inspect events not dispatched by the rust engine: " + ", ".join(events)

    return True, "rust engine supports this project, state options, kwargs, and inspect breakpoints"


def _unsupported_inspect_events(state) -> list[str]:
    """Inspect events *state* already has breakpoints for that Rust cannot dispatch.

    Registering one of these on a Rust-owned state raises (see
    ``_format_unsupported_event_msg``, ``rust_state_proxy.py``), so a seeding
    state that carries one is a Python workload — the raise would otherwise be
    the regression the moment the default engine flips.
    """
    inspector = getattr(state, "inspect", None)
    breakpoints = getattr(inspector, "_breakpoints", None)
    if not breakpoints:
        return []
    from angr.exploration.rust_state_proxy import _RUST_INSPECT_SUPPORTED_EVENTS

    return sorted(event for event, bps in breakpoints.items() if bps and event not in _RUST_INSPECT_SUPPORTED_EVENTS)


# Z3 context sharing: make Rust and Python use the same Z3 context
# to avoid AST translation overhead between solvers.
_z3_context_shared = False


def _setup_shared_z3_context():
    """Share Python's Z3 context with Rust, so both create ASTs in the same context."""
    global _z3_context_shared
    if _z3_context_shared:
        return
    try:
        import atexit

        import z3

        from angr.rustylib.vex_engine import reset_shared_z3_context, set_shared_z3_context

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


def _warn_if_parallel_nondeterministic() -> bool:
    """Warn when ``deterministic=True`` meets a multi-worker scheduler.

    ``RUST_PARALLEL_WORKERS`` > 1 runs the work-stealing pool, whose steal
    order is nondeterministic *by design*: the set of found states still
    matches a serial run, but which state is found first — and hence the
    order results are reported in — does not. Witness choice stays canonical
    (that part is per-state), so this is a warning, not an error: the
    combination is legitimate when only the found *set* matters.

    Returns True when the warning fired (multi-worker + deterministic).
    Deliberately does NOT touch Z3's parallel mode — see bd memory
    ``avoid-z3-parallel-enable``.
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
    (``angr::stepping=debug,angr::interpreter=info``). Per-module filters are
    honored — the Rust side delegates filter parsing to env_logger.
    Applies once per process.

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
    except Exception as e:
        # cat-(b) FALLBACK WITH LOSS: env-driven level could not be applied;
        # manager construction proceeds, but Rust-side log output stays at
        # whatever level was set previously (typically off).
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


from angr.exploration._constants import PAGE_SIZE
from angr.exploration.rust_callback_dispatch import (
    RustCallbackDispatchMixin,
    _simproc_dispatch_name,
    _unconstrained_stub_spec,
)
from angr.exploration.rust_disk_cache import RustDiskCacheManager
from angr.exploration.rust_state_cache import RustStateCacheMixin
from angr.exploration.rust_state_export import RustStateExportMixin
from angr.exploration.rust_state_sync import RustStateSyncMixin

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
        return f'<State errored at {hex(self.addr)} class={self.error_class} with "{self.error}">'


# Default active-stash safety cap (angr-o4q3). A divergent / path-exploding
# exploration grows the active stash without bound until it exhausts RAM and is
# OOM-killed (observed: ralph run 20260613 iters 28-30, a CADET_00001 explore
# leaking ~1.6 states/step). This is a *count* backstop that converts unbounded
# growth into bounded growth — it is NOT a precise memory limit (per-state size
# varies), so it is set far above any realistic workload (CTF explores peak in
# the hundreds–low thousands of active states). When the cap is hit, excess
# forks are pruned to the ``pruned`` stash and a one-time warning is logged.
# Pass ``max_active_states=None`` to disable, or a smaller int to bound tighter.
# Memory-precise bounding is tracked separately (angr-wcxi).
DEFAULT_MAX_ACTIVE_STATES = 100_000


class RustExplorationManager(
    RustCallbackDispatchMixin,
    RustStateSyncMixin,
    RustStateCacheMixin,
    RustStateExportMixin,
    RustDiskCacheManager,
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

    API stability:
        The public surface of this class is enumerated in
        ``angr/exploration/_public_api.py`` and governed by the semver
        + deprecation contract documented in
        ``docs/advanced-topics/rust_engine.rst`` under
        "API stability contract". Names with a leading underscore are
        private implementation detail; names tagged ``Experimental:``
        in their docstring may change shape in any minor release.
    """

    # Class-level cache for Python init results per binary
    _init_cache: dict[str, angr.SimState] = {}
    _init_cache_max = 10

    # ``_disk_key_cache`` (MD5-of-binary memo) is provided by RustDiskCacheManager.

    # Class-level cache for blank_state objects keyed by (binary_path, addr).
    # blank_state() is expensive (~1ms); caching + copy() is <0.1ms.
    _blank_state_cache: dict[tuple, angr.SimState] = {}
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
    _loader_pages_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    # SimProcedures known to write memory (need full state.copy() for changed_bytes)
    _MEMORY_WRITING_PROCS = frozenset(
        {
            "read",
            "recv",
            "fgets",
            "scanf",
            "__isoc99_scanf",
            "fread",
            "gets",
            "getchar",
            "fgetc",
            "getc",
            "strncpy",
            "strcpy",
            "memcpy",
            "memmove",
            "memset",
            "strcat",
            "strncat",
            "sprintf",
            "snprintf",
        }
    )

    def __init__(
        self,
        project: angr.Project,
        active_states: list | None = None,
        save_unconstrained: bool = False,
        solver_timeout_ms: int = 30000,
        max_active_states: int | None = DEFAULT_MAX_ACTIVE_STATES,
        max_history: int = 1000,
        clear_caches_on_cleanup: bool = False,
        exploration_strategy: str = "bfs",
        use_shared_lineage_solver: bool = False,
        deterministic: bool = False,
        use_callback_memory_proxy: bool | None = None,
        use_callback_register_proxy: bool | None = None,
        use_callback_solver_proxy: bool | None = None,
        use_callback_callstack_proxy: bool | None = None,
        use_export_callstack_proxy: bool | None = None,
        use_export_memory_proxy: bool | None = None,
        use_simproc_fork_via_rust: bool | None = None,
        prefer_native_library_hooks: bool | None = None,
        use_native_lift: bool = True,
        symlinks: dict | None = None,
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
            use_callback_memory_proxy: If True, install ``RustMemoryProxy``
                as ``state.memory`` on SimProcedure callback states so loads
                / stores route directly into Rust instead of going through
                the cached state's claripy SimMemory + ``CallbackMemoryTracker``
                diff-and-push (angr-4scu step 3). When ``None`` (default), the
                proxy is ON (angr-grji4 human-GO flip 2026-07-15); the env var
                ``ANGR_RUST_USE_CALLBACK_MEMORY_PROXY=0`` is the opt-OUT escape
                hatch that forces the existing tracker path back on.
            use_callback_register_proxy: If True, install ``RustRegisterProxy``
                as ``state.registers`` on SimProcedure callback states so
                ``state.regs.<name>`` reads/writes route directly into Rust
                instead of going through the bundle apply + post-callback
                ``_extract_register_changes`` diff-and-push (angr-qj30, write-
                through .2). When ``None`` (default), the env var
                ``ANGR_RUST_USE_CALLBACK_REGISTER_PROXY=1`` toggles it on;
                otherwise off. Off keeps the existing diff-and-push path live.
            use_callback_solver_proxy: If True, install
                ``RustSolverProxyPlugin`` as ``state.solver`` on SimProcedure
                callback states so ``state.solver.add(constraint)`` routes
                directly into the underlying Rust state's solver, and
                ``constraints`` / ``eval`` / ``satisfiable`` / ``min`` /
                ``max`` read through Rust (angr-8oiw, write-through .3). When
                ``None`` (default), the env var
                ``ANGR_RUST_USE_CALLBACK_SOLVER_PROXY=1`` toggles it on;
                otherwise off. Off keeps the existing
                ``_install_rust_solver_on_callback_state`` monkey-patch path
                live (which maintains a parallel Python claripy solver).
            use_callback_callstack_proxy: If True, install
                ``RustCallStackProxyPlugin`` as ``state.callstack`` on
                SimProcedure callback states so iteration / top-frame
                attribute access reads frames live from Rust by ``state_id``
                via ``get_state_call_stack`` (angr-6o9p, write-through .4).
                When ``None`` (default), the env var
                ``ANGR_RUST_USE_CALLBACK_CALLSTACK_PROXY=1`` toggles it on;
                otherwise off. Off leaves the cached state's (typically
                entry-state) ``CallStack`` plugin untouched at callback
                time.
            use_export_callstack_proxy: If True, install
                ``RustCallStackProxyPlugin`` as ``state.callstack`` on
                materialized states returned from Rust (stash exports /
                ``found`` / ``active`` / etc.) instead of reconstructing a
                ``CallStack`` linked-list via ``register_plugin`` at export
                time (angr-yk2g, write-through boundary). When ``None``
                (default), the env var
                ``ANGR_RUST_USE_EXPORT_CALLSTACK_PROXY=1`` toggles it on;
                otherwise off. Off keeps the eager
                ``_sync_rust_callstack_to_state`` reconstruction path live.
            use_export_memory_proxy: If True, install ``RustMemoryProxy``
                as ``state.memory`` on materialized states returned from
                Rust (stash exports / ``found`` / ``active`` / etc.)
                instead of pulling Rust pages back into the SimState's
                claripy memory via ``state.memory.store(...)`` at export
                time (angr-ul4k, write-through boundary). When ``None``
                (default), the env var
                ``ANGR_RUST_USE_EXPORT_MEMORY_PROXY=1`` toggles it on;
                otherwise off. Off keeps the eager
                ``_sync_rust_memory_to_state`` writeback path live.
            use_simproc_fork_via_rust: If True, route SimProcedure additional
                successors through ``fork_state_to_stash(parent_id, 'active')``
                (Rust-owned fork) instead of ``_add_rust_state('active', ...)``
                followed by ``add_constraints_to_pending(...)``. Default off
                keeps the eager Python-state-push path live. When on,
                ``_add_forked_state`` forks the parent Rust state directly
                and applies any path-specific constraints via
                ``add_constraints_to_state(new_id, ...)`` (angr-t3mr, write-
                through boundary). When ``None`` (default), the env var
                ``ANGR_RUST_USE_SIMPROC_FORK_VIA_RUST=1`` toggles it on;
                otherwise off.
            prefer_native_library_hooks: If True (default), let native
                procedures serve ``use_sim_procedures`` hooks whose address
                lands inside a *non-main* loaded object (libc &c. under
                ``auto_load_libs``). Main-object hooks always stay on Python so
                ``proj.hook()`` overrides win. When ``None`` (the default), env
                var ``ANGR_RUST_PREFER_NATIVE_LIBRARY_HOOKS=0`` forces the old
                behavior of bouncing every in-object hook to Python.
            use_native_lift: If True (default), lift cold blocks in-process
                through the native libVEX seam instead of the pyvex Python
                callback. Requires a ``--features libvex-ffi`` build and an
                AMD64 target; on any other build/arch the flag is inert and
                lifting stays on the callback path. Blocks whose bytes are
                not fully concrete in the Rust memory sidecar (and blocks
                with a VEX opt-level override) fall back to the callback
                with an identical IRSB, so setting this False only costs
                speed, never fidelity. See
                docs/advanced-topics/rust_libvex_ffi.rst.
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
        # Init pipeline sequenced into four named phases (angr-wqao.4).
        # boot:       construct the Rust manager + apply basic config.
        # config:     resolve gate flags, init counters/caches, register
        #             callbacks + simprocedures.
        # link_state: multi-stage-reuse short-circuit (Python SimState↔Rust id).
        # activate:   add initial states to the Rust 'active' stash.
        self._phase_boot(
            project,
            save_unconstrained,
            clear_caches_on_cleanup,
            solver_timeout_ms,
            max_active_states,
            max_history,
            exploration_strategy,
            deterministic,
            use_native_lift,
        )
        self._phase_config(
            use_shared_lineage_solver,
            use_callback_memory_proxy,
            use_callback_register_proxy,
            use_callback_solver_proxy,
            use_callback_callstack_proxy,
            use_export_callstack_proxy,
            use_export_memory_proxy,
            use_simproc_fork_via_rust,
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
        if symlinks:
            for link, target in symlinks.items():
                self._pending_symlinks[str(link)] = (
                    target if isinstance(target, (bytes, bytearray)) else str(target).encode()
                )
        if self._phase_link_state(active_states):
            return
        self._phase_activate(active_states)

    def _phase_boot(
        self,
        project,
        save_unconstrained,
        clear_caches_on_cleanup,
        solver_timeout_ms,
        max_active_states,
        max_history,
        exploration_strategy,
        deterministic,
        use_native_lift,
    ):
        """Phase 1 (boot): construct the Rust manager and apply basic config."""
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
            _warn_if_parallel_nondeterministic()
        self._deterministic = bool(deterministic)

        if not RUST_EXPLORATION_AVAILABLE:
            raise ImportError("RustExplorationManager not available. Build with vex-engine feature enabled.")

        self._project = project
        self._save_unconstrained = save_unconstrained
        # See ``cleanup()`` — only honored when the manager has a real Rust
        # backend (i.e. not the multi-stage-reuse early return below).
        self._clear_caches_on_cleanup = clear_caches_on_cleanup
        is_le = project.arch.memory_endness == "Iend_LE"
        self._rust_mgr = _RustExplorationManager(project.arch.name, little_endian=is_le)

        # angr-krp1: plumb the SimOS name to Rust so the syscall dispatcher
        # can route DECREE CGC binaries (x86 syscall numbers 1-7) through
        # the CGC table instead of the Linux i386 table. project.simos.name
        # is "Linux"/"CGC"/"Windows"/"Java"/... — lowercased by the Rust
        # setter so case differences don't matter. Default ("linux") covers
        # the common case so most paths are unaffected.
        simos_name = getattr(getattr(project, "simos", None), "name", None) or ""
        if simos_name:
            self._rust_mgr.set_os_name(simos_name)

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

        # z087y Stage-3: native (in-process) libVEX cold-block lifting. Only
        # flip the flag when the .so was compiled with the `libvex-ffi`
        # feature (probed via `libvex_ffi_enabled()`) AND the target is AMD64
        # — the only arch the native lifter marshals today. On by default
        # since angr-op0dn.2.2 (bench evidence in
        # docs/advanced-topics/rust_libvex_ffi.rst): no bench regressed, wins
        # up to 13%, and any block the native lifter cannot serve falls back
        # to the pyvex callback with an identical IRSB. `libvex-ffi` is still
        # a non-default cargo feature, so on a stock build the probe is False
        # and this is inert.
        self._use_native_lift = bool(use_native_lift)
        if self._use_native_lift:
            try:
                from angr.rustylib.vex_engine import libvex_ffi_enabled
            except ImportError:
                libvex_ffi_enabled = None
            if libvex_ffi_enabled is not None and libvex_ffi_enabled() and project.arch.name == "AMD64":
                self._rust_mgr.set_native_lift_enabled(True)

        # Configure exploration strategy. Reuses the post-init setter so the
        # validation and FFI-call shape live in one place. Always invoke so a
        # bad value raises ValueError eagerly during construction; the
        # default-bfs path is cheap (one FFI hop into the Rust setter).
        self.set_exploration_strategy(exploration_strategy)

    def _phase_config(
        self,
        use_shared_lineage_solver,
        use_callback_memory_proxy,
        use_callback_register_proxy,
        use_callback_solver_proxy,
        use_callback_callstack_proxy,
        use_export_callstack_proxy,
        use_export_memory_proxy,
        use_simproc_fork_via_rust,
        prefer_native_library_hooks,
    ):
        """Phase 2 (config): gate flags, counters/caches, callbacks + simprocedures."""
        # angr-3ms1 step 1b: opt-in flag for fork-time
        # SharedLineageSolver materialization. Stashed here so
        # _add_rust_state can push the value onto every seed state's
        # solver context. Default off keeps slice-1c's gate inert (and
        # the v5a5-slice-4c.3-retry-failed-bfs-thrash-fundamental
        # baby-re regression out of CI).
        self._use_shared_lineage_solver = bool(use_shared_lineage_solver)

        # angr-4scu step 3: gate for installing RustMemoryProxy as
        # ``state.memory`` on SimProcedure callback states. Default off keeps
        # the existing CallbackMemoryTracker diff-and-push path live. When
        # on, ``_create_state_for_callback`` swaps ``state.memory`` with a
        # ``RustMemoryProxy``, so loads/stores during the SimProc route
        # directly into Rust and the post-callback tracked-writes replay is
        # skipped (writes already landed in Rust). Default ON as of the
        # angr-grji4 human-GO flip (2026-07-15); env var
        # ``ANGR_RUST_USE_CALLBACK_MEMORY_PROXY=0`` is the opt-OUT escape
        # hatch that forces the legacy CallbackMemoryTracker path back on
        # when the kwarg is left at its default ``None``. Multi-session epic
        # (see bd memory boundary-4scu-simmem-spike).
        self._use_callback_memory_proxy = _resolve_env_flag(
            use_callback_memory_proxy, "ANGR_RUST_USE_CALLBACK_MEMORY_PROXY", default=True
        )

        # angr-qj30 (write-through .2): gate for installing
        # ``RustRegisterProxy`` as ``state.registers`` on SimProcedure
        # callback states. Default off keeps the existing bundle-apply +
        # ``_extract_register_changes`` diff-and-push path live. When on,
        # ``_create_state_for_callback`` skips the bundle register apply
        # and installs the proxy so every ``state.regs.<name>`` read/write
        # routes directly into Rust by state_id, and the matching site in
        # ``_handle_simprocedure_callback`` skips ``reg_changes``
        # extraction (writes already landed in Rust). Env var
        # ``ANGR_RUST_USE_CALLBACK_REGISTER_PROXY=1`` toggles default-on
        # when the kwarg is left at its default ``None``.
        self._use_callback_register_proxy = _resolve_env_flag(
            use_callback_register_proxy, "ANGR_RUST_USE_CALLBACK_REGISTER_PROXY"
        )

        # angr-8oiw (write-through .3): gate for installing
        # ``RustSolverProxyPlugin`` as ``state.solver`` on SimProcedure
        # callback states. Default off keeps the existing
        # ``_install_rust_solver_on_callback_state`` monkey-patch path live
        # (which mirrors constraints into a parallel Python claripy solver).
        # When on, ``_create_state_for_callback`` skips the monkey-patch
        # install and installs the proxy so every ``state.solver.add`` goes
        # straight to Rust by state_id, and reads come from the Rust state
        # directly (no parallel claripy solver). Env var
        # ``ANGR_RUST_USE_CALLBACK_SOLVER_PROXY=1`` toggles default-on when
        # the kwarg is left at its default ``None``.
        self._use_callback_solver_proxy = _resolve_env_flag(
            use_callback_solver_proxy, "ANGR_RUST_USE_CALLBACK_SOLVER_PROXY"
        )

        # angr-6o9p (write-through .4): gate for installing
        # ``RustCallStackProxyPlugin`` as ``state.callstack`` on
        # SimProcedure callback states. Default off leaves the cached
        # state's CallStack plugin untouched at callback time. When on,
        # ``_create_state_for_callback`` installs the proxy so iteration
        # / top-frame attribute access reads frames live from Rust by
        # ``state_id`` via ``get_state_call_stack``. Env var
        # ``ANGR_RUST_USE_CALLBACK_CALLSTACK_PROXY=1`` toggles default-on
        # when the kwarg is left at its default ``None``.
        self._use_callback_callstack_proxy = _resolve_env_flag(
            use_callback_callstack_proxy, "ANGR_RUST_USE_CALLBACK_CALLSTACK_PROXY"
        )

        # angr-yk2g (write-through boundary): gate for installing
        # ``RustCallStackProxyPlugin`` as ``state.callstack`` on every
        # materialized state returned from a Rust stash (instead of
        # rebuilding a ``CallStack`` linked-list via ``register_plugin`` in
        # ``_sync_rust_callstack_to_state``). Default off keeps the eager
        # reconstruction path live; when on, the export pipeline installs
        # the proxy and ``state.callstack`` reads frames live from Rust by
        # ``state_id`` via ``get_state_call_stack``. Env var
        # ``ANGR_RUST_USE_EXPORT_CALLSTACK_PROXY=1`` toggles default-on
        # when the kwarg is left at its default ``None``.
        self._use_export_callstack_proxy = _resolve_env_flag(
            use_export_callstack_proxy, "ANGR_RUST_USE_EXPORT_CALLSTACK_PROXY"
        )

        # angr-ul4k (write-through boundary): gate for installing
        # ``RustMemoryProxy`` as ``state.memory`` on every materialized
        # state returned from a Rust stash (instead of writing Rust pages
        # back into the SimState's claripy memory via
        # ``state.memory.store(...)`` in ``_sync_rust_memory_to_state``).
        # Default off keeps the eager writeback path live; when on, the
        # export pipeline installs the proxy and ``state.memory.load(...)``
        # reads bytes live from Rust by ``state_id`` via the existing
        # ``get_state_memory_ast`` FFI. Env var
        # ``ANGR_RUST_USE_EXPORT_MEMORY_PROXY=1`` toggles default-on when
        # the kwarg is left at its default ``None``.
        self._use_export_memory_proxy = _resolve_env_flag(use_export_memory_proxy, "ANGR_RUST_USE_EXPORT_MEMORY_PROXY")

        # angr-t3mr (write-through boundary): gate for routing SimProcedure
        # additional successors through ``fork_state_to_stash(parent_id,
        # 'active')`` (Rust-owned fork) instead of the legacy
        # ``_add_rust_state`` + ``add_constraints_to_pending`` chain. Default
        # off keeps the eager Python-state push live; when on,
        # ``_add_forked_state`` (rust_callback_dispatch.py) forks the parent
        # Rust state directly and applies any path-specific constraints via
        # ``add_constraints_to_state(new_id, ...)``. Env var
        # ``ANGR_RUST_USE_SIMPROC_FORK_VIA_RUST=1`` toggles default-on when
        # the kwarg is left at its default ``None``.
        self._use_simproc_fork_via_rust = _resolve_env_flag(
            use_simproc_fork_via_rust, "ANGR_RUST_USE_SIMPROC_FORK_VIA_RUST"
        )

        # angr-a8epx / angr-gorvf.3.2: native-dispatch gate for library hooks.
        # Off, native procedures fire only for hooks OUTSIDE every loaded object
        # (the extern-object stubs). On a dynamically-linked binary loaded with
        # ``auto_load_libs=True, use_sim_procedures=True`` every libc hook lands
        # inside the loaded libc's .text, so that would send all of them to
        # Python. On (the default since angr-gorvf.6) the native registry serves
        # those hooks too; main-object hooks still go to Python, preserving
        # ``proj.hook()`` overrides. It was off while the native string procs
        # scanned a wider symbolic window than angr's Python ones and could
        # fork-storm on symbolic data; ``MAX_SYMBOLIC_SCAN_BYTES`` (c94941fff)
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
        # symbols) the callback-memory-proxy gate would otherwise miss.
        self._stats_proxy_mem_fallback_python_load = 0
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
        self._registered_hooks: set = set()

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

        # Concrete byte strings this manager has spliced into posix.stdin as a
        # Rust-read packet (angr-psrxs). A materialized child state is built by
        # copying its parent's cached SimState, so it inherits the parent's
        # injected packet; the child's own Rust symbol list is cumulative and
        # already covers those bytes, so the inherited packet must be dropped
        # before the fresh one is appended. See `_inject_rust_stdin_inner`.
        self._rust_stdin_packets: set[bytes] = set()

        # angr-lyvf2: seeds the Python init skip fast-forwarded past their own
        # PC, as (pre-init addr, un-advanced SimState, rust id of the advanced
        # state). `explore()` consumes this once, then drops the refs.
        self._preinit_seeds: list[tuple[int, angr.SimState, int | None]] = []

        # Lazy SimState references handed out by _get_stash_states. Keyed by
        # Rust state id so repeated stash reads return the same wrapper
        # (preserves the `mgr.active[0] is mgr.active[0]` invariant). Wrappers
        # materialize their SimState on first attribute access — see
        # `_LazySimStateRef` in rust_state_export.py. Pruned in
        # `_cleanup_state_cache` alongside _state_cache.
        from angr.exploration.rust_state_export import _LazySimStateRef

        self._lazy_state_refs: dict[int, _LazySimStateRef] = {}

        # Maximum state cache size. Cache is bounded to the in-flight callback
        # state plus a small LRU window of recently-mutated states; root states
        # are pinned and never count against the cap. Pre-angr-qm7w this was
        # 500 and the cache tracked O(active_states), making it the dominant
        # driver of Python-side memory growth.
        self._max_state_cache_size = 8

        # Cache claripy AST -> Z3 AST pointer for register sync.
        # Skips redundant z3_backend.convert(reg_val) + .as_ast().value lookups
        # when the same symbolic register is re-imported across SimProcedure
        # callbacks. Holds a strong ref to the z3 object so the AST pointer
        # stays valid (Z3 ASTs are refcounted; the shared context outlives the
        # manager). Bounded size with simple drop-and-rebuild eviction.
        self._z3_ptr_cache: dict[tuple, tuple] = {}
        self._z3_ptr_cache_max = 1024
        self._z3_ptr_cache_hits = 0
        self._z3_ptr_cache_misses = 0

        # Track current callback state for memory access during callbacks
        # This allows memory_load callback to access the correct symbolic state
        self._callback_state: angr.SimState | None = None

        # Cache bundle register values from _create_state_for_callback for
        # reuse as register snapshot (avoids reading registers back from state)
        self._last_bundle_registers: dict | None = None

        # Per-state metadata (symbolic_pages / hook_symbolic_memory /
        # addr_to_ast) is now stored on the Rust side in `RustSimState`. Access
        # goes through `self._rust_mgr.{get,set}_state_*` PyO3 methods so the
        # storage and the state lifetime are unified — when Rust drops a state,
        # its metadata is freed automatically.
        self._max_symbolic_pages_cache = 100  # Retained for back-compat hooks.

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
        self._state_roots: dict[int, int] = {}

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
        # and follow the 5-touchpoint pattern documented there.
        # See docs/advanced-topics/rust_engine.rst for the MVP scope.
        from angr.exploration.rust_state_proxy import (
            _RUST_INSPECT_EVENT_BITS,
            _RUST_INSPECT_SUPPORTED_EVENTS,
        )

        self._inspect_breakpoints: dict[str, list] = {evt: [] for evt in _RUST_INSPECT_SUPPORTED_EVENTS}
        self._INSPECT_EVENT_BITS = dict(_RUST_INSPECT_EVENT_BITS)
        # Reentrancy guard: when a user action callback runs, suppress
        # nested inspect dispatch on the same manager. uq4n.4 covers
        # the full guard test.
        self._inspect_dispatch_depth = 0
        # Lazy RustInspectProxy instance (one per manager, shared across proxies).
        self._inspect_proxy: RustInspectProxy | None = None

        # Cached memory layout from disk cache for fast _sync_memory_to_rust
        self._mem_cache: dict | None = None

        # Pre-computed register dict from disk cache for fast register sync
        self._precomputed_regs: dict | None = None

        # Track stdin BVS variables for state export.
        # List of (claripy_bvs, size_ast) tuples from SimPacketsStream.content.
        # Populated during SimProcedure callbacks that read from stdin (fgets, read, etc.).
        # Used to restore stdin content on found states that were forked purely in Rust.
        self._stdin_content: list = []

    def _phase_link_state(self, active_states):
        """Phase 3 (link_state): multi-stage-reuse short-circuit.

        Returns True when an existing Rust manager was reused (init should
        stop here), False when a fresh manager must add its initial states.
        """
        # Multi-stage explore reuse: if the initial state came from a previous
        # RustExplorationManager for the same project, reuse the old Rust manager
        # instead of creating a new one. This avoids lossy constraint transfer
        # that fails after ~50 stages (Z3 dedup causes UNSAT, claripy loses constraints).
        self._reused_from = None
        if active_states:
            _single = active_states[0] if isinstance(active_states, (list, tuple)) else active_states
            old_mgr = getattr(getattr(_single, "scratch", None), "rust_mgr", None)
            old_state_id = getattr(getattr(_single, "scratch", None), "rust_found_state_id", None)
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
                    self._perf_stats.set_init_phase("total", time.perf_counter_ns() - self._init_start)
                    return True
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: multi-stage manager reuse failed;
                    # falls back to building a fresh Rust manager from this state.
                    # Already debug-logs the cause.
                    l.debug(f"Multi-stage reuse failed, falling back to normal init: {e}")

        return False

    def _phase_activate(self, active_states):
        """Phase 4 (activate): add initial states to the Rust 'active' stash."""
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
                self._initial_seed_states = [s.copy() for s in active_states]
            for state in active_states:
                # Detect state options
                if hasattr(state, "options"):
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
                        avoid_multi_reads = o.AVOID_MULTIVALUED_READS in state.options
                        avoid_multi_writes = o.AVOID_MULTIVALUED_WRITES in state.options
                        # Read Python's strategy limits from memory plugin
                        read_limit = 1024  # Python default
                        write_limit = 128  # Python default
                        if hasattr(state, "memory"):
                            mem = state.memory
                            if hasattr(mem, "read_strategies") and mem.read_strategies:
                                for strat in mem.read_strategies:
                                    if hasattr(strat, "_limit"):
                                        read_limit = strat._limit
                                        break
                            if hasattr(mem, "write_strategies") and mem.write_strategies:
                                for strat in mem.write_strategies:
                                    if hasattr(strat, "_limit"):
                                        write_limit = strat._limit
                                        break
                        self._rust_mgr.configure_concretization_strategies(
                            use_approx,
                            read_limit,
                            write_limit,
                            sym_write,
                            avoid_multi_reads,
                            avoid_multi_writes,
                        )
                        l.debug(
                            "Configured concretization: approx=%s, read_limit=%d, "
                            "write_limit=%d, sym_write=%s, avoid_reads=%s, "
                            "avoid_writes=%s",
                            use_approx,
                            read_limit,
                            write_limit,
                            sym_write,
                            avoid_multi_reads,
                            avoid_multi_writes,
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
                state = self._run_python_init_if_needed(state)
                self._perf_stats.add_init_phase("python_run", time.perf_counter_ns() - _t0)

                _t0 = time.perf_counter_ns()
                sid = self._add_rust_state("active", state)
                self._perf_stats.add_init_phase("add_rust_state", time.perf_counter_ns() - _t0)

                # angr-lyvf2: the init skip fast-forwards the seed past its own
                # PC, so Rust's pre-step find check can never see `seed_addr`.
                # Park the un-advanced seed so `explore()` can route it to FOUND
                # if the caller's find targets that address (vanilla angr's
                # Explorer matches it at filter() time, before any step).
                # The angr-027h phase-2 retry re-activates copies of the same
                # seeds mid-explore; parking them again would double-report the
                # find (angr-bdeqa), so only the original activation parks.
                if state.addr != seed_addr and not getattr(self, "_phase2_reseeding", False):
                    self._preinit_seeds.append((seed_addr, seed_state, sid))

        self._install_python_servable_pages()

        self._perf_stats.set_init_phase("total", time.perf_counter_ns() - self._init_start)

    def _install_python_servable_pages(self) -> None:
        """Tell Rust which pages ``_cb_fetch_page`` could actually serve.

        angr-gorvf.4.6: every page the run loop still fetches on the ZeroPy
        FAIL benches is a lazy-stack page Python DECLINES (measured: 84/84
        across the four benches whose whole residual GIL was this one site).
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
        # bytes. (Measured when it did: csgames2018 lost 27% — the fabricated
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
        # speedup analysis. All five paths are instrumented from the public
        # _handle_* entry points in rust_callback_dispatch.py.
        for label, key in (
            ("Syscall", "syscall"),
            ("Find predicate", "find_predicate"),
            ("Avoid predicate", "avoid_predicate"),
            ("Symbolic branch", "symbolic_branch"),
            ("Python VEX fallback", "vex_fallback"),
            ("Posix inject", "posix"),  # angr-afbx
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

    def get_exploration_summary(self) -> str:
        """Return a high-level summary of the exploration run."""
        s = self._perf_stats
        lines = ["=== Exploration Summary ==="]

        # Duration
        explore_ns = getattr(self, "_time_in_explore_ns", 0)
        init_ns = s.get("init_total_ns", 0)
        total_ns = init_ns + explore_ns
        lines.append(
            f"Total time: {total_ns / 1e6:.1f}ms (init: {init_ns / 1e6:.1f}ms, explore: {explore_ns / 1e6:.1f}ms)"
        )

        # Steps and throughput
        explore_s = explore_ns / 1e9 if explore_ns > 0 else 0
        steps_per_sec = f" ({self._stats_ffi_crossings / explore_s:.0f} steps/sec)" if explore_s > 0.001 else ""
        lines.append(f"Steps: {self._stats_ffi_crossings}{steps_per_sec}")

        # State counts
        try:
            n_found = len(self._rust_mgr.get_state_ids("found"))
            n_active = len(self._rust_mgr.get_state_ids("active"))
            n_deadended = len(self._rust_mgr.get_state_ids("deadended"))
            n_avoided = len(self._rust_mgr.get_state_ids("avoided"))
            lines.append(f"States: {n_found} found, {n_active} active, {n_avoided} avoided, {n_deadended} deadended")
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: state-count snapshot failed;
            # the summary skips the per-stash count line.
            pass

        # Callback breakdown
        cb_total = self._stats_callback_count
        sp_count = s.get("callback_simprocedure_count", 0)
        native_count = cb_total - sp_count  # memory/lift/fetch callbacks
        lines.append(f"Callbacks: {cb_total} total ({sp_count} SimProcedure, {native_count} other)")
        if self._stats_time_in_callbacks_ns > 0:
            lines.append(f"  Time in callbacks: {self._stats_time_in_callbacks_ns / 1e6:.1f}ms")

        # Per-procedure breakdown (top 5)
        if self._procedure_times:
            sorted_procs = sorted(self._procedure_times.items(), key=lambda x: -x[1]["execute_ns"])
            lines.append(f"SimProcedure breakdown ({len(sorted_procs)} unique):")
            for pname, pt in sorted_procs[:5]:
                lines.append(f"  {pname}: {pt['count']}x, {pt['execute_ns'] / 1e6:.1f}ms")

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
        # angr-5rjbq: under the memory-proxy gate the store callbacks above are
        # no-ops (state.memory *is* Rust memory), so tell Rust to keep such
        # stores in rust_memory itself instead of dropping them on the floor.
        if hasattr(callbacks, "set_memory_is_rust_proxy"):
            callbacks.set_memory_is_rust_proxy(bool(getattr(self, "_use_callback_memory_proxy", False)))
        if hasattr(callbacks, "set_memory_load_symbolic_full"):
            callbacks.set_memory_load_symbolic_full(self._cb_memory_load_symbolic_full)
        # state.inspect MVP (angr-uq4n.2, angr-d46u) — register dispatchers
        # even when no BPs are set so the Rust side has a target if
        # instrumentation fires unexpectedly. The bitmask gates dispatch.
        # Iterate the single source of truth so adding a new event
        # auto-wires registration. Read directly from the module rather
        # than self._INSPECT_EVENT_BITS so this method works even before
        # __init__ has finished. The hasattr() check tolerates older .so
        # builds. Skip events whose dispatch fires from Python (angr-xmfj —
        # simprocedure/syscall/dirty); they have no PythonCallbacks slot
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

                # angr-hcok: when the callback-memory-proxy gate is on, the
                # cached state's ``memory`` plugin is a ``RustMemoryProxy``
                # that re-enters ``_rust_mgr.get_state_memory_ast(...)`` —
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
        # angr-hcok: under the callback-memory-proxy gate, ``state.memory``
        # is a ``RustMemoryProxy`` that routes ``store(...)`` back into
        # ``_rust_mgr.set_state_memory_concrete(...)`` — but we're inside
        # ``_rust_mgr.run()`` which holds ``&mut self``, so the re-entry
        # raises "Already mutably borrowed". Rust already performed the
        # store internally before invoking this callback, so the Python
        # shadow is redundant under the gate; just acknowledge and return.
        if _is_rust_memory_proxy(state.memory):
            return
        try:
            val = claripy.BVV(int.from_bytes(data, "little"), len(data) * 8)
            state.memory.store(addr, val, endness=state.arch.memory_endness)
        except (SimError, ClaripyError) as e:
            # cat-(c) WRONG-ANSWER RISK: memory store silently dropped on Sim/
            # Claripy error — subsequent loads see stale data. Already warns.
            l.warning(f"Memory store error at 0x{addr:x}: {e}")

    def _cb_lift_block(self, addr: int, opt_level: int = None, dirty_bytes: bytes = None) -> str:
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
                block = self._project.factory.block(addr, **kwargs)
                irsb = block.vex
                if self._inspect_breakpoints.get("vex_lift"):
                    self._cb_inspect_vex_lift(-1, "after", addr, irsb.size)
                return self._serialize_irsb(irsb)
            except (SimEngineError, ClaripyError, PyVEXError) as e:
                # cat-(c) WRONG-ANSWER RISK: lift returned empty IRSB; Rust will
                # treat the block as a no-op step. Already warns.
                l.warning(f"Lift error at 0x{addr:x}: {e}")
                return "{}"
        finally:
            self._perf_stats.record_lift_block(time.perf_counter_ns() - _lb_start)

    def _cb_fetch_page(self, page_addr: int) -> tuple:
        _fp_start = time.perf_counter_ns()
        try:
            state = self._get_default_state()
            if state is None:
                return (bytes(4096), 0, False)
            # angr-hcok: under the callback-memory-proxy gate, ``state.memory``
            # is a ``RustMemoryProxy`` that calls back into ``_rust_mgr`` —
            # invalid while ``run()`` holds ``&mut self``. Decline the page;
            # Rust will fall back to its own zero-fill / static-binary page
            # source rather than asking Python for a copy.
            if _is_rust_memory_proxy(state.memory):
                return (bytes(4096), 0, False)
            try:
                # angr-gorvf.4.5: UltraPage side tables answer this without
                # materializing a page-wide AST. See _fetch_page_from_ultrapage.
                fast = self._fetch_page_from_ultrapage(state, page_addr)
                if fast is not None:
                    return fast
                data = state.memory.load(page_addr, 4096, endness=state.arch.memory_endness)
                is_symbolic = getattr(data, "symbolic", False)
                if is_symbolic:
                    if _DBG:
                        l.debug(f"fetch_page 0x{page_addr:x}: has symbolic data, declining")
                    return (bytes(4096), 0, False)
                concrete = state.solver.eval(data).to_bytes(4096, "little")
                return (concrete, 7, True)
            except (SimError, ClaripyError):
                # cat-(b) FALLBACK WITH LOSS: page fetch failed; return empty page
                # with perms=0 so Rust marks it inaccessible (a load there will
                # error rather than silently succeed). Already debug-logs.
                l.debug("fetch_page 0x%x: failed to load/eval, returning empty", page_addr, exc_info=True)
                return (bytes(4096), 0, False)
        finally:
            self._perf_stats.record_fetch_page(time.perf_counter_ns() - _fp_start)

    def _cb_get_register(self, offset: int, size: int) -> tuple[bytes, bool, object | None]:
        state = self._get_callback_state() or self._get_default_state()
        if state is None:
            return (bytes(size), False, None)
        try:
            val = state.registers.load(offset, size, endness=state.arch.register_endness)
            is_sym = getattr(val, "symbolic", False)
            concrete = state.solver.eval(val).to_bytes(size, "little")
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
            val = claripy.BVV(int.from_bytes(data, "little"), len(data) * 8)
            state.registers.store(offset, val, endness=state.arch.register_endness)
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: register write failed — Python state
            # diverges from Rust on this register. Already warns.
            l.warning(f"put_register error at offset {offset}: {e}")

    def _cb_dirty_call(self, name: str, args: list, ret_ty_bits: int) -> tuple[bytes, bool, object | None]:
        state = self._get_callback_state() or self._get_default_state()
        if state is None:
            return (bytes(ret_ty_bits // 8), False, None)
        # state.inspect dispatch (angr-xmfj). state_id falls back to -1
        # because dirty calls run from a Rust interpreter dispatch where the
        # state-id is not threaded through this callback yet; the inspect
        # proxy falls back to the cached default state, which is the same
        # SimState the handler sees here. BPs that read `state.inspect.*`
        # see the live attrs.
        sid = getattr(self, "_current_callback_state_id", None)
        if sid is None:
            sid = -1
        try:
            from angr.engines.vex.heavy import dirty as dirty_module

            if not hasattr(dirty_module, name):
                l.warning(f"No dirty call handler for {name}")
                return (bytes(ret_ty_bits // 8), False, None)

            handler = getattr(dirty_module, name)
            claripy_args = [claripy.BVV(arg, 64) for arg in args]

            self._cb_inspect_dirty(sid, "before", name, handler, claripy_args, None)

            result, constraints = handler(state, *claripy_args)

            if constraints:
                for c in constraints:
                    state.solver.add(c)

            override = self._cb_inspect_dirty(sid, "after", name, handler, claripy_args, result)
            if override is not None:
                # User BP set state.inspect.dirty_result — honor the override
                # in place of the handler's return value (angr-uy32).
                result = override

            if result is None:
                return (bytes(ret_ty_bits // 8), False, None)

            is_sym = getattr(result, "symbolic", False)
            concrete_val = state.solver.eval(result)
            num_bytes = ret_ty_bits // 8
            concrete_bytes = concrete_val.to_bytes(num_bytes, "little")

            if is_sym:
                self._register_handle(id(result), result)
                return (concrete_bytes, True, result)
            return (concrete_bytes, False, None)

        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: dirty call handler raised; Rust sees
            # zero bytes for the result. Already warns.
            l.warning(f"dirty_call {name} error: {e}")
            return (bytes(ret_ty_bits // 8), False, None)

    def _cb_resolve_function(self, addr: int, name: str | None) -> tuple[str, int, bool] | None:
        """Resolve an unmodeled function call."""
        if hasattr(self._project, "_sim_procedures"):
            if addr in self._project._sim_procedures:
                proc = self._project._sim_procedures[addr]
                proc_name = proc.__class__.__name__ if hasattr(proc, "__class__") else str(proc)
                num_args = getattr(proc, "num_args", 0) or 0
                no_ret = getattr(proc, "NO_RET", False)
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
                                    got_addr = insn.address + insn.size + op.mem.disp
                                    if got_addr in got_to_sym:
                                        sym_name = got_to_sym[got_addr]
                                        if sym_name in proc_by_name:
                                            proc_addr, proc = proc_by_name[sym_name]
                                            num_args = getattr(proc, "num_args", 0) or 0
                                            no_ret = getattr(proc, "NO_RET", False)
                                            l.debug(f"Resolved PLT at 0x{addr:x} to {sym_name} (GOT 0x{got_addr:x})")
                                            return (sym_name, num_args, no_ret)
                                    else:
                                        state = self._get_default_state()
                                        if state:
                                            got_val = state.memory.load(got_addr, 8, endness="Iend_LE")
                                            extern_addr = state.solver.eval(got_val)
                                            if extern_addr in self._project._sim_procedures:
                                                proc = self._project._sim_procedures[extern_addr]
                                                proc_name = (
                                                    proc.__class__.__name__ if hasattr(proc, "__class__") else str(proc)
                                                )
                                                num_args = getattr(proc, "num_args", 0) or 0
                                                no_ret = getattr(proc, "NO_RET", False)
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
                        num_args = getattr(proc_class, "num_args", 0) or 0
                        no_ret = getattr(proc_class, "NO_RET", False)
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
                            no_ret = getattr(proc_class, "NO_RET", False)
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
                    if section.is_executable and section.min_addr <= addr < section.max_addr:
                        if section.name not in (".plt", ".plt.got", ".plt.sec"):
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
                state.memory.store(addr, val, endness="Iend_LE")
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

    def _fetch_page_from_ultrapage(self, state, page_addr: int) -> tuple | None:
        """Serve one page fetch straight off the UltraPage side tables.

        Returns the ``(data, perms, is_mapped)`` triple Rust expects, or None
        when this state's memory backend isn't an UltraPage and the caller must
        fall back to ``memory.load`` + ``solver.eval``.

        angr-gorvf.4.5: the load-based path builds a 32768-bit AST for the whole
        page just to answer "is it symbolic?". Every page the ZeroPy FAIL benches
        still fetch is a lazy-stack page that IS symbolic (measured: 7/7 on
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
        concrete_data = getattr(page_obj, "concrete_data", None)
        symbolic_bitmap = getattr(page_obj, "symbolic_bitmap", None)
        if not isinstance(concrete_data, bytearray) or len(concrete_data) != 4096:
            return None  # non-UltraPage backend — caller uses the slow path
        if getattr(page_obj, "symbolic_data", None):
            return (bytes(4096), 0, False)
        if not zero_fill and (symbolic_bitmap is None or any(symbolic_bitmap)):
            # Uninitialized bytes on the page would default-fill symbolic.
            return (bytes(4096), 0, False)
        return (bytes(concrete_data), 7, True)

    def _cb_batch_fetch_pages(self, page_addrs: list) -> list:
        state = self._get_callback_state() or self._get_default_state()
        if state is None:
            return [(bytes(4096), 0, True) for _ in page_addrs]

        # angr-hcok: proxy gate — decline; Rust falls back to its own
        # zero-fill / static-binary page source.
        if _is_rust_memory_proxy(state.memory):
            return [(bytes(4096), 0, False) for _ in page_addrs]

        results = []
        for page_addr in page_addrs:
            try:
                fast = self._fetch_page_from_ultrapage(state, page_addr)
                if fast is not None:
                    results.append(fast)
                    continue
                data = state.memory.load(page_addr, 4096, endness="Iend_LE")
                if getattr(data, "symbolic", False):
                    # angr-gorvf.4.3: a symbolic page is DECLINED (is_mapped=False)
                    # — Rust keeps serving it through the per-load memory_load
                    # callback, which preserves the AST. Concretizing it here was
                    # pure waste: a `solver.eval` of a 32768-bit symbolic AST (a
                    # full Z3 solve; 745ms of the 788ms callback GIL on
                    # google2016_unbreakable_1) whose result was then thrown away
                    # by `fetch_pages_batch`, which skips every !is_mapped entry.
                    # `_cb_fetch_page` already declined without evaluating.
                    results.append((bytes(4096), 0, False))
                else:
                    concrete = state.solver.eval(data).to_bytes(4096, "little")
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
        # angr-hcok: proxy gate — Rust already holds the symbolic AST in
        # its own memory; skip the Python shadow write.
        if _is_rust_memory_proxy(state.memory):
            if hasattr(ast, "length") and ast.length:
                self._register_handle(id(ast), ast, addr=addr, size=ast.length // 8)
            return
        try:
            if hasattr(ast, "length") and ast.length:
                size = ast.length // 8
                state.memory.store(addr, ast, endness=state.arch.memory_endness, inspect=False, disable_actions=True)
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
        # angr-hcok: proxy gate — the Multi-cell fast path above already
        # handled the write directly in Rust; if it declined, skip the
        # Python shadow store to avoid the &mut self re-entry.
        if _is_rust_memory_proxy(state.memory):
            self._register_handle(id(data_ast), data_ast)
            return
        try:
            state.memory.store(
                addr_ast, data_ast, endness=state.arch.memory_endness, inspect=False, disable_actions=True
            )
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
        # angr-hcok: proxy gate — synthesize a fresh BVS rather than
        # re-entering Rust via the proxy mid-run().
        if _is_rust_memory_proxy(state.memory):
            ast = claripy.BVS(f"sym_addr_load_{size}", size * 8, explicit_name=False)
            self._register_handle(id(ast), ast, size=size)
            return ast
        try:
            ast = state.memory.load(
                addr_ast, size, endness=state.arch.memory_endness, inspect=False, disable_actions=True
            )
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
        if self._callbacks is None or not hasattr(self._callbacks, "set_inspect_enabled"):
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
                return RustStateProxy(self._rust_mgr, state_id, self._project, python_mgr=self)
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

        Returns the RustInspectProxy after dispatch so callers can read
        back attributes the user's BP action may have mutated (value
        injection — angr-uy32). Returns ``None`` when no BP fired (reentrancy
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

    def _cb_inspect_mem_read(
        self,
        state_id: int,
        when: str,
        addr: int,
        size: int,
        value_ast,
        endness: str,
    ):
        """PyO3 callback target for mem_read events from Rust.

        Returns the possibly-mutated ``mem_read_expr`` AST when the user's
        BP action overrode it (value injection — angr-uy32); the Rust caller
        substitutes it for the loaded value. Returns ``None`` when unchanged,
        so the original load result stands.
        """
        try:
            proxy = self._dispatch_inspect_event(
                "mem_read",
                state_id,
                when,
                mem_read_address=self._addr_attr_for(addr),
                mem_read_length=size,
                mem_read_expr=value_ast,
                mem_read_endness=endness,
            )
            if proxy is not None:
                mutated = proxy.mem_read_expr
                # Identity check: only round-trip back to Rust when the user
                # actually swapped the object — avoids a needless AST->RustBV
                # conversion (and its width checks) on every untouched read.
                if mutated is not value_ast:
                    return mutated
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: user BP action errored. Log and
            # swallow so the engine keeps stepping; the user can see the
            # warning in stderr.
            l.warning("inspect mem_read dispatch failed: %s: %s", type(e).__name__, e)
        return None

    def _cb_inspect_mem_write(
        self,
        state_id: int,
        when: str,
        addr: int,
        size: int,
        value_ast,
        endness: str,
    ):
        """PyO3 callback target for mem_write events from Rust.

        Returns the possibly-mutated ``mem_write_expr`` AST when a BP_BEFORE
        action overrode it (value injection — angr-inh0); the Rust caller
        substitutes it for the stored value before commit. Returns ``None``
        when unchanged (and always for ``when='after'``, post-commit), so the
        original store value stands.
        """
        try:
            proxy = self._dispatch_inspect_event(
                "mem_write",
                state_id,
                when,
                mem_write_address=self._addr_attr_for(addr),
                mem_write_length=size,
                mem_write_expr=value_ast,
                mem_write_endness=endness,
            )
            if proxy is not None:
                mutated = proxy.mem_write_expr
                # Identity check: only round-trip back to Rust when the user
                # actually swapped the object — avoids a needless AST->RustBV
                # conversion (and its width checks) on every untouched store.
                if mutated is not value_ast:
                    return mutated
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # mem_write event is dropped (no breakpoint fired). Exception
            # type is open since handlers are user code.
            l.warning("inspect mem_write dispatch failed: %s: %s", type(e).__name__, e)
        return None

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
                "reg_read",
                state_id,
                when,
                reg_read_offset=offset,
                reg_read_length=size,
                reg_read_expr=value_ast,
                reg_read_condition=None,
                reg_read_endness=None,
            )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # reg_read event is dropped (no breakpoint fired).
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
                "reg_write",
                state_id,
                when,
                reg_write_offset=offset,
                reg_write_length=size,
                reg_write_expr=value_ast,
                reg_write_condition=None,
                reg_write_endness=None,
            )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # reg_write event is dropped (no breakpoint fired).
            l.warning("inspect reg_write dispatch failed: %s: %s", type(e).__name__, e)

    def _cb_inspect_instruction(self, state_id: int, when: str, addr: int):
        """PyO3 callback target for instruction events (one per IMark)."""
        try:
            self._dispatch_inspect_event(
                "instruction",
                state_id,
                when,
                instruction=addr,
            )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # instruction event is dropped (no breakpoint fired).
            l.warning("inspect instruction dispatch failed: %s: %s", type(e).__name__, e)

    def _cb_inspect_irsb(self, state_id: int, when: str, addr: int):
        """PyO3 callback target for irsb events (one per basic block)."""
        try:
            self._dispatch_inspect_event(
                "irsb",
                state_id,
                when,
                address=addr,
            )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # irsb event is dropped (no breakpoint fired).
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
                "exit",
                state_id,
                when,
                exit_target=self._addr_attr_for(target),
                exit_guard=guard_ast,
                exit_jumpkind=jumpkind,
            )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # exit event is dropped (no breakpoint fired).
            l.warning("inspect exit dispatch failed: %s: %s", type(e).__name__, e)

    def _cb_inspect_call(self, state_id: int, when: str, function_address: int):
        """PyO3 callback target for Ijk_Call events (function entry).

        Mirrors Python `callstack.py:386/419` — fires `when='before'` with
        the resolved call target, then `when='after'` once the Rust
        call_stack push has happened. `function_address` is wrapped in a
        word-sized BVV so user code may compare it with `state.regs._ip`
        which is also symbolic.
        """
        try:
            self._dispatch_inspect_event(
                "call",
                state_id,
                when,
                function_address=self._addr_attr_for(function_address),
            )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # call event is dropped (no breakpoint fired).
            l.warning("inspect call dispatch failed: %s: %s", type(e).__name__, e)

    def _cb_inspect_return(self, state_id: int, when: str, function_address: int):
        """PyO3 callback target for Ijk_Ret events (function exit).

        Mirrors Python `callstack.py:430/432` — fires `when='before'` with
        the func_addr of the frame being popped, then `when='after'`.
        function_address is 0 when the Rust call stack is empty (popping
        from an unentered frame), matching the int convention.
        """
        try:
            self._dispatch_inspect_event(
                "return",
                state_id,
                when,
                function_address=self._addr_attr_for(function_address),
            )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # return event is dropped (no breakpoint fired).
            l.warning("inspect return dispatch failed: %s: %s", type(e).__name__, e)

    def _cb_inspect_tmp_read(self, state_id: int, when: str, tmp_num: int, value_ast):
        """PyO3 callback target for tmp_read events (VEX `RdTmp`).

        Fired `when='after'` once the tmp slot has been read. `value_ast`
        is the claripy reconstruction of the stored RustBV.
        """
        try:
            self._dispatch_inspect_event(
                "tmp_read",
                state_id,
                when,
                tmp_read_num=tmp_num,
                tmp_read_expr=value_ast,
            )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # tmp_read event is dropped (no breakpoint fired).
            l.warning("inspect tmp_read dispatch failed: %s: %s", type(e).__name__, e)

    def _cb_inspect_tmp_write(self, state_id: int, when: str, tmp_num: int, value_ast):
        """PyO3 callback target for tmp_write events (VEX `WrTmp`).

        Fired `when='after'` with the value about to be stored into the
        tmp. Note: the slot mutation happens after this callback returns,
        so user BP_AFTER overrides are not honored — same MVP gap as the
        Python-dispatched events documented in `rust_engine.rst`.
        """
        try:
            self._dispatch_inspect_event(
                "tmp_write",
                state_id,
                when,
                tmp_write_num=tmp_num,
                tmp_write_expr=value_ast,
            )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # tmp_write event is dropped (no breakpoint fired).
            l.warning("inspect tmp_write dispatch failed: %s: %s", type(e).__name__, e)

    def _cb_inspect_statement(self, state_id: int, when: str, stmt_idx: int):
        """PyO3 callback target for statement events (per VEX IR statement).

        Fired `when='before'` from `execute_block_with_callbacks` for each
        statement in the IRSB, with `stmt_idx` (the position in
        `irsb.statements`) as the only attr. Matches Python's
        `SimInspectMixin._handle_vex_stmt` BP_BEFORE attr signature.
        The BP_AFTER mirror is not wired — same MVP scope as `instruction`.
        """
        try:
            self._dispatch_inspect_event(
                "statement",
                state_id,
                when,
                statement=stmt_idx,
            )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # statement event is dropped (no breakpoint fired).
            l.warning("inspect statement dispatch failed: %s: %s", type(e).__name__, e)

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
        try:
            self._dispatch_inspect_event(
                "expr",
                state_id,
                when,
                expr=None,
                expr_result=expr_result,
            )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # expr event is dropped (no breakpoint fired).
            l.warning("inspect expr dispatch failed: %s: %s", type(e).__name__, e)

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
        try:
            self._dispatch_inspect_event(
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
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # address_concretization event is dropped (no breakpoint fired).
            l.warning("inspect address_concretization dispatch failed: %s: %s", type(e).__name__, e)

    def _cb_inspect_fork(self, state_id: int, when: str):
        """PyO3 callback target for fork events.

        Fired `when='after'` once per forked state created by the
        deferred-fork processing in `exploration/stepping.rs`
        (`handle_block_end` + `process_deferred_forks_into`). `state_id`
        is the FORKED state's id (matching Python's
        `engines/successors.py:203` where the BP fires on the newly-added
        successor, not the parent). The fork event has no attrs in
        angr's `inspect_attributes` table; the BP just gets the per-state
        proxy via `state.inspect`. UNSAT-pruned forks still fire the BP
        (the Rust dispatch is pre-satisfiability check, matching
        Python's pre-discard fire).
        """
        try:
            self._dispatch_inspect_event("fork", state_id, when)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # fork event is dropped (no breakpoint fired).
            l.warning("inspect fork dispatch failed: %s: %s", type(e).__name__, e)

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
        try:
            self._dispatch_inspect_event(
                "symbolic_variable",
                state_id,
                when,
                symbolic_name=name,
                symbolic_size=size,
                symbolic_expr=expr_ast,
            )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # symbolic_variable event is dropped (no breakpoint fired).
            l.warning("inspect symbolic_variable dispatch failed: %s: %s", type(e).__name__, e)

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
        try:
            self._dispatch_inspect_event(
                "simprocedure",
                state_id,
                when,
                simprocedure_name=sp_name,
                simprocedure_addr=sp_addr,
                simprocedure=sp_inst,
                simprocedure_result=sp_result,
            )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # simprocedure event is dropped (no breakpoint fired).
            l.warning("inspect simprocedure dispatch failed: %s: %s", type(e).__name__, e)

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
        try:
            self._dispatch_inspect_event(
                "syscall",
                state_id,
                when,
                syscall_name=syscall_name,
                simprocedure=sp_inst,
            )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # syscall event is dropped (no breakpoint fired).
            l.warning("inspect syscall dispatch failed: %s: %s", type(e).__name__, e)

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
        action overrode it (short-circuit / value injection — angr-uy32);
        `_cb_dirty_call` substitutes it for the handler's result. Returns
        ``None`` when unchanged (or no BP fired), so the original result
        stands.
        """
        try:
            proxy = self._dispatch_inspect_event(
                "dirty",
                state_id,
                when,
                dirty_name=dirty_name,
                dirty_handler=dirty_handler,
                dirty_args=dirty_args,
                dirty_result=dirty_result,
            )
            if proxy is not None:
                mutated = proxy.dirty_result
                if mutated is not dirty_result:
                    return mutated
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # dirty event is dropped (no breakpoint fired).
            l.warning("inspect dirty dispatch failed: %s: %s", type(e).__name__, e)
        return None

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
        try:
            proxy = self._dispatch_inspect_event(
                "constraints",
                state_id,
                when,
                added_constraints=added_constraints,
            )
            if proxy is not None and when == "before":
                mutated = proxy.added_constraints
                if mutated is not added_constraints:
                    return mutated
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # constraints event is dropped (no breakpoint fired).
            l.warning("inspect constraints dispatch failed: %s: %s", type(e).__name__, e)
        return None

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
            self._dispatch_inspect_event(
                "vex_lift",
                state_id,
                when,
                vex_lift_addr=self._addr_attr_for(addr),
                vex_lift_size=size,
                vex_lift_buff=buff,
            )
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: user inspect handler raised; this
            # vex_lift event is dropped (no breakpoint fired).
            l.warning("inspect vex_lift dispatch failed: %s: %s", type(e).__name__, e)

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
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: executable region not loaded into Rust;
                    # attempts to lift a block in this region will fall through to
                    # Python via lift_block. Debug-logs.
                    name = getattr(region, "name", repr(region))
                    l.debug(f"Could not load region {name}: {e}")

        self._rust_mgr.load_binary_regions(regions)

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
        """Register SimProcedures with the Rust manager."""
        procs = []
        # display_name -> return width, for the ReturnUnconstrained stubs the
        # native registry serves itself (see _unconstrained_stub_spec).
        stubs: dict[str, int] = {}

        # Get hooked addresses from project
        if hasattr(self._project, "_sim_procedures"):
            for addr, proc in self._project._sim_procedures.items():
                name = _simproc_dispatch_name(proc)
                num_args = getattr(proc, "num_args", 0) or 0
                no_return = getattr(proc, "NO_RET", False)
                procs.append((addr, name, num_args, no_return))
                # Track this hook as registered
                self._registered_hooks.add(addr)
                spec = _unconstrained_stub_spec(proc, self._project.arch)
                if spec is not None:
                    stubs.setdefault(*spec)

        if procs:
            self._rust_mgr.register_simprocedures(procs)
        if stubs:
            self._rust_mgr.register_unconstrained_stubs(list(stubs.items()))

    # =========================================================================
    # Persistent disk cache for Python init results
    #
    # The full disk-cache subsystem — both the save path (``_disk_cache_dir``
    # / ``_disk_cache_key`` / ``_save_init_to_disk_cache`` + module-level
    # ``_extract_*`` helpers) and the load path (``_load_init_from_disk_cache``
    # / ``_load_init_pickle`` / ``_deserialize_init_state`` /
    # ``_apply_init_side_effects`` + the ``_state_has_user_symbolic`` guard) —
    # lives in the RustDiskCacheManager mixin (``rust_disk_cache.py``).
    # ``_get_cached_blank_state`` stays here because it owns the host-class
    # ``_blank_state_cache`` pool; the mixin reaches it via ``self``.
    # =========================================================================

    def _get_cached_blank_state(self, addr: int) -> angr.SimState:
        """Get a blank state, using class-level cache when possible.

        blank_state() is expensive (~1ms) due to plugin initialization.
        Caching + copy() is <0.1ms.
        """
        binary_path = getattr(self._project.loader.main_object, "binary", None) or ""
        cache_key = (binary_path, addr)
        cached = RustExplorationManager._blank_state_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        state = self._project.factory.blank_state(addr=addr)
        if (
            binary_path
            and len(RustExplorationManager._blank_state_cache) < RustExplorationManager._blank_state_cache_max
        ):
            RustExplorationManager._blank_state_cache[cache_key] = state.copy()
        return state

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

    def _run_python_init_if_needed(self, state: angr.SimState) -> angr.SimState:
        """Run initialization in Python if the state starts at a loader address.

        When a state starts at a loader/init address (e.g., from full_init_state),
        the C++ init sequence (constructors, .init_array, etc.) is too complex for
        the Rust engine. Run it in Python first, then return the state at main.
        """
        main_obj = self._project.loader.main_object
        addr = state.addr
        cache_key = getattr(main_obj, "binary", None) or ""
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
            if obj is not None and obj.binary is not None and not obj.binary.startswith("cle##"):
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

    def _apply_state_metadata(self, src_state: angr.SimState, dst_state: angr.SimState) -> None:
        """Copy constraints, globals, and LAZY_SOLVES / STRICT_PAGE_ACCESS /
        ENABLE_NX / NO_IP_CONCRETIZATION / NO_SYMBOLIC_JUMP_RESOLUTION /
        KEEP_IP_SYMBOLIC / TRACK_ACTION_HISTORY / ZERO_FILL_UNCONSTRAINED_* /
        SYMBOL_FILL_UNCONSTRAINED_* options from src to dst.

        Options are mirrored — added when src has them, removed when src
        doesn't. The remove half matters for the in-memory init cache: a
        cached state populated from a prior STRICT_PAGE_ACCESS run on the
        same binary would otherwise leak that option to a subsequent caller
        that didn't request it (and downstream `set_enforce_permissions(True)`
        would then surface spurious permission errors). Same reasoning for
        ENABLE_NX → `set_enforce_nx(True)`, NO_IP_CONCRETIZATION →
        `set_no_ip_concretization(True)`, NO_SYMBOLIC_JUMP_RESOLUTION →
        `set_no_symbolic_jump_resolution(True)`, and KEEP_IP_SYMBOLIC →
        `set_keep_ip_symbolic(True)`. TRACK_ACTION_HISTORY is mirrored
        for preconstrainer.py compatibility (angr-fkvt, 2026-06-06) — it
        is a metadata flag consulted by preconstrainer's clear/restore
        pattern and must survive the init cache round-trip so AEG
        workloads see it on the seed state.
        """
        for c in src_state.solver.constraints:
            dst_state.solver.add(c)
        if "globals" in src_state.plugins:
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
                o.TRACK_ACTION_HISTORY,
                # angr-z21g0: the unconstrained-fill policy. The init-cache
                # dst_state is a blank_state, whose default options do NOT
                # include these. Without the mirror, every Python-side load of
                # an unmapped byte during a SimProcedure callback mints a fresh
                # `mem_*` BVS instead of a zero — those symbolic bytes flow back
                # into Rust and fork-storm the exploration (xmllint/libxml2:
                # 1 state -> 60+ actives at step 27).
                o.ZERO_FILL_UNCONSTRAINED_MEMORY,
                o.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                o.SYMBOL_FILL_UNCONSTRAINED_MEMORY,
                o.SYMBOL_FILL_UNCONSTRAINED_REGISTERS,
                # angr-kzjv6: symex-relevant options that native SimProcedures
                # consult via RustSimState.has_option (e.g. SHORT_READS). Without
                # this they get stripped here before `_add_rust_state` mirrors
                # them onto the Rust state. _NATIVE_SIMOPTIONS holds the raw
                # option-name strings, which `state.options` accepts directly.
                *_NATIVE_SIMOPTIONS,
            ):
                if opt in src_state.options:
                    dst_state.options.add(opt)
                else:
                    dst_state.options.discard(opt)
        except (ImportError, AttributeError):
            # cat-(a) EXPECTED CONTROL FLOW: sim_options unavailable or one
            # of the listed option names not present in this angr build;
            # without it the option-mirror step is skipped.
            pass

    def _compute_disk_init_key(self, state: angr.SimState, cache_key: str) -> str:
        """Compute disk init cache key. Empty string means caching is disabled
        (no binary path, or state has user symbolic data that blank_state can't
        round-trip)."""
        if not cache_key:
            return ""
        if self._state_has_user_symbolic(state):
            return ""
        arch_name = getattr(self._project.arch, "name", "") or ""
        return self._disk_cache_key(cache_key, arch_name)

    def _compute_mem_init_key(self, state: angr.SimState, cache_key: str) -> str:
        """In-memory init cache key. Returns '' (caching disabled) when the
        state has user-created symbolic data, mirroring _compute_disk_init_key.
        Without this gate, a user-symbolic store on the input state survives
        through Python init and ends up in the cached post-init state. Later
        callers that hit the cache via .copy() inherit those stores while
        their own stores are silently lost — _apply_state_metadata copies
        constraints/options but not memory pages.
        """
        if not cache_key:
            return ""
        if self._state_has_user_symbolic(state):
            return ""
        return cache_key

    def _try_in_memory_init_cache(self, state: angr.SimState, cache_key: str) -> angr.SimState | None:
        """Try the per-process init cache (~180ms savings). Returns ready state or None."""
        if not cache_key or cache_key not in RustExplorationManager._init_cache:
            return None
        cached = RustExplorationManager._init_cache[cache_key]
        l.info(f"Init cache hit for {cache_key}, copying state at 0x{cached.addr:x}")
        new_state = cached.copy()
        self._apply_state_metadata(state, new_state)
        return new_state

    def _try_disk_init_cache(self, state: angr.SimState, disk_key: str) -> angr.SimState | None:
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

    def _resolve_main_address(self) -> int | None:
        """Find main function address; for stripped binaries, parse _start's PUT(rdi)."""
        main_sym = self._project.loader.find_symbol("main")
        if main_sym:
            return main_sym.rebased_addr
        try:
            entry_block = self._project.factory.block(self._project.entry)
            vex = entry_block.vex
            rdi_offset = self._project.arch.registers.get("rdi", (None,))[0]
            if rdi_offset is None:
                rdi_offset = self._project.arch.registers.get("edi", (None,))[0]
            if rdi_offset is not None:
                for stmt in reversed(vex.statements):
                    s = str(stmt)
                    if f"PUT(offset={rdi_offset})" in s or "PUT(rdi)" in s:
                        import re

                        m_const = re.search(r"0x([0-9a-fA-F]+)", s)
                        if m_const:
                            candidate = int(m_const.group(1), 16)
                            main_obj = self._project.loader.main_object
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
                    main_obj = self._project.loader.main_object
                    if (
                        main_obj.min_addr <= tgt <= main_obj.max_addr
                        and tgt != self._project.entry
                        and tgt not in self._project._sim_procedures
                    ):
                        l.info(f"Resolved main=0x{tgt:x} from entry's direct call target")
                        return tgt
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: main address extraction from _start
            # disassembly failed; caller resolves None and uses the post-init
            # state at whatever PC step_python_to_main lands on. Debug-logs.
            l.debug(f"Could not extract main from _start: {e}")
        return None

    def _save_init_state_to_caches(self, result: angr.SimState, cache_key: str, disk_key: str) -> None:
        """Persist a freshly-built init state to in-memory and disk caches."""
        if cache_key and len(RustExplorationManager._init_cache) < RustExplorationManager._init_cache_max:
            RustExplorationManager._init_cache[cache_key] = result.copy()
        if disk_key:
            self._save_init_to_disk_cache(disk_key, result)

    def _step_python_to_main(
        self, state: angr.SimState, main_addr: int | None, cache_key: str, disk_key: str, main_obj
    ) -> angr.SimState:
        """Run Python SimulationManager until reaching main, then cache+return."""
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

        for step in range(500):
            if not sm.active:
                break

            if main_addr is not None:
                at_main = [s for s in sm.active if s.addr == main_addr]
                if at_main:
                    l.info(f"Python init complete: state reached main at 0x{main_addr:x} after {step} steps")
                    result = at_main[0]
                    self._extract_continuation_data(result)
                    self._save_init_state_to_caches(result, cache_key, disk_key)
                    return result

            # No main symbol: pick the first state inside the main binary that
            # isn't at _start, isn't at a SimProcedure, and is past the prologue.
            if main_addr is None and step > 10:
                in_main = [
                    s
                    for s in sm.active
                    if main_min <= s.addr <= main_max
                    and s.addr not in init_addrs
                    and s.addr not in self._project._sim_procedures
                ]
                if in_main:
                    l.info(f"Python init complete: state at 0x{in_main[0].addr:x} after {step} steps")
                    result = in_main[0]
                    self._extract_continuation_data(result)
                    self._save_init_state_to_caches(result, cache_key, disk_key)
                    return result

            sm.step()

        # Couldn't reach main within budget — fall back to whatever we have.
        if sm.active:
            best = sm.active[0]
            self._extract_continuation_data(best)
            l.warning(f"Python init: didn't reach main after 500 steps, using state at 0x{best.addr:x}")
            return best
        if sm.deadended:
            l.warning("Python init: all states deadended")
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
            "engine cannot provide. Drop use_rust_engine=True (or fix these "
            "options on state.options) and rerun with the Python "
            "engine. See docs/advanced-topics/rust_engine.rst for the "
            "full matrix."
        )

    # Maximum concretized file size (bytes) eligible for the Rust-side
    # symbolic-content export (angr-0xyq2 Phase 3 v1 scope gate). One RustBV
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
            if not content:
                return
            asts = []
            for data, _size in content:
                if not isinstance(data, claripy.ast.BV) or data.length % 8:
                    return
                asts.extend(data.chop(8))  # chop(8)[0] is the byte at stream offset 0
            if not asts or len(asts) > self._FS_EXPORT_MAX_FILE_SIZE:
                return
            self._rust_mgr.seed_stdin_content(state_id, asts)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: stdin seeding failed; native read(0)
            # mints fresh symbols, so the harness's BVS stays unconstrained on
            # exported states (the pre-angr-mb09c behavior). Debug-logs.
            l.debug("stdin seed push failed for state %d: %s: %s", state_id, type(e).__name__, e)

    def _export_fs_files_to_rust(self, angr_state: angr.SimState, state_id: int) -> None:
        """Export eligible ``state.fs._files`` entries into the Rust
        FileSystem's path-keyed symbolic-content registry (angr-0xyq2
        Phase 3), so a native guest ``open()`` attaches the content and
        reads are served in Rust instead of bouncing to Python.

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
            self._rust_mgr.set_fs_cwd(state_id, fs.cwd.decode("utf-8"))
            files = fs._files
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
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: this file stays Python-served
                # (the pre-Phase-3 behavior). Debug-logs.
                self._stats_symfile_export_skips["error"] += 1
                l.debug("fs file export failed for %r: %s: %s", path, type(e).__name__, e)
        self._stats_symfile_exports += exported
        if exported:
            l.debug("Exported %d symbolic file(s) to Rust state %d", exported, state_id)

    def _add_rust_state(self, stash: str, angr_state: angr.SimState):
        """Add an angr state to a Rust stash.

        Note: Rust internally forks the state, so we need to get the actual
        state ID from Rust after adding to properly cache the angr state.

        Returns the resolved Rust state id (the id-diff result, or the
        Python-side fallback id when the diff was inconclusive) so callers
        like ``merge()`` can post-process the freshly added state
        (angr-qluof lineage-aware demotion).
        """
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
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: posix.brk push to Rust failed; Rust's
            # brk syscall will base-from its hardcoded default and may overlap
            # mapped memory. Debug-logs.
            l.debug("posix.brk init push failed: %s", e)

        # Push state.heap.heap_location so Rust's malloc bump allocator
        # (heap_alloc) starts past any Python-side allocation. Symmetric with
        # the posix.brk push above and the export-side
        # _sync_rust_heap_brk_to_state: a Python fallback SimProcedure that
        # mallocs bumps heap_location, and without this push a subsequent
        # native alloc in Rust would hand out an overlapping address
        # (angr-um39j). Only push a plain int that's past Rust's default base.
        try:
            py_loc = getattr(getattr(angr_state, "heap", None), "heap_location", None)
            if isinstance(py_loc, int):
                rust_state.heap_brk = py_loc
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: heap_location push to Rust failed;
            # Rust's native allocator bases from its default and may overlap a
            # Python-allocated region. Debug-logs.
            l.debug("heap.heap_location init push failed: %s", e)

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
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: ctype table-ptr push failed; the
            # native __ctype_*_loc procs will defer to Python. Debug-logs.
            l.debug("libc ctype table-ptr init push failed: %s", e)

        # Push the loader-resolved guest addresses of the getopt(3) extern
        # globals (optind/optarg/optopt) so the native getopt proc (bead
        # angr-bhk0a.2) can write the cursor/optarg/optopt back to guest
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
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: getopt extern-addr push failed; the
            # native getopt proc will defer to Python. Debug-logs.
            l.debug("getopt extern-addr init push failed: %s", e)

        # Sync registers (use precomputed dict from disk cache when available)
        _t_reg = time.perf_counter_ns()
        precomputed = self._precomputed_regs
        self._precomputed_regs = None  # Consume once
        self._sync_registers_to_rust(angr_state, rust_state, precomputed_regs=precomputed)
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
        # through forks (see SymbolicMemory::fork in native/angr/src/memory/mod.rs).
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
                for _opt in _NATIVE_SIMOPTIONS:
                    if _opt in angr_state.options:
                        rust_state.set_option(_opt, True)
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
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: symlink push failed; native
                # readlink for this link returns -1 (the pre-6.2 behavior).
                l.debug("symlink seed push failed for %r: %s", link, e)

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

            # Install constraints into the new Rust state with a single call
            # (angr-tvpk; rust-write-through-supersedes-diff-push). Two paths,
            # mutually exclusive:
            #   - Cross-manager transfer (state came from a previous
            #     RustExplorationManager and the early-reuse branch in __init__
            #     did NOT take): copy Z3 ASTs by pointer from the old solver.
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

            # Lineage-aware demotion (angr-qluof pt2): on a cross-manager
            # transfer the export above re-arms any symbolic-file path the
            # source Rust manager's lineage had demoted. Re-apply from the old
            # manager's state so the guest keeps seeing the Python fallback.
            self._reapply_demoted_paths(old_rust_mgr, old_state_id, actual_state_id)

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
                src_opts = getattr(angr_state, "options", None)
                inner = getattr(src_opts, "_options", None)
                if isinstance(inner, dict):
                    self._py_state_options[actual_state_id] = {name for name, value in inner.items() if value is True}
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: seeding _py_state_options from source
                # state failed; child uses an empty options set on first access.
                # Debug-logs.
                l.debug("seed py_state_options(sid=%d) failed: %s: %s", actual_state_id, type(e).__name__, e)
            try:
                if "globals" in getattr(angr_state, "plugins", {}):
                    self._py_state_globals[actual_state_id] = dict(angr_state.globals)
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
                        self._register_handle(
                            id(ast),
                            ast,
                            addr=addr,
                            size=ast.length // 8 if hasattr(ast, "length") else 1,
                            state_id=actual_state_id,
                        )
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
                        # Track the ORIGINAL (non-reversed) AST for identity preservation
                        self._register_handle(id(ast), ast, addr=addr, size=ast.length // 8, state_id=actual_state_id)
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
        symbolic_pages = self._extract_symbolic_pages(angr_state)
        if symbolic_pages:
            self._rust_mgr.set_state_symbolic_pages(rust_state.state_id, symbolic_pages)
            # Import symbolic regions to Rust's symbolic memory
            for addr, ast in symbolic_pages.items():
                try:
                    import_ast = claripy.Reverse(ast) if hasattr(ast, "length") and ast.length > 8 else ast
                    self._rust_mgr.import_symbolic_to_state(rust_state.state_id, addr, import_ast)
                    self._register_handle(
                        id(ast),
                        ast,
                        addr=addr,
                        size=ast.length // 8 if hasattr(ast, "length") else 1,
                        state_id=rust_state.state_id,
                    )
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
        except Exception as e:
            # Non-fatal: a failed query just means the re-added state may
            # re-arm a native write-demotion (the documented, content-harmless
            # v1 limitation).
            l.debug("get_demoted_paths(%s) failed: %s: %s", source_sid, type(e).__name__, e)
            return
        for path in demoted:
            try:
                if self._rust_mgr.demote_file_path(target_sid, path):
                    self._stats_symfile_redemotions += 1
            except Exception as e:
                l.debug("re-demote %r on state %s failed: %s: %s", path, target_sid, type(e).__name__, e)

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
        callback would never fire at all (angr-gorvf.15 — the SimProcedure
        bounces that used to chop fauxware into many iterations are gone).
        """
        cb = getattr(self, "_progress_callback", None)
        if cb is not None:
            interval = getattr(self, "_progress_interval", 100)
            last = getattr(self, "_progress_last_fired", 0)
            if steps_taken - last >= interval:
                self._progress_last_fired = steps_taken
                counts = self._rust_mgr.stash_counts()
                try:
                    cb(
                        {
                            "step_count": steps_taken,
                            "active_count": counts.get("active", 0),
                            "found_count": counts.get("found", 0),
                            "deadended_count": counts.get("deadended", 0),
                            "elapsed_seconds": time.time() - start_time,
                        }
                    )
                except Exception:
                    # cat-(b) FALLBACK WITH LOSS: progress callback raised; suppress so
                    # user code can't break exploration. The callback's view skips this
                    # tick.
                    pass

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
                active-state selection; angr-a32jl.2 prototype), 'coverage'
                (new-block-first; angr-m9fpp prototype), 'loop_head'
                (round-robin over (loop-head, callstack-class) buckets;
                angr-caplg prototype), 'directed' (CFG-distance beam search;
                angr-a32jl.4), or 'find_directed' (novelty/CFG-distance find-first
                dispatch under num_find=1; angr-lnzcu). All non-default strategies
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
        fast-forwarded counterpart from active, exactly as ``filter()`` moves
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
        for seed_addr, seed_state, sid in seeds:
            if seed_addr not in targets:
                self._preinit_seeds.append((seed_addr, seed_state, sid))
                continue
            if sid is not None:
                try:
                    if self._rust_mgr.drop_state_from_stash(sid, "active"):
                        self._state_cache.pop(sid, None)
                        self._state_roots.pop(sid, None)
                except Exception as e:
                    # cat-(b) BENIGN: the advanced state stays in active and is
                    # explored redundantly; the seed is still reported as found.
                    l.debug("preinit seed drop(sid=%d) failed: %s: %s", sid, type(e).__name__, e)
            self._add_rust_state("found", seed_state)
            routed = True
        if routed:
            self._invalidate_state_export_cache()

    def _replay_preinit_prefix_for_find(self, find_addrs: list[int], avoid_addrs: list[int]) -> None:
        """Re-run the skipped Python init prefix, matching find addresses inside it.

        ``_run_python_init_if_needed`` fast-forwards an entry-point seed to
        ``main`` in Python (often straight out of a cache, without stepping at
        all), so an address the prefix traverses — inside ``__libc_csu_init``,
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
        main_addr = self._resolve_main_address()
        routed = False

        for _seed_addr, seed_state, _sid in self._preinit_seeds:
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
            self._invalidate_state_export_cache()

    def explore(
        self,
        find: int | list | Callable | None = None,
        avoid: int | list | Callable | None = None,
        num_find: int | None = 1,
        until: Callable | None = None,
        timeout: float | None = None,
        max_steps: int | None = None,
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
            **kwargs: Additional arguments (ignored for compatibility).

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
        if num_find is None:
            num_find = sys.maxsize

        # Re-entry into Rust execution invalidates the state-export cache:
        # any previously-cached Python mirrors are about to go stale.
        self._invalidate_state_export_cache()

        # Ensure predicate attributes exist (may not be set if find/avoid not provided)
        if not hasattr(self, "_find_predicate"):
            self._find_predicate = None
        if not hasattr(self, "_avoid_predicate"):
            self._avoid_predicate = None

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
                if hasattr(state, "options") and o.LAZY_SOLVES in state.options:
                    self._rust_mgr.set_lazy_solves(True)
                    l.debug("Enabled lazy_solves from cached state options at explore() time")
                    break
        except (ImportError, AttributeError):
            # cat-(a) EXPECTED CONTROL FLOW: sim_options unavailable or
            # LAZY_SOLVES symbol absent in this angr build; lazy_solves stays
            # at the value set during construction.
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
            result = self._explore_with_predicates(num_find, until, timeout, max_steps)
        else:
            result = self._explore_with_addresses(num_find, until, timeout, max_steps)

        # angr-bdeqa: an address-based find that came up empty may be targeting
        # an address strictly inside the Python init prefix, which Rust never
        # steps through. Replay the prefix once, then release the parked seeds.
        if find is not None and not callable(find):
            if self._preinit_seeds and self._found_count() == 0:
                self._replay_preinit_prefix_for_find(self._extract_addrs(find), avoid_addrs)
            self._preinit_seeds = []
        return result

    def _explore_with_predicates(self, num_find, until, timeout, max_steps):
        """Exploration loop for callable predicates or active techniques.

        Runs in batches of 50 steps, evaluating predicates between batches.
        Callbacks are handled immediately when Rust returns need_callback events.
        """
        self._rust_mgr.set_find_needs_python(self._find_predicate is not None)
        self._rust_mgr.set_avoid_needs_python(False)
        # Keep terminal states alive so predicates can check them
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
            if self._has_technique_step_state_hooks():
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
                if self._has_technique_step_state_hooks():
                    event = self._run_with_step_state_hooks(batch_size)
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
                try:
                    self._rust_mgr.clear_stash("unconstrained")
                except (RuntimeError, KeyError):
                    # cat-(b) FALLBACK WITH LOSS: clear_stash on the unconstrained
                    # stash failed (no such stash, race with Rust); states may persist
                    # in 'unconstrained' even though save_unconstrained=False.
                    pass

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
            elif event.event_type == "errored":
                if _DBG:
                    l.debug(f"Exploration error (state deadended): {event.callback_reason}")
            elif event.event_type == "step_complete":
                self._cleanup_symbolic_pages_cache()

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

        return self

    def _maybe_phase2_eager_retry(self, num_find):
        """angr-027h: re-seed the initial states in eager-fork mode (phase 2).

        Fires at most once per explore, only for address-based finds that
        exhausted phase 1 (deferred) without reaching the target. Returns True
        when a retry was seeded (the caller should reset its step/time budget and
        continue), False otherwise.
        """
        if getattr(self, "_phase2_retried", False):
            return False
        if not getattr(self, "_explore_find_addrs", None):
            return False
        if self._found_count() >= num_find:
            return False
        seeds = getattr(self, "_initial_seed_states", None)
        if not seeds:
            return False
        self._phase2_retried = True
        l.debug(
            "angr-027h: phase 1 (deferred) exhausted without find; re-seeding %d state(s) in eager mode", len(seeds)
        )
        try:
            self._rust_mgr.set_use_deferred_forks(False)
        except AttributeError:
            # Rust extension predates set_use_deferred_forks; cannot retry.
            return False
        # Re-seed from copies so the pristine originals stay reusable.
        self._phase2_reseeding = True
        try:
            self._phase_activate([s.copy() for s in seeds])
        finally:
            self._phase2_reseeding = False
        self._rust_mgr.sync_state_index()
        # Guard against a no-op re-seed (would otherwise spin active_empty).
        if not self._rust_mgr.get_state_ids("active"):
            return False
        return True

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

    def fork_state_for_copy(self, source_state_id: int) -> int:
        """Rust-side CoW fork backing ``RustStateProxy.copy()``.

        Mints a new state by forking ``source_state_id`` into the dedicated
        ``_copies`` stash. The new state's Python-side metadata (options set,
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
        new_id = self._rust_mgr.fork_state_to_stash(source_state_id, "_copies")
        src_opts = self._py_state_options.get(source_state_id)
        if src_opts is not None:
            self._py_state_options[new_id] = set(src_opts)
        src_glb = self._py_state_globals.get(source_state_id)
        if src_glb is not None:
            self._py_state_globals[new_id] = dict(src_glb)
        tracker = getattr(self, "_stdout_tracker", None)
        if tracker is not None and source_state_id in tracker:
            tracker[new_id] = tracker[source_state_id]
        return new_id

    def drop_copy(self, copy_state_id: int) -> bool:
        """Drop a clone produced by ``fork_state_for_copy`` from ``_copies``
        (angr-yhe0).

        Removes the Rust-side state from the ``_copies`` stash, clears the
        per-state Python metadata (options set, globals dict, stdout tracker
        entry), and frees Rust-side per-state metadata via
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
            dropped = self._rust_mgr.drop_state_from_stash(copy_state_id, "_copies")
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
        tracker = getattr(self, "_stdout_tracker", None)
        if tracker is not None:
            tracker.pop(copy_state_id, None)
        try:
            self._rust_mgr.clear_state_metadata(copy_state_id)
        except Exception as e:
            # cat-(a) EXPECTED CONTROL FLOW: metadata clear is best-effort.
            l.debug("clear_state_metadata(sid=%d) after drop_copy failed: %s: %s", copy_state_id, type(e).__name__, e)
        return True

    def _set_frontier_residency(self, enabled):
        """Toggle the Rust steady-state frontier-residency engagement flag.

        No-op on older Rust builds without the pymethod (steady mode absent).
        """
        try:
            self._rust_mgr.set_parallel_frontier_residency(bool(enabled))
        except AttributeError:
            pass

    def _finalize_parallel_session(self):
        """Drain any live steady-state session's resident frontier back into the
        stashes (angr-nkoct). No-op without a live session or on older Rust
        builds."""
        try:
            self._rust_mgr.finalize_parallel_session()
        except AttributeError:
            pass

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
        try:
            avoid_set = set(self._rust_mgr.get_state_ids("avoid"))
            deadended_set = set(self._rust_mgr.get_state_ids("deadended"))
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

        In deferred-fork mode the Rust engine normally DROPS the loop-exit
        forks accumulated when a state goes unconstrained (too many symbolic
        jump targets). For a find-guided ``explore()`` that drop is desirable —
        it lets the two-phase eager retry (angr-027h) detect ``active_empty``
        and re-seed in eager mode. But a bare step-loop that bypasses
        ``explore()`` (CADET solve.py phase 3: ``while True: sm.step(); break
        if any active.addr == TARGET``) has no find target and no retry, so the
        dropped forks collapse the active stash to empty and the loop spins
        forever.

        When enabled, those forks are instead materialized eagerly and routed
        to the active stash, so the step-loop keeps progressing toward a target
        behind the symbolic loop exit. Pair with ``set_block_granular(True)`` so
        the target block is observable at a step boundary before the
        materialized subtree explodes.

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
            if self._has_technique_step_state_hooks():
                event = self._run_with_step_state_hooks(1)
            elif self._has_technique_step_hooks():
                event = self._run_with_step_hooks(1)
            else:
                event = self._rust_mgr.run(1)
            self._rust_mgr.sync_state_index()

            if event.event_type == "need_callback":
                if self._dispatch_callback(event):
                    break
                steps_taken += 1
            elif event.event_type == "active_empty":
                break
            elif event.event_type == "errored":
                l.warning(f"Step error: {event.callback_reason}")
                break
            elif event.event_type in ("step_complete", "found"):
                steps_taken += 1
                if self._active_techniques:
                    self._apply_technique_filters()
            else:
                steps_taken += 1

        return self

    def _found_count(self) -> int:
        """Fast count of found states without triggering full state export/sync."""
        count = len(self._rust_mgr.get_state_ids("found"))
        if hasattr(self, "_predicate_found") and self._predicate_found:
            count += len(self._predicate_found)
        return count

    @property
    def active(self) -> list:
        """Get states in the active stash as angr SimStates.

        For SimulationManager API compatibility, this returns full angr states.
        """
        return self._get_stash_states("active")

    @property
    def found(self) -> list:
        """Get states in the found stash as angr SimStates.

        For SimulationManager API compatibility, this returns full angr states
        that can be used with state.solver.eval(), state.posix.dumps(), etc.
        Includes states found via callable predicates.
        """
        states = self._get_stash_states("found")
        # Include states found via callable predicates that may not be in Rust stash
        if hasattr(self, "_predicate_found") and self._predicate_found:
            existing_ids = {id(s) for s in states}
            for s in self._predicate_found:
                if id(s) not in existing_ids:
                    states.append(s)
        return states

    @property
    def avoid(self) -> list:
        """Get states in the avoid stash as angr SimStates."""
        return self._get_stash_states("avoid")

    @property
    def deadended(self) -> list:
        """Get states in the deadended stash as angr SimStates."""
        return self._get_stash_states("deadended")

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
    def unconstrained(self) -> list:
        """Get states in the unconstrained stash as angr SimStates.

        These are states where a symbolic jump target (e.g., ret instruction
        with symbolic return address) had too many possible concrete values
        to enumerate and fork.
        """
        return self._get_stash_states("unconstrained")

    @property
    def pruned(self) -> list:
        """Get states in the pruned stash as angr SimStates.

        These are states that were determined to be unsatisfiable during
        exploration (e.g., both branches of a conditional were infeasible
        given the current constraints).
        """
        return self._get_stash_states("pruned")

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
            stdin_vars=getattr(self, "_stdin_vars", None),
            stdout_tracker=getattr(self, "_stdout_tracker", {}),
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

        stdin_vars = getattr(self, "_stdin_vars", None)
        stdout_tracker = getattr(self, "_stdout_tracker", {}) or {}
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
        return self._stash_proxies("found")

    def active_proxies(self) -> list:
        """Return the ``active`` stash as ``list[RustStateProxy]``. See
        :meth:`found_proxies` for when to prefer proxies over full states."""
        return self._stash_proxies("active")

    def avoid_proxies(self) -> list:
        """Return the ``avoid`` stash as ``list[RustStateProxy]``. See
        :meth:`found_proxies` for when to prefer proxies over full states."""
        return self._stash_proxies("avoid")

    def deadended_proxies(self) -> list:
        """Return the ``deadended`` stash as ``list[RustStateProxy]``. See
        :meth:`found_proxies` for when to prefer proxies over full states."""
        return self._stash_proxies("deadended")

    def unconstrained_proxies(self) -> list:
        """Return the ``unconstrained`` stash as ``list[RustStateProxy]``.
        See :meth:`found_proxies` for when to prefer proxies over full
        states."""
        return self._stash_proxies("unconstrained")

    def eval_register(self, state_id: int, name: str) -> int | None:
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

    def one_found_state(self) -> angr.SimState | None:
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
        result["callback_count"] = self._stats_callback_count
        result["ffi_crossings"] = self._stats_ffi_crossings
        result["state_creations"] = self._stats_state_creations
        result["cache_hits"] = self._stats_cache_hits
        result["cache_misses"] = self._stats_cache_misses
        result["technique_filter_calls"] = self._stats_technique_filter_calls
        result["hook_sync_calls"] = self._stats_hook_sync_calls
        result["hook_sync_skips"] = self._stats_hook_sync_skips
        result["time_in_callbacks"] = self._stats_time_in_callbacks_ns / 1e9  # seconds
        result["z3_ptr_cache_hits"] = self._z3_ptr_cache_hits
        result["z3_ptr_cache_misses"] = self._z3_ptr_cache_misses
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
        # angr-h0dv: defensive counter for Path A (rust_solver_ctx attach)
        # regressions. The other legacy constraint-sync counters were retired
        # after a 20-bench soak proved Path B was dead code.
        result["rust_ctx_missing"] = self._stats_rust_ctx_missing
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

    def _has_technique_step_hooks(self) -> bool:
        """True iff an active technique has a non-native step() hook to dispatch."""
        if not self._active_techniques:
            return False
        from angr.exploration.rust_techniques import manager_has_step_hooks

        return manager_has_step_hooks(self)

    def _run_with_step_hooks(self, batch_size):
        """Run one batch under ExplorationTechnique step() hook composition.

        Returns the captured ExplorationEvent. When the step-hook stack never
        delegates to simgr.step() (e.g. a stash-only step hook), falls back to
        a direct Rust run so the loop still makes progress.
        """
        from angr.exploration.rust_techniques import dispatch_step_with_hooks

        event = dispatch_step_with_hooks(self, batch_size)
        if event is None:
            event = self._rust_mgr.run(batch_size)
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

    def _run_with_step_state_hooks(self, batch_size):
        """Run one batch under ExplorationTechnique step_state() hook composition.

        Returns the ExplorationEvent (real, over the natively-advanced declined
        states, or a synthetic step_complete when every state was merged).
        Bumps the Veritesting dispatch/applied counters (angr-op0dn.11.7).
        """
        from angr.exploration.rust_techniques import dispatch_step_state_with_hooks

        self._stats_veritesting_dispatches += 1
        event, applied = dispatch_step_state_with_hooks(self, batch_size)
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
            return self.explore(**kwargs)

        # Bounded / step_func mode: step one at a time. step_func (used by
        # Callable for concrete_only pruning) is applied after each step,
        # matching Python SimulationManager.run() behavior. Keep terminal
        # states so they land in their stash (deadended/errored) instead of
        # being dropped, matching Python.
        self._rust_mgr.set_drop_terminal_states(False)
        try:
            stash = kwargs.pop("stash", "active")
            until = kwargs.pop("until", None)
            import itertools

            for _ in itertools.count() if n is None else range(n):
                if not self._rust_mgr.get_state_ids(stash):
                    break
                self.step(**kwargs)
                if step_func is not None:
                    step_func(self)
                if until and until(self):
                    break
        finally:
            self._rust_mgr.set_drop_terminal_states(True)
        return self

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

                    proxy = RustStateProxy(self._rust_mgr, state_id, self._project, python_mgr=self)
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

    def stash(self, filter_func=None, from_stash="active", to_stash="stashed") -> RustExplorationManager:
        """Stash some states. Alias for move() with different defaults."""
        return self.move(from_stash, to_stash, filter_func=filter_func)

    def unstash(self, filter_func=None, to_stash="active", from_stash="stashed") -> RustExplorationManager:
        """Unstash some states. Alias for move() with different defaults."""
        return self.move(from_stash, to_stash, filter_func=filter_func)

    def filter(self, stash: str = "active", filter_func=None) -> RustExplorationManager:
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

                proxy = RustStateProxy(self._rust_mgr, state_id, self._project, python_mgr=self)
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
                self._rust_mgr.move_state(state_id, stash, "pruned")
            except (RuntimeError, KeyError):
                # cat-(a) EXPECTED CONTROL FLOW: state may already have moved.
                pass

        return self

    def prune(self, stash: str = "active", filter_func=None) -> RustExplorationManager:
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
                    self._rust_mgr.move_state(state_id, stash, "pruned")
                except (RuntimeError, KeyError):
                    # cat-(a) EXPECTED CONTROL FLOW: state may already have moved.
                    pass
            return self

        return self.filter(stash=stash, filter_func=filter_func)

    def drop(self, stash: str = "active", filter_func=None) -> RustExplorationManager:
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
                self._rust_mgr.move_states(stash, "deadended", None)
        else:
            # Drop states matching predicate
            state_ids = list(self._rust_mgr.get_state_ids(stash))

            for state_id in state_ids:
                try:
                    # Try lightweight proxy first (avoids expensive full state
                    # export). Falls back to full export if the filter accesses
                    # something the proxy doesn't support. Mirrors filter().
                    from angr.exploration.rust_state_proxy import RustStateProxy

                    proxy = RustStateProxy(self._rust_mgr, state_id, self._project, python_mgr=self)
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
                            self._rust_mgr.move_state(state_id, stash, "deadended")
                        except (RuntimeError, KeyError):
                            # cat-(a) EXPECTED CONTROL FLOW: state may already have moved.
                            pass
                except Exception as e:
                    # cat-(b) FALLBACK WITH LOSS: drop filter raised; that state is
                    # left in the source stash. Debug-logs.
                    if _DBG:
                        l.debug(f"drop filter error for state {state_id}: {e}")

        return self

    def split(
        self, stash_from: str = "active", stash_to: str = "stashed", limit: int = 8, filter_func=None
    ) -> RustExplorationManager:
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
        for stash_name in ["active", "found", "avoid", "deadended", "errored", "unconstrained", "pruned", "stashed"]:
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

    def copy(self) -> RustExplorationManager:
        """Return self for SimulationManager API compatibility.

        RustExplorationManager is stateful and backed by a single Rust object,
        so a true deep copy isn't possible. Return self to satisfy callers like
        angr.callable that store a reference to the manager.
        """
        return self

    def _merge_native(self, stash: str, state_ids: list[int]) -> bool:
        """Merge same-pc native states in-Rust, no export round trip (M3-4).

        Groups ``state_ids`` by program counter (the default ``merge_key``,
        ``s.addr``) using the lightweight ``get_state_pc_by_id`` accessor, then
        calls ``merge_states`` per multi-member group. The Rust ``merge_states``
        forks + merges the source states and pushes the result to ``stash`` but
        does NOT remove the sources, so we drop them afterwards (matching the
        Python path's ``_merge_drop`` dance).

        The native path never exports state, so the angr-qluof re-demotion dance
        is unnecessary: the merged state inherits its base's demoted
        symbolic-file paths directly (no ``_export_fs_files_to_rust`` re-arms
        them). ``states_merged_native`` is surfaced Rust-side by ``merge_states``.

        Returns:
            True if the native path handled the merge (caller returns), False if
            a pc lookup failed and the caller should fall back to the Python
            export path.
        """
        # Group by pc; a missing pc means we cannot group natively -> fall back.
        groups: dict[int, list[int]] = {}
        for sid in state_ids:
            pc = self._rust_mgr.get_state_pc_by_id(sid)
            if pc is None:
                return False
            groups.setdefault(pc, []).append(sid)

        # Nothing reconverges -> no merge needed, but we still handled it.
        if all(len(g) <= 1 for g in groups.values()):
            return True

        merged_any = False
        for group in groups.values():
            if len(group) <= 1:
                # Singleton stays in the stash untouched.
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
                try:
                    self._rust_mgr.move_state(sid, stash, "_merge_drop")
                except (RuntimeError, KeyError):
                    # cat-(a) EXPECTED CONTROL FLOW: source already moved.
                    pass
        if merged_any:
            try:
                self._rust_mgr.clear_stash("_merge_drop")
            except (RuntimeError, KeyError):
                # cat-(a) EXPECTED CONTROL FLOW: intermediate stash never created.
                pass
        return True

    def merge(
        self, stash: str = "active", merge_func=None, merge_key=None, prune=True, **kwargs
    ) -> RustExplorationManager:
        """Merge states in a stash.

        Exports states to Python, performs merge via claripy, then replaces
        the stash with merged states. Falls back to keeping all states
        unmerged if merge fails.
        """
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
            if self._merge_native(stash, state_ids):
                return self

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
                    py_states.append(py_state)
                    sid_by_state_key[id(py_state)] = sid
                    try:
                        demoted_by_sid[sid] = self._rust_mgr.get_demoted_paths(sid)
                    except Exception as e:
                        # Non-fatal: a missing demoted-path query just means the
                        # merged state may re-arm a native write-demotion (the
                        # documented, content-harmless v1 limitation).
                        l.debug("get_demoted_paths(%d) failed: %s: %s", sid, type(e).__name__, e)

            if len(py_states) <= 1:
                return self

            # Group by merge key (default: PC)
            if merge_key is None:
                merge_key = lambda s: s.addr

            groups = {}
            for s in py_states:
                key = merge_key(s)
                groups.setdefault(key, []).append(s)

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

            for key, group in groups.items():
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
                    except (AttributeError, TypeError, ValueError):
                        # cat-(b) FALLBACK WITH LOSS: built-in state.merge() failed;
                        # keep the group unmerged. Already warns.
                        l.warning("State merge failed for group at %s, keeping unmerged", key)
                        merged.extend((s, _group_demoted([s])) for s in group)

            # Clear the Rust stash and re-add merged states
            for sid in state_ids:
                try:
                    self._rust_mgr.move_state(sid, stash, "_merge_drop")
                except (RuntimeError, KeyError):
                    # cat-(a) EXPECTED CONTROL FLOW: source stash entry already moved.
                    pass
            try:
                self._rust_mgr.clear_stash("_merge_drop")
            except (RuntimeError, KeyError):
                # cat-(a) EXPECTED CONTROL FLOW: clear of intermediate _merge_drop
                # stash failed (already empty / never created).
                pass

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
                            except Exception as e:
                                l.debug(
                                    "re-demote %r on state %s failed: %s: %s",
                                    path,
                                    new_sid,
                                    type(e).__name__,
                                    e,
                                )
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
        deferred-fork subtree below it is lost (angr-op0dn.13.14 — a serial
        resume drained 4-6 of 8 leaves where a live run drains 8).
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
        """Restore a stash-manager snapshot from ``path`` (opt-in,
        angr-x04s.1.4).

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
        self._invalidate_state_export_cache()
        # angr-op0dn.13.12: the restored frontier — not the state this manager
        # was constructed with — is now the exploration's root. The angr-027h
        # phase-2 eager retry re-seeds `_initial_seed_states`, i.e. the caller's
        # pre-load placeholder state (`load_from_disk` passes none at all), which
        # would restart the exploration from a path the user never asked to
        # resume and additionally flip `use_deferred_forks` off globally (on the
        # parallel path that drops the stored branch conditions and forks
        # unconstrained). The mid-path frontier cannot be replayed eagerly from
        # its own states, so phase 2 is simply disabled for a resumed manager.
        self._initial_seed_states = None
        self._phase2_retried = True

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
                — there is no cross-check today (binary-hash field is a
                planned ``angr-x04s.2`` follow-up).
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
            if getattr(self, "_clear_caches_on_cleanup", False):
                self.cleanup()
        except Exception:
            pass

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

        try:
            return self._rust_mgr.get_state_ids(name)
        except (RuntimeError, KeyError):
            # cat-(a) EXPECTED CONTROL FLOW: stash name unknown to Rust;
            # raise AttributeError so getattr-style probes return the default.
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
