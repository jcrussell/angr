"""The Rust engine as a :class:`angr.sim_manager.SimulationManager` stepping backend (angr-mx00p.136).

:class:`RustBackend` decides per state and per step whether the Rust engine
steps a state (:meth:`RustBackend.claims`); a declined state runs on the
Python engine. A claimed state is stepped one block in a private
:class:`~angr.exploration.rust_manager.RustExplorationManager`
(:meth:`RustBackend.step_state`), and its successors come back as exported
SimStates. When nothing hooks the manager's ``step_state``, all the states
it claims in one step go through one :meth:`RustBackend.step_states` call
(angr-mx00p.136.13). An address-only ``explore()`` whose states it would all
claim runs to its end in the private manager's round-based explore instead
(:meth:`RustBackend.try_run`, angr-mx00p.136.14). Between steps every Rust state the backend handed out waits in one
private stash, kept only while its export is referenced: once that export is
collected (dropped from every stash and not held), the state is dropped before
the next step (angr-mx00p.136.9).
"""

from __future__ import annotations

import logging
import sys
import weakref
from collections import Counter
from typing import TYPE_CHECKING

from angr.engines import UberEngine
from angr.errors import SimEngineError, SimError, SimMemoryError
from angr.exploration.rust_manager import (
    _STASH_ACCESSORS,
    RustErrorRecord,
    RustExplorationManager,
    _unsupported_inspect_events,
    is_python_init_seed,
    resolve_main_address,
    rust_state_option_refusals,
    rust_supports_arch,
    sim_error_for_rust_message,
)
from angr.exploration.rust_state_binding import binding_of
from angr.exploration.rust_techniques import _NATIVE_STEP_TECH_NAMES
from angr.sim_manager import ErrorRecord
from angr.sim_manager_backend import SimulationBackend

if TYPE_CHECKING:
    from angr.sim_manager import SimulationManager
    from angr.sim_state import SimState
    from angr.state_plugins import SimStateEdgeHitmap

l = logging.getLogger(name=__name__)

# The private Rust stash every state the backend holds between steps waits in, so a step's
# _step_ids_once parks nothing but the stepped state's own children.
_PARKED = "_simgr_backend_parked"

# The run argument a claimed step honors; any other declines the state.
_STOP_POINTS = "extra_stop_points"

# Python init prefix: a lineage seeded at an init seed runs on Python until it reaches main.
_IN_INIT = "init"
_PAST_INIT = "past-init"

# Ownership of a lineage the backend has claimed. A lineage handed back to Python stays there
# (owner decision on angr-mx00p.136.1).
_OWNED = "owned"
_HANDED_BACK = "handed-back"

# With no main symbol, a state counts as at main once it is this deep inside the main object,
# matching RustExplorationManager._step_python_to_main's at_main heuristic.
_NO_MAIN_MIN_DEPTH = 10


class _LineageMarks:
    """A mark per state lineage, inherited by every descendant.

    A mark is set on a state's history node; a state reads the mark of its
    nearest marked ancestor node. Each lookup caches its answer on every node
    it walked, so stepping a lineage costs one parent hop per step instead of a
    walk to the root.
    """

    def __init__(self):
        self._marks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def get(self, state: SimState) -> str | None:
        walked = []
        node = state.history
        mark = None
        while node is not None:
            if node in self._marks:
                mark = self._marks[node]
                break
            walked.append(node)
            node = node.parent
        for n in walked:
            self._marks[n] = mark
        return mark

    def set(self, state: SimState, mark: str) -> None:
        self._marks[state.history] = mark


def _queue_release(backend_ref, state_id: int) -> None:
    backend = backend_ref()
    if backend is not None:
        backend._export_finalizers.pop(state_id, None)
        backend._released.append(state_id)


class RustBackend(SimulationBackend):
    """Steps the states the Rust engine supports, one block per step.

    :meth:`claims` declines a state, naming the reason, when its arch is not
    implemented in Rust, its options or inspect breakpoints are refused, the
    step passes a ``successor_func`` or other run arguments, a technique on
    the manager overrides ``successors()`` and is not natively handled, the
    project uses a custom default engine, the state is still in the Python
    init prefix, or its lineage was handed back to Python. Each answer is
    counted in :attr:`stats` and debug-logged.
    """

    def __init__(
        self,
        project,
        *,
        max_active_states: int | None = None,
        parallel_workers: int = 1,
        deterministic: bool = False,
        solver_timeout_ms: int = 30000,
        max_history: int = 1000,
        use_native_lift: bool = True,
        prefer_native_library_hooks: bool | None = None,
        symlinks: dict | None = None,
        save_unconstrained: bool = True,
    ):
        """The keyword arguments configure the private manager as they do a
        :class:`~angr.exploration.rust_manager.RustExplorationManager`, except that
        ``max_active_states`` defaults to no cap: between steps that active stash
        only ever holds one step's children, and the Python engine has none.
        """
        self._project = project
        self._manager_kwargs = {
            "max_active_states": max_active_states,
            "parallel_workers": parallel_workers,
            "deterministic": deterministic,
            "solver_timeout_ms": solver_timeout_ms,
            "max_history": max_history,
            "use_native_lift": use_native_lift,
            "prefer_native_library_hooks": prefer_native_library_hooks,
            "symlinks": symlinks,
            "save_unconstrained": save_unconstrained,
        }
        # The manager whose stashes the stash-scoped members (`<stash>_proxies`, `coverage_*`)
        # read: the last one that asked this backend anything.
        self._simgr: weakref.ref | None = None
        self._main_addr_resolved = False
        self._main_addr: int | None = None
        self._phase = _LineageMarks()
        self._ownership = _LineageMarks()
        self._counters = {"claimed": 0, "declined": Counter(), "handed_back": 0, "fast_path_runs": 0}
        self._mgr: RustExplorationManager | None = None
        # Ids of the Rust states waiting in _PARKED.
        self._resident: set[int] = set()
        # The finalizer of the export each resident id was handed out as; it queues the id in
        # _released when that export is collected.
        self._export_finalizers: dict[int, weakref.finalize] = {}
        # Resident ids whose export is gone, dropped from _PARKED before the next step. A
        # finalizer can run inside a Rust call (a callback's GC), so it only queues the id.
        self._released: list[int] = []
        # The imported roots each resident id descends from (several when a native run mixed
        # lineages), and the resident ids per root: a root's `_lineage_templates` entry is
        # dropped with its last resident id.
        self._root_of: dict[int, tuple[int, ...]] = {}
        self._lineage_refs: Counter[int] = Counter()
        # Python-lifted block size per address, for the extra_stop_points check.
        self._block_sizes: dict[int, int] = {}

    def claims(self, simgr: SimulationManager, state: SimState, successor_func, run_args: dict) -> tuple[bool, str]:
        self._simgr = weakref.ref(simgr)
        reason = self.decline_reason(simgr, state, successor_func, run_args)
        if reason is None:
            self._ownership.set(state, _OWNED)
            self._counters["claimed"] += 1
            return True, ""
        if self._ownership.get(state) == _OWNED:
            self._ownership.set(state, _HANDED_BACK)
            self._counters["handed_back"] += 1
        self._counters["declined"][reason] += 1
        l.debug("simgr backend: python (%s)", reason)
        return False, reason

    def decline_reason(self, simgr: SimulationManager, state: SimState, successor_func, run_args: dict) -> str | None:
        """Why the Rust engine cannot step *state* this step, or None when it can."""
        if self._ownership.get(state) == _HANDED_BACK:
            return "lineage was handed back to python"
        arch_name = state.arch.name
        if not rust_supports_arch(arch_name):
            return f"arch {arch_name} not implemented by the rust engine"
        refused = rust_state_option_refusals(state.options)
        if refused:
            return "state options not honored by the rust engine: " + ", ".join(refused)
        events = _unsupported_inspect_events(state)
        if events:
            return "state.inspect events not dispatched by the rust engine: " + ", ".join(events)
        if successor_func is not None:
            return "step passes a successor_func"
        unhonored = sorted(k for k in run_args if k != _STOP_POINTS)
        if unhonored:
            return "step passes run arguments: " + ", ".join(unhonored)
        hooks = sorted(
            type(tech).__name__
            for tech in simgr._techniques
            if tech._is_overridden("successors") and type(tech).__name__ not in _NATIVE_STEP_TECH_NAMES
        )
        if hooks:
            return "techniques override successors(): " + ", ".join(hooks)
        engine = self._project.factory.default_engine_factory
        if engine is not UberEngine:
            return f"project uses a custom SimEngine {engine.__name__}"
        if self._in_init_prefix(state):
            return "state is in the python init prefix"
        if self._stop_point_inside_block(state, run_args.get(_STOP_POINTS)):
            return "an extra_stop_points address lies inside the block"
        return None

    def _stop_point_inside_block(self, state: SimState, stops) -> bool:
        """True when a stop point would end the Python engine's block early.

        Rust stops only at block boundaries, which a block-granular step does
        anyway, so a stop point is honored unless it lies strictly inside the
        block at the state's pc: there Python splits the block and Rust cannot.
        """
        if not stops:
            return False
        addr = state.addr
        if addr in self._project._sim_procedures:
            return False
        size = self._block_sizes.get(addr)
        if size is None:
            try:
                size = self._project.factory.block(addr).size
            except (SimEngineError, SimMemoryError):
                # SILENT(cat-a): an unliftable pc has no interior for a stop point to split; the step
                # itself reports the lift failure.
                size = 0
            self._block_sizes[addr] = size
        return any(addr < stop < addr + size for stop in stops)

    def step_state(self, simgr: SimulationManager, state: SimState, error_list: list, **run_args) -> dict:
        """Step *state* one block in Rust and return its successors as exported SimStates.

        ``extra_stop_points`` needs no work here: :meth:`claims` declined the
        state if one lies inside its block. A step whose only outcome is the
        state itself deadending returns no successors, so
        :meth:`SimulationManager.step` deadends the source as it does for the
        Python engine. When the state itself errors, its SimError
        (:func:`~angr.exploration.rust_manager.sim_error_for_rust_message`)
        is raised, so :meth:`SimulationManager.step_state`'s resilience
        decides whether it is recorded or propagates; an errored fork of it
        becomes an :class:`ErrorRecord` in *error_list*, and so propagates
        too when the resilience does not cover it.
        """
        (outcome,) = self.step_states(simgr, [state], **run_args)
        return outcome(error_list)

    def step_states(self, simgr: SimulationManager, states: list[SimState], **run_args) -> list:
        """Step every state of *states* one block in one ``_step_ids_once`` call (angr-mx00p.136.13).

        Each state's successors are exported here; its callable replays what
        :meth:`step_state` gives for it alone: it appends the errored forks'
        records, then returns the successors or raises the state's own error.
        """
        mgr = self._manager()
        rust = mgr._rust_mgr
        sids = self._activate(mgr, states)
        children = {}
        try:
            children = mgr._step_ids_once(sids)
        finally:
            # Python owns stash membership, so every child waits in _PARKED whatever its Rust
            # stash; one that is not handed out below is dropped.
            messages = {err_id: message for _, message, err_id in rust.get_errors()}
            rust.clear_errors()
            rust.move_states("active", _PARKED, None)
            unexported = set(sids)
            for sid in sids:
                roots = self._root_of[sid]
                for name, cids in children.get(sid, {}).items():
                    for cid in cids:
                        if name != "active":
                            rust.move_state(cid, name, _PARKED)
                        self._adopt(cid, roots)
                    unexported.update(cids)
            self._resident.update(unexported)
        try:
            return [self._outcome(mgr, simgr, sid, children.get(sid, {}), messages, unexported) for sid in sids]
        finally:
            for cid in unexported:
                self._release(mgr, cid)

    def try_run(self, simgr: SimulationManager, stash: str, n: int | None, until, run_args: dict) -> bool:
        """Run an address-only ``explore()`` to its end in one native run (angr-mx00p.136.14).

        Taken when the run has no ``n``, ``until`` or step arguments, the
        manager's only techniques are one :class:`Explorer` whose find and
        avoid are addresses (no ``cfg``, no ``avoid_priority``) and the
        default Suggestions, and every state of *stash* would be claimed this
        step. The private manager's ``explore`` runs them, then every state it
        holds is exported into the manager's stashes. Suggestions is treated
        as transparent: it only reports errored states. Counted in
        ``stats["fast_path_runs"]``.
        """
        self._simgr = weakref.ref(simgr)
        explorer = self._fast_path_explorer(simgr, stash, n, until, run_args)
        states = simgr._stashes[stash]
        if explorer is None or any(self.decline_reason(simgr, s, None, {}) is not None for s in states):
            return False
        for state in states:
            self._ownership.set(state, _OWNED)
        self._explore_native(simgr, stash, states, explorer)
        self._counters["fast_path_runs"] += 1
        return True

    @staticmethod
    def _fast_path_explorer(simgr: SimulationManager, stash: str, n, until, run_args: dict):
        """The run's Explorer when the fast path can stand in for it, else None."""
        from angr.exploration_techniques import Explorer, Suggestions
        from angr.sim_manager import SimulationManager

        if n is not None or until is not None or run_args:
            return None
        cls = type(simgr)
        if any(
            getattr(cls, m) is not getattr(SimulationManager, m) for m in ("step", "step_state", "filter", "selector")
        ):
            return None
        explorers = [t for t in simgr._techniques if type(t) is Explorer]
        if len(explorers) != 1 or any(type(t) not in (Explorer, Suggestions) for t in simgr._techniques):
            return None
        (explorer,) = explorers
        static_find = getattr(explorer, "static_find", None)
        static_avoid = getattr(explorer, "static_avoid", None)
        if static_find is None or static_avoid is None or explorer.cfg is not None or explorer.avoid_priority:
            return None
        if len({stash, explorer.find_stash, explorer.avoid_stash}) != 3:
            return None
        return explorer

    def _explore_native(self, simgr: SimulationManager, stash: str, states: list[SimState], explorer) -> None:
        """Explore *states* in the private manager and export every state it ends with into *simgr*."""
        mgr = self._manager()
        rust = mgr._rust_mgr
        sids = self._activate(mgr, states)
        roots = tuple(sorted({root for sid in sids for root in self._root_of[sid]}))
        num_find = explorer.num_find - len(simgr._stashes.get(explorer.find_stash, ()))
        drop_terminal = rust.stats()["drop_terminal_states"]
        rust.set_drop_terminal_states(False)
        try:
            mgr.explore(
                find=sorted(explorer.static_find) or None,
                avoid=sorted(explorer.static_avoid) or None,
                num_find=max(num_find, 1),
                # In rounds, one native block step of every active state each, so the run stops on
                # the same step, with the same frontier, as SimulationManager's own loop.
                n=sys.maxsize,
            )
        finally:
            rust.set_drop_terminal_states(drop_terminal)
            messages = {err_id: message for _, message, err_id in rust.get_errors()}
            rust.clear_errors()
            ended = {}
            for name in rust.stash_counts():
                # Underscored stashes are the manager's own (_PARKED, the held sources' forks, the
                # step-round queues), never states this run ended with.
                if not name.startswith("_"):
                    ended[name] = list(rust.get_state_ids(name))
                    rust.move_states(name, _PARKED, None)
            for cids in ended.values():
                for cid in cids:
                    self._adopt(cid, roots)
                    self._resident.add(cid)
            ended_ids = {cid for cids in ended.values() for cid in cids}
            for sid in sids:
                if sid not in ended_ids:
                    self._unref(sid)
        targets = {"active": stash, "found": explorer.find_stash, "avoid": explorer.avoid_stash}
        simgr._stashes[stash] = []
        uncovered = None
        for name, cids in ended.items():
            if name == "errored":
                for cid in cids:
                    message = messages.get(cid, "rust engine error")
                    error = sim_error_for_rust_message(message, RustErrorRecord._classify(message))
                    simgr._errored.append(ErrorRecord(self._export(mgr, cid), error, None))
                    if uncovered is None and not isinstance(error, simgr._resilience):
                        uncovered = error
                continue
            exports = [self._export(mgr, cid) for cid in cids]
            if exports or name in targets:
                simgr._store_states(targets.get(name, name), exports)
        if uncovered is not None:
            raise uncovered

    def _outcome(self, mgr, simgr, sid, children, messages, unexported):
        """The callable :meth:`step_states` returns for *sid*: export its successors now, replay them later."""
        records = []
        error = None
        stashes = {}
        try:
            stashes = self._successors(mgr, simgr, sid, children, messages, records, unexported)
        except SimError as e:
            error = e

        def replay(error_list: list) -> dict:
            error_list.extend(records)
            if error is not None:
                raise error
            return stashes

        return replay

    def _successors(self, mgr, simgr, sid, children, messages, error_list, unexported) -> dict:
        """Export the children of *sid*'s step, discarding from *unexported* each id handed out."""
        if children == {"deadended": [sid]}:
            return {}
        stashes = {}
        for name, cids in children.items():
            if name == "errored":
                for cid in sorted(cids, key=lambda c: c != sid):
                    message = messages.get(cid, "rust engine error")
                    error = sim_error_for_rust_message(message, RustErrorRecord._classify(message))
                    if cid == sid or not isinstance(error, simgr._resilience):
                        raise error
                    error_list.append(ErrorRecord(self._export(mgr, cid), error, None))
                    unexported.discard(cid)
                continue
            stashes[None if name == "active" else name] = [self._export(mgr, cid) for cid in cids]
            unexported.difference_update(cids)
        return stashes

    def _release(self, mgr: RustExplorationManager, sid: int) -> None:
        """Drop resident *sid* from ``_PARKED`` with its per-state metadata."""
        if sid in self._resident:
            self._resident.discard(sid)
            mgr.drop_copy(sid, _PARKED)
            self._unref(sid)

    def _adopt(self, sid: int, roots: tuple[int, ...]) -> None:
        """Count *sid*, a new id of a step, as descending from the imported *roots*."""
        if sid not in self._root_of:
            self._root_of[sid] = roots
            self._lineage_refs.update(roots)

    def _unref(self, sid: int) -> None:
        """Forget *sid*'s roots, dropping a root's lineage template with its last id."""
        for root in self._root_of.pop(sid):
            self._lineage_refs[root] -= 1
            if not self._lineage_refs[root]:
                del self._lineage_refs[root]
                self._mgr._lineage_templates.pop(root, None)

    def _activate(self, mgr: RustExplorationManager, states: list[SimState]) -> list[int]:
        """Move *states* from ``_PARKED`` to the private manager's active stash, returning their ids."""
        self._drop_released(mgr)
        sids = [self._resident_id(mgr, state) for state in states]
        for state, sid in zip(states, sids):
            # The source is consumed: the step's children are exported anew, and a held source
            # reads its own fork from here on.
            finalizer = self._export_finalizers.pop(sid, None)
            if finalizer is not None:
                finalizer.detach()
            self._detach_source(mgr, state, sid)
        mgr._sync_hooks_before_step()
        for sid in sids:
            mgr._rust_mgr.move_state(sid, _PARKED, "active")
            self._resident.discard(sid)
        return sids

    def _drop_released(self, mgr: RustExplorationManager) -> None:
        """Drop the resident ids whose export was collected since the last step."""
        released, self._released = self._released, []
        for sid in released:
            self._release(mgr, sid)

    def _manager(self) -> RustExplorationManager:
        """The private Rust manager, built on first use with no states and the constructor's knobs."""
        if self._mgr is None:
            self._mgr = RustExplorationManager(self._project, **self._manager_kwargs)
        return self._mgr

    # The Rust-only API (angr-mx00p.136.16). A RustExplorationManager's `backend` is the manager
    # itself, so `mgr.backend.<member>` reads the same on either kind of manager.

    @property
    def backend(self) -> RustBackend:
        """This backend, as :attr:`RustExplorationManager.backend` is that manager."""
        return self

    @property
    def stats(self) -> dict:
        """The private manager's :attr:`RustExplorationManager.stats` plus this backend's own
        counters: ``claimed``, ``declined`` (reason -> count), ``handed_back`` and
        ``fast_path_runs``."""
        result = dict(self._mgr.stats) if self._mgr is not None else {}
        result.update(self._counters)
        return result

    def perf_report(self) -> str:
        """:meth:`RustExplorationManager.perf_report` of the private manager."""
        return self._manager().perf_report()

    def get_solver_stats(self) -> dict:
        """:meth:`RustExplorationManager.get_solver_stats` of the private manager."""
        return self._manager().get_solver_stats()

    def set_exploration_strategy(self, *args, **kwargs):
        """:meth:`RustExplorationManager.set_exploration_strategy` on the private manager."""
        return self._manager().set_exploration_strategy(*args, **kwargs)

    def register_uniqueness_filter(self, register_names: list[str]):
        """:meth:`RustExplorationManager.register_uniqueness_filter` on the private manager."""
        return self._manager().register_uniqueness_filter(register_names)

    def disable_uniqueness_filter(self):
        """:meth:`RustExplorationManager.disable_uniqueness_filter` on the private manager."""
        return self._manager().disable_uniqueness_filter()

    def uniqueness_filter_enabled(self) -> bool:
        """:meth:`RustExplorationManager.uniqueness_filter_enabled` of the private manager."""
        return self._manager().uniqueness_filter_enabled()

    def uniqueness_set_size(self) -> int:
        """:meth:`RustExplorationManager.uniqueness_set_size` of the private manager."""
        return self._manager().uniqueness_set_size()

    def coverage_blocks(self, stashes: tuple[str, ...] = ("found",)) -> list[int]:
        """:meth:`RustExplorationManager.coverage_blocks` over the Rust states in the manager's *stashes*."""
        return self._manager()._coverage_blocks_of(self._stash_ids(stashes))

    def coverage_report(self, stashes: tuple[str, ...] = ("found",)) -> SimStateEdgeHitmap:
        """:meth:`RustExplorationManager.coverage_report` over the Rust states in the manager's *stashes*."""
        return self._manager()._coverage_report_of(self._stash_ids(stashes))

    def _stash_proxies(self, stash: str) -> list:
        """A ``RustStateProxy`` per Rust state in the manager's *stash*."""
        return self._manager()._proxies_of(self._stash_ids((stash,)))

    def _stash_ids(self, stashes) -> list[int]:
        """The ids of the Rust states in *stashes* of the manager this backend last served.

        A state stepped on Python, or not yet stepped, has no Rust id and is
        left out; so is everything before the backend has served a manager.
        """
        simgr = self._simgr() if self._simgr is not None else None
        if simgr is None or self._mgr is None:
            return []
        ids = (self._parked_id(self._mgr, state) for stash in stashes for state in simgr._stashes.get(stash, ()))
        return [sid for sid in ids if sid is not None]

    def _parked_id(self, mgr: RustExplorationManager, state: SimState) -> int | None:
        """The id of *state* in ``_PARKED`` when it is a root export of a parked state, else None."""
        binding = binding_of(state)
        if (
            binding is not None
            and getattr(binding.owner, "manager", None) is mgr
            and not binding.is_virtual
            and binding._released is None
            and binding.raw_id in self._resident
        ):
            return binding.raw_id
        return None

    def _resident_id(self, mgr: RustExplorationManager, state: SimState) -> int:
        """The id of *state* in ``_PARKED``, importing it first when it is not a view of a parked state.

        A root export of a parked state is that state; anything else (a
        foreign SimState, a copy of an export, an export of another manager)
        is imported as the constructor imports a seed.
        """
        sid = self._parked_id(mgr, state)
        if sid is not None:
            return sid
        rust = mgr._rust_mgr
        mgr._phase_activate([state])
        (sid,) = rust.get_state_ids("active")
        rust.move_states("active", _PARKED, None)
        self._resident.add(sid)
        # The import's cached mirror is its lineage's plugin template, kept out of
        # `_state_cache` so it is never handed out as an export of sid.
        mgr._lineage_templates[sid] = mgr._state_cache.pop(sid)
        self._adopt(sid, (sid,))
        return sid

    def _detach_source(self, mgr: RustExplorationManager, state: SimState, sid: int) -> None:
        """Move *state*, a root export of *sid*, onto a fork of its own before *sid* steps.

        A non-forking step keeps the source's id, so without this a SimState
        the caller holds would follow the stepped state; on the Python engine
        it keeps showing the pre-step state (angr-mx00p.136.7). Its copies
        follow it, as they are views of it. The fork is dropped with *state*.
        An imported state is not bound to *sid* and needs nothing.
        """
        binding = binding_of(state)
        owner = mgr._copy_stash_fork_owner
        if binding is None or binding.owner is not owner or binding.is_virtual or binding.raw_id != sid:
            return
        cache = mgr._state_cache
        for key in [k for k, v in cache.items() if v is state]:
            del cache[key]
        owner.detach(binding)

    def _export(self, mgr: RustExplorationManager, state_id: int) -> SimState:
        """A new stash-mirror SimState of *state_id*, owned by the backend.

        The cached mirror is dropped first so the step's source, which keeps
        the same id when it does not fork, is never returned as its own
        successor.
        """
        mgr._state_cache.pop(state_id, None)
        export = mgr._materialize_single_state(state_id)
        # Not cached: the export alone keeps the Rust state, which is released with it.
        mgr._state_cache.pop(state_id, None)
        self._export_finalizers[state_id] = weakref.finalize(
            binding_of(export), _queue_release, weakref.ref(self), state_id
        )
        self._phase.set(export, _PAST_INIT)
        self._ownership.set(export, _OWNED)
        return export

    def _in_init_prefix(self, state: SimState) -> bool:
        """True while a lineage seeded at an init seed has not reached main.

        The init seeds are the ones :func:`is_python_init_seed` names; the
        loader's own SimProcedures run on Python wherever they appear.
        """
        addr = state.addr
        proc = self._project._sim_procedures.get(addr)
        if proc is not None and type(proc).__module__.startswith("angr.procedures.linux_loader"):
            return True
        phase = self._phase.get(state)
        if phase is None:
            phase = _IN_INIT if is_python_init_seed(self._project, addr) else _PAST_INIT
            self._phase.set(state, phase)
        if phase == _PAST_INIT:
            return False
        if self._at_main(state):
            self._phase.set(state, _PAST_INIT)
            return False
        return True

    def _at_main(self, state: SimState) -> bool:
        if not self._main_addr_resolved:
            self._main_addr = resolve_main_address(self._project)
            self._main_addr_resolved = True
        addr = state.addr
        if self._main_addr is not None:
            return addr == self._main_addr
        main_obj = self._project.loader.main_object
        init_addrs = {obj.entry for obj in self._project.loader.all_objects if getattr(obj, "entry", None)}
        return (
            state.history.depth > _NO_MAIN_MIN_DEPTH
            and main_obj.min_addr <= addr <= main_obj.max_addr
            and addr not in init_addrs
            and addr not in self._project._sim_procedures
        )


def _install_stash_proxies(cls):
    """Give *cls* a ``<stash>_proxies()`` per stash RustExplorationManager has one for."""
    for stash, (_doc, _emit_property, emit_proxies) in _STASH_ACCESSORS.items():
        if not emit_proxies:
            continue

        def _proxies(self, _stash=stash):
            return self._stash_proxies(_stash)

        _proxies.__name__ = f"{stash}_proxies"
        _proxies.__qualname__ = f"{cls.__qualname__}.{stash}_proxies"
        _proxies.__doc__ = f"A ``RustStateProxy`` per Rust state in the manager's ``{stash}`` stash."
        setattr(cls, _proxies.__name__, _proxies)


_install_stash_proxies(RustBackend)
