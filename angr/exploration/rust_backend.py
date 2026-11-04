"""The Rust engine as a :class:`angr.sim_manager.SimulationManager` stepping backend (angr-mx00p.136).

:class:`RustBackend` decides per state and per step whether the Rust engine
steps a state (:meth:`RustBackend.claims`); a declined state runs on the
Python engine. Stepping itself is not wired yet, so every state is declined,
an eligible one with the reason ``"stepping not implemented"``.
"""

from __future__ import annotations

import logging
import weakref
from collections import Counter
from typing import TYPE_CHECKING

from angr.engines import UberEngine
from angr.exploration.rust_manager import (
    _unsupported_inspect_events,
    is_python_init_seed,
    resolve_main_address,
    rust_state_option_refusals,
    rust_supports_arch,
)
from angr.exploration.rust_techniques import _NATIVE_STEP_TECH_NAMES
from angr.sim_manager_backend import SimulationBackend

if TYPE_CHECKING:
    from angr.sim_manager import SimulationManager
    from angr.sim_state import SimState

l = logging.getLogger(name=__name__)

_NOT_IMPLEMENTED = "stepping not implemented"

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

    def __init__(self, project):
        self._project = project
        self._main_addr_resolved = False
        self._main_addr: int | None = None
        self._phase = _LineageMarks()
        self._ownership = _LineageMarks()
        self.stats = {"claimed": 0, "declined": Counter(), "handed_back": 0}

    def claims(self, simgr: SimulationManager, state: SimState, successor_func, run_args: dict) -> tuple[bool, str]:
        reason = self.decline_reason(simgr, state, successor_func, run_args)
        if reason is None:
            self._ownership.set(state, _OWNED)
            reason = _NOT_IMPLEMENTED
        elif self._ownership.get(state) == _OWNED:
            self._ownership.set(state, _HANDED_BACK)
            self.stats["handed_back"] += 1
        self.stats["declined"][reason] += 1
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
        if run_args:
            return "step passes run arguments: " + ", ".join(sorted(run_args))
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
        return None

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
