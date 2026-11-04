from __future__ import annotations

import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sim_manager import SimulationManager
    from .sim_state import SimState


class SimulationBackend:
    """
    A stepping backend for :class:`angr.sim_manager.SimulationManager`.

    A backend is asked, per state and per step, whether it will step that state. When it claims the state, the
    manager's ``step_state`` delegates to :meth:`step_state`; otherwise the state takes the ordinary Python path.
    Exploration technique ``step_state`` hooks wrap the manager's ``step_state`` and so see every state, claimed or
    not, and the manager's resilience handling applies to errors the backend raises. When nothing hooks or overrides
    ``step_state``, the manager hands every state it claimed in one step to :meth:`step_states` at once.

    This base class claims nothing, so a manager with it behaves exactly like one without a backend.
    """

    def claims(self, simgr: SimulationManager, state: SimState, successor_func, run_args: dict) -> tuple[bool, str]:
        """
        Decide whether this backend steps ``state``.

        :param simgr:           The manager stepping the state.
        :param state:           The state about to be stepped.
        :param successor_func:  The ``successor_func`` passed to the step, or None.
        :param run_args:        The remaining keyword arguments of the step.
        :return:                ``(True, "")`` to step the state here, or ``(False, reason)`` to decline it.
        """
        return False, "the base SimulationBackend claims no state"

    def try_run(self, simgr: SimulationManager, stash: str, n: int | None, until, run_args: dict) -> bool:
        """
        Offer to run ``stash`` to completion in one call, in place of the rest of :meth:`SimulationManager.run`.

        Asked before each step of the run loop. A backend that takes the run leaves the manager's stashes as the
        remaining steps would have, and returns True; the loop then ends.

        :param simgr:       The running manager.
        :param stash:       The stash being run.
        :param n:           The steps left of the run's step bound, or None.
        :param until:       The run's ``until`` predicate, or None.
        :param run_args:    The keyword arguments the run passes to each step.
        :return:            True when the run is done, False to take the next step as usual.
        """
        return False

    def step_state(self, simgr: SimulationManager, state: SimState, error_list: list, **run_args) -> dict:
        """
        Step a claimed state once.

        :return: A dict shaped like the one :meth:`SimulationManager.step_state` returns: the key None maps to the
                 flat successors, and a stash name (``"unsat"``, ``"unconstrained"``, ``"pruned"``, ...) maps to the
                 states routed there.
        """
        raise NotImplementedError

    def step_states(self, simgr: SimulationManager, states: list[SimState], **run_args) -> list:
        """
        Step claimed states once, together.

        :return: One callable per state, in order. Called with the step's error list, it appends the records
                 :meth:`step_state` would have appended for that state and returns its result or raises its error,
                 so the manager handles each state exactly as it does on the per-state path. This default steps
                 each state through :meth:`step_state` when its callable is called.
        """
        return [functools.partial(self.step_state, simgr, state, **run_args) for state in states]

    def detach(self, state: SimState) -> SimState:
        """
        Return a plain SimState for ``state``, with no tie to this backend.
        """
        return state
