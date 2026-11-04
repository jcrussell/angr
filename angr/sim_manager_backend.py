from __future__ import annotations

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
    not, and the manager's resilience handling applies to errors the backend raises.

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

    def step_state(self, simgr: SimulationManager, state: SimState, error_list: list, **run_args) -> dict:
        """
        Step a claimed state once.

        :return: A dict shaped like the one :meth:`SimulationManager.step_state` returns: the key None maps to the
                 flat successors, and a stash name (``"unsat"``, ``"unconstrained"``, ``"pruned"``, ...) maps to the
                 states routed there.
        """
        raise NotImplementedError

    def detach(self, state: SimState) -> SimState:
        """
        Return a plain SimState for ``state``, with no tie to this backend.
        """
        return state
