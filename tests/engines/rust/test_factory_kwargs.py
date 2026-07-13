"""Property test: ``factory.simulation_manager(use_rust_engine=True, **kwargs)``
never *silently drops* a kwarg.

Before angr-op0dn.14.5.1 the Rust branch of
:meth:`angr.factory.AngrObjectFactory.simulation_manager` popped
``save_unconstrained`` and threw the rest of ``**kwargs`` away, so
``simulation_manager(state, use_rust_engine=True, resilience=True)``
constructed a manager that quietly ignored ``resilience``. The contract now
is: every kwarg the Python ``SimulationManager`` accepts either reaches the
constructed Rust manager or raises loudly.
"""

from __future__ import annotations

import inspect

import pytest

from angr.exploration import (
    RustExplorationManager,
    rust_unsupported_options,
    unsupported_rust_manager_kwargs,
)
from angr.sim_manager import SimulationManager

# Rust availability guard lives in the parent conftest, which also supplies
# the module-scoped ``fauxware_project`` fixture used below (angr-7gdp).
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


def _simulation_manager_kwarg_names() -> list[str]:
    """Named (non-positional-only, non-var) params of ``SimulationManager.__init__``,
    minus the ones the factory supplies itself."""
    params = inspect.signature(SimulationManager.__init__).parameters
    return [
        name
        for name, p in params.items()
        if name not in ("self", "project", "active_states")
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]


# A representative value per SimulationManager kwarg. Only the *routing*
# matters here (reached vs. raised), not the semantics of the value.
_SENTINELS = {
    "stashes": None,
    "hierarchy": None,
    "resilience": True,
    "save_unsat": True,
    "auto_drop": ["avoid"],
    "errored": [],
    "completion_mode": all,
    "techniques": None,
    "suggestions": False,
}


class TestFactoryKwargsAreNeverDropped:
    @pytest.mark.parametrize("name", _simulation_manager_kwarg_names())
    def test_kwarg_reaches_manager_or_raises(self, fauxware_project, name):
        """No SimulationManager kwarg is silently swallowed by the Rust branch."""
        value = _SENTINELS[name]
        state = fauxware_project.factory.entry_state()
        try:
            fauxware_project.factory.simulation_manager(state, use_rust_engine=True, **{name: value})
        except NotImplementedError as exc:
            # Loud refusal is the accepted outcome for SimulationManager-only kwargs.
            assert name in str(exc)
            return
        # Otherwise it must actually have reached the constructed manager,
        # i.e. RustExplorationManager names it explicitly in __init__.
        accepted = inspect.signature(RustExplorationManager.__init__).parameters
        assert name in accepted, f"{name!r} was accepted by the factory but is not a RustExplorationManager param"

    def test_save_unconstrained_still_reaches_the_manager(self, fauxware_project):
        """The one kwarg the old code did forward keeps working."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True, save_unconstrained=True)
        assert isinstance(mgr, RustExplorationManager)
        # Attribute access on a manager falls through to the stash lookup, so
        # read the private field the constructor actually stores.
        assert mgr._save_unconstrained is True

    def test_rust_only_kwarg_is_forwarded(self, fauxware_project):
        """A RustExplorationManager-native kwarg survives the factory."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(
            state, use_rust_engine=True, exploration_strategy="dfs", deterministic=True
        )
        assert isinstance(mgr, RustExplorationManager)
        assert mgr._deterministic is True

    def test_unknown_kwarg_raises_rather_than_dropping(self, fauxware_project):
        state = fauxware_project.factory.entry_state()
        with pytest.raises(NotImplementedError, match="totally_bogus_kwarg"):
            fauxware_project.factory.simulation_manager(state, use_rust_engine=True, totally_bogus_kwarg=1)

    def test_python_path_unaffected(self, fauxware_project):
        """Zero behavior change for the Python engine: kwargs still flow through."""
        state = fauxware_project.factory.entry_state()
        sm = fauxware_project.factory.simulation_manager(state, resilience=True)
        assert isinstance(sm, SimulationManager)
        assert not isinstance(sm, RustExplorationManager)


class TestRaiseOptionPredicate:
    """The raise-option list has exactly one source of truth."""

    def test_predicate_flags_offending_options(self):
        assert rust_unsupported_options({"TRACK_MEMORY_ACTIONS", "SYMBOLIC"}) == ["TRACK_MEMORY_ACTIONS"]
        assert rust_unsupported_options({"SYMBOLIC"}) == []
        assert rust_unsupported_options(None) == []

    def test_predicate_is_sorted_and_complete(self):
        offending = rust_unsupported_options({"CONCRETIZE", "CALLLESS", "SYMBOLIC"})
        assert offending == ["CALLLESS", "CONCRETIZE"]

    def test_constructor_consumes_the_predicate(self, fauxware_project):
        """The manager's own guard raises for exactly what the predicate flags."""
        state = fauxware_project.factory.entry_state(add_options={"TRACK_MEMORY_ACTIONS"})
        assert rust_unsupported_options(state.options) == ["TRACK_MEMORY_ACTIONS"]
        with pytest.raises(NotImplementedError, match="TRACK_MEMORY_ACTIONS"):
            RustExplorationManager(fauxware_project, [state])

    def test_kwargs_predicate_matches_the_signature(self):
        assert unsupported_rust_manager_kwargs({"save_unconstrained": True, "deterministic": True}) == []
        assert unsupported_rust_manager_kwargs({"hierarchy": None, "resilience": True}) == [
            "hierarchy",
            "resilience",
        ]
