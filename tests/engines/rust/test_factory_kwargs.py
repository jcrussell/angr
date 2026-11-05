"""Property test: ``factory.simulation_manager(engine="rust", **kwargs)``
never *silently drops* a kwarg.

The contract: a Rust route builds a ``SimulationManager`` with a
``RustBackend`` (angr-mx00p.136.19), so every kwarg the Python
``SimulationManager`` accepts reaches that manager exactly as on the Python
route, the ``RustBackend`` knobs reach the backend, and anything else raises
TypeError, so ``simulation_manager(state, engine="rust", hierarchy=...)``
never constructs a manager that quietly ignores ``hierarchy``
(angr-op0dn.14.5.1).
"""

from __future__ import annotations

import inspect
from collections import defaultdict

import pytest

from angr.exploration import (
    RustExplorationManager,
    rust_unsupported_options,
    unsupported_rust_manager_kwargs,
)
from angr.exploration.rust_backend import RustBackend
from angr.exploration_techniques import DFS
from angr.exploration_techniques.suggestions import Suggestions
from angr.sim_manager import SimulationManager
from angr.sim_manager_backend import SimulationBackend
from angr.state_hierarchy import StateHierarchy

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


_ERRORED_SENTINEL = object()

# Per SimulationManager kwarg: a non-default value (built fresh per test) and a predicate that holds
# on a manager the value reached.
_SENTINELS = {
    "stashes": (lambda: defaultdict(list), lambda mgr, value: mgr._stashes is value),
    "hierarchy": (StateHierarchy, lambda mgr, value: mgr._hierarchy is value),
    "resilience": (lambda: True, lambda mgr, _value: bool(mgr._resilience)),
    "save_unsat": (lambda: True, lambda mgr, _value: mgr._save_unsat is True),
    "auto_drop": (lambda: ["avoid"], lambda mgr, _value: "avoid" in mgr._auto_drop),
    "errored": (lambda: [_ERRORED_SENTINEL], lambda mgr, _value: _ERRORED_SENTINEL in mgr._errored),
    "completion_mode": (lambda: all, lambda mgr, _value: mgr.completion_mode is all),
    "techniques": (lambda: [DFS()], lambda mgr, value: value[0] in mgr._techniques),
    "suggestions": (
        lambda: False,
        lambda mgr, _value: not any(isinstance(t, Suggestions) for t in mgr._techniques),
    ),
}


class TestFactoryKwargsAreNeverDropped:
    def test_sentinels_cover_every_kwarg(self):
        assert set(_SENTINELS) | {"backend"} == set(_simulation_manager_kwarg_names())

    @pytest.mark.parametrize("name", sorted(_SENTINELS))
    def test_kwarg_reaches_manager(self, fauxware_project, name):
        """No SimulationManager kwarg is silently swallowed by the Rust route."""
        make, reached = _SENTINELS[name]
        state = fauxware_project.factory.entry_state()
        value = make()
        mgr = fauxware_project.factory.simulation_manager(state, engine="rust", **{name: value})
        assert type(mgr) is SimulationManager
        assert isinstance(mgr.backend, RustBackend)
        assert reached(mgr, value)

    def test_backend_kwarg_conflicts_with_the_rust_route(self, fauxware_project):
        """The Rust route supplies the backend itself; a caller's backend is refused, not dropped."""
        state = fauxware_project.factory.entry_state()
        with pytest.raises(TypeError, match="backend"):
            fauxware_project.factory.simulation_manager(state, engine="rust", backend=SimulationBackend())

    def test_save_unconstrained_still_reaches_the_backend(self, fauxware_project):
        """A RustBackend knob configures the backend's private manager."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, engine="rust", save_unconstrained=False)
        assert isinstance(mgr.backend, RustBackend)
        assert mgr.backend._manager_kwargs["save_unconstrained"] is False

    def test_save_unconstrained_also_reaches_the_manager(self, fauxware_project):
        """``save_unconstrained=False`` drops the unconstrained stash on a Python manager; a Rust-routed
        manager steps declined states on Python, so it must drop it too."""
        state = fauxware_project.factory.entry_state()
        python = fauxware_project.factory.simulation_manager(state, save_unconstrained=False)
        rust = fauxware_project.factory.simulation_manager(state, engine="rust", save_unconstrained=False)
        assert "unconstrained" in python._auto_drop
        assert "unconstrained" in rust._auto_drop

    def test_rust_only_kwarg_is_forwarded(self, fauxware_project):
        """A RustBackend knob survives the factory."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, engine="rust", deterministic=True)
        assert isinstance(mgr.backend, RustBackend)
        assert mgr.backend._manager_kwargs["deterministic"] is True

    @pytest.mark.parametrize("name", ["totally_bogus_kwarg", "exploration_strategy"])
    def test_unknown_kwarg_raises_rather_than_dropping(self, fauxware_project, name):
        """Neither a SimulationManager kwarg nor a RustBackend knob: refused, never dropped.
        ``exploration_strategy`` is ``mgr.backend.set_exploration_strategy()`` now."""
        state = fauxware_project.factory.entry_state()
        with pytest.raises(TypeError, match=name):
            fauxware_project.factory.simulation_manager(state, engine="rust", **{name: 1})

    def test_python_path_unaffected(self, fauxware_project):
        """Zero behavior change for the Python engine: kwargs still flow through."""
        state = fauxware_project.factory.entry_state()
        sm = fauxware_project.factory.simulation_manager(state, resilience=True)
        assert isinstance(sm, SimulationManager)
        assert not isinstance(sm, RustExplorationManager)


class TestRaiseOptionPredicate:
    """The raise-option list has exactly one source of truth."""

    # Every mode bundle ships EXTENDED_IROP_SUPPORT, and its *absence* is itself
    # an offense (_REQUIRED_OPTION_NAMES; see test_default_bundle_options.py),
    # so a realistic option set carries it (angr-op0dn.14.7).
    def test_predicate_flags_offending_options(self):
        base = {"SYMBOLIC", "EXTENDED_IROP_SUPPORT"}
        assert rust_unsupported_options(base | {"TRACK_MEMORY_ACTIONS"}) == ["TRACK_MEMORY_ACTIONS"]
        assert rust_unsupported_options(base) == []
        assert rust_unsupported_options(None) == []

    def test_predicate_is_sorted_and_complete(self):
        offending = rust_unsupported_options({"CONCRETIZE", "CALLLESS", "SYMBOLIC", "EXTENDED_IROP_SUPPORT"})
        assert offending == ["CALLLESS", "CONCRETIZE"]

    def test_constructor_consumes_the_predicate(self, fauxware_project):
        """The manager's own guard raises for exactly what the predicate flags."""
        state = fauxware_project.factory.entry_state(add_options={"TRACK_MEMORY_ACTIONS"})
        assert rust_unsupported_options(state.options) == ["TRACK_MEMORY_ACTIONS"]
        with pytest.raises(NotImplementedError, match="TRACK_MEMORY_ACTIONS"):
            RustExplorationManager(fauxware_project, [state])

    def test_kwargs_predicate_matches_the_signature(self):
        assert unsupported_rust_manager_kwargs({"save_unconstrained": True, "deterministic": True}) == []
        assert unsupported_rust_manager_kwargs({"hierarchy": None, "save_unsat": True}) == [
            "hierarchy",
            "save_unsat",
        ]
