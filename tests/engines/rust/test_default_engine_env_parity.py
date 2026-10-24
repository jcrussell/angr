"""Routing assertions that only make sense with ``ANGR_DEFAULT_ENGINE=rust`` set.

The sibling ``test_default_engine_flag.py`` pins the flag's behavior from a
*stock* process (env unset) — it opts individual projects in via
``Project(engine="rust")``. This module pins the other half: the whole-process
regime the ``rust-nightly.yml::default_engine_parity`` job runs the stock test subset
under, where every new ``Project`` defaults to the Rust manager.

The invariant that job exists to protect: raising the *default* must not weaken
the eligibility gate. An ineligible workload (the canonical case being a
``TRACK_MEMORY_ACTIONS`` state, which needs SimActions the native engine does
not record) has to route back to Python transparently — no exception, no
test edits, and observably so via the factory's debug log.

Skipped unless the env var is actually set, so a normal local run is a no-op::

    ANGR_DEFAULT_ENGINE=rust pytest tests/engines/rust/test_default_engine_env_parity.py
"""

from __future__ import annotations

import logging
import os

import pytest

from angr import sim_options as o
from angr.exploration import RustExplorationManager
from angr.sim_manager import SimulationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, load_fauxware

pytestmark = [
    pytest.mark.skipif(
        not RUST_EXPLORATION_AVAILABLE,
        reason="Rust exploration not available",
    ),
    pytest.mark.skipif(
        os.environ.get("ANGR_DEFAULT_ENGINE", "").strip().lower() != "rust",
        reason="requires ANGR_DEFAULT_ENGINE=rust (rust-nightly.yml default_engine_parity job)",
    ),
]


@pytest.fixture
def env_project():
    """A project built with *no* engine argument — the env var is what opts it in."""
    return load_fauxware()


def _routing_log(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records if r.name == "angr.factory")


def test_eligible_workload_routes_to_rust(env_project, caplog):
    with caplog.at_level(logging.DEBUG, logger="angr.factory"):
        mgr = env_project.factory.simulation_manager(env_project.factory.entry_state())
    assert isinstance(mgr, RustExplorationManager)
    assert "rust engine supports this project" in _routing_log(caplog)


def test_track_memory_actions_routes_back_to_python(env_project, caplog):
    """The acceptance case: an ineligible state falls back silently, not loudly."""
    state = env_project.factory.entry_state(add_options={o.TRACK_MEMORY_ACTIONS})
    with caplog.at_level(logging.DEBUG, logger="angr.factory"):
        mgr = env_project.factory.simulation_manager(state)
    assert type(mgr) is SimulationManager
    assert "TRACK_MEMORY_ACTIONS" in _routing_log(caplog)


def test_explicit_rust_stays_loud(env_project):
    state = env_project.factory.entry_state(add_options={o.TRACK_MEMORY_ACTIONS})
    with pytest.raises(NotImplementedError, match="TRACK_MEMORY_ACTIONS"):
        env_project.factory.simulation_manager(state, engine="rust")


def test_default_engine_factory_untouched(env_project):
    """The env flip swaps the MANAGER, not the SimEngine — block()/CFG stay put."""
    from angr.engines import UberEngine

    assert env_project.factory.default_engine_factory is UberEngine
