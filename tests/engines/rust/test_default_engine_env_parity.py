"""Routing assertions that only make sense with ``ANGR_DEFAULT_ENGINE=rust`` set.

The sibling ``test_default_engine_flag.py`` pins the flag's behavior from a
*stock* process (env unset) — it opts individual projects in via
``Project(engine="rust")``. This module pins the other half: the whole-process
regime the nightly ``default_engine_parity`` job runs the stock test subset
under, where every new ``Project`` defaults to the Rust manager.

The invariant that job exists to protect: raising the *default* must not weaken
the eligibility gate. An ineligible workload (the canonical case being a
``TRACK_MEMORY_ACTIONS`` state, which needs SimActions the native engine does
not record) has to route back to Python transparently — no exception, no
test edits, and observably so via ``mgr.dispatch_reason``.

Skipped unless the env var is actually set, so a normal local run is a no-op::

    ANGR_DEFAULT_ENGINE=rust pytest tests/engines/rust/test_default_engine_env_parity.py
"""

from __future__ import annotations

import os

import pytest

import angr
from angr import sim_options as o
from angr.exploration import RustExplorationManager
from angr.sim_manager import SimulationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, TEST_BINARIES_DIR

pytestmark = [
    pytest.mark.skipif(
        not RUST_EXPLORATION_AVAILABLE,
        reason="Rust exploration not available",
    ),
    pytest.mark.skipif(
        os.environ.get("ANGR_DEFAULT_ENGINE", "").strip().lower() != "rust",
        reason="requires ANGR_DEFAULT_ENGINE=rust (nightly default_engine_parity job)",
    ),
]


@pytest.fixture
def env_project():
    """A project built with *no* engine argument — the env var is what opts it in."""
    binary_path = os.path.join(TEST_BINARIES_DIR, "fauxware")
    if not os.path.exists(binary_path):
        pytest.skip("fauxware binary not found")
    return angr.Project(binary_path, auto_load_libs=False)


def test_eligible_workload_routes_to_rust(env_project):
    mgr = env_project.factory.simulation_manager(env_project.factory.entry_state())
    assert isinstance(mgr, RustExplorationManager)
    assert "rust engine supports this project" in mgr.dispatch_reason


def test_track_memory_actions_routes_back_to_python(env_project):
    """The acceptance case: an ineligible state falls back silently, not loudly."""
    state = env_project.factory.entry_state(add_options={o.TRACK_MEMORY_ACTIONS})
    mgr = env_project.factory.simulation_manager(state)
    assert type(mgr) is SimulationManager
    assert "TRACK_MEMORY_ACTIONS" in mgr.dispatch_reason


def test_explicit_false_still_wins(env_project):
    mgr = env_project.factory.simulation_manager(env_project.factory.entry_state(), use_rust_engine=False)
    assert type(mgr) is SimulationManager
    assert mgr.dispatch_reason == "explicit use_rust_engine=False"


def test_default_engine_factory_untouched(env_project):
    """The env flip swaps the MANAGER, not the SimEngine — block()/CFG stay put."""
    from angr.engines import UberEngine

    assert env_project.factory.default_engine_factory is UberEngine
