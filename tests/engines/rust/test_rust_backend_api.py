"""The Rust-only API on ``simgr.backend`` (angr-mx00p.136.16).

A :class:`RustBackend` forwards the members code reads off a
RustExplorationManager (stats, perf_report, get_solver_stats, the
``<stash>_proxies()`` accessors, coverage, exploration strategy and the
uniqueness filter) to its private manager, scoped to the Rust states in the
SimulationManager's stashes; a RustExplorationManager's ``backend`` is the
manager itself, so ``mgr.backend.<member>`` reads the same on both.
"""

from __future__ import annotations

import pytest

import angr
from angr.exploration import rust_backend
from angr.exploration.rust_backend import RustBackend
from angr.exploration.rust_manager import RustExplorationManager
from angr.exploration.rust_state_binding import binding_of
from tests.engines.conftest import FAUXWARE_PATH, RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")

FIND = 0x4006ED
AVOID = [0x4006AA, 0x4006FD]
# A bound no run reaches, which keeps explore() on the per-step backend path.
PER_STEP_N = 10**6
COUNTERS = ("claimed", "declined", "handed_back", "fast_path_runs")
FORWARDED = (
    "stats",
    "perf_report",
    "get_solver_stats",
    "set_exploration_strategy",
    "register_uniqueness_filter",
    "disable_uniqueness_filter",
    "uniqueness_filter_enabled",
    "uniqueness_set_size",
    "coverage_blocks",
    "coverage_report",
    "active_proxies",
    "found_proxies",
    "avoid_proxies",
    "deadended_proxies",
    "unconstrained_proxies",
)


@pytest.fixture(scope="module")
def project():
    return angr.Project(FAUXWARE_PATH, auto_load_libs=False)


@pytest.fixture(scope="module")
def explored(project):
    backend = RustBackend(project)
    simgr = angr.SimulationManager(project, [project.factory.entry_state()], backend=backend)
    simgr.explore(find=FIND, avoid=AVOID, n=PER_STEP_N)
    assert simgr.found, "fauxware must reach the authenticated branch"
    return simgr, backend


def _blocks(states):
    return sorted({addr for state in states for addr in state.history.bbl_addrs})


def test_backend_of_a_rust_manager_is_the_manager(project):
    mgr = RustExplorationManager(project, [project.factory.entry_state()])
    assert mgr.backend is mgr
    for name in FORWARDED:
        assert hasattr(mgr.backend, name), name


def test_backend_is_itself_and_has_every_forwarded_member(project):
    backend = RustBackend(project)
    assert backend.backend is backend
    for name in FORWARDED:
        assert hasattr(backend, name), name


def test_stats_merge_the_manager_counters_with_the_backend_counters(explored):
    _, backend = explored
    stats = backend.stats
    assert stats["claimed"] > 0
    assert stats["declined"]
    # The private manager's own counters come along.
    assert "callback_count" in stats
    assert "ffi_crossings" in stats
    manager_stats = backend._mgr.stats
    assert not set(COUNTERS) & set(manager_stats), "a backend counter would shadow a manager counter"


def test_stats_before_any_step_are_the_backend_counters(project):
    stats = RustBackend(project).stats
    assert set(stats) == set(COUNTERS)
    assert stats["claimed"] == 0


def test_reports_forward_to_the_private_manager(explored):
    _, backend = explored
    assert isinstance(backend.perf_report(), str)
    assert isinstance(backend.get_solver_stats(), dict)


def test_proxies_are_the_rust_states_of_each_stash(explored):
    simgr, backend = explored
    for stash in ("found", "avoid", "deadended", "active"):
        proxies = getattr(backend, f"{stash}_proxies")()
        # The avoid state's lineage is handed back to Python (Explorer's avoid address lies inside a
        # block, an extra_stop_points decline), so not every state is a Rust state.
        rust_states = [s for s in simgr.stashes[stash] if binding_of(s) is not None]
        assert sorted(p.addr for p in proxies) == sorted(s.addr for s in rust_states), stash
    assert len(backend.found_proxies()) == len(simgr.found)
    assert backend.unconstrained_proxies() == []
    assert backend.found_proxies()[0].addr == FIND


def test_proxies_leave_out_a_state_on_the_python_engine(project, explored):
    simgr, backend = explored
    simgr.stashes["found"].append(project.factory.blank_state(addr=FIND))
    try:
        assert len(backend.found_proxies()) == len(simgr.found) - 1
    finally:
        simgr.stashes["found"].pop()


def test_coverage_matches_a_rust_manager_on_the_same_exploration(project, explored):
    simgr, backend = explored
    mgr = RustExplorationManager(project, [project.factory.entry_state()])
    mgr.explore(find=FIND, avoid=AVOID)
    # Both cover the Rust history only: the blocks of the Python init prefix are not in it.
    assert backend.coverage_blocks() == mgr.coverage_blocks()
    assert set(backend.coverage_blocks()) < set(_blocks(simgr.found))
    assert backend.coverage_report().edge_hitmap == mgr.coverage_report().edge_hitmap
    assert backend.coverage_report().edge_hitmap.count(0xFF) > 0
    assert backend.coverage_blocks(("found", "deadended")) == mgr.coverage_blocks(("found", "deadended"))


def test_stash_scoped_members_are_empty_before_the_backend_serves_a_manager(project):
    backend = RustBackend(project)
    assert backend.active_proxies() == []
    assert backend.coverage_blocks(("active",)) == []


def test_uniqueness_filter_and_strategy_reach_the_private_manager(project):
    backend = RustBackend(project)
    assert not backend.uniqueness_filter_enabled()
    backend.register_uniqueness_filter(["rip"])
    assert backend.uniqueness_filter_enabled()
    assert backend._mgr._rust_mgr.uniqueness_filter_enabled()
    assert backend.uniqueness_set_size() == 0
    backend.disable_uniqueness_filter()
    assert not backend._mgr._rust_mgr.uniqueness_filter_enabled()
    backend.set_exploration_strategy("dfs")
    with pytest.raises(ValueError):
        backend.set_exploration_strategy("no-such-strategy")


def test_construction_knobs_reach_the_private_manager(project, monkeypatch):
    seen = {}

    class Recording(RustExplorationManager):
        def __init__(self, proj, **kwargs):
            seen.update(kwargs)
            super().__init__(proj, **kwargs)

    monkeypatch.setattr(rust_backend, "RustExplorationManager", Recording)
    knobs = {
        "max_active_states": 64,
        "parallel_workers": 1,
        "deterministic": True,
        "solver_timeout_ms": 1234,
        "max_history": 77,
        "use_native_lift": False,
        "prefer_native_library_hooks": False,
        "symlinks": {"/a": "/b"},
        "save_unconstrained": False,
    }
    backend = RustBackend(project, **knobs)
    simgr = angr.SimulationManager(project, [project.factory.entry_state()], backend=backend)
    simgr.explore(find=FIND, avoid=AVOID, n=PER_STEP_N)
    assert seen == knobs
    assert simgr.found
    assert backend.stats["claimed"] > 0


def test_default_private_manager_has_no_active_cap(project, monkeypatch):
    seen = {}

    class Recording(RustExplorationManager):
        def __init__(self, proj, **kwargs):
            seen.update(kwargs)
            super().__init__(proj, **kwargs)

    monkeypatch.setattr(rust_backend, "RustExplorationManager", Recording)
    RustBackend(project)._manager()
    assert seen["max_active_states"] is None
