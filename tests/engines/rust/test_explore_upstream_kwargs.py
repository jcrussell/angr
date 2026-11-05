#!/usr/bin/env python3
"""SimulationManager's step/explore/run keywords are honored or refused, never dropped (angr-vmi2l.243).

``explore()`` and ``__init__`` used to swallow every keyword they did not
name in ``**kwargs``: ``explore(n=)`` ran to the find (a hang on a loop),
``find_stash=`` left the find in ``found``, and an engine argument changed
nothing.

A Rust-routed manager is a ``SimulationManager`` with a ``RustBackend``
(angr-mx00p.136.19): it honors every manager-level keyword itself, so a
keyword is either honored exactly as on the Python engine or, for one that
reaches the per-state step (``successor_func``, engine run arguments), a
strict ``engine="rust"`` manager raises at that step.
"""

from __future__ import annotations

import inspect

import pytest

import angr
from angr.exploration import RustExplorationManager
from angr.exploration.rust_backend import RustBackend
from angr.exploration_techniques import Explorer
from angr.sim_manager import SimulationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, load_fauxware

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)

FAUXWARE_FIND = 0x4006ED
# The step from main in which fauxware's authenticated path reaches FAUXWARE_FIND.
FAUXWARE_FIND_ROUND = 27
BASE = 0x1000


def _named_params(fn) -> list[str]:
    return [
        name
        for name, p in inspect.signature(fn).parameters.items()
        if name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    ]


def _fauxware_mgr(rust: bool):
    project = load_fauxware()
    state = project.factory.entry_state()
    if rust:
        return project.factory.simulation_manager(state, engine="rust")
    return project.factory.simulation_manager(state)


def _counts(mgr) -> dict[str, int]:
    return {k: len(v) for k, v in mgr.stashes.items() if v}


def _addrs(mgr) -> dict[str, list[int]]:
    return {k: sorted(s.addr for s in v) for k, v in mgr.stashes.items() if v}


def _main_mgr(rust: bool):
    """A fauxware manager whose one active state is at ``main``, past the init prefix the backend declines."""
    project = load_fauxware()
    pre = project.factory.simulation_manager(project.factory.entry_state())
    pre.explore(find=project.loader.find_symbol("main").rebased_addr)
    return project.factory.simulation_manager(pre.found[0], engine="rust" if rust else None)


# A non-default value per upstream step() parameter besides stash/n/successor_func; each changes where the
# step's states land, so a dropped one shows in the stashes.
_STEP_VALUES = {
    "target_stash": lambda: "other",
    "selector_func": lambda: lambda _s: True,
    "step_func": lambda: lambda m: m.move("active", "stepped"),
    "error_list": list,
    "until": lambda: lambda _m: True,
    "filter_func": lambda: lambda _s: "filtered",
}


class TestSignatures:
    def test_explore_names_every_upstream_parameter(self):
        missing = set(_named_params(SimulationManager.explore)) - set(_named_params(RustExplorationManager.explore))
        assert not missing

    def test_run_names_upstream_parameters_or_forwards_them(self):
        """run() forwards to step()/explore(), so each upstream run parameter reaches one that names or refuses it."""
        mgr = _fauxware_mgr(rust=True)
        mgr.run(stash="active", n=1, until=lambda _m: True)

    def test_step_values_cover_every_other_upstream_parameter(self):
        assert set(_STEP_VALUES) == set(_named_params(SimulationManager.step)) - {"stash", "n", "successor_func"}

    @pytest.mark.parametrize("name", sorted(_STEP_VALUES))
    def test_step_honors_each_other_upstream_parameter_as_python(self, name):
        results = []
        for rust in (False, True):
            mgr = _main_mgr(rust)
            mgr.step(**{name: _STEP_VALUES[name]()})
            results.append(_addrs(mgr))
        assert results[0] == results[1]
        assert isinstance(mgr.backend, RustBackend)
        if name != "filter_func":
            # filter_func moves the state aside unstepped, so only the other parameters reach the backend.
            assert mgr.backend.stats["claimed"] >= 1

    def test_step_refuses_successor_func(self):
        mgr = _fauxware_mgr(rust=True)
        with pytest.raises(NotImplementedError, match="successor_func"):
            mgr.step(successor_func=lambda s: s.step())

    def test_init_takes_no_unknown_kwargs(self):
        project = load_fauxware()
        with pytest.raises(TypeError, match="hierarchy"):
            RustExplorationManager(project, active_states=[project.factory.entry_state()], hierarchy=None)


class TestExploreHonors:
    def test_find_stash_matches_python(self):
        results = []
        for rust in (False, True):
            mgr = _fauxware_mgr(rust)
            mgr.explore(find=FAUXWARE_FIND, find_stash="win")
            results.append((len(mgr.win), len(mgr.stashes.get("found", [])), mgr.win[0].addr))
        assert results[0] == results[1] == (1, 0, FAUXWARE_FIND)

    def test_n_bounds_the_run_as_python(self):
        counts = []
        for rust in (False, True):
            mgr = _fauxware_mgr(rust)
            mgr.explore(find=FAUXWARE_FIND, n=3)
            counts.append(_counts(mgr))
        assert counts[0] == counts[1] == {"active": 1}

    @pytest.mark.parametrize("n", [1, FAUXWARE_FIND_ROUND, FAUXWARE_FIND_ROUND + 1])
    @pytest.mark.parametrize(
        "targets",
        [
            {"find": FAUXWARE_FIND},
            {"find": 0x1234, "avoid": FAUXWARE_FIND},
            {"find": lambda s: s.addr == FAUXWARE_FIND},
            {"find": lambda _s: False, "avoid": lambda s: s.addr == FAUXWARE_FIND},
        ],
        ids=["find", "avoid", "find_predicate", "avoid_predicate"],
    )
    def test_n_budget_boundary_matches_python(self, targets, n):
        """A target reached in the budget's last step stays active: Explorer filters before it steps (angr-mx00p.136.35)."""
        stashes = []
        for rust in (False, True):
            project = load_fauxware()
            pre = project.factory.simulation_manager(project.factory.entry_state())
            pre.explore(find=project.loader.find_symbol("main").rebased_addr)
            mgr = project.factory.simulation_manager(pre.found[0], engine="rust" if rust else None)
            mgr.explore(n=n, **targets)
            stashes.append({k: sorted(s.addr for s in v) for k, v in mgr.stashes.items() if v})
        if n == FAUXWARE_FIND_ROUND:
            assert FAUXWARE_FIND in stashes[0]["active"]
        assert stashes[0] == stashes[1]

    def test_n_returns_on_an_unreachable_find(self):
        """``jmp $`` never reaches the find; n= is the only stop."""
        project = angr.load_shellcode(b"\xeb\xfe\x90\x90", arch="amd64", load_address=BASE)
        mgr = project.factory.simulation_manager(project.factory.blank_state(addr=BASE), engine="rust")
        mgr.explore(find=BASE + 2, n=20)
        assert _counts(mgr) == {"active": 1}


class TestExploreRefuses:
    @pytest.mark.parametrize(
        "kwargs",
        [{"num_inst": 1}],
        ids=["num_inst"],
    )
    def test_explicit_manager_raises(self, kwargs):
        mgr = _fauxware_mgr(rust=True)
        with pytest.raises(NotImplementedError, match=next(iter(kwargs))):
            mgr.explore(find=FAUXWARE_FIND, **kwargs)


class TestExploreHonorsManagerKwargs:
    @pytest.mark.parametrize(
        "kwargs",
        [{"selector_func": lambda _s: True}, {"stash": "other"}, {"avoid_priority": True}],
        ids=["selector_func", "stash", "avoid_priority"],
    )
    def test_explicit_manager_matches_python(self, kwargs):
        results = []
        for rust in (False, True):
            mgr = _main_mgr(rust)
            mgr.explore(find=FAUXWARE_FIND, n=FAUXWARE_FIND_ROUND + 1, **kwargs)
            results.append(_addrs(mgr))
        assert results[0] == results[1]
        assert isinstance(mgr.backend, RustBackend)

    def test_default_route_honors_avoid_priority(self):
        project = load_fauxware(engine="rust")
        mgr = project.factory.simulation_manager(project.factory.entry_state())
        assert isinstance(mgr.backend, RustBackend)
        mgr.explore(find=FAUXWARE_FIND, avoid_priority=True)
        assert type(mgr) is SimulationManager
        assert len(mgr.found) == 1
        assert mgr.backend.stats["claimed"] >= 1


class TestExplorerTechnique:
    def test_default_route_hands_off_for_find_stash(self):
        project = load_fauxware(engine="rust")
        mgr = project.factory.simulation_manager(project.factory.entry_state())
        mgr.use_technique(Explorer(find=FAUXWARE_FIND, find_stash="mine"))
        assert type(mgr) is SimulationManager
        mgr.run()
        assert len(mgr.mine) == 1

    def test_explicit_manager_honors_find_stash(self):
        mgr = _fauxware_mgr(rust=True)
        mgr.use_technique(Explorer(find=FAUXWARE_FIND, find_stash="mine"))
        mgr.run()
        assert len(mgr.mine) == 1
        assert not mgr.stashes.get("found")
        # run() takes the backend's fast path here, which steps on the private manager without per-state claims.
        stats = mgr.backend.stats
        assert stats["claimed"] + stats["fast_path_runs"] >= 1
