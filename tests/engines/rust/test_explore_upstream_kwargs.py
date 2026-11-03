#!/usr/bin/env python3
"""SimulationManager's step/explore/run keywords are honored or refused, never dropped (angr-vmi2l.243).

``explore()`` and ``__init__`` used to swallow every keyword they did not
name in ``**kwargs``: ``explore(n=)`` ran to the find (a hang on a loop),
``find_stash=`` left the find in ``found``, and an engine argument changed
nothing.
"""

from __future__ import annotations

import inspect

import pytest

import angr
from angr.exploration import RustExplorationManager
from angr.exploration_techniques import Explorer
from angr.sim_manager import SimulationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, load_fauxware

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)

FAUXWARE_FIND = 0x4006ED
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


class TestSignatures:
    def test_explore_names_every_upstream_parameter(self):
        missing = set(_named_params(SimulationManager.explore)) - set(_named_params(RustExplorationManager.explore))
        assert not missing

    def test_run_names_upstream_parameters_or_forwards_them(self):
        """run() forwards to step()/explore(), so each upstream run parameter reaches one that names or refuses it."""
        mgr = _fauxware_mgr(rust=True)
        mgr.run(stash="active", n=1, until=lambda _m: True)

    @pytest.mark.parametrize(
        "name",
        sorted(set(_named_params(SimulationManager.step)) - {"stash", "n"}),
    )
    def test_step_refuses_each_other_upstream_parameter(self, name):
        mgr = _fauxware_mgr(rust=True)
        with pytest.raises(NotImplementedError, match=name):
            mgr.step(**{name: object()})

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

    def test_n_returns_on_an_unreachable_find(self):
        """``jmp $`` never reaches the find; n= is the only stop."""
        project = angr.load_shellcode(b"\xeb\xfe\x90\x90", arch="amd64", load_address=BASE)
        mgr = project.factory.simulation_manager(project.factory.blank_state(addr=BASE), engine="rust")
        mgr.explore(find=BASE + 2, n=20)
        assert _counts(mgr) == {"active": 1}


class TestExploreRefuses:
    @pytest.mark.parametrize(
        "kwargs",
        [{"selector_func": lambda _s: True}, {"num_inst": 1}, {"stash": "other"}, {"cfg": object()}],
        ids=["selector_func", "num_inst", "stash", "cfg"],
    )
    def test_explicit_manager_raises(self, kwargs):
        mgr = _fauxware_mgr(rust=True)
        with pytest.raises(NotImplementedError, match=next(iter(kwargs))):
            mgr.explore(find=FAUXWARE_FIND, **kwargs)

    def test_avoid_priority_raises(self):
        mgr = _fauxware_mgr(rust=True)
        with pytest.raises(NotImplementedError, match="avoid_priority"):
            mgr.explore(find=FAUXWARE_FIND, avoid_priority=True)

    def test_default_route_hands_off_and_honors(self):
        project = load_fauxware(engine="rust")
        mgr = project.factory.simulation_manager(project.factory.entry_state())
        assert isinstance(mgr, RustExplorationManager)
        mgr.explore(find=FAUXWARE_FIND, avoid_priority=True)
        assert type(mgr) is SimulationManager
        assert len(mgr.found) == 1


class TestExplorerTechnique:
    def test_default_route_hands_off_for_find_stash(self):
        project = load_fauxware(engine="rust")
        mgr = project.factory.simulation_manager(project.factory.entry_state())
        mgr.use_technique(Explorer(find=FAUXWARE_FIND, find_stash="mine"))
        assert type(mgr) is SimulationManager
        mgr.run()
        assert len(mgr.mine) == 1

    def test_explicit_manager_warns_for_find_stash(self, caplog):
        mgr = _fauxware_mgr(rust=True)
        with caplog.at_level("WARNING"):
            mgr.use_technique(Explorer(find=FAUXWARE_FIND, find_stash="mine"))
        assert "find_stash" in caplog.text
