"""Every state.inspect phase Rust accepts fires, and every other phase is refused (angr-vmi2l.203).

``_INSPECT_EVENT_SPECS[event]["when_fired"]`` is the table
``RustInspectProxy.add_breakpoint`` reads: a BP on a phase missing from it
raises, so the table must name every phase the dispatch sites pass and no
other. These tests check it against the engine end to end, on fauxware.
"""

from __future__ import annotations

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager
from angr.exploration.rust_state_proxy import _INSPECT_EVENT_SPECS
from tests.engines.conftest import fauxware_project  # noqa: F401

# Rust-origin events fauxware's first steps raise: the entry block's
# reads, writes and exits, the call into __libc_start_main and the first
# branch in main. The Python-dispatched events and the ones that need a
# symbolic address or a fresh symbol are not reached here.
_REACHED_EVENTS = [
    "mem_read",
    "mem_write",
    "reg_read",
    "reg_write",
    "tmp_read",
    "tmp_write",
    "expr",
    "instruction",
    "irsb",
    "statement",
    "exit",
    "call",
    "vex_lift",
]

_FIRED = [(event, when) for event in _REACHED_EVENTS for when in sorted(_INSPECT_EVENT_SPECS[event]["when_fired"])]
_UNFIRED = [
    (event, when)
    for event, spec in sorted(_INSPECT_EVENT_SPECS.items())
    for when in ("before", "after")
    if when not in spec["when_fired"]
]


def _manager(project):
    return RustExplorationManager(project, [project.factory.entry_state()])


@pytest.mark.parametrize(("event", "when"), _FIRED)
def test_every_accepted_phase_fires(fauxware_project, event, when):  # noqa: F811
    hits = []
    mgr = _manager(fauxware_project)
    mgr.active_proxies()[0].inspect.b(event, when=when, action=lambda _st: hits.append(1))
    mgr.run(max_steps=8)
    assert hits, f"{event} when={when} is in when_fired but never fired"


@pytest.mark.parametrize(("event", "when"), _UNFIRED)
def test_an_unfired_phase_is_refused(fauxware_project, event, when):  # noqa: F811
    mgr = _manager(fauxware_project)
    inspect = mgr.active_proxies()[0].inspect
    with pytest.raises(NotImplementedError, match=f"would miss the {when} phase"):
        inspect.b(event, when=when, action=lambda _st: None)
    assert not mgr._inspect_breakpoints[event]


@pytest.mark.parametrize("event", ["mem_read", "instruction"])
def test_default_when_and_bp_both_on_a_one_phase_event_are_refused(fauxware_project, event):  # noqa: F811
    """Upstream's default ``when`` is ``BP_BEFORE``; ``BP_BOTH`` would fire half as often as on Python."""
    inspect = _manager(fauxware_project).active_proxies()[0].inspect
    with pytest.raises(NotImplementedError):
        inspect.b(event, when=angr.BP_BOTH, action=lambda _st: None)
    if event == "mem_read":
        with pytest.raises(NotImplementedError):
            inspect.b(event, action=lambda _st: None)


def test_bp_both_on_a_two_phase_event_fires_in_each(fauxware_project):  # noqa: F811
    whens = []
    mgr = _manager(fauxware_project)
    mgr.active_proxies()[0].inspect.b("mem_write", when=angr.BP_BOTH, action=lambda st: whens.append(1))
    mgr.run(max_steps=1)
    assert whens and len(whens) % 2 == 0


def test_a_seed_bp_on_an_unfired_phase_is_refused(fauxware_project):  # noqa: F811
    """A seed handed straight to RustExplorationManager skips the eligibility routing."""
    state = fauxware_project.factory.entry_state()
    state.inspect.b("mem_read", action=lambda _st: None)
    with pytest.raises(NotImplementedError, match="would miss the before phase"):
        RustExplorationManager(fauxware_project, [state])
