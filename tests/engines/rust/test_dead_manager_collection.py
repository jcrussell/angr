"""When building a manager runs the cyclic collector to free finished ones.

A finished manager is freed only by a full collection, which costs far more
than building a small manager, so ``_collect_dead_managers`` runs one only
after the native heap in use has grown by ``_GC_HEAP_SLACK`` since the
last (angr-mx00p.96). A run of ``Callable`` calls, one manager each, used to
spend a third of its time collecting.
"""

from __future__ import annotations

import pytest

from angr.exploration import rust_manager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

BASE = 1 << 30
SLACK = rust_manager._GC_HEAP_SLACK


class _Manager:
    """Stands in for a live manager: the policy only asks whether one exists."""


@pytest.fixture
def policy(monkeypatch):
    """The collection policy with its inputs under the test's control.

    ``policy.in_use`` is what the allocator reports as in use (None: it does not say);
    ``policy.collections`` counts the collections run. The duty-cycle window is closed, so only the
    heap-growth condition decides.
    """

    class Policy:
        in_use: int | None = BASE
        collections = 0

        def build_manager(self):
            rust_manager._collect_dead_managers()

    state = Policy()

    def collect():
        state.collections += 1

    live = _Manager()
    monkeypatch.setattr(rust_manager, "_live_managers", {live})
    monkeypatch.setattr(rust_manager, "_native_heap_bytes", lambda: state.in_use)
    monkeypatch.setattr(rust_manager.gc, "collect", collect)
    monkeypatch.setattr(rust_manager, "_GC_DUTY", 0.0)
    monkeypatch.setattr(rust_manager, "_last_collect_end", 0.0)
    monkeypatch.setattr(rust_manager, "_last_collect_cost", 0.0)
    monkeypatch.setattr(rust_manager, "_collect_heap_floor", None)
    return state


def test_first_build_with_a_live_manager_collects(policy):
    policy.build_manager()
    assert policy.collections == 1


def test_no_live_manager_never_collects(policy, monkeypatch):
    monkeypatch.setattr(rust_manager, "_live_managers", set())
    policy.in_use = BASE + 10 * SLACK
    policy.build_manager()
    assert policy.collections == 0


def test_small_managers_share_one_collection(policy):
    policy.build_manager()
    for step in range(1, 8):
        policy.in_use = BASE + step * (SLACK // 8)
        policy.build_manager()
    assert policy.collections == 1, "growth below the slack must not collect"

    policy.in_use = BASE + SLACK
    policy.build_manager()
    assert policy.collections == 2, "growth of the whole slack must collect"

    policy.in_use = BASE + SLACK + SLACK // 2
    policy.build_manager()
    assert policy.collections == 2, "the slack is measured from the last collection"


def test_slack_is_measured_from_the_low_water_mark(policy):
    policy.build_manager()
    policy.in_use = BASE - 4 * SLACK
    policy.build_manager()
    assert policy.collections == 1

    policy.in_use = BASE - 3 * SLACK
    policy.build_manager()
    assert policy.collections == 2, "heap regained after it shrank counts as growth"


def test_unreported_heap_use_collects_every_time(policy):
    policy.in_use = None
    for _ in range(3):
        policy.build_manager()
    assert policy.collections == 3


def test_duty_cycle_still_defers_a_collection_the_slack_allows(policy, monkeypatch):
    policy.build_manager()
    assert policy.collections == 1
    policy.in_use = BASE + 2 * SLACK
    monkeypatch.setattr(rust_manager, "_GC_DUTY", 2.0)
    monkeypatch.setattr(rust_manager, "_last_collect_end", rust_manager.time.perf_counter())
    monkeypatch.setattr(rust_manager, "_last_collect_cost", 3600.0)
    policy.build_manager()
    assert policy.collections == 1


def test_native_heap_bytes_follows_an_allocation():
    if rust_manager._mallinfo2() is None:
        pytest.skip("the C library has no mallinfo2")
    before = rust_manager._native_heap_bytes()
    block = bytearray(4 * SLACK)
    assert rust_manager._native_heap_bytes() - before >= 4 * SLACK
    del block
    assert rust_manager._native_heap_bytes() - before < SLACK
