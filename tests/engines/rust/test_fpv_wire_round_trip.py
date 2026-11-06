# pylint: disable=missing-class-docstring,no-self-use
"""FPV constants survive the WireDag serde round trip (angr-vmi2l.274).

Snapshots and cross-worker state migration carry every symbolic value as a
``WireDag``. FPV constants used to travel as JSON numbers: serde_json writes
NaN and infinities as ``null`` and cannot read them back, so a snapshot failed
to load and a parallel run dropped the state, and its default float parser
moved some finite doubles by one ULP. They now travel by bit pattern.
"""

from __future__ import annotations

import math
import random
import struct

import pytest

from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

FAUXWARE_ACCEPTED_ADDR = 0x4006ED
STORE_ADDR = 0x600000


def _fp_term(value):
    """A term that keeps ``FPV(value)`` symbolic, so it is not folded away."""
    x = claripy.FPS("x", claripy.FSORT_DOUBLE)
    zero = claripy.FPV(0.0, claripy.FSORT_DOUBLE)
    return claripy.fpToIEEEBV(claripy.If(claripy.fpLT(x, zero), claripy.FPV(value, claripy.FSORT_DOUBLE), x))


def _awkward_doubles():
    rng = random.Random(1)
    randoms = [struct.unpack("<d", struct.pack("<Q", rng.getrandbits(62)))[0] for _ in range(40)]
    return [-math.inf, math.inf, math.nan, -0.0, 5e-324, 2.2250738585072014e-308, 0.1, 1 / 3, *randoms]


class TestFpvWireRoundTrip:
    def test_snapshot_keeps_every_fpv_constant(self, fauxware_project, tmp_path):
        values = _awkward_doubles()
        state = fauxware_project.factory.entry_state()
        for i, value in enumerate(values):
            state.memory.store(STORE_ADDR + 8 * i, _fp_term(value))
        before = [str(state.memory.load(STORE_ADDR + 8 * i, 8)) for i in range(len(values))]

        path = tmp_path / "fpv.snap"
        RustExplorationManager(fauxware_project, [state]).dump_snapshot(str(path))
        loaded = RustExplorationManager(fauxware_project, [fauxware_project.factory.entry_state()])
        loaded.load_snapshot(str(path))

        after_state = loaded.active[0]
        after = [str(after_state.memory.load(STORE_ADDR + 8 * i, 8)) for i in range(len(values))]
        assert after == before

    @pytest.mark.parametrize("value", [-math.inf, math.nan, 0.1])
    def test_parallel_run_keeps_a_state_holding_the_constant(self, fauxware_project, value, monkeypatch):
        monkeypatch.setenv("RUST_PARALLEL_WORKERS", "2")
        state = fauxware_project.factory.entry_state()
        state.memory.store(STORE_ADDR, _fp_term(value))
        mgr = RustExplorationManager(fauxware_project, [state])
        assert mgr.stats["parallel_real_workers"] == 2
        mgr.explore(find=FAUXWARE_ACCEPTED_ADDR, num_find=1)
        assert len(mgr.found) == 1
        assert str(mgr.found[0].memory.load(STORE_ADDR, 8)) == str(state.memory.load(STORE_ADDR, 8))
