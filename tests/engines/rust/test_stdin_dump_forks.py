# pylint: disable=missing-class-docstring,no-self-use
"""``posix.dumps(0)`` must not double-count a forked state's stdin (angr-psrxs).

A materialized child SimState is built by ``.copy()``-ing its parent's cached
state (the parent-root path in ``rust_state_export._materialize_state``), so it
inherits whatever stdin packet Rust already injected for the parent. Rust's
per-state stdin symbol list is *cumulative* — it covers every byte the parent
read too — so appending the child's packet on top of the inherited one made
``posix.dumps(0)`` return two copies of the read: a junk prefix (the parent's
eval, made under the parent's weaker path condition) followed by the child's
real bytes.

Seen on the ``unmapped_analysis`` angr-example, whose own ``test()`` asserts the
two 40-byte segfault keys and got 80-byte dumps under the Rust engine. Vehicle
here is the pbounce synthetic, stepped one block at a time while reading each
active state's ``dumps(0)`` — the read caches a materialized parent, and every
state forked off it afterwards is a copy of that cached frame.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

_SYNTH_NAME = "fork_solve_pbounce_W3_S2_M8_B1"
_SYNTH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmarks",
    "synthetic_examples",
    _SYNTH_NAME,
    _SYNTH_NAME,
)
_READ_BYTES = 32  # the synthetic's single read(0, b, 32)
_LEAVES = 8


class _TrapPoint(angr.SimProcedure):
    """Identity hook for ``trap_point(x) -> x`` -- forces a Python bounce."""

    def run(self, x):  # pylint: disable=arguments-differ
        return x


@pytest.fixture(scope="module")
def stepped_dumps():
    """Step block-by-block, reading every active state's stdin dump as we go.

    Reading ``dumps(0)`` materializes and caches the state, so the states forked
    off it on later steps are copies that already carry its injected packet.
    Returns (dumps seen while stepping, dumps of the found leaves).
    """
    if not os.path.exists(_SYNTH_PATH):
        pytest.skip(f"synthetic bench binary not found: {_SYNTH_PATH}")
    project = angr.Project(_SYNTH_PATH, auto_load_libs=False)
    project.hook(project.loader.find_symbol("trap_point").rebased_addr, _TrapPoint())
    target = project.loader.find_symbol("reach_target").rebased_addr

    mgr = RustExplorationManager(project, [project.factory.entry_state()])
    stepped = []
    for _ in range(24):
        mgr.step(n=1)
        if not mgr.active:
            break
        stepped.extend(state.posix.dumps(0) for state in mgr.active)
    mgr.explore(find=target, num_find=_LEAVES, n=2048)
    return stepped, [state.posix.dumps(0) for state in mgr.found]


class TestForkedStdinDump:
    def test_stepped_dumps_are_one_read_each(self, stepped_dumps):
        """A state inspected mid-run dumps exactly its single read(0, ., 32)."""
        stepped, _ = stepped_dumps
        assert stepped
        assert all(len(d) == _READ_BYTES for d in stepped), (
            "a state forked off a materialized parent must not inherit its stdin packet"
        )

    def test_found_dumps_are_one_read_each(self, stepped_dumps):
        """The found leaves dump their own solution, not a parent-prefixed one."""
        _, found = stepped_dumps
        assert found
        for data in found:
            assert len(data) == _READ_BYTES
            # The find gate needs a nonzero byte among the branched-on three; a
            # stale parent prefix would put its own (weaker) eval in front.
            assert any(data[:3]), "dump must satisfy this leaf's own path condition"
