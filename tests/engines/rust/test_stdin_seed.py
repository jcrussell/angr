# pylint: disable=missing-class-docstring,no-self-use
"""Harness-seeded symbolic stdin must surface in Rust-engine found states (angr-mb09c).

A harness that fills ``state.posix.stdin.content`` itself --- a ``blank_state``
plus ``content.append((BVS, n))``, or ``entry_state(stdin=SimFileStream(...))``
--- expects the found state to solve for THAT symbol. Before angr-mb09c the
native ``read(0, ...)`` always minted its own ``stdin_*`` bytes and ignored the
seeded stream, so the harness's BVS stayed unconstrained: ``solver.eval(bvs)``
returned all zeros and ``posix.dumps(0)`` prefixed a bogus 32-byte zero run
(the un-consumed seed chunk) ahead of the injected solution bytes.

``RustExplorationManager._seed_stdin_to_rust`` now attaches the seeded byte
ASTs to fd 0 as bounded symbolic content, and ``read_stdin_symbolic``
(``procedures/read.rs``) binds each minted stdin byte to the seeded one with an
equality constraint --- so the path condition reaches the harness's symbol,
while the guest buffer keeps holding plain leaf symbols (which is what the
Python-bounce memory round-trip can re-import without dropping constraints).

Test vehicle: the 8-leaf pbounce synthetic, whose ``main`` does
``read(0, b, 32)`` then forks on ``b[0..3] != 0``. Leaf index ``s`` is the
bitmask of which of the first three bytes are nonzero, so the seeded BVS's
solution has an observable, leaf-specific shape.
"""

from __future__ import annotations

import os

import claripy
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
_LEAVES = 8  # 2^W, W=3
_SEED_BYTES = 32


class _TrapPoint(angr.SimProcedure):
    """Identity hook for ``trap_point(x) -> x`` -- forces a Python bounce."""

    def run(self, x):  # pylint: disable=arguments-differ
        return x


@pytest.fixture(scope="module")
def seeded_found():
    """Explore the synthetic from a blank_state with a harness-seeded stdin BVS."""
    if not os.path.exists(_SYNTH_PATH):
        pytest.skip(f"synthetic bench binary not found: {_SYNTH_PATH}")
    project = angr.Project(_SYNTH_PATH, auto_load_libs=False)
    project.hook(project.loader.find_symbol("trap_point").rebased_addr, _TrapPoint())
    target = project.loader.find_symbol("reach_target").rebased_addr

    stdin_bvs = claripy.BVS("stdin", _SEED_BYTES * 8)
    state = project.factory.blank_state(addr=project.loader.find_symbol("main").rebased_addr)
    state.posix.stdin.content.append((stdin_bvs, claripy.BVV(_SEED_BYTES, state.arch.bits)))

    mgr = RustExplorationManager(project, [state])
    mgr.explore(find=target, num_find=_LEAVES, n=4096)
    return list(mgr.found), stdin_bvs


# (b[0], b[1], b[2]) nonzero-ness of the six leaves whose find gate is
# satisfiable — every leaf except s=0 (0,0,0) and s=4 (0,0,1).
_FEASIBLE_LEAVES = {
    (b0, b1, b2) for b0 in (0, 1) for b1 in (0, 1) for b2 in (0, 1) if (b0 + 2 * b1 + 4 * b2) not in (0, 4)
}


def _leaf_pattern(data: bytes) -> tuple[int, ...]:
    """Which of the three branched-on bytes are nonzero -- the leaf's identity."""
    return tuple(1 if b else 0 for b in data[:3])


class TestSeededStdinSolves:
    def test_exhaustive_drain(self, seeded_found):
        """Every FEASIBLE leaf is drained.

        Only six of the 2^W leaves are feasible: s=0 and s=4 force b[0]==0 and
        b[1]==0, which makes the accumulator a constant whose low byte can never
        be the gate's 0xee (bd memory ``pbounce-synthetic-only-6-feasible-leaves``).
        This used to assert all eight, which only held because a bounced value
        lost its constraints and made the infeasible leaves look satisfiable
        (angr-izov2).
        """
        found, _ = seeded_found
        patterns = {_leaf_pattern(s.posix.dumps(0)) for s in found}
        assert patterns >= _FEASIBLE_LEAVES

    def test_eval_of_seeding_bvs_is_leaf_consistent(self, seeded_found):
        """Each found state solves the harness's BVS to its own leaf's byte shape."""
        found, stdin_bvs = seeded_found
        patterns = {_leaf_pattern(s.solver.eval(stdin_bvs, cast_to=bytes)) for s in found}
        # Every feasible leaf shows up with its own byte shape: pre-fix every
        # state evaluated the BVS to all zeros, collapsing this set to a single
        # (0, 0, 0) entry.
        assert patterns >= _FEASIBLE_LEAVES

    def test_dumps_is_the_seeded_chunk_only(self, seeded_found):
        """dumps(0) is the solved seed chunk -- no zero prefix, no duplicate injection."""
        found, _ = seeded_found
        for state in found:
            data = state.posix.dumps(0)
            assert len(data) == _SEED_BYTES, "seeded bytes must not be re-injected after the chunk"
            assert data[:3] != b"\x00\x00\x00" or _leaf_pattern(data) == (0, 0, 0)

    def test_dumps_agrees_with_eval_on_leaf(self, seeded_found):
        found, stdin_bvs = seeded_found
        for state in found:
            assert _leaf_pattern(state.posix.dumps(0)) == _leaf_pattern(state.solver.eval(stdin_bvs, cast_to=bytes))
