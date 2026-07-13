# pylint: disable=missing-class-docstring,no-self-use
"""Wide (>64-bit) eval on a Rust-engine state must come from ONE model (angr-ue4ro).

Two ways the old code handed back a value that satisfied no constraint set:

* ``SymContext::eval_wide`` ran a fresh ``check()`` per call and ignored the
  model cache, so evaluating the same 256-bit stdin BVS twice on the same state
  returned two different models --- and ``solver.eval(bvs)`` disagreed with
  ``posix.dumps(0)``.
* ``RustSolverFallback._rust_eval``'s >64-bit path issued one ``eval()`` PER
  BYTE. Each byte satisfied its own byte-local constraints, but the
  concatenation could violate a CROSS-BYTE one. That path now goes through
  ``RustSolverContext.eval_batch``, which does one check-sat and reads every
  byte out of that single model.

Plus the silent-zeros hazard: when the Rust context is unsat, Rust eval returns
None and the Python fallback --- whose constraint set is empty, since the
constraints live in Rust --- happily evaluates the BVS to zeros. ``eval`` now
raises ``SimUnsatError`` there instead, like ``min``/``max`` already did.

Test vehicle: the pbounce synthetic. Its find gate is ``(acc & 0xff) == 0xee``
over an accumulator mixed from ``b[0]`` and ``b[1]`` --- a genuine cross-byte
constraint, so a per-byte model is observably wrong.
"""

from __future__ import annotations

import os

import claripy
import pytest

import angr
from angr.errors import SimUnsatError
from angr.exploration.rust_manager import RustExplorationManager

_SYNTH_NAME = "fork_solve_pbounce_W3_S2_M8_B1"
_SYNTH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmarks",
    "synthetic_examples",
    _SYNTH_NAME,
    _SYNTH_NAME,
)
_SEED_BYTES = 32
_U32 = 0xFFFFFFFF


def _reaches_target(data: bytes) -> bool:
    """Mirror of the synthetic's ``main``: does this stdin actually hit reach_target?"""
    s = (1 if data[0] else 0) + (2 if data[1] else 0) + (4 if data[2] else 0)
    acc = (s + 0x1234567) & _U32
    for i in (0, 1):
        acc = ((acc * 1103515245 + 12345) & _U32) ^ (acc >> 3)
        acc = (acc + data[i]) & _U32
    return (acc & 0xFF) == 0xEE


@pytest.fixture(scope="module")
def found_states():
    if not os.path.exists(_SYNTH_PATH):
        pytest.skip(f"synthetic bench binary not found: {_SYNTH_PATH}")
    project = angr.Project(_SYNTH_PATH, auto_load_libs=False)
    target = project.loader.find_symbol("reach_target").rebased_addr

    stdin_bvs = claripy.BVS("stdin", _SEED_BYTES * 8)
    state = project.factory.blank_state(addr=project.loader.find_symbol("main").rebased_addr)
    state.posix.stdin.content.append((stdin_bvs, claripy.BVV(_SEED_BYTES, state.arch.bits)))

    mgr = RustExplorationManager(project, [state])
    mgr.explore(find=target, num_find=8, n=4096)
    return list(mgr.found), stdin_bvs


def _eval_or_unsat(state, expr):
    """Return the model bytes, or None when the state is (correctly) reported unsat."""
    try:
        return state.solver.eval(expr, cast_to=bytes)
    except SimUnsatError:
        return None


class TestWideEvalSingleModel:
    def test_wide_eval_is_stable(self, found_states):
        """Two evals of the same wide BVS on the same state agree (one cached model)."""
        found, bvs = found_states
        assert found
        for state in found:
            first = _eval_or_unsat(state, bvs)
            if first is None:
                continue
            assert state.solver.eval(bvs, cast_to=bytes) == first

    def test_wide_eval_agrees_with_dumps(self, found_states):
        """solver.eval(seed_bvs) and posix.dumps(0) read off the same model."""
        found, bvs = found_states
        for state in found:
            value = _eval_or_unsat(state, bvs)
            if value is None:
                continue
            assert state.posix.dumps(0) == value

    def test_model_satisfies_cross_byte_gate(self, found_states):
        """The model must satisfy the (acc & 0xff) == 0xee gate, not just per-byte constraints.

        This is the angr-ue4ro regression: a per-byte model gets the branched-on
        bytes right and the gate wrong. A state whose Rust context is unsat must
        raise rather than hand back zeros.
        """
        found, bvs = found_states
        checked = 0
        for state in found:
            value = _eval_or_unsat(state, bvs)
            if value is None:
                continue
            assert _reaches_target(value), f"model {value[:4].hex()} does not reach the target"
            checked += 1
        assert checked, "no satisfiable found state produced a model"

    def test_unsat_state_never_evals_to_silent_zeros(self, found_states):
        """An unsat Rust context raises SimUnsatError instead of falling back to an empty solver."""
        found, bvs = found_states
        for state in found:
            if state.solver.satisfiable():
                continue
            with pytest.raises(SimUnsatError):
                state.solver.eval(bvs, cast_to=bytes)


class TestFoundStatesAreSatisfiable:
    """No unsat state may reach the found stash (angr-3ag1l).

    ``resume_after_symbolic_branch`` used to prime both branch states'
    sat cache with ``true``, on the strength of a ``can_be_true`` /
    ``can_be_false`` check that the non-deferred symbolic-branch path in
    ``statements.rs`` never runs. Every downstream ``satisfiable()`` gate then
    hit that poisoned cache, so the two infeasible pbounce leaves (b[0]==0 and
    b[1]==0 force a constant accumulator whose low byte can never be 0xee)
    were reported as found.
    """

    def test_every_found_state_is_satisfiable(self, found_states):
        found, _ = found_states
        assert found
        unsat = [i for i, s in enumerate(found) if not s.solver.satisfiable()]
        assert not unsat, f"found stash carries unsat states at indices {unsat}"

    def test_every_found_model_reaches_the_target(self, found_states):
        """The acceptance check: feed each model back in as concrete stdin."""
        found, _ = found_states
        for state in found:
            data = state.posix.dumps(0)
            assert _reaches_target(data), f"found-state model {data[:4].hex()} misses reach_target"

    def test_infeasible_leaves_are_never_found(self, found_states):
        """Leaves s=0 / s=4 (b[0]==0 and b[1]==0) are unsat and must not appear."""
        found, _ = found_states
        for state in found:
            data = state.posix.dumps(0)
            assert data[0] or data[1], "an infeasible (b[0]==0, b[1]==0) leaf reached the found stash"
