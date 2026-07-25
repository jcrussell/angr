"""Regression tests for caller-SimState mutation isolation (angr-hv4lt.7).

RustExplorationManager construction must not mutate the SimState objects the
caller passes in. Historically ``_add_rust_state`` concretized a symbolic
stack pointer / base pointer *in place* and, for a symbolic sp, appended a
permanent ``sp == <concrete>`` equality constraint to the caller's own solver
— silently corrupting the symbolic identity + constraint set of any object the
caller still held a reference to.

The fix operates on a private copy inside ``_add_rust_state``; Rust still sees
the concretized+constrained view (the copy becomes the cached Python mirror),
but the caller's original stays pristine.
"""

from __future__ import annotations

import claripy
import pytest

import angr
from tests.engines.conftest import (
    RUST_EXPLORATION_AVAILABLE,
    RustExplorationManager,
)

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


class TestCallerStateMutationIsolation:
    def test_symbolic_sp_not_concretized_on_caller(self):
        """A symbolic rsp stays symbolic and unconstrained on the caller's
        state after merely constructing the manager.

        Pre-fix: rsp flips to concrete 0x7fffffff0000 and a
        ``my_sp == 0x7fffffff0000`` constraint appears on state.solver.
        """
        proj = angr.load_shellcode(b"\xc3", arch="amd64")
        state = proj.factory.blank_state(addr=0x1000)
        sym_sp = claripy.BVS("my_sp", 64)
        state.regs.rsp = sym_sp

        assert state.regs.rsp.symbolic
        n_constraints_before = len(state.solver.constraints)

        RustExplorationManager(proj, [state])

        # Caller's object is untouched: rsp still symbolic, no new constraint.
        assert state.regs.rsp.symbolic, "rsp was concretized on the caller's state"
        assert len(state.solver.constraints) == n_constraints_before, (
            "a stack-pointer equality constraint leaked onto the caller's solver"
        )

    def test_symbolic_bp_not_concretized_on_caller(self):
        """A symbolic rbp (angr's default_filler fills it on blank_state) stays
        symbolic on the caller's state after construction.

        Pre-fix: rbp flips from ``reg_rbp_0_64`` to concrete 0x0.
        """
        proj = angr.load_shellcode(b"\xc3", arch="amd64")
        state = proj.factory.blank_state(addr=0x1000)
        # default_filler_mixin fills rbp with a fresh unconstrained BVS.
        assert state.regs.rbp.symbolic

        RustExplorationManager(proj, [state])

        assert state.regs.rbp.symbolic, "rbp was concretized on the caller's state"
