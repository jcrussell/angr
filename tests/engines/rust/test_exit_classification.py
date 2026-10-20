# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests for how the Rust engine classifies block exits.

* Out-of-image fall-through (angr-et24a): a plain fall-through to an address
  outside the loaded image must not be treated as an unmodeled call — a
  "generic skip" to a return address of 0 that also zeroes the return
  register. Python just leaves the state at the fall-through target.
* SimProcedure return (angr-x39ae): a SimProcedure return must pop the
  call-stack frame its call pushed; otherwise a later ``ret`` to a symbolic
  return address no longer sees an empty call stack, misses the
  unconstrained-return guard, and the state is deadended instead of landing
  in ``unconstrained``. The manager must also keep unconstrained states by
  default, like Python.
"""

from __future__ import annotations

import os

import pytest

import angr
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")


class TestFallThroughOutOfImage:
    def test_fall_through_keeps_state_and_registers(self):
        # 0x1000: ldrneh r0, [r1] — the block's fall-through (0x1004) is past
        # the end of the 4-byte image.
        proj = angr.load_shellcode(bytes.fromhex("b000d111"), arch="ARMEL", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        state.regs.r1 = 0x4000
        state.memory.store(0x4000, 0xABCD, size=2, endness="Iend_LE")
        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=1)
        # Rust's step chains on into 0x1004, which has no code, so the state
        # may end deadended there rather than active; what matters is that it
        # sits at the fall-through target with the loaded value intact.
        finals = list(mgr.active) + list(mgr.deadended) + [e.state for e in mgr.errored]
        assert len(finals) == 1, {k: len(v) for k, v in mgr.stashes.items() if v}
        s = finals[0]
        assert s.solver.eval(s.regs.pc) == 0x1004
        assert s.solver.eval(s.regs.r0) == 0xABCD


class TestReturnThroughSymbolicAddress:
    def _call_then_ret(self, **mgr_kwargs):
        # 0x400000: call 0x400006 ; 0x400005: ret
        # 0x400006: jmp 0x400010    ; a PLT-style stub, like cow_fork_scaling's read@plt
        # 0x400010: (hooked) f
        code = bytes.fromhex("e801000000c3eb08") + b"\x90" * 8 + b"\x90"
        proj = angr.load_shellcode(code, arch="amd64", load_address=0x400000)
        proj.hook(0x400010, angr.SIM_PROCEDURES["stubs"]["ReturnUnconstrained"]())
        state = proj.factory.blank_state(addr=0x400000)
        mgr = RustExplorationManager(proj, [state], **mgr_kwargs)
        mgr.run()
        return mgr

    def test_ret_after_hooked_call_is_unconstrained(self):
        mgr = self._call_then_ret(save_unconstrained=True)
        assert len(mgr.unconstrained) == 1, {k: len(v) for k, v in mgr.stashes.items() if v}
        assert not mgr.deadended

    def test_unconstrained_kept_by_default(self):
        """Stock SimulationManager keeps unconstrained states by default."""
        mgr = self._call_then_ret()
        assert len(mgr.unconstrained) == 1, {k: len(v) for k, v in mgr.stashes.items() if v}

    def test_cow_fork_scaling_leaves_are_unconstrained(self):
        here = os.path.dirname(os.path.abspath(__file__))
        binary = os.path.join(here, "..", "..", "benchmarks", "synthetic_examples", "cow_fork_scaling", "fork_tree_8")
        if not os.path.exists(binary):
            pytest.skip("fork_tree_8 not present")
        from angr import claripy

        proj = angr.Project(binary, auto_load_libs=False)
        state = proj.factory.blank_state(addr=proj.loader.find_symbol("main").rebased_addr)
        stdin = claripy.BVS("stdin", 16 * 8)
        state.posix.stdin.content.append((stdin, claripy.BVV(16, state.arch.bits)))
        mgr = RustExplorationManager(proj, [state])
        mgr.run()
        assert len(mgr.unconstrained) == 256, {k: len(v) for k, v in mgr.stashes.items() if v}
        assert not mgr.deadended
