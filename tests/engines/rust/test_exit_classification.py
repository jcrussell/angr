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
* Python no-return procedure exit (angr-mx00p.132.8): a state a Python
  ``SimProcedure.exit`` ends reports ``Ijk_Exit`` and the ``terminate``
  event's status, also when the exit-only fast path skips the procedure.
"""

from __future__ import annotations

import os

import pytest

import angr
from tests.engines.conftest import FORK_TREE_8_PATH, RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")


class TestFallThroughOutOfImage:
    def test_fall_through_keeps_state_and_registers(self):
        # 0x1ffc: ldrneh r0, [r1] — the block's fall-through (0x2000) is past
        # the end of the 4-byte image and of its page (the rest of the page
        # is zero padding, which both engines execute; angr-vmi2l.159).
        proj = angr.load_shellcode(bytes.fromhex("b000d111"), arch="ARMEL", load_address=0x1FFC)
        state = proj.factory.blank_state(addr=0x1FFC)
        state.regs.r1 = 0x4000
        state.memory.store(0x4000, 0xABCD, size=2, endness="Iend_LE")
        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=1)
        # Rust's step chains on into 0x2000, which has no code, so the state
        # may end deadended there rather than active; what matters is that it
        # sits at the fall-through target with the loaded value intact.
        finals = list(mgr.active) + list(mgr.deadended) + [e.state for e in mgr.errored]
        assert len(finals) == 1, {k: len(v) for k, v in mgr.stashes.items() if v}
        s = finals[0]
        assert s.solver.eval(s.regs.pc) == 0x2000
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
        binary = FORK_TREE_8_PATH
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


class TestPythonProcedureExit:
    @staticmethod
    def _assert_exited(state, exit_code):
        assert state.history.jumpkind == "Ijk_Exit"
        terminations = [event for event in state.history.events if event.type == "terminate"]
        assert len(terminations) == 1
        assert state.solver.eval_one(terminations[0].objects["exit_code"]) == exit_code

    def test_fastfail_reports_exit_and_status(self):
        # __fastfail is a Python NO_RET procedure; the first state runs it,
        # the second hits the cached exit-only fast path.
        proj = angr.load_shellcode(b"\x00" * 4, arch="aarch64", simos="windows")
        states = [proj.factory.call_state(proj.simos.fastfail.addr, 0) for _ in range(2)]
        mgr = RustExplorationManager(proj, states)
        mgr.run()
        assert len(mgr.deadended) == 2
        for state in mgr.deadended:
            self._assert_exited(state, 0xC0000409)
            assert list(state.history.bbl_addrs)[-1] == proj.simos.fastfail.addr


class TestVsyscallHookSyscall:
    """A Python ``_vsyscall`` hook leaves its successor at an ``Ijk_Sys`` exit
    whose ip ``SimSuccessors`` pointed at the kernel stub; Rust must run the
    syscall and continue at its return address, not lift the kernel object
    (angr-mx00p.132.7)."""

    VSYSCALL = 0x100

    def _run(self, code):
        proj = angr.load_shellcode(code.ljust(self.VSYSCALL, b"\x90"), arch="x86", simos="linux")
        proj.hook(self.VSYSCALL, angr.SIM_PROCEDURES["linux_kernel"]["_vsyscall"]())
        mgr = RustExplorationManager(proj, [proj.factory.blank_state(addr=0)])
        mgr.run()
        return proj, mgr

    def _call_vsyscall(self, at):
        return b"\xe8" + (self.VSYSCALL - (at + 5)).to_bytes(4, "little", signed=True)

    def test_exit_group_through_vsyscall_terminates(self):
        code = b"\xbb\x07\x00\x00\x00\xb8\xfc\x00\x00\x00"  # mov ebx, 7; mov eax, exit_group
        code += self._call_vsyscall(len(code))
        _, mgr = self._run(code)
        assert len(mgr.deadended) == 1, {k: len(v) for k, v in mgr.stashes.items() if v}
        TestPythonProcedureExit._assert_exited(mgr.deadended[0], 7)

    def test_syscall_through_vsyscall_returns_after_the_call(self):
        # getpid, then exit_group with its result as the status.
        code = b"\xb8\x14\x00\x00\x00"  # mov eax, getpid
        code += self._call_vsyscall(len(code))
        code += b"\x89\xc3\xb8\xfc\x00\x00\x00"  # mov ebx, eax; mov eax, exit_group
        code += self._call_vsyscall(len(code))
        _, mgr = self._run(code)
        assert len(mgr.deadended) == 1, {k: len(v) for k, v in mgr.stashes.items() if v}
        state = mgr.deadended[0]
        terminations = [event for event in state.history.events if event.type == "terminate"]
        assert len(terminations) == 1
        assert terminations[0].objects["exit_code"].concrete_value == state.posix.pid
