"""Tests for RustExplorationManager.

This module tests the Rust-native exploration manager for symbolic execution.
"""

from __future__ import annotations

import pytest

import angr

# Rust availability guard, binary-path resolution, and the module-scoped
# fauxware_project fixture all live in tests/engines/conftest.py (angr-7gdp).
from tests.engines.conftest import (  # noqa: F401
    RUST_EXPLORATION_AVAILABLE,
    TEST_BINARIES_DIR,
    ExplorationEvent,
    PythonCallbacks,
    RustExplorationManager,
    RustSimState,
    _RustExplorationManager,
)

# All tests in this module require the Rust extension; skip the whole module
# when it is unavailable (matches tests/engines/test_rust_public_api.py).
pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


class TestSymbolicLibcProcedures:
    """Integration tests for symbolic libc SimProcedures via RustExplorationManager.

    Native Rust unit tests in `native/angr/src/procedures/*.rs` exercise each
    procedure directly on a `RustSimState`. These Python-level tests cover the
    surrounding integration path that the Rust tests do not: claripy↔Z3
    bridging, ITE-chain construction across the FFI boundary, callback
    dispatch + name-matching from `proj._sim_procedures`, Z3 timeout, and
    final rax export back to the Python `SimState`.

    Each test hooks fauxware's `.fini` area (a mapped-but-unused address) with
    a Python angr SimProcedure, sets up registers + memory with symbolic
    input, runs one step on the Rust manager, and verifies rax constraints.

    The hook address is intentionally inside fauxware's loaded segment so the
    Rust manager's `_run_python_init_if_needed` short-circuits — otherwise an
    out-of-binary PC triggers Python init to main and the libc procedure
    never runs.
    """

    HOOK_ADDR = 0x4008C0  # in fauxware's mapped .fini area, no normal exec
    RET_ADDR = 0x4008B0  # mapped, used as bogus ret target after the hook
    BUF_ADDR = 0x601100  # past .bss, lazy-mapped via filler mixin

    def _make_state(self, proj, sim_proc):
        import claripy

        proj.hook(self.HOOK_ADDR, sim_proc, replace=True)
        state = proj.factory.blank_state(
            addr=self.HOOK_ADDR,
            add_options={
                angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
            },
        )
        state.memory.store(state.regs.rsp, claripy.BVV(self.RET_ADDR, 64), endness="Iend_LE")
        return state

    def _run_one_step(self, proj, state):
        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=1)
        states = mgr.active + mgr.deadended + mgr.errored
        assert len(states) == 1, (
            f"expected exactly one post-call state, got {len(states)} (stash counts: {mgr.stash_counts()})"
        )
        return states[0]

    def test_strlen_symbolic_constrained_min_max(self, fauxware_project):
        """strlen on a 5-byte symbolic buffer (each byte constrained non-null,
        terminator at offset 5) must produce min=max=5."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["strlen"]())
            sym = claripy.BVS("strlen_input", 8 * 5)
            state.memory.store(self.BUF_ADDR, sym)
            state.memory.store(self.BUF_ADDR + 5, claripy.BVV(0, 8))
            for i in range(5):
                state.solver.add(sym.get_byte(i) != 0)
            state.regs.rdi = self.BUF_ADDR
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 5
            assert s.solver.max(s.regs.rax) == 5
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_strcmp_symbolic_constrained_equal(self, fauxware_project):
        """strcmp(s1, s2) where s1[0] is symbolic and s2 is "X\\0", with
        s1[0] constrained to 'X', must return 0 (equal)."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["strcmp"]())
            sym = claripy.BVS("strcmp_b", 8)
            state.memory.store(self.BUF_ADDR, sym)
            state.memory.store(self.BUF_ADDR + 1, claripy.BVV(0, 8))
            state.memory.store(self.BUF_ADDR + 0x10, b"X\x00")
            state.regs.rdi = self.BUF_ADDR
            state.regs.rsi = self.BUF_ADDR + 0x10
            state.solver.add(sym == ord("X"))
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 0
            assert s.solver.max(s.regs.rax) == 0
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_strchr_symbolic_finds_target(self, fauxware_project):
        """strchr("a?bX\\0", 'X') with '?' symbolic constrained to non-X,
        non-null must return BUF_ADDR + 3 (the index of 'X')."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["strchr"]())
            state.memory.store(self.BUF_ADDR, b"a")
            sym = claripy.BVS("strchr_q", 8)
            state.memory.store(self.BUF_ADDR + 1, sym)
            state.memory.store(self.BUF_ADDR + 2, b"bX\x00")
            state.solver.add(sym != ord("X"))
            state.solver.add(sym != 0)
            state.regs.rdi = self.BUF_ADDR
            state.regs.rsi = ord("X")
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == self.BUF_ADDR + 3
            assert s.solver.max(s.regs.rax) == self.BUF_ADDR + 3
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_memchr_symbolic_finds_target(self, fauxware_project):
        """memchr(buf, 'X', 5) where buf[2] is symbolic constrained to 'X'
        must return BUF_ADDR + 2."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["memchr"]())
            state.memory.store(self.BUF_ADDR, b"AB")
            sym = claripy.BVS("memchr_q", 8)
            state.memory.store(self.BUF_ADDR + 2, sym)
            state.memory.store(self.BUF_ADDR + 3, b"CD")
            state.solver.add(sym == ord("X"))
            state.regs.rdi = self.BUF_ADDR
            state.regs.rsi = ord("X")
            state.regs.rdx = 5
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == self.BUF_ADDR + 2
            assert s.solver.max(s.regs.rax) == self.BUF_ADDR + 2
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_memcmp_symbolic_constrained_equal(self, fauxware_project):
        """memcmp(b1, b2, 4) where each buffer's middle byte is a distinct
        symbolic, constrained equal — must return 0."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["memcmp"]())
            sym1 = claripy.BVS("memcmp_b1", 8)
            sym2 = claripy.BVS("memcmp_b2", 8)
            state.memory.store(self.BUF_ADDR, b"AB")
            state.memory.store(self.BUF_ADDR + 2, sym1)
            state.memory.store(self.BUF_ADDR + 3, b"D")
            state.memory.store(self.BUF_ADDR + 0x10, b"AB")
            state.memory.store(self.BUF_ADDR + 0x12, sym2)
            state.memory.store(self.BUF_ADDR + 0x13, b"D")
            state.solver.add(sym1 == sym2)
            state.regs.rdi = self.BUF_ADDR
            state.regs.rsi = self.BUF_ADDR + 0x10
            state.regs.rdx = 4
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 0
            assert s.solver.max(s.regs.rax) == 0
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_atoi_symbolic_constrained_to_value(self, fauxware_project):
        """atoi("7\\0") with '7' as a symbolic byte constrained to ASCII '7'
        must produce min=max=7."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["atoi"]())
            sym = claripy.BVS("atoi_d", 8)
            state.memory.store(self.BUF_ADDR, sym)
            state.memory.store(self.BUF_ADDR + 1, claripy.BVV(0, 8))
            state.solver.add(sym == ord("7"))
            state.regs.rdi = self.BUF_ADDR
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 7
            assert s.solver.max(s.regs.rax) == 7
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_strtol_symbolic_hex_digit(self, fauxware_project):
        """strtol("a\\0", NULL, 16) with 'a' as a symbolic byte constrained
        to ASCII 'a' must produce min=max=10."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["strtol"]())
            sym = claripy.BVS("strtol_d", 8)
            state.memory.store(self.BUF_ADDR, sym)
            state.memory.store(self.BUF_ADDR + 1, claripy.BVV(0, 8))
            state.solver.add(sym == ord("a"))
            state.regs.rdi = self.BUF_ADDR
            state.regs.rsi = 0
            state.regs.rdx = 16
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 10
            assert s.solver.max(s.regs.rax) == 10
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_strtoul_symbolic_decimal_digit(self, fauxware_project):
        """strtoul("5\\0", NULL, 10) with '5' as a symbolic byte constrained
        to ASCII '5' must produce min=max=5."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["strtoul"]())
            sym = claripy.BVS("strtoul_d", 8)
            state.memory.store(self.BUF_ADDR, sym)
            state.memory.store(self.BUF_ADDR + 1, claripy.BVV(0, 8))
            state.solver.add(sym == ord("5"))
            state.regs.rdi = self.BUF_ADDR
            state.regs.rsi = 0
            state.regs.rdx = 10
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 5
            assert s.solver.max(s.regs.rax) == 5
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_isdigit_symbolic_returns_true_for_digit(self, fauxware_project):
        """isdigit on a 32-bit symbolic int constrained to ASCII '5' must
        return rax = 1."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["isdigit"]())
            sym = claripy.BVS("isdigit_x", 32)
            state.solver.add(sym == ord("5"))
            state.regs.rdi = sym.zero_extend(32)
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 1
            assert s.solver.max(s.regs.rax) == 1
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_isalpha_symbolic_returns_true_for_letter(self, fauxware_project):
        """isalpha on a 32-bit symbolic int constrained to ASCII 'a' must
        return rax = 1."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["isalpha"]())
            sym = claripy.BVS("isalpha_x", 32)
            state.solver.add(sym == ord("a"))
            state.regs.rdi = sym.zero_extend(32)
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 1
            assert s.solver.max(s.regs.rax) == 1
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_posix_fork_symbolic_flag_returns_both_branches(self, fauxware_project):
        """angr-q6r1: posix.fork SimProcedure dispatches correctly under the
        Rust manager and the symbolic-flag return value yields both the
        parent pid (1338) and the child (0) branches.

        Python angr's ``posix.fork`` is symbolic-flag-only: it returns
        ``If(BoolS("fork_parent"), BVV(1338, sizeof[int]), BVV(0, sizeof[int]))``
        without actually forking the SimState. Both engines share that
        semantics; this test confirms the SimProcedure runs end-to-end under
        the Rust engine (claripy ITE round-trips through the FFI boundary,
        the return value lands in rax) and documents the v1.0 contract.
        """
        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["posix"]["fork"]())
            s = self._run_one_step(proj, state)
            # SimProcedure return value is an int (32-bit on amd64), which the
            # calling convention zero-extends to 64-bit rax. Both branches of
            # the symbolic-flag ITE must concretize.
            assert s.solver.min(s.regs.rax) == 0
            assert s.solver.max(s.regs.rax) == 1338
        finally:
            proj.unhook(self.HOOK_ADDR)


class TestNativeFileDescriptorProcedures:
    """Integration test for native pipe/dup/dup2 dispatched through the Rust manager.

    Native open/close/lseek/dup/dup2/pipe are registered by default in
    `NativeProcedureRegistry::new` and live in
    `native/angr/src/procedures/fileops.rs`. Cargo unit tests in that module
    cover concrete arg handling and error returns. This Python-level test
    asserts the full integration path: the manager dispatches the native
    procedure when a hook is set at an out-of-binary address (i.e. extern
    object range, which is skipped by `_load_binary_regions`), the procedure
    mutates the per-state `FileSystem`, and the FD layout observed via
    `get_state_open_fds` matches POSIX semantics for `pipe(2) + dup2(2)`.
    """

    # Hook addrs must satisfy two things:
    #   1. `find_object_containing(addr)` returns a real binary (not cle##*),
    #      so `_run_python_init_if_needed` short-circuits and our state actually
    #      starts at the hook (otherwise an extern/loader addr triggers Python
    #      init from main first).
    #   2. The addr is NOT in `binary_regions` — only executable sections of
    #      real binaries are loaded there, so a hook in fauxware's `.ctors`
    #      (non-executable, in-binary) qualifies. `handle_simprocedure` then
    #      tries the native registry first and dispatches NativePipe / NativeDup2.
    PIPE_ADDR = 0x600E30  # in fauxware's .ctors (non-executable)
    DUP2_ADDR = 0x600E3C
    # Bottom-of-call return target: any address that won't loop back into a
    # hook. We run for max_steps that just covers the procedure dispatches —
    # afterwards the state's PC lands here and we stop without lifting blocks.
    DEAD_ADDR = 0x4008B0
    BUF_ADDR = 0x601100  # past .bss, lazy-mapped

    @staticmethod
    def _all_fds(mgr):
        """Collect FD info from any stash that holds states. After the native
        procedure dispatches and sets PC to DEAD_ADDR, the next step may walk
        into a ret-with-empty-call-stack and land the state in 'unconstrained'
        (angr-3uye.2); the manager must be built with save_unconstrained=True
        for the state to remain inspectable. 'deadended'/'errored' are also
        possible if the post-DEAD_ADDR block lifts cleanly but ends terminally."""
        for stash in ("active", "deadended", "errored", "unconstrained"):
            ids = mgr._rust_mgr.get_state_ids(stash)
            if ids:
                sid = ids[0]
                return sid, {
                    fd: (name, flags, is_open)
                    for fd, name, _pos, flags, _len, is_open in mgr._rust_mgr.get_state_open_fds(sid)
                }
        raise AssertionError(f"no state in any stash: {mgr.stash_counts()}")

    def test_pipe_native_dispatch_creates_two_fds(self, fauxware_project):
        """pipe(buf) dispatched through the native registry must allocate two
        consecutive fds (read end at 3, write end at 4) with the correct flags."""
        import claripy

        proj = fauxware_project

        class pipe(angr.SimProcedure):
            num_args = 1

            def run(self, pipefd):  # pylint: disable=arguments-differ
                return 0

        proj.hook(self.PIPE_ADDR, pipe(), replace=True)
        try:
            state = proj.factory.blank_state(
                addr=self.PIPE_ADDR,
                add_options={
                    angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                    angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                },
            )
            state.regs.rdi = self.BUF_ADDR
            state.memory.store(state.regs.rsp, claripy.BVV(self.DEAD_ADDR, 64), endness="Iend_LE")
            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            # Confirm native dispatch fired (not Python fallback).
            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("pipe", 0) == 1, f"expected native pipe dispatch, got stats={stats}"

            # FD layout: pre-existing 0,1,2 + newly allocated 3 (read) and 4 (write).
            _sid, fds = self._all_fds(mgr)
            assert set(fds.keys()) == {0, 1, 2, 3, 4}, f"unexpected fd set after pipe: {sorted(fds)}"
            # ReadOnly=0, WriteOnly=1 (FdFlags::to_posix).
            assert fds[3][0] == "<pipe:r>" and fds[3][1] == 0 and fds[3][2]
            assert fds[4][0] == "<pipe:w>" and fds[4][1] == 1 and fds[4][2]
        finally:
            proj.unhook(self.PIPE_ADDR)

    def test_dup2_native_dispatch_redirects_stdin(self, fauxware_project):
        """dup2(0, 7) dispatched natively must create fd 7 as a copy of fd 0
        (stdin), with the original /dev/stdin name preserved."""
        import claripy

        proj = fauxware_project

        class dup2(angr.SimProcedure):
            num_args = 2

            def run(self, oldfd, newfd):  # pylint: disable=arguments-differ
                return newfd

        proj.hook(self.DUP2_ADDR, dup2(), replace=True)
        try:
            state = proj.factory.blank_state(
                addr=self.DUP2_ADDR,
                add_options={
                    angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                    angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                },
            )
            state.regs.rdi = 0  # oldfd = stdin
            state.regs.rsi = 7  # newfd = 7
            state.memory.store(state.regs.rsp, claripy.BVV(self.DEAD_ADDR, 64), endness="Iend_LE")
            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("dup2", 0) == 1, f"expected native dup2 dispatch, got stats={stats}"

            _sid, fds = self._all_fds(mgr)
            # Original three plus the new fd 7.
            assert set(fds.keys()) == {0, 1, 2, 7}, f"unexpected fd set after dup2(0, 7): {sorted(fds)}"
            # fd 7 is a clone of fd 0 (stdin).
            assert fds[7][0] == "/dev/stdin" and fds[7][1] == 0 and fds[7][2]
        finally:
            proj.unhook(self.DUP2_ADDR)


class TestNativeFileDescriptorErrorReturns:
    """Error-return contract for native fileops procedures (angr-95up.3).

    Each native proc in `native/angr/src/procedures/fileops.rs` returns -1
    (the EBADF-style sentinel) when handed a never-opened fd. The Cargo
    unit tests pin the per-proc behavior; this Python-level test pins the
    full integration contract that's relevant to Python callers:

      1. The native registry dispatches the procedure (no Python fallback)
         even on the error path — `native_procedure_stats` `call_counts`
         ticks under the registered name.
      2. The return register holds the -1 sentinel (0xFFFFFFFFFFFFFFFF on
         amd64), so a C-level `if (close(fd) == -1)` check would fire.
      3. The post-call `FileSystem` is unchanged for the bad fd: the
         error path early-returns before any state mutation, so the bad
         fd does NOT silently get registered as open.
    """

    # Same in-binary non-executable address pattern as
    # TestNativeFileDescriptorProcedures (see that class docstring for
    # the address-selection rationale).
    HOOK_ADDR = 0x600E30
    DEAD_ADDR = 0x4008B0

    @staticmethod
    def _make_stub(proc_name: str, num_args: int):
        """Build a Python SimProcedure stub whose class name matches the
        native-registry key. The stub's `run` is only invoked on Python
        fallback — if native dispatch fires (as expected here), `run` is
        bypassed, but the stub still has to exist for `proj.hook` to be
        valid and for `_register_simprocedures` to forward the name to
        the Rust manager."""
        runs = {
            1: lambda self, a0: 0,
            2: lambda self, a0, a1: 0,
            3: lambda self, a0, a1, a2: 0,
        }
        return type(
            proc_name,
            (angr.SimProcedure,),
            {"num_args": num_args, "run": runs[num_args]},
        )

    @pytest.mark.parametrize(
        "proc_name,num_args,reg_args,bad_fd",
        [
            ("close", 1, (99,), 99),  # close(99) — fd never opened
            ("dup", 1, (99,), 99),  # dup(99)   — fd never opened
            ("dup2", 2, (99, 7), 7),  # dup2(99, 7) — newfd 7 must NOT be created
            ("lseek", 3, (99, 0, 0), 99),  # lseek(99, 0, SEEK_SET)
        ],
    )
    def test_native_error_returns_minus_one(
        self,
        fauxware_project,
        proc_name,
        num_args,
        reg_args,
        bad_fd,
    ):
        import claripy

        proj = fauxware_project

        stub_cls = self._make_stub(proc_name, num_args)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = proj.factory.blank_state(
                addr=self.HOOK_ADDR,
                add_options={
                    angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                    angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                },
            )
            # Load args into the AMD64 SysV argument registers in order.
            for reg, val in zip(("rdi", "rsi", "rdx"), reg_args):
                setattr(state.regs, reg, val)
            state.memory.store(
                state.regs.rsp,
                claripy.BVV(self.DEAD_ADDR, 64),
                endness="Iend_LE",
            )

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            # (1) Native dispatch fired (the Python stub's `run` was
            # bypassed). No `python_fallbacks` bump is required by the
            # contract — what matters is the call_count tick.
            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get(proc_name, 0) == 1, (
                f"expected native {proc_name} dispatch on error path, got stats={stats}"
            )

            sid = None
            for stash in ("active", "deadended", "errored", "unconstrained"):
                ids = mgr._rust_mgr.get_state_ids(stash)
                if ids:
                    sid = ids[0]
                    break
            assert sid is not None, f"no state in any stash: {mgr.stash_counts()}"

            # (2) Return value sentinel: -1 as u64 == 0xFFFFFFFFFFFFFFFF.
            rax = mgr._rust_mgr.get_state_register(sid, "rax")
            assert rax == 0xFFFFFFFFFFFFFFFF, (
                f"expected rax=-1 sentinel from native {proc_name} error path, got rax={rax:#x}"
            )

            # (3) Post-call FileSystem invariant: the bad fd was not
            # silently registered.
            fds = {
                fd: (name, flags, is_open)
                for fd, name, _pos, flags, _len, is_open in mgr._rust_mgr.get_state_open_fds(sid)
            }
            assert bad_fd not in fds, f"native {proc_name} error path must NOT register bad fd {bad_fd}; got fds={fds}"
        finally:
            proj.unhook(self.HOOK_ADDR)


class TestNativeMemoryAlignedAllocators:
    """Integration tests for NativeMemalign / NativePosixMemalign (angr-f16h.2).

    angr has no Python SimProcedure for `memalign` / `posix_memalign` — they
    are declared in `procedures/definitions/common/glibc.json` but no Python
    file backs them. The native registry is the only path that handles these
    calls, so these tests exercise both the dispatch and the alignment math.

    Same in-binary non-executable hook-addr pattern as TestNativeFile-
    DescriptorProcedures (see that class docstring for the rationale).
    """

    HOOK_ADDR = 0x600E30  # in fauxware's .ctors (non-executable, in-binary)
    DEAD_ADDR = 0x4008B0
    BUF_ADDR = 0x601100  # past .bss, lazy-mapped

    @staticmethod
    def _make_stub(proc_name: str, num_args: int):
        """Python SimProcedure stub whose class name matches the native key.
        The `run` is bypassed when native dispatch fires (the expected path),
        but the class must exist so `proj.hook` is valid."""
        runs = {
            2: lambda self, a0, a1: 0,
            3: lambda self, a0, a1, a2: 0,
        }
        return type(
            proc_name,
            (angr.SimProcedure,),
            {"num_args": num_args, "run": runs[num_args]},
        )

    def test_memalign_returns_aligned_pointer(self, fauxware_project):
        """memalign(64, 100) must dispatch natively, return an aligned
        non-zero pointer in rax, and tick the call_counts['memalign']
        counter exactly once."""
        import claripy

        proj = fauxware_project

        stub_cls = self._make_stub("memalign", 2)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = proj.factory.blank_state(
                addr=self.HOOK_ADDR,
                add_options={
                    angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                    angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                },
            )
            state.regs.rdi = 64  # alignment
            state.regs.rsi = 100  # size
            state.memory.store(state.regs.rsp, claripy.BVV(self.DEAD_ADDR, 64), endness="Iend_LE")

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("memalign", 0) == 1, f"expected native memalign dispatch, got stats={stats}"

            sid = None
            for stash in ("active", "deadended", "errored", "unconstrained"):
                ids = mgr._rust_mgr.get_state_ids(stash)
                if ids:
                    sid = ids[0]
                    break
            assert sid is not None, f"no state in any stash: {mgr.stash_counts()}"

            rax = mgr._rust_mgr.get_state_register(sid, "rax")
            assert rax != 0, "memalign returned NULL"
            assert rax % 64 == 0, f"rax={rax:#x} not aligned to 64"
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_posix_memalign_writes_pointer_and_returns_zero(self, fauxware_project):
        """posix_memalign(memptr, 32, 80) must dispatch natively, return 0
        in rax, and write an aligned pointer to *memptr in memory."""
        import claripy

        proj = fauxware_project

        stub_cls = self._make_stub("posix_memalign", 3)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = proj.factory.blank_state(
                addr=self.HOOK_ADDR,
                add_options={
                    angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                    angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                },
            )
            state.regs.rdi = self.BUF_ADDR  # memptr (void**)
            state.regs.rsi = 32  # alignment
            state.regs.rdx = 80  # size
            state.memory.store(state.regs.rsp, claripy.BVV(self.DEAD_ADDR, 64), endness="Iend_LE")

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("posix_memalign", 0) == 1, (
                f"expected native posix_memalign dispatch, got stats={stats}"
            )

            sid = None
            for stash in ("active", "deadended", "errored", "unconstrained"):
                ids = mgr._rust_mgr.get_state_ids(stash)
                if ids:
                    sid = ids[0]
                    break
            assert sid is not None, f"no state in any stash: {mgr.stash_counts()}"

            # eax holds the errno-style return; 0 on success.
            rax = mgr._rust_mgr.get_state_register(sid, "rax")
            assert rax & 0xFFFFFFFF == 0, f"expected eax=0 success, got rax={rax:#x}"

            # *memptr must hold an aligned non-zero pointer.
            stored = mgr._rust_mgr.get_state_memory(sid, self.BUF_ADDR, 8)
            stored_addr = int.from_bytes(stored, "little")
            assert stored_addr != 0, "*memptr not written"
            assert stored_addr % 32 == 0, f"*memptr={stored_addr:#x} not aligned to 32"
        finally:
            proj.unhook(self.HOOK_ADDR)


class TestNativeStringToNumericProcedures:
    """Integration tests for NativeStrtoll / NativeStrtoull / NativeStrtod
    (angr-f16h.3).

    angr has no Python SimProcedure for these three — they are declared in
    `procedures/definitions/common/glibc.json` but no Python file backs them.
    The native registry is the only path that handles these calls.

    Same in-binary non-executable hook-addr + stub-class pattern as
    TestNativeMemoryAlignedAllocators (see that class docstring for the
    rationale of using a fake hook in fauxware's `.ctors`).
    """

    HOOK_ADDR = 0x600E30  # in fauxware's .ctors (non-executable, in-binary)
    DEAD_ADDR = 0x4008B0
    STRING_ADDR = 0x601100  # past .bss, lazy-mapped
    ENDPTR_ADDR = 0x601200  # storage for *endptr

    @staticmethod
    def _make_stub(proc_name: str, num_args: int):
        runs = {
            1: lambda self, a0: 0,
            2: lambda self, a0, a1: 0,
            3: lambda self, a0, a1, a2: 0,
        }
        return type(
            proc_name,
            (angr.SimProcedure,),
            {"num_args": num_args, "run": runs[num_args]},
        )

    def _setup_state(self, proj, string_bytes: bytes, *, set_endptr: bool = False):
        import claripy

        state = proj.factory.blank_state(
            addr=self.HOOK_ADDR,
            add_options={
                angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
            },
        )
        buf = string_bytes + b"\x00"
        for i, b in enumerate(buf):
            state.memory.store(self.STRING_ADDR + i, claripy.BVV(b, 8))
        state.memory.store(state.regs.rsp, claripy.BVV(self.DEAD_ADDR, 64), endness="Iend_LE")
        return state

    def _first_state_id(self, mgr):
        for stash in ("active", "deadended", "errored", "unconstrained"):
            ids = mgr._rust_mgr.get_state_ids(stash)
            if ids:
                return ids[0]
        return None

    def test_strtoll_64bit_concrete_value(self, fauxware_project):
        """strtoll on a value beyond i32 range must produce the i64 result
        in rax via native dispatch."""
        proj = fauxware_project

        stub_cls = self._make_stub("strtoll", 3)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = self._setup_state(proj, b"9223372036854775000")
            state.regs.rdi = self.STRING_ADDR  # nptr
            state.regs.rsi = 0  # endptr (NULL)
            state.regs.rdx = 10  # base

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("strtoll", 0) == 1, f"expected native strtoll dispatch, got stats={stats}"
            sid = self._first_state_id(mgr)
            assert sid is not None, f"no state: {mgr.stash_counts()}"
            rax = mgr._rust_mgr.get_state_register(sid, "rax")
            assert rax == 9223372036854775000, f"rax={rax:#x}"
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_strtoull_value_above_i64_max(self, fauxware_project):
        """strtoull must accept values above i64::MAX (interpreted as u64)."""
        proj = fauxware_project

        stub_cls = self._make_stub("strtoull", 3)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = self._setup_state(proj, b"18446744073709551000")
            state.regs.rdi = self.STRING_ADDR
            state.regs.rsi = 0
            state.regs.rdx = 10

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("strtoull", 0) == 1, f"expected native strtoull dispatch, got stats={stats}"
            sid = self._first_state_id(mgr)
            assert sid is not None
            rax = mgr._rust_mgr.get_state_register(sid, "rax")
            assert rax == 18446744073709551000, f"rax={rax:#x}"
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_strtoll_writes_endptr_past_parsed_prefix(self, fauxware_project):
        """strtoll with a non-null endptr must store nptr + consumed bytes."""
        proj = fauxware_project

        stub_cls = self._make_stub("strtoll", 3)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = self._setup_state(proj, b"123abc")
            state.regs.rdi = self.STRING_ADDR
            state.regs.rsi = self.ENDPTR_ADDR
            state.regs.rdx = 10

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            sid = self._first_state_id(mgr)
            assert sid is not None
            rax = mgr._rust_mgr.get_state_register(sid, "rax")
            assert rax == 123
            stored = mgr._rust_mgr.get_state_memory(sid, self.ENDPTR_ADDR, 8)
            stored_addr = int.from_bytes(stored, "little")
            assert stored_addr == self.STRING_ADDR + 3, f"endptr={stored_addr:#x} expected {self.STRING_ADDR + 3:#x}"
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_strtod_writes_xmm0_concrete_double(self, fauxware_project):
        """strtod must dispatch natively, write the IEEE-754 bit pattern of
        the parsed double to xmm0's low 64 bits, and leave rax untouched."""
        proj = fauxware_project

        stub_cls = self._make_stub("strtod", 2)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = self._setup_state(proj, b"3.141592653589793")
            state.regs.rdi = self.STRING_ADDR
            state.regs.rsi = 0

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("strtod", 0) == 1, f"expected native strtod dispatch, got stats={stats}"
            sid = self._first_state_id(mgr)
            assert sid is not None
            xmm0 = mgr._rust_mgr.get_state_register(sid, "xmm0")
            # xmm0 is 128 bits; low 64 carry the scalar double return.
            low64 = xmm0 & ((1 << 64) - 1)
            import struct

            parsed = struct.unpack("<d", low64.to_bytes(8, "little"))[0]
            assert parsed == 3.141592653589793, f"xmm0 low64 → {parsed!r}"
        finally:
            proj.unhook(self.HOOK_ADDR)


class TestNativeStdioStatusAndWrite:
    """Integration tests for NativeFeof / NativeFerror / NativeFputs
    (angr-f16h.1).

    feof and ferror dispatch off `FILE._fileno` (amd64 offset 112) and return
    int flags. fputs reuses fwrite's write_fd path with a NUL-terminated
    source string. ferror is decl-only in glibc.json (no Python proc), the
    other two have Python references at `procedures/libc/{feof,fputs}.py`.

    Same in-binary non-executable hook-addr + stub-class pattern as
    TestNativeStringToNumericProcedures (see that class docstring for the
    rationale of using a fake hook in fauxware's `.ctors`).
    """

    HOOK_ADDR = 0x600E30  # in fauxware's .ctors (non-executable, in-binary)
    DEAD_ADDR = 0x4008B0
    STRING_ADDR = 0x601100  # past .bss, lazy-mapped
    FILE_PTR = 0x601200  # FILE struct; +112 = _fileno on amd64

    @staticmethod
    def _make_stub(proc_name: str, num_args: int):
        runs = {
            1: lambda self, a0: 0,
            2: lambda self, a0, a1: 0,
        }
        return type(
            proc_name,
            (angr.SimProcedure,),
            {"num_args": num_args, "run": runs[num_args]},
        )

    def _setup_state(self, proj, *, fileno: int, source_bytes: bytes | None = None):
        import claripy

        state = proj.factory.blank_state(
            addr=self.HOOK_ADDR,
            add_options={
                angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
            },
        )
        # Write FILE._fileno as a 32-bit little-endian int at FILE_PTR + 112.
        state.memory.store(
            self.FILE_PTR + 112,
            claripy.BVV(fileno & 0xFFFFFFFF, 32),
            endness="Iend_LE",
        )
        if source_bytes is not None:
            buf = source_bytes + b"\x00"
            for i, b in enumerate(buf):
                state.memory.store(self.STRING_ADDR + i, claripy.BVV(b, 8))
        # Return address for the stub call.
        state.memory.store(state.regs.rsp, claripy.BVV(self.DEAD_ADDR, 64), endness="Iend_LE")
        return state

    def _first_state_id(self, mgr):
        for stash in ("active", "deadended", "errored", "unconstrained"):
            ids = mgr._rust_mgr.get_state_ids(stash)
            if ids:
                return ids[0]
        return None

    def test_feof_returns_one_on_empty_fd(self, fauxware_project):
        """fd=1 (stdout) starts empty: position 0 >= content_len 0 → EOF."""
        proj = fauxware_project

        stub_cls = self._make_stub("feof", 1)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = self._setup_state(proj, fileno=1)
            state.regs.rdi = self.FILE_PTR

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("feof", 0) == 1, f"expected native feof dispatch, got stats={stats}"
            sid = self._first_state_id(mgr)
            assert sid is not None, f"no state: {mgr.stash_counts()}"
            rax = mgr._rust_mgr.get_state_register(sid, "rax")
            assert rax == 1, f"feof(empty fd) → rax={rax}, expected 1"
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_ferror_always_returns_zero(self, fauxware_project):
        """ferror has no Python proc — the native entry returns 0 (no error)."""
        proj = fauxware_project

        stub_cls = self._make_stub("ferror", 1)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = self._setup_state(proj, fileno=1)
            state.regs.rdi = self.FILE_PTR

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("ferror", 0) == 1, f"expected native ferror dispatch, got stats={stats}"
            sid = self._first_state_id(mgr)
            assert sid is not None
            rax = mgr._rust_mgr.get_state_register(sid, "rax")
            assert rax == 0, f"ferror → rax={rax}, expected 0"
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_fputs_writes_to_stdout_and_returns_one(self, fauxware_project):
        """fputs on stdout must append the NUL-terminated string to fd 1's
        buffer and return 1."""
        proj = fauxware_project

        stub_cls = self._make_stub("fputs", 2)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            payload = b"hello world"
            state = self._setup_state(proj, fileno=1, source_bytes=payload)
            state.regs.rdi = self.STRING_ADDR
            state.regs.rsi = self.FILE_PTR

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("fputs", 0) == 1, f"expected native fputs dispatch, got stats={stats}"
            sid = self._first_state_id(mgr)
            assert sid is not None
            rax = mgr._rust_mgr.get_state_register(sid, "rax")
            assert rax == 1, f"fputs success → rax={rax}, expected 1"
            stdout_bytes = bytes(mgr._rust_mgr.get_state_fd_output(sid, 1))
            assert stdout_bytes == payload, f"stdout={stdout_bytes!r} expected {payload!r}"
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_fputs_negative_fileno_returns_minus_one(self, fauxware_project):
        """fputs with a closed/sentinel fd (-1) must return -1 without writing."""
        proj = fauxware_project

        stub_cls = self._make_stub("fputs", 2)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = self._setup_state(proj, fileno=-1, source_bytes=b"discarded")
            state.regs.rdi = self.STRING_ADDR
            state.regs.rsi = self.FILE_PTR

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("fputs", 0) == 1, f"expected native fputs dispatch, got stats={stats}"
            sid = self._first_state_id(mgr)
            assert sid is not None
            rax = mgr._rust_mgr.get_state_register(sid, "rax")
            # -1 as a 64-bit unsigned value
            assert rax == (1 << 64) - 1, f"fputs(fd=-1) → rax={rax:#x}"
            stdout_bytes = bytes(mgr._rust_mgr.get_state_fd_output(sid, 1))
            assert stdout_bytes == b"", f"stdout should be untouched, got {stdout_bytes!r}"
        finally:
            proj.unhook(self.HOOK_ADDR)


class TestNativeReadCacheSync:
    """Regression for angr-3tek.2: re-enabling NativeRead/NativeWrite
    requires the cached Python SimState to be invalidate-and-replayed per
    dirty page in `_create_state_for_callback`. Without the replay,
    symbolic bytes written by NativeRead are invisible to a later Python
    SimProc fallback (the original failure that kept NativeRead disabled —
    see angr-mme3 / `avoid-enabling-native-read` memory)."""

    def test_native_read_dispatches_during_fauxware_exploration(self, fauxware_project):
        """NativeRead is registered by default and dispatches when fauxware
        calls read() during authenticate(). The fauxware backdoor path
        (find=0x4006ed) must remain reachable end-to-end — this exercises
        the read+strcmp interaction that previously broke when NativeRead
        was first attempted."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED, avoid=0x4006FD, max_steps=50000)

        assert len(mgr.found) > 0, "fauxware backdoor must remain reachable with NativeRead enabled"
        stats = mgr._rust_mgr.native_procedure_stats()
        assert stats["call_counts"].get("read", 0) >= 1, (
            f"expected at least one native read dispatch, got stats={stats}"
        )

    def test_pending_memory_load_symbolic_page_api_exposed(self, fauxware_project):
        """The new `pending_memory_load_symbolic_page` PyO3 method must be
        exposed and callable (a no-pending-callback error is acceptable
        outside a callback; what we're guarding against is a missing
        wrapper, which would surface as AttributeError)."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        # Calling outside a callback raises RuntimeError("no pending callback state")
        # — that's fine; what we're verifying is wrapper presence.
        assert hasattr(mgr._rust_mgr, "pending_memory_load_symbolic_page"), (
            "pending_memory_load_symbolic_page wrapper missing on Rust manager"
        )
        try:
            mgr._rust_mgr.pending_memory_load_symbolic_page(0x500000)
        except RuntimeError as e:
            assert "no pending" in str(e), f"unexpected error: {e}"


class TestNativeExtendedStringProcedures:
    """Integration tests for NativeStrrchr / NativeStrpbrk / NativeStrspn /
    NativeStrcspn (angr-f16h.5).

    angr has no Python SimProcedure for these four — they're declared in
    `procedures/definitions/common/glibc.json` but no Python file backs
    them, so the native registry is the only handler. Test pattern matches
    TestNativeStringToNumericProcedures: stub in fauxware's `.ctors`, set
    up two concrete buffers, single-step, verify native dispatch fired and
    rax holds the expected value.
    """

    HOOK_ADDR = 0x600E30  # in fauxware's .ctors
    DEAD_ADDR = 0x4008B0
    S_ADDR = 0x601100  # haystack / scan target
    SET_ADDR = 0x601200  # accept / reject set

    @staticmethod
    def _make_stub(proc_name: str):
        return type(
            proc_name,
            (angr.SimProcedure,),
            {"num_args": 2, "run": lambda self, a0, a1: 0},
        )

    def _setup_state(self, proj, s_bytes: bytes, set_bytes: bytes):
        import claripy

        state = proj.factory.blank_state(
            addr=self.HOOK_ADDR,
            add_options={
                angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
            },
        )
        for i, b in enumerate(s_bytes + b"\x00"):
            state.memory.store(self.S_ADDR + i, claripy.BVV(b, 8))
        for i, b in enumerate(set_bytes + b"\x00"):
            state.memory.store(self.SET_ADDR + i, claripy.BVV(b, 8))
        state.memory.store(state.regs.rsp, claripy.BVV(self.DEAD_ADDR, 64), endness="Iend_LE")
        return state

    def _first_state_id(self, mgr):
        for stash in ("active", "deadended", "errored", "unconstrained"):
            ids = mgr._rust_mgr.get_state_ids(stash)
            if ids:
                return ids[0]
        return None

    def _run_one(self, proj, name, s_bytes, set_bytes, c_or_set_addr):
        """Hook, single-step, return (call_count, rax)."""

        stub_cls = self._make_stub(name)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = self._setup_state(proj, s_bytes, set_bytes)
            state.regs.rdi = self.S_ADDR
            state.regs.rsi = c_or_set_addr
            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)
            stats = mgr._rust_mgr.native_procedure_stats()
            count = stats["call_counts"].get(name, 0)
            sid = self._first_state_id(mgr)
            assert sid is not None, f"no state: {mgr.stash_counts()}"
            rax = mgr._rust_mgr.get_state_register(sid, "rax")
            return count, rax
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_strrchr_native_dispatch(self, fauxware_project):
        # "hello" has two 'l's; last is at S_ADDR + 3.
        count, rax = self._run_one(fauxware_project, "strrchr", b"hello", b"", ord("l"))
        assert count == 1, "expected native strrchr dispatch"
        assert rax == self.S_ADDR + 3, f"rax={rax:#x}"

    def test_strrchr_native_dispatch_not_found(self, fauxware_project):
        count, rax = self._run_one(fauxware_project, "strrchr", b"hello", b"", ord("z"))
        assert count == 1
        assert rax == 0, f"rax={rax:#x}"

    def test_strpbrk_native_dispatch(self, fauxware_project):
        # First vowel in "hello world" is 'e' at offset 1.
        count, rax = self._run_one(fauxware_project, "strpbrk", b"hello world", b"aeiou", self.SET_ADDR)
        assert count == 1, "expected native strpbrk dispatch"
        assert rax == self.S_ADDR + 1, f"rax={rax:#x}"

    def test_strspn_native_dispatch(self, fauxware_project):
        # Prefix "abc" of "abc123" is in accept set {a,b,c}.
        count, rax = self._run_one(fauxware_project, "strspn", b"abc123", b"abc", self.SET_ADDR)
        assert count == 1, "expected native strspn dispatch"
        assert rax == 3, f"rax={rax}"

    def test_strcspn_native_dispatch(self, fauxware_project):
        # First reject byte ',' in "abc,def" is at offset 3.
        count, rax = self._run_one(fauxware_project, "strcspn", b"abc,def", b",;", self.SET_ADDR)
        assert count == 1, "expected native strcspn dispatch"
        assert rax == 3, f"rax={rax}"
