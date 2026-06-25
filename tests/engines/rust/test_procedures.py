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

    def test_strncmp_symbolic_bound_ignores_byte_past_n(self, fauxware_project):
        """strncmp(s1, s2, 2) with s1 = "AB" + symbolic + "\\0" and s2 = "AB\\0"
        must return 0: only the first 2 (equal) bytes are compared, so the
        symbolic byte at offset 2 cannot affect the result. Distinguishes the
        bounded strncmp from strcmp, which would read past offset 1."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["strncmp"]())
            state.memory.store(self.BUF_ADDR, b"AB")
            sym = claripy.BVS("strncmp_b", 8)
            state.memory.store(self.BUF_ADDR + 2, sym)
            state.memory.store(self.BUF_ADDR + 3, claripy.BVV(0, 8))
            state.memory.store(self.BUF_ADDR + 0x10, b"AB\x00")
            state.regs.rdi = self.BUF_ADDR
            state.regs.rsi = self.BUF_ADDR + 0x10
            state.regs.rdx = 2
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 0
            assert s.solver.max(s.regs.rax) == 0
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_strcasecmp_symbolic_case_insensitive_equal(self, fauxware_project):
        """strcasecmp(s1, s2) where s1[0] is symbolic constrained to 'A'
        (uppercase) and s2 is "a\\0" (lowercase) must return 0: the case-fold
        on both operands makes the bytes equal. strcmp would return non-zero."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["posix"]["strcasecmp"]())
            sym = claripy.BVS("strcasecmp_b", 8)
            state.memory.store(self.BUF_ADDR, sym)
            state.memory.store(self.BUF_ADDR + 1, claripy.BVV(0, 8))
            state.memory.store(self.BUF_ADDR + 0x10, b"a\x00")
            state.regs.rdi = self.BUF_ADDR
            state.regs.rsi = self.BUF_ADDR + 0x10
            state.solver.add(sym == ord("A"))
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 0
            assert s.solver.max(s.regs.rax) == 0
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_strncasecmp_bounded_case_insensitive_equal(self, fauxware_project):
        """strncasecmp(s1, s2, 2) with s1 = "Ab" + symbolic + "\\0" and
        s2 = "aB\\0" must return 0: the first 2 bytes match case-insensitively
        (A==a, b==B), and the symbolic byte at offset 2 is past n so cannot
        affect the result. Distinguishes the bounded case-insensitive variant
        from strncmp (case-sensitive → nonzero) and strcasecmp (unbounded →
        would read the symbolic byte)."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["posix"]["strncasecmp"]())
            state.memory.store(self.BUF_ADDR, b"Ab")
            sym = claripy.BVS("strncasecmp_b", 8)
            state.memory.store(self.BUF_ADDR + 2, sym)
            state.memory.store(self.BUF_ADDR + 3, claripy.BVV(0, 8))
            state.memory.store(self.BUF_ADDR + 0x10, b"aB\x00")
            state.regs.rdi = self.BUF_ADDR
            state.regs.rsi = self.BUF_ADDR + 0x10
            state.regs.rdx = 2
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

    def test_strchrnul_symbolic_finds_target(self, fauxware_project):
        """strchrnul("a?bX\\0", 'X') with '?' symbolic constrained to non-X,
        non-null must return BUF_ADDR + 3 (the index of 'X')."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["strchrnul"]())
            state.memory.store(self.BUF_ADDR, b"a")
            sym = claripy.BVS("strchrnul_q", 8)
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

    def test_strchrnul_not_found_returns_nul_ptr(self, fauxware_project):
        """strchrnul("ab\\0", 'X') must return a pointer to the terminating
        NUL (BUF_ADDR + 2), not NULL — the defining strchrnul-vs-strchr
        difference."""
        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["strchrnul"]())
            state.memory.store(self.BUF_ADDR, b"ab\x00")
            state.regs.rdi = self.BUF_ADDR
            state.regs.rsi = ord("X")
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == self.BUF_ADDR + 2
            assert s.solver.max(s.regs.rax) == self.BUF_ADDR + 2
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_rawmemchr_symbolic_finds_target(self, fauxware_project):
        """rawmemchr(buf, 'X') where buf[2] is symbolic constrained to 'X'
        must return BUF_ADDR + 2 (rawmemchr has no length bound; the caller
        guarantees the byte is present)."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["rawmemchr"]())
            state.memory.store(self.BUF_ADDR, b"AB")
            sym = claripy.BVS("rawmemchr_q", 8)
            state.memory.store(self.BUF_ADDR + 2, sym)
            state.memory.store(self.BUF_ADDR + 3, b"CD")
            state.solver.add(sym == ord("X"))
            state.regs.rdi = self.BUF_ADDR
            state.regs.rsi = ord("X")
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == self.BUF_ADDR + 2
            assert s.solver.max(s.regs.rax) == self.BUF_ADDR + 2
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_memrchr_symbolic_finds_last(self, fauxware_project):
        """memrchr(buf, 'X', 4) with 'X' at index 1 (concrete) and index 3
        (symbolic, constrained to 'X') must return the LAST match,
        BUF_ADDR + 3 — the reverse-scan that distinguishes memrchr."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["memrchr"]())
            state.memory.store(self.BUF_ADDR, b"AXc")
            sym = claripy.BVS("memrchr_q", 8)
            state.memory.store(self.BUF_ADDR + 3, sym)
            state.solver.add(sym == ord("X"))
            state.regs.rdi = self.BUF_ADDR
            state.regs.rsi = ord("X")
            state.regs.rdx = 4
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == self.BUF_ADDR + 3
            assert s.solver.max(s.regs.rax) == self.BUF_ADDR + 3
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_strstr_finds_substring(self, fauxware_project):
        """strstr("abcdef\\0", "cde\\0") must return BUF_ADDR + 2 — a pointer to
        the first occurrence of the needle inside the haystack. Concrete strings
        keep the Python strstr SimProc on a single path (it forks on symbolic
        input)."""
        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["strstr"]())
            state.memory.store(self.BUF_ADDR, b"abcdef\x00")
            state.memory.store(self.BUF_ADDR + 0x10, b"cde\x00")
            state.regs.rdi = self.BUF_ADDR
            state.regs.rsi = self.BUF_ADDR + 0x10
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == self.BUF_ADDR + 2
            assert s.solver.max(s.regs.rax) == self.BUF_ADDR + 2
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_strstr_not_found_returns_null(self, fauxware_project):
        """strstr("abc\\0", "xyz\\0") must return NULL (0) — the needle is absent
        from the haystack."""
        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["strstr"]())
            state.memory.store(self.BUF_ADDR, b"abc\x00")
            state.memory.store(self.BUF_ADDR + 0x10, b"xyz\x00")
            state.regs.rdi = self.BUF_ADDR
            state.regs.rsi = self.BUF_ADDR + 0x10
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 0
            assert s.solver.max(s.regs.rax) == 0
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_stpncpy_concrete_returns_nul_ptr(self, fauxware_project):
        """stpncpy(dst, "hi\\0", 8) must NUL-pad the 8-byte window and return a
        pointer to the written NUL (dst + 2 = strlen("hi")), unlike strncpy
        which returns dst. src must be concrete (native proc gate)."""
        proj = fauxware_project
        dst = self.BUF_ADDR + 0x40
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["stpncpy"]())
            state.memory.store(self.BUF_ADDR, b"hi\x00")
            state.regs.rdi = dst
            state.regs.rsi = self.BUF_ADDR
            state.regs.rdx = 8
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == dst + 2
            assert s.solver.max(s.regs.rax) == dst + 2
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_sprintf_formats_decimal_and_returns_length(self, fauxware_project):
        """sprintf(dest, "%d", 42) must format "42" into dest and return 2 (the
        formatted length), exercising the native format_string core + vararg
        read from rdx across the FFI boundary."""
        proj = fauxware_project
        dst = self.BUF_ADDR + 0x40
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["sprintf"]())
            state.memory.store(self.BUF_ADDR, b"%d\x00")
            state.regs.rdi = dst
            state.regs.rsi = self.BUF_ADDR
            state.regs.rdx = 42  # first vararg
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 2
            assert s.solver.max(s.regs.rax) == 2
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_snprintf_returns_untruncated_length(self, fauxware_project):
        """snprintf(dest, 2, "%d", 999) must return the would-have-been length
        3 (not the truncated 1), the defining snprintf return contract."""
        proj = fauxware_project
        dst = self.BUF_ADDR + 0x40
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["snprintf"]())
            state.memory.store(self.BUF_ADDR, b"%d\x00")
            state.regs.rdi = dst
            state.regs.rsi = 2  # size limit
            state.regs.rdx = self.BUF_ADDR  # format
            state.regs.rcx = 999  # first vararg
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 3
            assert s.solver.max(s.regs.rax) == 3
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_asprintf_returns_formatted_length(self, fauxware_project):
        """asprintf(strp, "hi") allocates the buffer and returns the formatted
        length 2 (rax), matching Python asprintf's malloc + strcpy + strlen."""
        proj = fauxware_project
        strp = self.BUF_ADDR + 0x40
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["asprintf"]())
            state.memory.store(self.BUF_ADDR, b"hi\x00")
            state.regs.rdi = strp
            state.regs.rsi = self.BUF_ADDR
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 2
            assert s.solver.max(s.regs.rax) == 2
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_vsprintf_raw_copies_format_and_returns_length(self, fauxware_project):
        """vsprintf(str, "hello", ap) does NOT %-substitute (va_list unmodeled):
        it raw-copies the format string and returns strlen("hello") = 5,
        matching Python vsprintf (strcpy + strlen)."""
        proj = fauxware_project
        dst = self.BUF_ADDR + 0x40
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["vsprintf"]())
            state.memory.store(self.BUF_ADDR, b"hello\x00")
            state.regs.rdi = dst
            state.regs.rsi = self.BUF_ADDR  # format
            state.regs.rdx = 0  # va_list (unused by the raw-copy impl)
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 5
            assert s.solver.max(s.regs.rax) == 5
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_vsnprintf_stub_returns_one(self, fauxware_project):
        """vsnprintf(str, size, fmt, ap) is a no-op stub (va_list unmodeled):
        size>0 stores a single NUL at str and returns 1, matching Python
        vsnprintf.py."""
        proj = fauxware_project
        dst = self.BUF_ADDR + 0x40
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["vsnprintf"]())
            state.memory.store(self.BUF_ADDR, b"hello\x00")
            state.regs.rdi = dst
            state.regs.rsi = 5  # size > 0
            state.regs.rdx = self.BUF_ADDR  # format (unused by the stub)
            state.regs.rcx = 0  # va_list
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 1
            assert s.solver.max(s.regs.rax) == 1
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_vsnprintf_chk_forwards_maxlen_to_vsnprintf(self, fauxware_project):
        """__vsnprintf_chk(dest, maxlen, flag, slen, fmt, ap) drops flag+slen and
        forwards maxlen as the vsnprintf size; maxlen>0 yields the stub return 1
        (rax), matching Python __vsnprintf_chk -> vsnprintf."""
        proj = fauxware_project
        dst = self.BUF_ADDR + 0x40
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"]["__vsnprintf_chk"]())
            state.memory.store(self.BUF_ADDR, b"hello\x00")
            state.regs.rdi = dst
            state.regs.rsi = 5  # maxlen > 0
            state.regs.rdx = 1  # flag (dropped)
            state.regs.rcx = 8  # slen (dropped)
            state.regs.r8 = self.BUF_ADDR  # format (unused by the stub)
            state.regs.r9 = 0  # va_list
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == 1
            assert s.solver.max(s.regs.rax) == 1
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

    def _run_ctype_predicate(self, proj, name, char, expected):
        """Shared helper: run a 1-arg ctype proc on a 32-bit symbolic int
        constrained to ``ord(char)`` and assert rax concretizes to ``expected``.
        """
        import claripy

        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["libc"][name]())
            sym = claripy.BVS(f"{name}_x", 32)
            state.solver.add(sym == ord(char))
            state.regs.rdi = sym.zero_extend(32)
            s = self._run_one_step(proj, state)
            assert s.solver.min(s.regs.rax) == expected
            assert s.solver.max(s.regs.rax) == expected
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_isspace_symbolic_returns_true_for_space(self, fauxware_project):
        """isspace on a symbolic int constrained to ASCII ' ' returns rax = 1."""
        self._run_ctype_predicate(fauxware_project, "isspace", " ", 1)

    def test_isspace_symbolic_returns_false_for_letter(self, fauxware_project):
        """isspace on a symbolic int constrained to ASCII 'a' returns rax = 0."""
        self._run_ctype_predicate(fauxware_project, "isspace", "a", 0)

    def test_isalnum_symbolic_returns_true_for_digit(self, fauxware_project):
        """isalnum on a symbolic int constrained to ASCII '7' returns rax = 1."""
        self._run_ctype_predicate(fauxware_project, "isalnum", "7", 1)

    def test_isupper_symbolic_returns_true_for_upper(self, fauxware_project):
        """isupper on a symbolic int constrained to ASCII 'Q' returns rax = 1."""
        self._run_ctype_predicate(fauxware_project, "isupper", "Q", 1)

    def test_isupper_symbolic_returns_false_for_lower(self, fauxware_project):
        """isupper on a symbolic int constrained to ASCII 'q' returns rax = 0."""
        self._run_ctype_predicate(fauxware_project, "isupper", "q", 0)

    def test_islower_symbolic_returns_true_for_lower(self, fauxware_project):
        """islower on a symbolic int constrained to ASCII 'q' returns rax = 1."""
        self._run_ctype_predicate(fauxware_project, "islower", "q", 1)

    def test_isxdigit_symbolic_returns_true_for_hex_letter(self, fauxware_project):
        """isxdigit on a symbolic int constrained to ASCII 'f' returns rax = 1."""
        self._run_ctype_predicate(fauxware_project, "isxdigit", "f", 1)

    def test_isxdigit_symbolic_returns_false_for_nonhex(self, fauxware_project):
        """isxdigit on a symbolic int constrained to ASCII 'g' returns rax = 0."""
        self._run_ctype_predicate(fauxware_project, "isxdigit", "g", 0)

    def test_isprint_symbolic_returns_true_for_printable(self, fauxware_project):
        """isprint on a symbolic int constrained to ASCII 'A' returns rax = 1."""
        self._run_ctype_predicate(fauxware_project, "isprint", "A", 1)

    def test_isprint_symbolic_returns_false_for_control(self, fauxware_project):
        """isprint on a symbolic int constrained to a tab returns rax = 0."""
        self._run_ctype_predicate(fauxware_project, "isprint", "\t", 0)

    def test_toupper_symbolic_shifts_lowercase(self, fauxware_project):
        """toupper on a symbolic int constrained to 'a' returns rax = ord('A')."""
        self._run_ctype_predicate(fauxware_project, "toupper", "a", ord("A"))

    def test_toupper_symbolic_leaves_digit_unchanged(self, fauxware_project):
        """toupper on a symbolic int constrained to '5' returns rax = ord('5')."""
        self._run_ctype_predicate(fauxware_project, "toupper", "5", ord("5"))

    def test_tolower_symbolic_shifts_uppercase(self, fauxware_project):
        """tolower on a symbolic int constrained to 'A' returns rax = ord('a')."""
        self._run_ctype_predicate(fauxware_project, "tolower", "A", ord("a"))

    def test_isascii_symbolic_returns_true_for_ascii(self, fauxware_project):
        """isascii on a symbolic int constrained to 'A' returns rax = 1."""
        self._run_ctype_predicate(fauxware_project, "isascii", "A", 1)

    def test_isascii_symbolic_returns_false_for_high_byte(self, fauxware_project):
        """isascii on a symbolic int constrained to 0x80 returns rax = 0."""
        self._run_ctype_predicate(fauxware_project, "isascii", "\x80", 0)

    def test_isblank_symbolic_returns_true_for_tab(self, fauxware_project):
        """isblank on a symbolic int constrained to a tab returns rax = 1."""
        self._run_ctype_predicate(fauxware_project, "isblank", "\t", 1)

    def test_isblank_symbolic_returns_false_for_newline(self, fauxware_project):
        """isblank on a symbolic int constrained to a newline returns rax = 0."""
        self._run_ctype_predicate(fauxware_project, "isblank", "\n", 0)

    def test_iscntrl_symbolic_returns_true_for_del(self, fauxware_project):
        """iscntrl on a symbolic int constrained to DEL (0x7f) returns rax = 1."""
        self._run_ctype_predicate(fauxware_project, "iscntrl", "\x7f", 1)

    def test_iscntrl_symbolic_returns_false_for_space(self, fauxware_project):
        """iscntrl on a symbolic int constrained to ' ' returns rax = 0."""
        self._run_ctype_predicate(fauxware_project, "iscntrl", " ", 0)

    def test_isgraph_symbolic_returns_true_for_punct(self, fauxware_project):
        """isgraph on a symbolic int constrained to '!' returns rax = 1."""
        self._run_ctype_predicate(fauxware_project, "isgraph", "!", 1)

    def test_isgraph_symbolic_returns_false_for_space(self, fauxware_project):
        """isgraph on a symbolic int constrained to ' ' returns rax = 0."""
        self._run_ctype_predicate(fauxware_project, "isgraph", " ", 0)

    def test_ispunct_symbolic_returns_true_for_symbol(self, fauxware_project):
        """ispunct on a symbolic int constrained to '@' returns rax = 1."""
        self._run_ctype_predicate(fauxware_project, "ispunct", "@", 1)

    def test_ispunct_symbolic_returns_false_for_letter(self, fauxware_project):
        """ispunct on a symbolic int constrained to 'a' returns rax = 0."""
        self._run_ctype_predicate(fauxware_project, "ispunct", "a", 0)

    def test_htonl_concrete_swaps_low_four_bytes(self, fauxware_project):
        """htonl on a concrete 0x01020304 returns rax = 0x04030201 (amd64 is
        little-endian, so the native proc byte-swaps the low 32 bits and
        zero-extends; mirrors angr/procedures/libc/htonl.py)."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["posix"]["htonl"]())
            state.regs.rdi = claripy.BVV(0x01020304, 64)
            s = self._run_one_step(proj, state)
            assert s.solver.eval_one(s.regs.rax) == 0x04030201
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_htons_concrete_swaps_low_two_bytes(self, fauxware_project):
        """htons on a concrete 0x0102 returns rax = 0x0201; only the low 16
        bits are converted (mirrors angr/procedures/libc/htons.py)."""
        import claripy

        proj = fauxware_project
        try:
            state = self._make_state(proj, angr.SIM_PROCEDURES["posix"]["htons"]())
            state.regs.rdi = claripy.BVV(0x0102, 64)
            s = self._run_one_step(proj, state)
            assert s.solver.eval_one(s.regs.rax) == 0x0201
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


class TestCallbackSolverConcretizationFallback:
    """Regression: callback solver min()/max() must handle a ``None`` return
    from the Rust solver context (it only returns ``Option<u128>``), not just an
    exception.

    Surfaced on real-binary symex of xmllint: a SimProcedure (sscanf→strtol on
    the unconstrained-symbolic ``getenv()`` return) dereferenced a value the
    forked Rust ctx could not bound. The callback solver shim's ``_rust_min``/
    ``_rust_max`` (rust_callback_dispatch.py) only fell back on an *exception*,
    but ``rust_ctx.min()`` *returns None* (solving_ops.rs) when the ctx is unsat
    or the expr width > 128. That ``None`` reached the range concretization
    strategy's ``mx - mn`` → ``None - None`` TypeError, aborting ``explore()``.

    The two ``None`` causes are handled differently (also in the
    state-export ``RustSolverFallback`` shim):
      - **width > 128** — ``u128`` can't hold the extremum but claripy's big-int
        solver can: fall back to Python.
      - **unsat** — the state is dead; claripy would only re-derive the same
        unsat. Raise ``SimUnsatError`` directly (the sat result is cached, so
        the check is free) instead of wasting a Python solve.
    """

    HOOK_ADDR = 0x4008C0  # fauxware .fini area (mapped, not normally executed)
    RET_ADDR = 0x4008B0

    def test_min_max_fall_back_when_rust_returns_none(self, fauxware_project):
        """During a callback, ``state.solver.min``/``max`` on an expression the
        Rust ctx cannot bound (``rust_ctx.min`` returns ``None``, not raises)
        must fall back to claripy, not return ``None``.

        Deterministic trigger: a >128-bit BVS — ``RustSolverContext.min``/``max``
        return ``None`` for ``width > 128`` (solving_ops.rs, angr-cxw7), the same
        ``None`` contract the production trigger (an unconstrained ``getenv()``
        symbol unknown to the forked Rust ctx) hits. Pre-fix the shim returned
        that ``None`` straight through, which crashed the range concretization
        strategy on ``mx - mn``. Asserts (a) the trigger is real (Rust returns
        ``None``) and (b) the shim recovers the true claripy bounds.
        """
        import claripy

        proj = fauxware_project
        captured = {}

        class CallMinMaxWideSym(angr.SimProcedure):
            def run(self):
                wide = claripy.BVS("wide_gt128", 160)
                # Sanity: confirm the Rust ctx really returns None here, so a
                # green test proves the *fallback* (not Rust) produced the bound.
                captured["rust_min"] = self.state.scratch.rust_solver_ctx.min(wide)
                # The shimmed solver.min/max — must not propagate the None.
                captured["mn"] = self.state.solver.min(wide)
                captured["mx"] = self.state.solver.max(wide)
                self.state.regs.rax = claripy.BVV(0, 64)

        try:
            proj.hook(self.HOOK_ADDR, CallMinMaxWideSym(), replace=True)
            state = proj.factory.blank_state(
                addr=self.HOOK_ADDR,
                add_options={
                    angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                    angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                },
            )
            state.memory.store(state.regs.rsp, claripy.BVV(self.RET_ADDR, 64), endness="Iend_LE")

            mgr = RustExplorationManager(proj, [state])
            mgr.run(max_steps=1)

            # Trigger is genuine: the Rust ctx could not bound the wide symbol.
            assert captured.get("rust_min") is None, (
                f"expected rust_ctx.min None for >128-bit BVS, got {captured.get('rust_min')}"
            )
            # The fix: shimmed min/max fell back to claripy's true bounds, not None.
            assert captured["mn"] == 0, f"min fell through as {captured['mn']!r}"
            assert captured["mx"] == (1 << 160) - 1, f"max fell through as {captured['mx']!r}"
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_unsat_raises_without_python_round_trip(self, fauxware_project):
        """On an unsat callback ctx, ``state.solver.min`` must raise
        ``SimUnsatError`` *natively* — without round-tripping to the Python
        claripy solver just to re-derive the same (known) unsat.

        Distinguishes the refined fix from a naive "fall back on any None":
        the naive version reaches the same ``SimUnsatError`` but only after a
        wasted Python solve. The spy on the Python frontend's ``min`` asserts
        it is never consulted.
        """
        import claripy

        from angr.errors import SimUnsatError

        proj = fauxware_project
        captured = {}

        class MinOnUnsatCtx(angr.SimProcedure):
            def run(self):
                x = claripy.BVS("unsat_x", 64)
                # Forwarded to the Rust ctx via the _rust_add shim -> unsat.
                self.state.solver.add(x == 1)
                self.state.solver.add(x == 2)
                # Spy: flip a flag if the Python claripy frontend's min is hit.
                py_frontend = self.state.solver._solver
                orig_py_min = py_frontend.min

                def spy_min(*a, **k):
                    captured["py_min_called"] = True
                    return orig_py_min(*a, **k)

                py_frontend.min = spy_min
                try:
                    try:
                        self.state.solver.min(x)
                        captured["raised"] = None
                    except SimUnsatError:
                        captured["raised"] = "SimUnsatError"
                    except Exception as e:
                        captured["raised"] = type(e).__name__
                finally:
                    py_frontend.min = orig_py_min
                self.state.regs.rax = claripy.BVV(0, 64)

        try:
            proj.hook(self.HOOK_ADDR, MinOnUnsatCtx(), replace=True)
            state = proj.factory.blank_state(
                addr=self.HOOK_ADDR,
                add_options={
                    angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                    angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                },
            )
            state.memory.store(state.regs.rsp, claripy.BVV(self.RET_ADDR, 64), endness="Iend_LE")

            mgr = RustExplorationManager(proj, [state])
            mgr.run(max_steps=1)

            assert captured.get("raised") == "SimUnsatError", (
                f"expected SimUnsatError on unsat ctx, got {captured.get('raised')!r}"
            )
            # The optimization: no wasted Python solve to re-derive the unsat.
            assert captured.get("py_min_called") is not True, (
                "shim round-tripped to the Python solver on a known-unsat ctx"
            )
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


class TestNativeIdentityGetters:
    """Python-boundary coverage for native getuid/geteuid/getgid/getegid
    (angr-ae54t.9).

    Angr's posix SimProcedures each `return 1000`. The native side previously
    only had these as *syscall* handlers (`syscalls/identity.rs`); a PLT libc
    call dispatches through the procedure registry, so without the native
    procedures (`procedures/getid.rs`) every call round-tripped to Python.
    This asserts native dispatch (call_counts ticks) and the constant 1000 in
    rax for all four zero-arg getters.
    """

    HOOK_ADDR = 0x600E30
    DEAD_ADDR = 0x4008B0

    @staticmethod
    def _make_stub(proc_name: str):
        return type(
            proc_name,
            (angr.SimProcedure,),
            {"num_args": 0, "run": lambda self: 1000},
        )

    @pytest.mark.parametrize(
        "proc_name",
        ["getuid", "geteuid", "getgid", "getegid"],
    )
    def test_identity_getter_returns_1000(self, fauxware_project, proc_name):
        import claripy

        proj = fauxware_project

        stub_cls = self._make_stub(proc_name)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = proj.factory.blank_state(
                addr=self.HOOK_ADDR,
                add_options={
                    angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                    angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                },
            )
            state.memory.store(
                state.regs.rsp,
                claripy.BVV(self.DEAD_ADDR, 64),
                endness="Iend_LE",
            )

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get(proc_name, 0) == 1, (
                f"expected native {proc_name} dispatch, got stats={stats}"
            )

            sid = None
            for stash in ("active", "deadended", "errored", "unconstrained"):
                ids = mgr._rust_mgr.get_state_ids(stash)
                if ids:
                    sid = ids[0]
                    break
            assert sid is not None, f"no state in any stash: {mgr.stash_counts()}"

            rax = mgr._rust_mgr.get_state_register(sid, "rax")
            assert rax == 1000, f"expected rax=1000 from native {proc_name}, got rax={rax:#x}"
        finally:
            proj.unhook(self.HOOK_ADDR)


class TestNativeSetbuf:
    """Python-boundary coverage for native setbuf (angr-ae54t.19).

    Angr's libc SimProcedure (`setbuf.py`) is a void no-op (ignores stream/buf
    and returns nothing). The native side previously had none, so a PLT libc
    call round-tripped to Python. This asserts native dispatch (call_counts
    ticks) — the proc writes no return register, so there is no rax to assert.
    """

    HOOK_ADDR = 0x600E30
    DEAD_ADDR = 0x4008B0

    def test_setbuf_dispatches_native(self, fauxware_project):
        import claripy

        proj = fauxware_project

        stub_cls = type(
            "setbuf",
            (angr.SimProcedure,),
            {"num_args": 2, "run": lambda self, stream, buf: None},
        )
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = proj.factory.blank_state(
                addr=self.HOOK_ADDR,
                add_options={
                    angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                    angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                },
            )
            state.regs.rdi = claripy.BVV(0x1000, 64)
            state.regs.rsi = claripy.BVV(0, 64)
            state.memory.store(
                state.regs.rsp,
                claripy.BVV(self.DEAD_ADDR, 64),
                endness="Iend_LE",
            )

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("setbuf", 0) == 1, f"expected native setbuf dispatch, got stats={stats}"
        finally:
            proj.unhook(self.HOOK_ADDR)


class TestNativeSyslog:
    """Python-boundary coverage for native openlog/closelog (angr-ae54t.20).

    Angr's libc SimProcedures (`openlog.py`, `closelog.py`) are void no-op stubs
    (return nothing). The native side previously had none, so a PLT libc call
    round-tripped to Python. This asserts native dispatch (call_counts ticks) —
    each proc writes no return register, so there is no rax to assert.
    """

    HOOK_ADDR = 0x600E30
    DEAD_ADDR = 0x4008B0

    def _dispatch(self, proj, name, num_args, run, setup_regs):
        import claripy

        stub_cls = type(name, (angr.SimProcedure,), {"num_args": num_args, "run": run})
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = proj.factory.blank_state(
                addr=self.HOOK_ADDR,
                add_options={
                    angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                    angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                },
            )
            setup_regs(state, claripy)
            state.memory.store(
                state.regs.rsp,
                claripy.BVV(self.DEAD_ADDR, 64),
                endness="Iend_LE",
            )
            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)
            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get(name, 0) == 1, f"expected native {name} dispatch, got stats={stats}"
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_openlog_dispatches_native(self, fauxware_project):
        def setup(state, claripy):
            state.regs.rdi = claripy.BVV(0x1000, 64)
            state.regs.rsi = claripy.BVV(0, 64)
            state.regs.rdx = claripy.BVV(8, 64)

        self._dispatch(
            fauxware_project,
            "openlog",
            3,
            lambda self, ident, option, facility: None,
            setup,
        )

    def test_closelog_dispatches_native(self, fauxware_project):
        self._dispatch(
            fauxware_project,
            "closelog",
            0,
            lambda self: None,
            lambda state, claripy: None,
        )


class TestNativeTime:
    """Python-boundary coverage for native libc time() (angr-ae54t.21).

    Angr's `procedures/libc/time.py` inline_calls the linux_kernel time
    syscall. The native side already models that syscall but had no native
    libc `time` procedure, so a PLT time() call round-tripped to Python. The
    native proc forwards to the same `fresh_monotonic_time` model and returns a
    symbolic time_t in rax. This asserts native dispatch (call_counts ticks).
    """

    HOOK_ADDR = 0x600E30
    DEAD_ADDR = 0x4008B0

    def test_time_dispatches_native(self, fauxware_project):
        import claripy

        proj = fauxware_project
        stub_cls = type("time", (angr.SimProcedure,), {"num_args": 1, "run": lambda self, tloc: 0})
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = proj.factory.blank_state(
                addr=self.HOOK_ADDR,
                add_options={
                    angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                    angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                },
            )
            # tloc = NULL: no store, just a symbolic return.
            state.regs.rdi = claripy.BVV(0, 64)
            state.memory.store(
                state.regs.rsp,
                claripy.BVV(self.DEAD_ADDR, 64),
                endness="Iend_LE",
            )
            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)
            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("time", 0) == 1, f"expected native time dispatch, got stats={stats}"
        finally:
            proj.unhook(self.HOOK_ADDR)


class TestNativeSleep:
    """Python-boundary coverage for native sleep/usleep (angr-ae54t.15).

    Angr's posix SimProcedures (`sleep.py`/`usleep.py`) ignore their argument
    and `return 0`. The native side previously had neither, so a PLT libc call
    round-tripped to Python. This asserts native dispatch (call_counts ticks)
    and rax=0 for both one-arg timers.
    """

    HOOK_ADDR = 0x600E30
    DEAD_ADDR = 0x4008B0

    @staticmethod
    def _make_stub(proc_name: str):
        return type(
            proc_name,
            (angr.SimProcedure,),
            {"num_args": 1, "run": lambda self, n: 0},
        )

    @pytest.mark.parametrize("proc_name", ["sleep", "usleep"])
    def test_timer_returns_zero(self, fauxware_project, proc_name):
        import claripy

        proj = fauxware_project

        stub_cls = self._make_stub(proc_name)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = proj.factory.blank_state(
                addr=self.HOOK_ADDR,
                add_options={
                    angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                    angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                },
            )
            state.regs.rdi = claripy.BVV(42, 64)
            state.memory.store(
                state.regs.rsp,
                claripy.BVV(self.DEAD_ADDR, 64),
                endness="Iend_LE",
            )

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get(proc_name, 0) == 1, (
                f"expected native {proc_name} dispatch, got stats={stats}"
            )

            sid = None
            for stash in ("active", "deadended", "errored", "unconstrained"):
                ids = mgr._rust_mgr.get_state_ids(stash)
                if ids:
                    sid = ids[0]
                    break
            assert sid is not None, f"no state in any stash: {mgr.stash_counts()}"

            rax = mgr._rust_mgr.get_state_register(sid, "rax")
            assert rax == 0, f"expected rax=0 from native {proc_name}, got rax={rax:#x}"
        finally:
            proj.unhook(self.HOOK_ADDR)


class TestNativeSystem:
    """Python-boundary coverage for native system() (angr-ae54t.16).

    Angr's libc SimProcedure (`system.py`) cannot run a real shell, so it
    returns an unconstrained 8-bit exit status zero-extended to 32-bit int.
    The native side previously had no `system`, so a PLT libc call round-tripped
    to Python. This asserts native dispatch (call_counts ticks) — the return is
    symbolic, so there is no concrete rax to assert.
    """

    HOOK_ADDR = 0x600E30
    DEAD_ADDR = 0x4008B0

    def test_system_dispatches_native(self, fauxware_project):
        import claripy

        proj = fauxware_project

        stub_cls = type(
            "system",
            (angr.SimProcedure,),
            {"num_args": 1, "run": lambda self, cmd: 0},
        )
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = proj.factory.blank_state(
                addr=self.HOOK_ADDR,
                add_options={
                    angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                    angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                },
            )
            state.regs.rdi = claripy.BVV(0x1000, 64)
            state.memory.store(
                state.regs.rsp,
                claripy.BVV(self.DEAD_ADDR, 64),
                endness="Iend_LE",
            )

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("system", 0) == 1, f"expected native system dispatch, got stats={stats}"
        finally:
            proj.unhook(self.HOOK_ADDR)


class TestNativeAccess:
    """Python-boundary coverage for native access() (angr-ae54t.17).

    Angr's libc SimProcedure (`libc/access.py`) has no real filesystem, so it
    returns a fresh symbolic int constrained to be either 0 or -1. The native
    side previously had no `access`, so a PLT libc call round-tripped to Python.
    This asserts native dispatch (call_counts ticks) and that the symbolic
    return is feasibly 0 and -1 but no other value.
    """

    HOOK_ADDR = 0x600E30
    DEAD_ADDR = 0x4008B0

    def test_access_dispatches_native(self, fauxware_project):
        import claripy

        proj = fauxware_project

        stub_cls = type(
            "access",
            (angr.SimProcedure,),
            {"num_args": 2, "run": lambda self, path, mode: 0},
        )
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            state = proj.factory.blank_state(
                addr=self.HOOK_ADDR,
                add_options={
                    angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                    angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                },
            )
            state.regs.rdi = claripy.BVV(0x1000, 64)  # path
            state.regs.rsi = claripy.BVV(0, 64)  # mode
            state.memory.store(
                state.regs.rsp,
                claripy.BVV(self.DEAD_ADDR, 64),
                endness="Iend_LE",
            )

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("access", 0) == 1, f"expected native access dispatch, got stats={stats}"
            # The symbolic-int {0,-1} constraint is verified directly in the
            # cargo test access_tests.rs::test_access_constrained_to_zero_or_minus_one.
        finally:
            proj.unhook(self.HOOK_ADDR)


class TestNativeFreadBoundary:
    """Python-boundary regression for native fread dispatch (angr-c42t7).

    `feat(procedures/fread)` (d4d1ad1c4) shipped 7 Cargo unit tests in
    `native/angr/src/procedures/fread_tests.rs` that call `NativeFread.call`
    directly, but no Python-level test. The concrete-content *serving* path
    cannot be reached from a clean Python integration test: the only seeder
    of FS content is `FileSystem::open_with_content`, which has no Python
    caller (native `fopen`/`open` both use the content-less `open`, and no
    PyO3 fd-content setter is exposed). What the Cargo tests do NOT cover is
    the FFI boundary that Python callers actually hit:

      1. Name-based native dispatch of `fread` through the manager
         (`native_procedure_stats` `call_counts['fread']` ticks).
      2. 4-arg marshaling off the AMD64 SysV registers
         (rdi=dst, rsi=size, rdx=nmemb, rcx=stream) — a wider arg list than
         any other fileops proc here.
      3. `FILE._fileno` resolution from a Python-written struct (read at the
         AMD64 `_IO_FILE` offset 112 by `read_fileno`).
      4. The two deterministic native-success returns that need no FS
         content: the zero-count early return, and the invalid-`_fileno`
         (fd < 0) "0 items read" return.

    Same in-binary non-executable hook-addr pattern as
    TestNativeFileDescriptorProcedures (see that class docstring for the
    address-selection rationale).
    """

    HOOK_ADDR = 0x600E30  # in fauxware's .ctors (non-executable, in-binary)
    DEAD_ADDR = 0x4008B0
    FILE_PTR = 0x601100  # past .bss, lazy-mapped: holds the _IO_FILE struct
    DST_ADDR = 0x601300  # fread destination buffer (distinct page region)

    # AMD64 _IO_FILE._fileno offset (mirrors AMD64_FILENO_OFF in
    # native/angr/src/procedures/fread_tests.rs / io_file_for_arch).
    FILENO_OFF = 112

    @staticmethod
    def _fread_stub():
        """SimProcedure stub named `fread` (4 args). `run` is bypassed when
        native dispatch fires (the expected path), but the class must exist
        so `proj.hook` is valid and the name is forwarded to the manager."""
        return type(
            "fread",
            (angr.SimProcedure,),
            {
                "num_args": 4,
                "run": lambda self, dst, size, nmemb, stream: 0,
            },
        )

    def _setup(self, proj, *, dst, size, nmemb, stream):
        import claripy

        state = proj.factory.blank_state(
            addr=self.HOOK_ADDR,
            add_options={
                angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
            },
        )
        # fread(dst, size, nmemb, stream) — AMD64 SysV argument registers.
        state.regs.rdi = dst
        state.regs.rsi = size
        state.regs.rdx = nmemb
        state.regs.rcx = stream
        state.memory.store(state.regs.rsp, claripy.BVV(self.DEAD_ADDR, 64), endness="Iend_LE")
        return state

    @staticmethod
    def _run(proj, state):
        mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
        mgr.run(max_steps=1)
        for stash in ("active", "deadended", "errored", "unconstrained"):
            ids = mgr._rust_mgr.get_state_ids(stash)
            if ids:
                return mgr, ids[0]
        raise AssertionError(f"no state in any stash: {mgr.stash_counts()}")

    def test_fread_zero_count_returns_zero(self, fauxware_project):
        """fread(dst, size=4, nmemb=0, stream) — total = size*nmemb = 0, so
        the native body early-returns 0 items BEFORE touching the stream
        pointer. Exercises dispatch + 4-arg marshaling + rax export with no
        FS content required (a deliberately bogus stream=0 confirms the
        early return never dereferences it)."""
        proj = fauxware_project
        proj.hook(self.HOOK_ADDR, self._fread_stub()(), replace=True)
        try:
            state = self._setup(proj, dst=self.DST_ADDR, size=4, nmemb=0, stream=0)
            mgr, sid = self._run(proj, state)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("fread", 0) == 1, f"expected native fread dispatch, got stats={stats}"
            assert mgr._rust_mgr.get_state_register(sid, "rax") == 0
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_fread_invalid_fileno_returns_zero(self, fauxware_project):
        """fread(dst, 1, 4, stream) where stream->_fileno is -1 (0xFFFFFFFF):
        `read_fileno` resolves the 32-bit field from the Python-written FILE
        struct, the fd < 0 branch fires, and the call returns 0 items. This
        is the only path that round-trips the FILE._fileno memory read across
        the FFI boundary without needing seeded FS content."""
        import claripy

        proj = fauxware_project
        proj.hook(self.HOOK_ADDR, self._fread_stub()(), replace=True)
        try:
            state = self._setup(proj, dst=self.DST_ADDR, size=1, nmemb=4, stream=self.FILE_PTR)
            # Invalid fd sentinel in the _fileno field — read_fileno returns -1.
            state.memory.store(
                self.FILE_PTR + self.FILENO_OFF,
                claripy.BVV(0xFFFFFFFF, 32),
                endness="Iend_LE",
            )
            mgr, sid = self._run(proj, state)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("fread", 0) == 1, f"expected native fread dispatch, got stats={stats}"
            # fd < 0 -> 0 items read.
            assert mgr._rust_mgr.get_state_register(sid, "rax") == 0
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

    def test_perror_writes_user_string_to_stderr(self, fauxware_project):
        """perror(s) writes only the user string to stderr (fd 2), no errno
        suffix — mirrors Python posix/perror.py (write(2, s, strlen(s)))."""
        proj = fauxware_project

        stub_cls = self._make_stub("perror", 1)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            payload = b"open failed"
            state = self._setup_state(proj, fileno=2, source_bytes=payload)
            state.regs.rdi = self.STRING_ADDR  # const char *s

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("perror", 0) == 1, f"expected native perror dispatch, got stats={stats}"
            sid = self._first_state_id(mgr)
            assert sid is not None
            stderr_bytes = bytes(mgr._rust_mgr.get_state_fd_output(sid, 2))
            assert stderr_bytes == payload, f"stderr={stderr_bytes!r} expected {payload!r}"
            # stdout untouched.
            stdout_bytes = bytes(mgr._rust_mgr.get_state_fd_output(sid, 1))
            assert stdout_bytes == b"", f"stdout should be untouched, got {stdout_bytes!r}"
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_vprintf_alias_dispatches_native_to_stdout(self, fauxware_project):
        """vprintf(format, ap) is a declare_proc! alias of NativePrintf
        (printf.rs): the va_list is unmodeled, so it writes the RAW format
        string to stdout and returns its length. This asserts the alias name
        passed through the Python SimProc registration dispatches natively
        (call_counts["vprintf"], not "printf") — the Python-boundary half of
        the registry-level test_vprintf_family_aliases_dispatch unit test."""
        proj = fauxware_project

        stub_cls = self._make_stub("vprintf", 1)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            payload = b"hello world"
            state = self._setup_state(proj, fileno=1, source_bytes=payload)
            state.regs.rdi = self.STRING_ADDR  # const char *format

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("vprintf", 0) == 1, (
                f"expected native vprintf alias dispatch, got stats={stats}"
            )
            sid = self._first_state_id(mgr)
            assert sid is not None
            rax = mgr._rust_mgr.get_state_register(sid, "rax")
            assert rax == len(payload), f"vprintf → rax={rax}, expected {len(payload)}"
            stdout_bytes = bytes(mgr._rust_mgr.get_state_fd_output(sid, 1))
            assert stdout_bytes == payload, f"stdout={stdout_bytes!r} expected {payload!r}"
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_vfprintf_alias_dispatches_native_to_stream(self, fauxware_project):
        """vfprintf(stream, format, ap) is a declare_proc! alias of
        NativeFprintf (printf.rs): it resolves stream->_fileno and writes the
        RAW format string to that fd, returning the byte count. Asserts the
        alias name dispatches natively (call_counts["vfprintf"]) and routes to
        the stream's fd, mirroring the fputs stream test."""
        proj = fauxware_project

        stub_cls = self._make_stub("vfprintf", 2)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            payload = b"stream out"
            state = self._setup_state(proj, fileno=1, source_bytes=payload)
            state.regs.rdi = self.FILE_PTR  # FILE *stream
            state.regs.rsi = self.STRING_ADDR  # const char *format

            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)

            stats = mgr._rust_mgr.native_procedure_stats()
            assert stats["call_counts"].get("vfprintf", 0) == 1, (
                f"expected native vfprintf alias dispatch, got stats={stats}"
            )
            sid = self._first_state_id(mgr)
            assert sid is not None
            rax = mgr._rust_mgr.get_state_register(sid, "rax")
            assert rax == len(payload), f"vfprintf → rax={rax}, expected {len(payload)}"
            fd_bytes = bytes(mgr._rust_mgr.get_state_fd_output(sid, 1))
            assert fd_bytes == payload, f"fd1={fd_bytes!r} expected {payload!r}"
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

    def test_pending_memory_load_symbolic_page_returns_page_symbols(self):
        """`pending_memory_load_symbolic_page` must return the symbolic
        objects on the queried page as (addr, ast) pairs AND exclude objects
        on other pages — driven inside a live pending callback.

        De-vacuifies the former existence-only test (angr-a116s.9): the old
        body only asserted ``hasattr(...)`` plus a conditional "no pending"
        error string, so a regression that drops the
        ``symbolic_objects_iter()`` loop, the page filter, or the
        ``rustbv_to_claripy`` conversion (pending_api.rs
        ``_pending_memory_load_symbolic_page``) stayed green because the
        method was only ever called OUTSIDE a callback, where ``with_pending``
        short-circuits before the body runs.

        A live ``pending_callback`` is established only by the run loop's
        SimProcedure-Python fallback (run_loop.rs sets ``pending_callback =
        Some(PendingCallback::with_context(...))`` immediately before emitting
        the ``need_simprocedure`` event). A symbolic argument register forces
        that fallback, so after ``mgr.run`` returns the event, ``with_pending``
        succeeds and the wrapper reads ``pending.state``'s memory.
        """
        import claripy

        mgr = _RustExplorationManager("amd64")

        HOOK = 0x500000
        SYM_PAGE = 0x510000
        SYM_ADDR = SYM_PAGE + 0x40
        STACK_BASE = 0x7FFF0000

        callbacks = PythonCallbacks()
        callbacks.set_memory_load(lambda a, s: (bytes(s), False, None))
        callbacks.set_memory_store(lambda a, d: None)
        callbacks.set_lift_block(lambda a: "{}")
        mgr.set_callbacks(callbacks)

        invocations = []

        def proc(args):
            invocations.append(tuple(args))
            return 0

        mgr.register_python_procedure("sym_proc", num_args=1, no_return=False, callable=proc)
        mgr.register_simprocedure(HOOK, "sym_proc", num_args=1, no_return=False)

        state = RustSimState("amd64")
        state.map_memory(STACK_BASE, 0x1000, 7)
        state.memory_store(STACK_BASE, (0xDEADC0DE).to_bytes(8, "little"))
        state.set_register("rsp", STACK_BASE)
        state.map_memory(SYM_PAGE, 0x1000, 7)
        state.pc = HOOK

        # Symbolic RDI → extract_concrete_arg returns SymbolicArgument → the
        # dispatcher hands the SimProcedure to Python, which sets a live
        # pending_callback (see docstring).
        sym_arg = claripy.BVS("sym_arg0", 64)
        state.set_register_symbolic("rdi", claripy.backends.z3.convert(sym_arg).as_ast().value, 64)

        mgr.add_state("active", state)
        # add_state forks the supplied state and mints a fresh monotonic id
        # (state_lifecycle::_add_state), so state.state_id is NOT the id the
        # manager tracks — query the active stash for the forked copy's id.
        active_ids = mgr.get_state_ids("active")
        assert len(active_ids) == 1, f"expected exactly one active state, got {active_ids}"
        sid = active_ids[0]

        # Store a multi-byte symbolic object at a known address on SYM_PAGE.
        # store_concrete_automap registers it in memory.symbolic_objects (the
        # source symbolic_objects_iter walks).
        sym_mem = claripy.BVS("sym_mem", 64)
        ok = mgr.state_memory_store_symbolic_multi(sid, claripy.BVV(SYM_ADDR, 64), sym_mem)
        assert ok, "symbolic store to a concrete mapped addr must land in Rust memory"

        event = mgr.run(5)
        assert event.callback_reason == "simprocedure", (
            f"need a live SimProcedure pending callback; got reason={event.callback_reason}"
        )
        assert invocations == [], f"native callable must NOT run for a symbolic arg; got {invocations}"

        # Inside the live pending callback: the wrapper returns the symbolic
        # object on SYM_PAGE as an (addr, ast) pair.
        entries = mgr.pending_memory_load_symbolic_page(SYM_PAGE)
        by_addr = dict(entries)
        assert SYM_ADDR in by_addr, (
            f"symbolic object at 0x{SYM_ADDR:x} missing from page-0x{SYM_PAGE:x} replay; got {sorted(by_addr)}"
        )
        ast = by_addr[SYM_ADDR]
        assert ast is not None and getattr(ast, "op", None) is not None, (
            f"expected a converted claripy AST for the symbolic object, got {ast!r}"
        )
        assert ast.symbolic, "replayed object must remain symbolic (rustbv_to_claripy must not concretize it)"

        # The page filter must EXCLUDE objects whose base address is not on the
        # queried page — querying HOOK's page returns nothing for SYM_ADDR.
        other = mgr.pending_memory_load_symbolic_page(HOOK & ~0xFFF)
        assert SYM_ADDR not in [a for a, _ in other], (
            f"page filter leaked 0x{SYM_ADDR:x} into the wrong page (0x{HOOK & ~0xFFF:x})"
        )


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


# SimProcedure self.call() continuation constants (angr-aca6y regression).
# Laid out inside a single load_shellcode blob so the test stays binary-free.
_SELFCALL_LOAD = 0x400000
_SELFCALL_HOOK = 0x400000  # entry; hooked with the self.call() procedure
_SELFCALL_CALLEE = 0x400010  # a lone `ret` the procedure calls into
_SELFCALL_RETTGT = 0x400020  # self-loop the top-level procedure rets to
_SELFCALL_MARK = 0x400100  # after_call writes the sentinel here
_SELFCALL_SENTINEL = 0xC0FFEE


class _SelfCallContinuationProc(angr.SimProcedure):
    """Hooked procedure that issues a ``self.call`` into a lone ``ret`` and
    only writes the sentinel from its continuation (``after_call``).

    The continuation runs iff the return address ``self.call`` pushed actually
    reached Rust state: pre-fix (angr-aca6y) the push was lost by the
    state-copy in ``cc.setup_callsite``, the callee's ``ret`` popped a stale
    word and jumped to garbage, and ``after_call`` never ran.
    """

    def run(self):
        self.call(_SELFCALL_CALLEE, [], "after_call", prototype="void x()")

    def after_call(self):
        import claripy

        self.state.memory.store(_SELFCALL_MARK, claripy.BVV(_SELFCALL_SENTINEL, 64), endness="Iend_LE")


class TestSimProcedureSelfCallContinuation:
    """Regression for angr-aca6y: a SimProcedure ``self.call`` continuation
    must survive the FFI boundary.

    ``SimProcedure.call`` runs ``cc.setup_callsite`` on a *copy* of the
    executing state, so the continuation return address it pushes bypasses
    ``CallbackMemoryTracker``'s wrapped store. In register-snapshot mode the
    pushed word never reached Rust, so the callee's ``ret`` jumped to a stale
    value and deadended instead of re-entering ``after_call``. The fix
    (``rust_state_sync._resume_with_state``) reads the grown stack region on an
    ``Ijk_Call`` successor and folds it into the memory changes.
    """

    def test_self_call_continuation_reaches_after_call(self):
        import claripy

        shellcode = bytearray(b"\x90" * 0x200)
        shellcode[_SELFCALL_CALLEE - _SELFCALL_LOAD] = 0xC3  # callee: ret
        # Self-loop (jmp $) so the continuation's return lands on a stable,
        # constrained PC -- a bare ret would drain to the unconstrained stash.
        shellcode[_SELFCALL_RETTGT - _SELFCALL_LOAD] = 0xEB
        shellcode[_SELFCALL_RETTGT - _SELFCALL_LOAD + 1] = 0xFE
        proj = angr.load_shellcode(bytes(shellcode), arch="AMD64", load_address=_SELFCALL_LOAD)
        proj.hook(_SELFCALL_HOOK, _SelfCallContinuationProc(), replace=True)
        state = proj.factory.blank_state(
            addr=_SELFCALL_HOOK,
            add_options={
                angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
            },
        )
        # Return slot for the top-level procedure (consumed after after_call).
        state.memory.store(state.regs.rsp, claripy.BVV(_SELFCALL_RETTGT, 64), endness="Iend_LE")

        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=6)

        states = mgr.active + mgr.deadended + mgr.errored + mgr.unconstrained
        marks = [s.solver.eval(s.memory.load(_SELFCALL_MARK, 8, endness="Iend_LE")) for s in states]
        assert _SELFCALL_SENTINEL in marks, (
            "after_call continuation never ran -> self.call() return-address "
            f"push lost across FFI (angr-aca6y); marks={[hex(m) for m in marks]}, "
            f"stash counts={mgr.stash_counts()}"
        )


class TestNativeMemoryCopyAndSet:
    """Integration tests for NativeMemcpy / NativeMemmove / NativeMemset
    (angr-8vfnz) plus the string-copy family NativeStrcpy / NativeStrncpy /
    NativeStrcat (angr-0ndrs).

    `procedures/memcpy.rs` and `procedures/memset.rs` ship extensive Cargo
    unit tests for the concrete + symbolic-byte/size/address paths, but had
    no Python-boundary coverage — a grep of tests/engines/ found zero
    memcpy/memset/memmove references. This class exercises the native
    dispatch + concrete fast paths end-to-end through RustExplorationManager:
    stub fauxware's `.ctors`, lay out concrete src/dst buffers, single-step,
    and verify (a) native dispatch fired (call_counts), (b) the destination
    bytes match the C contract via get_state_memory, and (c) rax holds the
    returned destination pointer.

    Pattern mirrors TestNativeExtendedStringProcedures.
    """

    HOOK_ADDR = 0x600E30  # in fauxware's .ctors
    DEAD_ADDR = 0x4008B0
    SRC_ADDR = 0x601100
    DST_ADDR = 0x601200

    @staticmethod
    def _make_stub(proc_name: str):
        return type(
            proc_name,
            (angr.SimProcedure,),
            {"num_args": 3, "run": lambda self, a0, a1, a2: 0},
        )

    def _blank_state(self, proj):
        state = proj.factory.blank_state(
            addr=self.HOOK_ADDR,
            add_options={
                angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
                angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
            },
        )
        import claripy

        state.memory.store(state.regs.rsp, claripy.BVV(self.DEAD_ADDR, 64), endness="Iend_LE")
        return state

    def _first_state_id(self, mgr):
        for stash in ("active", "deadended", "errored", "unconstrained"):
            ids = mgr._rust_mgr.get_state_ids(stash)
            if ids:
                return ids[0]
        return None

    def _run(self, proj, name, state):
        """Hook `name`, single-step, return (call_count, state_id, mgr)."""
        stub_cls = self._make_stub(name)
        proj.hook(self.HOOK_ADDR, stub_cls(), replace=True)
        try:
            mgr = RustExplorationManager(proj, [state], save_unconstrained=True)
            mgr.run(max_steps=1)
            stats = mgr._rust_mgr.native_procedure_stats()
            count = stats["call_counts"].get(name, 0)
            sid = self._first_state_id(mgr)
            assert sid is not None, f"no state: {mgr.stash_counts()}"
            return count, sid, mgr
        finally:
            proj.unhook(self.HOOK_ADDR)

    def test_memcpy_native_dispatch_copies_bytes(self, fauxware_project):
        import claripy

        payload = b"angr-rust!"
        state = self._blank_state(fauxware_project)
        for i, b in enumerate(payload):
            state.memory.store(self.SRC_ADDR + i, claripy.BVV(b, 8))
        state.regs.rdi = self.DST_ADDR
        state.regs.rsi = self.SRC_ADDR
        state.regs.rdx = len(payload)

        count, sid, mgr = self._run(fauxware_project, "memcpy", state)
        assert count == 1, "expected native memcpy dispatch"
        rax = mgr._rust_mgr.get_state_register(sid, "rax")
        assert rax == self.DST_ADDR, f"rax={rax:#x}"
        got = bytes(mgr._rust_mgr.get_state_memory(sid, self.DST_ADDR, len(payload)))
        assert got == payload, f"dst={got!r}"

    def test_memcpy_zero_size_leaves_dst_untouched(self, fauxware_project):
        import claripy

        # Pre-seed dst with a sentinel; a zero-size memcpy must not change it.
        state = self._blank_state(fauxware_project)
        state.memory.store(self.SRC_ADDR, claripy.BVV(0xAA, 8))
        state.memory.store(self.DST_ADDR, claripy.BVV(0x55, 8))
        state.regs.rdi = self.DST_ADDR
        state.regs.rsi = self.SRC_ADDR
        state.regs.rdx = 0

        count, sid, mgr = self._run(fauxware_project, "memcpy", state)
        assert count == 1
        rax = mgr._rust_mgr.get_state_register(sid, "rax")
        assert rax == self.DST_ADDR, f"rax={rax:#x}"
        got = bytes(mgr._rust_mgr.get_state_memory(sid, self.DST_ADDR, 1))
        assert got == b"\x55", f"dst={got!r}"

    def test_memmove_overlapping_forward(self, fauxware_project):
        import claripy

        # Overlapping move: dst = src + 2, copy 4 bytes. memmove must read the
        # pre-store source bytes, so dst[0:4] == original src[0:4].
        base = self.SRC_ADDR
        original = b"ABCD"
        state = self._blank_state(fauxware_project)
        for i, b in enumerate(original):
            state.memory.store(base + i, claripy.BVV(b, 8))
        state.regs.rdi = base + 2  # dst overlaps src tail
        state.regs.rsi = base
        state.regs.rdx = len(original)

        count, sid, mgr = self._run(fauxware_project, "memmove", state)
        assert count == 1, "expected native memmove dispatch"
        rax = mgr._rust_mgr.get_state_register(sid, "rax")
        assert rax == base + 2, f"rax={rax:#x}"
        got = bytes(mgr._rust_mgr.get_state_memory(sid, base + 2, len(original)))
        assert got == original, f"moved={got!r}"

    def test_memset_native_dispatch_fills_bytes(self, fauxware_project):
        state = self._blank_state(fauxware_project)
        state.regs.rdi = self.DST_ADDR
        state.regs.rsi = 0x41  # fill byte 'A'
        state.regs.rdx = 6

        count, sid, mgr = self._run(fauxware_project, "memset", state)
        assert count == 1, "expected native memset dispatch"
        rax = mgr._rust_mgr.get_state_register(sid, "rax")
        assert rax == self.DST_ADDR, f"rax={rax:#x}"
        got = bytes(mgr._rust_mgr.get_state_memory(sid, self.DST_ADDR, 6))
        assert got == b"A" * 6, f"dst={got!r}"

    def test_memset_uses_low_byte_of_value(self, fauxware_project):
        # memset's value arg is an int; only the low 8 bits are written.
        state = self._blank_state(fauxware_project)
        state.regs.rdi = self.DST_ADDR
        state.regs.rsi = 0x12CC  # low byte 0xCC
        state.regs.rdx = 3

        count, sid, mgr = self._run(fauxware_project, "memset", state)
        assert count == 1
        got = bytes(mgr._rust_mgr.get_state_memory(sid, self.DST_ADDR, 3))
        assert got == b"\xcc" * 3, f"dst={got!r}"

    def test_bzero_native_dispatch_zeros_bytes(self, fauxware_project):
        # bzero(s, n) forwards to native memset(s, 0, n). Pre-seed dst with a
        # 0xFF sentinel; the first n bytes must be zeroed, byte n untouched.
        import claripy

        state = self._blank_state(fauxware_project)
        for i in range(7):
            state.memory.store(self.DST_ADDR + i, claripy.BVV(0xFF, 8))
        state.regs.rdi = self.DST_ADDR
        state.regs.rsi = 5  # n

        count, sid, mgr = self._run(fauxware_project, "bzero", state)
        assert count == 1, "expected native bzero dispatch"
        got = bytes(mgr._rust_mgr.get_state_memory(sid, self.DST_ADDR, 7))
        assert got == b"\x00" * 5 + b"\xff" * 2, f"dst={got!r}"

    def _store_cstr(self, state, addr, data: bytes):
        """Lay out a NUL-terminated C string at `addr`."""
        import claripy

        for i, b in enumerate(data + b"\x00"):
            state.memory.store(addr + i, claripy.BVV(b, 8))

    def test_strcpy_copies_string_with_terminator(self, fauxware_project):
        # strcpy(dst, src): copies src incl. its NUL, returns dst.
        payload = b"hello"
        state = self._blank_state(fauxware_project)
        self._store_cstr(state, self.SRC_ADDR, payload)
        state.regs.rdi = self.DST_ADDR
        state.regs.rsi = self.SRC_ADDR

        count, sid, mgr = self._run(fauxware_project, "strcpy", state)
        assert count == 1, "expected native strcpy dispatch"
        rax = mgr._rust_mgr.get_state_register(sid, "rax")
        assert rax == self.DST_ADDR, f"rax={rax:#x}"
        got = bytes(mgr._rust_mgr.get_state_memory(sid, self.DST_ADDR, len(payload) + 1))
        assert got == payload + b"\x00", f"dst={got!r}"

    def test_strncpy_nul_pads_window(self, fauxware_project):
        # strncpy(dst, src, n) with n > strlen(src) NUL-pads the whole window.
        payload = b"hi"
        n = 5
        state = self._blank_state(fauxware_project)
        self._store_cstr(state, self.SRC_ADDR, payload)
        # Pre-seed dst with sentinel bytes the pad must overwrite.
        import claripy

        for i in range(n):
            state.memory.store(self.DST_ADDR + i, claripy.BVV(0x55, 8))
        state.regs.rdi = self.DST_ADDR
        state.regs.rsi = self.SRC_ADDR
        state.regs.rdx = n

        count, sid, mgr = self._run(fauxware_project, "strncpy", state)
        assert count == 1, "expected native strncpy dispatch"
        rax = mgr._rust_mgr.get_state_register(sid, "rax")
        assert rax == self.DST_ADDR, f"rax={rax:#x}"
        got = bytes(mgr._rust_mgr.get_state_memory(sid, self.DST_ADDR, n))
        assert got == payload + b"\x00" * (n - len(payload)), f"dst={got!r}"

    def test_strcat_appends_after_existing_string(self, fauxware_project):
        # strcat(dst, src) appends src at dst's existing NUL, returns dst.
        existing = b"foo"
        suffix = b"bar"
        state = self._blank_state(fauxware_project)
        self._store_cstr(state, self.DST_ADDR, existing)
        self._store_cstr(state, self.SRC_ADDR, suffix)
        state.regs.rdi = self.DST_ADDR
        state.regs.rsi = self.SRC_ADDR

        count, sid, mgr = self._run(fauxware_project, "strcat", state)
        assert count == 1, "expected native strcat dispatch"
        rax = mgr._rust_mgr.get_state_register(sid, "rax")
        assert rax == self.DST_ADDR, f"rax={rax:#x}"
        joined = existing + suffix
        got = bytes(mgr._rust_mgr.get_state_memory(sid, self.DST_ADDR, len(joined) + 1))
        assert got == joined + b"\x00", f"dst={got!r}"

    def test_mempcpy_returns_dst_plus_n(self, fauxware_project):
        # mempcpy(dst, src, n) copies n bytes like memcpy but returns dst + n
        # (a pointer to the byte after the last copied), not dst.
        import claripy

        payload = b"mempcpy!"
        state = self._blank_state(fauxware_project)
        for i, b in enumerate(payload):
            state.memory.store(self.SRC_ADDR + i, claripy.BVV(b, 8))
        state.regs.rdi = self.DST_ADDR
        state.regs.rsi = self.SRC_ADDR
        state.regs.rdx = len(payload)

        count, sid, mgr = self._run(fauxware_project, "mempcpy", state)
        assert count == 1, "expected native mempcpy dispatch"
        rax = mgr._rust_mgr.get_state_register(sid, "rax")
        assert rax == self.DST_ADDR + len(payload), f"rax={rax:#x}"
        got = bytes(mgr._rust_mgr.get_state_memory(sid, self.DST_ADDR, len(payload)))
        assert got == payload, f"dst={got!r}"

    def test_stpcpy_returns_pointer_to_written_nul(self, fauxware_project):
        # stpcpy(dst, src) copies src incl. its NUL like strcpy, but returns
        # dst + strlen(src) (a pointer to the written terminating NUL).
        payload = b"world"
        state = self._blank_state(fauxware_project)
        self._store_cstr(state, self.SRC_ADDR, payload)
        state.regs.rdi = self.DST_ADDR
        state.regs.rsi = self.SRC_ADDR

        count, sid, mgr = self._run(fauxware_project, "stpcpy", state)
        assert count == 1, "expected native stpcpy dispatch"
        rax = mgr._rust_mgr.get_state_register(sid, "rax")
        assert rax == self.DST_ADDR + len(payload), f"rax={rax:#x}"
        got = bytes(mgr._rust_mgr.get_state_memory(sid, self.DST_ADDR, len(payload) + 1))
        assert got == payload + b"\x00", f"dst={got!r}"

    def test_strcoll_aliases_strcmp_equal_strings(self, fauxware_project):
        # strcoll(s1, s2) degenerates to strcmp in the C locale (native alias);
        # equal strings compare equal (rax == 0) and call_counts ticks under the
        # hooked `strcoll` name (aliases register on the NativeStrcmp impl).
        payload = b"locale"
        state = self._blank_state(fauxware_project)
        self._store_cstr(state, self.SRC_ADDR, payload)
        self._store_cstr(state, self.DST_ADDR, payload)
        state.regs.rdi = self.SRC_ADDR
        state.regs.rsi = self.DST_ADDR

        count, sid, mgr = self._run(fauxware_project, "strcoll", state)
        assert count == 1, "expected native strcoll (strcmp alias) dispatch"
        rax = mgr._rust_mgr.get_state_register(sid, "rax")
        # strcmp returns a 32-bit int; equal strings -> 0 (low 32 bits).
        assert rax & 0xFFFFFFFF == 0, f"rax={rax:#x}"

    def test_strxfrm_copies_window_and_returns_src_len(self, fauxware_project):
        # strxfrm(dst, src, n) degenerates to strncpy(dst, src, n) + return
        # strlen(src) in the C locale. With n > strlen, dst holds the NUL-padded
        # window and rax == strlen(src).
        payload = b"abc"
        n = 6
        state = self._blank_state(fauxware_project)
        self._store_cstr(state, self.SRC_ADDR, payload)
        import claripy

        for i in range(n):
            state.memory.store(self.DST_ADDR + i, claripy.BVV(0x55, 8))
        state.regs.rdi = self.DST_ADDR
        state.regs.rsi = self.SRC_ADDR
        state.regs.rdx = n

        count, sid, mgr = self._run(fauxware_project, "strxfrm", state)
        assert count == 1, "expected native strxfrm dispatch"
        rax = mgr._rust_mgr.get_state_register(sid, "rax")
        assert rax == len(payload), f"rax={rax:#x} (expected strlen={len(payload)})"
        got = bytes(mgr._rust_mgr.get_state_memory(sid, self.DST_ADDR, n))
        assert got == payload + b"\x00" * (n - len(payload)), f"dst={got!r}"

    def test_strndup_truncates_to_n_and_nul_terminates(self, fauxware_project):
        # strndup(s, n) = strnlen(s, n) + malloc(len+1) + copy, always
        # NUL-terminated. With n < strlen(s), the result is the truncated
        # prefix + NUL in a fresh heap buffer (rax points at it).
        src = b"hello"
        n = 3
        state = self._blank_state(fauxware_project)
        self._store_cstr(state, self.SRC_ADDR, src)
        state.regs.rdi = self.SRC_ADDR
        state.regs.rsi = n

        count, sid, mgr = self._run(fauxware_project, "strndup", state)
        assert count == 1, "expected native strndup dispatch"
        rax = mgr._rust_mgr.get_state_register(sid, "rax")
        assert rax != 0, "strndup returned NULL"
        got = bytes(mgr._rust_mgr.get_state_memory(sid, rax, n + 1))
        assert got == src[:n] + b"\x00", f"dup={got!r}"
