#!/usr/bin/env python3
from __future__ import annotations

__package__ = __package__ or "tests.procedures.libc"  # pylint:disable=redefined-builtin

import unittest

from cle.backends.externs.simdata.io_file import io_file_data_for_arch

import angr
from angr import claripy


def _state():
    p = angr.load_shellcode(b"\x90", arch="amd64")
    return p, p.factory.blank_state()


class TestVprintf(unittest.TestCase):
    """The v*printf family writes its format string as is: va_list is unmodeled."""

    def test_vprintf_writes_format_to_stdout(self):
        _, state = _state()
        fmt = 0x200000
        state.memory.store(fmt, b"hello %d\n\x00")
        ret = angr.SIM_PROCEDURES["libc"]["vprintf"]().execute(state, arguments=[fmt, 0]).ret_expr
        assert state.solver.eval(ret) == 9
        assert state.posix.dumps(1) == b"hello %d\n"

    def test_vfprintf_writes_to_the_stream_fd(self):
        p, state = _state()
        fmt, file_ptr = 0x200000, 0x400000
        state.memory.store(fmt, b"err\x00")
        fd_offset = io_file_data_for_arch(p.arch)["fd"]
        state.memory.store(file_ptr + fd_offset, claripy.BVV(2, 32), endness=p.arch.memory_endness)
        ret = angr.SIM_PROCEDURES["libc"]["vfprintf"]().execute(state, arguments=[file_ptr, fmt, 0]).ret_expr
        assert state.solver.eval(ret) == 3
        assert state.posix.dumps(2) == b"err"

    def test_vsprintf_copies_format_with_nul(self):
        _, state = _state()
        dst, fmt = 0x300000, 0x200000
        state.memory.store(fmt, b"abc\x00")
        ret = angr.SIM_PROCEDURES["libc"]["vsprintf"]().execute(state, arguments=[dst, fmt, 0]).ret_expr
        assert state.solver.eval(ret) == 3
        assert state.solver.eval(state.memory.load(dst, 4), cast_to=bytes) == b"abc\x00"

    def test_vsnprintf_chk_forwards_maxlen(self):
        _, state = _state()
        dst, fmt = 0x300000, 0x200000
        state.memory.store(dst, b"\xff")
        chk = angr.SIM_PROCEDURES["libc"]["__vsnprintf_chk"]
        # maxlen 0 writes nothing; slen (0x100) must not stand in for it.
        ret = chk().execute(state, arguments=[dst, 0, 1, 0x100, fmt, 0]).ret_expr
        assert state.solver.eval(ret) == 0
        assert state.solver.eval(state.memory.load(dst, 1)) == 0xFF
        ret = chk().execute(state, arguments=[dst, 8, 1, 0x100, fmt, 0]).ret_expr
        assert state.solver.eval(ret) == 1
        assert state.solver.eval(state.memory.load(dst, 1)) == 0


if __name__ == "__main__":
    unittest.main()
