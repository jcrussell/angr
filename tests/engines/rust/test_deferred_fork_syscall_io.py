"""A deferred-fork sibling does not inherit the taken path's syscall I/O (angr-vmi2l.195).

The interpreter takes one side of a symbolic branch and defers the other. When
the taken path then ran a native ``write``/``read`` syscall in the same step,
the sibling was materialized off the post-syscall state, so it carried the
taken path's stdout bytes and stdin reads: the 'B' path below reported
``dumps(1) == b'QQQZ'`` and ``dumps(0) == b'B\\x00\\x00'``.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
BUF = 0x600000
M1 = 0x603000
M2 = 0x603100


def _sc(num: int, a: int, b: int, c: int) -> bytes:
    q = lambda v: struct.pack("<Q", v)  # noqa: E731
    return b"\x48\xb8" + q(num) + b"\x48\xbf" + q(a) + b"\x48\xbe" + q(b) + b"\x48\xba" + q(c) + b"\x0f\x05"


def _found(engine: str, shape: str):
    # read(0, BUF, 1); if (BUF[0] == 'B') write(1, "Z", 1); else <shape>
    head = _sc(0, 0, BUF, 1) + b"\x80\x3c\x25" + struct.pack("<I", BUF) + b"\x42"
    fall = _sc(1, 1, M2, 3) if shape == "write" else _sc(0, 0, BUF + 1, 2)
    tgt = _sc(1, 1, M1, 1)
    fall += b"\xeb" + bytes([len(tgt)])
    code = head + b"\x74" + bytes([len(fall)]) + fall + tgt + b"\xeb\xfe"
    end = LOAD + len(code) - 2
    proj = angr.load_shellcode(code, arch="amd64", simos="linux", load_address=LOAD)
    state = proj.factory.entry_state(
        addr=LOAD,
        add_options={
            angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
            angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
        },
    )
    state.memory.store(BUF, b"\0" * 16)
    state.memory.store(M1, b"Z")
    state.memory.store(M2, b"QQQ")
    if engine == "rust":
        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=end, num_find=2)
        found = list(mgr.found)
    else:
        mgr = proj.factory.simulation_manager(state)
        mgr.explore(find=end, num_find=2)
        found = list(mgr.found)
    assert len(found) == 2
    return found


def _b_path(found):
    (state,) = [s for s in found if s.solver.eval_upto(s.memory.load(BUF, 1) == 0x42, 2) == [True]]
    return state


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_b_path_stdout_excludes_sibling_write(engine):
    state = _b_path(_found(engine, "write"))
    assert state.posix.dumps(1) == b"Z"


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_b_path_stdin_excludes_sibling_read(engine):
    state = _b_path(_found(engine, "read"))
    assert state.posix.dumps(0) == b"B"
    assert state.posix.dumps(1) == b"Z"
