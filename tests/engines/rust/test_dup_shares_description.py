"""dup/dup2 share one open file description in the Rust engine (angr-vmi2l.196).

POSIX ``dup`` makes two fd numbers refer to one description (offset and
content), as ``procedures/posix/dup.py`` does by aliasing one
``SimFileDescriptor``. The native syscalls used to clone the descriptor, so a
write through ``dup(1)`` never reached ``posix.dumps(1)`` and a read through a
``dup2`` copy did not advance the original's offset.
"""

from __future__ import annotations

import struct

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
BUF = 0x600000
PATH = 0x602000
MSG = 0x603000
SYSCALL = b"\x0f\x05"


def _mov(opcode: bytes, value: int) -> bytes:
    return opcode + struct.pack("<Q", value)


def _rax(v):
    return _mov(b"\x48\xb8", v)


def _rdi(v):
    return _mov(b"\x48\xbf", v)


def _rsi(v):
    return _mov(b"\x48\xbe", v)


def _rdx(v):
    return _mov(b"\x48\xba", v)


MOV_RDI_RAX = b"\x48\x89\xc7"
MOV_R12_RAX = b"\x49\x89\xc4"
MOV_RDI_R12 = b"\x4c\x89\xe7"


def _run(code: bytes, file_content: bytes | None = None):
    code += b"\xeb\xfe"
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
    state.memory.store(PATH, b"/g\0")
    state.memory.store(MSG, b"hiyo")
    if file_content is not None:
        state.fs.insert(b"/g", angr.SimFile(b"/g", content=file_content, size=len(file_content)))
    mgr = RustExplorationManager(proj, [state])
    mgr.explore(find=end)
    found = list(mgr.found)
    assert len(found) == 1
    return found[0]


def test_write_through_dup_of_stdout_reaches_dumps_1():
    # d = dup(1); write(d, "hi", 2); write(1, "yo", 2)
    code = (
        _rax(32)
        + _rdi(1)
        + SYSCALL
        + MOV_RDI_RAX
        + _rsi(MSG)
        + _rdx(2)
        + _rax(1)
        + SYSCALL
        + _rax(1)
        + _rdi(1)
        + _rsi(MSG + 2)
        + _rdx(2)
        + SYSCALL
    )
    assert _run(code).posix.dumps(1) == b"hiyo"


def test_read_through_dup2_advances_the_shared_offset():
    # fd = open("/g"); dup2(fd, 9); read(9, BUF, 4); read(fd, BUF + 8, 4)
    code = (
        _rax(2)
        + _rdi(PATH)
        + _rsi(0)
        + _rdx(0)
        + SYSCALL
        + MOV_R12_RAX
        + MOV_RDI_R12
        + _rsi(9)
        + _rax(33)
        + SYSCALL
        + _rdi(9)
        + _rsi(BUF)
        + _rdx(4)
        + _rax(0)
        + SYSCALL
        + MOV_RDI_R12
        + _rsi(BUF + 8)
        + _rdx(4)
        + _rax(0)
        + SYSCALL
    )
    st = _run(code, file_content=b"ABCDEFGH")
    assert st.solver.eval(st.memory.load(BUF, 12), cast_to=bytes) == b"ABCD\0\0\0\0EFGH"
