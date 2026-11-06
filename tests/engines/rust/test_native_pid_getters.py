"""The native getpid/gettid/getppid return a user-set ``state.posix.pid`` / ``ppid`` (angr-vmi2l.149).

They returned SimSystemPosix's defaults (1337 / 1336) whatever the seed
state's posix plugin held. Each case runs amd64 shellcode that calls the
three syscalls into ``r12``-``r14`` and compares both engines.
"""

from __future__ import annotations

import struct

import pytest

import angr
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000


def _syscall_into(number: int, reg: bytes) -> bytes:
    # mov rax, number; syscall; mov reg, rax
    return b"\x48\xb8" + struct.pack("<Q", number) + b"\x0f\x05" + reg


CODE = (
    _syscall_into(39, b"\x49\x89\xc4")  # getpid -> r12
    + _syscall_into(186, b"\x49\x89\xc5")  # gettid -> r13
    + _syscall_into(110, b"\x49\x89\xc6")  # getppid -> r14
)


def _run(engine: str, pid: int | None, ppid: int | None):
    project = angr.load_shellcode(CODE + b"\xeb\xfe", arch="AMD64", simos="linux", load_address=LOAD)
    state = project.factory.entry_state(addr=LOAD)
    if pid is not None:
        state.posix.pid = pid
        state.posix.ppid = ppid
    mgr = RustExplorationManager(project, [state]) if engine == "rust" else project.factory.simulation_manager(state)
    mgr.explore(find=LOAD + len(CODE), n=50)
    assert len(mgr.found) == 1
    found = mgr.found[0]
    return tuple(found.solver.eval(getattr(found.regs, r)) for r in ("r12", "r13", "r14"))


@pytest.mark.parametrize(("pid", "ppid"), [(None, None), (4242, 4241), (1, 0)])
def test_pid_getters_match_python(pid, ppid):
    expected = _run("python", pid, ppid)
    assert _run("rust", pid, ppid) == expected
    if pid is not None:
        assert expected == (pid, pid, ppid)
