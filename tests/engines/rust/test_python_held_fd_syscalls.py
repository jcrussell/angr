"""fd syscalls on an fd only Python holds fall back instead of answering EBADF (angr-vmi2l.137).

The Rust fd table is not seeded from ``posix.fd``, so an fd the harness opens
before the run lives only in Python. The native ``dup`` / ``dup2`` / ``dup3`` /
``fcntl(F_GETFL)`` answered ``-EBADF`` for it, and ``close`` / ``fstat`` once
did the same. Each case runs on both engines and compares the syscall's return
and Python's fd table.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr.exploration import RustExplorationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000


def _code(nr, rsi):
    # mov eax, nr; mov edi, 3; mov esi, rsi (or rsi = rsp - 0x200 when None); syscall; mov rbx, rax
    set_rsi = b"\x48\x89\xe6\x48\x81\xee\x00\x02\x00\x00" if rsi is None else b"\xbe" + struct.pack("<I", rsi)
    return b"\xb8" + struct.pack("<I", nr) + b"\xbf\x03\x00\x00\x00" + set_rsi + b"\x0f\x05\x48\x89\xc3"


def _run(nr, rsi, engine):
    code = _code(nr, rsi)
    project = angr.load_shellcode(code + b"\xeb\xfe", arch="AMD64", simos="linux", load_address=LOAD)
    state = project.factory.entry_state(addr=LOAD)
    state.fs.insert(b"/f3", angr.SimFile(b"/f3", content=b"XYZ", size=3))
    assert state.solver.eval(state.posix.open(b"/f3", 0)) == 3
    mgr = RustExplorationManager(project, [state]) if engine == "rust" else project.factory.simulation_manager(state)
    mgr.explore(find=LOAD + len(code), n=20)
    assert len(mgr.found) == 1, (engine, mgr.errored)
    found = mgr.found[0]
    return found.solver.eval(found.regs.rbx), sorted(found.posix.fd)


@pytest.mark.parametrize(
    ("nr", "rsi"),
    [
        pytest.param(3, None, id="close"),
        pytest.param(5, None, id="fstat"),
        pytest.param(32, None, id="dup"),
        pytest.param(33, 7, id="dup2"),
        pytest.param(292, 7, id="dup3"),
        pytest.param(72, 3, id="fcntl_getfl"),
    ],
)
def test_syscall_on_a_python_held_fd_matches_python(nr, rsi):
    got = {engine: _run(nr, rsi, engine) for engine in ("python", "rust")}
    assert got["rust"] == got["python"], got
