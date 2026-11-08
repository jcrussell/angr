"""Native stream syscalls honour SHORT_READS and the CGC fd/length/fdwait options (angr-vmi2l.139, angr-vmi2l.141).

The native ``read`` / ``readv`` syscalls on stdin and CGC ``receive`` returned
the full ``count`` under ``SHORT_READS``, where Python's ``SimPackets.read``
returns a symbolic length. CGC ``transmit`` / ``receive`` ignored
``CGC_ENFORCE_FD``, and ``receive`` ignored ``CGC_NO_SYMBOLIC_RECEIVE_LENGTH``'s
``max_receive_size`` clamp. Each case runs on both engines and compares.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr.exploration import RustExplorationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

o = angr.options

LOAD = 0x08048000
BUF = 0x0804A000
RX = BUF + 0x100
MSG = BUF + 0x200


def _d(v):
    return struct.pack("<I", v & 0xFFFFFFFF)


def _run(project, state, end, engine):
    mgr = RustExplorationManager(project, [state]) if engine == "rust" else project.factory.simulation_manager(state)
    mgr.explore(find=end, n=50)
    assert len(mgr.found) == 1, (engine, mgr.errored)
    return mgr.found[0]


def _both(build, outcome):
    got = {engine: outcome(_run(*build(), engine)) for engine in ("python", "rust")}
    assert got["rust"] == got["python"], got
    return got["python"]


# ---------------------------------------------------------------- amd64 read


def _amd64(code, options):
    project = angr.load_shellcode(code + b"\xeb\xfe", arch="AMD64", simos="linux", load_address=0x400000)
    state = project.factory.entry_state(addr=0x400000, add_options=options)
    return project, state, 0x400000 + len(code)


# mov eax, 0; mov edi, 0; mov rsi, rsp; sub rsi, 0x100; mov edx, 10; syscall; mov rbx, rax
_READ = (
    b"\xb8\x00\x00\x00\x00\xbf\x00\x00\x00\x00\x48\x89\xe6\x48\x81\xee\x00\x01\x00\x00"
    b"\xba\x0a\x00\x00\x00\x0f\x05\x48\x89\xc3"
)


def _rbx_range(found):
    rbx = found.regs.rbx
    return found.solver.min(rbx), found.solver.max(rbx)


def test_read_syscall_on_stdin_returns_a_short_length_under_short_reads():
    assert _both(lambda: _amd64(_READ, {o.SHORT_READS}), _rbx_range) == (0, 10)


def test_read_syscall_on_stdin_returns_count_by_default():
    assert _both(lambda: _amd64(_READ, set()), _rbx_range) == (10, 10)


# ------------------------------------------------------------------- CGC


def _cgc(code, options, max_receive_size=0):
    project = angr.load_shellcode(code + b"\xeb\xfe", arch="X86", simos="cgc", load_address=LOAD)
    state = project.factory.entry_state(
        addr=LOAD,
        add_options={o.ZERO_FILL_UNCONSTRAINED_REGISTERS, o.ZERO_FILL_UNCONSTRAINED_MEMORY} | set(options),
    )
    state.memory.map_region(BUF, 0x1000, 3)
    state.memory.store(BUF, b"\0" * 0x300)
    state.memory.store(MSG, b"hi")
    state.cgc.max_receive_size = max_receive_size
    return project, state, LOAD + len(code)


def _syscall(num, ebx, ecx, edx, esi):
    # mov eax, num; mov ebx, ..; mov ecx, ..; mov edx, ..; mov esi, ..; int 0x80
    return (
        b"\xb8" + _d(num) + b"\xbb" + _d(ebx) + b"\xb9" + _d(ecx) + b"\xba" + _d(edx) + b"\xbe" + _d(esi) + b"\xcd\x80"
    )


def _receive(fd):
    return _syscall(3, fd, BUF, 4, RX)


def _rx_range(found):
    rx = found.memory.load(RX, 4, endness="Iend_LE")
    return found.solver.min(rx), found.solver.max(rx)


def test_transmit_under_enforce_fd_writes_stdout():
    def outcome(found):
        return found.posix.dumps(1), found.posix.dumps(2)

    build = lambda: _cgc(_syscall(2, 2, MSG, 2, 0), {o.CGC_ENFORCE_FD})  # noqa: E731
    assert _both(build, outcome) == (b"hi", b"")


def test_receive_under_enforce_fd_reads_stdin():
    assert _both(lambda: _cgc(_receive(3), {o.CGC_ENFORCE_FD}), _rx_range) == (4, 4)


def test_receive_under_no_symbolic_receive_length_clamps_to_max_receive_size():
    build = lambda: _cgc(_receive(0), {o.CGC_NO_SYMBOLIC_RECEIVE_LENGTH}, max_receive_size=2)  # noqa: E731
    assert _both(build, _rx_range) == (2, 2)


def test_receive_under_short_reads_returns_a_short_length():
    assert _both(lambda: _cgc(_receive(0), {o.SHORT_READS}), _rx_range) == (0, 4)


def _fdwait(nfds, readfds, writefds, readyfds):
    # mov edi, readyfds; then fdwait(nfds, readfds, writefds, timeout=0, readyfds)
    return b"\xbf" + _d(readyfds) + _syscall(4, nfds, readfds, writefds, 0)


def test_fdwait_under_non_blocking_fds_runs_natively():
    # angr-vmi2l.141: the handler gates on CGC_NON_BLOCKING_FDS, which reaches
    # the Rust state only through _NATIVE_SIMOPTIONS.
    def outcome(found):
        words = [found.memory.load(addr, 4, endness="Iend_LE") for addr in (BUF, BUF + 4, RX)]
        return tuple(found.solver.eval_one(w) for w in words)

    build = lambda: _cgc(_fdwait(3, BUF, BUF + 4, RX), {o.CGC_NON_BLOCKING_FDS})  # noqa: E731
    assert _both(build, outcome) == (0b111, 0b111, 6)

    project, state, end = build()
    mgr = RustExplorationManager(project, [state])
    mgr.explore(find=end, n=50)
    assert len(mgr.found) == 1, mgr.errored
    assert mgr.stats.get("syscall_native_by_num", {}).get(4) == 1, mgr.stats.get("syscall_python_fallback_by_num")


def test_every_native_has_option_literal_is_mirrored():
    # A `has_option("X")` in native prod code is dead unless X is mirrored onto
    # the Rust state (angr-vmi2l.141).
    import re
    from pathlib import Path

    from angr.exploration.rust_manager import _NATIVE_SIMOPTIONS

    native = Path(__file__).resolve().parents[3] / "native"
    names = set()
    for path in native.rglob("src/**/*.rs"):
        if path.name.endswith("_tests.rs") or any(part == "tests" or part.endswith("_tests") for part in path.parts):
            continue
        names.update(re.findall(r'has_option\(\s*"([A-Z_0-9]+)"', path.read_text()))
    assert names, "no has_option literals found; the scan is broken"
    assert names <= _NATIVE_SIMOPTIONS, sorted(names - _NATIVE_SIMOPTIONS)
