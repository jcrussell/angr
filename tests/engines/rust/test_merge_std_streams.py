"""``mgr.merge()`` keeps each arm's stdin and stdout as the Python engine does (angr-vmi2l.199).

The native merge keeps one arm's fd table, so two arms that printed different
bytes merged into one state whose ``posix.dumps(1)`` gave the longer arm's
output under both merge conditions. A group whose std streams differ now goes
to the Python merge: ``SimPackets.merge`` ITE-merges equal packet counts and
raises ``SimMergeError`` (the group stays unmerged) otherwise. Each case runs
amd64 shellcode that reads one stdin byte and branches on it, explores to the
common exit on both engines, merges ``found`` and compares, per arm, the
satisfiable ``dumps(0)`` / ``dumps(1)`` pairs.
"""

from __future__ import annotations

import struct

import pytest

import angr
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
BUF = 0x600000
MSG_A = 0x603000
MSG_B = 0x603100


def _sys3(nr: int, a: int, b: int, c: int) -> bytes:
    movabs = (b"\x48\xb8", b"\x48\xbf", b"\x48\xbe", b"\x48\xba")
    return b"".join(op + struct.pack("<Q", v) for op, v in zip(movabs, (nr, a, b, c))) + b"\x0f\x05"


def _read(addr: int) -> bytes:
    return _sys3(0, 0, addr, 1)


def _write(addr: int, n: int) -> bytes:
    return _sys3(1, 1, addr, n)


def _branchy(then_: bytes, else_: bytes) -> bytes:
    """read(0, BUF, 1); if BUF[0] == 'A': then_ else: else_; spin at the exit."""
    else_ += b"\xeb" + bytes([len(then_)])
    cmp_a = b"\x80\x3c\x25" + struct.pack("<I", BUF) + b"\x41"
    return _read(BUF) + cmp_a + b"\x74" + bytes([len(else_)]) + else_ + then_ + b"\xeb\xfe"


def _run(engine: str, code: bytes):
    proj = angr.load_shellcode(code, arch="amd64", simos="linux", load_address=LOAD)
    st = proj.factory.entry_state(
        addr=LOAD,
        add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS, angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY},
    )
    st.memory.store(BUF, b"\0" * 8)
    st.memory.store(MSG_A, b"XY")
    st.memory.store(MSG_B, b"ZZ")
    end = LOAD + len(code) - 2
    mgr = RustExplorationManager(proj, [st]) if engine == "rust" else proj.factory.simulation_manager(st)
    mgr.explore(find=end, num_find=2)
    assert len(list(mgr.found)) == 2
    mgr.merge(stash="found")
    arms = []
    for state in mgr.found:
        first = state.memory.load(BUF, 1)
        for label, cond in (("A", first == 0x41), ("other", first != 0x41)):
            if state.solver.satisfiable(extra_constraints=[cond]):
                # The other arm's stdin byte is whatever the model picks, so
                # only the A arm's is compared.
                stdin = state.posix.dumps(0, extra_constraints=[cond])[:1] if label == "A" else None
                arms.append((label, stdin, state.posix.dumps(1, extra_constraints=[cond])))
    return len(list(mgr.found)), sorted(arms, key=repr), mgr


@pytest.mark.parametrize(
    ("else_", "merged_count"),
    [
        # Same packet count, different bytes: Python ITE-merges.
        pytest.param(_write(MSG_B, 1), 1, id="stdout_bytes_differ"),
        # The other arm reads a second stdin byte and prints two: SimMergeError.
        pytest.param(_read(BUF + 1) + _write(MSG_B, 2), 2, id="stdin_and_stdout_differ"),
    ],
)
def test_divergent_std_streams_merge_like_python(else_, merged_count):
    code = _branchy(_write(MSG_A, 1), else_)
    py_count, py_arms, _ = _run("py", code)
    rust_count, rust_arms, _ = _run("rust", code)
    assert py_count == merged_count
    assert rust_count == py_count
    assert rust_arms == py_arms
    assert ("A", b"A", b"X") in rust_arms


def test_equal_std_streams_still_merge_natively():
    # Both arms print the same byte; only a register differs.
    code = _branchy(_write(MSG_A, 1) + b"\x48\x31\xdb", _write(MSG_A, 1) + b"\x48\xc7\xc3\x07\x00\x00\x00")
    py_count, py_arms, _ = _run("py", code)
    rust_count, rust_arms, mgr = _run("rust", code)
    assert py_count == rust_count == 1
    # Stdout only: an export carries stdin as evaluated bytes, which
    # dumps(0, extra_constraints=...) cannot re-solve.
    assert [(label, out) for label, _, out in rust_arms] == [(label, out) for label, _, out in py_arms]
    assert mgr.stats.get("states_merged_native") == 2
