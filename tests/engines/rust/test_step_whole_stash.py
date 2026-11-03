#!/usr/bin/env python3
"""``step()`` advances every active state once, as SimulationManager.step does (angr-vmi2l.246).

The Rust step used to advance one policy-selected state, so a frontier of two
or more states fell behind the Python engine's after each ``step()`` and
``run(n=N)``.
"""

from __future__ import annotations

from collections import Counter

import pytest

import angr
from angr.exploration import RustExplorationManager
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)

BASE = 0x1000


def _rel8(opcode: int, pc: int, target: int) -> bytes:
    return bytes([opcode, (target - (pc + 2)) & 0xFF])


def _branch_tree() -> bytes:
    """A call-free branch tree whose paths reach the join at +0x40 in uneven block counts.

    +0x00 forks on rdi (taken: the two-block +0x20/+0x28 arm), +0x05 forks on
    rsi (taken: +0x30, else +0x0a); every arm jumps to +0x40.
    """
    code = bytearray(b"\x90" * 0x48)
    for off, insn in (
        (0x00, b"\x48\x85\xff" + _rel8(0x74, 0x03, 0x20)),  # test rdi, rdi; je +0x20
        (0x05, b"\x48\x85\xf6" + _rel8(0x74, 0x08, 0x30)),  # test rsi, rsi; je +0x30
        (0x0A, _rel8(0xEB, 0x0A, 0x40)),  # jmp +0x40
        (0x20, b"\x48\xff\xc0" + _rel8(0xEB, 0x23, 0x28)),  # inc rax; jmp +0x28
        (0x28, b"\x48\xff\xc0" + _rel8(0xEB, 0x2B, 0x40)),  # inc rax; jmp +0x40
        (0x30, _rel8(0xEB, 0x30, 0x40)),  # jmp +0x40
    ):
        code[off : off + len(insn)] = insn
    return bytes(code)


def _forked(rust: bool):
    """A manager one step in: two active states, at +0x05 and +0x20."""
    kwargs = {"engine": "rust"} if rust else {}
    project = angr.load_shellcode(_branch_tree(), arch="amd64", load_address=BASE, **kwargs)
    mgr = project.factory.simulation_manager(project.factory.blank_state(addr=BASE))
    mgr.step()
    return mgr


def _addrs(mgr) -> Counter:
    return Counter(s.addr for s in mgr.active)


@pytest.fixture
def forked_pair():
    py, rust = _forked(False), _forked(True)
    assert isinstance(rust, RustExplorationManager)
    assert _addrs(rust) == _addrs(py) == Counter([BASE + 0x05, BASE + 0x20])
    return py, rust


class TestStepAdvancesWholeStash:
    def test_one_step_matches_python(self, forked_pair):
        py, rust = forked_pair
        before = _addrs(rust)

        py.step()
        rust.step()

        assert _addrs(rust) == _addrs(py)
        assert not set(_addrs(rust)) & set(before)

    def test_run_n_matches_python(self, forked_pair):
        py, rust = forked_pair

        py.run(n=2)
        rust.run(n=2)

        assert _addrs(rust) == _addrs(py) == Counter({BASE + 0x40: 3})

    def test_round_stashes_are_emptied(self, forked_pair):
        _, rust = forked_pair
        rust.step()
        counts = rust.stash_counts()
        assert not any(v for k, v in counts.items() if k.startswith("_step_round"))
