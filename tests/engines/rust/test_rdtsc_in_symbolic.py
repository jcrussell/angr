"""RDTSC and IN keep both sides of a branch over their result (angr-vmi2l.69).

Python's dirty helpers return a fresh symbol for each (``RDTSC`` and ``IN``),
so a timing check on a TSC delta or a test of a port value is feasible both
ways. Rust used to answer them natively with a concrete counter and 0xFF,
which silently dropped one side.
"""

from __future__ import annotations

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
TAIL = b"\xeb\xfe\xeb\xfe"  # fall-through: jmp $; taken: jmp $

PROGRAMS = {
    # rdtsc; mov rsi, rax; rdtsc; sub rax, rsi; cmp rax, 0x10000; ja taken
    ("amd64", "rdtsc"): b"\x0f\x31\x48\x89\xc6\x0f\x31\x48\x29\xf0\x48\x3d\x00\x00\x01\x00\x77\x02",
    # rdtsc; mov esi, eax; rdtsc; sub eax, esi; cmp eax, 0x10000; ja taken
    ("x86", "rdtsc"): b"\x0f\x31\x89\xc6\x0f\x31\x29\xf0\x3d\x00\x00\x01\x00\x77\x02",
    # in al, dx; cmp al, 5; je taken
    ("amd64", "in"): b"\xec\x3c\x05\x74\x02",
    ("x86", "in"): b"\xec\x3c\x05\x74\x02",
}


def _reaches(engine: str, arch: str, insn: str, target: int) -> bool:
    code = PROGRAMS[(arch, insn)] + TAIL
    proj = angr.load_shellcode(code, arch=arch, load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD)
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=LOAD + len(code) - target, n=10)
    return len(mgr.found) == 1


@pytest.mark.parametrize("engine", ["py", "rust"])
@pytest.mark.parametrize("arch", ["amd64", "x86"])
@pytest.mark.parametrize("insn", ["rdtsc", "in"])
@pytest.mark.parametrize("target", [4, 2], ids=["fallthrough", "taken"])
def test_branch_over_dirty_result_keeps_both_sides(engine, arch, insn, target):
    assert _reaches(engine, arch, insn, target)
