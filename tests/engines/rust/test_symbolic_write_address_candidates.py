"""A symbolic-address store reaches every candidate Python writes (angr-vmi2l.324).

Under SYMBOLIC_WRITE_ADDRESSES Python's ``Range(128)`` writes ``ITE(addr == a,
data, mem[a])`` at every candidate, mapping each fresh page it touches. Rust
used to skip candidates on pages neither side held and to hand sets over its
ITE cap to the host's full store, which a proxy-backed state dropped. A CAS at
a symbolic address took the same host store even for a single candidate.
"""

from __future__ import annotations

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
TAIL = b"\xeb\xfe\xeb\xfe"  # fall-through: jmp $; taken: jmp $


def _store_program(mask: int) -> bytes:
    # and rdi, mask; mov byte [rdi+0x10000], 1; movzx eax, byte [0x10005];
    # cmp al, 1; je taken
    return (
        b"\x48\x83\xe7"
        + bytes([mask])
        + b"\xc6\x87\x00\x00\x01\x00\x01"
        + b"\x0f\xb6\x04\x25\x05\x00\x01\x00"
        + b"\x3c\x01\x74\x02"
    )


# mov cl, 7; and rdi, 0x3f; lock cmpxchg byte [rdi+0x10000], cl;
# cmp byte [rdi+0x10000], 7; je taken  (al is symbolic, so the compare is
# undecided and the CAS stores an ITE). A dword CAS here errors in Python
# itself (an ITE over mismatched widths).
CAS_PROGRAM = (
    b"\xb1\x07"
    + b"\x48\x83\xe7\x3f"
    + b"\xf0\x0f\xb0\x8f\x00\x00\x01\x00"
    + b"\x80\xbf\x00\x00\x01\x00\x07"
    + b"\x74\x02"
)


def _reaches_taken(engine: str, body: bytes, options: set) -> bool:
    code = body + TAIL
    proj = angr.load_shellcode(code, arch="amd64", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD, add_options=options | {angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY})
    mgr = RustExplorationManager(proj, [state]) if engine == "rust" else proj.factory.simulation_manager(state)
    mgr.explore(find=LOAD + len(code) - 2, n=10)
    return len(mgr.found) == 1


@pytest.mark.parametrize("engine", ["py", "rust"])
@pytest.mark.parametrize(
    ("mask", "expected"),
    [(0x3F, True), (0x7F, True), (0xFF, False)],
    ids=["64_candidates", "128_candidates", "over_range_takes_max"],
)
def test_symbolic_write_addresses_store_reaches_every_candidate(engine, mask, expected):
    options = {angr.options.SYMBOLIC_WRITE_ADDRESSES}
    assert _reaches_taken(engine, _store_program(mask), options) is expected


@pytest.mark.parametrize("engine", ["py", "rust"])
def test_cas_at_symbolic_address_lands(engine):
    assert _reaches_taken(engine, CAS_PROGRAM, set())
