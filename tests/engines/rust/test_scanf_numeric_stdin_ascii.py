"""A numeric scanf conversion on stdin records ASCII digits (angr-vmi2l.108).

Native scanf minted one free value per numeric conversion and recorded it as
a stdin symbol, so ``posix.dumps(0)`` spliced the value's little-endian bytes
where a real scanf(3) reads digits, and the format's literal separators were
missing. Python reads the digit bytes and constrains them to spell the value
(``FormatString._spell``); native scanf now does the same. Each found state's
``dumps(0)`` is replayed through ``int()`` against the value the binary
checks, under both engines.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager
from tests.common import bin_location

BIN_DIR = os.path.join(bin_location, "tests", "x86_64")
M32 = 2**32

# dumps(1) -> (base, field values the binary accepts, mod 2**32)
SINGLE = {
    b"%%07x\n": (16, [0xAAAA]),
    b"%%07x and negative numbers\n": (16, [-0xCDCD]),
    b"%%d\n": (10, [133337]),
    b"%%d and negative numbers\n": (10, [-1337]),
    b"%%u\n": (10, [0xAAAA]),
    b"%%u and negative numbers\n": (10, [-0xCDCD]),
}
MULTI = {
    b"%%04x.%%04x.%%04x\n": (16, [0xAAAA, 0xBBBB, 0xCCCC]),
    b"%%04x.%%04x.%%04x and negative numbers\n": (16, [-0xCD] * 3),
    b"%%d.%%d.%%d\n": (10, [133337, 1337, 13337]),
    b"%%d.%%d.%%d and negative numbers\n": (10, [-1337] * 3),
    b"%%u.%%u.%%u\n": (10, [0xAAAA, 0xBBBB, 0xCCCC]),
    b"%%u and negative numbers\n": (10, [-0xCDCD] * 3),
}
CASES = {
    "scanf_test": (0x4007F3, (), SINGLE),
    "scanf_multi_test": (0x40083E, (0x4006DB, 0x400776, 0x40080B), MULTI),
}


def _found(engine, binary):
    find, avoid, expected = CASES[binary]
    proj = angr.Project(os.path.join(BIN_DIR, binary), auto_load_libs=False)
    state = proj.factory.entry_state()
    if engine == "py":
        simgr = proj.factory.simulation_manager(state)
    else:
        simgr = RustExplorationManager(proj, [state])
    simgr.explore(find=find, avoid=avoid, num_find=len(expected) + 4)
    return {s.posix.dumps(1): s.posix.dumps(0) for s in simgr.found}, expected


@pytest.mark.skipif(not os.path.isdir(BIN_DIR), reason="binaries checkout not available")
@pytest.mark.parametrize("binary", sorted(CASES))
@pytest.mark.parametrize("engine", ["py", "rust"])
def test_dumps0_parses_to_the_checked_value(engine, binary):
    outputs, expected = _found(engine, binary)
    for out, (base, values) in expected.items():
        assert out in outputs, f"{engine}: {out!r} not reached"
        stdin = outputs[out]
        # The first byte is the switch character the binary reads with %c.
        fields = stdin[1:].decode("latin-1").split(".")
        assert len(fields) == len(values), f"{engine} {out!r}: {stdin!r}"
        got = [int(field, base) % M32 for field in fields]
        assert got == [v % M32 for v in values], f"{engine} {out!r}: {stdin!r}"
