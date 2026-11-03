"""A stdin packet a bounced Python proc reads keeps its place (angr-azp2b).

scanf_test reads its switch character with a native ``read(0, &c, 1)``, then
calls ``scanf``; with native scanf disabled the bounced Python scanf reads the
number. The bounce used to re-read the packet Rust's read was injected as
(IndexError, errored state), and each found state's ``posix.dumps(0)`` either
lost the scanf packet or put it before the switch character. Every found
state must give the same ``dumps(0)`` as the Python engine: the switch
character, then the digits the checked value parses from.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr.exploration.rust_manager import RustExplorationManager
from tests.common import bin_location

BIN = os.path.join(bin_location, "tests", "x86_64", "scanf_test")
FIND = 0x4007F3
M32 = 2**32

# dumps(1) -> (switch, base, value the binary accepts, mod 2**32)
EXPECTED = {
    b"%%07x\n": (b"x", 16, 0xAAAA),
    b"%%07x and negative numbers\n": (b"x", 16, -0xCDCD),
    b"%%d\n": (b"d", 10, 133337),
    b"%%d and negative numbers\n": (b"d", 10, -1337),
    b"%%u\n": (b"u", 10, 0xAAAA),
    b"%%u and negative numbers\n": (b"u", 10, -0xCDCD),
}


def _explore(engine):
    proj = angr.Project(BIN, auto_load_libs=False)
    state = proj.factory.entry_state()
    if engine == "py":
        simgr = proj.factory.simulation_manager(state)
    else:
        simgr = RustExplorationManager(proj, [state])
        for name in ("scanf", "__isoc99_scanf"):
            simgr.disable_native_procedure(name)
    simgr.explore(find=FIND, num_find=len(EXPECTED) + 4)
    return simgr


@pytest.mark.skipif(not os.path.isfile(BIN), reason="binaries checkout not available")
@pytest.mark.parametrize("engine", ["py", "rust"])
def test_bounced_scanf_packet_follows_native_read(engine):
    simgr = _explore(engine)
    assert not simgr.errored, simgr.errored
    outputs = {s.posix.dumps(1): s.posix.dumps(0) for s in simgr.found}
    for out, (switch, base, value) in EXPECTED.items():
        assert out in outputs, f"{engine}: {out!r} not reached"
        stdin = outputs[out]
        assert stdin[:1] == switch, f"{engine} {out!r}: {stdin!r}"
        assert int(stdin[1:].decode("latin-1"), base) % M32 == value % M32, f"{engine} {out!r}: {stdin!r}"
