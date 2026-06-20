#!/usr/bin/env python3
from __future__ import annotations

__package__ = __package__ or "tests.procedures.libc"  # pylint:disable=redefined-builtin

import os
import unittest

import angr


class TestSscanf(unittest.TestCase):
    def test_sscanf_p_spec(self):
        # Regression for the scanf interpret() path: %p reads a hex pointer and
        # must parse like %x (base 16) instead of raising SimProcedureError.
        # Binary-free: drive the sscanf SimProcedure directly. See commit
        # 2f8deb4ca (xmllint cross-engine characterization, bead angr-6d3l).
        p = angr.load_shellcode(b"\x90", arch="amd64")
        state = p.factory.blank_state()
        src, fmt, out = 0x100000, 0x200000, 0x300000
        state.memory.store(src, b"0xdeadbeef\x00")
        state.memory.store(fmt, b"%p\x00")

        sscanf = angr.SIM_PROCEDURES["libc"]["sscanf"]()
        sscanf.execute(state, arguments=[src, fmt, out])

        val = state.solver.eval(state.memory.load(out, 8, endness=p.arch.memory_endness))
        assert val == 0xDEADBEEF
        assert len(state.solver.constraints) == 0 or state.satisfiable()

    def test_sscanf(self):
        from tests.common import bin_location

        test_location = os.path.join(bin_location, "tests")
        test_bin = os.path.join(test_location, "x86_64", "sscanf_test")
        b = angr.Project(test_bin, auto_load_libs=False)
        pg = b.factory.simulation_manager()
        # find the end of main
        expected_outputs = {
            b"0x worked\n",
            b"+0x worked\n",
            b"base +16 worked\n",
            b"base 16 worked\n",
            b"-0x worked\n",
            b"base -16 worked\n",
            b"base 16 length 2 worked\n",
            b"Nope x\n",
            b"base 8 worked\n",
            b"base +8 worked\n",
            b"base +10 worked\n",
            b"base 10 worked\n",
            b"base -8 worked\n",
            b"base -10 worked\n",
            b"Nope u\n",
            b"No switch\n",
        }
        pg.run()
        assert len(pg.deadended) == len(expected_outputs)
        assert {f.posix.dumps(1) for f in pg.deadended} == expected_outputs
        assert len(pg.active) == 0
        assert len(pg.errored) == 0


if __name__ == "__main__":
    unittest.main()
