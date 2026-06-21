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

    def test_sscanf_o_spec(self):
        # Regression for the scanf interpret() path: %o reads an octal integer and
        # must parse like the scanf-inject path (base 8) instead of raising
        # SimProcedureError. %o was recognized everywhere except interpret().
        # Binary-free: drive the sscanf SimProcedure directly. Bead angr-4jufb,
        # mirrors the %p fix in commit 2f8deb4ca (bead angr-6d3l).
        p = angr.load_shellcode(b"\x90", arch="amd64")
        state = p.factory.blank_state()
        src, fmt, out = 0x100000, 0x200000, 0x300000
        state.memory.store(src, b"755\x00")
        state.memory.store(fmt, b"%o\x00")

        sscanf = angr.SIM_PROCEDURES["libc"]["sscanf"]()
        sscanf.execute(state, arguments=[src, fmt, out])

        val = state.solver.eval(state.memory.load(out, 4, endness=p.arch.memory_endness))
        assert val == 0o755
        assert len(state.solver.constraints) == 0 or state.satisfiable()

    def test_sscanf_X_spec(self):
        # Regression for the scanf interpret() path: %X (uppercase hex) is
        # recognized by _match_spec (basic_spec/int_sign) but was unhandled at
        # the dispatch sites — the non-SimPackets interpret() path raised
        # SimProcedureError and the SimPackets path mis-parsed it as base 10.
        # Must parse like %x (base 16). Binary-free. Bead angr-vp1dg.
        p = angr.load_shellcode(b"\x90", arch="amd64")
        state = p.factory.blank_state()
        src, fmt, out = 0x100000, 0x200000, 0x300000
        state.memory.store(src, b"DEADBEEF\x00")
        state.memory.store(fmt, b"%X\x00")

        sscanf = angr.SIM_PROCEDURES["libc"]["sscanf"]()
        sscanf.execute(state, arguments=[src, fmt, out])

        val = state.solver.eval(state.memory.load(out, 4, endness=p.arch.memory_endness))
        assert val == 0xDEADBEEF
        assert len(state.solver.constraints) == 0 or state.satisfiable()

    def test_sprintf_hex_octal_no_digit_strip(self):
        # Regression for the printf replace() path: commit 129bd9e645 refactored
        # hex(c_val)[2:] -> f"{c_val:x}"[2:] (and the octal analog). hex()/oct()
        # carry a 2-char prefix that [2:] stripped, but the f-string forms have
        # no prefix, so [2:] silently dropped the first two significant digits
        # (0xFF -> ""). Also covers the new %X spec. Binary-free. Bead angr-vp1dg.
        import claripy

        p = angr.load_shellcode(b"\x90", arch="amd64")
        fmt, out = 0x200000, 0x300000
        sprintf = angr.SIM_PROCEDURES["libc"]["sprintf"]()

        for spec, value, expected in [
            (b"%x\x00", 0xFF, b"ff"),
            (b"%X\x00", 0xFF, b"FF"),
            (b"%x\x00", 0xDEAD, b"dead"),
            (b"%o\x00", 0o755, b"755"),
        ]:
            state = p.factory.blank_state()
            state.memory.store(fmt, spec)
            sprintf.execute(state, arguments=[out, fmt, claripy.BVV(value, 64)])
            data = state.solver.eval(state.memory.load(out, len(expected) + 1), cast_to=bytes)
            assert data == expected + b"\x00", (spec, data)

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
