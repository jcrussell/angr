#!/usr/bin/env python3
from __future__ import annotations

import unittest

import archinfo

from angr import SIM_PROCEDURES, SimState
from tests.common import minimal_project

RLIM_BUF = 0x100000
RLIMIT_STACK = 3


def getrlimit(state, arguments):
    return SIM_PROCEDURES["linux_kernel"]["getrlimit"]().execute(state, arguments=arguments)


class TestGetrlimit(unittest.TestCase):
    def test_rlimit_stack_layout(self):
        # struct rlimit is two unsigned longs, in the arch's memory endianness: 8 bytes on a 32-bit ABI,
        # 16 on a 64-bit one (angr-vmi2l.152).
        for arch in (
            archinfo.ArchX86(),
            archinfo.ArchMIPS32(archinfo.Endness.BE),
            archinfo.ArchARMEL(),
            archinfo.ArchAMD64(),
        ):
            with self.subTest(arch=arch.name, endness=arch.memory_endness):
                state = SimState(project=minimal_project(arch), mode="symbolic")
                state.memory.store(RLIM_BUF, b"\xff" * (2 * arch.bytes + 8))

                assert state.solver.eval(getrlimit(state, [RLIMIT_STACK, RLIM_BUF]).ret_expr) == 0

                def load(off, size):
                    return state.memory.load(RLIM_BUF + off, size, endness=arch.memory_endness)

                assert state.solver.eval_one(load(0, arch.bytes)) == 8388608
                rlim_max = load(arch.bytes, arch.bytes)
                assert any(v.startswith("rlim_max") for v in rlim_max.variables), rlim_max
                assert state.solver.eval_one(load(2 * arch.bytes, 8)) == 0xFFFFFFFFFFFFFFFF


if __name__ == "__main__":
    unittest.main()
