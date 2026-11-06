#!/usr/bin/env python3
from __future__ import annotations

import unittest

import archinfo

from angr import SIM_PROCEDURES, SimFile, SimState
from tests.common import minimal_project

STAT_BUF = 0x100000


def fstat64(state, arguments):
    return SIM_PROCEDURES["linux_kernel"]["fstat64"]().execute(state, arguments=arguments)


class TestFstat64(unittest.TestCase):
    def test_mips32_stat64_layout(self):
        # o32 struct stat64, arch/mips/include/uapi/asm/stat.h: st_mode@0x18, st_nlink@0x1c, st_size@0x38,
        # st_blksize@0x58, st_blocks@0x60, 0x68 bytes in all. Both byte orders, since a field stored wider than
        # its kernel width reads back as its value's high half on big-endian.
        for endness in (archinfo.Endness.BE, archinfo.Endness.LE):
            with self.subTest(endness=endness):
                arch = archinfo.ArchMIPS32(endness)
                state = SimState(project=minimal_project(arch), mode="symbolic")
                state.fs.insert("/tmp/f", SimFile(name="f", content=b"hello"))
                fd = state.posix.open(b"/tmp/f", 0)
                state.memory.store(STAT_BUF, b"\xff" * 0x70)

                assert state.solver.eval(fstat64(state, [fd, STAT_BUF]).ret_expr) == 0

                def load(off, size):
                    return state.memory.load(STAT_BUF + off, size, endness=arch.memory_endness)

                mode = load(0x18, 4)
                size = load(0x38, 8)
                assert any(v.startswith("st_mode") for v in mode.variables), mode
                assert any(v.startswith("st_size") for v in size.variables), size
                assert state.solver.eval(load(0x58, 4)) == 0x400
                for off in range(0, 0x68, 4):
                    if off in (0x18, 0x38, 0x3C, 0x58):
                        continue
                    assert state.solver.eval_one(load(off, 4)) == 0, hex(off)
                assert state.solver.eval_one(load(0x68, 8)) == 0xFFFFFFFFFFFFFFFF


if __name__ == "__main__":
    unittest.main()
