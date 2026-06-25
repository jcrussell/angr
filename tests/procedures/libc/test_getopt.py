#!/usr/bin/env python3
from __future__ import annotations

__package__ = __package__ or "tests.procedures.libc"  # pylint:disable=redefined-builtin

import struct
import unittest

import claripy

import angr


class TestGetopt(unittest.TestCase):
    # pylint: disable=no-self-use

    @staticmethod
    def _project():
        b = angr.load_shellcode(b"\x90\x90", "AMD64")
        # getopt's globals are data symbols the program reads back; create them
        # as externs so the SimProc can resolve their guest addresses.
        for name in ("optind", "optarg", "optopt", "opterr"):
            b.loader.extern_object.make_extern(name)
        return b

    @staticmethod
    def _layout(state, strings, base=0x700000):
        """Write a list of byte strings and a NULL-terminated argv array.

        Returns (argc, argv_ptr) and a dict name->addr for each string element.
        """
        addrs = []
        cur = base
        for s in strings:
            state.memory.store(cur, s + b"\x00")
            addrs.append(cur)
            cur += len(s) + 1
        argv_ptr = (cur + 0xF) & ~0xF
        ptr_fmt = "<Q"
        for i, a in enumerate(addrs):
            state.memory.store(argv_ptr + i * 8, struct.pack(ptr_fmt, a))
        state.memory.store(argv_ptr + len(addrs) * 8, struct.pack(ptr_fmt, 0))
        return len(addrs), argv_ptr, addrs

    def _getopt(self, state, b, argc, argv_ptr, optstr_addr, klass="getopt", *extra):
        proc = angr.SIM_LIBRARIES["libc.so.6"][0].get(klass, arch=b.arch)
        proc.state = state
        return proc.run(argc, argv_ptr, optstr_addr, *extra)

    def _read_optind(self, state, b):
        addr = b.loader.find_symbol("optind").rebased_addr
        return state.solver.eval(state.memory.load(addr, 4, endness=b.arch.memory_endness))

    def _read_optarg(self, state, b):
        addr = b.loader.find_symbol("optarg").rebased_addr
        return state.solver.eval(state.memory.load(addr, 8, endness=b.arch.memory_endness))

    def test_short_basic(self):
        b = self._project()
        state = b.factory.blank_state()
        # argv = ["prog", "-a", "-bvalue", "foo", "-c"], optstring "ab:c"
        argc, argv, addrs = self._layout(state, [b"prog", b"-a", b"-bvalue", b"foo", b"-c"])
        optstr = 0x680000
        state.memory.store(optstr, b"ab:c\x00")

        r1 = self._getopt(state, b, argc, argv, optstr)
        assert state.solver.eval(r1) == ord("a")
        assert self._read_optind(state, b) == 2

        r2 = self._getopt(state, b, argc, argv, optstr)
        assert state.solver.eval(r2) == ord("b")
        # optarg points at "value" inside argv[2] (just past "-b")
        assert self._read_optarg(state, b) == addrs[2] + 2
        assert self._read_optind(state, b) == 3

        # "foo" is a non-option operand -> POSIX non-permuting stop
        r3 = self._getopt(state, b, argc, argv, optstr)
        assert state.solver.eval(r3) == 0xFFFFFFFF  # -1

    def test_short_grouped_and_separate_arg(self):
        b = self._project()
        state = b.factory.blank_state()
        # argv = ["p", "-ab", "x"], optstring "ab:" -> 'a', then 'b' optarg="x"
        argc, argv, addrs = self._layout(state, [b"p", b"-ab", b"x"])
        optstr = 0x680000
        state.memory.store(optstr, b"ab:\x00")

        assert state.solver.eval(self._getopt(state, b, argc, argv, optstr)) == ord("a")
        r = self._getopt(state, b, argc, argv, optstr)
        assert state.solver.eval(r) == ord("b")
        # required arg taken from the next argv element "x"
        assert self._read_optarg(state, b) == addrs[2]
        assert self._read_optind(state, b) == 3

    def test_short_unknown_option(self):
        b = self._project()
        state = b.factory.blank_state()
        argc, argv, _ = self._layout(state, [b"p", b"-z"])
        optstr = 0x680000
        state.memory.store(optstr, b"ab\x00")
        r = self._getopt(state, b, argc, argv, optstr)
        assert state.solver.eval(r) == ord("?")
        addr = b.loader.find_symbol("optopt").rebased_addr
        assert state.solver.eval(state.memory.load(addr, 4, endness=b.arch.memory_endness)) == ord("z")

    def test_double_dash_terminates(self):
        b = self._project()
        state = b.factory.blank_state()
        argc, argv, _ = self._layout(state, [b"p", b"--", b"-a"])
        optstr = 0x680000
        state.memory.store(optstr, b"a\x00")
        r = self._getopt(state, b, argc, argv, optstr)
        assert state.solver.eval(r) == 0xFFFFFFFF
        # optind advanced past the "--"
        assert self._read_optind(state, b) == 2

    def test_symbolic_argv_falls_back(self):
        b = self._project()
        state = b.factory.blank_state()
        optstr = 0x680000
        state.memory.store(optstr, b"a\x00")
        sym_argv = claripy.BVS("argv", 64)
        r = self._getopt(state, b, 2, sym_argv, optstr)
        # unconstrained: no concrete commitment, both -1 and an option char possible
        assert len(state.solver.eval_upto(r, 3)) > 1

    @staticmethod
    def _write_option(state, addr, name_ptr, has_arg, flag_ptr, val):
        state.memory.store(addr, struct.pack("<Q", name_ptr))
        state.memory.store(addr + 8, struct.pack("<i", has_arg))
        state.memory.store(addr + 16, struct.pack("<Q", flag_ptr))
        state.memory.store(addr + 24, struct.pack("<i", val))

    def test_getopt_long(self):
        b = self._project()
        state = b.factory.blank_state()
        # argv = ["p", "--verbose", "--file=out", "x"]
        argc, argv, addrs = self._layout(state, [b"p", b"--verbose", b"--file=out", b"x"])
        optstr = 0x680000
        state.memory.store(optstr, b"\x00")  # empty short optstring
        # longopts table: {"verbose",0,NULL,'v'},{"file",1,NULL,'f'},{NULL,...}
        names = 0x690000
        state.memory.store(names, b"verbose\x00")
        state.memory.store(names + 16, b"file\x00")
        table = 0x6A0000
        self._write_option(state, table, names, 0, 0, ord("v"))
        self._write_option(state, table + 32, names + 16, 1, 0, ord("f"))
        self._write_option(state, table + 64, 0, 0, 0, 0)
        longindex = 0x6B0000

        r1 = self._getopt(state, b, argc, argv, optstr, "getopt_long", table, longindex)
        assert state.solver.eval(r1) == ord("v")
        assert state.solver.eval(state.memory.load(longindex, 4, endness=b.arch.memory_endness)) == 0

        r2 = self._getopt(state, b, argc, argv, optstr, "getopt_long", table, longindex)
        assert state.solver.eval(r2) == ord("f")
        # optarg = "out" (just past "--file=")
        assert self._read_optarg(state, b) == addrs[2] + len(b"--file=")
        assert state.solver.eval(state.memory.load(longindex, 4, endness=b.arch.memory_endness)) == 1

        r3 = self._getopt(state, b, argc, argv, optstr, "getopt_long", table, longindex)
        assert state.solver.eval(r3) == 0xFFFFFFFF


if __name__ == "__main__":
    unittest.main()
