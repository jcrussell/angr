"""Tests for the Windows DLL search-path shim (angr/misc/z3_dll.py).

The shim exists so the rustylib extension resolves the SAME libz3 that claripy
loads — see docs/advanced-topics/rust_wheel_distribution.rst. It cannot be
exercised for real off Windows, so these tests pin the two things that are
platform-independent: the z3 lib directory is discoverable, and the guard rails
(no-op off win32, no-op without z3, idempotent) hold.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

from angr.misc import z3_dll


class TestZ3LibDirDiscovery(unittest.TestCase):
    def test_finds_the_installed_z3_lib_dir(self):
        lib_dir = z3_dll._z3_lib_dir()
        assert lib_dir is not None, "z3-solver is a runtime dep; its lib dir must be discoverable"
        assert lib_dir.is_dir()
        assert lib_dir.name == "lib"

        # It must be the directory holding the shared library claripy loads —
        # that is the whole point of handing it to os.add_dll_directory().
        libs = {p.name for p in lib_dir.iterdir()}
        assert any(name.startswith(("libz3", "z3.dll")) for name in libs), libs

    def test_returns_none_when_z3_is_not_installed(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            assert z3_dll._z3_lib_dir() is None

    def test_returns_none_when_z3_has_no_lib_dir(self):
        fake = mock.Mock(submodule_search_locations=["/nonexistent/z3"])
        with mock.patch("importlib.util.find_spec", return_value=fake):
            assert z3_dll._z3_lib_dir() is None


class TestAddZ3DllDirectory(unittest.TestCase):
    def setUp(self):
        self._saved = (z3_dll._dll_directory, z3_dll._added_dir)
        z3_dll._dll_directory = None
        z3_dll._added_dir = None

    def tearDown(self):
        z3_dll._dll_directory, z3_dll._added_dir = self._saved

    def test_noop_off_windows(self):
        assert sys.platform != "win32", "this branch is the one that runs in CI here"
        with mock.patch("os.add_dll_directory", create=True) as add:
            assert z3_dll.add_z3_dll_directory() is None
        add.assert_not_called()

    def test_registers_the_z3_lib_dir_on_windows(self):
        lib_dir = Path("C:/site-packages/z3/lib")
        with (
            mock.patch.object(z3_dll.sys, "platform", "win32"),
            mock.patch.object(z3_dll, "_z3_lib_dir", return_value=lib_dir),
            mock.patch("os.add_dll_directory", create=True) as add,
        ):
            assert z3_dll.add_z3_dll_directory() == lib_dir
            # Idempotent: a second call must not register the directory twice.
            assert z3_dll.add_z3_dll_directory() == lib_dir
        add.assert_called_once_with(str(lib_dir))
        # The cookie is retained: closing it would undo the registration.
        assert z3_dll._dll_directory is add.return_value

    def test_noop_on_windows_without_z3(self):
        with (
            mock.patch.object(z3_dll.sys, "platform", "win32"),
            mock.patch.object(z3_dll, "_z3_lib_dir", return_value=None),
            mock.patch("os.add_dll_directory", create=True) as add,
        ):
            assert z3_dll.add_z3_dll_directory() is None
        add.assert_not_called()


if __name__ == "__main__":
    unittest.main()
