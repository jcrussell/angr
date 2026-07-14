"""Windows DLL search-path shim for the ``angr.rustylib`` extension.

The Rust engine hands raw Z3 ``Ast`` pointers across the FFI boundary, and an
``Ast`` is only valid inside the ``Z3_context`` that minted it. So the extension
and claripy's ``z3`` bindings must load *one* libz3, never two. On ELF and
Mach-O ``native/angr/build.rs`` guarantees that by baking a runpath into the
extension that points at the ``z3-solver`` package's ``lib`` directory (the
prebuilt wheels relativize it to ``$ORIGIN/../z3/lib``).

PE has no runpath. The Windows equivalent is to put that same directory on the
DLL search path *before* the extension is loaded: since CPython 3.8, extension
modules are loaded with ``LoadLibraryEx`` restricted to the directories
registered via :func:`os.add_dll_directory`, and claripy's ``z3`` loads its
``libz3.dll`` by ``ctypes`` out of that very directory — so both halves resolve
the same file.

See ``docs/advanced-topics/rust_wheel_distribution.rst``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# The cookie returned by os.add_dll_directory() removes the directory from the
# search path when closed; hold it for the life of the process.
_dll_directory = None
_added_dir: Path | None = None


def _z3_lib_dir() -> Path | None:
    """Locate the ``lib`` directory of the installed ``z3-solver`` package.

    Uses :func:`importlib.util.find_spec` rather than ``import z3`` so that
    resolving the path at ``angr`` import time does not drag the (heavy) z3
    bindings in as a side effect.
    """
    try:
        spec = importlib.util.find_spec("z3")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        lib_dir = Path(location) / "lib"
        if lib_dir.is_dir():
            return lib_dir
    return None


def add_z3_dll_directory() -> Path | None:
    """Put the ``z3-solver`` package's ``lib`` directory on the DLL search path.

    No-op (returns ``None``) off Windows, where the extension carries a runpath,
    and when ``z3-solver`` is not installed — in the latter case letting the
    extension fail its own import with the platform loader's error is more
    informative than anything raised from here.

    Idempotent: repeated calls return the directory added by the first one.
    """
    global _dll_directory, _added_dir

    if sys.platform != "win32":
        return None
    if _added_dir is not None:
        return _added_dir

    lib_dir = _z3_lib_dir()
    if lib_dir is None:
        return None

    _dll_directory = os.add_dll_directory(str(lib_dir))
    _added_dir = lib_dir
    return lib_dir
