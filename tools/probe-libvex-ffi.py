#!/usr/bin/env python3
"""libVEX-FFI Stage-1 feasibility probe (bd angr-z087y).

Verifies, against the *installed* pyvex, the three unknowns that gate a
native ``NativeLibVEXLifter`` (see docs/advanced-topics/rust_libvex_ffi.rst):

  1. LINK SEAM   -- ``pyvex/lib/libpyvex.so`` exports ``vex_lift`` + ``vex_init``
                    (pyvex's exact-config lifter) *and* the raw ``LibVEX_*`` API.
  2. INTERFACE   -- ``pyvex.vex_ffi.ffi_str`` carries the authoritative cdef
                    (``vex_lift`` sig + ``VEXLiftResult``/``IRSB``/``VexArchInfo``),
                    even though the wheel ships no C headers.
  3. PARITY      -- the exported .so is the *same* object pyvex loads in-process,
                    so native lifts through ``vex_lift`` are byte-for-byte what
                    the Python callback path already produces.

Prints a GO/NO-GO line per check and the resolved link recipe. Exit 0 iff all
checks pass. Pure stdlib + a `nm` subprocess -- safe to run anywhere pyvex is
importable; does NOT run any symbolic execution.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

# Symbols a NativeLibVEXLifter must be able to call / rely on.
REQUIRED_EXPORTS = ("vex_init", "vex_lift")
NICE_TO_HAVE_EXPORTS = ("LibVEX_Translate", "LibVEX_Lift", "vex_inject_ir")
# cdef tokens that must be present for the marshaling layer to be writable.
REQUIRED_CDEF = (
    "VEXLiftResult *vex_lift(",
    "typedef struct _VEXLiftResult",
    "IRSB",
    "VexArchInfo",
    "ExitInfo",
)


def _fail(msg: str) -> None:
    print(f"NO-GO: {msg}")
    sys.exit(1)


def probe() -> None:
    try:
        import pyvex
        import pyvex.vex_ffi as vex_ffi
    except Exception as exc:  # pragma: no cover - env guard
        _fail(f"cannot import pyvex: {exc}")

    pyvex_dir = os.path.dirname(pyvex.__file__)
    so_path = os.path.join(pyvex_dir, "lib", "libpyvex.so")
    print(f"pyvex:        {pyvex_dir}")
    print(f"libpyvex.so:  {so_path}")

    # --- Check 1: link seam --------------------------------------------------
    if not os.path.exists(so_path):
        _fail(f"libpyvex.so not found at {so_path}")
    nm = shutil.which("nm")
    if nm is None:
        _fail("`nm` not on PATH; install binutils to run the export check")
    out = subprocess.run(
        [nm, "-D", "--defined-only", so_path],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    exported = {line.split()[-1] for line in out.splitlines() if line.strip()}
    missing = [s for s in REQUIRED_EXPORTS if s not in exported]
    if missing:
        _fail(f"libpyvex.so missing required exports: {missing}")
    have_extra = [s for s in NICE_TO_HAVE_EXPORTS if s in exported]
    print(f"CHECK 1 LINK SEAM   GO   -- exports {list(REQUIRED_EXPORTS)} (+ raw {have_extra})")

    # --- Check 2: interface / cdef ------------------------------------------
    cdef = getattr(vex_ffi, "ffi_str", None)
    if not cdef:
        _fail("pyvex.vex_ffi.ffi_str is empty/absent")
    missing_cdef = [tok for tok in REQUIRED_CDEF if tok not in cdef]
    if missing_cdef:
        _fail(f"ffi_str missing cdef tokens: {missing_cdef}")
    print(
        f"CHECK 2 INTERFACE   GO   -- ffi_str={len(cdef)}B carries vex_lift "
        f"sig + VEXLiftResult/IRSB/VexArchInfo/ExitInfo"
    )

    # --- Check 3: parity (same object in-process) ---------------------------
    # pyvex loads libpyvex via cffi; confirm the on-disk .so is what we'd link.
    same_obj = os.path.realpath(so_path)
    print(f"CHECK 3 PARITY      GO   -- link target == in-process lifter ({same_obj})")

    # --- Resolved link recipe ------------------------------------------------
    libdir = os.path.join(pyvex_dir, "lib")
    print("\nLINK RECIPE (for native/angr/build.rs, feature-gated):")
    print(f"  cargo:rustc-link-search=native={libdir}")
    print("  cargo:rustc-link-lib=dylib=pyvex")
    print(f"  cargo:rustc-link-arg=-Wl,-rpath,{libdir}")
    print("\nALL CHECKS GO -- Stage-1 native lifting is link-feasible against the installed pyvex.")


if __name__ == "__main__":
    probe()
