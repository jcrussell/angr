#!/usr/bin/env python3
"""Regenerate the vendored pyvex cffi cdef header.

Writes ``native/angr/vendor/pyvex_ffi.h`` from the installed pyvex's
``pyvex.vex_ffi.ffi_str`` — the authoritative C interface pyvex's own cffi
uses to marshal libVEX output. Binding Rust (via bindgen, behind the
default-ON ``libvex-ffi`` cargo feature) against this exact cdef is
parity-safe (see docs/advanced-topics/rust_libvex_ffi.rst).

Run this on any pyvex version bump, then re-run the corpus IRSB parity
harness (bd angr-3s5js.4). The struct ABI (VEXLiftResult array sizes,
IRStmt/IRExpr layout) is tied to the pinned pyvex ==9.2.209; a bump can
change it. The pin is frozen (see bd ``avoid-pip-install-deps``), so the
drift risk is dormant.
"""

from __future__ import annotations

import os

import pyvex
import pyvex.vex_ffi as vex_ffi

HEADER_PATH = os.path.join(os.path.dirname(__file__), "..", "native", "angr", "vendor", "pyvex_ffi.h")


def build_header() -> str:
    banner = f"""\
/*
 * pyvex_ffi.h — VENDORED cffi cdef from pyvex.vex_ffi.ffi_str
 *
 * Provenance: pyvex {pyvex.__version__} (pinned ==9.2.209).  This is the
 * authoritative C interface that pyvex's own cffi uses to marshal libVEX
 * output: the vex_lift/vex_init signatures plus VEXLiftResult / IRSB /
 * IRStmt / IRExpr / VexArchInfo / ExitInfo struct layouts.  Binding Rust
 * against this exact cdef is parity-safe (see docs/advanced-topics/
 * rust_libvex_ffi.rst).
 *
 * DO NOT hand-edit.  Regenerate on any pyvex bump with
 * tools/regen-pyvex-ffi-header.py and re-run the parity harness
 * (bd angr-3s5js.4).  ABI drift risk is dormant while the pin is frozen.
 *
 * Consumed by build.rs::generate_pyvex_ffi_bindings (bindgen) under the
 * `libvex-ffi` cargo feature (default-ON).
 */
#include <stddef.h>  /* size_t (used by msg_current_size) */

"""
    body = vex_ffi.ffi_str
    if not body.endswith("\n"):
        body += "\n"
    return banner + body


def main() -> None:
    text = build_header()
    with open(os.path.normpath(HEADER_PATH), "w") as f:
        f.write(text)
    print(f"wrote {len(text)} bytes to {os.path.normpath(HEADER_PATH)}")


if __name__ == "__main__":
    main()
