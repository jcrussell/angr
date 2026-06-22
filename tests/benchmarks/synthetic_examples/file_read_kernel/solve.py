#!/usr/bin/env python
"""File-I/O open()/read() harness (angr-w5llj — evidence for angr-11djq.7).

A single concrete path that pre-seeds ``/data/secret.txt`` on the Python
side (``state.fs.insert``), then open()s + read()s it and branches on the
magic prefix. The purpose is to give ``collect_simproc_fallbacks.py`` a
READ-side, pre-seeded-file workload so it can measure whether the Rust
native filesystem serves the content or falls back to a Python syscall
handler.

This is the harness that T2-MEASURE (angr-11djq.4) asked for and that the
sibling angr-11djq.7 (RustPosixState fd/file sync) is evidence-gated on.
The earlier candidates were the wrong layer (bd memory
``djq7-evidence-needs-file-io-harness``):

  * ``write_stream_heavy`` is WRITE-side cle stdout/stderr stdio only;
  * ``xmllint_getenv`` reaches getenv before any file open();

both read ZERO syscall fd-sync fallbacks. This kernel is the missing
READ-side pre-seeded-file harness.

The vendored binary is built with::

    gcc -O2 -fno-stack-protector -no-pie -o file_read_kernel file_read_kernel.c

Run it via::

    python tests/benchmarks/run_single.py file_read_kernel --engine rust
    python tests/benchmarks/run_single.py file_read_kernel --dump-counters

and read the ``syscall_*`` / ``*fallback*`` counters: a non-zero syscall
fd-sync fallback while still reaching the magic-prefix success path is the
evidence that funds angr-11djq.7.
"""

from __future__ import annotations

import os

import angr
from angr.storage.file import SimFile

# Offset of main's `ret` from main's entry (from objdump). The pre-seeded
# content makes the magic compare concrete, so the whole run is a single
# non-forking path; we explore to the ret as the find target.
RET_OFFSET = 0x5C
MAX_STEPS = 200

SECRET_PATH = "/data/secret.txt"
SECRET_CONTENT = b"MAGIC-bytes-pre-seeded-on-python-side\n"


def solve():
    bin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file_read_kernel")

    proj = angr.Project(bin_path, auto_load_libs=False)
    assert proj.arch.name == "AMD64", proj.arch.name
    main_sym = proj.loader.find_symbol("main")
    assert main_sym is not None, "main symbol not found"
    main_addr = main_sym.rebased_addr

    state = proj.factory.entry_state(addr=main_addr)
    # Pre-seed the file on the Python side. angr-11djq.7's question is
    # whether this insert is visible to the Rust-native open()/read() path.
    state.fs.insert(SECRET_PATH, SimFile(SECRET_PATH, content=SECRET_CONTENT, size=len(SECRET_CONTENT)))

    sm = proj.factory.simulation_manager(state)
    sm.explore(find=main_addr + RET_OFFSET, num_find=1, n=MAX_STEPS)

    return len(sm.found)


def test():
    found = solve()
    assert found == 1, f"expected to reach main's ret once, found {found}"


if __name__ == "__main__":
    found = solve()
    print(f"file_read_kernel reached main ret: {found} found state(s)")
