#!/usr/bin/env python
"""Write-heavy cle-stream characterisation benchmark (angr-csyy9).

A single concrete path that issues many stdio writes (fputs / fputc /
fwrite) against the cle-loaded ``stdout`` / ``stderr`` FILE* externs. Under
the Rust engine the native write-side SimProcedures (puts.rs::fputs,
puts.rs::fputc, fwrite.rs::fwrite) call ``read_fileno_for_stream``, which hits an
unmapped lazy page for the cle stdout/stderr FILE* and returns
``ProcedureError::Memory`` — the dispatcher then falls back to Python.
The fallback is CORRECT (Python resolves the right fd); this bench
exists to *quantify* that per-call cost (measured ~2.8ms/call: 192
fallbacks, time_in_callbacks 0.534s — NOT the ~100ms/call the
superseded write-side-fileno-fallback-correct memory estimated) so a
human can decide whether
the proper native fix (loader symbol resolution exposed to SimProcedure
context) is worth building. See bd memory write-side-fileno-fallback-correct
and task angr-csyy9.

The loop bound (48 iters x 4 writes = ~192 cle-stream write calls) is
concrete, so execution stays on a single non-forking path: the bench
measures per-write fallback cost, not fork scaling. Because every write
routes through Python under Rust, this bench is EXPECTED to run slower under
Rust than Python — it is a known-regression demonstrator, NOT a speed win,
and is deliberately kept OUT of the gated baseline_timings.json suite.

Run it via::

    python tests/benchmarks/run_single.py write_stream_heavy --both
    python tests/benchmarks/run_single.py write_stream_heavy --dump-counters

and read ``simprocedure_python_fallback`` / ``python_fallbacks`` in the
counter dump to see the fallback volume.
"""

from __future__ import annotations

import os

import angr

# Step cap: the program is concrete and returns from main after the write
# loop; cap the manager so a stray unconstrained return after main cannot
# diverge. ~48 iters of straight-line writes settle well under this.
MAX_STEPS = 400


def solve():
    bin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "write_stream_heavy")

    proj = angr.Project(bin_path, auto_load_libs=False)
    main_sym = proj.loader.find_symbol("main")
    assert main_sym is not None, "main symbol not found"

    state = proj.factory.blank_state(addr=main_sym.rebased_addr)

    sm = proj.factory.simulation_manager(state)
    steps = 0
    while sm.active and steps < MAX_STEPS:
        sm.step()
        steps += 1

    return steps


def test():
    steps = solve()
    # Sanity: the write loop must actually execute (more than a handful of
    # blocks) before the path settles. A near-zero count means the binary
    # never reached the write loop (load/entry regression).
    assert steps > 10, f"expected the write loop to execute many steps, got {steps}"


if __name__ == "__main__":
    print(f"steps = {solve()}")
