#!/usr/bin/env python
"""Synthetic scanf/fgets stdin bench with a replay oracle (angr-vmi2l.342).

``replay_oracle.py`` replays a bench's found stdin on the real binary, but no
gated bench reached the native ``scanf`` or ``fgets`` procedures: baby-re hooks
scanf with its own proc and fauxware / defcamp_r100 read through other paths.
A native scanf that stored the wrong bytes in ``posix.dumps(0)`` therefore
passed every gate.

The checked-in ELF (``cli_scanf_fgets.c``, gcc ``-O0 -no-pie`` x86-64) reads a
number with ``scanf("%d")`` and a line with ``fgets``, and reaches ``win`` only
for 1337 followed by a line starting with ``open``. test() replays the found
stdin on the binary and asserts the marker ``win`` prints.

stdin is left to angr's default symbolic stream (no ``stdin=`` seed), because a
harness seed makes a numeric scanf defer to Python (``do_scanf`` in
angr-symex-native procedures/scanf.rs) and the bench would not reach the native
procedure. ``auto_load_libs=False`` keeps the libc hooks in the extern region,
where the Rust engine dispatches native procedures.

``rust_only=True`` in ``run_regression.FAST_SUITE``: the bench gates the
recorded stdin, not a speed claim.

Run it:
    python tests/benchmarks/run_single.py cli_scanf_fgets --engine rust --test
    python tests/benchmarks/run_single.py cli_scanf_fgets --dump-counters
"""

from __future__ import annotations

import os
import subprocess

import angr

# The path forks a handful of times; the cap is a backstop against a fork storm.
STEP_BUDGET = 200


def _binary_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "cli_scanf_fgets")


def solve():
    proj = angr.Project(_binary_path(), auto_load_libs=False, use_sim_procedures=True)
    win = proj.loader.find_symbol("win")
    assert win is not None, "win symbol not found"

    state = proj.factory.entry_state()
    sm = proj.factory.simulation_manager(state)
    sm.explore(find=win.rebased_addr, num_find=1, n=STEP_BUDGET)
    return sm


def test():
    sm = solve()
    assert sm.found, "did not reach win within the step budget"
    # Reaching win is not enough: the recorded stdin must drive the real
    # binary there too.
    stdin = sm.found[0].posix.dumps(0)
    out = subprocess.run([_binary_path()], input=stdin, capture_output=True, timeout=10, check=False).stdout
    assert b"SCANF FGETS OK" in out, f"replaying stdin {stdin!r} printed {out!r}"


if __name__ == "__main__":
    sm = solve()
    print(f"cli_scanf_fgets: reached win -> {len(sm.found)} found state(s)")
