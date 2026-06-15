"""CADET_00001 wrapper that runs the convergent subset of phases.

Upstream ``solve.py`` (``angr-examples/examples/CADET_00001/solve.py``)
runs three phases against the CGC easter-egg/buffer-overflow binary:

1. Buffer overflow: ``while len(sm.unconstrained)==0: sm.step()`` until an
   unconstrained state is reached, then ``posix.dumps(0)``.
2. Easter egg: ``sm.explore(find=0x804833E)`` then ``posix.dumps(0/1)``.
3. Easter egg *again*, via a raw ``while True: sm.step(); break if any
   active.addr == 0x804833E`` loop that bypasses ``explore()``.

Phases 1 and 2 converge cleanly under the Rust engine (phase 2 via the
two-phase eager-retry in :class:`RustExplorationManager`, angr-027h).
Phase 3, however, is **pathological under Rust**: it only converges with
``set_block_granular(True)`` + ``set_materialize_unconstrained_forks(True)``
(angr-bmyx / angr-ckdy), and even then takes ~158 s over ~538
block-granular steps with a growing active stash — and those two flags
are mutually exclusive with the fork-drop behaviour phase 2's explore
relies on, so no single manager config runs all three phases. Running the
full upstream script under Rust TIMEOUTs (>280 s vs Python's ~22 s).

This wrapper runs only the convergent subset (phases 1+2) so
``run_single.py`` / ``run_regression.py`` can record a representative,
fair Rust-vs-Python number for the idiomatic ``explore(find=)`` workload.
See bd memory ``benchmark-cadet-phase3-not-a-bench`` and ``angr-027h``.

The binary lives upstream (``CADET_00001/CADET_00001``), not in this
synthetic-examples directory, so we ``chdir`` to the upstream example dir
and load ``./CADET_00001`` relative to that cwd — matching the upstream
solve.py and the cmu_binary_bomb_partial wrapper convention.
"""

from __future__ import annotations

import os

import angr

_EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")
_UPSTREAM_DIR = os.path.join(_EXAMPLES_DIR, "CADET_00001")
_BINARY = os.path.join(_UPSTREAM_DIR, "CADET_00001")

if not os.path.exists(_BINARY):
    raise FileNotFoundError(
        f"CADET_00001 binary not found at {_BINARY}. Set ANGR_EXAMPLES_DIR or clone angr/angr-examples."
    )

os.chdir(_UPSTREAM_DIR)

# Easter-egg basic block (where the "EASTER EGG!" text is printed).
EASTER_EGG_ADDR = 0x804833E


def main():
    project = angr.Project("./CADET_00001", auto_load_libs=False)

    # Phase 1: find the buffer overflow (overwriting the return address
    # produces an unconstrained state). save_unconstrained keeps it.
    print("finding the buffer overflow...")
    sm = project.factory.simulation_manager(save_unconstrained=True)
    while len(sm.unconstrained) == 0:
        sm.step()
    crashing_input = sm.unconstrained[0].posix.dumps(0)
    print("buffer overflow found!")
    print(repr(crashing_input))

    # Phase 2: find the easter egg via the idiomatic explore(find=).
    print("finding the easter egg...")
    sm = project.factory.simulation_manager(project.factory.entry_state())
    sm.explore(find=EASTER_EGG_ADDR)
    found = sm.found[0]
    solution1 = found.posix.dumps(0)
    stdout1 = found.posix.dumps(1)
    print("easter egg found!")
    print(repr(solution1))
    print(repr(stdout1))

    # Phase 3 (upstream raw step-loop egg hunt) intentionally skipped — see
    # module docstring: pathological under Rust, no shared config with phase 2.

    return (crashing_input, solution1, stdout1)


if __name__ == "__main__":
    print(repr(main()))
