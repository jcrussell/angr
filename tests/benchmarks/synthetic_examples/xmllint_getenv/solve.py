#!/usr/bin/env python
"""Bounded real-binary bench: reach ``getenv`` in xmllint (angr-11djq.3 / T1c).

Unlike the CTF crackmes that dominate ``baseline_timings.json``, this target is
a real-world, libc-heavy command-line utility — ``xmllint`` from libxml2. The
driver runs the *de-risked* bounded slice validated by the iter-11 tractability
probe (bd memory ``xmllint-path-b-tractability-probe``): with a symbolic 16-byte
stdin and the standard CLI options, symbolic execution stays **single-state**
through the deterministic startup and reaches ``getenv@plt`` in ~185 steps
(FAST tier, no state explosion, well inside the 4 GB cap).

This gives us (a) real-workload regression coverage on a non-CTF binary and
(b) a syscall / native-coverage fallback *measurement surface* on real libc
code — inspect it with ``run_single.py xmllint_getenv --dump-counters``
(``syscall_python_fallback_count`` etc.). It explicitly does NOT make the
fuzzer-vs-simproc a/b call tracked in angr-75mc — that stays a human decision.

``getenv`` sits on the deterministic CLI-option-parsing path *before* the
symbolic stdin is consumed, so the symbolic input is carried but not yet
branched on: the run is a clean mechanics smoke, not an entity-resolution
exercise (the parser is not reached this cheaply — see the probe memory).

Provenance: the binary is vendored in angr-examples (``xmllint/xmllint_bin``,
an x86-64 PIE built from libxml2). Resolved via ``ANGR_EXAMPLES_DIR`` — the
same env var ``run_single.py`` and CI already set — so the driver does not
duplicate the 80 KB artifact into this repo.

The driver builds its simulation manager through
``proj.factory.simulation_manager`` so ``run_single.py --engine rust`` swaps in
``RustExplorationManager`` transparently and ``--both`` can time the Python
engine against it.

Run it:
    python tests/benchmarks/run_single.py xmllint_getenv --engine rust
    python tests/benchmarks/run_single.py xmllint_getenv --both
    python tests/benchmarks/run_single.py xmllint_getenv --dump-counters
"""

from __future__ import annotations

import os

import claripy

import angr
from angr import sim_options as so

# Hard backstop on total steps. The probe reached getenv in ~185 steps on a
# single state; this cap protects the bench from any unexpected fork storm
# deeper in the option parser.
STEP_BUDGET = 400


def _binary_path():
    base = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")
    return os.path.join(base, "xmllint", "xmllint_bin")


def solve():
    target = _binary_path()
    if not os.path.exists(target):
        raise FileNotFoundError(
            f"xmllint binary not found at {target}. Set ANGR_EXAMPLES_DIR to the "
            "angr-examples examples/ directory (CI does this automatically)."
        )

    # xmllint CLI: read XML from stdin ('-'), quiet/no-network/recover/entities.
    xmllint_args = [target, "--noout", "--nonet", "--recover", "--noent", "-"]

    # use_sim_procedures=True keeps libc startup cheap and gives the native
    # SimProcedures (and their Python fallbacks) a measurement surface.
    proj = angr.Project(target, auto_load_libs=True, use_sim_procedures=True)

    # 16 symbolic stdin bytes: carried through startup but not yet branched on
    # when getenv is reached, so the run stays single-state.
    stdin = claripy.BVS("stdin", 16 * 8)
    state = proj.factory.entry_state(
        args=xmllint_args,
        stdin=stdin,
        add_options={
            so.ZERO_FILL_UNCONSTRAINED_MEMORY,
            so.ZERO_FILL_UNCONSTRAINED_REGISTERS,
        },
    )

    # With use_sim_procedures=True angr hooks getenv inside libc and redirects
    # the call straight to the hook, bypassing the local PLT stub — so target
    # the resolved libc symbol address (where the SimProcedure lives), not the
    # main-object PLT entry (which is never executed).
    getenv_sym = proj.loader.find_symbol("getenv")
    assert getenv_sym is not None, "getenv symbol not found in loaded libc"
    getenv_addr = getenv_sym.rebased_addr

    sm = proj.factory.simulation_manager(state)
    # Bounded explore: stop at the first state to reach getenv, or after the
    # step budget is exhausted. n= is honored by both the Python
    # SimulationManager and the Rust manager.
    sm.explore(find=getenv_addr, num_find=1, n=STEP_BUDGET)
    return sm


def test():
    sm = solve()
    assert sm.found, "did not reach getenv within the step budget"


if __name__ == "__main__":
    sm = solve()
    print(f"xmllint: reached getenv -> {len(sm.found)} found state(s)")
