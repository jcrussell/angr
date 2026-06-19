#!/usr/bin/env python
"""Showcase driver: drive a real-world x86-64 static binary (busybox) under
the Rust engine (angr-4n26m.1).

Unlike the CTF crackmes that dominate ``baseline_timings.json``, this target
is a piece of *real software* — the BusyBox multi-call utility, the same
binary shipped in Debian/Ubuntu rescue images and most container base layers.
It exercises the Rust VEX interpreter on a large, optimized, stripped,
statically-linked C program rather than a hand-built or contest binary.

Provenance (note, do not vendor — BusyBox is GPLv2 and the binary is a
distro artifact, so the driver loads the system copy and documents it):

    path        /usr/bin/busybox
    package     busybox-static  1:1.36.1-6ubuntu3.1  (Ubuntu 24.04, amd64)
    version     BusyBox v1.36.1
    sha256      dbac288c29ba568459550a2da9e7ae0ded6b1fc728ee9fad3044c44e62d6ac14
    ELF         ELF 64-bit LSB executable, x86-64, statically linked, stripped

A statically-linked, non-PIE target is the cleanest "raw speed" subject: the
whole program (including libc) is in the image, so there are no PLT stubs or
loader-resolution gaps — every executed instruction is real native code the
Rust interpreter lifts and runs. Startup (``_start`` -> ``__libc_start_main``
-> applet dispatch) is deterministic with concrete argv, so a bounded step
budget stays single-state and well inside the fast-tier memory/time envelope
(no state explosion — same property the xmllint tractability probe found,
bd memory ``xmllint-path-b-tractability-probe``).

The driver is engine-agnostic: it builds the simulation manager through
``proj.factory.simulation_manager`` so ``run_single.py --engine rust`` can
swap in ``RustExplorationManager`` transparently, and ``--both`` can time the
Python engine against it for the showcase speed table (angr-4n26m.5).

Run it:
    python tests/benchmarks/run_single.py busybox_static --engine rust
    python tests/benchmarks/run_single.py busybox_static --both
"""

from __future__ import annotations

import os

import claripy

import angr

# Prefer the static system busybox; fall back to the initramfs copy (dynamic
# PIE) only if the static one is absent.
_CANDIDATES = (
    "/usr/bin/busybox",
    "/usr/lib/initramfs-tools/bin/busybox",
)

# Bounded step budget. The deterministic libc/applet-dispatch startup is
# single-state, so this stays cheap; the cap is a hard backstop against any
# symbolic fork storm deeper in the applet logic.
STEP_BUDGET = 60


def _find_busybox():
    for path in _CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No system busybox found. Install it with "
        "`apt install busybox-static` (provides /usr/bin/busybox). "
        f"Looked in: {', '.join(_CANDIDATES)}"
    )


def solve():
    binpath = _find_busybox()

    # auto_load_libs=False: for the static binary there is nothing external to
    # load anyway; for the dynamic fallback it keeps the run bounded. NOTE:
    # the static glibc inside busybox carries IRELATIVE/IFUNC relocations that
    # angr resolves at load time by *executing* the resolver through a
    # Callable, which builds an internal simulation manager. Under
    # run_single.py the engine-swap monkeypatch now routes those internal
    # Callables through the Python engine (its caller-frame guard excludes
    # /angr/callable.py); without that, the Rust manager would reject the
    # resolver state's default SimOptions and the binary would fail to load.
    proj = angr.Project(binpath, auto_load_libs=False)
    assert proj.arch.name == "AMD64", proj.arch.name

    # Concrete applet selector ("echo") keeps the startup + dispatch
    # single-state; a symbolic trailing argv arg gives the engine real
    # symbolic data to carry through memory (echo copies argv to stdout)
    # without inducing an argv-dispatch fork storm. Mirrors the proven
    # ais3_crackme symbolic-argv pattern that runs cleanly under Rust.
    arg = claripy.BVS("arg", 8 * 16)
    state = proj.factory.entry_state(args=["busybox", "echo", arg])
    # Defensive: the Rust manager raises on SYMBOL_FILL_UNCONSTRAINED_REGISTERS
    # (concrete-zero register reads, see _RAISE_OPTION_NAMES in rust_manager.py).
    # busybox's entry_state does not set it, but dropping it keeps the driver
    # robust across angr configs and runs identically under both engines.
    from angr import sim_options as o

    state.options.discard(o.SYMBOL_FILL_UNCONSTRAINED_REGISTERS)

    sm = proj.factory.simulation_manager(state)
    # Bounded run: step at most STEP_BUDGET times (or until the active stash
    # empties). Works identically on the Python SimulationManager and the
    # Rust manager (both honor run(n=...)).
    sm.run(n=STEP_BUDGET)

    # Total states the engine has produced across all stashes — proves the
    # interpreter lifted and executed real busybox basic blocks.
    return sum(len(states) for states in sm.stashes.values())


def test():
    total = solve()
    assert total >= 1, f"expected the engine to retain >=1 state, got {total}"


if __name__ == "__main__":
    total = solve()
    print(f"busybox: engine retained {total} state(s) after bounded run")
