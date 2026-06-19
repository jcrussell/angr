#!/usr/bin/env python3
"""Multi-arch breadth showcase demo for the Rust engine (angr-4n26m.6).

Runs the **same logical crackme** — "find the symbolic input ``x`` that drives
the program to its success block" — across every architecture the Rust engine
*Supports* in the arch matrix (rust_engine.rst, "Architecture support matrix",
~300-395), and verifies each run reaches the find address AND solves the input
to the one correct value (42). The point is **breadth / parity correctness**,
NOT speed: there is deliberately no Python baseline and no timing claim here.

Honesty framing (epic angr-4n26m): multi-arch is a "runs across N arches"
PARITY story, not a "faster on arch X" story. The only real non-x86 ELF bench
in the repo (android_arm_license_validation) is ~0.8x — *slower* under Rust —
so a non-x86 speed claim would be dishonest. This demo therefore makes a
correctness claim only: the engine decodes the instruction stream, syncs
registers, resolves the conditional branch, and the solver recovers the unique
satisfying input — identically across 32/64-bit and little/big-endian targets.

Each target is a tiny hand-assembled program (blob or inline ELF) lifted
verbatim from the arch integration tests in
``tests/engines/rust/test_multiarch.py`` (so the byte encodings are already
CI-verified). No external binaries or cross-compilers are required — every
target is built in-memory, which is what makes this demo reproducible from a
clean checkout. The shared crackme shape is::

    x = input
    if 2*x + 16 == 100:   # i.e. x == 42
        success()
    else:
        fail()

(MIPS uses a direct ``ADDIU t0, 42`` / ``BEQ`` compare to the same effect.)

Usage::

    python tests/benchmarks/show_multiarch.py             # human report
    python tests/benchmarks/show_multiarch.py --json \
        > tests/benchmarks/multiarch_numbers.json         # durable artifact
    python tests/benchmarks/show_multiarch.py --quick     # 3-arch subset (faster)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import resource
import struct
import sys
import tempfile
import time

EXPECTED_INPUT = 42  # every target's unique satisfying input


# ---------------------------------------------------------------------------
# Target builders. Each returns (project, state, symbol, find, avoid) with a
# fresh symbolic input already wired into the entry register / memory. The
# byte encodings are copied from tests/engines/rust/test_multiarch.py, where
# they are decode-verified end-to-end in CI.
# ---------------------------------------------------------------------------


def _build_amd64(tmpdir):
    """x86-64 (AMD64), 32-bit LE blob: 2*eax + 16 == 100 -> eax == 42."""
    import claripy

    import angr

    code = bytes.fromhex(
        "01C0"  # add eax, eax
        "83C010"  # add eax, 0x10
        "83F864"  # cmp eax, 0x64
        "7406"  # je  +6  -> 0x10 (found)
        "EB0C"  # jmp +12 -> 0x18 (avoid)
        "90909090"
        "9090909090909090"
        "90"
    )
    path = os.path.join(tmpdir, "amd64_branch.bin")
    with open(path, "wb") as fh:
        fh.write(code)
    proj = angr.Project(
        path,
        main_opts={"backend": "blob", "arch": "amd64", "base_addr": 0x400000},
        auto_load_libs=False,
    )
    state = proj.factory.blank_state(addr=0x400000)
    eax = claripy.BVS("eax", 32)
    state.regs.eax = eax
    return proj, state, eax, 0x400010, 0x400018


def _build_x86(tmpdir):
    """x86 (32-bit) LE blob: 2*eax + 16 == 100 -> eax == 42."""
    import claripy

    import angr

    code = bytes.fromhex("01C083C01083F8647406EB0C90909090909090909090909090")
    path = os.path.join(tmpdir, "x86_branch.bin")
    with open(path, "wb") as fh:
        fh.write(code)
    proj = angr.Project(
        path,
        main_opts={"backend": "blob", "arch": "x86", "base_addr": 0x400000},
        auto_load_libs=False,
    )
    state = proj.factory.blank_state(addr=0x400000)
    eax = claripy.BVS("eax", 32)
    state.regs.eax = eax
    return proj, state, eax, 0x400010, 0x400018


def _build_armeb(tmpdir):
    """ARM big-endian (ARMEB) blob: 2*r0 + 16 == 100 -> r0 == 42."""
    import claripy

    import angr

    code = struct.pack(
        ">IIIIIIIII",
        0xE0800000,  # add r0, r0, r0
        0xE2800010,  # add r0, r0, #16
        0xE3A01064,  # mov r1, #100
        0xE1500001,  # cmp r0, r1
        0x0A000000,  # beq +0 -> 0x18 (found)
        0xEA000001,  # b   +4 -> 0x20 (avoid)
        0xE1A00000,  # nop (found)
        0xE1A00000,
        0xE1A00000,  # nop (avoid)
    )
    path = os.path.join(tmpdir, "armeb_branch.bin")
    with open(path, "wb") as fh:
        fh.write(code)
    proj = angr.Project(
        path,
        main_opts={"backend": "blob", "arch": "armeb", "base_addr": 0x10000},
        auto_load_libs=False,
    )
    state = proj.factory.blank_state(addr=0x10000)
    r0 = claripy.BVS("r0", 32)
    state.regs.r0 = r0
    return proj, state, r0, 0x10018, 0x10020


def _elf64(code, e_machine, e_flags, entry_off):
    """Assemble a minimal little-endian ELF64 with one PT_LOAD over the code."""
    BASE = 0x400000
    EHDR_SIZE = 64
    PHDR_SIZE = 56
    total = EHDR_SIZE + PHDR_SIZE + len(code)
    entry = BASE + EHDR_SIZE + PHDR_SIZE + entry_off
    ehdr = b"\x7fELF" + bytes([2, 1, 1, 0, 0]) + b"\x00" * 7
    ehdr += struct.pack(
        "<HHIQQQIHHHHHH",
        2,  # e_type = ET_EXEC
        e_machine,
        1,
        entry,
        EHDR_SIZE,  # e_phoff
        0,
        e_flags,
        EHDR_SIZE,
        PHDR_SIZE,
        1,
        0,
        0,
        0,
    )
    phdr = struct.pack("<IIQQQQQQ", 1, 5, 0, BASE, BASE, total, total, 0x1000)
    return ehdr + phdr + code, entry


def _build_aarch64(tmpdir):
    """AArch64 inline ELF: calls double_it (2*w0) then checks == 84 -> w0 == 42."""
    import claripy

    import angr

    code = struct.pack(
        "<IIIIIIIIII",
        0x0B000000,  # add w0, w0, w0   (double_it)
        0xD65F03C0,  # ret
        0x97FFFFFE,  # bl -8 -> double_it
        0x52800A81,  # movz w1, #84
        0x6B01001F,  # cmp w0, w1
        0x54000040,  # b.eq +8 -> found
        0x14000003,  # b +12 -> avoid
        0xD503201F,  # nop (found)
        0xD503201F,
        0xD503201F,  # nop (avoid)
    )
    elf_bytes, _entry = _elf64(code, 0xB7, 0, entry_off=8)  # entry skips double_it
    path = os.path.join(tmpdir, "aarch64_call.elf")
    with open(path, "wb") as fh:
        fh.write(elf_bytes)
    proj = angr.Project(path, auto_load_libs=False)
    state = proj.factory.blank_state(addr=proj.entry)
    x0 = claripy.BVS("x0", 64)
    state.regs.x0 = x0
    return proj, state, x0, 0x400094, 0x40009C


# MIPS code is identical for MIPS32/MIPS64 here (5-bit regs, ADDIU sign-extends).
_MIPS_CODE = struct.pack(
    "<IIIIIII",
    0x2408002A,  # addiu t0, zero, 42
    0x10880003,  # beq a0, t0, +3 -> +0x14 (found)
    0x00000000,  # nop (delay slot)
    0x10000002,  # b +2 -> +0x18 (avoid)
    0x00000000,
    0x00000000,  # nop (found)
    0x00000000,  # nop (avoid)
)


def _build_mips64(tmpdir):
    """MIPS64 LE inline ELF: ADDIU/BEQ compare a0 == 42."""
    import claripy

    import angr

    elf_bytes, entry = _elf64(_MIPS_CODE, 0x08, 0x60000000, entry_off=0)
    path = os.path.join(tmpdir, "mips64le_branch.elf")
    with open(path, "wb") as fh:
        fh.write(elf_bytes)
    proj = angr.Project(path, auto_load_libs=False)
    state = proj.factory.blank_state(addr=proj.entry)
    a0 = claripy.BVS("a0", 64)
    state.regs.a0 = a0
    return proj, state, a0, entry + 0x14, entry + 0x18


def _build_mips32(tmpdir):
    """MIPS32 LE inline ELF32: ADDIU/BEQ compare a0 == 42."""
    import claripy

    import angr

    BASE = 0x400000
    EHDR_SIZE = 52
    PHDR_SIZE = 32
    total = EHDR_SIZE + PHDR_SIZE + len(_MIPS_CODE)
    entry = BASE + EHDR_SIZE + PHDR_SIZE
    ehdr = b"\x7fELF" + bytes([1, 1, 1, 0, 0]) + b"\x00" * 7
    ehdr += struct.pack(
        "<HHIIIIIHHHHHH",
        2,
        0x08,  # EM_MIPS
        1,
        entry,
        EHDR_SIZE,
        0,
        0x50001000,  # EF_MIPS_ARCH_32 | O32
        EHDR_SIZE,
        PHDR_SIZE,
        1,
        0,
        0,
        0,
    )
    phdr = struct.pack("<IIIIIIII", 1, 0, BASE, BASE, total, total, 5, 0x1000)
    path = os.path.join(tmpdir, "mips32le_branch.elf")
    with open(path, "wb") as fh:
        fh.write(ehdr + phdr + _MIPS_CODE)
    proj = angr.Project(path, auto_load_libs=False)
    state = proj.factory.blank_state(addr=proj.entry)
    a0 = claripy.BVS("a0", 32)
    state.regs.a0 = a0
    return proj, state, a0, entry + 0x14, entry + 0x18


# Ordered so --quick takes a 32-bit-LE, 64-bit-LE, 32-bit-BE spread.
TARGETS = [
    ("AMD64", "64", "LE", _build_amd64),
    ("AARCH64", "64", "LE", _build_aarch64),
    ("ARMEB", "32", "BE", _build_armeb),
    ("X86", "32", "LE", _build_x86),
    ("MIPS32", "32", "LE", _build_mips32),
    ("MIPS64", "64", "LE", _build_mips64),
]
QUICK = {"AMD64", "AARCH64", "ARMEB"}


def run_target(name, build, max_steps=100):
    """Build, explore, and verify one arch target. Returns a result dict."""
    from angr.exploration import RustExplorationManager

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        proj, state, sym, find, avoid = build(tmpdir)
        arch_name = proj.arch.name
        endness = proj.arch.memory_endness
        mgr = RustExplorationManager(proj, [state])
        mgr.explore(find=find, avoid=avoid, num_find=1, max_steps=max_steps)
        found_ok = len(mgr.found) >= 1
        solved = None
        sat = None
        if found_ok:
            f = mgr.found[0]
            sat = bool(f.solver.satisfiable())
            solved = int(f.solver.eval(sym))
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    ok = bool(found_ok and sat and solved == EXPECTED_INPUT)
    return {
        "label": name,
        "arch": arch_name,
        "endness": endness,
        "found": found_ok,
        "satisfiable": sat,
        "solved_input": solved,
        "expected_input": EXPECTED_INPUT,
        "correct": ok,
        "elapsed_ms": elapsed_ms,
    }


def characterize(quick=False):
    results = []
    for name, _bits, _end, build in TARGETS:
        if quick and name not in QUICK:
            continue
        results.append(run_target(name, build))
    return {
        "demo": "multiarch_parity",
        "claim": "correctness/parity only; NOT a speed claim",
        "expected_input": EXPECTED_INPUT,
        "n_arches": len(results),
        "all_correct": all(r["correct"] for r in results),
        "results": results,
    }


def _print_report(res):
    print("=" * 72)
    print("Rust engine multi-arch PARITY demo (angr-4n26m.6)")
    print("Same crackme (input==42 reaches success) across architectures.")
    print("Breadth/correctness only — NOT a speed claim (no Python baseline).")
    print("=" * 72)
    print(f"{'arch':<10} {'bits/end':<10} {'found':<6} {'solved':<8} {'correct':<8} {'ms':>7}")
    print("-" * 72)
    for r in res["results"]:
        solved = "-" if r["solved_input"] is None else str(r["solved_input"])
        print(
            f"{r['label']:<10} {r['endness']:<10} "
            f"{r['found']!s:<6} {solved:<8} {r['correct']!s:<8} {r['elapsed_ms']:>7}"
        )
    print("-" * 72)
    verdict = "PASS" if res["all_correct"] else "FAIL"
    print(f"{res['n_arches']} arches, all_correct={res['all_correct']}  [{verdict}]")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--quick", action="store_true", help="run a 3-arch subset (AMD64/AARCH64/ARMEB)")
    args = parser.parse_args(argv)

    # Self-cap address space at 3 GB: this runs angr in-process and the loop
    # box has no swap. Subprocess invocation inherits this guard. The targets
    # are tiny, so this is a backstop, not a real constraint.
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))

    res = characterize(quick=args.quick)

    if args.json:
        json.dump(res, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_report(res)

    return 0 if res["all_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
