#!/usr/bin/env python3
"""Generate + compile the PARTIAL-bounce fork/solve benchmark (angr-nkoct).

Sibling of ``fork_solve_trap/build_trap.py``. Adds a ``--b`` (bounce-fraction
exponent k) knob and interleaves the mixing rounds with the trap levels.
Substitutes ``__W__`` (width), ``__S__`` (solve cost), ``__M__`` (mask width),
``__T__`` (trap levels) and ``__B__`` (k) into ``fork_solve_pbounce_template.c``
and compiles a ``-O0 -no-pie`` x86-64 ELF (same flags as the base bench). Ships
both the substituted ``.c`` and the ELF so the bench needs no compiler at test
time.

Usage:
    python build_pbounce.py --w 6 --s 8 --m 12 --t 4 --b 2  # committed variant

Unlike the full-bounce trap bench, only ``2^-k`` of the ``2^W`` leaves bounce at
each level (the gate tests ``k`` concrete bits of the leaf index ``s``); the
rest keep stepping worker-locally. That partial overlap is what the steady-state
loop converts into a wall-clock win. ``T`` and ``B`` are baked into the ``.c``
and named in the committed dir (``fork_solve_pbounce_W<W>_S<S>_M<M>_B<k>``).
"""

from __future__ import annotations

import argparse
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "fork_solve_pbounce_template.c")

# LCG multiplier / increment (glibc rand()); nonlinear mixing with the xor-shift
# makes the final equality a hard bitvector problem for Z3.
_MUL = 1103515245
_INC = 12345

_DEFAULT_M = 12
_PATTERN = 0xC0FFEE  # masked to M bits at build time


def _mix_round(k: int) -> str:
    return f"    acc = (acc * {_MUL}u + {_INC}u) ^ (acc >> 3); acc += b[{k & 31}];"


def gen_c_source(w: int, s: int, m: int, t: int, b: int) -> str:
    with open(TEMPLATE) as f:
        src = f.read()

    branches = [f"    if (b[{i}] != 0) s += {1 << i}u;" for i in range(w)]

    # Distribute S mixing rounds across T levels (remainder folded into the
    # last level), then a gated trap per level. The gate tests `b` concrete bits
    # of the leaf index `s`, rotating the tested field by level so the bouncing
    # subset differs each level. Fraction that bounces per level = 2^-b.
    gate_mask = (1 << b) - 1
    per_level = s // t
    remainder = s % t
    levels: list[str] = []
    mix_k = 0
    for level in range(t):
        rounds = per_level + (remainder if level == t - 1 else 0)
        levels.append(f"    /* level {level}: {rounds} mixing round(s) then a gated trap */")
        for _ in range(rounds):
            levels.append(_mix_round(mix_k))
            mix_k += 1
        shift = level % w
        # Concrete per-leaf gate: no fork, no solve. 2^-b of leaves bounce.
        levels.append(f"    if (((s >> {shift}u) & {gate_mask:#x}u) == 0u) acc = trap_point(acc);")

    mask = (1 << m) - 1
    pattern = _PATTERN & mask
    gate = f"(acc & {mask:#x}u) == {pattern:#x}u"

    src = src.replace("    /* __BRANCHES__ */", "\n".join(branches))
    src = src.replace("    /* __LEVELS__ */", "\n".join(levels))
    src = src.replace("/* __GATE__ */ 0", gate)
    return src


def build(w: int, s: int, m: int, t: int, b: int, out_dir: str) -> str:
    name = f"fork_solve_pbounce_W{w}_S{s}_M{m}_B{b}"
    c_path = os.path.join(out_dir, f"{name}.c")
    bin_path = os.path.join(out_dir, name)
    with open(c_path, "w") as f:
        f.write(gen_c_source(w, s, m, t, b))
    subprocess.check_call(
        ["gcc", "-O0", "-no-pie", "-o", bin_path, c_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    print(f"built {bin_path} (W={w}, S={s}, M={m}, T={t}, B={b} -> 2^-{b} bounce fraction)")
    return bin_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--w", type=int, default=6, help="branch count (2^W leaves)")
    p.add_argument("--s", type=int, default=8, help="total nonlinear mixing rounds")
    p.add_argument("--m", type=int, default=_DEFAULT_M, help="find-gate mask width in bits")
    p.add_argument("--t", type=int, default=4, help="number of interleaved gated trap levels")
    p.add_argument("--b", type=int, default=2, help="bounce-fraction exponent k (fraction = 2^-k)")
    p.add_argument("--out-dir", default=HERE)
    args = p.parse_args()
    if args.b >= args.w:
        p.error("--b must be < --w (else no leaf ever bounces)")
    build(args.w, args.s, args.m, args.t, args.b, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
