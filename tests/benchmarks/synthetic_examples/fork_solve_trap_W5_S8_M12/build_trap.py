#!/usr/bin/env python3
"""Generate + compile the wide-AND-slow fork/solve benchmark WITH the trap.

Extends ``fork_solve_W6_S8/build.py`` with a ``--t`` (trap) knob. Substitutes
``__W__`` (width), ``__S__`` (per-state solve cost), ``__M__`` (mask width) and
``__T__`` (number of Python-bounce ``trap_point`` call sites) into
``fork_solve_trap_template.c`` and compiles a ``-O0 -no-pie`` x86-64 ELF, the
same flags the base bench uses. Ships both the substituted ``.c`` and the ELF
next to ``solve.py`` so the bench needs no compiler at test time.

Usage:
    python build_trap.py --w 5 --s 8 --m 12 --t 4    # committed variant (reproduces the ELF)

The trap count ``T`` is kept out of the file name (the committed bench dir is
``fork_solve_trap_W5_S8_M12``); the exact ``T`` lives in the ``.c`` and in
``solve.py``. Each ``trap_point(acc)`` call, once hooked by an identity
SimProcedure in ``solve.py``, bounces the state to Python and (at workers>1)
re-migrates the whole active frontier — so more T => stronger migration
regression.
"""

from __future__ import annotations

import argparse
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "fork_solve_trap_template.c")

# LCG multiplier / increment (glibc rand()); nonlinear mixing with the xor-shift
# makes the final equality a hard bitvector problem for Z3.
_MUL = 1103515245
_INC = 12345

# Default match-mask width (bits). Committed variant bakes M=12.
_DEFAULT_M = 12
_PATTERN = 0xC0FFEE  # masked to M bits at build time


def gen_c_source(w: int, s: int, m: int, t: int) -> str:
    with open(TEMPLATE) as f:
        src = f.read()
    branches = [f"    if (b[{i}] != 0) s += {1 << i}u;" for i in range(w)]
    mix = [f"    acc = (acc * {_MUL}u + {_INC}u) ^ (acc >> 3); acc += b[{k & 31}];" for k in range(s)]
    trap = ["    acc = trap_point(acc);" for _ in range(t)]
    mask = (1 << m) - 1
    pattern = _PATTERN & mask
    gate = f"(acc & {mask:#x}u) == {pattern:#x}u"
    src = src.replace("    /* __BRANCHES__ */", "\n".join(branches))
    src = src.replace("    /* __MIX__ */", "\n".join(mix))
    src = src.replace("    /* __TRAP__ */", "\n".join(trap))
    src = src.replace("/* __GATE__ */ 0", gate)
    return src


def build(w: int, s: int, m: int, t: int, out_dir: str) -> str:
    # T is baked into the source but kept out of the file name so the committed
    # variant stays fork_solve_trap_W<W>_S<S>_M<M>; the exact T lives in the .c.
    name = f"fork_solve_trap_W{w}_S{s}_M{m}"
    c_path = os.path.join(out_dir, f"{name}.c")
    bin_path = os.path.join(out_dir, name)
    with open(c_path, "w") as f:
        f.write(gen_c_source(w, s, m, t))
    subprocess.check_call(
        ["gcc", "-O0", "-no-pie", "-o", bin_path, c_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    print(f"built {bin_path} (W={w}, S={s}, M={m}, T={t})")
    return bin_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--w", type=int, default=5, help="branch count (2^W leaves); keep 5 for exact-drain 32")
    p.add_argument("--s", type=int, default=8, help="nonlinear mixing rounds (per-state solve cost)")
    p.add_argument("--m", type=int, default=_DEFAULT_M, help="match-mask width in bits (Z3 check hardness)")
    p.add_argument("--t", type=int, default=4, help="number of trap_point() Python-bounce call sites")
    p.add_argument("--out-dir", default=HERE)
    args = p.parse_args()
    build(args.w, args.s, args.m, args.t, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
