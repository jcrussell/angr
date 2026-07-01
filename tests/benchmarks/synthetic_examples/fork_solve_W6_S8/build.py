#!/usr/bin/env python3
"""Generate + compile a wide-AND-slow fork/solve benchmark variant.

Substitutes the ``__W__`` (width) and ``__S__`` (per-state solve cost) knobs
into ``fork_solve_template.c`` and compiles a ``-O0 -no-pie`` x86-64 ELF, the
same flags ``cow_fork_scaling`` uses. Ships both the substituted ``.c`` and the
ELF next to ``solve.py`` so the bench needs no compiler at test time.

Usage:
    python build.py --w 6 --s 64            # builds fork_solve_W6_S64[.c]
    python build.py --w 5 --s 96 --keep-c   # a lighter/harder variant

The generated binary reads 32 fully-symbolic bytes from stdin, runs them
through W independent ``if (b[i] != 0)`` branches (2^W leaves), then S rounds of
nonlinear mixing, and calls ``reach_target()`` iff the mixed accumulator equals
a fixed constant -- an expensive Z3 check at every leaf.
"""

from __future__ import annotations

import argparse
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "fork_solve_template.c")

# LCG multiplier / increment (glibc rand()); nonlinear mixing with the xor-shift
# makes the final equality a hard bitvector problem for Z3.
_MUL = 1103515245
_INC = 12345


# Default match-mask width (bits) baked into the committed variant. The
# find-address check requires the low M bits of the mixed accumulator to equal a
# fixed pattern; wider M => harder (but still satisfiable) Z3 check.
_DEFAULT_M = 20
_PATTERN = 0xC0FFEE  # masked to M bits at build time


def gen_c_source(w: int, s: int, m: int) -> str:
    with open(TEMPLATE) as f:
        src = f.read()
    branches = [f"    if (b[{i}] != 0) s += {1 << i}u;" for i in range(w)]
    mix = [f"    acc = (acc * {_MUL}u + {_INC}u) ^ (acc >> 3); acc += b[{k & 31}];" for k in range(s)]
    mask = (1 << m) - 1
    pattern = _PATTERN & mask
    gate = f"(acc & {mask:#x}u) == {pattern:#x}u"
    src = src.replace("    /* __BRANCHES__ */", "\n".join(branches))
    src = src.replace("    /* __MIX__ */", "\n".join(mix))
    src = src.replace("/* __GATE__ */ 0", gate)
    return src


def build(w: int, s: int, m: int, out_dir: str) -> str:
    # Mask width M is baked into the source but kept out of the file name so the
    # committed variant stays fork_solve_W<W>_S<S>; the exact M lives in the .c.
    name = f"fork_solve_W{w}_S{s}"
    c_path = os.path.join(out_dir, f"{name}.c")
    bin_path = os.path.join(out_dir, name)
    with open(c_path, "w") as f:
        f.write(gen_c_source(w, s, m))
    subprocess.check_call(
        ["gcc", "-O0", "-no-pie", "-o", bin_path, c_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    print(f"built {bin_path} (W={w}, S={s}, M={m})")
    return bin_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--w", type=int, default=6, help="branch count (2^W leaves); keep <= 6")
    p.add_argument("--s", type=int, default=64, help="nonlinear mixing rounds (per-state solve cost)")
    p.add_argument("--m", type=int, default=_DEFAULT_M, help="match-mask width in bits (Z3 check hardness)")
    p.add_argument("--out-dir", default=HERE)
    args = p.parse_args()
    build(args.w, args.s, args.m, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
