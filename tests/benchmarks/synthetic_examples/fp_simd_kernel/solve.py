#!/usr/bin/env python
"""FP/SIMD-heavy synthetic benchmark (angr-amtxu).

A single concrete, non-forking path that drives SSE *scalar* FP
(``sqrtsd`` / ``divsd`` / ``mulsd`` / ``addsd`` / ``ucomisd``) and -O3
auto-vectorized *packed* SIMD (``mulps`` / ``addps``) VEX ops through
the Rust interpreter. The rest of the corpus is integer/string-bound,
so ``vex_fallback_count`` and ``vecret_gsptr_fallback_count`` read ZERO
everywhere (bd memory ``benchmark-vecret-gsptr-corpus-zero``) and the
native FP/SIMD interpreter path has no measured coverage. This fixture
gives ``collect_simproc_fallbacks.py`` a workload that exercises it, so
a non-zero reading re-opens native FP/SIMD handler work.

The kernel makes NO libc calls (no ``printf``), so the only VEX ops
between ``main`` entry and return are FP/SIMD arithmetic — SimProcedure
fallbacks cannot pollute the measurement. The vendored binary is built
with::

    gcc -O3 -fno-math-errno -fno-stack-protector -no-pie \\
        -o fp_simd_kernel fp_simd_kernel.c

``-fno-math-errno`` lets GCC lower ``__builtin_sqrt`` to the ``sqrtsd``
instruction instead of a libm call; ``volatile`` seeds in ``main``
defeat constant-folding so the scalar ops survive into the binary.

Run it via::

    python tests/benchmarks/run_single.py fp_simd_kernel --engine rust
    python tests/benchmarks/run_single.py fp_simd_kernel --dump-counters

and read ``vex_fallback_count`` / ``vecret_gsptr_fallback_count`` in the
counter dump to see whether the native interpreter handled every
FP/SIMD op natively (expected: yes, both 0).
"""

from __future__ import annotations

import os

import angr

# Offset of main's `ret` from main's entry (from objdump). The kernel
# input is a `volatile` concrete seed, so the whole compute is a single
# concrete non-forking path; we explore to the ret as the find target
# rather than step blindly, because blank_state leaves rsp unconstrained
# and the final `ret` would otherwise diverge on an unconstrained address.
RET_OFFSET = 0x2DE
MAX_STEPS = 200


def solve():
    bin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fp_simd_kernel")

    proj = angr.Project(bin_path, auto_load_libs=False)
    assert proj.arch.name == "AMD64", proj.arch.name
    main_sym = proj.loader.find_symbol("main")
    assert main_sym is not None, "main symbol not found"
    main_addr = main_sym.rebased_addr

    state = proj.factory.blank_state(addr=main_addr)
    # A concrete 16-aligned stack pointer keeps the path single: the -O3
    # vectorized loops emit alignment-peel checks on the array (stack)
    # addresses, which fork if rsp is left symbolic (blank_state default).
    state.regs.rsp = 0x7FFFFFFFF000

    sm = proj.factory.simulation_manager(state)
    sm.explore(find=main_addr + RET_OFFSET, num_find=1, n=MAX_STEPS)

    return len(sm.found)


def test():
    found = solve()
    assert found == 1, f"expected to reach main's ret once, found {found}"


if __name__ == "__main__":
    found = solve()
    print(f"fp_simd_kernel reached main ret: {found} found state(s)")
