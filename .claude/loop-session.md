# Loop session notes (2026-05-07, 131st loop session)

## Task: angr-n28w — FP transcendentals: sin / cos / exp / log / pow / etc.

### Status: closed (commit ef020d101)

### What landed
Added concrete-only fast paths for x87 FPU transcendentals plus
ARM AArch64 FRECPX. New module `native/angr/src/vex/transcendentals.rs`
with `try_concrete_binop_rm` and `try_concrete_triop_rm`. Wired into
`VEXOps::binop` (Raw arm) and `VEXOps::binop_with_rm` (new Raw arm
before the falls-through-to-binop wildcard).

Opcodes covered (all hex from libvex_ir.h, validated via pyvex.const):
- Binops:  SinF64 0x14e6, CosF64 0x14e7, TanF64 0x14e8,
           2xm1F64 0x14e9, RecpExpF64 0x14fa, RecpExpF32 0x14fb
- Triops:  AtanF64 0x14de, Yl2xF64 0x14df, Yl2xp1F64 0x14e0,
           ScaleF64 0x14e5

Symbolic inputs return None → existing fresh-sym fallback in
`expressions.rs` (Z3 has no transcendental theory).

### Why this fixes a real bug
Previously `IROp::Raw(opcode)` returned `Err(RawOpcode)` and the
fallback in `IRExpr::Binop`/`IRExpr::Triop` substituted concrete 0
for concrete inputs (silently wrong) or fresh-unconstrained-symbolic
for symbolic. The bead description called this "Python fallback" —
this was inaccurate. Concrete-input transcendentals were returning 0,
not getting executed in Python.

### Validation
- Rust unit tests: 9 new in vex::transcendentals, all pass
- Python integration: 261/261 in test_rust_exploration.py
- fauxware: OK rust 0.35s
- ekopartyctf2016_sokohashv2 (uses fyl2x/fscale/f2xm1):
  Before: 12.091s (silent-zero path), After: 11.29s (correct concrete)

### Memories saved
- invariant-vex-transcendentals — opcode constants, dispatch sites,
  how to add new transcendentals
- avoid-silent-zero-raw-fallback — describes the prior buggy behavior
  so future debug sessions recognize the smell
- build-env-pyo3-workaround — required env steps because pip install
  is broken in .venv (PYO3_PYTHON env, target dir is workspace root)

### Build env (still broken — use workaround)
- `PYO3_PYTHON=$(which python) cargo build --release --manifest-path native/angr/Cargo.toml`
- `cp ./target/release/librustylib.so angr/rustylib.cpython-312-x86_64-linux-gnu.so`
- Tests need `PYTHONPATH=$(pwd)` to find `angr` from the repo

## Status: complete
