## Session log: 2026-05-14 — angr-jzn8 (ARM64 promotion benchmark)

### Status: complete (commit e0d2746d1, bd angr-jzn8 closed)

### Outcome
Added 1 AArch64 LE inline-ELF synthetic benchmark to the FAST_SUITE.
- Test bench: `aarch64_le_branch` — 10-instruction AArch64 program: ADD +
  ADD imm + MOVZ + CMP chain with symbolic w0; expected w0=42.
- baseline_timings.json now has 24 entries (was 23). Only the new
  `aarch64_le_branch` is added — every other entry is untouched.
- ARM64 promoted Experimental → Supported in the arch matrix.

### Design rationale
- Same shape as angr-2xfz (mips32_le_branch): 10 instructions, find +
  avoid via cmp/branch on a symbolic register.
- Avoids NEON ops (still NeonUnimplemented scaffold per
  invariant-neon-scaffolding-panic-not-fallback memory) — only base
  scalar ops (ADD, MOVZ, CMP, B.cond, B, NOP).
- Marked `rust_only=True`. Reason: 10 instructions is too short to
  amortize Rust's ~250 ms PyO3 init tax. python_time=null skips the
  SLA gate; the 15% threshold gate against rust_time=0.51 still catches
  AArch64 lift/exec regressions.
- 5 trial repeats on Rust: stable rust_time=0.51s, w0=42 every time.

### Files changed (commit e0d2746d1)
- `tests/benchmarks/synthetic_examples/aarch64_le_branch/solve.py` (new)
- `tests/benchmarks/synthetic_examples/aarch64_le_branch/.gitignore` (new — ignores generated .elf)
- `tests/benchmarks/run_regression.py` (FAST_SUITE += aarch64_le_branch)
- `tests/benchmarks/baseline_timings.json` (add aarch64_le_branch entry)
- `docs/advanced-topics/rust_engine.rst` (ARM64: Experimental → Supported)
- `CLAUDE.md` (benchmark count 23 → 24)

### Verified
- 394/395 unit tests pass — 1 flake (`test_model_stability_constraint_order`,
  pre-existing Z3 order-stability flake, unrelated; passes when re-run).
- `run_single.py aarch64_le_branch --both` → Python OK (x0=42),
  Rust OK (x0=42).
- 5 trial repeats on Rust: stable x0=42 every time.
- CI-mode regression run (`--rust-only --skip-bimodal --threshold 0.15`):
  `aarch64_le_branch` not in failures list; failures are pre-existing
  SLA / threshold drift on local machine that don't reflect CI machine.
