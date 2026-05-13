## Session log: 2026-05-13 — angr-ikt9 (Add NEON criterion microbench group) — CLOSED

### Task

**angr-ikt9** (P3, CLOSED) — Add `rustbv_neon_ops` criterion bench
group to `native/angr/benches/vex_engine.rs` so the NEON ops landed
in angr-bkcs.2 (commit `da0966893`: Mul8x16 + VGetElem + VSetElem)
have microbench coverage.

### What landed

7 benches in a new `rustbv_neon_ops` group:

  mul64_concrete_baseline         18.6 ns   (scalar Iop_Mul I64)
  mul8x16_concrete                68.1 ns   (~3.7x scalar, 16-lane)
  mul8x16_symbolic                 2.0 µs   (Z3 AST construction)
  get_elem8x16_concrete_idx       19.9 ns   (bit-slice fast path)
  get_elem8x16_symbolic_idx        1.3 µs   (ITE over 16 lanes)
  set_elem8x16_concrete_idx       21.0 ns   (bit-twiddle)
  set_elem8x16_symbolic_idx        1.9 µs   (ITE + concat_le rebuild)

Concrete paths are essentially free vs scalar; symbolic paths dominated
by ITE-chain construction, not Z3 solve.

### Files touched

- `native/angr/benches/vex_engine.rs` — new `bench_rustbv_neon_ops` fn,
  imports for `IROp`/`IRType`/`VEXOps`, group registered in
  `criterion_group!`.
- `tests/benchmarks/profile_rust_bench.sh` — added `rustbv_neon_ops`
  line to the `--list` output.

### Verification

- `cargo check --release --benches` ✓
- `cargo bench --no-run` → bench binary built ✓
- `./target/release/deps/vex_engine-* --list | grep neon` shows all 7
  benches registered ✓
- Bench binary ran end-to-end with 1s warm-up + 1s measurement on
  every NEON sub-bench ✓
- `tools/rebuild-rust.sh --cargo-only` to rebuild the Python .so ✓
- `pytest tests/engines/test_rust_exploration.py` — 389 pass, same
  3 pre-existing failures (`dcas`, `pipe_native`, `dup2_native`),
  unrelated to this change.

### Tricky bit found

`Iop_SetElem*` is a VEX Triop but does NOT carry a rounding mode.
In Rust, it dispatches through `VEXOps::binop_with_rm` (the rm slot
is reinterpreted as the vector operand — see `vex/ops.rs:694`). Calls
from a bench/test must use `binop_with_rm`, not `binop` or any
hypothetical `triop` API. Saved to memory `invariant-vsetelem-bench-via-binop-with-rm`.

### Commit

`526482ae3` bench(rust-neon): add rustbv_neon_ops criterion group (angr-ikt9)

### Memories saved

- NEW `benchmark-neon-ops-baseline` — initial timings + how to run
  them. Use these as the regression baseline for future NEON op work.
- NEW `invariant-vsetelem-bench-via-binop-with-rm` — calling-convention
  invariant for VSetElem and other non-rm Triops.

### Still-open followup from prior sessions (unchanged)

Three pre-existing test failures (`dcas`, `pipe native`, `dup2 native`)
remain. Reproducible after `tools/rebuild-rust.sh --cargo-only`. No bd
issue covers them yet; deferred pending fix to broken venv pip rebuild
(`avoid-broken-venv-pip-rebuild`) so we can compare paths.
