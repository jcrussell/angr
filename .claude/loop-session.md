## Session log: 2026-05-08, 163rd loop session

### Task: angr-k5mj (closed) — Sync profile_rust_bench.sh --list with vex_engine.rs criterion ids

P3 bug filed in the previous session. Fixed by renaming criterion ids
in `native/angr/benches/vex_engine.rs` so that every name advertised by
`tests/benchmarks/profile_rust_bench.sh --list` resolves to a real
bench id under criterion's regex/substring `--filter`.

### What changed

Commit ee08a03d4 — `bench(rust): rename criterion ids to match
profile_rust_bench.sh --list`:

- `rustbv_build_z3_ast` → `rustbv_z3`
- benchmark_group `symcontext` / fn `fork_2constraints` →
  benchmark_group `symcontext_fork` / fn `2_constraints`
- `symcontext_check_branch_feasibility` → `symcontext_check_branch`
- `symcontext_assume_true` → `symcontext_assume`
- `memory_symbolic_load_16range` → `memory_symbolic_load`
- `memory_fork_16pages` → `memory_fork`

### Verification

Built the bench (`cargo bench --bench vex_engine --no-run`) and
listed ids via `target/release/deps/vex_engine-* --bench --list`.
Confirmed each --list-advertised name now appears as a real bench
id, and `--list <name>` filters resolve to the expected entries.

`python -m pytest tests/engines/test_rust_exploration.py` →
**332/332 passing**.

### Disambiguation gotcha (saved to bd memory)

`--filter symcontext_fork` is a substring match — it now matches both
`symcontext_fork/2_constraints` AND `symcontext_fork_scaling/*`. The
same applies to any name that is a prefix of another. Anchor with
`--filter '^symcontext_fork$'` or `--filter symcontext_fork/` to
isolate the simple fork bench. Memory
`invariant-profile-rust-bench-list-vs-criterion` updated with the
post-fix list and this gotcha.

### Files changed

- `native/angr/benches/vex_engine.rs` (7 lines)

### Tests

332/332 passing on `tests/engines/test_rust_exploration.py`.
