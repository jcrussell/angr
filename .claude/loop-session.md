## Session log: 2026-05-08, 162nd loop session

### Tasks: angr-13ir + angr-j1nk (both closed) — CLAUDE.md docs additions

Two related docs tasks bundled because both touch CLAUDE.md and the second
finished the first one's narrative arc (profiling -> debugging).

### angr-13ir — Document profile_rust_bench.sh in CLAUDE.md (P3)

**Outcome: closed.** Commit 247de1f0b adds a "Profiling Rust Benches"
section to CLAUDE.md describing tests/benchmarks/profile_rust_bench.sh:
auto-tool detection (cargo-flamegraph > perf > callgrind), --secs/--filter
flags, output dir target/profile/<filter-or-all>/, tool prerequisites.

**Surprise finding (saved to bd memory invariant-profile-rust-bench-list-vs-criterion):**
The script's `--list` output advertises group names like `rustbv_z3`,
`symcontext_fork`, `symcontext_check_branch`, but the actual criterion
benchmark_group / bench_function names in vex_engine.rs are
`rustbv_build_z3_ast`, `symcontext` (group) / `fork_2constraints` (fn),
`symcontext_check_branch_feasibility`. So `--filter rustbv_z3` matches
nothing, and `--filter symcontext_fork` misses the symcontext group.
Filed angr-k5mj (P3 bug) to sync names.

The CLAUDE.md docs intentionally avoid republishing the misleading list —
they tell users to run `--list` instead.

### angr-j1nk — ANGR_RUST_LOG env var + Debugging section (P3)

**Outcome: closed.** Commit 1fa809945:
- New section "Debugging Rust-side Logs" in CLAUDE.md
- Code: `_apply_rust_log_env()` helper in rust_manager.py reads
  `ANGR_RUST_LOG` once on first manager construction and calls
  `set_rust_log_level()`
- Verified end-to-end: `ANGR_RUST_LOG=debug` surfaces `[rust:DEBUG]`
  z3::ast / z3::solver lines on stderr through fauxware

**Key clarification (saved to bd memory invariant-rust-log-no-env-logger):**
The engine uses a hand-rolled StderrLogger (engine.rs:1476-1516) via
`log::set_logger` + `log::set_max_level` — NOT env_logger. Two consequences
documented in CLAUDE.md:
1. Standard `RUST_LOG` env var is **not** honored
2. Per-module log filters are not supported, only a single global level

### Files changed
- `CLAUDE.md` — added "Profiling Rust Benches" + "Debugging Rust-side Logs"
- `angr/exploration/rust_manager.py` — `_apply_rust_log_env()` and call
  site in `RustExplorationManager.__init__`

### bd memories saved
- `invariant-profile-rust-bench-list-vs-criterion` — script --list ≠ actual
  criterion ids; filter prefixes that work vs. don't
- `invariant-rust-log-no-env-logger` — engine uses custom StderrLogger,
  RUST_LOG silently ignored, ANGR_RUST_LOG is the one to use

### Followups filed
- `angr-k5mj` (P3 bug) — Sync profile_rust_bench.sh --list with actual
  criterion group names in vex_engine.rs

### Tests
332/332 passing on tests/engines/test_rust_exploration.py.
