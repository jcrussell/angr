## Session log: 2026-05-13 — angr-nivm (Move user-facing Rust docs into Sphinx tree) — CLOSED

### Task

**angr-nivm** (P2, CLOSED) — Per the 2026-05-12 comment redirection, the
goal is not a CLAUDE.md/DEVELOPMENT.md split but rather: move user-facing
Rust engine documentation out of CLAUDE.md and into the docs/ Sphinx tree
so it integrates with upstream docs; keep CLAUDE.md narrow to
Claude-operator content (build/test/profile/log commands, file pointers).

### What landed

Moved into `docs/advanced-topics/rust_engine.rst` (which already covered
SimOption matrix, state.inspect, and known slower benchmarks):

- **Usage** — quickstart example using `RustExplorationManager` and the
  `use_rust_engine=True` form.
- **Architecture** — Rust core / Python wrapper / PyO3 FFI / shared Z3 /
  feature-flag bullets.
- **Architecture support matrix** — full per-arch table plus
  Skeleton/Experimental/Supported definitions and the wired-up-but-
  unverified list. Includes the Cdecl x86 latent-bug caveat.
- **Z3 solver API** — `RustSolverContext` usage example.
- **Z3 solver profiling counters** — full `get_solver_stats()` reference.

CLAUDE.md sections removed (replaced with a single "User-facing
documentation" pointer block):

- Architecture
- Architecture Support Matrix
- SimOption Coverage
- state.inspect Unsupported
- Rust Symbolic Execution
- Z3 Solver Integration (z3-rs 0.19)
- Z3 Solver Profiling Counters

CLAUDE.md retained (operator content): Makefile shortcuts, Building the
Rust Extension (prereqs, bootstrap, build commands, common issues),
Running Tests, Benchmark gates in CI, Profiling Rust Benches, Debugging
Rust-side Logs, Key Files, Branch: rust-symex, Current Status
(test/bench/perf table).

### Files touched

- `CLAUDE.md` — 367 → 259 lines (−108).
- `docs/advanced-topics/rust_engine.rst` — 569 → 760 lines (+191).

### Verification

- `git diff --stat` confirms net +83 lines: 208 insertions, 125
  deletions across the two files.
- New rst markup is consistent with surrounding sections (top-level
  `---` underlines, `.. code-block:: python`, `.. list-table::`).
- Sphinx is not installed locally (memory `invariant-sphinx-build-workaround`)
  so the build itself cannot be verified here; visual review only.

### Commit

`027faaad7` docs(rust-engine): move user-facing rust engine docs into
Sphinx tree (angr-nivm)

### Memories saved

- NEW `rust-engine-docs-canonical-rst` — points future work at
  `docs/advanced-topics/rust_engine.rst` as the canonical user-facing
  doc; CLAUDE.md should cite it rather than duplicate prose.

### Note on acceptance criteria

Original criteria said "CLAUDE.md <= 120 lines". After the redirection
the practical target became "remove user-facing content"; CLAUDE.md is
259 lines, dominated by Makefile/build/test/profile/log
*operator* sections, which are intentionally retained. The redirection
trumped the line-count number.
