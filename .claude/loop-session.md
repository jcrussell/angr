## Session log: 2026-05-13 — angr-4ffe (Add Makefile)

### Task
P2 DX: Add a repo-root Makefile of canonical targets (rebuild, test,
profile-bench, lint, …) so contributors do not have to remember script
paths in `tools/` and `tests/benchmarks/`. Cross-reference from CLAUDE.md.

### What was done
- New `Makefile` at the repo root. Targets:
  - Build:  `rebuild`, `rebuild-clean`, `rebuild-cargo`, `check`, `clean`
  - Tests:  `test` / `test-quick` (alias), `test-verbose`, `test-full`
  - Bench:  `bench-regression`, `bench-single EXAMPLE=… ARGS=…`,
            `profile-bench FILTER=… SECS=… TOOL=…`
  - Lint:   `lint`, `lint-changed`, `fmt`, `fmt-check`
  - `make help` auto-generates the menu from `##` comments.
- `CLAUDE.md` gained a "Makefile Shortcuts" table near the top that
  forwards to the existing tools/scripts docs (those remain
  authoritative).

### Design choices
- Thin Makefile — every target shells out to the existing
  `tools/rebuild-rust.sh`, `tests/benchmarks/run_*.py`, or
  `profile_rust_bench.sh`. The Makefile is an index, not a build
  system.
- PY points at `.venv/bin/python` so `make` works without a manual
  `source .venv/bin/activate`.
- Python lint/format routes through `pre-commit` (the canonical tool
  per `.pre-commit-config.yaml`) rather than installing ruff
  separately. This matches CI exactly.
- `fmt-check` only checks cargo fmt (Python is covered by `make lint`).

### Validation
- `make help` — lists all targets cleanly.
- `make check` — clean (release cargo check, ~7s after warm cache).
- `make fmt-check` — clean (cargo fmt --check passes).
- `make rebuild-cargo` — succeeded (pip is broken in this venv, the
  documented fallback path works).
- `make test` — 389 passed, 3 pre-existing failures
  (`test_pipe_native_dispatch_creates_two_fds`,
   `test_dup2_native_dispatch_redirects_stdin`,
   `test_dcas_cmpxchg16b_no_match_keeps_memory`). Matches baseline.
- `make bench-single EXAMPLE=fauxware ARGS="--engine rust"` — ran.
- `make bench-regression` — ran 12 benches, 2 noise regressions on
  defcamp_r100 (unrelated to this change).

### Memories to save
- (none — this is plumbing; nothing surprising. The pre-existing
  `avoid-pip-install-broken-venv` memory already covers the pip
  failure mode that drove `make rebuild-cargo`.)

### Prior session
`angr-govb + angr-wvxj` (auto-memory housekeeping) closed at 226590a10.
