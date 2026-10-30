# Agent Notes

Project instructions for agents and contributors on the `rust-symex`
branch (`CLAUDE.md` imports this file). It holds workflow and pointers
only. How the engine *works* lives in
[`docs/advanced-topics/rust_engine.rst`](docs/advanced-topics/rust_engine.rst)
and the pages it lists; how a gate works lives in that gate's own
docstring; why a rejected alternative was rejected lives in
[`docs/advanced-topics/rust_engine_decisions.rst`](docs/advanced-topics/rust_engine_decisions.rst).
When a fact has a better home, link to it from here instead of copying it.

## Issue Tracking

This project uses **bd (beads)** for issue tracking.
Run `bd prime` for workflow context, or install hooks (`bd hooks install`) for auto-injection.

**Quick reference:**
- `bd ready` - Find unblocked work
- `bd create "Title" --type <type> --priority <P>` - Create issue
- `bd close <id>` - Complete work
- `bd dolt push` - Push beads to remote

**Filing issues:** type and priority follow [`TRIAGE.md`](TRIAGE.md)
(P1 = silently wrong and reachable; crashes and omissions are P2 or lower;
chores are never P1). Every bug ends with a `Why P<n>:` line, and a re-rating
is recorded with `bd note`, not a silent edit. Read it before `bd create` or
`bd update --priority`.

For full workflow details: `bd prime`. Active development happens on
`rust-symex`; PRs target `master`.

## Memory system

Project memory lives in **bd**, not in Claude's per-project auto-memory
markdown files. The sibling checkout `~/repos/angr-memories/` holds the
synced `memories.jsonl` archive and the pruning tooling (see its README).

- **Recall** a memory: `bd recall <key>` (keys are kebab-case).
- **Search**: `bd memories <keyword>` (no arg lists all).
- **Save**: `bd remember "<insight>" --key <kebab-key>` — always pass
  `--key` (`avoid-bd-remember-without-key-flag`); it updates in place.
- **Remove**: `bd forget <key>`.
- **Prime context**: `bd prime`.

This **overrides** the default auto-memory instruction to write
per-memory markdown files. Do not write those files for this project.

### Memory, comment and doc hygiene

1. **Rename-sweep.** A commit that renames or moves a file or public symbol
   runs `bd memories <old-name>` for every plausible old name and repairs the
   hits in the same commit (`refactor-memory-sweep-rule`,
   `bd-memory-citation-repair-pattern`). Keyword search has recall gaps, so
   widen with `bd recall <key> </dev/null` over candidate keys rather than
   grepping an argless dump, which truncates bodies.
2. **Symbol anchors, never line numbers.** Cite `fn`/`struct`/`method` names
   in memories, in-source comments and `docs/**/*.rst`.
   `tools/check_line_citations.py` gates new `file:NNN` citations against a
   baseline that should only ever shrink.
3. **No cardinality claims.** Enumerate members instead of counting them, so
   adding one leaves a visibly incomplete list rather than a quietly wrong
   number.
4. **Provenance is a bead id and nothing else.** A comment recording *why*
   cites `angr-xxxxx` (optionally `angr-xxxxx Phase N`). Never write
   `angr-xxxxx tracks <future work>` — the bead closes and the comment lies.
   To explain why a divergence *stays*, cite the invariant (a memory key or
   the sibling symbol's doc), not a bead.
5. **Pointers, not copies.** A comment, docstring or memory that would
   restate a design doc points at the section and keeps only the code-local
   invariant. Every topic has one home — see *Where context lives*.
6. **No transient prose.** No dates, counts, timings, commit hashes or
   "as of" in docs, comments or memories. Measurements live in bd beads and
   git history; a doc quotes the file or command that holds the live value.

## Static gates (Rust)

Every gate runs in `ci.yml::rust_check` (the `steady_guard_checked` row is
compile-time). Each script's docstring is the authority for what it
matches and why; the right-hand column is the canonical API or doc to
reach for. Exempt a genuine case with the marker, not `--update-baseline`;
the baselines marked *empty* must stay empty. CI runs every script with
`--self-test` ahead of the real check: a gate never seen to fail is
indistinguishable from one that cannot. The newer scripts
(`audit_silent_fallback_test_pairing.py`, `audit_rounding_mode_threading.py`,
`audit_sp_default_zero.py`, `audit_overflow_wraparound.py`,
`audit_page_shift.py`, `audit_test_import_guard.py`, `audit_core_pyo3_free.py`) import comment blanking, test-file filtering and
`fn`-name resolution from `tools/rust_source_utils.py` — add helpers there;
`check_line_citations.py`, `audit_silent_fallback.py` and
`audit_satresult.py` predate that module and keep their own scanners.

| Gate | Catches | Exempt with | Canonical API / doc |
|------|---------|-------------|---------------------|
| `tools/check_line_citations.py` | `file:NNN` / "line NNN" citations in Rust comments and `docs/**/*.rst` (baseline) | `--update-baseline`, only for an external line number that cannot drift, or to prune dead rows after a repair sweep (deletions-only diff) | rule 2 above |
| `tools/audit_silent_fallback.py` | `return Ok(None)` / `.ok();` discards (bare, or as a `let` tail) without a `// SILENT(cat-a\|b\|c): <why>` tag (baseline) | the tag | `silent_default!` doc in `native/angr/src/lib.rs`: `cat_b` logs debug, `cat_c` logs warn, there is no `cat_a` arm; `let _ = fallible` is denied by clippy (`let_underscore_must_use`), so spell expected discards `drop(expr)` + tag, or `#[expect(clippy::let_underscore_must_use, reason = "SILENT(cat-a): …")]` for `Copy` results. Wildcard `_ =>` arms are not auto-detected: at a trust boundary fail loud. |
| `tools/audit_silent_fallback_test_pairing.py` | a new `cat_b`/`cat_c` site in the diff with no regression test in the same diff (diff-based) | — | same macro doc |
| `tools/audit_satresult.py` | raw `z3::SatResult::{Sat,Unsat}` matches outside test code (`*_tests.rs`, `*_tests/`) | `// satresult-exempt: <why>` on the line or either of the two above | `SatOutcome::decided` in `symbolic/solver_build.rs` (`invariant-z3-unknown-not-unsat`): `Unknown` is never `Unsat` |
| `#[angr_macros::steady_guard_checked]` | any `&mut self` method, whatever its visibility, on a `#[pymethods] impl RustExplorationManager` block (`exploration/manager_methods*.rs`) lacking `#[steady_guarded]` or `#[steady_guard_exempt(reason = "…")]` → `compile_error!` | the `exempt` attribute with a reason | macro doc in `native/angr-macros/src/lib.rs`; `angr-macros-call-site-invariants` |
| `tools/audit_rounding_mode_threading.py` | a closure in a `fn *_rm` that ignores its rounding-mode parameter, or an rm operand bound to `_rm` and dropped (`vex/`, `interpreter/`; empty baseline) | `rm-ignored: <why>` on the line above, or as the site's trailing comment | `VEXOps::binop_with_rm` / `qop_with_rm`, `ops::float_arith::apply_rounding` |
| `tools/audit_sp_default_zero.py` | an SP / return-address accessor chained to `.unwrap_or(0)` / `.unwrap_or_default()` (empty baseline) | `sp-default-ok: <why>` on the line above, or as the site's trailing comment | `AddrOrSymbolic::or_log` in `arch/mod.rs` — the typed accessors have no `Default`/`From<u64>`, `.concrete()` is the only escape hatch |
| `tools/audit_overflow_wraparound.py` | a bare `+`/`-`/`+=`/`-=` on an address/size/offset/count-shaped operand in `native/angr/src/{arch,memory,interpreter,symbolic,state,syscalls,procedures,exploration}/` and `native/angr-bv/src/` (empty baseline) | `overflow-ok: <why>` on the flagged line or either of the two above (for a multi-line expression the flagged line is the continuation carrying the operator) | address arithmetic wraps, sizes and identities refuse: `invariant-rust-concrete-arith-must-wrap`, `invariant-proc-address-arith-wrapping`, `invariant-overflow-fix-refuse-not-saturate-identities`; `cargo test --profile release-checked` has overflow-checks on but is a runtime net only |
| `tools/audit_page_shift.py` | an open-coded page shift (`>> 12`, `PageIndex::SHIFT`) or a `0x1000`/`4096`/`0xFFF`/`4095` literal next to an address/page-shaped operand, all of `native/angr/src/` (empty baseline) | `page-shift-ok: <why>` on the flagged line or either of the two above | `Address::{page_num, page_base, page_offset}`, `PageIndex::{of, base_addr, range_covering}`, `PAGE_SIZE`/`PAGE_MASK` in `memory/page.rs` |
| `tools/audit_core_pyo3_free.py` | any PyO3 token in code (`pyo3`, `Python::initialize` / `attach`, `Py<…>`, `#[pyclass]`, …) in any file, test or prod, under `native/angr/src/{arch,memory,symbolic,vex,interpreter,state}/` and `native/angr-bv/src/` (no baseline; the set is empty) | only the upstream-owned files in the script's `ALLOWED` map, each with its reason | `interpreter::EngineHost` and its test double `interpreter::TestHost`; `state::HostAst`; the shell's `state_bridge`, `callbacks`, `claripy_bridge` |
| `tools/audit_test_import_guard.py` | a `py.import("angr.…")` in Rust test code: `cargo test` cannot import `angr`, so the guarded test returns early and passes (no baseline; the set is empty) | `import-guard-ok: <why>` on the line or the one above | `claripy_bridge::claripy_module` / `claripy_object` in `claripy_bridge/mod.rs` |

Preferred fix order for the "invariant upheld at some call sites but not
others" bug family, strongest first: make the unchecked value
unrepresentable (`BoundedArg`, `AddrOrSymbolic`), then a compile-time
macro over the whole impl/struct (`steady_guard_checked`, `MergePolicy`),
then a baseline-gated script. Why, in the "Which enforcement shape to
reach for" section of the `native/angr-macros/src/lib.rs` crate doc.

## Where context lives

Docs are authoritative for stable topics; bd memories carry the
per-decision detail. Look here first, then `bd recall`.

| Topic | Docs (authoritative) | bd memory (entry point) |
|-------|----------------------|-------------------------|
| Using the engine: options, inspect, techniques, errors | docs/advanced-topics/rust_engine.rst | core-goal-design-philosophy |
| Rejected alternatives | docs/advanced-topics/rust_engine_decisions.rst | — |
| Building, tests, bench gates, profiling, logging | docs/advanced-topics/rust_build.rst | venv-rebuild-cargo-direct-copy |
| Z3 sharing & solver counters | docs/advanced-topics/rust_z3_sharing.rst | constraint-export-no-pre-pin |
| Lazy memory design | docs/advanced-topics/rust_lazy_memory_design.rst | — |
| State proxy & exports | docs/advanced-topics/rust_proxy_writes_design.rst | rust-proxy-architecture-decision |
| Bimodal benchmark variance | docs/advanced-topics/rust_bimodal_variance.rst | benchmark-bimodal-variance-rules |
| Parallel / Send-Sync design | docs/advanced-topics/rust_parallel_design.rst | rust-python-boundary-audit |
| Native libVEX lift | docs/advanced-topics/rust_libvex_ffi.rst | — |
| Wheel distribution | docs/advanced-topics/rust_wheel_distribution.rst | avoid-pip-install-deps |
| VEX op contribution | docs/extending-angr/rust_vex_ops.rst | invariant-vex-op-no-concrete-zero-fallback |
| Native SimProcedures & syscalls | docs/extending-angr/simprocedures.rst, docs/extending-angr/native_coverage_matrix.rst | invariant-syscall-arg-extraction |
| Proc-macro invariants | native/angr-macros/src/lib.rs crate doc | angr-macros-call-site-invariants |
| Bead priority / type rubric | TRIAGE.md (repo root) | triage-rubric |

Live numbers never go in prose: `tests/benchmarks/baseline_timings.json`
for timings and speedups, `grep -rc 'def test_' tests/engines/rust/` for the
test count, `bd memories | head -1` for the memory count.

### Quick keys (durable entry points)

Mandatory-keep anchors that other memories and docs reference:

- `core-goal-design-philosophy` — Rust symex engine goal + design rules
- `rust-proxy-architecture-decision` — RustStateProxy vs SimState sync
- `rust-engine-agents-md-pointer` — what stays in memory vs AGENTS.md vs docs
- `rust-python-boundary-audit` — what bridge code must stay Python
- `constraint-export-no-pre-pin` — pre-pinning is dangerous; use Rust solver eval fallback
- `characterization-vs-fix-pattern` — separate characterization tasks from regression bisects
- `benchmark-bimodal-variance-rules` — what moves together when a bench joins `BIMODAL_BENCHMARKS` (the set and its pinned timing, `COUNT_EXEMPT`, the rst)
- `env-venv-corruption` / `venv-rebuild-cargo-direct-copy` — venv recovery and the cargo-direct rebuild
- `bd-update-notes-overwrites` — `bd update --notes` overwrites; append with `bd note`
- `avoid-bd-remember-without-key-flag` — `bd remember` without `--key` silently clobbers
- `9maq-bisect-method` — bisect pattern for memory-leak regressions (codified by `run_leak_check.py`)
- `avoid-pip-install-deps` — why the ecosystem pins (`archinfo`, `claripy`, `cle`, `pyvex`) stay put

## Building, testing, debugging

All of it is in [`docs/advanced-topics/rust_build.rst`](docs/advanced-topics/rust_build.rst):
prerequisites, bootstrap, `make` targets, the test and benchmark commands,
the CI gate table, profiling and Rust-side logging. `make help` lists every
target. Two local facts worth knowing up front:

- This checkout's `.venv/` `pip` is unreliable: rebuild with
  `make rebuild-cargo`.
- `make check` is not the CI gate; the rst's "Incremental builds" section
  has the exact commands.

## Key Files

Directory-level map. Each Rust directory's `mod.rs` carries a `//!` doc
that indexes its files; start there. `solver/` and `fuzzer/` use a sibling
module root instead (`solver.rs`, `fuzzer.rs`), and `fuzzer.rs` has no
`//!` index.

- **Build**: `pyproject.toml`, `setup.py`, `native/angr/Cargo.toml`,
  `native/angr/build.rs`, `rust-toolchain.toml`, `Makefile`,
  `tools/rebuild-rust.sh`.
- **Python side** (`angr/exploration/`): `rust_manager.py` (the
  `RustExplorationManager` wrapper and the SimOption ledgers; its module
  docstring holds the cross-mixin invariants), `rust_state_proxy.py`,
  `rust_state_binding.py` (copy-on-write proxy bindings),
  `rust_state_export.py`, `rust_state_sync.py`, `rust_callback_dispatch.py`,
  `rust_techniques.py`,
  `rust_perf_tracker.py`, `rust_state_cache.py`,
  `_constants.py`.
- **Rust engine** (`native/angr/src/`): `exploration/` (manager
  `#[pymethods]` split across `manager_methods*.rs`, the run loops,
  stepping, the work-stealing scheduler), `interpreter/` (VEX execution),
  `vex/` (IR, op semantics, lifting), `symbolic/` (`SymContext`, `ToZ3`,
  Z3 plumbing, and the `RustBV` re-export of `angr_bv::SymBV`), `solver/` (the `RustSolverContext` Python surface),
  `claripy_bridge/` (AST import/export and its caches), `memory/`,
  `state/`, `arch/`, `syscalls/`, `procedures/`, `callbacks/`,
  `fuzzer/`, `automaton/`.
- **Value layer** (`native/angr-bv/`): `SymBV` (the engine's `RustBV`) and
  its ops, with no solver dependency; `src/lib.rs` indexes its files.
- **Compiler-enforced invariants**: `native/angr-macros/` (`steady_guarded`
  / `steady_guard_checked`, `MergePolicy`, `callback_setters!`,
  `inspect_test_entries!`, `inspect_dispatch!`); the crate doc explains
  each.
- **Gates and tooling**: `tools/audit_*.py`, `tools/check_line_citations.py`,
  `tools/rust_source_utils.py`, `tools/ralph-gate.sh`; `tools/upstream_extract.py`
  scrubs fork-process references from upstream PR branches.
- **Tests and benches**: `tests/engines/rust/`, `tests/benchmarks/`
  (`run_single.py`, `run_regression.py`, `bench_diff.py`, the leak and
  find-all gates, `parity_census.py`), `native/angr/benches/`.
- **Docs**: `docs/advanced-topics/rust_*.rst`, `docs/extending-angr/`.

## Autonomous loop (`ralph`)

The optimization loop runs through [`ralph`](https://github.com/jcrussell/ralph)
(local checkout `~/repos/ralph`, symlinked to `~/.local/bin/ralph`).
Everything it needs is under `.ralph/`:

- `config.toml` — the overrides over ralph's built-in defaults (memory
  cap, gate timeout, the `tools/ralph-claude.sh` runner wrapper, model, and
  `master` as the review base branch); the file documents each.
- `prompts/{_header,_footer,clean,dirty,revert,review}.md` — per-state
  agent prompts; `_footer.md` carries the workflow steps and the MEMORY
  SAFETY rules.
- `hooks/states/{clean,dirty}/gate` — thin wrappers around
  `tools/ralph-gate.sh`, whose header is the authority on the checks it
  runs, in order: `cargo test --profile release-checked` and the fast-tier benchmark regression with the
  PR-gate flags. `revert/gate` and `review/gate` are no-op stubs that
  exit 0.
- `state/` — runtime state (FSM, logs, `session.md` handoff), gitignored.

```bash
ralph doctor                      # verify prereqs
ralph run --once --dry-run        # route + render prompt, skip the runner
ralph run --once                  # one real iteration
ralph run                         # autonomous session
ralph status                      # FSM state, iteration count, cost
ralph trace <iter>                # drill into one iteration
ralph hook run states/clean/gate  # run the gate by hand
```
