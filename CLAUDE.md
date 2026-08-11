# Claude Code Notes

## Memory system

Project memory lives in **bd**, not in Claude's per-project auto-memory
markdown files. Run `bd memories | head -1` for the current count
(1000 as of 2026-07-20, after the round-4 prune). Audit/cleanup tooling
and pre-prune snapshots live at `~/angr-memories/` — see its README if
you ever need to recover or re-run a prune.

- **Recall** a specific memory: `bd recall <key>` (keys are kebab-case).
- **Search** memories: `bd memories <keyword>` (no arg lists all).
- **Save** a new memory: `bd remember "<insight>" --key <kebab-key>` — updates
  in place if `--key` already exists.
- **Remove**: `bd forget <key>`.
- **Prime context**: `bd prime` injects bd's workflow context (~80 lines) and
  is auto-called by `bd hooks install` at session start where installed.

This **overrides** the default Claude auto-memory instructions to
write per-memory markdown files. Do not write those files for this
project; they fragment across accounts and drift from bd.

### Keeping memories from rotting (refactor-time rules)

Memories cite code; renames silently break those citations. Two standing
rules (full rationale in bd memory `refactor-memory-sweep-rule`):

- **Rename-sweep:** any commit that renames or moves a file or public
  symbol must `bd memories <old-name>` and repair the hits in the **same
  commit**. Keyword search has recall gaps, so run it for *every* plausible
  old name — and do **not** fall back to grepping an argless `bd memories`
  dump, which prints truncated one-line summaries and misses tokens buried
  mid-body. Widen instead with `bd recall <key> </dev/null` over the
  candidate keys. Repair per `bd-memory-citation-repair-pattern`.
- **Symbol-anchor convention:** when authoring a memory, anchor code
  references to symbol names (`fn`/`struct`/`method`) rather than raw line
  numbers. Audits find line refs drift 10–600 lines across refactors while
  symbol anchors stay resolvable.

The same convention applies to in-source Rust comments (`///`, `//!`, `//`):
cite the function/struct/const name a comment is cross-referencing, not a raw
`file.rs:NNN` or "line ~NNN". The angr-9ke6b full-review audit found a whole
class of these gone stale (`.12`, `.21`, `.35`, `.54`, `.122`, `.181`, and
more). `tools/check_line_citations.py` gates new occurrences in CI
(`rust_check`) against a checked-in baseline (`tools/line_citations_baseline.txt`)
of the ones that already exist — it does not require fixing those
retroactively, only stops the class from growing. Regenerate the baseline
with `--update-baseline` after intentionally adding one (rare — e.g. citing
an external, non-repo line number that can't drift). Since angr-kw5f6 the
same check also scans **every line of `docs/**/*.rst`** — prose rots the same
way comments do (angr-sqfj8.126 found `rust_engine.rst`'s typed-exception
table 60-95 lines off with no gate signal), so the symbol-anchor convention
applies when writing docs too.

**Provenance-tag convention:** a comment that records *why* a change was made
cites the bd bead id (`angr-xxxxx`, optionally `angr-xxxxx Phase N` for a
multi-phase bead) and nothing else. The pre-bd review-round labels (`P5 fix`,
`D2 Fix`, bare `Phase 2 Fix`) that came in with the v2 port resolved to
nothing a reader could look up and were stripped in angr-9ke6b.46 — do not
reintroduce a second numbering scheme. A bead id is provenance for what *was*
done, so never write `angr-xxxxx tracks <future work>` — the bead closes,
usually having done something else, and the comment silently becomes a lie a
reader will trust (angr-c7xno.35 found two of these pointing at angr-ph300.73).
When the comment's job is to explain why the duplication/divergence *stays*,
cite the invariant — the bd memory key or the sibling symbol's doc — not a
bead.

### Silent-fallback tagging (Rust)

Sites in `native/angr/src/` that discard an error/absent value and continue
with a degraded result (the `.100`/`.172`/`.144`/`.194`-class "silently wrong
instead of loud error" bugs from the angr-9ke6b audit) must carry a
`// SILENT(cat-a|b|c): <rationale>` comment above them — `cat-a` = expected
control flow, `cat-b` = fallback with loss, `cat-c` = wrong-answer risk (must
also `log::warn!`).

For the two *logged* categories, reach for the `silent_default!` macro
(defined in `native/angr/src/lib.rs`, textual scope so it needs no import)
rather than hand-writing the `unwrap_or_else` + `log::warn!` boilerplate — the
whole reason the silent form kept winning is that it was the shorter one
(angr-91vj9.6). The category argument picks the level (`cat_b` → `log::debug!`,
`cat_c` → `log::warn!`) and *is* the tag, so no separate comment is needed;
`audit_silent_fallback.py` accepts an invocation in place of one. There is
deliberately no `cat_a` arm — expected control flow needs no log, so it stays a
plain `match`/`unwrap_or` with the comment. Both fallible shapes are supported:
`silent_default!(cat_c, opt_expr, default, "msg {ctx}")` for `Option`, and
`silent_default!(cat_b, res_expr, default, |err| "msg {err}")` for `Result`
(the binder must be named by the caller — macro hygiene would hide a
macro-defined one from the message). `$default` sits in tail position, so
`return`/`continue` are legal defaults. Canonical call sites:
`VEXInterpreter::get_stack_pointer_or_log`, both `get_return_addr_or_log`s,
`vex::opcode_map::parse_type_or_log`, and the `Result` form in
`exploration/state_api.rs::push_assumed_constraint_or_log`.

`tools/audit_silent_fallback.py` gates the currently-narrow
`return Ok(None)` / `.ok();` shapes in CI (`rust_check`) against
`tools/silent_fallback_baseline.txt`, same baseline-audit pattern as the
line-citation check above. The `let _ = <fallible expr>;` shape needs no
script — since angr-91vj9.11 the workspace `[lints.clippy]` block in the
repo-root `Cargo.toml` carries `let_underscore_must_use = "deny"`, so clippy
rejects it outright. A discard that really is expected control flow spells
itself `drop(expr)` with a `// SILENT(cat-a):` rationale comment; a lossy one
uses `silent_default!`. When the discarded value is `Copy`
(`compare_exchange`'s `Result<u64, u64>`), `drop` is itself a no-op clippy
rejects, so those keep `let _ =` under an
`#[expect(clippy::let_underscore_must_use, reason = "SILENT(cat-a): ...")]` —
the attribute's `reason` is the tag. The noisier `.unwrap_or*`/wildcard
`_ =>` match-arm shapes are deliberately NOT auto-detected (too many
legitimate uses, e.g. matching a handful of variants out of a huge
externally-defined enum like `IROp`) — for those, apply the same judgment in
code review instead: a "shouldn't happen" case at a trust boundary
(interpreter dispatch, syscall arg decoding) should fail loud
(`unreachable!()`, `Err(...)`, or a logged warning), not silently return a
default value.

### Steady-guard coverage (Rust)

`#[angr_macros::steady_guarded]` is opt-in per function, so a *new*
`RustExplorationManager` config mutator silently inherits the bug the macro
exists to prevent (round-3 audit caught `set_max_history` this way —
angr-c7xno.21). `tools/audit_steady_guard_coverage.py` gates
`native/angr/src/exploration/manager_methods*.rs` in CI (`rust_check`) against
`tools/steady_guard_baseline.txt`, same baseline-audit pattern as the two
checks above. Eligible = `pub fn` + `&mut self` + a mutator name prefix
(`set_`/`clear_`/`register_`/`add_`/… — see `MUTATOR_PREFIXES`). Exempt =
carries the attribute, **or** documents the exception with a doc line matching
`NOT ... steady_guarded` (the convention `load_snapshot_bytes` and
`set_parallel_frontier_residency` already use, for the cases where the guard
cannot be the unconditional first statement the macro injects). Fixing a
baselined gap means deleting its line, not just adding the attribute.

### VEX rounding-mode threading (Rust)

The `VEXOps::*_rm` helpers hand a concrete closure to a shared `*_rm`
dispatcher (`float_to_int_rm`, `float_to_float_rm`) whose whole purpose is to
thread the VEX rounding mode into the concrete path so it agrees with the
symbolic Z3 FP path. A closure that drops the mode compiles cleanly — naming
the parameter `_rm` is Rust's *sanctioned* spelling of "unused" — so
`f64_to_f32_rm` silently computed round-to-nearest-even for RZ/RU/RD while
every sibling routed through `apply_rounding_f32`/`apply_rounding_f64`
(angr-c7xno.85). `tools/audit_rounding_mode_threading.py` gates that shape in
CI (`rust_check`) against `tools/rounding_mode_baseline.txt` — same
baseline-audit pattern as the three checks above, but the baseline is **empty**
and should stay so. Two detectors, over both `native/angr/src/vex/` and
`native/angr/src/interpreter/`: (1) a closure inside a `fn *_rm` whose last
parameter is `_`-prefixed, or is named `rm` and never referenced in the body;
(2) a `let` binding whose name is `_`-prefixed and ends in `rm` — an rm operand
evaluated and then discarded. Detector 2 exists because `eval_qop` bound the
Qop's rm to `_rm` and called the rm-less `VEXOps::qop`, so FMAdd/FMSub always
computed RNE, outside detector 1's `vex/`-only, closure-only reach
(angr-03vl4.30); the fix is `VEXOps::qop_with_rm`, the Qop mirror of
`binop_with_rm`. Exempt by
documenting the reason with an `rm-ignored: <why>` comment on the line above
(for the genuine cases — e.g. a libm-backed transcendental where Rust only
offers RNE); prefer that over `--update-baseline`. `--self-test` classifies a
synthetic threaded/dropped/exempt trio and runs ahead of the real check in CI —
same reasoning as the valgrind gate's self-test.

### SP / return-address default-to-zero (Rust)

Reading the stack pointer or the return address can fail (symbolic register,
unreadable `[sp]` slot), and collapsing that failure to the literal `0` is a
wrong-answer risk: the value is exported verbatim (e.g. as
`CallStackEntry.stack_ptr`) and downstream code cannot tell it from a genuine
SP of 0. Three logged `SILENT(cat-c)` helpers exist for it —
`VEXInterpreter::get_stack_pointer_or_log`, `VEXInterpreter::get_return_addr_or_log`,
`RustExplorationManager::get_return_addr_or_log` — plus
`exploration::helpers::advance_sp_past_return_addr` for the native-proc return
path, which bumps a symbolic SP *symbolically* instead of concretizing it. The
helpers were still bypassed three separate times (angr-sqfj8.62, angr-sqfj8.63,
angr-c7xno.29), because the raw accessor and the helper are equally easy to
reach for. `tools/audit_sp_default_zero.py` gates the shape in CI
(`rust_check`) against `tools/sp_default_baseline.txt` — same baseline-audit
pattern as the checks above, and like the rounding-mode one the baseline is
**empty** and should stay so. Flags an SP/return-address accessor
(`get_sp`/`get_sp_value`/`get_stack_pointer`/`get_return_addr`, plus
`get_offset_u64` when its args are SP-ish) followed by a method chain ending in
`.unwrap_or(0)`/`.unwrap_or_default()`. Exempt by documenting the reason with
an `sp-default-ok: <why>` comment on the line above; prefer that over
`--update-baseline`. `--self-test` runs ahead of the real check in CI, same
reasoning as the rounding-mode gate.

The four audit scripts share comment/string blanking, test-file filtering and
`fn`-name resolution via `tools/rust_source_utils.py` — put new helpers there
rather than copying a fifth blanker.

### Where context lives

When you need background that isn't in CLAUDE.md, look here first, then
fall back to `bd recall`. The docs are authoritative for stable topics;
bd memories cover the per-decision detail.

| Topic                          | Docs (authoritative)                                       | bd memory key (entry point)         |
|--------------------------------|------------------------------------------------------------|-------------------------------------|
| Engine architecture & usage    | docs/advanced-topics/rust_engine.rst                       | core-goal-design-philosophy         |
| Z3 sharing & solver counters   | docs/advanced-topics/rust_z3_sharing.rst                   | constraint-export-no-pre-pin        |
| Lazy memory design             | docs/advanced-topics/rust_lazy_memory_design.rst           | lazy-memory-sidecar-architecture    |
| State proxy & exports          | docs/advanced-topics/rust_proxy_writes_design.rst          | rust-proxy-architecture-decision    |
| Bimodal benchmark variance     | docs/advanced-topics/rust_bimodal_variance.rst             | benchmark-bimodal-variance-rules    |
| Parallel/Send-Sync design      | docs/advanced-topics/rust_parallel_design.rst              | rust-python-boundary-audit          |
| VEX op contribution            | docs/extending-angr/rust_vex_ops.rst                       | invariant-syscall-arg-extraction    |
| Native SimProcedures           | docs/extending-angr/simprocedures.rst                      | —                                   |

### Quick keys (durable entry points)

These memories are mandatory-keep — anchors that other memories and
docs reference:

- `core-goal-design-philosophy` — Rust symex engine goal + design rules
- `rust-proxy-architecture-decision` — RustStateProxy vs SimState sync
- `rust-engine-claude-md-pointer` — what stays in memory vs CLAUDE.md vs docs
- `rust-python-boundary-audit` — what bridge code must stay Python
- `constraint-export-no-pre-pin` — pre-pinning is dangerous; use Rust solver eval fallback
- `characterization-vs-fix-pattern` — separate char tasks from regression bisects
- `benchmark-bimodal-variance-rules` — handling the bimodal-Z3 benches (5; see `BIMODAL_BENCHMARKS`)
- `env-venv-corruption` — venv recovery from corruption
- `bd-update-notes-overwrites` — `bd update --notes` overwrites; never append-via-update
- `avoid-bd-remember-without-key-flag` — `bd remember <text>` without `--key` silently clobbers
- `9maq-bisect-method` — bisect pattern for memory-leak regressions
- `venv-rebuild-cargo-direct-copy` — cargo-direct rebuild when venv pip is broken
- `avoid-pip-install-deps` — never bump claripy past 9.2.209 — z3-solver SONAME risk

## Makefile Shortcuts

Common dev workflows are wired up in the repo-root `Makefile`. Run `make help`
for the full list. Most-used targets:

| Target | What it does |
|--------|--------------|
| `make rebuild` | Incremental Rust .so rebuild via pip editable install. |
| `make rebuild-cargo` | Cargo-direct rebuild (broken-venv fallback). |
| `make rebuild-fast` | Inner-loop rebuild via `[profile.release-fast]` (~3s warm vs ~36s). NOT for bench gates. |
| `make check` | `cargo check --release` — fast type/borrow check. Default features only, so it does **not** cover `fuzzer`/`fuzzing`/`libvex-ffi` code; it is not the CI gate. Reproduce that with `cargo clippy --manifest-path native/angr/Cargo.toml --all-targets --all-features -- -D warnings`. |
| `make check-no-z3` | `cargo check -D warnings --all-targets` over the four `--no-default-features` combos CI's `rust_feature_flags` gates. Neither `make check` nor clippy `--all-features` compiles the `#[cfg(not(feature = "vex-engine-z3"))]` arms — this is the only local way to type-check them, and (since angr-sqfj8.139) the only gate holding them to the zero-warning bar. `--all-targets` since angr-c7xno.99: without it only the *lib* is compiled, and the no-z3 test/bench targets had rotted to 41 errors unseen. |
| `make test` (`test-quick`) | Run `tests/engines/rust/` (~1-2 min). |
| `make test-libvex` | libVEX-FFI unit + corpus parity tests (`--features libvex-ffi`, default-off in cargo). |
| `make test-fuzzer` | icicle/libafl fuzzer unit tests (`--features fuzzer`, default-off in cargo; nightly lane `fuzzer_feature_tests`). |
| `make test-python-baseline` | Fast (~5min) vanilla Python-engine regression subset (needs `../binaries`). |
| `make test-full` | Full angr test suite (long). |
| `make gate-findall` | Find-all worker-count-invariance gate (nightly lane `findall_worker_invariance`). |
| `make bench-regression` | Fast-tier benchmark regression check. |
| `make bench-single EXAMPLE=fauxware ARGS="--both"` | Run one bench in a subprocess. |
| `make profile-bench FILTER=... SECS=... TOOL=...` | Wrap the criterion bench with a profiler. |
| `make lint` | `pre-commit run --all-files` (ruff, pyupgrade, formatters). |
| `make fmt` | Apply pre-commit autofixes + `cargo fmt`. |

The Makefile is a thin shell over `tools/rebuild-rust.sh`,
`tests/benchmarks/run_*.py`, and `profile_rust_bench.sh` — the sections below
remain the authoritative docs for those scripts.

## Building the Rust Extension

This project uses **setuptools-rust** (not maturin) to build the Rust native extension.

### Prerequisites
- Python 3.10+ (3.12 verified)
- Rust toolchain (rustup) at `~/.cargo/bin/` — pinned via `rust-toolchain.toml`
- Z3 development headers — `apt install libz3-dev` on Debian/Ubuntu, `dnf install z3-devel` on Fedora, `brew install z3` on macOS. The PyPI `z3-solver` wheel ships `libz3.so` but **not** the C headers; `z3-sys` needs the headers to generate bindings.
- `pkg-config` (used by `z3-sys` to locate the system Z3)
- `libclang` — `bindgen` (used by both `z3-sys` and, since libVEX FFI is default-ON per angr-3trr7, the `libvex-ffi` feature) generates bindings via libclang. `apt install libclang-dev` / `dnf install clang-devel` / `brew install llvm`. Set `LIBCLANG_PATH` if it lives outside the default search path. Building with `ANGR_LIBVEX_FFI=0` still needs it for `z3-sys`.
- A `.venv/` virtualenv populated with the angr ecosystem deps (`claripy`, `pyvex`, `archinfo`, `cle`, etc.)

### Bootstrap from a clean state

If `.venv/` is missing, corrupted, or wiped (it has happened — see `tools/restore-venv.sh` and the `env-venv-corruption` bd memory), the recovery sequence is:

```bash
# Option A: restore from a recent tarball backup (fastest if available)
#   Look in ~/repos/angr.tar.gz or similar; extract just the .venv subtree:
#     cd /home/ubuntu/repos && tar -xzf angr.tar.gz angr/.venv
#   Verify with: .venv/bin/python -c "import angr; print(angr.__file__)"

# Option B: fresh install from scratch
sudo apt install libz3-dev pkg-config rustup        # or distro equivalent
rustup show                                          # picks up rust-toolchain.toml
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools setuptools-rust
.venv/bin/pip install -e .                           # pulls deps into venv
.venv/bin/pip install -e . --no-build-isolation --no-deps   # rebuild Rust .so

# Verify
.venv/bin/python -c "import angr; print(angr.__file__)"
.venv/bin/python -m pytest tests/engines/rust/ --tb=short -q
```

**Note:** `pyproject.toml` pins the four angr-ecosystem deps (`archinfo`, `claripy`, `cle`, `pyvex`) to `==9.2.209`. Do not bump these pins yet. Audit (bd bead `angr-3mkf`, 2026-06-05): claripy 9.2.221 (latest on PyPI) still pins `z3-solver==4.13.0.0`, same SONAME as 9.2.209, so the shared-Z3-context concern is dormant. Two active blockers remain: (1) upstream master pins `9.2.222.dev0`, which is not on PyPI and would require source builds of all four ecosystem repos; (2) our `.venv/` is hand-assembled via cargo-direct-copy (`pip list` only shows pip / platformdirs / rust-demangler), so any pin change requires a venv rebuild rather than `pip install -U`. Re-evaluate when upstream cuts a stable 9.2.222+ release on PyPI. See bd memory `avoid-pip-install-deps`.

### Build Commands (incremental)

```bash
# Ensure Rust is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Activate virtualenv
source .venv/bin/activate

# Rebuild Rust .so (editable mode, after changes in native/angr/src/)
pip install -e . --no-build-isolation --no-deps
```

`native/angr/build.rs` auto-detects the Z3 shared library from the active venv's `z3-solver` install and sets the runpath so Python and Rust load the same `libz3.so` (required for AST passthrough). For headers, `setup.py::_resolve_z3_header` probes the venv `z3/include`, then `pkg-config --variable=includedir z3`, then `/usr/include`, `/usr/local/include`, `/opt/homebrew/include`, `/opt/local/include`, and sets `Z3_SYS_Z3_HEADER` before cargo runs. If nothing matches it prints an install hint (apt/dnf/brew) to stderr and lets z3-sys fall back to its own pkg-config probe.

### Common Issues

**"can't find Rust compiler"**: Add `~/.cargo/bin` to PATH before running pip install.

**Stale .so file**: Run `tools/rebuild-rust.sh`. The script wipes
`angr/rustylib*.so` and `build/`, runs `cargo clean`, then re-runs the editable
pip install. Pass `--keep-cargo-cache` to skip `cargo clean` (faster), or
`--cargo-only` to skip pip and build via `cargo build --release` + copy
(recovery path when the venv's pip/setuptools is corrupt — see
`venv-rebuild-cargo-direct-copy` memory).

**"Unable to generate bindings: NotExist ...z3.h"**: `setup.py::_resolve_z3_header` couldn't find `z3.h` in any standard location. Install the Z3 dev package: `apt install libz3-dev pkg-config` / `dnf install z3-devel pkgconf-pkg-config` / `brew install z3 pkg-config`. To override discovery, set `Z3_SYS_Z3_HEADER=/path/to/z3.h` before invoking `pip install` or `cargo`. When using `tools/rebuild-rust.sh --cargo-only` (setup.py bypassed), the shell script also probes the venv path then `/usr/include/z3.h` as a fallback.

**Linker can't find libz3** at runtime: build.rs sets the runpath to the venv's `z3/lib`, so an `import angr` from inside the venv works. Outside the venv, set `LD_LIBRARY_PATH` to that directory or override with `Z3_LIBRARY_PATH_OVERRIDE=/path/to/lib`.

## Running Tests

```bash
# Run RustExplorationManager tests (see Current Status below for live count)
python -m pytest tests/engines/rust/ -v --tb=short

# Run benchmark regression tests (7 fast-tier benchmarks)
python tests/benchmarks/run_regression.py

# Skip the five bimodal-Z3 benches (see BIMODAL_BENCHMARKS in run_regression.py:
# unbreakable_1 / fairlight / sokohashv2 / angry-reverser / CADET_00001_partial)
# — matches the PR-time CI gate, useful when chasing a regression locally.
python tests/benchmarks/run_regression.py --rust-only --skip-bimodal --threshold 0.15

# Run single benchmark (safe, subprocess with 4GB memory limit)
python tests/benchmarks/run_single.py fauxware --both

# Dump every counter from mgr.stats() grouped by category — useful for
# per-bench attribution without a flamegraph (Rust engine only).
python tests/benchmarks/run_single.py fauxware --dump-counters

# Machine-readable variant: emits the full stats dict as JSON between an
# `OK rust ...` header line and end-of-file. Suppresses the curated stats
# sections so the JSON is the only structured payload on stdout.
python tests/benchmarks/run_single.py fauxware --counters-json

# Diff two --counters-json captures by counter delta (largest |delta| first,
# filters noise below 5% AND below 10 abs). load_counters strips the
# `OK rust ...` preamble so the raw run_single output can be fed in directly.
python tests/benchmarks/run_single.py fauxware --counters-json > /tmp/base.json
# ... apply candidate change, rebuild ...
python tests/benchmarks/run_single.py fauxware --counters-json > /tmp/curr.json
python tests/benchmarks/bench_diff.py /tmp/base.json /tmp/curr.json

# Refresh baseline_counters.json (used by run_regression.py to auto-emit the
# diff on a timing regression) without touching baseline_timings.json. Use
# --full so MEDIUM_SUITE benches (mma_howtouse, csaw_wyvern, flareon2015_5/10,
# ekopartyctf2016_rev250, codegate_2017-angrybird, sym-write) get a snapshot
# too — nightly-ci.yml runs --full, so without these the diff is silently
# skipped on a nightly medium-tier regression (angr-amoi). Drop --full for a
# fast-tier-only refresh.
python tests/benchmarks/run_regression.py --full --rust-only --skip-bimodal --update-counters

# Point at a custom angr-examples checkout (defaults to ~/repos/angr-examples)
ANGR_EXAMPLES_DIR=/path/to/angr-examples/examples python tests/benchmarks/run_regression.py
```

`--update` does **not** overwrite a baseline listed in
`run_regression.py::PINNED_RUST_TIMES` — those are deliberately pessimistic
(slow-mode / load-sensitive) pins that a single refresh run would otherwise
demote to whichever mode it happened to measure, which is how
`cow_fork_scaling`'s documented 2.7 pin became 2.5 and reddened the ralph gate
for a dozen iterations (angr-x6t9o). A held-back pin prints a
`pinned baseline kept:` line naming the measurement it declined to write.
Change a pin by editing that table, so the new value lands in a reviewable diff.

### Benchmark gates in CI

- **`.github/workflows/ci.yml::benchmark_regression`** (PR-time): runs the
  fast-tier suite with `--rust-only --skip-bimodal --threshold 0.15`, capped at
  5-minute timeout. Fails a PR when any non-bimodal fast-tier bench is more than
  15% slower than `baseline_timings.json` **and** more than `--regression-floor`
  (default 50ms) slower in absolute terms — at a 0.16s baseline 15% is ~24ms,
  inside the host noise floor, which failed the ralph gate twice on
  `sharif7_rev50` alone (angr-z8p3x). Benches above ~0.33s are unaffected by the
  floor; a suppressed excursion still prints a `noise-floor:` line. A second,
  independent bar covers the case the floor cannot (angr-mc8pw): before any
  retry, timing failures are re-checked against a baseline scaled by the
  **suite median** `current/baseline` ratio, so a uniformly slow host no longer
  reds the gate on whichever bench it hit hardest — iter24 failed on
  `cow_fork_scaling` at +24% while the whole 22-bench suite ran 1.083x slow and
  took 47.4s against a normal 31-35s. Normalization is inert inside a 1.02x
  dead-band (quiet hosts decide exactly as before) and refuses to engage past
  `--host-factor-cap` (default 1.25), where a uniform shift is equally
  consistent with a global regression; that case prints `SUITE-WIDE SLOWDOWN`
  and keeps the failures. `--no-host-normalize` restores the raw comparison.
  Warns (but does not fail) when the
  Rust-vs-Python speedup, computed against cached `python_time`, slips into the
  0.5x–1.0x band.
- **`.github/workflows/nightly-ci.yml::benchmark_regression`** (nightly): runs
  the same suite without `--skip-bimodal` (bimodal benches still tracked for
  drift, just expected to vary).
- Both jobs checkout `angr/angr-examples` and set `ANGR_EXAMPLES_DIR` so
  `tests/benchmarks/run_single.py` finds the example corpus. Without that the
  script falls back to `~/repos/angr-examples/examples`.
- On a timing regression, the gate auto-emits a `bench_diff` table comparing
  the per-bench counter dict against `baseline_counters.json` (soft dep —
  missing file just suppresses the diff). The table caps at 20 rows by
  default and shows the counters that moved most in absolute terms; ns-level
  timing counters always lead because they vary slightly across runs.

### Nightly find-all worker-count-invariance gate

`.github/workflows/nightly-ci.yml::findall_worker_invariance` runs
`tests/benchmarks/run_findall_gate.py` — the parallel scheduler's correctness
contract: an exhaustive `explore(find=..., num_find=all-leaves)` must produce
the **same found-set** regardless of `RUST_PARALLEL_WORKERS`.

The gate is **structural, not wall-clock** (wall-clock acceptance lives on the
spike bead): each run exports the multiset of found pcs via
`ANGR_BENCH_FOUND_FINGERPRINT=1`, and the fingerprint must be identical across
every worker count and rep. An empty found-set fails rather than passing
vacuously. Default corpus is the two in-repo synthetic find-all benches
(`fork_solve_pbounce_W6_S8_M12_B2`, `fork_solve_trap_W5_S8_M12`) swept over
workers {1,2,4} x 3 reps — 18 `run_single.py` children, ~12 min.

```bash
make gate-findall                                        # full default sweep
python tests/benchmarks/run_findall_gate.py --only fork_solve_pbounce_W6_S8_M12_B2 --workers 1 2 --reps 1
python tests/benchmarks/run_findall_gate.py --json findall.json
```

### Nightly RSS leak check

`.github/workflows/nightly-ci.yml::rss_leak_check` runs
`tests/benchmarks/run_leak_check.py` against a Callable-heavy bench
(default `mma_howtouse`, 45 invocations per `main()`) **N=10 times** in a
single process under a 4 GB `RLIMIT_AS`. Tracks `ru_maxrss` after each
`main()` call and fails when
`peak_rss(iterN) / peak_rss(iter1) > --threshold` (default **1.5x**).

Codifies the `9maq-bisect-method` bd memory into a gate — protects
against future Callable / cache regressions.

```bash
# Local invocation (~50s for N=10, ~15s for N=3)
python tests/benchmarks/run_leak_check.py --iters 10
python tests/benchmarks/run_leak_check.py --iters 3 --threshold 1.3
python tests/benchmarks/run_leak_check.py --example flareon2015_10 --json
```

The harness installs the same engine-swap monkeypatch `run_single.py` uses
(shared in `tests/benchmarks/engine_patch.py`), so the bench's Callables run
on the **Rust** engine; it fails if zero `RustExplorationManager`s were built.
`--engine python` measures the Python engine for comparison.

**Threshold tuning notes:**
- Healthy ratio is ~1.00-1.05x (Z3+claripy caches warm up but don't grow
  unboundedly). Measured on Rust, mma_howtouse N=10: **1.054** (285 -> 301 MB).
- 1.5x is the recommended default: large enough to avoid false-positives
  from cache warmup, small enough to catch a ~2x regression on the first
  nightly after it lands.
- If a future legitimate change pushes healthy ratio above ~1.3x (e.g. a
  larger steady-state cache), raise the threshold incrementally — do not
  lower below 1.2x without first ruling out warmup variance.
- For benches outside `mma_howtouse`, expect different baselines; rerun
  N=10 once locally before picking a threshold.

### Nightly valgrind leak check

`.github/workflows/nightly-ci.yml::valgrind_leak_check` runs
`tests/benchmarks/run_valgrind_leak_check.py`, a memcheck gate on the **Rust
core** (angr-c4xcs.4). It complements the RSS gate above: RSS cannot see a
steady small leak the allocator satisfies from arenas it already owns.

It drives `native/angr/examples/leak_probe.rs` — a **pure-Rust** binary (no
Python in the process) that churns SymContext + z3 solver, RustBV,
SymbolicMemory and RustSimState fork, dropping everything it allocates. Running
memcheck through CPython instead would bury real findings under interpreter
false-positives; the Python-driven Callable path stays covered by the RSS gate.

The gate is a per-iteration **slope**, not an absolute byte count — the probe
frees everything, so a correct engine leaks a *constant* baseline (one-time z3
globals, lazy statics) that varies by distro. The harness runs the probe at N
and 10N and fails on growth of `definitely lost + indirectly lost`.
`possibly lost` / `still reachable` are reported but not gated.

```bash
python tests/benchmarks/run_valgrind_leak_check.py              # gate (needs valgrind)
python tests/benchmarks/run_valgrind_leak_check.py --json
python tests/benchmarks/run_valgrind_leak_check.py --self-test  # prove it can fail
```

Baseline (2026-07-14, valgrind 3.22): **0 B** definite+indirect at both N=20
and N=200 → **0.00 B/iter**; threshold 1.0 B/iter. `--self-test` injects a
deliberate 64 B/iter leak and asserts the gate trips (it measures exactly
64.00 B/iter). Keep that self-test running *before* the real check in CI — a
leak gate that has never been seen to fail is indistinguishable from one that
cannot, and this one initially could not (release LLVM elided the injector).

## Profiling Rust Benches

`tests/benchmarks/profile_rust_bench.sh` wraps the criterion bench in
`native/angr/benches/vex_engine.rs` with whichever sampling profiler is
available. It auto-detects the best tool (cargo-flamegraph > perf > callgrind),
builds the bench in `--release` with frame pointers and full debuginfo
(overriding the workspace `strip="symbols"`), and writes results under
`target/profile/<filter-or-all>/` (gitignored).

```bash
# Profile everything for 10s with the best tool available
tests/benchmarks/profile_rust_bench.sh

# Scope to a single bench group (any prefix works), pick the tool
tests/benchmarks/profile_rust_bench.sh --filter symcontext_fork --secs 30
tests/benchmarks/profile_rust_bench.sh --tool callgrind --filter rustbv_

# List bench groups (rustbv_*, symcontext_*, memory_*, state_fork)
tests/benchmarks/profile_rust_bench.sh --list
```

Tool prerequisites:
- **cargo-flamegraph** (preferred): `cargo install flamegraph`. Produces
  `flamegraph.svg` directly.
- **perf**: needs `kernel.perf_event_paranoid <= 2`
  (`sudo sysctl -w kernel.perf_event_paranoid=2`). Writes `perf.data` plus a
  `perf.txt` top-symbols report. To turn `perf.script` into an SVG:
  `cat perf.script | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg`.
- **callgrind**: 10–50× slower but needs no kernel privileges. Writes
  `callgrind.out` plus an annotated `callgrind.txt`.

If `Z3_SYS_Z3_HEADER` is unset and the venv's `z3/include/z3.h` is missing,
the script falls back to `/usr/include/z3.h` so `bindgen` does not fail.

## Debugging Rust-side Logs

The Rust engine emits `log::debug!` / `log::info!` / etc. through a stderr
logger backed by `env_logger::filter::Filter` (defined in
`native/angr/src/engine.rs`). By default the log level is unset and nothing
is printed.

Two ways to turn it on:

```python
# From Python, programmatically
from angr.exploration.rust_manager import set_rust_log_level
set_rust_log_level("debug")    # or "error" / "warn" / "info" / "trace" / "off"
# Or a full RUST_LOG-style spec with per-module filters:
set_rust_log_level("rustylib::stash=warn,rustylib::exploration=info,off")

mgr = angr.exploration.RustExplorationManager(proj, [state])
mgr.run(max_steps=100)
# stderr will carry lines like:  [rust:DEBUG] rustylib::exploration: ...
```

```bash
# From the shell — applied on first RustExplorationManager construction.
# Both env vars are honored; RUST_LOG wins when both are set.
RUST_LOG=debug python tests/benchmarks/run_single.py fauxware --engine rust
ANGR_RUST_LOG=debug python tests/benchmarks/run_single.py fauxware --engine rust
RUST_LOG="rustylib::stash=warn,off" python ...   # per-module filter
```

Caveats:
- Rust modules show up under the `rustylib` crate (e.g.
  `rustylib::exploration`, `rustylib::stash`), not `angr::*` — the crate
  is named `rustylib` for Python import compat.
- `RUST_LOG` / `ANGR_RUST_LOG` are read once per process by
  `_apply_rust_log_env()` (`angr/exploration/rust_manager.py`); changing
  them after the first manager is constructed has no effect — call
  `set_rust_log_level()` explicitly instead.
- Stderr format is `[rust:LEVEL] target: msg` (not env_logger's default;
  we wrap `Filter` inside a custom `log::Log` impl to preserve this).

## Key Files

- **Build config**: `pyproject.toml`, `native/angr/Cargo.toml`
- **Compiler-enforced call-site invariants**: `native/angr-macros/` — proc-macro
  crate (`syn`/`quote`/`proc-macro2`) closing the "same invariant upheld at
  some call sites but not others" bug family from the angr-9ke6b/angr-sqfj8
  audits. `#[angr_macros::steady_guarded]` injects the
  `steady_config_guard()` call `manager_methods*.rs` mutators must make;
  `#[derive(angr_macros::MergePolicy)]` requires every `RustSimState` field to
  carry a documented `#[merge_policy = "..."]` (validation-only — the actual
  merge logic stays hand-written in `state/fork.rs`, since several fields
  need whole-slice/cross-field logic a per-field derive can't express
  safely). Since angr-91vj9.3 the same derive also guards `SymbolicMemory`,
  whose `delegate` label at the `RustSimState` level used to end the
  guarantee one level above where the bug family kept recurring; its
  behavioural half is `memory/tests/merge_sidecars.rs`. See bd memory
  `angr-macros-call-site-invariants`.
- **Rust exploration**: `angr/exploration/rust_manager.py`, `native/angr/src/exploration/` (entry point `mod.rs` — struct def + module wiring only; the `#[pymethods] impl RustExplorationManager` surface lives in sibling `manager_methods.rs` plus its `manager_methods_{hooks,state,constraints,procedures,techniques,export,run,stats}.rs` blocks — one file per former section banner, enabled by PyO3's `multiple-pymethods` feature, angr-nbim4.1 / angr-9ke6b.50 — plus run_loop.rs, stepping.rs, resume.rs, helpers.rs, state_api.rs, stats_api.rs, pending_api.rs, constraints.rs, execution_env.rs, memory_config.rs, profiling.rs, state_lifecycle.rs, state_id.rs, scheduler.rs (work-stealing parallel pool, angr-1ilq.3 — root keeps `CancelToken` / `TaskOutcome` / `TerminalSummary`; the angr-9ke6b.59 split put metrics in `scheduler_stats.rs`, wave-mode `WorkTransport`/`WaveJob` in `scheduler_transport.rs`, the session protocol + `PersistentPool`/`worker_thread` in `scheduler_pool.rs`, and the per-worker loop in `scheduler_worker.rs`), step_core.rs (Send+Sync StepContext + run_interpreter_step_core, angr-1ilq.3 increment 2b-i), and the angr-zel8z.3 data-model split: event.rs (ExplorationEvent pyclass), callback_types.rs (CallbackReason + PendingCallback), native_technique.rs (NativeTechnique enum))
- **Z3 solver**: `native/angr/src/symbolic/context.rs`, `native/angr/src/solver.rs`
  (parent keeps the Python error mapping + claripy-AST API;
  `solver/z3_ast_extract.rs` holds the raw `Z3_ast`-pointer plumbing and every
  `unsafe` on that surface — not to be confused with `symbolic/z3_ast_ptr.rs`,
  which owns the refcounted `Z3AstPtr` type itself (angr-03vl4.75),
  `solver/handle_api.rs` the handle-based claripy-bypass `op_*` API —
  angr-9ke6b.205)
- **Claripy bridge**: `native/angr/src/claripy_bridge/` (entry point `mod.rs` + cache.rs, import.rs, export.rs submodules)
- **VEX interpreter**: `native/angr/src/interpreter/` (mod.rs, execution.rs, expressions.rs, statements.rs, exits.rs, bv_utils.rs, pending_store.rs, prefetch.rs), `native/angr/src/vex/` — contributor guide for adding a new VEX op in [`docs/extending-angr/rust_vex_ops.rst`](docs/extending-angr/rust_vex_ops.rst)
- **Native SimProcedures**: `native/angr/src/procedures/` (strlen, memcpy, strcmp, malloc, free, etc.) — contributor guide in [`docs/extending-angr/simprocedures.rst`](docs/extending-angr/simprocedures.rst) ("Native (Rust) SimProcedures" section)
- **State proxy**: `angr/exploration/rust_state_proxy.py`
- **State export**: `angr/exploration/rust_state_export.py`
- **State sync**: `angr/exploration/rust_state_sync.py`
- **Tests**: `tests/engines/rust/`

## Branch: rust-symex

Active development branch. Work is tracked via beads (`bd ready`).

## Autonomous loop (`ralph`)

The optimization loop runs through [`ralph`](https://github.com/jcrussell/ralph)
(local checkout at `~/repos/ralph`, symlinked to `~/.local/bin/ralph`). Config
lives in `.ralph/`:

- `.ralph/config.toml` — overrides for the runner wrapper (`tools/ralph-claude.sh`)
  and the master base branch. All other knobs (max_iterations=30, timeout=3600s,
  memory=7G, dirty_revert_threshold=3, gate soft_fail, run_when="commits-only")
  use ralph's built-in defaults.
- `.ralph/prompts/{_header,_footer,clean,dirty,revert}.md` — per-state agent
  prompts. `_footer.md` carries the workflow steps, MEMORY SAFETY rules, and
  key-files map.
- `.ralph/hooks/states/{clean,dirty}/gate` — thin wrappers around
  `tools/ralph-gate.sh`, the shared gate body. Fires only on iterations that
  produced commits. Two checks, fail-fast in order:
  1. `cargo test --release` — the Rust unit + integration suite. These encode
     contracts the Python suite never exercises; a real regression once survived
     a full iteration because the gate skipped them (bd `angr-8kk32`, memory
     `cargo-test-not-in-ralph-gate`). ~15s warm, ~2m when an iteration touched
     `native/` and the test binaries must rebuild.
  2. `tests/benchmarks/run_regression.py --rust-only --skip-bimodal
     --threshold 0.15 --retry-failures 2` — matches the CI benchmark gate.
- `.ralph/state/` — runtime state (FSM, logs, session.md handoff). Gitignored
  by ralph init.

Common commands:

```bash
ralph doctor                      # verify prereqs (claude, bd, systemd-run, .ralph/)
ralph run --once --dry-run        # route + render prompt, skip the runner
ralph run --once                  # one real iteration
ralph run                         # autonomous session
ralph status                      # current FSM state, iter count, cost
ralph trace <iter>                # drill into one iteration
ralph hook run states/clean/gate  # standalone gate hook test
```

## Current Status

- **Tests:** over a thousand `def test_` functions in the
  `tests/engines/rust/` package (split from the former monolithic
  `test_rust_exploration.py`; see angr-yg2m). Run
  `grep -rc 'def test_' tests/engines/rust/ | awk -F: '{s+=$2} END {print s}'`
  for the live count, or
  `pytest --collect-only -q tests/engines/rust/ | tail -1`
  for the parametrize-expanded total.
- **Benchmarks:** `tests/benchmarks/baseline_timings.json` is the
  authoritative list. Mix of angr-examples and in-repo synthetic arch
  smoke benches (the `*_branch` entries have `python_time=null` by
  design).
- **Performance:** for the live `python_time / rust_time` ratios, read
  `baseline_timings.json` directly. Known slow benches and their root
  causes are documented in
  [`docs/advanced-topics/rust_engine.rst`](docs/advanced-topics/rust_engine.rst);
  bimodal benches and their variance handling are in
  [`docs/advanced-topics/rust_bimodal_variance.rst`](docs/advanced-topics/rust_bimodal_variance.rst).

## User-facing documentation

End-user documentation for the Rust engine lives in
[`docs/advanced-topics/rust_engine.rst`](docs/advanced-topics/rust_engine.rst).
Refer users there for:

- Usage example (`RustExplorationManager`, `use_rust_engine=True`)
- Architecture overview (Rust core, Python wrapper, PyO3 FFI, shared Z3)
- Architecture support matrix (which arches are Skeleton / Experimental / Supported)
- `RustSolverContext` Z3 API example
- Z3 solver profiling counters (`get_solver_stats`)
- SimOption coverage matrix (honored / inherited / ignored)
- `state.inspect` limitation (raises `NotImplementedError`)
- Known slower benchmarks and their root causes

Keep that file authoritative; cite it from new commits rather than
duplicating prose into CLAUDE.md.
