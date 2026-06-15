# Claude Code Notes

## Memory system

Project memory lives in **bd**, not in Claude's per-project auto-memory
markdown files. Run `bd memories | head -1` for the current count
(~414 as of 2026-06-05). Audit/cleanup tooling and pre-prune snapshots
live at `~/angr-memories/` — see its README if you ever need to recover
or re-run a prune.

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
- `benchmark-bimodal-variance-rules` — handling the bimodal-Z3 benches (4; see `BIMODAL_BENCHMARKS`)
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
| `make check` | `cargo check --release` — fast type/borrow check. |
| `make test` (`test-quick`) | Run `tests/engines/test_rust_exploration.py` (~1-2 min). |
| `make test-python-baseline` | Fast (~5min) vanilla Python-engine regression subset (needs `../binaries`). |
| `make test-full` | Full angr test suite (long). |
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
.venv/bin/python -m pytest tests/engines/test_rust_exploration.py --tb=short -q
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
python -m pytest tests/engines/test_rust_exploration.py -v --tb=short

# Run benchmark regression tests (7 fast-tier benchmarks)
python tests/benchmarks/run_regression.py

# Skip the four bimodal-Z3 benches (see BIMODAL_BENCHMARKS in run_regression.py:
# unbreakable_1 / fairlight / sokohashv2 / angry-reverser)
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
# diff on a timing regression) without touching baseline_timings.json.
python tests/benchmarks/run_regression.py --rust-only --skip-bimodal --update-counters

# Point at a custom angr-examples checkout (defaults to ~/repos/angr-examples)
ANGR_EXAMPLES_DIR=/path/to/angr-examples/examples python tests/benchmarks/run_regression.py
```

### Benchmark gates in CI

- **`.github/workflows/ci.yml::benchmark_regression`** (PR-time): runs the
  fast-tier suite with `--rust-only --skip-bimodal --threshold 0.15`, capped at
  3-minute timeout. Fails a PR when any non-bimodal fast-tier bench is more than
  15% slower than `baseline_timings.json`. Warns (but does not fail) when the
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

**Threshold tuning notes:**
- Healthy ratio is ~1.00-1.05x (Z3+claripy caches warm up but don't grow
  unboundedly).
- 1.5x is the recommended default: large enough to avoid false-positives
  from cache warmup, small enough to catch a ~2x regression on the first
  nightly after it lands.
- If a future legitimate change pushes healthy ratio above ~1.3x (e.g. a
  larger steady-state cache), raise the threshold incrementally — do not
  lower below 1.2x without first ruling out warmup variance.
- For benches outside `mma_howtouse`, expect different baselines; rerun
  N=10 once locally before picking a threshold.

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
- **Rust exploration**: `angr/exploration/rust_manager.py`, `native/angr/src/exploration/` (13 modules; entry point `mod.rs`, plus run_loop.rs, stepping.rs, resume.rs, helpers.rs, state_api.rs, stats_api.rs, pending_api.rs, constraints.rs, execution_env.rs, memory_config.rs, profiling.rs, state_lifecycle.rs)
- **Z3 solver**: `native/angr/src/symbolic/context.rs`, `native/angr/src/solver.rs`
- **Claripy bridge**: `native/angr/src/claripy_bridge.rs`
- **VEX interpreter**: `native/angr/src/interpreter/` (mod.rs, execution.rs, expressions.rs, statements.rs, exits.rs, helpers.rs, pending_store.rs, prefetch.rs), `native/angr/src/vex/` — contributor guide for adding a new VEX op in [`docs/extending-angr/rust_vex_ops.rst`](docs/extending-angr/rust_vex_ops.rst)
- **Native SimProcedures**: `native/angr/src/procedures/` (strlen, memcpy, strcmp, malloc, free, etc.) — contributor guide in [`docs/extending-angr/simprocedures.rst`](docs/extending-angr/simprocedures.rst) ("Native (Rust) SimProcedures" section)
- **State proxy**: `angr/exploration/rust_state_proxy.py`
- **State export**: `angr/exploration/rust_state_export.py`
- **State sync**: `angr/exploration/rust_state_sync.py`
- **Tests**: `tests/engines/test_rust_exploration.py`

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
- `.ralph/hooks/states/{clean,dirty}/gate` — benchmark regression check
  (`tests/benchmarks/run_regression.py --rust-only --skip-bimodal --threshold 0.15`),
  matching the CI gate. Fires only on iterations that produced commits.
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

- **Tests:** several hundred `def test_` functions in
  `tests/engines/test_rust_exploration.py`. Run
  `grep -c 'def test_' tests/engines/test_rust_exploration.py` for the
  live count, or
  `pytest --collect-only -q tests/engines/test_rust_exploration.py | tail -1`
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
