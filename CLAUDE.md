# Claude Code Notes

## Makefile Shortcuts

Common dev workflows are wired up in the repo-root `Makefile`. Run `make help`
for the full list. Most-used targets:

| Target | What it does |
|--------|--------------|
| `make rebuild` | Incremental Rust .so rebuild via pip editable install. |
| `make rebuild-cargo` | Cargo-direct rebuild (broken-venv fallback). |
| `make check` | `cargo check --release` — fast type/borrow check. |
| `make test` (`test-quick`) | Run `tests/engines/test_rust_exploration.py` (~1-2 min). |
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

**Note:** `pyproject.toml` pins the four angr-ecosystem deps (`archinfo`, `claripy`, `cle`, `pyvex`) to `==9.2.209`. The previous pin (`9.2.210.dev0`) was unobtainable on PyPI (which jumps 9.2.209 → 9.2.211), so fresh installs without cache failed. `9.2.209` is the latest available in the 9.2.20x range and is what existing working venvs already have installed. Bumping past `9.2.209` is risky because newer claripy releases may bundle a different `z3-solver` version that would break the Rust↔Python shared Z3 context — see bd memory `avoid-pip-install-deps`.

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
# Run RustExplorationManager tests (332/332 passing as of 2026-05-08)
python -m pytest tests/engines/test_rust_exploration.py -v --tb=short

# Run benchmark regression tests (7 fast-tier benchmarks)
python tests/benchmarks/run_regression.py

# Skip the three bimodal-Z3 benches (unbreakable_1 / fairlight / sokohashv2)
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

# Point at a custom angr-examples checkout (defaults to ~/repos/angr-examples)
ANGR_EXAMPLES_DIR=/path/to/angr-examples/examples python tests/benchmarks/run_regression.py

# AVOID run_comparison_10.py on <16GB machines (OOM risk)
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

The Rust engine emits `log::debug!` / `log::info!` / etc. through a custom
`StderrLogger` (defined at `native/angr/src/engine.rs:1476`). By default the
log level is unset and nothing is printed.

Two ways to turn it on:

```python
# From Python, programmatically
from angr.exploration.rust_manager import set_rust_log_level
set_rust_log_level("debug")    # or "error" / "warn" / "info" / "trace" / "off"

mgr = angr.exploration.RustExplorationManager(proj, [state])
mgr.run(max_steps=100)
# stderr will carry lines like:  [rust:DEBUG] angr::exploration: ...
```

```bash
# From the shell — applied on first RustExplorationManager construction
ANGR_RUST_LOG=debug python tests/benchmarks/run_single.py fauxware --engine rust
```

Caveats:
- The standard `RUST_LOG` env var is **not** honored. The engine uses
  `log::set_max_level` plus a hand-rolled stderr logger, not `env_logger`.
- Per-module filters (e.g. `RUST_LOG=angr::stepping=debug`) are not
  supported — only a single global level.
- `ANGR_RUST_LOG` is read once per process by `_apply_rust_log_env()`
  (`angr/exploration/rust_manager.py`); changing it after the first
  manager is constructed has no effect — call `set_rust_log_level()`
  explicitly instead.

## Key Files

- **Build config**: `pyproject.toml`, `native/angr/Cargo.toml`
- **Rust exploration**: `angr/exploration/rust_manager.py`, `native/angr/src/exploration/` (mod.rs, run_loop.rs, stepping.rs, resume.rs, helpers.rs, pyapi.rs)
- **Z3 solver**: `native/angr/src/symbolic/context.rs`, `native/angr/src/solver.rs`
- **Claripy bridge**: `native/angr/src/claripy_bridge.rs`
- **VEX interpreter**: `native/angr/src/interpreter.rs`, `native/angr/src/interpreter_cb/` (mod.rs, execution.rs, expressions.rs, statements.rs, exits.rs, constraints.rs, helpers.rs, prefetch.rs), `native/angr/src/vex/` — contributor guide for adding a new VEX op in [`docs/extending-angr/rust_vex_ops.rst`](docs/extending-angr/rust_vex_ops.rst)
- **Native SimProcedures**: `native/angr/src/procedures/` (strlen, memcpy, strcmp, malloc, free, etc.) — contributor guide in [`docs/extending-angr/simprocedures.rst`](docs/extending-angr/simprocedures.rst) ("Native (Rust) SimProcedures" section)
- **State proxy**: `angr/exploration/rust_state_proxy.py`
- **State export**: `angr/exploration/rust_state_export.py`
- **State sync**: `angr/exploration/rust_state_sync.py`
- **Tests**: `tests/engines/test_rust_exploration.py`

## Branch: rust-symex

Ported from `rust-engine-v2` (176 commits condensed to clean port). Tracked via beads (`bd ready`).

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

The legacy `run_optimization_loop.py` is kept in the repo root during a soak
period — do not run both simultaneously.

## Current Status

**Tests:** 484/484 passing (`grep -c 'def test_' tests/engines/test_rust_exploration.py`; 492 after pytest parametrize expansion)
**Benchmarks:** 26 benchmarks tracked in `tests/benchmarks/baseline_timings.json` (22 angr-examples + 4 in-repo synthetics: ARM, MIPS32, MIPS64, AArch64)
**Performance:** 17/22 faster than Python (≥1.0x; arm_le_branch, mips32_le_branch and aarch64_le_branch are synthetic rust_only smoke benchmarks)

| Example | Speedup | Notes |
|---------|---------|-------|
| csaw_wyvern | 16.9x | Best case |
| ekopartyctf2016_rev250 | 15.7x | |
| flareon2015_5 | 10.3x | |
| defcamp_r100__dfs | 4.5x | DFS variant |
| defcamp_r100 | 4.4x | |
| ais3_crackme | 3.0x | |
| whitehatvn2015_re400 | 2.6x | |
| codegate_2017-angrybird | 2.4x | Was XFAIL, now passing |
| strcpy_find | 2.3x | Was 0.2x, fixed CFG interception |
| sym-write | 2.3x | |
| defcon2016quals_baby-re | 2.0x | |
| csgames2018 | 1.6x | |
| google2016_unbreakable_0 | 1.5x | |
| flareon2015_10 | 1.4x | Callable flow |
| fauxware | 1.4x | Was 0.9x; flipped after NativeRead/NativeWrite enabled by default (angr-3tek.2) cut callbacks from 6 to 1 |
| unmapped_analysis | 1.2x | |
| flareon2015_2 | 1.1x | 32-bit x86 |
| securityfest_fairlight | 1.0x | Rust interpreter parity for symbolic-heavy blocks |
| hackcon2016_angry-reverser | 0.7x | angr-8t45 (2026-05-18) per-state lazy-zero cap; angr-tlvl (2026-05-19) fixed latent Reverse leaf-emission bug — Z3 shape now matches claripy but hackcon residual gap unchanged (5-sample median ~14.84s). See `docs/advanced-topics/rust_engine.rst`. |
| mma_howtouse | 0.6x | 45 isolated `Callable` invocations; AST-cache hypothesis invalidated 2026-05-17 (claripy `clear_all_caches()` removed; Rust-side cache hook delivered 0%). Residual gap unattributed. See [`docs/advanced-topics/rust_engine.rst`](docs/advanced-topics/rust_engine.rst) |
| google2016_unbreakable_1 | 0.5x | High variance; regressed from 3.3x |
| ekopartyctf2016_sokohashv2 | 0.4x | x87 transcendentals fall back to Python; bimodal Z3 variance. See [`docs/advanced-topics/rust_engine.rst`](docs/advanced-topics/rust_engine.rst) |

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
