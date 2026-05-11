# Claude Code Notes

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

**Caveat:** `pyproject.toml` pins `claripy==9.2.210.dev0` etc., but the `.dev0` versions don't exist on PyPI (PyPI jumps 9.2.209 → 9.2.211). Existing working venvs use `9.2.209` from cache and the editable install tolerates the version mismatch in practice. A fresh `pip install` with no cache may resolve to `9.2.213+` and fail the pin — in that case relax the pins or restore from tarball. See bd memory `env-venv-fully-wiped-2026-05-05` for the historical incident.

### Build Commands (incremental)

```bash
# Ensure Rust is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Activate virtualenv
source .venv/bin/activate

# Rebuild Rust .so (editable mode, after changes in native/angr/src/)
pip install -e . --no-build-isolation --no-deps
```

`native/angr/build.rs` auto-detects the Z3 shared library from the active venv's `z3-solver` install and sets the runpath so Python and Rust load the same `libz3.so` (required for AST passthrough). Header discovery is delegated to `z3-sys`, which probes `pkg-config` and falls back to system include paths.

### Common Issues

**"can't find Rust compiler"**: Add `~/.cargo/bin` to PATH before running pip install.

**Stale .so file**: Run `tools/rebuild-rust.sh`. The script wipes
`angr/rustylib*.so` and `build/`, runs `cargo clean`, then re-runs the editable
pip install. Pass `--keep-cargo-cache` to skip `cargo clean` (faster), or
`--cargo-only` to skip pip and build via `cargo build --release` + copy
(recovery path when the venv's pip/setuptools is corrupt — see
`venv-rebuild-cargo-direct-copy` memory).

**"Unable to generate bindings: NotExist ...z3.h"**: The Z3 C headers are missing. Install the system package (`libz3-dev` / `z3-devel` / `brew z3`). To override discovery, set `Z3_SYS_Z3_HEADER=/path/to/z3.h` before invoking `pip install` or `cargo`.

**Linker can't find libz3** at runtime: build.rs sets the runpath to the venv's `z3/lib`, so an `import angr` from inside the venv works. Outside the venv, set `LD_LIBRARY_PATH` to that directory or override with `Z3_LIBRARY_PATH_OVERRIDE=/path/to/lib`.

## Running Tests

```bash
# Run RustExplorationManager tests (332/332 passing as of 2026-05-08)
python -m pytest tests/engines/test_rust_exploration.py -v --tb=short

# Run benchmark regression tests (7 fast-tier benchmarks)
python tests/benchmarks/run_regression.py

# Run single benchmark (safe, subprocess with 4GB memory limit)
python tests/benchmarks/run_single.py fauxware --both

# AVOID run_comparison_10.py on <16GB machines (OOM risk)
```

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
- **VEX interpreter**: `native/angr/src/interpreter.rs`, `native/angr/src/interpreter_cb/` (mod.rs, execution.rs, expressions.rs, statements.rs, exits.rs, constraints.rs, helpers.rs, prefetch.rs), `native/angr/src/vex/`
- **Native SimProcedures**: `native/angr/src/procedures/` (strlen, memcpy, strcmp, malloc, free, etc.)
- **State proxy**: `angr/exploration/rust_state_proxy.py`
- **State export**: `angr/exploration/rust_state_export.py`
- **State sync**: `angr/exploration/rust_state_sync.py`
- **Tests**: `tests/engines/test_rust_exploration.py`

## Branch: rust-symex

Ported from `rust-engine-v2` (176 commits condensed to clean port). Tracked via beads (`bd ready`).

## Current Status

**Tests:** 146/146 passing
**Benchmarks:** 16 benchmarks, all correct (16/16)
**Performance:** 15/16 faster than Python

| Example | Speedup | Notes |
|---------|---------|-------|
| csaw_wyvern | 15.6x | Best case |
| ekopartyctf2016_rev250 | 7.8x | |
| flareon2015_5 | 5.9x | |
| google2016_unbreakable_1 | 3.3x | High variance |
| ais3_crackme | 2.9x | |
| whitehatvn2015_re400 | 2.5x | |
| codegate_2017-angrybird | 2.4x | Was XFAIL, now passing |
| sym-write | 2.2x | |
| strcpy_find | 1.7x | Was 0.2x, fixed CFG interception |
| csgames2018 | 1.6x | |
| google2016_unbreakable_0 | 1.5x | |
| defcamp_r100 | 1.5x | |
| fauxware | 1.4x | Was 0.9x; flipped after NativeRead/NativeWrite enabled by default (angr-3tek.2) cut callbacks from 6 to 1 |
| flareon2015_10 | 1.3x | Callable flow |
| flareon2015_2 | 1.1x | 32-bit x86 |
| securityfest_fairlight | 0.4x | Rust interpreter slower for symbolic-heavy blocks |

## Architecture

- **Rust engine** (`native/angr/src/`): VEX interpreter, Z3 solver, state management
- **Python wrapper** (`angr/exploration/rust_manager.py`): Callback dispatch, state sync, technique support
- **FFI boundary**: PyO3 bindings in `native/angr/src/lib.rs`
- **Shared Z3 context**: Python and Rust share Z3 context for AST passthrough
- **Feature flag**: `use_rust_engine=True` on `proj.factory.simulation_manager()`

## Architecture Support Matrix

Only AMD64 is exercised end-to-end. Other archs have register/state plumbing
and (mostly) calling-convention definitions, but no binary-driven integration
tests and no benchmarks. Treat them as experimental until that changes.

| Arch        | Unit tests | Integration tests | Benchmarks | Calling conv      | Status       |
|-------------|------------|-------------------|------------|-------------------|--------------|
| AMD64       | ~110+      | ~268 (fauxware)   | 15/16      | SystemV, MS x64   | Supported    |
| x86 (32-bit)| 2          | 1 (Cdecl ret reg) | 1 (flareon2015_2) | Cdecl    | Experimental |
| ARM (32-bit)| 2          | 2 (validate, native-proc) | 0  | ARMEABI           | Experimental |
| ARM64       | 1          | 3 (blob branch, real ELF, native-proc) | 0 | AArch64 | Experimental |
| MIPS32      | 4          | 2 (blob branch, native-proc) | 0 | MipsO32        | Experimental |
| MIPS64      | 0          | 0                 | 0          | none (falls back to SystemV) | Skeleton |

**What "Skeleton" means:** `RustSimState("arm64")` etc. constructs successfully,
register reads/writes round-trip, and `fork()` preserves isolation, but no
test runs VEX through the interpreter on a real binary for these archs and
no calling convention is actually exercised. The Cdecl x86 return-register
bug (commit 5329d8222) was latent for months precisely because no end-to-end
x86 test ran — assume the same risk for ARM64/MIPS until coverage lands.

**What's wired up but unverified:**
- Register offsets for all six arches in `native/angr/src/arch/*.rs`
- Endianness flag (MIPS32 BE smoke-tested; ARM/ARM64/MIPS64 BE untested)
- ARMEABI / AArch64 / MipsO32 calling conventions defined in
  `calling_conventions.rs`. MIPS64 still falls back to `SystemVAMD64`,
  which will silently misbehave on any SimProcedure that takes args.

To promote an arch from Skeleton → Experimental: add at least one
integration test that loads a real binary, runs `mgr.run(...)`, and
verifies a found-state result. To promote Experimental → Supported:
add a benchmark and ensure it stays green in regression runs.

## SimOption Coverage

The Rust engine honors only a handful of `angr.sim_options` flags
(`LAZY_SOLVES`, `ZERO_FILL_UNCONSTRAINED_MEMORY`, `APPROXIMATE_MEMORY_INDICES`,
`SYMBOLIC_WRITE_ADDRESSES`, `STRICT_PAGE_ACCESS`). Most other options are
silent no-ops. See [`docs/RUST_SIMOPTION_COVERAGE.md`](docs/RUST_SIMOPTION_COVERAGE.md)
for the per-option matrix (honored / inherited / ignored — divergence-risk /
ignored — no-op).

## state.inspect Unsupported

`state.inspect` breakpoints are not dispatched under the Rust engine.
Registering one through a `RustStateProxy` raises `NotImplementedError`
at call time rather than silently never firing. Use the Python engine
(`proj.factory.simulation_manager(state)` without `use_rust_engine=True`)
for breakpoint-driven analyses. See
[`docs/RUST_STATE_INSPECT.md`](docs/RUST_STATE_INSPECT.md) for the
affected API, the rationale, and the historical decision trail.

## Rust Symbolic Execution

```python
import angr
from angr.exploration import RustExplorationManager

proj = angr.Project("/path/to/binary", auto_load_libs=False)
state = proj.factory.entry_state()
mgr = RustExplorationManager(proj, [state])
mgr.set_find_addresses([0x401234])
mgr.set_avoid_addresses([0x401000])
mgr.run(max_steps=10000)

found_states = mgr.found
for state in found_states:
    print(f"Found at {hex(state.addr)}")
```

## Z3 Solver Integration (z3-rs 0.19)

```python
from angr.rustylib.vex_engine import RustSolverContext
import claripy

ctx = RustSolverContext()
print(f'Z3 available: {ctx.z3_available()}')  # True

x = claripy.BVS('x', 32)
ctx.add_constraint_ast(x > 10)
ctx.add_constraint_ast(x < 20)
print(f'satisfiable: {ctx.satisfiable()}')
print(f'min: {ctx.min(x, signed=False)}')  # 11
print(f'max: {ctx.max(x, signed=False)}')  # 19
```

### Z3 Solver Profiling Counters

Process-wide atomics in `native/angr/src/symbolic/context.rs` track every
`solver.check()` call. Read or reset them via the manager:

```python
mgr = RustExplorationManager(proj, [state])
mgr.reset_solver_stats()
mgr.explore(find=...)
stats = mgr.get_solver_stats()
# stats includes:
#   z3_check_count        — total solver.check() calls
#   z3_check_time_ns      — total time in solver.check()
#   z3_sat_count / z3_unsat_count / z3_timeout_count — by SatResult
#   z3_materialize_count, z3_materialize_time_ns — lazy-fork solver materialization
#   z3_assume_concrete / z3_assume_symbolic — assume_true/false fast-path counters
#   z3_branch_check / z3_branch_concrete / z3_branch_model_hit / z3_branch_model_miss
#   z3_ast_build          — AST construction count
#   z3_site_<name>_count, z3_site_<name>_time_ns — per-call-site breakdown
#     (sites: satisfiable, branch_true, branch_false, eval, eval_upto,
#      min_init, min_search, max_init, max_search)
```

Counters are global (shared across SymContexts). Call
`mgr.reset_solver_stats()` to zero them at the start of a measured window.
The same dict is also merged into `mgr.stats` for convenience.
