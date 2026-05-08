# Claude Code Notes

## Building the Rust Extension

This project uses **setuptools-rust** (not maturin) to build the Rust native extension.

### Prerequisites
- Python virtualenv at `.venv/`
- Rust toolchain (rustup) at `~/.cargo/bin/`
- Z3 development headers — `apt install libz3-dev` on Debian/Ubuntu, `dnf install z3-devel` on Fedora, `brew install z3` on macOS. The PyPI `z3-solver` wheel ships `libz3.so` but **not** the C headers; `z3-sys` needs the headers to generate bindings.

### Build Commands

```bash
# Ensure Rust is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Activate virtualenv
source .venv/bin/activate

# Build and install (editable mode)
pip install -e . --no-build-isolation --no-deps
```

`native/angr/build.rs` auto-detects the Z3 shared library from the active venv's `z3-solver` install and sets the runpath so Python and Rust load the same `libz3.so` (required for AST passthrough). Header discovery is delegated to `z3-sys`, which probes `pkg-config` and falls back to system include paths.

### Common Issues

**"can't find Rust compiler"**: Add `~/.cargo/bin` to PATH before running pip install.

**Stale .so file**: Delete `angr/rustylib.*.so` and rebuild with `pip install -e .`

**"Unable to generate bindings: NotExist ...z3.h"**: The Z3 C headers are missing. Install the system package (`libz3-dev` / `z3-devel` / `brew z3`). To override discovery, set `Z3_SYS_Z3_HEADER=/path/to/z3.h` before invoking `pip install` or `cargo`.

**Linker can't find libz3** at runtime: build.rs sets the runpath to the venv's `z3/lib`, so an `import angr` from inside the venv works. Outside the venv, set `LD_LIBRARY_PATH` to that directory or override with `Z3_LIBRARY_PATH_OVERRIDE=/path/to/lib`.

## Running Tests

```bash
# Run RustExplorationManager tests (146/146 passing)
python -m pytest tests/engines/test_rust_exploration.py -v --tb=short

# Run benchmark regression tests (7 fast-tier benchmarks)
python tests/benchmarks/run_regression.py

# Run single benchmark (safe, subprocess with 4GB memory limit)
python tests/benchmarks/run_single.py fauxware --both

# AVOID run_comparison_10.py on <16GB machines (OOM risk)
```

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
**Performance:** 14/16 faster than Python

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
| flareon2015_10 | 1.3x | Callable flow |
| flareon2015_2 | 1.1x | 32-bit x86 |
| fauxware | 0.9x | Per-callback FFI overhead |
| securityfest_fairlight | 0.4x | Rust interpreter slower for symbolic-heavy blocks |

## Architecture

- **Rust engine** (`native/angr/src/`): VEX interpreter, Z3 solver, state management
- **Python wrapper** (`angr/exploration/rust_manager.py`): Callback dispatch, state sync, technique support
- **FFI boundary**: PyO3 bindings in `native/angr/src/lib.rs`
- **Shared Z3 context**: Python and Rust share Z3 context for AST passthrough
- **Feature flag**: `use_rust_engine=True` on `proj.factory.simulation_manager()`

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
