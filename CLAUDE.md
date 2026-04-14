# Claude Code Notes

## Building the Rust Extension

This project uses **setuptools-rust** (not maturin) to build the Rust native extension.

### Prerequisites
- Python virtualenv at `.venv/`
- Rust toolchain (rustup) at `~/.cargo/bin/`

### Build Commands

```bash
# Ensure Rust is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Activate virtualenv
source .venv/bin/activate

# Build and install (editable mode)
pip install -e . --no-build-isolation --no-deps
```

### Common Issues

**"can't find Rust compiler"**: Add `~/.cargo/bin` to PATH before running pip install.

**Stale .so file**: Delete `angr/rustylib.*.so` and rebuild with `pip install -e .`

## Running Tests

```bash
# Run RustExplorationManager tests (82/82 passing)
python -m pytest tests/engines/test_rust_exploration.py -v --tb=short

# Run benchmark regression tests (5 fast-tier benchmarks)
python tests/benchmarks/run_regression.py

# Run single benchmark (safe, subprocess with 4GB memory limit)
python tests/benchmarks/run_single.py fauxware --both

# AVOID run_comparison_10.py on <16GB machines (OOM risk)
```

## Key Files

- **Build config**: `pyproject.toml`, `native/angr/Cargo.toml`
- **Rust exploration**: `angr/exploration/rust_manager.py`, `native/angr/src/exploration.rs`
- **Z3 solver**: `native/angr/src/symbolic/context.rs`, `native/angr/src/solver.rs`
- **Claripy bridge**: `native/angr/src/claripy_bridge.rs`
- **VEX interpreter**: `native/angr/src/interpreter.rs`, `native/angr/src/vex/`
- **Native SimProcedures**: `native/angr/src/procedures/` (strlen, memcpy, strcmp, malloc, free, etc.)
- **State proxy**: `angr/exploration/rust_state_proxy.py`
- **Tests**: `tests/engines/test_rust_exploration.py`

## Branch: rust-symex

Ported from `rust-engine-v2` (176 commits condensed to clean port). Tracked via beads (`bd ready`).

## Current Status

**Tests:** 82/82 passing
**Benchmarks:** 12/12 correct results (5 regression-tested)
**Performance:** 7/12 faster than Python

| Example | Speedup | Notes |
|---------|---------|-------|
| wyvern | 14.7x | Best case |
| ekoparty | 8.2x | |
| flareon5 | 5.85x | |
| ais3 | 3.55x | |
| whitehatvn | 2.7x | Partial output divergence |
| defcamp | 2.04x | |
| flareon10 | 1.37x | Callable flow |
| fairlight | 1.15x | |
| fauxware | 0.91x | Per-callback FFI (lazy fork_base) |
| strcpy_find | 0.21x | Callable predicate overhead |
| sym-write | 0.08x | Eager symbolic store |
| hackcon | varies | Z3 AST structure |

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
