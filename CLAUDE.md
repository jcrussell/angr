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

# Clean old artifacts (if rebuilding)
rm -rf build/ target/ angr/rustylib.cpython-312-x86_64-linux-gnu.so

# Build and install (editable mode)
pip install -e .
```

### Common Issues

**"can't find Rust compiler"**: Add `~/.cargo/bin` to PATH before running pip install.

**Stale .so file**: If `maturin develop` was used previously, it installs to site-packages but Python loads from the source tree. Delete `angr/rustylib.*.so` and rebuild with `pip install -e .`

## Running Tests

```bash
# Run RustExplorationManager tests
python -m pytest tests/engines/test_rust_exploration.py -v --tb=short

# Run a comparison against Python engine
python tests/benchmarks/run_comparison_10.py
```

## Key Files

- **Build config**: `pyproject.toml` (setuptools-rust at lines 96-99)
- **Rust exploration**: `angr/exploration/rust_manager.py`, `native/angr/src/exploration.rs`
- **Z3 solver**: `native/angr/src/symbolic/context.rs`, `native/angr/src/solver.rs`
- **Claripy bridge**: `native/angr/src/claripy_bridge.rs`
- **Tests**: `tests/engines/test_rust_exploration.py`

## Rust Symbolic Execution

The Rust engine provides full symbolic execution with Python callbacks for SimProcedures:

```python
import angr
from angr.exploration import RustExplorationManager

proj = angr.Project("/path/to/binary", auto_load_libs=False)
state = proj.factory.entry_state()

# Create Rust exploration manager
mgr = RustExplorationManager(proj, [state])

# Set exploration targets
mgr.set_find_addresses([0x401234])  # Target addresses
mgr.set_avoid_addresses([0x401000])  # Addresses to avoid

# Run exploration (Rust handles symex, Python handles SimProcedures)
mgr.run(max_steps=10000)

# Get found states
found_states = mgr.found
for state in found_states:
    print(f"Found at {hex(state.addr)}")
```

## Z3 Solver Integration (z3-rs 0.19)

The Rust engine uses z3-rs 0.19+ for real Z3 constraint solving:

- **Thread-local context**: z3-rs 0.19+ uses a thread-local context model
- **No lifetime parameters**: Z3 types (Solver, BV, Bool) don't have lifetime parameters
- **Unsendable pyclass**: `RustSolverContext` uses `#[pyclass(unsendable)]` due to Z3's non-Send types

### Testing Z3 Integration

```python
from angr.rustylib.vex_engine import RustSolverContext
import claripy

ctx = RustSolverContext()
print(f'Z3 available: {ctx.z3_available()}')  # Should print True

x = claripy.BVS('x', 32)
ctx.add_constraint_ast(x > 10)
ctx.add_constraint_ast(x < 20)
print(f'satisfiable: {ctx.satisfiable()}')
print(f'min: {ctx.min(x, signed=False)}')  # 11
print(f'max: {ctx.max(x, signed=False)}')  # 19
```

## Current Test Status

**RustExplorationManager Tests:** 14/14 passing

Tests include unit tests for state management, stash handling, and an integration test with the fauxware example.
