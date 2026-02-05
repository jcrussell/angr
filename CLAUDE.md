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

## Running Differential Tests

```bash
# Run Tier 3 extended tests
python -m pytest tests/engines/differential/test_tier3_extended.py -v --tb=short

# Check pass rate from report
python3 -c "
import json
with open('tests/engines/differential/reports/tier3_extended_report.json') as f:
    report = json.load(f)
s = report['summary']
print(f'Pass rate: {s[\"pass_rate\"]:.1%} ({s[\"passed\"]}/{s[\"total\"]})')
"
```

## Key Files

- **Build config**: `pyproject.toml` (setuptools-rust at lines 96-99)
- **Rust source**: `native/angr/src/vex/opcode_map.rs`, `native/angr/src/vex/ops.rs`
- **Z3 solver**: `native/angr/src/symbolic/context.rs`, `native/angr/src/solver.rs`
- **Tests**: `tests/engines/differential/test_tier3_extended.py`
- **Rust engine docs**: `docs/rust_vex_engine.md`

## Z3 Solver Integration (z3-rs 0.19)

The Rust VEX engine uses z3-rs 0.19+ for real Z3 constraint solving. Key points:

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

**Tier 3 Extended Tests:** 100% pass rate (920/920)
