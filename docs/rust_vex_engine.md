# Rust VEX Execution Engine

A high-performance VEX execution engine implemented in Rust for angr.

## Overview

The Rust VEX engine provides an alternative execution backend for angr's symbolic execution. Key benefits:

- **O(1) State Forking**: Copy-on-Write memory using immutable data structures
- **Fast Concrete Operations**: ~9ns for bitvector operations
- **Multi-Architecture Support**: AMD64, x86, ARM, ARM64, MIPS32, MIPS64
- **Full Symbolic Execution**: Z3-based symbolic constraint solving

## Installation

The Rust VEX engine is built as part of angr's native library. Ensure you have Rust 1.70+ installed:

```bash
cd native/angr
cargo build --release --features vex-engine
```

## Usage

### Using with angr

```python
import angr
from angr.engines import UberEngineRust, RUST_ENGINE_AVAILABLE

if RUST_ENGINE_AVAILABLE:
    # Create project with Rust engine
    proj = angr.Project("/path/to/binary", auto_load_libs=False)

    # Use Rust engine for simulation
    state = proj.factory.entry_state()
    simgr = proj.factory.simulation_manager(state, engine=UberEngineRust(proj))
    simgr.run()
```

### Direct Engine Access

For lower-level access:

```python
from angr.engines.rust_vex import RustVEXEngineWrapper

# Create engine
engine = RustVEXEngineWrapper("amd64")

# Set registers
engine.set_register("rax", 0x12345678)
engine.set_register("rsp", 0x7FFFFFFFE000)
engine.pc = 0x401000

# Map memory
engine.map_memory(0x400000, 0x10000)  # .text
engine.map_memory(0x7FFFFFFFE000, 0x10000)  # stack

# Write to memory
engine.write_memory(0x401000, b"\x48\x31\xc0")  # xor rax, rax

# Fork state (O(1) operation)
forked = engine.fork()
forked.set_register("rax", 0xDEADBEEF)

# Read registers
print(engine.get_registers())
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Python (angr)                                  │
│  ┌─────────────────────────────────────────┐    │
│  │ RustVEXMixin / UberEngineRust           │    │
│  │ - SimEngine protocol implementation     │    │
│  │ - State synchronization                 │    │
│  │ - Fallback to Python VEX when needed    │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
                    ↓ PyO3 FFI
┌─────────────────────────────────────────────────┐
│  Rust (native/angr)                             │
│  ┌─────────────────────────────────────────┐    │
│  │ RustVEXEngine (engine.rs)               │    │
│  │ - PyO3 bindings                         │    │
│  │ - Block cache management                │    │
│  └─────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────┐    │
│  │ VEXInterpreter (interpreter.rs)         │    │
│  │ - Statement/expression evaluation       │    │
│  │ - Branch handling                       │    │
│  └─────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────┐    │
│  │ SymbolicMemory (memory.rs)              │    │
│  │ - Copy-on-Write pages (im crate)        │    │
│  │ - O(1) fork operation                   │    │
│  └─────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────┐    │
│  │ RustBV (symbolic/value.rs)              │    │
│  │ - Concrete/Symbolic bitvectors          │    │
│  │ - Z3 AST backend                        │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

## Performance

Benchmark results on AMD64:

| Operation | Time | Notes |
|-----------|------|-------|
| memory_fork | 38 µs | O(1) CoW fork |
| memory_read (8B) | 41 ns | Concrete read |
| memory_write (8B) | 42 ns | Concrete write |
| register_access | 39 ns | Read/write cycle |
| bv_add (concrete) | 9 ns | 64-bit addition |
| bv_mul (concrete) | 9 ns | 64-bit multiply |
| interpreter_create | 139 ns | Fresh interpreter |
| irsb_execute | 179 ns | Simple block |
| 100 forks | 70 µs | Linear scaling |
| 500 forks | 1.7 ms | ~3.4 µs/fork |

### Fork Performance

The key advantage is O(1) state forking. Creating 500 states with modifications:

- **Rust Engine**: 1.7ms (3.4µs per fork)
- **Python Engine**: ~500ms (1ms per copy) - estimated

**~300x speedup** for fork-heavy workloads.

## Supported Features

### VEX Operations (~75 parameterized ops)

- **Arithmetic**: Add, Sub, Mul, Div (signed/unsigned)
- **Bitwise**: And, Or, Xor, Not, Shl, Shr, Sar
- **Comparison**: Eq, Ne, Lt, Le, Gt, Ge (signed/unsigned)
- **Conversion**: Sign extend, Zero extend, Truncate
- **SIMD**: Vector add, mul, shift, compare, interleave
- **Floating Point**: Basic float operations (concrete)

### Architectures

| Architecture | Status | Guest State Size |
|--------------|--------|------------------|
| AMD64 | ✓ Full | 768 bytes |
| x86 | ✓ Full | 340 bytes |
| ARM | ✓ Full | 380 bytes |
| ARM64 | ✓ Full | 848 bytes |
| MIPS32 | ✓ Full | 296 bytes |
| MIPS64 | ✓ Full | 592 bytes |

### Memory Model

- Page-based (4KB pages)
- Copy-on-Write using `im::OrdMap`
- Mixed concrete/symbolic values
- Configurable endianness

## Limitations

1. **VEX Lifting**: Currently uses pyvex for lifting. Native VEX lifting planned.
2. **Floating Point**: Limited to concrete operations.
3. **Syscalls**: Handled by Python side via SimProcedures.
4. **Hooks**: Require return to Python for execution.

## Development

### Running Tests

```bash
# All tests
cargo test --features vex-engine

# Specific module
cargo test --features vex-engine memory::tests
```

### Running Benchmarks

```bash
cargo bench --features vex-engine
```

### Adding New Architectures

1. Create `src/arch/<arch>.rs` with VEX guest state offsets
2. Implement `Arch` trait
3. Add to `arch/mod.rs` exports and factory functions
4. Add tests

## API Reference

### RustVEXEngineWrapper

```python
class RustVEXEngineWrapper:
    def __init__(self, arch: str = "amd64"): ...

    # Properties
    pc: int  # Program counter

    # Registers
    def get_register(self, name: str) -> int: ...
    def set_register(self, name: str, value: int) -> None: ...
    def get_registers(self) -> dict[str, int]: ...

    # Memory
    def map_memory(self, addr: int, size: int, permissions: int = 7) -> None: ...
    def map_memory_data(self, addr: int, data: bytes, permissions: int = 7) -> None: ...
    def read_memory(self, addr: int, size: int) -> bytes: ...
    def write_memory(self, addr: int, data: bytes) -> None: ...

    # Execution
    def step(self) -> ExecutionEvent: ...
    def fork(self) -> RustVEXEngineWrapper: ...

    # Hooks
    def add_hook(self, addr: int) -> None: ...
    def remove_hook(self, addr: int) -> None: ...

    # Debugging
    def snapshot(self) -> StateSnapshot: ...
    def stats(self) -> dict[str, Any]: ...
```

### ExecutionEvent

```python
class ExecutionEvent:
    event_type: str  # "block_end", "symbolic_branch", "syscall", "hook", "error"
    next_addr: int | None
    jumpkind: str | None
    syscall_num: int | None
    error: str | None
    true_target: int | None  # For symbolic branches
    false_target: int | None
```

## Future Work

- [ ] Native VEX lifting (libvex-rs)
- [ ] Symbolic floating point
- [ ] JIT compilation for hot paths
- [ ] Parallel state exploration
- [ ] Memory-mapped files
