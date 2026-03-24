# rust-engine-v2 Branch Status

## Architecture: Rust Owns Memory

```
Python init → Rust SymbolicMemory → VEX execution → Python callbacks for externals only
```

- **Init**: Python state pages + symbolic values imported to Rust SymbolicMemory
- **Execute**: VEX loads/stores go directly to Rust (no Python FFI for memory)
- **Callback**: SimProcedures get [SP] + 8 args synced from Rust, plus Python cached state
- **Fork**: Rust CoW SymbolicMemory fork (O(1))
- **Export**: Found states use Python cached state with Rust constraints

## Passing Examples (8)

| Example | Time | Output |
|---------|------|--------|
| fauxware | 1.1s | SOSNEAKY |
| ais3_crackme | 17.0s | ais3{I_tak3_g00d_n0t3s} |
| ekoparty | 152s | correct |
| cmu_binary_bomb p1 | 7s | correct |
| crackme0x00a | 1.2s | g00dJ0B! |
| crackme0x01 | 1.3s | 5274 |
| crackme0x02 | 1.3s | 338724 |
| crackme0x03 | 1.2s | 338724 |

## Performance Profile

- 94% time in Rust VEX+Z3 interpreter
- 6% time in Python callbacks
- Python boundary is NOT the bottleneck

## Build & Test

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release
cp target/release/librustylib.so angr/rustylib.cpython-312-x86_64-linux-gnu.so

# Quick test:
python -c "
import angr; from angr.exploration import RustExplorationManager
p = angr.Project('path/to/fauxware', auto_load_libs=False)
m = RustExplorationManager(p, [p.factory.entry_state()])
m.explore(find=0x4006ed, avoid=0x40073d)
print('PASS' if m.found else 'FAIL')
"
```

## Next Steps (Rust-side engineering)

1. **Concrete execution fast-path**: skip Z3 for all-concrete VEX blocks
2. **Z3 result caching**: cache solver results for repeated constraint patterns
3. **Unicorn integration**: hardware-accelerated concrete execution
4. **Incremental solving**: reuse Z3 solver state across steps
