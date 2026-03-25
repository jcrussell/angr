# rust-engine-v2 Branch Status

## Architecture: Rust Owns Memory

```
Python init → Rust SymbolicMemory → VEX execution → Python callbacks for externals only
```

- **Init**: Python state pages + symbolic values imported to Rust SymbolicMemory
- **Execute**: VEX loads/stores go directly to Rust (no Python FFI for memory)
- **Callback**: Stack synced from Rust + symbolic registers converted to claripy ASTs
- **Fork**: Rust CoW SymbolicMemory fork (O(1))
- **Export**: Found states use Python cached state with Rust constraints

## Passing Examples (11)

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
| asisctffinals2015_fake | 0.9s | correct |
| sharif7_rev50 | 1.0s | correct |
| hitcon2017_sakura | 1.6s | correct |

## Performance Profile

- 94% time in Rust VEX+Z3 interpreter
- 6% time in Python callbacks
- Python boundary is NOT the bottleneck
- Concrete VEX execution: ~0.06ms/block (3.2s for 50K blocks)

## Key Commits (29 total)

- Phase 1: Wire Rust memory into interpreter (fauxware)
- Phase 2: Symbolic value import + SimProcedure memory sync (ais3)
- Phase 3: State export (ekoparty)
- Phase 4-6: Testing, optimization, per-example diagnosis
- Phase 7: Memory proxy attempt (reverted — needs MemoryMixin)
- Symbolic register sync via rustbv_to_claripy
- Sat check on found states
- load_concrete/store_concrete single-page optimization

## Build & Test

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release
cp target/release/librustylib.so angr/rustylib.cpython-312-x86_64-linux-gnu.so

# Quick regression (8 examples, ~30s):
python -c "
import angr, claripy; from angr.exploration import RustExplorationManager
tests = [
    ('fauxware', 'fauxware/fauxware', None, 0x4006ed, 0x40073d),
    ('ais3', 'ais3_crackme/ais3_crackme', ['./c', claripy.BVS('a',800)], 0x400602, None),
    ('crackme0x01', 'CSCI-4968-MBE/challenges/crackme0x01/crackme0x01', None, 0x0804844e, 0x08048434),
]
base = '/home/ubuntu/repos/angr-examples/examples/'
for name, path, args, find, avoid in tests:
    p = angr.Project(base+path, auto_load_libs=False)
    s = p.factory.entry_state(args=args) if args else p.factory.entry_state()
    m = RustExplorationManager(p, [s])
    m.explore(find=find, avoid=avoid, max_steps=5000)
    assert m.found, f'{name} failed'
print('ALL PASS')
"
```

## Remaining Work

### Performance (Rust-side)
- Concrete fast-path: unicorn or native execution for all-concrete blocks
- Z3 caching: reuse solver results for repeated constraint patterns
- Incremental solving: reuse Z3 state across steps

### Correctness
- Dual-solver constraint propagation (Rust Z3 vs Python claripy)
- Full memory proxy (MemoryMixin, not monkey-patch)
- Windows PE support (different init sequence)
- ARM architecture support

### Features
- Callable find/avoid predicates
- Custom ExplorationTechniques with filter()
- SimFile support for file-based symbolic input
