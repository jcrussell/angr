# Loop session notes (2026-05-07, 105th loop session)

## Task: angr-75y3 — FxHasher for SymbolicMemory hot maps (CLOSED)

Bead closed (commit df3a5c53e).

### What changed

Added `rustc-hash = "2.1"` as a direct dep on the angr crate (was
already transitively present via z3-sys/syn).

`native/angr/src/memory/mod.rs`: swapped four u64-keyed std collections
to their rustc_hash equivalents.

| Field            | Before                       | After                          |
|------------------|------------------------------|--------------------------------|
| symbolic_objects | HashMap<u64, RustBV>         | FxHashMap<u64, RustBV>         |
| symbolic_spans   | HashMap<u64, (u64, u32)>     | FxHashMap<u64, (u64, u32)>     |
| dirty_pages      | HashSet<u64>                 | FxHashSet<u64>                 |
| imported_addrs   | HashSet<u64>                 | FxHashSet<u64>                 |

The local HashSet<u64> temporaries in `try_merge_with` are off the hot
path and stay explicitly std::collections-qualified.

### Measured impact (criterion --baseline before)

```
memory_concrete/store_8bytes : 53.28 ns -> 33.04 ns  (-38.0%, p < 0.05)
memory_fork_16pages          : 45.30 ns -> 38.24 ns  (-15.5%, p < 0.05)
state_fork                   :283.80 ns -> 256.29 ns ( -9.8%, p < 0.05)
memory_concrete/load_8bytes  : 27.98 ns -> 27.49 ns  ( -1.6%, noise)
memory_concrete/load_1byte   : 21.35 ns -> 21.24 ns  ( -0.6%, noise)
memory_symbolic_load_16range : 1.629 ms -> 1.630 ms  (no change, Z3-bound)
```

store_8bytes is the headline: every store hits dirty_pages.insert plus
a symbolic_objects.remove on the no-symbolic path; both are now ~2-3x
faster on u64 keys. fork_16pages drops because cloning two
FxHashMaps + two FxHashSets is meaningfully cheaper than the std
SipHasher-based versions. load is mostly empty-table .get() so the
hash itself dominates and SipHasher already short-circuits a near-empty
table — small win there.

### Tests

- 254 Python tests pass (tests/engines/test_rust_exploration.py)
- 517 Rust unit tests pass
- 11/12 regression benchmarks pass; csgames2018 was already broken on
  master per `avoid-csgames2018-as-regression-signal` memory.

### Memories saved

- `benchmark-memory-fxhash` — bench numbers + reasoning per field.
- `invariant-fxhash-internal-keys` — pattern: u64-keyed internal
  collections on hot paths should use FxHasher; remaining candidates
  (RegisterFile.symbolic, HeapMetadata, RustExplorationManager
  internal id maps).
- `reference-rustc-hash-dep` — how to add the dep + that it's already
  transitive so lockfile delta is one line.

### Build note

Same as last session: `pip install -e .` broken in venv. Workaround:

    Z3_SYS_Z3_HEADER=/usr/include/z3.h cargo build --manifest-path \
        native/angr/Cargo.toml --release --lib
    cp target/release/librustylib.so \
        angr/rustylib.cpython-312-x86_64-linux-gnu.so

The run_single.py harness needs `PYTHONPATH=/home/ubuntu/repos/angr`
when invoked from a multiprocessing-spawn context — the spawned child
doesn't inherit the activated venv's site-packages path even though the
parent does.

## Suggested next slices

- `angr-6n56` (P2) — Arc-tree teardown for transient RustBV results,
  36% of symbolic arithmetic time. Bigger refactor (likely needs an
  arena).
- Apply same FxHasher swap to other u64-keyed hot maps:
  - `RegisterFile.symbolic: HashMap<u32, RustBV>` (2.48% in fork)
  - `HeapMetadata` internals (1% in fork)
  - `RustExplorationManager` id maps (state_fork still has fork
    overhead from these)
- `angr-pufm` (P1) lazy guarded-entries (still open, deferred multiple
  times — needs split per the audit note).
