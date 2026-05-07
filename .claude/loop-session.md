# Loop session notes (2026-05-07, 106th loop session)

## Task: angr-z84i — FxHasher for RegisterFile.symbolic & HeapMetadata.allocated (CLOSED)

Bead closed (commit 26a9a6676).

### What changed

Extended the angr-75y3 FxHasher pattern to two more u64/u32-keyed std
HashMaps on the fork-and-step hot path.

| Field                         | Before                | After                |
|-------------------------------|-----------------------|----------------------|
| RegisterFile.symbolic         | HashMap<u32, RustBV>  | FxHashMap<u32, RustBV> |
| HeapMetadata.allocated        | HashMap<u64, u64>     | FxHashMap<u64, u64>  |

The local HashSet<u32> in RegisterFile::merge (arch/mod.rs:445) is off
the hot path and stays std::collections-qualified per the
`invariant-fxhash-internal-keys` pattern.

### Measured impact (criterion --baseline before)

```
state_fork                : 256.29 ns -> 204.59 ns (-20.3%, p < 0.05)
memory_concrete/store_8b  :  33.04 ns -> 33.23 ns  (no change, +0.6% noise)
memory_concrete/load_1b   :  21.24 ns -> 21.17 ns  (no change)
memory_fork_16pages       :  38.24 ns -> 39.97 ns  (+4.3%, allocator noise on a 40 ns workload)
```

The state_fork win is the headline. state.fork() clones the
RegisterFile (which holds .symbolic) and HeapMetadata (which holds
.allocated). After this session both are FxHashMap-cloned instead of
SipHasher-cloned, dropping the per-fork hash-table copy cost.

memory_fork_16pages should not have been touched by this change. The
+4.3% drift is allocator/heat noise (40 ns workload, 8 high outliers).
Worth re-measuring next session if it persists.

### Tests

- 254/254 Python tests pass (tests/engines/test_rust_exploration.py)
- 517/517 Rust unit tests pass
- 11/12 regression benchmarks pass; csgames2018 was already broken on
  master per `avoid-csgames2018-as-regression-signal`.

### Memories saved

- `benchmark-fxhash-register-heap` — bench numbers + reasoning per field.
- `invariant-fxhash-internal-keys` — updated with new completed targets;
  remaining candidates (PredictiveEngine.symbolic_registers, RustExplorationManager
  id maps, Arc-wrapped fds/hooks/environment).
- `reference-z3-header-build-workaround` — Z3_SYS_Z3_HEADER + copy-to-angr/
  workaround for venv builds.

### Build note

Same as last session: `pip install -e .` broken in venv. Workaround:

    Z3_SYS_Z3_HEADER=/usr/include/z3.h cargo build --manifest-path \
        native/angr/Cargo.toml --release --lib
    cp -f target/release/librustylib.so \
        angr/rustylib.cpython-312-x86_64-linux-gnu.so

The run_single.py harness needs `PYTHONPATH=/home/ubuntu/repos/angr`
when invoked from a multiprocessing-spawn context.

## Suggested next slices

- Continue FxHasher swap on remaining candidates:
  - `engine.rs:187` PredictiveEngine.symbolic_registers (HashMap<u32, RustBV>)
  - `state.rs:167/620/653` Arc-wrapped fds/hooks/environment maps
  - RustExplorationManager internal state-id maps (exploration/mod.rs)
- `angr-6n56` (P2) — Arc-tree teardown for transient RustBV results,
  36% of symbolic arithmetic time. Bigger refactor (likely needs an
  arena).
- `angr-pufm` (P1) lazy guarded-entries (still open, needs split per
  the audit note).
