# Loop session notes (2026-05-07, 107th loop session)

## Task: angr-07rg — FxHasher swap on CallbackInterpreter per-step HashMaps (CLOSED)

Bead closed (commit 76c76c1b3).

### What changed

Continued the invariant-fxhash-internal-keys pattern on 7 u64-keyed std
HashMaps inside `CallbackInterpreter` (interpreter_cb/mod.rs) plus
`PendingStoreBuffer.byte_index` (pending_store.rs). The signatures that
cross between interpreter_cb, exploration/mod.rs (PendingCallback), and
exploration/stepping.rs were updated together so the FxHashMap type
flows end-to-end.

| Field                                                         | Before                | After |
|---------------------------------------------------------------|-----------------------|---------|
| CallbackInterpreter.all_flushed_stores                        | HashMap<u64, Vec<u8>> | FxHashMap |
| CallbackInterpreter.all_flushed_symbolic_stores               | HashMap<u64, RustBV>  | FxHashMap |
| CallbackInterpreter.pending_symbolic_stores                   | HashMap<u64, RustBV>  | FxHashMap |
| CallbackInterpreter.load_prefetch_cache                       | HashMap<(u64,usize),..>| FxHashMap |
| CallbackInterpreter.stored_conditions                         | HashMap<u64, RustBV>  | FxHashMap |
| CallbackInterpreter.fork_snapshots                            | HashMap<u64, BranchSnapshot> | FxHashMap |
| CallbackInterpreter.concretize_cache                          | HashMap<u64, Arc<...>>| FxHashMap |
| PendingStoreBuffer.byte_index                                 | HashMap<u64, usize>   | FxHashMap |
| PendingCallback.{stored_conditions,fork_snapshots}            | HashMap<u64, ...>     | FxHashMap |

`take_stored_conditions` / `take_fork_snapshots` now return FxHashMap;
five stepping.rs helper signatures and `process_deferred_forks_into`
were updated to match.

### Measured impact (criterion --baseline before)

```
state_fork                   : 204.59 ns -> 199.66 ns (-2.1%, p<0.05)
memory_concrete/store_8b     :  33.04 ns ->  33.12 ns (no change)
memory_concrete/load_1b      :  21.22 ns ->  21.05 ns (-0.6%, noise)
memory_fork_16pages          :  39.97 ns ->  39.22 ns (-2.0%, recovers
                                                       last session +4.3%
                                                       drift)
memory_symbolic_load_16range :   1.66 ms ->   1.69 ms (+2.2%, 1.7ms test
                                                       on heavy Z3 work,
                                                       not on the FxHash
                                                       path; noise)
```

state_fork's smaller win this round is expected: state.fork() doesn't
clone the interpreter — these maps live inside CallbackInterpreter and
only matter on per-step insert/lookup, which we don't have a microbench
for. The fork bench picks up only the secondary effect of FxHashMap's
smaller default capacity on the rest of the workload.

### Tests

- 254/254 Python tests pass (tests/engines/test_rust_exploration.py)
- 517/517 Rust unit tests pass
- 11/12 regression benchmarks pass; csgames2018 was already broken on
  master per `avoid-csgames2018-as-regression-signal`.

### Memories saved

- `benchmark-fxhash-interpreter-cb` — bench numbers + reasoning for the
  per-step interpreter swap.
- `invariant-fxhash-cross-module-signatures` — when changing internal
  Rust HashMap signatures across the stepping.rs / interpreter_cb /
  exploration/mod.rs boundary, change them all in one pass; the
  signatures flow end-to-end and skipping any one site triggers ~11
  E0308 mismatches at call sites.
- Updated `invariant-fxhash-internal-keys` with new completed targets
  and remaining candidates.

### Build note

Same pattern as last sessions: `pip install -e .` is broken in venv.
Working command:

    Z3_SYS_Z3_HEADER=/usr/include/z3.h cargo build --manifest-path \
        native/angr/Cargo.toml --release --lib
    cp -f target/release/librustylib.so \
        angr/rustylib.cpython-312-x86_64-linux-gnu.so

The run_single.py harness needs `PYTHONPATH=/home/ubuntu/repos/angr`
when invoked from a multiprocessing-spawn context.

## Suggested next slices

- Continue FxHasher swap on remaining candidates from
  `invariant-fxhash-internal-keys`:
  - `engine.rs:187` RustVEXEngine.symbolic_registers (HashMap<u32, RustBV>)
    — only matters on `engine.fork()`, not `state.fork()`. Smaller win
    but cheap to do.
  - exploration/mod.rs RustExplorationManager id maps
    (state→id maps, vex_fallback_addrs, simprocedures).
- `angr-6n56` (P2) — Arc-tree teardown for transient RustBV results,
  36% of symbolic arithmetic time. Bigger refactor (likely needs an
  arena).
- `angr-pufm` (P1) lazy guarded-entries — still open; the audit notes
  recommend splitting into a fallback-strategy enum and a separate
  lazy guarded-entries bead before claiming. Either could become a
  session task once split.
