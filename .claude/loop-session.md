# Loop session notes (2026-05-07, 104th loop session)

## Task: angr-ar8r — HashMap clone audit in RustSimState::fork (CLOSED)

Bead closed (commit 76aef1ca2).

### What changed

`native/angr/src/state.rs`:

- Wrapped `FileSystem.fds: HashMap<u32, FileDescriptor>` in
  `Arc<HashMap<...>>`. Mutating methods (open / open_with_content / close /
  write / read / seek) call `Arc::make_mut` lazily; close/read/seek peek
  the read path first to skip CoW when the op would be a no-op.
- Wrapped `RustSimState.hooks: HashSet<u64>` in `Arc<HashSet<u64>>`.
  add_hook/remove_hook/clear_hooks use Arc::make_mut. clear_hooks skips
  the clone when already empty.
- Wrapped `RustSimState.environment: HashMap<Vec<u8>, Vec<u8>>` in
  `Arc<HashMap<...>>`. setenv uses Arc::make_mut.

### Measured impact (state_fork criterion bench)

```
state_fork: 338.86 ns -> 283.80 ns (-16.4%, p < 0.05)
```

Other fork-related benches unchanged (symcontext_fork_scaling 215 ns at
all sizes — still O(1) post angr-w6nq). 254 Python tests + 517 Rust unit
tests + 11/12 regression benchmarks pass (csgames2018 was already
failing on master pre-change with "list index out of range" — verified
by stashing the change and reproducing).

### Memories saved

- `benchmark-state-fork-arc-wrap` — bench numbers + still-extant clone
  hot spots (RegisterFile fields, HeapMetadata, SymbolicMemory internals).
- `invariant-arc-make-mut-cow` — pattern for read-mostly RustSimState
  fields: wrap in Arc, peek read path first in mutators to skip CoW for
  no-op calls. Avoid Arc-wrap for fields mutated on every fork.
- `avoid-csgames2018-as-regression-signal` — csgames2018 is broken on
  master, not a regression signal.

### Build note

Same as last session: `pip install -e .` is broken in venv. Workaround:

    Z3_SYS_Z3_HEADER=/usr/include/z3.h cargo build --manifest-path \
        native/angr/Cargo.toml --release --lib
    cp target/release/librustylib.so \
        angr/rustylib.cpython-312-x86_64-linux-gnu.so

(The Python z3 package install is missing headers, so we point the
build at the system z3 headers via the env var. The .cargo/config.toml
rule honors Z3_SYS_Z3_HEADER if set.)

## Suggested next slices

- `angr-6n56` (Arc-tree teardown for transient RustBV results) — 36% of
  symbolic arithmetic time. Bigger refactor (likely needs an arena).
- `angr-pufm` lazy guarded-entries (still open).
- Smaller fork-time wins still on the table:
  - `RegisterFile.data: Vec<u8>` (2.51% in fork) — but mutated every
    register write, so Arc::make_mut might break even.
  - `RegisterFile.symbolic: HashMap<u32, RustBV>` (2.48%) — mutated
    every symbolic write.
  - `HeapMetadata` (1%) — mutated every malloc/free.
  - `SymbolicMemory` internals (HashMap 2.70%, HashSet 1.71%).
