# Loop session notes (2026-05-07, 94th loop session)

## Task: angr-0lre — Split memory/mod.rs into focused modules (CLOSED)

This session finished the split by extracting the load_*/store_*
families. Bead is now closed (commit 12d8caaa2).

### Extractions this session

- `memory/load.rs` (662 lines, new): load, load_concrete,
  load_symbolic, load_concrete_or_unconstrained,
  load_symbolic_unified, load_concrete_automap,
  load_concrete_automap_internal, load_concrete_lazy,
  load_concrete_lazy_inner, plus apply_pending_writes_concrete /
  apply_pending_writes_symbolic helpers.
- `memory/store.rs` (495 lines, new): store, store_concrete,
  store_symbolic, store_strided, store_symbolic_unified,
  store_with_concretization, store_conditional_multiple,
  store_concrete_lazy, store_concrete_automap,
  store_concrete_automap_internal.

`mod.rs` declarations updated to `mod load; mod store;`.

### Visibility tweaks

- `apply_pending_writes_concrete`, `apply_pending_writes_symbolic`,
  `load_concrete_lazy_inner` → `pub(super)` (called from siblings).
- `store_strided`, `store_conditional_multiple` → `pub(super)`
  (called from store.rs's own `store_with_concretization` /
  `store_symbolic_unified`).
- All other fields/methods unchanged. Private struct fields stay
  reachable from child modules under Rust's privacy rules — no pub
  needed for `pages`, `symbolic_objects`, `symbolic_spans`, etc.

### Final mod.rs contents (762 lines)

PendingWrite + MemoryError + SymbolicMemory struct/new, perms
helpers, map/unmap/is_mapped/map_data, fork+Clone, pending_writes
API (incl. flush_pending_writes ~60 lines), page mgmt + lazy
regions + map_page + auto_map_zero_page, merge.

### Module layout (final)

```
memory/
  mod.rs              762 lines  (was 2881 at start of bead)
  load.rs             662 lines  (NEW this session)
  store.rs            495 lines  (NEW this session)
  tests.rs            992 lines  (extracted previous session)
  page.rs             251 lines
  ite_builder.rs      219 lines
  symbolic_objects.rs 115 lines
  concretize_glue.rs   75 lines
```

### Tests / build

- `Z3_SYS_Z3_HEADER=/usr/include/z3.h cargo build --release` clean
- `cargo test --release --lib memory::` → 31 memory tests pass
- pytest tests/engines/test_rust_exploration.py → 243/243 pass

### Build env note (still applies)

`pip install -e .` is broken on this venv. Use:
`Z3_SYS_Z3_HEADER=/usr/include/z3.h cargo build --release` then
`cp target/release/librustylib.so angr/rustylib.cpython-312-x86_64-linux-gnu.so`.
NOTE: `target/` is at the repo root, NOT under `native/angr/target/`.

For pytest/benchmarks, prefix with `PYTHONPATH=/home/ubuntu/repos/angr` if
the editable install is missing from `.venv` site-packages.

## Bead state

`angr-0lre` CLOSED (commit 12d8caaa2).

bd memories written this session:
- `invariant-rust-child-mod-privacy` — child modules of M see private
  items of structs defined in M; pub(super) only needed across siblings.
- `memory-mod-layout-2026-05-07` — final layout snapshot.

## Suggested next slices

angr-0lre is done. If anyone wants to push further (not required by
acceptance criteria):
- `memory/pending_writes.rs` would catch flush_pending_writes (~60
  lines) and the small accessors (count/get/add/drain).
- `memory/merge.rs` would catch the merge() impl (~110 lines).

Both are pure cuts — no shared helpers between them and the rest of
mod.rs.
