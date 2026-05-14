## Session log: 2026-05-14 — angr-n082 CLOSED (Phase 1.2 Multi-cell collapse)

### Closed task

**angr-n082** — Phase 1.2: Multi-cell collapse in load_concrete_lazy_inner.
Commit `5030e0989` ("feat(rust-symex): Multi-cell collapse in
load_concrete_lazy_inner (angr-n082)").

### What landed

- `native/angr/src/memory/load.rs`:
  * `load_concrete_lazy_inner` now detects any Multi byte in
    `[addr, addr+size)` (cheap `multi_objects.is_empty()` short-circuit
    in the common case). On hit, runs `check_perms_range` and
    dispatches to the new helper.
  * `assemble_load_with_multi` walks each byte:
    - Multi byte: right-fold over `payload.alternatives()` building an
      ITE chain with the page's concrete byte as the final `else`.
      Calls `record_mem_ite_depth(payload.len())` per memory
      `invariant-mem-ite-depth-counter`.
    - Plain Symbolic byte: extract via symbolic_objects or
      symbolic_spans (mirrors `try_byte_merge_load`).
    - Concrete byte: 8-bit `RustBV::concrete`.
    - Concatenate endianness-correctly: LE folds high→low, BE folds
      low→high.
- `native/angr/src/memory/tests.rs`:
  * 4 new tests (`test_multi_cell_load_*`): single-byte LE, 4-byte
    mixed LE, 4-byte mixed BE, two Multi bytes in one load.

### Validation

- `cargo test --release --lib`: 751 / 751 passing.
- `pytest tests/engines/test_rust_exploration.py`: 396 / 396 passing.

### Memories saved this session

- `invariant-multi-load-collapse` — right-fold construction details,
  page-byte default else, endianness concat pattern.
- `invariant-multi-load-not-in-load_concrete` — load_concrete (the
  sibling fn at line 37) was NOT patched; Phase 1.3+ will need a
  parallel detection block there once production stores emit Multi.

### Suggested next work

- `bd ready` for follow-on tasks. Phase 1.3 (`angr-aija` — store
  helpers) and Phase 1.4 (`angr-5zw8` — strchr wiring) both block on
  this commit landing.
