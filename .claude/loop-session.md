## Session log: 2026-05-14 (Phase 1.1 — angr-me3z CLOSED)

### Task: angr-me3z — Phase 1.1: MultiPayload data structure + sidecar storage

CLOSED. Lands the data-structure foundation for lazy symbolic memory
(parent angr-czph). Load-side collapse and store helpers are follow-up
sub-beads (see "What this unblocks" below).

### Sub-bead split for angr-czph

Phase 1 was too large for a single session. Split into four sequential
sub-beads, dependency-wired so `bd ready` surfaces them in order:

- angr-me3z (this session)  — MultiPayload data structure       ✓ CLOSED
- angr-n082 — load_concrete_lazy_inner Multi-cell collapse      next
- angr-aija — store helpers + flush_pending_writes update
- angr-5zw8 — wire strchr SimProcedure (closes angr-czph)

### Files modified / added

- native/angr/src/memory/multi.rs        (NEW — MultiAlternative, MultiPayload, SymbolicMemory public API)
- native/angr/src/memory/mod.rs          (mod multi; multi_objects sidecar field; fork() clones it)
- native/angr/src/memory/page.rs         (multi_bitmap field; mark_multi/is_multi/clear_multi; store_concrete clears bit; fork() copies bitmap)
- native/angr/src/memory/tests.rs        (6 new Phase 1.1 tests)

### Architecture decision (logged as memory `lazy-memory-sidecar-architecture`)

Design doc said "Add Multi variant to byte-cell enum in memory/page.rs"
but page.rs has no byte enum — it stores concrete bytes + a separate
bitmap, with symbolic data in a SymbolicMemory sidecar map. Phase 1.1
mirrors that pattern instead of refactoring to an enum:

- `SymbolicMemory.multi_objects: FxHashMap<u64, MultiPayload>` —
  sidecar parallel to `symbolic_objects`.
- `MemoryPage.multi_bitmap` — parallel to `symbolic_bitmap`,
  Option<Box<[u64; BITMAP_WORDS]>> for zero-cost when unused.

Four valid cell states (memory `invariant-multi-vs-symbolic-cell-states`):
Concrete / Symbolic / Multi / inconsistent (bug). set_multi_alternatives
enforces "Multi supersedes Symbolic" by clearing the symbolic_objects /
symbolic_spans entries before installing.

### Counter contract

Per memory `invariant-mem-ite-depth-counter`,
set_multi_alternatives calls record_mem_ite_depth(payload.len()) on
every insertion. Empty payload is a no-op (clears the cell, does not
bump the counter). Verified by test_multi_payload_records_ite_depth.

### Validation

- `cargo check --release` clean
- `cargo test --lib memory::` 42/42 pass (36 prior + 6 new)
- `pytest tests/engines/test_rust_exploration.py` 396/396 pass

### Commit

ca509f925 feat(rust-symex): MultiPayload data structure for lazy symbolic memory (angr-me3z)

### Notes for the next session

- Pip is still broken (`ImportError: RequirementInformation from
  pip._vendor.resolvelib.structs`). Use `tools/rebuild-rust.sh
  --cargo-only --keep-cargo-cache` to rebuild the .so. The script
  works fine — it copies the .so into angr/ directly.
- Phase 1.2 (angr-n082) is next-ready. Scope: teach
  `load_concrete_lazy_inner` (memory/load.rs:517) to detect Multi
  cells via `page.is_multi(offset)`, look up the payload in
  `multi_objects`, and collapse via the existing balanced ITE builder
  in memory/ite_builder.rs. Per-byte. End-of-load record
  `record_mem_ite_depth(alternatives.len())` on each collapse.
- The Multi-cell entry-count counter on `MemoryPage` is currently per
  byte, not per cell. That is fine for the data structure but the
  baseline metric `mem_ite_depth_max` is process-global; Phase 0
  baseline numbers (sym-write max=2, total=16) come from
  `store_conditional_multiple` not Multi cells, so they remain valid
  baselines to compare against once Phase 1.4 wires the strchr path.
