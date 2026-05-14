## Session log: 2026-05-14 — angr-fv81 closed (sokohashv2 fully working)

### Status: fixed, committed pending

### Outcome
Identified and fixed the second sokohashv2 bug (the IndexError after angr-ctct).

### Root cause
`UltraPage.symbolic_data` is a SortedDict keyed by the START offset of each
symbolic region. A single `init.memory.load(addr, 8)` filler-materialised
value stores ONE dict entry covering 8 bytes (via `symbolic_bitmap`).

The angr-ctct fallback walked only `sd.keys()`, extracting just ONE byte per
region. Bytes 1..N of each region silently collapsed to concrete zero on the
Rust side.

For sokohashv2 this turned a 15-term hash AST (using all 16-bit halves of every
8-byte input) into a 4-term AST (low byte only). Explored states still reached
to_find but the hash conjunction with WIN_HASH was unsat, so
`solver.eval_upto` returned `[]` → IndexError on `eval` access.

### Fix shipped
`angr/exploration/rust_state_sync.py::_extract_from_ultrapage`: walk each
`symbolic_data` entry's contiguous symbolic_bitmap extent, not just the head
byte. Per-entry extent capped at 64 bytes to skip 4 KB page-fillers from
SYMBOL_FILL_UNCONSTRAINED_MEMORY (which would tank ais3_crackme /
google2016_unbreakable_0 by ~40% if extracted byte-by-byte).

### Verified
- sokohashv2 (Rust): IndexError → OK, 12.9s, correct solution
- 395 unit tests pass (added `test_filler_materialised_multibyte_symbolic_preserved`)
- Benchmark regression: my fix is performance-neutral; pre-existing regressions
  remain (unmapped_analysis, ais3_crackme, etc. — unrelated to this task)

### Memories saved
- sokohashv2-fv81-root-cause
- invariant-rust-state-satisfiable-no-extras
- avoid-unbounded-extraction-fallback
