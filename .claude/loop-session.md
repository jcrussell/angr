## Session log: 2026-05-14 — angr-ctct partial fix (extract_symbolic_pages misses filler-materialised values)

### Status: partial fix shipped

### Outcome
Identified the root cause of angr-ctct sokohashv2 bug: `_extract_symbolic_pages`
walked `all_bytes_changed_in_history()`, which only sees bytes touched by
`store()`. Values materialised by the `SYMBOL_FILL_UNCONSTRAINED_MEMORY` filler
at load time (e.g. `init.memory.load(addr, 8)` in solve.py to capture symbolic
input variables) live in the UltraPage's `symbolic_data` SortedDict but never
get added to changed-history. As a result, the symbolic input vars at
0x7fff0080..0x7fff009F were NEVER cached, never imported to Rust, and never
restored on the Python callback path. The do_repmovsd hook read 32 CONCRETE
witnesses from [esi] (the source area), stored them concretely to [edi], and
the hash routine then operated on zeros.

### Fix shipped
In `_extract_from_ultrapage` (angr/exploration/rust_state_sync.py): when
`all_bytes_changed_in_history` returns no segments but `symbolic_data` is
small (≤ 64 entries), walk the SortedDict directly. This covers user-seeded
filler-materialised symbols at init time without burning per-state cost when
the dict is full of runtime filler entries.

### Verified
- All 394 unit tests pass.
- Benchmark suite has same 10 pre-existing regressions; my fix only adds
  ~3% to defcamp_r100 (within noise).
- Sokohashv2 now fails differently: was AssertionError (hash mismatch),
  now IndexError (no found state) — exposes a downstream bug, not the
  same root cause.

### Regression test added
`test_hook_copies_symbolic_memory_preserves_symbolicity` in
TestExplorationIntegration — proves that a hook copying filler-symbolic
memory preserves symbolic identity at the destination.
