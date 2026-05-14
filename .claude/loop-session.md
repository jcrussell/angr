## Session log: 2026-05-14 — angr-7vcx (sokohashv2) partial fix + angr-ctct spawned

### Outcome

- **angr-7vcx CLOSED** with partial fix in commit `293aa8163`.
- **angr-ctct CREATED** for the remaining sokohashv2-specific root cause.
- 393/393 tests pass.
- sokohashv2 still fails on the Rust engine (separate bug now tracked).

### What I shipped (commit 293aa8163)

Two distinct memory-sync bugs in `angr/exploration/rust_state_sync.py`,
both surfacing as "Rust never sees my user-mapped page":

1. `_find_user_symbolic_pages` fallback used `any(symbolic_bitmap)` —
   UltraPage initialises that bitmap to all-ones for a fresh
   `map_region`, so every user-mapped page got flagged "symbolic" and
   then skipped by both `_overlay_python_state_pages` and
   `_sync_extra_python_pages`. Fixed by scanning
   `UltraPage.symbolic_data` (the explicit-symbolic-stores dict)
   instead.

2. `_sync_extra_python_pages` dropped all-zero pages via
   `any(concrete)`. A user `map_region(addr, ...)` with no follow-up
   store has zero bytes; the filter de-maps it on the Rust side, so
   the first Rust store hits `MemoryError::Unmapped` and is silently
   discarded in `flush_stores`' `let _ = ...`. Fixed by dropping the
   `any(concrete)` check.

Added `TestRustConcreteMemoryStoreRoundTrip` exercising
`mov [0x4216c0], 0x12345678` against a `load_shellcode` binary with
`state.memory.map_region(0x421000, ...)`. Pre-fix: read returns 0.
Post-fix: 0x12345678 round-trips.

### What I confirmed about sokohashv2 (the deeper bug)

Used a temporary Rust `log::warn!` at the store/load dispatch path to
trace what happens during a full sokohashv2 run.

**Loads from the stack region 0x7FFF002C..0x7FFF004A return CONCRETE
witnesses** (e.g. `0x7fff002c -> 0xa002d concrete`). Those addresses
are where `do_repmovsd` (hook at `0x401028`) was supposed to land
symbolic copies of the user variables. So the symbolic bytes are
being lost between the hook's `state.memory.store(edi, buf)` and
the inline hash routine's load.

**Stores at 0x4216c0/0x4216c4 all fire `sym=false val=0x0`** —
correctly, given the operands are already concrete by the time they
reach the store. Not a store bug; a load bug.

The dispatch path (`_handle_callback_with_successors` in
`angr/exploration/rust_callback_dispatch.py`) does the right shape:
1. `CallbackMemoryTracker` records (base_addr, big_ast) in
   `tracked_symbolic_writes` AND (concrete_addr, byte_data) in
   `tracked_writes` for each Python state.memory.store.
2. `_extract_memory_changes` yields byte-by-byte
   `('symbolic', byte_addr, 1, data, handle_id, byte_val)` for small
   symbolic regions.
3. `all_symbolic_addrs` = {32 byte addrs} ∪ {tracker base}.
4. `mem_changes` filter drops byte-level entries.
5. `resume_after_simprocedure` applies (filtered) `mem_changes`.
6. `import_symbolic_to_state` runs for each (addr, ast) in
   `all_sym_imports`.

Steps look correct on paper. But the Rust state ends up with
concrete bytes. The mismatch is the open question.

### Hypothesis for next session (angr-ctct)

Likely paths:
- `import_symbolic_value` in `native/angr/src/memory/symbolic_objects.rs:54`
  collides between the 32 byte-level imports and the tracker's
  256-bit base import. Last-write-wins on the `symbolic_objects`
  map; the spans for the big AST cover bytes that ALSO have their
  own symbolic_objects entries, so `load_symbolic_unified` may
  pick the wrong one.
- The `concrete_changes` in `_extract_memory_changes` may emit
  AFTER the symbolic at the same addr (race in the iteration
  order). Check `_emit_memory_region` line 1438-onwards in
  `rust_state_sync.py`.
- Could be the `_handle_callback_no_successors` mirror path
  (line ~1133 in `rust_callback_dispatch.py`) — verify which one
  fires for `do_nothing`/`do_repmovsd` hooks.

### Files touched / state

- Committed (293aa8163): `angr/exploration/rust_state_sync.py`,
  `tests/engines/test_rust_exploration.py`
- Reverted: temporary `log::warn!` in
  `native/angr/src/interpreter_cb/{statements,expressions}.rs`
- Working tree: clean (only `.venv/` untracked)

### Memories saved

- `invariant-ultrapage-symbolic-bitmap` — pitfall: don't conflate
  bitmap with "has symbolic data"
- `sokohashv2-deep-investigation-2026-05-14` — full trace of what
  happens during the sokohashv2 hook flow

### bd state

- angr-7vcx: CLOSED with partial-fix reason
- angr-ctct: open, captures the remaining sokohashv2 root-cause hunt
