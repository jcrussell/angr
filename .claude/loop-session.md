# Loop session notes (2026-05-05, forty-sixth loop session)

## Task: angr-jdz9 — DONE
Add test that wide symbolic load whose two pinned address solutions
each straddle a 4 KiB page boundary returns the correct LE
concatenation under each pinned addr.

## What I added
`native/angr/src/memory.rs` (mod tests) —
`test_symbolic_load_cross_page_multiple_solutions`.

Map pages 0x1000/0x2000/0x3000 RWX, store distinct concrete dwords at
each boundary (offsets 0xFFC and 0x000). Build a symbolic 64-bit `addr`
with `(addr == 0x1FFC) | (addr == 0x2FFC)` asserted on the SymContext.
`load_symbolic_unified(addr, 8, ctx, concretizer)` is called; the
result is then evaluated under forked contexts pinning addr to each
solution. Each pinned eval must reproduce the LE concatenation of the
4 bytes from the lower page + 4 bytes from the higher page; eval_upto
is also asserted to enumerate both possible 8-byte values.

## Result
Test PASSES. Locks in current correct behaviour. The bead acceptance
criterion was "passes (locks in correct behaviour) or fails (drives a
fix bead)" — first branch.

Path under the hood: two solutions with stride 0x1000 trip the Strided
concretization branch (NOT Multiple) in
`load_symbolic_unified` → `load_strided_balanced` →
`build_strided_ite_tree` → per-leaf `load_concrete_lazy`, which
handles cross-page concrete loads correctly.

## Verification
- 391/391 Rust unit tests pass (`cargo test --release --lib`).
- 214/214 pytest passing in `tests/engines/test_rust_exploration.py`.

## Files modified
- `native/angr/src/memory.rs` — added one new test in mod tests {}.

## Beads / memory
- angr-jdz9 claimed → closed.
- `avoid-rust-mgr-extra-pages-symbolic-skip` saved: the Python-level
  `RustExplorationManager` integration path skips user-stored pages
  whose symbolic_bitmap still has untouched bits, so Rust unit-test
  against SymbolicMemory directly is the right level for this kind of
  test.
- `invariant-two-solution-stride-strided-branch` saved: two pinned
  address solutions with stride == page size go through the Strided
  ITE branch, not Multiple.

## False starts (kept here for context, not in bd)
- First wrote a Python integration test through RustExplorationManager
  with `mov rax, [rdi]; ret`; rax came back 0 because pages 0x2000 and
  0x3000 (manually populated via state.memory.store) were detected as
  "symbolic pages" by `_find_user_symbolic_pages` (symbolic_bitmap
  stays True for untouched bytes) and skipped during sync. Switched to
  a Rust unit test, which hits the right code paths and is much
  faster.
