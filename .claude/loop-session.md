# Loop session notes (2026-05-06, eighty-first loop session — DONE)

## Status: COMPLETE — angr-lrdr closed

## Task: angr-lrdr (P3) — Native amd64 brk syscall handler

### What landed
- `native/angr/src/syscalls/brk.rs` — `NativeBrkSyscall` mirroring
  `procedures/linux_kernel/brk.py` (state.posix.set_brk semantics).
- `native/angr/src/state.rs`:
  - new `posix_brk: u64` field (default 0x1B00000), distinct from
    `heap_brk` (the malloc bump allocator at 0xC0000000).
  - threaded through all 6 RustSimState struct literals
    (3 ctors + fork + fork_true + fork_false + fork_from_snapshot + merge).
  - `posix_brk()` getter and `set_posix_brk(addr)` setter.
- `native/angr/src/syscalls/mod.rs`:
  - registered amd64 syscall 12 → brk.
  - default-registry test asserts brk (12) is present.

### Handler semantics
- Symbolic new_brk → Err(SymbolicArgument) → Python fallback.
- new_brk < current → Continue { ret: current } (no-op; covers brk(0)).
- Otherwise: set posix_brk = new_brk; if growing across page boundary,
  map new pages with `Permission::RWX`; return new_brk.
- Collision (any to-be-mapped page already mapped) → Err so Python's
  SimMemoryError-driven alternate-brk fixup runs (Rust's `map` is
  idempotent and can't surface that signal).

### Verification
- cargo test (lib, syscalls): 23/23 (10 new brk tests added).
- pytest tests/engines/test_rust_exploration.py: 243/243 (8.21s).
- fauxware smoke (rust engine): no regression in callback timings.

### Commit
- fad1b14d2 feat(syscalls): native amd64 brk dispatch (angr-lrdr)

### Memories saved/updated
- `invariant-posix-brk-vs-heap-brk` (NEW) — two distinct brk fields.
- `invariant-syscall-fallback-on-collision` (NEW) — pattern for Err
  return when Rust semantics can't reproduce Python's error paths.
- `invariant-native-syscall-dispatch` (UPDATED) — now lists brk (12).

## Next-up (still ready)
- angr-eygl (P1) Differential test harness — substantial infra
- angr-pufm (P1) Symbolic address concretization fallback — architectural
- angr-prem (P2) MemoryLayer trait
- angr-fk0m (P2) Unify rust_state_sync / cache / export mixins
- angr-fbl0 (P2) Native SimProcedure coverage gaps (audit needed)
- angr-3zs6 (P2) FallbackStrategy enum (depends on angr-m2hf)
- angr-nnov (P2) Fill out RustStateProxy
- angr-4j5u (P2) Decompose Rust-side RustExplorationManager
- angr-m2hf (P2) Unified error trait + single PyO3 conversion site
- angr-vybt (P4) Native amd64 mmap/munmap syscall handlers
- angr-0z34 (P4) Native amd64 read/write syscall handlers

### Build environment notes (carry-over)
- `Z3_SYS_Z3_HEADER=/usr/include/z3.h` for cargo
- `cp -f target/release/librustylib.so angr/rustylib.cpython-312-x86_64-linux-gnu.so`
  (note: `target/release/`, NOT `native/angr/target/release/` — workspace
  uses repo-root target dir)
- pytest needs `PYTHONPATH=.`
- run_single.py needs `PYTHONPATH=/home/ubuntu/repos/angr` (subprocess)
- pip install path is broken (resolvelib import error); use cargo + cp .so path
