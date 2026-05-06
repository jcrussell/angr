# Loop session notes (2026-05-06, eightieth loop session — DONE)

## Status: COMPLETE — angr-uzla closed

## Task: angr-uzla (P3) — Native amd64 mprotect syscall handler

### What landed
- `native/angr/src/syscalls/mprotect.rs` — `NativeMprotectSyscall`
  mirrors Python `procedures/linux_kernel/mprotect.py`:
  - 3 concrete args (addr, length, prot); symbolic falls back to Python.
  - addr & 0xFFF != 0 → return -1 (misaligned).
  - any unmapped page in [addr, page_end) → return -1.
  - else apply `prot & 7` to every covered page → return 0.
  - zero length → return 0 no-op.
- `native/angr/src/memory.rs` — added `page_permissions(page_num)` and
  `set_page_permissions(page_num, perm)` helpers on `SymbolicMemory`
  (page_num is `addr >> 12`).
- `native/angr/src/syscalls/mod.rs`:
  - registered amd64 syscall 10 → mprotect.
  - `SyscallOutcome` now derives `Debug` so `expect_err` works in tests.
  - default-registry test asserts mprotect (10) is present.

### Encoding gotcha (saved as memory)
Linux PROT bits (READ=0x1, WRITE=0x2, EXEC=0x4) and Rust
`Permission::from_bits` (read=0x4, write=0x2, execute=0x1) are
REVERSED on bits 0x1/0x4. mprotect explicitly translates rather
than going through `from_bits`.

### Verification
- cargo test (lib, syscalls): 13/13 (8 new mprotect tests)
- pytest tests/engines/test_rust_exploration.py: 243/243 (8.29s)
- Baselines unchanged: fauxware 0.38s, defcamp_r100 0.23s,
  ais3_crackme 0.87s.

### Commit
- 63cad7f93 feat(syscalls): native amd64 mprotect dispatch (angr-uzla)

### Memories saved/updated
- `invariant-linux-prot-vs-rust-permission` (NEW) — bit-encoding mismatch.
- `invariant-symbolic-memory-page-helpers` (NEW) — page_permissions /
  set_page_permissions accessors.
- `invariant-native-syscall-dispatch` (UPDATED) — now lists mprotect (10)
  and notes SyscallOutcome derives Debug.

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
- angr-lrdr (P3) Native amd64 brk syscall handler
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
