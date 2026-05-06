# Loop session notes (2026-05-06, seventy-ninth loop session — DONE)

## Status: COMPLETE — angr-imy1 closed

## Task: angr-imy1 (P2) — Native syscall coverage (partial)

Delivered the dispatch infrastructure plus exit/exit_group on amd64.
Remaining syscalls (read, write, brk, mmap, mprotect) split into
follow-up beads (each warrants its own; they interact with broader
state).

### What landed

New module `native/angr/src/syscalls/` with:
- `mod.rs`: `NativeSyscallRegistry` (HashMap keyed by `(arch, num)`),
  `NativeSyscall` trait, `SyscallOutcome::{Continue { ret }, Exit}`.
- `exit.rs`: `NativeExitSyscall` — returns `Exit`, dispatcher routes to
  `STASH_DEADENDED`. Mirrors libc.exit (NO_RET) semantics; doesn't
  bother extracting the exit code, matching angr Python.

Wiring:
- `lib.rs`: `pub mod syscalls`.
- `exploration/mod.rs`: `native_syscalls: NativeSyscallRegistry` field
  on `RustExplorationManager`, initialized via `NativeSyscallRegistry::new()`.
- `exploration/stepping.rs`: `RunResult::Syscall` arm tries native
  dispatch before creating `PendingCallback`. Continue → set return
  register + process deferred forks; Exit → push to deadended +
  process deferred forks.

### Verification
- pytest: 243/243 pass (8.46s)
- cargo test: 467/467 pass (was 463; +4 syscall unit tests)
- Benchmarks: fauxware 0.38s, defcamp_r100 0.23s, ais3_crackme 0.87s
  (all within baseline)

### Commit
- eb4f5f8f3 feat(syscalls): native exit/exit_group dispatch (angr-imy1)

### Memories saved
- `invariant-native-syscall-dispatch` — registry/dispatch shape
- `invariant-amd64-syscall-abi` — r10 vs rcx for 4th arg; existing
  extract_procedure_args is ≤3-arg-safe only
- `invariant-native-syscall-pc-contract` — PC already advanced by the
  time Syscall arm fires; handlers only set the return register

### Follow-up beads created
- angr-lrdr (P3) Native amd64 brk syscall handler
- angr-uzla (P3) Native amd64 mprotect syscall handler
- angr-vybt (P4) Native amd64 mmap/munmap syscall handlers
- angr-0z34 (P4) Native amd64 read/write syscall handlers

### Build environment notes (carry-over)
- `Z3_SYS_Z3_HEADER=/usr/include/z3.h` for cargo
- `cp -f target/release/librustylib.so angr/rustylib.cpython-312-x86_64-linux-gnu.so`
- pytest needs `PYTHONPATH=.`
- run_single.py needs `PYTHONPATH=/home/ubuntu/repos/angr` (subprocess)
- pip install path is broken (resolvelib import error); use cargo +
  cp .so path

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
- angr-lrdr (P3) Native amd64 brk syscall handler [new]
- angr-uzla (P3) Native amd64 mprotect syscall handler [new]
