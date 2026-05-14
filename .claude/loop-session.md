## Session log: 2026-05-14

### Completed this session

1. **angr-hzs0** (NEON Dup/Widen/Narrow ops) — CLOSED in commit 7630b1d66.
   12 unit tests + 2 Z3 symbolic tests covering VDup/VWiden/VNarrowUn/
   VNarrowBin/VQNarrowUn/VQNarrowBin. Updated test_neon_unimplemented_routing
   and test_neon_does_not_shadow_existing_mappings.
   Memories: invariant-neon-routing-tests, vex-qnarrow-naming.

2. **angr-7xms** (Native syscall ABI dispatchers) — CLOSED in commit 33dbc05fa.
   Registered existing arch-agnostic handlers (read/write/exit/exit_group/
   brk/mprotect/munmap/gettimeofday/time/clock_gettime/rt_sigaction; ARM64
   gets mmap) for X86, ARM, ARM64, MIPS32. Added Cdecl::syscall_arg_registers
   override (EBX/ECX/EDX/ESI/EDI/EBP) and ARMEABI::syscall_arg_registers
   override (R0-R5). 5 per-arch dispatch tests + 2 CC tests.
   Memories: syscall-cc-override-needed, syscall-handlers-arch-agnostic.

### Tests / Build state at session end
- Rust: 739/739 passing (full suite, with vex-engine-z3).
- Python: 395/395 passing (test_rust_exploration.py).
- Pre-existing flaky tests: syscalls::brk::tests::fork_preserves_posix_brk,
  syscalls::mmap::tests::fork_preserves_mmap_base. Both fail on HEAD with
  pyo3 Python interpreter init errors when run individually — order-dependent.
  Not introduced this session.

### Next picks
- angr-g9hy (MIPS32 $t0 symbolic accumulation bug) — non-deterministic;
  investigate stepping.rs symbolic register fork/cache path. Workaround
  documented; bench unblocked. Likely needs careful repro setup.
- angr-myty (perf dashboard) — substantial CI/Pages work.
- angr-pogf (lazy memory design) — research/design task.
