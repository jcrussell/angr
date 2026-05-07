# Loop session notes (2026-05-07, 129th loop session)

## Task: angr-vybt — Native amd64 mmap/munmap syscall handlers

### Status: closed (partial — mmap split to angr-2jyk)

### What landed (commit 129facc20)

1. `CallingConvention::syscall_arg_registers()` — defaults to
   arg_registers(); SystemVAMD64 overrides to RDI/RSI/RDX/R10/R8/R9.
   Required for 4+ arg syscalls because Linux amd64 syscall ABI
   replaces RCX with R10 at arg 4.
2. `RustExplorationManager::extract_syscall_args()` — reads syscall
   registers; no stack fallback (Linux amd64 syscalls cap at 6 args).
3. `stepping.rs:153` syscall dispatch now calls extract_syscall_args
   (was extract_procedure_args, which would have routed RCX to arg 4
   for any future 4+ arg syscall — silent miscompile risk).
4. `NativeMunmapSyscall` at AMD64/11 — mirrors Python's no-op `return 0`.
5. Tests: 3 new in syscalls::munmap, 2 new in calling_conventions.

### Tests
- 29 passed in syscalls::* (cargo)
- 6 passed in arch::calling_conventions::tests (cargo)
- 261 passed in tests/engines/test_rust_exploration.py (Python)
- fauxware sanity: OK rust 0.36s — no regression

### Why mmap was split off (now angr-2jyk)
- mmap needs `state.heap.mmap_base` mirrored into RustSimState
  (default 0xC1000000 = 0xC0000000 + 0x00800000*2).
- Cross-engine sync risk: when mmap falls back to Python (file-backed,
  symbolic args, MAP_FIXED collision), Python reads heap.mmap_base.
  Rust's mirror must propagate back, or Python allocates at a stale
  base and collides with Rust's region. Same problem class as
  angr-0z34 (read/write fd-state sync).
- Without mmap_base, the addr=0 (kernel-chooses) path can't run
  natively — and that's the common case in real binaries.

### Memories saved
- invariant-syscall-arg-extraction (replaces obsolete invariant-amd64-syscall-abi)
- avoid-pip-install-editable-broken-venv (env-specific build workaround)

### Build env gotcha
- `.venv/lib/python3.12/site-packages/setuptools` is corrupted
  (version "0.dev0+unknown", missing Lorem ipsum.txt). pip install -e
  fails with "invalid command 'dist_info'". Worked around by
  `cargo build --release` + cp librustylib.so → angr/rustylib.cpython-*.so,
  and using PYTHONPATH=/home/ubuntu/repos/angr for test invocation.

## Status: complete
