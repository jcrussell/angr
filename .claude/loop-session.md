# Loop session notes (2026-05-07, 130th loop session)

## Task: angr-2jyk — Native amd64 mmap syscall handler

### Status: closed (commit 7391b2426)

### What landed
1. RustSimState.mmap_base: u64 field (default 0xC100_0000 = heap_base
   0xC0000000 + heap_size 0x00800000 * 2). Mirrored in 3 constructors
   + 5 fork/merge sites alongside posix_brk. Accessors mmap_base() /
   set_mmap_base() in native/angr/src/state.rs.
2. NativeMmapSyscall at AMD64/9, mirroring procedures/posix/mmap.py.
   Native subset: concrete addr/length/prot/flags/fd/offset, anonymous
   (MAP_ANONYMOUS, fd[31:0]==-1), exactly one of MAP_SHARED/MAP_PRIVATE.
   addr=0 → allocate from mmap_base + bump page-aligned;
   addr!=0 → map at addr if range unmapped.
3. Bad-flags fast path returns -1 directly (matches Python).
4. Falls back (Err) for: symbolic any arg, file-backed, MAP_FIXED+
   collision, addr=0 collision (Python loops to find alt addr).
5. Tests: 20 new in syscalls::mmap (cargo).

### Why no cross-engine sync was added
- mmap_base advances are NOT propagated back to Python state.heap.mmap_base
  on fallback (same drift pattern as posix_brk).
- Acceptable while syscall fallbacks rarely interleave with successful
  native calls. Future cross-engine sync work (angr-0z34 area) should
  address both fields together at the syscall callback boundary.
- Documented inline in mmap.rs module comment + state.rs field comment.

### Tests
- 49 passed in syscalls::* (cargo, was 29)
- 261 passed in tests/engines/test_rust_exploration.py
- fauxware sanity: OK rust 0.36s — no regression

### Memories saved
- invariant-mmap-base-mirror — field/accessors/sync caveat
- invariant-mmap-syscall-semantics — fd[31:0] mask, bad-flags fast
  path, prot[2:0] mask

### Build env (still broken)
Same workaround as 129th session: pip install -e fails (broken
setuptools in .venv). Used `cargo build --release` + cp librustylib.so
→ angr/rustylib.cpython-312-x86_64-linux-gnu.so + PYTHONPATH for tests.

## Status: complete
