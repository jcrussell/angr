## Session log: 2026-05-13 — angr-7vcx (sokohashv2 fails on Rust engine) — IN PROGRESS (claimed, not closed)

### Status

Bug investigation made significant progress but is NOT a quick fix.
Issue remains claimed (not closed). Detailed findings below for the next session.

### What I confirmed

1. **ef020d101 (transcendentals) is NOT the root cause.** Disabling the
   transcendentals fast paths (returning Err for IROp::Raw in both
   binop and binop_with_rm) reproduces the SAME failure. So the
   regression is elsewhere.

2. **The state DOES traverse the correct path.** Callback log shows
   exactly 9 user hooks visited in order:
   0x401028 → 0x40102c → 0x401033 → 0x40103e → 0x401215 → 0x40121e →
   0x401225 → 0x401239 → 0x40123c → 0x40123e (find).

3. **Register state is correct.** ESP at 0x40103E = 0x7ffefff0; at
   0x401215 = 0x7ffeffec. The delta matches the expected execution
   path: block 0x401043 starts with `add esp, 4` (canceling the push
   at 0x401039), then no more stack ops until block 0x401203 does 2
   pushes. ESI changes from 0x7fff0080 to 0x500008 — consistent with
   `mov esi, dword ptr [0x413008]` at 0x401203 reading the import
   stub address.

4. **Memory writes are LOST.** State at 'found' has page 0x421000
   (the BSS page containing 0x4216C0) in its snapshot, but the bytes
   are all zeros. Block 0x4011AB is supposed to write the hash via
   `mov dword ptr [eax], ...` with eax=0x4216C0. The writes don't
   make it to the exported page.

5. **Minimal reproducer (different bug?):** load_shellcode with
   `mov [0x4216c0], 0x12345678` shows that for a shellcode binary
   (where page 0x421000 is NOT in the binary's image), writes via
   mov-immediate to absolute address don't propagate. But this is a
   DIFFERENT scenario from sokohashv2 (where page 0x421000 IS in the
   PE binary's image).

### Stats anomaly

`steps: 0, rust_step_count: 9, rust_blocks_executed: 0`. The 0
blocks is a **stats bug** unrelated to the correctness bug:
`run_until_event` discards `blocks_executed` on early-return paths
(SimProcedure/Hook), only accumulating to global stats on MaxBlocks.
See native/angr/src/interpreter_cb/execution.rs:316. Not the cause.

### Hypothesis for next session

The hash routine block 0x4011AB has a tight inner loop:
```
0x4011ba: lea ebx, [ebx]              ; setup
0x4011c0: mov dword ptr [eax], 0       ; clear lo
0x4011c8: mov dword ptr [eax + 4], 0   ; clear hi
0x4011cf: nop
0x4011d0: movzx edx, word ptr [ebp + esi*2 - 0x48]
0x4011d5: movzx ecx, word ptr [edi]
0x4011d8: imul edx, ecx
0x4011db: and edx, 0xffff
0x4011e1: add dword ptr [eax], edx     ; accumulate hash
0x4011e3: adc dword ptr [eax + 4], 0
0x4011e7: inc esi
0x4011e8: add edi, 2
0x4011eb: cmp esi, 0x10
0x4011ee: jl 0x4011d0
```

Suspect: maybe `add dword ptr [eax], edx` (read-modify-write) is
being handled incorrectly — possibly the load reads from one memory
layer but the store goes to another, or the loop is unrolled in a
way that skips iterations.

### Next steps

1. **Add a unit test** in tests/engines/test_rust_exploration.py
   exercising `mov [0x4216c0], 0x12345678` (using a PE-like binary
   with that page mapped) — establish whether this is sokohashv2-
   specific or a general bug.

2. **Bisect** against cafc8e702 (last known good HEAD, 2026-05-09).
   Likely suspects from git log since then:
   - 84ab8fad3 skip ultrapage walk when no symbolic bytes
   - 86ee3dfa8 KISS split _extract_symbolic_pages 
   - 2622dcd43 NativeRead/NativeWrite re-enablement
   - 1a233418b fast-path _concretize_stack_registers
   - 974e2daeb ret-with-empty-call-stack to unconstrained

3. **Add Rust log::debug at MemoryStore** in interpreter_cb to see
   whether the stores at 0x4216C0 actually execute. Run with
   ANGR_RUST_LOG=debug. If stores happen but page bytes are still
   zero, the issue is in the export path. If no store events at
   0x4216C0, the block isn't running properly.

### Files touched

- (none committed). Restored ops.rs after temporary disable.

### Memories saved

- `rust-memory-sync-bug-2026-05-13` — high-level finding
- `sokohashv2-deep-investigation-2026-05-13` — deep findings

### bd state

- angr-7vcx: in_progress, claimed by me, notes updated with findings
