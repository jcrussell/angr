## Session log: 2026-05-17 — angr-kcf.1 BFS divergence triage

### Status: CLOSED — research task, no code changes. 444 tests still pass (no code touched).

### Task

**angr-kcf.1 (P2, task)** — Confirm or refute BFS path-divergence as the
root cause for codegate_2017-angrybird Rust vs Python divergence.
Deliverable: short writeup saying (a) BFS ordering is the cause and the
fix is in the manager queue, OR (b) something else.

### Conclusion: (b) BFS path-divergence is NOT the cause.

Empirical test: ran codegate_2017-angrybird Rust under both `--strategy
bfs` (FIFO pop_front, default) and `--strategy dfs` (LIFO pop_back). Both
produce byte-identical wrong output:
`b'*%\xac\x0c\`\xfe\xff\x80\x06 \xc0\xff\xff\xca4\x07\xffpK\x05'`.
Python (correct): `b'Im_so_cute&pretty_:)'`.

If exploration order were the cause, swapping to DFS should reach
find_addr (0x404fab) via a different path through the binary and
produce different stdin — it doesn't. Byte-identical bytes → same
found state → same path under both orderings.

### Where the real bug lives

Aligns with the parent task angr-kcf's stated root cause:
> The Rust engine doesn't track symbolic stdin BVS variables created
> during entry_state(). posix.dumps(0) returns b'' because _stdin_vars
> is never populated.

The 250-block Rust history that the found state carries is dominated by
extern PLT addresses (0x700010 puts ×250, 0x700030 fgets ×1 — see
"history-granularity" finding below). Stdin BVS variables created in
entry_state are still in their default symbolic form when find_addr is
reached. Python (correct) accumulates 382 constraints on the found
state; Rust shows 0 via `state.solver.constraints` (the Rust solver
might have more, but the Python view is empty — and the eval call to
the Rust solver via the proxy fallback produces garbage bytes, meaning
the underlying constraints don't pin stdin).

### Additional finding: Rust history granularity is NOT per-block

While investigating, discovered that `RustSimState.history` (returned by
`get_history()` / `RustHistoryProxy.recent_bbl_addrs` / via the proxy
path of `mgr.found`) records ONE entry per `step_state_with_skip` call,
not per IRSB block. Each step can lift+execute many blocks via
`interpreter.run_until_event` before terminating at an event
(Hook/SimProcedure/Syscall/MaxBlocks). The history shows the
state.pc AFTER the step, which is typically a SimProcedure entry
address. Verified on fauxware: Python history=40 (mixed real code +
extern), Rust history=9 (all extern PLT addrs).

Implication: `bbl_addrs` comparison between engines is not meaningful.
Use `detailed_history` (per-IRSB with jumpkinds) or instrument run_loop
directly if a per-block trace is needed in the future.

Saved as memory `rust-history-granularity`.

### What landed

No code changes. Bd updates:
- Memory `angr-kcf-not-bfs-divergence` — full reasoning + byte evidence.
- Memory `rust-history-granularity` — heads-up for future agents.
- `angr-kcf.1` closed with reason.
- `angr-kcf.2` (the follow-up "fix-or-doc") notes updated to re-point
  the fix at stdin BVS variable tracking, not the manager queue.

### Followup work for next session

`bd ready` after this close:
- angr-prem (P2) / angr-prem.1 (P3) — MemoryLayer trait. Parent notes
  argue the trait premise is wrong; deferred multiple times.
- angr-fk0m (P2) — refactor sync/cache/export mixins. Notes say "purely
  cosmetic"; deferred multiple times. Probably skip.
- angr-34w.12 (P2) — grub OOM. Big. Already documented in memory
  `grub-investigation-depth` as requiring DFS or targeted path
  prioritization.
- angr-kcf.2 (P2, now unblocked) — the actual stdin BVS tracking fix.
  Real work, well-scoped per parent task description.
