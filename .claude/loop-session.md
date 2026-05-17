## Session log: 2026-05-17 — angr-gra3 validated, cleanup hook is no-op

### Status: CLOSED — revert landed (commit 17ab6787a). 437/437 pass.

### Task

**angr-gra3 (P2, task)** — "Validate angr-518z cleanup() benefit on
mma_howtouse post-9maq fix". With angr-9maq closed, re-measure
whether the cleanup() flag delivers the predicted 23% gain from
`mma-howtouse-cache-clear-speedup` memory.

### Findings

| cfg                | wall    | peak  |
|--------------------|---------|-------|
| python             | 4.29s   | 226MB |
| rust cleanup=False | 7.38s   | 277MB |
| rust cleanup=True  | 7.32s   | 277MB |

The cleanup hook delivers **0% measurable benefit** on current HEAD.
Rust runs at 0.58x of Python.

### Why the prediction failed

1. The 23% gain came from `claripy.clear_all_caches()`. That API
   **no longer exists** in claripy — replaced with WeakValueDictionary
   caches that GC naturally.
2. The angr-518z `clear_ast_cache()` PyO3 export clears only the
   **Rust-side** translation LRUs (AST_CACHE, CLARIPY_AST_CACHE,
   EXPRESSION_CACHE, EXPRESSION_BY_OPERANDS_PTR) — not equivalent to
   the original Python-side cache flush.
3. The Rust LRUs are bounded at 10000 entries; 45 Callable
   invocations don't fill them enough to cause cold-line misses.

### Action

- Reverted `tests/benchmarks/run_single.py` to construct
  `RustExplorationManager(project, states)` with default kwargs
  (removed `clear_caches_on_cleanup=True` opt-in that implied a
  benefit that isn't real).
- Kept the angr-518z infrastructure intact (PyO3 export,
  cleanup() method, ctor flag, 5 TestRustManagerCleanup tests). It
  remains a valid opt-in hygiene primitive in case a future workload
  actually fills the LRUs.

### Files modified

- `tests/benchmarks/run_single.py` — -8 lines, +1 line (remove opt-in).

### Memories saved

- `benchmark-mma-howtouse-2026-05-17-final` — definitive numbers on
  HEAD 17ab6787a; mma_howtouse at 0.58x permanently until the
  remaining Python-vs-Rust gap is investigated separately.
- `avoid-trusting-stale-cache-clear-speedup` — warns future sessions
  not to plan against the stale 23% prediction.
- `claripy-cache-clear-api` — what cache structures actually exist in
  current claripy (WeakValueDictionary, no global flush API).

### Followup work for next session

- **mma_howtouse remains 0.58x slower than Python**. The gap is no
  longer attributable to AST cache lookup (validated this session).
  A fresh investigation could profile the Callable construction path,
  per-manager teardown, or the Rust↔Python FFI cost for
  short-lived states. File a new bead if there's appetite to pursue.
- The previous session's secondary followup ("check angr-7vcx's
  `_scan_symbolic_pages` change for similar untested cost") is still
  open and worth a 30-min look.
