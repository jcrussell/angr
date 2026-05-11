## Session log: 2026-05-11 — angr-ed7j (slow-benchmark documentation)

### Task
Investigate mma_howtouse (0.65x) and ekopartyctf2016_sokohashv2 (slow floor) regressions vs Python. Acceptance: each bench either above 1.0x OR has a written rationale + table entry.

### Approach
Investigation was already substantially complete via prior bd memories:
- mma-howtouse-leak-source (memory leak fixed in 342df4a7f; remaining 6.5s vs 4.2s is AST cache lookup overhead)
- mma-howtouse-cache-clear-speedup (clear_all_caches() reduces 6.62s → 5.07s but only helps benchmark-style isolated calls)
- avoid-silent-zero-raw-fallback (sokohashv2 uses x87 fyl2x/fscale/f2xm1; Raw fallback was silently 0, fixed)
- invariant-bimodal-variance-benchmarks (sokohashv2 bimodal ~9.5s/~15.4s due to Z3 nondeterminism)

Chose option (b) — document root cause + table entry. A Rust fix is non-trivial (per-manager AST cache scoping or native x87 transcendentals) and the workloads are degenerate/niche.

### Changes
- Created `docs/RUST_KNOWN_SLOWER_BENCHMARKS.md` with detailed rationale for mma_howtouse and sokohashv2, plus a brief table covering the other three sub-1.0x benchmarks.
- Updated CLAUDE.md table notes for both benchmarks to point to the new doc (removed "See angr-ed7j" sentinels).

### Files modified
- docs/RUST_KNOWN_SLOWER_BENCHMARKS.md (new)
- CLAUDE.md (table notes for mma_howtouse and ekopartyctf2016_sokohashv2)

### Commit / bead
- Pending commit.
- angr-ed7j to be closed after commit.

### Status
Doc-only change, no Rust rebuild required. Will commit + close.
