## Session log: 2026-05-13 — angr-govb + angr-wvxj (auto-memory housekeeping)

Two related memory-housekeeping tasks closed in the same session.

### angr-govb — project_rust_engine.md refresh
P1 chore. Auto-memory `project_rust_engine.md` was 32 days old and
claimed "14/14 tests passing" (real count: 389 tests, 22 benchmarks,
17/22 faster than Python). Rewrote as a thin pointer to CLAUDE.md
(the live doc). Rationale: snapshots in memory rot; pointers do not.
Updated MEMORY.md index entry to match.

### angr-wvxj — endianness_bug distillation
P1 chore. Auto-memory `project_endianness_bug.md` documented three
bugs (wide symbolic import endianness, Rust solver eval fallback,
pre-pinning overconstrain) that are all FIXED in code. Verified the
three referenced commits exist (bb1a908dc, 1f823ba68, 2f0e8164f) and
that `_attach_rust_solver_fallback` is still wired (rust_state_export,
rust_state_proxy, rust_state_cache).

Replaced with `feedback_constraint_export.md`, framed as a lesson:
- Rule: don't pre-pin tracked symbols with ast==BVV during export.
- Why: pre-pinning overconstrains because get_state_memory() concrete
  values can disagree with the actual model — broke fauxware and
  flareon2015_5 in the rolled-back attempt.
- How to apply: ensure `_attach_rust_solver_fallback` is wired on
  every new export path; Rust eval returns LE bytes (reverse for
  cast_to=bytes); wide symbolic loads need Iend_BE + Reverse().

Deleted the original .md, updated MEMORY.md index, cross-ref in
project_rust_engine.md.

### Verification
No code changes. Memory files live outside the repo. `git status`
clean (only untracked `.venv/`). No build/test required.

### Memories saved this session
- `invariant-memory-pointer-vs-snapshot` — snapshots in memory rot;
  prefer pointers to CLAUDE.md for live state.
- `avoid-prepinning-export-constraints` — the pre-pinning anti-pattern
  with the rollback history.

### Prior session
`angr-w2je` (unsat_core FFI) closed at 190186a32.

### Next ready work
- `angr-7ylc` (P1, contributor guide: native SimProcedures) — has
  comment redirecting output to .rst tree.
- `angr-2bjx` (P1, contributor guide: VEX ops) — also redirected to
  .rst.
- `angr-zv0a` (P1, migrate docs/RUST_*.md into Sphinx tree).
- `angr-qdwt` (P1, pre-PR docs sweep) — runs LAST, blocked on
  in-flight docs/test work.
