## Session log: 2026-05-14 (later)

### Task: angr-pogf — RESEARCH: Design lazy symbolic memory model — CLOSED

Wrote `docs/advanced-topics/rust_lazy_memory_design.rst` and linked it
into the toctree. The doc is the design-phase deliverable that
unblocks the implementation children:
- angr-czph (lazy LOAD)
- angr-qh5u (lazy STORE)

### Key design decisions captured in the doc

1. Recommend **Option A** (per-byte MultiValues port, Python-equivalent)
   over Option B (Z3 array primitive) and Option C (Python fallback
   hybrid). Reuses existing balanced ITE builder, concretization
   strategy chain, and fork/merge plumbing; only the page byte cell
   representation changes.

2. Two-phase plan:
   - Phase 0: ite_depth_max counter as gate metric
   - Phase 1 (angr-czph): Multi cell variant + read-time collapse;
     wire one annotated SimProcedure (strchr) as test driver
   - Phase 2 (angr-qh5u): make store_symbolic_unified emit Multi
     cells by default for Multiple/Strided results
   - Phase 3 (Z3 arrays): only if Phase 2 doesn't close the gap

3. Acceptance criteria per phase, with sym-write ≥ 2x current Rust
   time as the Phase 2 soft blocker.

### Findings worth remembering across sessions
- Python is NOT actually lazy on address resolution; it uses the
  same Range→Any/Range→Max strategy chain Rust mirrors. The laziness
  is at the PAGE level: MultiValues stores a set of alternative
  values per byte, collapsed to ITE only at LOAD time.
- PendingWrite is scaffolded but inactive in execution paths (only
  drained on export). The lazy-load-overlay-fails memory explains
  why this scaffolding is not the path forward.
- z3-rs 0.19 (already in Cargo.toml) does expose Array::store/select
  — no version bump needed for Option B if Phase 3 ever runs.
- Existing strcpy_find is now 2.3x (was 0.21x) after the CFG fix in
  angr-3tek; the lazy memory work targets sym-write (6.9x slower)
  primarily.

### Files modified
- docs/advanced-topics/rust_lazy_memory_design.rst (new, 488 lines)
- docs/advanced-topics/index.rst (added to toctree)

### Status at session end
- Doc passes manual heading-length check (Python script confirmed
  underlines match titles; em-dashes count as 1 char correctly).
- No sphinx in venv; couldn't render. Manual eyeball of RST cross-
  references (:doc:`rust_engine`, etc.) matches existing patterns.
- Nothing else changed; no tests need to run.
