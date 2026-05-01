# Loop session notes (2026-05-01, sixth session)

## Closed this session

### angr-jmiz — Fix max_active_states + delete dead files — commit 15960a013

Two-part fix for the bug surfaced during last session:

**Part 1 — Delete dead source files (~2.8K lines removed):**
- `native/angr/src/exploration/pyapi.rs`
- `native/angr/src/exploration/resume.rs`
- `native/angr/src/exploration/run_loop.rs`

These claimed in their header comments to be "included into mod.rs"
but were never declared as modules. Their content had been duplicated
into the live mod.rs ages ago, and the dead copies were diverging
silently. Verified via diff that no live functionality was lost
(only `py_eval_stdin_symbol` differed — a thin wrapper around the
live `eval_stdin_symbol`, unused from Python).

**Part 2 — Restore max_active_states enforcement:**
- Restored `push_to_active_or_drop` helper in `exploration/helpers.rs`
  using `sm.push()` (which indexes) rather than raw `push_back`.
- Replaced 8 direct STASH_ACTIVE push sites in mod.rs with helper:
  - native procedure return path (~line 2580)
  - regular successors after step (~line 2648)
  - deferred forks at find/avoid boundaries (~line 2725)
  - regular successors final path (~line 3106)
  - deadend deferred forks (~line 3212)
  - resume_after_symbolic_branch bulk push (~line 3447)
  - resume_find_predicate not-matched (~line 3494)
  - resume_avoid_predicate not-matched (~line 3520)

**Part 3 — Functional test:**
Added `test_max_active_states_prunes_forks` — sets limit=1 on
fauxware, explores, asserts `counts['pruned'] > 0`. The prior test
only checked `active <= limit` which passed even when the limit
was a silent no-op.

Tests: 208/208 Python passing; 364/364 cargo lib tests; cargo
release build warning-free.

Memories updated:
- `invariant-active-stash-push` (NEW) — must use helper at new push sites
- `invariant-dead-source-files` — updated: now resolved
- `max-active-states-test-pattern` (NEW) — how to verify wiring
- `bug-max-active-states-unenforced` — forgotten (resolved)
- `invariant-pyapi-dead-code` — forgotten (resolved by deletion)

## Closed previously

### angr-gste — Remove dead code — commit 1992563dd
8 dead items removed; surfaced angr-jmiz (this session's work).

### angr-742d — Register accessor macros — commit a25cdc3ef
Replaced hand-rolled VexArch match with Arch trait methods.

## Ready P-tasks remaining

- angr-vt0t (P3 categorize remaining ~280 except blocks)
- angr-8em4 (P3 panic audit — 543 sites)
- angr-3ijo (P3 bincode for IRSB serialization spike)
- angr-bgv0 (P3 Z3 FP theory)
- angr-awm3 (P3 CAS/LLSC statement handling)
- angr-7c9j (P3 feature flag correctness in CI)
- angr-dja4 (P3 expand benchmark baseline)
- angr-sc8h (P3 replace solver fallback monkey-patching)
- angr-wpi7 (P3 consolidate P1-P19/GAP fix workarounds)
- angr-v4db (P3 extract god-methods)
