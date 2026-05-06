# Loop session notes (2026-05-06, seventy-second loop session — DONE)

## Status: COMPLETE — angr-mjc0 closed

## Task: angr-mjc0 (P3) — Encapsulate _perf_stats behind explicit accessors

### What changed
New module `angr/exploration/rust_perf_tracker.py` (PerformanceTracker class).
Wraps the previously-bare ``self._perf_stats`` dict. Writers now use named
methods; reads still work via ``__getitem__`` / ``.get()`` so reporters and
debug scripts didn't need touching.

API:
  - ``set_init_phase(phase, ns)`` / ``add_init_phase(phase, ns)``
  - ``record_simprocedure_call(total_ns)`` (count++ + total_ns)
  - ``increment_simprocedure_count()`` (orphan VEX-fallback site)
  - ``add_simprocedure_phase(phase, ns)`` for sub-phases
    (state_create / execute / sync_back / state_copy)
  - ``record_memory_load(ns)`` / ``record_fetch_page(ns)`` / ``record_lift_block(ns)``
  - ``__getitem__`` / ``get`` / ``as_dict`` for read-side compat

Files touched:
  - angr/exploration/rust_perf_tracker.py  (new, ~100 lines)
  - angr/exploration/rust_manager.py       (-24 init dict + 9 writer sites)
  - angr/exploration/rust_callback_dispatch.py  (13 writer sites)

### Scope decision
Bead title mentions `_perf_stats / _state_cache / _rust_mgr`. Scoped to ONLY
`_perf_stats` because:
  - bead body and AC mention only PerformanceTracker
  - `_state_cache` (71 refs) and `_rust_mgr` (222 refs) are larger refactors
  - AC is "No direct dict access to _perf_stats outside its owning class"
  - Future bead can pick up the others if desired

### Verification
- pytest tests/engines/test_rust_exploration.py: 243/243 passing (8.20s)
- run_single.py fauxware --engine rust: perf_report renders correctly with all
  phases reporting non-zero where expected (Init=15.1ms, SimProcedures=6, etc)

### Build note
Recovered .venv at session start. The venv was missing bin/ scripts and many
.py files. Site-packages had only __pycache__/*.cpython-312.pyc files. Wrote
/tmp/recover_pycache.sh to copy ``__pycache__/foo.cpython-312.pyc`` to
``foo.pyc`` at the parent level (Python 3.x can find them there via PEP 488),
recovered 549 files. Recreated minimal activate script. No pip available.
Worked around missing editable-install .pth via PYTHONPATH=. for run_single.py.

### Commits
- (pending) refactor(rust_manager): encapsulate _perf_stats behind PerformanceTracker (angr-mjc0)

### Memories saved
- venv-recovery-pattern (how to recover py-stripped venv)

## Next-up (still ready)
- angr-eygl (P1) Differential test harness — multi-session
- angr-pufm (P1) Symbolic address concretization fallback — multi-session
- angr-prem (P2) MemoryLayer trait
- angr-fk0m (P2) Unify rust_state_sync / cache / export mixins
- angr-3zs6 (P2) FallbackStrategy enum
- angr-imy1 (P2) Native syscall coverage
- angr-fbl0 (P2) Native SimProcedure coverage
- angr-zrq1 (P2) Dirty-call coverage stubs
- angr-4j5u (P2) Decompose RustExplorationManager
- angr-3vrj (P3) StateMetadata dataclass
- (many P3 — see `bd ready -n 50`)
