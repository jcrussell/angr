# Loop session notes (2026-05-07, 115th loop session)

## Task: angr-khth — separate init-pipeline phases (DONE)

### What landed (commit aed1ec175)

Split `_load_init_from_disk_cache` (rust_manager.py:1398) — previously a
single ~80-line function fusing pickle I/O, SimState deserialization, and
manager-owned metadata mutation — into three independently testable phases:

- `_load_init_pickle` (~1398) — pure I/O; returns data dict or None on miss
  or read failure (catches OSError / UnpicklingError / EOFError).
- `_deserialize_init_state` (~1411) — pure SimState + mem_cache construction
  from data dict; NO manager metadata mutation.
- `_apply_init_side_effects` (~1465) — populates
  `self._pending_procedure_data` (continuation slots) and
  `self._precomputed_regs` (fast Rust register sync).
- `_load_init_from_disk_cache` (~1476) — thin orchestrator preserving the
  prior all-or-nothing semantic (deserialize failure → metadata untouched,
  return (None, None)).

### Why I rejected the "overlay symbolic pages / hook memory / stdin" half

`_save_init_to_disk_cache` does NOT store those — by design. The disk cache
serves a *blank* state at main, BEFORE user symbolic data exists. A binary's
init sequence runs purely concrete. Symbolic pages, hook memory, and stdin
metadata are populated DURING exploration via Python callbacks, not at init.

### Cache validator already done

`_disk_cache_key` (rust_manager.py:1305) already combines binary hash +
RUST_CACHE_VERSION + PYTHON_METADATA_VERSION + arch_name (4 axes). The
bead's "feature flags" axis is already absorbed by the two version axes
(bump version when a feature changes the cache layout).

### Other init-pipeline phases were already modular

- `_compute_disk_init_key` (rust_manager.py:1571) — caller-side guard.
- `_save_init_to_disk_cache` (rust_manager.py:1335).
- `_run_python_init_if_needed` (rust_manager.py:1506) — orchestrator.
- `_try_in_memory_init_cache` (rust_manager.py:1582).
- `_try_disk_init_cache` (rust_manager.py:1593).
- `_apply_state_metadata` (rust_manager.py:1544).
- `_extract_continuation_data` (rust_manager.py:1482).

### Test results

- 261/261 Python tests pass.
- fauxware benchmark (rust engine) runs cleanly post-refactor.
- No Rust changes; no cargo build needed.

### Memories saved

- `invariant-init-pipeline-phases` — full phase map with line numbers and
  the rule "if you save a NEW field to disk cache, decide whether
  deserialization (pure) or side-effect (manager mutation) phase owns it.
  Don't mix the two."

### Next session candidates

P1 / P2 ready (still unblocked):
- angr-pufm (P1) symbolic concretization fallback — multi-session.
- angr-prem (P2) MemoryLayer trait refactor — multi-session.
- angr-fk0m (P2) unify state mixin classes (rust_state_sync /
  rust_state_cache / rust_state_export).
- angr-4j5u (P2) decompose 95-field god struct — multi-session.
- angr-m2hf (P2) unified error trait — 11 error enums, multi-Rust-file change.
- angr-wqao (P2) split rust_manager.py 2949 lines — multi-session.

P3 ready:
- angr-3vrj — StateMetadata dataclass (~86 metadata refs across 6 files, larger
  than the bead's "56 sites" estimate; could do a focused first slice with just
  the 3 fields the bead lists).
- angr-csd1 — split PythonCallbacks into focused traits (48 callsites across
  12 Rust files).
- angr-x3xu — SolverBridge protocol; previous session noted manager has no
  direct RustSolverContext coupling, so the refactor is mostly a Protocol
  declaration. Limited value unless paired with a generic-over-backend test
  harness.
- angr-qrhl — VEX op trait dispatcher (needs perf measurement).
- angr-czph, angr-bkcs, angr-n28w — feature work, multi-session.

Already addressed and closed in prior sessions:
- angr-ja0b (StepOutcome trait) — pattern-matching is the natural Rust
  expression here; rejected with rationale.
- angr-khth (init-pipeline split) — done THIS session.
