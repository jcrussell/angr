# Loop session notes (2026-05-03, twenty-sixth session)

## Task: angr-v4db
Extract god-methods in Python bridge layer.

## Status
- DONE. Closed angr-v4db.
- Two commits: 7def453e3 (IRSB + init), 129ce886d (sync_memory).

## What landed

### 1. _serialize_irsb -> rust_irsb_serializer.py (commit 7def453e3)
Created angr/exploration/rust_irsb_serializer.py with:
- serialize_irsb(irsb) -> str: entry point
- _serialize_const / _serialize_descr / _serialize_cee
- _serialize_expr / _serialize_stmt
- _CONST_TYPES, _EXPR_FIELDS, _STMT_FIELDS tables
The class method now just forwards to serialize_irsb.

### 2. _run_python_init_if_needed (commit 7def453e3)
Decomposed ~195-line method (4 try-blocks, duplicated cache-load
metadata handling) into 6 helpers + the dispatcher (~33 lines):

| Helper                          | Purpose                                    |
|---------------------------------|--------------------------------------------|
| `_apply_state_metadata`         | Copy constraints/globals/LAZY_SOLVES (3x)  |
| `_compute_disk_init_key`        | Gate disk caching on user symbolic data    |
| `_try_in_memory_init_cache`     | Per-process class-level init cache         |
| `_try_disk_init_cache`          | Persistent disk cache                      |
| `_resolve_main_address`         | Symbol or _start PUT(rdi) extraction       |
| `_save_init_state_to_caches`    | Write-through to both caches               |
| `_step_python_to_main`          | SimulationManager loop (500 step budget)   |

### 3. _sync_memory_to_rust (commit 129ce886d)
Decomposed ~350-line method into a 36-line dispatcher + 11 phase
helpers + 1 static utility:

| Helper                          | Phase                                       |
|---------------------------------|---------------------------------------------|
| `_try_fast_memory_sync`         | Cached layout from disk init cache          |
| `_find_user_symbolic_pages`     | Pages skipped during loader map (callback)  |
| `_map_loader_pages`             | Batched FFI mapping of segments             |
| `_overlay_relocated_sections`   | Small concrete sections (GOT, etc.)         |
| `_overlay_python_state_pages`   | Python state on loader (multi-stage fix)    |
| `_add_loader_lazy_regions`      | fetch_page coverage                         |
| `_setup_stack_region`           | Compute SP page + lazy mark                 |
| `_sync_stack_page`              | Pre-populate page at SP (fast/slow path)    |
| `_extract_stack_symbolic_from_sd` | Targeted symbolic_data scan               |
| `_sync_extra_python_pages`      | Non-loader/non-stack pages (multi-stage)    |
| `_scan_user_symbolic_pages`     | Wide region import                          |
| `_has_user_symbolic_var`(static)| Dedupe mem_/reg_/unconstrained filter (3x)  |

## Tests
- 208/208 passing throughout
- Smoke benchmarks: fauxware (finds SOSNEAKY), ais3_crackme (0.84s,
  finds flag), defcamp_r100 (0.23s OK), csgames2018 (0.96s OK)

## Files modified
- angr/exploration/rust_irsb_serializer.py (new, 260 lines)
- angr/exploration/rust_manager.py (-372/+187 lines)
- angr/exploration/rust_state_sync.py (-220/+266 lines)

## Memories saved
- `sync-memory-pipeline-phases`: 11-phase pipeline order + threaded state
- `invariant-init-cache-user-symbolic`: disk init cache must be gated on
  _state_has_user_symbolic; _compute_disk_init_key enforces both load+save
- `irsb-serializer-module`: location and structure of the new module

## Other ready P3 tasks
- angr-w4os: Python bridge cleanup (split sync/export/cache/init methods)
  — overlaps with what we did but title also mentions "state cache eviction"
- angr-cbko: native exit/abort SimProcs
- angr-8em4: replace panic patterns
- angr-3ijo: bincode for VEX IRSB serialization (would build on the new
  serializer module)
- angr-bgv0: Z3 floating point theory
- angr-awm3: CAS/LLSC statement handling
