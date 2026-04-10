# Rust Symex Engine — Next Session Plan

**Branch:** `rust-engine-v2`
**Date:** 2026-04-10
**Current Score:** 8/10 passing, 3 failing with partial fixes applied
**Previous sessions:** 6/10 → 8/10 (session 1), then three targeted fixes applied (session 2)

## What Was Fixed Last Session (2026-04-05)

Three root causes were identified and fixed, but each benchmark has a second-layer issue:

1. **Flareon NO_RET deadend** (`rust_manager.py` ~line 2800): Zero-length NO_RET hooks
   (CallReturn) now check `name in ('exit','_exit','abort','__stack_chk_fail','CallReturn')`
   BEFORE the zero-length hook path. CallReturn deadends with correct PC via skip_hook.

2. **Grub filter re-check** (`rust_techniques.py` apply_technique_filters): Added
   `_filtered_state_ids` set to skip already-filtered states. CheckUniqueness no longer
   prunes all states.

3. **Ekoparty Rust solver delegation** (`rust_manager.py` _install_rust_solver_on_callback_state):
   Replaced `_sync_rust_constraints_to_python` with monkey-patching state.solver methods to
   delegate to the forked Rust solver context. Also fixed Explorer technique address extraction
   for raw int/list find/avoid. Removed active state cap of 5.

## Remaining Second-Layer Issues (priority order)

### 1. Ekoparty — State cache miss for deeply-forked states
- Exploration reaches `get_flag` (0x700028) — correct path found!
- But state 106 is not in `_state_cache` → blank fallback → callback fails
- `_get_pending_root_state_id()` and ancestry chain can't find cached ancestor
- **Root cause:** When many states fork, child IDs aren't cached. Only the initial state (ID 1)
  and states that went through callbacks are cached.
- **Fix approach:** When a callback fires for an uncached state, use the Rust engine to find the
  root state ID, then copy the cached root state and sync registers/memory from Rust. The code
  already tries this (`_get_pending_ancestry()`) but may be failing silently.
- **Debug:** Add logging to `_create_state_for_callback` to trace why state 106's root isn't found.
  Check if `get_pending_root_state_id()` returns None or if the root is also uncached.

### 2. Grub — Unhooked external function at 0xb01038
- CheckUniqueness filter fix works (exploration progresses instead of pruning)
- But execution loops on "Lift error at 0xb01038" — an external function not in the binary
- The grub example only hooks grub_memset, grub_getkey, grub_xputs. Other external symbols
  (grub_env_get, grub_error, grub_fatal, grub_free, grub_malloc, etc.) are NOT hooked.
- **Root cause:** The Rust engine hits an unhooked call to an external function. It tries to lift
  code there, fails, but doesn't properly handle it. States at external addresses should be
  handled by ReturnUnconstrained (angr's default for unresolved externals).
- **Fix approach:** Check how the Python engine handles unresolved external calls. In angr, the
  extern object provides hooks for imported symbols. These may not be registered with the Rust
  engine. Register all extern-object SimProcedures with Rust, not just project._sim_procedures.
- **Debug:** Check `project.loader.extern_object` for hooks at 0xb01038 and nearby addresses.

### 3. Flareon — Snapshot export missing pages + VEX execution wrong
- Two separate issues:
  a) `_snapshot_to_angr` creates blank state, loads pages from snapshot. But the page at
     ARRAY_ADDRESS (0x29f210) shows all zeros in Python, while Rust memory has non-zero bytes.
     The snapshot may not include Rust-modified pages (written via resume_after_simprocedure).
  b) The Rust memory at ARRAY_ADDRESS shows `b'!\x9a5\x0b\xb3\xb4\x14\xf6'` repeated —
     not the expected decrypted text. The VEX execution of tea_decrypt produces wrong output.
     This could be a VEX interpretation bug (endianness, rotation, XOR).
- **Fix approach for (a):** Check export_state() in Rust — does it include pages modified by
  apply_changes()? If not, dirty page tracking needs to include externally-modified pages.
- **Fix approach for (b):** Compare VEX trace of tea_decrypt between Python and Rust engines to
  find where they diverge. The TEA algorithm uses 32-bit XOR, shift, and add — check if any
  of these VEX operations produce different results in the Rust interpreter.

## Current Benchmark Status (8/10 passing)

| Example | Status | Notes |
|---------|--------|-------|
| flareon2015_5 | ✓ 5.03x | |
| securityfest_fairlight | ✓ 0.96x | |
| fauxware | ✓ | |
| ais3_crackme | ✓ 0.67x | |
| csaw_wyvern | ✓ | |
| hackcon2016_angry-reverser | ✓ 0.17x | Slow (every branch → Python) |
| sym-write | ✓ | |
| ekopartyctf2016_rev250 | ✗ | State cache miss for forked states |
| grub | ✗ | Unhooked external loop |
| flareon2015_10 | ✗ | Snapshot export + VEX execution |

## Key Files with Uncommitted Changes

- `angr/exploration/rust_manager.py` — NO_RET fix, Rust solver delegation, active cap removal
- `angr/exploration/rust_techniques.py` — filter re-check prevention, Explorer addr extraction
- `angr/exploration/rust_state_export.py` — (unchanged this session but relevant)
- `native/angr/src/interpreter_cb.rs` — (unchanged this session)
- `native/angr/src/exploration.rs` — (unchanged this session)

## Build/Test Commands

```bash
export PATH="$HOME/.cargo/bin:$PATH" && source .venv/bin/activate
pip install -e .
python tests/benchmarks/run_comparison_10.py
python -m pytest tests/engines/test_rust_exploration.py -v --tb=short
```
