# Loop session notes (2026-05-01, third session)

## Closed this session

### angr-i1wg — Silent exception swallowing audit (commits 4f16e3792, 1e00e4919)

Addressed the 4 explicitly-listed CRITICAL/HIGH sites:

1. **rust_state_export.py:_sync_rust_registers_to_state** —
   `export_state` failure now warns; per-register sync failure
   downgraded from bare `pass` to `l.debug` (VEX internal regs).

2. **rust_state_export.py:_sync_rust_memory_to_state** — flushed/
   unflushed double-fallback now logs both errors. The "neither
   worked, return silently" path was leaving Python memory stale.

3. **rust_state_export.py:satisfiable_with_fallback** AND
   **_attach_rust_solver_primary:satisfiable_rust_primary** —
   was returning False (= UNSAT, a wrong answer) on double exception.
   Now re-raises so callers see the solver failure. This is a
   behavioral change but the previous behavior was a silent wrong
   answer, not a fallback.

4. **rust_state_sync.py:_sync_registers_to_rust** — symbolic Z3
   conversion failure now warns (Rust would otherwise see uninit
   BVS). Outer bare except split into AttributeError (debug) +
   Exception (warn).

Tests: 207/207 passing. fauxware benchmark unchanged (~0.35s).

Saved memories:
- `satisfiable-wrong-answer` — why returning False on exception
  is wrong, not a fallback.
- `invariant-bridge-except-categorization` — 3-bucket taxonomy
  for classifying remaining bridge excepts.

Created angr-vt0t (P3) for the broader categorization of the
remaining ~280 except blocks across the bridge.

## Closed previously

(see prior session notes — angr-2q74, angr-ur31, etc.)

## Ready P2/P3 tasks remaining

- angr-vt0t P3 NEW (categorize remaining ~280 except blocks)
- angr-8em4 P3 (panic audit — 543 sites)
- angr-742d P3 (register accessor macros — only 2 sites)
- angr-3ijo P3 (bincode for IRSB serialization spike)
- angr-bgv0 P3 (Z3 FP theory)
- angr-awm3 P3 (CAS/LLSC statement handling)
- angr-7c9j P3 (feature flag correctness in CI)
- angr-dja4 P3 (expand benchmark baseline)
- angr-sc8h P3 (replace solver fallback monkey-patching)
- angr-wpi7 P3 (consolidate P1-P19/GAP fix workarounds)
