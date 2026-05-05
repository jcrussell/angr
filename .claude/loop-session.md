# Loop session notes (2026-05-05, forty-fifth loop session)

## Task: angr-ho6i — DONE
Add test that pins a symbolic load address to an unmapped value
(0xDEADBEEF) and asserts the state does NOT remain in `active`.

## What I added
`tests/engines/test_rust_exploration.py` —
`TestErrorRecovery::test_symbolic_load_to_unmapped_address_does_not_stay_active`.

- Build a 4-byte shellcode (`mov rax, [rdi]; ret`) with
  `angr.load_shellcode(... arch="AMD64", load_address=0x1000)`.
  Only [0x1000, 0x2000) is mapped (the shellcode page).
- Create blank_state with `STRICT_PAGE_ACCESS` and pin
  `state.regs.rdi == 0xDEADBEEF` via `state.solver.add(...)`.
- Run with `RustExplorationManager`.
- Assert `stash_counts["active"] == 0`.

## Result
Test PASSES. Current behaviour: state lands in `deadended` (not
`errored`). It does NOT silently stay in `active`, so the worst
divergence flagged in the bead description (fall back to addr 0 / fresh
symbol) is not present today.

Path under the hood (saved as `unsat-load-fallback-stash` memory):
Rust concretization Failed → `CbExecutionError::Unsupported` →
`RunResult::NeedPythonVEX` → Python VEX fallback fills unconstrained
bytes at the unmapped page (default_filler_mixin) and continues; a
later stack/control-flow load (`0x7fffffff0000`) fails, lift at 0x0
fails, and the state deadends. Python angr would raise
`SimMemoryAddressError` under STRICT_PAGE_ACCESS, so we still diverge,
but not via `active`.

## Verification
- 214/214 pytest passing in `tests/engines/test_rust_exploration.py`.

## Files modified
- `tests/engines/test_rust_exploration.py` — added one new test.

## Beads / memory
- angr-ho6i claimed → closed.
- Memory `unsat-load-fallback-stash` saved (deadended-not-errored
  divergence vs. Python angr's SimMemoryAddressError).
