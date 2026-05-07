# Loop session notes (2026-05-07, 121st loop session)

## Task: angr-p5hl — Benchmark gate still broken: angr-lbze fix didn't land  ✓ CLOSED

Bead premise was incorrect. The angr-lbze fix (commit a6b4f6615) DID land
and works correctly — verified end-to-end via direct invocation of
`run_benchmark_gate('2G')`: 12/12 benchmarks passed in 19.6s.

The reason iter 48-50 of the previous loop run still showed 12 fake
"No module named 'angr'" failures is that those iterations belonged to
a long-running orchestrator process started BEFORE the fix landed.
Python doesn't reload its source code on file edits, so the orchestrator
kept executing pre-fix bytecode for the rest of its run. The current
orchestrator (PID 929591, started 2026-05-07 16:17:33 — after the fix)
correctly applies PYTHONPATH and the gate passes.

### Diagnosis evidence

- Manual reproduction at 16:19 with same env/cmd as `run_benchmark_gate`:
  12/12 benchmarks pass in 19.3s.
- Manual reproduction with PYTHONPATH stripped from env: 12/12 fail with
  "No module named 'angr'" in 0.7s — exactly matches the iter 48-50 logs.
- `gate_broken` field absent from iter50 JSON output → that iter ran
  pre-fix code (the field was added by angr-lbze).
- Currently running `run_benchmark_gate` test: passed=12, failed=0,
  gate_broken=False, duration=19.6s.

### Fix added (defensive)

Added a stale-orchestrator detector in `run_optimization_loop.py`:
- `_orchestrator_source_hash()` SHA-256s `__file__` on disk.
- Captured at startup, compared at each iteration; warns once when the
  hashes diverge so the operator can spot pre-fix bytecode reuse.
- Startup log line now includes `source_sha=...`.

### Files modified

- `run_optimization_loop.py` (+33 lines): hashlib import, helper, startup
  capture, per-iter freshness check.

### Validation

- `python -m pytest tests/engines/test_rust_exploration.py`: 261/261 pass.
- `python -c "import run_optimization_loop"`: imports cleanly.
- `_orchestrator_source_hash()` returns 12-char prefix as expected.
- Direct `run_benchmark_gate('2G')` call: passed=12 failed=0 gate_broken=False.

## Status: complete
