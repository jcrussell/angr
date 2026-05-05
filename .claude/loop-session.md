# Loop session notes (2026-05-05, forty-eighth loop session)

## Task: angr-q2dk — DONE
Profile securityfest_fairlight to confirm Z3 FP-theory dominates the
+62% post-FP-wiring overhead.

## What I did
Ran `python tests/benchmarks/run_single.py securityfest_fairlight
--engine rust --timeout 60` three times for stability. Captured
`rust profiling` and `z3 solver stats` blocks. Cross-checked with
`objdump -d fairlight | grep -c xmm` to confirm FP density (616
xmm-touching instructions).

## Result
- Total runtime: 12.9 / 14.0 / 12.9 s (peak_mem ~330MB).
- **Z3 dominates ~95% of runtime.** z3_check umbrella 12.3-13.3s
  across 35 calls.
- Top 3 sites:
  1. z3_site_branch_true: 9.1-10.6s / 13 calls (~700-820 ms/call)
  2. z3_site_branch_false: 1.2-2.1s / 2 calls
  3. z3_site_satisfiable: 1.0-1.3s / 16 calls
- Non-Z3 work negligible: expr_eval 14ms (11109 calls), lift 51ms,
  python_callback 46ms (109 calls), FFI crossings 3.
- FP-theory inferred from binary (xmm-heavy) + cost shape (very
  high per-call branch_true).
- Headline insight: branch_true per-call cost is the lever; this is
  about path-condition accumulation, not single-FP-op cost.

## Files changed
None — pure measurement. Bead notes + 2 memories updated.

## Beads / memory
- angr-q2dk claimed → closed with full report in notes.
- `fairlight-bottleneck-2026-05` updated with 2026-05-05 numbers.
- `fairlight-bottleneck` consolidated across measurements (5/01, 5/05).

## Caution / inherited mess
While searching for memories I accidentally ran `bd remember
fairlight-bottleneck` and `bd remember fairlight-bottleneck-2026-05`
without an insight argument — those overwrote the memory bodies with
the keys themselves. Restored both with fresh content from this run.
**Do NOT use `bd remember <key>` to read; use `bd memories <pattern>`.**
The remember subcommand stores its first positional as the body.

## Next-up
- **angr-2aih (P0)** is now UNBLOCKED. 22-benchmark sweep +
  baseline_timings.json refresh. Long-running task — give it a full
  session. Use `python tests/benchmarks/run_regression.py --full
  --update` per `project_benchmark_status.md`. Confirm `cargo test
  --release --lib` and pytest pass at the end.
- **angr-fbxi (P1)** can also unblock now (single-bench baseline
  refresh for unbreakable_1 and hackcon angry-reverser). Smaller
  scope; could fold into 2aih or run separately.
- Optimization-side ready candidates: angr-8kht (avoid double-Z3 in
  solutions+range), angr-nwbx (cache claripy↔Z3 conversion for
  register sync), angr-6uhh (eliminate per-step clones).
