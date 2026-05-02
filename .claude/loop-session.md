# Loop session notes (2026-05-02, twentieth session)

## Task: angr-rsfv (CLOSED)
[bench] Investigate or refresh 8 baselines that consistently fail by 18-71%

## Outcome
- Verified non-regression: checked out baseline-set commit (15395dd6b),
  rebuilt .so, ran ais3_crackme — reproduced HEAD timings (0.83-0.84s),
  not the recorded 0.498s baseline. fauxware (not in regression list) ran
  at 0.37-0.38s on both HEAD and 15395dd6b vs baseline 0.356s — confirms
  no system-wide perf change.
- Refreshed all 12 FAST_SUITE baselines via `run_regression.py --update
  --rust-only`. 12/12 pass with --check-counts; 208/208 pytest passes.
- Commit: d200d4901

## Key insight
- The original 0.498s baseline came from an abnormally fast
  --update --full pass; the same exact source code does not reproduce
  it on subsequent runs. This is the same family of issue noted in
  benchmark-update-variance memory.
- Memory baselines also moved: most went up (ais3_crackme 237→364 MB,
  unbreakable_0 249→426 MB, etc.), unmapped_analysis dropped 876→474
  MB (GC leak fix 342df4a7f finally credited in baseline).

## Memories saved
- `benchmark-staleness-verification`: how to confirm a regression is
  stale-baseline vs real-regression by rebuilding at baseline commit
- `benchmark-peak-memory-2026-05-02`: peak memory values updated in
  d200d4901 (mostly upward), so future peak regression alerts have a
  prior

## Other ready tasks
- angr-3tek (P2): native read/write SimProcs blocked by stale-cache
- angr-w4os (P3): Python bridge cleanup
- angr-2fs0 (P3): decompose _handle_simprocedure_callback
- angr-1f8s (P3): refactor stepping.rs InterpreterStepResult struct
- angr-cbko (P3): native exit/abort SimProcs
- angr-8em4 (P3): replace panic patterns
- angr-3ijo (P3): bincode for VEX IRSB serialization
- angr-bgv0 (P3): Z3 floating point theory
- angr-awm3 (P3): CAS/LLSC statement handling
