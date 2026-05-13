## Session log: 2026-05-13 — angr-w2yr (Characterize bimodal Z3 variance) — CLOSED

### Task

**angr-w2yr** (P2, CLOSED) — Three benchmarks were tagged "bimodal Z3
nondeterministic": ekopartyctf2016_sokohashv2 (9.5s OR 15.4s),
securityfest_fairlight (7.8s OR 15s), google2016_unbreakable_1
(1.4–3.3s). Goal: run 20× each, produce mode-distribution histogram,
decide whether to tighten baselines.

### Findings (20 runs each, rust engine, 2026-05-13)

- **google2016_unbreakable_1**: NO LONGER bimodal. 20/20 in 3.01–3.06s
  (stdev 0.01s). Has stabilized at what used to be the slow mode.
  Baseline 3.5s already gives ~15% headroom — left alone.
- **securityfest_fairlight**: still bimodal, but slow mode drifted
  from ~15s to ~21.4s. 15× slow / 5× fast (~7.95s). Baseline raised
  16.0s → 22.0s.
- **ekopartyctf2016_sokohashv2**: REGRESSED. 3/3 deterministic
  failures at ~0.75s with empty AssertionError. Python engine still
  passes. Cannot measure variance. Opened **angr-7vcx**. Likely
  triggered by commit ef020d101 (x87 transcendental libm fast paths).

Pre-existing test failures discovered on rust-symex HEAD (not caused
by this session's changes, confirmed via git stash):

- TestErrorRecovery::test_dcas_cmpxchg16b_no_match_keeps_memory
- TestNativeFileDescriptorProcedures::test_pipe_native_dispatch_creates_two_fds
- TestNativeFileDescriptorProcedures::test_dup2_native_dispatch_redirects_stdin

All three exhibit "no state in any stash" — same signature family as
sokohashv2's failure. Opened **angr-ruay**.

### What landed (commit 85c80b55c)

- ``tests/benchmarks/bimodal_variance.py`` — new subprocess-driven
  runner. Captures elapsed time for both OK and FAIL outcomes; prints
  1s-bin ASCII histogram; optional ``--json`` dump.
- ``tests/benchmarks/run_single.py`` — capture elapsed time in the
  FAIL return shape so variance runs can record timing even when an
  in-script assertion fires due to Z3 picking a different valid model.
- ``docs/advanced-topics/rust_bimodal_variance.rst`` — full campaign
  writeup, histograms, baseline decisions.
- ``docs/advanced-topics/index.rst`` — toctree entry.
- ``docs/advanced-topics/rust_engine.rst`` — refreshed two rows
  ("google2016_unbreakable_1" and "securityfest_fairlight") in the
  "Other benchmarks below 1.0x" table to cite the new doc.
- ``tests/benchmarks/baseline_timings.json`` — fairlight rust_time
  16.0 → 22.0.

### Memories saved

- NEW ``bimodal-variance-2026-05-13`` — full campaign findings.
- UPDATED ``invariant-bimodal-variance-benchmarks`` — superseded the
  historical 7.8s/15s and 9.5s/15.4s figures; now points at the new doc.

### Beads opened during session

- **angr-7vcx** (P2 bug) — sokohashv2 deterministic 0.75s
  AssertionError on Rust engine; Python still works.
- **angr-ruay** (P2 bug) — 3 pre-existing test failures sharing the
  no-state-in-stash signature.

### Verification

- ``python tests/benchmarks/bimodal_variance.py --runs 2`` smoke-tests
  pass.
- ``python tests/benchmarks/run_single.py fauxware --engine rust`` and
  ``ais3_crackme --engine python`` both still produce ``OK`` lines.
- ``pytest tests/engines/test_rust_exploration.py`` is 389 pass /
  3 fail; the 3 failures are pre-existing on HEAD (confirmed by
  re-running under ``git stash``).
- No Rust code changed, no rebuild needed.
