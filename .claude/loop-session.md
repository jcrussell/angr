## Session log: 2026-05-09 — angr-ony8 run_single.py spawn PYTHONPATH self-heal (205th loop session, COMPLETE)

### Task
angr-ony8 (P2): tests/benchmarks/run_single.py spawn-child failed with
'No module named angr' unless PYTHONPATH was set. Workaround was
documented in memory `venv-rebuild-cargo-only-2026-05-09` but never
codified into the script.

### Root cause
multiprocessing.get_context('spawn') children get sys.path[0] = the
directory of the executed module (`tests/benchmarks/`), NOT the parent's
cwd. The editable angr install ships no .pth file (only the
`__editable___angr_*finder.pyc` finder, with nothing to register it via
site.py). The parent works because it runs from repo root with '' on
sys.path; the child does not.

### What landed (commit 99901512d)
In `_run_in_child` (tests/benchmarks/run_single.py:81), prepend
`os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`
(three levels up = repo root) to sys.path before `import angr`. 7 lines
added.

### Verification
- `python tests/benchmarks/run_single.py fauxware --engine rust` works
  without PYTHONPATH (was the reported failure case).
- `python tests/benchmarks/run_single.py fauxware --both` works.
- `python tests/benchmarks/run_regression.py` reuses _run_in_child, so it
  is fixed too: 12/13 SLA pass (the 1 failure is google2016_unbreakable_1
  bimodal-variance, unrelated to this fix).
- `pytest tests/engines/test_rust_exploration.py`: 369/369 pass.

### Findings saved (bd remember)
- `invariant-spawn-child-pythonpath` — new invariant; future spawn entry
  points must self-heal sys.path.
- `venv-rebuild-cargo-only-2026-05-09` — updated to note PYTHONPATH is no
  longer required after this fix.

### Files modified
- `tests/benchmarks/run_single.py` — 7 lines, _run_in_child only.

### Status
COMPLETE. Bead closed with commit reference. Run benchmarks no longer
need PYTHONPATH set.
