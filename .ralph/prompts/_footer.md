## Workflow

For each task, follow this loop:

 1. `bd update <id> --claim`
 2. `bd show <id>` — read the description carefully
 3. `bd memories` — check for relevant invariants/pitfalls before coding
 4. Implement the change (edit Rust and/or Python files as described)
 5. Build: `cargo check --manifest-path native/angr/Cargo.toml --release`
 6. If Rust changed: `pip install -e . --no-build-isolation --no-deps`
 7. Test: `python -m pytest tests/engines/test_rust_exploration.py -v --tb=short`
 8. If tests pass: `git add <changed files> && git commit -m "<description>"`
 9. `bd close <id> --reason="<what was done>"`
10. **MANDATORY — save what you learned.** Run `bd remember` for each that applies:
    - Root cause was surprising or non-obvious? `--key <topic>-root-cause`
    - An approach failed before the one that worked? `--key avoid-<thing>`
    - A constraint/invariant future work must respect? `--key invariant-<thing>`
    - Profiling showed bottleneck was NOT where expected? `--key <topic>-bottleneck`
    - A benchmark number changed significantly? `--key benchmark-<topic>`

    Do NOT skip this step. Context dies between sessions; memories are the only
    bridge.
11. If you discover new work needed: `bd create --title="<title>"
    --description="<desc>" --type=task`
12. Append a one-paragraph entry to `.ralph/state/session.md` describing what
    you did, what you learned, and what's next. This is the human's morning
    briefing and the next iteration's handoff note.

## Key files

- **Rust:** `native/angr/src/` (exploration/, interpreter_cb.rs, callbacks.rs,
  state.rs, symbolic/context.rs, symbolic/value.rs)
- **Python:** `angr/exploration/rust_manager.py`, `rust_state_export.py`,
  `rust_state_sync.py`, `rust_state_proxy.py`
- **Tests:** `tests/engines/test_rust_exploration.py`
- **Single runner:** `tests/benchmarks/run_single.py <example> [--engine rust|python] [--both]`
- **Regression:** `tests/benchmarks/run_regression.py`

## Rules

- Never push to remote (no network to github)
- Never skip pre-commit hooks (`--no-verify` is forbidden)
- Run tests after every change
- If a build fails, fix it before moving on
- If tests fail, investigate and fix before closing the task
- Store findings in `bd remember`, not in markdown files
- **CHECKPOINT:** every ~30 minutes of work, commit any working changes (even
  partial) with a WIP commit message. This prevents losing work if you hit a
  rate limit or crash.

## MEMORY SAFETY (8GB machine, no swap)

- **NEVER** run `tests/benchmarks/run_comparison_10.py` — it OOM-kills the
  orchestrator.
- **NEVER** run angr exploration/debug scripts directly (e.g. `python debug_*.py`,
  `python my_script.py`). They run in-process with NO memory limit and WILL
  OOM-kill this machine (no swap). This has happened multiple times — the
  script balloons to 6GB+ and kills the loop agent.
- **ALWAYS** use `run_single.py` to run any angr exploration, even for debugging:
  ```
  python tests/benchmarks/run_single.py <example> --engine rust
  ```
  `run_single.py` spawns a subprocess with `RLIMIT_AS=4GB` — safe even if the
  example OOMs.
- If you need custom debug logic, ADD it to `run_single.py` or write a wrapper
  that uses `subprocess + resource.setrlimit(RLIMIT_AS, 4GB)` — NEVER run
  angr directly in your process.
- Safe quick-check examples: `fauxware`, `ais3_crackme`, `defcamp_r100`.
- AVOID running: `grub` (OOM/crash), `hackcon2016_angry-reverser` (67s), `sym-write` (30s).
- For engine comparison: `python tests/benchmarks/run_single.py <example> --both`.
