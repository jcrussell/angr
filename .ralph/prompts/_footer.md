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

See the **Key Files** section in `CLAUDE.md` (already in your context).

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
- **Hard turn cap:** complete each iteration in ≤80 tool calls. If you're
  past 60 turns and still investigating, commit a WIP and stop — let the next
  iter pick up. Do NOT keep reading files searching for context.
- **Log handling:** do NOT cat full logs or test outputs. Pipe through
  `tail -200`, `head -200`, or `grep -C3 PATTERN`. Tests:
  `pytest -q --tb=short`, never `-v --tb=long` in this loop.
- **Don't re-read injected context:** CLAUDE.md, _footer.md, and session.md
  were injected at iter start. Do not re-open them mid-iteration.
- **Output discipline:** no recap of completed steps, no narrative summaries
  between tool calls. End-of-turn: ≤2 sentences (what changed, what's next).

## MEMORY SAFETY (8GB box, no swap — these will OOM the loop)

- **NEVER** run angr scripts in-process. Use
  `python tests/benchmarks/run_single.py <example> --engine rust` (subprocess
  + `RLIMIT_AS=4GB`). For custom debug logic, add it to `run_single.py` or
  wrap with `resource.setrlimit(RLIMIT_AS, 4GB)`.
- **NEVER** run `tests/benchmarks/run_comparison_10.py`.
- Safe quick benches: `fauxware`, `ais3_crackme`, `defcamp_r100`.
- Avoid: `grub` (OOM), `hackcon2016_angry-reverser` (67s), `sym-write` (30s).
- Engine comparison: `python tests/benchmarks/run_single.py <example> --both`.
