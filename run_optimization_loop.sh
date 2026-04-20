#!/bin/bash
# Autonomous rust-symex optimization loop.
# Each iteration is a fresh claude -p session that picks up work from beads.
# Usage: bash run_optimization_loop.sh [max_iterations]

set -euo pipefail

MAX_ITERATIONS=${1:-20}
ITERATION=0
LOG_DIR="/home/ubuntu/repos/angr/.claude/loop-logs"
mkdir -p "$LOG_DIR"

PROMPT='You are continuing autonomous work on the rust-symex optimization epic in /home/ubuntu/repos/angr.

FIRST: Run these setup commands:
  export PATH="$HOME/.cargo/bin:$PATH"
  source /home/ubuntu/repos/angr/.venv/bin/activate
  cd /home/ubuntu/repos/angr
  bd prime

THEN: Check for in-flight work:
  bd list --status=in_progress

If something is in_progress, continue it. Otherwise:
  bd ready
Pick the highest priority unblocked task.

SCOPE: Complete ONE task per session. Only take a second task if it is closely
related to the first (same files, shared context) AND the first task completed
quickly (<15 minutes). This keeps sessions focused and limits blast radius from
rate limits or crashes.

RESUMABILITY: Check .claude/loop-session.md for context from the previous session.
This file contains the task ID, progress notes, and what was being done when the
session ended. Read it BEFORE doing anything else (after setup commands).

Then run git status and git diff --stat. If there are uncommitted changes:
  1. Read .claude/loop-session.md and the diff to understand what was being done
  2. Check bd list --status=in_progress to find the associated task
  3. Decide: compile, test, and commit if the work is complete/useful, OR
     revert if the changes are broken/partial and start fresh
  4. Use your judgment — you have full context the shell script does not

SESSION LOG: Throughout your work, keep .claude/loop-session.md updated with:
  - Task ID and title you are working on
  - Current step (investigating / implementing / testing / committing)
  - Key findings or decisions made so far
  - Files you have modified
This file is your handoff note to the next session. Update it at each major
milestone, not just at the end. Write it as if briefing a colleague who will
pick up exactly where you left off.

For each task, follow this loop:
1. bd update <id> --claim
2. bd show <id> — read the description carefully
3. bd memories — check for relevant invariants/pitfalls before coding
4. Implement the change (edit Rust and/or Python files as described)
5. Build: cargo check --manifest-path native/angr/Cargo.toml --release
6. If Rust changed: pip install -e . --no-build-isolation --no-deps
7. Test: python -m pytest tests/engines/test_rust_exploration.py -v --tb=short
8. If tests pass: git add <changed files> && git commit -m "<description>"
9. bd close <id> --reason="<what was done>"
10. MANDATORY — save what you learned. Run bd remember for EACH of these that applies:
    - Root cause was surprising or non-obvious? Save it. (--key <topic>-root-cause)
    - An approach FAILED before the one that worked? Save why. (--key avoid-<thing>)
    - A constraint/invariant that future work must respect? Save it. (--key invariant-<thing>)
    - Profiling showed the bottleneck was NOT where expected? Save it. (--key <topic>-bottleneck)
    - A benchmark number changed significantly? Save before/after. (--key benchmark-<topic>)
    Do NOT skip this step. Context dies between sessions; memories are the only bridge.
11. If you discover new work needed: bd create --title="<title>" --description="<desc>" --type=task
12. Write a brief session summary to stdout before exiting.

Key files:
- Rust: native/angr/src/ (exploration/, interpreter_cb.rs, callbacks.rs, state.rs, symbolic/context.rs, symbolic/value.rs)
- Python: angr/exploration/rust_manager.py, rust_state_export.py, rust_state_sync.py, rust_state_proxy.py
- Tests: tests/engines/test_rust_exploration.py (85 tests)
- Single runner: tests/benchmarks/run_single.py <example> [--engine rust|python] [--both]
- Regression: tests/benchmarks/run_regression.py

Rules:
- Never push to remote (no network to github)
- Never skip pre-commit hooks
- Run tests after every change
- If a build fails, fix it before moving on
- If tests fail, investigate and fix before closing the task
- Store findings in bd remember, not in markdown files
- CHECKPOINT: Every ~30 minutes of work, commit any working changes (even partial) with a WIP commit message. This prevents losing work if you hit a rate limit or crash.

MEMORY SAFETY (8GB machine, no swap):
- NEVER run tests/benchmarks/run_comparison_10.py — it OOM-kills the orchestrator
- NEVER run angr exploration/debug scripts directly (e.g. python debug_*.py, python my_script.py).
  They run in-process with NO memory limit and WILL OOM-kill this machine (no swap).
  This has happened multiple times — the script balloons to 6GB+ and kills the loop agent.
- ALWAYS use run_single.py to run any angr exploration, even for debugging:
    python tests/benchmarks/run_single.py <example> --engine rust
  run_single.py spawns a subprocess with RLIMIT_AS=4GB — safe even if the example OOMs.
- If you need custom debug logic, ADD it to run_single.py or write a wrapper that uses
  subprocess + resource.setrlimit(RLIMIT_AS, 4GB) — NEVER run angr directly in your process.
- Safe quick-check examples: fauxware, ais3_crackme, defcamp_r100
- AVOID running: grub (OOM/crash), hackcon2016_angry-reverser (67s), sym-write (30s)
- For engine comparison: python tests/benchmarks/run_single.py <example> --both'

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
  ITERATION=$((ITERATION + 1))
  LOGFILE="$LOG_DIR/iteration-${ITERATION}-$(date +%Y%m%d-%H%M%S).log"

  # Check if there's any work to do before burning tokens
  IN_PROGRESS=$(cd /home/ubuntu/repos/angr && bd list --status=in_progress --json 2>/dev/null | jq 'length')
  READY=$(cd /home/ubuntu/repos/angr && bd ready --json 2>/dev/null | jq 'length')

  if [ "$IN_PROGRESS" -eq 0 ] && [ "$READY" -eq 0 ]; then
    echo "=== No tasks in_progress or ready — exiting loop at $(date) ==="
    break
  fi

  echo "=== Iteration $ITERATION/$MAX_ITERATIONS at $(date) (in_progress=$IN_PROGRESS, ready=$READY) ==="
  echo "=== Log: $LOGFILE ==="

  stdbuf -oL claude -p "$PROMPT" \
    --dangerously-skip-permissions \
    --model opus \
    < /dev/null \
    2>&1 | tee "$LOGFILE"

  EXIT_CODE=${PIPESTATUS[0]}

  if [ $EXIT_CODE -eq 0 ]; then
    echo "--- Clean exit at $(date), next iteration in 10s ---"
    sleep 10
  else
    echo "--- Error exit ($EXIT_CODE) at $(date), backing off 60s ---"
    sleep 60
  fi
done

echo "=== Loop complete: $ITERATION iterations ==="
echo "=== Check progress: cd /home/ubuntu/repos/angr && bd stats ==="
