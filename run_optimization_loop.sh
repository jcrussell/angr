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
11. If you discover new work needed: bd create --title="<title>" --description="<desc>" --type=task --parent=angr-34w
12. If budget remains, pick the next task from bd ready

IMPORTANT: Read the detailed plan before implementing:
  cat /home/ubuntu/.claude/plans/fluffy-snacking-music.md
It has specific file paths, line numbers, and implementation details for each task.

Key files:
- Plan: /home/ubuntu/.claude/plans/fluffy-snacking-music.md
- Rust: native/angr/src/ (exploration.rs, interpreter_cb.rs, callbacks.rs, state.rs, concretize.rs)
- Python: angr/exploration/rust_manager.py (5059 lines), rust_state_proxy.py, rust_techniques.py
- Tests: tests/engines/test_rust_exploration.py (14 tests)
- Single runner: tests/benchmarks/run_single.py <example> [--engine rust|python] [--both]

Rules:
- Never push to remote (no network to github)
- Never skip pre-commit hooks
- Run tests after every change
- If a build fails, fix it before moving on
- If tests fail, investigate and fix before closing the task
- Store findings in bd remember, not in markdown files

MEMORY SAFETY (8GB machine, no swap):
- NEVER run tests/benchmarks/run_comparison_10.py — it OOM-kills the orchestrator
- For benchmarking use: python tests/benchmarks/run_single.py <example> --engine rust
- Safe quick-check examples: fauxware, ais3_crackme, defcamp_r100
- AVOID running: grub (OOM/crash), hackcon2016_angry-reverser (67s), sym-write (30s)
- For engine comparison: python tests/benchmarks/run_single.py <example> --both
- run_single.py runs in a subprocess with 4GB memory limit — safe even if example OOMs'

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
  ITERATION=$((ITERATION + 1))
  LOGFILE="$LOG_DIR/iteration-${ITERATION}-$(date +%Y%m%d-%H%M%S).log"

  echo "=== Iteration $ITERATION/$MAX_ITERATIONS at $(date) ==="
  echo "=== Log: $LOGFILE ==="

  claude -p "$PROMPT" \
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
echo "=== Check progress: cd /home/ubuntu/repos/angr && bd status ==="
