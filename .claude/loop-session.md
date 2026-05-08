## Session log: 2026-05-08, 156th loop session

### Task: angr-0xho — CI loop-orchestrator smoke test

Goal: Add CI coverage so orchestrator regressions (broken benchmark gate,
stale-orchestrator bytecode, rate-limit classification gaps) are caught
before they burn loop budget.

### Changes
1. run_optimization_loop.py:875-892 — moved --dry-run check before prereq
   verification, systemd-run probe, pidfile acquisition, and stale scope
   cleanup. Dry-run now exercises ONLY: module imports, argparse, logging,
   signal handlers, detect_git_state, build_prompt. No side effects.
2. .github/workflows/ci.yml — added orchestrator_smoke job (ubuntu-latest)
   running --dry-run with header/length assertions plus --stress-test --dry-run
   to validate the stress-test preset path.

### Status
done — committed (b9be0325d) and closed.

322/322 tests pass. Local dry-run + stress-test+dry-run both exit 0.
Saved invariant-orchestrator-dryrun-side-effects memory.

### Coverage gap (deferred)
The smoke test does NOT exercise the iteration loop, claude subprocess
invocation, or actual systemd-run cgroup creation. Catching those would
require mocking the claude CLI in CI — future work.
