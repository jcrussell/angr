#!/usr/bin/env python3
"""Autonomous optimization loop for angr rust-symex with OOM isolation.

Replaces run_optimization_loop.sh with:
- cgroup-based OOM isolation via systemd-run --user --scope
- 60-minute session timeouts
- Structured JSON logging per iteration
- Benchmark regression gate after commits
- Rate-limit detection with smart backoff
- Clean vs dirty state prompts

Usage:
    python run_optimization_loop.py                    # Run with defaults (30 iterations)
    python run_optimization_loop.py --max-iterations 5 # Quick run
    python run_optimization_loop.py --dry-run          # Print prompt, don't run claude
    python run_optimization_loop.py --skip-benchmarks  # Skip regression gate
    python run_optimization_loop.py --timeout 300      # 5-min timeout (for testing)
    python run_optimization_loop.py --memory-limit 256M # Low limit (for OOM testing)
"""

import argparse
import datetime
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ── Constants ──────────────────────────────────────────────────────────────────

REPO_DIR = Path("/home/ubuntu/repos/angr")
LOG_DIR = REPO_DIR / ".claude" / "loop-logs-v2"
SESSION_FILE = REPO_DIR / ".claude" / "loop-session.md"
VENV_PYTHON = REPO_DIR / ".venv" / "bin" / "python"
CARGO_BIN = Path.home() / ".cargo" / "bin"

DEFAULT_MEMORY_LIMIT = "7G"
DEFAULT_TIMEOUT_SECS = 3600  # 60 minutes
DEFAULT_MAX_ITERATIONS = 30
RATE_LIMIT_BASE_BACKOFF = 300  # 5 minutes
MAX_BACKOFF = 4800  # 80 minutes

CLAUDE_MODEL = "opus"

log = logging.getLogger("loop")

# Track current systemd scope for signal handler cleanup
_current_scope_unit: Optional[str] = None


# ── Prompts ────────────────────────────────────────────────────────────────────

PROMPT_FOOTER = r'''
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
- Tests: tests/engines/test_rust_exploration.py (146 tests)
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
- For engine comparison: python tests/benchmarks/run_single.py <example> --both
'''

PROMPT_CLEAN_OPENING = r'''You are continuing autonomous work on the rust-symex optimization epic in /home/ubuntu/repos/angr.

FIRST: Run these setup commands:
  export PATH="$HOME/.cargo/bin:$PATH"
  source /home/ubuntu/repos/angr/.venv/bin/activate
  cd /home/ubuntu/repos/angr
  bd prime

Read .claude/loop-session.md for context from the previous session.

THEN: Check for in-flight work:
  bd list --status=in_progress

If something is in_progress, continue it. Otherwise:
  bd ready
Pick the highest priority unblocked task.

SCOPE: Complete ONE task per session. Only take a second task if it is closely
related to the first (same files, shared context) AND the first task completed
quickly (<15 minutes). This keeps sessions focused and limits blast radius from
rate limits or crashes.

SESSION LOG: Throughout your work, keep .claude/loop-session.md updated with:
  - Task ID and title you are working on
  - Current step (investigating / implementing / testing / committing)
  - Key findings or decisions made so far
  - Files you have modified
This file is your handoff note to the next session. Update it at each major
milestone, not just at the end. Write it as if briefing a colleague who will
pick up exactly where you left off.
'''

PROMPT_DIRTY_OPENING = r'''You are continuing autonomous work on the rust-symex optimization epic in /home/ubuntu/repos/angr.
The previous session left UNCOMMITTED CHANGES. Your first priority is to assess and resolve them.

FIRST: Run these setup commands:
  export PATH="$HOME/.cargo/bin:$PATH"
  source /home/ubuntu/repos/angr/.venv/bin/activate
  cd /home/ubuntu/repos/angr
  bd prime

THEN: Assess the uncommitted changes:
  1. Read .claude/loop-session.md — understand what the previous session was doing
  2. Run: git status && git diff --stat
  3. Run: bd list --status=in_progress — find the associated task
  4. Try to build: cargo check --manifest-path native/angr/Cargo.toml --release

DECISION MATRIX (spend at most 15 minutes on this):
  - If it compiles AND tests pass → finish the work, commit, close the task
  - If it compiles but tests fail → investigate briefly, fix if straightforward or revert
  - If it does NOT compile → attempt fix (max 10 min), then revert if stuck
  - If changes are clearly broken with no clear path → git checkout -- . && git clean -fd
  After resolution, proceed to pick a new task from bd ready.

SESSION LOG: Throughout your work, keep .claude/loop-session.md updated with:
  - Task ID and title you are working on
  - Current step (assessing dirty state / implementing / testing / committing)
  - Key findings or decisions made so far
  - Files you have modified
This file is your handoff note to the next session.
'''


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_cmd(cmd: list[str], cwd: Optional[Path] = None, timeout: int = 30) -> tuple[int, str, str]:
    """Run a command and return (exit_code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=cwd or REPO_DIR, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError:
        return -2, "", f"command not found: {cmd[0]}"


def monitor_memory() -> float:
    """Return orchestrator RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except (OSError, ValueError):
        pass
    return 0.0


def detect_git_state() -> dict:
    """Detect current git state: dirty files, HEAD sha, etc."""
    _, status_out, _ = _run_cmd(["git", "status", "--porcelain"])
    _, diff_out, _ = _run_cmd(["git", "diff", "--stat"])
    _, log_out, _ = _run_cmd(["git", "log", "-1", "--format=%H %s"])

    dirty_files = [line.strip() for line in status_out.strip().splitlines() if line.strip()]
    # Filter out untracked files that are always there (debug scripts, .claude, etc.)
    modified_files = [f for f in dirty_files if not f.startswith("??")]

    parts = log_out.strip().split(" ", 1) if log_out.strip() else ["", ""]
    return {
        "is_dirty": len(modified_files) > 0,
        "dirty_file_count": len(modified_files),
        "dirty_files_summary": status_out.strip()[:500],
        "diff_stat": diff_out.strip()[:500],
        "head_sha": parts[0][:12],
        "head_msg": parts[1] if len(parts) > 1 else "",
    }


def check_tasks_available() -> dict:
    """Check beads for available work."""
    _, ip_out, _ = _run_cmd(["bd", "list", "--status=in_progress", "--json"])
    _, ready_out, _ = _run_cmd(["bd", "ready", "--json"])

    def count_json(s):
        try:
            return len(json.loads(s))
        except (json.JSONDecodeError, TypeError):
            return 0

    return {
        "in_progress": count_json(ip_out),
        "ready": count_json(ready_out),
    }


def build_prompt(git_state: dict) -> str:
    """Build the appropriate prompt based on clean vs dirty repo state."""
    if git_state["is_dirty"]:
        opening = PROMPT_DIRTY_OPENING
    else:
        opening = PROMPT_CLEAN_OPENING
    return opening + PROMPT_FOOTER


def detect_rate_limit(stdout: str, stderr: str) -> tuple[bool, Optional[float]]:
    """Detect rate limit and return (is_limited, seconds_until_reset_or_None)."""
    combined = stdout + stderr
    if "hit your limit" not in combined.lower() and "rate limit" not in combined.lower():
        return False, None

    # Try to parse reset time: "resets 4am (UTC)" or "resets 8am (UTC)"
    match = re.search(r"resets\s+(\d{1,2})\s*(am|pm)\s*\(UTC\)", combined, re.IGNORECASE)
    if match:
        hour = int(match.group(1))
        if match.group(2).lower() == "pm" and hour != 12:
            hour += 12
        elif match.group(2).lower() == "am" and hour == 12:
            hour = 0
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        reset = now_utc.replace(hour=hour, minute=0, second=0, microsecond=0)
        if reset <= now_utc:
            reset += datetime.timedelta(days=1)
        secs = (reset - now_utc).total_seconds() + 60  # 60s buffer
        return True, secs

    return True, None


def detect_oom(exit_code: int, stderr: str, scope_unit: Optional[str]) -> bool:
    """Detect if the session was killed by OOM."""
    if exit_code in (137, 143, -9, -15, 9, 15):
        return True
    if "killed" in stderr.lower() or "oom" in stderr.lower():
        return True
    # Check cgroup memory.events if we know the scope unit
    if scope_unit:
        events_path = f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/{scope_unit}/memory.events"
        try:
            with open(events_path) as f:
                for line in f:
                    if line.startswith("oom_kill") and int(line.split()[1]) > 0:
                        return True
        except (OSError, ValueError, IndexError):
            pass
    return False


def calculate_backoff(rate_limited: bool, rate_limit_reset_secs: Optional[float],
                      killed_by_oom: bool, killed_by_timeout: bool,
                      exit_code: int, consecutive_failures: int) -> int:
    """Calculate backoff seconds."""
    if rate_limited:
        if rate_limit_reset_secs is not None:
            return min(int(rate_limit_reset_secs), MAX_BACKOFF)
        # Exponential backoff: 5m, 10m, 20m, 40m, 80m
        return min(RATE_LIMIT_BASE_BACKOFF * (2 ** min(consecutive_failures, 4)), MAX_BACKOFF)
    if killed_by_oom:
        return 30
    if killed_by_timeout:
        return 60
    if exit_code != 0:
        return 60
    return 10  # Clean exit


# ── Core Functions ─────────────────────────────────────────────────────────────

def run_claude_session(prompt: str, iteration: int, timeout: int,
                       memory_limit: str) -> dict:
    """Launch a claude -p session inside a cgroup scope with timeout."""
    global _current_scope_unit

    # Use predictable scope name so we always know how to kill it (fixes F1/F2)
    scope_name = f"angr-loop-iter{iteration}"
    scope_unit = f"{scope_name}.scope"
    _current_scope_unit = scope_unit

    cmd = [
        "systemd-run", "--user", "--scope",
        f"--unit={scope_name}",
        "-p", f"MemoryMax={memory_limit}",
        "-p", "MemorySwapMax=0",
        "--",
        "claude", "-p", prompt,
        "--dangerously-skip-permissions",
        "--model", CLAUDE_MODEL,
    ]

    env = os.environ.copy()
    env["PATH"] = f"{CARGO_BIN}:{env['PATH']}"
    # Ensure virtualenv is active
    env["VIRTUAL_ENV"] = str(REPO_DIR / ".venv")
    env["PATH"] = f"{REPO_DIR / '.venv' / 'bin'}:{env['PATH']}"

    log.info(f"Starting claude session (scope={scope_unit}, memory_limit={memory_limit}, timeout={timeout}s)")
    start = time.monotonic()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        cwd=str(REPO_DIR),
        env=env,
    )

    killed_by_timeout = False
    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        killed_by_timeout = True
        log.warning(f"Session timed out after {timeout}s, killing scope {scope_unit}")

        # Kill the entire cgroup scope — we know the name because we chose it
        try:
            subprocess.run(
                ["systemctl", "--user", "kill", scope_unit],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
        proc.terminate()
        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_bytes, stderr_bytes = proc.communicate(timeout=5)
    finally:
        _current_scope_unit = None

    duration = time.monotonic() - start
    stdout_str = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr_str = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

    exit_code = proc.returncode
    rate_limited, rate_reset = detect_rate_limit(stdout_str, stderr_str)
    killed_by_oom = detect_oom(exit_code, stderr_str, scope_unit)

    # Reset the finished scope so it doesn't linger as "failed" in systemd
    try:
        subprocess.run(
            ["systemctl", "--user", "reset-failed", scope_unit],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass

    # Try to parse JSON output from claude
    claude_json = None
    try:
        claude_json = json.loads(stdout_str)
    except (json.JSONDecodeError, TypeError):
        pass

    result = {
        "exit_code": exit_code,
        "duration_secs": round(duration, 1),
        "killed_by_timeout": killed_by_timeout,
        "killed_by_oom": killed_by_oom,
        "rate_limited": rate_limited,
        "rate_limit_reset_secs": rate_reset,
        "scope_unit": scope_unit,
        "stdout_tail": stdout_str[-2000:] if len(stdout_str) > 2000 else stdout_str,
        "stderr_tail": stderr_str[-1000:] if len(stderr_str) > 1000 else stderr_str,
    }

    # Extract cost info from claude JSON output if available
    if claude_json and isinstance(claude_json, dict):
        result["claude_cost_usd"] = claude_json.get("cost_usd")
        result["claude_input_tokens"] = claude_json.get("input_tokens")
        result["claude_output_tokens"] = claude_json.get("output_tokens")
        result["claude_num_turns"] = claude_json.get("num_turns")

    return result


def run_benchmark_gate(memory_limit: str) -> dict:
    """Run regression suite in a cgroup scope. Returns result dict."""
    log.info("Running benchmark regression gate...")
    start = time.monotonic()

    cmd = [
        "systemd-run", "--user", "--scope",
        "-p", f"MemoryMax={memory_limit}",
        "-p", "MemorySwapMax=0",
        "--",
        str(VENV_PYTHON), str(REPO_DIR / "tests" / "benchmarks" / "run_regression.py"),
        "--rust-only",
    ]

    env = os.environ.copy()
    env["PATH"] = f"{CARGO_BIN}:{REPO_DIR / '.venv' / 'bin'}:{env['PATH']}"
    env["VIRTUAL_ENV"] = str(REPO_DIR / ".venv")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(REPO_DIR), env=env,
            timeout=600,  # 10 min max for benchmarks
        )
        exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        return {
            "ran": True, "exit_code": -1, "passed": 0, "failed": 0,
            "regressions": ["benchmark gate timed out after 600s"],
            "duration_secs": 600, "output": "",
        }

    duration = round(time.monotonic() - start, 1)

    # Parse results from stdout
    passed = 0
    failed = 0
    regressions = []

    # Look for summary line: "Benchmarks: 7, Passed: 7, Failed: 0"
    match = re.search(r"Passed:\s*(\d+).*Failed:\s*(\d+)", stdout)
    if match:
        passed = int(match.group(1))
        failed = int(match.group(2))

    # Collect failure lines
    in_failures = False
    for line in stdout.splitlines():
        if line.strip().startswith("FAILURES:"):
            in_failures = True
            continue
        if in_failures and line.strip().startswith("- "):
            regressions.append(line.strip()[2:])

    # Also check for REGRESSION lines
    for line in stdout.splitlines():
        if "REGRESSION:" in line:
            regressions.append(line.strip())

    return {
        "ran": True,
        "exit_code": exit_code,
        "passed": passed,
        "failed": failed,
        "regressions": regressions,
        "duration_secs": duration,
        "output": stdout[-2000:],
    }


def log_iteration(iteration: int, git_before: dict, git_after: dict,
                   session: dict, prompt_type: str, tasks: dict,
                   benchmark: Optional[dict], backoff: int,
                   orchestrator_rss: float) -> None:
    """Write structured JSON log for this iteration."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    entry = {
        "iteration": iteration,
        "timestamp": ts,
        "duration_secs": session["duration_secs"],
        "prompt_type": prompt_type,
        "session": {
            "exit_code": session["exit_code"],
            "killed_by_oom": session["killed_by_oom"],
            "killed_by_timeout": session["killed_by_timeout"],
            "rate_limited": session["rate_limited"],
            "cost_usd": session.get("claude_cost_usd"),
            "num_turns": session.get("claude_num_turns"),
        },
        "commits_made": git_before["head_sha"] != git_after["head_sha"],
        "git_head_before": git_before["head_sha"],
        "git_head_after": git_after["head_sha"],
        "git_dirty_after": git_after["is_dirty"],
        "benchmark_gate": benchmark if benchmark else {"ran": False},
        "tasks": tasks,
        "orchestrator_rss_mb": round(orchestrator_rss, 1),
        "backoff_secs": backoff,
    }

    # Append to summary JSONL
    summary_file = LOG_DIR / "summary.jsonl"
    with open(summary_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Write full per-iteration log (includes claude output)
    ts_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    detail_file = LOG_DIR / f"iter-{iteration:04d}-{ts_file}.json"
    full_entry = {**entry, "session_stdout_tail": session.get("stdout_tail", ""),
                  "session_stderr_tail": session.get("stderr_tail", "")}
    with open(detail_file, "w") as f:
        json.dump(full_entry, f, indent=2)

    log.info(f"Logged iteration {iteration} to {detail_file.name}")


# ── Signal Handling ────────────────────────────────────────────────────────────

def _setup_signal_handlers():
    """Kill child scope on SIGTERM/SIGINT for clean shutdown."""
    def handler(signum, frame):
        log.info(f"Received signal {signum}, cleaning up...")
        if _current_scope_unit:
            try:
                subprocess.run(
                    ["systemctl", "--user", "kill", _current_scope_unit],
                    capture_output=True, timeout=5,
                )
            except Exception:
                pass
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _cleanup_stale_scopes():
    """Kill active and reset failed angr-loop-* scopes from previous runs."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "list-units", "--type=scope",
             "--no-legend", "--no-pager"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if not parts:
                continue
            unit = parts[0]
            if not unit.startswith("angr-loop-"):
                continue
            # Determine state: "active" means processes still running, "failed" means exited
            state = parts[3] if len(parts) > 3 else ""
            if state == "failed":
                log.info(f"Resetting failed scope: {unit}")
                try:
                    subprocess.run(
                        ["systemctl", "--user", "reset-failed", unit],
                        capture_output=True, timeout=5,
                    )
                except Exception:
                    pass
            else:
                log.warning(f"Killing stale active scope: {unit}")
                try:
                    subprocess.run(
                        ["systemctl", "--user", "kill", unit],
                        capture_output=True, timeout=5,
                    )
                except Exception:
                    pass
    except Exception as e:
        log.warning(f"Stale scope cleanup failed: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous rust-symex optimization loop with OOM isolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
                        help=f"Maximum loop iterations (default: {DEFAULT_MAX_ITERATIONS})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECS,
                        help=f"Per-session timeout in seconds (default: {DEFAULT_TIMEOUT_SECS})")
    parser.add_argument("--memory-limit", default=DEFAULT_MEMORY_LIMIT,
                        help=f"cgroup MemoryMax for sessions (default: {DEFAULT_MEMORY_LIMIT})")
    parser.add_argument("--skip-benchmarks", action="store_true",
                        help="Skip benchmark regression gate after commits")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the prompt that would be used and exit")
    args = parser.parse_args()

    # Setup logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(LOG_DIR / "orchestrator.log"),
        ],
    )

    _setup_signal_handlers()

    # Verify prerequisites
    for cmd_name in ["claude", "bd", "git", "systemd-run"]:
        rc, _, _ = _run_cmd(["which", cmd_name])
        if rc != 0:
            log.error(f"Required command not found: {cmd_name}")
            sys.exit(2)

    # Verify systemd-run --user works
    rc, _, stderr = _run_cmd(["systemd-run", "--user", "--scope", "true"])
    if rc != 0:
        log.error(f"systemd-run --user --scope failed: {stderr}")
        log.error("cgroup isolation not available. Cannot proceed safely on 8GB/0-swap machine.")
        sys.exit(2)

    # Clean up stale scopes from previous crashes (F4)
    _cleanup_stale_scopes()

    # Dry run mode
    if args.dry_run:
        git_state = detect_git_state()
        prompt = build_prompt(git_state)
        prompt_type = "dirty" if git_state["is_dirty"] else "clean"
        print(f"--- Prompt type: {prompt_type} ---")
        print(f"--- Git state: dirty={git_state['is_dirty']}, HEAD={git_state['head_sha']} ---")
        print(prompt)
        return

    log.info(f"Starting optimization loop: max_iterations={args.max_iterations}, "
             f"timeout={args.timeout}s, memory_limit={args.memory_limit}")

    consecutive_failures = 0
    iteration = 0

    for iteration in range(1, args.max_iterations + 1):
        rss = monitor_memory()
        log.info(f"{'='*60}")
        log.info(f"Iteration {iteration}/{args.max_iterations} | RSS={rss:.1f}MB")

        # Check task availability
        tasks = check_tasks_available()
        log.info(f"Tasks: in_progress={tasks['in_progress']}, ready={tasks['ready']}")
        if tasks["in_progress"] == 0 and tasks["ready"] == 0:
            log.info("No tasks available. Exiting loop.")
            break

        # Detect git state and choose prompt
        git_before = detect_git_state()
        prompt = build_prompt(git_before)
        prompt_type = "dirty" if git_before["is_dirty"] else "clean"
        log.info(f"Prompt: {prompt_type} | HEAD: {git_before['head_sha']} {git_before['head_msg'][:60]}")
        if git_before["is_dirty"]:
            log.info(f"Dirty files ({git_before['dirty_file_count']}): {git_before['dirty_files_summary'][:200]}")

        # Run claude session in cgroup scope
        session = run_claude_session(prompt, iteration, args.timeout, args.memory_limit)
        log.info(f"Session done: exit={session['exit_code']}, duration={session['duration_secs']}s, "
                 f"oom={session['killed_by_oom']}, timeout={session['killed_by_timeout']}, "
                 f"rate_limited={session['rate_limited']}")

        # Detect git state after
        git_after = detect_git_state()
        commits_made = git_before["head_sha"] != git_after["head_sha"]
        if commits_made:
            log.info(f"New commits: {git_before['head_sha']} -> {git_after['head_sha']} ({git_after['head_msg'][:60]})")

        # Benchmark gate
        benchmark = None
        if commits_made and not args.skip_benchmarks:
            benchmark = run_benchmark_gate(args.memory_limit)
            if benchmark["failed"] > 0:
                log.warning(f"BENCHMARK REGRESSIONS ({benchmark['failed']}): {benchmark['regressions']}")
            else:
                log.info(f"Benchmark gate passed: {benchmark['passed']} benchmarks OK in {benchmark['duration_secs']}s")

        # Classify outcome and backoff
        backoff = calculate_backoff(
            session["rate_limited"], session.get("rate_limit_reset_secs"),
            session["killed_by_oom"], session["killed_by_timeout"],
            session["exit_code"], consecutive_failures,
        )

        if session["exit_code"] == 0 and not session["rate_limited"]:
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        # Log structured JSON
        try:
            log_iteration(iteration, git_before, git_after, session, prompt_type,
                          tasks, benchmark, backoff, rss)
        except OSError as e:
            log.warning(f"Failed to write iteration log: {e}")

        # Backoff
        if backoff > 10:
            log.info(f"Backing off {backoff}s ({backoff/60:.0f}m) before next iteration...")
        time.sleep(backoff)

    log.info(f"Loop complete after {iteration} iterations.")
    log.info(f"Review logs: cat {LOG_DIR / 'summary.jsonl'} | python -m json.tool --no-ensure-ascii")


if __name__ == "__main__":
    main()
