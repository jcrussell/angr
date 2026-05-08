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
import glob
import hashlib
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
DEAD_SESSION_DURATION_SECS = 10  # exit_code != 0 + duration < this counts as dead
DEAD_SESSION_STREAK_THRESHOLD = 3  # consecutive dead sessions before long backoff
FAILED_OUTPUT_RETENTION = 20  # keep stdout/stderr for last N failed sessions

# Sentinel returned by calculate_backoff to signal "exit the loop, don't sleep"
BACKOFF_EXIT = -1

CLAUDE_MODEL = "opus"
DIRTY_REVERT_THRESHOLD = 3  # Auto-revert after this many consecutive dirty iterations
PID_FILE = Path("/home/ubuntu/repos/angr/.claude/loop.pid")

log = logging.getLogger("loop")

# Track current systemd scope for signal handler cleanup
_current_scope_unit: Optional[str] = None
# Unique run ID for scope names (prevents cross-run collisions)
_run_id: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


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


def _orchestrator_source_hash() -> Optional[str]:
    """SHA256 of run_optimization_loop.py on disk (12-char prefix).

    Returned None on read failure. Used to detect when the running orchestrator
    process is stale relative to the current source file (e.g. the fix in
    angr-lbze was applied to disk but the long-running loop kept executing the
    old in-memory bytecode for 28+ iterations, silently masking gate failures).
    """
    try:
        with open(__file__, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except OSError:
        return None


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


def _parse_reset_seconds(text: str) -> Optional[float]:
    """Parse 'resets 4am (UTC)' style text into seconds-until-reset."""
    match = re.search(r"resets\s+(\d{1,2})\s*(am|pm)\s*\(UTC\)", text, re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group(1))
    if match.group(2).lower() == "pm" and hour != 12:
        hour += 12
    elif match.group(2).lower() == "am" and hour == 12:
        hour = 0
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    reset = now_utc.replace(hour=hour, minute=0, second=0, microsecond=0)
    if reset <= now_utc:
        reset += datetime.timedelta(days=1)
    return (reset - now_utc).total_seconds() + 60  # 60s buffer


def detect_failure_mode(claude_json: Optional[dict],
                        exit_code: int) -> tuple[str, Optional[float]]:
    """Classify a session outcome from the structured JSON envelope.

    Returns (mode, reset_secs_or_None) where mode is one of:
        ok, rate_limit, budget_exhausted, auth_error, model_overloaded,
        unknown_error
    """
    if not isinstance(claude_json, dict):
        # JSON parse failed (CLI crashed before emitting result, or non-JSON output)
        if exit_code == 0:
            return "ok", None
        return "unknown_error", None

    is_error = bool(claude_json.get("is_error"))
    subtype = str(claude_json.get("subtype") or "").lower()

    if not is_error:
        return "ok", None

    # Build a haystack from subtype + errors[] + result for substring matching.
    # The CLI puts monthly-limit messages in `result` (e.g. "You've hit your
    # org's monthly usage limit") with subtype="success" + is_error=true, so
    # the errors[] list is empty in that path.
    errors_list = claude_json.get("errors") or []
    errors_text = " ".join(str(e) for e in errors_list).lower()
    result_text = str(claude_json.get("result") or "").lower()
    api_error_status = claude_json.get("api_error_status")
    haystack = f"{subtype} {errors_text} {result_text}"

    # Monthly usage limit comes back as api_error_status=429 with the text in
    # `result`. Distinguish from burst rate-limit 429s, which are recoverable.
    if "monthly" in haystack and "limit" in haystack:
        return "budget_exhausted", None
    if subtype == "error_max_budget_usd" or "budget" in subtype:
        return "budget_exhausted", None
    if api_error_status == 429:
        return "rate_limit", _parse_reset_seconds(haystack)
    if "rate" in haystack or ("limit" in haystack and "budget" not in subtype):
        return "rate_limit", _parse_reset_seconds(haystack)
    if "auth" in haystack or "credential" in haystack:
        return "auth_error", None
    if "overload" in haystack:
        return "model_overloaded", None
    if subtype.startswith("error_") or is_error:
        return "unknown_error", None
    return "ok", None


def derive_exit_reason(session: dict) -> str:
    """Granular outcome label for telemetry (alongside `mode`).

    Distinguishes terminal vs recoverable 429s, OOM, timeout, dead-loop hangs,
    and clean exits. Read from summary.jsonl to investigate loop behavior.
    """
    mode = session.get("mode", "unknown_error")
    if mode == "ok":
        return "ok"
    if mode == "budget_exhausted":
        return "budget_exhausted"
    if mode == "rate_limit":
        return "rate_limited"
    if mode == "auth_error":
        return "auth_error"
    if mode == "model_overloaded":
        return "model_overloaded"
    if session.get("killed_by_oom"):
        return "oom"
    if session.get("killed_by_timeout"):
        return "timeout"
    return "unknown_error"


_BEAD_ID_RE = re.compile(r"\bangr-[a-z0-9]+(?:\.\d+)?\b")


def extract_bead_id(git_after: dict, git_before: dict) -> Optional[str]:
    """Best-effort: pull the bead the session was working on.

    Priority:
    1. New HEAD commit message (only when commits were made this iteration).
    2. `## Task: <bead_id>` line from .claude/loop-session.md (works even
       without commits, since the prompt asks the agent to update it).
    """
    if git_before.get("head_sha") != git_after.get("head_sha"):
        msg = git_after.get("head_msg") or ""
        match = _BEAD_ID_RE.search(msg)
        if match:
            return match.group(0)

    try:
        text = SESSION_FILE.read_text(errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Task:") or stripped.startswith("## Task "):
            match = _BEAD_ID_RE.search(stripped)
            if match:
                return match.group(0)
    return None


def detect_oom(exit_code: int, stderr: str, scope_unit: Optional[str]) -> bool:
    """Detect if the session was killed by OOM.

    Only returns True for confirmed OOM kills (cgroup oom_kill event or
    exit code 137/-9). Exit 143 is NOT assumed to be OOM — it can also be
    Claude CLI self-termination or other SIGTERM sources.
    """
    if exit_code in (137, -9):
        return True
    if "oom" in stderr.lower():
        return True
    # Check cgroup memory.events for actual oom_kill count
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


def calculate_backoff(mode: str, rate_limit_reset_secs: Optional[float],
                      killed_by_oom: bool, killed_by_timeout: bool,
                      exit_code: int, consecutive_failures: int,
                      short_dead_streak: int) -> int:
    """Calculate backoff seconds. Returns BACKOFF_EXIT to signal terminal failure."""
    if mode in ("budget_exhausted", "auth_error"):
        return BACKOFF_EXIT
    if mode == "rate_limit":
        if rate_limit_reset_secs is not None:
            return min(int(rate_limit_reset_secs), MAX_BACKOFF)
        return min(RATE_LIMIT_BASE_BACKOFF * (2 ** min(consecutive_failures, 4)), MAX_BACKOFF)
    if mode == "model_overloaded":
        return RATE_LIMIT_BASE_BACKOFF
    if killed_by_oom:
        return 30
    if killed_by_timeout:
        return 60
    if mode == "unknown_error" and short_dead_streak >= DEAD_SESSION_STREAK_THRESHOLD:
        # Safety net: structured detection missed something; treat like rate limit
        return min(RATE_LIMIT_BASE_BACKOFF * (2 ** min(short_dead_streak - DEAD_SESSION_STREAK_THRESHOLD, 4)), MAX_BACKOFF)
    if exit_code != 0:
        return 60
    return 10  # Clean exit


# ── Core Functions ─────────────────────────────────────────────────────────────

def run_claude_session(prompt: str, iteration: int, timeout: int,
                       memory_limit: str) -> dict:
    """Launch a claude -p session inside a cgroup scope with timeout."""
    global _current_scope_unit

    # Use predictable scope name with run_id to avoid cross-run collisions
    scope_name = f"angr-loop-{_run_id}-iter{iteration}"
    scope_unit = f"{scope_name}.scope"
    _current_scope_unit = scope_unit

    cmd = [
        "systemd-run", "--user", "--scope",
        f"--unit={scope_name}",
        "-p", f"MemoryMax={memory_limit}",
        "-p", "MemorySwapMax=0",
        "--",
        "claude", "-p", prompt,
        "--output-format=json",
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
    killed_by_oom = detect_oom(exit_code, stderr_str, scope_unit)

    # Reset the finished scope so it doesn't linger as "failed" in systemd
    try:
        subprocess.run(
            ["systemctl", "--user", "reset-failed", scope_unit],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass

    # Parse the structured JSON envelope (--output-format=json). On crash before
    # emission, claude_json stays None and detect_failure_mode falls through to
    # unknown_error, which the dead-session heuristic in calculate_backoff catches.
    claude_json = None
    try:
        claude_json = json.loads(stdout_str)
    except (json.JSONDecodeError, TypeError):
        pass

    mode, rate_reset = detect_failure_mode(claude_json, exit_code)

    result = {
        "exit_code": exit_code,
        "duration_secs": round(duration, 1),
        "killed_by_timeout": killed_by_timeout,
        "killed_by_oom": killed_by_oom,
        "mode": mode,
        "rate_limited": (mode == "rate_limit"),  # derived alias for backwards compat
        "rate_limit_reset_secs": rate_reset,
        "scope_unit": scope_unit,
        "stdout_tail": stdout_str[-2000:] if len(stdout_str) > 2000 else stdout_str,
        "stderr_tail": stderr_str[-1000:] if len(stderr_str) > 1000 else stderr_str,
        "_stdout_full": stdout_str,
        "_stderr_full": stderr_str,
    }

    # Extract cost info from claude JSON output if available
    if claude_json and isinstance(claude_json, dict):
        # The new envelope uses total_cost_usd; older builds used cost_usd
        result["claude_cost_usd"] = claude_json.get("total_cost_usd") or claude_json.get("cost_usd")
        usage = claude_json.get("usage") or {}
        result["claude_input_tokens"] = usage.get("input_tokens") or claude_json.get("input_tokens")
        result["claude_output_tokens"] = usage.get("output_tokens") or claude_json.get("output_tokens")
        result["claude_num_turns"] = claude_json.get("num_turns")
        result["claude_subtype"] = claude_json.get("subtype")
        result["claude_api_error_status"] = claude_json.get("api_error_status")

    return result


def persist_failed_session_output(iteration: int, session: dict) -> None:
    """Write stdout/stderr of a failed session to disk for later debugging.

    Caps retention at FAILED_OUTPUT_RETENTION sessions to bound disk usage.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    stdout_path = LOG_DIR / f"iter-{iteration:04d}-{ts}-stdout.txt"
    stderr_path = LOG_DIR / f"iter-{iteration:04d}-{ts}-stderr.txt"

    try:
        stdout_path.write_text(session.get("_stdout_full", ""))
        stderr_path.write_text(session.get("_stderr_full", ""))
    except OSError as e:
        log.warning(f"Failed to persist session output: {e}")
        return

    # Retain only the most recent FAILED_OUTPUT_RETENTION pairs
    for pattern in ("iter-*-stdout.txt", "iter-*-stderr.txt"):
        files = sorted(glob.glob(str(LOG_DIR / pattern)))
        for old in files[:-FAILED_OUTPUT_RETENTION]:
            try:
                os.unlink(old)
            except OSError:
                pass


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
    # Ensure the editable angr install is importable from spawn-multiprocessing
    # children regardless of cwd or sys.path[0]. systemd-run --scope preserves
    # cwd, but `python tests/benchmarks/run_regression.py ...` sets sys.path[0]
    # to that script's directory, and spawn workers inherit it — leaving REPO_DIR
    # off sys.path. PYTHONPATH lands in front of every interpreter started in
    # the scope, including spawn workers.
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(REPO_DIR)
    )

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

    # Detect "gate broken" pattern: every failure is an import / harness error
    # rather than a real benchmark regression. When the gate itself can't run
    # (e.g. angr import fails), we'd otherwise silently treat 12 fake failures
    # as 12 real regressions. Treat this as a distinct, infrastructural failure.
    gate_broken = False
    if failed > 0 and regressions:
        broken_markers = (
            "No module named",
            "ImportError",
            "ModuleNotFoundError",
            "subprocess crashed",
        )
        if all(any(m in r for m in broken_markers) for r in regressions):
            gate_broken = True

    return {
        "ran": True,
        "exit_code": exit_code,
        "passed": passed,
        "failed": failed,
        "regressions": regressions,
        "duration_secs": duration,
        "output": stdout[-2000:],
        "gate_broken": gate_broken,
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
        "bead_id": extract_bead_id(git_after, git_before),
        "exit_reason": derive_exit_reason(session),
        "session": {
            "exit_code": session["exit_code"],
            "killed_by_oom": session["killed_by_oom"],
            "killed_by_timeout": session["killed_by_timeout"],
            "mode": session.get("mode", "unknown_error"),
            "rate_limited": session["rate_limited"],  # derived alias, kept for one cycle
            "subtype": session.get("claude_subtype"),
            "cost_usd": session.get("claude_cost_usd"),
            "num_turns": session.get("claude_num_turns"),
            "api_error_status": session.get("claude_api_error_status"),
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


# ── PID Lockfile ──────────────────────────────────────────────────────────────

def _acquire_pidfile() -> bool:
    """Write PID file, return False if another instance is running."""
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            # Check if the process is still running
            os.kill(old_pid, 0)
            log.error(f"Another loop instance is running (PID {old_pid}). Exiting.")
            return False
        except (ProcessLookupError, ValueError):
            log.info(f"Removing stale PID file (PID was gone)")
        except PermissionError:
            log.error(f"Another loop instance is running (PID file exists, permission denied)")
            return False
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    return True


def _release_pidfile():
    """Remove PID file."""
    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


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
        _release_pidfile()
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
    parser.add_argument("--stress-test", action="store_true",
                        help="Stress-test mode: 300s timeout, 512M memory, 3 iterations, skip benchmarks")
    args = parser.parse_args()

    # Apply stress-test presets
    if args.stress_test:
        args.timeout = 300
        args.memory_limit = "512M"
        args.max_iterations = 3
        args.skip_benchmarks = True

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

    if args.stress_test:
        log.info("STRESS TEST MODE: timeout=%ds, memory=%s, iterations=%d, benchmarks=off",
                 args.timeout, args.memory_limit, args.max_iterations)

    # Dry run mode — exercises module import, argparse, logging, signal handler
    # setup, git state detection, and prompt template without requiring
    # claude/bd/systemd-run binaries or pidfile/cgroup state. This is the CI
    # smoke-test entry point (see .github/workflows/ci.yml orchestrator_smoke).
    if args.dry_run:
        git_state = detect_git_state()
        prompt = build_prompt(git_state)
        prompt_type = "dirty" if git_state["is_dirty"] else "clean"
        print(f"--- Prompt type: {prompt_type} ---")
        print(f"--- Git state: dirty={git_state['is_dirty']}, HEAD={git_state['head_sha']} ---")
        print(prompt)
        return

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

    # Acquire PID lockfile
    if not _acquire_pidfile():
        sys.exit(1)

    # Clean up stale scopes from previous crashes (F4)
    _cleanup_stale_scopes()

    startup_source_hash = _orchestrator_source_hash()
    log.info(f"Starting optimization loop: max_iterations={args.max_iterations}, "
             f"timeout={args.timeout}s, memory_limit={args.memory_limit}, "
             f"source_sha={startup_source_hash}")

    consecutive_failures = 0
    consecutive_dirty = 0
    short_dead_streak = 0
    iteration = 0
    terminal_failure: Optional[str] = None  # set when calculate_backoff returns BACKOFF_EXIT

    stale_warned = False
    for iteration in range(1, args.max_iterations + 1):
        rss = monitor_memory()
        log.info(f"{'='*60}")
        log.info(f"Iteration {iteration}/{args.max_iterations} | RSS={rss:.1f}MB")

        # Detect stale orchestrator: if the source file on disk has been
        # modified since process start, the running bytecode is out of date.
        # angr-lbze masked a real bug for 28+ iterations exactly because the
        # orchestrator kept running pre-fix code while the patched file sat
        # untouched on disk. Warn once so logs make this obvious.
        if not stale_warned and startup_source_hash is not None:
            current_hash = _orchestrator_source_hash()
            if current_hash is not None and current_hash != startup_source_hash:
                log.warning(
                    f"ORCHESTRATOR SOURCE STALE: run_optimization_loop.py changed on disk "
                    f"(startup={startup_source_hash}, on-disk={current_hash}). "
                    f"This process is still running pre-edit bytecode — restart to pick up changes."
                )
                stale_warned = True

        # Check task availability
        tasks = check_tasks_available()
        log.info(f"Tasks: in_progress={tasks['in_progress']}, ready={tasks['ready']}")
        if tasks["in_progress"] == 0 and tasks["ready"] == 0:
            log.info("No tasks available. Exiting loop.")
            break

        # Detect git state and choose prompt
        git_before = detect_git_state()

        # Dirty state circuit breaker
        if git_before["is_dirty"]:
            consecutive_dirty += 1
            if consecutive_dirty >= DIRTY_REVERT_THRESHOLD:
                log.warning(f"Dirty state persisted for {consecutive_dirty} iterations — auto-reverting")
                # Capture what we're reverting for the record
                _, diff_stat, _ = _run_cmd(["git", "diff", "--stat"], timeout=10)
                _run_cmd(["git", "checkout", "--", "."], timeout=10)
                _run_cmd(["git", "clean", "-fd"], timeout=10)
                # Defer any in-progress tasks so the agent doesn't re-claim them
                rc, ip_out, _ = _run_cmd(["bd", "list", "--status=in_progress", "--json"])
                try:
                    for issue in json.loads(ip_out):
                        issue_id = issue.get("id", "")
                        title = issue.get("title", "unknown")
                        if issue_id:
                            _run_cmd(["bd", "defer", issue_id], timeout=10)
                            log.warning(f"Deferred stuck task {issue_id} ({title})")
                            # Record why so future sessions don't repeat
                            reason = f"Auto-deferred: {consecutive_dirty} dirty iterations, agent couldn't commit. Files: {diff_stat.strip()[:200]}"
                            _run_cmd(["bd", "update", issue_id,
                                      f"--notes={reason}"], timeout=10)
                            _run_cmd(["bd", "remember",
                                      f"--key=avoid-stuck-{issue_id}",
                                      f"Task {issue_id} ({title}) was auto-deferred after {consecutive_dirty} consecutive dirty iterations. The agent could not complete build/test/commit cycle within session time limits. May need to be broken into smaller subtasks or done manually."],
                                     timeout=10)
                except (json.JSONDecodeError, TypeError):
                    pass
                consecutive_dirty = 0
                git_before = detect_git_state()  # Re-detect after revert
        else:
            consecutive_dirty = 0

        prompt = build_prompt(git_before)
        prompt_type = "dirty" if git_before["is_dirty"] else "clean"
        log.info(f"Prompt: {prompt_type} | HEAD: {git_before['head_sha']} {git_before['head_msg'][:60]}")
        if git_before["is_dirty"]:
            log.info(f"Dirty files ({git_before['dirty_file_count']}): {git_before['dirty_files_summary'][:200]}")

        # Run claude session in cgroup scope
        session = run_claude_session(prompt, iteration, args.timeout, args.memory_limit)
        log.info(f"Session done: exit={session['exit_code']}, duration={session['duration_secs']}s, "
                 f"oom={session['killed_by_oom']}, timeout={session['killed_by_timeout']}, "
                 f"mode={session['mode']}")

        # Detect git state after
        git_after = detect_git_state()
        commits_made = git_before["head_sha"] != git_after["head_sha"]
        if commits_made:
            log.info(f"New commits: {git_before['head_sha']} -> {git_after['head_sha']} ({git_after['head_msg'][:60]})")

        # Benchmark gate
        benchmark = None
        if commits_made and not args.skip_benchmarks:
            benchmark = run_benchmark_gate(args.memory_limit)
            if benchmark.get("gate_broken"):
                log.error(
                    "BENCHMARK GATE BROKEN: every failure is an import/harness "
                    f"error, not a regression. failed={benchmark['failed']} "
                    f"regressions[0]={benchmark['regressions'][0] if benchmark['regressions'] else '?'} "
                    "— gate is not validating commits"
                )
            elif benchmark["failed"] > 0:
                log.warning(f"BENCHMARK REGRESSIONS ({benchmark['failed']}): {benchmark['regressions']}")
            else:
                log.info(f"Benchmark gate passed: {benchmark['passed']} benchmarks OK in {benchmark['duration_secs']}s")

        # Update dead-session streak: short failed sessions with no commits indicate
        # the CLI is bouncing without actually running. Reset on success or any
        # session that ran long enough to have done real work.
        is_dead = (
            session["exit_code"] != 0
            and session["duration_secs"] < DEAD_SESSION_DURATION_SECS
            and not commits_made
        )
        if is_dead:
            short_dead_streak += 1
        else:
            short_dead_streak = 0

        # Persist stdout/stderr for any non-ok session
        if session["mode"] != "ok":
            persist_failed_session_output(iteration, session)

        # Classify outcome and backoff
        backoff = calculate_backoff(
            session["mode"], session.get("rate_limit_reset_secs"),
            session["killed_by_oom"], session["killed_by_timeout"],
            session["exit_code"], consecutive_failures, short_dead_streak,
        )

        if session["mode"] == "ok":
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        # Log structured JSON
        try:
            log_iteration(iteration, git_before, git_after, session, prompt_type,
                          tasks, benchmark, max(backoff, 0), rss)
        except OSError as e:
            log.warning(f"Failed to write iteration log: {e}")

        # Terminal failure: log and exit the loop
        if backoff == BACKOFF_EXIT:
            terminal_failure = session["mode"]
            log.error(f"Terminal failure ({terminal_failure}); stopping loop. "
                      f"See {LOG_DIR} for captured stdout/stderr.")
            break

        # Backoff
        if backoff > 10:
            log.info(f"Backing off {backoff}s ({backoff/60:.0f}m) before next iteration...")
        time.sleep(backoff)

    log.info(f"Loop complete after {iteration} iterations.")
    log.info(f"Review logs: cat {LOG_DIR / 'summary.jsonl'} | python -m json.tool --no-ensure-ascii")
    _release_pidfile()
    if terminal_failure is not None:
        sys.exit(2)


if __name__ == "__main__":
    main()
