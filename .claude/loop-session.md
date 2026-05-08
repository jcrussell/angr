# Loop session notes (2026-05-08, 150th loop session)

## Task: angr-zcvu — Loop telemetry: bead_id and granular exit_reason in summary.jsonl

### Status: investigating → implementing

### Goal
Update run_optimization_loop.py so that:
1. summary.jsonl entries include `bead_id` (when known) and `exit_reason`
2. `api_error_status: 429` with "monthly usage limit" text → exit_reason='budget_exhausted' (terminal)
3. Other 429s → exit_reason='rate_limited' (recoverable)
4. Orchestrator aborts on budget_exhausted (this already exists; just need correct classification)

### Root cause of bug being fixed
The Claude CLI returns:
```json
{"subtype":"success","is_error":true,"api_error_status":429,
 "result":"You've hit your org's monthly usage limit",...}
```

`detect_failure_mode` currently only inspects `subtype` + `errors[]`. The
monthly-limit text lives in `result` (not `errors`), and `subtype` is "success"
not "error_*". So the function falls through to `unknown_error`, which means
the orchestrator does exponential backoff (10→20→40→80 min) instead of
aborting. Iters 29/30/31 in summary.jsonl are exactly this — three
back-to-back monthly-limit 429s misclassified as `unknown_error`.

### Plan
1. Update `detect_failure_mode` to also pull `result` and `api_error_status`
   into the haystack. Add explicit branch: `api_error_status == 429`
   AND ("monthly" in haystack) → `budget_exhausted`. Other `api_error_status
   == 429` → `rate_limit`.
2. Add `exit_reason` field to log entries (alongside `mode`, more granular).
3. Add `bead_id` extraction: parse new HEAD commit message after session
   for `angr-[a-z0-9]+` pattern (most reliable when commits made); fall
   back to reading `.claude/loop-session.md` `## Task: <id>` line.

### Files
- run_optimization_loop.py
