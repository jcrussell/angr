# Clean state

The working tree is clean. Your job: drain one task from the bd queue.

## Bootstrap

1. `bd prime` — load the current memory / context dump.
2. Read `.ralph/state/session.md` for the previous session's handoff notes.
3. Check for in-flight work: `bd list --status=in_progress`.
4. If something is in_progress, continue it. Otherwise: `bd ready` and pick
   the highest-priority unblocked **leaf** task (lowest P-number). **Skip
   `[epic]` rows** — epics are containers, not drainable work, and `bd ready`
   lists them *above* leaves in the same priority band. If a candidate still has
   open children (`bd children <id>`), it's a container too — descend to its
   lowest-P open leaf. Pick the first row that is itself actionable.

## Maintenance fallback (only when step 4 finds no actionable leaf)

Evaluate this table **only after** step 4 confirms nothing is in_progress and
`bd ready` yields no lint-clean, unblocked, actionable leaf. The trigger is
**objective** — tie "not actionable" to `bd lint` output and
`[epic]`/open-children status, never to a subjective "this looks vague" call.
A lint-clean unblocked leaf is **always** taken (do the task; skip this table).

| Signal (objective)                                             | Do this        |
|---------------------------------------------------------------|----------------|
| A ready **leaf** exists (not `[epic]`, no open children, passes `bd lint`) | do the task (step 4) |
| Ready rows are **only** `[epic]` / containers                 | **decompose** one epic |
| Ready leaves exist but **all fail `bd lint`** (missing sections) | **groom**: fill sections / re-scope |
| Ready leaves are all **blocked**                              | **unblock**: verify deps, drop stale `bd dep` links |
| `bd ready` non-empty but nothing above applies                | **groom one item**, else **idle** |

Rules:

- **Exactly one maintenance action per iteration** (mirrors the one-task
  discipline). Menu, non-destructive first, using verified bd subcommands:
  1. Decompose a ready epic into concrete leaf tasks
     (`bd create ... --parent <epic>`).
  2. Fill missing template sections flagged by `bd lint` on high-priority beads.
  3. Refresh stale beads (`bd stale`) — re-scope / reprioritize drifted ones.
  4. Verify blockers — drop stale links with `bd dep remove` so blocked work
     becomes ready.
  5. Flag duplicates (`bd find-duplicates` / `bd duplicates`) — prefer a
     `bd note`/label pointing at the canonical bead over an autonomous merge.
- **Closing/deleting is a last resort**: only when *clearly* obsolete, always
  with a `bd note` on the bead explaining why (bd is gitignored → a bad close
  leaves no diff to review). Reserve `bd remember` for genuinely cross-cutting
  invariants — the memory count is actively managed, so default to a per-bead
  `bd note`, not a new memory.
- **No git commit expected.** A maintenance iteration normally leaves the tree
  clean with no commit — that's fine, do not fabricate one. Update `session.md`.

**Anti-churn stop condition (the key rule).** A bd mutation resets ralph's
no-op streak, so unbounded grooming would never trip `MaxNoopIters=10` and
could burn every iteration on low-value churn. Therefore: groom **only** for
obvious, high-value items (an epic that clearly needs breakdown, a P0/P1 bead
failing lint, a genuinely stale bead). Once those are exhausted, **prefer a
true idle iteration** — write a one-line "nothing actionable, nothing worth
grooming" note to `session.md`, make **no bd mutation**, and end. That lets the
no-op streak accrue so `MaxNoopIters` can self-terminate the loop. Do **not**
groom just to have done something.

## Scope discipline

Complete **ONE task per session**. Only take a second task if it is closely
related to the first (same files, shared context) AND the first task completed
quickly (<15 minutes). This keeps sessions focused and limits blast radius from
rate limits or crashes.

## Session log

Throughout your work, keep `.ralph/state/session.md` updated with:

- Task ID and title you are working on
- Current step (investigating / implementing / testing / committing)
- Key findings or decisions made so far
- Files you have modified

This file is your handoff note to the next session. Update it at each major
milestone, not just at the end. Write it as if briefing a colleague who will
pick up exactly where you left off.
