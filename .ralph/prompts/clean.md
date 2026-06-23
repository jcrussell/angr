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
