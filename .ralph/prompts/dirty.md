# Dirty state

The previous session left **uncommitted changes**. Your first priority is to
assess and resolve them. Do NOT start new work in this session — only resolve
the dirty state.

## Bootstrap

1. `bd prime` — load context.
2. Read `.ralph/state/session.md` to understand what the previous session was
   doing.
3. Run `git status && git diff --stat` to see exactly what's modified.
4. `bd list --status=in_progress` — find the associated task.
5. Try to build: `cargo check --manifest-path native/angr/Cargo.toml --release`

## Decision matrix (time-box this to 15 minutes)

| Compiles? | Tests pass? | Action                                                   |
|-----------|-------------|----------------------------------------------------------|
| yes       | yes         | Finish the work, commit, close the task.                 |
| yes       | no          | Investigate briefly; fix if ≤10 min or revert.           |
| no        | n/a         | Attempt fix (max 10 min), then revert if stuck.          |
| obviously broken | n/a  | `git checkout -- . && git clean -fd` immediately.        |

If the decision is unclear after 15 minutes, revert and `bd defer <id>` with
a `bd remember --key avoid-<topic>` explaining what went wrong.

## Session log

Keep `.ralph/state/session.md` updated with task ID, current step (assessing /
fixing / reverting), and key decisions. This is the handoff to the next session.
