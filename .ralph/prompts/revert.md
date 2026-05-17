# Revert state (one-shot)

Three consecutive iterations left the working tree dirty without progress —
ralph reset the tree to HEAD ({{.GitHead}}). This prompt runs once. Use it to
defer the stuck task and capture what we learned.

1. **Verify clean.** `git status` should be clean. If not, `git checkout -- .
   && git clean -fd`.
2. **Defer in-progress beads.** For each task from `bd list --status=in_progress`,
   `bd defer <id>` with a note explaining the cause (read
   `.ralph/state/session.md` for context).
3. **Save the lesson.** `bd remember --key avoid-<topic> "..."` describing
   the failure pattern so future iterations can search for it. Be specific —
   vague memories don't help.
4. **Note in session.md.** Append a one-line entry to `.ralph/state/session.md`
   summarizing the revert and what was deferred.

Do NOT start new work and do NOT immediately re-attempt the deferred bead. The
next iteration will run `clean.md` and pick fresh work.
