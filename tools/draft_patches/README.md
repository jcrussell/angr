# Draft patches (internal, offline-prepared)

This directory holds patches and runbooks prepared offline against
**this repo** for bd tasks that are blocked on a runtime
measurement, a CI artifact, or some other piece of data that can
only be obtained when network or a heavier environment is
available.

This is the **internal sibling** of `tools/upstream_patches/` —
the upstream variant ships PRs to angr-ecosystem repos when github
is unreachable; this one stages local changes when the data needed
to finalize them is unavailable. Both follow the same offline-
prepare slice pattern (`bd recall upstream-pr-offline-prepare-slice-pattern`).

## How to apply when the missing data is available

For each `<topic>.patch` + `<topic>_RUNBOOK.md` pair:

```bash
# 1. Read the runbook to learn what value(s) need to be measured
cat tools/draft_patches/<topic>_RUNBOOK.md

# 2. Run the measurement workflow described there
#    (e.g. coverage baseline run, benchmark capture, …)

# 3. Edit the patch in-place to replace the placeholder(s) — the
#    runbook calls out every TBD-* marker by name.

# 4. Verify the patch still applies cleanly against current HEAD
git apply --check tools/draft_patches/<topic>.patch

# 5. Apply, sanity-test, commit
git apply tools/draft_patches/<topic>.patch
# Run any verification commands listed in the runbook
git add -A && git commit
# Close the parent bd bead with the measured value(s) noted
```

## Convention

* Patches use unified diff format (git-apply + GNU `patch -p1`
  compatible). Each carries a leading `Subject:` block to
  document intent.
* Runbooks live next to the patch with the suffix `_RUNBOOK.md`.
  They list every placeholder by name and explain how to derive
  its value.
* Patches are **never** applied automatically by the autonomous
  loop or by CI — they sit dormant until a human (or a future
  iteration with the right environment) runs the runbook.
* Each artifact pair has a corresponding bd slice bead. The
  PARENT bead (the one the patch will eventually fix) stays open
  until the patch lands; the SLICE bead closes once the artifacts
  are committed.

## Current artifacts

### `ywu7_coverage_fail_under.patch` + `ywu7_coverage_fail_under_RUNBOOK.md`

Target: `angr-ywu7` (CI: enforce coverage minimum via
`tool.coverage.report.fail_under`).

Adds the patch shape — `pyproject.toml` `[tool.coverage.report]`
`fail_under = TBD-PY-FLOOR` and a `coverage.yml` report-step
enforcement — with the threshold left as a placeholder. The
runbook explains how to measure the baseline (per the blocked
sibling `angr-439q`) and pick the value.

Downstream unblock: once baseline lands and the patch is applied,
`angr-ywu7` closes (CI gates coverage drops > 2pp).
