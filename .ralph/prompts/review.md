# Iteration {{.Iter}} — review

Review mode on `{{.Review.Branch}}` (base `{{.Review.Base}}`). The
single review state covers four sub-flows; pick exactly one per
iteration based on the current signal:

| Signal                                | Sub-flow         |
|---------------------------------------|------------------|
| `bd ready -l review:{{.Review.Branch}}` has fresh items | **ingest**  |
| any open finding in progress           | **fix**          |
| `{{.GateResult}}` == `fail`            | **address-gate** |
| no open findings + tree clean + gate ok | **file-merge**   |

There are {{.Review.OpenFindings}} open findings against
`review:{{.Review.Branch}}`. Last gate result: `{{.GateResult}}`.
Working tree is {{if .GitDirty}}dirty{{else}}clean{{end}}.

**Ingest.** New issues from review came in. Read each, decide
priority, and either fix in this branch or defer with a clear note.
Do not let the queue accumulate uncategorized findings.

**Fix.** Pick one finding (`bd ready -l review:{{.Review.Branch}}`),
claim it, implement, test, commit, close. Same discipline as
clean-state iterations — one finding per iteration.

**Address-gate.** The last gate failed. Read the gate output (it's
in {{.LastIter}}). Reproduce locally. Fix the cause, not the symptom.
Re-run the gate manually before committing.

**File-merge.** All findings closed + tree clean + gate green. File
a merge-ready bd note (`bd remember --key
review:{{.Review.Branch}}:ready "..."`) and the orchestrator will
finalize as done{queue_empty} on the next routing tick.

Never bypass the gate. Never force-push.
