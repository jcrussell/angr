# Offline-prepare: decision briefs

Fourth pattern in the offline-prepare family (siblings:
`upstream_patches/`, `draft_patches/`, the aggregation-doc
v1-partial convention). See `../OFFLINE_WORKFLOWS.md` for the
full decision matrix.

## When to use

When a bd bead is blocked on a **binary decision between two
viable paths** — not on missing access, missing data, or a
missing sibling — and neither path is safe to apply
autonomously from the loop.

Examples:

- "Enable Cargo feature X by default OR write an alternative
  workload that doesn't need X." Both paths are work; only a
  human can weigh wheel-build / dep-inflation risk.
- "Land downstream monkey-patch OR wait for upstream PR
  merge." Both are valid; one is faster but technical-debt,
  the other is principled but slow.

When the blocker is "I need to know what value to type into a
config file", that's a `draft_patches/` runbook, not a
decision brief.

## File shape

One markdown per decision, named `<bead-id>_<topic>.md`.
Required sections:

- **Status** — link back to the bead, current state.
- **Context** — one paragraph on why the bead is blocked.
- **Options** — at minimum two, with tradeoffs honest about
  what we *can* and *cannot* verify offline.
- **Recommendation** — pick one, justify briefly. If the
  recommendation is "escalate to human", say that and explain
  why neither is loop-applicable.
- **Implementation sketch** — for the recommended path, the
  files/changes needed. Doesn't need to be the patch itself;
  this is a brief, not a runbook.

## Bookkeeping

- Update the bead notes (`bd update <id> --notes`) to point
  at the brief. Do NOT create a new tracking bead — the brief
  *is* the deliverable for this slice.
- The bead stays open until the decision is applied. The
  brief lives in-tree as the durable artifact.
- If the human picks the path and it lands, leave the brief
  in place with a closing "Resolved: applied path X on
  <date>, commit <hash>" line — this is the receipt for
  future audits.
