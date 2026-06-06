# Offline workflows for stuck-queue drainage

Three patterns in the kit for turning bd tasks that are blocked
on **external resources** (github, CI artifacts, runtime
measurements, sibling beads) into **productive offline drainage**.

All three share the same underlying move: **file a sibling slice
bead** for the portion of the work that *can* be done now, ship
that slice, and leave the parent bead open until the blocker
clears. The slice closes on artifact commit; the parent closes
when the real-world action lands.

## Decision matrix — which pattern fits?

| Blocker class | Pattern | Artifact destination |
|---|---|---|
| External repo unreachable (e.g. `gh` blocked but PR is ready to file) | **upstream-PR offline-prepare** | `tools/upstream_patches/` |
| Runtime measurement / CI artifact / data missing (patch shape known, value not) | **internal-patches offline-prepare** | `tools/draft_patches/` |
| Aggregation-doc bead has one blocked sub-bead but most siblings done | **aggregation-doc v1-partial** | The aggregation doc itself, with a stub for the blocked section |

Pick by **what's missing**, not by what the work looks like:
- Missing *access* → patch + PR body offline, file later.
- Missing *data* → patch with placeholder offline, fill later.
- Missing *one sibling's result* → ship the aggregation now with
  a marked stub.

## Shared bookkeeping convention

For all three patterns:

- **Slice bead** is filed as a sibling of the parent (use
  `bd dep relate <slice-id> <parent-id>` to link). It closes
  when the offline artifact lands in-tree.
- **Parent bead** stays open. It only closes when the
  real-world action completes (PR merged, measurement landed,
  blocked sibling fills in).
- Pre-commit hooks and bench-regression gates apply normally —
  these slices are real commits, not WIP.
- Each artifact pair carries enough self-documenting context
  (runbook / README entry / stub markers) that a future iter
  can finish the work without re-reading bd history.

## Pattern A — Upstream-PR offline-prepare

**Use when**: a bd task is to file a PR against an angr-ecosystem
upstream repo, but `github.com` (or `gh auth`) is unreachable.

**Artifacts**: `tools/upstream_patches/<topic>.patch` plus
`tools/upstream_patches/<topic>_PR.md`. See
[`tools/upstream_patches/README.md`](upstream_patches/README.md)
for the apply-and-file workflow.

**Reference**: bd memory
`upstream-pr-offline-prepare-slice-pattern`.

**Example**: `angr-adtv` (slice of `angr-b3sc`, archinfo AArch64
BE alias support).

## Pattern B — Internal-patches offline-prepare

**Use when**: the patch shape is known but a placeholder requires
a measurement (CI artifact, benchmark capture, coverage baseline)
that the current environment can't produce.

**Artifacts**: `tools/draft_patches/<topic>.patch` plus
`tools/draft_patches/<topic>_RUNBOOK.md`. See
[`tools/draft_patches/README.md`](draft_patches/README.md) for
the placeholder-substitution workflow.

**Reference**: bd memory
`internal-patches-offline-prepare-pattern`.

**Example**: `angr-zlzw` (slice of `angr-ywu7`, coverage
`fail_under` gate — patch ships with `TBD-PY-FLOOR` placeholder
that fills in once baseline lands).

## Pattern C — Aggregation-doc v1-partial

**Use when**: a documentation-aggregation bead is blocked by one
sub-bead but ≥3 sibling sub-beads have shipped with substantive
findings. The writeup itself is the value; waiting on the last
sibling stalls a doc that's already useful.

**Artifacts**: the aggregation doc lands at its final path with
a **clearly marked stub** for the blocked section (e.g.
`*Pending angr-XXXX — see bead for status*`). No subdir under
`tools/`.

**Reference**: bd memory `aggregation-doc-v1-partial-pattern`.

**Example**: `angr-j8sr` (slice of `angr-kvn0`, Rust engine
characterization writeup — shipped `.1/.3/.4` sections, stubbed
`.2` pending the xmllint sub-bead chain).

## Why these patterns exist

The autonomous loop runs in an environment with constrained
network reach and limited heavy compute. Without these patterns,
any bead gated on an offline-impossible step ages indefinitely
in the `bd ready` queue, generating no value. With them, the
*offline-doable* fraction of every blocked task gets committed
incrementally — leaving the final "online" action as a small,
well-scoped commit when the blocker clears.

Pattern lineage tracked in bd via memory keys
`upstream-pr-offline-prepare-slice-pattern`,
`internal-patches-offline-prepare-pattern`, and
`aggregation-doc-v1-partial-pattern`. Update those when the
patterns evolve; keep this file as the contributor-facing
entry point.
