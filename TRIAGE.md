# Triage: priority and type for beads

Rules for filing and re-rating beads (`bd create`, `bd update --priority`,
`bd update --type`). Priority and type are independent axes: **priority is the
consequence for the engine's users; type is the kind of change.** Neither
encodes how much work something is or how much we want it. Priority also
drives scheduling (the ralph loop drains the lowest P-number first), so an
inflated band displaces real bugs.

## Priority = consequence × reachability

| Band | Rule |
|------|------|
| **P0** | **Blocks all work.** CI or the ralph gate is red on the branch, the venv or build is broken, the benchmark baseline is invalidated. A state of the repo, not a bug severity. Rare and short-lived. |
| **P1** | **Silently wrong and reachable.** The engine returns a wrong answer, accepts a wrong state, or loses states/constraints with no error, no panic and no log, on a path real input reaches: a supported arch, the public Python API, an ordinary syscall or libc call, default options. A gate or test that cannot fail belongs here too: it silently returns a wrong CI answer. |
| **P2** | **Loud failure, or silent-wrong with narrowed reach.** Crash, panic, process abort, UB. Hang, OOM, unbounded allocation from a guest-controlled size. A leak that grows per iteration. A perf regression past the gate threshold. A missing feature on a supported path that fails visibly (`NotImplementedError`, Python fallback). Silent-wrong that needs a non-default option, an unsupported arch or an unusual width to reach. |
| **P3** | **Latent correctness and correctness-flavoured hygiene.** Silent-wrong on a dead or currently-unreachable path, or held off only by a caller invariant. `debug_assert!` where release needs a real check. A missing `SILENT(cat-x)` tag. A test-coverage gap on a live path. Stale docs that would mislead onboarding. Dead code. A missing feature with no current consumer. |
| **P4** | **Dormant, defensive, cosmetic.** Defense-in-depth for a caller that does not exist yet, frozen features, naming, doc nits, coverage for defensive arms, anything whose precondition is "only if X lands". |

### Modifiers

Each moves one band at most. Say which one applied in the rationale line.

- **Reachability demotes.** Not reachable from real input today: P1 becomes P2
  when reach is merely narrow, P3 when the path is latent or dead; P3 becomes P4
  when it is pure defense-in-depth.
- **Blast radius promotes.** Every call, every arch, every state affected: one
  band up. Never into P0, which is reserved for repo blockers.
- **Loudness demotes.** If a `log::warn!`, an error or a panic already fires,
  it is not silent. P2 at most.

### Decision ladder

Ask in order; stop at the first yes.

1. Does it stop CI, the ralph loop or the build? **P0.**
2. Is the result wrong with no signal, and does real input reach it? **P1.**
   Latent or dead path: **P3.**
3. Does it crash, hang, abort, leak or regress perf? **P2.** Latent: **P3.**
4. Is functionality missing? A supported path needs it now: **P2.** Otherwise
   **P3** or **P4**.
5. Anything else is hygiene: **P3** if it protects a live path, **P4** if it is
   dormant or cosmetic.

## Type = kind of change

| Type | Use when | Usual band |
|------|----------|------------|
| `bug` | Observed behaviour differs from the spec: Python-engine parity, VEX or hardware semantics, a documented contract, the benchmark baseline. Silent or loud, crash or leak, and gates that cannot fail. | P1–P3; P4 only when dormant |
| `chore` | No behaviour change: docs and comments, dead code, DRY/split/rename, clippy, baseline refresh, `SILENT` tags, test coverage for behaviour that already exists. | P3–P4, never P1 |
| `task` | Changes behaviour or produces knowledge but fixes no defect: bisects, measurements, roadmap steps, migrations, flag flips, re-verification, hardening with new tests. | P2–P3 |
| `feature` | New capability: a new arch, native SimProcedure or syscall, a new API. | P2–P3 |
| `spike` | Timeboxed go/no-go investigation. | P2–P3 |
| `epic` | Container. Takes the band of its highest-priority open child. | — |
| `decision` | Architecture decision record. | — |

Rules of thumb:

- A title that would start with `bug:` or `[bug]` is type `bug`.
- A title that would start with `docs:`, `dead-code:`, `DRY:`, `test-coverage:`,
  `clippy:` or `refactor:` is type `chore`.
- A `task` is never P1. If a task is the fix vehicle for a P1 defect, file the
  bug and make the task its child.
- `bd lint` requires *Steps to Reproduce* and *Acceptance Criteria* on a bug and
  *Acceptance Criteria* on a task or feature. Keep satisfying it.

## Required rationale line

Every bug's description ends with one line:

```
Why P<n>: <consequence>, <reachability>
```

For example `Why P1: silent wrong-answer, reachable via ordinary mmap on amd64`
or `Why P3: silent wrong-answer, but only on the dead load_concrete path`.
When a verifier or a later reader changes the band, append a note rather than
editing silently:

```
bd note <id> "Severity P2->P3: <reason>"
```

## Worked examples

Read with `bd show <id>`; each shows the band and the reasoning that put it there.

- **P1, silent and reachable:** `angr-c7xno.50` (merge drops symbolic objects),
  `angr-sqfj8.121` (constraint double-listing corrupts unsat cores),
  `angr-hv4lt.15` (a leak gate that could not fail).
- **P2, loud:** `angr-fs8kb.28` (process abort on a width-0 BV),
  `angr-mi56k` (unbounded allocation from a format width).
  **P2, silent with narrowed reach:** `angr-hrcds` (its note records the
  P3 to P2 reasoning).
- **P3, latent:** `angr-fs8kb.65` (a "why P3 not P2" reachability trace),
  `angr-03vl4.69` (fallback with no `SILENT` tag).
- **P4, dormant:** `angr-myzjx.13` (divergence on a path nothing takes),
  `angr-j60q0` (defense-in-depth epic).
- **P0, blocker:** `angr-1c4z` (formatting drift that reddened CI).
